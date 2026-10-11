"""
单股 AI 分析（计划 2026-10-10 第 8 项）：给一个代码，本机把数据备齐，交给 claude.exe（Opus 5.5）
结合最新新闻公告做一次详尽分析，输出结构化结论。**只给人看，不进清单、不进邮件、不进模型。**

    python tools/ai_analyze.py 300829            终端看
    python tools/ai_analyze.py 300829 --json     最后一行 JSON（控制台用）
    python tools/ai_analyze.py 300829 --no-web   不联网查新闻（便宜、快，只用本机数据）

本机备的材料（AI 不碰生产表，拿到的是算好的事实）：
  · 模型怎么看：predict_one 的分数 / 名次 / 够不够格 / 近 20 天分数 / 这一档的历史命中
  · 近 60 根 K 线的事实：涨跌幅、量比、换手、距 52 周高低点、近 20 日涨停次数、连板、回撤
  · 股东户数最近几期
  · 减持 / 定增（corp_events，能拉就现拉，拉不到写明）
  · 市场环境（regime_daily 近 10 天：成交额、涨停家数、上涨占比、滚动基准率）
AI 自己查：最新新闻、公告、题材（WebSearch / WebFetch，只读）。
输出 schema：立场（看多 / 看空 / 观望）、置信度、时间范围、三条依据、三条风险、
每条依据标明是数据 / 新闻 / 推断、引用链接、什么情况下会改判、一句话结论。
结果落 state/ai_analysis/<代码>_<打分日>.json；用量记账在 src/ai_usage.py。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import ai_usage      # noqa: E402

DATA = ROOT / "data" / "breakout"
OUT = ROOT / "state" / "ai_analysis"

SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": ["看多", "看空", "观望"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "horizon": {"type": "string"},
        "one_line": {"type": "string"},
        "reasons": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object",
            "properties": {"point": {"type": "string"},
                           "basis": {"type": "string", "enum": ["数据", "新闻", "推断"]},
                           "source": {"type": "string"}},
            "required": ["point", "basis"]}},
        "risks": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object",
            "properties": {"point": {"type": "string"},
                           "basis": {"type": "string", "enum": ["数据", "新闻", "推断"]},
                           "source": {"type": "string"}},
            "required": ["point", "basis"]}},
        "news": {"type": "array", "maxItems": 8, "items": {
            "type": "object",
            "properties": {"date": {"type": "string"}, "title": {"type": "string"},
                           "url": {"type": "string"}, "impact": {"type": "string"}},
            "required": ["title"]}},
        "change_mind": {"type": "string"},
        "model_vs_ai": {"type": "string"},
        "unverified": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["stance", "confidence", "horizon", "one_line", "reasons", "risks",
                 "change_mind", "model_vs_ai"],
}

SYSTEM = (
    "你是一位严谨的 A 股分析师，只按证据说话。用户是中文使用者，不要投资建议式的套话，"
    "不要「值得关注」「有望受益」这类词，要具体事实和数字。每一条依据都要标明来源类型："
    "数据（我给你的材料）、新闻（你查到的，附链接）、推断（你自己的判断）。"
    "查新闻时只用 WebSearch 和 WebFetch，优先交易所公告、公司公告、主流财经媒体，"
    "查不到就写进 unverified，不要编造。时间范围固定为「未来 20 个交易日」。"
)


def _k_facts(code: str) -> dict:
    px = pd.read_parquet(DATA / "daily.parquet", filters=[("code", "==", code)])
    if px.empty:
        return {}
    px = px.sort_values("date").reset_index(drop=True)
    c = px["close"].to_numpy(float)
    h, lo, v = px["high"].to_numpy(float), px["low"].to_numpy(float), px["volume"].to_numpy(float)
    n = len(px)
    import datasource as ds
    pct = ds.limit_pct(code, "")
    prev = np.r_[np.nan, c[:-1]]
    lim = ds.limit_price_arr(pd.Series(prev), pct, pd.Series([code] * n)).to_numpy(float)
    lu = (c >= lim - 1e-9) & np.isfinite(lim)
    streak = 0
    for x in lu[::-1]:
        if x:
            streak += 1
        else:
            break

    def ret(k: int) -> float | None:
        return round(100 * (c[-1] / c[-1 - k] - 1), 2) if n > k else None

    w = px.tail(60)
    out = {
        "last_date": str(px["date"].iloc[-1]), "close": round(float(c[-1]), 2),
        "ret_1d_pct": ret(1), "ret_5d_pct": ret(5), "ret_20d_pct": ret(20), "ret_60d_pct": ret(60),
        "vol_vs_20d_avg": round(float(v[-1] / max(v[-21:-1].mean(), 1)), 2) if n > 21 else None,
        "turnover_pct_last": round(100 * float(px["turnover"].iloc[-1]), 2) if "turnover" in px and np.isfinite(px["turnover"].iloc[-1]) else None,
        "turnover_pct_20d_avg": round(100 * float(px["turnover"].tail(20).mean()), 2) if "turnover" in px else None,
        "dist_52w_high_pct": round(100 * (c[-1] / h[-250:].max() - 1), 1),
        "dist_52w_low_pct": round(100 * (c[-1] / lo[-250:].min() - 1), 1),
        "limit_up_closes_20d": int(lu[-20:].sum()), "limit_up_streak": int(streak),
        "max_drawdown_20d_pct": round(100 * float((c[-20:] / np.maximum.accumulate(c[-20:]) - 1).min()), 1) if n >= 20 else None,
        "limit_pct": pct,
        "recent_20_bars": [{"d": str(r.date)[5:], "o": round(float(r.open), 2), "h": round(float(r.high), 2),
                            "l": round(float(r.low), 2), "c": round(float(r.close), 2),
                            "v_rel": round(float(r.volume / max(w["volume"].mean(), 1)), 2)}
                           for r in px.tail(20).itertuples()],
    }
    if "outstanding_share" in px and np.isfinite(px["outstanding_share"].iloc[-1]):
        out["float_mcap_yi"] = round(float(px["outstanding_share"].iloc[-1] * c[-1]) / 1e8, 1)
    return out


def _holders(code: str) -> list[dict]:
    p = DATA / "holders.parquet"
    if not p.exists():
        return []
    try:
        d = pd.read_parquet(p, filters=[("code", "==", code)])
        if d.empty:
            return []
        cols = [c for c in d.columns if c != "code"]
        return d.sort_values(cols[0]).tail(4)[cols].astype(str).to_dict("records")
    except Exception:  # noqa: BLE001
        return []


def _corp(code: str, asof: str, fetch: bool) -> dict:
    import corp_events as CE
    out = {"fetched": False}
    if fetch:
        try:
            since = (dt.date.fromisoformat(asof) - dt.timedelta(days=730)).isoformat()
            CE.refresh(code, since, asof)
            out["fetched"] = True
        except Exception as e:  # noqa: BLE001
            out["fetch_error"] = str(e)[:120]
    try:
        v, why = CE.verdict(code, asof)
        out.update({"verdict": v, "why": why})
        c = CE.load(code) or {}
        out["reduce_hits_90d"] = CE.reduce_hits(c, asof, 90)[:5] if c else []
        p = CE.pending_plan(c, asof, 730) if c else None
        out["pending_placement"] = {k: p[k] for k in ("start", "art") if k in p} if p else None
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)[:120]
    return out


def _regime() -> list[dict]:
    import regime as R
    rows = R.load(10)
    keep = ("date", "turnover_yi", "adv_share", "limit_up", "limit_down", "ret_median_pct", "base20")
    return [{k: r.get(k) for k in keep} for r in rows]


def gather(code: str, fetch_corp: bool, with_name: bool) -> dict:
    import predict_one as P
    model = P.predict(code, with_name=with_name)
    asof = model.get("score_date") or dt.date.today().isoformat()
    return {"code": code, "name": model.get("name", ""), "asof": asof,
            "model": {k: model.get(k) for k in ("ok", "error", "score", "p", "rank", "n_pool", "n_qualified",
                                                 "qualified", "streak", "board", "is_st", "perf_bin",
                                                 "perf_streak", "base", "history")},
            "kline": _k_facts(code), "holders": _holders(code),
            "corp_events": _corp(code, asof, fetch_corp), "market": _regime()}


def build_prompt(m: dict, web: bool) -> str:
    facts = json.dumps(m, ensure_ascii=False, default=str)
    web_txt = ("请先用 WebSearch 查这只票最近两周的新闻、公告、所属题材和板块动向（至少两次搜索，"
               "中文关键词用「代码 名称」和「名称 公告」），用 WebFetch 读最相关的 1~3 篇。" if web
               else "这次不联网，只按材料分析，新闻那一项留空。")
    return (
        f"分析 A 股 {m['code']} {m.get('name', '')}，判断它未来 20 个交易日的走势。\n"
        f"{web_txt}\n\n"
        "下面是本机算好的材料（JSON）。model 是我们自己的起涨预测模型对它的打分："
        "分数是 0~100 的全市场排名，≥97 分且当天前 10 才会上清单；perf_bin 是这一分数档在验证集上"
        "「次日开盘买入、20 个交易日内最高价涨超 50%」的历史命中率，base 是随便买一只的比例。"
        "kline 是近 60 根 K 线算出来的事实（前复权价），recent_20_bars 的 v_rel 是当日量相对近 60 日均量。"
        "corp_events 是减持 / 定增核查，market 是近 10 天的全市场环境。\n\n"
        f"{facts}\n\n"
        "要求：立场只能是看多 / 看空 / 观望之一；置信度 0~100；依据和风险各给 2~5 条，每条标明"
        "数据 / 新闻 / 推断并尽量给来源；news 列出你查到的最相关的几条（日期、标题、链接、影响）；"
        "model_vs_ai 说明你的判断和模型分数一致还是分歧、为什么；change_mind 写什么情况下你会改判；"
        "查不到或无法核实的放进 unverified。结论要具体，不要套话。"
    )


def fmt(code: str, m: dict, r: dict) -> str:
    L = [f"{code} {m.get('name', '')}  打分日 {m['asof']}  模型 {m['model'].get('score')} 分 / 第 {m['model'].get('rank')} 名",
         f"AI：{r['stance']}（置信 {r['confidence']}）  {r['horizon']}",
         f"一句话：{r['one_line']}", "依据："]
    L += [f"  - [{x.get('basis')}] {x['point']}" + (f"  <{x['source']}>" if x.get("source") else "")
          for x in r.get("reasons", [])]
    L.append("风险：")
    L += [f"  - [{x.get('basis')}] {x['point']}" + (f"  <{x['source']}>" if x.get("source") else "")
          for x in r.get("risks", [])]
    if r.get("news"):
        L.append("新闻：")
        L += [f"  - {x.get('date', '')} {x['title']}  {x.get('impact', '')}  {x.get('url', '')}" for x in r["news"]]
    L.append(f"和模型的关系：{r.get('model_vs_ai', '')}")
    L.append(f"什么情况下改判：{r.get('change_mind', '')}")
    if r.get("unverified"):
        L.append("没核实到：" + "；".join(r["unverified"]))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--no-corp", action="store_true", help="不现拉减持 / 定增，只用缓存")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    import importlib
    importlib.import_module("localenv")
    code = str(a.code).strip().zfill(6)
    if not (code.isdigit() and len(code) == 6):
        print("代码要是 6 位数字")
        return 1
    if not ai_usage.available():
        print("本机没有 claude 命令行，装好并登录后再用")
        return 1
    print(f"备材料 {code} …", flush=True)
    m = gather(code, fetch_corp=not a.no_corp, with_name=True)
    print(f"模型 {m['model'].get('score')} 分，第 {m['model'].get('rank')} 名；减持 / 定增：{m['corp_events'].get('verdict')}；"
          f"交给 Claude（Opus，{'联网' if not a.no_web else '不联网'}）…", flush=True)
    r, outer, err = ai_usage.run_claude(build_prompt(m, not a.no_web), SCHEMA, tools="" if a.no_web else "WebSearch,WebFetch",
                                        timeout=a.timeout, system=SYSTEM, kind="analyze", subject=code)
    usd = float((outer or {}).get("total_cost_usd") or 0)
    if err or not r:
        print(f"分析失败：{err or '没有结构化结果'}（本次 {usd:.2f} 美元）")
        if a.json:
            print(json.dumps({"ok": False, "code": code, "error": err, "usd": usd}, ensure_ascii=False))
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"ok": True, "code": code, "name": m.get("name", ""), "asof": m["asof"],
           "at": ai_usage.now_bj().isoformat(timespec="seconds"), "usd": round(usd, 3),
           "web": not a.no_web, "result": r, "model": m["model"]}
    (OUT / f"{code}_{m['asof']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
    print(fmt(code, m, r))
    t = ai_usage.totals()
    print(f"\n本次 {usd:.2f} 美元；今天 {t['today_usd']} 美元 / 本月 {t['month_usd']} 美元"
          + ("  <<< 超过警戒线" if t["warn"] else ""))
    if a.json:
        print(json.dumps(rec, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
