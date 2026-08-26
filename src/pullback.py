"""
「放量启动 -> 缩量回调 -> 再放量」形态扫描。每交易日 17:00 BJT 收盘后跑。

三段式条件（用户 2026-08-26 给定）
---------------------------------
  启动日 S    涨停或涨幅 >= 5%；成交量 >= 前一交易日 1.5 倍；换手 5%~10%
  调整期      S+1 .. T-1，长度 1~6 个交易日；缩量；期间最低价 >= S 日最低价
  执行日 T    涨幅 >= 5%；成交量 >= 前一交易日 1.5 倍；换手 5%~10%

「涨停或涨幅 >= 5%」只判后者。非 ST 票的涨停幅度是 10/20/30%，全都 >= 5%，
涨停是「涨幅 >= 5%」的真子集；ST 本来就排除在外。

「量能」一律指**成交量（手）**，不是成交额。成交额受价格影响，同样的换手在
涨停日会显得更大，用它判「放大 1.5 倍」会系统性偏松。

换手率从哪来
------------
腾讯批量行情的 index 38 是换手率，今天的直接读。历史日线两条路里只有东财带
换手率，腾讯 K 线不带，所以统一用今天反推：

    历史换手率 = 今日换手率 x (历史成交量 / 今日成交量)

流通股本在 7 个交易日内基本不变，这个恒等式成立。期间有过增发或大额解禁的
票会失真，属于已知误差，不为它单独去拉股本表。东财那路真带换手率时优先用真值。

为什么先用批量行情粗筛
----------------------
全市场 5548 只逐个拉日线要十几分钟。执行日三个条件里的涨幅和换手率，一次
批量行情就能算（1600 只 3.6 秒），先把池子砍到几十只，再对幸存者拉日线。

和竞价流水线的关系
------------------
完全独立：独立的 workflow、独立的 out_pullback/ 目录、独立的数据目录，
共用的只有 datasource / mailer 的底层和 GitHub Pages 站点。竞价那条线出问题
不会牵连这条，反之亦然。
"""
from __future__ import annotations

import os
import sys
import json
import math
import logging
import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import datasource as ds                                    # noqa: E402
from datasource import _turnover as turnover_pct           # noqa: E402
from mailer import send_alert                              # noqa: E402
from pullback_export import write_panel, write_blocks, send_pullback  # noqa: E402

OUT = ROOT / "out_pullback"
BJ = ZoneInfo("Asia/Shanghai")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pullback")


def now_bj() -> dt.datetime:
    return dt.datetime.now(BJ)


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def target(hms: str) -> dt.datetime:
    h, m, s = (int(x) for x in hms.split(":"))
    return now_bj().replace(hour=h, minute=m, second=s, microsecond=0)


def sleep_until(hms: str, label: str) -> None:
    """早到就等。cron 不会提前触发，但手动 dispatch 会——盘中跑一遍扫到的是
    实时数据，涨幅和量都还没定型，出来的榜是假的。宁可空转等到收盘。"""
    import time
    d = (target(hms) - now_bj()).total_seconds()
    if d <= 0:
        return
    log.info("等待 %s (%s)，%.0fs", label, hms, d)
    while (rem := (target(hms) - now_bj()).total_seconds()) > 0:
        time.sleep(min(rem, 20))


# ---------------------------------------------------------------------
#  日线归一化：东财和腾讯两条路的列不一样
# ---------------------------------------------------------------------
_NEED = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]


def norm_hist(h: pd.DataFrame) -> pd.DataFrame:
    """统一成 _NEED（+ 可选换手率），按日期升序，日期为 YYYY-MM-DD 字符串。"""
    if h is None or not len(h):
        return pd.DataFrame(columns=_NEED)
    miss = [c for c in _NEED if c not in h.columns]
    if miss:
        return pd.DataFrame(columns=_NEED)
    cols = _NEED + (["换手率"] if "换手率" in h.columns else [])
    d = h[cols].copy()
    d["日期"] = d["日期"].astype(str).str.slice(0, 10)
    for c in cols[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["收盘", "成交量"]).sort_values("日期").reset_index(drop=True)


def hist_turnover(row: pd.Series, today_to: float, today_vol: float) -> float:
    """历史某日换手率。东财带真值就用真值，否则按今日等比反推。"""
    v = row.get("换手率")
    if v is not None and not pd.isna(v) and float(v) > 0:
        return float(v)
    if today_vol <= 0:
        return 0.0
    return today_to * float(row["成交量"]) / today_vol


# ---------------------------------------------------------------------
#  形态判定
# ---------------------------------------------------------------------
def find_pattern(h: pd.DataFrame, today_to: float, today_vol: float,
                 pb: dict) -> dict | None:
    """
    在 h（不含今日，升序）里找启动日 + 校验调整期。

    从最近的候选启动日往前找，取**第一个**整段都成立的。取最近而不是取最强，
    因为形态讲的是「上一次启动之后的这一段回调」，中间再插一个更早的启动日
    会让「调整期」跨过一次新的放量，那就不是同一段结构了。
    """
    lc, ac = pb["launch"], pb["adjust"]
    n = len(h)
    lo_i = max(1, n - ac["max_days"] - 1)          # 调整最多 max_days 天
    hi_i = n - ac["min_days"] - 1                  # 调整至少 min_days 天
    for i in range(hi_i, lo_i - 1, -1):
        s, prev = h.iloc[i], h.iloc[i - 1]
        if float(s["涨跌幅"]) < lc["gain_pct_min"]:
            continue
        if float(prev["成交量"]) <= 0:
            continue
        s_vr = float(s["成交量"]) / float(prev["成交量"])
        if s_vr < lc["vol_ratio_min"]:
            continue
        s_to = hist_turnover(s, today_to, today_vol)
        if not (lc["turnover_min"] <= s_to <= lc["turnover_max"]):
            continue

        adj = h.iloc[i + 1:]                        # 调整期 S+1 .. T-1
        if not len(adj):
            continue
        s_vol = float(s["成交量"])
        if float(adj["成交量"].max()) >= s_vol * ac["vol_day_max_ratio"]:
            continue                                # 期间有一天没缩量
        vm = float(adj["成交量"].mean()) / s_vol
        if vm > ac["vol_mean_max_ratio"]:
            continue
        a_low = float(adj["最低"].min())
        if a_low < float(s["最低"]) * ac["low_floor_ratio"]:
            continue                                # 跌破启动日最低价

        return {
            "launch_date": str(s["日期"]),
            "launch_gain": round(float(s["涨跌幅"]), 2),
            "launch_vol_ratio": round(s_vr, 2),
            "launch_turnover": round(s_to, 2),
            "launch_low": round(float(s["最低"]), 2),
            "launch_high": round(float(s["最高"]), 2),
            "launch_close": round(float(s["收盘"]), 2),
            "adjust_days": int(n - 1 - i),
            "adjust_vol_mean_ratio": round(vm, 3),
            "adjust_low": round(a_low, 2),
            "adjust_drawdown_pct": round(
                (a_low / float(s["收盘"]) - 1) * 100, 2),
        }
    return None


# ---------------------------------------------------------------------
#  打分（纯确定性，LLM 不参与，和竞价那条线同一条硬约束）
# ---------------------------------------------------------------------
def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _strength(gain: float, vr: float, gain_min: float, limit: float,
              vr_min: float, vr_full: float) -> float:
    """涨幅和量比各占一半。量比走对数刻度：vr_min -> 0，vr_full -> 1。"""
    g = _clamp((gain - gain_min) / max(0.1, limit - gain_min))
    v = _clamp(math.log(max(vr, vr_min) / vr_min) / math.log(vr_full / vr_min))
    return 0.5 * g + 0.5 * v


def score_one(r: dict, pb: dict) -> dict:
    w, sc = pb["weights"], pb["scoring"]
    limit = ds.limit_pct(r["code"], r["name"])

    trig = _strength(r["gain_pct"], r["vol_ratio"], pb["trigger"]["gain_pct_min"],
                     limit, pb["trigger"]["vol_ratio_min"], sc["vol_ratio_full"])
    lau = _strength(r["launch_gain"], r["launch_vol_ratio"],
                    pb["launch"]["gain_pct_min"], limit,
                    pb["launch"]["vol_ratio_min"], sc["vol_ratio_full"])
    # 缩得越狠越好：均量比等于上限时 0 分，等于 0 时满分
    mx = pb["adjust"]["vol_mean_max_ratio"]
    shrink = _clamp((mx - r["adjust_vol_mean_ratio"]) / mx)
    # 回调最低点在启动日 K 线里的相对位置，越靠上说明抛压越轻
    span = max(0.01, r["launch_high"] - r["launch_low"])
    hold = _clamp((r["adjust_low"] - r["launch_low"]) / span)
    # 调整越短，趋势被破坏的可能越小
    md = pb["adjust"]["max_days"]
    speed = _clamp((md + 1 - r["adjust_days"]) / md)

    parts = {"trigger": trig, "launch": lau, "shrink": shrink,
             "hold": hold, "speed": speed}
    raw = 100.0 * sum(w[k] * v for k, v in parts.items())
    return {**r, "score": round(raw, 1),
            "parts": {k: round(v, 3) for k, v in parts.items()}}


# ---------------------------------------------------------------------
def scan(c: dict) -> tuple[list[dict], dict]:
    pb = c["pullback"]
    tg = pb["trigger"]
    stat = {}

    codes = ds.load_code_list()
    q = ds.fetch_quotes([ds.to_symbol(x) for x in codes])
    stat["quotes"] = len(q)
    log.info("全市场快照 %d 只", len(q))

    cand: dict[str, dict] = {}
    for v in q.values():
        nm = (v.name or "").upper()
        if not v.prev_close or not v.price:
            continue
        if "ST" in nm or "退" in nm:
            continue
        to = turnover_pct(v)
        if v.chg_pct < tg["gain_pct_min"]:
            continue
        if not (tg["turnover_min"] <= to <= tg["turnover_max"]):
            continue
        cand[v.code] = {"code": v.code, "name": v.name,
                        "close": round(v.price, 2),
                        "gain_pct": round(v.chg_pct, 2),
                        "turnover": round(to, 2),
                        "vol_hand": float(v.volume_hand),
                        "amount_wan": round(v.amount_wan, 1)}
    stat["after_today"] = len(cand)
    log.info("今日涨幅/换手初筛通过 %d 只", len(cand))
    if not cand:
        return [], stat

    today = now_bj().strftime("%Y-%m-%d")
    # daily_hist 内部要求至少 25 根 K 线，窗口给足 90 个自然日
    start = (now_bj() - dt.timedelta(days=pb["lookback_days"])).strftime("%Y-%m-%d")
    hs = ds.daily_hist_many(list(cand), start, today, workers=4)
    stat["hist_ok"] = sum(1 for v in hs.values() if v is not None and len(v))
    log.info("日线拉取 %d/%d 成功，来源 %s", stat["hist_ok"], len(cand),
             ds.hist_source_stats())

    rows: list[dict] = []
    for code, r in cand.items():
        h = norm_hist(hs.get(code))
        if len(h) < 8:
            continue
        # 今日那根一律以行情快照为准，日线里若已有今日先剔掉，避免两个源混用
        h = h[h["日期"] < today].reset_index(drop=True)
        if len(h) < pb["adjust"]["max_days"] + 2:
            continue

        y_vol = float(h.iloc[-1]["成交量"])
        if y_vol <= 0:
            continue
        vr = r["vol_hand"] / y_vol
        if vr < tg["vol_ratio_min"]:
            continue

        pat = find_pattern(h, r["turnover"], r["vol_hand"], pb)
        if not pat:
            continue

        row = {**r, **pat, "vol_ratio": round(vr, 2),
               "prev_vol_hand": y_vol,
               "break_launch_high": bool(r["close"] > pat["launch_high"])}
        rows.append(score_one(row, pb))

    stat["matched"] = len(rows)
    rows.sort(key=lambda x: -x["score"])
    log.info("形态匹配 %d 只", len(rows))
    return rows, stat


# ---------------------------------------------------------------------
def write_prompt(sel: list[dict], date: str) -> None:
    brief = [{k: r[k] for k in
              ("code", "name", "close", "gain_pct", "turnover", "vol_ratio",
               "launch_date", "launch_gain", "adjust_days",
               "adjust_drawdown_pct", "break_launch_high", "score")}
             for r in sel]
    OUT.mkdir(exist_ok=True)
    (OUT / "brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = (ROOT / "prompts" / "pullback_analyst.md").read_text(encoding="utf-8")
    (OUT / "prompt.md").write_text(f"# 日期：{date}\n\n{tpl}", encoding="utf-8")
    (OUT / "commentary.json").write_text("{}", encoding="utf-8")


def stage_scan(c: dict) -> int:
    today = now_bj().strftime("%Y-%m-%d")
    try:
        if today not in ds.trade_dates():
            log.info("%s 非交易日", today)
            return 0
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历失败(%s)，按工作日兜底", e)
        if now_bj().weekday() >= 5:
            return 0

    pb = c["pullback"]
    sleep_until(pb["run_at"], "收盘后扫描")
    if now_bj() > now_bj().replace(
            hour=int(pb["hard_deadline"][:2]), minute=int(pb["hard_deadline"][3:5]),
            second=0, microsecond=0):
        send_alert(f"{today} 形态扫描启动过晚（超过 {pb['hard_deadline']}），本次放弃。")
        return 0

    rows, stat = scan(c)
    OUT.mkdir(exist_ok=True)
    if rows:
        d = ROOT / "data" / today[:7]
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / f"pullback_{today}.parquet", index=False)
        pd.DataFrame(rows).to_csv(OUT / "detail.csv", index=False,
                                  encoding="utf-8-sig")
    else:
        # 空榜也要落一个空文件，否则幂等检查以为今天没跑过
        d = ROOT / "data" / today[:7]
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["code"]).to_parquet(
            d / f"pullback_{today}.parquet", index=False)

    sel = rows[: c["pullback"]["output"]["top_n"]]
    (OUT / "selected.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "run_meta.json").write_text(
        json.dumps({"date": today, "n": len(sel), "matched": len(rows), **stat},
                   ensure_ascii=False), encoding="utf-8")
    write_prompt(sel, today)
    log.info("入选 %d 只，等待 Claude 分析", len(sel))
    return 0


def stage_send(c: dict) -> int:
    today = now_bj().strftime("%Y-%m-%d")
    f = OUT / "selected.json"
    if not f.exists():
        log.info("无 selected.json（今日未运行或非交易日），跳过")
        return 0
    # out_pullback/ 是提交进仓库的，checkout 会带着上一交易日的产物。
    # 没有今天的日期戳就说明本次扫描没跑完，绝不能拿旧清单发信。
    try:
        stamp = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stamp = {}
    if stamp.get("date") != today:
        log.warning("out_pullback/ 里是 %s 的产物（今天 %s），本次不发信",
                    stamp.get("date", "未知"), today)
        return 0

    sel = json.loads(f.read_text(encoding="utf-8"))
    texts, notice = {}, ""
    try:
        texts = json.loads((OUT / "commentary.json").read_text(encoding="utf-8"))
        if not isinstance(texts, dict):
            raise ValueError("格式错误")
    except Exception as e:  # noqa: BLE001
        log.warning("Claude 输出不可用: %s", e)
    if not texts:
        notice = "本次无 LLM 分析（模型调用失败或额度耗尽），以下为纯量化结果"
        log.warning(notice)

    blocks = write_blocks(sel, OUT, today) if sel else []
    write_panel(sel, texts, OUT, today, notice, stamp)

    owner = os.environ.get("GH_OWNER", "")
    repo = os.environ.get("GH_REPO", "")
    page = f"https://{owner.lower()}.github.io/{repo}/pullback.html" if owner else ""

    att = list(blocks)
    if (OUT / "detail.csv").exists():
        att.append(OUT / "detail.csv")

    send_pullback(today, sel, texts, notice=notice, attachments=att,
                  page_url=page, stat=stamp)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "send"], required=True)
    a = ap.parse_args()
    c = cfg()
    return stage_scan(c) if a.stage == "scan" else stage_send(c)


if __name__ == "__main__":
    sys.exit(main())
