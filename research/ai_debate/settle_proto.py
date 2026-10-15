"""第 3 件：AI 判断的向前结算 + 输入哈希缓存（研究原型，不落生产）。

设计要点（详细在 report.md 第 3 节）：
  ledger   每次 AI 分析 / 讨论 / 辩论给出立场，追加一行到账本（生产时 state/ai_judgments.jsonl）。
           主键 (code, asof, source)：同一只票同一材料日同一来源只记第一次，免得用户反复点把样本刷多。
  settle   asof 之后满 20 根 K 线（buyable.open_labels 里 y_open 不是 NaN）就结算，写回 y_open / r20_open。
           只认 asof 之后才落账的判断（at > asof 收盘），回头补的历史日期一律 replay=True、不进成绩。
  stats    按立场分组；满 30 个结算样本才给命中率，不满写「仅供观察」；按天聚类标准误（教训 33）；
           对照组 = 同一天模型清单 A 的命中率（同期同口径），不和全市场比。
  cache    键 = sha256(模型, 推理等级, 系统提示, schema, 工具, 提示词)。提示词里不许有时分秒
           （ai_chat.first_prompt 现在写了「今天北京时间 %H:%M」，生产化时要改成只到日期，否则永远不命中）。
           命中只省钱，不改账本：同一 (code, asof, source) 的判断本来就只记一次。

    python research/ai_debate/settle_proto.py --selftest     离线自测（临时目录，假判断 + 真标签）
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

MIN_N = 30
STANCES = ("看多", "看空", "观望", "待复核")


# ---------------------------------------------------------------
#  缓存
# ---------------------------------------------------------------
def cache_key(*, model: str, effort: str, system: str, schema: dict | None, tools: str, prompt: str) -> str:
    blob = json.dumps({"m": model, "e": effort or "", "s": system, "j": schema, "t": tools, "p": prompt},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_get(cache_dir: Path, key: str) -> dict | None:
    p = cache_dir / key[:2] / f"{key}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def cache_put(cache_dir: Path, key: str, structured: dict, outer_meta: dict) -> None:
    p = cache_dir / key[:2] / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"result": structured, "meta": outer_meta}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def cached_run(run, cache_dir: Path, *, model, effort, system, schema, tools, prompt, **kw):
    """生产里包住 ai_usage.run_claude：命中就不调 CLI（记账里记一行 usd=0 cached=True），
    没命中且成功才写缓存（失败不缓存，否则一次网络抖动会被永久记住）。会话型（resume）不走缓存。"""
    key = cache_key(model=model, effort=effort, system=system, schema=schema, tools=tools, prompt=prompt)
    hit = cache_get(cache_dir, key)
    if hit is not None:
        return hit["result"], {"cached": True, "key": key, "total_cost_usd": 0.0}, ""
    r, outer, err = run(prompt, schema, model=model, effort=effort, system=system, tools=tools, **kw)
    if r and not err:
        cache_put(cache_dir, key, r, {"usd": (outer or {}).get("total_cost_usd"), "model": model})
    return r, outer, err


# ---------------------------------------------------------------
#  账本
# ---------------------------------------------------------------
def _rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()]


def record(ledger: Path, *, code: str, asof: str, source: str, stance: str, confidence: int | None,
           model: str, at: str, model_score: float | None = None, model_rank: int | None = None,
           in_list_a: bool = False, input_key: str = "") -> bool:
    """落一条判断。stance 不在四个取值里就记「待复核」。返回是否新增（重复的不记）。

    at 是落账时刻（北京时间 ISO）。落账晚于材料日的次日开盘（买点已过，或者是回头重放历史日期）
    记 replay=True，不进成绩；只有买点之前落账的才算向前。"""
    if stance not in STANCES:
        stance = "待复核"
    key = (code, asof, source)
    if any((r["code"], r["asof"], r["source"]) == key for r in _rows(ledger)):
        return False
    replay = at[:10] > _next_open_guess(asof)  # 材料日的次日开盘之后才落账 = 买点已过，不算向前
    row = {"code": code, "asof": asof, "source": source, "stance": stance, "confidence": confidence,
           "model": model, "at": at, "model_score": model_score, "model_rank": model_rank,
           "in_list_a": in_list_a, "input_key": input_key, "replay": replay,
           "y_open": None, "r20_open": None, "settled": False}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def _next_open_guess(asof: str) -> str:
    """简化：生产里用 datasource 的交易日历取 asof 的下一个交易日；这里原型按自然日 +3 天放宽
    （周末 / 假日也落在里面）。判据的意思是：用户在次日开盘前看到这条判断，它才有资格被结算。"""
    import datetime as dt
    return (dt.date.fromisoformat(asof) + dt.timedelta(days=3)).isoformat()


def settle(ledger: Path, labels) -> int:
    """labels = buyable.open_labels()（code, date, y_open, r20_open）。返回本次新结算条数。"""
    import numpy as np
    lab = labels.set_index(["code", "date"])[["y_open", "r20_open"]]
    rows = _rows(ledger)
    n = 0
    for r in rows:
        if r["settled"] or r["stance"] == "待复核":
            continue
        k = (r["code"], r["asof"])
        if k not in lab.index:
            continue
        y, r20 = lab.loc[k, "y_open"], lab.loc[k, "r20_open"]
        if not np.isfinite(y):
            continue          # 还没满 20 根
        r.update(y_open=int(y), r20_open=round(float(r20), 4), settled=True)
        n += 1
    if n:
        tmp = ledger.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(ledger)
    return n


def _day_clustered(rows: list[dict]) -> tuple[float, float, int]:
    """先按天求命中率，再对天求均值和标准误（教训 33：点估计和标准误来自同一条按天序列）。"""
    by: dict[str, list[int]] = {}
    for r in rows:
        by.setdefault(r["asof"], []).append(r["y_open"])
    s = [sum(v) / len(v) for v in by.values()]
    m = sum(s) / len(s)
    se = math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1) / len(s)) if len(s) > 1 else float("nan")
    return m, se, len(s)


def stats(ledger: Path, list_a_hit_by_day: dict[str, float] | None = None) -> dict:
    """控制台显示用。list_a_hit_by_day：同一天清单 A 的命中率（对照组）。"""
    rows = [r for r in _rows(ledger) if r["settled"] and not r["replay"]]
    out = {"n_settled": len(rows), "pending": sum(1 for r in _rows(ledger) if not r["settled"]),
           "replay_excluded": sum(1 for r in _rows(ledger) if r["replay"]), "groups": {}}
    for st in ("看多", "看空", "观望"):
        g = [r for r in rows if r["stance"] == st]
        if len(g) < MIN_N:
            out["groups"][st] = {"n": len(g), "show": f"仅供观察（{len(g)}/{MIN_N}）"}
            continue
        m, se, nd = _day_clustered(g)
        item = {"n": len(g), "days": nd, "hit": round(m, 4), "se_day": round(se, 4),
                "show": f"命中 {100 * m:.1f}% ± {100 * se:.1f}（{len(g)} 只 / {nd} 天）"}
        if list_a_hit_by_day:
            diffs = []
            by: dict[str, list[int]] = {}
            for r in g:
                by.setdefault(r["asof"], []).append(r["y_open"])
            for d, v in by.items():
                if d in list_a_hit_by_day:
                    diffs.append(sum(v) / len(v) - list_a_hit_by_day[d])
            if len(diffs) > 1:
                md = sum(diffs) / len(diffs)
                sd = math.sqrt(sum((x - md) ** 2 for x in diffs) / (len(diffs) - 1) / len(diffs))
                item["vs_list_a"] = {"diff": round(md, 4), "se": round(sd, 4), "days": len(diffs),
                                     "t": round(md / sd, 2) if sd > 0 else None}
        out["groups"][st] = item
    return out


def prior_note(ledger: Path, code: str, before: str) -> str:
    """下次分析同一只票时带进提示词的一句（TradingAgents v0.6.0 的「到期结算 + 反思」，这里只给事实不给反思）。"""
    rs = [r for r in _rows(ledger) if r["code"] == code and r["asof"] < before and r["settled"]]
    if not rs:
        return ""
    r = rs[-1]
    return (f"上次（{r['asof']}，{r['source']}）你的立场是{r['stance']}（置信 {r['confidence']}），"
            f"之后 20 根买得到口径最高涨幅 {100 * r['r20_open']:.1f}%，"
            f"{'达到' if r['y_open'] else '没达到'} 50%。")


# ---------------------------------------------------------------
def selftest() -> int:
    import pandas as pd
    import buyable
    bad = 0
    lab = buyable.open_labels()
    lab["date"] = lab["date"].astype(str)
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "j.jsonl"
        t = pd.read_parquet(ROOT / "data/breakout/raw/exp/w1_scale_avg/top_base.parquet")
        t["date"] = t["date"].astype(str)
        t = t[(t["rank"] <= 10) & (t["date"] >= "2025-01-01") & (t["date"] <= "2025-06-30")]
        # 假判断：按模型名次给立场（只为跑通流程，不是成绩）
        for r in t.itertuples():
            st = "看多" if r.rank <= 3 else "观望" if r.rank <= 6 else "看空"
            record(led, code=str(r.code).zfill(6), asof=r.date, source="analyze", stance=st, confidence=60,
                   model="fake", at=r.date + "T20:00:00+08:00", model_score=r.score, model_rank=r.rank,
                   in_list_a=True)
        n0 = len(_rows(led))
        # 重复记不增加
        r0 = _rows(led)[0]
        if record(led, code=r0["code"], asof=r0["asof"], source="analyze", stance="看多", confidence=1,
                  model="fake", at=r0["at"]):
            print("断言失败：重复判断被记了两次"); bad += 1
        # 非法立场 -> 待复核；回头重放 -> replay
        record(led, code="000001", asof="2025-03-03", source="chat", stance="买入", confidence=None,
               model="fake", at="2026-10-14T20:00:00+08:00")
        last = _rows(led)[-1]
        if last["stance"] != "待复核" or not last["replay"]:
            print("断言失败：非法立场没记成待复核，或回头重放没标 replay", last); bad += 1
        n = settle(led, lab)
        if n != n0:
            print(f"断言失败：结算 {n} 条，应为 {n0}"); bad += 1
        if settle(led, lab) != 0:
            print("断言失败：结算不是跑过就不再跑"); bad += 1
        # 结算值和 top_base 里的 y_open 一致
        m = {(str(r.code).zfill(6), r.date): r.y_open for r in t.itertuples()}
        mism = sum(1 for r in _rows(led) if r["settled"] and m.get((r["code"], r["asof"])) != r["y_open"])
        if mism:
            print(f"断言失败：结算和产物 y_open 不一致 {mism} 条"); bad += 1
        day_hit = t.groupby("date")["y_open"].mean().to_dict()
        s = stats(led, day_hit)
        print(json.dumps(s, ensure_ascii=False, indent=1))
        if s["groups"]["看多"]["n"] < MIN_N or "hit" not in s["groups"]["看多"]:
            print("断言失败：满 30 样本没显示命中率"); bad += 1
        # 不满 30 只显示仅供观察
        led2 = Path(td) / "k.jsonl"
        for r in list(t.itertuples())[:10]:
            record(led2, code=str(r.code).zfill(6), asof=r.date, source="analyze", stance="看多", confidence=50,
                   model="fake", at=r.date + "T20:00:00+08:00")
        settle(led2, lab)
        if "仅供观察" not in stats(led2)["groups"]["看多"]["show"]:
            print("断言失败：不满 30 也显示了命中率"); bad += 1
        print("prior_note 示例：", prior_note(led, r0["code"], "2026-01-01"))
        # 缓存键：时分秒进了提示词就永远不命中
        k1 = cache_key(model="m", effort="", system="s", schema=None, tools="", prompt="今天 2026-10-14 20:01")
        k2 = cache_key(model="m", effort="", system="s", schema=None, tools="", prompt="今天 2026-10-14 20:02")
        if k1 == k2:
            print("断言失败：缓存键没区分提示词"); bad += 1
        calls = []

        def fake_run(prompt, schema, **kw):
            calls.append(1)
            return {"stance": "看多"}, {"total_cost_usd": 1.5}, ""
        cd = Path(td) / "cache"
        for _ in range(3):
            cached_run(fake_run, cd, model="m", effort="", system="s", schema=None, tools="", prompt="同一份材料")
        if len(calls) != 1:
            print(f"断言失败：同样输入调了 {len(calls)} 次 CLI"); bad += 1

        def fail_run(prompt, schema, **kw):
            calls.append(1)
            return None, None, "网络错误"
        calls.clear()
        for _ in range(2):
            cached_run(fail_run, cd, model="m", effort="", system="s", schema=None, tools="", prompt="失败的")
        if len(calls) != 2:
            print("断言失败：失败结果被缓存了"); bad += 1
    print(f"断言失败 {bad} 个")
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    print(__doc__)
