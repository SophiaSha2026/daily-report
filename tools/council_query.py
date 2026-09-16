"""
学习会诊的只读查询工具。LLM 视角进程通过 `Bash(python tools/council_query.py:*)`
调它看更多切片；它能做的事全在这里，**只读**，改不了任何东西。

    python tools/council_query.py help
    python tools/council_query.py evidence --date 2026-09-16 --key morning.summary
    python tools/council_query.py lists                       起涨清单逐份真值
    python tools/council_query.py picks --date 2026-09-07     某份清单逐只
    python tools/council_query.py stock --code 688137 [--from 2026-08-01]   日线
    python tools/council_query.py stock-lists --code 688137   这只票上过哪些清单、之后走势
    python tools/council_query.py morning-day --date 2026-09-16 [--top 30]  当日早盘全表（按分数）
    python tools/council_query.py morning-code --code 600000 --date 2026-09-16
    python tools/council_query.py verdicts                    闸门裁决记录
    python tools/council_query.py wf [--kind W5]              走向前成绩表
    python tools/council_query.py importance                  起涨模型特征重要性
    python tools/council_query.py feature-ic --name chip_conc90   起涨特征逐月 IC
    python tools/council_query.py regimes                     归因 day_regime 逐日
    python tools/council_query.py proposals                   提案台账
    python tools/council_query.py rules                       两条线的规则和常量（不用翻源码）

输出一律 JSON（UTF-8）。行数上限 200，免得把上下文撑爆。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

STATE = ROOT / "state"
LIMIT = 200


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _out(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _df_rows(df, limit: int = LIMIT) -> list[dict]:
    import pandas as pd
    df = df.head(limit)
    return json.loads(df.to_json(orient="records", force_ascii=False,
                                 date_format="iso")) if isinstance(df, pd.DataFrame) else []


def q_evidence(a) -> None:
    dirs = sorted((STATE / "council").glob("20*"))
    d = STATE / "council" / a.date if a.date else (dirs[-1] if dirs else None)
    if not d:
        return _out({"error": "没有证据包"})
    obj = _json(d / "evidence.json")
    for k in (a.key or "").split("."):
        if k:
            obj = obj.get(k, {}) if isinstance(obj, dict) else {}
    if isinstance(obj, list):
        obj = obj[:LIMIT]
    _out(obj)


def q_lists(a) -> None:
    t = _json(STATE / "breakout" / "truth.json")
    _out(t.get("lists", []))


def q_picks(a) -> None:
    t = _json(STATE / "breakout" / "truth.json")
    _out([p for p in t.get("picks", []) if p.get("date") == a.date][:LIMIT])


def q_stock(a) -> None:
    import pandas as pd
    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet")
    s = px[px["code"] == str(a.code).zfill(6)].sort_values("date")
    if a.date_from:
        s = s[s["date"].astype(str) >= a.date_from]
    _out(_df_rows(s.tail(LIMIT)))


def q_stock_lists(a) -> None:
    t = _json(STATE / "breakout" / "truth.json")
    code = str(a.code).zfill(6)
    _out([p for p in t.get("picks", []) if p.get("code") == code])


def q_morning_day(a) -> None:
    import cfg as C
    from learn import dataset, vscore
    c = C.load()
    df = dataset.build([a.date], c["learning"]["neutralize"])
    if df.empty:
        return _out({"error": f"{a.date} 没有快照或标签"})
    sc, rej = vscore.score_df(df, c)
    d = df.assign(score=sc, rejected=rej).sort_values("score", ascending=False)
    cols = [x for x in ["code", "name", "score", "rejected", "gap_pct", "auc_ratio",
                        "auc_amount", "slope", "monotonic", "dive", "pos_pct_60d",
                        "ma_bull", "breakout", "prev_limit_up", "board_height",
                        "sector", "sector_members", "r", "y", "ytil"] if x in d.columns]
    _out({"date": a.date, "pool": int(len(d)), "admitted": int((~d["rejected"]).sum()),
          "rows": _df_rows(d[~d["rejected"]][cols], a.top or 30)})


def q_morning_code(a) -> None:
    import cfg as C
    from learn import dataset, vscore
    c = C.load()
    df = dataset.build([a.date], c["learning"]["neutralize"])
    code = str(a.code).zfill(6)
    d = df[df["code"] == code]
    if d.empty:
        return _out({"error": "没有这只票"})
    sc, rej = vscore.score_df(d, c)
    parts = vscore.parts(vscore.prepare(d), c)
    row = json.loads(d.to_json(orient="records", force_ascii=False))[0]
    row["score"] = float(sc[0])
    row["rejected"] = bool(rej[0])
    row["parts"] = {k: float(v[0]) for k, v in parts.items()}
    _out(row)


def q_verdicts(a) -> None:
    p = STATE / "verdict_log.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    _out(rows[-50:])


def q_wf(a) -> None:
    g = _json(ROOT / "out_breakout" / "window_grid.json")
    rows = g.get("grid", [])
    if a.kind:
        rows = [r for r in rows if r.get("kind") == a.kind]
    _out({"days": g.get("days"), "base": g.get("base"), "rows": rows})


def q_importance(a) -> None:
    e = sorted((STATE / "council").glob("20*"))
    obj = _json(e[-1] / "evidence.json") if e else {}
    m = (obj.get("breakout") or {}).get("model") or {}
    _out({"importance": m.get("importance"), "by_group": m.get("importance_by_group"),
          "fit_date": m.get("fit_date")})


def q_feature_ic(a) -> None:
    # 生产模型旁边那份（带 fingerprint）。out_breakout 那份是 09-12 的实验产物，
    # 和现在入模的 51 列对不上（见 evidence.py 同处注释）。
    fs = _json(ROOT / "state" / "breakout" / "feature_select.json")
    tab = fs.get("ic_table") or {}
    hits = {k: v for k, v in tab.items() if a.name in k} if isinstance(tab, dict) else tab
    _out(hits)


def q_regimes(a) -> None:
    rows = []
    for p in sorted((STATE / "llm_eval").glob("*.json")):
        j = _json(p)
        rows.append({"date": p.stem, "day_regime": j.get("day_regime"),
                     "day_note": j.get("day_note"),
                     "causes": sorted({i.get("cause") for i in j.get("items", [])})})
    _out(rows)


def q_rules(a) -> None:
    """两条线的规则和常量，免得 LLM 去翻源码。"""
    import cfg as C
    c = C.load()
    sc, out_ = c.get("screen", {}), c.get("output", {})
    rules = {
        "morning": {
            "admission": {"gap_pct": [sc.get("gap_pct_min"), sc.get("gap_pct_max")],
                          "auc_ratio": [sc.get("auc_ratio_min"), sc.get("auc_ratio_max")],
                          "liangbi_equiv": [round(240 * sc.get("auc_ratio_min", 0), 1),
                                            round(240 * sc.get("auc_ratio_max", 0), 1)],
                          "min_auc_amount_wan": sc.get("min_auc_amount_wan"),
                          "hard_rejects_in_order": ["公告黑名单", "一字板", "价格<=0", "涨幅区间外",
                                                    "量比区间外", "竞价额不足", "假涨停撤单",
                                                    "尾盘跳水", "斜率<=0(可选)"]},
            "weights": c.get("scoring", {}).get("weights"),
            "shape": {k: sc.get(k) for k in ("gap_pct_peak", "auc_ratio_score_hi", "auc_ratio_decay",
                                             "sector_min_members", "concept_prev_limitup_min",
                                             "pos_pct_60d_max_for_lowbase")},
            "penalties": c.get("scoring", {}).get("penalties"),
            "output": {k: out_.get(k) for k in ("top_n", "top_n_a", "top_n_b", "min_score", "merge_groups")},
            "learnable": sorted((c.get("learning", {}).get("box") or {}).keys()),
            "propose_only": c.get("learning", {}).get("propose_only"),
            "label": "开盘买(竞价成交价)收盘卖，y = r - 当日中位数（缩尾），ytil = y / 当日 MAD",
            "learned_yaml_active": bool(C.diff()),
        },
        "breakout": {},
    }
    try:
        sys.path.insert(0, str(ROOT / "src" / "breakout"))
        import daily as D
        import label as L
        rules["breakout"] = {
            "list_a": f"score >= {D.SCORE_MIN} 且当天前 {D.CAP_A} 名（按预测值）；连续按上榜天数排序",
            "score": "训练集再平衡样本上的预测值分位数 0~100；预测值先乘板块系数 BOARD_ADJ",
            "board_adj": D.BOARD_ADJ,
            "label": f"之后 {L.UP_WINDOW} 根自身 K 线最高价相对收盘涨幅 > {L.UP_THRESHOLD:.0%}",
            "list_b": f"进 A 池满 {D.MIN_HOLD_DAYS} 个交易日、涨过 {D.RISE_MIN:.0%}、从高点回落 8%~20% 且高点在最近 10 天",
            "pool_days": D.POOL_DAYS, "model_refit_days": D.MODEL_MAX_AGE,
            "train_end_gap_days": D.TRAIN_END_GAP,
            "risk_filter": f"ST / 30 天内减持公告 / 30 天内解禁占流通 >= {D.RELEASE_MIN_SHARE:.0%} / 60 天内增发",
            "overrides": D.load_overrides(),
            "drop_features": D.DROP_FEATURES,
        }
    except Exception as e:  # noqa: BLE001
        rules["breakout"] = {"error": str(e)}
    _out(rules)


def q_proposals(a) -> None:
    p = STATE / "council" / "proposals.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    _out(rows[-LIMIT:])


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?", default="help")
    ap.add_argument("--date", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--code", default="")
    ap.add_argument("--from", dest="date_from", default="")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--kind", default="")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    table = {"evidence": q_evidence, "lists": q_lists, "picks": q_picks,
             "stock": q_stock, "stock-lists": q_stock_lists,
             "morning-day": q_morning_day, "morning-code": q_morning_code,
             "verdicts": q_verdicts, "wf": q_wf, "importance": q_importance,
             "feature-ic": q_feature_ic, "regimes": q_regimes,
             "proposals": q_proposals, "rules": q_rules}
    if a.cmd not in table:
        print(__doc__)
        return 0
    try:
        table[a.cmd](a)
    except Exception as e:  # noqa: BLE001
        _out({"error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
