"""「突破平台」这个标志到底带来了什么：先出证据，不改代码。

用户 2026-09-16 定：先出证据再决定要不要消融（会诊提案 20260916-75d86e）。

三件事分开算，因为 breakout 在打分里有**两个**通道，提案只提到了第一个：
  通道 1：f_position 的 A 组里 +0.5 分（提案说的「加分」）
  通道 2：score.py:205 / vscore.py:137 的 A/B 组划分
          is_a = 昨涨停 | 连板 | breakout。把加分置 0 不会动这一条，
          只靠 breakout 进 A 组的票仍然按 A 组打分（低位那组是另一套曲线）。

反事实两个臂都**复用 vscore 的函数**算，不另写一份打分器（教训 34）：
  掐加分：delta = 100 * w_position * (f_position(原) − f_position(breakout=False))，
          扣分项与 breakout 无关，所以 新分 = max(0, 原分 − delta)，这是恒等式不是近似。
  连组划分一起去掉：直接把 d["breakout"] 全置 False 再调 vscore.score。

输出写 out_learn/platform_evidence.json。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\home\daily-report")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "learn")]

import cfg  # noqa: E402
from learn import vscore  # noqa: E402
from learn.optimize import production_order  # noqa: E402

C = cfg.load()
TOP_K = int(C["learning"]["objective"]["top_k"])


def day_se(per_day) -> tuple[float, float, int]:
    """按天聚类：调用方给的是每天一个数，这里对天求均值和标准误（教训 33）。"""
    a = np.asarray([x for x in per_day if x == x], float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(a[0]), float("nan"), 1
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(n)), n


def score_no_bonus(d: dict, c: dict) -> np.ndarray:
    """掐掉 f_position 里 breakout 的 +0.5，组划分保持原样。"""
    sc = c["screen"]
    gb = vscore.assign_group_b(d, sc)
    zero = np.zeros(len(d["breakout"]), bool)
    f_old = vscore.f_position(d["pos_pct_60d"], d["ma_bull"], d["breakout"], gb)
    f_new = vscore.f_position(d["pos_pct_60d"], d["ma_bull"], zero, gb)
    delta = 100.0 * float(c["scoring"]["weights"]["position"]) * (f_old - f_new)
    return np.maximum(0.0, vscore.score(d, c) - delta)


def score_no_flag(d: dict, c: dict) -> np.ndarray:
    """连 A/B 组划分里的 breakout 一起去掉。"""
    d2 = dict(d)
    d2["breakout"] = np.zeros(len(d["breakout"]), bool)
    return vscore.score(d2, c)


def analyse(df: pd.DataFrame, tag: str, rcol: str) -> dict:
    rows, cf_rows, rows_pool = [], [], []
    for day, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        d = vscore.prepare(g)
        sc = vscore.score(d, C)
        rej = vscore.hard_reject(d, C["screen"])
        ok = g[~rej]
        if ok.empty:
            continue
        sel = production_order(sc, rej, C, TOP_K)
        top = g.iloc[sel]
        if top.empty:
            continue
        pool_mean = float(ok[rcol].mean())
        # 过准入口径：会诊那条提案分的是这一组（17 天 87/364），不是前 10。
        # 这个标志只在打分里起作用、不决定谁进池，所以前 10 才是它真正影响的集合；
        # 两个口径都算出来并排放，才看得出会诊那个 2.5 SE 是在哪一层得到的。
        pb = ok["breakout"].to_numpy(bool)
        rows_pool.append({
            "date": str(day),
            "n_bo": int(pb.sum()), "n_no": int((~pb).sum()),
            "exc_bo": 100 * (float(ok[rcol][pb].mean()) - pool_mean)
            if pb.any() else float("nan"),
            "exc_no": 100 * (float(ok[rcol][~pb].mean()) - pool_mean)
            if (~pb).any() else float("nan"),
        })
        bo = top["breakout"].to_numpy(bool)
        only_bo = bo & (~top["prev_limit_up"].to_numpy(bool)) \
            & (top["board_height"].to_numpy(float) < 1)
        rows.append({
            "date": str(day),
            "admitted": int(len(ok)), "picks": int(len(top)),
            "n_bo": int(bo.sum()), "n_only_bo": int(only_bo.sum()),
            "exc_bo": 100 * (float(top[rcol][bo].mean()) - pool_mean)
            if bo.any() else float("nan"),
            "exc_no": 100 * (float(top[rcol][~bo].mean()) - pool_mean)
            if (~bo).any() else float("nan"),
            "exc_only_bo": 100 * (float(top[rcol][only_bo].mean()) - pool_mean)
            if only_bo.any() else float("nan"),
            "exc_top": 100 * (float(top[rcol].mean()) - pool_mean),
        })
        for key, fn in (("cf_bonus", score_no_bonus), ("cf_both", score_no_flag)):
            s2 = fn(d, C)
            sel2 = production_order(s2, rej, C, TOP_K)
            t2 = g.iloc[sel2]
            cf_rows.append({
                "date": str(day), "arm": key, "picks": int(len(t2)),
                "exc_top": 100 * (float(t2[rcol].mean()) - pool_mean)
                if len(t2) else float("nan"),
                "changed": int(len(set(top["code"]) ^ set(t2["code"]))),
            })

    R = pd.DataFrame(rows)
    if R.empty:
        return {"tag": tag, "days": 0}
    CF = pd.DataFrame(cf_rows)

    def pack(col):
        m, se, n = day_se(R[col])
        return {"mean_pct": round(m, 3) if m == m else None,
                "se_day_clustered": round(se, 3) if se == se else None,
                "n_days": n}

    out = {
        "tag": tag, "days": int(len(R)),
        "picks_total": int(R["picks"].sum()),
        "picks_breakout": int(R["n_bo"].sum()),
        "picks_only_breakout": int(R["n_only_bo"].sum()),
        "share_breakout_pct": round(100 * R["n_bo"].sum() / max(1, R["picks"].sum()), 1),
        "excess_vs_pool": {"breakout_yes": pack("exc_bo"),
                           "breakout_no": pack("exc_no"),
                           "only_breakout": pack("exc_only_bo"),
                           "all_picks": pack("exc_top")},
    }
    P = pd.DataFrame(rows_pool)
    mp, sp, npd = day_se(P["exc_bo"] - P["exc_no"])
    out["admitted_pool_split"] = {
        "note": "会诊提案 20260916-75d86e 用的就是这个口径",
        "n_yes": int(P["n_bo"].sum()), "n_no": int(P["n_no"].sum()),
        "yes": {k: v for k, v in zip(("mean_pct", "se_day_clustered", "n_days"),
                                     [round(x, 3) if x == x else None
                                      for x in day_se(P["exc_bo"])])},
        "no": {k: v for k, v in zip(("mean_pct", "se_day_clustered", "n_days"),
                                    [round(x, 3) if x == x else None
                                     for x in day_se(P["exc_no"])])},
        "diff": {"mean_pct": round(mp, 3) if mp == mp else None,
                 "se_day_clustered": round(sp, 3) if sp == sp else None,
                 "n_days": npd,
                 "t": round(mp / sp, 2) if (sp == sp and sp > 0) else None}}
    m, se, n = day_se(R["exc_bo"] - R["exc_no"])
    out["diff_yes_minus_no"] = {
        "mean_pct": round(m, 3) if m == m else None,
        "se_day_clustered": round(se, 3) if se == se else None,
        "n_days": n, "t": round(m / se, 2) if (se == se and se > 0) else None}
    for arm in ("cf_bonus", "cf_both"):
        j = R.merge(CF[CF["arm"] == arm], on="date", suffixes=("", "_cf"))
        m2, se2, n2 = day_se(j["exc_top_cf"] - j["exc_top"])
        out[arm] = {
            "delta_top_excess_pct": round(m2, 3) if m2 == m2 else None,
            "se_day_clustered": round(se2, 3) if se2 == se2 else None,
            "n_days": n2, "t": round(m2 / se2, 2) if (se2 == se2 and se2 > 0) else None,
            "days_list_changed": int((j["changed"] > 0).sum()),
            "avg_names_changed": round(float(j["changed"].mean()), 2)}
    return out


def main() -> int:
    res = {}
    f = sorted((ROOT / "data" / "train").glob("backfill_*.parquet"))[-1]
    bf = pd.read_parquet(f)
    bf["date"] = bf["date"].astype(str)
    res["backfill"] = analyse(bf, f"回填 {f.name}", "r")
    res["backfill"]["flag_rate_pool_pct"] = round(100 * float(bf["breakout"].mean()), 2)

    from learn import dataset
    on = dataset.build(None, C["learning"]["neutralize"])
    if not on.empty:
        on["date"] = on["date"].astype(str)
        res["online"] = analyse(on, "在线真值日", "y")
        res["online"]["flag_rate_pool_pct"] = round(100 * float(on["breakout"].mean()), 2)
    else:
        res["online"] = {"days": 0}

    res["note"] = (
        "excess_vs_pool = 前 10 里该组的当日收益均值 − 当天过准入全部的均值，"
        "按天聚类。cf_bonus = 只掐 f_position 的 +0.5；cf_both = 连 A/B 组划分里的 "
        "breakout 一起去掉。delta 为正 = 掐掉之后前 10 的超额更高。"
        "两个反事实都只在这个脚本里算，没有改 score.py / vscore.py。")
    p = ROOT / "out_learn" / "platform_evidence.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
