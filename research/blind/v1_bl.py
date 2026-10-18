"""V1 盲测 + 逐票验证（criteria.txt 第 1 节）。

    PYTHONIOENCODING=utf-8 python research/blind/v1_bl.py [--score lgb_price]

看好 = s(code, T−20) 在 T−20 当天候选池内分位 >= q；走出来 = T 日 ret20 名次 <= K；出票 = 看好 ∩ 走出来 按 s(T) 或 ret20 降序每天前 n。
诊断：同名次档内按 s(T−20) 分位分组的精确率（两段，kind=diag）。自检：T−20 换成 T；日期逐行断言；hit 每天内打乱。
筛选集网格全部记 kind=screen；开发集全部配置先落盘 results_v1_dev.json 不打印；只读筛选集最好的一个（kind=arm）。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common_bl as B

QS = (0.80, 0.90, 0.95, 0.98)
RANK_BINS = [(1, 3), (4, 10), (11, 30), (31, 50)]


def group_of(pct: np.ndarray) -> np.ndarray:
    g = np.full(len(pct), "缺", dtype="U8")
    ok = np.isfinite(pct)
    g[ok & (pct >= 0.95)] = ">=.95"
    g[ok & (pct >= 0.80) & (pct < 0.95)] = ".80-.95"
    g[ok & (pct < 0.80)] = "<.80"
    return g


def diag_table(t: pd.DataFrame, pct: np.ndarray, title: str, L: list[str]) -> None:
    L.append("\n%s（精确率%% / 只数）" % title)
    L.append("%-8s %-8s | %-20s %-20s %-20s %-20s" % ("段", "名次档", ">=.95", ".80-.95", "<.80", "T-20不在池"))
    grp = group_of(pct)
    for seg in ("screen", "dev"):
        for lo, hi in RANK_BINS:
            m = (t["seg"].to_numpy() == seg) & (t["rank_ret20"].to_numpy() >= lo) & (t["rank_ret20"].to_numpy() <= hi)
            cells = []
            for gname in (">=.95", ".80-.95", "<.80", "缺"):
                mm = m & (grp == gname)
                n = int(mm.sum())
                p = 100 * t["hit"].to_numpy()[mm].mean() if n else np.nan
                cells.append("%5.1f%% (%6d)" % (p, n))
            L.append("%-8s %-8s | %s" % (seg, "%d-%d" % (lo, hi), "  ".join("%-20s" % c for c in cells)))
    # 按天配对的组间差（>=.95 对 <.80，在同一名次档、同一天都有两组时）
    L.append("按天配对差（>=.95 − <.80，同名次档同一天都有两组的天）：")
    for seg in ("screen", "dev"):
        for lo, hi in RANK_BINS:
            m = (t["seg"].to_numpy() == seg) & (t["rank_ret20"].to_numpy() >= lo) & (t["rank_ret20"].to_numpy() <= hi)
            d = pd.DataFrame({"date": t["date"].to_numpy()[m], "g": grp[m], "h": t["hit"].to_numpy()[m]})
            d = d[d["g"].isin([">=.95", "<.80"])]
            pv = d.pivot_table(index="date", columns="g", values="h", aggfunc="mean").dropna()
            if len(pv) < 30:
                L.append("  %-6s %-6s 配得上的天 %d（不足）" % (seg, "%d-%d" % (lo, hi), len(pv)))
                continue
            u = (pv[">=.95"] - pv["<.80"]).to_numpy()
            L.append("  %-6s %-6s 天 %3d  差 %+5.1fpp  NW t %+5.2f" % (seg, "%d-%d" % (lo, hi), len(pv), 100 * u.mean(), B.PR.nw_t(u)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="lgb_price")
    a = ap.parse_args()
    t0 = time.time()
    t = B.load_keys()
    s = B.load_scores(a.score).astype(float)
    n = len(t)
    assert len(s) == n
    ptr = B.lag_pointer(t, 20)
    has = ptr >= 0
    date = t["date"].to_numpy().astype("U10")
    nbar = t["nbar"].to_numpy(np.int64)
    # 自检 (ii)：T−20 那一行的日期 < T，根数差正好 20，分数月份 = 该行日期月份（walk_gt 按行日期的月份打分）
    assert (date[ptr[has]] < date[has]).all()
    assert (nbar[has] - nbar[ptr[has]] == 20).all()
    cfg = json.loads((B.G.SCORES / (a.score + ".json")).read_text(encoding="utf-8"))
    months_scored = {m["month"] for m in cfg["months"]}
    pct_T = B.day_pct(t, s)
    pct_lag = np.full(n, np.nan)
    pct_lag[has] = pct_T[ptr[has]]
    s_lag = np.full(n, np.nan)
    s_lag[has] = s[ptr[has]]
    in_eval = t["seg"].isin(["screen", "dev"]).to_numpy() & (t["rank_ret20"].to_numpy() <= 50)
    lag_months = set(pd.unique(pd.Series(date[ptr[has & in_eval]]).str[:7]))
    assert lag_months <= months_scored, lag_months - months_scored
    L = ["V1 盲测 + 逐票验证  分数 %s  表 %d 行  T−20 在池 %.1f%%（前 50 行 %.1f%%）  %.0fs" % (
        a.score, n, 100 * has.mean(), 100 * has[in_eval].mean(), time.time() - t0)]
    L.append("自检通过：T−20 行日期 < T 全部成立；根数差 == 20 全部成立；T−20 的分数月份 ⊆ 已打分月份 %s..%s" % (min(months_scored), max(months_scored)))
    B.say(L[-1])
    gap_days = (pd.to_datetime(date[has]) - pd.to_datetime(date[ptr[has]])).days
    L.append("T 与 T−20 行的日历间隔：中位 %d 天，90 分位 %d 天（>45 天的 %.2f%%，是停牌 / 缺根）" % (
        np.median(gap_days), np.quantile(gap_days, 0.9), 100 * (gap_days > 45).mean()))

    tt = t.copy()
    # 诊断：同名次档内按 s(T−20) 分位分组
    diag_table(tt, pct_lag, "诊断 A：按 s(T−20) 当天分位分组（盲测窗之前的模型看法）", L)
    B.ledger_add({"kind": "diag", "what": "V1 名次档 × s(T−20) 分位组 两段精确率", "score": a.score})
    diag_table(tt, pct_T, "诊断 B（对照）：按 s(T) 当天分位分组（模型看全部数据，= 模型重排）", L)
    B.ledger_add({"kind": "diag", "what": "V1 名次档 × s(T) 分位组 两段精确率（对照）", "score": a.score})
    # 再往前：s(T−40) 作安慰剂（看好的持续性）
    ptr40 = B.lag_pointer(t, 40)
    pct_lag40 = np.full(n, np.nan)
    h40 = ptr40 >= 0
    pct_lag40[h40] = pct_T[ptr40[h40]]
    diag_table(tt, pct_lag40, "诊断 C（安慰剂）：按 s(T−40) 当天分位分组", L)
    B.ledger_add({"kind": "diag", "what": "V1 名次档 × s(T−40) 分位组 两段精确率（安慰剂）", "score": a.score})

    # 筛选集网格
    rank = t["rank_ret20"].to_numpy()
    ret20 = t["ret20"].to_numpy(float)
    rows_screen, dev_store = [], {}
    L.append("\n筛选集网格（kind=screen）：K × q × n × 排序键")
    for K in B.KS:
        for q in QS:
            cand = (rank <= K) & np.isfinite(pct_lag) & (pct_lag >= q)
            for order, key in (("sT", s), ("ret20", ret20)):
                for nn in B.NS:
                    sel = B.topn_by(t, cand, key, nn)
                    name = "V1 K%d q%.2f n%d %s" % (K, q, nn, order)
                    r = B.eval_both(t, sel, boot=False)
                    rs = B.screen_row(name, r["screen"])
                    rows_screen.append(rs)
                    dev_store[name] = {"screen": rs, "dev": B.screen_row(name, r["dev"]),
                                       "comb": {k: v for k, v in r.get("comb", {}).items()} if "comb" in r else None}
                    B.ledger_add({"kind": "screen", **rs})
                    L.append(B.G.fmt(name, r["screen"], "screen"))
    (B.HERE / "results_v1_dev.json").write_text(json.dumps(dev_store, ensure_ascii=False, default=float, indent=0), encoding="utf-8")
    # 自检 (i)：T−20 换成 T
    L.append("\n自检 (i)：看好改用 s(T) 分位（同一网格里 K=50 的 20 个配置，筛选集）")
    for q in QS:
        cand = (rank <= 50) & (pct_T >= q)
        for nn in B.NS:
            sel = B.topn_by(t, cand, s, nn)
            r = B.eval_both(t, sel, boot=False)
            L.append(B.G.fmt("V1chk K50 q%.2f n%d sT" % (q, nn), r["screen"], "screen"))
    B.ledger_add({"kind": "shuffle", "what": "V1 自检 (i) T−20 换成 T，筛选集 20 个配置"})
    # 开发集：只读筛选集最好的一个（过量门槛）
    best = B.pick_best(rows_screen)
    L.append("\n筛选集最好的（过量门槛 V>=150 天>=60）：%s" % json.dumps(best, ensure_ascii=False))
    if best is not None:
        K = int(best["name"].split()[1][1:])
        q = float(best["name"].split()[2][1:])
        nn = int(best["name"].split()[3][1:])
        order = best["name"].split()[4]
        cand = (rank <= K) & np.isfinite(pct_lag) & (pct_lag >= q)
        sel = B.topn_by(t, cand, s if order == "sT" else ret20, nn)
        r = B.eval_both(t, sel, boot=True)
        L.append(B.fmt_both(best["name"], r, dev=True))
        L.append("判定：%s" % B.verdict(r))
        B.ledger_add({"kind": "arm", "name": best["name"], "screen": B.screen_row(best["name"], r["screen"]),
                      "dev": B.screen_row(best["name"], r["dev"]), "comb_t": r["comb"]["t"], "boot_p": r["comb"]["boot_p"],
                      "verdict": B.verdict(r)})
        # 打乱自检：hit 每天内置换，同配置筛选集 ΔP 应回到 0 附近
        ts = B.dev_shuffle_within_day(t)
        rsh = B.eval_both(ts, sel, boot=False)
        L.append("打乱自检（hit 每天内置换，同配置）：筛选 ΔP %+.2fpp (t %.2f)  开发 ΔP %+.2fpp (t %.2f)" % (
            100 * rsh["screen"]["dP"], rsh["screen"]["t"], 100 * rsh["dev"]["dP"], rsh["dev"]["t"]))
        B.ledger_add({"kind": "shuffle", "what": "V1 最好配置 hit 每天内置换", "screen_dP": rsh["screen"]["dP"], "dev_dP": rsh["dev"]["dP"]})
        # 这一臂两段的名单和 B(k̄) 的重合度
        for seg in ("screen", "dev"):
            e = r[seg]
            L.append("%s：出票 %d，命中 %d，每天 %.2f 只，命中票 20 根最大涨幅 q10/50/90 %s" % (
                seg, e["V"], int(sum(e["h"])), e["vbar"], {k: round(100 * v) for k, v in e["gain_q"].items()}))
    (B.HERE / "results_v1.txt").write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.progress("v1_bl：诊断 3 张 + 筛选网格 %d 配置 + 自检 + 开发集 1 臂 -> results_v1.txt（%.0fs）" % (len(rows_screen), time.time() - t0))


if __name__ == "__main__":
    main()
