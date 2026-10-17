"""
功效：对 ret20 基线按天配对的 NW se 随「和基线前 10 的重合度」怎么变，以及对应 MDE。

    PYTHONIOENCODING=utf-8 python research/polish/power.py

合成臂：每天把 ret20 前 10 里的 m 只换成 ret20 第 11~50 名里随机抽的（同一候选池、同一天），m = 1..10，
重复 20 个种子，取 se 的均值。真实近邻臂（不看命中，只看 se）：ret10 / ret40 / ret20 剔今日涨停。
MDE(80%) = (c + 0.84) * se，c = 2.5。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P


def main():
    df = P.load_pool(["date", "code", "board", "y_open", "y_up", "pool", "lret20", "ret10", "ret40", "lu_close"])
    df = df[df["pool"] & (df["date"] >= "2022-01-01")].reset_index(drop=True)
    base = P.rank_by(df, df["lret20"].to_numpy(), keep=50)
    hb = P.hit_series(base)
    P.say("ret20 基线：%d 天，前 10 命中 %.2f%%（筛选 %.2f%% / 开发 %.2f%%）" % (
        len(hb), 100 * hb.mean(), 100 * hb[hb.index < "2024-06"].mean(), 100 * hb[hb.index >= "2024-06"].mean()))
    top = base[base["rank"] <= 10]
    rest = base[(base["rank"] > 10) & (base["rank"] <= 50)]
    lines = ["合成臂：换掉前 10 里的 m 只（换成第 11~50 名随机）", "  m  重合  NW se(pp)  MDE80(c=2.5)  MDE80(c=2.8)"]
    rng = np.random.default_rng(1)
    rest_g = {d: g["y_open"].to_numpy() for d, g in rest.groupby("date")}
    top_g = {d: g["y_open"].to_numpy() for d, g in top.groupby("date")}
    days = sorted(top_g)
    for m in (1, 2, 3, 4, 5, 6, 8, 10):
        ses = []
        for s in range(20):
            diffs = np.empty(len(days))
            for i, d in enumerate(days):
                t = top_g[d]
                r = rest_g.get(d, np.array([]))
                if len(r) < m:
                    diffs[i] = 0.0
                    continue
                keep = rng.permutation(len(t))[: len(t) - m]
                new = np.concatenate([t[keep], rng.choice(r, m, replace=False)])
                diffs[i] = np.nanmean(new) - np.nanmean(t)
            ses.append(P.nw_se(diffs))
        se = float(np.mean(ses))
        lines.append("  %2d  %3d%%  %6.2f     %+5.2fpp      %+5.2fpp" % (m, 100 - 10 * m, 100 * se, 100 * 3.34 * se, 100 * 3.64 * se))
    lines.append("真实近邻臂（只看 se 和重合度，命中不印）")
    for name, sc, el in (("ret10", df["ret10"].to_numpy(), None), ("ret40", df["ret40"].to_numpy(), None),
                         ("ret20 剔今日涨停", df["lret20"].to_numpy(), (df["lu_close"].to_numpy() < 0.5))):
        t = P.rank_by(df, sc, keep=10, elig=el)
        h = P.hit_series(t)
        d = (h - hb).dropna()
        ja = []
        sa = base[base["rank"] <= 10].groupby("date")["code"].apply(set)
        sb = t.groupby("date")["code"].apply(set)
        for k in sa.index:
            if k in sb.index:
                ja.append(len(sa[k] & sb[k]) / 10)
        se = P.nw_se(d.to_numpy())
        lines.append("  %-18s 重合 %3.0f%%  NW se %.2fpp  MDE80 %+.2fpp (c=2.5)" % (name, 100 * np.mean(ja), 100 * se, 100 * 3.34 * se))
    txt = "\n".join(lines)
    P.say(txt)
    (P.HERE / "results_power.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
