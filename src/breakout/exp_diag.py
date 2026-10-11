"""
实验 20 · 薄弱点诊断（计划 2026-10-10 第 7 项第一步）。先量化模型在哪里弱，再决定试什么。

全部在开发集（evalkit DEV，2024-06..2025-12）的现状臂上看（raw/exp/w1_scale_avg/top_base.parquet，
每天前 200 名 + 买得到标签），按买得到口径 y_open：

    --light   分数单调性、名次曲线、月度残差、板块残差、「已经在动」对「静止」、连续天数（只读小表）
    --heavy   命中对未命中在特征空间的差异、市值分位残差（要读 train.parquet 的特征列）

产物 raw/exp/diag/summary.json，结论写 docs/breakout_log.md 实验 20。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
SRC = K.EXP / "w1_scale_avg" / "top_base.parquet"
FEATS = K.DATA / "raw" / "exp_acc" / "feats.parquet"
OUT = K.EXP / "diag"


def _tab(name: str, g) -> None:
    print(f"\n{name}")
    for k, v in g.iterrows():
        print("  %-14s n=%5d  命中 %5.2f%%  ±%.2f" % (str(k), v["n"], 100 * v["hit"], 100 * v["se"]))


def by(picks: pd.DataFrame, key, y: str = "y_open") -> pd.DataFrame:
    rows = {}
    for k, g in picks.groupby(key):
        c = K.clustered(g, y)
        rows[k] = {"n": c["n"], "hit": c["hit"], "se": c["se"]}
    return pd.DataFrame(rows).T


def light() -> dict:
    t = pd.read_parquet(SRC)
    t["month"] = t["date"].str[:7]
    res = {}
    top10 = t[t["rank"] <= 10]
    print("开发集 %d 天，前 10 名额 %d，买得到口径 %.2f%%，收盘口径 %.2f%%" % (
        t["date"].nunique(), len(top10), 100 * top10["y_open"].mean(), 100 * top10["y_up"].mean()))

    # 1. 名次曲线：第 1 名到第 200 名的命中率衰减有多快
    t["rb"] = pd.cut(t["rank"], [0, 1, 3, 5, 10, 20, 50, 100, 200],
                     labels=["1", "2-3", "4-5", "6-10", "11-20", "21-50", "51-100", "101-200"])
    g = by(t, "rb")
    _tab("名次曲线（买得到口径）", g)
    res["rank_curve"] = g.to_dict("index")

    # 2. 分数单调性（前 200 里的分数）
    t["sb"] = pd.cut(t["score"], [-1, 89, 94, 96, 97, 98, 99, 100],
                     labels=["<90", "90-94", "95-96", "97", "98", "99", "100"])
    g = by(t[t["rank"] <= 50], "sb")
    _tab("分数分档（前 50 名内）", g)
    res["score_bins"] = g.to_dict("index")

    # 3. 月度残差：前 10 命中 − 当月全市场基准（基准用 labels_open）
    import buyable as B
    lab = B.open_labels()
    lab = lab[lab["date"].between(t["date"].min(), t["date"].max())]
    base_m = lab.groupby(lab["date"].str[:7])["y_open"].mean()
    hit_m = top10.groupby("month")["y_open"].mean()
    mo = pd.DataFrame({"hit": hit_m, "base": base_m}).dropna()
    mo["lift"] = mo["hit"] / mo["base"]
    mo["excess"] = mo["hit"] - mo["base"]
    print("\n月度：前 10 命中 / 基准 / 倍数")
    for m, r in mo.iterrows():
        print("  %s  %5.1f%%  %5.2f%%  %4.1fx" % (m, 100 * r["hit"], 100 * r["base"], r["lift"]))
    print("  倍数的月间中位 %.1fx，最低 %.1fx，最高 %.1fx；倍数和基准的相关 %.2f" % (
        mo["lift"].median(), mo["lift"].min(), mo["lift"].max(), mo["lift"].corr(mo["base"])))
    res["monthly"] = mo.to_dict("index")

    # 4. 板块
    g = by(top10, "board")
    _tab("板块（前 10）", g)
    res["board"] = g.to_dict("index")

    # 5. 已经在动 vs 静止（涨停历史、近 20 日涨停、近 5 日涨幅由 feats.parquet 给）
    fx = pd.read_parquet(FEATS, columns=["code", "date", "x_lu_streak", "x_lu20", "x_zb20",
                                          "x_since_resume"])
    m = top10.merge(fx, on=["code", "date"], how="left")
    m["moving"] = np.select([m["x_lu_streak"] > 0, m["x_lu20"] > 0], ["连板中", "近20日封过板"], "没封过板")
    g = by(m, "moving")
    _tab("已经在动 vs 静止（前 10）", g)
    res["moving"] = g.to_dict("index")
    share = m["moving"].value_counts(normalize=True).round(3).to_dict()
    print("  前 10 名额构成：", share)
    res["moving_share"] = share

    # 6. 连续够格天数
    ok = t[(t["score"] >= 97) & (t["rank"] <= 10)].copy()
    ok["_i"] = ok["date"].map({d: i for i, d in enumerate(sorted(t["date"].unique()))})
    streak = []
    bycode = {c: set(g["_i"]) for c, g in ok.groupby("code")}
    for c, i in zip(ok["code"], ok["_i"]):
        s, k = 1, i - 1
        while k in bycode[c]:
            s += 1
            k -= 1
        streak.append(min(s, 4))
    ok["streak"] = streak
    g = by(ok, "streak")
    _tab("连续够格天数（生产规则名额）", g)
    res["streak"] = g.to_dict("index")

    # 7. 命中的票之后怎么走：r20_open 分布（赢多少、输多少）
    lab2 = B.open_labels()[["code", "date", "r20_open"]]
    m2 = top10.merge(lab2, on=["code", "date"], how="left")
    q = m2["r20_open"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
    print("\n前 10 名额次日开盘起算的 20 日最高涨幅分位：", q)
    res["r20_open_quantiles"] = q
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_light.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    return res


def heavy() -> dict:
    """命中 vs 未命中在特征空间里差在哪（前 10 名额），市值分位残差。读 train.parquet。"""
    import pyarrow.parquet as pq
    t = pd.read_parquet(SRC)
    top = t[t["rank"] <= 10].copy()
    pf = pq.ParquetFile(K.DATA / "train.parquet")
    feats = [c for c in pf.schema.names if "__" in c]
    dates = set(top["date"])
    tr = pd.read_parquet(K.DATA / "train.parquet",
                         columns=["code", "date", "float_mcap", "close"] + feats,
                         filters=[("date", "in", sorted(dates))])
    tr["code"] = tr["code"].astype(str).str.zfill(6)
    tr["mcap_pct"] = tr.groupby("date")["float_mcap"].rank(pct=True)
    m = top.merge(tr, on=["code", "date"], how="left")
    res = {}
    m["mq"] = pd.cut(m["mcap_pct"], [0, .25, .5, .75, 1.0], labels=["小", "中小", "中大", "大"])
    g = by(m, "mq")
    _tab("市值分位（前 10）", g)
    res["mcap"] = g.to_dict("index")
    # 特征差：命中组和未命中组的均值差 / 全体标准差，取绝对值前 15
    h, n = m[m["y_open"] == 1], m[m["y_open"] == 0]
    rows = []
    for c in feats:
        a, b = h[c].astype(float), n[c].astype(float)
        sd = m[c].astype(float).std()
        if sd and np.isfinite(sd) and sd > 0:
            rows.append((c, float((a.mean() - b.mean()) / sd), float(a.mean()), float(b.mean())))
    rows.sort(key=lambda r: -abs(r[1]))
    print("\n命中 vs 未命中（前 10 名额）差异最大的特征（标准化均值差）")
    for c, d, am, bm in rows[:15]:
        print("  %-28s %+.2f  命中 %.3f  未命中 %.3f" % (c, d, am, bm))
    res["feature_gap"] = [{"feat": c, "d": d, "hit_mean": am, "miss_mean": bm} for c, d, am, bm in rows[:30]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary_heavy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    a = ap.parse_args()
    if a.light:
        light()
    if a.heavy:
        heavy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
