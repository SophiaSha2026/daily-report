"""
一行字的规则 对 现状模型：稳健性核对（ceiling.py 第 3 节的延伸）。

ceiling.py 发现「每天取 20 日涨幅最高的 10 只」16.9% > 现状模型 11.8%。这里核：
  1. 剔 ST（和模型回测同一份静态名单）、剔前复权价 < 2 元；
  2. 分期：含 924 / 去 924 / 2025 起；
  3. 规则 − 模型 的按天配对差 NW t；
  4. 筛选集 2022-01..2024-05（五年里另外 29 个月）上同样的规则 对 同循环 base（8.33%）；
  5. 规则清单的持续性：每月不重复的票数、同一只连续在榜天数。
    PYTHONIOENCODING=utf-8 python research/rethink/rules.py > research/rethink/rules.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nw_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
T0 = time.time()
ST = {str(c).zfill(6) for c in json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8")).get("codes", [])}


def say(*a):
    print(*a, flush=True)


def build_feats(px: pd.DataFrame) -> pd.DataFrame:
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    px["ret1"] = g["close"].pct_change()
    px["ret20"] = g["close"].pct_change(20)
    px["ret10"] = g["close"].pct_change(10)
    lr = np.log(px["close"]).groupby(px["code"]).diff()
    px["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std())
    px["amp"] = px["high"] / px["low"] - 1
    px["amp20"] = px.groupby("code")["amp"].transform(lambda s: s.rolling(20).mean())
    px["turn20"] = px.groupby("code")["turnover"].transform(lambda s: s.rolling(20).mean())
    px["mcap"] = px["close"] * px["outstanding_share"]
    thr = np.where(px["code"].str[:2].isin(["30", "68"]), 0.195,
                   np.where(px["code"].str[:1].isin(["9", "8", "4"]), 0.29, 0.097))
    px["lu"] = (px["ret1"] >= thr).astype(float)
    px["lu20"] = px.groupby("code")["lu"].transform(lambda s: s.rolling(20).sum())
    return px


RULES = {
    "ret20": lambda r: r["ret20"],
    "ret10": lambda r: r["ret10"],
    "vol60": lambda r: r["vol60"],
    "amp20": lambda r: r["amp20"],
    "mcap_small": lambda r: -r["mcap"],
    "vol60+ret20": lambda r: r["_rk_vol60"] + r["_rk_ret20"],
    "amp20+ret20": lambda r: r["_rk_amp20"] + r["_rk_ret20"],
    "vol60+ret20+turn20": lambda r: r["_rk_vol60"] + r["_rk_ret20"] + r["_rk_turn20"],
    "model": lambda r: r["p"],
}


def prep(r: pd.DataFrame) -> pd.DataFrame:
    for c in ("vol60", "ret20", "amp20", "turn20"):
        r["_rk_" + c] = r.groupby("date")[c].rank(pct=True)
    return r


def daily_hit(r: pd.DataFrame, score: pd.Series, k: int, elig: pd.Series) -> pd.Series:
    s = score.where(elig, -np.inf)
    rk = s.groupby(r["date"]).rank(ascending=False, method="first")
    sub = r[(rk <= k) & elig]
    return sub.groupby("date")["y"].mean()


def analyse(tag: str, r: pd.DataFrame, k: int = 10):
    r = prep(r)
    elig_all = pd.Series(True, index=r.index)
    elig_st = ~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"]) & np.isfinite(r["vol60"])
    dbase = r.groupby("date")["y"].mean()
    out = {}
    say(f"\n== {tag}：每天前 {k}，买得到口径；行 {len(r)}，天 {r['date'].nunique()}，基准 {100 * dbase.mean():.2f}% ==")
    say("%-22s %8s %8s | %8s %8s %8s %8s | %8s %8s" % (
        "规则", "全部%", "剔ST<2元%", "去924%", "2025起%", "超额pp", "零天%", "对模型pp", "NW t"))
    model_hit = daily_hit(r, RULES["model"](r), k, elig_st)
    for nm, fn in RULES.items():
        sc = fn(r)
        h_all = daily_hit(r, sc, k, elig_all)
        h = daily_hit(r, sc, k, elig_st)
        m = h.index.str[:7]
        h_no924 = h[m != "2024-09"]
        h_2025 = h[h.index >= "2025-01-01"]
        ex = (h - dbase.reindex(h.index))
        dlt = (h - model_hit.reindex(h.index)).dropna()
        mu, se, t = nw_t(dlt.values) if nm != "model" else (0, 0, 0)
        out[nm] = h
        say("%-22s %7.2f%% %8.2f%% | %7.2f%% %7.2f%% %+7.2f %7.1f%% | %+8.2f %8.2f" % (
            nm, 100 * h_all.mean(), 100 * h.mean(), 100 * h_no924.mean() if len(h_no924) else np.nan,
            100 * h_2025.mean() if len(h_2025) else np.nan, 100 * ex.mean(), 100 * (h == 0).mean(), 100 * mu, t))
    return out


def persistence(tag: str, r: pd.DataFrame, k: int = 10):
    r = prep(r)
    elig = ~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"]) & np.isfinite(r["vol60"])
    say(f"\n-- {tag}：清单的持续性（每天前 {k}）--")
    say("%-22s %10s %12s %10s %10s" % ("规则", "每月不重复票", "日均在榜连续天", "前10 Jaccard 相邻日", "命中票占不重复票%"))
    for nm in ("ret20", "vol60+ret20", "amp20", "model"):
        sc = RULES[nm](r).where(elig, -np.inf)
        rk = sc.groupby(r["date"]).rank(ascending=False, method="first")
        sub = r[(rk <= k) & elig][["date", "code", "y"]].sort_values(["date", "code"])
        per_month = sub.assign(m=sub["date"].str[:7]).groupby("m")["code"].nunique().mean()
        sets = sub.groupby("date")["code"].apply(set)
        ds = list(sets.index)
        jac = np.mean([len(sets[ds[i]] & sets[ds[i + 1]]) / len(sets[ds[i]] | sets[ds[i + 1]]) for i in range(len(ds) - 1)])
        # 连续在榜天数
        sub = sub.sort_values(["code", "date"])
        dnum = sub["date"].map({d: i for i, d in enumerate(ds)})
        sub["grp"] = (dnum - sub.groupby("code").cumcount())
        streak = sub.groupby(["code", "grp"]).size()
        uniq_hit = sub.groupby("code")["y"].max().mean()
        say("%-22s %10.1f %12.2f %10.2f %10.1f%%" % (nm, per_month, streak.mean(), jac, 100 * uniq_hit))


# ---------------------------------------------------------------- 开发集
lab = pq.read_table(EXP / "leak" / "labels_open_snap.parquet",
                    filters=[("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")]).to_pandas()
lab["code"] = lab["code"].astype(str).str.zfill(6)
px = pq.read_table(EXP / "leak" / "daily_snap.parquet",
                   filters=[("date", ">=", "2024-02-01"), ("date", "<=", "2025-12-31")]).to_pandas()
px["code"] = px["code"].astype(str).str.zfill(6)
px = build_feats(px)
base = pq.read_table(EXP / "feat_final_v1" / "all_base_s7.parquet",
                     filters=[("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")]).to_pandas()
base["code"] = base["code"].astype(str).str.zfill(6)
cand = pq.read_table(EXP / "feat_final_v1" / "all_cand_s7.parquet",
                     filters=[("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")]).to_pandas()
cand["code"] = cand["code"].astype(str).str.zfill(6)
r = base.merge(lab, on=["code", "date"]).merge(
    px[["code", "date", "close", "ret20", "ret10", "vol60", "amp20", "turn20", "mcap", "lu20"]], on=["code", "date"], how="left")
r = r[np.isfinite(r["y_open"])].copy()
r["y"] = r["y_open"] > 0.5
dev10 = analyse("开发集 2024-06..2025-12（模型 = 同循环 base_s7）", r, 10)
dev30 = analyse("开发集，每天前 30", r, 30)
persistence("开发集", r, 10)

# 候选 a158_ovh_cs4 和 ret20 规则的重合
rc = r.merge(cand.rename(columns={"p": "p_cand"}), on=["code", "date"])
elig = ~rc["code"].isin(ST) & (rc["close"] >= 2.0) & np.isfinite(rc["ret20"])
for nm, sc in [("cand", rc["p_cand"]), ("ret20", rc["ret20"]), ("model", rc["p"])]:
    rc["_rk_" + nm] = sc.where(elig, -np.inf).groupby(rc["date"]).rank(ascending=False, method="first")
t_c = rc[rc["_rk_cand"] <= 10]
say("\n候选 a158_ovh_cs4 前 10（剔 ST/<2 元）命中 %.2f%%；其中落在 ret20 规则前 10 的 %.1f%%、前 30 的 %.1f%%、前 100 的 %.1f%%" % (
    100 * t_c.groupby("date")["y"].mean().mean(),
    100 * (t_c["_rk_ret20"] <= 10).mean(), 100 * (t_c["_rk_ret20"] <= 30).mean(), 100 * (t_c["_rk_ret20"] <= 100).mean()))
t_m = rc[rc["_rk_model"] <= 10]
say("现状模型前 10 里落在 ret20 规则前 10 / 30 / 100 的：%.1f%% / %.1f%% / %.1f%%" % (
    100 * (t_m["_rk_ret20"] <= 10).mean(), 100 * (t_m["_rk_ret20"] <= 30).mean(), 100 * (t_m["_rk_ret20"] <= 100).mean()))
say("候选前 10 的 ret20 中位 %.3f，现状模型前 10 的 ret20 中位 %.3f，全市场 %.3f" % (
    t_c["ret20"].median(), t_m["ret20"].median(), rc["ret20"].median()))
hc = t_c.groupby("date")["y"].mean()
hr = dev10["ret20"].reindex(hc.index)
mu, se, t = nw_t((hc - hr).dropna().values)
say("候选 − ret20 规则：%+.2fpp，NW t %.2f" % (100 * mu, t))

# ---------------------------------------------------------------- 筛选集
say("\n\n######## 筛选集 2022-01..2024-05 ########")
labs = pq.read_table(EXP / "screen" / "labels_screen.parquet", columns=["code", "date", "y_open", "r20_open"],
                     filters=[("date", ">=", "2022-01-01"), ("date", "<=", "2024-05-31")]).to_pandas()
labs["code"] = labs["code"].astype(str).str.zfill(6)
pxs = pq.read_table(EXP / "screen" / "px_screen.parquet",
                    columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share"],
                    filters=[("date", ">=", "2021-09-01"), ("date", "<=", "2024-05-31")]).to_pandas()
pxs["code"] = pxs["code"].astype(str).str.zfill(6)
pxs = build_feats(pxs)
bs = pq.read_table(EXP / "screen_base_v1" / "all_base_s7.parquet").to_pandas()
bs["code"] = bs["code"].astype(str).str.zfill(6)
rs = bs.merge(labs, on=["code", "date"]).merge(
    pxs[["code", "date", "close", "ret20", "ret10", "vol60", "amp20", "turn20", "mcap", "lu20"]], on=["code", "date"], how="left")
rs = rs[np.isfinite(rs["y_open"])].copy()
rs["y"] = rs["y_open"] > 0.5
scr10 = analyse("筛选集 2022-01..2024-05（模型 = screen_base_v1 base_s7）", rs, 10)
scr30 = analyse("筛选集，每天前 30", rs, 30)
persistence("筛选集", rs, 10)

# 按半年
say("\n按半年：ret20 规则 / vol60+ret20 / 模型（每天前 10，剔 ST/<2 元）")
for tag, d10 in [("筛选集", scr10), ("开发集", dev10)]:
    for nm in ("ret20", "vol60+ret20", "amp20", "model"):
        h = d10[nm]
        hh = h.groupby(h.index.str[:4] + np.where(h.index.str[5:7] <= "06", "H1", "H2")).mean()
        say("  %-6s %-14s %s" % (tag, nm, "  ".join("%s %.1f" % (k, 100 * v) for k, v in hh.items())))

say("\n耗时 %.0f 秒" % (time.time() - T0))
