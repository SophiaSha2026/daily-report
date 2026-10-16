"""
从根源上估天花板：日线量价对「20 根内最高价涨 50%」这个标签到底能解释多少。

只读已有产物（开发集 2024-06..2025-12），不训模型，不读 2026 年标签。
    python research/rethink/ceiling.py  >  research/rethink/ceiling.txt

数据：
  data/breakout/raw/exp/feat_final_v1/all_base_s{7,9,10}.parquet   现状模型（同循环）全市场预测值
  data/breakout/raw/exp/leak_v1/all_{base,cand}_pit_s7.parquet     干净价格上的现状 / 候选
  data/breakout/raw/exp/leak/labels_open_snap.parquet              买得到口径标签（只到 2025-12-31）
  data/breakout/raw/exp/leak/daily_snap.parquet                    日线快照（算几条一行字的规则）
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
DEV0, DEV1 = "2024-06-01", "2025-12-31"
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def read_dev(path, cols=None):
    t = pq.read_table(path, columns=cols,
                      filters=[("date", ">=", DEV0), ("date", "<=", DEV1)])
    return t.to_pandas()


sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nw_se, nw_t  # noqa: E402


def auc_fast(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ap_fast(y, s):
    y = np.asarray(y, bool)
    o = np.argsort(-np.asarray(s, float), kind="stable")
    yy = y[o]
    if yy.sum() == 0:
        return np.nan
    cum = np.cumsum(yy)
    prec = cum / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def binormal_precision(auc: float, base: float, frac: float) -> float:
    """双正态模型：负样本 N(0,1)、正样本 N(d,1)，d = sqrt(2) Φ⁻¹(AUC)。
    取分数最高的 frac 比例，命中率是多少。"""
    d = np.sqrt(2) * norm.ppf(auc)
    # 找阈值 c 使 base*(1-Φ(c-d)) + (1-base)*(1-Φ(c)) = frac
    lo, hi = -10, 15
    for _ in range(80):
        c = (lo + hi) / 2
        f = base * norm.sf(c - d) + (1 - base) * norm.sf(c)
        if f > frac:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2
    tp = base * norm.sf(c - d)
    return tp / max(frac, 1e-12)


def auc_for_precision(target: float, base: float, frac: float) -> float:
    lo, hi = 0.5, 0.9999
    for _ in range(60):
        m = (lo + hi) / 2
        if binormal_precision(m, base, frac) < target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


# ----------------------------------------------------------------------------
say("== 0. 读数据 ==")
lab = read_dev(EXP / "leak" / "labels_open_snap.parquet")
lab["code"] = lab["code"].astype(str).str.zfill(6)
say("标签行", len(lab), "日期", lab["date"].min(), lab["date"].max(),
    "y_open 均值 %.4f" % lab["y_open"].mean())

base = read_dev(EXP / "feat_final_v1" / "all_base_s7.parquet")
base["code"] = base["code"].astype(str).str.zfill(6)
df = base.merge(lab, on=["code", "date"], how="inner")
df = df[np.isfinite(df["y_open"])].copy()
df["y"] = df["y_open"] > 0.5
df["rank"] = df.groupby("date")["p"].rank(ascending=False, method="first")
days = df["date"].nunique()
say("现状 base_s7 可评行", len(df), "天数", days, "每天只数中位", df.groupby("date").size().median())

# ----------------------------------------------------------------------------
say("\n== 1. 现状模型的整体分辨力：AUC / AP / 精度-召回曲线（全市场，买得到口径） ==")
auc_pool = auc_fast(df["y"], df["p"])
ap_pool = ap_fast(df["y"], df["p"])
d_auc = df.groupby("date").apply(lambda g: auc_fast(g["y"], g["p"]))
d_ap = df.groupby("date").apply(lambda g: ap_fast(g["y"], g["p"]))
base_rate = df["y"].mean()
d_base = df.groupby("date")["y"].mean()
say("合并 AUC %.4f  AP %.4f  基准 %.4f（AP/基准 %.1f 倍）" % (auc_pool, ap_pool, base_rate, ap_pool / base_rate))
say("按天 AUC 均值 %.4f（p10 %.3f p90 %.3f）  按天 AP 均值 %.4f" % (
    d_auc.mean(), d_auc.quantile(.1), d_auc.quantile(.9), d_ap.mean()))

ks = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
pos_per_day = df.groupby("date")["y"].sum()
say("每天正样本数中位 %.0f，均值 %.1f" % (pos_per_day.median(), pos_per_day.mean()))
say("%6s %8s %8s %8s %10s" % ("前k", "命中率", "每天命中", "召回", "倍数(按天)"))
for k in ks:
    sub = df[df["rank"] <= k]
    hit = sub.groupby("date")["y"].mean().mean()
    nh = sub.groupby("date")["y"].sum()
    rec = (nh / pos_per_day.reindex(nh.index)).mean()
    lift = (sub.groupby("date")["y"].mean() / d_base.reindex(nh.index)).mean()
    say("%6d %7.2f%% %8.2f %7.2f%% %9.2f" % (k, 100 * hit, nh.mean(), 100 * rec, lift))

say("\n-- 双正态模型：同样的 AUC，理论上前 10（每天约 5300 只，0.19%）该有多少命中 --")
frac = 10 / df.groupby("date").size().mean()
for a in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    say("  AUC %.2f -> 前 10 理论命中 %.1f%%" % (a, 100 * binormal_precision(a, base_rate, frac)))
say("  实际合并 AUC %.3f -> 理论 %.1f%%，实测 %.1f%%" % (
    auc_pool, 100 * binormal_precision(auc_pool, base_rate, frac),
    100 * df[df["rank"] <= 10].groupby("date")["y"].mean().mean()))
for tgt in [0.15, 0.20, 0.30, 0.50]:
    say("  要前 10 命中 %.0f%%，双正态下需要 AUC %.3f" % (100 * tgt, auc_for_precision(tgt, base_rate, frac)))

# ----------------------------------------------------------------------------
say("\n== 2. 「当天」解释了多少：前 10 命中 对 当天全市场基准率 ==")
t10 = df[df["rank"] <= 10].groupby("date")["y"].mean()
d = pd.DataFrame({"hit": t10, "base": d_base.reindex(t10.index)})
c = np.corrcoef(d["hit"], d["base"])[0, 1]
say("前 10 命中序列 均值 %.4f sd %.4f；和当天基准率相关 %.3f（R2 %.3f）" % (
    d["hit"].mean(), d["hit"].std(), c, c * c))
d["ex"] = d["hit"] - d["base"]
say("超额（命中 − 当天基准）均值 %.4f sd %.4f；超额序列和基准率相关 %.3f" % (
    d["ex"].mean(), d["ex"].std(), np.corrcoef(d["ex"], d["base"])[0, 1]))
d["q"] = pd.qcut(d["base"], 3, labels=["冷", "中", "热"])
say(d.groupby("q", observed=True).agg(days=("hit", "size"), base=("base", "mean"), hit=("hit", "mean"),
                                       excess=("ex", "mean"), zero_days=("hit", lambda s: (s == 0).mean())).round(4))
say("前 10 一个都没中的天 %.1f%%；若 10 只独立二项 p=%.3f 应为 %.1f%%" % (
    100 * (t10 == 0).mean(), t10.mean(), 100 * (1 - t10.mean()) ** 10))
t30 = df[df["rank"] <= 30].groupby("date")["y"].mean()
say("前 30 一个都没中的天 %.1f%%" % (100 * (t30 == 0).mean()))
# 去掉 2024-09
m = d.index.str[:7]
dd = d[m != "2024-09"]
say("去掉 2024-09：前 10 命中 %.4f，基准 %.4f，倍数 %.2f" % (dd["hit"].mean(), dd["base"].mean(), dd["hit"].mean() / dd["base"].mean()))

# ----------------------------------------------------------------------------
say("\n== 3. 一行字的规则能到多少（同一张表、同一尺子：每天前 10 买得到命中） ==")
px = read_dev(EXP / "leak" / "daily_snap.parquet")
px["code"] = px["code"].astype(str).str.zfill(6)
# 需要往前 60 天的历史：读 2024-03 起
px_hist = pq.read_table(EXP / "leak" / "daily_snap.parquet",
                        filters=[("date", ">=", "2024-02-01"), ("date", "<=", DEV1)]).to_pandas()
px_hist["code"] = px_hist["code"].astype(str).str.zfill(6)
px_hist = px_hist.sort_values(["code", "date"]).reset_index(drop=True)
g = px_hist.groupby("code", sort=False)
px_hist["ret1"] = g["close"].pct_change()
px_hist["ret5"] = g["close"].pct_change(5)
px_hist["ret20"] = g["close"].pct_change(20)
px_hist["ret60"] = g["close"].pct_change(60)
lr = np.log(px_hist["close"]).groupby(px_hist["code"]).diff()
px_hist["vol20"] = lr.groupby(px_hist["code"]).transform(lambda s: s.rolling(20).std())
px_hist["vol60"] = lr.groupby(px_hist["code"]).transform(lambda s: s.rolling(60).std())
px_hist["amp"] = px_hist["high"] / px_hist["low"] - 1
px_hist["amp20"] = px_hist.groupby("code")["amp"].transform(lambda s: s.rolling(20).mean())
px_hist["turn20"] = px_hist.groupby("code")["turnover"].transform(lambda s: s.rolling(20).mean())
px_hist["amt20"] = px_hist.groupby("code")["amount"].transform(lambda s: s.rolling(20).mean())
px_hist["vr5_20"] = px_hist.groupby("code")["volume"].transform(lambda s: s.rolling(5).mean() / s.rolling(20).mean())
px_hist["mcap"] = px_hist["close"] * px_hist["outstanding_share"]
px_hist["hi60"] = px_hist.groupby("code")["high"].transform(lambda s: s.rolling(60).max())
px_hist["dist_hi60"] = px_hist["close"] / px_hist["hi60"] - 1
thr = np.where(px_hist["code"].str[:2].isin(["30", "68"]), 0.195,
               np.where(px_hist["code"].str[:1].isin(["9", "8", "4"]), 0.29, 0.097))
px_hist["lu"] = (px_hist["ret1"] >= thr).astype(float)
px_hist["lu5"] = px_hist.groupby("code")["lu"].transform(lambda s: s.rolling(5).sum())
px_hist["lu20"] = px_hist.groupby("code")["lu"].transform(lambda s: s.rolling(20).sum())
feat = px_hist[px_hist["date"] >= DEV0][["code", "date", "ret5", "ret20", "ret60", "vol20", "vol60", "amp20",
                                          "turn20", "amt20", "vr5_20", "mcap", "dist_hi60", "lu5", "lu20"]]
r = df[["code", "date", "y", "r20_open", "p", "rank"]].merge(feat, on=["code", "date"], how="left")
r["board"] = np.where(r["code"].str[:2].isin(["30"]), "chinext",
                      np.where(r["code"].str[:2] == "68", "star",
                               np.where(r["code"].str[:1].isin(["9", "8", "4"]), "bj", "main")))


def top10_rule(col, asc=False, name=None):
    s = r[col] if not asc else -r[col]
    rk = s.groupby(r["date"]).rank(ascending=False, method="first")
    sub = r[rk <= 10]
    hit = sub.groupby("date")["y"].mean()
    return float(hit.mean()), float(nw_se(hit.values)), float(sub["r20_open"].mean())


say("规则（每天取该量最大的 10 只）        命中%  NW se  前10 r20 均值")
rules = [("vol60 最高", "vol60", False), ("vol20 最高", "vol20", False), ("amp20 最高", "amp20", False),
         ("ret20 最高", "ret20", False), ("ret5 最高", "ret5", False), ("ret60 最高", "ret60", False),
         ("turn20 最高", "turn20", False), ("vr5_20 最高", "vr5_20", False), ("lu20 最多", "lu20", False),
         ("lu5 最多", "lu5", False), ("mcap 最小", "mcap", True), ("amt20 最小", "amt20", True),
         ("dist_hi60 最高（贴近 60 日高）", "dist_hi60", False), ("模型 p 最高（现状）", "p", False)]
for nm, col, asc in rules:
    h, se, rr = top10_rule(col, asc)
    say("%-34s %6.2f%% %5.2f  %6.3f" % (nm, 100 * h, 100 * se, rr))
# 组合：当日秩的和
for combo in [("vol60", "ret20"), ("vol60", "lu20"), ("amp20", "turn20"), ("vol60", "ret20", "turn20"),
              ("vol20", "amp20", "lu20", "ret20")]:
    z = sum(r[c].groupby(r["date"]).rank(pct=True) for c in combo)
    r["_z"] = z
    h, se, rr = top10_rule("_z")
    say("%-34s %6.2f%% %5.2f  %6.3f" % ("+".join(combo) + " 秩和", 100 * h, 100 * se, rr))
# 条件基准率：按板块、按 vol60 十分位、按 lu20
say("\n条件基准率（买得到口径，按天均值再平均）")
for col, q in [("vol60", 10), ("amp20", 10), ("ret20", 10), ("turn20", 10), ("mcap", 5)]:
    r["_q"] = r.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False))
    tab = r.groupby("_q")["y"].mean()
    say("  %-7s 分位 0..%d: %s" % (col, q - 1, " ".join("%.1f" % (100 * v) for v in tab.values)))
say("  板块:", r.groupby("board")["y"].mean().round(4).to_dict())
say("  lu20=0 / 1 / 2 / 3+: %s" % r.groupby(r["lu20"].clip(upper=3))["y"].mean().round(4).to_dict())
# 模型前 10 里和最高 vol60 十分位的重合
r["_vq"] = r.groupby("date")["vol60"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
say("  模型前 10 里落在 vol60 最高十分位的比例 %.1f%%，最高两档 %.1f%%" % (
    100 * (r.loc[r["rank"] <= 10, "_vq"] == 9).mean(), 100 * (r.loc[r["rank"] <= 10, "_vq"] >= 8).mean()))
say("  模型前 10 里 lu20≥1 的比例 %.1f%%，全市场 %.1f%%" % (
    100 * (r.loc[r["rank"] <= 10, "lu20"] >= 1).mean(), 100 * (r["lu20"] >= 1).mean()))

# ----------------------------------------------------------------------------
say("\n== 4. 清单长度 k：命中率、名单稳定性（换种子的 Jaccard）、每天至少中一只的概率 ==")
seeds = {}
for s in (7, 9, 10):
    a = read_dev(EXP / "feat_final_v1" / f"all_base_s{s}.parquet")
    a["code"] = a["code"].astype(str).str.zfill(6)
    a["rank"] = a.groupby("date")["p"].rank(ascending=False, method="first")
    seeds[s] = a
say("%5s %8s %10s %12s %12s" % ("k", "命中%", "日均命中只", "≥1 只的天%", "Jaccard s7/s9/s10"))
for k in [5, 10, 20, 30, 50, 100]:
    sub = df[df["rank"] <= k]
    hit = sub.groupby("date")["y"].mean().mean()
    nh = sub.groupby("date")["y"].sum()
    sets = {s: a[a["rank"] <= k].groupby("date")["code"].apply(set) for s, a in seeds.items()}
    jac = []
    for (s1, s2) in [(7, 9), (7, 10), (9, 10)]:
        common = sets[s1].index.intersection(sets[s2].index)
        jac.append(np.mean([len(sets[s1][d] & sets[s2][d]) / len(sets[s1][d] | sets[s2][d]) for d in common]))
    say("%5d %7.2f%% %10.2f %11.1f%% %12.2f" % (k, 100 * hit, nh.mean(), 100 * (nh >= 1).mean(), np.mean(jac)))
# 三种子平均排名 vs 单种子
avg = seeds[7][["date", "code", "p"]].rename(columns={"p": "p7"})
for s in (9, 10):
    avg = avg.merge(seeds[s][["date", "code", "p"]].rename(columns={"p": f"p{s}"}), on=["date", "code"])
avg["pm"] = avg[["p7", "p9", "p10"]].rank(pct=True).mean(axis=1) if False else avg[["p7", "p9", "p10"]].mean(axis=1)
avg = avg.merge(lab[["code", "date", "y_open"]], on=["code", "date"])
avg = avg[np.isfinite(avg["y_open"])]
avg["rank"] = avg.groupby("date")["pm"].rank(ascending=False, method="first")
for k in [10, 30]:
    h = avg[avg["rank"] <= k].groupby("date")["y_open"].mean().mean()
    say("三种子平均预测值 前 %d 命中 %.2f%%" % (k, 100 * h))

# ----------------------------------------------------------------------------
say("\n== 5. 校准：预测值分档的真实命中（全市场，p 的当日名次档） ==")
bins = [0, 10, 30, 100, 200, 500, 1000, 2000, 1e9]
r["_rb"] = pd.cut(r["rank"], bins, labels=["1-10", "11-30", "31-100", "101-200", "201-500", "501-1000", "1001-2000", "2000+"])
tab = r.groupby("_rb", observed=True).agg(n=("y", "size"), hit=("y", "mean"), r20_mean=("r20_open", "mean"),
                                          r20_med=("r20_open", "median"), p_mean=("p", "mean"))
say(tab.round(4))
# 预测值绝对分档（跨天）
r["_pb"] = pd.qcut(r["p"], [0, .5, .9, .99, .995, .998, .999, 1.0], labels=False, duplicates="drop")
tab = r.groupby("_pb").agg(n=("y", "size"), p_mean=("p", "mean"), hit=("y", "mean"))
say("预测值全样本分位档（0 最低）\n", tab.round(4))

# ----------------------------------------------------------------------------
say("\n== 6. 同一对臂（干净价格 cand_pit 对 base_pit，种子 7）在不同尺子下的功效 ==")
bp = read_dev(EXP / "leak_v1" / "all_base_pit_s7.parquet")
cp = read_dev(EXP / "leak_v1" / "all_cand_pit_s7.parquet")
for a in (bp, cp):
    a["code"] = a["code"].astype(str).str.zfill(6)
bp = bp.merge(lab, on=["code", "date"]).dropna(subset=["y_open"])
cp = cp.merge(lab, on=["code", "date"]).dropna(subset=["y_open"])
for a in (bp, cp):
    a["rank"] = a.groupby("date")["p"].rank(ascending=False, method="first")
    a["y"] = a["y_open"] > 0.5
    a["lr"] = np.log1p(a["r20_open"].clip(lower=-0.99))


def daily_metrics(a):
    out = {}
    gb = a.groupby("date")
    for k in (10, 30, 100):
        out[f"top{k}_hit"] = a[a["rank"] <= k].groupby("date")["y"].mean()
        out[f"top{k}_lr"] = a[a["rank"] <= k].groupby("date")["lr"].mean()
    out["auc"] = gb.apply(lambda g: auc_fast(g["y"], g["p"]))
    out["ap"] = gb.apply(lambda g: ap_fast(g["y"], g["p"]))
    out["spearman_lr"] = gb.apply(lambda g: np.corrcoef(rankdata(g["p"]), rankdata(g["lr"]))[0, 1])
    n = gb.size()
    out["prec1pct"] = a[a["rank"] <= np.ceil(0.01 * n.reindex(a["date"]).values)].groupby("date")["y"].mean()
    return pd.DataFrame(out)


mb, mc = daily_metrics(bp), daily_metrics(cp)
say("%-14s %10s %10s %10s %8s %8s" % ("尺子", "base", "cand", "差", "NW se", "NW t"))
for col in mb.columns:
    dlt = (mc[col] - mb[col]).dropna()
    m_, se, t = nw_t(dlt.values)
    say("%-14s %10.4f %10.4f %+10.4f %8.4f %8.2f" % (col, mb[col].mean(), mc[col].mean(), m_, se, t))
say("（前 10 命中是登记的主尺子；其余是同一对臂的辅助尺子。全市场 AUC/AP 的差基本看不见，差别在尾部。）")

# ----------------------------------------------------------------------------
say("\n== 7. 前向累积要多久：按现有 NW se 外推（增量臂 0.87pp、大改 1.85pp、种子级 0.46pp，387 天） ==")
for nm, se387 in [("特征大改", 1.85), ("同族增量", 0.87), ("换种子级", 0.46)]:
    for eff in (2, 3, 5):
        for c in (2.0, 2.5):
            n = 387 * ((c + 0.84) * se387 / eff) ** 2
            say("  %-8s 真效应 +%dpp 门槛 %.1f 功效 80%%：需要 %5.0f 天（%.1f 年）" % (nm, eff, c, n, n / 243))
say("  （单侧 c=2.0 是「预登记 + 前向影子跑 + 开发集先验同号」三件合在一起时能接受的下限；见方案第三节）")

say("\n耗时 %.0f 秒" % (time.time() - T0))
