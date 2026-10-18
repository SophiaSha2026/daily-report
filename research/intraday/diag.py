"""步骤 3 / 4：日内特征在 ret20 前 50 内的诊断（不训正式模型、不立臂、不看确认集）。

    research/datasrc/.venv/Scripts/python.exe research/intraday/diag.py

A 条件命中：每个特征在前 50 内按天中位数分高 / 低组，按天配对的命中差（y_open），NW t（滞后 20），两段 + 合并，另印去 2024-09。
B 增量：同一日线条件内（收在最高价 at_high 0/1、涨停收盘 lu_close 0/1、名次段 1~10 / 11~50）再按天-层中位数分组，
   按天配对差。日线条件已经知道的信息被层固定住，剩下的差才是分时的增量。
C 过滤：把特征当过滤条件加在前 10 上（阈值是事先定的固定数，T 日可算），看精确率（命中 / 出票）和出票量；
   另一种用法：前 50 先过滤再按名次取 10。
D 粗合成：留一月（LOMO）逻辑回归，日内特征 + 名次，前 50 内每天 AUC 和「模型前 10」命中，对 ret20 前 10。只作方向参考。
输出：results_diag.txt、results_filter.txt。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "src" / "breakout", ROOT / "src"):
    sys.path.insert(0, str(p))
import evalkit as K  # noqa: E402

FEATS = {
    # 名字: (说明, 二值?)
    "c_vwap": ("收盘相对日内均价", False),
    "ret_last30": ("尾盘 30 分钟涨幅", False),
    "ret_last60": ("尾盘 60 分钟涨幅", False),
    "ret_first30": ("早盘 30 分钟涨幅（对开盘）", False),
    "ret_am": ("上午对开盘", False),
    "ret_pm": ("下午对上午收盘", False),
    "dd_intra": ("日内最大回撤（越接近 0 越好）", False),
    "c_hi": ("收盘离日内最高（0 = 收在最高）", False),
    "t_high": ("日内高点出现时刻（1 = 收盘）", False),
    "t_low": ("日内低点出现时刻", False),
    "rv": ("1 分钟收益实现波动", False),
    "up_share": ("上涨分钟占比", False),
    "v_first30": ("早盘 30 分钟量占比（含竞价）", False),
    "v_last30": ("尾盘 30 分钟量占比", False),
    "v_last5": ("最后 5 分钟量占比", False),
    "v_am": ("上午量占比", False),
    "v_top10": ("量最大 10 分钟占比", False),
    "v_zero": ("零成交分钟占比", False),
    "v_above_vwap": ("均价之上成交的量占比", False),
    "buy_share": ("主动买占主动成交", False),
    "buy_share_pm": ("14:30 后主动买占比", False),
    "big_net": ("大单（>=100 万）净买占总量", False),
    "big_share": ("大单成交占比", False),
    "auc_share": ("集合竞价量占比", False),
    "tick_size": ("平均每笔手数", False),
    "lu_touch": ("日内触及涨停", True),
    "t_first_lu": ("首次触及涨停时刻（触板的票内）", False),
    "n_open": ("开板次数（触板的票内，越少越好）", False),
    "lu_hold": ("在板分钟占比（触板的票内）", False),
    "v_at_lu": ("在板分钟量占比（触板的票内）", False),
    "lu_seal_close": ("收盘封住（触板的票内）", True),
}
LINES = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def nw_t(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return K.nw_t(d) if len(d) >= 25 else float("nan")


def paired_split(df: pd.DataFrame, f: str, binary: bool, by=("date",), min_n: int = 2) -> pd.Series:
    """按 by（天 或 天+层）把 f 分高 / 低两组，返回按天的配对差序列（一天多层时取层的均值）。"""
    d = df[np.isfinite(df[f].to_numpy(float)) & np.isfinite(df["y_open"].to_numpy(float))]
    if binary:
        hi = d[f] > 0.5
    else:
        med = d.groupby(list(by))[f].transform("median")
        hi = d[f] > med
        # 中位数上的并列：> 中位为高，<= 为低；离散特征会偏，二值的另走 binary
    g = d.assign(_hi=hi).groupby(list(by) + ["_hi"])["y_open"].agg(["mean", "size"]).unstack("_hi")
    if True not in g["mean"].columns or False not in g["mean"].columns:
        return pd.Series(dtype=float)
    ok = (g["size"][True] >= min_n) & (g["size"][False] >= min_n)
    diff = (g["mean"][True] - g["mean"][False])[ok]
    if len(by) > 1:
        diff = diff.groupby(level=0).mean()
    return diff.dropna()


def seg_stats(diff: pd.Series) -> dict:
    out = {}
    for s, m in (("screen", diff.index < "2024-06-01"), ("dev", diff.index >= "2024-06-01"), ("all", np.ones(len(diff), bool))):
        x = diff[m]
        out[s] = (100 * x.mean() if len(x) else np.nan, nw_t(x), len(x))
    x = diff[diff.index.str[:7] != "2024-09"]
    out["ex924"] = (100 * x.mean() if len(x) else np.nan, nw_t(x), len(x))
    return out


def fmt(st: dict) -> str:
    return "筛选 %+6.2f (t %5.2f, %3d天)  开发 %+6.2f (t %5.2f, %3d天)  合并 %+6.2f (t %5.2f)  去924 %+6.2f (t %5.2f)" % (
        *st["screen"], *st["dev"], st["all"][0], st["all"][1], st["ex924"][0], st["ex924"][1])


def day_auc(df: pd.DataFrame, f: str) -> float:
    """按天 AUC 的均值（当天两类都有才算）。"""
    from sklearn.metrics import roc_auc_score
    vals = []
    for _, g in df.groupby("date"):
        g = g[np.isfinite(g[f].to_numpy(float))]
        y = g["y_open"].to_numpy()
        if len(g) >= 5 and 0 < y.sum() < len(y):
            vals.append(roc_auc_score(y, g[f].to_numpy(float)))
    return float(np.mean(vals)) if vals else float("nan")


def section_a(df):
    say("\n== A. 前 50 内高组对低组（按天中位数分组，按天配对差，单位 pp；y_open 买得到口径）==")
    say("%-14s %-30s %s" % ("特征", "说明", "筛选 差pp (NW t, 天)  开发 差pp (NW t, 天)  合并 差pp (t)  去2024-09 差pp (t)  按天AUC"))
    rows = []
    for f, (desc, binary) in FEATS.items():
        diff = paired_split(df, f, binary)
        if len(diff) < 25:
            say("%-14s %-30s 样本不够（%d 天）" % (f, desc, len(diff)))
            continue
        st = seg_stats(diff)
        auc = day_auc(df, f)
        say("%-14s %-30s %s  %.3f" % (f, desc, fmt(st), auc))
        rows.append({"feat": f, "desc": desc, "screen": st["screen"][0], "t_screen": st["screen"][1],
                     "dev": st["dev"][0], "t_dev": st["dev"][1], "all": st["all"][0], "t_all": st["all"][1],
                     "ex924": st["ex924"][0], "t_ex924": st["ex924"][1], "auc": auc})
    return pd.DataFrame(rows)


def section_b(df):
    say("\n== B. 日线条件固定后的增量（同一天同一层内按中位数分组；层 = 收在最高价 / 涨停收盘 / 名次段）==")
    strata = {
        "at_high": ("收在最高价 0/1", df["at_high"]),
        "lu_close": ("涨停收盘 0/1", df["lu_close"]),
        "rank_band": ("名次 1~10 / 11~50", (df["rank_ret20"] <= 10).astype(int)),
        "lu_close&at_high": ("涨停 x 收最高 四格", df["lu_close"] * 2 + df["at_high"]),
    }
    # 先印条件本身的命中（前 50 内，按天配对）
    for name, (desc, col) in strata.items():
        if name == "rank_band":
            continue
        d2 = df.assign(_s=col.astype(float))
        if name == "lu_close&at_high":
            for v, lab in ((0, "不涨停不收最高"), (1, "不涨停收最高"), (2, "涨停不收最高"), (3, "涨停收最高")):
                s = d2[d2["_s"] == v]
                say("  日线条件 %-12s：%6d 个股票-日，命中 筛选 %.1f%% 开发 %.1f%%" % (
                    lab, len(s), 100 * s.loc[s["seg"] == "screen", "y_open"].mean(), 100 * s.loc[s["seg"] == "dev", "y_open"].mean()))
            continue
        diff = paired_split(d2, "_s", True)
        say("  日线条件 %-12s 1 对 0：%s" % (desc, fmt(seg_stats(diff))))
    for f, (desc, binary) in FEATS.items():
        out = []
        for name, (sdesc, col) in strata.items():
            d2 = df.assign(_s=col.to_numpy())
            diff = paired_split(d2, f, binary, by=("date", "_s"))
            if len(diff) < 25:
                out.append("%s: n/a" % name)
                continue
            st = seg_stats(diff)
            out.append("%s: 筛 %+.2f(t%.1f) 开 %+.2f(t%.1f) 合 %+.2f(t%.1f)" % (
                name, st["screen"][0], st["screen"][1], st["dev"][0], st["dev"][1], st["all"][0], st["all"][1]))
        say("%-14s %-26s | %s" % (f, desc[:13], " | ".join(out)))


def section_b2(df):
    say("\n== B2. 只看「涨停收盘」那一层里的分时差异（封板质量）和「非涨停」那一层里的尾盘特征 ==")
    for lab, m in (("涨停收盘", df["lu_close"] == 1), ("非涨停", df["lu_close"] == 0)):
        d = df[m]
        say("  [%s] %d 个股票-日，%d 天" % (lab, len(d), d["date"].nunique()))
        for f in ("t_first_lu", "n_open", "lu_hold", "v_at_lu", "v_zero", "auc_share", "buy_share", "big_net", "c_vwap",
                  "ret_last30", "v_last30", "dd_intra", "t_high", "v_first30", "up_share"):
            diff = paired_split(d, f, FEATS[f][1])
            if len(diff) < 25:
                continue
            say("    %-12s %-28s %s" % (f, FEATS[f][0][:14], fmt(seg_stats(diff))))


def section_b3(df):
    """rv / dd_intra 这类「日内波动」信号，日线的振幅 range_u、amp20、vol60、市值已经知道多少？在这些量的当日三分位内再分。"""
    say("\n== B3. 日内波动类特征对日线已有量（当日振幅 / 20 日振幅 / 60 日波动 / 市值 / 当日换手）的增量：当日三分位层内再按中位数分 ==")
    def terc(col):
        r = df.groupby("date")[col].rank(pct=True)
        return np.minimum((r * 3).astype(int), 2)
    strata = {"range_u": terc("range_u"), "amp20": terc("amp20"), "vol60": terc("vol60"), "mcap": terc("mcap"), "turn1": terc("turn1"),
              "range_u&amp20": terc("range_u") * 3 + terc("amp20")}
    for name, col in strata.items():
        diff = paired_split(df.assign(_s=col.to_numpy()), "_s", False, by=("date",))
        say("  日线量 %-14s 高半对低半：%s" % (name, fmt(seg_stats(diff))))
    for f in ("rv", "dd_intra", "ret_pm", "buy_share_pm", "c_hi", "big_share", "tick_size", "ret_first30", "ret_am", "v_first30", "auc_share"):
        out = []
        for name, col in strata.items():
            diff = paired_split(df.assign(_s=col.to_numpy()), f, FEATS[f][1], by=("date", "_s"))
            st = seg_stats(diff)
            out.append("%s: 筛 %+.2f(t%.1f) 开 %+.2f(t%.1f) 合 %+.2f(t%.1f)" % (
                name, st["screen"][0], st["screen"][1], st["dev"][0], st["dev"][1], st["all"][0], st["all"][1]))
        say("%-12s | %s" % (f, " | ".join(out)))
    # rv 对 range_u 的回归残差（全样本一条线，只看 rv 里 range_u 解释不了的那部分）
    x = np.log(df["range_u"].to_numpy(float) + 1e-4)
    y = np.log(df["rv"].to_numpy(float) + 1e-6)
    m = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[m], y[m], 1)
    res = y - np.polyval(b, x)
    say("  log rv ~ log range_u：斜率 %.3f，R² %.3f" % (b[0], 1 - np.var(res[m]) / np.var(y[m])))
    diff = paired_split(df.assign(rv_res=res), "rv_res", False)
    say("  rv 残差（去掉当日振幅能解释的部分）高对低：%s" % fmt(seg_stats(diff)))
    for name in ("range_u", "amp20", "mcap"):
        diff = paired_split(df.assign(rv_res=res, _s=strata[name].to_numpy()), "rv_res", False, by=("date", "_s"))
        say("  rv 残差 在 %s 三分位层内：%s" % (name, fmt(seg_stats(diff))))


def precision_table(df, top_all, filters):
    say("\n== C. 当过滤条件（前 10 上；阈值事先定死，T 日可算）==")
    base = top_all[top_all["rank_ret20"] <= 10]
    for seg in ("screen", "dev", "all"):
        b = base if seg == "all" else base[base["seg"] == seg]
        d = df[df["rank_ret20"] <= 10]
        d = d if seg == "all" else d[d["seg"] == seg]
        nd = b["date"].nunique()
        say("  [%s] 基线前 10（含北交所）：%d 天，%d 票，精确率 %.2f%%；本样本（不含北交所、分时齐全）%d 票 精确率 %.2f%%" % (
            seg, nd, len(b), 100 * b["y_open"].mean(), len(d), 100 * d["y_open"].mean()))
        say("  %-34s %6s %7s %7s %8s %6s %7s" % ("过滤", "出票", "日均", "空榜天", "精确率", "命中", "按天差"))
        for name, fn in filters.items():
            m = fn(d)
            k = d[m]
            if len(k) == 0:
                say("  %-34s 0" % name)
                continue
            # 按天配对：过滤后精确率 - 当天基线前 10 精确率（当天有出票的天）
            pk = k.groupby("date")["y_open"].mean()
            pb = d.groupby("date")["y_open"].mean()
            dd = (pk - pb.reindex(pk.index)).dropna()
            say("  %-34s %6d %7.2f %7d %7.2f%% %6d %+6.2f (t %.2f)" % (
                name, len(k), len(k) / nd, nd - k["date"].nunique(), 100 * k["y_open"].mean(), int(k["y_open"].sum()),
                100 * dd.mean(), nw_t(dd)))


def precision_top50(df, filters):
    say("\n== C2. 前 50 先过滤再按名次取前 10 ==")
    for seg in ("screen", "dev", "all"):
        d = df if seg == "all" else df[df["seg"] == seg]
        nd = d["date"].nunique()
        base = d[d["rank_ret20"] <= 10].groupby("date")["y_open"].mean()
        say("  [%s] 本样本基线前 10 精确率 %.2f%%（%d 天）" % (seg, 100 * d[d["rank_ret20"] <= 10]["y_open"].mean(), nd))
        for name, fn in filters.items():
            k = d[fn(d)].sort_values(["date", "rank_ret20"])
            k = k[k.groupby("date").cumcount() < 10]
            if len(k) == 0:
                continue
            pk = k.groupby("date")["y_open"].mean()
            dd = (pk - base.reindex(pk.index)).dropna()
            say("  %-34s 出票 %5d 日均 %5.2f 精确率 %6.2f%% 按天差 %+6.2f (t %.2f) 名次中位 %.0f" % (
                name, len(k), len(k) / nd, 100 * k["y_open"].mean(), 100 * dd.mean(), nw_t(dd), k["rank_ret20"].median()))


def section_d(df):
    say("\n== D. 粗合成：留一月逻辑回归（日内特征 + 名次），前 50 内 ==")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [f for f, (_, b) in FEATS.items() if f not in ("t_first_lu", "n_open", "lu_hold", "v_at_lu", "lu_seal_close")]
    X = df[cols].copy()
    X["lu_seal_close"] = df["lu_seal_close"].fillna(0)
    X["n_open"] = df["n_open"].fillna(0)
    X["rank"] = df["rank_ret20"]
    X["lu_close"] = df["lu_close"]
    X["at_high"] = df["at_high"]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = df["y_open"].to_numpy()
    mon = df["date"].str[:7]
    for variant, use in (("只有 名次+日线条件", ["rank", "lu_close", "at_high"]), ("名次+日线条件+日内", list(X.columns))):
        p = np.full(len(df), np.nan)
        for m in sorted(mon.unique()):
            tr, te = (mon != m).to_numpy(), (mon == m).to_numpy()
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
            mdl.fit(X.loc[tr, use], y[tr])
            p[te] = mdl.predict_proba(X.loc[te, use])[:, 1]
        d2 = df.assign(_p=p)
        aucs, hits_m, hits_b = [], [], []
        for _, g in d2.groupby("date"):
            yy = g["y_open"].to_numpy()
            if 0 < yy.sum() < len(yy):
                aucs.append(roc_auc_score(yy, g["_p"]))
            hits_m.append(g.nlargest(10, "_p")["y_open"].mean())
            hits_b.append(g[g["rank_ret20"] <= 10]["y_open"].mean())
        dd = pd.Series(np.array(hits_m) - np.array(hits_b), index=sorted(d2["date"].unique()))
        st = seg_stats(dd)
        say("  %-22s 按天 AUC %.3f  模型前 10 命中 %.2f%% 对 ret20 前 10 %.2f%%  %s" % (
            variant, np.mean(aucs), 100 * np.mean(hits_m), 100 * np.nanmean(hits_b), fmt(st)))


GATE_FEATS = ["c_vwap", "ret_last30", "ret_last60", "ret_first30", "ret_am", "ret_pm", "dd_intra", "c_hi", "t_high", "t_low",
              "rv", "up_share", "v_first30", "v_last30", "v_last5", "v_am", "v_top10", "v_zero", "v_above_vwap",
              "buy_share", "buy_share_pm", "big_net", "big_share", "auc_share", "tick_size",
              "t_first_lu", "n_open", "lu_hold", "v_at_lu"]
QS = (0.5, 0.7, 0.8, 0.9, 0.95)


def section_e(df):
    """绝对门槛：阈值取筛选集的分位数（事先定的分位，不看标签挑），两个方向，在前 10 / 前 50 上看精确率和出票量；开发集只套同一阈值。"""
    say("\n== E. 绝对门槛前沿（阈值 = 筛选集分位数；过线才出，可以空榜）==")
    rows = []
    scr = df[df["seg"] == "screen"]
    nd = {s: max(1, df.loc[df["seg"] == s, "date"].nunique()) for s in ("screen", "dev")}
    for K_, lab in ((10, "前10"), (50, "前50")):
        u = df[df["rank_ret20"] <= K_]
        for s in ("screen", "dev"):
            us = u[u["seg"] == s]
            rows.append({"universe": lab, "feat": "(基线)", "dir": "", "q": np.nan, "thr": np.nan, "seg": s,
                         "n": len(us), "hits": int(us["y_open"].sum()), "prec": us["y_open"].mean(),
                         "per_day": len(us) / nd[s], "days_nonempty": us["date"].nunique() / nd[s]})
        for ref_name, m in (("涨停收盘", u["lu_close"] == 1), ("收在最高价", u["at_high"] == 1), ("涨停且没开板", (u["lu_close"] == 1) & (u["n_open"] == 0))):
            for s in ("screen", "dev"):
                us = u[m & (u["seg"] == s)]
                rows.append({"universe": lab, "feat": ref_name, "dir": "日线", "q": np.nan, "thr": np.nan, "seg": s,
                             "n": len(us), "hits": int(us["y_open"].sum()), "prec": us["y_open"].mean() if len(us) else np.nan,
                             "per_day": len(us) / nd[s], "days_nonempty": us["date"].nunique() / nd[s]})
        for f in GATE_FEATS:
            x = scr.loc[scr["rank_ret20"] <= K_, f].to_numpy(float)
            x = x[np.isfinite(x)]
            if len(x) < 100:
                continue
            for q in QS:
                for d_, thr in ((">=", np.quantile(x, q)), ("<=", np.quantile(x, 1 - q))):
                    m = (u[f] >= thr) if d_ == ">=" else (u[f] <= thr)
                    for s in ("screen", "dev"):
                        us = u[m & (u["seg"] == s)]
                        rows.append({"universe": lab, "feat": f, "dir": d_, "q": q, "thr": thr, "seg": s,
                                     "n": len(us), "hits": int(us["y_open"].sum()),
                                     "prec": us["y_open"].mean() if len(us) else np.nan,
                                     "per_day": len(us) / nd[s], "days_nonempty": us["date"].nunique() / nd[s]})
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results_gate.csv", index=False)
    R["q"] = R["q"].fillna(-1)
    R["thr"] = R["thr"].fillna(-1)
    keys = ["universe", "feat", "dir", "q", "thr"]
    vals = ["n", "hits", "prec", "per_day", "days_nonempty"]
    Rs = R[R["seg"] == "screen"][keys + vals].rename(columns={v: v + "_screen" for v in vals})
    Rd = R[R["seg"] == "dev"][keys + vals].rename(columns={v: v + "_dev" for v in vals})
    W = Rs.merge(Rd, on=keys, how="outer")
    for v in ("n", "hits"):
        W[v + "_screen"] = W[v + "_screen"].fillna(0).astype(int)
        W[v + "_dev"] = W[v + "_dev"].fillna(0).astype(int)
    for v in ("per_day", "days_nonempty"):
        W[v + "_screen"] = W[v + "_screen"].fillna(0)
        W[v + "_dev"] = W[v + "_dev"].fillna(0)
    for lab in ("前10", "前50"):
        w = W[W["universe"] == lab].copy()
        base = w[w["feat"] == "(基线)"].iloc[0]
        say("  [%s] 基线精确率 筛选 %.2f%%（%d 票）开发 %.2f%%（%d 票）" % (
            lab, 100 * base["prec_screen"], base["n_screen"], 100 * base["prec_dev"], base["n_dev"]))
        say("  %-14s %-3s %-5s %9s | 筛选 精确率 命中/出票 日均 | 开发 精确率 命中/出票 日均 | 两段都高于基线 +2pp?" % ("特征", "向", "分位", "阈值"))
        w["gain_s"] = w["prec_screen"] - base["prec_screen"]
        w["gain_d"] = w["prec_dev"] - base["prec_dev"]
        w["both"] = (w["gain_s"] >= 0.02) & (w["gain_d"] >= 0.02)
        w["min_gain"] = np.minimum(w["gain_s"], w["gain_d"])
        # 先印日线参照，再印两段最小增益最高的 25 条（出票日均 >= 0.3 的），再印全部两段都 +2pp 的
        for _, r in w[w["dir"] == "日线"].iterrows():
            say("  %-14s %-3s %-5s %9s | %6.2f%% %4d/%-5d %5.2f | %6.2f%% %4d/%-5d %5.2f | %s" % (
                r["feat"], "", "", "", 100 * r["prec_screen"], r["hits_screen"], r["n_screen"], r["per_day_screen"],
                100 * r["prec_dev"], r["hits_dev"], r["n_dev"], r["per_day_dev"], "是" if r["both"] else ""))
        cand = w[(w["dir"].isin([">=", "<="])) & (w["per_day_screen"] >= 0.3) & (w["per_day_dev"] >= 0.3)]
        say("  -- 两段最小增益最高的 25 条（日均出票 >= 0.3）--")
        for _, r in cand.sort_values("min_gain", ascending=False).head(25).iterrows():
            say("  %-14s %-3s %-5.2f %9.4f | %6.2f%% %4d/%-5d %5.2f | %6.2f%% %4d/%-5d %5.2f | %s" % (
                r["feat"], r["dir"], r["q"], r["thr"], 100 * r["prec_screen"], r["hits_screen"], r["n_screen"], r["per_day_screen"],
                100 * r["prec_dev"], r["hits_dev"], r["n_dev"], r["per_day_dev"], "是" if r["both"] else ""))
        say("  两段都比基线高 >= 2pp 的门槛数：%d / %d（日均出票 >= 0.3 的里 %d / %d）" % (
            int(w.loc[w["dir"].isin([">=", "<="]), "both"].sum()), int(w["dir"].isin([">=", "<="]).sum()),
            int(cand["both"].sum()), len(cand)))


def main():
    F = pd.read_parquet(HERE / "feat.parquet")
    top = pd.read_parquet(ROOT / "research" / "altdata" / "top60.parquet", columns=["code", "date", "board", "rank_ret20", "y_open"])
    top = top[top["rank_ret20"] <= 50].copy()
    top["seg"] = np.where(top["date"] < "2024-06-01", "screen", "dev")
    top = top[top["date"].isin(F["date"].unique())]
    df = F[F["ok"] & np.isfinite(F["y_open"].to_numpy(float)) & (F["rank_ret20"] <= 50)].copy()
    pool = pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "exp" / "polish" / "pool.parquet",
                           columns=["code", "date", "amp20", "vol60", "turn20"], filters=[("date", "in", list(df["date"].unique()))])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    df = df.merge(pool, on=["code", "date"], how="left")
    say("样本：%d 个股票-日（前 50 含北交所 %d），%d 天；筛选 %d 天 / 开发 %d 天；前 50 命中 筛选 %.2f%% 开发 %.2f%%" % (
        len(df), len(top), df["date"].nunique(), df.loc[df["seg"] == "screen", "date"].nunique(),
        df.loc[df["seg"] == "dev", "date"].nunique(),
        100 * df.loc[df["seg"] == "screen", "y_open"].mean(), 100 * df.loc[df["seg"] == "dev", "y_open"].mean()))
    say("月份：", sorted(df["date"].str[:7].unique()))
    say("触及涨停 %.1f%%，涨停收盘 %.1f%%，收在最高价 %.1f%%，一字 %.1f%%" % (
        100 * df["lu_touch"].mean(), 100 * df["lu_close"].mean(), 100 * df["at_high"].mean(), 100 * df["yizi_today"].mean()))
    tab = section_a(df)
    tab.to_csv(HERE / "results_a.csv", index=False)
    section_b(df)
    section_b2(df)
    section_b3(df)
    (HERE / "results_diag.txt").write_text("\n".join(LINES), encoding="utf-8")
    LINES.clear()
    # 固定阈值过滤（事先定：符号 / 整数 / 常识数，不从标签调）
    filters = {
        "收盘在均价之上 c_vwap>0": lambda d: d["c_vwap"] > 0,
        "收盘在均价之上 >1%": lambda d: d["c_vwap"] > 0.01,
        "尾盘 30 分钟不跌 ret_last30>=0": lambda d: d["ret_last30"] >= 0,
        "尾盘 30 分钟涨 >1%": lambda d: d["ret_last30"] > 0.01,
        "日内回撤 >-5%": lambda d: d["dd_intra"] > -0.05,
        "日内回撤 >-3%": lambda d: d["dd_intra"] > -0.03,
        "高点在下午 t_high>0.5": lambda d: d["t_high"] > 0.5,
        "高点在最后 30 分钟": lambda d: d["t_high"] > 210 / 240,
        "主动买占比 >0.5": lambda d: d["buy_share"] > 0.5,
        "14:30 后主动买 >0.5": lambda d: d["buy_share_pm"] > 0.5,
        "大单净买 >0": lambda d: d["big_net"] > 0,
        "尾盘 30 分钟量占比 <20%": lambda d: d["v_last30"] < 0.20,
        "早盘 30 分钟量占比 >30%": lambda d: d["v_first30"] > 0.30,
        "均价之上成交量占比 >0.5": lambda d: d["v_above_vwap"] > 0.5,
        "上涨分钟占比 >0.5": lambda d: d["up_share"] > 0.5,
        "涨停收盘": lambda d: d["lu_close"] == 1,
        "涨停收盘且没开过板": lambda d: (d["lu_close"] == 1) & (d["n_open"] == 0),
        "涨停收盘且 11:30 前封": lambda d: (d["lu_close"] == 1) & (d["t_first_lu"] <= 0.5),
        "涨停收盘且开板 <=1 次": lambda d: (d["lu_close"] == 1) & (d["n_open"] <= 1),
        "非涨停且收盘在均价上": lambda d: (d["lu_close"] == 0) & (d["c_vwap"] > 0),
        "收在最高价（日线）": lambda d: d["at_high"] == 1,
        "触板没封住 剔除": lambda d: ~((d["lu_touch"] == 1) & (d["lu_close"] == 0)),
        "c_vwap>0 且 大单净买>0": lambda d: (d["c_vwap"] > 0) & (d["big_net"] > 0),
        "c_vwap>0 且 ret_last30>=0": lambda d: (d["c_vwap"] > 0) & (d["ret_last30"] >= 0),
    }
    precision_table(df, top, filters)
    precision_top50(df, filters)
    section_d(df)
    (HERE / "results_filter.txt").write_text("\n".join(LINES), encoding="utf-8")
    LINES.clear()
    section_e(df)
    (HERE / "results_gate.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
