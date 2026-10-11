"""
实验 21 · 第 7 项「本质提升」候选（计划 2026-10-10）。走 evalkit，预登记，过线才上。

诊断（实验 20，exp_diag）给的线索：
  · 前 50 名里 97 分命中 11.7%，98 分 15.3%，99 分 19.9%：分数线 97 放进来一半 11% 的票
  · 北交所按买得到口径 19.0%（最强），主板 10.8%，科创 7.4%；板块系数却是按收盘口径估的，
    把北交所压到 0.886
  · 连续够格 2 天 21.0%、3 天 23.7%，对首日 14.6%
  · 前 10 名额次日开盘起算 20 日最高涨幅中位 +15%、75 分位 +33%：标签 +50% 太稀，+20% 是
    一半以上名额都能到的目标

臂
--
  事后（不重训，用 w1 的现状预测值）：
    thr98        分数线 97 -> 98（其余不变）
    adj_open     收缩板块系数按买得到口径的名额估（滚动、只用过去月份），其余不变
    thr98_open   两者都上
  要训练：
    y20          训练目标换成「次日开盘起算 20 根最高涨幅 ≥ +20%」（评估仍看 +50% 买得到口径）
    y20_blend    y20 模型和现状模型的名次平均（同一天两份名次取均值）
    pool_small   只在流通市值当日排名前 75%（剔掉最大的四分之一）的票上训练和选票
    lr_blend     现状 LightGBM 和线性模型（L0）的名次平均（异质集成）

过线：前 10 买得到口径按天 t ≥ 2.5，最差三个月不比现状差，空榜不翻倍（evalkit.GATE）。
生产规则的改动（分数线、板块系数口径）还要看名额数和空榜天数，不能只看命中。

    python src/breakout/exp_w3.py --post      事后三个臂
    python src/breakout/exp_w3.py --fit       四个训练臂（约 1 小时）
    python src/breakout/exp_w3.py --analyze
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import board_adj as BA  # noqa: E402
import evalkit as K     # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
NAME = "w3_essence"
BASE_NAME = "w1_scale_avg"
HYP = ("thr98 / adj_open / thr98_open 在不重训的情况下抬高生产规则的买得到命中且空榜不翻倍；"
       "y20 / pool_small / lr_blend 之一在前 10 买得到口径上过线（t ≥ 2.5，最差三月不差，空榜不翻倍）")
ARMS = {"thr98": "分数线 98", "adj_open": "板块系数按买得到口径估", "thr98_open": "两者都上",
        "y20": "训练目标 +20%", "y20_blend": "y20 和现状名次平均",
        "pool_small": "剔掉市值最大四分之一再训练和选票", "lr_blend": "树模型和线性模型名次平均"}


# ---------------------------------------------------------------
#  事后臂
# ---------------------------------------------------------------
def load_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    top = pd.read_parquet(K.EXP / BASE_NAME / "top_base.parquet")
    allp = pd.read_parquet(K.EXP / BASE_NAME / "all_base.parquet")
    return top, allp


def rescore_with_adj(top: pd.DataFrame, allp: pd.DataFrame, hit_col: str) -> pd.DataFrame:
    """按滚动收缩板块系数（hit_col 口径）重算 p、分数、名次。

    分数刻度：w1 没落每月的分位点，但 top 表里每月有 ~4000 对（预测值, 分数），分数是
    预测值的单调阶梯函数，用观测对反推（取不大于 p 的最近观测点的分数），95~100 分那段
    每月有上千个样本点，够准。top 只有每天前 200，系数 ≤1.4 不会把 200 名外的票抬进前 10。"""
    t = top.copy()
    t["month"] = t["date"].str[:7]
    parts, hist = [], []
    for m, g in t.groupby("month", sort=True):
        pairs = g[["p", "score"]].sort_values("p")
        px = pairs["p"].to_numpy(float)
        sx = np.maximum.accumulate(pairs["score"].to_numpy(float))
        amap = BA.factors_before(pd.concat(hist, ignore_index=True), m) if hist else {}
        adj = g["board"].map(amap).fillna(1.0).to_numpy(float) if amap else np.ones(len(g))
        g = g.assign(p=g["p"].to_numpy(float) * adj)
        idx = np.clip(np.searchsorted(px, g["p"].to_numpy(float), side="right") - 1, 0, len(px) - 1)
        g["score"] = np.where(g["p"].to_numpy(float) < px[0], 0.0, sx[idx])
        g = g.sort_values(["date", "p"], ascending=[True, False])
        g["rank"] = g.groupby("date").cumcount() + 1
        parts.append(g)
        prod = g[(g["score"] >= 97) & (g["rank"] <= 10)]
        hist.append(prod[["date", "board", hit_col]].rename(columns={hit_col: "y_up"}))
    return pd.concat(parts, ignore_index=True)


def post() -> dict:
    import daily as D
    top, allp = load_base()
    res = {}
    base_prod = K.prod_picks(top)
    arms = {
        "base": top,
        "thr98": top.assign(score=np.where(top["score"] >= 98, top["score"], 0)),
        "adj_open": rescore_with_adj(top, allp, "y_open"),
    }
    arms["thr98_open"] = arms["adj_open"].assign(
        score=np.where(arms["adj_open"]["score"] >= 98, arms["adj_open"]["score"], 0))
    # 对照：现状的收缩系数也是滚动估的，用 y_up 重算一遍应当和 base 接近（自检）
    arms["adj_close"] = rescore_with_adj(top, allp, "y_up")
    print("事后臂（生产规则 ≥%d 分前 %d，买得到口径）" % (D.SCORE_MIN, D.CAP_A))
    for name, t in arms.items():
        prod = K.prod_picks(t)
        c = K.clustered(prod, "y_open")
        cc = K.clustered(prod, "y_up")
        days = t["date"].nunique()
        mo = K.monthly(prod, "y_open")
        res[name] = {"n": c["n"], "hit_open": c["hit"], "se": c["se"], "hit_close": cc["hit"],
                     "empty": int(days - prod["date"].nunique()),
                     "per_day": c["n"] / days, "month_worst": float(mo.min()) if len(mo) else None,
                     "month_median": float(mo.median()) if len(mo) else None}
        cmp_ = K.compare(prod.assign(rank=1), base_prod.assign(rank=1), y="y_open", k=1)
        res[name]["vs_base"] = {k: cmp_[k] for k in ("diff_day", "t_day", "wins", "months")}
        print("  %-10s n=%4d 命中 %5.2f%% ±%.2f（收盘 %5.2f%%） 空榜 %3d 每天 %.1f 月中位 %5.1f%% 最差 %4.1f%% | 对现状按天 %+.2f t=%.2f 赢 %d/%d" % (
            name, c["n"], 100 * c["hit"], 100 * c["se"], 100 * cc["hit"], res[name]["empty"],
            res[name]["per_day"], 100 * (res[name]["month_median"] or 0), 100 * (res[name]["month_worst"] or 0),
            100 * cmp_["diff_day"], cmp_["t_day"], cmp_["wins"], cmp_["months"]))
    K.record(NAME, {"post": res})
    return res


# ---------------------------------------------------------------
#  训练臂
# ---------------------------------------------------------------
def fit_predict(tr, te, feats, month, state):
    import model as M
    out = {}
    # y20：目标换成 +20%（买得到口径），评估仍是 +50%
    tr20 = tr[np.isfinite(tr["y20"])]
    trs20 = M.stratified_sample(tr20, "y20")
    m20 = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs20, feats, "y20")
    p20 = m20.predict_proba(te)
    out["y20"] = (p20, m20.predict_proba(trs20 if len(trs20) <= 600_000 else trs20.sample(600_000, random_state=1)))
    # 现状（本月重训一份，和 y20 同一批测试行，名次平均要用）
    trs = M.stratified_sample(tr, "y_up")
    m0 = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, feats, "y_up")
    p0 = m0.predict_proba(te)
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    r0 = m0.predict_proba(ref)
    out["base_re"] = (p0, r0)
    # 名次平均：同一天内各自的名次取均值，再换回「越大越好」的分数
    d = te[["date"]].copy()
    d["r0"] = pd.Series(p0, index=te.index).groupby(d["date"]).rank(ascending=False)
    d["r20"] = pd.Series(p20, index=te.index).groupby(d["date"]).rank(ascending=False)
    blend = 1.0 / (1.0 + (d["r0"] + d["r20"]) / 2.0)
    out["y20_blend"] = (blend.to_numpy(), np.quantile(blend, np.linspace(0, 1, 101)))
    # 线性模型 + 树模型
    l0 = M.L0Logistic().fit(trs, feats, "y_up")
    pl = l0.predict_proba(te)
    d["rl"] = pd.Series(pl, index=te.index).groupby(d["date"]).rank(ascending=False)
    lr = 1.0 / (1.0 + (d["r0"] + d["rl"]) / 2.0)
    out["lr_blend"] = (lr.to_numpy(), np.quantile(lr, np.linspace(0, 1, 101)))
    # 候选池：剔掉当日流通市值最大的四分之一
    small_tr = tr[tr["mcap_pct"] <= 0.75]
    trs_s = M.stratified_sample(small_tr, "y_up")
    ms = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs_s, feats, "y_up")
    ps = ms.predict_proba(te)
    ps = np.where(te["mcap_pct"].to_numpy() <= 0.75, ps, 0.0)   # 大票不进候选池
    out["pool_small"] = (ps, ms.predict_proba(trs_s if len(trs_s) <= 600_000 else trs_s.sample(600_000, random_state=1)))
    return out


def fit() -> None:
    import buyable as B
    df = K.load_frame(with_confirm=False)
    lab = B.open_labels()[["code", "date", "r20_open"]]
    if "r20_open" not in df.columns:
        df = df.merge(lab, on=["code", "date"], how="left")
    df["y20"] = np.where(np.isfinite(df["r20_open"]), (df["r20_open"] >= 0.20).astype(float), np.nan)
    df["mcap_pct"] = df.groupby("date")["float_mcap"].rank(pct=True).astype("float32")
    K.walk_forward(df, K.months_in(K.DEV), fit_predict, NAME)


def analyze() -> dict:
    base = pd.read_parquet(K.EXP / BASE_NAME / "top_base.parquet")
    tops = {"base": base}
    for arm in ("base_re", "y20", "y20_blend", "lr_blend", "pool_small"):
        f = K.EXP / NAME / f"top_{arm}.parquet"
        if f.exists():
            tops[arm] = pd.read_parquet(f)
    res = {}
    for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
        print("[%s]" % lbl)
        for arm, t in tops.items():
            m = K.metrics(t, since)
            res[f"{arm}|{lbl}"] = m
            print(K.fmt_metrics(arm, m))
        for arm in tops:
            if arm == "base":
                continue
            c = K.compare(tops[arm], base, since=since)
            res[f"cmp_{arm}|{lbl}"] = c
            print(K.fmt_compare(arm, c))
        print()
    K.record(NAME, {"fit": res})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    if a.post:
        post()
    if a.fit:
        fit()
    if a.analyze:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
