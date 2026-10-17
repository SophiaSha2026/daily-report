"""
天花板估计（不是臂）：日线 + 现有特征在 ret20 前 50 内还能挑出多少。

    PYTHONIOENCODING=utf-8 python research/polish/ceiling.py

1 名次参照：ret20 第 1 名 / 前 3 名的命中 = 「把 10 只都挑成第 1 名那样」的参照
2 同期拟合上界（有时间泄漏）：在 ret20 前 50 上，同一段里按股票 5 折交叉拟合 LightGBM（扩展特征 + 名次），
  同期的 regime 信息留着、股票记忆去掉 -> 日线特征在这个段上的乐观上界
3 全样本内拟合（完全过拟合上界，只作参照）
4 双正态：前 50 内要把前 10 命中从 x 提到 y 需要的段内 AUC
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P
import rerank as R


def top10_from(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    o = df[["date", "code", "board", "y_open", "y_up"]].copy()
    t = P.rank_by(o, p, keep=10)
    return P.hit_series(t, 10)


def main():
    import lightgbm as lgb
    df = R.load_topk(50)
    df = df[(df["date"] >= "2022-01-01") & np.isfinite(df["y_open"])].reset_index(drop=True)
    df["seg"] = P.seg_of(df["date"])
    cols = R.feat_list(df, "ext")
    X = np.nan_to_num(df[cols].to_numpy(np.float32), nan=0.0)
    y = df["y_open"].to_numpy(float)
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))
    for seg in ("screen", "dev"):
        m = (df["seg"] == seg).to_numpy()
        d = df[m].reset_index(drop=True)
        say("== %s（%d 天）==" % (seg, d["date"].nunique()))
        for k in (1, 3, 5, 10, 20, 50):
            say("  ret20 前 %2d 命中 %.2f%%" % (k, 100 * d[d["rank_ret20"] <= k].groupby("date")["y_open"].mean().mean()))
        nh = d.groupby("date")["y_open"].sum().mean()
        say("  前 50 里每天平均 %.2f 只命中 -> 完美挑 10 只的绝对上限 %.1f%%" % (nh, 100 * min(nh, 10) / 10))
        base = top10_from(d, -d["rank_ret20"].to_numpy(float))
        # 股票 5 折（同期）
        codes = d["code"].unique()
        rng = np.random.default_rng(3)
        fold = pd.Series(rng.integers(0, 5, len(codes)), index=codes)
        f = d["code"].map(fold).to_numpy()
        p = np.zeros(len(d))
        params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100, "feature_fraction": 0.7,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7, "num_threads": 8}
        for k in range(5):
            tr, te = f != k, f == k
            b = lgb.train(params, lgb.Dataset(X[m][tr], label=y[m][tr]), num_boost_round=200)
            p[te] = b.predict(X[m][te])
        h = top10_from(d, p)
        dd = (h - base).dropna()
        say("  同期按股票 5 折交叉拟合（时间泄漏，乐观上界）：前 10 命中 %.2f%% 对 ret20 %.2f%%，差 %+.2fpp（NW t %.2f）" % (
            100 * h.mean(), 100 * base.mean(), 100 * dd.mean(), P.nw_t(dd)))
        # 同期按时间 5 折（块）：去掉同一天的信息
        mon = d["date"].str[:7]
        ms = sorted(mon.unique())
        fold_m = {mm: i % 5 for i, mm in enumerate(ms)}
        f2 = mon.map(fold_m).to_numpy()
        p2 = np.zeros(len(d))
        for k in range(5):
            tr, te = f2 != k, f2 == k
            b = lgb.train(params, lgb.Dataset(X[m][tr], label=y[m][tr]), num_boost_round=200)
            p2[te] = b.predict(X[m][te])
        h2 = top10_from(d, p2)
        dd2 = (h2 - base).dropna()
        say("  同期按月 5 折交叉拟合（月份交错，含未来月）：前 10 命中 %.2f%%，差 %+.2fpp（NW t %.2f）" % (100 * h2.mean(), 100 * dd2.mean(), P.nw_t(dd2)))
        b = lgb.train(params, lgb.Dataset(X[m], label=y[m]), num_boost_round=200)
        h3 = top10_from(d, b.predict(X[m]))
        say("  全样本内拟合（纯过拟合参照）：前 10 命中 %.2f%%" % (100 * h3.mean()))
        # 双正态
        from scipy.stats import norm
        pi = d["y_open"].mean()
        q = 10 / 50
        say("  前 50 内基准 %.2f%%；双正态下前 10（20%% 截取）命中 对应 AUC：" % (100 * pi))
        for auc in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            dprime = np.sqrt(2) * norm.ppf(auc)
            # 混合分布的 80% 分位截取点，数值求解
            xs = np.linspace(-5, 8, 4000)
            cdf = (1 - pi) * norm.cdf(xs) + pi * norm.cdf(xs - dprime)
            c = xs[np.searchsorted(cdf, 1 - q)]
            hit = pi * (1 - norm.cdf(c - dprime)) / q
            say("    AUC %.2f -> 前 10 命中 %.1f%%" % (auc, 100 * hit))
    (P.HERE / "results_ceiling.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
