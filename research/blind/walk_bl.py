"""逐月滚动出样本外分数（research/blind 的几种变体），walk_gt.py 同一条流水（36 个月训练、净化、负样本抽样 15% 权重补回、浅树、种子 7）。

    PYTHONIOENCODING=utf-8 python research/blind/walk_bl.py --name v2_lag20 --mode lag     # V2：特征取 T−20 那一行（lag 根数可改 --lag）
    ... --mode fe        L2：init_score = logit(当天候选池基准率)，预测输出 raw margin（当天内排序用）
    ... --mode reg       L3：回归 clip(log1p(r20_open), −0.5, 1.1)
    ... --mode q90       L3：分位回归 alpha 0.9，同一目标
    ... --mode multi     L3：r20_open > 0.3 / 0.5 / 0.8 三个二分类，分数 = 三个当天内分位的平均
    ... --shuffle day    标签每天内置换（自检）
输出 research/blind/scores/<name>.npy（和 research/gate/table.parquet 同序，没打分的行 NaN）和 <name>.json。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common_bl as B

G = B.G
LGB = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=10.0, n_estimators=300, verbose=-1, num_threads=8)


def purge_cut(dates_u: np.ndarray, month: str, gap: int = 20) -> str:
    first = int(np.searchsorted(dates_u, month + "-01"))
    return str(dates_u[max(first - gap, 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="lag", choices=["lag", "fe", "reg", "q90", "multi", "plain"])
    ap.add_argument("--lag", type=int, default=20)
    ap.add_argument("--feats", default="price", choices=["price", "price_nomkt"])
    ap.add_argument("--neg", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle", default="none", choices=["none", "day"])
    ap.add_argument("--months", default="%s,%s" % G.TEST_MONTHS)
    a = ap.parse_args()
    B.PL.wait_prod_window()
    t0 = time.time()
    feats = G.FEATS_PRICE if a.feats == "price" else [c for c in G.FEATS_PRICE if c not in G.DAY_COLS]
    cols = ["code", "date", "end20", "y_open", "r20_open", "nbar"] + [c for c in feats if c not in ("nbar",)]
    t = G.load_table(cols)
    t["code"] = t["code"].astype(str).str.zfill(6)
    n = len(t)
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    ym = dates.astype("U7")
    y = t["y_open"].to_numpy(float)
    r20 = t["r20_open"].to_numpy(float)
    X = t[feats].to_numpy(np.float32)
    if a.mode == "lag":
        ptr = B.lag_pointer(t, a.lag)
        has = ptr >= 0
        # 自检：T−lag 行日期 < T 且根数差 == lag
        nb = t["nbar"].to_numpy(np.int64)
        assert (dates[ptr[has]] < dates[has]).all() and (nb[has] - nb[ptr[has]] == a.lag).all()
        Xl = np.full_like(X, np.nan)
        Xl[has] = X[ptr[has]]
        # 自检：特征逐行等于 T−lag 行的特征（抽 1000 行）
        rng_c = np.random.default_rng(1)
        chk = rng_c.choice(np.flatnonzero(has), 1000, replace=False)
        assert np.array_equal(np.nan_to_num(Xl[chk]), np.nan_to_num(X[ptr[chk]]))
        X = Xl
        uni = has
        B.say("lag 模式：特征取 T−%d 行，可用 %.1f%%；自检通过" % (a.lag, 100 * has.mean()))
    else:
        uni = np.ones(n, bool)
    # 当天候选池基准率（fe 模式的偏移；用全部候选池行，不是抽样后的）
    day_base = pd.Series(y).groupby(dates).transform("mean").to_numpy()
    rng0 = np.random.default_rng(a.seed)
    if a.shuffle == "day":
        # y 和 r20 一起按天置换（保持配对）
        perm = np.arange(n)
        for _, idx in pd.Series(np.arange(n)).groupby(dates).indices.items():
            perm[idx] = idx[rng0.permutation(len(idx))]
        y, r20 = y[perm], r20[perm]
    del t
    dates_u = np.unique(dates)
    months = G.months_between(*a.months.split(","))
    s = np.full(n, np.nan, np.float32)
    log = []
    import lightgbm as lgb
    B.say("表 %d 行 %d 特征 模式 %s %.0fs；月份 %s..%s" % (n, len(feats), a.mode, time.time() - t0, months[0], months[-1]))
    for m in months:
        t1 = time.time()
        cut = purge_cut(dates_u, m)
        lo = G.month_shift(m, -12 * G.TRAIN_YEARS) + "-01"
        tm = (dates < cut) & (end20 < m + "-01") & (dates >= lo) & np.isfinite(y) & uni
        test = (ym == m) & uni
        if test.sum() == 0 or tm.sum() < 1000:
            continue
        rng = np.random.default_rng(a.seed * 1000 + int(m[:4]) * 12 + int(m[5:7]))
        pos = tm & (y > 0.5)
        neg = tm & (y < 0.5)
        keep_neg = neg & (rng.random(n) < a.neg)
        idx = np.flatnonzero(pos | keep_neg)
        w = np.where(y[idx] > 0.5, 1.0, 1.0 / a.neg)
        Xtr = X[idx]
        if a.mode in ("lag", "plain"):
            clf = lgb.LGBMClassifier(objective="binary", random_state=a.seed, **LGB)
            clf.fit(Xtr, y[idx], sample_weight=w)
            s[test] = clf.predict_proba(X[test])[:, 1]
        elif a.mode == "fe":
            off = np.log(np.clip(day_base[idx], 1e-4, 0.5) / (1 - np.clip(day_base[idx], 1e-4, 0.5)))
            clf = lgb.LGBMClassifier(objective="binary", random_state=a.seed, **LGB)
            clf.fit(Xtr, y[idx], sample_weight=w, init_score=off)
            s[test] = clf.predict(X[test], raw_score=True)
        elif a.mode in ("reg", "q90"):
            tgt = np.clip(np.log1p(np.clip(r20[idx], -0.99, None)), -0.5, 1.1)
            if a.mode == "reg":
                mdl = lgb.LGBMRegressor(objective="regression", random_state=a.seed, **LGB)
            else:
                mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.9, random_state=a.seed, **LGB)
            mdl.fit(Xtr, tgt, sample_weight=w)
            s[test] = mdl.predict(X[test])
        elif a.mode == "multi":
            # 三个阈值各训一个，分数 = 当天内分位的平均；负样本抽样按 50% 标签抽的那份（三个共用 idx / w）
            acc = np.zeros(int(test.sum()))
            td = dates[test]
            for thr in (0.3, 0.5, 0.8):
                yy = (r20[idx] > thr).astype(float)
                if yy.sum() < 50:
                    continue
                clf = lgb.LGBMClassifier(objective="binary", random_state=a.seed, **LGB)
                clf.fit(Xtr, yy, sample_weight=w)
                p = clf.predict_proba(X[test])[:, 1]
                acc += pd.Series(p).groupby(td).rank(pct=True).to_numpy()
            s[test] = acc / 3.0
        log.append({"month": m, "train_rows": int(len(idx)), "pos": int(pos.sum()), "test_rows": int(test.sum()),
                    "cut": cut, "sec": round(time.time() - t1, 1)})
        B.say("%s 训练 %d 行（正 %d）测试 %d 行 净化线 %s %.0fs" % (m, len(idx), pos.sum(), test.sum(), cut, time.time() - t1))
    np.save(B.SCORES / (a.name + ".npy"), s)
    (B.SCORES / (a.name + ".json")).write_text(json.dumps({"args": vars(a), "feats": feats, "months": log,
                                                           "total_sec": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    B.say("-> %s  %.0fs" % (B.SCORES / (a.name + ".npy"), time.time() - t0))
    B.progress("样本外分数 %s：模式 %s 特征 %s(%d) 打乱 %s，%d 个月 %.0f 分钟" % (
        a.name, a.mode, a.feats, len(feats), a.shuffle, len(log), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
