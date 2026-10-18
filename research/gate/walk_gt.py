"""逐月滚动出样本外分数（criteria.txt 第 1 节 a / b）。

    PYTHONIOENCODING=utf-8 python research/gate/walk_gt.py --name lgb_price --model lgb --feats price [--neg 0.15]
        [--universe all|top300] [--depth shallow|deep] [--seed 7] [--shuffle none|global|day]

输出 research/gate/scores/<name>.npy（float32，和 table.parquet 同序，没打分的行 NaN）和 <name>.json（配置 + 每月耗时）。
训练：每个测试月 m 用 m 之前 36 个月、按标签窗口净化的行（common_rc.train_mask 同一条规则），负样本按 --neg 抽样、权重补回。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common_gt as G

FEATS = {"price": G.FEATS_PRICE, "alt": G.FEATS_PRICE + G.ALT_COLS, "lr": G.FEATS_LR,
         "price_nomkt": [c for c in G.FEATS_PRICE if c not in G.DAY_COLS],
         "core": ["lret20", "rank_ret20", "lvol60", "lturn20", "amp20", "lmcap", "board_i", "nlu20", "dist_hi60", "turn1", "ret5"]}
LGB_PARAMS = {
    "shallow": dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, n_estimators=300, verbose=-1, num_threads=8),
    "deep": dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, n_estimators=500, verbose=-1, num_threads=8),
}


def purge_cut(dates_u: np.ndarray, month: str, gap: int = 20) -> str:
    first = int(np.searchsorted(dates_u, month + "-01"))
    return str(dates_u[max(first - gap, 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lgb", choices=["lgb", "lr"])
    ap.add_argument("--feats", default="price", choices=list(FEATS))
    ap.add_argument("--neg", type=float, default=0.15)
    ap.add_argument("--universe", default="all", choices=["all", "top300", "top50", "top20"])
    ap.add_argument("--depth", default="shallow", choices=["shallow", "deep"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle", default="none", choices=["none", "global", "day"])
    ap.add_argument("--months", default="%s,%s" % G.TEST_MONTHS)
    ap.add_argument("--extra", default="", help="按 (code,date) 并入的额外特征表（research/gate 下的 parquet），只在 universe != all 时用")
    a = ap.parse_args()
    G.PL.wait_prod_window()
    t0 = time.time()
    feats = FEATS[a.feats]
    cols = ["code", "date", "end20", "y_open", "rank_ret20"] + [c for c in feats if c not in ("rank_ret20",)]
    t = G.load_table(cols)
    n = len(t)
    idx_uni = np.arange(n)
    if a.universe != "all":
        idx_uni = np.flatnonzero(t["rank_ret20"].to_numpy() <= int(a.universe[3:]))
        t = t.iloc[idx_uni].reset_index(drop=True)
    if a.extra:
        ex = pd.read_parquet(G.HERE / a.extra)
        excols = [c for c in ex.columns if c not in ("code", "date")]
        t = t.merge(ex, on=["code", "date"], how="left")
        assert len(t) == len(idx_uni)
        feats = feats + excols
        G.say("并入额外特征 %d 列" % len(excols))
    n_full = n
    n = len(t)
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    ym = dates.astype("U7")
    y = t["y_open"].to_numpy(float)
    rng0 = np.random.default_rng(a.seed)
    if a.shuffle == "global":
        ok = np.isfinite(y)
        y = y.copy()
        yy = y[ok].copy()
        rng0.shuffle(yy)
        y[ok] = yy
    elif a.shuffle == "day":
        ys = pd.Series(y)
        y = ys.groupby(t["date"].to_numpy()).transform(lambda s: s.sample(frac=1.0, random_state=int(rng0.integers(1 << 30))).to_numpy()).to_numpy()
    X = t[feats].to_numpy(np.float32)
    uni = np.ones(n, bool)
    del t
    dates_u = np.unique(dates)
    months = G.months_between(*a.months.split(","))
    s = np.full(n, np.nan, np.float32)
    log = []
    G.say("表 %d 行 %d 特征 %.0fs；月份 %s..%s" % (n, len(feats), time.time() - t0, months[0], months[-1]))
    for m in months:
        t1 = time.time()
        cut = purge_cut(dates_u, m)
        lo = G.month_shift(m, -12 * G.TRAIN_YEARS) + "-01"
        tm = (dates < cut) & (end20 < m + "-01") & (dates >= lo) & np.isfinite(y) & uni
        test = (ym == m) & uni
        if test.sum() == 0:
            continue
        rng = np.random.default_rng(a.seed * 1000 + int(m[:4]) * 12 + int(m[5:7]))
        pos = tm & (y > 0.5)
        neg = tm & (y < 0.5)
        keep_neg = neg & (rng.random(n) < a.neg)
        idx = np.flatnonzero(pos | keep_neg)
        w = np.where(y[idx] > 0.5, 1.0, 1.0 / a.neg)
        Xtr, ytr = X[idx], y[idx]
        if a.model == "lgb":
            import lightgbm as lgb
            params = dict(LGB_PARAMS[a.depth], random_state=a.seed)
            clf = lgb.LGBMClassifier(**params)
            clf.fit(Xtr, ytr, sample_weight=w)
            s[test] = clf.predict_proba(X[test])[:, 1].astype(np.float32)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            med = np.nanmedian(Xtr, axis=0)
            Xa = np.where(np.isfinite(Xtr), Xtr, med)
            Xb = np.where(np.isfinite(X[test]), X[test], med)
            # board_i 做哑变量
            bi = feats.index("board_i") if "board_i" in feats else -1
            if bi >= 0:
                def dum(M):
                    b = M[:, bi].astype(int)
                    D = np.stack([(b == k).astype(np.float32) for k in (1, 2, 3)], axis=1)
                    return np.concatenate([np.delete(M, bi, axis=1), D], axis=1)
                Xa, Xb = dum(Xa), dum(Xb)
            sc = StandardScaler().fit(Xa)
            clf = LogisticRegression(C=1.0, max_iter=500)
            clf.fit(sc.transform(Xa), ytr, sample_weight=w)
            s[test] = clf.predict_proba(sc.transform(Xb))[:, 1].astype(np.float32)
        log.append({"month": m, "train_rows": int(len(idx)), "pos": int(pos.sum()), "test_rows": int(test.sum()),
                    "cut": cut, "sec": round(time.time() - t1, 1)})
        G.say("%s 训练 %d 行（正 %d）测试 %d 行 净化线 %s %.0fs" % (m, len(idx), pos.sum(), test.sum(), cut, time.time() - t1))
    full = np.full(n_full, np.nan, np.float32)
    full[idx_uni] = s
    np.save(G.SCORES / (a.name + ".npy"), full)
    (G.SCORES / (a.name + ".json")).write_text(json.dumps({"args": vars(a), "feats": feats, "months": log,
                                                           "total_sec": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    G.say("-> %s  %.0fs" % (G.SCORES / (a.name + ".npy"), time.time() - t0))
    G.progress("样本外分数 %s：模型 %s 特征 %s(%d) 负样本 %.2f 宇宙 %s 深度 %s 打乱 %s，%d 个月 %.0f 分钟" % (
        a.name, a.model, a.feats, len(feats), a.neg, a.universe, a.depth, a.shuffle, len(log), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
