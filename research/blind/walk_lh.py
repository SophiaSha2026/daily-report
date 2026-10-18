"""L1 长历史：在 research/longhist/table_lh.parquet 上逐月滚动训练，分数按 (code, date) 对齐到 research/gate/table.parquet 的行序。

    PYTHONIOENCODING=utf-8 python research/blind/walk_lh.py --mode roll36 | rollall | fixed [--from 2006-01]

roll36  每个测试月前 36 个月（同 gate 流水，同一张长历史特征表上的对照）
rollall 每个测试月之前从 --from 起的全部历史（2010 起多段行情）
fixed   只训一次：--from 起到 2021-06（end20 < 2021-07-01，date < 2021-06 净化线），给 2021-07..2025-12 全部打分
训练行：表里 2020-12-23 之前的负样本已按 15% 抽样（w 补回），之后的行在这里按同样 15% 抽（权重 1/0.15）；浅树、种子 7。
输出 research/blind/scores/lh_<mode>.npy（gate 表同序，没对上的行 NaN）和 .json。
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
           lambda_l2=10.0, n_estimators=300, verbose=-1, num_threads=10)
TABLE = B.ROOT / "research" / "longhist" / "table_lh.parquet"
NEG = 0.15


def purge_cut(dates_u: np.ndarray, month: str, gap: int = 20) -> str:
    first = int(np.searchsorted(dates_u, month + "-01"))
    return str(dates_u[max(first - gap, 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["roll36", "rollall", "fixed"])
    ap.add_argument("--from", dest="start", default="2006-01")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--months", default="%s,%s" % G.TEST_MONTHS)
    a = ap.parse_args()
    B.PL.wait_prod_window()
    t0 = time.time()
    feats = list(G.FEATS_PRICE)
    cols = ["code", "date", "end20", "y_open", "w"] + [c for c in feats]
    t = pd.read_parquet(TABLE, columns=cols)
    t = t[t["date"] >= a.start + "-01"].reset_index(drop=True)
    n = len(t)
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    ym = dates.astype("U7")
    y = t["y_open"].to_numpy(float)
    w0 = t["w"].to_numpy(float)
    X = t[feats].to_numpy(np.float32)
    keys = t[["code", "date"]].copy()
    del t
    dates_u = np.unique(dates)
    months = G.months_between(*a.months.split(","))
    s = np.full(n, np.nan, np.float32)
    log = []
    import lightgbm as lgb
    B.say("长历史表 %d 行（%s 起）%d 特征 %.0fs；测试月 %s..%s" % (n, a.start, len(feats), time.time() - t0, months[0], months[-1]))
    rng = np.random.default_rng(a.seed)
    late = dates >= "2020-12-23"
    samp = ~late | (y > 0.5) | (rng.random(n) < NEG)        # 2020-12-23 起的负样本在这里抽
    w = np.where(late & (y < 0.5), 1.0 / NEG, 1.0) * w0

    def fit(tm):
        idx = np.flatnonzero(tm & samp)
        clf = lgb.LGBMClassifier(objective="binary", random_state=a.seed, **LGB)
        clf.fit(X[idx], y[idx], sample_weight=w[idx])
        return clf, idx

    if a.mode == "fixed":
        m0 = months[0]
        cut = purge_cut(dates_u, m0)
        tm = (dates < cut) & (end20 < m0 + "-01") & np.isfinite(y)
        clf, idx = fit(tm)
        test = np.isin(ym, months)
        s[test] = clf.predict_proba(X[test])[:, 1]
        log.append({"month": "fixed", "train_rows": int(len(idx)), "pos": int((y[idx] > 0.5).sum()), "test_rows": int(test.sum()),
                    "cut": cut, "sec": round(time.time() - t0, 1)})
        B.say("fixed：训练 %d 行（正 %d，%s..%s）测试 %d 行 %.0fs" % (len(idx), (y[idx] > 0.5).sum(), str(np.sort(dates[idx])[0]), str(np.sort(dates[idx])[-1]), test.sum(), time.time() - t0))
    else:
        for m in months:
            t1 = time.time()
            cut = purge_cut(dates_u, m)
            lo = (G.month_shift(m, -36) + "-01") if a.mode == "roll36" else (a.start + "-01")
            tm = (dates < cut) & (end20 < m + "-01") & (dates >= lo) & np.isfinite(y)
            test = ym == m
            if test.sum() == 0:
                continue
            clf, idx = fit(tm)
            s[test] = clf.predict_proba(X[test])[:, 1]
            log.append({"month": m, "train_rows": int(len(idx)), "pos": int((y[idx] > 0.5).sum()), "test_rows": int(test.sum()),
                        "cut": cut, "sec": round(time.time() - t1, 1)})
            B.say("%s 训练 %d 行（正 %d，起 %s）测试 %d 行 %.0fs" % (m, len(idx), (y[idx] > 0.5).sum(), str(np.sort(dates[idx])[0]), test.sum(), time.time() - t1))
    # 对齐到 gate 表
    g = G.load_table(["code", "date"])
    g["code"] = g["code"].astype(str).str.zfill(6)
    keys["s"] = s
    mm = g.merge(keys, on=["code", "date"], how="left")
    assert len(mm) == len(g)
    out = mm["s"].to_numpy(np.float32)
    name = "lh_" + a.mode
    np.save(B.SCORES / (name + ".npy"), out)
    (B.SCORES / (name + ".json")).write_text(json.dumps({"args": vars(a), "feats": feats, "months": log, "total_sec": round(time.time() - t0),
                                                         "aligned_share": float(np.isfinite(out).mean())}, ensure_ascii=False, indent=1), encoding="utf-8")
    B.say("-> %s  对齐到 gate 表 %.1f%% 行有分数  %.0fs" % (B.SCORES / (name + ".npy"), 100 * np.isfinite(out).mean(), time.time() - t0))
    B.progress("walk_lh %s（%s 起）：%d 次训练，对齐 gate 表 %.1f%%，%.0f 分钟" % (a.mode, a.start, len(log), 100 * np.isfinite(out).mean(), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
