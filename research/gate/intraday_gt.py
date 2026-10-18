"""F6：并入 research/intraday/features_intraday.parquet（抽样 12 个月、ret20 前 50、无北交所）的配对检查。

    PYTHONIOENCODING=utf-8 python research/gate/intraday_gt.py

抽样月按时间顺序：每个抽样月 m 用之前的抽样月（标签窗口已关）训练两个 LightGBM：A = 日线原值特征，B = A + 36 列日内特征，
同一批行、同一超参、同一种子。看 (1) 每月前 N 行精确率 A 对 B（池化、按天序列 NW t），(2) 按天 AUC 的配对差，
(3) B 的分数做门槛（校准窗 = 之前的抽样月）能不能出 >= 20% 的集合。前 2 个抽样月没有训练数据，只作训练。
功效有限（筛选集 4 个可测月、开发集 6 个），只到方向。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_gt as G

INTRA = G.ROOT / "research" / "intraday" / "features_intraday.parquet"
ICOLS = ["c_vwap", "ret_last30", "ret_last60", "ret_first30", "ret_am", "ret_pm", "dd_intra", "c_hi", "c_lo", "t_high", "t_low",
         "rv", "up_share", "range_u", "v_first30", "v_first5", "v_last30", "v_last5", "v_am", "v_top10", "v_zero", "v_above_vwap",
         "lu_touch", "t_first_lu", "n_open", "lu_hold", "lu_seal_close", "v_at_lu", "buy_share", "buy_share_pm", "big_net",
         "big_share", "auc_share", "tick_size", "tx_n"]


def day_auc(df: pd.DataFrame, col: str) -> pd.Series:
    from sklearn.metrics import roc_auc_score
    out = {}
    for d, g in df.groupby("date"):
        if g["hit"].nunique() == 2 and len(g) >= 10:
            out[d] = roc_auc_score(g["hit"], g[col])
    return pd.Series(out)


def main():
    import lightgbm as lgb
    fi = pd.read_parquet(INTRA)
    fi["code"] = fi["code"].astype(str).str.zfill(6)
    t = G.load_table(["code", "date", "end20", "y_open", "hit", "seg", "rank_ret20", "r20_open"] + [c for c in G.FEATS_PRICE if c != "rank_ret20"])
    t = t[t["rank_ret20"] <= 50].merge(fi[["code", "date"] + ICOLS], on=["code", "date"], how="inner").reset_index(drop=True)
    t["ym"] = t["date"].str[:7]
    months = sorted(t["ym"].unique())
    G.say("并上日内特征的行 %d，月份 %s" % (len(t), months))
    featsA = G.FEATS_PRICE
    featsB = G.FEATS_PRICE + ICOLS
    t["sA"] = np.nan
    t["sB"] = np.nan
    params = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, n_estimators=200, verbose=-1, num_threads=8, random_state=7)
    for i, m in enumerate(months):
        if i < 2:
            continue
        tr = (t["ym"] < m) & (t["end20"] < m + "-01") & np.isfinite(t["y_open"])
        te = t["ym"] == m
        for nm, feats in (("sA", featsA), ("sB", featsB)):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(t.loc[tr, feats].to_numpy(np.float32), t.loc[tr, "y_open"].to_numpy())
            t.loc[te, nm] = clf.predict_proba(t.loc[te, feats].to_numpy(np.float32))[:, 1]
        G.say("%s 训练 %d 行（正 %d）测试 %d 行" % (m, tr.sum(), int(t.loc[tr, "y_open"].sum()), te.sum()))
    e = t[np.isfinite(t["sA"])].copy()
    L = ["F6 日内特征配对检查（抽样月 %s..%s，可测 %d 个月，行 %d）" % (months[2], months[-1], len(months) - 2, len(e))]
    for seg in ("screen", "dev"):
        s = e[e["seg"] == seg]
        if not len(s):
            continue
        aA, aB = day_auc(s, "sA"), day_auc(s, "sB")
        d = (aB - aA).dropna()
        L.append("%s：按天 AUC A %.3f  B %.3f  差 %+.3f（NW t %.2f，%d 天）" % (seg, aA.mean(), aB.mean(), d.mean(), G.PR.nw_t(d.to_numpy()) if len(d) > 25 else np.nan, len(d)))
        for N in (3, 5, 10):
            rows = []
            for nm in ("sA", "sB"):
                sel = s.groupby("date")[nm].rank(method="first", ascending=False) <= N
                hA = s[sel].groupby("date")["hit"].sum()
                rows.append((hA, s[sel]["hit"].mean(), int(sel.sum())))
            dd = (rows[1][0] - rows[0][0]).reindex(s["date"].unique()).fillna(0)
            # 同量名次基线：每天前 N 名（ret20 名次）
            bsel = s["rank_ret20"] <= N
            L.append("  每天前 %2d：A %.1f%% (V %d)  B %.1f%% (V %d)  B−A 按天 NW t %.2f   名次前 %d：%.1f%%" % (
                N, 100 * rows[0][1], rows[0][2], 100 * rows[1][1], rows[1][2], G.PR.nw_t(dd.to_numpy()), N, 100 * s[bsel]["hit"].mean()))
        # 门槛：B 的分数 >= 之前抽样月里累计精确率 >= 0.2 的线（支持 >= 50 行）
        for nm in ("sA", "sB"):
            emitted = []
            for m in sorted(s["ym"].unique()):
                cal = e[(e["ym"] < m) & (e["end20"] < m + "-01")]
                if len(cal) < 200:
                    continue
                sc = cal[nm].to_numpy()
                o = np.argsort(-sc)
                cumh = np.cumsum(cal["hit"].to_numpy()[o])
                k = np.arange(1, len(sc) + 1)
                prec = cumh / k
                okk = (prec >= 0.2) & (k >= 50)
                if not okk.any():
                    continue
                thr = sc[o][np.flatnonzero(okk).max()]
                emitted.append(s[(s["ym"] == m) & (s[nm] >= thr)])
            if emitted:
                em = pd.concat(emitted)
                L.append("  门槛 τ=0.20（%s）：出 %d 只 / %d 天，精确率 %.1f%%" % (nm, len(em), em["date"].nunique(), 100 * em["hit"].mean()))
            else:
                L.append("  门槛 τ=0.20（%s）：没有出票" % nm)
    txt = "\n".join(L)
    print(txt)
    (G.HERE / "results_intraday.txt").write_text(txt + "\n", encoding="utf-8")
    G.ledger_add({"kind": "diag", "name": "intraday_pair", "note": "F6 日内特征配对检查（抽样月，两段都看）"})
    G.progress("F6 日内特征配对检查完成（results_intraday.txt）")


if __name__ == "__main__":
    main()
