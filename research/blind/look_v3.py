"""V3 那一臂（前 20 里可信度最高的 1 只）的稳健性加看：更长滞后的 NW t、按票整簇重抽的自举、逐月命中、名单重合。

    PYTHONIOENCODING=utf-8 python research/blind/look_v3.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common_bl as B
from diag_v3 import cred


def nw_t_lag(u: np.ndarray, lag: int) -> float:
    import evalkit as EK
    return float(np.mean(u) / EK._nw_se(np.asarray(u, float), lag))


def main():
    t0 = time.time()
    t = B.load_keys()
    s = B.load_scores("lgb_price").astype(float)
    y = t["y_open"].to_numpy(float)
    rank = t["rank_ret20"].to_numpy()
    c, hist_n = cred(t, s, y)
    cpct = B.day_pct(t, np.where(hist_n > 0, c, np.nan))
    sel = B.topn_by(t, (rank <= 20) & (cpct >= 0.9), c, 1)
    r = B.eval_both(t, sel, boot=False)
    L = ["V3 K20 cp.9 n1 c 稳健性加看"]
    u_all = np.concatenate([r["screen"]["u"], r["dev"]["u"]])
    for lag in (20, 40, 60, 120):
        L.append("合并 969 天 u 序列 NW t：滞后 %3d -> %.2f（筛选 %.2f，开发 %.2f）" % (
            lag, nw_t_lag(u_all, lag), nw_t_lag(r["screen"]["u"], lag), nw_t_lag(r["dev"]["u"], lag)))
    # 按票整簇重抽：以「不重复的票」为单位做自举（每只票的全部出票天一起进出），对 ΔP 的分布
    rng = np.random.default_rng(5)
    for seg in ("screen", "dev"):
        m = sel & (t["seg"].to_numpy() == seg)
        rows = t[m][["date", "code", "hit"]].copy()
        B_ = B.G.baseline(seg)
        di = pd.Series(np.arange(B_.D), index=B_.dates)
        hb = B_.hits_k(1.0)                                      # 同量基线 B(1) 的每日命中
        rows["hb"] = hb[di[rows["date"]].to_numpy()]
        rows["u"] = rows["hit"] - rows["hb"]
        codes = rows["code"].unique()
        per = rows.groupby("code")["u"].sum()
        nper = rows.groupby("code").size()
        vals = []
        for _ in range(4000):
            pick = rng.choice(codes, len(codes), replace=True)
            vals.append(per.loc[pick].sum() / nper.loc[pick].sum())
        vals = 100 * np.array(vals)
        L.append("%s：按票整簇自举（%d 只票）ΔP 对 B(1)：点 %+.1fpp，95%% 区间 [%+.1f, %+.1f]，P(ΔP<=0) %.3f" % (
            seg, len(codes), 100 * rows["u"].sum() / len(rows), *np.quantile(vals, [0.025, 0.975]), float((vals <= 0).mean())))
        # 去掉贡献最大的 3 只票之后
        top3 = per.sort_values(ascending=False).head(3)
        rest = rows[~rows["code"].isin(top3.index)]
        L.append("   贡献最大的 3 只票 %s 合计 u %+.0f；去掉它们后 ΔP %+.1fpp（%d 行）" % (
            list(top3.index), top3.sum(), 100 * rest["u"].sum() / max(len(rest), 1), len(rest)))
        # 逐月
        mo = rows.groupby(rows["date"].str[:7]).agg(n=("hit", "size"), h=("hit", "sum"), hb=("hb", "sum"))
        L.append("   逐月（出票/命中/基线命中）：%s" % " ".join("%s:%d/%d/%d" % (k[2:], v["n"], v["h"], v["hb"]) for k, v in mo.iterrows()))
        L.append("   赢基线的月 %d / %d" % (int((mo["h"] > mo["hb"]).sum()), len(mo)))
    (B.HERE / "results_look_v3.txt").write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.ledger_add({"kind": "look", "what": "V3 臂稳健性：长滞后 NW t、按票整簇自举、去最大 3 只、逐月"})
    B.progress("look_v3：V3 臂稳健性加看 -> results_look_v3.txt（%.0fs）" % (time.time() - t0))


if __name__ == "__main__":
    main()
