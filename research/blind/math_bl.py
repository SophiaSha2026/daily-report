"""第二部分 2a：底层瓶颈的数字（criteria.txt 第 2a 节公式，数字从数据里算）。

    PYTHONIOENCODING=utf-8 python research/blind/math_bl.py

正样本率、按天聚集的设计效应 DE、有效样本、分辨 2pp 要多少票 / 多少年、「挑日子」分解、正样本按天的集中度、当天内 AUC 和双正态天花板。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

import common_bl as B

K = B.G.PR.P  # common_pl


def nw_lrv(x: np.ndarray, lag: int = 20) -> float:
    """长程方差（Bartlett，滞后 lag）：evalkit._nw_se 给的是均值的 se，lrv = se^2 · n。"""
    import evalkit as EK
    return EK._nw_se(np.asarray(x, float), lag) ** 2 * len(x)


def auc_day(score: np.ndarray, y: np.ndarray, dates: np.ndarray) -> pd.Series:
    df = pd.DataFrame({"d": dates, "s": score, "y": y})
    df = df[np.isfinite(df["s"]) & np.isfinite(df["y"])]
    df["r"] = df.groupby("d")["s"].rank()
    g = df.groupby("d").agg(n=("y", "size"), n1=("y", "sum"), rs=("r", lambda r: 0.0))
    pos = df[df["y"] > 0].groupby("d")["r"].sum()
    g["rs"] = pos.reindex(g.index).fillna(0)
    n1, n0 = g["n1"], g["n"] - g["n1"]
    auc = (g["rs"] - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc[(n1 >= 5) & (n0 >= 50)]


def binormal_top(pi: float, N: int, k: int, dprime: float) -> float:
    """双正态（正类均值 d'，方差 1）下，N 只里取分数前 k 的精确率。"""
    frac = k / N

    def f(z):
        return pi * norm.sf(z - dprime) + (1 - pi) * norm.sf(z) - frac
    z = brentq(f, -10, 10)
    tp = pi * norm.sf(z - dprime)
    return tp / frac


def main():
    t0 = time.time()
    L = ["底层瓶颈的数字（research/blind/math_bl.py）"]
    t = B.load_keys()
    s = B.load_scores("lgb_price").astype(float)
    y = t["y_open"].to_numpy(float)
    dates = t["date"].to_numpy().astype("U10")
    seg = t["seg"].to_numpy()
    rank = t["rank_ret20"].to_numpy()
    lret20 = np.log1p(np.clip(t["ret20"].to_numpy(float), -0.99, None))
    for sg in ("screen", "dev"):
        m = (seg == sg) & np.isfinite(y)
        d = pd.DataFrame({"d": dates[m], "y": y[m]})
        day = d.groupby("d")["y"].agg(["mean", "sum", "size"])
        pi_bar = float(d["y"].mean())
        L.append("\n== %s：%d 天，候选池 %d 行/天，正样本率 π̄ %.2f%%，每天正样本 %.0f 只（中位 %.0f，10/90 分位 %.0f/%.0f）" % (
            sg, len(day), day["size"].mean(), 100 * pi_bar, day["sum"].mean(), day["sum"].median(),
            day["sum"].quantile(0.1), day["sum"].quantile(0.9)))
        w = day["sum"] / day["sum"].sum()
        L.append("   正样本按天的集中度：最热 10%% 的天占 %.0f%% 的正样本，有效天数 1/Σw² = %.0f（名义 %d 天）；日基准率 π_d 的 sd %.2fpp，自相关(1) %.2f" % (
            100 * w.sort_values(ascending=False).head(max(1, len(w) // 10)).sum(), 1 / (w ** 2).sum(), len(day),
            100 * day["mean"].std(), day["mean"].autocorr(1)))
        # 设计效应：前 10 / 前 2 / 前 1 的按天命中率序列
        for k in (10, 2, 1):
            mk = m & (rank <= k)
            hd = pd.Series(y[mk]).groupby(dates[mk]).mean().reindex(day.index).fillna(0)
            P = float(y[mk].mean())
            iid = P * (1 - P) / k
            lrv = nw_lrv(hd.to_numpy())
            DE = lrv / iid
            V = int(mk.sum())
            L.append("   ret20 前 %2d：P %.1f%%，V %d，按天序列 sd %.3f，NW 长程方差 / iid 方差 = 设计效应 DE %.2f，有效票数 V/DE = %.0f，"
                     "单臂 se = sqrt(DE·P(1−P)/V) = %.2fpp（对账 NW se）" % (k, 100 * P, V, hd.std(), DE, V / DE, 100 * np.sqrt(DE * P * (1 - P) / V)))
            if k == 10:
                DE10, P10 = DE, P
        # 分辨 2pp 需要多少
        zc = 2.5 + 0.84
        for rho in (0.0, 0.5, 0.9):
            Vn = 2 * (1 - rho) * DE10 * P10 * (1 - P10) * zc ** 2 / 0.02 ** 2
            L.append("   分辨 Δ=2pp（P≈%.0f%%，DE %.1f，两臂重合 ρ=%.1f，c=2.5 功效 80%%）需要有效票数 %.0f，= 每天 10 只跑 %.1f 年" % (
                100 * P10, DE10, rho, Vn, Vn / (10 * 242)))
        # 「挑日子」分解：lgb_price 全池每月事后前 N 行；和 每天固定前 10
        ms = m & np.isfinite(s)
        dm = pd.DataFrame({"ym": dates[ms].astype("U7"), "d": dates[ms], "s": s[ms], "y": y[ms]})
        dm["r"] = dm.groupby("ym")["s"].rank(ascending=False, method="first")
        pick = dm[dm["r"] <= 20]
        nd = pick.groupby("d").size()
        pid = day["mean"].reindex(nd.index)
        between = float((nd * pid).sum() / nd.sum())
        P_pick = float(pick["y"].mean())
        L.append("   「挑日子」分解（lgb_price 全池每月事后前 20 行，共 %d 只、落在 %d 天）：实际精确率 %.1f%% = 当天基准率按出票量加权 %.1f%%（挑日子）"
                 " + 当天内增量 %.1fpp；池平均 π̄ %.2f%%" % (len(pick), len(nd), 100 * P_pick, 100 * between, 100 * (P_pick - between), 100 * pi_bar))
        mk = m & (rank <= 10)
        P10r = float(y[mk].mean())
        L.append("   同一分解，ret20 每天前 10：%.1f%% = 当天基准 %.2f%%（每天同量，挑日子贡献 0）+ 当天内增量 %.1fpp" % (
            100 * P10r, 100 * pi_bar, 100 * (P10r - pi_bar)))
        # 当天内 AUC
        a_m = auc_day(s[m], y[m], dates[m])
        a_r = auc_day(lret20[m], y[m], dates[m])
        L.append("   当天内 AUC（候选池全体，中位 / 均值）：lgb_price %.3f / %.3f，lret20 %.3f / %.3f" % (
            a_m.median(), a_m.mean(), a_r.median(), a_r.mean()))
        # 双正态天花板：用实际的每天 π_d 和 N_d
        for auc in (0.60, 0.656, 0.70, 0.75, 0.80):
            dp = np.sqrt(2) * norm.ppf(auc)
            vals = [binormal_top(float(p), int(n), 10, dp) for p, n in zip(day["mean"], day["size"]) if p > 0]
            vals1 = [binormal_top(float(p), int(n), 1, dp) for p, n in zip(day["mean"], day["size"]) if p > 0]
            L.append("   双正态：当天内 AUC %.3f -> 每天前 10 精确率 %.1f%%，前 1 %.1f%%" % (auc, 100 * np.mean(vals), 100 * np.mean(vals1)))
    (B.HERE / "results_math.txt").write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.progress("math_bl：瓶颈数字 -> results_math.txt（%.0fs）" % (time.time() - t0))


if __name__ == "__main__":
    main()
