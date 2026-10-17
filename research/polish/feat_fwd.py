"""
扩展特征的单一实现（前向样本用；和 build_pool.py 的公式一致，verify() 在 2025-12 的重叠行上对账）。

ext_features(px) 要 code date open high low close volume amount turnover outstanding_share，
可选 close_u / low_u（不复权价；没有就用前复权价按 1 分容差判涨停，见 CLAUDE.md 长期调整突破一节）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ext_features(px: pd.DataFrame) -> pd.DataFrame:
    from datasource import limit_price_arr
    px = px.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    g = px.groupby("code", sort=False)
    c = px["close"]
    out = pd.DataFrame({"code": px["code"], "date": px["date"]})
    for k in (1, 5, 10, 20, 40, 60):
        out["ret%d" % k] = g["close"].pct_change(k).astype(np.float32)
    out["lret20"] = np.log1p(out["ret20"].astype(float).clip(lower=-0.99)).astype(np.float32)
    out["ret20_lag5"] = (g["close"].shift(5) / g["close"].shift(20) - 1).astype(np.float32)
    lr = np.log(c).groupby(px["code"]).diff()
    out["vol20"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(20).std()).astype(np.float32)
    out["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std()).astype(np.float32)
    amp = px["high"] / px["low"] - 1
    out["amp20"] = amp.groupby(px["code"]).transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    out["turn20"] = g["turnover"].transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    out["nbar"] = (g.cumcount() + 1).astype(np.int64)
    for w in (20, 60, 120):
        hi = g["high"].transform(lambda s, w=w: s.rolling(w).max())
        out["dist_hi%d" % w] = (c / hi - 1).astype(np.float32)
    lo20 = g["low"].transform(lambda s: s.rolling(20).min())
    hi20 = g["high"].transform(lambda s: s.rolling(20).max())
    out["pos20"] = ((c - lo20) / (hi20 - lo20 + 1e-12)).astype(np.float32)
    cmax20 = g["close"].transform(lambda s: s.rolling(20).max())
    nh = (c >= cmax20 - 1e-9).astype(float)
    out["nh20_5"] = nh.groupby(px["code"]).transform(lambda s: s.rolling(5).sum()).astype(np.float32)
    r1 = out["ret1"].astype(float)
    out["up20"] = (r1 > 0).astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["max1_20"] = r1.groupby(px["code"]).transform(lambda s: s.rolling(20).max()).astype(np.float32)
    out["amt20"] = g["amount"].transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    out["amt1"] = px["amount"].astype(np.float32)
    out["amt_ratio"] = (px["amount"] / out["amt20"]).astype(np.float32)
    out["turn1"] = px["turnover"].astype(np.float32)
    out["turn5"] = g["turnover"].transform(lambda s: s.rolling(5).mean()).astype(np.float32)
    v5 = g["volume"].transform(lambda s: s.rolling(5).mean())
    v20 = g["volume"].transform(lambda s: s.rolling(20).mean())
    out["vr5_20"] = (v5 / v20).astype(np.float32)
    cu = px["close_u"] if "close_u" in px.columns else px["close"]
    lu_ = px["low_u"] if "low_u" in px.columns else px["low"]
    out["mcap"] = (px["outstanding_share"] * cu).astype(np.float32)
    out["price_u"] = cu.astype(np.float32)
    out["gap1"] = (px["open"] / g["close"].shift(1) - 1).astype(np.float32)
    out["cr1"] = (px["close"] / px["open"] - 1).astype(np.float32)
    b2 = px["code"].str[:2]
    pct = np.where(b2.isin(["30", "68"]), 20.0, np.where(px["code"].str[:1].isin(["9", "8", "4"]), 30.0, 10.0))
    out["lim_pct"] = pct.astype(np.float32)
    prev_u = cu.groupby(px["code"]).shift(1)
    tol = 0.005 if "close_u" in px.columns else 0.011
    lim = limit_price_arr(prev_u, pct, px["code"])
    lu = (cu >= lim - tol) & np.isfinite(prev_u)
    out["lu_close"] = lu.astype(np.float32)
    out["yizi_today"] = ((lu_ >= lim - tol) & lu).astype(np.float32)
    lu_i = lu.astype(int).to_numpy()
    codes = px["code"].to_numpy()
    streak = np.zeros(len(px), np.int32)
    for i in range(len(px)):
        if lu_i[i] and i > 0 and codes[i] == codes[i - 1]:
            streak[i] = streak[i - 1] + 1
        else:
            streak[i] = lu_i[i]
    out["lu_streak"] = streak.astype(np.float32)
    out["nlu20"] = lu.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["nlu5"] = lu.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(5).sum()).astype(np.float32)
    lim_dn = prev_u * (1 - pct / 100)
    dn = (cu <= lim_dn + tol) & np.isfinite(prev_u)
    out["nld20"] = dn.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["ret20_norm"] = (out["ret20"] / (out["lim_pct"] / 100)).astype(np.float32)
    out["lret20_vol"] = (out["lret20"] / (out["vol60"] + 1e-6)).astype(np.float32)
    # 停牌缺口：按全市场开市日编号，最近 20 根里相邻两根隔 >= 5 个开市日
    n_by_day = px.groupby("date")["code"].size().sort_index()
    open_days = n_by_day[n_by_day >= 0.5 * n_by_day.rolling(20, min_periods=1).max()].index
    didx = pd.Series(np.arange(len(open_days)), index=list(open_days))
    di = px["date"].map(didx)
    gap = (di - di.groupby(px["code"]).shift(1) - 1).fillna(0)
    out["susp20"] = (gap.groupby(px["code"]).transform(lambda s: s.rolling(20, min_periods=1).max()) >= 5).to_numpy()
    out["close"] = c.astype(np.float32)
    out["board"] = np.where(b2 == "30", "chinext", np.where(b2 == "68", "star", np.where(px["code"].str[:1].isin(["9", "8", "4"]), "bj", "main")))
    return out


def verify(sample_month: str = "2025-11") -> None:
    """对账：px_screen 上重算 2025-11 的行，和 pool.parquet 的列逐位比。"""
    import common_pl as P
    px = pd.read_parquet(P.EXP / "screen" / "px_screen.parquet")
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px[px["date"] >= "2024-06-01"]
    f = ext_features(px)
    f = f[f["date"].str[:7] == sample_month]
    pool = P.load_pool()
    pool = pool[pool["date"].str[:7] == sample_month]
    m = f.merge(pool, on=["code", "date"], suffixes=("", "_p"))
    bad = []
    for c in [c for c in f.columns if c + "_p" in m.columns and c not in ("board", "code", "date")]:
        a, b = m[c].to_numpy(float), m[c + "_p"].to_numpy(float)
        ok = np.isfinite(a) == np.isfinite(b)
        ok &= ~np.isfinite(a) | np.isclose(a, b, rtol=1e-5, atol=1e-6)
        if ok.mean() < 0.9999:
            bad.append((c, float(1 - ok.mean())))
    print("对账 %s：%d 行，不一致列 %s" % (sample_month, len(m), bad or "无"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    verify()
