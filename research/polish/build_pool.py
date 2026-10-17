"""
候选池表：research/rulecore 的 feat.parquet（候选池、标签、四个输入）+ px_screen 上算的扩展特征 + 事件剔除标志。

    PYTHONIOENCODING=utf-8 python research/polish/build_pool.py

只留 2021-10-01 之后（两段 + 训练回看用的前 3 个月留给需要的臂自己处理）、elig0 的行。
扩展特征全部只用 T 日及之前的数据（按根数滑窗，不按日历）。涨停判定用不复权价 close_u / low_u 和
datasource.limit_price_arr（北交所向下取整），和生产同一口径。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common_pl as P

RC = P.EXP / "rulecore"
SCR = P.EXP / "screen"
START = "2020-12-01"   # 重排臂的训练回看要 3 年


def main():
    t0 = time.time()
    px = pd.read_parquet(SCR / "px_screen.parquet")
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    P.say("px %d 行 %.0fs" % (len(px), time.time() - t0))
    g = px.groupby("code", sort=False)
    c = px["close"]
    out = pd.DataFrame({"code": px["code"], "date": px["date"]})
    for k in (1, 5, 10, 40, 60):
        out["ret%d" % k] = g["close"].pct_change(k).astype(np.float32)
    out["ret20_lag5"] = (g["close"].shift(5) / g["close"].shift(20) - 1).astype(np.float32)
    lr = np.log(c).groupby(px["code"]).diff()
    out["vol20"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(20).std()).astype(np.float32)
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
    out["mcap"] = (px["outstanding_share"] * px["close_u"]).astype(np.float32)
    out["price_u"] = px["close_u"].astype(np.float32)
    out["gap1"] = (px["open"] / g["close"].shift(1) - 1).astype(np.float32)
    out["cr1"] = (px["close"] / px["open"] - 1).astype(np.float32)
    # 涨停判定（不复权价）
    from datasource import limit_price_arr
    b2 = px["code"].str[:2]
    pct = np.where(b2.isin(["30", "68"]), 20.0, np.where(px["code"].str[:1].isin(["9", "8", "4"]), 30.0, 10.0))
    out["lim_pct"] = pct.astype(np.float32)
    prev_u = g["close_u"].shift(1)
    lim = limit_price_arr(prev_u, pct, px["code"])
    lu = (px["close_u"] >= lim - 0.005) & np.isfinite(prev_u)
    out["lu_close"] = lu.astype(np.float32)
    out["yizi_today"] = ((px["low_u"] >= lim - 0.005) & lu).astype(np.float32)
    # 连板高度：连续涨停收盘天数
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
    # 近 20 根里有几天收盘跌停附近（用对称幅度）
    lim_dn = prev_u * (1 - pct / 100)
    dn = (px["close_u"] <= lim_dn + 0.005) & np.isfinite(prev_u)
    out["nld20"] = dn.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    P.say("扩展特征算完 %.0fs" % (time.time() - t0))
    del px, g

    feat = pd.read_parquet(RC / "feat.parquet")
    feat["code"] = feat["code"].astype(str).str.zfill(6)
    feat = feat[(feat["date"] >= START) & feat["elig0"]].reset_index(drop=True)
    df = feat.merge(out, on=["code", "date"], how="left")
    del out, feat
    P.say("候选池 elig0 行 %d（%s..%s）%.0fs" % (len(df), df["date"].min(), df["date"].max(), time.time() - t0))
    import common_rc as C
    ev = C.event_flags(df)
    for col in ("reduce30", "release30", "placement60", "any"):
        df["ev_" + col] = ev[col].to_numpy()
    df["pool"] = df["elig0"].to_numpy() & ~df["ev_any"].to_numpy()
    df["ret20_norm"] = (df["ret20"] / (df["lim_pct"] / 100)).astype(np.float32)
    df["lret20_vol"] = (df["lret20"] / (df["vol60"] + 1e-6)).astype(np.float32)
    P.say("事件剔除 减持 %.2f%% 解禁 %.2f%% 增发 %.2f%%；pool %.1f%% %.0fs" % (
        100 * df["ev_reduce30"].mean(), 100 * df["ev_release30"].mean(), 100 * df["ev_placement60"].mean(),
        100 * df["pool"].mean(), time.time() - t0))
    df.to_parquet(P.PL / "pool.parquet", index=False)
    P.say("-> %s  %d 行 %d 列 %.0fs" % (P.PL / "pool.parquet", len(df), df.shape[1], time.time() - t0))
    P.progress("pool.parquet 建好：%d 行 %s..%s，%d 列" % (len(df), df["date"].min(), df["date"].max(), df.shape[1]))


if __name__ == "__main__":
    main()
