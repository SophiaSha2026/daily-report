"""
阶段 2：新特征族，从 data/breakout/daily.parquet 自己算（只读到 2025-12-31），当日全市场排名归一。
每族一个文件 data/breakout/raw/exp/features/fam_<族>.parquet，列名 f_<族>_<名>。

族（只用日线 OHLCV、成交额、换手率、流通股本）：
    on    隔夜 / 日内收益分离：隔夜 = 开 / 昨收 - 1，日内 = 收 / 开 - 1，5/20/60 日和、差、波动、隔夜为正的天数
    vc    波动收缩和平台：收益波动 5/60、10/120，振幅 5/60，20 日箱体宽 / 120 日箱体宽，60 日平台宽，
          离前 20 / 60 日最高价（不含今天）的突破距离，60 日里收盘价在今价 ±10% 内的天数
    hi    多周期高低点：离 120/250 日最高、最低的距离，距 60/120/250 日最高点的天数，250 日区间位置
    ext   极端收益和分布：20 日最大 3 天均值、60 日最大 1 天、20 日最小 1 天，偏度 20/60、峰度 60，
          特质波动（对等权市场回归残差）20/60、贝塔 60、特质 / 总波动
    dly   价格延迟（Hou-Moskowitz D1）：60 日 r ~ m_t + m_t-1 + m_t-2 对 r ~ m_t 的 R² 差；滞后系数和
    turn  换手结构：换手 5/60、5/250，20 日最大 / 20 日总和（集中度），换手相对 60 日的 z 冲击，
          250 日均换手水平，Amihud 非流动性 20 日（|收益| / 成交额）
    vp    量价相关和背离：20 日 corr(收益, 换手变化)、corr(|收益|, 换手)、corr(收益_t, 换手_t-1)，
          20 日价位排名减换手排名（价高量低 = 背离）
    rm    残差动量：扣市场（贝塔调整）的 20/60/120 日累计残差收益（跳过最近 5 天的 120 日版也给）、
          扣板块中位的 20/60 日收益、残差动量 / 特质波动

价格一律用比值（前复权对比值无影响）；不用价格水平和平盘天数（前复权取整的颗粒度带未来除权信息，见审计）。
停牌：滑窗按该票自己的 K 线数（同 a158 和生产）。

    python research/features/build_new.py [族 ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_feats as BF  # noqa: E402
import features as F  # noqa: E402

KEEP_FROM, KEEP_TO = "2023-06-01", "2025-12-31"
roll = BF.roll


def S(x, d, pos):
    return roll(x, d, lambda v: v.sum(axis=1), pos)


def M(x, d, pos):
    return roll(x, d, lambda v: v.mean(axis=1), pos)


def SD(x, d, pos):
    return roll(x, d, lambda v: v.std(axis=1), pos)


def MX(x, d, pos):
    return roll(x, d, lambda v: v.max(axis=1), pos)


def MN(x, d, pos):
    return roll(x, d, lambda v: v.min(axis=1), pos)


def lag(x, k, pos):
    y = np.r_[np.full(k, np.nan), x[:-k]]
    y[pos < k] = np.nan
    return y


def load():
    px = pd.read_parquet(C.ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover",
                                  "outstanding_share"])
    px["date"] = px["date"].astype(str).str[:10]
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px[px["date"] <= KEEP_TO].sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    px["turnover"] = px["turnover"].clip(0, 0.6)
    pos = px["_pos"].to_numpy()
    c = px["close"].to_numpy(float)
    cp = lag(c, 1, pos)
    r = c / cp - 1
    r[~np.isfinite(r)] = np.nan
    r = np.clip(r, -0.35, 0.35)
    px["r"] = r
    # 等权市场收益（当天有收益的票），按市场交易日对齐滞后
    m = px.groupby("date")["r"].mean()
    m = m[px.groupby("date")["r"].count() >= 1000]
    px["m0"] = px["date"].map(m).to_numpy(float)
    px["m1"] = px["date"].map(m.shift(1)).to_numpy(float)
    px["m2"] = px["date"].map(m.shift(2)).to_numpy(float)
    return px


def fam_on(px, f):
    pos = px["_pos"].to_numpy()
    o, c = px["open"].to_numpy(float), px["close"].to_numpy(float)
    cp = lag(c, 1, pos)
    on = np.clip(o / cp - 1, -0.35, 0.35)
    idr = np.clip(c / o - 1, -0.35, 0.35)
    f["on_1"] = on
    for d in (5, 20, 60):
        f[f"on_sum{d}"] = S(on, d, pos)
        f[f"id_sum{d}"] = S(idr, d, pos)
    f["on_m_id20"] = f["on_sum20"] - f["id_sum20"]
    f["on_m_id60"] = f["on_sum60"] - f["id_sum60"]
    f["on_std20"] = SD(on, 20, pos)
    f["id_std20"] = SD(idr, 20, pos)
    f["on_pos20"] = M(np.where(np.isfinite(on), (on > 0).astype(float), np.nan), 20, pos)
    # 目录 F007：未回补跳空、大幅高开天数；F005：换手变化相对市场的「硬币」加权收益
    h, l_ = px["high"].to_numpy(float), px["low"].to_numpy(float)
    hp = lag(h, 1, pos)
    f["gapup20"] = M(np.where(np.isfinite(hp), (l_ > hp).astype(float), np.nan), 20, pos)
    f["gap3_20"] = M(np.where(np.isfinite(on), (on > 0.03).astype(float), np.nan), 20, pos)
    t = px["turnover"].to_numpy(float)
    dtr = t - lag(t, 1, pos)
    mdtr = pd.Series(dtr).groupby(px["date"].to_numpy()).transform("mean").to_numpy()
    f["coin20"] = M(px["r"].to_numpy() * np.sign(dtr - mdtr), 20, pos)


def fam_vc(px, f):
    pos = px["_pos"].to_numpy()
    h, l_, c = (px[k].to_numpy(float) for k in ("high", "low", "close"))
    r = px["r"].to_numpy()
    f["rstd5_60"] = SD(r, 5, pos) / SD(r, 60, pos)
    f["rstd10_120"] = SD(r, 10, pos) / SD(r, 120, pos)
    rg = (h - l_) / c
    f["rng5_60"] = M(rg, 5, pos) / M(rg, 60, pos)
    box20 = MX(h, 20, pos) / MN(l_, 20, pos) - 1
    box120 = MX(h, 120, pos) / MN(l_, 120, pos) - 1
    f["box20_120"] = box20 / box120
    f["box60"] = MX(c, 60, pos) / MN(c, 60, pos) - 1
    hp = lag(h, 1, pos)
    f["brk20"] = c / MX(hp, 20, pos) - 1
    f["brk60"] = c / MX(hp, 60, pos) - 1
    # 目录 F008/F009：多均线离散度（价格、换手）
    mas = np.vstack([c] + [M(c, d, pos) for d in (5, 10, 20, 60, 120)])
    f["pcf_conv"] = -np.log1p(mas.std(axis=0) / c)
    t = px["turnover"].to_numpy(float)
    mts = np.vstack([t] + [M(t, d, pos) for d in (5, 10, 20, 60, 120)])
    with np.errstate(all="ignore"):
        f["trcf_conv"] = -np.log1p(mts.std(axis=0) / mts[-1])
    f["band60"] = roll(c, 60, lambda v: (np.abs(v / v[:, -1:] - 1) <= 0.10).mean(axis=1), pos)


def fam_hi(px, f):
    pos = px["_pos"].to_numpy()
    h, l_, c = (px[k].to_numpy(float) for k in ("high", "low", "close"))
    for d in (120, 250):
        f[f"dh{d}"] = c / MX(h, d, pos) - 1
        f[f"dl{d}"] = c / MN(l_, d, pos) - 1
    for d in (60, 120, 250):
        f[f"since_hi{d}"] = (d - 1 - roll(h, d, lambda v: v.argmax(axis=1).astype(float), pos)) / d
    mx, mn = MX(h, 250, pos), MN(l_, 250, pos)
    f["pos250"] = (c - mn) / (mx - mn + 1e-12)


def fam_ext(px, f):
    pos = px["_pos"].to_numpy()
    r = px["r"].to_numpy()
    m0 = px["m0"].to_numpy()
    f["max3_20"] = roll(r, 20, lambda v: np.sort(v, axis=1)[:, -3:].mean(axis=1), pos)
    f["max1_60"] = MX(r, 60, pos)
    f["min1_20"] = MN(r, 20, pos)

    def skew(v):
        z = v - v.mean(axis=1, keepdims=True)
        s = np.sqrt((z * z).mean(axis=1))
        with np.errstate(all="ignore"):
            return (z ** 3).mean(axis=1) / s ** 3

    def kurt(v):
        z = v - v.mean(axis=1, keepdims=True)
        s2 = (z * z).mean(axis=1)
        with np.errstate(all="ignore"):
            return (z ** 4).mean(axis=1) / s2 ** 2
    # 目录 F018：过去 250 天里已关闭的窗口内出现过 20 天涨 50% 吗（窗口 u-19..u 的最高 / u-19 开盘）
    h, o = px["high"].to_numpy(float), px["open"].to_numpy(float)
    z = MX(h, 20, pos) / lag(o, 19, pos) - 1
    f["jackpot250"] = roll(np.nan_to_num(z, nan=0.0), 231, lambda v: v.max(axis=1), pos)
    f["skew20"] = roll(r, 20, skew, pos)
    f["skew60"] = roll(r, 60, skew, pos)
    f["kurt60"] = roll(r, 60, kurt, pos)
    for d in (20, 60):
        sr, sm = S(r, d, pos), S(m0, d, pos)
        srm, smm, srr = S(r * m0, d, pos), S(m0 * m0, d, pos), S(r * r, d, pos)
        cov = srm / d - sr * sm / d ** 2
        vm = smm / d - (sm / d) ** 2
        vr = srr / d - (sr / d) ** 2
        with np.errstate(all="ignore"):
            b = cov / vm
            iv = np.sqrt(np.maximum(vr - b * b * vm, 0))
        f[f"ivol{d}"] = iv
        if d == 60:
            f["beta60"] = b
            with np.errstate(all="ignore"):
                f["iv_share60"] = iv / np.sqrt(np.maximum(vr, 1e-12))


def fam_dly(px, f):
    pos = px["_pos"].to_numpy()
    r = px["r"].to_numpy()
    X = [np.ones(len(r)), px["m0"].to_numpy(), px["m1"].to_numpy(), px["m2"].to_numpy()]
    d = 60
    k = len(X)
    A = np.empty((len(r), k, k), np.float64)
    bvec = np.empty((len(r), k), np.float64)
    for i in range(k):
        for j in range(i, k):
            s = S(X[i] * X[j], d, pos)
            A[:, i, j] = s
            A[:, j, i] = s
        bvec[:, i] = S(X[i] * r, d, pos)
    srr = S(r * r, d, pos)
    sr = bvec[:, 0]
    sst = srr - sr * sr / d
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(bvec).all(axis=1) & (sst > 1e-12)
    r2f = np.full(len(r), np.nan)
    lagsum = np.full(len(r), np.nan)
    idx = np.flatnonzero(ok)
    for s0 in range(0, len(idx), 500_000):
        ii = idx[s0:s0 + 500_000]
        try:
            beta = np.linalg.solve(A[ii], bvec[ii][..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = np.stack([np.linalg.lstsq(A[j], bvec[j], rcond=None)[0] for j in ii])
        ssr_fit = (beta * bvec[ii]).sum(axis=1) - sr[ii] ** 2 / d   # 回归平方和
        r2f[ii] = ssr_fit / sst[ii]
        lagsum[ii] = beta[:, 2] + beta[:, 3]
    del A, bvec
    sm, smm, srm = S(X[1], d, pos), S(X[1] * X[1], d, pos), S(X[1] * r, d, pos)
    with np.errstate(all="ignore"):
        cov = srm - sr * sm / d
        r2r = cov * cov / ((smm - sm * sm / d) * sst)
        f["d1"] = 1 - r2r / r2f
    f["lagbeta"] = lagsum


def fam_turn(px, f):
    pos = px["_pos"].to_numpy()
    t = px["turnover"].to_numpy(float)
    r = px["r"].to_numpy()
    amt = px["amount"].to_numpy(float)
    m5, m60, m250 = M(t, 5, pos), M(t, 60, pos), M(t, 250, pos)
    f["t5_60"] = m5 / m60
    f["t5_250"] = m5 / m250
    f["tconc20"] = MX(t, 20, pos) / S(t, 20, pos)
    f["tz60"] = (t - m60) / SD(t, 60, pos)
    f["tlvl250"] = m250
    with np.errstate(all="ignore"):
        f["amihud20"] = M(np.abs(r) / np.where(amt > 0, amt, np.nan) * 1e8, 20, pos)


def fam_vp(px, f):
    pos = px["_pos"].to_numpy()
    t = px["turnover"].to_numpy(float)
    r = px["r"].to_numpy()
    c = px["close"].to_numpy(float)
    dt_ = np.log((t + 1e-6) / (lag(t, 1, pos) + 1e-6))
    f["c_r_dt20"] = BF.roll2(r, dt_, 20, pos)
    f["c_ar_t20"] = BF.roll2(np.abs(r), t, 20, pos)
    f["c_r_tl20"] = BF.roll2(r, lag(t, 1, pos), 20, pos)
    pr = roll(c, 20, lambda v: (v <= v[:, -1:]).mean(axis=1), pos)
    tr = roll(t, 20, lambda v: (v <= v[:, -1:]).mean(axis=1), pos)
    f["div20"] = pr - tr


def fam_rm(px, f):
    pos = px["_pos"].to_numpy()
    r = px["r"].to_numpy()
    m0 = px["m0"].to_numpy()
    d = 120
    sr, sm = S(r, d, pos), S(m0, d, pos)
    srm, smm = S(r * m0, d, pos), S(m0 * m0, d, pos)
    with np.errstate(all="ignore"):
        b = (srm / d - sr * sm / d ** 2) / (smm / d - (sm / d) ** 2)
    # 残差 = r - b*m（b 用截至今天的 120 日贝塔；只用过去）
    e = r - b * m0
    for k in (20, 60, 120):
        f[f"rres{k}"] = S(e, k, pos)
    e_l5 = lag(e, 5, pos)
    f["rres120x5"] = S(e_l5, 115, pos)
    with np.errstate(all="ignore"):
        f["rres60_iv"] = f["rres60"] / (SD(e, 60, pos) * np.sqrt(60))
    c = px["close"].to_numpy(float)
    board = px["code"].map(F.board_of).to_numpy()
    for k in (20, 60):
        rk = c / lag(c, k, pos) - 1
        tmp = pd.DataFrame({"date": px["date"].to_numpy(), "b": board, "x": rk})
        med = tmp.groupby(["date", "b"])["x"].transform("median").to_numpy()
        f[f"rboard{k}"] = rk - med


def fam_sz(px, f):
    """目录 F047/F048/F044/F049：规模、名义价位一律用不复权均价 AMT/V（不用前复权价，防未来除权）。"""
    pos = px["_pos"].to_numpy()
    v, amt = px["volume"].to_numpy(float), px["amount"].to_numpy(float)
    fs = px["outstanding_share"].to_numpy(float)
    with np.errstate(all="ignore"):
        vw = np.where(v > 0, amt / v, np.nan)
        f["nominal_price"] = np.log(vw)
        f["size_raw"] = np.log(fs * vw)
    f["listing_age"] = np.minimum(pos + 1, 750).astype(float)
    t = px["turnover"].to_numpy(float)
    f["tr_own_rank"] = roll(t, 250, lambda x: (x <= x[:, -1:]).mean(axis=1), pos)


def fam_amp(px, f):
    """目录 F040 理想振幅、F041 振幅切割动量、F052 开盘推力、F053 真实波幅上下比。"""
    pos = px["_pos"].to_numpy()
    o, h, l_, c = (px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    r = px["r"].to_numpy()
    amp = h / l_ - 1

    def ideal(v):
        d = v.shape[1] // 2
        cc, aa = v[:, :d], v[:, d:]
        o_ = np.argsort(cc, axis=1)
        k = max(int(round(0.25 * d)), 1)
        lo = np.take_along_axis(aa, o_[:, :k], axis=1).mean(axis=1)
        hi = np.take_along_axis(aa, o_[:, -k:], axis=1).mean(axis=1)
        return hi - lo
    n = len(c)
    # 把 (收盘, 振幅) 两条序列拼成一个 40 宽的窗口：用两次 swv 再拼
    from numpy.lib.stride_tricks import sliding_window_view as swv
    out = np.full(n, np.nan)
    vc, va = swv(c, 20), swv(amp, 20)
    for s0 in range(0, len(vc), 300_000):
        out[19 + s0:19 + s0 + len(vc[s0:s0 + 300_000])] = ideal(np.hstack([vc[s0:s0 + 300_000], va[s0:s0 + 300_000]]))
    out[pos < 19] = np.nan
    f["ideal_amp20"] = out

    def cut(v):
        d = v.shape[1] // 2
        aa, rr = v[:, :d], v[:, d:]
        o_ = np.argsort(aa, axis=1)
        k = int(0.3 * d)
        return np.take_along_axis(rr, o_[:, :k], axis=1).sum(axis=1) - np.take_along_axis(rr, o_[:, -k:], axis=1).sum(axis=1)
    out = np.full(n, np.nan)
    va, vr = swv(amp, 120), swv(r, 120)
    for s0 in range(0, len(va), 100_000):
        out[119 + s0:119 + s0 + len(va[s0:s0 + 100_000])] = cut(np.hstack([va[s0:s0 + 100_000], vr[s0:s0 + 100_000]]))
    out[pos < 119] = np.nan
    f["ampcut120"] = out
    f["open_push20"] = S(h - o, 20, pos) / (S(o - l_, 20, pos) + 1e-9)
    cp = lag(c, 1, pos)
    f["tr_up20"] = S(np.maximum(0, h - cp), 20, pos) / (S(np.maximum(0, cp - l_), 20, pos) + 1e-9)


FAMS = {"sz": fam_sz, "amp": fam_amp, "on": fam_on, "vc": fam_vc, "hi": fam_hi, "ext": fam_ext, "dly": fam_dly, "turn": fam_turn,
        "vp": fam_vp, "rm": fam_rm}


def main():
    t0 = time.time()
    fams = sys.argv[1:] or list(FAMS)
    px = load()
    keep = (px["date"] >= KEEP_FROM).to_numpy()
    print("日线 %d 行，%.0fs" % (len(px), time.time() - t0), flush=True)
    for fam in fams:
        f = {}
        FAMS[fam](px, f)
        out = pd.DataFrame({"code": px["code"].to_numpy()[keep], "date": px["date"].to_numpy()[keep]})
        for k, v in f.items():
            v = np.asarray(v, float)[keep]
            v[~np.isfinite(v)] = np.nan
            out[f"f_{fam}_{k}"] = v.astype("float32")
        cols = [c for c in out.columns if c.startswith("f_")]
        out[cols] = out.groupby("date")[cols].rank(pct=True).astype("float32")
        out.to_parquet(C.FE / f"fam_{fam}.parquet", index=False)
        print("  %s：%d 列，缺失 %.1f%%，%.0fs" % (fam, len(cols), 100 * out[cols].isna().mean().mean(),
                                                time.time() - t0), flush=True)
        del out, f


if __name__ == "__main__":
    main()
