"""
换条路研究（research/altpath）第 0 步：从 data/breakout/daily.parquet 自己算三份东西。

    a158.parquet   Alpha158 风格量价特征（qlib 口径，窗口 5/10/20/30/60），当日全市场排名归一
    lu.parquet     涨停生态：个股层（首板 / 连板 / 炸板 / 近期涨停次数）+ 市场层（同一天所有票同值）
    env.parquet    每日环境（只用当天收盘及以前）：成交额相对 60 日、涨停家数、炸板率、上涨占比、
                   昨日涨停今日表现、连板高度、已关闭窗口的滚动基准率

只读 daily.parquet 和 buyable 的标签缓存，不联网，不碰 src/ 和训练表。输出行只保留
2023-12-01 .. 2025-12-31（训练表起点到开发集终点），确认集 2026 年整段不落盘。

    python research/altpath/build_feats.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
import datasource as ds  # noqa: E402

OUT = ROOT / "data" / "breakout" / "raw" / "exp" / "altpath"
KEEP_FROM, KEEP_TO = "2023-12-01", "2025-12-31"
WINS = (5, 10, 20, 30, 60)
CHUNK = 300_000


def roll(x: np.ndarray, d: int, fn, pos: np.ndarray) -> np.ndarray:
    """长表上的滑窗（按 code,date 排好序）：fn(视图块 (m, d)) -> (m,)。
    窗口跨了 code 边界的行（该票自己的第 d 根之前）置 NaN。"""
    n = len(x)
    out = np.full(n, np.nan)
    if n < d:
        return out
    v = swv(x, d)                         # 第 j 行 = x[j : j+d]，对应输出行 j+d-1
    res = np.empty(len(v))
    for s in range(0, len(v), CHUNK):
        res[s:s + CHUNK] = fn(v[s:s + CHUNK])
    out[d - 1:] = res
    out[pos < d - 1] = np.nan
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(all="ignore"):
        return np.where(den > 1e-12, (a * b).sum(axis=1) / den, np.nan)


def roll2(x: np.ndarray, y: np.ndarray, d: int, pos: np.ndarray) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    vx, vy = swv(x, d), swv(y, d)
    res = np.empty(len(vx))
    for s in range(0, len(vx), CHUNK):
        a, b = vx[s:s + CHUNK], vy[s:s + CHUNK]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        r = _corr(np.nan_to_num(a), np.nan_to_num(b))
        res[s:s + CHUNK] = np.where(ok, r, np.nan)
    out[d - 1:] = res
    out[pos < d - 1] = np.nan
    return out


def _slope_r2_resid(v: np.ndarray):
    d = v.shape[1]
    t = np.arange(d, dtype=float)
    t = t - t.mean()
    ym = v.mean(axis=1, keepdims=True)
    yc = v - ym
    stt = (t * t).sum()
    b = (yc * t).sum(axis=1) / stt
    fit = b[:, None] * t
    sse = ((yc - fit) ** 2).sum(axis=1)
    sst = (yc * yc).sum(axis=1)
    with np.errstate(all="ignore"):
        r2 = np.where(sst > 1e-12, 1 - sse / sst, np.nan)
    resid_last = yc[:, -1] - fit[:, -1]
    return b, r2, resid_last


class _F(dict):
    """只存要落盘的那些行，float32，省内存（4.4M 行 x 158 列 float64 要 5.5 GB）。"""
    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def __setitem__(self, k, v):
        v = np.asarray(v, float)[self.keep]
        v[~np.isfinite(v)] = np.nan
        super().__setitem__(k, v.astype("float32"))


def alpha158(px: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    o, h, l, c = (px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = px["volume"].to_numpy(float)
    pos = px["_pos"].to_numpy()
    first = pos == 0
    cp = np.r_[np.nan, c[:-1]]
    cp[first] = np.nan
    vp = np.r_[np.nan, v[:-1]]
    vp[first] = np.nan
    f = _F(keep)
    eps = 1e-12
    hl = h - l + eps
    f["a_kmid"] = (c - o) / o
    f["a_klen"] = (h - l) / o
    f["a_kmid2"] = (c - o) / hl
    f["a_kup"] = (h - np.maximum(o, c)) / o
    f["a_kup2"] = (h - np.maximum(o, c)) / hl
    f["a_klow"] = (np.minimum(o, c) - l) / o
    f["a_klow2"] = (np.minimum(o, c) - l) / hl
    f["a_ksft"] = (2 * c - h - l) / o
    f["a_ksft2"] = (2 * c - h - l) / hl
    f["a_open0"] = o / c
    f["a_high0"] = h / c
    f["a_low0"] = l / c
    # VWAP0 不要：amount/volume 是不复权价，除以前复权收盘价量到的是累计分红因子，不是量价
    ret = c / cp - 1
    lv = np.log(v + 1)
    dlv = np.log(v / np.where(vp > 0, vp, np.nan) + 1)
    up = np.where(np.isfinite(ret), np.maximum(c - cp, 0), np.nan)
    dn = np.where(np.isfinite(ret), np.maximum(cp - c, 0), np.nan)
    ab = np.abs(c - cp)
    vup = np.maximum(v - vp, 0)
    vdn = np.maximum(vp - v, 0)
    vab = np.abs(v - vp)
    wv = np.abs(ret) * v
    sw = lambda a, d: roll(a, d, lambda x: x.sum(axis=1), pos)  # noqa: E731
    for d in WINS:
        cs = np.r_[np.full(d, np.nan), c[:-d]]
        cs[pos < d] = np.nan
        f[f"a_roc{d}"] = cs / c
        f[f"a_ma{d}"] = roll(c, d, lambda x: x.mean(axis=1), pos) / c
        f[f"a_std{d}"] = roll(c, d, lambda x: x.std(axis=1), pos) / c
        reg = {}

        def _reg(x, reg=reg):
            b, r2, rs = _slope_r2_resid(x)
            reg.setdefault("b", []).append(b)
            reg.setdefault("r2", []).append(r2)
            reg.setdefault("rs", []).append(rs)
            return b
        roll(c, d, _reg, pos)
        for k, name in (("b", "beta"), ("r2", "rsqr"), ("rs", "resi")):
            arr = np.full(len(c), np.nan)
            arr[d - 1:] = np.concatenate(reg[k])
            arr[pos < d - 1] = np.nan
            f[f"a_{name}{d}"] = arr / c if name != "rsqr" else arr
        mx = roll(h, d, lambda x: x.max(axis=1), pos)
        mn = roll(l, d, lambda x: x.min(axis=1), pos)
        f[f"a_max{d}"] = mx / c
        f[f"a_min{d}"] = mn / c
        f[f"a_qtlu{d}"] = roll(c, d, lambda x: np.quantile(x, 0.8, axis=1), pos) / c
        f[f"a_qtld{d}"] = roll(c, d, lambda x: np.quantile(x, 0.2, axis=1), pos) / c
        f[f"a_rank{d}"] = roll(c, d, lambda x: (x <= x[:, -1:]).mean(axis=1), pos)
        f[f"a_rsv{d}"] = (c - mn) / (mx - mn + eps)
        imax = roll(h, d, lambda x: x.argmax(axis=1).astype(float), pos)
        imin = roll(l, d, lambda x: x.argmin(axis=1).astype(float), pos)
        # qlib 的 IdxMax 是「距今多少根」：窗口最后一根 = 0
        f[f"a_imax{d}"] = (d - 1 - imax) / d
        f[f"a_imin{d}"] = (d - 1 - imin) / d
        f[f"a_imxd{d}"] = (imax - imin) / d
        f[f"a_corr{d}"] = roll2(c, lv, d, pos)
        f[f"a_cord{d}"] = roll2(c / cp, dlv, d, pos)
        f[f"a_cntp{d}"] = roll(np.where(np.isfinite(ret), (ret > 0).astype(float), np.nan), d,
                               lambda x: x.mean(axis=1), pos)
        f[f"a_cntn{d}"] = roll(np.where(np.isfinite(ret), (ret < 0).astype(float), np.nan), d,
                               lambda x: x.mean(axis=1), pos)
        sab = sw(ab, d) + eps
        f[f"a_sump{d}"] = sw(up, d) / sab
        f[f"a_sumn{d}"] = sw(dn, d) / sab
        f[f"a_vma{d}"] = roll(v, d, lambda x: x.mean(axis=1), pos) / (v + eps)
        f[f"a_vstd{d}"] = roll(v, d, lambda x: x.std(axis=1), pos) / (v + eps)
        f[f"a_wvma{d}"] = (roll(wv, d, lambda x: x.std(axis=1), pos)
                           / (roll(wv, d, lambda x: x.mean(axis=1), pos) + eps))
        svab = sw(vab, d) + eps
        f[f"a_vsump{d}"] = sw(vup, d) / svab
        f[f"a_vsumn{d}"] = sw(vdn, d) / svab
        print("    a158 窗口 %d 完成" % d, flush=True)
    out = pd.DataFrame(dict(f))
    out.insert(0, "date", px["date"].to_numpy()[keep])
    out.insert(0, "code", px["code"].to_numpy()[keep])
    return out


def limit_flags(px: pd.DataFrame) -> pd.DataFrame:
    """涨停 / 炸板 / 跌停，口径同 pullback.prepare（收在最高价且不低于涨停价 1.5 分以上）。"""
    pos = px["_pos"].to_numpy()
    c = px["close"].to_numpy(float)
    h = px["high"].to_numpy(float)
    l_ = px["low"].to_numpy(float)
    cp = np.r_[np.nan, c[:-1]]
    cp[pos == 0] = np.nan
    codes = pd.Series(px["code"].unique())
    lim = dict(zip(codes, codes.map(lambda x: ds.limit_pct(x, ""))))
    pct = px["code"].map(lim).astype(float)
    lp = np.asarray(ds.limit_price_arr(pd.Series(cp), pct.reset_index(drop=True),
                                       px["code"].reset_index(drop=True)), float)
    dp = np.asarray(ds.limit_price_arr(pd.Series(cp), -pct.reset_index(drop=True),
                                       px["code"].reset_index(drop=True)), float) \
        if False else cp * (1 - pct.to_numpy() / 100.0)
    gain = c / cp - 1
    ok = np.isfinite(lp)
    lu = ok & (c >= h - 0.005) & (c >= lp - 0.015) & (gain > 0)
    zb = ok & (h >= lp - 0.005) & ~lu
    ld = ok & (c <= np.round(dp, 2) + 0.015) & (gain < 0) & (c <= l_ + 0.005)
    return pd.DataFrame({"lu": lu, "zb": zb, "ld": ld, "ret": gain})


def lu_features(px: pd.DataFrame, fl: pd.DataFrame, lab: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = px["_pos"].to_numpy()
    code = px["code"].to_numpy()
    lu = fl["lu"].to_numpy()
    zb = fl["zb"].to_numpy()
    n = len(px)
    # 连板高度：今天封板则 = 昨天高度 + 1
    streak = np.zeros(n)
    for i in range(n):
        if lu[i]:
            streak[i] = (streak[i - 1] + 1) if (i > 0 and pos[i] > 0) else 1
    luf = lu.astype(float)
    zbf = zb.astype(float)
    cnt = {}
    for d in (5, 20, 60):
        cnt[d] = roll(luf, d, lambda x: x.sum(axis=1), pos)
    zb20 = roll(zbf, 20, lambda x: x.sum(axis=1), pos)
    # 首板：今天封板，此前 10 根没封过
    prev10 = roll(np.r_[0.0, luf[:-1]], 10, lambda x: x.sum(axis=1), pos)
    prev10[pos == 0] = 0
    first_board = lu & (np.nan_to_num(prev10, nan=0) == 0)
    # 距上次封板几根（不含今天），上限 60
    since = np.full(n, 60.0)
    last = -10 ** 9
    for i in range(n):
        if pos[i] == 0:
            last = -10 ** 9
        since[i] = min(60.0, i - last) if last > -10 ** 9 else 60.0
        if lu[i]:
            last = i
    # 昨日涨停、今日的个股表现（赚钱效应的个股版：自己上次封板后的次日涨幅）
    ret = fl["ret"].to_numpy()
    after_lu = np.full(n, np.nan)
    lastret = np.nan
    for i in range(n):
        if pos[i] == 0:
            lastret = np.nan
        if i > 0 and pos[i] > 0 and lu[i - 1]:
            lastret = ret[i]
        after_lu[i] = lastret
    ind = pd.DataFrame({"code": code, "date": px["date"].to_numpy(),
                        "l_lu": luf, "l_zb": zbf, "l_streak": streak,
                        "l_first": first_board.astype(float),
                        "l_cnt5": cnt[5], "l_cnt20": cnt[20], "l_cnt60": cnt[60], "l_zb20": zb20,
                        "l_since": since, "l_after_lu_ret": after_lu})
    # 市场层（同一天同值）
    m = pd.DataFrame({"date": px["date"].to_numpy(), "lu": lu, "zb": zb, "ld": fl["ld"].to_numpy(),
                      "ret": ret, "streak": streak, "amount": px["amount"].to_numpy(float)})
    m["prev_lu"] = np.r_[False, lu[:-1]] & (pos > 0)
    m["prev_streak"] = np.r_[0.0, streak[:-1]] * (pos > 0)
    g = m.groupby("date")
    env = pd.DataFrame({
        "n_lu": g["lu"].sum(), "n_zb": g["zb"].sum(), "n_ld": g["ld"].sum(),
        "adv_share": g["ret"].apply(lambda s: float((s > 0).mean())),
        "max_streak": g["streak"].max(), "amount": g["amount"].sum(),
        "n": g["ret"].size(),
    })
    env["zb_rate"] = env["n_zb"] / (env["n_lu"] + env["n_zb"]).clip(lower=1)
    pl = m[m["prev_lu"]].groupby("date")["ret"].mean()
    env["prev_lu_ret"] = pl                     # 昨日涨停今日平均涨幅（赚钱效应）
    ps = m[m["prev_streak"] >= 2].groupby("date")["ret"].mean()
    env["prev_lb_ret"] = ps                     # 昨日连板今日平均涨幅
    env = env.sort_index()
    # 只保留开市日（有 K 线的票 >= 1000）
    env = env[env["n"] >= 1000]
    env["amt_ratio60"] = env["amount"] / env["amount"].rolling(60, min_periods=20).mean().shift(1)
    env["lu_share"] = env["n_lu"] / env["n"]
    # 已关闭窗口的滚动基准率：第 t 天能知道的只有标签窗口已关闭的日子（t-21 及更早）
    lab = lab[np.isfinite(lab["y_open"])]
    br = lab.groupby("date")["y_open"].mean().reindex(env.index)
    env["base_closed20"] = br.shift(21).rolling(20, min_periods=10).mean()
    env["prev_lu_ret"] = env["prev_lu_ret"].astype(float)
    env.index.name = "date"
    env = env.reset_index()
    # 个股层并上市场层（模型臂用）
    mk = env[["date", "n_lu", "zb_rate", "max_streak", "prev_lu_ret", "prev_lb_ret",
              "adv_share", "amt_ratio60", "lu_share"]].rename(
        columns=lambda x: x if x == "date" else "m_" + x)
    ind = ind.merge(mk, on="date", how="left")
    return ind, env


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    px["date"] = px["date"].astype(str).str[:10]
    px["code"] = px["code"].astype(str).str.zfill(6)
    # 确认集之后的日子一律不读（特征最多看到 2025-12-31）
    px = px[px["date"] <= KEEP_TO]
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    print("日线 %d 行，%.0fs" % (len(px), time.time() - t0), flush=True)

    import buyable as B
    lab = B.open_labels()
    lab = lab[lab["date"] <= KEEP_TO]
    fl = limit_flags(px)
    ind, env = lu_features(px, fl, lab)
    env.to_parquet(OUT / "env.parquet", index=False)
    keep = (px["date"] >= KEEP_FROM).to_numpy()
    assert (ind["date"].to_numpy() == px["date"].to_numpy()).all()
    ind = ind[keep].reset_index(drop=True)
    # 个股层的连续量做当日排名（教训：绝对计数随行情漂，实验 16 的 g1）
    for c in ("l_cnt20", "l_cnt60", "l_zb20", "l_since", "l_after_lu_ret"):
        ind[c + "_pct"] = ind.groupby("date")[c].rank(pct=True)
    num = [c for c in ind.columns if c not in ("code", "date")]
    ind[num] = ind[num].astype("float32")
    ind.to_parquet(OUT / "lu.parquet", index=False)
    print("涨停生态 %d 行 %d 列，env %d 天，%.0fs" % (len(ind), len(num), len(env), time.time() - t0),
          flush=True)
    del ind

    a = alpha158(px, keep)
    cols = [c for c in a.columns if c.startswith("a_")]
    print("a158 原值 %d 列，开始当日排名 %.0fs" % (len(cols), time.time() - t0), flush=True)
    for i in range(0, len(cols), 20):
        cc = cols[i:i + 20]
        a[cc] = a.groupby("date")[cc].rank(pct=True).astype("float32")
    a.to_parquet(OUT / "a158.parquet", index=False)
    print("a158 %d 行 %d 列，%.0fs -> %s" % (len(a), len(cols), time.time() - t0, OUT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
