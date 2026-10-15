"""
按天配对差值序列的稳健推断（研究用，不进生产）。

evalkit.compare 的 t_day = mean(d) / (sd(d)/sqrt(n))，把每天当独立观测。标签是未来 20 根的最高价，
相邻交易日的名单和结果高度重叠，d 有正自相关，这个 t 偏大。这里给四样东西：

    iid_t         evalkit 原口径（对账用）
    nw_t          Newey-West（Bartlett 核），滞后 L 可选，报 5/10/20/30 的敏感性
    mbb           移动块自举（块长 20 个交易日），双侧 p（中心化）和百分位置信区间
    acf / n_eff   自相关前 25 阶、有效样本量 n / (1 + 2 Σ (1-k/(L+1)) ρ_k)（和 NW 同一个核）

pooled_boot 是给「子集命中 vs 另一子集命中」这种不平衡按天聚类的比较用：
按天块重抽，比的是两个合并命中率之差（分子分母都跟着重抽）。

    python research/stats/hac.py        AR(1) 模拟自检：rho=0 时 NW≈iid，rho=0.5 时 iid 的 t 偏大约 sqrt(3) 倍
"""
from __future__ import annotations

import math

import numpy as np

LAGS = (5, 10, 20, 30)
BLOCK = 20
B = 5000
SEED = 20261014


def _arr(d) -> np.ndarray:
    x = np.asarray(d, float)
    return x[np.isfinite(x)]


def iid_t(d) -> dict:
    x = _arr(d)
    n = len(x)
    if n < 3:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "t": float("nan")}
    se = float(x.std(ddof=1) / math.sqrt(n))
    return {"n": n, "mean": float(x.mean()), "se": se, "t": float(x.mean() / se) if se > 0 else float("nan")}


def autocov(x: np.ndarray, k: int) -> float:
    n = len(x)
    xc = x - x.mean()
    return float((xc[k:] * xc[:n - k]).sum() / n)


def nw_t(d, lag: int) -> dict:
    """Newey-West：长期方差 = γ0 + 2 Σ_{k=1..L} (1 - k/(L+1)) γk，se = sqrt(LRV / n)。"""
    x = _arr(d)
    n = len(x)
    if n < lag + 3:
        return {"lag": lag, "se": float("nan"), "t": float("nan")}
    lrv = autocov(x, 0)
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * autocov(x, k)
    lrv = max(lrv, 1e-18)
    se = math.sqrt(lrv / n)
    return {"lag": lag, "se": se, "t": float(x.mean() / se)}


def acf(d, kmax: int = 25) -> list[float]:
    x = _arr(d)
    if len(x) < 3:
        return [float("nan")] * kmax
    g0 = autocov(x, 0)
    if g0 <= 0:
        return [float("nan")] * kmax
    return [autocov(x, k) / g0 if k < len(x) - 1 else float("nan") for k in range(1, kmax + 1)]


def n_eff(d, lag: int = 20) -> float:
    """有效样本量：和 NW(lag) 同一个 Bartlett 核，n_eff = n * γ0 / LRV。"""
    x = _arr(d)
    n = len(x)
    if n < lag + 3:
        return float("nan")
    g0 = autocov(x, 0)
    nw = nw_t(x, lag)
    if not np.isfinite(nw["se"]) or g0 <= 0:
        return float("nan")
    lrv = nw["se"] ** 2 * n
    return float(n * g0 / lrv)


def _mbb_index(n: int, block: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """移动块自举的下标矩阵 (b, n)：每行由 ceil(n/block) 个长 block 的连续块拼成再截断。"""
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(b, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, nb * block)
    return idx[:, :n]


def mbb(d, block: int = BLOCK, b: int = B, seed: int = SEED) -> dict:
    """均值的移动块自举。p 是双侧中心化 p：P(|m* - m| >= |m|)。"""
    x = _arr(d)
    n = len(x)
    if n < 2 * block:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "t": float("nan")}
    rng = np.random.default_rng(seed)
    m = float(x.mean())
    ms = x[_mbb_index(n, block, b, rng)].mean(axis=1)
    se = float(ms.std(ddof=1))
    p = float((np.abs(ms - m) >= abs(m)).mean())
    lo, hi = np.quantile(ms, [0.025, 0.975])
    return {"p": max(p, 1.0 / b), "lo": float(lo), "hi": float(hi), "se": se, "t": m / se if se > 0 else float("nan"),
            "block": block, "b": b}


def pooled_boot(ka, na, kb=None, nb=None, ref=None, block: int = BLOCK, b: int = B, seed: int = SEED) -> dict:
    """按天块重抽的合并命中率（之差）。

    ka/na：每天 A 子集的命中数 / 名额数（同一条日期轴，没名额的天填 0）。
    kb/nb 给了就比 A - B；ref 给了（每天的基准率，按 A 的名额加权）就比 A - ref。
    返回点估计、自举 se、95% 区间、双侧 p（中心化）。"""
    ka, na = np.asarray(ka, float), np.asarray(na, float)
    n = len(ka)
    rng = np.random.default_rng(seed)
    idx = _mbb_index(n, block, b, rng)

    def stat(ii):
        sa = ka[ii].sum(axis=-1) / np.maximum(na[ii].sum(axis=-1), 1e-12)
        if kb is not None:
            sb = np.asarray(kb, float)[ii].sum(axis=-1) / np.maximum(np.asarray(nb, float)[ii].sum(axis=-1), 1e-12)
            return sa - sb
        if ref is not None:
            r = np.asarray(ref, float)
            return sa - (r[ii] * na[ii]).sum(axis=-1) / np.maximum(na[ii].sum(axis=-1), 1e-12)
        return sa

    m = float(stat(np.arange(n)))
    ms = stat(idx)
    se = float(ms.std(ddof=1))
    lo, hi = np.quantile(ms, [0.025, 0.975])
    p = float((np.abs(ms - m) >= abs(m)).mean())
    return {"est": m, "se": se, "lo": float(lo), "hi": float(hi), "p": max(p, 1.0 / b),
            "t": m / se if se > 0 else float("nan")}


def full(d, label: str = "") -> dict:
    """一条差值序列的全套：iid / NW 四个滞后 / MBB / ACF 前 25 阶 / n_eff。"""
    x = _arr(d)
    out = {"label": label, **{f"iid_{k}": v for k, v in iid_t(x).items()}}
    for L in LAGS:
        out[f"nw{L}_t"] = nw_t(x, L)["t"]
    bb = mbb(x)
    out.update({"mbb_p": bb["p"], "mbb_lo": bb["lo"], "mbb_hi": bb["hi"], "mbb_t": bb["t"]})
    a = acf(x)
    out["acf"] = a
    out["acf1"], out["acf5"], out["acf10"], out["acf20"] = a[0], a[4], a[9], a[19]
    out["acf_sum1_20"] = float(np.nansum(a[:20]))
    out["n_eff20"] = n_eff(x, 20)
    return out


def norm_p(t: float) -> float:
    """双侧正态 p。"""
    if not np.isfinite(t):
        return float("nan")
    return float(math.erfc(abs(t) / math.sqrt(2)))


def _selftest() -> None:
    rng = np.random.default_rng(1)
    for rho in (0.0, 0.5, 0.8):
        ts_iid, ts_nw, rej_iid, rej_nw, rej_bb = [], [], 0, 0, 0
        R = 300
        for _ in range(R):
            e = rng.normal(size=400)
            x = np.empty(400)
            x[0] = e[0]
            for i in range(1, 400):
                x[i] = rho * x[i - 1] + e[i]
            a, w = iid_t(x)["t"], nw_t(x, 20)["t"]
            ts_iid.append(a)
            ts_nw.append(w)
            rej_iid += abs(a) > 1.96
            rej_nw += abs(w) > 1.96
            rej_bb += mbb(x, b=400, seed=int(rng.integers(1e9)))["p"] < 0.05
        print("AR(1) rho=%.1f  原假设下拒绝率（名义 5%%）：iid %.1f%%  NW20 %.1f%%  块自举 %.1f%%  iid t 的离散 %.2f  NW %.2f"
              % (rho, 100 * rej_iid / R, 100 * rej_nw / R, 100 * rej_bb / R, np.std(ts_iid), np.std(ts_nw)))


if __name__ == "__main__":
    _selftest()
