"""
损失函数。完整推导见 docs/learning.md 第 4 节。

核心两句话：

  1. **优化平滑代理，汇报硬指标。** 真前 10 的收益是参数的阶跃函数，
     梯度几乎处处为 0、在名次交换点上跳变，直接优化它就是在噪声面上
     随机游走。软 Top-K 有连续曲面，最优方向一致。

  2. **跨天用 Huber M 估计，不是算术平均。** 任何单日对目标的影响力
     被截在 c·MAD 以内。某天头号选票封涨停（ỹ≈+8），它对参数的推力
     和一个普通好日子完全一样。算术平均没有这个性质。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
#  软 Top-K
# ---------------------------------------------------------------------
def solve_tau(s: np.ndarray, k: float, tol: float = 0.01,
              iters: int = 60) -> float:
    """二分求温度 τ，使 softmax(s/τ) 的困惑度 exp(H) = k。

    每天单独解而不是固定一个 τ：候选池大小和分数分布天天在变，
    固定温度会让池子大的日子实际持仓分散、池子小的日子过度集中。
    """
    n = s.size
    if n <= k:
        return float("inf")            # 池子本来就不够 k 只，等权
    rng = float(s.max() - s.min())
    if rng <= 0:
        return float("inf")
    lo, hi = rng * 1e-4, rng * 1e3
    for _ in range(iters):
        mid = (lo * hi) ** 0.5         # 几何二分，τ 跨好几个数量级
        p = _softmax(s / mid)
        perp = np.exp(-np.sum(p * np.log(p + 1e-300)))
        if abs(perp - k) < tol:
            return mid
        if perp > k:                   # 太分散 -> 降温
            hi = mid
        else:
            lo = mid
    return (lo * hi) ** 0.5


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def day_G(scores: np.ndarray, ytil: np.ndarray, k: int,
          tol: float = 0.01) -> float:
    """单日目标：按分数集中到约 k 只票上的组合，当天的标准化超额收益。

    极限行为：τ→0 全押第一名；τ→∞ 等权全池 -> 0（ỹ 已日内中性化）；
    分数与收益无关时期望为 0。
    """
    if scores.size == 0:
        return 0.0
    if scores.size <= k:
        return float(np.mean(ytil))
    tau = solve_tau(scores, k, tol)
    if not np.isfinite(tau):
        return float(np.mean(ytil))
    return float(np.dot(_softmax(scores / tau), ytil))


# ---------------------------------------------------------------------
#  跨天聚合
# ---------------------------------------------------------------------
def huber_location(x: np.ndarray, w: np.ndarray | None = None,
                   c: float = 1.345, iters: int = 40) -> float:
    """加权 Huber M 估计的位置参数。IRLS 求解。

    σ̂ 用 MAD。c=1.345 是对高斯 95% 效率的标准取值。
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    if w.sum() <= 0:
        return 0.0
    m = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - m)))
    if not np.isfinite(s) or s <= 0:
        return float(np.average(x, weights=w))
    for _ in range(iters):
        u = (x - m) / s
        # Huber 权：|u|<=c 时 1，超出按 c/|u| 衰减 -> 影响力上限 c·s
        hw = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        ww = w * hw
        if ww.sum() <= 0:
            break
        new = float(np.dot(ww, x) / ww.sum())
        if abs(new - m) < 1e-12 * max(1.0, abs(m)):
            m = new
            break
        m = new
    return m


def G_hat(day_scores: list[np.ndarray], day_ytil: list[np.ndarray],
          day_w: np.ndarray | None, k: int, huber_c: float,
          tol: float = 0.01) -> tuple[float, np.ndarray]:
    """返回 (Ĝ, 每日 G_d)。"""
    g = np.array([day_G(s, y, k, tol) for s, y in zip(day_scores, day_ytil)])
    return huber_location(g, day_w, huber_c), g


# ---------------------------------------------------------------------
#  正则
# ---------------------------------------------------------------------
def sigma_of(box: dict[str, list]) -> dict[str, float]:
    """每个参数的尺度 = 箱宽的一半。把所有参数放到同一量纲。"""
    return {k: max((hi - lo) / 2.0, 1e-12) for k, (lo, hi) in box.items()}


def lambda_anchor(lam0: float, n_days: int, prior_days: int) -> float:
    """按证据量衰减的锚定强度：λ₀ · n₀/(n₀+N)。

    标准贝叶斯收缩。好处是不需要人为设「多少天后开始学」的开关，
    权重连续过渡，而且早期天然保守：N=8 时还有 0.94 λ₀，基本冻结。
    """
    return lam0 * prior_days / max(prior_days + n_days, 1)


def penalty(theta: dict[str, float], theta0: dict[str, float],
            theta_prev: dict[str, float], box: dict[str, list],
            lam_a: float, lam_1: float) -> float:
    sig = sigma_of(box)
    l2 = sum(((theta[k] - theta0[k]) / sig[k]) ** 2 for k in box)
    l1 = sum(abs(theta[k] - theta_prev[k]) / sig[k] for k in box)
    return lam_a * l2 + lam_1 * l1


def project(theta: dict[str, float], box: dict[str, list],
            weight_prefix: str = "scoring.weights.") -> dict[str, float]:
    """投影回可行域：所有参数夹进箱，且权重那组的和恰好为 1。

    「先夹箱再按比例归一」是**错的**，不收敛：归一会把值推出箱，再夹回来
    和又不是 1，来回震荡。自测里 scoring.weights.position 就是这么跑出去的。

    正解是投影到「箱 ∩ 超平面 {Σw=1}」的交集上。欧氏投影的解形如

        w_k' = clip(w_k + λ, lo_k, hi_k)

    λ 是唯一的拉格朗日乘子。左边关于 λ 单调不减，所以二分必然收敛，
    而且给出的是**离原点最近**的可行点，不会无谓地扰动其他参数。
    """
    t = {k: float(np.clip(v, box[k][0], box[k][1])) for k, v in theta.items()}
    wk = [k for k in t if k.startswith(weight_prefix)]
    if not wk:
        return t

    lo = np.array([box[k][0] for k in wk], float)
    hi = np.array([box[k][1] for k in wk], float)
    w = np.array([theta[k] for k in wk], float)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        # 箱本身就装不下「和为 1」。配置写错了，退回等比缩放而不是崩掉。
        v = np.clip(w, lo, hi)
        v = v / v.sum() if v.sum() > 0 else np.full_like(v, 1.0 / v.size)
        for k, x in zip(wk, v):
            t[k] = float(x)
        return t

    def total(lam: float) -> float:
        return float(np.clip(w + lam, lo, hi).sum())

    a, b = float((lo - w).min()), float((hi - w).max())
    for _ in range(80):
        mid = (a + b) / 2.0
        if total(mid) < 1.0:
            a = mid
        else:
            b = mid
    for k, x in zip(wk, np.clip(w + (a + b) / 2.0, lo, hi)):
        t[k] = float(x)
    return t
