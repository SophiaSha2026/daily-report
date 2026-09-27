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
    if not np.all(np.isfinite(s)):
        # 分数里混进 NaN 的话 rng 是 NaN，下面 60 轮二分的比较全是 False，
        # 最后返回 NaN，调用方又把 NaN 当成「池子不够 k 只」静默等权。
        # 2026-09-15 的 auc_ratio 那次就是这么让 404 天里 159 天的目标函数
        # 和参数无关的（vscore.py:103 的注释）。宁可炸也不要再静默一次。
        raise ValueError(
            f"solve_tau: 分数含非有限值 {int((~np.isfinite(s)).sum())} 行")
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

    极限行为：τ→0 全押第一名；τ→∞ 等权**硬性排除之后的那个子集** -> 该子集
    的 mean(ỹ)。注意不是 0：ỹ 是按当日全池中位数中性化的，准入后的子集均值
    系统性偏正（413 天实测 p10/p50/p90 = −0.51 / +0.21 / +1.01，75% 的天
    |均值| > 0.2）。以前这里和 docs 都写成「→ 0」，是个不成立的定理（F1-7）。

    空池天返回 NaN 而不是 0.0：一只不剩就是当天没有持仓，对参数没有任何
    信息，不是「超额收益 0」。0.0 会被当成一个正常观测挤进 Huber 聚合和
    自助（闸门 7 的在线 Problem 只有 17 天，一个假 0 就占 1/17 权重）。
    """
    if scores.size == 0:
        return float("nan")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(ytil)):
        # 上游 Problem._day_arrays 已经按 isfinite 过滤过，走到这里说明有新的
        # 入口绕开了它。静默退化（返回 mean(ytil)，与 θ 无关）比崩掉难查得多。
        raise ValueError(
            "day_G: 分数/收益含非有限值 "
            f"{int((~np.isfinite(scores)).sum())}/{int((~np.isfinite(ytil)).sum())} 行")
    if scores.size <= k:
        return float(np.mean(ytil))
    tau = solve_tau(scores, k, tol)
    if not np.isfinite(tau):
        return float(np.mean(ytil))    # 只剩「分数全相同」这一条路
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
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    # w=0 的天（LLM 判「数据异常」）必须在**算起点和尺度之前**就剔掉。
    # 以前只有下面的 IRLS 带权，median 和 MAD 用的是含 w=0 天的全部 x：
    # 那些天照样决定 σ̂，进而决定其它天的截断阈值 c·σ̂。实测注入 10 个
    # w=0 的极端日，闸门 2 的 ΔĜ 漂 3.7e-3（当次闸门差距的 32%）；异常日
    # 过半时 σ̂ 完全由它们决定，Huber 直接退化成算术平均（F1-4）。
    # bootstrap_better 一直是先剔再抽，剔完两个闸门口径才一致。
    keep = w > 0
    x, w = x[keep], w[keep]
    if x.size == 0 or w.sum() <= 0:
        return 0.0
    m = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - m)))
    if not np.isfinite(s) or s <= 0:
        # MAD=0：半数以上样本取同一个值。退回**加权中位数**（Huber c→0 的
        # 极限），不是加权均值——均值在这个分支上完全没有影响力上限，
        # 6 个 0 加一个 5.0 就被拖到 0.502，「单日暴走不能主导」当场失效（F1-3）
        o = np.argsort(x, kind="stable")
        cw = np.cumsum(w[o])
        return float(x[o][min(int(np.searchsorted(cw, cw[-1] / 2.0)),
                              x.size - 1)])
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


def aggregate(g: np.ndarray, day_w: np.ndarray | None = None,
              huber_c: float = 1.345) -> float:
    """把逐日 G_d 聚成 Ĝ。

    非有限的天整天剔除：day_G 只在「硬性排除后一只不剩」时返回 NaN，那天
    根本没有持仓，对参数没有任何信息（F1-7）。day_w 按同一个掩码对齐，
    所以调用方拿到的 g 数组长度不变、和 dates 一一对应。
    """
    g = np.asarray(g, float)
    ok = np.isfinite(g)
    w = None if day_w is None else np.asarray(day_w, float)[ok]
    return huber_location(g[ok], w, huber_c)


def G_hat(day_scores: list[np.ndarray], day_ytil: list[np.ndarray],
          day_w: np.ndarray | None, k: int, huber_c: float,
          tol: float = 0.01) -> tuple[float, np.ndarray]:
    """返回 (Ĝ, 每日 G_d)。g 原样返回（含空池天的 NaN），按 dates 对齐。"""
    g = np.array([day_G(s, y, k, tol) for s, y in zip(day_scores, day_ytil)])
    return aggregate(g, day_w, huber_c), g


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
