"""板块校正因子：经验贝叶斯收缩（随机效应），只用过去的月份估。

为什么要有这个模块
------------------
生产按分数跨板块混排取前 10，所以**分数在各板块之间必须是一个意思**。
模型对北交所严重超配，而各板块的真实命中率不一样，于是有了 `daily.BOARD_ADJ`。
在 2026-09-16 之前，那四个数是人按验证集 2025-03~12 的命中率÷整体命中率
手写死的，有两个毛病：

1. **样本内**。`exp_window.py` 又在同一段数据上乘着它评成绩，邮件里印的
   命中率含样本内增益（实测连续≥2 天那一档约 3.9 个百分点是这么来的）。
2. **小样本会掀桌子**。`exp_window.board_factors` 那条只用过去月份的滚动臂
   解决了第 1 点，但它对每个板块**各信各的**：北交所在验证集里只有 19 个名额，
   命中率 26.3% 直接变成因子 2.37，清单构成从「主板 67% / 科创 30%」变成
   「主板 54% / 北交 31%」—— 那已经是另一套策略，不是同一套策略的更诚实估法。
   同一批数据劈两半重算，bj 因子 1.52 -> 0.62、main 0.69 -> 1.53，段间抖动
   远大于因子之间的差距。

收缩解决的就是第 2 点：**每个板块该被信多少，由它自己的样本量和板块之间
真实差异的大小共同决定**，不用人去挑 min_board 阈值。

怎么估
------
把各板块的命中率看成一个随机效应模型：p_b ~ (p0, τ²)，观测 p̂_b 的抽样方差
是 p0(1−p0)/n_b。τ² 用 DerSimonian-Laird 矩估计（meta 分析里的标准做法）：

    w_b = n_b / (p0(1−p0))            # 抽样方差的倒数
    Q   = Σ w_b (p̂_b − p̄_w)²
    τ²  = max(0, (Q − (k−1)) / (Σw − Σw² / Σw))

然后每个板块的收缩权重 B_b = τ² / (τ² + p0(1−p0)/n_b)，

    p̂ᴮ_b = B_b · p̂_b + (1 − B_b) · p0
    因子   = p̂ᴮ_b / p0，再夹到 clamp 区间里

三个性质正好是我们要的：

- **样本大的板块基本保留自己的估计**（B→1），样本小的被拉回全市场（B→0）。
  n=19 的北交所拿不到 2.37 这种因子。
- **板块之间看不出真实差异时 τ²=0，所有因子恰好等于 1**，即「证据不支持
  区分板块」，自动退化成不校正。这是它比「挑个 min_board 阈值」优雅的地方：
  阈值是人拍的，τ² 是数据算的。
- **冷启动安全**：前几个月样本少，Q 小 -> τ²=0 -> 全是 1。

时间上和滚动臂一样只用 `before_month` 之前**已结算**的名额，拟合窗口和
评估窗口不重叠，所以成绩里没有样本内增益。

clamp 是最后一道保险，不是主力：收缩之后因子本来就很难跑到边界上，
真跑到了说明某个板块的证据极强，那时候夹一下也不亏。
"""
from __future__ import annotations

CLAMP = (0.60, 1.40)
MIN_TOTAL = 100      # 全部板块加起来不够这么多名额就不校正
MIN_BOARD = 5        # 一个板块少于这么多名额就不单独估（并进整体）


def shrink_factors(counts: dict[str, tuple[int, int]],
                   clamp: tuple[float, float] = CLAMP,
                   min_total: int = MIN_TOTAL,
                   min_board: int = MIN_BOARD) -> dict[str, float]:
    """counts = {板块: (命中数, 名额数)} -> {板块: 因子}。

    样本不够就返回 {}（= 不校正）。返回的 dict 只含有资格单独估的板块，
    没进来的板块在调用方 `.fillna(1.0)`，等于按全市场对待。
    """
    rows = [(b, int(h), int(n)) for b, (h, n) in counts.items()
            if int(n) >= min_board and int(n) > 0]
    total_n = sum(n for _, _, n in rows)
    total_h = sum(h for _, h, _ in rows)
    if total_n < min_total or len(rows) < 2 or not total_h:
        return {}
    p0 = total_h / total_n
    var0 = p0 * (1.0 - p0)
    if var0 <= 0:
        return {}

    w = [n / var0 for _, _, n in rows]                 # 抽样方差的倒数
    sw = sum(w)
    p = [h / n for _, h, n in rows]
    pbar = sum(wi * pi for wi, pi in zip(w, p)) / sw   # 加权均值
    q = sum(wi * (pi - pbar) ** 2 for wi, pi in zip(w, p))
    denom = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - (len(rows) - 1)) / denom) if denom > 0 else 0.0

    out = {}
    for (b, h, n), pi in zip(rows, p):
        v = var0 / n                                   # 这个板块的抽样方差
        shrink = tau2 / (tau2 + v) if (tau2 + v) > 0 else 0.0
        post = shrink * pi + (1.0 - shrink) * p0
        f = post / p0
        out[str(b)] = float(min(clamp[1], max(clamp[0], f)))
    return out


def diagnostics(counts: dict[str, tuple[int, int]],
                min_board: int = MIN_BOARD) -> dict:
    """把中间量也吐出来，面板和会诊证据包要印「每个板块被信了多少」。"""
    rows = [(b, int(h), int(n)) for b, (h, n) in counts.items()
            if int(n) >= min_board and int(n) > 0]
    total_n = sum(n for _, _, n in rows)
    total_h = sum(h for _, h, _ in rows)
    if not total_n or not total_h or len(rows) < 2:
        return {"tau2": 0.0, "p0": (total_h / total_n) if total_n else 0.0,
                "boards": {}, "note": "样本不够，不校正"}
    p0 = total_h / total_n
    var0 = p0 * (1.0 - p0)
    w = [n / var0 for _, _, n in rows]
    sw = sum(w)
    p = [h / n for _, h, n in rows]
    pbar = sum(wi * pi for wi, pi in zip(w, p)) / sw
    q = sum(wi * (pi - pbar) ** 2 for wi, pi in zip(w, p))
    denom = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - (len(rows) - 1)) / denom) if denom > 0 else 0.0
    f = shrink_factors(counts, min_board=min_board)
    out = {"tau2": tau2, "Q": q, "df": len(rows) - 1, "p0": p0, "boards": {}}
    for (b, h, n), pi in zip(rows, p):
        v = var0 / n
        out["boards"][str(b)] = {
            "n": n, "hits": h, "raw_rate": pi,
            "raw_factor": pi / p0,
            "shrink_weight": (tau2 / (tau2 + v)) if (tau2 + v) > 0 else 0.0,
            "factor": f.get(str(b), 1.0)}
    return out


def factors_before(picks, before_month: str, hit_col: str = "y_up",
                   board_col: str = "board", date_col: str = "date",
                   **kw) -> dict[str, float]:
    """`exp_window.board_factors` 的收缩版：只用 before_month 之前的月份。

    picks 是逐日名额表（一行一个名额，hit_col 是 0/1 命中）。
    """
    past = picks[picks[date_col].astype(str).str[:7] < before_month]
    if past.empty:
        return {}
    counts = {}
    for b, g in past.groupby(board_col):
        counts[str(b)] = (int(g[hit_col].sum()), int(len(g)))
    return shrink_factors(counts, **kw)
