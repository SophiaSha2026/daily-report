"""
`score.py` 的向量化孪生体。

存在的唯一理由是速度：优化器一次目标函数求值要给几十万行重新打分，
逐行调 `score_one` 是 3 秒，Nelder-Mead 跑 500 次迭代就是 25 分钟。
numpy 版本是 10 毫秒量级。

**它不是「另一套打分逻辑」，是同一套逻辑的另一种写法。**
`selftest_learn.py` 用 2000 个随机样本断言两者逐位一致（差 < 1e-9）。
那条断言一红，整个学习系统的结论作废——因为学到的参数会被
生产打分器用另一套语义执行。

改 `score.py` 就必须同步改这里，反之亦然。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 与 score.py 的 AuctionFeature 字段一一对应，顺序无关
NEEDED = [
    "limit_pct", "prev_close", "auc_price", "gap_pct", "auc_ratio", "auc_amount",
    "t1_chg", "t3_chg", "slope", "monotonic", "dive",
    "pos_pct_60d", "ma_bull", "breakout", "prev_limit_up",
    "prev_broken_board", "board_height", "sector_members",
    "sector_prev_limitups", "blacklisted", "one_word",
]


# ---------------------------------------------------------------------
#  分项打分。签名和 score.py 里的同名函数一致，只是吃数组。
# ---------------------------------------------------------------------
def f_gap(gap: np.ndarray, lo: float, hi: float, peak: float) -> np.ndarray:
    """钟形，两臂各自按自己的跨度归一。见 score.py::f_gap 的注释。"""
    half = np.where(gap < peak, peak - lo, hi - peak)
    v = 1.0 - np.abs(gap - peak) / half
    v = np.maximum(0.0, v)
    # 写成「在区间内才留 v」而不是「在区间外置 0」：NaN 和任何数比较都是 False，
    # 后一种写法让 NaN 保住 v（=NaN），而 score.py 的 max(0.0, nan) 是 0.0。
    # 同 hard_reject 里 gap_in 的写法。
    return np.where(~((gap > lo) & (gap < hi)), 0.0, v)


def f_volume(ratio: np.ndarray, lo: float, hi: float, sat: float,
             decay: float) -> np.ndarray:
    """对数刻度，衰减速率由 decay 独立给。见 score.py::f_volume 的注释。"""
    if lo <= 0:
        # 维度停用（抢救模式），和 score.py 一样给 0
        return np.zeros_like(np.asarray(ratio, dtype=float))
    # ratio<=0 会让 log 发散；这些行最后会被区间判据置零，先夹一下避免 warning。
    # NaN 也先换成 0：score.py 对 NaN 走到 max(0, nan) 得 0，这里要一样。
    r = np.where(np.isfinite(ratio), ratio, 0.0)
    safe = np.maximum(r, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        rise = np.log(safe / lo) / np.log(sat / lo)
        fall = np.maximum(0.0, 1.0 - decay * np.log(safe / sat))
    v = np.where(safe <= sat, rise, fall)
    return np.where(~((r >= lo) & (r <= hi)), 0.0, v)


def f_trend(slope: np.ndarray, monotonic: np.ndarray,
            limit: np.ndarray) -> np.ndarray:
    # 斜率缺失按 0，和 score.py::f_trend 同一口径（np.clip(nan) 是 nan，
    # 而标量那边 min(1.0, nan) 是 1.0 —— 两边都不对，统一成「没有轨迹证据」）
    slope = np.where(np.isfinite(slope), slope, 0.0)
    s = np.clip(slope / (limit * 0.3), -1.0, 1.0)
    return np.minimum(1.0, (s + 1.0) / 2.0 + np.where(monotonic, 0.15, 0.0))


def f_position(pos: np.ndarray, ma_bull: np.ndarray, breakout: np.ndarray,
               group_b: np.ndarray) -> np.ndarray:
    b = np.minimum(1.0, np.maximum(0.0, 1.0 - pos / 0.40)
                   + np.where(ma_bull, 0.2, 0.0))
    a = (np.where(breakout, 0.5, 0.0) + np.where(ma_bull, 0.3, 0.0)
         + np.where((pos >= 0.55) & (pos <= 0.95), 0.2, 0.0))
    return np.where(group_b, b, np.minimum(1.0, a))


def f_sector(members: np.ndarray, prev_limitups: np.ndarray,
             min_members: int, min_prev: int) -> np.ndarray:
    s = np.minimum(1.0, (members - 1.0) / (min_members + 1.0))
    s = np.where(prev_limitups >= min_prev, np.minimum(1.0, s + 0.3), s)
    return np.where(members <= 1, 0.0, s)


def f_continuity(board_height: np.ndarray,
                 prev_limit_up: np.ndarray) -> np.ndarray:
    return np.select(
        [board_height >= 3, board_height == 2, prev_limit_up],
        [0.6, 1.0, 0.85],
        default=0.3,
    )


# ---------------------------------------------------------------------
#  硬性排除 / 分组 / 总分
# ---------------------------------------------------------------------
def _liangbi(auc_ratio: np.ndarray, sc: dict) -> np.ndarray:
    return auc_ratio * sc.get("liangbi_per_auc_ratio", 240)


def hard_reject(d: dict[str, np.ndarray], sc: dict) -> np.ndarray:
    """返回布尔数组：True = 被剔除。

    这里只需要「过没过」，不需要原因（原因由 score.py 在生产路径上给），
    所以不必复刻 score.py 里的判定顺序。
    """
    lim = d["limit_pct"]
    # 区间判定写成「不在 [lo, hi] 内」而不是「< lo 或 > hi」：NaN 和任何数比较
    # 都是 False，后一种写法会让 NaN 悄悄通过，而 score.py 的
    # `not (lo <= x <= hi)` 对 NaN 是剔除。回填表里有 6950 行 auc_ratio 为 NaN，
    # 2026-09-15 前这些行在这里不剔除、分数变 NaN，404 天里 159 天的目标函数
    # 因此和参数无关。
    gap_in = (d["gap_pct"] >= sc["gap_pct_min"]) & (d["gap_pct"] <= sc["gap_pct_max"])
    ratio_in = ((d["auc_ratio"] >= sc["auc_ratio_min"])
                & (d["auc_ratio"] <= sc["auc_ratio_max"]))
    # 直接索引，和 score.py::hard_reject 一样缺键就响。两边同时 .get(…,0) 时
    # 等价性断言抓不到：生产和回测会一起把这条准入线静默关掉，而训练表里的
    # sector_members 是按有下限算的，一次拟合里两个口径就分叉（教训 30）。
    amt_ok = d["auc_amount"] >= sc["min_auc_amount_wan"] * 1e4
    return (
        d["blacklisted"]
        | d["one_word"]
        | (d["prev_close"] <= 0) | (d["auc_price"] <= 0)
        | ~gap_in
        | ~ratio_in
        | ~amt_ok
        | ((d["t1_chg"] >= lim * sc["fake_limit_t1_frac"])
           & (d["t3_chg"] < lim * sc["fake_limit_t3_frac"]))
        | (d["dive"] >= sc["last_min_dive_max"])
        | (bool(sc["require_positive_slope"]) & (d["slope"] <= 0))
    )


def assign_group_b(d: dict[str, np.ndarray], sc: dict) -> np.ndarray:
    """True = B 组（低位首板预备）。"""
    is_a = d["prev_limit_up"] | (d["board_height"] >= 1) | d["breakout"]
    return (~is_a) & (d["pos_pct_60d"] <= sc["pos_pct_60d_max_for_lowbase"])


_BOOL_COLS = ("monotonic", "ma_bull", "breakout", "prev_limit_up",
              "prev_broken_board", "blacklisted", "one_word")


def prepare(df: "pd.DataFrame") -> dict[str, np.ndarray]:
    """DataFrame -> 列数组字典。一次准备，多次求值时不用反复转换。

    按**列名**分流，不按 dtype。以前写的是
    `col.astype(bool) if col.dtype == bool or col.dtype == object else ...`，
    一个数值列只要以 object 到达（concat 时 dtype 漂了、或混进一个 None），
    就会被 astype(bool) 变成 True/False：gap_pct 变 bool 后
    `True >= 2.0` 恒为 False，整天 100% 被硬性排除，优化器每次目标函数求值
    都是零幸存者；pos_pct_60d 变 bool 则实测 12 只通过者里 9 只分数变了、
    最大差 40.72 分。全程不抛异常，退出码 0。
    数值列一律 astype(float)，object 里的 None 会正确变成 NaN 而不是 False。
    """
    out: dict[str, np.ndarray] = {}
    for k in NEEDED:
        col = df[k].to_numpy()
        out[k] = col.astype(bool) if k in _BOOL_COLS else col.astype(float)
    return out


def parts(d: dict[str, np.ndarray], c: dict) -> dict[str, np.ndarray]:
    sc = c["screen"]
    gb = assign_group_b(d, sc)
    return {
        "gap": f_gap(d["gap_pct"], sc["gap_pct_min"], sc["gap_pct_max"],
                     sc["gap_pct_peak"]),
        "volume": f_volume(d["auc_ratio"], sc["auc_ratio_min"],
                           sc["auc_ratio_max"], sc["auc_ratio_score_hi"],
                           sc.get("auc_ratio_decay", 0.40)),
        "trend": f_trend(d["slope"], d["monotonic"], d["limit_pct"]),
        "position": f_position(d["pos_pct_60d"], d["ma_bull"], d["breakout"], gb),
        "sector": f_sector(d["sector_members"], d["sector_prev_limitups"],
                           sc["sector_min_members"], sc["concept_prev_limitup_min"]),
        "continuity": f_continuity(d["board_height"], d["prev_limit_up"]),
    }


def score(d: dict[str, np.ndarray], c: dict) -> np.ndarray:
    """总分，和 score.py::score_one 的 'score' 字段一致（含扣分和 0 下限）。"""
    sc, w, pen = c["screen"], c["scoring"]["weights"], c["scoring"]["penalties"]
    p = parts(d, c)
    raw = 100.0 * sum(w[k] * v for k, v in p.items())

    penalty = np.zeros_like(raw)
    penalty += np.where(d["dive"] >= sc["last_min_dive_max"] * 0.6,
                        pen["last_min_dive"] * 0.5, 0.0)
    penalty += np.where(
        (d["pos_pct_60d"] > 0.9)
        & (_liangbi(d["auc_ratio"], sc) >= sc.get("high_pos_liangbi_min", 8.0)),
        pen["high_pos_extreme_volume"], 0.0)
    penalty += np.where(d["prev_broken_board"],
                        pen["yesterday_broken_board"], 0.0)
    return np.maximum(0.0, raw - penalty)


def score_df(df: "pd.DataFrame", c: dict) -> tuple[np.ndarray, np.ndarray]:
    """便捷入口：返回 (分数, 是否被硬性剔除)。"""
    d = prepare(df)
    return score(d, c), hard_reject(d, c["screen"])
