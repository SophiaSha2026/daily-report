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

from typing import Any

import numpy as np
import pandas as pd

# 与 score.py 的 AuctionFeature 字段一一对应，顺序无关
NEEDED = [
    "limit_pct", "prev_close", "auc_price", "gap_pct", "auc_ratio",
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
    return np.where((gap <= lo) | (gap >= hi), 0.0, v)


def f_volume(ratio: np.ndarray, lo: float, hi: float, sat: float,
             decay: float) -> np.ndarray:
    """对数刻度，衰减速率由 decay 独立给。见 score.py::f_volume 的注释。"""
    # ratio<=0 会让 log 发散；这些行最后会被区间判据置零，先夹一下避免 warning
    safe = np.maximum(ratio, 1e-12)
    rise = np.log(safe / lo) / np.log(sat / lo)
    fall = np.maximum(0.0, 1.0 - decay * np.log(safe / sat))
    v = np.where(safe <= sat, rise, fall)
    return np.where((ratio < lo) | (ratio > hi), 0.0, v)


def f_trend(slope: np.ndarray, monotonic: np.ndarray,
            limit: np.ndarray) -> np.ndarray:
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
    return (
        d["blacklisted"]
        | d["one_word"]
        | (d["prev_close"] <= 0) | (d["auc_price"] <= 0)
        | (d["gap_pct"] < sc["gap_pct_min"]) | (d["gap_pct"] > sc["gap_pct_max"])
        | (d["auc_ratio"] < sc["auc_ratio_min"])
        | (d["auc_ratio"] > sc["auc_ratio_max"])
        | ((d["t1_chg"] >= lim * sc["fake_limit_t1_frac"])
           & (d["t3_chg"] < lim * sc["fake_limit_t3_frac"]))
        | (d["dive"] >= sc["last_min_dive_max"])
        | (bool(sc["require_positive_slope"]) & (d["slope"] <= 0))
    )


def assign_group_b(d: dict[str, np.ndarray], sc: dict) -> np.ndarray:
    """True = B 组（低位首板预备）。"""
    is_a = d["prev_limit_up"] | (d["board_height"] >= 1) | d["breakout"]
    return (~is_a) & (d["pos_pct_60d"] <= sc["pos_pct_60d_max_for_lowbase"])


def prepare(df: "pd.DataFrame") -> dict[str, np.ndarray]:
    """DataFrame -> 列数组字典。一次准备，多次求值时不用反复转换。"""
    out: dict[str, np.ndarray] = {}
    for k in NEEDED:
        col = df[k].to_numpy()
        out[k] = col.astype(bool) if col.dtype == bool or col.dtype == object \
            else col.astype(float)
    for k in ("monotonic", "ma_bull", "breakout", "prev_limit_up",
              "prev_broken_board", "blacklisted", "one_word"):
        out[k] = df[k].to_numpy().astype(bool)
    for k in ("board_height", "sector_members", "sector_prev_limitups"):
        out[k] = df[k].to_numpy().astype(float)
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
