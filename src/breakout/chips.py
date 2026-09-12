"""
筹码分布自算。

为什么能自己算
--------------
筹码分布**不是交易所公布的数据**，是各家行情软件用日线自己算出来的，
算法公开。2026-09-12 的第一版设计里写"拿不到"是错的：东财的
push2his/kline 路径被掐，但那只挡住了**抄现成结果**这条路，
算法本身只需要日线 OHLCV 加换手率，两样都有。

算法（各家通用的三角分布法）
----------------------------
    每个交易日：
      1. 当日成交量按 [low, high] 的三角分布撒到价格网格，峰在 (h+l+c)/3
      2. 历史筹码按当日换手率衰减：chip *= (1 - turnover * decay)
      3. 新筹码 = 衰减后的历史 + 当日分配

decay 是唯一的自由参数，各家取 0.8~1.0，差别在老筹码衰减的快慢。
本项目取 1.0（config.yaml 可调，将来可由学习线提案）。

正确性怎么保证
--------------
没法和东财对比（接口被掐），所以靠**内部一致性检验**，六条都是数学性质，
不依赖外部数据。selftest_breakout.py 把它们钉住：

    平均成本 ≈ 60 日 VWAP        相关系数 > 0.9
    筹码质量守恒                  恒等于 1
    创新高时获利盘 ≈ 100%
    创新低时获利盘 ≈ 0%
    获利盘与价格正相关            秩相关 > 0.5
    集中度落在合理区间

这六条同时成立的实现基本不可能是错的。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 160        # 价格网格档数。太少精度不够，太多没有额外信息且变慢
DECAY = 1.0


def chip_features(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  turnover: np.ndarray, n_bins: int = N_BINS,
                  decay: float = DECAY) -> pd.DataFrame:
    """逐日算筹码分布，返回五个特征。

    turnover 是**小数**形式的换手率（0.03 表示 3%），不是百分数。
    传错的话筹码几乎不衰减，集中度会假性偏高 —— 这是最容易犯的错，
    所以下面有一条断言。

    返回列：
      chip_avg      筹码平均成本
      chip_win      获利盘比例（成本低于现价的筹码占比）
      chip_conc90   90% 成本区间宽度 / 平均成本，越小越集中
      chip_dev      close / chip_avg - 1，现价相对平均成本的位置
      chip_peak     主力筹码峰价 / close - 1，上方套牢峰的距离
    """
    n = len(close)
    assert len(high) == len(low) == len(turnover) == n, "长度不一致"
    assert np.nanmax(turnover) <= 1.5, (
        f"换手率最大 {np.nanmax(turnover):.2f}，看着像百分数。"
        "这里要小数形式（0.03 = 3%）")

    lo, hi = float(np.nanmin(low)) * 0.98, float(np.nanmax(high)) * 1.02
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.DataFrame(np.nan, index=range(n),
                            columns=["chip_avg", "chip_win", "chip_conc90",
                                     "chip_dev", "chip_peak"])
    grid = np.linspace(lo, hi, n_bins)
    w = grid[1] - grid[0]

    chip = np.zeros(n_bins)
    out = np.full((n, 5), np.nan)

    for i in range(n):
        h, l, c, t = high[i], low[i], close[i], turnover[i]
        if not np.isfinite(h + l + c):
            continue
        # 当日成交按三角分布撒开。半宽至少一个格，否则一字板那天
        # （h == l）会得到全零分布。
        peak = (h + l + c) / 3.0
        half = max((h - l) / 2.0, w)
        tri = np.maximum(0.0, 1.0 - np.abs(grid - peak) / half)
        tri[(grid < l - w) | (grid > h + w)] = 0.0
        s = tri.sum()
        if s <= 0:
            continue
        tri /= s

        k = float(np.clip(t * decay, 0.0, 1.0)) if np.isfinite(t) else 0.0
        chip = chip * (1.0 - k) + tri * k
        tot = chip.sum()
        if tot <= 0:
            continue
        p = chip / tot
        cum = np.cumsum(p)

        avg = float(grid @ p)
        win = float(p[grid <= c].sum())
        i5 = int(np.searchsorted(cum, 0.05))
        i95 = int(min(np.searchsorted(cum, 0.95), n_bins - 1))
        conc = (grid[i95] - grid[i5]) / max(avg, 1e-9)
        dev = c / max(avg, 1e-9) - 1.0
        pk = grid[int(np.argmax(p))]
        out[i] = (avg, win, conc, dev, pk / max(c, 1e-9) - 1.0)

    return pd.DataFrame(out, columns=["chip_avg", "chip_win", "chip_conc90",
                                      "chip_dev", "chip_peak"])


def add_chips(df: pd.DataFrame, decay: float = DECAY) -> pd.DataFrame:
    """给一只票的日线表加筹码特征。要求已含 turnover 列（小数）。

    这里不做分组循环：调用方按 code 分组后逐只调用，因为筹码是
    **路径依赖**的（今天的分布取决于昨天的），没法跨股票向量化。
    """
    f = chip_features(df["high"].to_numpy(float), df["low"].to_numpy(float),
                      df["close"].to_numpy(float),
                      df["turnover"].to_numpy(float), decay=decay)
    f.index = df.index
    return pd.concat([df, f], axis=1)
