"""
标注：起涨事件（清单 A 的标签）和见顶事件（清单 B 的标签）。

这个模块和 features.py 是**物理隔离**的
--------------------------------------
标注只看 t 之后，特征只看 t 及之前。两者各自接收自己允许的那段数据，
不共享 DataFrame、不互相 import。

这不是洁癖。量化里最贵的错误就是前视偏差，而它的典型形态是
「特征函数里顺手用了一个后面才算出来的列」—— 代码看着没问题，
回测结果惊艳，实盘一塌糊涂。隔离成两个模块，这种错就写不出来。

定义（2026-09-12 与用户确认）
-----------------------------
起涨：t 日收盘买入，未来 20 个交易日内**最高价**涨幅 > 50%
      窗口 20 个交易日 ≈ 一个自然月；用最高价不用收盘价，
      因为用户说的是「上涨超过 50%」，触及即算

见顶：一段波段的**最高点当天及其前一天**（用户确认的读法 B）
      而不是「回撤 20% 那天的前两天」—— 那时股价已经跌下来了，
      信号没有交易价值
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UP_WINDOW = 20          # 交易日，约一个自然月
UP_THRESHOLD = 0.50     # 涨幅阈值
DRAWDOWN_END = 0.20     # 从最高点回撤多少算波段结束


def label_up(close: np.ndarray, high: np.ndarray,
             window: int = UP_WINDOW,
             threshold: float = UP_THRESHOLD) -> np.ndarray:
    """y_up[t] = 未来 window 个交易日内最高价相对 close[t] 涨幅是否超阈值。

    只接收 close 和 high，**不接收任何特征**。末尾 window 天因为看不到
    完整的未来窗口，标为 NaN 而不是 0：把"不知道"当成"没发生"会让模型
    在最近的数据上系统性地学到负样本。
    """
    n = len(close)
    y = np.full(n, np.nan)
    if n <= window:
        return y
    # 未来 window 天的最高价。注意是 t+1 开始，不含当天。
    fut = np.full(n, np.nan)
    for i in range(n - window):
        fut[i] = high[i + 1:i + 1 + window].max()
    ok = np.isfinite(fut) & np.isfinite(close) & (close > 0)
    y[ok] = ((fut[ok] / close[ok] - 1.0) > threshold).astype(float)
    return y


def first_days(y_up: np.ndarray) -> np.ndarray:
    """把连续的正样本区间收缩成它的第一天（真正的起涨点 t0）。

    y_up 对一段真实行情会连续亮很多天（今天买涨 50%，明天买也涨 50%），
    直接拿去训练等于给同一个事件重复计数几十次，模型会被少数几段大行情
    主导。真正的"起涨前 5 日"只有一个，就是这段的第一天。
    """
    y = np.nan_to_num(y_up, nan=0.0) > 0
    out = np.zeros(len(y))
    prev = False
    for i, v in enumerate(y):
        out[i] = 1.0 if (v and not prev) else 0.0
        prev = v
    out[~np.isfinite(y_up)] = np.nan
    return out


def label_top(close: np.ndarray, high: np.ndarray,
              t0_mask: np.ndarray,
              window: int = UP_WINDOW,
              drawdown: float = DRAWDOWN_END) -> np.ndarray:
    """见顶标签：每段起涨波段的最高点当天及前一天标 1。

    波段的界定：
        从 t0 起往后找最高价那一天 peak（最多找 3 倍窗口，防止一路走成
        长牛之后把很远的高点也算进同一段）
        peak 之后首次 close < peak_high * (1-drawdown) 才算这段结束
        找不到结束点（还在高位）就不标 —— 这段还没走完，标了就是猜
    """
    n = len(close)
    y = np.zeros(n)
    horizon = window * 3
    t0s = np.where(np.nan_to_num(t0_mask, nan=0.0) > 0)[0]
    for t0 in t0s:
        seg_end = min(t0 + horizon, n)
        if seg_end - t0 < 3:
            continue
        seg = high[t0:seg_end]
        peak = t0 + int(np.argmax(seg))
        peak_px = high[peak]
        # 必须真的跌下来过，这段才算走完
        after = close[peak + 1:seg_end]
        if len(after) == 0 or not (after < peak_px * (1 - drawdown)).any():
            continue
        y[peak] = 1.0
        if peak - 1 >= 0:
            y[peak - 1] = 1.0
    # 末尾标 NaN 的范围只能是 window，不能是 horizon。
    # 原来写的是 y[n-horizon:] = nan，horizon = window*3 = 60，序列短于 60 天时
    # **整个数组**都成了 NaN，见顶标签一个都出不来 —— selftest_breakout
    # 的「见顶标签有命中」就是为了钉住这个。
    # 判断一段波段走没走完靠的是上面的 after 检查（必须真的跌下来过），
    # 不是靠末尾一刀切。
    y[max(n - window, 0):] = np.nan
    return y


def label_frame(df: pd.DataFrame) -> pd.DataFrame:
    """给一只票的日线表加三列标签。df 需含 close / high，按日期升序。

    返回的是**只含标签**的新表，不带任何输入列 —— 再次强调隔离：
    调用方必须显式地把标签和特征拼起来，不会"顺手"拿到。
    """
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    y_up = label_up(c, h)
    t0 = first_days(y_up)
    y_top = label_top(c, h, t0)
    return pd.DataFrame({"y_up": y_up, "y_t0": t0, "y_top": y_top},
                        index=df.index)
