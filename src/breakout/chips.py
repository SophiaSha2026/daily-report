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

N_BINS = 360        # 对数网格档数：起始覆盖 1/10~10 倍，每格约 1.3%（2026-09-15 起）
DECAY = 1.0
# 筹码算法版本。进 daily.feature_fingerprint：网格口径变了而列名没变，
# 指纹不带版本号的话旧模型会拿新语义的列继续打分最多 30 天。
CHIPS_VERSION = 2   # 2026-09-16：网格越界从「丢弃那天」改成按同格宽外接


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

    # 网格只能由**过去**决定。第一版用整段序列的 min/max 定网格：格宽 w 取决于
    # 未来的最高价，一只票未来涨 50% 创新高（正是标签为 1 的情形），它之前每一天
    # 的筹码特征都被这个 w 系统性地扰动过，是一种隐蔽的前视偏差（2026-09-15
    # 审计实测：随机 40 只票，只用 [..t] 和用全序列算出的 chip_conc90 差最大
    # 0.022，差值与「未来最高/过去最高」相关 0.79）。
    # 现在按**对数价格**建网格，锚在第一根有效收盘价上，起始覆盖 1/10 ~ 10 倍；
    # 越界的那天**按同一个格宽往外接**，不丢弃。第一版把越界日整天跳过
    # （连衰减都不做），2026-09-16 实测：5516 只里 68 只涨过 10 倍、5 只跌破
    # 1/10，训练表 2739 行筹码全 NaN，而这些行的 y_up 率是全表的 2.6 倍
    # （8.9% vs 3.4%）—— 正是模型要找的那一撮票。更隐蔽的是没 NaN 但值错的
    # 1991 行：涨过 10 倍再跌回来的票，10 倍以上的成交从没记进分布，
    # 「上方套牢峰」不存在、获利盘被算成 100%（603629 生产 1.000 / 真值 0.303）。
    # 外接只由 [..t] 决定，补零的格不改变 avg/win/conc/peak，因果性不变。
    # 对数网格下格宽是恒定的比例，三角核也在对数空间撒。
    ok = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) \
        & (low > 0) & (high > 0) & (close > 0)
    if not ok.any():
        return pd.DataFrame(np.nan, index=range(n),
                            columns=["chip_avg", "chip_win", "chip_conc90",
                                     "chip_dev", "chip_peak"])
    c0 = float(close[np.argmax(ok)])
    lo, hi = np.log(c0 / 10.0), np.log(c0 * 10.0)
    w = (hi - lo) / (n_bins - 1)
    grid = lo + w * np.arange(n_bins)      # 起始网格；越界时按同样的 w 往外接
    price = np.exp(grid)

    chip = np.zeros(n_bins)
    out = np.full((n, 5), np.nan)

    for i in range(n):
        if not ok[i]:
            continue
        h, l, c, t = np.log(high[i]), np.log(low[i]), np.log(close[i]), turnover[i]
        # 外接网格。补的是零质量的格，历史分布一格不动
        if l - w < grid[0]:
            pad = int(np.ceil((grid[0] - (l - w)) / w))
            grid = np.concatenate([grid[0] - w * np.arange(pad, 0, -1), grid])
            chip = np.concatenate([np.zeros(pad), chip])
            price = np.exp(grid)
        if h + w > grid[-1]:
            pad = int(np.ceil((h + w - grid[-1]) / w))
            grid = np.concatenate([grid, grid[-1] + w * np.arange(1, pad + 1)])
            chip = np.concatenate([chip, np.zeros(pad)])
            price = np.exp(grid)
        # 当日成交按三角分布撒开。半宽至少一个格，否则一字板那天
        # （h == l）会得到全零分布。
        peak = (h + l + c) / 3.0
        half = max((h - l) / 2.0, w)
        tri = np.maximum(0.0, 1.0 - np.abs(grid - peak) / half)
        tri[(grid < l - w) | (grid > h + w)] = 0.0
        s = tri.sum()
        # 网格已经外接到 [l-w, h+w]，峰必在网格内，s 不可能再是 0。
        # 以前这里是 continue：越界那天既不衰减也不写 out[i]，分布冻结、
        # 特征 NaN，而且一声不吭。静默跳过必须变成炸。
        assert s > 0, "网格外接后仍无质量（l=%.4f h=%.4f）" % (l, h)
        tri /= s

        k = float(np.clip(t * decay, 0.0, 1.0)) if np.isfinite(t) else 0.0
        chip = chip * (1.0 - k) + tri * k
        tot = chip.sum()
        if tot <= 0:
            continue
        p = chip / tot
        cum = np.cumsum(p)

        cp = float(close[i])
        avg = float(price @ p)
        win = float(p[grid <= c].sum())
        i5 = int(np.searchsorted(cum, 0.05))
        i95 = int(min(np.searchsorted(cum, 0.95), len(grid) - 1))
        conc = (price[i95] - price[i5]) / max(avg, 1e-9)
        dev = cp / max(avg, 1e-9) - 1.0
        pk = price[int(np.argmax(p))]
        out[i] = (avg, win, conc, dev, pk / max(cp, 1e-9) - 1.0)

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
