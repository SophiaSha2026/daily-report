"""rethink 系列脚本共用的两个统计函数（Newey-West，滞后 20）。"""
from __future__ import annotations

import numpy as np


def nw_se(d, lag: int = 20) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    x = d - d.mean()
    s = x @ x / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(max(s, 0) / n))


def nw_t(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    se = nw_se(d)
    return d.mean(), se, (d.mean() / se if se > 0 else np.nan)
