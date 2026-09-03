"""
装载多天数据 + 日内中性化。

中性化是**收敛的第一道保险**，比损失函数本身更重要：
大盘跌 2% 的那天几乎所有票的 close/open 都是负的，不减掉的话，
损失完全被「今天大盘怎么走」主导，而我们的特征根本不预测大盘。

三步（见 docs/learning.md 3.1）：

    y  = r − median_d(r)         全池中位数。用均值会被涨跌停拽偏
    y  = clip(y, q1_d, q99_d)    缩尾。一只跌停票 y ≈ −10σ，能翻转整天梯度
    ỹ = y / MAD_d(y)            除掉当日波动。安静日和暴动日等权
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"


def snapshot_days() -> list[str]:
    """所有已存的竞价快照日期。"""
    return sorted(p.stem.replace("auction_", "")
                  for p in DATA.glob("2*/auction_*.parquet"))


def load_snapshot(date: str) -> pd.DataFrame:
    p = DATA / date[:7] / f"auction_{date}.parquet"
    d = pd.read_parquet(p)
    d["date"] = date
    return d


def mad(x: np.ndarray) -> float:
    """1.4826 × median(|x − median x|)，正态下与 std 同尺度。"""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def neutralize(df: pd.DataFrame, nz: dict, *,
               salvage_guard: bool = True) -> pd.DataFrame:
    """按天做中性化，写入列 y（中心化+缩尾）和 ytil（再除 MAD）。

    整天被丢弃的条件：可用样本少于 min_pool，或者 MAD 为 0
    （全池同涨同跌，尺度没有意义）。

    salvage_guard 只对**在线快照**打开。回填表的 t1/t2/t3 是竞价段的
    open/vwap/close 代理值，单价竞价（指示价全程没变）的票天然三者相等，
    正常日就有 39% 左右；2026-09-04 复查发现这个守卫把 2025-04-08、
    10-30、12-12 三个 51~55% 的日子当成抢救日丢了。守卫认的是数据来源，
    不是数据模式，所以由调用方按来源决定开关。
    """
    q = float(nz.get("winsor_q", 0.01))
    use_med = nz.get("center", "median") == "median"
    use_mad = nz.get("scale", "mad") == "mad"
    min_pool = int(nz.get("min_pool", 200))

    out = []
    for date, g in df.groupby("date", sort=True):
        # 抢救模式的快照里四个 T 全指向同一次采样：T1=T2=T3、斜率 0。
        # 正常日子这种票只占 ~6%（流动性差的），抢救日是 100%。
        # 拿抢救日训练等于教系统「斜率永远是 0」，整天丢弃。
        if salvage_guard and "t1_chg" in g.columns and len(g) > 0:
            frac = float(((g["t1_chg"] == g["t3_chg"])
                          & (g["t2_chg"] == g["t3_chg"])).mean())
            if frac > 0.5:
                log.info("%s 疑似抢救模式快照（%.0f%% 行 T1=T2=T3），整天丢弃",
                         date, frac * 100)
                continue
        ok = g[~g["dirty"].astype(bool)].copy()
        if len(ok) < min_pool:
            log.info("%s 可用样本 %d < %d，整天丢弃", date, len(ok), min_pool)
            continue
        r = ok["r"].to_numpy(float)
        center = np.median(r) if use_med else np.mean(r)
        y = r - center
        lo, hi = np.quantile(y, [q, 1 - q])
        y = np.clip(y, lo, hi)
        scale = mad(y) if use_mad else float(np.std(y))
        if not np.isfinite(scale) or scale <= 0:
            log.info("%s 离散度为 0，整天丢弃", date)
            continue
        ok["y"] = y
        ok["ytil"] = y / scale
        ok["day_center"] = center
        ok["day_scale"] = scale
        out.append(ok)
    if not out:
        return df.iloc[0:0].assign(y=[], ytil=[])
    return pd.concat(out, ignore_index=True)


def build(dates: list[str] | None = None, nz: dict | None = None
          ) -> pd.DataFrame:
    """快照 + 标签 -> 中性化后的训练表。"""
    from learn import labels as L
    nz = nz or {}
    lab = L.load_all()
    if lab.empty:
        return pd.DataFrame()
    have = set(lab["date"].unique())
    dates = [d for d in (dates or snapshot_days()) if d in have]
    if not dates:
        return pd.DataFrame()

    frames = []
    for d in dates:
        snap = load_snapshot(d)
        lb = lab[lab["date"] == d][["code", "r", "dirty"]]
        frames.append(snap.merge(lb, on="code", how="inner"))
    df = pd.concat(frames, ignore_index=True)
    df["dirty"] = df["dirty"].fillna(True).astype(bool)
    return neutralize(df, nz)
