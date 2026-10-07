"""
买得到口径的标签（实验 16 第 1 条，计划第 2.1 节）。

    y_open    次日开盘买入，之后 UP_WINDOW 根 K 线最高价相对次日开盘涨超 UP_THRESHOLD，
              且次日不是一字板（全天最低价没低于涨停价）
    r20_open  次日开盘起算的 20 根最高涨幅（连续值）
    yizi1     次日一字板

不进模型指纹：label.py / build.py 不动，这里从 daily.parquet 现算，按 (code, date) 并到
训练表或清单上。exp_acc.build_labels 是它的第一版，对账过 353 万行零不一致；
以后只留这一份实现（教训 34）。缓存 data/breakout/raw/labels_open.parquet，
日线表比缓存新就重算。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label import UP_THRESHOLD, UP_WINDOW  # noqa: E402

DATA = ROOT / "data" / "breakout"
CACHE = DATA / "raw" / "labels_open.parquet"


def _per_code(g: pd.DataFrame, window: int) -> pd.DataFrame:
    n = len(g)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    opn = g["open"].to_numpy(float)
    fut = np.full(n, np.nan)
    if n > window:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        fut[:n - window] = swv(high[1:], window).max(axis=1)
    return pd.DataFrame({"code": g["code"].to_numpy(), "date": g["date"].to_numpy(),
                         "close": g["close"].to_numpy(float), "fut_max": fut,
                         "open1": np.append(opn[1:], np.nan),
                         "low1": np.append(low[1:], np.nan)})


def compute(daily: pd.DataFrame, window: int = UP_WINDOW,
            threshold: float = UP_THRESHOLD) -> pd.DataFrame:
    import datasource as ds
    px = daily[["code", "date", "open", "high", "low", "close"]].copy()
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    d = pd.concat([_per_code(g, window) for _, g in px.groupby("code", sort=False)],
                  ignore_index=True)
    pct = d["code"].map(lambda c: ds.limit_pct(c, ""))
    lim1 = ds.limit_price_arr(d["close"], pct, d["code"])
    d["yizi1"] = (d["low1"] >= lim1 - 1e-9) & np.isfinite(d["low1"])
    ok = np.isfinite(d["fut_max"]) & np.isfinite(d["open1"]) & (d["open1"] > 0)
    d["r20_open"] = np.where(ok, d["fut_max"] / d["open1"] - 1.0, np.nan)
    d["y_open"] = np.where(ok, ((d["r20_open"] > threshold) & ~d["yizi1"]).astype(float), np.nan)
    return d[["code", "date", "y_open", "r20_open", "yizi1"]]


def open_labels(refresh: bool = False) -> pd.DataFrame:
    dp = DATA / "daily.parquet"
    if CACHE.exists() and not refresh and CACHE.stat().st_mtime >= dp.stat().st_mtime:
        return pd.read_parquet(CACHE)
    daily = pd.read_parquet(dp, columns=["code", "date", "open", "high", "low", "close"])
    out = compute(daily)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


if __name__ == "__main__":
    o = open_labels(refresh=True)
    v = o[np.isfinite(o["y_open"])]
    print("%d 行，y_open 正样本率 %.3f%%，次日一字 %.2f%% -> %s" % (
        len(o), 100 * v["y_open"].mean(), 100 * v["yizi1"].mean(), CACHE))
