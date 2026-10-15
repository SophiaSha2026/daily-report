"""
研究面板：开发集 2024-06..2025-12 的每个股票-日，带模型分数、买得到标签、一字标记。
只读生产产物，不写 data/ state/。产物 research/announce/cache/panel.parquet。

  模型分数   data/breakout/raw/exp/w1_scale_avg/all_base.parquet（实验 17 的现状臂，
             按月等权 = 生产口径，evalkit 逐月走向前，prereg.json 已核）
  模型前 10  同目录 top_base.parquet 的 rank（剔 ST，和 evalkit.metrics 同口径）
  标签       buyable.compute 的口径（labels_open.parquet 缓存），读进来立刻切到 <= 2025-12-31，
             确认集 2026-01..08 的行不进内存以外的任何计算
  yizi0      t 当天一字涨停（全天最低价不低于按昨收算的涨停价）
  hit_raw    r20_open > 50%，不管次日一字（量化「买不到」丢了多少）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
import datasource as ds  # noqa: E402

OUT = Path(__file__).resolve().parent / "cache"
EXP = ROOT / "data" / "breakout" / "raw" / "exp" / "w1_scale_avg"
DEV_LO, DEV_HI = "2024-06-01", "2025-12-31"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_parquet(EXP / "all_base.parquet")
    a = a[(a["date"] >= DEV_LO) & (a["date"] <= DEV_HI)].copy()
    a["code"] = a["code"].astype(str).str.zfill(6)
    a["date"] = a["date"].astype(str)
    a["pct_rank"] = a.groupby("date")["p"].rank(pct=True, ascending=True)
    a["rank_all"] = a.groupby("date")["p"].rank(ascending=False, method="first")
    t = pd.read_parquet(EXP / "top_base.parquet", columns=["date", "code", "rank", "score", "board"])
    t["code"] = t["code"].astype(str).str.zfill(6)
    a = a.merge(t, on=["date", "code"], how="left")

    lab = pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "labels_open.parquet")
    lab = lab[(lab["date"] >= DEV_LO) & (lab["date"] <= DEV_HI)]
    a = a.merge(lab, on=["code", "date"], how="left")

    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "low", "close", "high", "open"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = px["date"].astype(str)
    px = px[(px["date"] >= "2024-03-01") & (px["date"] <= DEV_HI)].sort_values(["code", "date"])
    px["prev_close"] = px.groupby("code")["close"].shift(1)
    pct = px["code"].map(lambda c: ds.limit_pct(c, ""))
    lim0 = ds.limit_price_arr(px["prev_close"], pct, px["code"])
    px["yizi0"] = (px["low"] >= lim0 - 1e-9) & np.isfinite(px["prev_close"])
    px["ret0"] = px["close"] / px["prev_close"] - 1
    a = a.merge(px[["code", "date", "yizi0", "ret0"]], on=["code", "date"], how="left")
    a["hit_raw"] = np.where(np.isfinite(a["r20_open"]), (a["r20_open"] > 0.5).astype(float), np.nan)
    a = a[np.isfinite(a["y_open"])]
    if "board" not in a or a["board"].isna().any():
        a["board"] = a["code"].map(board_of)
    a.to_parquet(OUT / "panel.parquet", index=False)
    print(len(a), "行", a["date"].nunique(), "天", "y_open 基准 %.3f%%" % (100 * a["y_open"].mean()),
          "次日一字 %.2f%%" % (100 * a["yizi1"].mean()))


def board_of(c: str) -> str:
    if c[:3] in ("688", "689"):
        return "科创"
    if c[:2] in ("30",):
        return "创业"
    if c[:2] in ("92", "83", "87", "88", "43"):
        return "北交"
    return "主板"


if __name__ == "__main__":
    main()
