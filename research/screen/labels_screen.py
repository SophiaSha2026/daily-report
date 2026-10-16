"""
第 3 步：标签。同一份 pit 价格上
    y_up / y_t0 / y_top   label.label_frame（只递 close/high，和 build.assemble 同一种调法）
    y_open r20_open yizi1 buyable.compute（pit 的 OHLC）
    yizi1_u               buyable.compute 在不复权价上算的一字板（涨停价是按真实价取整的，pit 价不取整会在
                          边界上翻转；只用来核对，y_open 按任务要求用 pit 那份）
产物 data/breakout/raw/exp/screen/labels_screen.parquet（code date + 7 列）。

    python research/screen/labels_screen.py [--px px_screen_fake.parquet --out labels_screen_fake.parquet]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import buyable as BY  # noqa: E402
from label import label_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", default="px_screen.parquet")
    ap.add_argument("--out", default="labels_screen.parquet")
    a = ap.parse_args()
    t0 = time.time()
    px = pd.read_parquet(CS.SCR / a.px, columns=["code", "date", "open", "high", "low", "close", "close_u", "low_u", "q"])
    px = px[(px["date"] >= CS.DATA_START) & (px["date"] <= CS.KEEP_TO)].sort_values(["code", "date"]).reset_index(drop=True)
    parts = []
    for _, g in px.groupby("code", sort=False):
        lab = label_frame(g[["close", "high"]])
        lab.insert(0, "date", g["date"].to_numpy())
        lab.insert(0, "code", g["code"].to_numpy())
        parts.append(lab)
    lab = pd.concat(parts, ignore_index=True)
    print("y_up 算完 %d 行，%.0fs" % (len(lab), time.time() - t0), flush=True)
    bo = BY.compute(px[["code", "date", "open", "high", "low", "close"]])
    # 一字板按不复权价再判一次（涨停价 = round(昨收_u × (1+pct), 2)，和交易所一致）
    pu = px[["code", "date"]].copy()
    pu["open"] = px["open"] * px["q"]
    pu["high"] = px["high"] * px["q"]
    pu["low"] = px["low_u"]
    pu["close"] = px["close_u"]
    bu = BY.compute(pu)[["code", "date", "yizi1"]].rename(columns={"yizi1": "yizi1_u"})
    out = lab.merge(bo, on=["code", "date"], how="left").merge(bu, on=["code", "date"], how="left")
    assert len(out) == len(lab)
    v = out[np.isfinite(out["y_open"])]
    dis = v[v["yizi1"] != v["yizi1_u"]]
    flip = int(((dis["r20_open"] > 0.5)).sum())
    print("y_open %d 行有效，正样本率 %.3f%%（y_up %.3f%%）；次日一字 pit %.3f%% / 不复权 %.3f%%，两者不一致 %d 行，"
          "其中 r20>50%% 的（会翻 y_open）%d 行 = %.4f%%" % (
              len(v), 100 * v["y_open"].mean(), 100 * np.nanmean(out["y_up"]), 100 * v["yizi1"].mean(),
              100 * v["yizi1_u"].mean(), len(dis), flip, 100 * flip / max(len(v), 1)), flush=True)
    out["y_open_u"] = np.where(np.isfinite(out["r20_open"]), ((out["r20_open"] > 0.5) & ~out["yizi1_u"].fillna(False).astype(bool)).astype(float), np.nan)
    out.to_parquet(CS.SCR / a.out, index=False)
    print("-> %s %d 行，%.0fs" % (CS.SCR / a.out, len(out), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
