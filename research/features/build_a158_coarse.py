"""
审计追加：取整颗粒度泄漏的决定性检验。

前复权把 T 日之后的除权按比例缩进之前的价格，再取整到分：之后除权越多，前复权价越低，最小价位（0.01 / 价）
越粗，颗粒度里带了未来信息。这里把 OHLC 统一取整到对数价 0.5% 的网格上（比绝大多数票的一分钱颗粒都粗），
再用同一份 alpha158 算一遍、当日排名。颗粒度信息被统一抹掉之后，如果 base_a158 的增益还在，就不是取整泄漏。

    python research/features/build_a158_coarse.py
产物 data/breakout/raw/exp/features/a158_coarse.parquet（列名同 a158，前缀 a_）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_feats as BF  # noqa: E402

GRID = 0.005


def main():
    t0 = time.time()
    px = pd.read_parquet(C.ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume"])
    px["date"] = px["date"].astype(str).str[:10]
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px[px["date"] <= BF.KEEP_TO].sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    for k in ("open", "high", "low", "close"):
        v = px[k].to_numpy(float)
        with np.errstate(all="ignore"):
            px[k] = np.exp(np.round(np.log(v) / GRID) * GRID)
    keep = (px["date"] >= BF.KEEP_FROM).to_numpy()
    a = BF.alpha158(px, keep)
    cols = [c for c in a.columns if c.startswith("a_")]
    for i in range(0, len(cols), 20):
        cc = cols[i:i + 20]
        a[cc] = a.groupby("date")[cc].rank(pct=True).astype("float32")
    a.to_parquet(C.FE / "a158_coarse.parquet", index=False)
    print("a158_coarse %d 行 %d 列 %.0fs" % (len(a), len(cols), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
