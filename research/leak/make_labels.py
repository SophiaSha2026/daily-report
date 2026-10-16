"""
买得到口径标签 y_open：用 buyable.compute 在 daily_snap 上算（不碰 data/breakout/raw/labels_open.parquet 缓存）。
两臂同一份。标签用快照的 daily 价（= fwd），取整对 >50% 的判定影响可忽略，且两臂相同。

    python research/leak/make_labels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import buyable as B  # noqa: E402


def main():
    d = pd.read_parquet(CL.LEAK / "daily_snap.parquet", columns=["code", "date", "open", "high", "low", "close"])
    out = B.compute(d)
    out = out[out["date"] <= CL.KEEP_TO]
    out.to_parquet(CL.LEAK / "labels_open_snap.parquet", index=False)
    v = out[np.isfinite(out["y_open"])]
    print("%d 行（<= %s），y_open 正样本率 %.3f%%，次日一字 %.2f%%" % (
        len(out), CL.KEEP_TO, 100 * v["y_open"].mean(), 100 * v["yizi1"].mean()))


if __name__ == "__main__":
    main()
