"""把 research/deepdata 的六张特征表并到 ret20 前 300 名的行上（F6b），写 research/gate/table_dd.parquet（gitignore）。

    PYTHONIOENCODING=utf-8 python research/gate/build_dd.py
"""
import glob

import numpy as np
import pandas as pd

import common_gt as G

DD = G.ROOT / "research" / "deepdata"


def main():
    t = G.load_table(["code", "date", "rank_ret20"])
    keep = t[t["rank_ret20"] <= 300][["code", "date"]].reset_index(drop=True)
    G.say("前 300 行 %d" % len(keep))
    cols_all = []
    for f in sorted(glob.glob(str(DD / "features_*.parquet"))):
        d = pd.read_parquet(f)
        d["code"] = d["code"].astype(str).str.zfill(6)
        num = [c for c in d.columns if c not in ("code", "date") and d[c].dtype != object and not str(d[c].dtype).startswith("string")]
        d = d[["code", "date"] + num]
        for c in num:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(np.float32)
        keep = keep.merge(d, on=["code", "date"], how="left")
        cols_all += num
        G.say("%s：%d 列，并后覆盖 %.3f" % (f.split("\\")[-1], len(num), float(np.isfinite(keep[num[0]]).mean())))
    keep.to_parquet(G.HERE / "table_dd.parquet", index=False)
    (G.HERE / "dd_cols.txt").write_text("\n".join(cols_all), encoding="utf-8")
    G.say("-> table_dd.parquet %d 行 %d 列" % (len(keep), keep.shape[1]))
    G.progress("deepdata 六张表并到前 300 名行：table_dd.parquet %d 行，%d 个特征（dd_cols.txt）" % (len(keep), len(cols_all)))


if __name__ == "__main__":
    main()
