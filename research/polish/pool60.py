"""
候选池放宽到上市 >= 60 根（研究池 120、生产约 240）：ret20 规则在放宽池上的成绩。只印筛选集；开发集算了落盘不印。

    PYTHONIOENCODING=utf-8 python research/polish/pool60.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P


def main():
    import validate as V
    import common_rc as C
    feat = pd.read_parquet(P.EXP / "rulecore" / "feat.parquet",
                           columns=["code", "date", "board", "close", "nbar", "susp20", "lret20", "y_open", "y_up", "elig0"])
    feat["code"] = feat["code"].astype(str).str.zfill(6)
    feat = feat[(feat["date"] >= "2022-01-01") & (feat["nbar"] >= 60) & (feat["nbar"] < 120)].reset_index(drop=True)
    st = V.st_codes()
    feat["elig60"] = (~feat["code"].isin(st) & (feat["close"] >= 2.0) & ~feat["susp20"] & np.isfinite(feat["lret20"]))
    feat = feat[feat["elig60"]].reset_index(drop=True)
    ev = C.event_flags(feat)
    feat = feat[~ev["any"].to_numpy()].reset_index(drop=True)
    P.say("60~119 根的加入行 %d（每天 %.0f 只）" % (len(feat), feat.groupby("date").size().mean()))
    pool = P.load_pool(["date", "code", "board", "y_open", "y_up", "pool", "lret20", "nbar"])
    pool = pool[pool["pool"] & (pool["date"] >= "2022-01-01")]
    allr = pd.concat([pool[["date", "code", "board", "y_open", "y_up", "lret20", "nbar"]], feat[["date", "code", "board", "y_open", "y_up", "lret20", "nbar"]]],
                     ignore_index=True)
    t = P.rank_by(allr, allr["lret20"].to_numpy(), keep=30)
    h, h30 = P.hit_series(t, 10), P.hit_series(t, 30)
    H = pd.read_parquet(P.PL / "rules_hits10.parquet")
    H30 = pd.read_parquet(P.PL / "rules_hits30.parquet")
    H["pool_nbar60"], H30["pool_nbar60"] = h.reindex(H.index), h30.reindex(H30.index)
    H.to_parquet(P.PL / "rules_hits10.parquet")
    H30.to_parquet(P.PL / "rules_hits30.parquet")
    base, base30 = H["base_ret20"], H30["base_ret20"]
    d, d30 = (h - base).dropna(), (h30 - base30).dropna()
    scr = d.index < "2024-06-01"
    top = t[t["rank"] <= 10].merge(allr[["date", "code", "nbar"]], on=["date", "code"], how="left")
    young = (top["nbar"] < 120)
    P.say("pool_nbar60 筛选集：命中 %.2f%% 对 base %.2f%%，差 %+.2fpp NW t %.2f；前 30 差 %+.2f；前 10 里 60~119 根的占 %.1f%%，其命中 %.1f%%（筛选集）" % (
        100 * h[h.index < "2024-06-01"].mean(), 100 * base[scr].mean(), 100 * d[scr].mean(), P.nw_t(d[scr]), 100 * d30[scr].mean(),
        100 * young[top["date"] < "2024-06-01"].mean(), 100 * top[young & (top["date"] < "2024-06-01")]["y_open"].mean()))


if __name__ == "__main__":
    main()
