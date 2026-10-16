"""
第 3 步：在某一版价格上重算 Alpha158（research/altpath/build_feats.alpha158）和 on/vc/hi 三族
（research/features/build_new.FAMS），当日全市场排名归一，和原来两个脚本同一套代码、同一段日期
（a158 2023-12-01 起、三族 2023-06-01 起，都只到 2025-12-31）。

产物 data/breakout/raw/exp/leak/alt_<ver>.parquet（code date + a_* 142 列 + f_on_*/f_vc_*/f_hi_* 33 列，float32）。

    python research/leak/build_alt.py --ver fwd|pit|ctl [--px px_fake.parquet] [--sample 200]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_feats as BF  # noqa: E402
import build_new as BN  # noqa: E402

FAMS = ("on", "vc", "hi")


def load_px(ver: str, px: str, sample: int) -> pd.DataFrame:
    cols = ["code", "date"] + [f"{k}_{ver}" for k in CL.PX] + ["volume", "amount", "turnover", "outstanding_share"]
    p = pd.read_parquet(CL.LEAK / px, columns=cols).rename(columns={f"{k}_{ver}": k for k in CL.PX})
    p = p[p["date"] <= CL.KEEP_TO]
    if sample:
        keep = sorted(p["code"].unique())[:sample]
        p = p[p["code"].isin(keep)]
    p = p.sort_values(["code", "date"]).reset_index(drop=True)
    p["_pos"] = p.groupby("code").cumcount().to_numpy()
    # 照抄 build_new.load 的后半段
    p["turnover"] = p["turnover"].clip(0, 0.6)
    pos = p["_pos"].to_numpy()
    c = p["close"].to_numpy(float)
    r = c / BN.lag(c, 1, pos) - 1
    r[~np.isfinite(r)] = np.nan
    p["r"] = np.clip(r, -0.35, 0.35)
    m = p.groupby("date")["r"].mean()
    m = m[p.groupby("date")["r"].count() >= (1000 if not sample else 1)]
    p["m0"] = p["date"].map(m).to_numpy(float)
    p["m1"] = p["date"].map(m.shift(1)).to_numpy(float)
    p["m2"] = p["date"].map(m.shift(2)).to_numpy(float)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", required=True, choices=CL.VERS)
    ap.add_argument("--px", default="px.parquet")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    px = load_px(a.ver, a.px, a.sample)
    print("[%s] 日线 %d 行 %d 只，%.0fs" % (a.ver, len(px), px["code"].nunique(), time.time() - t0), flush=True)
    keep_a = (px["date"] >= BF.KEEP_FROM).to_numpy()
    al = BF.alpha158(px, keep_a)
    acols = [c for c in al.columns if c.startswith("a_")]
    for i in range(0, len(acols), 20):
        cc = acols[i:i + 20]
        al[cc] = al.groupby("date")[cc].rank(pct=True).astype("float32")
    print("  a158 %d 行 %d 列，%.0fs" % (len(al), len(acols), time.time() - t0), flush=True)
    keep_f = (px["date"] >= BN.KEEP_FROM).to_numpy()
    fam = pd.DataFrame({"code": px["code"].to_numpy()[keep_f], "date": px["date"].to_numpy()[keep_f]})
    for f_ in FAMS:
        f = {}
        BN.FAMS[f_](px, f)
        for k, v in f.items():
            v = np.asarray(v, float)[keep_f]
            v[~np.isfinite(v)] = np.nan
            fam[f"f_{f_}_{k}"] = v.astype("float32")
        del f
    fcols = [c for c in fam.columns if c.startswith("f_")]
    fam[fcols] = fam.groupby("date")[fcols].rank(pct=True).astype("float32")
    print("  三族 %d 行 %d 列，%.0fs" % (len(fam), len(fcols), time.time() - t0), flush=True)
    del px
    out = fam.merge(al, on=["code", "date"], how="left")
    del fam, al
    dst = CL.LEAK / (a.out or f"alt_{a.ver}.parquet")
    out.to_parquet(dst, index=False)
    print("[%s] -> %s %d 行 %d 列，%.1f 分钟" % (a.ver, dst, len(out), out.shape[1] - 2, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
