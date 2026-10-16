"""
第 0 步：核实三套特征都只用价格比值（对价格整体缩放不变）。

生产特征（features.per_stock + add_chip_block + 三层变换前的原值）、Alpha158（build_feats.alpha158）、
on/vc/hi 三族（build_new.FAMS）、标签（label_frame / buyable.compute）：把四个价统一乘 k（3.7 和 1/2.3），
看每列最大相对差。不是比值的列会在这里现形。

    python research/leak/check_invariance.py
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build as B  # noqa: E402
import build_feats as BF  # noqa: E402
import build_new as BN  # noqa: E402
import features as F  # noqa: E402
from label import label_frame  # noqa: E402

L = []


def rel(a: pd.DataFrame, b: pd.DataFrame, cols) -> list[tuple[str, float]]:
    out = []
    for c in cols:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if (np.isfinite(x) != np.isfinite(y)).any():
            out.append((c, float("inf")))
            continue
        if not ok.any():
            out.append((c, 0.0))
            continue
        d = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(x[ok]), 1e-6)
        out.append((c, float(d.max())))
    return out


def main():
    daily = pd.read_parquet(CL.LEAK / "daily_snap.parquet")
    daily["date"] = daily["date"].astype(str).str[:10]
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily = daily[daily["date"] <= CL.KEEP_TO]
    rng = np.random.default_rng(3)
    codes = sorted(rng.choice(daily["code"].unique(), 40, replace=False))
    sub = daily[daily["code"].isin(codes)].sort_values(["code", "date"]).reset_index(drop=True)
    sub = B.attach_turnover(sub, None)
    raw_cols = [c for c in F.feature_columns() if c not in {n for *_, n in F.INTERACTIONS}]
    for k in (3.7, 1 / 2.3):
        s2 = sub.copy()
        for p in CL.PX:
            s2[p] = s2[p] * k
        # float_mcap 是 attach_turnover 用 amount/volume 把前复权价折回去算的，缩放后应几乎不变；这里重算一遍核
        s2 = B.attach_turnover(s2.drop(columns=["float_mcap", "turn_est"]), None)
        a_, b_ = [], []
        for code in codes:
            g1 = sub[sub["code"] == code].reset_index(drop=True)
            g2 = s2[s2["code"] == code].reset_index(drop=True)
            x1 = F.add_chip_block(F.per_stock(g1))
            x2 = F.add_chip_block(F.per_stock(g2))
            x1 = pd.concat([x1, label_frame(g1[["close", "high"]])], axis=1)
            x2 = pd.concat([x2, label_frame(g2[["close", "high"]])], axis=1)
            a_.append(x1)
            b_.append(x2)
        a, b = pd.concat(a_, ignore_index=True), pd.concat(b_, ignore_index=True)
        cols = [c for c in raw_cols if c in a.columns and c not in F.MARKET_WIDE and c not in ("rs20", "rs60", "rs_accel")]
        cols += ["ret20", "ret60", "float_mcap", "y_up", "y_t0", "y_top", "chip_avg"]
        r = sorted(rel(a, b, cols), key=lambda x: -x[1])
        CL.say("生产特征 价格 x %.3f：%d 列，最大相对差前 6：%s" % (
            k, len(cols), "; ".join("%s %.2e" % x for x in r[:6])), L)
        bad = [x for x in r if x[1] > 1e-6 and x[0] != "chip_avg"]
        CL.say("  超过 1e-6 的列（chip_avg 是绝对价位，不进特征，排除）：%s" % (bad or "无"), L)

    # Alpha158 + 三族
    px = sub[["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share"]].copy()
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    keep = np.ones(len(px), bool)

    def prep(p):
        p = p.copy()
        p["turnover"] = p["turnover"].clip(0, 0.6)
        pos = p["_pos"].to_numpy()
        c = p["close"].to_numpy(float)
        r = c / BN.lag(c, 1, pos) - 1
        r[~np.isfinite(r)] = np.nan
        p["r"] = np.clip(r, -0.35, 0.35)
        m = p.groupby("date")["r"].mean()
        p["m0"] = p["date"].map(m).to_numpy(float)
        p["m1"] = p["date"].map(m.shift(1)).to_numpy(float)
        p["m2"] = p["date"].map(m.shift(2)).to_numpy(float)
        return p
    for k in (3.7, 1 / 2.3):
        p2 = px.copy()
        for p in CL.PX:
            p2[p] = p2[p] * k
        a1, a2 = BF.alpha158(px, keep), BF.alpha158(p2, keep)
        cols = [c for c in a1.columns if c.startswith("a_")]
        r = sorted(rel(a1, a2, cols), key=lambda x: -x[1])
        CL.say("Alpha158 价格 x %.3f：%d 列，最大相对差前 6：%s" % (k, len(cols), "; ".join("%s %.2e" % x for x in r[:6])), L)
        CL.say("  超过 1e-5 的列：%s" % ([x for x in r if x[1] > 1e-5] or "无"), L)
        f1, f2 = {}, {}
        q1, q2 = prep(px), prep(p2)
        for fam in ("on", "vc", "hi"):
            BN.FAMS[fam](q1, f1)
            BN.FAMS[fam](q2, f2)
        d1 = pd.DataFrame({k_: np.asarray(v, float) for k_, v in f1.items()})
        d2 = pd.DataFrame({k_: np.asarray(v, float) for k_, v in f2.items()})
        r = sorted(rel(d1, d2, list(d1.columns)), key=lambda x: -x[1])
        CL.say("on/vc/hi 价格 x %.3f：%d 列，最大相对差前 6：%s" % (k, len(r), "; ".join("%s %.2e" % x for x in r[:6])), L)
        CL.say("  超过 1e-5 的列：%s" % ([x for x in r if x[1] > 1e-5] or "无"), L)
    (CL.HERE / "results_invariance.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
