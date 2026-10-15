"""a158 / base_a158 的事后稳健性（不重训）：去掉 2024-09、前 5/20/30、板块构成、和现状名单重合、一字板占比。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", Path(__file__).parent, ROOT / "research" / "stats"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd  # noqa: E401,E402
import evalkit as K  # noqa: E402
import hac  # noqa: E402
from exp_altpath import daily_diff, nw_t  # noqa: E402

d = K.EXP / "altpath_v1"
T = {a: pd.read_parquet(d / f"top_{a}.parquet") for a in ("base", "a158", "base_a158")}
lab = pd.read_parquet(K.DATA / "raw" / "labels_open.parquet", columns=["code", "date", "yizi1", "r20_open"])
fl = pd.read_parquet(K.DATA / "train.parquet", columns=["code", "date", "float_mcap", "close"],
                     filters=[("date", ">=", K.DEV[0]), ("date", "<", K.CONFIRM[0])])
fl["code"] = fl["code"].astype(str).str.zfill(6)
out = []
for arm in ("a158", "base_a158"):
    a, b = T[arm], T["base"]
    for lbl, f in (("全部", lambda x: x), ("去 2024-09", lambda x: x[x["date"].str[:7] != "2024-09"]),
                   ("去 2024-09/10", lambda x: x[~x["date"].str[:7].isin(["2024-09", "2024-10"])])):
        for k in (5, 10, 20, 30):
            dd = daily_diff(f(a), f(b), k=k)
            t, _ = nw_t(dd)
            out.append("%-9s %-14s 前%-2d %+.2fpp NW t=%.2f 块自举 p=%.3f" % (arm, lbl, k, 100 * dd.mean(), t, hac.mbb(dd)["p"]))
for arm, t in T.items():
    x = t[t["rank"] <= 10].merge(lab, on=["code", "date"], how="left").merge(fl, on=["code", "date"], how="left")
    bd = x["board"].value_counts(normalize=True).round(3).to_dict()
    hb = x.groupby("board")["y_open"].mean().round(3).to_dict()
    out.append("%-9s 板块占比 %s 板块命中 %s 次日一字 %.1f%% 流通市值中位 %.1f 亿 价格中位 %.1f r20 中位 %.3f" % (
        arm, bd, hb, 100 * x["yizi1"].mean(), x["float_mcap"].median() / 1e8 if x["float_mcap"].median() > 1e6 else x["float_mcap"].median(),
        x["close"].median(), x["r20_open"].median()))
s0 = T["base"][T["base"]["rank"] <= 10].groupby("date")["code"].apply(set)
for arm in ("a158", "base_a158"):
    s1 = T[arm][T[arm]["rank"] <= 10].groupby("date")["code"].apply(set)
    ov = np.mean([len(s0[i] & s1[i]) / 10 for i in s0.index if i in s1.index])
    out.append("%s 前 10 和现状重合 %.1f%%" % (arm, 100 * ov))
print("\n".join(out))
(Path(__file__).parent / "check_a158.txt").write_text("\n".join(out), encoding="utf-8")
