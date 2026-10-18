"""a03：按板块分层抽样 >=200 只真实代码（含 ST、除权多次、停牌过、新股），写 raw/sample.csv。只读代码表 / 名单 / 两张价格表。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261018)

codes = pd.read_csv(ROOT / "cache/codes.csv", dtype=str)["code"].str.zfill(6)
st = set(json.load(open(ROOT / "cache/st_codes.json", encoding="utf-8"))["codes"])
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date"])
d["code"] = d["code"].astype(str).str.zfill(6)
dates = np.array(sorted(d["date"].unique()))
di = pd.Series(np.arange(len(dates)), index=dates)
d = d.sort_values(["code", "date"])
d["_di"] = d["date"].map(di).to_numpy()
gap = d["_di"] - d.groupby("code")["_di"].shift(1) - 1
gap_codes = set(d.loc[gap >= 5, "code"].unique())
per = d.groupby("code")["date"].agg(["min", "max", "size"])
new_codes = set(per.index[per["min"] >= "2025-06-01"])
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "qfq_factor"])
u["code"] = u["code"].astype(str).str.zfill(6)
u = u.sort_values(["code", "date"])
step = (u.groupby("code")["qfq_factor"].shift(1) / u["qfq_factor"] - 1).abs() > 1e-6
nx = step.groupby(u["code"]).sum()
nx3 = u.loc[step & (u["date"] >= "2023-01-01"), "code"].value_counts()
multi_x = set(nx3.index[nx3 >= 3])          # 近三年除权 >= 3 次


def board(c):
    if c.startswith("30"):
        return "chinext"
    if c.startswith("68"):
        return "star"
    if c[0] in "849":
        return "bj"
    return "sh_main" if c[0] == "6" else "sz_main"


df = pd.DataFrame({"code": codes})
df["board"] = df["code"].map(board)
df["st"] = df["code"].isin(st)
df["gap"] = df["code"].isin(gap_codes)
df["new"] = df["code"].isin(new_codes)
df["multi_x"] = df["code"].isin(multi_x)
df["nx_total"] = df["code"].map(nx).fillna(0).astype(int)
df["in_daily"] = df["code"].isin(per.index)
pick = []
for b, n in (("sh_main", 50), ("sz_main", 50), ("chinext", 45), ("star", 30), ("bj", 30)):
    pool = df[(df["board"] == b) & df["in_daily"]]
    pick += list(rng.choice(pool["code"].to_numpy(), size=min(n, len(pool)), replace=False))
for col, n in (("st", 20), ("multi_x", 25), ("gap", 20), ("new", 15)):
    pool = df[df[col] & df["in_daily"] & ~df["code"].isin(pick)]
    pick += list(rng.choice(pool["code"].to_numpy(), size=min(n, len(pool)), replace=False))
s = df[df["code"].isin(pick)].copy()
s["tag"] = s.apply(lambda r: ",".join(k for k in ("st", "gap", "new", "multi_x") if r[k]), axis=1)
s = s.sort_values("code").reset_index(drop=True)
(OUT / "raw").mkdir(exist_ok=True)
s.to_csv(OUT / "raw/sample.csv", index=False)
print("样本", len(s), "板块", s["board"].value_counts().to_dict(), "ST", int(s["st"].sum()), "停牌过", int(s["gap"].sum()),
      "新股", int(s["new"].sum()), "近三年除权>=3", int(s["multi_x"].sum()))
print("全市场：停牌过(缺>=5日)", len(gap_codes), "新股(2025-06 后起)", len(new_codes), "近三年除权>=3", len(multi_x), "ST", len(st))
