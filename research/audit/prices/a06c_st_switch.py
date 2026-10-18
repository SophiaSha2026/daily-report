"""a06c：主板 ST 涨跌幅从 5% 换到 10% 的日期。当前 ST 名单里的主板票，2026-06-01..2026-08-31 逐日：触 5% 板只数、触 10% 板只数、|ret| 在 (5.2%,9.8%) 的只数。"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import datasource as ds  # noqa: E402

st = set(json.load(open(ROOT / "cache/st_codes.json", encoding="utf-8"))["codes"])
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "high_u", "low_u", "close_u", "volume"])
u["code"] = u["code"].astype(str).str.zfill(6)
u = u[(u["volume"] > 0)].sort_values(["code", "date"])
u["pc"] = u.groupby("code")["close_u"].shift(1)
x = u[u["code"].isin(st) & (u["code"].map(ds.board_of) == "main") & (u["date"] >= "2026-06-01") & (u["date"] <= "2026-08-31")].copy()
for p, nm in ((0.05, "5"), (0.10, "10")):
    lim = ds.limit_price_arr(x["pc"], pd.Series(p * 100, index=x["pc"].index), x["code"])
    dn = ds.limit_price_arr(x["pc"], pd.Series(-p * 100, index=x["pc"].index), x["code"])
    x["up" + nm] = (x["high_u"] - lim).abs() < 1e-9
    x["dn" + nm] = (x["low_u"] - dn).abs() < 1e-9
    x["beyond" + nm] = (x["high_u"] > lim + 1e-9) | (x["low_u"] < dn - 1e-9)
g = x.groupby("date").agg(n=("code", "size"), up5=("up5", "sum"), dn5=("dn5", "sum"), beyond5=("beyond5", "sum"), up10=("up10", "sum"), dn10=("dn10", "sum"), beyond10=("beyond10", "sum"))
pd.set_option("display.width", 200)
s = g.to_string()
print(s)
first_beyond = g.index[g["beyond5"] >= 5]
print("主板 ST 当日超出 ±5% 的只数 >=5 的第一天：", first_beyond[0] if len(first_beyond) else None)
print("之后 beyond5 每天只数的中位数：", g.loc[first_beyond[0]:, "beyond5"].median() if len(first_beyond) else None)
print("之前（6 月）beyond5 合计：", g.loc[:"2026-06-30", "beyond5"].sum(), "，beyond10 合计（整个窗口）：", g["beyond10"].sum())
(OUT / "results_a06c.txt").write_text(s + "\n换挡日（beyond5>=5 的第一天）：%s\n" % (first_beyond[0] if len(first_beyond) else None), encoding="utf-8")
