"""a12b：静态 ST 名单（cache/st_codes.json，2026-09-16）vs BaoStock 逐日 isST（raw/bs_isst.parquet，沪深）。
    量：(1) 逐日 ST 只数 vs 静态名单只数；(2) 静态名单剔掉的 (code,日) 里当天其实不是 ST 的（误剔）、
    静态名单没剔但当天是 ST 的（漏剔）；(3) 这两类在候选池口径（收盘 >= 2 元、有 y_up）里的 y_up 正样本率；
    (4) 对涨跌停幅度的影响：主板 ST 日在 2026-07-06 前按 5%。可以只跑部分（续跑中途）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


bsd = pd.read_parquet(OUT / "raw/bs_isst.parquet")
bsd = bsd[bsd["date"].notna()].copy()
bsd["isST"] = bsd["isST"].astype(int)
bsd["trading"] = bsd["tradestatus"].astype(int) == 1
say("BaoStock 覆盖 %d 只 %d 行 %s..%s（扫描%s）" % (bsd["code"].nunique(), len(bsd), bsd["date"].min(), bsd["date"].max(),
                                              "完成" if bsd["code"].nunique() >= 5000 else "进行中，部分样本"))
st = set(json.load(open(ROOT / "cache/st_codes.json", encoding="utf-8"))["codes"])
bsd["static_st"] = bsd["code"].isin(st)
tr = bsd[bsd["trading"]]
# (1) 逐日
day = tr.groupby("date").agg(n=("code", "size"), st_daily=("isST", "sum"), st_static=("static_st", "sum"))
for dt in ("2020-07-01", "2021-06-30", "2022-06-30", "2023-06-30", "2024-06-28", "2025-06-30", "2026-06-30", "2026-10-16"):
    if dt in day.index:
        r = day.loc[dt]
        say("  %s：在市 %d，当天 ST %d，静态名单内 %d" % (dt, r["n"], r["st_daily"], r["st_static"]))
# (2) 误剔 / 漏剔
tr = tr.copy()
tr["wrong_excl"] = tr["static_st"] & (tr["isST"] == 0)
tr["missed"] = ~tr["static_st"] & (tr["isST"] == 1)
by_year = tr.groupby(tr["date"].str[:4]).agg(rows=("code", "size"), st_rows=("isST", "sum"), wrong_excl=("wrong_excl", "sum"), missed=("missed", "sum"))
by_year["wrong_excl_pct"] = (100 * by_year["wrong_excl"] / by_year["rows"]).round(3)
by_year["missed_pct"] = (100 * by_year["missed"] / by_year["rows"]).round(3)
say("按年（交易日行）：\n" + by_year.to_string())
say("  误剔（静态名单内、当天不是 ST）的只数 %d；漏剔（当天 ST、不在名单）的只数 %d" % (tr.loc[tr["wrong_excl"], "code"].nunique(), tr.loc[tr["missed"], "code"].nunique()))
# ST 日的涨跌停：主板 ST 2026-07-06 前 5%
main = tr["code"].str[:2].isin(["60", "00"])
st_main_before = tr[(tr["isST"] == 1) & main & (tr["date"] < "2026-07-06")]
say("  主板 ST 日（2026-07-06 前，涨跌停实为 5%%）共 %d 行，其中不在静态名单（代码按 10%% 算）%d 行；在名单内的 %d 行（回测剔了，但 limit_pct 仍按 10%%）" % (
    len(st_main_before), (~st_main_before["static_st"]).sum(), st_main_before["static_st"].sum()))
# (3) 对标签 / 候选池的影响：并上生产标签重算（a07）和收盘价
lab = pd.read_parquet(OUT / "raw/a07_mine_prod.parquet", columns=["code", "date", "my_y_up", "my_y_openB"])
px = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "close"])
px["code"] = px["code"].astype(str).str.zfill(6)
m = tr.merge(lab, on=["code", "date"]).merge(px[["code", "date", "close"]], on=["code", "date"], suffixes=("_bs", ""))
m = m[(m["close"] >= 2) & np.isfinite(m["my_y_openB"]) & (m["date"] >= "2023-12-01")]
say("2023-12 起、收盘 >= 2 元、有标签的行 %d：" % len(m))
for tag, mask in (("静态名单外且当天非 ST（候选池主体）", ~m["static_st"] & (m["isST"] == 0)),
                  ("漏剔：当天 ST 但不在名单（回测当成普通票）", m["missed"]),
                  ("误剔：名单内但当天非 ST（回测白剔了）", m["wrong_excl"]),
                  ("名单内且当天 ST", m["static_st"] & (m["isST"] == 1))):
    x = m[mask]
    say("  %s：%d 行（%.3f%%），y_openB 正样本率 %.3f%%，y_up %.3f%%" % (tag, len(x), 100 * len(x) / max(len(m), 1), 100 * x["my_y_openB"].mean() if len(x) else 0, 100 * x["my_y_up"].mean() if len(x) else 0))
# 按天：漏剔的票每天平均几只、最多几只
dm = m[m["missed"]].groupby("date").size()
say("  漏剔按天：有漏剔的天数 %d / %d，日均 %.1f 只，最多 %d 只（%s）" % (len(dm), m["date"].nunique(), dm.mean() if len(dm) else 0, dm.max() if len(dm) else 0, dm.idxmax() if len(dm) else ""))
(OUT / "results_a12b.txt").write_text("\n".join(L), encoding="utf-8")
