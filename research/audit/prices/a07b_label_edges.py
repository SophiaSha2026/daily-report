"""a07b：标签的边界情形量化（生产日线上，用 a07 的重算结果）：
    停牌跨越（20 根 K 线跨 >20 个市场日）的行数和它们的正样本率；窗口不足（末尾 NaN）；
    次日停牌（open1 不是下一个交易日）；新股前 5 日；退市整理期（名字带「退」）。"""
import json
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


m = pd.read_parquet(OUT / "raw/a07_mine_prod.parquet")
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "open", "close", "high"])
d["code"] = d["code"].astype(str).str.zfill(6)
d = d.sort_values(["code", "date"]).reset_index(drop=True)
dates = np.array(sorted(d["date"].unique()))
didx = pd.Series(np.arange(len(dates)), index=dates)
d["_di"] = d["date"].map(didx).to_numpy()
d["gap1"] = d.groupby("code")["_di"].shift(-1) - d["_di"] - 1        # t 到 t+1 之间缺的市场日
m = m.merge(d[["code", "date", "gap1", "_di"]], on=["code", "date"])
v = m[np.isfinite(m["my_y_up"])].copy()
say("有 y_up 的行 %d，正样本率 %.4f%%" % (len(v), 100 * v["my_y_up"].mean()))
for lo, hi in ((21, 25), (26, 40), (41, 1000)):
    x = v[(v["span"] >= lo) & (v["span"] <= hi)]
    say("  窗口跨 %d~%d 个市场日的行 %d：y_up 正样本率 %.3f%%，y_openB %.3f%%" % (lo, hi, len(x), 100 * x["my_y_up"].mean(), 100 * x["my_y_openB"].mean()))
x = v[v["span"] == 20]
say("  窗口正好 20 个市场日的行 %d：y_up %.4f%%，y_openB %.4f%%" % (len(x), 100 * x["my_y_up"].mean(), 100 * x["my_y_openB"].mean()))
# 次日停牌：t+1 那根不是下一个市场日 -> 「次日开盘买」其实买不到
x = v[v["gap1"] >= 1]
say("  t 之后紧接停牌（t+1 根不是下一个市场日）的行 %d：y_openB 正样本 %d（%.2f%%），这些「次日开盘」实际是复牌日开盘" % (
    len(x), int(x["my_y_openB"].sum()), 100 * x["my_y_openB"].mean()))
x5 = v[(v["gap1"] >= 5)]
say("    其中停 >=5 个市场日的 %d 行，正样本 %d（%.2f%%）；>=20 的 %d 行，正样本 %d" % (
    len(x5), int(x5["my_y_openB"].sum()), 100 * x5["my_y_openB"].mean(), (v["gap1"] >= 20).sum(), int(v.loc[v["gap1"] >= 20, "my_y_openB"].sum())))
# 正样本里有多少靠停牌复牌的跳空：窗口内有 >=5 日缺口的正样本
pos = v[v["my_y_openB"] == 1]
say("  y_openB 正样本 %d 行，其中窗口跨 >20 市场日的 %d（%.2f%%）、跨 >=25 的 %d" % (len(pos), (pos["span"] > 20).sum(), 100 * (pos["span"] > 20).mean(), (pos["span"] >= 25).sum()))
# 新股：每只前 5 根
nb = d.groupby("code").cumcount()
first = d.groupby("code")["date"].transform("min")
newrows = d[(nb < 5) & (first > "2023-06-01")][["code", "date"]]
x = v.merge(newrows, on=["code", "date"])
say("  2023-06 后上市的票前 5 根 K 线 %d 行：y_up %.2f%%，y_openB %.2f%%（上市首日 / 前 5 日没涨跌停，一字板判据不适用）" % (len(x), 100 * x["my_y_up"].mean(), 100 * x["my_y_openB"].mean()))
# 退市整理期 / 名字带退：代码表没有名字，用 2026-10 的 ST 名单文件也没有名字；跳过，只量低价票
lowp = v.merge(d[["code", "date", "close"]], on=["code", "date"])
x = lowp[lowp["close"] < 2]
say("  收盘 < 2 元的行 %d：y_up %.2f%%，y_openB %.2f%%（生产候选池剔 <2 元）" % (len(x), 100 * x["my_y_up"].mean(), 100 * x["my_y_openB"].mean()))
# 北交所 30% / 创业板 20% 一字板：按板块的一字板率
v["board"] = np.where(v["code"].str[:2] == "30", "chinext", np.where(v["code"].str[:2] == "68", "star", np.where(v["code"].str[0].isin(list("849")), "bj", "main")))
say("  次日一字板率按板块：%s" % v.groupby("board")["my_yzB"].mean().round(5).to_dict())
say("  y_up=1 且次日一字板（买不到）的行按板块：%s" % v[(v["my_y_up"] == 1) & v["my_yzB"]].groupby("board").size().to_dict())
(OUT / "results_a07b.txt").write_text("\n".join(L), encoding="utf-8")
