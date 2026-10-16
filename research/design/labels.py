"""
起涨预测改进设计（2026-10-16）：标签和样本结构的描述统计。只读快照，不训模型，不读确认集标签。

    python research/design/labels.py

读：
    data/breakout/raw/exp/leak/labels_open_snap.parquet   y_open / r20_open / yizi1（全表，按 date < 2026-01-01 过滤）
    data/breakout/raw/exp/leak/train_snap.parquet         只读 code/date/board/y_up/y_t0/close 六列
输出：research/design/labels.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LEAK = ROOT / "data" / "breakout" / "raw" / "exp" / "leak"
OUT = Path(__file__).resolve().parent / "labels.txt"
DEV0, DEV1 = "2024-06-01", "2026-01-01"
lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


lab = pd.read_parquet(LEAK / "labels_open_snap.parquet")
lab = lab[lab["date"] < DEV1].copy()
lab = lab[np.isfinite(lab["y_open"])]
lab["m"] = lab["date"].str[:7]
dev = lab[lab["date"] >= DEV0]

say("== 1. 买得到口径标签的月度结构（开发集 2024-06..2025-12） ==")
g = dev.groupby("m")
tab = pd.DataFrame({
    "行数": g.size(),
    "y_open%": 100 * g["y_open"].mean(),
    "r20>20%": 100 * g["r20_open"].apply(lambda s: (s > 0.20).mean()),
    "r20>30%": 100 * g["r20_open"].apply(lambda s: (s > 0.30).mean()),
    "r20 中位%": 100 * g["r20_open"].median(),
    "r20 p90%": 100 * g["r20_open"].quantile(0.90),
    "一字%": 100 * g["yizi1"].mean(),
})
say(tab.round(2).to_string())
say("全开发集：y_open %.3f%%  r20>20%% %.2f%%  r20>30%% %.2f%%  r20>50%% %.2f%%" % (
    100 * dev["y_open"].mean(), 100 * (dev["r20_open"] > 0.2).mean(),
    100 * (dev["r20_open"] > 0.3).mean(), 100 * (dev["r20_open"] > 0.5).mean()))
say("2024-09 占开发集正样本的 %.1f%%（行数占 %.1f%%）" % (
    100 * dev[dev["m"] == "2024-09"]["y_open"].sum() / dev["y_open"].sum(),
    100 * (dev["m"] == "2024-09").mean()))

say()
say("== 2. 正样本的「事件」结构：连续亮灯的段 ==")
dev_s = dev.sort_values(["code", "date"])
pos = dev_s["y_open"].to_numpy() > 0
code = dev_s["code"].to_numpy()
new_run = pos & ~(np.roll(pos, 1) & (np.roll(code, 1) == code))
n_pos, n_runs = int(pos.sum()), int(new_run.sum())
run_id = np.cumsum(new_run) * pos
rl = pd.Series(run_id[pos]).value_counts()
say("正样本行 %d，连续段 %d 个，平均每段 %.1f 行；段长中位 %d、p90 %d、最长 %d" % (
    n_pos, n_runs, n_pos / max(n_runs, 1), int(rl.median()), int(rl.quantile(0.9)), int(rl.max())))
say("=> 按「事件」数，开发集独立正样本约 %d 个（19 个月，每月约 %d 个），不是 %d 行" % (n_runs, n_runs // 19, n_pos))

say()
say("== 3. 每天有多少正样本（决定按天排序学习 / 按天 top-k 目标的可行性） ==")
pd_ = dev.groupby("date")["y_open"].sum()
say("每天正样本数：中位 %d  p10 %d  p25 %d  p75 %d  p90 %d  最大 %d；少于 10 个的天 %d/%d，少于 5 个的天 %d" % (
    pd_.median(), pd_.quantile(0.1), pd_.quantile(0.25), pd_.quantile(0.75), pd_.quantile(0.9), pd_.max(),
    (pd_ < 10).sum(), len(pd_), (pd_ < 5).sum()))
pd25 = dev[dev["date"] >= "2025-01-01"].groupby("date")["y_open"].sum()
say("2025 年：中位 %d  p10 %d  少于 10 个的天 %d/%d" % (pd25.median(), pd25.quantile(0.1), (pd25 < 10).sum(), len(pd25)))

say()
say("== 4. r20_open 的分布形状（回归 / 分位数目标的依据） ==")
r = dev["r20_open"].dropna()
qs = r.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
say("分位：" + "  ".join("p%g %.1f%%" % (100 * k, 100 * v) for k, v in qs.items()))
say("偏度 %.2f  峰度 %.1f（正态为 0）；log1p 后偏度 %.2f" % (r.skew(), r.kurt(), np.log1p(r).skew()))
say("按天把 r20_open 做秩百分位后，秩 ≥ 0.98 的行里 y_open 占 %.1f%%；y_open 行里秩中位 %.3f" % (
    100 * dev.assign(rk=dev.groupby("date")["r20_open"].rank(pct=True)).query("rk >= 0.98")["y_open"].mean(),
    dev.assign(rk=dev.groupby("date")["r20_open"].rank(pct=True)).query("y_open > 0")["rk"].median()))

say()
say("== 5. 收盘口径 y_up 对 买得到 y_open ==")
cols = ["code", "date", "board", "y_up", "y_t0", "close"]
tr = pq.read_table(LEAK / "train_snap.parquet", columns=cols, filters=[("date", ">=", DEV0), ("date", "<", DEV1)]).to_pandas()
tr["code"] = tr["code"].astype(str).str.zfill(6)
m = tr.merge(dev[["code", "date", "y_open", "yizi1"]], on=["code", "date"], how="inner")
m = m[np.isfinite(m["y_up"])]
ct = pd.crosstab(m["y_up"], m["y_open"])
say(ct.to_string())
say("y_up=1 里 y_open=0 的占 %.1f%%（其中次日一字 %.1f%%）；y_open=1 里 y_up=0 的占 %.1f%%" % (
    100 * (m[(m.y_up == 1)]["y_open"] == 0).mean(),
    100 * m[(m.y_up == 1) & (m.y_open == 0)]["yizi1"].mean(),
    100 * (m[(m.y_open == 1)]["y_up"] == 0).mean()))
say("按板块（开发集）：")
bb = m.groupby("board").agg(行数=("y_up", "size"), y_up=("y_up", "mean"), y_open=("y_open", "mean"), 一字=("yizi1", "mean"))
say((bb.assign(y_up=100 * bb.y_up, y_open=100 * bb.y_open, 一字=100 * bb["一字"])).round(2).to_string())
say("y_t0（起涨第一天）开发集 %d 个" % int(np.nansum(tr["y_t0"])))

say()
say("== 6. 价位：前复权价 < 1 元 / 取整颗粒度 ==")
c = tr["close"]
for px in (2, 5, 8, 10):
    say("前复权收盘 < %d 元的行 %.1f%%（一分钱颗粒 > %.2f%%）" % (px, 100 * (c < px).mean(), 100 / px))

OUT.write_text("\n".join(lines), encoding="utf-8")
print("->", OUT)

# ---- 追加（同次会话）：方差分解和阈值附近的「险些」 ----
lines2: list[str] = []


def say2(s: str = "") -> None:
    print(s)
    lines2.append(s)


say2()
say2("== 7. r20_open 的方差分解：多少是「当天」、多少是「这只票」 ==")
rr = dev[["date", "r20_open"]].dropna()
tot = rr["r20_open"].var()
between = rr.groupby("date")["r20_open"].transform("mean").var()
say2("总方差 %.4f，按天均值的方差 %.4f（占 %.1f%%）；log1p 口径占 %.1f%%" % (
    tot, between, 100 * between / tot,
    100 * np.log1p(rr["r20_open"]).groupby(rr["date"]).transform("mean").var() / np.log1p(rr["r20_open"]).var()))
dm = rr.groupby("date")["r20_open"].mean()
say2("按天均值 r20：中位 %.1f%%  p10 %.1f%%  p90 %.1f%%  最大 %.1f%%（%s）" % (
    100 * dm.median(), 100 * dm.quantile(0.1), 100 * dm.quantile(0.9), 100 * dm.max(), dm.idxmax()))
say2("绝对阈值 50%% 之下的「险些」：r20 在 40~50%% 的行占 %.2f%%，>50%% 的占 %.2f%%；30~50%% 占 %.2f%%" % (
    100 * rr["r20_open"].between(0.40, 0.50).mean(), 100 * (rr["r20_open"] > 0.5).mean(),
    100 * rr["r20_open"].between(0.30, 0.50).mean()))
say2("每天按 r20 排名取前 3%% 当正样本（相对标签）：和绝对 y_open 的重合（Jaccard）%.2f" % (
    (lambda a, b: (a & b).sum() / (a | b).sum())(
        dev.groupby("date")["r20_open"].rank(pct=True) >= 0.97, dev["y_open"] > 0)))
say2("2024-09 之外的月份里，相对标签前 3%% 的 r20 门槛（当天第 97 分位）：中位 %.1f%%，最低 %.1f%%" % (
    100 * dev[dev["m"] != "2024-09"].groupby("date")["r20_open"].quantile(0.97).median(),
    100 * dev[dev["m"] != "2024-09"].groupby("date")["r20_open"].quantile(0.97).min()))
OUT.write_text(OUT.read_text(encoding="utf-8") + "\n" + "\n".join(lines2), encoding="utf-8")
