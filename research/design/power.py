"""
起涨预测改进设计（2026-10-16）：统计功效 / 最小可检出效应。只读已有产物，不训模型。

    python research/design/power.py

读：
    research/stats/out/daily_hits.parquet            开发集 387 天各臂每天前 10 命中（买得到口径）
    data/breakout/raw/exp/feat_final_v1/top_*.parquet 最终候选和同循环对照（每天前 200）
输出：research/design/power.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "breakout"))
sys.path.insert(0, str(ROOT / "src"))
import evalkit as K  # noqa: E402

EXP = ROOT / "data" / "breakout" / "raw" / "exp"
OUT = Path(__file__).resolve().parent / "power.txt"
lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def acf(x: np.ndarray, k: int) -> float:
    xc = x - x.mean()
    return float((xc[k:] * xc[:-k]).sum() / (xc * xc).sum())


def nw_se(x: np.ndarray, lag: int = 20) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    g0 = K._autocov(x, 0)
    lrv = g0 + sum(2 * (1 - k / (lag + 1)) * K._autocov(x, k) for k in range(1, lag + 1))
    return float(np.sqrt(max(lrv, 1e-18) / n))


def daily_hit(t: pd.DataFrame, k: int = 10, y: str = "y_open") -> pd.Series:
    return t[t["rank"] <= k].groupby("date")[y].mean()


say("== 1. 开发集每天前 10 命中序列的离散和自相关（买得到口径） ==")
dh = pd.read_parquet(ROOT / "research" / "stats" / "out" / "daily_hits.parquet")
say("列：%s" % [c for c in dh.columns if c.startswith("W:")])
for col in [c for c in dh.columns if c.startswith("W:")][:3]:
    x = dh[col].dropna().to_numpy(float)
    say("%-16s n=%d 均值 %.4f 日标准差 %.4f  ACF1 %.2f ACF5 %.2f ACF10 %.2f ACF20 %.2f  NW se %.4f  iid se %.4f  n_eff≈%d" % (
        col, len(x), x.mean(), x.std(ddof=1), acf(x, 1), acf(x, 5), acf(x, 10), acf(x, 20),
        nw_se(x), x.std(ddof=1) / np.sqrt(len(x)), int(len(x) * (x.std(ddof=1) / np.sqrt(len(x)) / nw_se(x)) ** 2)))
p = 0.12
say("对照：若每天 10 只独立二项，p=%.2f，日标准差应为 %.4f；实际更大的部分是「当天共同因子」" % (p, np.sqrt(p * (1 - p) / 10)))

say()
say("== 2. 配对差序列（臂 - 对照）的离散：决定能检出多小的效应 ==")
fin = EXP / "feat_final_v1"
T = {n: pd.read_parquet(fin / f"top_{n}.parquet") for n in ("base_s7", "cand_s7", "a158_s7", "base_s9", "cand_s9")}
pairs = [("cand_s7", "base_s7", "特征大改（前 10 重合约 20%）"),
         ("a158_s7", "base_s7", "加 Alpha158（重合约 20%）"),
         ("cand_s7", "a158_s7", "候选 对 a158（同一大族，重合高）"),
         ("base_s9", "base_s7", "只换下采样种子（同特征同参数）")]
rows = []
for a, b, desc in pairs:
    da, db = daily_hit(T[a]), daily_hit(T[b])
    d = (da - db).dropna().to_numpy(float)
    ta, tb = T[a][T[a]["rank"] <= 10], T[b][T[b]["rank"] <= 10]
    ov = (ta.merge(tb, on=["date", "code"]).groupby("date").size()
          .reindex(sorted(set(ta["date"])), fill_value=0).mean() / 10)
    se = nw_se(d)
    rows.append((a, b, desc, len(d), d.mean(), d.std(ddof=1), acf(d, 1), se, ov))
    say("%-9s-%-9s %-34s n=%d 均值 %+.4f 日标准差 %.4f ACF1 %.2f NW se %.4f 前10重合 %.0f%%  NW t %.2f" % (
        a, b, desc, len(d), d.mean(), d.std(ddof=1), acf(d, 1), se, 100 * ov, d.mean() / se))

say()
say("== 3. 最小可检出效应 MDE（单侧，功效 80%：效应 ≥ (c + 0.84) × NW se） ==")
say("   门槛 c：2.5（下限）/ 2.8（族内最大 t）/ 3.2（登记簿 Bonferroni，m≈77~103）")
for a, b, desc, n, m, sd, r1, se, ov in rows:
    s = "%-34s 387 天 NW se %.2fpp -> MDE" % (desc, 100 * se)
    for c in (2.5, 2.8, 3.2):
        s += "  c=%.1f: %+.1fpp" % (c, 100 * (c + 0.84) * se)
    say(s)
say()
say("   反过来：要在 c=3.2、功效 80% 下检出 +2 / +3 / +5pp，需要多少个交易日（按 NW se ∝ 1/√n 外推）")
for a, b, desc, n, m, sd, r1, se, ov in rows:
    s = "%-34s" % desc
    for eff in (0.02, 0.03, 0.05):
        need = n * ((3.2 + 0.84) * se / eff) ** 2
        s += "  %+.0fpp: %5.0f 天（%.1f 年）" % (100 * eff, need, need / 243)
    say(s)

say()
say("== 4. 同一改动在不同种子下的差异（种子噪声的量级） ==")
for a, b in (("cand_s7", "base_s7"), ("cand_s9", "base_s9")):
    d = (daily_hit(T[a]) - daily_hit(T[b])).dropna()
    say("%s-%s：+%.2fpp（NW t %.2f）" % (a, b, 100 * d.mean(), d.mean() / nw_se(d.to_numpy())))
d7 = (daily_hit(T["cand_s7"]) - daily_hit(T["base_s7"])).dropna()
d9 = (daily_hit(T["cand_s9"]) - daily_hit(T["base_s9"])).dropna()
dd = (d7 - d9).dropna()
say("两个种子的差值序列之差：均值 %+.2fpp 日标准差 %.4f（种子本身给配对差带来的噪声）" % (100 * dd.mean(), dd.std(ddof=1)))

say()
say("== 5. 名次曲线（base_s7，开发集）：信号有多宽 ==")
tb = T["base_s7"]
for lo, hi in ((1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 200)):
    s = tb[(tb["rank"] >= lo) & (tb["rank"] <= hi)]
    say("  名次 %3d~%3d  n=%6d  买得到命中 %.2f%%  收盘口径 %.2f%%" % (lo, hi, len(s), 100 * s["y_open"].mean(), 100 * s["y_up"].mean()))

OUT.write_text("\n".join(lines), encoding="utf-8")
print("->", OUT)

# ---- 追加：换一把尺子功效有多大（同一对臂 cand_s7 对 base_s7） ----
lines3: list[str] = []


def say3(s: str = "") -> None:
    print(s)
    lines3.append(s)


say3()
say3("== 6. 同一对臂（cand_s7 对 base_s7），不同尺子的 NW se 和 t ==")
a, b = T["cand_s7"], T["base_s7"]


def series(t, k, fn):
    s = t[t["rank"] <= k]
    return s.groupby("date").apply(fn)


rulers = [
    ("前10 买得到命中", 10, lambda s: s["y_open"].mean()),
    ("前30 买得到命中", 30, lambda s: s["y_open"].mean()),
    ("前50 买得到命中", 50, lambda s: s["y_open"].mean()),
    ("前10 r20>30% 占比", 10, lambda s: (s["r20_open"] > 0.30).mean() if "r20_open" in s else np.nan),
    ("前10 收盘口径命中", 10, lambda s: s["y_up"].mean()),
]
# top_*.parquet 没有 r20_open，从 labels 快照并上
lab = pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "exp" / "leak" / "labels_open_snap.parquet",
                      columns=["code", "date", "r20_open"])
a = a.merge(lab, on=["code", "date"], how="left")
b = b.merge(lab, on=["code", "date"], how="left")
rulers += [("前10 平均 log1p(r20)", 10, lambda s: np.log1p(s["r20_open"].clip(lower=-0.99)).mean()),
           ("前10 r20 中位", 10, lambda s: s["r20_open"].median())]
for name, k, fn in rulers:
    d = (series(a, k, fn) - series(b, k, fn)).dropna().to_numpy(float)
    se = nw_se(d)
    say3("%-22s 差 %+.4f  NW se %.4f  NW t %.2f  |t| 相对前10命中的倍数 %.2f" % (name, d.mean(), se, d.mean() / se,
                                                                     (d.mean() / se) / 2.82))
say3()
say3("三种子平均后再比（cand 均值 对 base 均值）：")
A = pd.concat([daily_hit(T[f"cand_s{s}"]) for s in (7, 9)], axis=1).mean(axis=1)
B = pd.concat([daily_hit(T[f"base_s{s}"]) for s in (7, 9)], axis=1).mean(axis=1)
d = (A - B).dropna().to_numpy(float)
say3("两种子平均：差 %+.4f NW se %.4f NW t %.2f（单种子 se 0.0185）" % (d.mean(), nw_se(d), d.mean() / nw_se(d)))
OUT.write_text(OUT.read_text(encoding="utf-8") + "\n" + "\n".join(lines3), encoding="utf-8")
