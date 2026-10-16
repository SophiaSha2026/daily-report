"""
过线规则第 5 条两种写法的通过率（evalkit v3，2026-10-16）：
    v2  最差三个月的月差 >= 0（等价于最差一个月 >= 0）
    v3  任意连续 3 个月的滚动平均月差 >= -5pp

两种模拟：
    A 参数化：19 个月的月差 iid 正态，sd 10pp（特征大改臂）/ 4.5pp（同族增量臂），真效应 0 / +3 / +5pp
    B 半参数：真实的逐日差值序列（feat_final_v1 cand_s7 对 base_s7、cand_s7 对 a158_s7，开发集 387 天）去均值、
      加真效应，按 20 天块重抽，日期标签按位置沿用，再切成月。保留日内自相关和月份长短

只读 data/breakout/raw/exp/feat_final_v1/top_*.parquet（<= 2025-12-31），不训模型。

    python research/design/power_roll3.py     -> research/design/power_roll3.txt
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
OUT = Path(__file__).resolve().parent / "power_roll3.txt"
lines: list[str] = []
RULE3 = K.GATE_V3["roll3_min_diff"]


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def rules(md: np.ndarray) -> tuple[bool, bool]:
    """md：按时间顺序的月差。返回 (v2 最差月 >= 0, v3 滚动三月最低 >= -5pp)。"""
    r3 = np.convolve(md, np.ones(3) / 3, mode="valid")
    return bool(md.min() >= 0), bool(r3.min() >= RULE3)


say("== A. 参数化：19 个月 iid 正态月差 ==")
rng = np.random.default_rng(20261016)
N = 20000
for sd in (0.10, 0.045):
    for mu in (0.0, 0.03, 0.05):
        md = rng.normal(mu, sd, size=(N, 19))
        v2 = (md.min(axis=1) >= 0).mean()
        r3 = np.stack([np.convolve(x, np.ones(3) / 3, mode="valid") for x in md])
        v3 = (r3.min(axis=1) >= RULE3).mean()
        say("月差 sd %4.1fpp 真效应 %+.0fpp：v2 最差月>=0 通过 %5.1f%%   v3 滚动三月>=-5pp 通过 %5.1f%%" % (
            100 * sd, 100 * mu, 100 * v2, 100 * v3))

say()
say("== B. 半参数：真实逐日差值序列去均值 + 真效应，20 天块重抽，按位置沿用月份 ==")
T = {n: pd.read_parquet(EXP / "feat_final_v1" / f"top_{n}.parquet") for n in ("base_s7", "cand_s7", "a158_s7")}
for n, t in T.items():
    assert str(t["date"].max()) <= "2025-12-31", f"{n} 含开发集之后的日期"


def daily_hit(t: pd.DataFrame) -> pd.Series:
    return t[t["rank"] <= 10].groupby("date")["y_open"].mean()


B = 5000
for a, b, desc in (("cand_s7", "base_s7", "特征大改（NW se 1.85pp）"), ("cand_s7", "a158_s7", "同族增量（NW se 0.87pp）")):
    d = (daily_hit(T[a]) - daily_hit(T[b])).dropna().sort_index()
    months = d.index.str[:7].to_numpy()
    x = d.to_numpy(float)
    xc = x - x.mean()
    md_real = d.groupby(months).mean()
    say("%s 对 %s %s：%d 天 %d 个月，实际月差 sd %.1fpp，最差月 %+.1f，滚动三月最低 %+.1f" % (
        a, b, desc, len(x), len(md_real), 100 * md_real.std(ddof=1), 100 * md_real.min(),
        100 * np.convolve(md_real.to_numpy(), np.ones(3) / 3, mode="valid").min()))
    idx = K._mbb_index(len(x), K.BLOCK, B, np.random.default_rng(7))
    mon_codes, mon_inv = np.unique(months, return_inverse=True)
    for mu in (0.0, 0.03, 0.05):
        ok2 = ok3 = 0
        for i in range(B):
            xs = xc[idx[i]] + mu
            md = np.bincount(mon_inv, weights=xs, minlength=len(mon_codes)) / np.bincount(mon_inv, minlength=len(mon_codes))
            p2, p3 = rules(md)
            ok2 += p2
            ok3 += p3
        say("    真效应 %+.0fpp：v2 通过 %5.1f%%   v3 通过 %5.1f%%" % (100 * mu, 100 * ok2 / B, 100 * ok3 / B))

say()
say("读法：v2「最差月 >= 0」在真 +5pp 下大改臂几乎过不了（参数化 0.1%，半参数 0.5%），增量臂也只有 7%~44%。")
say("v3「滚动三月 >= -5pp」真 +5pp：大改臂参数化 61%、半参数（真实序列重抽）90%；真 0 效应误放参数化 12%、半参数 22%。")
say("设计文档 7.2 写的「约 75% / 约 28%」是按 iid 月差、有效窗口数约 6 的手算，两种模拟都不逐位对上，以本文件为准。")
say("第 5 条只是次级稳定性规则，主判定仍是 NW t >= 族门槛，误放的那两成要靠第 1~3 条挡。")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("->", OUT)
