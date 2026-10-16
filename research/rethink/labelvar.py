"""
标签变体下 规则核 / 现状模型 / 全市场 的命中（给「标签要什么」拍板用）：
  A 现标签：次日开盘买、20 根内最高价 ≥ +50%
  B 冲高且不崩：A 且第 20 根收盘 ≥ 买入价
  C 持续上涨：第 20 根收盘 ≥ +20%
  D 冲高 30%：最高价 ≥ +30%
另算 规则 − 模型 在开发集 + 筛选集 合并（逆方差加权）的差。
    PYTHONIOENCODING=utf-8 python research/rethink/labelvar.py > research/rethink/labelvar.txt
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules2 import load_set, top  # noqa: E402
from common import nw_t  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
ST = {str(c).zfill(6) for c in json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8")).get("codes", [])}
def say(*a): print(*a, flush=True)

LABELS = {"A 冲高50%": lambda d: d["y"], "B 冲高50%且20根收盘不亏": lambda d: d["y"] & (d["c20"] >= 0),
          "C 20根收盘≥+20%": lambda d: d["c20"] >= 0.2, "D 冲高30%": lambda d: d["r20_open"] >= 0.3}
diffs = {}
for tag, r in [
    ("开发集", load_set(EXP / "leak" / "labels_open_snap.parquet", EXP / "leak" / "daily_snap.parquet", "2024-02-01",
                     [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")],
                     EXP / "feat_final_v1" / "all_base_s7.parquet", [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")])),
    ("筛选集", load_set(EXP / "screen" / "labels_screen.parquet", EXP / "screen" / "px_screen.parquet", "2021-09-01",
                     [("date", ">=", "2022-01-01"), ("date", "<=", "2024-05-31")],
                     EXP / "screen_base_v1" / "all_base_s7.parquet", None))]:
    elig = ~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"]) & np.isfinite(r["c20"])
    r = r[elig].copy()
    say(f"\n== {tag} ==")
    say("%-26s %8s %8s %8s %8s %8s" % ("标签", "全市场", "规则前10", "规则前30", "模型前10", "模型前30"))
    picks = {"规则前10": top(r, r["ret20"], 10), "规则前30": top(r, r["ret20"], 30),
             "模型前10": top(r, r["p"], 10), "模型前30": top(r, r["p"], 30)}
    for nm, fn in LABELS.items():
        row = [fn(r).groupby(r["date"]).mean().mean()] + [fn(s).groupby(s["date"]).mean().mean() for s in picks.values()]
        say("%-26s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%" % tuple([nm] + [100 * v for v in row]))
    for k in (10, 30):
        a = picks[f"规则前{k}"].groupby("date")["y"].mean(); b = picks[f"模型前{k}"].groupby("date")["y"].mean()
        mu, se, t = nw_t((a - b.reindex(a.index)).dropna().values)
        diffs[(tag, k)] = (mu, se)
        say("  前 %d：规则 − 模型 %+.2fpp，NW se %.2f，t %.2f" % (k, 100 * mu, 100 * se, t))
for k in (10, 30):
    (m1, s1), (m2, s2) = diffs[("开发集", k)], diffs[("筛选集", k)]
    w1, w2 = 1 / s1 ** 2, 1 / s2 ** 2
    m = (m1 * w1 + m2 * w2) / (w1 + w2); se = 1 / np.sqrt(w1 + w2)
    say("合并（逆方差加权，969 天）前 %d：规则 − 模型 %+.2fpp，se %.2f，t %.2f" % (k, 100 * m, 100 * se, m / se))
