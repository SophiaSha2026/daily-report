"""
条件基准率的交叉格子（描述，不是清单规则）：板块 x 60 日波动十分位 x 20 日涨幅十分位。
看「不用模型、只按三个公开量切格子」的格子命中率能到多少、每天有几只。
    PYTHONIOENCODING=utf-8 python research/rethink/cells.py > research/rethink/cells.txt
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules2 import load_set  # noqa: E402
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
ST = {str(c).zfill(6) for c in json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8")).get("codes", [])}
T0 = time.time()
def say(*a): print(*a, flush=True)

for tag, r in [
    ("开发集", load_set(EXP / "leak" / "labels_open_snap.parquet", EXP / "leak" / "daily_snap.parquet", "2024-02-01",
                     [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")],
                     EXP / "feat_final_v1" / "all_base_s7.parquet", [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")])),
    ("筛选集", load_set(EXP / "screen" / "labels_screen.parquet", EXP / "screen" / "px_screen.parquet", "2021-09-01",
                     [("date", ">=", "2022-01-01"), ("date", "<=", "2024-05-31")],
                     EXP / "screen_base_v1" / "all_base_s7.parquet", None))]:
    r = r[~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"])].copy()
    px = pq.read_table(EXP / ("leak/daily_snap.parquet" if tag == "开发集" else "screen/px_screen.parquet"),
                       columns=["code", "date", "close"]).to_pandas()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"])
    lr = np.log(px["close"]).groupby(px["code"]).diff()
    px["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std())
    r = r.merge(px[["code", "date", "vol60"]], on=["code", "date"], how="left")
    r = r[np.isfinite(r["vol60"])]
    r["vq"] = r.groupby("date")["vol60"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    r["rq"] = r.groupby("date")["ret20"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    ndays = r["date"].nunique()
    say(f"\n== {tag}：{ndays} 天，基准 {100 * r['y'].mean():.2f}% ==")
    say("板块 x vol60 十分位（行=板块，列=0..9）：命中%")
    t = r.pivot_table(index="board", columns="vq", values="y", aggfunc="mean").reindex(["main", "chinext", "star", "bj"])
    say((100 * t).round(1).to_string())
    say("板块 x ret20 十分位：命中%")
    t = r.pivot_table(index="board", columns="rq", values="y", aggfunc="mean").reindex(["main", "chinext", "star", "bj"])
    say((100 * t).round(1).to_string())
    say("vol60 十分位 x ret20 十分位（全板块）：命中%（行 vq，列 rq）")
    t = r.pivot_table(index="vq", columns="rq", values="y", aggfunc="mean")
    say((100 * t).round(1).to_string())
    say("格子里的只数 / 天（vq=9 行）：", (r[r["vq"] == 9].groupby("rq").size() / ndays).round(1).to_dict())
    for cond, nm in [((r["vq"] == 9) & (r["rq"] == 9), "vol60 最高十分位 & ret20 最高十分位"),
                     ((r["vq"] >= 8) & (r["rq"] >= 8), "两者前两档"),
                     ((r["board"] == "bj") & (r["vq"] >= 8), "北交所 & vol60 前两档"),
                     ((r["board"] == "bj") & (r["rq"] >= 8), "北交所 & ret20 前两档"),
                     ((r["board"] != "main") & (r["vq"] == 9) & (r["rq"] == 9), "非主板 & 两者最高档")]:
        s = r[cond]
        h = s.groupby("date")["y"].mean()
        say("  %-32s 每天 %5.1f 只，命中 %.1f%%（按天均值 %.1f%%），之后 20 根收盘收益中位 %.1f%%，最大回撤中位 %.1f%%" % (
            nm, len(s) / ndays, 100 * s["y"].mean(), 100 * h.mean(), 100 * s["c20"].median(), 100 * s["dd20"].median()))
say("\n耗时 %.0f 秒" % (time.time() - T0))
