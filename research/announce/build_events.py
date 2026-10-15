"""
把 raw/em/*.json.gz 展开成事件表 cache/events.parquet：一行 = 一条公告 × 一只 A 股 × 一个类别。

可知日（avail，交易日）两种口径：
  main  （任务约定，保守）：notice_date 当天或之后第一个交易日 t，且入库时间 eiTime 早于 t 的下一个交易日 09:15
        （eiTime 晚到的补录不能拿 notice_date 回填，取两者较晚的那个交易日）
  ei    （敏感性）：只看 eiTime：最后一个交易日 t，使 eiTime 早于 t 的下一个交易日 09:15。
        上交所晚上发、notice_date 记成次日的公告，在这个口径下算 t（当晚可知、t+1 开盘可买）
eiTime 缺失时两种口径都退回 notice_date。
另存 stats：总条数、A 股展开行、细类分布、notice_date 和 eiTime 的关系。
"""
from __future__ import annotations

import datetime as dt
import glob
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from categories import classify  # noqa: E402

ROOT = HERE.parents[1]
CACHE = HERE / "cache"


def calendar() -> list[str]:
    d = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet", columns=["date"])
    days = sorted(set(d["date"].astype(str)))
    return [x for x in days if "2024-01-01" <= x <= "2026-01-31"]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    cal_np = np.array(cal)
    files = sorted(glob.glob(str(HERE / "raw" / "em" / "*.json.gz")))
    recs, n_ann, n_hits = [], 0, 0
    col_cnt: Counter = Counter()
    rel = Counter()
    for f in files:
        j = json.load(gzip.open(f, "rt", encoding="utf-8"))
        n_hits += j["hits"]
        for r in j["rows"]:
            n_ann += 1
            codes = [c for c, t in r["codes"] if str(t).startswith("A") and len(str(c)) == 6]
            cols = [c for c, _ in r["cols"]]
            for c, nm in r["cols"]:
                col_cnt[(c, nm)] += 1
            if not codes:
                continue
            cats = classify(cols, r["title"])
            ei = r.get("ei") or ""
            if ei[:10] and r["nd"]:
                dd = (dt.date.fromisoformat(ei[:10]) - dt.date.fromisoformat(r["nd"])).days
                rel[max(-3, min(dd, 3))] += 1
            for c in codes:
                recs.append((r["art"], c, r["nd"], ei, r["title"], "|".join(cols), "|".join(cats)))
    ev = pd.DataFrame(recs, columns=["art", "code", "nd", "ei", "title", "cols", "cats"])
    ev = ev.drop_duplicates(["art", "code"])
    # 交易日映射
    nd_idx = np.searchsorted(cal_np, ev["nd"].to_numpy(), side="left")          # nd 当天或之后
    ei_ts = pd.to_datetime(ev["ei"].str[:19], errors="coerce")
    # 最后一个交易日 t，使 eiTime < (t 的下一个交易日 09:15)：
    # b = 第一个「09:15 晚于 eiTime」的交易日；t = b 的前一个交易日
    open_ts = pd.to_datetime(pd.Series(cal_np) + " 09:15:00").to_numpy()
    b = np.searchsorted(open_ts, ei_ts.to_numpy(), side="right")
    ei_idx = b - 1
    has_ei = ei_ts.notna().to_numpy()
    ev["avail_ei"] = np.where(has_ei, ei_idx, nd_idx)
    ev["avail_main"] = np.maximum(nd_idx, ev["avail_ei"].to_numpy())
    ev["avail_main_date"] = cal_np[np.clip(ev["avail_main"], 0, len(cal) - 1)]
    ev["shift"] = ev["avail_main"] - ev["avail_ei"]
    ev.to_parquet(CACHE / "events.parquet", index=False)
    pd.Series(cal).to_frame("date").to_parquet(CACHE / "calendar.parquet", index=False)
    stats = {"files": len(files), "total_hits": n_hits, "announcements": n_ann,
             "a_share_rows": int(len(ev)), "a_share_codes": int(ev["code"].nunique()),
             "ei_minus_nd_days": {str(k): v for k, v in sorted(rel.items())},
             "main_minus_ei_tradingdays": {str(k): int(v) for k, v in ev["shift"].value_counts().sort_index().items()},
             "columns_top": [[c, n, k] for (c, n), k in col_cnt.most_common(300)],
             "cats": {k: int(v) for k, v in Counter(x for s in ev["cats"] for x in s.split("|") if x).most_common()}}
    (CACHE / "events_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "columns_top"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
