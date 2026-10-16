"""
一次性抢存同花顺两份带时点的涨停数据（docs/oss_survey_2026-10-16.md 第 1、2 条）。原样 JSON 落盘，不加工。

    limit_up   data.10jqka.com.cn/dataapi/limit_up/limit_up_pool   首封 / 末封时间、炸板次数、板型、几天几板、封单、原因、
                                                                  分时轨迹。只给最近约一年（2025-10-09 起），滚动窗口，每天丢一天
    harden     zx.10jqka.com.cn/event/api/getharden/date/<日>/     涨停票当日题材原因，约 2023-10 下旬起

产物 data/events/ths_<源>/YYYY-MM/<日>.json（data/events/ 在 gitignore），之后跑
`python tools/archive_events.py --import-research` 并成 parquet（和每日存档器同格式）。已存的日子跳过（跑过就不再跑），可反复运行。
每个请求间隔 0.6 秒；北京 15:55~18:30 不发请求（和 tools/archive_events.py 同一条规矩）。

    python research/datasrc/grab_ths.py limit_up 2025-09-25
    python research/datasrc/grab_ths.py harden 2023-10-01
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402
import datasource as ds  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
LU_FIELDS = "199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004"


def fetch_limit_up(day: str) -> dict:
    out, page = None, 1
    while True:
        p = {"page": page, "limit": 200, "field": LU_FIELDS, "filter": "HS,GEM2STAR",
             "order_field": "330324", "order_type": 0, "date": day.replace("-", "")}
        d = requests.get("https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool", params=p,
                         headers=UA, timeout=20).json()
        data = d.get("data") or {}
        if out is None:
            out = d
        else:
            out["data"]["info"].extend(data.get("info") or [])
        pg = data.get("page") or {}
        if page >= int(pg.get("count") or 1):
            return out
        page += 1
        time.sleep(0.6)


def fetch_harden(day: str) -> dict:
    return requests.get(f"https://zx.10jqka.com.cn/event/api/getharden/date/{day}/", headers=UA, timeout=20).json()


def n_rows(src: str, d: dict) -> int:
    if src == "limit_up":
        return len(((d.get("data") or {}).get("info")) or [])
    return len(d.get("data") or [])


def quiet() -> bool:
    t = bjtime.now_bj()
    return (t.hour, t.minute) >= (15, 55) and t.hour < 18 or (t.hour == 18 and t.minute < 30)


def main() -> int:
    src, start = sys.argv[1], sys.argv[2]
    fn = {"limit_up": fetch_limit_up, "harden": fetch_harden}[src]
    tds = ds.trade_dates()
    last = bjtime.last_closed(tds)
    days = sorted(d for d in tds if start <= d <= last)
    base = ROOT / "data" / "events" / f"ths_{src}"
    got = skip = empty = fail = 0
    for day in days:
        f = base / day[:7] / f"{day}.json"
        if f.exists() or f.with_suffix(".parquet").exists():      # 已被 archive_events --import-research 并成 parquet
            skip += 1
            continue
        if quiet():
            print("北京 15:55~18:30 不发请求，停在", day, flush=True)
            break
        try:
            d = fn(day)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(day, "失败", str(e)[:80], flush=True)
            time.sleep(3)
            continue
        n = n_rows(src, d)
        if n == 0:
            empty += 1                       # 窗口外 / 当天没数据：不落盘，下次还会再试
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            got += 1
        time.sleep(0.6)
    print("%s：交易日 %d，新存 %d，已有 %d，空 %d，失败 %d" % (src, len(days), got, skip, empty, fail), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
