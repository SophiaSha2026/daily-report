"""
全市场公告（东财 np-anotice-stock，f_node=0 全部大类）按自然日逐页拉，缓存到
research/announce/raw/em/YYYY-MM-DD.json.gz。研究用，不进生产。

纪律（任务约定）：
  - 最多 2 路并发，每个请求之后 sleep 0.5 秒
  - 连续失败 5 次全局停 10 分钟（教训 20：限流是累积配额）
  - 北京时间 2026-10-15 16:00 ~ 18:30 不发请求（生产流程在跑），到窗口就睡到 18:30
实测（2026-10-14 晚）：page_size 上限 100；同一天翻页稳定（两次拉 1569 条 art_code 完全一致）；
单次查询 total_hits 封顶 50000，所以按天切（年报截止日 2025-04-29 也只有 29973 条）。
每条留：art_code、notice_date、eiTime（东财入库时间，北京时间，和巨潮 announcementTime 同一分钟）、
title、columns（细类 code + name）、codes（stock_code + ann_type）。

用法：python research/announce/fetch_em.py 2024-04-01 2025-12-31
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "raw" / "em"
URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
WORKERS = 2
SLEEP = 0.5
BLACKOUT = (dt.datetime(2026, 10, 15, 16, 0), dt.datetime(2026, 10, 15, 18, 30))

_lock = threading.Lock()
_fail = {"n": 0}
_tls = threading.local()


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=8)


def guard() -> None:
    while True:
        n = now_bj()
        if BLACKOUT[0] - dt.timedelta(minutes=2) <= n < BLACKOUT[1]:
            print(f"[{n:%H:%M}] 生产窗口，睡到 18:30", flush=True)
            time.sleep(60)
            continue
        return


def get(params: dict) -> dict:
    s = getattr(_tls, "s", None) or requests.Session()
    _tls.s = s
    while True:
        guard()
        try:
            r = s.get(URL, params=params, headers=UA, timeout=25)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            d = r.json().get("data")
            if d is None:
                raise RuntimeError("data 为空")
            with _lock:
                _fail["n"] = 0
            time.sleep(SLEEP)
            return d
        except Exception as e:  # noqa: BLE001
            with _lock:
                _fail["n"] += 1
                n = _fail["n"]
            print(f"  失败 {n}：{e}", flush=True)
            if n >= 5:
                print(f"[{now_bj():%H:%M}] 连续失败 {n} 次，全局停 10 分钟", flush=True)
                time.sleep(600)
                with _lock:
                    _fail["n"] = 0
            else:
                time.sleep(3 * n)


def slim(it: dict) -> dict:
    return {"art": it.get("art_code", ""), "nd": str(it.get("notice_date", ""))[:10],
            "ei": str(it.get("eiTime", ""))[:19], "title": it.get("title", ""),
            "cols": [[c.get("column_code", ""), c.get("column_name", "")] for c in it.get("columns") or []],
            "codes": [[x.get("stock_code", ""), x.get("ann_type", "")] for x in it.get("codes") or []]}


def one_day(day: str) -> tuple[str, int, int]:
    for i in range(6):
        try:
            return _one_day(day)
        except OSError as e:
            print(f"  {day} 本机 OSError {e}，10 秒后重试", flush=True)
            time.sleep(10)
    return day, -2, -2


def _one_day(day: str) -> tuple[str, int, int]:
    f = OUT / f"{day}.json.gz"
    if f.exists():
        return day, -1, -1
    rows, page, hits = [], 1, 0
    while True:
        d = get({"sr": "-1", "page_size": "100", "page_index": str(page), "ann_type": "A",
                 "client_source": "web", "f_node": "0", "s_node": "0",
                 "begin_time": day, "end_time": day})
        lst = d.get("list") or []
        hits = int(d.get("total_hits") or 0)
        rows += [slim(x) for x in lst]
        if len(lst) < 100 or page * 100 >= hits:
            break
        page += 1
    uniq = {r["art"]: r for r in rows}
    tmp = f.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as g:
        json.dump({"day": day, "hits": hits, "pages": page, "rows": list(uniq.values())}, g,
                  ensure_ascii=False)
    tmp.replace(f)
    return day, hits, len(uniq)


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    days, d = [], dt.date.fromisoformat(a)
    while d <= dt.date.fromisoformat(b):
        days.append(d.isoformat())
        d += dt.timedelta(days=1)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for day, hits, n in ex.map(one_day, days):
            done += 1
            if hits >= 0:
                print(f"[{now_bj():%m-%d %H:%M}] {day} hits {hits} 得 {n}  进度 {done}/{len(days)}  "
                      f"{(time.time() - t0) / 60:.0f} 分钟", flush=True)
    print("完成", len(days), "天", f"{(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
