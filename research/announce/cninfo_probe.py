"""
巨潮 hisAnnouncement/query 可用性评估 + 和东财 eiTime 对时（研究用）。
拉指定自然日的全部公告（pageSize 30，实测上限），存 raw/cninfo/<日>.json，
再和 raw/em/ 同日及次日的东财记录按 (代码, 去前缀标题) 对上，比较巨潮 announcementTime 和东财 eiTime。
用法：python research/announce/cninfo_probe.py 2025-03-12
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
H = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"), "X-Requested-With": "XMLHttpRequest"}


def fetch(day: str) -> list[dict]:
    out, page = [], 1
    while True:
        j = requests.post(URL, data={"pageNum": str(page), "pageSize": "30", "column": "szse",
                                     "tabName": "fulltext", "plate": "", "stock": "", "searchkey": "",
                                     "secid": "", "category": "", "trade": "", "seDate": f"{day}~{day}",
                                     "sortName": "", "sortType": "", "isHLtitle": "true"},
                          headers=H, timeout=25).json()
        a = j.get("announcements") or []
        out += [{"code": x.get("secCode"), "title": x.get("announcementTitle"),
                 "ts": x.get("announcementTime"), "type": x.get("announcementType"),
                 "url": x.get("adjunctUrl")} for x in a]
        time.sleep(0.6)
        if not j.get("hasMore") or not a:
            return out
        page += 1


def norm(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t or "")
    t = re.sub(r"^[^:：]{1,20}[:：]", "", t)
    return re.sub(r"\s", "", t)


def main() -> None:
    day = sys.argv[1]
    d = HERE / "raw" / "cninfo"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{day}.json"
    t0 = time.time()
    if not f.exists():
        rows = fetch(day)
        f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    rows = json.loads(f.read_text(encoding="utf-8"))
    print(f"巨潮 {day}：{len(rows)} 条，{time.time() - t0:.0f} 秒")
    em = {}
    for k in (-1, 0, 1):
        g = HERE / "raw" / "em" / f"{(dt.date.fromisoformat(day) + dt.timedelta(days=k)).isoformat()}.json.gz"
        if g.exists():
            for r in json.load(gzip.open(g, "rt", encoding="utf-8"))["rows"]:
                for c, t in r["codes"]:
                    em[(c, norm(r["title"]))] = (r["nd"], r["ei"])
    diffs, miss = [], 0
    for r in rows:
        key = (r["code"], norm(r["title"]))
        if key not in em or not em[key][1]:
            miss += 1
            continue
        cts = dt.datetime.utcfromtimestamp(r["ts"] / 1000) + dt.timedelta(hours=8)
        ei = dt.datetime.fromisoformat(em[key][1])
        diffs.append((ei - cts).total_seconds() / 60)
    import numpy as np
    x = np.array(diffs)
    print(f"对上 {len(x)} 条，没对上 {miss} 条；东财 eiTime - 巨潮时间（分钟）："
          f"中位 {np.median(x):.1f}，5% {np.percentile(x, 5):.1f}，95% {np.percentile(x, 95):.1f}，"
          f"早于巨潮 10 分钟以上 {int((x < -10).sum())} 条，晚 60 分钟以上 {int((x > 60).sum())} 条")


if __name__ == "__main__":
    main()
