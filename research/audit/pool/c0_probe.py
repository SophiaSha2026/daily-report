"""审计 C0：联网探针（每个源 1 个请求，timeout 30s，顺序执行），把原始表存到 research/audit/pool/raw/ 供 C 脚本离线核。
北京 16:00~18:30 不跑。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

t = bjtime.now_bj()
hm = t.hour * 60 + t.minute
if t.weekday() < 5 and 16 * 60 <= hm < 18 * 60 + 30:
    raise SystemExit("禁区，不联网")
RAW = ROOT / "research/audit/pool/raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
log = []


def say(s):
    log.append(s)
    print(s, flush=True)


# 1 东财解禁表：一个过去的月（2025-03）看列和「实际解禁数量」对「解禁数量」
try:
    import akshare as ak
    import socket
    socket.setdefaulttimeout(30)
    d = ak.stock_restricted_release_detail_em(start_date="20250301", end_date="20250331")
    d.to_parquet(RAW / "release_2025-03.parquet", index=False)
    say("解禁 2025-03：%d 行，列 %s" % (len(d), list(d.columns)))
    time.sleep(1)
    d2 = ak.stock_restricted_release_detail_em(start_date="20261101", end_date="20261130")
    d2.to_parquet(RAW / "release_2026-11.parquet", index=False)
    say("解禁 2026-11（未来）：%d 行" % len(d2))
except Exception as e:  # noqa: BLE001
    say("解禁表失败：%s" % e)

# 2 东财全市场增发表：看 发行方式 / 发行日期 / 增发上市日期
try:
    time.sleep(1)
    q = ak.stock_qbzf_em()
    q.to_parquet(RAW / "qbzf.parquet", index=False)
    say("增发表：%d 行，列 %s" % (len(q), list(q.columns)))
except Exception as e:  # noqa: BLE001
    say("增发表失败：%s" % e)

# 3 巨潮全文检索：一个请求测可达性和字段（announcementTime 是真实发布时间戳）
try:
    r = requests.post("http://www.cninfo.com.cn/new/fulltextSearch/full",
                      params={"searchkey": "300829 减持", "sdate": "", "edate": "", "isfulltext": "false",
                              "sortName": "pubdate", "sortType": "desc", "pageNum": "1", "pageSize": "10"},
                      headers=UA, timeout=30)
    j = r.json()
    (RAW / "cninfo_probe.json").write_text(json.dumps(j, ensure_ascii=False)[:20000], encoding="utf-8")
    anns = j.get("announcements") or []
    say("巨潮全文检索 HTTP %d，%d 条；样例：%s" % (r.status_code, len(anns),
                                          [(a.get("secCode"), a.get("announcementTitle", "")[:30], pd.to_datetime(a.get("announcementTime"), unit="ms").strftime("%Y-%m-%d %H:%M")) for a in anns[:3]]))
except Exception as e:  # noqa: BLE001
    say("巨潮失败：%s" % e)

(RAW / "probe_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
