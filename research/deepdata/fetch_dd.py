"""东财 datacenter 系列全市场分块拉取（研究用，产物只在 research/deepdata/raw/，gitignore）。

    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_dd.py lhb_sbuy lhb_ssell yjyg yjkb repurchase bonus incentive holder_inc
    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_dd.py report          研报按日（reportapi），约 1050 天
    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_dd.py freeholders     十大流通股东（按 NOTICE_DATE 月块）

纪律（和 research/altdata/fetch.py 同一套）：每个请求 30 秒超时；失败重试 3 次；全局熔断（连续失败 5 次停 10 分钟，连停 3 次放弃）；
并发 <= 3；北京 15:55~18:30 不发请求（research/datasrc/common.check_blackout）。
每个块落 raw/<表>/parts/<块>.parquet，跑过的不重拉；拉完合并成 raw/<表>.parquet，删后验列（D1~D30、RISE_PROBABILITY_3DAY 等）。
字段名 / 过滤写法在 probe_dd.py 实测过（probe_result.jsonl，2026-10-18 03:40 北京）。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "datasrc"))
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402

RAW = HERE / "raw"
BEGIN, END = "2021-09-01", "2025-12-31"
UA = {"User-Agent": common.UA, "Referer": "https://data.eastmoney.com/"}
HOSTS = {"web": "https://datacenter-web.eastmoney.com/api/data/v1/get",
         "sec": "https://datacenter.eastmoney.com/securities/api/data/v1/get"}
WORKERS = int(os.environ.get("DD_WORKERS", "3"))
LOOKAHEAD = re.compile(r"^(D\d+_CLOSE_ADJCHRATE|BD\d+_CLOSE_ADJCHRATE|CHANGE_RATE_\d+DAYS|RCHANGE\d+DCP|Y1_CLOSE_ADJCHRATE|INTERVAL_CHRATE|"
                       r"RISE_PROBABILITY_3DAY|TOTAL_BUYER_SALESTIMES_3DAY|NEWEST_PRICE|CHANGE_RATE_QUOTES|NEWPRICE|CPRICE)$")

# 表名: 报表, 日期列, 排序列, 额外过滤, 分块, 页大小, 可得时点口径
TABLES = {
    "lhb_sbuy": dict(report="RPT_BILLBOARD_DAILYDETAILSBUY", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                     chunk="month", size=500, avail="TRADE_DATE 当天收盘后交易所公布，T 日收盘可用"),
    "lhb_ssell": dict(report="RPT_BILLBOARD_DAILYDETAILSSELL", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                      chunk="month", size=500, avail="同上"),
    "yjyg": dict(report="RPT_PUBLIC_OP_NEWPREDICT", datecol="NOTICE_DATE", sort="NOTICE_DATE,SECURITY_CODE", flt="",
                 chunk="month", size=500, avail="业绩预告，按 NOTICE_DATE（沪深晚间公告记次日），公告日当天收盘起可用", hosts=("web", "sec")),
    "yjkb": dict(report="RPT_FCI_PERFORMANCEE", datecol="UPDATE_DATE", sort="UPDATE_DATE,SECURITY_CODE", flt="",
                 chunk="month", size=500, avail="业绩快报，按 NOTICE_DATE（UPDATE_DATE 分块），公告日当天收盘起可用"),
    "repurchase": dict(report="RPTA_WEB_GETHGLIST_NEW", datecol="NOTICEDATE", sort="NOTICEDATE", flt="",
                       chunk="month", size=500, avail="回购，按 NOTICEDATE / REPURADVANCEDATE（预案日），公告日当天收盘起可用"),
    "bonus": dict(report="RPT_SHAREBONUS_DET", datecol="PLAN_NOTICE_DATE", sort="PLAN_NOTICE_DATE,SECURITY_CODE", flt="",
                  chunk="month", size=500, avail="分红送转，按 PLAN_NOTICE_DATE（预案公告日），公告日当天收盘起可用"),
    "incentive": dict(report="RPT_EQUITYINCENTIVE", datecol="NOTICE_DATE", sort="NOTICE_DATE,SECURITY_CODE", flt="",
                      chunk="all", size=500, avail="股权激励，按 NOTICE_DATE / PLAN_NOTICE_DATE，公告日当天收盘起可用"),
    "holder_inc": dict(report="RPT_SHARE_HOLDER_INCREASE", datecol="NOTICE_DATE", sort="NOTICE_DATE,SECURITY_CODE", flt='(DIRECTION="增持")',
                       chunk="month", size=500, avail="股东增持，按 NOTICE_DATE，公告日当天收盘起可用"),
    "freeholders": dict(report="RPT_F10_EH_FREEHOLDERS", datecol="NOTICE_DATE", sort="NOTICE_DATE,SECURITY_CODE", flt="",
                        chunk="month", size=500, avail="十大流通股东，按 NOTICE_DATE（定期报告公告日），公告日当天收盘起可用"),
    "report": dict(report="reportapi", datecol="publishDate", chunk="day", size=100, avail="研报，按 publishDate，当天收盘起可用（研报多在盘前 / 盘后发布，保守按当天收盘）"),
}

_lock = threading.Lock()
_breaker = common.Breaker(n=5, pause=600, max_trips=3, log=lambda *a: log(*a))
_stop = threading.Event()


def log(*a):
    s = "[%s BJ] " % common.now_bj().strftime("%m-%d %H:%M:%S") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with _lock:
        with (HERE / "fetch.log").open("a", encoding="utf-8") as f:
            f.write(s + "\n")


def get(url: str, params: dict, kind: str = "dc") -> dict:
    """一次请求：禁区检查、30 秒超时、3 次重试、全局熔断。"""
    last = None
    for attempt in range(3):
        if _stop.is_set():
            raise RuntimeError("stopped")
        common.check_blackout()
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            j = r.json()
            time.sleep(random.uniform(0.3, 0.6))
            if kind == "report":
                if not isinstance(j, dict) or "data" not in j:
                    raise RuntimeError(f"reportapi: {str(j)[:100]}")
                with _lock:
                    _breaker.ok()
                return j
            if j.get("result") is None:
                msg = j.get("message", "")
                if "为空" in msg:
                    with _lock:
                        _breaker.ok()
                    return {"data": [], "count": 0, "pages": 0}
                raise RuntimeError(f"{params['reportName']}: {msg}")
            with _lock:
                _breaker.ok()
            return j["result"]
        except common.Blackout:
            _stop.set()
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    with _lock:
        if not _breaker.bad(last):
            _stop.set()
    raise RuntimeError(f"放弃：{last}")


def pull_dc(cfg: dict, lo: str, hi: str) -> pd.DataFrame:
    flt = cfg["flt"] + (f"({cfg['datecol']}>='{lo}')({cfg['datecol']}<='{hi}')" if cfg["chunk"] != "all" else "")
    n_sort = len(cfg["sort"].split(","))
    p = {"reportName": cfg["report"], "columns": "ALL", "pageNumber": "1", "pageSize": str(cfg["size"]),
         "sortColumns": cfg["sort"], "sortTypes": ",".join(["1"] * n_sort), "source": "WEB", "client": "WEB"}
    if flt:
        p["filter"] = flt
    hosts = cfg.get("hosts", ("web",))
    err = None
    for h in hosts:
        try:
            res = get(HOSTS[h], p)
            err = None
            break
        except RuntimeError as e:
            err = e
    if err is not None:
        raise err
    rows = list(res["data"])
    pages = int(res.get("pages") or 0)
    for pg in range(2, pages + 1):
        p["pageNumber"] = str(pg)
        res = get(HOSTS[hosts[0]], p)
        rows.extend(res["data"])
    return pd.DataFrame(rows)


def pull_report(day: str) -> pd.DataFrame:
    url = "https://reportapi.eastmoney.com/report/list"
    p = {"industryCode": "*", "pageSize": "100", "pageNo": "1", "beginTime": day, "endTime": day, "qType": "0",
         "fields": "", "orgCode": "", "rcode": "", "p": "1", "pageNum": "1"}
    j = get(url, p, kind="report")
    rows = list(j.get("data") or [])
    pages = int(j.get("TotalPage") or 0)
    for pg in range(2, pages + 1):
        p["pageNo"] = str(pg)
        p["p"] = p["pageNum"] = str(pg)
        j = get(url, p, kind="report")
        rows.extend(j.get("data") or [])
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop(columns=[c for c in ("encodeUrl", "attachType", "attachSize", "attachPages") if c in df.columns])
    return df


def chunks(cfg: dict) -> list[tuple[str, str, str]]:
    if cfg["chunk"] == "all":
        return [("all", BEGIN, END)]
    if cfg["chunk"] == "month":
        out = []
        for per in pd.period_range(BEGIN[:7], END[:7], freq="M"):
            s, e = per.start_time.strftime("%Y-%m-%d"), per.end_time.strftime("%Y-%m-%d")
            out.append((per.strftime("%Y-%m"), max(s, BEGIN), min(e, END)))
        return out
    days = sorted(pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "exp" / "polish" / "pool.parquet", columns=["date"])["date"].unique())
    return [(d, d, d) for d in days if BEGIN <= d <= END]


def run_table(name: str) -> None:
    cfg = TABLES[name]
    d = RAW / name / "parts"
    d.mkdir(parents=True, exist_ok=True)
    allc = chunks(cfg)
    todo = [(k, lo, hi) for k, lo, hi in allc if not (d / f"{k}.parquet").exists() and not (d / f"{k}.empty").exists()]
    log(f"== {name}: {len(todo)} 块待拉（共 {len(allc)} 块）")
    t0 = time.time()
    done = 0

    def work(k, lo, hi):
        df = pull_report(lo) if name == "report" else pull_dc(cfg, lo, hi)
        if len(df):
            df.to_parquet(d / f"{k}.parquet", index=False)
        else:
            (d / f"{k}.empty").write_text("")
        return k, len(df)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, *t) for t in todo]
        for f in as_completed(futs):
            try:
                k, n = f.result()
                done += 1
                if done % 10 == 0 or cfg["chunk"] != "day":
                    log(f"  {name} {k}: {n} 行；进度 {done}/{len(todo)} {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001
                log(f"  {name} 块失败：{e}")
                if _stop.is_set():
                    for g in futs:
                        g.cancel()
                    break
    merge(name)


def merge(name: str) -> None:
    cfg = TABLES[name]
    d = RAW / name / "parts"
    parts = sorted(d.glob("*.parquet"))
    n_empty = len(list(d.glob("*.empty")))
    total = len(chunks(cfg))
    if not parts:
        log(f"  {name}: 一块都没有")
        return
    frames = []
    drop: list[str] = []
    for p in parts:
        x = pd.read_parquet(p)
        bad = [c for c in x.columns if LOOKAHEAD.match(c)]
        drop = sorted(set(drop) | set(bad))
        frames.append(x.drop(columns=bad))
    df = pd.concat(frames, ignore_index=True)
    dc = cfg["datecol"]
    df[dc] = df[dc].astype(str).str[:10]
    df.to_parquet(RAW / f"{name}.parquet", index=False)
    msg = (f"{name}（{cfg['report']}）合并 {len(df)} 行，{dc} {df[dc].min()}..{df[dc].max()}，"
           f"块 {len(parts)} + 空 {n_empty} / {total}，删后验列 {drop}；可得时点：{cfg['avail']}")
    log(msg)
    st = RAW / "status.json"
    status = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
    status[name] = {"rows": len(df), "span": [df[dc].min(), df[dc].max()], "parts": len(parts) + n_empty, "total_chunks": total,
                    "dropped": drop, "avail": cfg["avail"], "at": common.now_bj().strftime("%Y-%m-%d %H:%M")}
    st.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] 拉取 %s\n" % (common.now_bj().strftime("%Y-%m-%d %H:%M"), msg))


if __name__ == "__main__":
    names = sys.argv[1:]
    if names == ["merge"]:
        for n in TABLES:
            if (RAW / n / "parts").exists():
                merge(n)
        sys.exit(0)
    for n in names:
        if _stop.is_set():
            break
        run_table(n)
