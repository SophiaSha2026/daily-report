"""东财 datacenter-web 全市场按日期分块拉取（研究用，产物只在 research/altdata/raw/，不入库）。

    PYTHONIOENCODING=utf-8 python research/altdata/fetch.py lhb_detail lhb_org dzjy gdhs          几分钟
    PYTHONIOENCODING=utf-8 python research/altdata/fetch.py rzrq                                  约 7300 页，放后台
    PYTHONIOENCODING=utf-8 python research/altdata/fetch.py survey pledge

纪律：每个请求 30 秒超时；失败重试 3 次；全局熔断（连续失败 5 次停 10 分钟，连停 3 次放弃，教训 20）；
并发 3 线程（任务书上限）；北京 15:55~18:30 不发请求（research/datasrc/common.check_blackout）。
每个日期块落 parts/<块>.parquet，跑过的不重拉（可续跑）；拉完合并成 raw/<表>.parquet，删后验列（D1~D30 等），
把行数 / 日期范围 / 可得时点口径追加到 progress.txt。

字段名、过滤写法、页大小上限都是 probe.py 实测过的（2026-10-17 22:40 北京）：
    两个排序列要 sortTypes "1,1"；除大宗交易（5000）外页大小上限 500；日期过滤 (COL>='YYYY-MM-DD')(COL<='YYYY-MM-DD')。
    RPT_ORG_SURVEYNEW 只有近期数据（整段过滤只回 4833 行且从 2025-10 起），不能用；RPT_ORG_SURVEY 22:45 起一直「服务器繁忙」，
    这里对它两个主机都试。
"""
from __future__ import annotations

import json
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
UA = {"User-Agent": common.UA}
HOSTS = {"web": "https://datacenter-web.eastmoney.com/api/data/v1/get",
         "sec": "https://datacenter.eastmoney.com/securities/api/data/v1/get"}
import os
WORKERS = int(os.environ.get("ALT_WORKERS", "3"))   # 全机东财并发合计不超过 3
LOOKAHEAD = re.compile(r"^(D\d+_CLOSE_ADJCHRATE|CHANGE_RATE_\d+DAYS|RCHANGE\d+DCP|Y1_CLOSE_ADJCHRATE|INTERVAL_CHRATE|EXPLAIN)$")

# 表名: 报表, 日期列, 排序列, 额外过滤, 分块, 页大小, 可得时点口径
TABLES = {
    "lhb_detail": dict(report="RPT_DAILYBILLBOARD_DETAILSNEW", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                       chunk="month", size=500, avail="TRADE_DATE 当天收盘后（交易所盘后公布），T 日收盘可用"),
    "lhb_org": dict(report="RPT_ORGANIZATION_TRADE_DETAILS", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                    chunk="month", size=500, avail="TRADE_DATE 当天收盘后，T 日收盘可用"),
    "dzjy": dict(report="RPT_DATA_BLOCKTRADE", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                 chunk="month", size=5000, avail="TRADE_DATE 当天收盘后，T 日收盘可用"),
    "gdhs": dict(report="RPT_HOLDERNUM_DET", datecol="HOLD_NOTICE_DATE", sort="HOLD_NOTICE_DATE,SECURITY_CODE", flt="",
                 chunk="month", size=500, avail="按公告日 HOLD_NOTICE_DATE（教训 25），公告日当天收盘起可用（沪深晚间公告记次日）"),
    "rzrq": dict(report="RPTA_WEB_RZRQ_GGMX", datecol="DATE", sort="DATE,SCODE", flt="",
                 chunk="day", size=500, avail="DATE 为交易日，交易所 T+1 早上发布：T 日收盘只能用 DATE <= T-1 的行"),
    "pledge": dict(report="RPT_CSDC_LIST", datecol="TRADE_DATE", sort="TRADE_DATE,SECURITY_CODE", flt="",
                   chunk="month", size=500, avail="中登周度（TRADE_DATE 为周五），下周一起可用（保守按 +3 天）"),
    "survey": dict(report="RPT_ORG_SURVEY", datecol="NOTICE_DATE", sort="NOTICE_DATE", flt='(IS_SOURCE="1")',
                   chunk="month", size=500, avail="按 NOTICE_DATE 对齐（不用接待日）；公告日当天收盘起可用", hosts=("sec", "web")),
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


def get(url: str, params: dict) -> dict:
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
            if j.get("result") is None:
                msg = j.get("message", "")
                if "为空" in msg:       # 这一块真没数据
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


def pull_chunk(name: str, cfg: dict, lo: str, hi: str) -> pd.DataFrame:
    flt = f"{cfg['flt']}({cfg['datecol']}>='{lo}')({cfg['datecol']}<='{hi}')"
    n_sort = len(cfg["sort"].split(","))
    p = {"reportName": cfg["report"], "columns": "ALL", "pageNumber": "1", "pageSize": str(cfg["size"]),
         "sortColumns": cfg["sort"], "sortTypes": ",".join(["1"] * n_sort), "source": "WEB", "client": "WEB", "filter": flt}
    rows = []
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
    rows.extend(res["data"])
    pages = int(res.get("pages") or 0)
    for pg in range(2, pages + 1):
        p["pageNumber"] = str(pg)
        res = get(HOSTS[hosts[0]], p)
        rows.extend(res["data"])
    df = pd.DataFrame(rows)
    return df


def chunks(cfg: dict) -> list[tuple[str, str, str]]:
    if cfg["chunk"] == "month":
        out = []
        for per in pd.period_range(BEGIN[:7], END[:7], freq="M"):
            s, e = per.start_time.strftime("%Y-%m-%d"), per.end_time.strftime("%Y-%m-%d")
            out.append((per.strftime("%Y-%m"), max(s, BEGIN), min(e, END)))
        return out
    # 交易日从 polish 的 pool.parquet 取（2020-12 起）；daily.parquet 只有三年，从它取会漏掉 2021-09..2022-12
    days = sorted(pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "exp" / "polish" / "pool.parquet", columns=["date"])["date"].unique())
    return [(d, d, d) for d in days if BEGIN <= d <= END]


def run_table(name: str) -> None:
    cfg = TABLES[name]
    d = RAW / name / "parts"
    d.mkdir(parents=True, exist_ok=True)
    todo = [(k, lo, hi) for k, lo, hi in chunks(cfg) if not (d / f"{k}.parquet").exists() and not (d / f"{k}.empty").exists()]
    log(f"== {name}: {len(todo)} 块待拉（共 {len(chunks(cfg))} 块）")
    t0 = time.time()
    done = 0
    nrows = 0

    def work(k, lo, hi):
        df = pull_chunk(name, cfg, lo, hi)
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
                nrows += n
                if done % 5 == 0 or cfg["chunk"] == "month":
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
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    drop = [c for c in df.columns if LOOKAHEAD.match(c)]
    df = df.drop(columns=drop)
    dc = cfg["datecol"]
    df[dc] = df[dc].astype(str).str[:10]
    df.to_parquet(RAW / f"{name}.parquet", index=False)
    msg = (f"{name}（{cfg['report']}）合并 {len(df)} 行，{dc} {df[dc].min()}..{df[dc].max()}，"
           f"块 {len(parts) + n_empty}/{total}（空 {n_empty}），删后验列 {drop}；可得时点：{cfg['avail']}")
    log("  " + msg)
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] 取数：%s\n" % (common.now_bj().strftime("%Y-%m-%d %H:%M"), msg))
    status_f = RAW / "status.json"
    st = json.loads(status_f.read_text(encoding="utf-8")) if status_f.exists() else {}
    st[name] = {"rows": int(len(df)), "span": [str(df[dc].min()), str(df[dc].max())], "parts": len(parts) + n_empty,
                "total_chunks": total, "dropped": drop, "avail": cfg["avail"], "at": common.now_bj().strftime("%Y-%m-%d %H:%M")}
    status_f.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    names = sys.argv[1:] or list(TABLES)
    for n in names:
        if _stop.is_set():
            log("已停（禁区或熔断），下次续跑")
            break
        try:
            run_table(n)
        except common.Blackout as e:
            log(str(e))
            break


if __name__ == "__main__":
    main()
