"""
每日存档器（计划 docs/plan_2026-10-15.md 第 3 步；用户 2026-10-15 拍板：单独第四个计划任务）。

只存「接口只给近期、以后拿不到」的数据，给半年后的特征实验攒历史（research/datasrc/report.md）：

    ztpool_<池>        东财 push2ex 涨停 / 昨日涨停 / 强势 / 炸板 / 跌停 / 次新六池（只给近 30 个自然日）
    comment_evaluate   千股千评-主力控盘（超大单 / 主力流入流出、机构参与度），按 TRADE_DATE 取目标日
    comment_focus      千股千评-关注指数和全市场名次，按 TRADE_DATE 取目标日
    comment_snap       千股千评当日快照（全市场一行一只），TRADE_DATE 必须是目标日

    python tools/archive_events.py                 存最近一个已收盘交易日
    python tools/archive_events.py --if-needed     计划任务用：每一项都存过就秒退
    python tools/archive_events.py --import-research   把 research/datasrc/archive 里抢救的那批并进来（一次性）

产物 data/events/<项>/YYYY-MM/<日>.parquet（gitignore，不入库：一年约 1 GB）；
状态 state/events_status.json（gitignore），控制台计划任务区块读它。

规矩（都是踩过的坑）：
- 北京 15:55~18:30 不发请求（两条线在取数发信，东财按路径限流，别和生产抢），直接退出 0
- 每个请求 30 秒超时，整次最多 20 分钟（教训 43）；连续失败 3 次就停，下一次触发再试（教训 20）
- 产物的日期对不上目标日就不算存过（教训 22 / 27）：快照还没更新到目标日、池子还是空的，都等下一次
- 进程锁 state/lock/events.json，后来者退出 0（教训 23）
- 失败只记进状态文件，不抛到计划任务（fail-open，发信和它无关）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bjtime import now_bj  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TIMEOUT = 30
MAX_RUN = 20 * 60
BLACKOUT = (dt.time(15, 55), dt.time(18, 30))
POOLS = {  # 名字: (接口, 排序, 交易日里能不能是空的)
    "zt": ("getTopicZTPool", "fbt:asc", False),
    "zt_prev": ("getYesterdayZTPool", "zs:desc", False),
    "strong": ("getTopicQSPool", "zdp:desc", False),
    "zb": ("getTopicZBPool", "fbt:asc", True),
    "dt": ("getTopicDTPool", "fund:asc", True),
    "subnew": ("getTopicCXPooll", "ods:asc", False),
}
DC = {  # 名字: (表, 是否按 TRADE_DATE 过滤)
    "comment_evaluate": ("RPT_DMSK_TS_STOCKEVALUATE", True),
    "comment_focus": ("RPT_STOCK_MARKETFOCUS", True),
    "comment_snap": ("RPT_DMSK_TS_STOCKNEW", False),
}
POOL_DAYS = 25      # 涨停池接口只给近 30 个自然日，目标日再早就不去要了


def out_dir() -> Path:
    return ROOT / "data" / "events"


def status_path() -> Path:
    # 函数不是常量：自测把 ROOT 换成临时目录时跟着走（教训 44）
    return ROOT / "state" / "events_status.json"


def items() -> list[str]:
    return [f"ztpool_{k}" for k in POOLS] + list(DC)


def item_path(name: str, day: str) -> Path:
    return out_dir() / name / day[:7] / f"{day}.parquet"


def in_blackout(now: dt.datetime | None = None) -> bool:
    t = (now or now_bj()).time()
    return BLACKOUT[0] <= t <= BLACKOUT[1]


def target_day(now: dt.datetime | None = None) -> str:
    import local_run as L
    from bjtime import last_closed
    return last_closed(L.trade_dates(), now or now_bj())


def write_atomic(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def fetch_pool(s, name: str, day: str):
    import pandas as pd
    ep, sort, _ = POOLS[name]
    r = s.get(f"https://push2ex.eastmoney.com/{ep}", timeout=TIMEOUT, params={
        "ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt", "Pageindex": "0",
        "pagesize": "10000", "sort": sort, "date": day.replace("-", "")})
    r.raise_for_status()
    df = pd.DataFrame(((r.json().get("data") or {}).get("pool")) or [])
    if len(df) and "zttj" in df:
        df["zt_days"] = df["zttj"].map(lambda x: (x or {}).get("days"))
        df["zt_ct"] = df["zttj"].map(lambda x: (x or {}).get("ct"))
        df = df.drop(columns="zttj")
    for c in df.columns:
        if df[c].map(lambda x: isinstance(x, (dict, list))).any():
            df[c] = df[c].map(lambda x: json.dumps(x, ensure_ascii=False))
    if len(df):
        df.insert(0, "date", day)
        df["c"] = df["c"].astype(str).str.zfill(6)
    return df


def fetch_dc(s, name: str, day: str):
    import pandas as pd
    rep, by_day = DC[name]
    rows, n, pages = [], 1, None
    while pages is None or n <= pages:
        p = {"reportName": rep, "columns": "ALL", "pageNumber": str(n), "pageSize": "500",
             "sortColumns": "SECURITY_CODE", "sortTypes": "1", "source": "WEB", "client": "WEB"}
        if by_day:
            p["filter"] = f"(TRADE_DATE='{day}')"
        j = s.get("https://datacenter-web.eastmoney.com/api/data/v1/get", params=p, timeout=TIMEOUT).json()
        res = j.get("result") or {}
        if not res:
            break
        rows += res.get("data") or []
        pages = int(res.get("pages") or 0)
        n += 1
        time.sleep(0.6)
    df = pd.DataFrame(rows)
    if len(df):
        d = df["TRADE_DATE"].astype(str).str[:10]
        if not (d == day).all():
            raise ValueError(f"{rep} 的日期是 {sorted(set(d))[:3]}，不是目标日 {day}（还没更新，等下一次）")
        df.insert(0, "date", day)
    return df


def fetch(s, name: str, day: str):
    if name.startswith("ztpool_"):
        return fetch_pool(s, name[len("ztpool_"):], day)
    return fetch_dc(s, name, day)


def complete(name: str, df) -> bool:
    """这一项算不算存到了：池子里不许空的空了、千股千评一行没有，都不算（下一次再试）。"""
    if name.startswith("ztpool_"):
        return len(df) > 0 or POOLS[name[len("ztpool_"):]][2]
    return len(df) > 1000


def load_status() -> dict:
    try:
        return json.loads(status_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_status(st: dict) -> None:
    p = status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def run(day: str, if_needed: bool, session=None, clock=time.time, now=now_bj) -> int:
    st = load_status()
    todo = [n for n in items() if not (if_needed and item_path(n, day).exists())]
    old = (dt.date.fromisoformat(now().date().isoformat()) - dt.date.fromisoformat(day)).days > POOL_DAYS
    if old:
        todo = [n for n in todo if not n.startswith("ztpool_")]
    if not todo:
        print(f"数据存档：目标日 {day} 每一项都存过了，跳过")
        return 0
    if session is None:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = UA
    t0, fails = clock(), 0
    for name in todo:
        if in_blackout(now()):
            print("到了北京 15:55~18:30（两条线在取数发信），停下，18:30 后的下一次触发接着存")
            break
        if clock() - t0 > MAX_RUN:
            print(f"超过 {MAX_RUN // 60} 分钟，停下")
            break
        try:
            df = fetch(session, name, day)
            if not complete(name, df):
                raise ValueError(f"只有 {len(df)} 行，像是还没出，等下一次")
            write_atomic(item_path(name, day), df)
            st[name] = {"date": day, "ok": True, "rows": int(len(df)), "at": now().isoformat(timespec="seconds")}
            fails = 0
            print(f"{name} {day} {len(df)} 行")
        except Exception as e:  # noqa: BLE001
            st[name] = {"date": day, "ok": False, "err": repr(e)[:200], "at": now().isoformat(timespec="seconds")}
            fails += 1
            print(f"{name} {day} 没存到：{repr(e)[:160]}")
            if fails >= 3:
                print("连续失败 3 次，这次先停（下一次触发再试）")
                save_status(st)
                break
        save_status(st)
        time.sleep(0.5)
    return 0


def import_research() -> int:
    """research/datasrc/archive 里 2026-10-15 抢救的那批并进 data/events（已有的不覆盖）。"""
    import pandas as pd
    src = ROOT / "research" / "datasrc" / "archive"
    n = 0
    for pool in POOLS:
        for f in sorted((src / "ztpool" / pool).glob("*.parquet")):
            dst = item_path(f"ztpool_{pool}", f.stem)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
                n += 1
    for name, (rep, _) in DC.items():
        f = src / "dc" / f"{rep}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["date"] = df["TRADE_DATE"].astype(str).str[:10]
        for day, g in df.groupby("date"):
            dst = item_path(name, day)
            if not dst.exists():
                write_atomic(dst, g[["date"] + [c for c in g.columns if c != "date"]])
                n += 1
    print(f"并进来 {n} 个文件")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-needed", action="store_true")
    ap.add_argument("--date", default="")
    ap.add_argument("--import-research", action="store_true")
    a = ap.parse_args()
    if a.import_research:
        return import_research()
    if in_blackout():
        print("北京 15:55~18:30 不存档（两条线在取数发信），退出")
        return 0
    import local_run as L
    try:
        with L.file_lock("events", timeout=5):
            day = a.date or target_day()
            return run(day, a.if_needed)
    except L.LockBusy:
        print("另一个存档器在跑，退出")
        return 0


if __name__ == "__main__":
    sys.exit(main())
