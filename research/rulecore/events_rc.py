"""
规则核预登记判定用的公司事件表（减持 / 解禁 / 定增），全市场按日期拉一次，判定在本机按公告日做。

和生产 daily.risk_filter 同一份数据源、同一份判据：
    减持  corp_events.fetch_ann_market("7", ...) + fetch_rec(None, ...)，判据 corp_events.reduce_hits（30 天）
    解禁  akshare stock_restricted_release_detail_em，占比列由 daily.release_share_by_code 解析，未来 30 天合计 >= daily.RELEASE_MIN_SHARE
    定增  akshare stock_qbzf_em，发行日期在往前 60 天内
生产是「今天」查一次；这里把 2021-12..2026-01 整段拉下来，按 (代码, 日期) 判，不偷看：减持按公告日、
定增按发行日期、解禁按解禁日（排期是提前公告的）。

产物 data/breakout/raw/exp/rulecore/events/：
    reduce_ann.parquet   code date title            减持相关公告（node 7，含所有持股变动，判据在 judge 时套 reduce_hits）
    reduce_rec.parquet   code date holder ...       交易所减持记录
    release.parquet      code date share            解禁日 + 占解禁前流通市值比例（小数）
    placement.parquet    code date                  增发发行日期
    status.json          每张表的覆盖区间和拉取时间；拉不全的月份记在 missing 里

    PYTHONIOENCODING=utf-8 python research/rulecore/events_rc.py
可续跑：每个月的减持公告单独落 parts/，已有的不重拉。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "breakout"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import corp_events as CE  # noqa: E402

OUT = ROOT / "data" / "breakout" / "raw" / "exp" / "rulecore" / "events"
PARTS = OUT / "parts"
BEGIN, END = "2021-12-01", "2026-01-31"        # 筛选集 2022-01 往前 30 天起，到开发集末尾 + 30 天解禁窗口


def say(*a):
    print(*a, flush=True)


def month_edges(a: str, b: str):
    m = pd.period_range(a[:7], b[:7], freq="M")
    for p in m:
        s, e = p.start_time.strftime("%Y-%m-%d"), p.end_time.strftime("%Y-%m-%d")
        yield p.strftime("%Y-%m"), max(s, a), min(e, b)


def reduce_ann(status: dict) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    miss = []
    for m, s, e in month_edges(BEGIN, END):
        f = PARTS / f"ann7_{m}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        try:
            rows = CE.fetch_ann_market("7", s, e, max_pages=400)
        except Exception as ex:  # noqa: BLE001
            say(f"  {m} 减持公告拉不到：{ex}")
            miss.append(m)
            continue
        d = pd.DataFrame(rows, columns=["art", "date", "title", "node", "cats", "code"])
        d["cats"] = d["cats"].map(lambda x: ",".join(x) if isinstance(x, list) else str(x))
        d.to_parquet(f, index=False)
        say(f"  {m} 持股变动公告 {len(d)} 行 {time.time() - t0:.0f}s")
    parts = sorted(PARTS.glob("ann7_*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()
    if len(d):
        d = d.drop_duplicates(["art", "code"]).sort_values(["code", "date"]).reset_index(drop=True)
        d.to_parquet(OUT / "reduce_ann.parquet", index=False)
    status["reduce_ann"] = {"rows": int(len(d)), "months": len(parts), "missing": miss,
                            "span": [str(d["date"].min()), str(d["date"].max())] if len(d) else None}


def reduce_rec(status: dict) -> None:
    f = OUT / "reduce_rec.parquet"
    if f.exists():
        status["reduce_rec"] = {"rows": int(len(pd.read_parquet(f))), "cached": True}
        return
    t0 = time.time()
    rows = CE.fetch_rec(None, BEGIN)
    d = pd.DataFrame(rows, columns=["code", "date", "holder", "free_ratio", "start", "end"])
    d.to_parquet(f, index=False)
    status["reduce_rec"] = {"rows": int(len(d)), "span": [str(d["date"].min()), str(d["date"].max())] if len(d) else None}
    say(f"  减持记录 {len(d)} 行 {time.time() - t0:.0f}s")


def release(status: dict) -> None:
    import akshare as ak
    import daily as D
    f = OUT / "release.parquet"
    if f.exists():
        status["release"] = {"rows": int(len(pd.read_parquet(f))), "cached": True}
        return
    out, miss = [], []
    for m, s, e in month_edges(BEGIN, END):
        try:
            d = ak.stock_restricted_release_detail_em(start_date=s.replace("-", ""), end_date=e.replace("-", ""))
        except Exception as ex:  # noqa: BLE001
            say(f"  {m} 解禁表拉不到：{ex}")
            miss.append(m)
            continue
        if d is None or not len(d):
            continue
        cc = next((c for c in d.columns if "代码" in c), None)
        dc = next((c for c in d.columns if "解禁时间" in c), None)
        sc = next((c for c in d.columns if "占解禁前流通市值比例" in c), None)
        assert cc and dc and sc, list(d.columns)
        _ = D.release_share_by_code(d)        # 只为复用生产的列解析和单位告警，不用它的按票合计（这里按日期）
        out.append(pd.DataFrame({"code": d[cc].astype(str).str.zfill(6),
                                 "date": pd.to_datetime(d[dc]).dt.strftime("%Y-%m-%d"),
                                 "share": pd.to_numeric(d[sc], errors="coerce").fillna(0.0)}))
        time.sleep(0.5)
    d = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["code", "date", "share"])
    d.to_parquet(f, index=False)
    status["release"] = {"rows": int(len(d)), "missing": miss,
                         "span": [str(d["date"].min()), str(d["date"].max())] if len(d) else None}
    say(f"  解禁 {len(d)} 行，缺 {miss}")


def placement(status: dict) -> None:
    import akshare as ak
    f = OUT / "placement.parquet"
    if f.exists():
        status["placement"] = {"rows": int(len(pd.read_parquet(f))), "cached": True}
        return
    d = ak.stock_qbzf_em()
    cc = next((c for c in d.columns if "股票代码" in c), None)
    dc = next((c for c in d.columns if "发行日期" in c), None)
    assert cc and dc, list(d.columns)
    o = pd.DataFrame({"code": d[cc].astype(str).str.zfill(6),
                      "date": pd.to_datetime(d[dc], errors="coerce").dt.strftime("%Y-%m-%d")}).dropna()
    o.to_parquet(f, index=False)
    status["placement"] = {"rows": int(len(o)), "span": [str(o["date"].min()), str(o["date"].max())]}
    say(f"  增发 {len(o)} 行")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / "status.json"
    status = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    for name, fn in (("placement", placement), ("release", release), ("reduce_rec", reduce_rec), ("reduce_ann", reduce_ann)):
        say(f"== {name} ==")
        try:
            fn(status)
        except Exception as ex:  # noqa: BLE001
            say(f"  {name} 失败：{ex}")
            status[name] = {"error": str(ex)[:200]}
        status["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        sf.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    say(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
