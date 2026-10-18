"""巨潮「投资评级」按日全市场（akshare stock_rank_forecast_cninfo，webapi.cninfo p_sysapi1089），研究用，产物只在 raw/。

    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_cninfo.py

按自然日逐天拉 2021-09-01..2025-12-31（研报周末也发），单线程（py_mini_racer 不能多线程，教训 19），每天一个 parts 文件可续跑；
请求超时 20 秒（research/datasrc/common 给 requests 补的默认超时）、失败重试 3 次、连续失败 5 次全局停 10 分钟、连停 3 次放弃；
北京 15:55~18:30 不发请求。字段（2026-10-18 03:40 实测）：证券代码 证券简称 发布日期 研究机构简称 研究员名称 投资评级 是否首次评级
评级变化 前一次投资评级 目标价格-下限 目标价格-上限；2022-01-05 123 行、2024-03-05 145 行（同日东财 reportapi 全市场只给 28 行，不全）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "datasrc"))
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402

RAW = HERE / "raw"
BEGIN, END = "2021-09-01", "2025-12-31"
COLS = ["code", "name", "date", "org", "analyst", "rating", "first_flag", "rating_change", "prev_rating", "target_lo", "target_hi"]
log = common.logger(HERE / "fetch_cninfo.log")


def main():
    import akshare as ak
    d = RAW / "rating_cninfo" / "parts"
    d.mkdir(parents=True, exist_ok=True)
    days = [x.strftime("%Y-%m-%d") for x in pd.date_range(BEGIN, END, freq="D")]
    todo = [x for x in days if not (d / f"{x}.parquet").exists() and not (d / f"{x}.empty").exists()]
    log(f"== rating_cninfo: {len(todo)} 天待拉 / {len(days)}")
    br = common.Breaker(n=5, pause=600, max_trips=3, log=log)
    t0 = time.time()
    n_done = 0
    for day in todo:
        err = None
        for attempt in range(3):
            try:
                common.check_blackout()
                df = ak.stock_rank_forecast_cninfo(date=day.replace("-", ""))
                err = None
                break
            except common.Blackout:
                log("禁区，存好进度退出")
                merge()
                return
            except Exception as e:  # noqa: BLE001
                err = e
                msg = repr(e)
                if "records" in msg or "KeyError" in msg or "columns" in msg:
                    # 没数据的日子 akshare 自己会抛（records 为空 / 列数不对），当空
                    df = pd.DataFrame()
                    err = None
                    break
                time.sleep(2 * (attempt + 1))
        if err is not None:
            log(f"  {day} 失败：{err!r}"[:200])
            if not br.bad(err):
                break
            continue
        br.ok()
        if len(df):
            df.columns = COLS
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["date"] = df["date"].astype(str).str[:10]
            df.to_parquet(d / f"{day}.parquet", index=False)
        else:
            (d / f"{day}.empty").write_text("")
        n_done += 1
        if n_done % 25 == 0:
            log(f"  {day}: {len(df)} 行；进度 {n_done}/{len(todo)} {time.time() - t0:.0f}s")
        time.sleep(0.6)
    merge()


def merge():
    d = RAW / "rating_cninfo" / "parts"
    parts = sorted(d.glob("*.parquet"))
    if not parts:
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(RAW / "rating_cninfo.parquet", index=False)
    n_empty = len(list(d.glob("*.empty")))
    total = len(pd.date_range(BEGIN, END, freq="D"))
    msg = (f"rating_cninfo（巨潮 p_sysapi1089）合并 {len(df)} 行，date {df['date'].min()}..{df['date'].max()}，"
           f"天 {len(parts)} + 空 {n_empty} / {total}；可得时点：发布日期当天收盘起可用（研报多为盘前 / 盘后，保守按当天收盘）")
    log(msg)
    st = RAW / "status.json"
    status = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
    status["rating_cninfo"] = {"rows": len(df), "span": [df["date"].min(), df["date"].max()], "parts": len(parts) + n_empty,
                               "total_chunks": total, "at": common.now_bj().strftime("%Y-%m-%d %H:%M")}
    st.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] 拉取 %s\n" % (common.now_bj().strftime("%Y-%m-%d %H:%M"), msg))


if __name__ == "__main__":
    main()
