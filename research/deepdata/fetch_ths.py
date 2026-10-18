"""同花顺「强势股题材」按日（zx.10jqka.com.cn/event/api/getharden/date/<日>/），每只涨停票当天的题材原因文本。研究用，产物只在 raw/。

    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_ths.py

历史起点约 2023-10 下旬（docs/oss_survey_2026-10-16.md；2026-10-18 03:40 实测 2023-10-25 72 行、2022-06-01 0 行），
所以只覆盖开发集 + 筛选集尾巴（2023-10-25..2024-05）。按交易日逐天拉 2023-10-01..2025-12-31，单线程、每请求后歇 0.8 秒、
20 秒超时、重试 3 次、连续失败 5 次停 10 分钟、北京 15:55~18:30 不发请求。
字段（实测）：id name code reason date close zhangdie zhangfu huanshou chengjiaoe chengjiaoliang ddejingliang market。
时点：收盘后快照（给的是当天涨停 / 大涨的票），reason 文本是否事后改写未核（只能当「当时的题材标签」近似）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research" / "datasrc"))
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402

RAW = HERE / "raw"
BEGIN, END = "2023-10-01", "2025-12-31"
log = common.logger(HERE / "fetch_ths.log")
H = {"User-Agent": common.UA, "Referer": "https://zx.10jqka.com.cn/"}


def main():
    d = RAW / "ths_harden" / "parts"
    d.mkdir(parents=True, exist_ok=True)
    days = sorted(pd.read_parquet(ROOT / "data" / "breakout" / "raw" / "exp" / "polish" / "pool.parquet", columns=["date"])["date"].unique())
    days = [x for x in days if BEGIN <= x <= END]
    todo = [x for x in days if not (d / f"{x}.parquet").exists() and not (d / f"{x}.empty").exists()]
    log(f"== ths_harden: {len(todo)} 天待拉 / {len(days)}")
    br = common.Breaker(n=5, pause=600, max_trips=3, log=log)
    t0 = time.time()
    n = 0
    for day in todo:
        err = None
        rows = None
        for attempt in range(3):
            try:
                common.check_blackout()
                r = requests.get(f"https://zx.10jqka.com.cn/event/api/getharden/date/{day}/", headers=H, timeout=20)
                j = r.json()
                data = j.get("data")
                rows = data if isinstance(data, list) else []
                err = None
                break
            except common.Blackout:
                log("禁区，存好进度退出")
                merge()
                return
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 * (attempt + 1))
        if err is not None:
            log(f"  {day} 失败：{err!r}"[:200])
            if not br.bad(err):
                break
            continue
        br.ok()
        if rows:
            df = pd.DataFrame(rows)
            df["code"] = df["code"].astype(str).str.zfill(6)
            df.to_parquet(d / f"{day}.parquet", index=False)
        else:
            (d / f"{day}.empty").write_text("")
        n += 1
        if n % 25 == 0:
            log(f"  {day}: {len(rows)} 行；进度 {n}/{len(todo)} {time.time() - t0:.0f}s")
        time.sleep(0.8)
    merge()


def merge():
    d = RAW / "ths_harden" / "parts"
    parts = sorted(d.glob("*.parquet"))
    if not parts:
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = df["date"].astype(str).str[:10]
    df.to_parquet(RAW / "ths_harden.parquet", index=False)
    n_empty = len(list(d.glob("*.empty")))
    msg = (f"ths_harden（同花顺强势股题材）合并 {len(df)} 行，date {df['date'].min()}..{df['date'].max()}，天 {len(parts)} + 空 {n_empty}；"
           f"可得时点：收盘后快照，reason 是否事后改写未核")
    log(msg)
    st = RAW / "status.json"
    status = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
    status["ths_harden"] = {"rows": len(df), "span": [df["date"].min(), df["date"].max()], "parts": len(parts) + n_empty,
                            "at": common.now_bj().strftime("%Y-%m-%d %H:%M")}
    st.write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] 拉取 %s\n" % (common.now_bj().strftime("%Y-%m-%d %H:%M"), msg))


if __name__ == "__main__":
    main()
