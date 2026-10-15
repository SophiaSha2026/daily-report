"""抢救：东财涨停板行情五个池（push2ex），接口只给近期，能拿到的每一天都存原始字段。

archive/ztpool/<池>/<YYYY-MM-DD>.parquet，原始 JSON 字段不改名（zttj 拆成 zt_days / zt_ct）。
价格字段原样是「元 × 1000」的整数（p / ztp / dtp 等），用的时候再除。
从最近交易日往回走，连续 8 个交易日空就停。跑过的日子不再跑（空的也记）。

python research/datasrc/rescue_ztpool.py
"""
import json

import pandas as pd
import requests

import common

POOLS = {
    "zt": ("getTopicZTPool", "fbt:asc"),        # 涨停池
    "zt_prev": ("getYesterdayZTPool", "zs:desc"),  # 昨日涨停池（今日表现）
    "strong": ("getTopicQSPool", "zdp:desc"),    # 强势股池
    "zb": ("getTopicZBPool", "fbt:asc"),         # 炸板池
    "dt": ("getTopicDTPool", "fund:asc"),        # 跌停池
    "subnew": ("getTopicCXPooll", "ods:asc"),    # 次新股池
}
OUT = common.ARCHIVE / "ztpool"
log = common.logger(common.ROOT / "rescue_ztpool.log")


def fetch(s, ep, sort, day):
    r = s.get(f"https://push2ex.eastmoney.com/{ep}", params={
        "ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt", "Pageindex": "0",
        "pagesize": "10000", "sort": sort, "date": day.replace("-", "")}, timeout=20)
    r.raise_for_status()
    j = r.json()
    pool = ((j.get("data") or {}).get("pool")) or []
    df = pd.DataFrame(pool)
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


def main():
    s = requests.Session()
    s.headers["User-Agent"] = common.UA
    prog = common.Progress(common.ROOT / "rescue_ztpool.progress.json")
    days = common.trade_days("2026-06-01")[::-1]
    br = common.Breaker(log=log)
    for name, (ep, sort) in POOLS.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        empty_run = 0
        for day in days:
            key = f"{name}/{day}"
            if key in prog.d["done"]:
                empty_run = empty_run + 1 if prog.d["done"][key] == 0 else 0
                if empty_run >= 8:
                    break
                continue
            try:
                with common.EMGate():
                    df = fetch(s, ep, sort, day)
                br.ok()
            except common.Blackout:
                raise
            except Exception as e:  # noqa: BLE001
                log(name, day, "失败", repr(e)[:200])
                if not br.bad(e):
                    break
                continue
            if len(df):
                df.to_parquet(OUT / name / f"{day}.parquet", index=False)
            prog.d["done"][key] = len(df)
            prog.save()
            log(name, day, len(df))
            empty_run = empty_run + 1 if len(df) == 0 else 0
            if empty_run >= 8:
                break
    got = {k: v for k, v in prog.d["done"].items() if v}
    log("完成", len(got), "个池日有数据")


if __name__ == "__main__":
    main()
