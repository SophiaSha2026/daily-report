"""抢救：东财 datacenter 里「全市场、只留近几十天」的表，按页整表拉下来（2026-10-15）。

  RPT_DMSK_TS_STOCKEVALUATE   千股千评-主力控盘（机构参与度 + 超大单 / 主力流入流出），约 40 个交易日
  RPT_STOCK_MARKETFOCUS       千股千评-用户关注指数 + 当日全市场名次，约 30 个交易日
  RPT_STOCK_HISTORYMARK       千股千评-综合评分历史，约 30 个交易日
  RPT_STOCK_PARTICIPATION     千股千评-市场参与意愿，约 5 个交易日
  RPT_DMSK_TS_STOCKNEW        千股千评当日快照（全市场一行一只）

archive/dc/<表>.parquet，原始列。每页 500 行，经 common.EMGate，页间 sleep。
页号进度 rescue_dc.progress.json，可续跑；整表拉完才合并。
"""
import json
import time

import pandas as pd
import requests

import common

TABLES = {
    "RPT_DMSK_TS_STOCKEVALUATE": "TRADE_DATE",
    "RPT_STOCK_MARKETFOCUS": "TRADE_DATE",
    "RPT_STOCK_HISTORYMARK": "DIAGNOSE_DATE",
    "RPT_STOCK_PARTICIPATION": "TRADE_DATE",
    "RPT_DMSK_TS_STOCKNEW": "SECURITY_CODE",
}
OUT = common.ARCHIVE / "dc"
log = common.logger(common.ROOT / "rescue_dc.log")


def page(rep, sort, n):
    r = requests.get("https://datacenter-web.eastmoney.com/api/data/v1/get", params={
        "reportName": rep, "columns": "ALL", "pageNumber": str(n), "pageSize": "500",
        "sortColumns": f"{sort},SECURITY_CODE" if sort != "SECURITY_CODE" else sort, "sortTypes": "1,1" if sort != "SECURITY_CODE" else "1",
        "source": "WEB", "client": "WEB"}, headers={"User-Agent": common.UA}, timeout=30)
    j = r.json()
    if not j.get("result"):
        raise RuntimeError(f"{j.get('message')} {j.get('code')}")
    return j["result"]["data"], j["result"]["pages"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    br = common.Breaker(log=log)
    for rep, sort in TABLES.items():
        if (OUT / f"{rep}.parquet").exists():
            continue
        tmp = OUT / f"{rep}.parts.jsonl"
        got = sum(1 for _ in open(tmp, encoding="utf-8")) if tmp.exists() else 0
        n, pages = got + 1, None
        while pages is None or n <= pages:
            try:
                with common.EMGate((0.8, 1.5)):
                    rows, pages = page(rep, sort, n)
                br.ok()
            except common.Blackout:
                raise
            except Exception as e:  # noqa: BLE001
                log(rep, n, "失败", repr(e)[:150])
                if not br.bad(e):
                    return
                continue
            with open(tmp, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rows, ensure_ascii=False) + "\n")
            if n % 50 == 0 or n == pages:
                log(rep, f"{n}/{pages}")
            n += 1
        rows = [r for line in open(tmp, encoding="utf-8") for r in json.loads(line)]
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / f"{rep}.parquet", index=False)
        tmp.unlink()
        d = df[sort].astype(str).str[:10] if sort != "SECURITY_CODE" else None
        log(rep, "完成", len(df), "行", df["SECURITY_CODE"].nunique(), "只",
            "" if d is None else f"{d.min()} ~ {d.max()} 共 {d.nunique()} 天")
        time.sleep(2)


if __name__ == "__main__":
    main()
