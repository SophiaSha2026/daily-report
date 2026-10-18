"""单次请求实测各表字段名 / 过滤写法 / 页大小上限 / 区间总条数（写取数代码之前先核实）。

    PYTHONIOENCODING=utf-8 python research/altdata/probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "datasrc"))
import common  # noqa: E402  研究用禁区 / 超时

UA = {"User-Agent": common.UA}
URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def dc(report, sort, flt, size=500, page=1, asc=True):
    p = {"reportName": report, "columns": "ALL", "pageNumber": str(page), "pageSize": str(size),
         "sortColumns": sort, "sortTypes": ",".join(["1" if asc else "-1"] * len(sort.split(","))), "source": "WEB", "client": "WEB", "filter": flt}
    common.check_blackout()
    t0 = time.time()
    r = requests.get(URL, params=p, headers=UA, timeout=30)
    j = r.json()
    dt_ = time.time() - t0
    if not j.get("result"):
        return None, j.get("message"), dt_
    return j["result"], None, dt_


TESTS = [
    ("lhb_detail", "RPT_DAILYBILLBOARD_DETAILSNEW", "TRADE_DATE,SECURITY_CODE", "(TRADE_DATE>='2024-01-02')(TRADE_DATE<='2024-01-31')"),
    ("lhb_org", "RPT_ORGANIZATION_TRADE_DETAILS", "TRADE_DATE,SECURITY_CODE", "(TRADE_DATE>='2024-01-02')(TRADE_DATE<='2024-01-31')"),
    ("rzrq", "RPTA_WEB_RZRQ_GGMX", "DATE,SCODE", "(DATE>='2024-01-02')(DATE<='2024-01-03')"),
    ("survey", "RPT_ORG_SURVEY", "NOTICE_DATE,SECURITY_CODE", "(NOTICE_DATE>='2024-01-02')(NOTICE_DATE<='2024-01-31')"),
    ("survey_tj", "RPT_ORG_SURVEYNEW", "NOTICE_DATE,SECURITY_CODE", "(NOTICE_DATE>='2024-01-02')(NOTICE_DATE<='2024-01-31')"),
    ("dzjy", "RPT_DATA_BLOCKTRADE", "TRADE_DATE,SECURITY_CODE", "(TRADE_DATE>='2024-01-02')(TRADE_DATE<='2024-01-31')"),
    ("pledge", "RPT_CSDC_LIST", "TRADE_DATE,SECURITY_CODE", "(TRADE_DATE>='2024-01-02')(TRADE_DATE<='2024-01-31')"),
    ("gdhs_det", "RPT_HOLDERNUM_DET", "HOLD_NOTICE_DATE,SECURITY_CODE", "(HOLD_NOTICE_DATE>='2024-01-02')(HOLD_NOTICE_DATE<='2024-01-31')"),
]


def main():
    out = {}
    for name, rep, sort, flt in TESTS:
        res, err, dt_ = dc(rep, sort, flt, size=500)
        if res is None:
            print(f"{name}: 失败 {err} ({dt_:.1f}s)")
            out[name] = {"err": err}
            continue
        rows = res["data"]
        print(f"{name}: 区间总条数 {res.get('count')} 页数 {res.get('pages')} 本页 {len(rows)} 行 {dt_:.1f}s")
        print("   列:", list(rows[0].keys()) if rows else None)
        print("   首行:", json.dumps(rows[0], ensure_ascii=False)[:600] if rows else None)
        # 页大小上限
        big = {}
        for sz in (1000, 5000):
            time.sleep(0.8)
            r2, e2, d2 = dc(rep, sort, flt, size=sz)
            big[sz] = (len(r2["data"]) if r2 else f"err {e2}", round(d2, 1))
        print("   页大小 1000/5000 ->", big)
        out[name] = {"count": res.get("count"), "cols": list(rows[0].keys()) if rows else [], "big": {str(k): v for k, v in big.items()}}
        time.sleep(0.8)
    (HERE / "probe_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
