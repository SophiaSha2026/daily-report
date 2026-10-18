"""候选数据源单次实测（字段、单位、时点、历史起点、页数），每个源 1~3 个请求，结果追加到 probe_result.jsonl。

    PYTHONIOENCODING=utf-8 python research/deepdata/probe_dd.py [源名 ...]

全部请求带 30 秒超时；北京 15:55~18:30 不发（research/datasrc/common.check_blackout）。
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

UA = {"User-Agent": common.UA, "Referer": "https://data.eastmoney.com/"}
HOSTS = {"web": "https://datacenter-web.eastmoney.com/api/data/v1/get",
         "sec": "https://datacenter.eastmoney.com/securities/api/data/v1/get"}
OUT = HERE / "probe_result.jsonl"


def say(*a):
    print(*a, flush=True)


def rec(name: str, **kw):
    kw = {"name": name, "at": common.now_bj().strftime("%Y-%m-%d %H:%M"), **kw}
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False, default=str) + "\n")
    say(json.dumps(kw, ensure_ascii=False, default=str)[:1500])


def dc(report: str, flt: str = "", sort: str = "", size: int = 50, page: int = 1, host: str = "web", asc=True) -> dict:
    common.check_blackout()
    p = {"reportName": report, "columns": "ALL", "pageNumber": str(page), "pageSize": str(size),
         "source": "WEB", "client": "WEB"}
    if sort:
        p["sortColumns"] = sort
        p["sortTypes"] = ",".join(["1" if asc else "-1"] * len(sort.split(",")))
    if flt:
        p["filter"] = flt
    r = requests.get(HOSTS[host], params=p, headers=UA, timeout=30)
    time.sleep(0.5)
    j = r.json()
    if j.get("result") is None:
        return {"error": j.get("message"), "code": j.get("code")}
    res = j["result"]
    d = res["data"]
    return {"count": res.get("count"), "pages": res.get("pages"), "n": len(d),
            "cols": list(d[0].keys()) if d else [], "sample": d[:2]}


def probe_dc(name, report, datecol, lo, hi, sort=None, host="web", flt_extra="", size=50):
    try:
        flt = f"{flt_extra}({datecol}>='{lo}')({datecol}<='{hi}')"
        r = dc(report, flt, sort or datecol, size=size, host=host)
        rec(name, report=report, window=[lo, hi], **r)
        return r
    except Exception as e:  # noqa: BLE001
        rec(name, report=report, error=repr(e)[:300])
        return {"error": repr(e)}


def earliest(name, report, datecol, host="web"):
    """最早一条：按日期升序取第一页。"""
    try:
        r = dc(report, "", datecol, size=5, host=host, asc=True)
        first = r.get("sample", [{}])[0].get(datecol) if r.get("sample") else None
        rec(name + "_earliest", report=report, first=first, count=r.get("count"), error=r.get("error"))
    except Exception as e:  # noqa: BLE001
        rec(name + "_earliest", report=report, error=repr(e)[:300])


P = {}


def p_lhb_seats():
    for rp in ("RPT_BILLBOARD_DAILYDETAILSBUY", "RPT_BILLBOARD_DAILYDETAILSSELL"):
        probe_dc("lhb_" + rp[-4:].lower(), rp, "TRADE_DATE", "2024-03-01", "2024-03-31", sort="TRADE_DATE,SECURITY_CODE")
    probe_dc("lhb_dept_active", "RPT_OPERATEDEPT_ACTIVE", "ONLIST_DATE", "2024-03-01", "2024-03-31", sort="ONLIST_DATE")
    # 营业部交易明细（按营业部看买了哪些票）
    probe_dc("lhb_dept_trade", "RPT_OPERATEDEPT_TRADE_DETAILSNEW", "TRADE_DATE", "2024-03-01", "2024-03-31", sort="TRADE_DATE")


P["lhb_seats"] = p_lhb_seats


def p_forecast():
    probe_dc("yjyg", "RPT_PUBLIC_OP_NEWPREDICT", "NOTICE_DATE", "2024-01-01", "2024-01-31", sort="NOTICE_DATE,SECURITY_CODE", host="sec")
    probe_dc("yjyg_web", "RPT_PUBLIC_OP_NEWPREDICT", "NOTICE_DATE", "2024-01-01", "2024-01-31", sort="NOTICE_DATE,SECURITY_CODE", host="web")
    probe_dc("yjkb", "RPT_FCI_PERFORMANCEE", "UPDATE_DATE", "2024-01-01", "2024-03-31", sort="UPDATE_DATE,SECURITY_CODE", host="web")
    probe_dc("yjkb_sec", "RPT_FCI_PERFORMANCEE", "UPDATE_DATE", "2024-01-01", "2024-03-31", sort="UPDATE_DATE,SECURITY_CODE", host="sec")


P["forecast"] = p_forecast


def p_repurchase():
    probe_dc("repurchase", "RPTA_WEB_GETHGLIST_NEW", "NOTICEDATE", "2024-03-01", "2024-03-31", sort="NOTICEDATE")
    probe_dc("repurchase_sec", "RPTA_WEB_GETHGLIST_NEW", "NOTICEDATE", "2024-03-01", "2024-03-31", sort="NOTICEDATE", host="sec")


P["repurchase"] = p_repurchase


def p_bonus():
    probe_dc("bonus", "RPT_SHAREBONUS_DET", "NOTICE_DATE", "2024-03-01", "2024-04-30", sort="NOTICE_DATE,SECURITY_CODE")
    probe_dc("bonus_plan", "RPT_SHAREBONUS_DET", "PLAN_NOTICE_DATE", "2024-03-01", "2024-04-30", sort="PLAN_NOTICE_DATE")


P["bonus"] = p_bonus


def p_incentive():
    for rp in ("RPT_GQJL_BASIC", "RPT_GQJL_INFO", "RPT_EQUITYINCENTIVE", "RPT_GQJL_DETAIL", "RPT_F10_EH_EQUITYINCENTIVE"):
        try:
            r = dc(rp, "", "", size=5)
            rec("incentive_" + rp, **{k: r.get(k) for k in ("error", "count", "cols", "sample")})
        except Exception as e:  # noqa: BLE001
            rec("incentive_" + rp, error=repr(e)[:200])


P["incentive"] = p_incentive


def p_holders_inc():
    """股东增持（增持方向），RPT_SHARE_HOLDER_INCREASE 生产只用了减持。"""
    probe_dc("holder_inc", "RPT_SHARE_HOLDER_INCREASE", "NOTICE_DATE", "2024-03-01", "2024-03-31", sort="NOTICE_DATE", flt_extra='(DIRECTION="增持")')


P["holder_inc"] = p_holders_inc


def p_report():
    """研报按日期范围（全市场），reportapi。"""
    url = "https://reportapi.eastmoney.com/report/list"
    for (lo, hi) in (("2024-03-01", "2024-03-31"), ("2022-01-04", "2022-01-04"), ("2021-09-01", "2021-09-01")):
        try:
            common.check_blackout()
            p = {"industryCode": "*", "pageSize": "100", "pageNo": "1", "beginTime": lo, "endTime": hi, "qType": "0",
                 "fields": "", "orgCode": "", "rcode": "", "p": "1", "pageNum": "1", "_": str(int(time.time() * 1000))}
            r = requests.get(url, params=p, headers=UA, timeout=30)
            j = r.json()
            d = j.get("data", [])
            rec("report_range", window=[lo, hi], hits=j.get("hits"), size=j.get("size"), total_page=j.get("TotalPage"),
                n=len(d), cols=list(d[0].keys()) if d else [], sample=d[:2])
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            rec("report_range", window=[lo, hi], error=repr(e)[:300])


P["report"] = p_report


def p_sw():
    url = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
    try:
        common.check_blackout()
        r = requests.get(url, headers={"User-Agent": common.UA}, timeout=60)
        f = HERE / "raw" / "sw_class_hist.xls"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(r.content)
        df = pd.read_excel(f)
        rec("sw_hist", status=r.status_code, bytes=len(r.content), rows=len(df), cols=list(df.columns), sample=df.head(3).to_dict("records"))
    except Exception as e:  # noqa: BLE001
        rec("sw_hist", error=repr(e)[:300])


P["sw"] = p_sw


def p_ths_harden():
    for d in ("2023-10-25", "2023-11-01", "2024-09-30", "2022-06-01"):
        url = f"https://zx.10jqka.com.cn/event/api/getharden/date/{d}/"
        try:
            common.check_blackout()
            r = requests.get(url, headers={"User-Agent": common.UA, "Referer": "https://zx.10jqka.com.cn/"}, timeout=30)
            j = r.json()
            data = j.get("data") if isinstance(j, dict) else None
            n = len(data) if isinstance(data, list) else (len(data) if isinstance(data, dict) else None)
            rec("ths_harden", date=d, status=r.status_code, keys=list(j.keys()) if isinstance(j, dict) else None, n=n,
                sample=(data[:2] if isinstance(data, list) else (dict(list(data.items())[:2]) if isinstance(data, dict) else str(j)[:500])))
            time.sleep(0.8)
        except Exception as e:  # noqa: BLE001
            rec("ths_harden", date=d, error=repr(e)[:300])


P["ths_harden"] = p_ths_harden


def p_cninfo_rating():
    import akshare as ak
    for d in ("20220105", "20230105"):
        try:
            common.check_blackout()
            df = ak.stock_rank_forecast_cninfo(date=d)
            rec("cninfo_rating", date=d, rows=len(df), cols=list(df.columns), sample=df.head(2).to_dict("records"))
        except Exception as e:  # noqa: BLE001
            rec("cninfo_rating", date=d, error=repr(e)[:300])


P["cninfo_rating"] = p_cninfo_rating


def p_fundhold():
    """基金持股明细（季报），看有没有公告日列。"""
    probe_dc("fundhold", "RPT_MAINDATA_MAIN_POSITIONDETAILS", "REPORT_DATE", "2024-03-31", "2024-03-31", sort="SECURITY_CODE", size=5)
    probe_dc("freeholders", "RPT_F10_EH_FREEHOLDERS", "END_DATE", "2024-03-31", "2024-03-31", sort="SECURITY_CODE", size=5)


P["fundhold"] = p_fundhold


def p_limitup_ths():
    for d in ("20251010", "20250901"):
        url = "https://data.10jqka.com.cn/dataapi/limit_up/limit_up_pool"
        try:
            common.check_blackout()
            r = requests.get(url, params={"page": 1, "limit": 15, "field": "199112,10,9001,330323,330324,330325,9002,330329,133971,133970,1968584,3475914,9003,9004", "filter": "HS,GEM2STAR", "order_field": "330324", "order_type": 0, "date": d}, headers={"User-Agent": common.UA, "Referer": "https://data.10jqka.com.cn/"}, timeout=30)
            j = r.json()
            info = j.get("data", {}).get("info", []) if isinstance(j, dict) else []
            rec("ths_limitup", date=d, status=r.status_code, n=len(info), total=j.get("data", {}).get("page", {}), sample=info[:1])
            time.sleep(0.8)
        except Exception as e:  # noqa: BLE001
            rec("ths_limitup", date=d, error=repr(e)[:300])


P["limitup_ths"] = p_limitup_ths


if __name__ == "__main__":
    names = sys.argv[1:] or list(P)
    for n in names:
        say("==", n)
        P[n]()
