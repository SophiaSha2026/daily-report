"""数据源普查（2026-10-15 北京时间凌晨实测，美东家宽）。

每个探针实际拉一次，记：行数、列名、日期列的最早 / 最晚、耗时、报错。
结果追加到 survey/results.jsonl（同名探针后写的覆盖前面的），样本前 5 行存 survey/samples/<名>.csv。

  python research/datasrc/survey.py              跑全部
  python research/datasrc/survey.py lhb dzjy     只跑名字里带这些词的
东财探针一律经 common.EMGate（全机并发闸 + 请求间 sleep）。
"""
from __future__ import annotations

import json
import sys
import time
import traceback

import pandas as pd
import requests

import common

OUT = common.ROOT / "survey"
(OUT / "samples").mkdir(parents=True, exist_ok=True)
RES = OUT / "results.jsonl"
UA = {"User-Agent": common.UA}
PROBES: list = []


def probe(name, host, datecol=None, em=False):
    def deco(fn):
        PROBES.append((name, host, datecol, em, fn))
        return fn
    return deco


def dc(report, sort, asc=True, flt="", size=3, host="datacenter-web.eastmoney.com", path="/api/data/v1/get"):
    """东财 datacenter 通用查询：按日期列升 / 降序取一页，回 (df, 总条数)。"""
    p = {"reportName": report, "columns": "ALL", "pageNumber": "1", "pageSize": str(size),
         "sortColumns": sort, "sortTypes": "1" if asc else "-1", "source": "WEB", "client": "WEB"}
    if flt:
        p["filter"] = flt
    r = requests.get(f"https://{host}{path}", params=p, headers=UA, timeout=20)
    j = r.json()
    if not j.get("result"):
        raise RuntimeError(f"{report}: {j.get('message')} code={j.get('code')}")
    return pd.DataFrame(j["result"]["data"]), j["result"].get("count")


def dc_span(report, sort, flt="", host="datacenter-web.eastmoney.com", path="/api/data/v1/get"):
    a, n = dc(report, sort, True, flt, host=host, path=path)
    time.sleep(1.0)  # 外层已经拿着 EMGate，这里不能再拿（单路时会自锁）
    b, _ = dc(report, sort, False, flt, host=host, path=path)
    df = pd.concat([a, b], ignore_index=True)
    return df, f"总条数 {n}"


# ---------------- 东财 datacenter（带公告日 / 交易日的表） ----------------
DC = [
    # 名字, 报表, 日期列, 说明
    ("lhb_detail", "RPT_DAILYBILLBOARD_DETAILSNEW", "TRADE_DATE", "龙虎榜每日明细（上榜原因、买卖额）"),
    ("lhb_seat_buy", "RPT_BILLBOARD_DAILYDETAILSBUY", "TRADE_DATE", "龙虎榜买入席位明细（营业部 / 机构专用）"),
    ("lhb_org", "RPT_ORGANIZATION_TRADE_DETAILS", "TRADE_DATE", "龙虎榜机构买卖每日统计"),
    ("lhb_yyb_active", "RPT_OPERATEDEPT_ACTIVE", "ONLIST_DATE", "每日活跃营业部（游资）"),
    ("dzjy", "RPT_DATA_BLOCKTRADE", "TRADE_DATE", "大宗交易明细"),
    ("lift", "RPT_LIFT_STAGE", "FREE_DATE", "限售解禁（含未来日程）"),
    ("pledge_ratio", "RPT_CSDC_LIST", "TRADE_DATE", "股权质押比例（中登周度）"),
    ("repurchase", "RPTA_WEB_GETHGLIST_NEW", "UPD", "股票回购"),
    ("holder_change", "RPT_SHARE_HOLDER_INCREASE", "NOTICE_DATE", "股东增减持（带公告日）"),
    ("exec_hold", "RPT_EXECUTIVE_HOLD_DETAILS", "CHANGE_DATE", "董监高持股变动"),
    ("jgdy", "RPT_ORG_SURVEY", "NOTICE_DATE", "机构调研明细（接待日 + 公告日）"),
    ("jgdy_tj", "RPT_ORG_SURVEYNEW", "NOTICE_DATE", "机构调研统计"),
    ("comment_snapshot", "RPT_DMSK_TS_STOCKNEW", "SECURITY_CODE", "千股千评（全市场当日快照）"),
    ("comment_focus", "RPT_STOCK_MARKETFOCUS", "TRADE_DATE", "千股千评-用户关注指数（单只历史）"),
    ("north_hold", "RPT_MUTUAL_STOCK_NORTHSTA", "TRADE_DATE", "北向持股个股"),
    ("rzrq_stock", "RPTA_WEB_RZRQ_GGMX", "DATE", "两融个股明细（东财）"),
    ("holder_num", "RPT_HOLDERNUMLATEST", "END_DATE", "股东户数最新（参照，生产已用 gdhs）"),
]
for _n, _r, _d, _desc in DC:
    def _mk(r=_r, d=_d, n=_n):
        def f():
            flt = '(SECURITY_CODE="600000")' if n == "comment_focus" else ""
            if n == "comment_snapshot":
                df, cnt = dc(r, d, True, size=5)
                return df, f"总条数 {cnt}，只有当日快照"
            return dc_span(r, d, flt)
        return f
    probe(f"em_{_n}", "datacenter-web.eastmoney.com", _d, em=True)(_mk())

probe("em_yjyg", "datacenter.eastmoney.com", "NOTICE_DATE", em=True)(
    lambda: dc_span("RPT_PUBLIC_OP_NEWPREDICT", "NOTICE_DATE", host="datacenter.eastmoney.com",
                    path="/securities/api/data/v1/get"))
probe("em_yjkb", "datacenter.eastmoney.com", "UPDATE_DATE", em=True)(
    lambda: dc_span("RPT_FCI_PERFORMANCEE", "UPDATE_DATE", host="datacenter.eastmoney.com",
                    path="/securities/api/data/v1/get"))
probe("em_yjbb", "datacenter-web.eastmoney.com", "NOTICE_DATE", em=True)(
    lambda: dc_span("RPT_LICO_FN_CPD", "NOTICE_DATE"))


# ---------------- akshare：东财其它主机 ----------------
def ak():
    import akshare
    return akshare


@probe("ak_research_report_em_300829", "reportapi.eastmoney.com", "日期", em=True)
def _():
    return ak().stock_research_report_em(symbol="300829"), ""


@probe("ak_research_report_em_600519", "reportapi.eastmoney.com", "日期", em=True)
def _():
    return ak().stock_research_report_em(symbol="600519"), ""


@probe("ak_concept_name_em", "push2.eastmoney.com", None, em=True)
def _():
    return ak().stock_board_concept_name_em(), "概念列表"


@probe("ak_concept_cons_em", "push2.eastmoney.com", None, em=True)
def _():
    return ak().stock_board_concept_cons_em(symbol="BK0815"), "BK0815 成分（只有当前成分，没有历史进出）"


@probe("raw_concept_hist_em", "push2his.eastmoney.com", "date", em=True)
def _():
    r = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
        "secid": "90.BK0815", "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101", "fqt": "0", "beg": "0", "end": "20500101", "lmt": "100000",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b"}, headers=UA, timeout=20)
    k = r.json()["data"]["klines"]
    return pd.DataFrame([x.split(",") for x in k]).rename(columns={0: "date"}), "概念 BK0815 日 K（push2his kline 路径）"


@probe("raw_industry_hist_em", "push2his.eastmoney.com", "date", em=True)
def _():
    r = requests.get("https://push2his.eastmoney.com/api/qt/stock/kline/get", params={
        "secid": "90.BK0475", "fields1": "f1,f2,f3", "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101", "fqt": "0", "beg": "0", "end": "20500101", "lmt": "100000",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b"}, headers=UA, timeout=20)
    k = r.json()["data"]["klines"]
    return pd.DataFrame([x.split(",") for x in k]).rename(columns={0: "date"}), "行业 BK0475 日 K"


@probe("raw_fflow_today_push2", "push2.eastmoney.com", "date", em=True)
def _():
    r = requests.get("https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get", params={
        "lmt": "0", "klt": "101", "secid": "1.600000", "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5"}, headers=UA, timeout=20)
    k = r.json()["data"]["klines"]
    return pd.DataFrame([x.split(",") for x in k]).rename(columns={0: "date"}), "push2 同路径只给当天 1 根"


@probe("raw_fflow_clist_today", "push2.eastmoney.com", None, em=True)
def _():
    r = requests.get("https://push2.eastmoney.com/api/qt/clist/get", params={
        "pn": "1", "pz": "100", "po": "1", "np": "1", "fltt": "2", "invt": "2", "fid": "f62",
        "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
        "fields": "f12,f14,f2,f3,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f124",
        "ut": "b2884a393a59ad64002292a3e90d46a5"}, headers=UA, timeout=20)
    j = r.json()["data"]
    return pd.DataFrame(j["diff"]), f"全市场当日主力净流入排名，总数 {j['total']}，分页 100"


@probe("ak_sector_fund_flow_rank", "push2.eastmoney.com", None, em=True)
def _():
    return ak().stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流"), "当日快照"


@probe("ak_hot_rank_em", "emappdata.eastmoney.com", None, em=True)
def _():
    return ak().stock_hot_rank_em(), "人气榜前 100 当前快照"


@probe("ak_hot_up_em", "emappdata.eastmoney.com", None, em=True)
def _():
    return ak().stock_hot_up_em(), "飙升榜快照"


@probe("ak_hot_keyword_em", "emappdata.eastmoney.com", "时间", em=True)
def _():
    return ak().stock_hot_keyword_em(symbol="SZ300829"), "个股热门关键词（概念）"


@probe("ak_hot_rank_detail_realtime_em", "emappdata.eastmoney.com", "时间", em=True)
def _():
    return ak().stock_hot_rank_detail_realtime_em(symbol="SZ300829"), "当天分钟级排名"


@probe("raw_hot_rank_bj", "emappdata.eastmoney.com", "calcTime", em=True)
def _():
    out = []
    for sym in ("BJ920001", "BJ920099", "SZ920001"):
        r = requests.post("https://emappdata.eastmoney.com/stockrank/getHisList", json={
            "appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "",
            "srcSecurityCode": sym, "yearType": "5"}, timeout=20)
        d = pd.DataFrame(r.json().get("data") or [])
        d["sym"] = sym
        out.append(d)
        time.sleep(1)
    df = pd.concat(out)
    return df, "北交所人气榜：" + ", ".join(f"{s}={len(d)}" for s, d in zip(("BJ920001", "BJ920099", "SZ920001"), out))


@probe("ak_news_em", "search-api-web.eastmoney.com", "发布时间", em=True)
def _():
    return ak().stock_news_em(symbol="300829"), "个股新闻"


@probe("ak_info_global_em", "np-weblist.eastmoney.com", "发布时间", em=True)
def _():
    return ak().stock_info_global_em(), "东财全球快讯"


@probe("raw_em_notice_2015", "np-anotice-stock.eastmoney.com", "notice_date", em=True)
def _():
    r = requests.get("https://np-anotice-stock.eastmoney.com/api/security/ann", params={
        "sr": "-1", "page_size": "50", "page_index": "1", "ann_type": "SHA,CYB,SZA,BJA",
        "client_source": "web", "f_node": "0", "s_node": "0",
        "begin_time": "2015-03-02", "end_time": "2015-03-02"}, headers=UA, timeout=20)
    j = r.json()["data"]
    df = pd.DataFrame(j["list"])
    if len(df):
        df["cols"] = df["columns"].map(lambda x: "|".join(c.get("column_name", "") for c in x))
    return df, f"全市场公告按日期，2015-03-02 总数 {j.get('total_hits')}"


# ---------------- akshare：交易所 / 新浪 / 巨潮 / 其它 ----------------
@probe("ak_margin_detail_sse_recent", "query.sse.com.cn", "信用交易日期")
def _():
    return ak().stock_margin_detail_sse(date="20261014"), ""


@probe("ak_margin_detail_sse_2012", "query.sse.com.cn", "信用交易日期")
def _():
    return ak().stock_margin_detail_sse(date="20120105"), "沪市两融明细 2012 年"


@probe("ak_margin_detail_szse_recent", "www.szse.cn", None)
def _():
    return ak().stock_margin_detail_szse(date="20261014"), ""


@probe("ak_margin_detail_szse_2012", "www.szse.cn", None)
def _():
    return ak().stock_margin_detail_szse(date="20120105"), "深市两融明细 2012 年"


@probe("ak_margin_detail_bse", "www.bse.cn", None)
def _():
    return ak().stock_margin_detail_bse(date="20261014"), ""


@probe("ak_lhb_sina_2015", "vip.stock.finance.sina.com.cn", None)
def _():
    return ak().stock_lhb_detail_daily_sina(date="20150105"), "新浪龙虎榜 2015"


@probe("ak_news_cctv_2016", "tv.cctv.com", "date")
def _():
    return ak().news_cctv(date="20160203"), ""


@probe("ak_news_cctv_recent", "tv.cctv.com", "date")
def _():
    return ak().news_cctv(date="20261013"), ""


@probe("ak_irm_cninfo", "irm.cninfo.com.cn", "提问时间")
def _():
    return ak().stock_irm_cninfo(symbol="300829"), "互动易提问"


@probe("ak_sns_sseinfo", "sns.sseinfo.com", None)
def _():
    return ak().stock_sns_sseinfo(symbol="603119"), "上证 e 互动"


@probe("ak_hot_follow_xq", "xueqiu.com", None)
def _():
    return ak().stock_hot_follow_xq(symbol="最热门"), "雪球关注榜快照"


@probe("ak_hot_tweet_xq", "xueqiu.com", None)
def _():
    return ak().stock_hot_tweet_xq(symbol="最热门"), "雪球讨论榜快照"


@probe("ak_info_global_cls", "www.cls.cn", "发布日期")
def _():
    return ak().stock_info_global_cls(symbol="全部"), "财联社电报"


@probe("ak_cninfo_disclosure_2010", "www.cninfo.com.cn", "公告时间")
def _():
    return ak().stock_zh_a_disclosure_report_cninfo(symbol="000001", market="沪深京", start_date="20100101",
                                                    end_date="20100630"), "巨潮公告检索 000001 2010 上半年"


@probe("raw_cninfo_market_day", "www.cninfo.com.cn", "announcementTime")
def _():
    r = requests.post("http://www.cninfo.com.cn/new/hisAnnouncement/query", data={
        "pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext", "plate": "",
        "stock": "", "searchkey": "", "secid": "", "category": "category_gddh_szsh;",
        "trade": "", "seDate": "2016-03-01~2016-03-31", "sortName": "", "sortType": "", "isHfTitle": "true"},
        headers={**UA, "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"},
        timeout=20)
    j = r.json()
    df = pd.DataFrame(j.get("announcements") or [])
    if len(df):
        df["announcementTime"] = pd.to_datetime(df["announcementTime"], unit="ms")
    return df, f"巨潮全市场按分类 + 日期（股东大会类，2016-03），totalAnnouncement={j.get('totalAnnouncement')}"


@probe("ak_rank_forecast_cninfo", "webapi.cninfo.com.cn", None)
def _():
    return ak().stock_rank_forecast_cninfo(date="20230817"), "巨潮投资评级（按日）"


@probe("ak_board_concept_name_ths", "q.10jqka.com.cn", None)
def _():
    return ak().stock_board_concept_name_ths(), "同花顺概念列表"


@probe("raw_sse_lhb_public", "query.sse.com.cn", None)
def _():
    r = requests.get("https://query.sse.com.cn/infodisplay/showTradePublicFile.do", params={
        "isPagination": "false", "searchDate": "2016-03-01", "_": str(int(time.time() * 1000))},
        headers={**UA, "Referer": "https://www.sse.com.cn/"}, timeout=20)
    j = r.json()
    return pd.DataFrame({"fileContents": [str(j.get("fileContents"))[:2000]]}), \
        f"上交所交易公开信息 2016-03-01，keys={list(j)[:8]}"


# ---------------- 非 akshare ----------------
@probe("baostock_k", "baostock.com", "date")
def _():
    import baostock as bs
    bs.login()
    rs = bs.query_history_k_data_plus("sh.600000", "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST,peTTM",
                                      start_date="1990-01-01", end_date="2026-10-14", frequency="d", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    return pd.DataFrame(rows, columns=rs.fields), f"err={rs.error_code} {rs.error_msg}"


@probe("baostock_forecast", "baostock.com", "profitForcastExpPubDate")
def _():
    import baostock as bs
    bs.login()
    rs = bs.query_forecast_report("sz.300829", start_date="2010-01-01", end_date="2026-10-14")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    rs2 = bs.query_profit_data(code="sh.600000", year=2008, quarter=1)
    rows2 = []
    while rs2.error_code == "0" and rs2.next():
        rows2.append(rs2.get_row_data())
    bs.logout()
    return pd.DataFrame(rows, columns=rs.fields), f"业绩预告；季频利润 2008Q1 600000：{rows2[:1]} 字段 {rs2.fields}"


@probe("baostock_5min", "baostock.com", "date")
def _():
    import baostock as bs
    bs.login()
    rs = bs.query_history_k_data_plus("sz.300829", "date,time,open,close,volume", start_date="2015-01-01",
                                      end_date="2026-10-14", frequency="5", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    return pd.DataFrame(rows, columns=rs.fields), "5 分钟线"


@probe("mootdx_1min", "通达信行情主机", "datetime")
def _():
    from mootdx.quotes import Quotes
    c = Quotes.factory(market="std", timeout=10)
    out = []
    for off in (0, 800 * 10, 800 * 30, 800 * 60):
        try:
            d = c.bars(symbol="600000", frequency=8, start=off, offset=800)
            if d is not None and len(d):
                out.append(d.assign(start=off))
        except Exception:  # noqa: BLE001
            pass
    df = pd.concat(out) if out else pd.DataFrame()
    return df.reset_index(drop=True), "1 分钟线，按 start 往回翻页：" + ", ".join(
        f"start={o['start'].iloc[0]}:{o['datetime'].min()}" for o in out)


@probe("mootdx_finance", "通达信行情主机", None)
def _():
    from mootdx.quotes import Quotes
    c = Quotes.factory(market="std", timeout=10)
    df = c.finance(symbol="300829")
    return df, "最新一期财务快照（无历史）"


@probe("mootdx_f10", "通达信行情主机", None)
def _():
    from mootdx.quotes import Quotes
    c = Quotes.factory(market="std", timeout=10)
    cat = c.F10C(symbol="300829")
    names = list(cat)[:20] if isinstance(cat, dict) else str(cat)[:300]
    return pd.DataFrame({"f10": [str(names)]}), "F10 目录"


@probe("mootdx_affair", "通达信财务文件", "filename")
def _():
    from mootdx.affair import Affair
    files = Affair.files()
    df = pd.DataFrame(files)
    return df, "历史季度财务包 gpcw*.zip 清单（只看清单不下载）"


@probe("adata_market", "adata 多源", "trade_date")
def _():
    import adata
    return adata.stock.market.get_market(stock_code="300829", start_date="2015-01-01", k_type=1), ""


@probe("adata_hot", "adata 多源", None)
def _():
    import adata
    return adata.sentiment.hot.pop_rank_100_east(), "东财人气榜 100（快照）"


@probe("dolthub_investment_data", "www.dolthub.com", None)
def _():
    q = ("select * from final_a_stock_eod_price where symbol='SZ300829' order by tradedate desc limit 5")
    r = requests.get("https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master",
                     params={"q": q}, headers=UA, timeout=30)
    j = r.json()
    return pd.DataFrame(j.get("rows") or []), f"query_execution_status={j.get('query_execution_status')} {j.get('query_execution_message', '')[:150]}"



# ---------------- 第二批（修参数 / 补探针） ----------------
for _d in ("20261013",):
    probe(f"ak_margin_detail_sse_{_d}", "query.sse.com.cn", "信用交易日期")(lambda d=_d: (ak().stock_margin_detail_sse(date=d), "T+1 早上才发"))
    probe(f"ak_margin_detail_szse_{_d}", "www.szse.cn", None)(lambda d=_d: (ak().stock_margin_detail_szse(date=d), ""))
    probe(f"ak_margin_detail_bse_{_d}", "www.bse.cn", None)(lambda d=_d: (ak().stock_margin_detail_bse(date=d), ""))
probe("ak_margin_detail_szse_2010", "www.szse.cn", None)(lambda: (ak().stock_margin_detail_szse(date="20100401"), "深市两融起点附近"))
probe("ak_margin_detail_sse_2010", "query.sse.com.cn", "信用交易日期")(lambda: (ak().stock_margin_detail_sse(date="20100401"), "沪市两融起点附近"))
probe("ak_news_cctv_2018", "tv.cctv.com", "date")(lambda: (ak().news_cctv(date="20180105"), ""))
probe("ak_news_cctv_2021", "tv.cctv.com", "date")(lambda: (ak().news_cctv(date="20210105"), ""))
probe("ak_lhb_sina_recent", "vip.stock.finance.sina.com.cn", None)(lambda: (ak().stock_lhb_detail_daily_sina(date="20261014"), ""))
for _f in ("stock_comment_detail_zhpj_lspf_em", "stock_comment_detail_zlkp_jgcyd_em", "stock_comment_detail_scrd_desire_em"):
    probe("ak_" + _f, "datacenter-web.eastmoney.com", None, em=True)(lambda f=_f: (getattr(ak(), f)(symbol="600000"), "千股千评单只历史"))
probe("ak_hot_rank_latest_em", "emappdata.eastmoney.com", None, em=True)(lambda: (ak().stock_hot_rank_latest_em(symbol="SZ300829"), ""))
probe("ak_hot_rank_relate_em", "emappdata.eastmoney.com", "时间", em=True)(lambda: (ak().stock_hot_rank_relate_em(symbol="SZ300829"), ""))


@probe("raw_news_em", "search-api-web.eastmoney.com", "date", em=True)
def _():
    import json as _j
    cb = "jQuery3510875346244069884_1668256937995"
    param = {"uid": "", "keyword": "300829", "type": ["cmsArticleWebOld"], "client": "web", "clientType": "web",
             "clientVersion": "curr", "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                                                                     "pageIndex": 1, "pageSize": 100, "preTag": "", "postTag": ""}}}
    r = requests.get("https://search-api-web.eastmoney.com/search/jsonp", params={"cb": cb, "param": _j.dumps(param)},
                     headers=UA, timeout=20)
    t = r.text
    j = _j.loads(t[t.find("(") + 1:t.rfind(")")])
    df = pd.DataFrame(j["result"]["cmsArticleWebOld"])
    return df, "东财个股新闻搜索（akshare 的 stock_news_em 在 pandas 3 下报 ArrowInvalid，这里直连）"


# ---------------- 跑 ----------------
def run_one(name, host, datecol, em, fn):
    t = time.time()
    rec = {"name": name, "host": host, "at_bj": f"{common.now_bj():%Y-%m-%d %H:%M}"}
    try:
        if em:
            with common.EMGate():
                df, note = fn()
        else:
            common.check_blackout()
            df, note = fn()
        rec.update(ok=True, rows=len(df), cols=[str(c) for c in df.columns][:40], note=note)
        if datecol and datecol in df and len(df):
            s = df[datecol].astype(str).str[:19]
            s = s[s.str.match(r"^\d{4}-?\d{2}")]
            if len(s):
                rec.update(first=s.min(), last=s.max())
        df.head(5).to_csv(OUT / "samples" / f"{name}.csv", index=False, encoding="utf-8-sig")
    except common.Blackout:
        raise
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, err=repr(e)[:300], tb=traceback.format_exc()[-400:])
    rec["sec"] = round(time.time() - t, 2)
    with open(RES, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("name", "ok", "rows", "first", "last", "sec", "note", "err")},
                     ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:]
    for p in PROBES:
        if not keys or any(k in p[0] for k in keys):
            run_one(*p)
