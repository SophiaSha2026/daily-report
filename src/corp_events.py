"""
股东减持、定向增发：长期调整突破剔票用的两类公司事件。

用户 2026-09-28：「剔除近期有股东减持或者定向增发的股票（定增价格已经确定的除外）」。

数据全来自东财，2026-09-28 本机实测可达（这条线只在本机跑，runner 上没测）：
  公告列表  np-anotice-stock.eastmoney.com/api/security/ann
            按股票（stock_list）+ 大类（f_node）+ 日期区间查，每条带东财打的细类标签：
              持股变动 f_node=7：减持公告在 001002007004003「股东/实际控制人股份减持」，
                预披露、计划、进展、结果都在这一类
              融资公告 f_node=2：001002001002 开头是增发，按阶段分：
                001001 预案 / 001002 方案修订 / 001004 获准 / 005001 发行结果 /
                006001、006002 上市公告书 / 007001 提示性 / 007002 其他 / 007003 终止
  减持记录  datacenter-web.eastmoney.com 的 RPT_SHARE_HOLDER_INCREASE（DIRECTION=减持），
            交易所披露的实际减持，带公告日
  公告正文  np-cnotice-stock.eastmoney.com/api/content/ann，每页 5000 字，
            只在要判定增定价方式时拉

不用巨潮的 stock_hold_change_cninfo：它是**股本变动**表（定期报告、回购、限售股上市），
不是股东减持表，2026-09-28 实测 5649 行的「变动原因」里没有一条是减持。起涨预测以前
就是拿它剔「近 30 天减持」，一只都没剔过；2026-09-28 起改用这里的
recent_reduction_reasons（全市场拉一次，判据同 reduce_hits）。

判定一律按**公告日**：回看历史时只用那一天之前已经公告的，不偷看。

  减持  目标日往前 reduce_days 天内（含当天）有减持公告或减持记录 -> 剔。
        不算：公司卖回购的库存股（标题带「回购」），「未实施 / 未减持 / 不减持」这类
        说明没减的公告。
  定增  目标日往前 placement_days 天内有增发预案，到目标日为止：
          没有发行结果 / 上市公告书 / 发行情况报告书（发完了，价格早定了）
          没有终止 / 撤回 / 失效
        就是还在走流程。再看预案（最新一份修订稿）的定价方式：
          「定价基准日为发行期首日」= 竞价，价格没定 -> 剔
          「定价基准日为董事会决议公告日」= 锁价，价格已定 -> 不剔（用户说的例外）
          认不出 -> 按价格没定，剔
        发行股份购买资产也在东财的「增发」里：购买资产那部分锁价，配套募集资金竞价，
        正文里只要有竞价条款就按价格没定。

网络和判定分开：prefetch() 联网把缓存补齐，verdict() 只读缓存。拉不到的票 verdict
返回 unknown，由调用方决定（当天的清单按剔处理，宁可为空）。

缓存 data/pullback/corp/<代码>.json（gitignore，能重拉），定价方式 _pricing.json。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "pullback" / "corp"
log = logging.getLogger("corp")

ANN_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
TEXT_URL = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
REC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
WORKERS = 4

SEO = "001002001002"                         # 增发的细类都以它开头
PLAN = SEO + "001001"                        # 增发预案
REVISE = SEO + "001002"                      # 增发方案修订
DONE = {SEO + "005001", SEO + "006001", SEO + "006002"}   # 发行结果、上市公告书
STOP = SEO + "007003"                        # 增发终止
DONE_TITLE = re.compile(r"发行情况报告书|上市公告书|发行结果|竞价结果")
STOP_TITLE = re.compile(r"终止|撤回|失效")
# 真正的方案文件。东财把股东会决议、董事会决议也打成「增发预案」，那些只有表决结果，
# 没有定价条款；年度股东会例行的「授权董事会办理以简易程序向特定对象发行」也不是一笔
# 具体的定增（2026-09-28 实测 300249 / 920964 / 300560 被这两种情形误判）
PLAN_DOC = re.compile(r"预案|募集说明书|发行方案|定向发行说明书|调整.{0,30}方案")
NOT_PLAN = re.compile(r"提示性|授权董事会办理|提请股东(大)?会授权|不特定")
# 只认定向（对特定对象）发行。东财的「增发」里还混着北交所上市的公开发行方案
# （向不特定合格投资者公开发行），那不是定增（2026-09-28 实测 920057 / 920249）
PRIVATE = re.compile(r"向特定对象|非公开发行|定向发行|定向增发|发行股份(及支付现金)?购买资产"
                     r"|募集配套资金")
# 标题带「减持」但说的是没减：计划到期没实施、承诺不减持
NOT_REDUCE = re.compile(r"未实施|未减持|不减持|未发生|没有减持|无减持|暂不")
# 定价方式。正文里常是表格、换行，比对前先去掉所有空白
BID = re.compile(r"定价基准日(为|指|：|:)[^。；]{0,40}?发行期首日")
LOCK = re.compile(r"定价基准日(为|指|：|:)[^。；]{0,60}?(董事会|股东会|股东大会)"
                  r"[^。；]{0,40}?决议公告日")
# 直接写死一个价（北交所定向发行常见）：没有竞价条款时也算锁价
FIXED = re.compile(r"发行价格(为|确定为|：|:)?(人民币)?[0-9]+(\.[0-9]+)?元/股")
TEXT_PAGES = 8                               # 预案正文最多看 8 页（4 万字）

_tls = threading.local()
_price_lock = threading.Lock()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = requests.Session()
    return s


def _get(url: str, params: dict, tries: int = 3) -> dict:
    last: Exception | None = None
    for i in range(tries):
        try:
            r = _session().get(url, params=params, headers=UA, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"{url.split('/')[2]} 重试 {tries} 次仍失败：{last}")


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def _day(s: str, k: int) -> str:
    return (dt.date.fromisoformat(s) + dt.timedelta(days=k)).isoformat()


# ---------------------------------------------------------------------
#  拉取
# ---------------------------------------------------------------------
def fetch_ann(code: str, node: str, begin: str, end: str) -> list[dict]:
    """一只票某个大类在 [begin, end] 的全部公告，自动翻页。"""
    out, page = [], 1
    while True:
        d = (_get(ANN_URL, {"sr": "-1", "page_size": "100", "page_index": str(page),
                            "ann_type": "A", "client_source": "web", "f_node": node,
                            "s_node": "0", "stock_list": code,
                            "begin_time": begin, "end_time": end}).get("data") or {})
        rows = d.get("list") or []
        for it in rows:
            out.append({"art": it.get("art_code", ""), "date": str(it.get("notice_date", ""))[:10],
                        "title": it.get("title", ""), "node": node,
                        "cats": [c.get("column_code", "") for c in it.get("columns") or []]})
        if len(rows) < 100 or page * 100 >= int(d.get("total_hits") or 0) or page >= 50:
            return out
        page += 1


def fetch_ann_market(node: str, begin: str, end: str, max_pages: int = 200) -> list[dict]:
    """全市场某个大类在 [begin, end] 的公告，逐页拉。一条公告挂几只票就展开成几行（带 code）。"""
    out, page = [], 1
    while True:
        d = (_get(ANN_URL, {"sr": "-1", "page_size": "100", "page_index": str(page),
                            "ann_type": "A", "client_source": "web", "f_node": node,
                            "s_node": "0", "begin_time": begin, "end_time": end}).get("data") or {})
        rows = d.get("list") or []
        for it in rows:
            base = {"art": it.get("art_code", ""), "date": str(it.get("notice_date", ""))[:10],
                    "title": it.get("title", ""), "node": node,
                    "cats": [c.get("column_code", "") for c in it.get("columns") or []]}
            for x in it.get("codes") or []:
                if str(x.get("ann_type", "")).startswith("A") and x.get("stock_code"):
                    out.append({**base, "code": str(x["stock_code"]).zfill(6)})
        if len(rows) < 100 or page * 100 >= int(d.get("total_hits") or 0):
            return out
        if page >= max_pages:
            raise RuntimeError(f"全市场公告超过 {max_pages} 页还没拉完（{begin}~{end}），不返回残表")
        page += 1


def fetch_rec(code: str | None, since: str) -> list[dict]:
    """交易所披露的实际减持记录（公告日 >= since）。code=None 拉全市场，每条带 code。"""
    flt = (f'(SECURITY_CODE="{code}")' if code else "") + \
        f'(DIRECTION="减持")(NOTICE_DATE>=\'{since}\')'
    out, page = [], 1
    while True:
        j = _get(REC_URL, {"sortColumns": "NOTICE_DATE", "sortTypes": "-1", "pageSize": "500",
                           "pageNumber": str(page), "reportName": "RPT_SHARE_HOLDER_INCREASE",
                           "columns": "SECURITY_CODE,NOTICE_DATE,HOLDER_NAME,CHANGE_FREE_RATIO,"
                                      "START_DATE,END_DATE",
                           "source": "WEB", "client": "WEB", "filter": flt})
        res = j.get("result") or {}          # 查不到时东财给 result: null
        for r in res.get("data") or []:
            out.append({"code": str(r.get("SECURITY_CODE") or code or "").zfill(6),
                        "date": str(r.get("NOTICE_DATE", ""))[:10],
                        "holder": r.get("HOLDER_NAME") or "",
                        "free_ratio": r.get("CHANGE_FREE_RATIO"),
                        "start": str(r.get("START_DATE") or "")[:10],
                        "end": str(r.get("END_DATE") or "")[:10]})
        if page >= int(res.get("pages") or 0):
            return out
        page += 1


def fetch_pricing(art: str) -> str:
    """预案正文里的定价方式：bid（竞价，价格没定）/ lock（锁价，价格已定）/ unknown。"""
    seen_lock = False
    for page in range(1, TEXT_PAGES + 1):
        d = (_get(TEXT_URL, {"art_code": art, "client_source": "web",
                             "page_index": str(page)}).get("data") or {})
        txt = re.sub(r"\s+", "", d.get("notice_content") or "")
        if BID.search(txt):
            return "bid"
        seen_lock = seen_lock or bool(LOCK.search(txt) or FIXED.search(txt))
        if page >= int(d.get("page_size") or 1):
            break
    return "lock" if seen_lock else "unknown"


# ---------------------------------------------------------------------
#  缓存
# ---------------------------------------------------------------------
def _path(code: str) -> Path:
    return CACHE / f"{code}.json"


def load(code: str) -> dict | None:
    try:
        return json.loads(_path(code).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _save(code: str, c: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = _path(code).with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _path(code))


def _pricing_all() -> dict:
    try:
        return json.loads((CACHE / "_pricing.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _pricing_save(d: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "_pricing.json"
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, p)


def refresh(code: str, since: str, until: str, fresh_min: float | None = None) -> dict:
    """把一只票的缓存补到覆盖 [since, until]。

    已经覆盖就不联网：历史事件的判定日早就过去了，那天之前的公告不会再变。
    fresh_min 给了（当天的候选）：缓存比这么多分钟旧就把最近那一段重拉。
    """
    today = now_bj().date().isoformat()
    c = load(code)
    if c and c.get("from", "9") <= since:
        age = (time.time() - float(c.get("ts") or 0)) / 60
        if c.get("through", "") >= until and (fresh_min is None or age <= fresh_min):
            return c
        begin = _day(c["through"], -7)       # 重叠一周：晚到的公告、改过日期的公告
    else:
        c, begin = None, since
    ann = fetch_ann(code, "7", begin, today) + fetch_ann(code, "2", begin, today)
    rec = fetch_rec(code, begin)
    if c:
        keep = {a["art"]: a for a in c.get("ann", [])}
        keep.update({a["art"]: a for a in ann})
        ann = sorted(keep.values(), key=lambda a: (a["date"], a["art"]))
        seen = {(r["date"], r["holder"], r["start"], r["end"], r["free_ratio"])
                for r in c.get("rec", [])}
        rec = c.get("rec", []) + [r for r in rec if (r["date"], r["holder"], r["start"],
                                                    r["end"], r["free_ratio"]) not in seen]
    out = {"code": code, "from": since if not c else c["from"], "through": today,
           "ts": time.time(), "ann": ann, "rec": rec}
    _save(code, out)
    return out


# ---------------------------------------------------------------------
#  判定（只读缓存）
# ---------------------------------------------------------------------
def reduce_hits(c: dict, asof: str, days: int) -> list[str]:
    """[asof-days, asof] 里的减持证据，每条一句人话。"""
    lo = _day(asof, -days)
    out = []
    for a in c.get("ann", []):
        t = a.get("title", "")
        if (a.get("node") == "7" and lo <= a.get("date", "") <= asof and "减持" in t
                and "回购" not in t and not NOT_REDUCE.search(t)):
            out.append(f"{a['date'][5:]} 公告{_short(t)}")
    for r in c.get("rec", []):
        if lo <= r.get("date", "") <= asof:
            out.append(f"{r['date'][5:]} {r.get('holder') or '股东'}减持")
    return out


def _short(title: str) -> str:
    """「公司名:公司名关于持股5%以上股东减持股份预披露公告」-> 「持股5%以上股东减持股份预披露」"""
    t = re.sub(r"^[^:：]*[:：]", "", title)
    t = re.sub(r"^.*?关于", "", t)
    return re.sub(r"(的)?(公告|提示性公告)$", "", t)[:24]


def recent_reduction_reasons(codes, asof: str, days: int) -> dict[str, str]:
    """codes 里 [asof-days, asof] 有股东减持的：{代码: 一句原因}。

    起涨预测的风险剔除用：它一天要核几十上百只够格的票，全市场拉一次（减持公告十来页、
    减持记录一两页）比逐只查省得多，而且只数不影响请求数。判据就是 reduce_hits，和
    长期调整突破同一份（教训 34）。拉不到就抛，由调用方决定跳过还是停。
    """
    cs = {str(c).zfill(6) for c in codes}
    begin = _day(asof, -days)
    by: dict[str, dict] = {}
    for a in fetch_ann_market("7", begin, asof):
        if a["code"] in cs:
            by.setdefault(a["code"], {"ann": [], "rec": []})["ann"].append(a)
    for r in fetch_rec(None, begin):
        if r["code"] in cs:
            by.setdefault(r["code"], {"ann": [], "rec": []})["rec"].append(r)
    out = {}
    for c, d in by.items():
        hits = reduce_hits(d, asof, days)
        if hits:
            out[c] = hits[0] + (f" 等 {len(hits)} 条" if len(hits) > 1 else "")
    return out


def pending_plan(c: dict, asof: str, days: int) -> dict | None:
    """到 asof 为止还在走流程的定增（没发完、没终止），返回定价看哪份文件；没有就 None。

    本轮的起点取窗口里最后一份方案文件（预案、修订稿、募集说明书、调整方案）：
    之后有发行结果 / 上市公告书 / 竞价结果就是发完了（价格早定了），
    有终止 / 撤回 / 失效就是不做了。
    """
    seo = [a for a in c.get("ann", [])
           if a.get("node") == "2" and a.get("date", "") <= asof
           and any(str(x).startswith(SEO) for x in a.get("cats", []))]
    lo = _day(asof, -days)
    docs = [a for a in seo if a["date"] >= lo
            and (PLAN in a.get("cats", []) or REVISE in a.get("cats", []))
            and PLAN_DOC.search(a.get("title", "")) and PRIVATE.search(a.get("title", ""))
            and not NOT_PLAN.search(a.get("title", ""))]
    if not docs:
        return None
    start = max(a["date"] for a in docs)
    for a in seo:
        if a["date"] < start:
            continue
        if set(a.get("cats", [])) & DONE or DONE_TITLE.search(a.get("title", "")):
            return None                                   # 发完了：价格早定了
        if STOP in a.get("cats", []) or STOP_TITLE.search(a.get("title", "")):
            return None                                   # 终止 / 撤回 / 失效
    # 定价看最新的一份；同一天优先预案 / 募集说明书全文（「调整方案的公告」常只写改了
    # 哪几条，摘要里可能没有配套募集资金那段的竞价条款）
    doc = max(docs, key=lambda a: (a["date"], bool(re.search(r"预案|说明书", a["title"])),
                                   "摘要" not in a["title"], a["art"]))
    return {"start": start, "art": doc["art"], "title": doc["title"]}


def verdict(code: str, asof: str, reduce_days: int = 90,
            placement_days: int = 730) -> tuple[str, str]:
    """(ok / out / unknown, 原因)。只读缓存，不联网。"""
    c = load(code)
    if not c or c.get("from", "9") > _day(asof, -placement_days) or c.get("through", "") < asof:
        return "unknown", "减持 / 定增数据没拉到，没法核"
    hits = reduce_hits(c, asof, reduce_days)
    if hits:
        return "out", f"近 {reduce_days} 天有股东减持（{hits[0]}" + (
            f" 等 {len(hits)} 条）" if len(hits) > 1 else "）")
    p = pending_plan(c, asof, placement_days)
    if p:
        kind = _pricing_all().get(p["art"], {}).get("kind")
        if kind is None:
            return "unknown", "定增的定价方式没拉到，没法核"
        if kind != "lock":
            how = "竞价，价格没定" if kind == "bid" else "预案里认不出定价方式，按价格没定算"
            return "out", f"定增进行中（{p['start'][5:]} 预案，{how}）"
    return "ok", ""


def prefetch(items: list[tuple[str, str]], placement_days: int = 730,
             fresh: set[str] | None = None, fresh_min: float = 120) -> dict[str, str]:
    """把 (代码, 判定日) 需要的缓存补齐。返回 {代码: 错误}，拉成功的不在里面。

    fresh 里的代码（当天的候选）缓存超过 fresh_min 分钟就重拉最近那段。
    还在走流程的定增，顺手把预案正文的定价方式拉下来（按公告编号缓存，拉一次就够）。
    """
    fresh = fresh or set()
    need: dict[str, str] = {}
    until: dict[str, str] = {}
    for code, asof in items:
        s = _day(asof, -placement_days)
        need[code] = min(need.get(code, s), s)
        until[code] = max(until.get(code, asof), asof)
    errs: dict[str, str] = {}

    def one(code: str) -> None:
        try:
            refresh(code, need[code], until[code], fresh_min if code in fresh else None)
        except Exception as e:  # noqa: BLE001
            errs[code] = str(e)[:160]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(one, sorted(need)))
    # 定价方式：只看还在走流程的那几份预案
    pricing = _pricing_all()
    arts = set()
    for code, asof in items:
        c = load(code)
        p = pending_plan(c, asof, placement_days) if c else None
        if p and p["art"] not in pricing:
            arts.add(p["art"])
    got: dict[str, dict] = {}

    def price(art: str) -> None:
        try:
            k = fetch_pricing(art)
            with _price_lock:
                got[art] = {"kind": k, "at": now_bj().isoformat(timespec="seconds")}
        except Exception as e:  # noqa: BLE001
            log.warning("定增正文 %s 拉不到：%s", art, e)

    if arts:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(price, sorted(arts)))
        if got:
            pricing = _pricing_all()
            pricing.update(got)
            _pricing_save(pricing)
    if errs:
        log.warning("减持 / 定增数据有 %d 只拉不到：%s", len(errs),
                    "；".join(f"{k} {v}" for k, v in list(errs.items())[:3]))
    return errs
