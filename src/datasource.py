"""
数据源层：腾讯批量行情（竞价窗口主力） + 东财快照/日线（盘前）。

设计要点
--------
1. 竞价窗口只有 5 分钟，必须用「批量」接口。腾讯 qt.gtimg.cn 单次可查 ~60 只，
   500 只候选池 = 9 个请求，通常 <2 秒完成。
2. 东财 push2 接口用于盘前（时间宽裕），走 akshare 封装。
3. 所有网络调用带重试 + 超时 + 降级，任何单点失败不阻断主流程。

⚠️ 首次部署必须跑一次 smoke_test.yml 验证腾讯字段索引。
   腾讯返回字段顺序历史上调整过，本文件只使用 index <= 38 的低位字段
   （相对稳定），涨停价由昨收自行推算而非读取。
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:                       # 只给类型标注用，运行时不导入 pandas
    import pandas as pd                 # noqa: F401  （竞价 quick 阶段要快起）

log = logging.getLogger(__name__)

TX_URL = "https://qt.gtimg.cn/q={codes}"
TX_BATCH = 60
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

_LINE = re.compile(r'v_(?P<sym>[a-z]{2}\d{6})="(?P<body>[^"]*)"')

_SESSION = requests.Session()
_SESSION.mount("https://", requests.adapters.HTTPAdapter(
    pool_connections=16, pool_maxsize=16, max_retries=0))


@dataclass
class Quote:
    """一只票在某一时刻的快照。price 在 9:15-9:25 期间为虚拟撮合参考价。"""
    code: str            # 6位代码
    market: str          # sh / sz / bj
    name: str
    price: float         # 当前价 / 竞价撮合价
    prev_close: float    # 昨收（已含除权调整）
    open_: float         # 今开（9:25 后才有效）
    volume_hand: float   # 累计成交量（手）
    amount_wan: float    # 累计成交额（万元）
    ts: str = ""         # 数据源时间戳
    raw: list[str] = field(default_factory=list, repr=False)

    @property
    def symbol(self) -> str:
        return f"{self.market}{self.code}"

    @property
    def chg_pct(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close * 100.0

    @property
    def amount_yuan(self) -> float:
        return self.amount_wan * 1e4


def to_symbol(code: str) -> str:
    """6位代码 -> 带市场前缀。北交所 8/4 开头归 bj。"""
    c = str(code).zfill(6)
    if c[0] == "6":
        return "sh" + c
    if c[0] in ("0", "3"):
        return "sz" + c
    if c[0] in ("8", "4", "9"):
        return "bj" + c
    return "sh" + c


def limit_pct(code: str, name: str) -> float:
    """当日涨停幅度（百分比）。ST 5%，创业板/科创板 20%，北交所 30%，其余 10%。"""
    c = str(code).zfill(6)
    if "ST" in name.upper():
        return 5.0
    if c.startswith("688") or c.startswith("30"):
        return 20.0
    if c[0] in ("8", "4"):
        return 30.0
    return 10.0


def limit_price(prev_close: float, pct: float) -> float:
    """涨停价：昨收 × (1+pct)，四舍五入到分。"""
    return round(prev_close * (1 + pct / 100.0) + 1e-9, 2)


def _parse_tx_body(sym: str, body: str) -> Quote | None:
    f = body.split("~")
    if len(f) < 40:
        return None
    try:
        return Quote(
            code=f[2],
            market=sym[:2],
            name=f[1],
            price=float(f[3] or 0),
            prev_close=float(f[4] or 0),
            open_=float(f[5] or 0),
            volume_hand=float(f[6] or 0),
            amount_wan=float(f[37] or 0),
            ts=f[30] if len(f) > 30 else "",
            raw=f,
        )
    except (ValueError, IndexError):
        return None


def _one_batch(batch: list[str], timeout: float, retries: int) -> dict[str, Quote]:
    url = TX_URL.format(codes=",".join(batch))
    for attempt in range(retries + 1):
        try:
            r = _SESSION.get(url, headers=UA, timeout=timeout)
            r.encoding = "gbk"
            got = {}
            for m in _LINE.finditer(r.text):
                q = _parse_tx_body(m.group("sym"), m.group("body"))
                if q and q.prev_close > 0:
                    got[q.symbol] = q
            return got
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                log.warning("腾讯批量失败 batch=%s err=%s", batch[:1], e)
                return {}
            time.sleep(0.15 * (attempt + 1))
    return {}


def fetch_quotes(
    symbols: Sequence[str],
    *,
    timeout: float = 6.0,
    retries: int = 2,
    workers: int = 5,
) -> dict[str, Quote]:
    """
    批量拉取实时/竞价快照。返回 {symbol: Quote}。

    并发说明：GitHub runner 在美国，到腾讯单次往返约 0.6s，1600 只串行要 17s。
    5 路并发压到 4s 左右，为竞价窗口留足余量。并发再高会触发限流，别调。
    单批失败只丢该批，不影响其余。
    """
    batches = [list(symbols[i:i + TX_BATCH])
               for i in range(0, len(symbols), TX_BATCH)]
    out: dict[str, Quote] = {}
    if not batches:
        return out

    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        futs = [ex.submit(_one_batch, b, timeout, retries) for b in batches]
        for f in as_completed(futs):
            out.update(f.result())

    log.info("fetch_quotes: 请求 %d 只，返回 %d 只", len(symbols), len(out))
    return out


# ---------------------------------------------------------------------
#  盘前用（时间宽裕，走 akshare / 东财）
# ---------------------------------------------------------------------

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
def _sina_code_list(timeout: float = 8.0) -> list[str]:
    """
    新浪分页行情列表（node=hs_a，含北交所）。每页 100 只，约 56 页。

    加这一路的原因：东财的两个代码表接口在 GitHub runner 上是「时好时坏」，
    2026-08-24 那轮冒烟测试里两个都被 ConnectionReset 掐断（同一次运行里
    第二次调用又成功了）。新浪这条通道和交易日历同源，实测稳定。
    """
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "Market_Center.getHQNodeData?page={p}&num=100&sort=symbol&asc=1&node=hs_a")
    hdr = {**UA, "Referer": "https://finance.sina.com.cn"}

    def one(p: int) -> list[str]:
        r = _SESSION.get(url.format(p=p), headers=hdr, timeout=timeout)
        r.encoding = "gbk"
        txt = r.text.strip()
        if not txt or txt in ("null", "[]"):
            return []
        import json as _json
        return [str(d["code"]).zfill(6) for d in _json.loads(txt)]

    codes: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        page = 1
        while page <= 120:               # 安全阀，正常 56 页就到底
            futs = {ex.submit(one, p): p for p in range(page, page + 8)}
            got = 0
            for f in as_completed(futs):
                try:
                    part = f.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("新浪列表第 %d 页失败: %s", futs[f], e)
                    continue
                codes += part
                got += len(part)
            if got == 0:
                break
            page += 8
    return sorted(set(codes))


def load_code_list() -> list[str]:
    """
    全市场 6 位代码表。优先读仓库里的缓存（每周刷新一次），
    缓存缺失时才现场拉取。代码表变动极慢，缓存完全够用。
    """
    import pandas as pd
    p = _ROOT / "cache" / "codes.csv"
    if p.exists():
        codes = pd.read_csv(p, dtype=str)["code"].tolist()
        if len(codes) > 3000:
            return codes
        log.warning("代码表缓存过短(%d)，重新拉取", len(codes))
    return refresh_code_list()


def refresh_code_list() -> list[str]:
    """
    多源兜底拉取代码表，成功即写入缓存。

    每个源重试 2 次：东财那两个接口在 runner 上是间歇性拒绝，
    一次失败不代表不可用（实测同一次运行里第二次调用就成功了）。
    """
    import pandas as pd

    def _ak_codes(fn_name: str) -> list[str]:
        import akshare as ak
        df = getattr(ak, fn_name)()
        col = next(c for c in df.columns if c in ("code", "代码"))
        return sorted({str(x).zfill(6) for x in df[col]})

    sources = (
        ("sina_hs_a", _sina_code_list),
        ("stock_info_a_code_name", lambda: _ak_codes("stock_info_a_code_name")),
        ("stock_zh_a_spot_em", lambda: _ak_codes("stock_zh_a_spot_em")),
    )
    for name, fn in sources:
        for attempt in range(2):
            try:
                codes = fn()
                if len(codes) > 3000:
                    out = _ROOT / "cache"
                    out.mkdir(exist_ok=True)
                    pd.DataFrame({"code": codes}).to_csv(
                        out / "codes.csv", index=False)
                    log.info("代码表已刷新 (%s): %d 只", name, len(codes))
                    return codes
                log.warning("%s 仅返回 %d 只", name, len(codes))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("%s 第%d次失败: %s", name, attempt + 1, e)
                time.sleep(0.8)
    raise RuntimeError("代码表获取失败：所有数据源均不可用")


def spot_all() -> "pd.DataFrame":  # noqa: F821
    """
    全市场快照，盘前用于构建候选池。

    ⚠️ 实测：GitHub runner 访问东财 push2 的 clist/get 批量接口会被
    RemoteDisconnected 掐断（但东财的**单只**日线接口 push2his 正常）。
    所以这里改用腾讯批量接口 + 本地代码表，走的是已验证可用的通道。

    盘前 08:23 调用时腾讯返回的是上一交易日收盘状态：
        当前价 = T-1 收盘价      成交额 = T-1 全天成交额
        涨跌%  = T-1 涨跌幅      昨收   = T-2 收盘价
    正好是候选池需要的字段。
    """
    import pandas as pd
    codes = load_code_list()
    syms = [to_symbol(c) for c in codes]
    q = fetch_quotes(syms)
    if len(q) < len(syms) * 0.6:
        raise RuntimeError(f"全市场快照过少: {len(q)}/{len(syms)}")

    rows = [{
        "代码": v.code, "名称": v.name, "最新价": v.price,
        "涨跌幅": v.chg_pct, "成交额": v.amount_yuan,
        "换手率": _turnover(v), "总市值": float("nan"),
    } for v in q.values()]
    df = pd.DataFrame(rows)
    log.info("全市场快照(腾讯): %d 只", len(df))
    return df


def _turnover(q: Quote) -> float:
    """换手率，腾讯字段 38。取不到返回 0，不影响主流程。"""
    try:
        return float(q.raw[38])
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------
#  日线历史：东财为主，腾讯为辅，中间加熔断
# ---------------------------------------------------------------------
#  东财 stock_zh_a_hist（push2his）给的是真实成交额和官方复权口径，优先用。
#  它在 runner 上是间歇性拒绝，不是永久失效——2026-08-23 那轮通过（153 根 K 线），
#  2026-08-24 那轮失败。所以单只失败要重试，不要一次就判死。
#
#  但盘前 stage2 要对 1600 只逐个拉。如果东财整段时间不通，每只都耗满重试
#  再降级，1600 只跑不完 40 分钟的 job 超时。所以加熔断：
#      连续 _EM_TRIP 只都失败 -> 本次进程内暂时跳过东财，直接走腾讯
#      每隔 _EM_RETRY_AFTER 只回探一次，东财恢复就切回去
#
#  腾讯 K 线返回 [日期, 开盘, 收盘, 最高, 最低, 成交量(手)]，**没有成交额**。
#  成交额按 收盘×成交量×100 估算——它只喂给 amount_ratio_5d 这个展示字段，
#  不进打分。竞价用的真实成交额来自腾讯实时快照，不是这里。
# ---------------------------------------------------------------------

# 2026-09-02：用 4 路并发连打 800 只之后，web.ifzq.gtimg.cn 的
# /appstock/app/fqkline/get 开始整片返回 HTTP 501（JS 挑战页），
# 而同一时刻 qt.gtimg.cn 批量行情、以及**去掉 web. 前缀**的裸主机
# ifzq.gtimg.cn 同一路径全部 200。所以那次限流是按「主机+路径」挂的，
# 不是按 IP。裸主机更不容易被挑战，改用它。
TX_KLINE = ("https://ifzq.gtimg.cn/appstock/app/fqkline/get"
            "?param={sym},day,{start},{end},{cnt},")

_HIST_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]

_EM_RETRIES = 3          # 单只东财重试次数
_EM_TRIP = 12            # 连续失败多少只后熔断
_EM_RETRY_AFTER = 150    # 熔断后每隔多少只回探一次
_em_state = {"fail_streak": 0, "tripped": False, "since_probe": 0, "em": 0, "tx": 0}
_em_lock = __import__("threading").Lock()


def _dash(d: str) -> str:
    d = str(d).replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else str(d)


def daily_hist_tx(code: str, start: str, end: str,
                  timeout: float = 8.0) -> "pd.DataFrame":  # noqa: F821
    """腾讯不复权日线。字段名与东财 stock_zh_a_hist 对齐，便于互换。"""
    import pandas as pd
    sym = to_symbol(code)
    url = TX_KLINE.format(sym=sym, start=_dash(start), end=_dash(end), cnt=640)
    r = _SESSION.get(url, headers=UA, timeout=timeout)
    js = r.json()
    node = (js.get("data") or {}).get(sym) or {}
    rows = node.get("day") or node.get("qfqday") or []
    if not rows:
        return pd.DataFrame(columns=_HIST_COLS)

    rec = []
    prev = None
    for it in rows:
        try:
            d, o, cl, hi, lo, vol = (it[0], float(it[1]), float(it[2]),
                                     float(it[3]), float(it[4]), float(it[5]))
        except (ValueError, IndexError):
            continue
        rec.append({
            "日期": d, "开盘": o, "收盘": cl, "最高": hi, "最低": lo,
            "成交量": vol, "成交额": cl * vol * 100.0,
            "涨跌幅": round((cl - prev) / prev * 100.0, 2) if prev else 0.0,
        })
        prev = cl
    return pd.DataFrame(rec, columns=_HIST_COLS)


SINA_KLINE = ("https://quotes.sina.cn/cn/api/json_v2.php/"
              "CN_MarketDataService.getKLineData"
              "?symbol={sym}&scale=240&ma=no&datalen=1023")


def daily_hist_sina(code: str, start: str, end: str,
                    timeout: float = 10.0) -> "pd.DataFrame":  # noqa: F821
    """新浪日线。第三路兜底，2026-09-02 加。

    单次给 1023 根，实测回溯到 2022-06，比腾讯那路的 640 根上限还长。
    代价是它**只有 OHLC 和成交量**：
      成交额  用 收盘×成交量 估算，和腾讯那路一样是估算不是真值
      涨跌幅  按相邻收盘价算，**除权日会错**（东财那路才是复权真值）
    所以顺序仍然是 东财 -> 腾讯 -> 新浪，它只在前两路都不通时顶上。
    """
    import pandas as pd
    sym = to_symbol(code)
    r = _SESSION.get(SINA_KLINE.format(sym=sym), headers=UA, timeout=timeout)
    js = r.json()
    if not isinstance(js, list) or not js:
        return pd.DataFrame(columns=_HIST_COLS)
    s0, e0 = _dash(start), _dash(end)
    rec, prev = [], None
    for it in js:
        d = str(it.get("day", ""))[:10]
        if not (s0 <= d <= e0):
            prev = float(it["close"])
            continue
        cl = float(it["close"])
        vol = float(it["volume"]) / 100.0     # 新浪给的是股，统一成手
        rec.append({
            "日期": d, "开盘": float(it["open"]), "收盘": cl,
            "最高": float(it["high"]), "最低": float(it["low"]),
            "成交量": vol, "成交额": cl * vol * 100.0,
            "涨跌幅": round((cl - prev) / prev * 100.0, 2) if prev else 0.0,
        })
        prev = cl
    return pd.DataFrame(rec, columns=_HIST_COLS)


def daily_hist_em(code: str, start: str, end: str,
                  retries: int = _EM_RETRIES) -> "pd.DataFrame":  # noqa: F821
    """东财不复权日线（akshare 封装）。成交额是真实值，腾讯那路是估算。"""
    import akshare as ak
    last = None
    for attempt in range(retries):
        try:
            return ak.stock_zh_a_hist(
                symbol=str(code).zfill(6), period="daily",
                start_date=str(start).replace("-", ""),
                end_date=str(end).replace("-", ""), adjust="",
            )
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.3 * (attempt + 1))
    raise last if last else RuntimeError("东财日线失败")


def _em_allowed() -> bool:
    """熔断开关。熔断后每 _EM_RETRY_AFTER 只放一只过去回探。"""
    with _em_lock:
        if not _em_state["tripped"]:
            return True
        _em_state["since_probe"] += 1
        if _em_state["since_probe"] >= _EM_RETRY_AFTER:
            _em_state["since_probe"] = 0
            return True
        return False


def _em_result(ok: bool) -> None:
    with _em_lock:
        if ok:
            _em_state["fail_streak"] = 0
            _em_state["em"] += 1
            if _em_state["tripped"]:
                _em_state["tripped"] = False
                log.info("东财日线已恢复，切回主源")
        else:
            _em_state["fail_streak"] += 1
            if not _em_state["tripped"] and _em_state["fail_streak"] >= _EM_TRIP:
                _em_state["tripped"] = True
                log.warning("东财日线连续 %d 只失败，本次熔断，改走腾讯"
                            "（每 %d 只回探一次）", _EM_TRIP, _EM_RETRY_AFTER)


def hist_source_stats() -> dict:
    """本次进程内各源命中数，跑完打日志用。"""
    with _em_lock:
        return {"东财": _em_state["em"], "腾讯": _em_state["tx"],
                "熔断中": _em_state["tripped"]}


def daily_hist(code: str, start: str, end: str) -> "pd.DataFrame":  # noqa: F821
    """
    个股不复权日线。用于算 5 日均量、60 日分位、平台高点。

    三路：东财（成交额是真值）-> 腾讯 -> 新浪。
    东财失败重试 3 次，整体不通时熔断。腾讯被限流返 501 时新浪顶上。
    """
    import pandas as pd
    if _em_allowed():
        try:
            h = daily_hist_em(code, start, end)
            if h is not None and len(h) >= 25:
                _em_result(True)
                return h
            _em_result(False)
        except Exception as e:  # noqa: BLE001
            _em_result(False)
            log.debug("东财日线(%s) 失败: %s", code, e)
    try:
        h = daily_hist_tx(code, start, end)
        if h is not None and len(h) >= 25:
            with _em_lock:
                _em_state["tx"] += 1
            return h
    except Exception as e:  # noqa: BLE001
        log.debug("腾讯日线(%s) 失败: %s", code, e)
    # 第三路。腾讯那路被限流返 501 时，这一路仍然通（2026-09-02 实测）。
    try:
        h = daily_hist_sina(code, start, end)
        if h is not None and len(h) >= 25:
            with _em_lock:
                _em_state["sina"] = _em_state.get("sina", 0) + 1
            return h
    except Exception as e:  # noqa: BLE001
        log.debug("新浪日线(%s) 失败: %s", code, e)
    return pd.DataFrame(columns=_HIST_COLS)


def daily_hist_many(codes: Sequence[str], start: str, end: str,
                    workers: int = 4) -> dict[str, "pd.DataFrame"]:  # noqa: F821
    """
    批量拉日线。盘前 stage2 用，1600 只串行要 9 分钟以上，4 路并发压到 2-3 分钟。

    并发数和 fetch_quotes 一样保守：免费接口并发一高就限流，别往上调。
    """
    out: dict[str, "pd.DataFrame"] = {}  # noqa: F821
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(daily_hist, c, start, end): c for c in codes}
        for f in as_completed(futs):
            code = futs[f]
            try:
                out[code] = f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("日线 %s 失败: %s", code, e)
    return out


def trade_dates() -> set[str]:
    """交易日历（YYYY-MM-DD）。带本地缓存兜底。"""
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    return {str(d) for d in df["trade_date"].astype(str)}
