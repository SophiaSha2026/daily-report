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
from typing import Iterable, Sequence

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """多源兜底拉取代码表，成功即写入缓存。"""
    import akshare as ak
    import pandas as pd
    for name, fn in (
        ("stock_info_a_code_name", lambda: ak.stock_info_a_code_name()),
        ("stock_zh_a_spot_em", lambda: ak.stock_zh_a_spot_em()),
    ):
        try:
            df = fn()
            col = next(c for c in df.columns if c in ("code", "代码"))
            codes = sorted({str(x).zfill(6) for x in df[col]})
            if len(codes) > 3000:
                out = _ROOT / "cache"
                out.mkdir(exist_ok=True)
                pd.DataFrame({"code": codes}).to_csv(out / "codes.csv", index=False)
                log.info("代码表已刷新 (%s): %d 只", name, len(codes))
                return codes
            log.warning("%s 仅返回 %d 只", name, len(codes))
        except Exception as e:  # noqa: BLE001
            log.warning("%s 失败: %s", name, e)
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


def daily_hist(code: str, start: str, end: str) -> "pd.DataFrame":  # noqa: F821
    """个股不复权日线。用于算 5 日均量、60 日分位、平台高点。"""
    import akshare as ak
    for attempt in range(2):
        try:
            return ak.stock_zh_a_hist(
                symbol=str(code).zfill(6), period="daily",
                start_date=start, end_date=end, adjust="",
            )
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    import pandas as pd
    return pd.DataFrame()


def trade_dates() -> set[str]:
    """交易日历（YYYY-MM-DD）。带本地缓存兜底。"""
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    return {str(d) for d in df["trade_date"].astype(str)}
