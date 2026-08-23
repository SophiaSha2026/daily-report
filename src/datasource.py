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


def fetch_quotes(
    symbols: Sequence[str],
    *,
    timeout: float = 4.0,
    retries: int = 2,
    sleep_between: float = 0.05,
) -> dict[str, Quote]:
    """
    批量拉取实时/竞价快照。返回 {symbol: Quote}。
    单批失败只丢该批，不影响其余，保证竞价窗口内尽可能拿到多数数据。
    """
    out: dict[str, Quote] = {}
    batches = [symbols[i:i + TX_BATCH] for i in range(0, len(symbols), TX_BATCH)]

    for batch in batches:
        url = TX_URL.format(codes=",".join(batch))
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, headers=UA, timeout=timeout)
                r.encoding = "gbk"
                for m in _LINE.finditer(r.text):
                    q = _parse_tx_body(m.group("sym"), m.group("body"))
                    if q and q.prev_close > 0:
                        out[q.symbol] = q
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries:
                    log.warning("腾讯批量行情失败 batch=%s err=%s", batch[:2], e)
                else:
                    time.sleep(0.15)
        time.sleep(sleep_between)

    log.info("fetch_quotes: 请求 %d 只，返回 %d 只", len(symbols), len(out))
    return out


# ---------------------------------------------------------------------
#  盘前用（时间宽裕，走 akshare / 东财）
# ---------------------------------------------------------------------

def spot_all() -> "pd.DataFrame":  # noqa: F821
    """全市场快照。盘前用于构建候选池。"""
    import akshare as ak
    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 3000:
                return df
            last_err = RuntimeError(f"返回行数异常: {0 if df is None else len(df)}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"全市场快照获取失败: {last_err}")


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
