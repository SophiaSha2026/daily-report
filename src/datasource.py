"""
数据源层：腾讯批量行情、代码表、交易日历，外加涨停价 / 涨跌停幅度这些口径函数。

2026-09-27 早盘系统归档之后，用它的是晚间两条线：
  · 起涨预测每天追加当天 K 线（breakout/backfill.py --stage update）走 fetch_quotes
  · 长期调整突破拿当前名称剔 ST（pullback.names_for / is_excluded）走 fetch_quotes，
    涨停判定（pullback.prepare）用 limit_pct / limit_price_arr
  · 所有流程判交易日走 trade_dates（接口 -> state/trade_dates.json 缓存）
历史日线不在这里：起涨预测的 breakout/backfill.py 自己拉（新浪为主、腾讯兜底）。
只给早盘用的全市场快照 spot_all、三路日线 daily_hist*、次新判定 is_new_listing
2026-09-27 删掉，要恢复早盘从 git 取回（见 archive/morning/RESTORE.md）。

腾讯返回字段顺序历史上调整过，本文件只使用 index <= 37 的低位字段
（相对稳定），涨停价由昨收自行推算而非读取。所有网络调用带重试 + 超时，
单批失败只丢那一批。改了字段解析，跑一次 tools/e2e_check.py（真起子进程拉快照）。
"""
from __future__ import annotations

import os
import re
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Sequence

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


# 第三方库里不带 timeout 的 requests 调用，补上的默认超时（秒）
HTTP_TIMEOUT = 30


def http_timeout(sec: float = HTTP_TIMEOUT) -> None:
    """给这个进程里所有**没写 timeout** 的 requests 调用补一个默认超时。

    akshare 的 stock_zh_a_daily（新浪日线）裸调 requests.get(url)，不带 timeout，
    连接半死的时候会永远等下去：2026-09-29 16:30 起涨预测整段重拉 43 只，
    一个子进程挂在这里 14 小时，那天的清单到次日 06:31 手动重跑才发出去
    （CLAUDE.md 教训 43）。我们自己的调用都写了 timeout，第三方库改不了，
    所以在 Session.request 这一层补：调用方显式给了 timeout 的原样不动。
    每个进程调一次（多进程的子进程要各自调），重复调用不会叠。
    """
    cur = requests.Session.request
    if getattr(cur, "_default_timeout", None) is not None:
        return

    def request(self, method, url, *args, **kw):
        # timeout 是 Session.request 的第 7 个位置参数（url 之后），
        # 位置传了就说明调用方自己给了
        if len(args) < 7 and kw.get("timeout") is None:
            kw["timeout"] = sec
        return cur(self, method, url, *args, **kw)

    request._default_timeout = sec
    requests.Session.request = request


@dataclass
class Quote:
    """一只票在某一时刻的快照。price 在 9:15-9:25 期间为虚拟撮合参考价。"""
    code: str            # 6位代码
    market: str          # sh / sz / bj
    name: str
    price: float         # 当前价 / 竞价撮合价
    prev_close: float    # 昨收（已含除权调整）
    open_: float         # 今开（9:25 后才有效）
    volume_hand: float   # 累计成交量（手）。腾讯对 688 段原始给「股」，
                         # 解析时已由 tx_vol_hand 折成手，三路日线同单位
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
    """当日涨停幅度（百分比）。按板块：创业板/科创板 20%，北交所 30%，其余 10%。

    **ST 不再一律 5%。** 2026-09-16 实测 201 只 ST：主板 141 只里 131 只
    自 2026-03 起有过 |日涨跌| > 5.5%（110 只 > 9.5%，最大 10.47%，无一
    超过 10.5%），创业/科创 57/57 全部 > 5.5%（47 只 > 10.5%，*ST清越
    688496 到过 ±20.5%），北交所 3/3（*ST康乐 920575 −30.00%）。
    腾讯 f[47]/f[4] 同口径：ST海王 1.10、ST南都 1.2005、*ST田野 1.2958。
    旧的「ST -> 5%」排在板块判断之前，把这 201 只全算成 5%。

    真正剩下的 5% 只有**未股改 S 股**（名字以 S 开头且不含 ST）：
    全市场只有 S佳通 600182，腾讯 f[47]=13.42/12.78=1.0501，
    daily.parquet 里 254 根 K 线最大 +5.024%、最小 −4.996%。
    旧实现给它 10%，它 +5% 封板时 prev_limit_up / 一字板 / 假涨停判据全错。
    """
    c = str(code).zfill(6)
    nm = str(name).upper().replace(" ", "")
    if nm.startswith("S") and "ST" not in nm:
        return 5.0
    # 科创板整段 68x 都是 20cm（688 主体 + 689 存托凭证）。以前只认 688，
    # 689 段落到 10%，和回填那份前缀表（"68" -> 20）分叉
    if c.startswith("68") or c.startswith("30"):
        return 20.0
    # 北交所：老代码段 8/4 开头，2024 年起的专属段 920xxx 以 9 开头。
    # 2026-09-15 前这里漏了 9，50 只北交所全按 10% 算：假涨停判据、
    # 斜率归一、一字板判定、昨日涨停全错。to_symbol 一直认 9，这里没同步。
    if c[0] in ("8", "4", "9"):
        return 30.0
    return 10.0


def is_st_name(name: str) -> bool:
    """名字层面判 ST / 退市整理。长期调整突破（pullback.is_excluded）用它。

    只留这一份判据：早盘系统归档前，盘前候选池和学习回填各写过一份
    （`contains("ST|退")` 和 `startswith(("ST","*ST"))`），两边剔掉的不是同一批票
    （2026-09-16 审计，教训 34）。
    """
    s = str(name).upper()
    return "ST" in s or "退" in str(name)


def _is_bj(code: str) -> bool:
    return str(code).zfill(6)[0] in ("8", "4", "9")


def limit_price(prev_close: float, pct: float, code: str = "") -> float:
    """涨停价。沪深四舍五入到分，**北交所向下取整到分**。

    北交所那条是实测出来的：data/breakout/daily.parquet 里 920 段 978 个封板日，
    `最高价×100 − 昨收×(100+pct)` 的平台落在 (-1, 0]（-0.9~0.0 各 45~90 次），
    而沪主板 21841 个封板日的平台是 [-0.4, +0.5]（.xx5 平局进位 1724 次）。
    逐票对得上：920002 昨收 84.63×1.3=110.019 封 110.01、920006 17.09→22.217
    封 22.21、920083 31.46→40.898 封 40.89、920090 2.85→3.705 封 3.70。
    腾讯 f[47] 对 338 只 920 里 170 只与四舍五入差 0.01，全部是四舍五入多 0.01。

    不传 code 时保持旧行为（四舍五入），所以只有真的知道是哪只票的调用方
    才会拿到北交所口径。按代码判而不是按 pct==30 判：to_symbol 认 8/4/9 三段。
    """
    x = prev_close * (1 + pct / 100.0)
    if code and _is_bj(code):
        # +1e-6 防浮点掉档：0.70*1.3 = 0.9099999999999999，直接 floor 会得 0.90
        return math.floor(x * 100 + 1e-6) / 100.0
    return round(x + 1e-9, 2)


def limit_price_arr(prev_close, pct, code=None):
    """limit_price 的向量化孪生体（pandas Series / numpy 数组）。

    code 给了就和标量版同口径（北交所向下取整）；不给就一律四舍五入。
    两个版本必须同口径：长期调整突破的涨停判定（pullback.prepare）用这个版本，
    它的自测造「封涨停」的数据用标量版，差一分就是两套语义（教训 30）。

    只留一份公式：早盘系统归档前，盘前候选池和学习回填各写过一遍
    `prev_close*(1+lim/100) - 0.01`，那个 −0.01 不是防浮点噪音，是把
    涨停价整体放宽一分 —— data/breakout/daily.parquet 428 万行实测，
    −0.01 判「昨日炸板」25418 行、精确判 21483 行，多出的 3935 行
    （15.5%）全部是「最高价 = 涨停价 − 0.01」的误判（600654 2026-09-14
    昨收 3.09、涨停价 3.40、最高 3.39 就被判成触及涨停）。
    +1e-9 与 limit_price 同一口径，428 万个收盘价 × 4 档涨停幅度零不一致。
    """
    x = prev_close * (1 + pct / 100.0)
    up = (x + 1e-9).round(2)
    if code is None:
        return up
    import numpy as np
    import pandas as pd
    bj = pd.Series(code).astype(str).str.zfill(6).str[0].isin(("8", "4", "9"))
    dn = np.floor(np.asarray(x, dtype=float) * 100.0 + 1e-6) / 100.0
    return pd.Series(np.where(bj.to_numpy(), dn, np.asarray(up, dtype=float)),
                     index=getattr(up, "index", None))


def tx_vol_hand(code: str, vol: float):
    """腾讯的成交量口径统一成「手」。

    腾讯**两个接口**（qt.gtimg.cn 快照 f[6]、fqkline 日 K 第 6 列）对
    科创板都给「股」，其余板块给「手」：2026-09-16 本机实测
    sh688008 f[6]=35064294 而 f[35]="199.89/35064294/6828993518"，
    成交额/(价×量)=0.97；同式 sz300750=99.73、sh600000=99.70、
    bj920060=96.63。腾讯日 K 与新浪日 K 同日之比 688008/688981/688256
    恰好 100.000，600000/000001/300750 恰好 1.000。
    用代码前缀而不是按 f[35]/f[37] 自校：09:15~09:25 竞价窗口 f[6]=f[37]=0，
    比值无定义。cache/codes.csv 里 688 段 614 只、没有 689 段，前缀够用。
    """
    return vol / 100.0 if str(code).zfill(6).startswith("688") else vol


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
            volume_hand=tx_vol_hand(f[2], float(f[6] or 0)),
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
            # 非 200（限流、挑战页）以前当成「成功的空批」静默吞掉，
            # 一批 60 只就这么消失。当成失败走重试。
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.encoding = "gbk"
            got = {}
            for m in _LINE.finditer(r.text):
                q = _parse_tx_body(m.group("sym"), m.group("body"))
                if q and q.prev_close > 0:
                    got[q.symbol] = q
            if not got and batch:
                raise RuntimeError("正文里没有任何行情行")
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
    批量拉取实时快照（收盘后就是当日收盘态）。返回 {symbol: Quote}。

    并发说明：到腾讯单次往返约 0.6s（美国出口），1600 只串行 17.2s，5 路并发
    3.6s（2026-08 实测）。并发再高会触发限流，别调（CLAUDE.md 硬约束 4）。
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
#  代码表
# ---------------------------------------------------------------------

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
def _sina_code_list(timeout: float = 8.0) -> list[str]:
    """
    新浪分页行情列表（node=hs_a，含北交所）。每页 100 只，约 56 页。

    加这一路的原因：东财的两个代码表接口在 GitHub runner 上是「时好时坏」，
    2026-08-24 那轮冒烟测试里两个都被 ConnectionReset 掐断（同一次运行里
    第二次调用又成功了）。新浪这条通道和交易日历同源，实测稳定。

    **一页失败就整体抛异常，绝不返回残表。** 页是按 symbol 排序切的，
    丢一页就是丢连续一段代码（离线复现：第 3 页超时 -> 5448 只，丢的
    920433~920837 一整段北交所）。残表会被 refresh_code_list 写进
    cache/codes.csv，之后起涨预测的回填和每日追加、长期调整突破全都看不见
    那一段，而唯一的信号是一行 log.warning（教训 16）。
    """
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "Market_Center.getHQNodeData?page={p}&num=100&sort=symbol&asc=1&node=hs_a")
    hdr = {**UA, "Referer": "https://finance.sina.com.cn"}

    def one(p: int) -> list[str]:
        last = None
        for attempt in range(3):
            try:
                r = _SESSION.get(url.format(p=p), headers=hdr, timeout=timeout)
                r.encoding = "gbk"
                txt = r.text.strip()
                if not txt or txt in ("null", "[]"):
                    return []                 # 真到底了，不是失败
                import json as _json
                return [str(d["code"]).zfill(6) for d in _json.loads(txt)]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"新浪列表第 {p} 页重试 3 次仍失败: {last}")

    codes: list[str] = []
    failed: list[int] = []
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
                    failed.append(futs[f])
                    continue
                codes += part
                got += len(part)
            if got == 0:
                break
            page += 8
    if failed:
        # 整组失败时 got==0 会 break，那条路同样不许把残表放出去
        raise RuntimeError(f"新浪列表 {len(failed)} 页失败 {sorted(failed)}，"
                           f"只拿到 {len(set(codes))} 只，不返回残表")
    return sorted(set(codes))


def refresh_code_list() -> list[str]:
    """
    多源兜底拉取代码表，成功即写入缓存。

    每个源重试 2 次：东财那两个接口在 runner 上是间歇性拒绝，
    一次失败不代表不可用（实测同一次运行里第二次调用就成功了）。

    写盘前还有一道「不许比现有表少超过 2%」的闸：`len > 3000` 太松，一次网络
    抖动削成 3200 只照样写进去，而且没有任何自动刷新会来纠正它（refresh_meta
    的周日 cron 2026-09-12 已停用，只剩控制台手点）。
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
    out = _ROOT / "cache"
    dst = out / "codes.csv"
    for name, fn in sources:
        for attempt in range(2):
            try:
                codes = fn()
                if len(codes) > 3000:
                    old_n = 0
                    if dst.exists():
                        try:
                            old_n = len(pd.read_csv(dst, dtype=str))
                        except Exception:  # noqa: BLE001
                            old_n = 0
                    if old_n and len(codes) < old_n * 0.98:
                        log.error("%s 新表 %d 只，比现有 %d 只少超过 2%%，拒绝覆盖",
                                  name, len(codes), old_n)
                        break               # 换下一个源
                    out.mkdir(exist_ok=True)
                    pd.DataFrame({"code": codes}).to_csv(dst, index=False)
                    log.info("代码表已刷新 (%s): %d 只", name, len(codes))
                    return codes
                log.warning("%s 仅返回 %d 只", name, len(codes))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("%s 第%d次失败: %s", name, attempt + 1, e)
                time.sleep(0.8)
    raise RuntimeError("代码表获取失败：所有数据源均不可用，或新表比现有表短太多")


# ---------------------------------------------------------------------
#  交易日历
# ---------------------------------------------------------------------
def _norm_trade_dates(col, today=None) -> set[str]:
    """把接口的 trade_date 列统一成 YYYY-MM-DD，归一不了就抛。

    akshare 1.18.94 的 tool_trade_date_hist_sina 返回 datetime.date，
    `str(d)` 恰好是 YYYY-MM-DD，所以一直没出事。但 requirements.txt 写的是
    `akshare>=1.16.0`，云端每次 pip install 都拉最新版，上游改成
    '20260916' / 20260916 / object 型 Timestamp 里的任何一种，
    `today not in s` 就恒真：起涨预测、长期调整突破全部每天「非交易日」
    退出 0，evening_check 也判非交易日不提醒，三层托底一起哑，
    而且坏日历会写进 state/trade_dates.json 把三处只读缓存的兜底一起污染
    （教训 15「恢复路径本身要能恢复」、教训 27「退出码 0 不等于做了事」）。

    校验放在写缓存**之前**：抛出去让调用方走缓存，缓存里还是上一份好的。
    """
    import re
    import datetime as _dt
    import pandas as pd
    s = set(pd.to_datetime(col.astype(str)).dt.strftime("%Y-%m-%d")) if len(col) else set()
    bad = [d for d in s if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(d))]
    if bad or not s:
        raise ValueError(f"交易日历格式异常: {sorted(bad)[:3] or '空'}")
    today = today or _dt.date.today()
    if not any(abs((_dt.date.fromisoformat(d) - today).days) <= 30 for d in s):
        raise ValueError(f"交易日历不覆盖 {today} 前后 30 天")
    return s


def trade_dates() -> set[str]:
    """交易日历（YYYY-MM-DD）。接口拿到就写 state/trade_dates.json，
    接口失败读缓存；两边都没有才抛异常，由调用方决定怎么兜底。

    缓存也给控制台用：它不能 import akshare（历史教训 19），只读这个文件。
    """
    import json
    cache = _ROOT / "state" / "trade_dates.json"
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        s = _norm_trade_dates(df["trade_date"])
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # 原子写。Path.write_text 是 open("w")：先把文件截断成 0 字节
            # 再写回 123KB，中间那约 1 毫秒里读方（控制台 gui/status.py、
            # local_run 的日历缓存）读到的是空文件，json.loads 直接失败。
            # 实测一写一读并发 3 秒 337 次读里 138 次读到 0 字节（41%）。
            # tmp 名带 pid：两个进程同时写不能共用一个临时文件。
            tmp = cache.with_name(f"trade_dates.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(sorted(s)), encoding="utf-8")
            try:
                os.replace(tmp, cache)
            finally:
                # Windows 上读方正持着句柄时 os.replace 会 PermissionError，
                # 此时旧文件仍然完整，丢掉 tmp 就行
                tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return s
    except Exception as e:  # noqa: BLE001
        if cache.exists():
            log.warning("交易日历接口失败（%s），用本地缓存", e)
            return set(json.loads(cache.read_text(encoding="utf-8")))
        raise
