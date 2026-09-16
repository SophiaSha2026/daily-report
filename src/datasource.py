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

import os
import re
import math
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
    """名字层面判 ST / 退市整理。

    生产 premarket.stage1 和回填 learn/backfill.build_pool 共用这一份判据。
    以前两边各写各的（生产 `contains("ST|退")`、回填 `startswith(("ST","*ST"))`），
    而回填那边的 names 长期是空字典，exclude_st 等于没执行：训练表里 11514 行
    （2.82%，201 只）ST 票混在池子里，每天前 10 平均 2.7% 席位是生产永远
    不会打分的票（2026-09-16 审计）。
    """
    s = str(name).upper()
    return "ST" in s or "退" in str(name)


def is_new_listing(name: str) -> bool:
    """次新：腾讯给的名字带 N/C 前缀（上市第 1~5 天）。

    旧写法 `^[NC] ` 要求 N/C 后面跟一个空格，而腾讯只在 2~3 字老票补位时
    插空格（`万  科Ａ`、`农 产 品`）；真正的次新是 `C频准` / `C绿控传动` /
    `C宇树-W`，一个空格都没有。那条正则从上线起一次都没生效，688836、
    688826、301655 都进过候选池并被 score.py 打了分（2026-09-16 审计）。
    后面跟一个汉字是为了不误伤以英文开头的老票名。
    """
    return bool(re.match(r"^[NC][一-鿿]", str(name)))


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
    两个版本必须同口径：生产 premarket.stage2 判昨日炸板用标量版，
    回填 learn/backfill 判同一件事用这个版本，差一分就是两套语义（教训 30）。

    只留一份公式：以前 premarket 和 learn/backfill 各写了一遍
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

    **一页失败就整体抛异常，绝不返回残表。** 页是按 symbol 排序切的，
    丢一页就是丢连续一段代码（离线复现：第 3 页超时 -> 5448 只，丢的
    920433~920837 一整段北交所）。残表会被 refresh_code_list 写进
    cache/codes.csv，之后早盘候选池、形态扫描、起涨预测的当日 K 线追加、
    学习回填全都看不见那一段，而唯一的信号是一行 log.warning（教训 16）。
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

    写盘前还有一道「不许比现有表少超过 2%」的闸（和 refresh_meta 里
    sector_map 的 0.9 闸同款）：`len > 3000` 太松，一次网络抖动削成 3200 只
    照样写进去，而且没有任何自动刷新会来纠正它（refresh_meta 的周日 cron
    2026-09-12 已停用，只剩控制台手点）。
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


def spot_all(expect_date: str | None = None) -> "pd.DataFrame":  # noqa: F821
    """
    全市场快照，盘前用于构建候选池。

    ⚠️ 实测：GitHub runner 访问东财 push2 的 clist/get 批量接口会被
    RemoteDisconnected 掐断（但东财的**单只**日线接口 push2his 正常）。
    所以这里改用腾讯批量接口 + 本地代码表，走的是已验证可用的通道。

    盘前 08:23 调用时腾讯返回的是上一交易日收盘状态：
        当前价 = T-1 收盘价      成交额 = T-1 全天成交额
        涨跌%  = T-1 涨跌幅      昨收   = T-2 收盘价
    正好是候选池需要的字段。

    **这个「还没翻篇」是个时间假设，不是事实**，以前一道校验都没有：
    早班触发全丢或机器晚醒时，池子可能 09:15 之后才建，那时腾讯给的是
    当日竞价态（成交额变成竞价额、涨跌幅变成竞价涨幅），按它算出来的
    昨日量能整体放大两个数量级，全榜会被量能条件剔光，而清单照发。
    expect_date 给了就用 f[30] 时间戳核对：正常态实测 5548/5548 只
    ts[:8] 全等于最近已收盘交易日（含 14 只成交额为 0 的停牌票），
    5% 的容差只留给零星未更新。同仓库 breakout/backfill.py 和 pullback.py
    早就把 f[30] 当日期权威用，spot_all 是最后一个没做的消费者。
    """
    import pandas as pd
    codes = load_code_list()
    syms = [to_symbol(c) for c in codes]
    q = fetch_quotes(syms)
    if len(q) < len(syms) * 0.6:
        raise RuntimeError(f"全市场快照过少: {len(q)}/{len(syms)}")
    if expect_date:
        key = str(expect_date).replace("-", "")[:8]
        bad = [str(v.ts) for v in q.values() if not str(v.ts).startswith(key)]
        if len(bad) > len(q) * 0.05:
            raise RuntimeError(
                f"快照已翻篇：{len(bad)}/{len(q)} 只的时间戳不是 {key}"
                f"（示例 {bad[0] if bad else '空'}），"
                f"此刻建池会把竞价态当成昨日收盘态")

    rows = [{
        "代码": v.code, "名称": v.name, "最新价": v.price,
        "涨跌幅": v.chg_pct, "成交额": v.amount_yuan,
        # 没有「总市值」这一列：腾讯低位字段里没有市值（高位字段顺序会变，
        # 硬约束 6 不许读）。以前恒填 NaN，一列永远是假的数据，
        # universe.min/max_mktcap_yi 设成非 0 就会把候选池静默筛空。
        # 现在干脆不输出，premarket.stage1 见到那种配置直接报错。
        "换手率": _turnover(v),
    } for v in q.values()]
    df = pd.DataFrame(rows)
    log.info("全市场快照(腾讯): %d 只", len(df))
    return df


def _turnover(q: Quote) -> float:
    """换手率，腾讯字段 38。取不到返回 0。

    2026-09-16 前这里注释写「不影响主流程」，其实反了：入池四条规则里
    「昨日换手 >= 5%」就是靠它，字段一缺整条规则被静默关掉（08-28 那天
    池子只有 635 只，正常 1100~1400）。现在 premarket.stage1 会检查这一列
    是否成片为 0，成片为 0 就报错发告警邮件，不静默放行。
    """
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
#  腾讯 K 线返回 [日期, 开盘, 收盘, 最高, 最低, 成交量]，**没有成交额**。
#  那个成交量：主板/创业板是手，**科创板是股**，已由 tx_vol_hand 折成手。
#  成交额只能估算。以前写「只喂 amount_ratio_5d 这个展示字段，不进打分」，
#  那句话对生产成立、对回填不成立：learn/backfill.py 拿 cache/hist_daily.parquet
#  的成交额当 prev_amount，也就是 auc_ratio 的分母，而 auc_ratio 是准入判据。
#  估算式从 收盘×量 改成 (最高+最低+收盘)/3×量：实测昨日涨停那一子群
#  收盘×量 的中位偏差 +2.79%，(H+L+C)/3 只有 −0.12%（2026-09-16，
#  基准是新浪日线店 data/breakout/daily.parquet 的真实成交额）。
#  回填那边现在优先并真值进来，估算只作补不上的兜底。
#  竞价用的真实成交额来自腾讯实时快照 index 37，不是这里。
# ---------------------------------------------------------------------

# 2026-09-02：用 4 路并发连打 800 只之后，web.ifzq.gtimg.cn 的
# /appstock/app/fqkline/get 开始整片返回 HTTP 501（JS 挑战页），
# 而同一时刻 qt.gtimg.cn 批量行情、以及**去掉 web. 前缀**的裸主机
# ifzq.gtimg.cn 同一路径全部 200。所以那次限流是按「主机+路径」挂的，
# 不是按 IP。裸主机更不容易被挑战，改用它。
TX_KLINE = ("https://ifzq.gtimg.cn/appstock/app/fqkline/get"
            "?param={sym},day,{start},{end},{cnt},")

# 成交量一律为「手」，成交额为「元」，三路一致（腾讯 688 由 tx_vol_hand 折算）。
# chg_adj：「涨跌幅」这一列是不是**复权口径**的真值。
#   东财 stock_zh_a_hist 的涨跌幅是服务端字段（除权后的真实涨跌幅）-> True
#   腾讯 / 新浪只给 OHLCV，涨跌幅只能按相邻**不复权**收盘价算 -> False
# 三路以前共用「涨跌幅」这一个列名却是两种语义，谁也标不出来：
# learn/backfill.py 曾按「chg 是复权涨跌幅」反解昨收，而 cache/hist_daily.parquet
# 里 2188865 个非首行 100% 是不复权比值，那段反解是恒等变换（2026-09-16 审计）。
_HIST_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
              "涨跌幅", "chg_adj"]

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
    """腾讯不复权日线。字段名与东财 stock_zh_a_hist 对齐，便于互换。

    成交量：腾讯对科创板给「股」，其余给「手」，由 tx_vol_hand 统一成手
    （不折算的话 688 的成交额估算会大 100 倍：688008 2026-09-16 估出
    7009 亿，真值 68.3 亿）。
    涨跌幅：按相邻**不复权**收盘价算，除权日会错，所以 chg_adj=False；
    首根没有前一根，给 NaN 而不是 0.0 —— 给 0.0 会让次日的 prev_gain
    变成「昨天平盘」这个假事实（2025-01-02 那批 440 只里 000001 实际 −2.64%）。
    """
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
                                     float(it[3]), float(it[4]),
                                     tx_vol_hand(code, float(it[5])))
        except (ValueError, IndexError):
            continue
        rec.append({
            "日期": d, "开盘": o, "收盘": cl, "最高": hi, "最低": lo,
            "成交量": vol, "成交额": (hi + lo + cl) / 3.0 * vol * 100.0,
            "涨跌幅": (round((cl - prev) / prev * 100.0, 2) if prev
                    else float("nan")),
            "chg_adj": False,
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
      成交额  用 (最高+最低+收盘)/3×成交量 估算，和腾讯那路同一口径，
              是估算不是真值（理由见上面那段注释）
      涨跌幅  按相邻收盘价算，**除权日会错**（东财那路才是复权真值），
              所以 chg_adj=False；窗口第一根若连前置行都没有就是 NaN
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
        hi, lo = float(it["high"]), float(it["low"])
        rec.append({
            "日期": d, "开盘": float(it["open"]), "收盘": cl,
            "最高": hi, "最低": lo,
            "成交量": vol, "成交额": (hi + lo + cl) / 3.0 * vol * 100.0,
            "涨跌幅": (round((cl - prev) / prev * 100.0, 2) if prev
                    else float("nan")),
            "chg_adj": False,
        })
        prev = cl
    return pd.DataFrame(rec, columns=_HIST_COLS)


def daily_hist_em(code: str, start: str, end: str,
                  retries: int = _EM_RETRIES) -> "pd.DataFrame":  # noqa: F821
    """东财不复权日线（akshare 封装）。成交额是真实值，腾讯那路是估算。

    它的「涨跌幅」是服务端字段（除权后的真实涨跌幅），所以 chg_adj=True ——
    只有这一路的涨跌幅能用来反解「已除权的昨收」。
    """
    import akshare as ak
    last = None
    for attempt in range(retries):
        try:
            h = ak.stock_zh_a_hist(
                symbol=str(code).zfill(6), period="daily",
                start_date=str(start).replace("-", ""),
                end_date=str(end).replace("-", ""), adjust="",
            )
            if h is not None:
                h = h.copy()
                h["chg_adj"] = True
            return h
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


def _em_result(ok: bool, used: bool = True) -> None:
    """ok = 东财**答复了**（决定熔断），used = 这次真的用了它的数据（决定计数）。

    两件事必须分开：新股/长期停牌只有十几根 K 线是「数据本身短」，不是
    「东财不通」。以前 <25 根和抛异常一样累加 fail_streak，连续 12 只短历史
    就把东财熔断掉，之后 150 只无声改走腾讯（腾讯那路的成交额是估算）。
    真实数据下凑不满 12 只（全市场 <25 根的只有 17 只，按代码排序最长连续
    3 只：688826/828/836），但语义错了就该改对。
    """
    with _em_lock:
        if ok:
            _em_state["fail_streak"] = 0
            if used:
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
    """本次进程内各源命中数，跑完打日志用。

    新浪那一路以前不报，1125 只里 780 只去向不明，看日志对不上账。
    """
    with _em_lock:
        return {"东财": _em_state["em"], "腾讯": _em_state["tx"],
                "新浪": _em_state.get("sina", 0),
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
            n = 0 if h is None else len(h)
            # 「通不通」看有没有答复，「用不用」才看够不够 25 根。
            # 空表仍算失败：东财软限流会返回 data=null。
            _em_result(n > 0, used=n >= 25)
            if n >= 25:
                return h
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


def _norm_trade_dates(col, today=None) -> set[str]:
    """把接口的 trade_date 列统一成 YYYY-MM-DD，归一不了就抛。

    akshare 1.18.94 的 tool_trade_date_hist_sina 返回 datetime.date，
    `str(d)` 恰好是 YYYY-MM-DD，所以一直没出事。但 requirements.txt 写的是
    `akshare>=1.16.0`，云端每次 pip install 都拉最新版，上游改成
    '20260916' / 20260916 / object 型 Timestamp 里的任何一种，
    `today not in s` 就恒真：早盘选股、盘前候选池、形态扫描全部每天
    「非交易日」退出 0，evening_check 也判非交易日不提醒，三层托底一起哑，
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
            # 学习闸门 learn/gate.py）读到的是空文件，json.loads 直接失败。
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
