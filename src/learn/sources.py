"""
回填数据源适配器。

两条路，下游代码看到的是同一张表，只是列的完整度不同：

    free      免费源（腾讯日线）。只能还原 H 块：gap / position / sector /
              continuity。竞价量能和竞价轨迹历史上取不到，2026-09-02 全渠道
              实测见 docs/learning.md 2.3。
    tushare   填了 TUSHARE_TOKEN 就走这条。两个接口按可用性自动降级：
                stk_auction_o  竞价段 OHLC -> 能补 volume + trend
                stk_auction    竞价成交量/额/量比/换手/流通股本 -> 只补 volume
              官网「集合竞价成交」¥500/年 对应的是后者，历史从 2025-01 起。
              前者是否在同一权限里不确定，所以运行时探测：先试 _o，
              401/权限不足就退到 stk_auction，都不通就退回 free。

表上带一列 `completeness`，取值 H 或 FULL，下游据此决定哪些维度可学。
换源不改任何下游代码。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "cache"

# H 块能还原的维度。免费源下 volume / trend 的参数不参与学习。
DIMS_H = ["gap", "position", "sector", "continuity"]
DIMS_ALL = DIMS_H + ["volume", "trend"]


def token() -> str:
    """按优先级找 Tushare token：环境变量 -> .secret -> tools/local.env。

    `.secret` 是裸值一行；`tools/local.env` 是 KEY=VALUE。两个都在
    .gitignore 里。token 只在这里读一次进内存，不写日志、不进任何产物。
    """
    t = os.environ.get("TUSHARE_TOKEN", "").strip()
    if t:
        return t
    root = Path(__file__).resolve().parent.parent.parent
    p = root / ".secret"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split("=", 1)[1].strip() if "=" in line else line
            if len(v) >= 32:
                os.environ["TUSHARE_TOKEN"] = v
                return v
    p = root / "tools" / "local.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                v = line.split("=", 1)[1].strip()
                if v and "FILLME" not in v:
                    os.environ["TUSHARE_TOKEN"] = v
                    return v
    return ""


def which() -> str:
    """当前可用的回填源。有 token 就用 tushare，否则 free。"""
    return "tushare" if token() else "free"


def completeness() -> str:
    """H = 只有 gap/position/sector/continuity；+V = 再加 volume；FULL = 全六维。"""
    if which() != "tushare":
        return "H"
    return {"o": "FULL", "plain": "H+V"}.get(probe_ts(), "H")


def learnable_dims() -> list[str]:
    """免费源 0.55 权重；stk_auction 再加 volume 到 0.80；_o 才是全部。

    竞价斜率(trend, 0.20) 需要 09:19/09:23 的盘中虚拟撮合价，属于 L1 快照流，
    Tushare 的竞价包和分钟线都不含（分钟线从 09:30 起）。只能靠线上积累。
    """
    return {"FULL": DIMS_ALL, "H+V": DIMS_H + ["volume"]}.get(
        completeness(), DIMS_H)


# ---------------------------------------------------------------------
#  免费源：日线
# ---------------------------------------------------------------------
def _normalize(h: pd.DataFrame, code: str) -> pd.DataFrame:
    """统一列类型。

    两个数据源的 `日期` 类型不一样：东财 stock_zh_a_hist 给的是
    datetime.date 对象，腾讯 K 线给的是字符串。混在一张表里
    pyarrow 直接报 ArrowTypeError，整批落盘失败——第一次跑就是这么
    把二十分钟的下载丢掉的。统一成 'YYYY-MM-DD' 字符串。
    """
    h = h.copy()
    h["日期"] = pd.to_datetime(h["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ("开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"):
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h["code"] = str(code).zfill(6)
    keep = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额",
            "涨跌幅", "code"]
    return h[[c for c in keep if c in h.columns]]


def fetch_daily(codes, start: str, end: str, workers: int = 4,
                out: Path | None = None, chunk: int = 800) -> Path:
    """拉日线并落一张长表。一只票一次调用覆盖整段（实测 405 根/次）。

    分块落盘：5548 只要跑二十来分钟，中途挂了不该从头再来。
    每 chunk 只写一次，重跑时按已有 code 续传。
    """
    import datasource as ds
    out = out or (CACHE / "hist_daily.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    done: set[str] = set()
    if out.exists():
        old = pd.read_parquet(out)
        done = set(old["code"].unique())
        frames.append(old)
        log.info("已有 %d 只，续传", len(done))

    todo = [c for c in codes if str(c).zfill(6) not in done]
    log.info("待拉 %d 只 %s ~ %s", len(todo), start, end)
    t0 = time.time()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        got = ds.daily_hist_many(part, start.replace("-", ""),
                                 end.replace("-", ""), workers=workers)
        for code, h in got.items():
            if h is not None and len(h):
                frames.append(_normalize(h, code))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        log.info("已落盘 %d/%d 只，累计 %.0fs", min(i + chunk, len(todo)),
                 len(todo), time.time() - t0)
    df = pd.read_parquet(out)
    log.info("日线完成 %s：%d 只 %d 行，%.0fs", out.name,
             df["code"].nunique(), len(df), time.time() - t0)
    return out


# ---------------------------------------------------------------------
#  付费源：Tushare 集合竞价
# ---------------------------------------------------------------------
_PROBE: str | None = None


def probe_ts() -> str:
    """探测 token 能开哪个竞价接口。返回 'o' / 'plain' / ''。

    不硬编码「买了哪个包」，因为权限是会变的：今天只有 stk_auction，
    明年补买了 _o，代码不该需要改。运行时问一次，几百毫秒。
    """
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    tk = token()
    if not tk:
        _PROBE = ""
        return ""
    try:
        import tushare as ts
    except ImportError:
        log.warning("有 TUSHARE_TOKEN 但没装 tushare，pip install tushare")
        return ""
    pro = ts.pro_api(tk)
    # 探测日必须是**确定的交易日**。第一版写的 20250602 是端午节休市，
    # 两个接口都合法返回 0 行，差点被误判成没权限。没权限时抛的是异常，
    # 返回空表只说明那天没数据——两者要分清。
    for name, tag in (("stk_auction_o", "o"), ("stk_auction", "plain")):
        try:
            df = getattr(pro, name)(trade_date="20250603")
            if df is not None and len(df):
                log.info("Tushare 竞价接口可用: %s", name)
                _PROBE = tag
                return tag
        except Exception as e:      # noqa: BLE001
            log.info("Tushare %s 不可用: %s", name, str(e)[:80])
    _PROBE = ""
    return ""


def fetch_auction_ts(start: str, end: str,
                     out: Path | None = None) -> Path | None:
    """拉历史竞价数据。接口按 probe_ts() 的结果自动选。

    没 token 或都没权限就返回 None，调用方退回 free 路径。
    按交易日循环，单次上限 8000 行，A 股 5500 只，一天一次调用够。
    """
    tag = probe_ts()
    if not tag:
        return None
    import tushare as ts
    import datasource as ds
    # 优先 stk_auction_o：它按交易日拉是 ~5400 行（A 股全量，不截断），
    # 而 stk_auction 含 ETF 会顶到单次 8000 行上限被静默截断。
    api = "stk_auction_o" if tag == "o" else "stk_auction"
    out = out or (CACHE / "hist_auction.parquet")
    pro = ts.pro_api(token())
    days = sorted(d for d in ds.trade_dates() if start <= d <= end)

    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if out.exists():
        old = pd.read_parquet(out)
        done = set(old["trade_date"].astype(str).unique())
        frames.append(old)

    for d in days:
        key = d.replace("-", "")
        if key in done:
            continue
        try:
            df = getattr(pro, api)(trade_date=key)
            if df is not None and len(df):
                frames.append(df)
            time.sleep(0.35)          # 接口有频次限制，别打太快
        except Exception as e:        # noqa: BLE001
            log.warning("%s %s 失败: %s", api, d, e)
    if not frames:
        return None
    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
    log.info("竞价历史落盘 %s", out.name)
    return out
