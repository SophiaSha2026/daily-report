"""
把历史日线 + 历史竞价重建成训练表。

产出的每一行 = 某一天某只票，列和 score.py 的 AuctionFeature 对齐，
再加上标签 r。下游（dataset / optimize / model_select）拿到的东西
和线上积累的那份**结构完全一致**，只是来源不同。

已知偏差（写在这里，报告里也会标）：
  1. 幸存者偏差。codes.csv 是今天的代码表，一年前退市的票不在里面。
     token 没有 stock_basic 权限，拿不到退市清单，补不了。
     单日持仓下影响轻微，但会略微高估收益。
  2. 板块表漂移。sector_map 是当前成分，用它标注一年前等于用了未来信息。
     sector 维度的历史结论要打折看。
  3. 候选池近似。线上 stage1 用实时快照的换手率和总市值筛，历史没有
     （daily_basic 也无权限）。这里用「昨日涨停 or 昨日涨幅>=5% or
     昨日成交额排名<=600」重建，和线上不完全一致。
  4. ST 判定。拿不到历史股票名，一律按非 ST 算涨停幅度。线上 universe
     本来就 exclude_st，影响面小。
  5. 竞价轨迹是代理值。stk_auction_o 给的是竞价段 OHLC，
     t1 用段内 open、t2 用 vwap、t3 用 close，不是线上那三个精确采样点。
     方向和量级对，绝对值不对。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "cache"
TRAIN = ROOT / "data" / "train"


def _limit_pct_by_code(code: str) -> float:
    """只按板块推涨停幅度。历史股票名拿不到，ST 一律当非 ST（已知偏差 4）。"""
    c = str(code).zfill(6)
    if c.startswith(("30", "68")):
        return 20.0
    if c.startswith(("83", "87", "88", "92", "43")):
        return 30.0
    return 10.0


def load_names() -> dict[str, str]:
    """当前在市股票的名字。只用于 ST 排除，拿不到就返回空。"""
    p = CACHE / "names.csv"
    if p.exists():
        d = pd.read_csv(p, dtype=str)
        return dict(zip(d["code"], d["name"]))
    return {}


def refresh_names(codes: list[str]) -> dict[str, str]:
    import datasource as ds
    q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
    m = {v.code: v.name for v in q.values()}
    pd.DataFrame({"code": list(m), "name": list(m.values())}).to_csv(
        CACHE / "names.csv", index=False, encoding="utf-8")
    log.info("代码-名称表刷新 %d 只", len(m))
    return m


# ---------------------------------------------------------------------
#  日线派生特征
# ---------------------------------------------------------------------
def daily_features(h: pd.DataFrame, breakout_lookback: int = 20
                   ) -> pd.DataFrame:
    """按 code 分组做滚动计算。2.2M 行用 groupby.transform，秒级。

    所有「昨日」类特征都要 shift(1)：竞价时点能看到的只有昨天收盘为止的信息。
    漏 shift 就是未来函数，回测会漂亮得不真实。
    """
    d = h.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                          "最高": "high", "最低": "low",
                          "成交量": "vol", "成交额": "amount",
                          "涨跌幅": "chg"}).copy()
    d["date"] = d["date"].astype(str).str[:10]
    for c in ("open", "close", "high", "low", "vol", "amount", "chg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("code", sort=False)

    # 除权修正：不复权收盘价在除权日会跳空，直接 shift 会把除权当成暴跌。
    # 数据源自带的 chg 是复权后的涨跌幅，用它反解出「可比的昨收」。
    d["prev_close"] = np.where(
        d["chg"].notna() & (d["chg"] > -100),
        d["close"] / (1.0 + d["chg"] / 100.0),
        g["close"].shift(1))
    d["prev_amount"] = g["amount"].shift(1)
    d["prev_vol"] = g["vol"].shift(1)
    d["prev_gain"] = g["chg"].shift(1)
    d["prev_close_1"] = g["close"].shift(1)
    d["prev_high"] = g["high"].shift(1)
    d["prev_close_2"] = g["close"].shift(2)

    d["lim"] = [_limit_pct_by_code(c) for c in d["code"]]
    lu = d["chg"] >= d["lim"] - 0.4
    d["is_limit_up"] = lu
    d["prev_limit_up"] = g["is_limit_up"].shift(1).fillna(False).astype(bool)

    # 连板高度：截至昨天为止的连续涨停根数。
    # 断点分组 + 组内累计，比逐行回溯快两个数量级。
    brk = (~lu).groupby(d["code"]).cumsum()
    d["streak"] = lu.groupby([d["code"], brk]).cumsum()
    d["board_height"] = g["streak"].shift(1).fillna(0).astype(int)

    # 昨日炸板：昨天盘中触涨停但收盘没封
    d["prev_broken_board"] = (
        (~d["prev_limit_up"])
        & (d["prev_high"] >= d["prev_close_2"] * (1 + d["lim"] / 100) - 0.01)
        & d["prev_close_2"].notna()).fillna(False)

    # 60 日位置：**不含今天**，用截至昨天的窗口
    cmin = g["close"].transform(lambda s: s.shift(1).rolling(60, min_periods=25).min())
    cmax = g["close"].transform(lambda s: s.shift(1).rolling(60, min_periods=25).max())
    rng = cmax - cmin
    d["pos_pct_60d"] = np.where(rng > 0, (d["prev_close_1"] - cmin) / rng, 0.5)

    for k in (5, 10, 20):
        d[f"ma{k}"] = g["close"].transform(
            lambda s, k=k: s.shift(1).rolling(k, min_periods=k).mean())
    d["ma_bull"] = ((d.ma5 > d.ma10) & (d.ma10 > d.ma20)).fillna(False)

    d["platform_high"] = g["high"].transform(
        lambda s: s.shift(1).rolling(breakout_lookback, min_periods=5).max())
    a5 = g["amount"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    d["amount_ratio_5d"] = np.where(a5 > 0, d["prev_amount"] / a5, 1.0)

    # 标签：开盘买、收盘卖。开盘价按定义就是竞价撮合价。
    d["r"] = np.where((d["open"] > 0) & (d["close"] > 0),
                      d["close"] / d["open"] - 1.0, np.nan)
    return d


# ---------------------------------------------------------------------
#  竞价段
# ---------------------------------------------------------------------
def merge_auction(d: pd.DataFrame, auc: pd.DataFrame | None) -> pd.DataFrame:
    """把 stk_auction_o 的竞价段 OHLC 接上去，算出竞价类特征。

    没有竞价数据时（免费源）这些列留 NaN，下游据此把 volume / trend
    两个维度排除出可学集合。
    """
    if auc is None or auc.empty:
        for c in ("auc_price", "auc_amount", "auc_open", "auc_high", "auc_vwap"):
            d[c] = np.nan
        return d
    a = auc.copy()
    a["code"] = a["ts_code"].str.slice(0, 6)
    a["date"] = (a["trade_date"].astype(str).str.slice(0, 4) + "-"
                 + a["trade_date"].astype(str).str.slice(4, 6) + "-"
                 + a["trade_date"].astype(str).str.slice(6, 8))
    a = a.rename(columns={"close": "auc_price", "open": "auc_open",
                          "high": "auc_high", "low": "auc_low",
                          "amount": "auc_amount", "vwap": "auc_vwap"})
    cols = ["code", "date", "auc_price", "auc_open", "auc_high", "auc_low",
            "auc_amount", "auc_vwap"]
    return d.merge(a[[c for c in cols if c in a.columns]],
                   on=["code", "date"], how="left")


def to_features(d: pd.DataFrame, sector: dict[str, str],
                names: dict[str, str]) -> pd.DataFrame:
    """拼出 AuctionFeature 的全部列。"""
    pc = d["prev_close"]
    # 竞价撮合价：有竞价数据用它，没有就用日线开盘（两者本该相等）
    d["auc_price"] = d["auc_price"].fillna(d["open"])
    d["gap_pct"] = np.where(pc > 0, (d["auc_price"] / pc - 1.0) * 100, np.nan)

    d["limit_pct"] = [
        5.0 if names.get(c, "").upper().replace(" ", "").startswith(("ST", "*ST"))
        else l for c, l in zip(d["code"], d["lim"])]
    d["gap_norm"] = d["gap_pct"] / d["limit_pct"]
    d["auc_ratio"] = np.where(d["prev_amount"] > 0,
                              d["auc_amount"] / d["prev_amount"], np.nan)

    # 竞价轨迹的代理值。线上 T1/T2/T3 是 09:19:40 / 09:23:30 / 09:25:10 的
    # 精确采样，历史没有；这里用竞价段的 开盘/vwap/收盘 顶替。
    # 方向和量级对得上，绝对值不对，所以 slope 只当序数用。
    def chg(x):
        return np.where(pc > 0, (x / pc - 1.0) * 100, np.nan)
    d["t1_chg"] = chg(d["auc_open"].fillna(d["auc_price"]))
    d["t2_chg"] = chg(d["auc_vwap"].fillna(d["auc_price"]))
    d["t3_chg"] = d["gap_pct"]
    d["slope"] = d["t3_chg"] - d["t1_chg"]
    d["monotonic"] = ((d["t1_chg"] <= d["t2_chg"] + 1e-9)
                      & (d["t2_chg"] <= d["t3_chg"] + 1e-9)).fillna(False)
    d["dive"] = (d["t2_chg"] - d["t3_chg"]).fillna(0.0)
    # 假涨停判据用段内最高价，比线上单点采样还准一些
    d["auc_high_chg"] = chg(d["auc_high"].fillna(d["auc_price"]))
    d["t1_chg"] = np.where(d["auc_high"].notna(),
                           np.maximum(d["t1_chg"], d["auc_high_chg"]),
                           d["t1_chg"])

    d["breakout"] = (d["auc_price"] > d["platform_high"]).fillna(False)
    d["sector"] = d["code"].map(sector).fillna("未分类")
    d["name"] = d["code"].map(names).fillna("")
    d["blacklisted"] = False       # 历史公告拿不到，一律 False（已知偏差）
    # 一字板：竞价段全程贴在涨停，买不进
    d["one_word"] = ((d["gap_pct"] >= d["limit_pct"] - 0.3)
                     & (d["auc_low"].fillna(0) >= d["auc_price"] - 1e-9)
                     ).fillna(False)
    return d


def build_pool(d: pd.DataFrame, c: dict) -> pd.DataFrame:
    """按天重建候选池。近似规则见模块 docstring 已知偏差 3。"""
    u = c["universe"]
    inc = u["include_if"]
    keep = pd.Series(False, index=d.index)
    if inc.get("yesterday_limit_up"):
        keep |= d["prev_limit_up"]
    if inc.get("yesterday_gain_pct_gte") is not None:
        keep |= d["prev_gain"] >= inc["yesterday_gain_pct_gte"]
    if inc.get("amount_ratio_5d_gte") is not None:
        keep |= d["amount_ratio_5d"] >= inc["amount_ratio_5d_gte"]
    rank = d.groupby("date")["prev_amount"].rank(ascending=False)
    keep |= rank <= 600

    d = d[keep & d["prev_gain"].gt(-9.0) & d["prev_close"].gt(0)].copy()
    if u.get("exclude_st"):
        d = d[~d["name"].str.upper().str.replace(" ", "").str.startswith(
            ("ST", "*ST"), na=False)]
    if u.get("exclude_bj") is False:
        pass
    else:
        d = d[~d["code"].str.startswith(("83", "87", "88", "92", "43"))]
    # 每天按成交额取前 max_candidates
    d["_rk"] = d.groupby("date")["prev_amount"].rank(ascending=False)
    return d[d["_rk"] <= u["max_candidates"]].drop(columns="_rk")


def add_sector_stats(d: pd.DataFrame, c: dict) -> pd.DataFrame:
    """板块共振：池内同板块只数、板块昨日涨停家数。

    「未分类」是占位符，必须排除在计数外——它会被 f_sector 当成一个
    几百只成分的巨型板块，白送满分（CLAUDE.md 历史教训第 9 条）。
    """
    real = d["sector"].ne("未分类")
    d["sector_members"] = np.where(
        real, d.groupby(["date", "sector"])["code"].transform("size"), 0)
    lu = d.groupby(["date", "sector"])["prev_limit_up"].transform("sum")
    d["sector_prev_limitups"] = np.where(real, lu, 0)
    return d


COLS = ["date", "code", "name", "limit_pct", "prev_close", "auc_price",
        "gap_pct", "gap_norm", "auc_amount", "prev_amount", "auc_ratio",
        "t1_chg", "t2_chg", "t3_chg", "slope", "monotonic", "dive",
        "pos_pct_60d", "ma_bull", "breakout", "prev_limit_up",
        "prev_broken_board", "board_height", "sector", "sector_members",
        "sector_prev_limitups", "blacklisted", "one_word", "r"]


def build(c: dict, out: Path | None = None) -> Path:
    hp, ap = CACHE / "hist_daily.parquet", CACHE / "hist_auction.parquet"
    if not hp.exists():
        raise FileNotFoundError("先跑 --stage backfill 拉日线")
    h = pd.read_parquet(hp)
    auc = pd.read_parquet(ap) if ap.exists() else None
    sector = dict(pd.read_parquet(CACHE / "sector_map.parquet")
                  [["code", "sector"]].values)
    names = load_names()

    log.info("日线 %d 行 %d 只；竞价 %s", len(h), h["code"].nunique(),
             f"{len(auc)} 行" if auc is not None else "无（免费源）")
    d = daily_features(h, c["screen"]["breakout_lookback"])
    d = merge_auction(d, auc)
    d = to_features(d, sector, names)
    d = build_pool(d, c)
    d = add_sector_stats(d, c)
    d = d[d["gap_pct"].notna() & d["r"].notna()]

    out = out or (TRAIN / f"backfill_{d['date'].min()}_{d['date'].max()}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = d[COLS].reset_index(drop=True)
    res.to_parquet(out, index=False)
    log.info("训练表落盘 %s：%d 行，%d 天，日均 %d 只", out.name, len(res),
             res["date"].nunique(), len(res) // max(res["date"].nunique(), 1))
    return out
