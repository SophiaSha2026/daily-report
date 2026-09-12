"""
特征工程：把原始日线变成可用特征。

三层变换（顺序不能换）
----------------------
    原始指标  →  ① 横截面百分位  →  ② 中性化  →  ③ 组内正交  →  模型

**① 是整套设计里最关键的一个决定。** 它解决时间聚集问题：抽样实测里
2024-09 一个月占了全部正样本的 25%（924 行情普涨），直接拿绝对值训练，
模型学到的是「今天大盘好」而不是「这只票强」。

横截面百分位是把每个特征换成它在**当日全市场**的分位排名。普涨那天
所有票的绝对涨幅、绝对量比都高，但百分位永远是 [0,1] 上的均匀分布，
「今天大盘好」这个信息在特征里根本不存在，模型想学也学不到。

顺带免费解决三件事：量纲统一、异常值天然截断、特征分布不随时间漂移
（2023 年的 0.9 分位和 2026 年的 0.9 分位是同一件事）。

关于 shift(1)
-------------
本线**不需要**对特征做 shift(1)。流水线在收盘后跑，t 日的收盘价、成交量
都是已知的，用 [..t] 合法。前视偏差的边界在标签那边：标签只看 [t+1..t+20]，
两边不重叠就不会漏未来信息。

真正需要延迟的是**低频数据**（股东人数、股本变更），它们有披露滞后，
必须按公告日对齐，见 holder_features。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chips import add_chips

# ---- 五组特征的成员，selftest 拿这个表核对接线 ----
GROUPS: dict[str, list[str]] = {
    "position": ["dist_52w_high", "dist_52w_low", "ma_align",
                 "ma20_slope5", "above_ma60"],
    "compression": ["range_compress", "vol_compress", "atr_ratio",
                    "boll_width"],
    "structure": ["chip_conc90", "chip_conc_chg20", "chip_win", "chip_dev",
                  "chip_peak", "turn_pct", "vol_ratio5", "turn_std20"],
    "holders": ["gdhs_chg1", "gdhs_chg3", "gdhs_down_streak",
                "gdhs_stale_days"],
    "regime": ["rs20", "rs60", "rs_accel", "mkt_breadth", "mkt_ret5"],
}
# 全市场共同变量：当天所有股票同值，做横截面排名会退化成常数，
# 所以刻意排除在 ① 之外。它们的作用正是给模型一个「今天什么天气」的坐标。
MARKET_WIDE = {"mkt_breadth", "mkt_ret5"}

INTERACTIONS = [
    ("gdhs_chg1", "chip_conc_chg20", "x_holder_chip"),
    ("range_compress", "vol_ratio5", "x_squeeze_burst"),
    ("chip_win", "dist_52w_high", "x_light_top"),
]


def board_of(code: str) -> str:
    if code[:3] == "688":
        return "star"
    if code[0] == "3":
        return "chinext"
    if code[:2] in ("83", "87", "88", "43", "92") or code[:3] == "920":
        return "bj"
    return "main"


# ---------------------------------------------------------------
#  单只票的时序特征
# ---------------------------------------------------------------
def per_stock(df: pd.DataFrame) -> pd.DataFrame:
    """一只票的时序特征。df 按日期升序，含 open/high/low/close/volume/turnover。

    这里只算**原始值**，不做任何横截面处理 —— 那是下一层的事。
    """
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    # --- Q1 位置 ---
    ma5, ma10 = c.rolling(5).mean(), c.rolling(10).mean()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    d["dist_52w_high"] = c / c.rolling(250, min_periods=60).max() - 1
    d["dist_52w_low"] = c / c.rolling(250, min_periods=60).min() - 1
    # 四个均线乖离 close/maN-1 彼此相关普遍 > 0.9，全部不用。
    # 只保留排列结构（离散）和斜率（方向），见设计文档 4.1。
    d["ma_align"] = ((ma5 > ma10).astype(float) + (ma10 > ma20).astype(float)
                     + (ma20 > ma60).astype(float))
    d["ma20_slope5"] = (ma20 - ma20.shift(5)) / c.replace(0, np.nan)
    above = (c > ma60).astype(float)
    # 连续站上 ma60 的天数，截断 60
    grp = (above != above.shift()).cumsum()
    d["above_ma60"] = (above.groupby(grp).cumsum() * above).clip(upper=60)

    # --- Q2 压缩 ---
    rng = (h - l) / c.replace(0, np.nan)
    d["range_compress"] = rng.rolling(5).mean() / rng.rolling(20).mean()
    d["vol_compress"] = v.rolling(5).mean() / v.rolling(20).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr14, atr60 = tr.rolling(14).mean(), tr.rolling(60).mean()
    d["atr_ratio"] = atr14 / atr60
    d["boll_width"] = c.rolling(20).std() * 4 / ma20

    # --- Q3 结构（量能部分，筹码在 add_chips 里加） ---
    d["vol_ratio5"] = v / v.rolling(5).mean()
    d["turn_pct"] = d["turnover"]
    d["turn_std20"] = (d["turnover"].rolling(20).std()
                       / d["turnover"].rolling(20).mean())

    # --- Q5 相对强度（个股部分，全市场部分在 add_market 里） ---
    d["ret20"] = c / c.shift(20) - 1
    d["ret60"] = c / c.shift(60) - 1
    return d


def add_chip_block(d: pd.DataFrame) -> pd.DataFrame:
    """筹码五个特征 + 集中度的变化方向。

    chip_conc_chg20 是本组的核心：**静态的集中度不如集中的过程有信息量**。
    一只票一直很集中，说明根本没人交易；正在从分散走向集中，才是有人在收。
    """
    d = add_chips(d)
    d = d.rename(columns={"chip_conc90": "chip_conc90"})
    d["chip_conc_chg20"] = d["chip_conc90"] - d["chip_conc90"].shift(20)
    return d


# ---------------------------------------------------------------
#  低频数据：股东人数
# ---------------------------------------------------------------
def holder_features(panel: pd.DataFrame,
                    holders: pd.DataFrame | None) -> pd.DataFrame:
    """把季度的股东户数对齐到日频。

    **披露滞后是这份数据最大的坑。** 2026-06-30 的股东户数要到 7 月中才
    公告，按报告期对齐的话模型在 7 月 1 日就用上了 —— 那是标准的前视偏差，
    而且极其隐蔽，因为数据本身没错，错的是时间。

    这里用「报告期 + 15 个自然日」作为可用日（法定披露期限的保守估计），
    再前向填充。宁可晚用几天，不可早用一天。
    """
    for c in ("gdhs_chg1", "gdhs_chg3", "gdhs_down_streak",
              "gdhs_stale_days"):
        panel[c] = np.nan
    if holders is None or not len(holders):
        return panel

    h = holders.copy()
    h["code"] = h["代码"].astype(str).str.zfill(6)
    h["period"] = pd.to_datetime(h["报告期"], format="%Y%m%d")
    h["avail"] = (h["period"] + pd.Timedelta(days=15)).dt.strftime("%Y-%m-%d")
    h["chg"] = pd.to_numeric(h["股东户数-增减比例"], errors="coerce") / 100.0
    h = h[["code", "avail", "chg"]].dropna().sort_values(["code", "avail"])
    h["chg3"] = h.groupby("code")["chg"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    down = (h["chg"] < 0).astype(int)
    h["streak"] = down.groupby(
        [h["code"], (down != down.groupby(h["code"]).shift()).cumsum()]
    ).cumsum() * down
    h["streak"] = h["streak"].clip(upper=6)

    out = []
    for code, g in panel.groupby("code", sort=False):
        hh = h[h["code"] == code]
        if not len(hh):
            out.append(g)
            continue
        g = g.sort_values("date")
        m = pd.merge_asof(
            g[["date"]].assign(_d=pd.to_datetime(g["date"])).sort_values("_d"),
            hh.assign(_d=pd.to_datetime(hh["avail"])).sort_values("_d"),
            on="_d", direction="backward")
        g = g.copy()
        g["gdhs_chg1"] = m["chg"].to_numpy()
        g["gdhs_chg3"] = m["chg3"].to_numpy()
        g["gdhs_down_streak"] = m["streak"].to_numpy()
        # 数据有多旧。不给这个特征，模型会把 5 天前刚披露的户数骤降
        # 和 80 天前的旧数据同等对待。
        g["gdhs_stale_days"] = (
            pd.to_datetime(g["date"]) - m["_d"].to_numpy()).dt.days.to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------
#  全市场层
# ---------------------------------------------------------------
def add_market(panel: pd.DataFrame) -> pd.DataFrame:
    """相对强度 + 市场环境。

    rs20/rs60 是横截面量（个股涨幅在全市场的分位），第 ① 层对它们幂等。
    mkt_* 是全市场共同变量，刻意不做百分位化，见 MARKET_WIDE。
    """
    panel["ret1"] = panel.groupby("code")["close"].pct_change()
    by_date = panel.groupby("date")
    panel["rs20"] = by_date["ret20"].rank(pct=True)
    panel["rs60"] = by_date["ret60"].rank(pct=True)
    panel["rs_accel"] = panel["rs20"] - panel["rs60"]
    breadth = by_date["ret1"].transform(lambda s: (s > 0).mean())
    panel["mkt_breadth"] = breadth
    mkt = panel.groupby("date")["ret1"].mean().rename("m")
    mkt5 = mkt.rolling(5).sum()
    panel["mkt_ret5"] = panel["date"].map(mkt5)
    return panel


# ---------------------------------------------------------------
#  ① 横截面百分位  ②中性化
# ---------------------------------------------------------------
def cross_section(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """① 每个特征换成它在当日全市场的分位排名。见模块 docstring。"""
    tgt = [c for c in cols if c not in MARKET_WIDE]
    panel[tgt] = panel.groupby("date")[tgt].rank(pct=True)
    return panel


def neutralize(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """② 对板块和市值中性化：减去同日同板块同市值档的均值。

    为什么不做逐日 OLS：417 万行上按日跑 800 次多元回归要几十分钟，
    而「分组去均值」是同一件事的离散版本，效果接近、代价是秒级。
    市值分 5 档已经足够吸收市值的单调效应。

    对应设计文档 2.5 的板块问题：北交所日涨停 30%，它的量价特征天然比
    主板极端。不中性化的话模型会把「是北交所」学成强特征，
    而那只是标签定义的副作用。
    """
    tgt = [c for c in cols if c not in MARKET_WIDE]
    panel["_mcap_bin"] = panel.groupby("date")["float_mcap"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False,
                          duplicates="drop"))
    g = panel.groupby(["date", "board", "_mcap_bin"], observed=True)
    panel[tgt] = panel[tgt] - g[tgt].transform("mean")
    return panel.drop(columns=["_mcap_bin"])


def add_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    """③ 三个有明确假设的交互项，见设计文档 4.2。

    五组两两组合能造出几百个交互项，那是特征工程最容易失控的地方。
    只留有假设的三个。其中第一个是股东人数这个特征的**真正用法**：
    单看户数变化噪音很大（股价跌了散户离场也会让户数减少），
    只有当它和筹码集中度同向时，才构成「有人在收集」的证据。
    """
    for a, b, name in INTERACTIONS:
        if a in panel.columns and b in panel.columns:
            panel[name] = panel[a] * panel[b]
    return panel


def feature_columns() -> list[str]:
    cols = [c for g in GROUPS.values() for c in g]
    return cols + [n for _, _, n in INTERACTIONS]
