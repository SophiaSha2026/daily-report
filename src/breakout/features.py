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
    # 成交量组（2026-09-15 用户要求加成交量）。此前成交量只以 vol_compress /
    # vol_ratio5 两个「相对自己」的比值出现，没有量的**方向**和**水平**：
    #   vol_ratio20     今天 / 20 日均量，放量是否超出近一个月的常态
    #   up_vol_share20  20 日内上涨日成交量占比，量价配合 —— 资金流入的代理
    #                   （东财资金流只有 120 天，够不着三年训练集）
    #   obv_slope20     20 日净签名成交量 / 20 日总量，吸筹还是派发
    #   amt_ma20        20 日均成交额（对数），流动性水平；横截面分位后就是
    #                   「今天全市场里它排多活跃」
    "volume": ["vol_ratio20", "up_vol_share20", "obv_slope20", "amt_ma20"],
    # 股东人数组。gdhs_level 是 2026-09-15 加的：户数 / 流通股本，散户密度，
    # 静态水平；原来四个都是变化量。
    "holders": ["gdhs_chg1", "gdhs_chg3", "gdhs_down_streak",
                "gdhs_stale_days", "gdhs_level"],
    "regime": ["rs20", "rs60", "rs_accel", "mkt_breadth", "mkt_ret5"],
}
# 上市前的股东名册不是二级市场散户。招股书/新三板阶段的名册最多一千二百户
# （2026-09 实测 holders.parquet：920175 上次 1214 户、920186 1116 户），
# 上市后真实户数最少 1598 户（920033），所以 1500 这条线在现有数据上零误杀，
# 而 500 会漏掉 6 只北交所。低于它的行是发行前数据，整行不要。
HOLDER_MIN = 1500
# 全市场共同变量：当天所有股票同值，做横截面排名会退化成常数，
# 所以刻意排除在 ① 之外。它们的作用正是给模型一个「今天什么天气」的坐标。
MARKET_WIDE = {"mkt_breadth", "mkt_ret5"}

# (a, sa, b, sb, name)：sa/sb 是假设指向的那一侧，+1 取高位、-1 取低位。
# chip_conc90 越小越集中，所以「集中」是 -1。
# 第一版写成 a * b，而 a、b 都是中性化之后**以 0 为中心**的量：(-)(+) 和
# (+)(-) 同为负、(-)(-) 和 (+)(+) 同为正，假设象限和它的镜像象限拿到同一个值。
# 2026-09-16 在 428 万行训练表上实测 x_light_top：获利盘高&远前高 y_up 7.91%
# vs 获利盘低&近前高 2.31%（差 3.4 倍）被合并成同一个负值；pooled Spearman
# 从方向保持写法的 -0.0217 被乘成 -0.0047，能排进保留特征前 5 的信号被乘成噪音。
INTERACTIONS = [
    ("gdhs_chg1", -1, "chip_conc_chg20", -1, "x_holder_chip"),    # 户数减 且 成本区间收窄
    ("range_compress", -1, "vol_ratio5", +1, "x_squeeze_burst"),  # 区间收敛 且 今天放量
    ("chip_win", -1, "dist_52w_high", +1, "x_light_top"),         # 获利盘低 且 近前高
]
INTERACTION_VERSION = 2   # 进 daily.feature_fingerprint：改公式就重训


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
    c, h, l = d["close"], d["high"], d["low"]
    # 量能一律用**换手率**，不用成交股数。daily.parquet 的价是前复权
    # （akshare adjust="qfq" 只除价格四列），volume / amount / outstanding_share
    # 都是原值：送转日成交股数按股本比例整段跳升而价格被除回去，量价口径不一致。
    # 实测每年约 290 次送转（集中在 5~7 月），10 送 10 之后 20 天里
    # vol_ratio20 从 2.0 衰减到 1.05 —— 换手率恒定也能凭空造出一段「放量」。
    # 换手率 = 成交股数 / 当日流通股本，两边同步更新，是唯一在送转前后可比的量。
    # 腾讯兜底源没有换手率时 build.attach_turnover 已经合成过一列。
    vol_raw = d["volume"]
    v = (d["turnover"] if "turnover" in d.columns and d["turnover"].notna().any()
         else vol_raw)

    # --- Q1 位置 ---
    ma5, ma10 = c.rolling(5).mean(), c.rolling(10).mean()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    d["dist_52w_high"] = c / c.rolling(250, min_periods=60).max() - 1
    d["dist_52w_low"] = c / c.rolling(250, min_periods=60).min() - 1
    # 四个均线乖离 close/maN-1 彼此相关普遍 > 0.9，全部不用。
    # 只保留排列结构（离散）和斜率（方向），见设计文档 4.1。
    # ma60 没定义的前 59 行必须是 NaN，不能是 0。`x > NaN` 是 False，
    # astype(float) 落成 0，看着像「均线全空头 / 从没站上 ma60」，而真相是
    # 「还不知道」。2026-09-16 实测：训练表里 22037 行是上市晚于面板起点的
    # 短历史票和成熟票同日排名，中性化后 above_ma60 中位 -0.171（95% 以上
    # 为负，钉在横截面底部），而这些行的 y_up 率 6.06% 是成熟行的 1.7 倍 ——
    # 模型拿到一条「均线垫底 <-> 次新 <-> 高命中」的假通道，而 above_ma60__mean
    # 是现役模型分裂次数第 4 的特征。NaN 一路穿过 rank/neutralize，
    # 到 model._xy 落成 0 = 组均值（中性），和 dist_52w_* 的暖机语义一致。
    d["ma_align"] = ((ma5 > ma10).astype(float) + (ma10 > ma20).astype(float)
                     + (ma20 > ma60).astype(float)).where(ma60.notna())
    d["ma20_slope5"] = (ma20 - ma20.shift(5)) / c.replace(0, np.nan)
    above = (c > ma60).astype(float)
    # 连续站上 ma60 的天数，截断 60
    grp = (above != above.shift()).cumsum()
    d["above_ma60"] = ((above.groupby(grp).cumsum() * above)
                       .clip(upper=60).where(ma60.notna()))

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

    # --- 成交量组 ---
    v20 = v.rolling(20).mean()
    d["vol_ratio20"] = v / v20.replace(0, np.nan)
    up = (c > c.shift()).astype(float)
    vsum20 = v.rolling(20).sum().replace(0, np.nan)
    d["up_vol_share20"] = (v * up).rolling(20).sum() / vsum20
    sign = np.sign(c.diff()).fillna(0.0)
    d["obv_slope20"] = (v * sign).rolling(20).sum() / vsum20
    amt = d["amount"] if "amount" in d.columns else vol_raw * c
    d["amt_ma20"] = np.log1p(amt.rolling(20).mean())

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

    可用日 = 东财表里的**公告日期**（2026-09-15 起）。第一版用「报告期 + 15
    个自然日」，实测 68208 条里 99.7% 的公告晚于这个日子：中位滞后 50 天，
    九成分位 115 天（年报里的户数要到次年四月底才公开）。也就是说模型
    平均提前一个多月「知道」了户数变化，而这一组特征的重要性排第二 ——
    之前验证集上的成绩里有一部分是这个偷看给的，见 breakout_log 实验 9。
    公告日期缺失的那行按报告期 + 120 天兜底，宁可晚用，不可早用。
    """
    for c in ("gdhs_chg1", "gdhs_chg3", "gdhs_down_streak",
              "gdhs_stale_days", "gdhs_level"):
        panel[c] = np.nan
    if holders is None or not len(holders):
        return panel

    h = holders.copy()
    h["code"] = h["代码"].astype(str).str.zfill(6)
    h["period"] = pd.to_datetime(h["报告期"], format="%Y%m%d")
    ann = pd.to_datetime(h.get("公告日期"), errors="coerce")
    late = h["period"] + pd.Timedelta(days=120)
    ann = ann.where(ann.notna() & (ann > h["period"]), late)
    h["avail"] = ann.dt.strftime("%Y-%m-%d")
    h["chg"] = pd.to_numeric(h["股东户数-增减比例"], errors="coerce") / 100.0
    h["cnt"] = pd.to_numeric(h.get("股东户数-本次"), errors="coerce")
    # 上市前的名册（几十户）整行丢；上市后第一期的「上次」还是招股书户数，
    # 东财算出来的增减比例是几万倍（实测中位 39220%、最大 1.63e8%，
    # 而正常行最大约 460%），置 NaN，户数本身保留给 gdhs_level。
    # 不这么做的话：当日 gdhs_chg3 前 1% 的行里 53.8% 是这种伪迹（y_up 10.49%
    # vs 同区间非伪迹 4.21%），模型学到的是「股东特征极端 = 次新 = 容易起涨」，
    # 最近 8 份清单 A 里有 4 份混进这样的票。
    prev = (pd.to_numeric(h["股东户数-上次"], errors="coerce")
            if "股东户数-上次" in h.columns
            else pd.Series(np.nan, index=h.index))
    h = h[h["cnt"] >= HOLDER_MIN].copy()
    h.loc[prev.reindex(h.index) < HOLDER_MIN, "chg"] = np.nan
    # 年报和一季报常同一天公告：同一 (code, avail) 两行，按报告期排在后面的
    # 才是新的一期。不加第二键的话，merge_asof 取到哪一行取决于排序算法
    # 碰巧稳不稳定。
    # dropna 按 cnt 不按 chg：上一行刚把上市后第一期的 chg 置成 NaN，
    # 按 chg 丢会把那一期整行丢掉，gdhs_level 跟着丢三个月。
    # chg 为 NaN 时 chg3 的 rolling(min_periods=1) 自动跳过、streak 判
    # chg<0 得 0，都安全。
    h = (h[["code", "avail", "period", "chg", "cnt"]].dropna(subset=["cnt"])
          .sort_values(["code", "avail", "period"]))
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
        # 右表的可用日单独留一列 ann_d：merge_asof 之后 _d 是左表的日期，
        # 拿它减自己永远是 0 —— 第一版的 gdhs_stale_days 就是这么变成常数的
        # （常数特征被筛掉，模型从没用上「数据有多旧」这个信息）。
        m = pd.merge_asof(
            g[["date"]].assign(_d=pd.to_datetime(g["date"])).sort_values("_d"),
            hh.assign(_d=pd.to_datetime(hh["avail"]),
                      ann_d=pd.to_datetime(hh["avail"])).sort_values("_d"),
            on="_d", direction="backward")
        g = g.copy()
        g["gdhs_chg1"] = m["chg"].to_numpy()
        g["gdhs_chg3"] = m["chg3"].to_numpy()
        g["gdhs_down_streak"] = m["streak"].to_numpy()
        # 散户密度：户数 / 流通股本。没有流通股本（腾讯兜底源）就 NaN
        os_ = (g["outstanding_share"].to_numpy(float)
               if "outstanding_share" in g.columns else np.full(len(g), np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            g["gdhs_level"] = np.where(os_ > 0, m["cnt"].to_numpy(float) / os_,
                                       np.nan)
        # 数据有多旧。不给这个特征，模型会把 5 天前刚披露的户数骤降
        # 和 80 天前的旧数据同等对待。
        g["gdhs_stale_days"] = (
            pd.to_datetime(g["date"]) - m["ann_d"].to_numpy()).dt.days.to_numpy()
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
    # 没有昨收的行（每只票在面板里的第一行）不进分母。`NaN > 0` 是 False，
    # 原来那句 `(s > 0).mean()` 把它们算成「没涨」：2026-09-16 实测面板首日
    # 2023-05-30 有 5039 行、其中 4151 行没有昨收，宽度被算成 0.098，
    # 真值 0.559 —— 凭空造出一个「全市场只有 9.8% 上涨」的假暴跌日，
    # 还通过 5 日窗口污染随后 5 天。下一行的 mean 本来就跳过 NaN，
    # 同一个函数里两个市场变量的口径必须一致。整日无有效 ret1 给 NaN。
    up = (panel["ret1"] > 0).astype(float).where(panel["ret1"].notna())
    panel["mkt_breadth"] = up.groupby(panel["date"]).transform("mean")
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

    用方向保持的模糊 AND（两个指向侧取 min），不用乘法：这两列在这一步
    已经过 ① 横截面百分位 + ② 中性化，都是以 0 为中心的有符号量，乘法
    分不出假设象限和它的镜像象限（见 INTERACTIONS 上面那段实测）。
    min 只在两个条件**同时**成立时为正，镜像象限为负，且对指向侧单调。
    """
    for a, sa, b, sb, name in INTERACTIONS:
        if a in panel.columns and b in panel.columns:
            panel[name] = np.minimum(sa * panel[a], sb * panel[b])
    return panel


def feature_columns() -> list[str]:
    cols = [c for g in GROUPS.values() for c in g]
    return cols + [n for *_, n in INTERACTIONS]
