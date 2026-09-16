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
  3. 候选池。入池四条规则（昨日涨停 / 昨涨>=5% / 昨日换手>=5% /
     昨日成交额排名<=600）和生产共用 premarket.include_mask，阈值都来自
     config。换手率来自新浪日线店 data/breakout/daily.parquet（回填窗口
     覆盖率 99.96%），总市值仍然拿不到（线上也拿不到，两边都不筛）。
  4. ST 判定。历史股票名只有「今天的名字」（cache/names.csv，缺失时退到
     data/breakout/holders.parquet），改过名的票会错判。拿不到名字就拒绝
     出表，不再静默把 ST 留在池里。
  5. 竞价只有一个点：撮合价。stk_auction_o 的 bar 是「09:25 撮合价 ->
     09:30 之后」，不是 09:15~09:25：实测 open == 日线开盘 99.4%，而
     close 只有 56%，high>open 38.8%、low<open 36.4%（219 万行，2026-09-16）。
     所以只取 open 当撮合价，close/high/low/vwap 一律不进特征 ——
     以前拿它们当 T1/T2/T3，slope 算的其实是**开盘后的涨跌**，
     和标签 r=close/open-1 相关 +0.15，是前视泄漏。
     现在 t1/t2/t3 一律等于撮合价、slope=dive=0、monotonic=False，
     和线上 traj_ok=False 时同一口径（run_auction.build_features）。
     后果：trend 维度在回填数据上不可学，sources.learnable_dims() 已经
     把它排除，箱约束要用 sources.restrict_box() 裁过再交给优化器。
  6. 假涨停规则在回填表上不可观测。竞价段拿不到 09:15~09:20 撤单前的虚拟
     撮合冲高，回填 4/407621 触发、线上 281/21036（准入后 0% vs 5.1%）。
     这批票在回填里混在合格样本中，r 均值低约 1.7 个百分点，且没有任何一列
     能识别它们。
  7. auc_amount 仍然含开盘后成交（stk_auction_o 的 bar 越过了撮合）：
     与线上 09:25:10 真值比中位高 25%~33%，auc_ratio 跟着偏大约 23%，
     量能准入两边判定不一致 16.6%。所以 volume 维度也不可学
     （completeness 里 _o 算 "H" 不算 "FULL"）。要修干净得换 stk_auction
     （非 _o）拿纯竞价额，换之前必须先用重叠日实测 median(比值) 落在
     [0.98, 1.02]。
  8. 除权日没有复权因子。日线三路里只有东财那路的 chg 是复权真值
     （datasource 现在用 `chg_adj` 这一列标出来），腾讯/新浪给的是相邻
     **原始**收盘价之比，拿它反解昨收是恒等式、等于没修 ——
     cache/hist_daily.parquet 的 2188865 个非首行 100% 是不复权比值。
     所以只有 chg_adj=True 的行才反解昨收；chg_adj=False 的行按
     「涨跌幅绝对值超过涨跌停幅度 0.5pp」判出除权日并把那一行丢掉
     （以前留下 236 行 gap_pct 低到 -66.76% 的不可能样本）；
     现金分红那种 0.5%~5% 的小除息判不出来，约 0.1% 的行 gap_pct 偏低
     （11443 个除权事件里 1805 个落在训练表内，gap_pct 误差中位 0.66pp、
     P90 5.07pp，按 2~5% 准入有 55 行判定翻转）。要修干净得换一个带
     「除权后昨收」的源（Tushare pro.daily 的 pre_close），换之前先查文档。
     判据用的涨跌停幅度是「今天的名字」算出来的，所以最近才被 ST 的票
     会多判出几个假除权日（那几行被丢掉，不会变成脏样本）。
  9. one_word 的定义和线上不同。回填按「gap_pct >= limit_pct − 0.3」判，
     线上 run_auction.py 判「|撮合价 − 涨停价| < 0.005」。回填的 prev_close
     是相邻收盘价（97% 的行不是两位小数），线上那种精确匹配复刻不了。
     407621 行实测两边分歧 455 行（0.11%），gap_pct 全部 >= 9.71，
     而准入上限 gap_pct_max=5 把它们**整体遮蔽**，vscore.hard_reject 逐行
     结果相同。gap_pct_max 若被抬到 9.7 以上，必须先统一这两个定义
     ——selftest_train.check_one_word_masked 有断言钉住这条边界。
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


def _limit_pct_by_code(code: str, name: str = "") -> float:
    """涨停幅度。只留一份实现：直接调生产那份 datasource.limit_pct。

    以前这里自己写了一张前缀表（30/68 -> 20，83/87/88/92/43 -> 30），
    和生产的首字符判据在现有代码表上恰好一只都不差，但 689xxx 这类
    两边会分叉，而且 ST 的 5% 只有生产那份认。同一个量两份实现迟早漂
    （硬约束 9 是同一个道理）。
    """
    import datasource as ds
    return ds.limit_pct(code, name)


def load_names() -> dict[str, str]:
    """当前在市股票的名字。只用于 ST 排除和涨停幅度。

    cache/names.csv 是 refresh_names() 写的，但全仓没有任何调用方，实际
    一直不存在 -> names={} -> 训练表 name 列 407621 行全空 -> exclude_st
    空转，11514 行 ST（2.82%）留在池里，每天前 10 里 2.7% 的席位是生产
    永远不会打分的票（2026-09-16 审计）。所以加一条离线兜底：
    data/breakout/holders.parquet 带东财的「代码/名称」（5429 只）。
    两份都是「今天的名字」，改过名的票仍然会错判（已知偏差 4）。
    """
    p = CACHE / "names.csv"
    if p.exists():
        d = pd.read_csv(p, dtype=str)
        return dict(zip(d["code"], d["name"]))
    hp = ROOT / "data" / "breakout" / "holders.parquet"
    if hp.exists():
        d = pd.read_parquet(hp, columns=["代码", "名称"])
        d = d.dropna(subset=["代码", "名称"]).drop_duplicates("代码")
        return {str(k).zfill(6): str(v) for k, v in zip(d["代码"], d["名称"])}
    return {}


def require_names(names: dict[str, str], c: dict) -> None:
    """拿不到股票名就别出表。

    names 为空时 exclude_st 空转、limit_pct 的 ST 5% 也判不出来：
    训练表里混进 11514 行 ST（2.82%），每天前 10 里 2.7% 的席位是生产
    永远不会打分的票，这些席位的 r 均值 -0.35% 而前 10 整体 +0.50%。
    静默跑完等于没跑（教训 16）。
    """
    if not names and c["universe"].get("exclude_st"):
        raise RuntimeError("拿不到股票名，exclude_st 执行不了，拒绝落训练表")


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
def attach_daily_truth(h: pd.DataFrame, real: pd.DataFrame | None
                       ) -> pd.DataFrame:
    """把新浪日线店的**真实成交额和换手率**并进 hist_daily。

    两件事都是「回测口径 != 生产口径」（教训 30）：

    成交额：cache/hist_daily.parquet 的成交额 99.9993% 是估算
      （腾讯/新浪 K 线不给成交额，datasource 按 (高+低+收)/3×量 估），
      而线上 prev_amount 是腾讯快照 index 37 的真值。估算不是白噪声，
      方向固定在强势票上：昨日涨停那一子群旧口径（收盘×量）中位偏高 2.79%。
      它是 auc_ratio 的分母，也就是量能准入判据 —— 两种分母下 vscore 的
      准入集有 2.2% 行翻转，404 天里 151 天模拟前 10 不同（2026-09-16 实测）。

    换手率：生产入池四条规则里有「昨日换手 >= 5%」，回填以前根本没有这一列，
      只能用一条生产没有的 amount_ratio_5d 顶替，池子日均只重合 70%。

    real 只取 amount / turnover 两列。那张表的 close 是前复权、volume 未复权，
    **不要顺手用它的价量**。
    """
    h = h.copy()
    if real is None or real.empty:
        # 一行真值都没有：整张表都是估算，标出来而不是假装它是真值
        h["amount_est"] = True
        return h
    r = real.rename(columns={"date": "日期", "amount": "_amt_real",
                             "turnover": "_to_real"}).copy()
    r["日期"] = r["日期"].astype(str).str[:10]
    r["code"] = r["code"].astype(str).str.zfill(6)
    r = r.drop_duplicates(["日期", "code"])
    h["日期"] = h["日期"].astype(str).str[:10]
    h["code"] = h["code"].astype(str).str.zfill(6)
    h = h.merge(r, on=["日期", "code"], how="left")
    cov = float((h["_amt_real"] > 0).mean())
    # 哪几行没接上真值要留痕：实测覆盖 99.97%，但「几乎全覆盖」和
    # 「全覆盖」是两回事，落进训练表的标记让后面能把估算行单独挑出来核对
    h["amount_est"] = ~(h["_amt_real"] > 0)
    h["成交额"] = h["_amt_real"].where(h["_amt_real"] > 0, h["成交额"])
    # 新浪的 turnover 是小数（0.05 = 5%），入池规则按百分点判
    h["换手率"] = pd.to_numeric(h["_to_real"], errors="coerce") * 100.0
    log.info("成交额真值覆盖 %.4f，其余保留 (高+低+收)/3×量 的估算", cov)
    return h.drop(columns=["_amt_real", "_to_real"])


def daily_features(h: pd.DataFrame, breakout_lookback: int = 20,
                   names: dict[str, str] | None = None) -> pd.DataFrame:
    """按 code 分组做滚动计算。2.2M 行用 groupby.transform，秒级。

    所有「昨日」类特征都要 shift(1)：竞价时点能看到的只有昨天收盘为止的信息。
    漏 shift 就是未来函数，回测会漂亮得不真实。
    """
    names = names or {}
    d = h.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                          "最高": "high", "最低": "low",
                          "成交量": "vol", "成交额": "amount",
                          "换手率": "turnover_pct",
                          "涨跌幅": "chg"}).copy()
    d["date"] = d["date"].astype(str).str[:10]
    if "turnover_pct" not in d.columns:
        d["turnover_pct"] = np.nan
    # 成交额有没有接上真值。没经过 attach_daily_truth 的表一律算估算，
    # 不许默认成 False 把估算值说成真值
    if "amount_est" not in d.columns:
        d["amount_est"] = True
    d["amount_est"] = d["amount_est"].fillna(True).astype(bool)
    for c in ("open", "close", "high", "low", "vol", "amount", "chg",
              "turnover_pct"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("code", sort=False)

    # 涨停幅度要在除权判据之前算出来（判据按跌停幅度定），ST 的 5% 靠 names
    d["lim"] = [_limit_pct_by_code(c, names.get(c, ""))
                for c in d["code"]]

    # 除权日：不复权收盘价在除权日跳空。**只有 chg_adj=True 的源**（东财）
    # 的 chg 是复权涨跌幅，才能拿它反解出「已除权的昨收」；腾讯/新浪那两路
    # 的 chg 就是相邻**原始**收盘价之比（datasource 自己写着「除权日会错」），
    # 实测 219 万行里 |chg − close/shift(close)+1| > 0.05 的是 0 行 ——
    # 2026-09-16 前无条件反解，对 100% 的数据是恒等变换，等于没修。
    # 不复权那两路只能退化处理：涨跌幅绝对值超过涨跌停幅度 0.5pp 判为除权日，
    # 昨收置 NaN -> gap_pct NaN -> 落盘前整行丢掉，而不是留下 236 行
    # gap_pct 低到 -66.76% 的假样本。
    # 判据不用「最高价 < 昨收×(1-lim)」：实测会把 -10.06% 的一字跌停
    # 误判成除权（831 行里 200 行）。
    adj = (d["chg_adj"].fillna(False).astype(bool) if "chg_adj" in d.columns
           else pd.Series(False, index=d.index))
    raw_prev = g["close"].shift(1)
    d["prev_close"] = np.where(
        adj & d["chg"].notna() & (d["chg"] > -100.0),
        d["close"] / (1.0 + d["chg"] / 100.0), raw_prev)
    d["exdiv"] = (raw_prev.notna() & ~adj
                  & (d["chg"].abs() > d["lim"] + 0.5)).fillna(False)
    d.loc[d["exdiv"], "prev_close"] = np.nan

    d["prev_amount"] = g["amount"].shift(1)
    # prev_amount 是**昨天**的成交额，它是不是估算值的标记也要跟着 shift(1)
    d["prev_amount_est"] = (g["amount_est"].shift(1)
                            .fillna(True).astype(bool))
    d["prev_vol"] = g["vol"].shift(1)
    d["prev_turnover_pct"] = g["turnover_pct"].shift(1)
    d["prev_gain"] = g["chg"].shift(1)
    # 除权次日的 prev_gain 是那个假跌幅（例 -66.94），会被「昨日跌停剔除」
    # 整行丢掉：698 个除权日的次日进训练表的是 0 行，其中 117 行昨日成交额
    # 排名 <=600，线上用复权 chg 会照常入池。置 NaN 让它留下。
    d.loc[g["exdiv"].shift(1).fillna(False).astype(bool), "prev_gain"] = np.nan
    d["prev_close_1"] = g["close"].shift(1)
    d["prev_high"] = g["high"].shift(1)
    d["prev_close_2"] = g["close"].shift(2)

    # 窗口内第几根日线 + 首根日期：min_listed_days 靠它执行（次新剔除）。
    # 不能裸用 cumcount：窗口起点之前就上市的老票头 59 行会被误杀。
    d["hist_i"] = g.cumcount()
    d["first_date"] = g["date"].transform("min")

    lu = d["chg"] >= d["lim"] - 0.4
    d["is_limit_up"] = lu
    d["prev_limit_up"] = g["is_limit_up"].shift(1).fillna(False).astype(bool)

    # 连板高度：截至昨天为止的连续涨停根数。
    # 断点分组 + 组内累计，比逐行回溯快两个数量级。
    brk = (~lu).groupby(d["code"]).cumsum()
    d["streak"] = lu.groupby([d["code"], brk]).cumsum()
    d["board_height"] = g["streak"].shift(1).fillna(0).astype(int)

    # 昨日炸板：昨天盘中触涨停但收盘没封。涨停价走 datasource.limit_price_arr
    # （四舍五入到分），和生产 premarket.stage2 同一份公式。以前两边都写
    # `昨收×(1+lim/100) − 0.01`，把阈值整体放宽一分：428 万行实测多判
    # 3935 行（15.5%），全部是「最高价 = 涨停价 − 0.01」，训练表这一列的
    # 正例里每 6 个就有 1 个是假的，影子模型（权重 −0.048）在假标签上学
    # 传 code：北交所封板价向下取整到分，四舍五入会高 0.01
    import datasource as _ds
    d["prev_broken_board"] = (
        (~d["prev_limit_up"])
        & (d["prev_high"] >= _ds.limit_price_arr(d["prev_close_2"], d["lim"],
                                                 d["code"]))
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
    # 昨日额 / **不含昨日**的前 5 日均额。分母以前写成 shift(1).rolling(5)，
    # 把昨日自己也算进去了，比值被数学上压死在 5 以内（实测上限 4.968），
    # 而生产 premarket.stage2 的 am.tail(6).iloc[:-1] 是不含昨日的。
    # 这一列只作展示和口径核对：它进不了 COLS，也不再是入池条件
    # （入池走 premarket.include_mask，和生产同一份）。
    a5 = g["amount"].transform(lambda s: s.shift(2).rolling(5, min_periods=5).mean())
    d["amount_ratio_5d"] = np.where(a5 > 0, d["prev_amount"] / a5, 1.0)

    # 标签：开盘买、收盘卖。开盘价按定义就是竞价撮合价。
    d["r"] = np.where((d["open"] > 0) & (d["close"] > 0),
                      d["close"] / d["open"] - 1.0, np.nan)
    return d


# ---------------------------------------------------------------------
#  竞价段
# ---------------------------------------------------------------------
def merge_auction(d: pd.DataFrame, auc: pd.DataFrame | None) -> pd.DataFrame:
    """接上竞价撮合价和竞价成交额。

    **只取 open 和 amount。** stk_auction_o 的 bar 不是 09:15~09:25 的竞价段，
    而是「09:25 撮合价 -> 09:30 之后」的一根早盘 K 线（模块 docstring 已知
    偏差 5）：集合竞价只有一个成交价，纯竞价 bar 必然 high==low==open==close，
    而实测 high==low 只占 45.5%；再和日线对表，open == 日线开盘 99.4%，
    close 只有 56%。第三方交叉验证也一致：线上 09:25:10 的真采样
    （data/2026-0*/auction_*.parquet）== 日线开盘 99.7%、== ts.open 99.6%、
    == ts.close 只有 42.7%。
    所以 close/high/low/vwap 一个都不 merge 进来 —— 留着就会有人再顺手用，
    而用它们算出来的 slope/dive 就是开盘后的涨跌，直接是标签的一部分。

    没有竞价数据时（免费源）两列留 NaN，下游据此把 volume 维度排除出
    可学集合。
    """
    if auc is None or auc.empty:
        for c in ("auc_price", "auc_amount"):
            d[c] = np.nan
        return d
    a = auc.copy()
    a["code"] = a["ts_code"].str.slice(0, 6)
    a["date"] = (a["trade_date"].astype(str).str.slice(0, 4) + "-"
                 + a["trade_date"].astype(str).str.slice(4, 6) + "-"
                 + a["trade_date"].astype(str).str.slice(6, 8))
    a = a.rename(columns={"open": "auc_price", "amount": "auc_amount"})
    cols = ["code", "date", "auc_price", "auc_amount"]
    return d.merge(a[[c for c in cols if c in a.columns]],
                   on=["code", "date"], how="left")


def check_auction_price(d: pd.DataFrame, rtol: float = 0.005,
                        max_bad: float = 0.01) -> float:
    """撮合价必须等于当日日线开盘价 —— 把注释里的「两者本该相等」变成断言。

    这条是数据源口径变没变的哨兵：实测不一致率 0.23%（219 万行，rtol 0.5%），
    而按旧口径（ts.close）是 41.6%。返回不一致占比，超过 max_bad 就抛。
    """
    m = d["auc_price"].notna() & (d["open"] > 0)
    if not bool(m.any()):
        return 0.0
    bad = float((~np.isclose(d.loc[m, "auc_price"].to_numpy(float),
                             d.loc[m, "open"].to_numpy(float),
                             rtol=rtol, atol=0.001)).mean())
    if bad > max_bad:
        raise ValueError(
            f"竞价撮合价与日线开盘价不一致 {bad:.1%}（上限 {max_bad:.1%}），"
            "数据源口径变了：stk_auction_o 只有 open 是撮合价")
    return bad


def to_features(d: pd.DataFrame, sector: dict[str, str],
                names: dict[str, str]) -> pd.DataFrame:
    """拼出 AuctionFeature 的全部列。"""
    pc = d["prev_close"]
    # 竞价撮合价：有竞价数据用它，没有就用日线开盘（两者本该相等）
    d["auc_price"] = d["auc_price"].fillna(d["open"])
    # 买入价（日线开盘）和特征描述的价（撮合价）偏离多少。口径逐字照抄在线
    # learn/labels.py::build：|open − auc_price| / auc_price × 100，两边都没有
    # 就留 NaN。**必须带 .abs()**：不带的话 open 低于撮合价的那一半行是负数，
    # 下游 `> 阈值` 判不脏，2026-09-16 审计实测会漏掉六成脏行。
    # 有这一列，eval_daily._bf_dirty 才能和在线判同一条规则（教训 30）；
    # 缺列时它只判一字板并打 warning，两边可用样本不是同一批。
    mis = (d["auc_price"] > 0) & (d["open"] > 0)
    d["open_mismatch_pct"] = ((d["open"] - d["auc_price"]).abs()
                              / d["auc_price"] * 100).where(mis)
    d["gap_pct"] = np.where(pc > 0, (d["auc_price"] / pc - 1.0) * 100, np.nan)

    # 涨停幅度（含 ST 的 5%）已经在 daily_features 里按 datasource.limit_pct
    # 算好了，这里不再自己判一次 ST：两份实现会漂。
    d["limit_pct"] = d["lim"].astype(float)
    d["gap_norm"] = d["gap_pct"] / d["limit_pct"]
    d["auc_ratio"] = np.where(d["prev_amount"] > 0,
                              d["auc_amount"] / d["prev_amount"], np.nan)

    # 竞价轨迹：历史上**取不到**。线上 T1/T2/T3 是 09:19:40 / 09:23:30 /
    # 09:25:10 的精确采样，stk_auction_o 只给得了 09:25 撮合价这一个点。
    # 以前拿竞价段的 开盘/vwap/收盘 顶替，实测代理值和线上真值没有关系：
    # corr(slope) -0.03、corr(dive) +0.008、monotonic 一致率 50%（抛硬币），
    # 尾盘跳水 >=2pp 线上 4.53% vs 回填 0.126%；而它和当日 open->close 收益
    # 相关 +0.15 —— 因为它算的就是开盘后的涨跌，是标签的一部分。
    # 宁可给中性值也不给假信号：这和 run_auction.build_features 里
    # traj_ok=False 时的生产口径完全一致（f_trend 恒 0.5，假涨停和跳水
    # 两条规则都不触发）。
    d["t3_chg"] = d["gap_pct"]
    d["t1_chg"] = d["gap_pct"]
    d["t2_chg"] = d["gap_pct"]
    d["slope"] = 0.0
    d["dive"] = 0.0
    # 线上 monotonic = traj_ok and (...)，没有轨迹证据时是 False。
    # 写 True 会给每一行白送趋势分（教训 9 同型）
    d["monotonic"] = False

    d["breakout"] = (d["auc_price"] > d["platform_high"]).fillna(False)
    d["sector"] = d["code"].map(sector).fillna("未分类")
    d["name"] = d["code"].map(names).fillna("")
    d["blacklisted"] = False       # 历史公告拿不到，一律 False（已知偏差）
    # 一字板：撮合价就等于涨停价。线上判的是 |撮合价 − 涨停价| < 0.005
    # （run_auction.py），不看撮合之后的走势，所以这里也只按撮合价判 ——
    # 以前还要求「段内最低价 >= 撮合价」，而那个最低价是开盘后的
    d["one_word"] = (d["gap_pct"] >= d["limit_pct"] - 0.3).fillna(False)
    return d


def build_pool(d: pd.DataFrame, c: dict) -> pd.DataFrame:
    """按天重建候选池。入池规则和生产共用 premarket.include_mask。

    剔除的顺序也要和生产 stage1 一致：先按名字/板块/停牌剔，**再**排成交额
    名次。以前这里在全量上排名，而生产是在过滤之后排，同一条「前 600」
    在两边选出的不是同一批票。
    """
    # 延迟导入：premarket 在模块级 basicConfig，学习线不该被它改掉日志格式
    from premarket import include_mask
    import datasource as ds

    u = c["universe"]
    inc = u["include_if"]
    if u.get("exclude_st"):
        d = d[~d["name"].map(ds.is_st_name)].copy()
    if u.get("min_listed_days"):
        # 名字层面的次新兜底，和 premarket.stage1 同一判据
        d = d[~d["name"].map(ds.is_new_listing)].copy()
        # 真判据：窗口内第几根日线。只对「窗口内才上市」的票生效 ——
        # 裸用 cumcount 会把窗口起点之前就上市的老票头 59 行一起砍掉
        # （实测会丢 15% 的表），而 first_date 晚于全表起点才是真上市
        ipo = d["first_date"] > d["date"].min()
        d = d[~(ipo & (d["hist_i"] < int(u["min_listed_days"])))]
    if u.get("exclude_bj"):
        # 判据和生产 stage1 一字不差：北交所老段 8/4 开头 + 920xxx 新段
        d = d[~d["code"].str[0].isin(["8", "4", "9"])]
    # 停牌：生产用「昨日成交额 > 0」判
    d = d[d["prev_amount"].fillna(0) > 0].copy()

    rank = d.groupby("date")["prev_amount"].rank(ascending=False)
    keep = include_mask(d["prev_limit_up"], d["prev_gain"],
                        d["prev_turnover_pct"], rank, inc)
    d = d[keep].copy()
    if u.get("exclude_yesterday_limit_down"):
        # `~le` 而不是 `gt`：NaN 要留下。除权次日的 prev_gain 被置成 NaN
        # （daily_features），用 gt 会把那批行连同生产照常入池的票一起丢掉
        d = d[~d["prev_gain"].le(u["yesterday_drop_pct_max"])]
    d = d[d["prev_close"].gt(0)].copy()
    # 每天按成交额取前 max_candidates
    d["_rk"] = d.groupby("date")["prev_amount"].rank(ascending=False)
    return d[d["_rk"] <= u["max_candidates"]].drop(columns="_rk")


def add_sector_stats(d: pd.DataFrame, c: dict) -> pd.DataFrame:
    """板块共振：池内同板块只数、板块昨日涨停家数。

    「未分类」是占位符，必须排除在计数外——它会被 f_sector 当成一个
    几百只成分的巨型板块，白送满分（CLAUDE.md 历史教训第 9 条）。
    """
    real = d["sector"].ne("未分类")
    # 口径必须和线上一样：run_auction.build_features 数的是**过了初筛**
    # （涨幅区间、量能区间、成交额下限）的同板块只数，不是整个候选池。
    # 以前这里数整个池子：回填里 94.6% 的行 f_sector 直接饱和到 1.0，
    # 线上只有 26.9%，优化器看到的板块维度几乎是常数，学出来的权重对
    # 生产打分器没有意义（2026-09-15 审计）。
    sc = c["screen"]
    prelim = (real
              & d["gap_pct"].between(sc["gap_pct_min"], sc["gap_pct_max"])
              & d["auc_ratio"].between(sc["auc_ratio_min"], sc["auc_ratio_max"])
              & (d["auc_amount"] >= sc["min_auc_amount_wan"] * 1e4))
    cnt = (d.assign(_p=prelim.astype(int))
            .groupby(["date", "sector"])["_p"].transform("sum"))
    d["sector_members"] = np.where(real, cnt, 0)
    lu = d.groupby(["date", "sector"])["prev_limit_up"].transform("sum")
    d["sector_prev_limitups"] = np.where(real, lu, 0)
    return d


COLS = ["date", "code", "name", "limit_pct", "prev_close", "auc_price",
        "gap_pct", "gap_norm", "auc_amount", "prev_amount", "auc_ratio",
        "t1_chg", "t2_chg", "t3_chg", "slope", "monotonic", "dive",
        "pos_pct_60d", "ma_bull", "breakout", "prev_limit_up",
        "prev_broken_board", "board_height", "sector", "sector_members",
        "sector_prev_limitups", "blacklisted", "one_word",
        # auc_ratio 的分母是不是估算值（实测 0.03% 的行接不上新浪真值）。
        # 有这一列才能在复盘时把估算行单独挑出来，而不是整表当成同一口径
        "prev_amount_est",
        # 买入价与撮合价的偏离（在线 labels.build 同名同口径）。
        # eval_daily._bf_dirty 按它判脏；不入表的话回填和在线的可用样本
        # 口径不一致，而这件事不报错（审计 F8-14）
        "open_mismatch_pct",
        "cauc_ratio_prev", "r"]


def merge_cauc(d: pd.DataFrame, cauc: pd.DataFrame | None) -> pd.DataFrame:
    """昨日尾盘集合竞价占比。stk_auction_c（14:57-15:00）是机构收盘
    定价行为，「昨天尾盘有人大额定价」对次日早盘有信息。
    没有数据（免费源）就整列 NaN，秩归一后填 0 = 中性。
    """
    if cauc is None or cauc.empty:
        d["cauc_ratio_prev"] = np.nan
        return d
    a = cauc.copy()
    a["code"] = a["ts_code"].str.slice(0, 6)
    a["date"] = (a["trade_date"].astype(str).str.slice(0, 4) + "-"
                 + a["trade_date"].astype(str).str.slice(4, 6) + "-"
                 + a["trade_date"].astype(str).str.slice(6, 8))
    a = a[["code", "date", "amount"]].rename(columns={"amount": "_cauc_amt"})
    d = d.merge(a, on=["code", "date"], how="left")
    g = d.sort_values(["code", "date"], kind="mergesort").groupby("code",
                                                                  sort=False)
    d["_cauc_prev"] = g["_cauc_amt"].shift(1)
    d["cauc_ratio_prev"] = np.where(d["prev_amount"] > 0,
                                    d["_cauc_prev"] / d["prev_amount"], np.nan)
    return d.drop(columns=["_cauc_amt", "_cauc_prev"])


def check_table(res: pd.DataFrame) -> None:
    """落盘前的硬不变量。三条都真的红过，红了宁可不出表（教训 16/26）。"""
    bad = int((res["gap_pct"].abs() > res["limit_pct"] + 1e-6).sum())
    if bad:
        raise ValueError(f"{bad} 行高开幅度超过涨跌停幅度，多半是除权日没剔干净")
    if not bool((res["slope"] == 0).all()) or bool(res["monotonic"].any()):
        raise ValueError("回填表里出现非零 slope / monotonic=True，"
                         "说明撮合之后的价格又混进了竞价轨迹")


def build(c: dict, out: Path | None = None) -> Path:
    hp, ap = CACHE / "hist_daily.parquet", CACHE / "hist_auction.parquet"
    cp = CACHE / "hist_auction_c.parquet"
    if not hp.exists():
        raise FileNotFoundError("先跑 --stage backfill 拉日线")
    h = pd.read_parquet(hp)
    auc = pd.read_parquet(ap) if ap.exists() else None
    cauc = pd.read_parquet(cp) if cp.exists() else None
    sector = dict(pd.read_parquet(CACHE / "sector_map.parquet")
                  [["code", "sector"]].values)
    names = load_names()
    require_names(names, c)

    rp = ROOT / "data" / "breakout" / "daily.parquet"
    real = (pd.read_parquet(rp, columns=["date", "code", "amount", "turnover"])
            if rp.exists() else None)
    if real is None:
        raise FileNotFoundError(
            "缺 data/breakout/daily.parquet：成交额只有估算值、换手率一列都没有，"
            "入池规则和 auc_ratio 的分母都会和生产不是一个口径")
    h = attach_daily_truth(h, real)

    log.info("日线 %d 行 %d 只；竞价 %s", len(h), h["code"].nunique(),
             f"{len(auc)} 行" if auc is not None else "无（免费源）")
    d = daily_features(h, c["screen"]["breakout_lookback"], names)
    d = merge_auction(d, auc)
    log.info("撮合价与日线开盘不一致 %.4f", check_auction_price(d))
    d = merge_cauc(d, cauc)
    d = to_features(d, sector, names)
    d = build_pool(d, c)
    d = add_sector_stats(d, c)
    d = d[d["gap_pct"].notna() & d["r"].notna()]

    out = out or (TRAIN / f"backfill_{d['date'].min()}_{d['date'].max()}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = d[COLS].reset_index(drop=True)
    check_table(res)
    res.to_parquet(out, index=False)
    log.info("训练表落盘 %s：%d 行，%d 天，日均 %d 只", out.name, len(res),
             res["date"].nunique(), len(res) // max(res["date"].nunique(), 1))
    return out
