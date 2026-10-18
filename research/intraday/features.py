"""步骤 3：从分时 / 分笔派生 T 日收盘可得的日内特征，和日线条件、标签拼成一张表。

    research/datasrc/.venv/Scripts/python.exe research/intraday/features.py

输入：raw/parts/{min,tx,meta}_*.parquet、research/altdata/top60.parquet（名次 / 标签）、
      research/polish pool.parquet（lim_pct lu_close yizi_today lu_streak cr1 gap1 turn1 amt_ratio）、
      research/screen px_screen.parquet（open high low close volume close_u，pit 价）。
输出：research/intraday/feat.parquet，一行一个 (code, date)。

分时 240 根：i 0..119 = 09:31..11:30（第 0 根含集合竞价），120..239 = 13:01..15:00。价是不复权原价。
分笔按分钟聚合：hm 09:25 是集合竞价，> 15:00 是科创 / 创业板盘后定价交易（不算进日内）。
涨停价 = round(昨收原价 x (1 + lim_pct/100), 2)，昨收原价 = px_screen 的 close_u 前一行；容 1 分。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARTS = HERE / "raw" / "parts"
EXP = ROOT / "data" / "breakout" / "raw" / "exp"


def say(*a):
    print(*a, flush=True)


def load_parts(prefix: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(PARTS / f"{prefix}_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["code"] = df["code"].astype(str)
    df["date"] = df["date"].astype(str)
    return df


def main():
    meta = load_parts("meta").drop_duplicates(["code", "date"], keep="last")
    mn = load_parts("min")
    tx = load_parts("tx")
    say("meta", len(meta), "ok", int(meta["ok"].sum()), "分时行", len(mn), "分笔分钟行", len(tx))
    keys = meta.loc[meta["ok"] & (meta["n_min"] == 240), ["code", "date"]].drop_duplicates()
    kid = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys))
    mn = mn.merge(keys.assign(_k=np.arange(len(keys))), on=["code", "date"])
    mn = mn.drop_duplicates(["_k", "i"])
    n = len(keys)
    P = np.full((n, 240), np.nan, np.float64)
    V = np.zeros((n, 240), np.float64)
    P[mn["_k"].to_numpy(), mn["i"].to_numpy()] = mn["price"].to_numpy(float)
    V[mn["_k"].to_numpy(), mn["i"].to_numpy()] = mn["vol"].to_numpy(float)
    ok = np.isfinite(P).all(axis=1) & (V.sum(axis=1) > 0)
    say("240 根齐全且有量", int(ok.sum()), "/", n)
    # 参考日线（pit 价，不复权 close_u）
    dates = sorted(keys["date"].unique())
    px = pd.read_parquet(EXP / "screen" / "px_screen.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume", "close_u"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px[px["code"].isin(keys["code"].unique())].sort_values(["code", "date"])
    px["pclose_u"] = px.groupby("code")["close_u"].shift(1)
    px = px[px["date"].isin(dates)]
    pool = pd.read_parquet(EXP / "polish" / "pool.parquet",
                           columns=["code", "date", "lim_pct", "lu_close", "yizi_today", "lu_streak", "cr1", "gap1",
                                    "turn1", "amt_ratio", "nlu20", "ret1", "mcap"],
                           filters=[("date", "in", dates)])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    top = pd.read_parquet(ROOT / "research" / "altdata" / "top60.parquet",
                          columns=["code", "date", "board", "rank_ret20", "y_open", "y_up", "ret20", "amt20"])
    top["code"] = top["code"].astype(str).str.zfill(6)
    ref = keys.merge(px, on=["code", "date"], how="left").merge(pool, on=["code", "date"], how="left")
    ref = ref.merge(top, on=["code", "date"], how="left")
    assert len(ref) == n
    # 原价换算：q = close_u / close（前复权 -> 原价）
    q = (ref["close_u"] / ref["close"]).to_numpy()
    open_u = ref["open"].to_numpy() * q
    high_u = ref["high"].to_numpy() * q
    low_u = ref["low"].to_numpy() * q
    close_u = ref["close_u"].to_numpy()
    lim = np.round(ref["pclose_u"].to_numpy() * (1 + ref["lim_pct"].to_numpy() / 100), 2)
    last = P[:, -1]
    chk = np.abs(last / close_u - 1)
    say("分时末价 对 close_u 偏差 >0.5%%：%d 个（中位 %.5f）" % (int((chk > 0.005).sum()), float(np.nanmedian(chk))))
    vsum = V.sum(axis=1)
    vref = ref["volume"].to_numpy() / 100.0
    vr = vsum / vref
    say("分时总量 / 日线量（手）中位 %.4f，偏差 >1%% %d 个" % (float(np.nanmedian(vr)), int((np.abs(vr - 1) > 0.01).sum())))

    F = pd.DataFrame({"code": keys["code"].to_numpy(), "date": keys["date"].to_numpy()})
    F["ok"] = ok
    # ---- 价格路径 ----
    vwap = (P * V).sum(axis=1) / np.maximum(vsum, 1e-9)
    F["c_vwap"] = last / vwap - 1                                  # 收盘相对日内均价
    F["ret_last30"] = P[:, 239] / P[:, 209] - 1                    # 尾盘 30 分钟涨幅
    F["ret_last60"] = P[:, 239] / P[:, 179] - 1
    F["ret_first30"] = P[:, 29] / open_u - 1                       # 早盘 30 分钟涨幅（对开盘）
    F["ret_am"] = P[:, 119] / open_u - 1                           # 上午收盘对开盘
    F["ret_pm"] = P[:, 239] / P[:, 119] - 1                        # 下午对上午收盘
    runmax = np.maximum.accumulate(P, axis=1)
    F["dd_intra"] = (P / runmax - 1).min(axis=1)                   # 日内最大回撤（从日内高点）
    F["c_hi"] = last / P.max(axis=1) - 1                           # 收盘离分时最高
    F["c_lo"] = last / P.min(axis=1) - 1
    F["t_high"] = (P.argmax(axis=1) + 1) / 240.0                   # 日内高点出现的时刻（0~1）
    F["t_low"] = (P.argmin(axis=1) + 1) / 240.0
    r1 = np.diff(np.log(P), axis=1)
    F["rv"] = np.sqrt((r1 ** 2).sum(axis=1))                       # 1 分钟收益实现波动
    F["up_share"] = (r1 > 0).mean(axis=1)                          # 上涨分钟占比
    F["c_open"] = last / open_u - 1                                # = 日线 cr1（核对用）
    F["range_u"] = high_u / low_u - 1
    # ---- 量的分布 ----
    F["v_first30"] = V[:, :30].sum(axis=1) / vsum                  # 早盘 30 分钟量占比（含竞价）
    F["v_first5"] = V[:, :5].sum(axis=1) / vsum
    F["v_last30"] = V[:, 210:].sum(axis=1) / vsum                  # 尾盘 30 分钟量占比
    F["v_last5"] = V[:, 235:].sum(axis=1) / vsum
    F["v_am"] = V[:, :120].sum(axis=1) / vsum
    Vs = np.sort(V, axis=1)
    F["v_top10"] = Vs[:, -10:].sum(axis=1) / vsum                  # 量最大的 10 分钟占比
    F["v_zero"] = (V == 0).sum(axis=1) / 240.0                     # 零成交分钟占比（封板会很高）
    # 量加权的价格位置：高价成交占比（在 vwap 之上成交的量）
    F["v_above_vwap"] = (V * (P > vwap[:, None])).sum(axis=1) / vsum
    # ---- 涨停相关（只在当天 high 触及涨停价时有意义）----
    at = P >= (lim[:, None] - 0.0051)
    touched = at.any(axis=1)
    F["lu_touch"] = touched.astype(float)
    first_i = np.where(touched, at.argmax(axis=1), -1)
    F["t_first_lu"] = np.where(touched, (first_i + 1) / 240.0, np.nan)   # 首次触及涨停时刻
    # 开板次数：触板后从「在板」掉到「不在板」的次数
    trans = (at[:, :-1] & ~at[:, 1:]).sum(axis=1)
    F["n_open"] = np.where(touched, trans, np.nan).astype(float)
    F["lu_hold"] = np.where(touched, at.mean(axis=1), np.nan)            # 在板分钟占比
    F["lu_seal_close"] = np.where(touched, at[:, -1], np.nan).astype(float)
    # 封板后的量（在板分钟的量占比：封死的板量应该很小）
    F["v_at_lu"] = np.where(touched, (V * at).sum(axis=1) / vsum, np.nan)
    # ---- 分笔：主动买卖 ----
    tx = tx[tx["hm"] <= "15:00"]
    tx["_pm"] = (tx["hm"] >= "14:30").astype(int)
    tx["_auc"] = (tx["hm"] == "09:25").astype(int)
    tx["_big_net"] = tx["big_buy"] - tx["big_sell"]
    g = tx.groupby(["code", "date"])
    agg = pd.DataFrame({
        "tx_buy": g["buy"].sum(), "tx_sell": g["sell"].sum(), "tx_neu": g["neu"].sum(),
        "tx_n": g["n"].sum(), "tx_bigb": g["big_buy"].sum(), "tx_bigs": g["big_sell"].sum(),
        "tx_buy_pm": (tx["buy"] * tx["_pm"]).groupby([tx["code"], tx["date"]]).sum(),
        "tx_sell_pm": (tx["sell"] * tx["_pm"]).groupby([tx["code"], tx["date"]]).sum(),
        "tx_auc": ((tx["buy"] + tx["sell"] + tx["neu"]) * tx["_auc"]).groupby([tx["code"], tx["date"]]).sum(),
    }).reset_index()
    F = F.merge(agg, on=["code", "date"], how="left")
    tot = F["tx_buy"] + F["tx_sell"] + F["tx_neu"]
    F["buy_share"] = F["tx_buy"] / (F["tx_buy"] + F["tx_sell"])                # 主动买占主动成交
    F["buy_share_pm"] = F["tx_buy_pm"] / (F["tx_buy_pm"] + F["tx_sell_pm"])    # 14:30 后主动买占比
    F["big_net"] = (F["tx_bigb"] - F["tx_bigs"]) / tot                        # 大单净买占总量
    F["big_share"] = (F["tx_bigb"] + F["tx_bigs"]) / tot                      # 大单成交占比
    F["auc_share"] = F["tx_auc"] / tot                                        # 集合竞价量占比
    F["tick_size"] = tot / F["tx_n"]                                          # 平均每笔手数
    # ---- 日线条件 / 标签 ----
    for c in ["board", "rank_ret20", "y_open", "y_up", "ret20", "amt20", "lim_pct", "lu_close", "yizi_today",
              "lu_streak", "cr1", "gap1", "turn1", "amt_ratio", "nlu20", "ret1", "mcap"]:
        F[c] = ref[c].to_numpy()
    F["at_high"] = (ref["close"].to_numpy() >= ref["high"].to_numpy() - 1e-9).astype(float)   # 收在最高价
    F["hi_touch_lu"] = (high_u >= lim - 0.0051).astype(float)                                # 日线最高价触及涨停
    F["lim_raw"] = lim
    F["seg"] = np.where(F["date"] < "2024-06-01", "screen", "dev")
    say("cr1 核对：分时 c_open 对 日线 cr1 偏差中位 %.5f" % float(np.nanmedian(np.abs(F["c_open"] - F["cr1"]))))
    say("lu_touch 对 日线 hi_touch_lu 一致率 %.4f；lu_seal_close 对 lu_close 一致率 %.4f" % (
        float((F["lu_touch"] == F["hi_touch_lu"]).mean()),
        float((F.loc[F["lu_touch"] == 1, "lu_seal_close"] == F.loc[F["lu_touch"] == 1, "lu_close"]).mean())))
    F.to_parquet(HERE / "feat.parquet", index=False)
    say("->", HERE / "feat.parquet", F.shape, "按段", F.groupby("seg").size().to_dict())
    # 给 research/gate/ 用的统一格式：只有 T 日收盘后可得的日内特征，不带标签 / 名次
    intraday_cols = ["c_vwap", "ret_last30", "ret_last60", "ret_first30", "ret_am", "ret_pm", "dd_intra", "c_hi", "c_lo",
                     "t_high", "t_low", "rv", "up_share", "range_u", "v_first30", "v_first5", "v_last30", "v_last5", "v_am",
                     "v_top10", "v_zero", "v_above_vwap", "lu_touch", "t_first_lu", "n_open", "lu_hold", "lu_seal_close",
                     "v_at_lu", "buy_share", "buy_share_pm", "big_net", "big_share", "auc_share", "tick_size",
                     "tx_n", "lim_raw"]
    G = F.loc[F["ok"], ["code", "date"] + intraday_cols].copy()
    G["code"] = G["code"].astype(str).str.zfill(6)
    G = G.sort_values(["date", "code"]).reset_index(drop=True)
    G.to_parquet(HERE / "features_intraday.parquet", index=False)
    write_readme(G, meta)
    say("->", HERE / "features_intraday.parquet", G.shape)


README = """research/intraday/features_intraday.parquet：ret20 前 50 抽样股票-日的日内（分时 / 分笔）特征。2026-10-18 生成。

键：code（6 位字符串）+ date（YYYY-MM-DD，交易日）。一行一个股票-日。值只用 T 日收盘后可得的信息（当天 240 根 1 分钟分时 + 当天分笔），
不含标签、不含名次。再跑 features.py 会整表重写（按 raw/parts 里已拉到的全部股票-日）。

来源：通达信行情主机（pytdx 1.72，get_history_minute_time_data / get_history_transaction_data），价是不复权原价，量单位手。
核对：分时总量 x100 = 新浪日线 volume（比值 1.0000），分时末价 = 日线不复权收盘，分时 c_open = 日线 cr1。

覆盖（抽样，不是全市场、不是全时段）：
  股票-日 = research/altdata/top60.parquet 里 rank_ret20 <= 50 的行（即 ret20 规则候选池内近 20 日涨幅前 50，候选池口径见
  research/rulecore/common_rc.py：非 ST、>= 2 元、>= 120 根、无停牌缺口、无减持 / 解禁 / 增发），限定月份：
  筛选集 %(screen)s，开发集 %(dev)s。
  北交所的票没有（通达信标准行情不提供北交所历史分时），一字板的票有（数据在，只是没法买）。
  本表 %(n)d 行，%(ncode)d 只票，%(ndate)d 个交易日，%(dmin)s .. %(dmax)s。拉取失败 / 分时不足 240 根的股票-日不在表里（meta 里 ok=False 的 %(nbad)d 个）。

缺失的含义：
  t_first_lu / n_open / lu_hold / lu_seal_close / v_at_lu 只在当天触及过涨停（lu_touch=1）的行有值，其余 NaN（不是「没拉到」）。
  buy_share / buy_share_pm / big_net / big_share / auc_share / tick_size / tx_n 来自分笔，分笔为空的行 NaN。
  其他列不应有 NaN。

分时 240 根：第 1..120 根 = 09:31..11:30（第 1 根含集合竞价的量），第 121..240 根 = 13:01..15:00。分笔 hm 09:25 = 集合竞价，> 15:00 的盘后定价交易不算。
涨停价 lim_raw = round(昨收原价 x (1 + 涨跌幅限制), 2)，判「在板」容 0.5 分。

列：
  c_vwap        收盘 / 日内成交量加权均价 - 1
  ret_last30    尾盘 30 分钟涨幅：第 240 根 / 第 210 根 - 1
  ret_last60    尾盘 60 分钟涨幅：第 240 根 / 第 180 根 - 1
  ret_first30   早盘 30 分钟涨幅：第 30 根 / 开盘价 - 1
  ret_am        上午收盘（第 120 根）/ 开盘价 - 1
  ret_pm        收盘 / 上午收盘 - 1
  dd_intra      日内最大回撤：min_t(价_t / 之前最高价 - 1)，<= 0
  c_hi          收盘 / 分时最高 - 1（0 = 收在最高）
  c_lo          收盘 / 分时最低 - 1
  t_high        日内最高价出现的时刻 / 240（1 = 最后一分钟）
  t_low         日内最低价出现的时刻 / 240
  rv            1 分钟对数收益平方和开方（日内实现波动）
  up_share      上涨分钟占比（239 个 1 分钟收益里 > 0 的比例）
  range_u       日线最高 / 最低 - 1（原价）
  v_first30     前 30 分钟成交量占全天比例（含集合竞价）
  v_first5      前 5 分钟成交量占比
  v_last30      最后 30 分钟成交量占比
  v_last5       最后 5 分钟成交量占比
  v_am          上午成交量占比
  v_top10       成交量最大的 10 分钟占全天比例
  v_zero        零成交分钟占比（封死的板会很高）
  v_above_vwap  在均价之上成交的量占比
  lu_touch      当天分时里是否触及涨停价（0/1）
  t_first_lu    首次触及涨停的时刻 / 240（触板的票内）
  n_open        开板次数：触板后从「在板」掉到「不在板」的次数（触板的票内）
  lu_hold       在板分钟占比（触板的票内）
  lu_seal_close 收盘那一分钟是否在板（触板的票内；= 日线的涨停收盘）
  v_at_lu       在板分钟的成交量占比（触板的票内）
  buy_share     主动买成交量 / (主动买 + 主动卖)，全天（分笔 buyorsell 0 买 1 卖 2 中性）
  buy_share_pm  14:30 之后的主动买占比
  big_net       (大单主动买 - 大单主动卖) / 全天成交量，大单 = 单笔成交额 >= 100 万元
  big_share     大单成交量占比
  auc_share     集合竞价成交量占比
  tick_size     平均每笔手数（全天手数 / 分笔笔数）
  tx_n          分笔笔数
  lim_raw       当天涨停价（原价）
"""


def write_readme(G: pd.DataFrame, meta: pd.DataFrame) -> None:
    import fetch as FT
    txt = README % {"screen": " / ".join(FT.MONTHS["screen"]), "dev": " / ".join(FT.MONTHS["dev"]),
                    "n": len(G), "ncode": G["code"].nunique(), "ndate": G["date"].nunique(),
                    "dmin": G["date"].min(), "dmax": G["date"].max(), "nbad": int((~meta["ok"] | (meta["n_min"] != 240)).sum())}
    (HERE / "features_README.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
