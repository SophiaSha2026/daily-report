"""龙虎榜席位（买方 / 卖方营业部）-> features_lhb_seats.parquet + 诊断 results_lhb_seats.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_lhb_seats.py

输入 raw/lhb_sbuy.parquet / raw/lhb_ssell.parquet（RPT_BILLBOARD_DAILYDETAILSBUY / SELL，2021-09..2025-12，后验列已删）。
席位的「知名度」不用任何外部名单，全部按时点从历史里算：
    freq250  = 该席位此前 250 个交易日（含当天）作为买方上榜的 (票, 日) 次数   —— 活跃游资
    hit250   = 该席位此前作为买方上榜、且标签窗口已经关上的那些 (票, 日) 的 y_open 命中率（只用 s <= t-21 的，窗口 250 日），
               收缩：(hits + 10 * 0.08) / (n + 10)                                    —— 战绩
席位类型按名称：机构专用 / 沪深股通（北向）/ 「拉萨」（东财拉萨营业部，散户聚集）。
全部 T 日收盘后可得（龙虎榜交易所盘后公布；hit 只用窗口已关的历史标签）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_dd as C  # noqa: E402

PRIOR, PRIOR_W = 0.08, 10.0
LAG = 21            # y_open 窗口 = s+1..s+20 根，s <= t-21 才算已知


def seat_type(name: pd.Series) -> pd.DataFrame:
    n = name.fillna("")
    return pd.DataFrame({"inst": n.str.contains("机构专用").to_numpy(),
                         "north": n.str.contains("沪股通|深股通|港股通|北向").to_numpy(),
                         "lasa": n.str.contains("拉萨").to_numpy()})


def seat_history(B: pd.DataFrame) -> pd.DataFrame:
    """B: seat / di（交易日序号）/ y（y_open 或 NaN）。返回同序的 freq250 / hit_n / hit_rate。"""
    B = B.copy()
    B["_i"] = np.arange(len(B))
    B = B.sort_values(["seat", "di"], kind="mergesort")
    freq = np.zeros(len(B))
    hn = np.zeros(len(B))
    hh = np.zeros(len(B))
    for _, grp in B.groupby("seat", sort=False):
        d = grp["di"].to_numpy()
        y = grp["y"].to_numpy(float)
        known = np.isfinite(y)
        cn = np.concatenate([[0], np.cumsum(known)])
        ch = np.concatenate([[0], np.cumsum(np.where(known, y, 0.0))])
        # 频次：[t-249, t]
        lo = np.searchsorted(d, d - 249, side="left")
        hi = np.searchsorted(d, d, side="right")
        fr = hi - lo
        # 战绩：s in [t-250-LAG, t-LAG]
        lo2 = np.searchsorted(d, d - 250 - LAG, side="left")
        hi2 = np.searchsorted(d, d - LAG, side="right")
        n_ = cn[hi2] - cn[lo2]
        h_ = ch[hi2] - ch[lo2]
        idx = grp["_i"].to_numpy()
        freq[idx] = fr
        hn[idx] = n_
        hh[idx] = h_
    out = pd.DataFrame({"freq250": freq, "hit_n": hn, "hit_rate": (hh + PRIOR_W * PRIOR) / (hn + PRIOR_W)})
    return out


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    cols = ["SECURITY_CODE", "TRADE_DATE", "OPERATEDEPT_CODE", "OPERATEDEPT_NAME", "BUY", "SELL", "ACCUM_AMOUNT", "TOTAL_BUYRIO", "TOTAL_SELLRIO"]
    B = pd.read_parquet(C.RAW / "lhb_sbuy.parquet", columns=cols)
    S = pd.read_parquet(C.RAW / "lhb_ssell.parquet", columns=cols)
    for X in (B, S):
        X["code"] = X["SECURITY_CODE"].astype(str).str.zfill(6)
        X["date"] = X["TRADE_DATE"].astype(str).str[:10]
        X["seat"] = X["OPERATEDEPT_CODE"].astype(str)
    B = B.sort_values("BUY", ascending=False).drop_duplicates(["code", "date", "seat"])
    S = S.sort_values("SELL", ascending=False).drop_duplicates(["code", "date", "seat"])
    B["di"] = g.day_index(B["date"])
    S["di"] = g.day_index(S["date"])
    B = B[B["di"] >= 0].reset_index(drop=True)
    S = S[S["di"] >= 0].reset_index(drop=True)
    C.say("买方 %d 行（%d 席位）卖方 %d 行 %.0fs" % (len(B), B["seat"].nunique(), len(S), time.time() - t0))
    # 标签只用来算席位历史战绩（窗口已关的）
    lab = pool[["code", "date", "y_open"]]
    B = B.merge(lab, on=["code", "date"], how="left")
    B["y"] = B["y_open"].astype(float)
    hist = seat_history(B[["seat", "di", "y"]])
    B = pd.concat([B, hist], axis=1)
    S = S.merge(B[["seat", "date", "freq250"]].drop_duplicates(["seat", "date"]), on=["seat", "date"], how="left")
    S["freq250"] = S["freq250"].fillna(0)
    tb, ts = seat_type(B["OPERATEDEPT_NAME"]), seat_type(S["OPERATEDEPT_NAME"])
    B = pd.concat([B, tb], axis=1)
    S = pd.concat([S, ts], axis=1)
    B["ratio"] = pd.to_numeric(B["TOTAL_BUYRIO"], errors="coerce")
    B["buy"] = pd.to_numeric(B["BUY"], errors="coerce")
    B["amt"] = pd.to_numeric(B["ACCUM_AMOUNT"], errors="coerce")
    S["sell"] = pd.to_numeric(S["SELL"], errors="coerce")
    S["amt"] = pd.to_numeric(S["ACCUM_AMOUNT"], errors="coerce")
    B["hot"] = (B["freq250"] >= 60).astype(float)        # 一年上榜 60 次以上 = 活跃游资席位
    B["bw"] = B["buy"] * B["hit_rate"]
    ab = B.groupby(["code", "date"], sort=False).agg(
        n_buy=("seat", "size"), inst_buy=("inst", "sum"), north_buy=("north", "sum"), lasa_buy=("lasa", "sum"),
        hot_buy=("hot", "sum"), famous_max=("freq250", "max"), famous_sum=("freq250", "sum"),
        hit_max=("hit_rate", "max"), hitn_max=("hit_n", "max"), bw=("bw", "sum"), buy_sum=("buy", "sum"),
        top_ratio=("ratio", "max"), amt=("amt", "max")).reset_index()
    ab["hit_wavg"] = ab["bw"] / ab["buy_sum"]
    ab["buy_share"] = ab["buy_sum"] / ab["amt"]
    as_ = S.groupby(["code", "date"], sort=False).agg(
        inst_sell=("inst", "sum"), lasa_sell=("lasa", "sum"), famous_sell=("freq250", "sum"), sell_sum=("sell", "sum"), amt=("amt", "max")).reset_index()
    as_["sell_share"] = as_["sell_sum"] / as_["amt"]
    a = ab.merge(as_.drop(columns=["amt"]), on=["code", "date"], how="outer")
    a["net_share"] = a["buy_share"].fillna(0) - a["sell_share"].fillna(0)
    a["di"] = g.day_index(a["date"])
    C.say("上榜 (票, 日) %d 条；席位 hit_n 中位 %.0f，freq250 中位 %.0f，hot 占比 %.1f%% %.0fs" % (
        len(a), B["hit_n"].median(), B["freq250"].median(), 100 * B["hot"].mean(), time.time() - t0))

    rows = pool[pool["date"] >= C.FEAT_START]
    di = g.day_index(rows["date"])
    code = rows["code"]
    f = rows[["code", "date"]].copy().reset_index(drop=True)
    on = (g.dense(a["code"], a["di"].to_numpy(), np.ones(len(a))) > 0).astype(float)
    f["lhbs_today"] = g.pick(on, code, di)
    nanon = on.copy()
    nanon[nanon == 0] = np.nan
    _, idx = C.Grid.ffill(nanon)
    f["lhbs_age"] = g.pick(np.where(idx >= 0, np.arange(on.shape[1])[None, :] - idx, np.nan), code, di)
    today_cols = ["n_buy", "inst_buy", "north_buy", "lasa_buy", "hot_buy", "famous_max", "famous_sum", "hit_max", "hit_wavg",
                  "top_ratio", "buy_share", "inst_sell", "lasa_sell", "famous_sell", "sell_share", "net_share"]
    for c in today_cols:
        A = g.dense(a["code"], a["di"].to_numpy(), a[c].to_numpy(float), agg="last")
        f["lhbs_" + c] = g.pick(A, code, di)
    # 5 日 / 20 日窗口（只在上榜日有值；窗口和 / 最大）
    def wmax(A, n):
        X = np.where(np.isfinite(A), A, -np.inf)
        out = X.copy()
        for k in range(1, n):
            out[:, k:] = np.maximum(out[:, k:], X[:, :-k])
        out[~np.isfinite(out)] = np.nan
        return out
    for c in ("inst_buy", "hot_buy", "north_buy"):
        A = g.dense(a["code"], a["di"].to_numpy(), a[c].to_numpy(float))
        f["lhbs_%s_n5" % c] = g.pick(C.Grid.wsum(A, 5), code, di)
        f["lhbs_%s_n20" % c] = g.pick(C.Grid.wsum(A, 20), code, di)
    for c in ("famous_max", "hit_max"):
        A = g.dense(a["code"], a["di"].to_numpy(), a[c].to_numpy(float), agg="last")
        f["lhbs_%s_5" % c] = g.pick(wmax(A, 5), code, di)
    A = g.dense(a["code"], a["di"].to_numpy(), a["net_share"].to_numpy(float))
    f["lhbs_net_share_5"] = g.pick(C.Grid.wsum(A, 5), code, di)
    A = g.dense(a["code"], a["di"].to_numpy(), a["inst_sell"].to_numpy(float))
    f["lhbs_inst_sell_n5"] = g.pick(C.Grid.wsum(A, 5), code, di)
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：东财龙虎榜买方 / 卖方席位明细（RPT_BILLBOARD_DAILYDETAILSBUY / SELL），2021-09..2025-12，同一 (票, 日, 席位) 多条原因只算一次。
时点：T 日收盘后交易所公布，T 日那行可用；席位战绩只用 s <= T-21 的历史标签（窗口已关）。
席位统计全部按时点自算（freq250 = 该席位此前 250 个交易日买方上榜次数；hit_rate = 此前 250 日买方上榜票的 y_open 命中率，
收缩 (hits + 0.8) / (n + 10)；hot = freq250 >= 60）。席位类型按名称：机构专用 / 沪深股通 / 拉萨。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31。不上榜的日子 lhbs_today = 0，当日席位列 NaN，窗口计数列 0。
列：
  lhbs_today / lhbs_age                 当天是否上榜（买方或卖方席位表里有）/ 距最近一次上榜的交易日数（没有 = NaN）
  lhbs_n_buy / inst_buy / north_buy / lasa_buy / hot_buy   当天买方席位数 / 机构专用数 / 沪深股通数 / 拉萨席位数 / 活跃游资席位数
  lhbs_famous_max / famous_sum          买方席位 freq250 最大 / 合计
  lhbs_hit_max / hit_wavg               买方席位收缩后历史命中率的最大 / 按买入额加权均值
  lhbs_top_ratio / buy_share            第一大买方占当日成交额比例 / 买方五席合计占比
  lhbs_inst_sell / lasa_sell / famous_sell / sell_share   卖方：机构专用数 / 拉萨数 / freq250 合计 / 五席合计占比
  lhbs_net_share                        买方合计占比 − 卖方合计占比
  lhbs_{inst_buy,hot_buy,north_buy}_n5 / _n20   近 5 / 20 个交易日累计（不上榜算 0）
  lhbs_famous_max_5 / lhbs_hit_max_5    近 5 个交易日上榜日里的最大值（5 日内没上榜 = NaN）
  lhbs_net_share_5 / lhbs_inst_sell_n5  近 5 日净占比合计 / 机构卖出席位数合计
缺失含义：当日席位列 NaN = 当天不在龙虎榜；lhbs_age NaN = 2021-09 以来没上过榜。
"""
    C.write_table("lhb_seats", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="lhb_seats")
    (HERE / "results_lhb_seats.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 lhb_seats -> results_lhb_seats.txt")


if __name__ == "__main__":
    main()
