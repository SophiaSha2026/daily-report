"""申万行业（带时点）x 板块共振特征 -> features_sw_industry.parquet，并做快速诊断 -> results_sw.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_sw.py

行业表：raw/sw_class_hist.xls（申万研究所 StockClassifyUse_stock.xls，2026-10-18 03:40 curl 下载，12929 行 / 5934 只），
四列 = 股票代码 / 计入日期 / 行业代码（6 位：前 2 位一级、前 4 位二级）/ 更新时间。T 日用的是「计入日期 <= T 的最后一条」，
2021-07-30 整体换成申万 2021 版，之前是 2014 版：同一天内口径一致，跨期同一行业代码可能含义不同（只做当天截面统计，不跨期比）。
所有行业统计只用 T 日及之前的价格（pool.parquet 的 ret20 / ret5 / lu_close 等都是 T 日收盘值），T 日收盘后可得。
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


def load_sw() -> pd.DataFrame:
    sw = pd.read_excel(C.RAW / "sw_class_hist.xls")
    sw.columns = ["code", "since", "ind", "updated"]
    sw["code"] = sw["code"].astype(int).astype(str).str.zfill(6)
    sw["since"] = pd.to_datetime(sw["since"])
    sw["ind"] = sw["ind"].astype(int).astype(str).str.zfill(6)
    sw = sw.sort_values(["code", "since"]).reset_index(drop=True)
    return sw


def main():
    t0 = time.time()
    pool = C.load_pool(extra=["ret5", "ret1", "dist_hi20", "nlu5"])
    sw = load_sw()
    C.say("申万表 %d 行 %d 只，计入日期 %s..%s" % (len(sw), sw["code"].nunique(), sw["since"].min().date(), sw["since"].max().date()))
    # 时点行业：merge_asof（按 code，since <= date）
    a = pool[["code", "date"]].copy()
    a["_i"] = np.arange(len(a))
    a["_d"] = pd.to_datetime(a["date"])
    a = a.sort_values("_d")
    m = pd.merge_asof(a, sw[["code", "since", "ind"]].sort_values("since"), left_on="_d", right_on="since", by="code", direction="backward")
    m = m.sort_values("_i")
    pool["ind2"] = m["ind"].str[:4].to_numpy()
    pool["ind1"] = m["ind"].str[:2].to_numpy()
    pool["ind3"] = m["ind"].to_numpy()
    cov = pool["ind2"].notna().mean()
    C.say("行业覆盖 %.1f%%（pool 行）；候选池内 %.1f%%" % (100 * cov, 100 * pool.loc[pool["pool"], "ind2"].notna().mean()))
    pool["lu"] = (pool["lu_close"].astype(float) > 0).astype(float)
    pool["in50"] = (pool["rank_ret20"] <= 50).astype(float)
    pool["nh20"] = (pool["dist_hi20"].astype(float) >= -1e-9).astype(float)     # 收在 20 日高点
    f = pool[["code", "date"]].copy()
    for lvl in ("ind2", "ind1"):
        key = [lvl, "date"]
        ok = pool[lvl].notna()
        g = pool[ok].groupby(key, sort=False)
        agg = g.agg(n=("code", "size"), lu_n=("lu", "sum"), in50_n=("in50", "sum"), nh20_n=("nh20", "sum"),
                    ret20_med=("ret20", "median"), ret5_med=("ret5", "median"), ret1_med=("ret1", "median")).reset_index()
        agg["lu_share"] = agg["lu_n"] / agg["n"]
        agg["nh20_share"] = agg["nh20_n"] / agg["n"]
        # 行业动量在当天所有行业里的百分位（只算成分 >= 5 只的行业）
        big = agg["n"] >= 5
        for c in ("ret20_med", "ret5_med", "lu_share"):
            s = agg.loc[big].groupby("date")[c].rank(pct=True)
            agg[c + "_pct"] = np.nan
            agg.loc[big, c + "_pct"] = s
        # 5 日 / 20 日累计涨停家数占比（按行业沿交易日滚动）
        agg = agg.sort_values([lvl, "date"])
        agg["lu_share5"] = agg.groupby(lvl)["lu_share"].transform(lambda s: s.rolling(5, min_periods=1).sum())
        agg["lu_share20"] = agg.groupby(lvl)["lu_share"].transform(lambda s: s.rolling(20, min_periods=1).sum())
        j = pool[key].merge(agg, on=key, how="left")
        pre = lvl + "_"
        f[pre + "n"] = j["n"].to_numpy()
        f[pre + "lu_n"] = j["lu_n"].to_numpy()
        f[pre + "lu_share"] = j["lu_share"].to_numpy()
        f[pre + "lu_share5"] = j["lu_share5"].to_numpy()
        f[pre + "lu_share20"] = j["lu_share20"].to_numpy()
        f[pre + "nh20_share"] = j["nh20_share"].to_numpy()
        f[pre + "ret20_med"] = j["ret20_med"].to_numpy()
        f[pre + "ret5_med"] = j["ret5_med"].to_numpy()
        f[pre + "ret20_pct"] = j["ret20_med_pct"].to_numpy()
        f[pre + "ret5_pct"] = j["ret5_med_pct"].to_numpy()
        f[pre + "lu_pct"] = j["lu_share_pct"].to_numpy()
        if lvl == "ind2":
            # 同行业里还有几只在 ret20 前 50（不含自己）、自己在行业内的 ret20 名次百分位、相对行业的超额 ret20
            f["ind2_in50_other"] = j["in50_n"].to_numpy() - pool["in50"].to_numpy()
            r = pool.loc[ok].groupby(key)["ret20"].rank(pct=True)
            f["ind2_ret20_rank_in"] = np.nan
            f.loc[ok, "ind2_ret20_rank_in"] = r.to_numpy()
            f["ind2_ret20_excess"] = pool["ret20"].to_numpy() - j["ret20_med"].to_numpy()
            f["ind2_lu_other"] = j["lu_n"].to_numpy() - pool["lu"].to_numpy()
    f["sw_ind2"] = pool["ind2"].to_numpy()
    f["sw_ind1"] = pool["ind1"].to_numpy()
    f = f[f["date"] >= C.FEAT_START].reset_index(drop=True)
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：申万行业归属变迁表（带计入日期，时点可知）+ pool.parquet 的 T 日收盘价量。全部为 T 日收盘后可得。
覆盖：pool.parquet 全部行（2021-12-01..2025-12-31）；行业缺失（新股 / 表里没有）的行所有行业列为 NaN。
列（ind2_ = 申万二级，ind1_ = 申万一级；每个行业当天在 pool.parquet 里的成分）：
  sw_ind2 / sw_ind1        T 日生效的申万二级 / 一级代码（字符串；2021-07-30 前为 2014 版口径）
  *_n                      同行业当天只数
  *_lu_n / *_lu_share      同行业当天收盘涨停只数 / 占比（lu_close，含自己）
  *_lu_share5 / 20         同行业涨停占比近 5 / 20 个交易日累计（板块持续活跃度）
  *_nh20_share             同行业收在 20 日新高的占比
  *_ret20_med / *_ret5_med 同行业 ret20 / ret5 中位数（行业动量）
  *_ret20_pct / *_ret5_pct / *_lu_pct   该行业的 ret20 中位 / ret5 中位 / 涨停占比 在当天全部行业（成分 >= 5 只）里的百分位
  ind2_in50_other          同二级行业里其他在 ret20 前 50 的只数（候选池内名次）
  ind2_lu_other            同二级行业里其他涨停的只数
  ind2_ret20_rank_in       自己 ret20 在同二级行业内的百分位（1 = 行业龙头）
  ind2_ret20_excess        自己 ret20 减行业中位
缺失含义：NaN = 没有行业归属或行业内无人有值；计数列 0 = 真 0。
"""
    C.write_table("sw_industry", f, readme)
    cols = ["ind2_lu_n", "ind2_lu_share", "ind2_lu_share5", "ind2_lu_share20", "ind2_nh20_share", "ind2_ret20_med", "ind2_ret5_med",
            "ind2_ret20_pct", "ind2_ret5_pct", "ind2_lu_pct", "ind2_in50_other", "ind2_lu_other", "ind2_ret20_rank_in", "ind2_ret20_excess",
            "ind1_lu_share", "ind1_lu_share5", "ind1_ret20_pct", "ind1_ret5_pct", "ind2_n"]
    txt = C.diag_table(f, pool, cols, label="sw_industry")
    (HERE / "results_sw.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 sw_industry -> results_sw.txt")


if __name__ == "__main__":
    main()
