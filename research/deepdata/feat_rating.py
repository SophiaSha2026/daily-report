"""研报评级（巨潮投资评级按日全市场）-> features_rating.parquet + 诊断 results_rating.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_rating.py

输入 raw/rating_cninfo.parquet（fetch_cninfo.py，2021-09-01..2025-12-31 逐自然日）。
时点：按发布日期，非交易日发布算下一个交易日，发布日收盘起可用（研报多在盘前 / 盘后发布，保守按收盘）。
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

BUY_WORDS = ("买入", "强烈推荐", "强推", "推荐", "增持", "优于大市", "跑赢", "强于大市", "审慎增持", "谨慎增持", "优于")


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    r = pd.read_parquet(C.RAW / "rating_cninfo.parquet")
    r["code"] = r["code"].astype(str).str.zfill(6)
    r = r[r["code"].isin(set(g.codes))]
    r["first"] = r["first_flag"].astype(str).str.contains("^是").astype(float)
    chg = r["rating_change"].astype(str)
    r["up"] = chg.str.contains("调高|上调").astype(float)
    r["down"] = chg.str.contains("调低|下调").astype(float)
    rt = r["rating"].astype(str)
    r["buy"] = rt.str.contains("|".join(BUY_WORDS)).astype(float)
    r["strong"] = rt.str.contains("买入|强烈推荐|强推").astype(float)
    r["tgt"] = pd.to_numeric(r["target_hi"], errors="coerce")
    r.loc[~(r["tgt"] > 0), "tgt"] = np.nan
    a = r.groupby(["code", "date"], sort=False).agg(n=("org", "size"), first=("first", "sum"), up=("up", "sum"), down=("down", "sum"),
                                                     buy=("buy", "sum"), strong=("strong", "sum"), tgt=("tgt", "max")).reset_index()
    ei = g.day_index(a["date"], how="next")
    a = a[ei >= 0]
    ei = ei[ei >= 0]
    C.say("研报 %d 篇 -> (票, 日) %d 条 %s..%s；评级取值 %s；变化取值 %s" % (
        len(r), len(a), a["date"].min(), a["date"].max(), rt.value_counts().head(8).to_dict(), chg.value_counts().head(6).to_dict()))
    rows = pool[pool["date"] >= C.FEAT_START]
    di = g.day_index(rows["date"])
    code = rows["code"]
    close = rows["close"].to_numpy(float)
    f = rows[["code", "date"]].copy().reset_index(drop=True)
    N = g.dense(a["code"], ei, a["n"].to_numpy(float))
    on = (N > 0).astype(float)
    f["rat_today"] = g.pick(on, code, di)
    for w in (5, 20, 60, 250):
        f["rat_n%d" % w] = g.pick(C.Grid.wsum(N, w), code, di)
    nanon = on.copy()
    nanon[nanon == 0] = np.nan
    _, idx = C.Grid.ffill(nanon)
    f["rat_age"] = g.pick(np.where(idx >= 0, np.arange(on.shape[1])[None, :] - idx, np.nan), code, di)
    for c, ws in (("first", (20, 60)), ("up", (20, 60)), ("down", (60,)), ("buy", (60,)), ("strong", (60,))):
        A = g.dense(a["code"], ei, a[c].to_numpy(float))
        for w in ws:
            f["rat_%s_n%d" % (c, w)] = g.pick(C.Grid.wsum(A, w), code, di)
    n60 = f["rat_n60"].to_numpy(float)
    f["rat_buy_share60"] = np.where(n60 > 0, f["rat_buy_n60"] / n60, np.nan)
    f["rat_nocov250"] = (f["rat_n250"] == 0).astype(float)
    f["rat_surge"] = np.where(f["rat_n250"] > 0, f["rat_n20"] / (f["rat_n250"] / 12.5), np.nan)     # 近 20 日研报数对一年均值的倍数
    T, age = C.dense_last_ffill(g, a["code"], ei, a["tgt"].to_numpy(float))
    tgt = g.pick(T, code, di)
    tage = g.pick(age, code, di)
    prem = np.where(np.isfinite(tgt) & (tage <= 60), tgt / close - 1, np.nan)
    f["rat_tgt_prem60"] = prem
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：巨潮资讯「投资评级」按日全市场（webapi.cninfo p_sysapi1089，akshare stock_rank_forecast_cninfo），2021-09-01..2025-12-31 逐自然日；
东财 reportapi 全市场按日只给约 1/5（2024-03-05：28 对 145），没用。
时点：发布日期，非交易日算下一交易日，发布日收盘起可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；250 日窗口在 2022-08 之前不完整（表从 2021-09 起）。
列：
  rat_today / rat_age                   当天有研报 / 距最近一篇研报的交易日数（表里从没有过 = NaN）
  rat_n5 / n20 / n60 / n250             近 5 / 20 / 60 / 250 个交易日研报篇数（同一天多篇都算）
  rat_first_n20 / n60                   「首次评级」篇数（巨潮字段「是否首次评级」）
  rat_up_n20 / n60 / rat_down_n60       评级变化含「调高 / 上调」/「调低 / 下调」的篇数
  rat_buy_n60 / rat_strong_n60 / rat_buy_share60   60 日内正面评级篇数（买入 增持 推荐 优于大市 等）/ 买入级篇数 / 正面占比
  rat_nocov250                          一年没有任何研报（无人覆盖）
  rat_surge                             近 20 日篇数 / (近 250 日篇数 / 12.5)，研报关注度突增倍数（一年没研报 = NaN）
  rat_tgt_prem60                        60 日内最近一篇给了目标价的：目标价上限 / 收盘 − 1（没有 = NaN）
缺失含义：计数列 0 = 真 0；rat_age NaN = 2021-09 以来无研报。
"""
    C.write_table("rating", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="rating")
    (HERE / "results_rating.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 rating -> results_rating.txt")


if __name__ == "__main__":
    main()
