"""同花顺强势股题材（涨停票当天的题材原因）-> features_ths_theme.parquet + 诊断 results_ths.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_ths.py

输入 raw/ths_harden.parquet（fetch_ths.py，2023-10-25 起，只覆盖开发集 + 筛选集尾巴 2023-10-25..2024-05，诊断的筛选段只有约 145 天）。
题材 = reason 按「+」切成的词；当天某题材的热度 = 当天带这个词的强势股只数。
时点：收盘后快照，T 日那行可用；一只票「最近一次上榜的题材」沿交易日前向沿用最多 20 个交易日，用来算它的题材今天热不热。
reason 文本是否事后改写没法核（只能当近似）。
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_dd as C  # noqa: E402

GENERIC = {"市场超500股涨停", "市场超1000股涨停", "市场超200股涨停", "超跌反弹", "超跌", "反弹", "ST板块", "st板块", "新股", "次新股", "北交所"}
LAG_MAX = 20


def tokens(s: str) -> list[str]:
    out = []
    for t in str(s).replace("＋", "+").split("+"):
        t = t.strip()
        if t and t not in GENERIC:
            out.append(t)
    return out


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    h = pd.read_parquet(C.RAW / "ths_harden.parquet", columns=["code", "date", "reason", "zhangfu"])
    h["code"] = h["code"].astype(str).str.zfill(6)
    h = h[h["code"].isin(set(g.codes))]
    h["di"] = g.day_index(h["date"])
    h = h[h["di"] >= 0].drop_duplicates(["code", "di"]).reset_index(drop=True)
    h["toks"] = h["reason"].map(tokens)
    start = h["date"].min()
    C.say("强势股 %d 条 %s..%s（%d 天）；词 %d 个" % (len(h), start, h["date"].max(), h["di"].nunique(),
                                               len(set(t for ts in h["toks"] for t in ts))))
    # 当天各题材热度
    heat = defaultdict(Counter)
    for d_, ts in zip(h["di"].to_numpy(), h["toks"]):
        for t in set(ts):
            heat[d_][t] += 1
    # 当天上榜票：自己题材的热度（不含自己）
    h["theme_max"] = [max((heat[d_][t] - 1 for t in set(ts)), default=0) for d_, ts in zip(h["di"], h["toks"])]
    h["theme_sum"] = [sum(heat[d_][t] - 1 for t in set(ts)) for d_, ts in zip(h["di"], h["toks"])]
    h["ntok"] = h["toks"].map(len)
    # 最近一次上榜的题材（按票沿交易日前推）
    last_tok = {(c, d_): ts for c, d_, ts in zip(h["code"], h["di"], h["toks"])}
    rows = pool[pool["date"] >= start].reset_index(drop=True)
    di = g.day_index(rows["date"])
    code = rows["code"]
    f = rows[["code", "date"]].copy()
    on = (g.dense(h["code"], h["di"].to_numpy(), np.ones(len(h))) > 0).astype(float)
    f["ths_today"] = g.pick(on, code, di)
    f["ths_n5"] = g.pick(C.Grid.wsum(on, 5), code, di)
    f["ths_n20"] = g.pick(C.Grid.wsum(on, 20), code, di)
    nanon = on.copy()
    nanon[nanon == 0] = np.nan
    _, idx = C.Grid.ffill(nanon)
    last_di = g.pick(idx.astype(float), code, di)
    age = np.where(last_di >= 0, di - last_di, np.nan)
    f["ths_age"] = age
    for c in ("theme_max", "theme_sum", "ntok"):
        A = g.dense(h["code"], h["di"].to_numpy(), h[c].to_numpy(float), agg="last")
        f["ths_" + c] = g.pick(A, code, di)
    # 最近一次题材在今天的热度（age <= 20）
    lag_heat = np.full(len(f), np.nan)
    lag_heat_max_other = np.full(len(f), np.nan)
    ok = np.isfinite(age) & (age <= LAG_MAX)
    codes_arr = code.to_numpy()
    for i in np.flatnonzero(ok):
        ts = last_tok.get((codes_arr[i], int(last_di[i])))
        if not ts:
            continue
        hh = heat[di[i]]
        vals = [hh[t] for t in set(ts)]
        self_on = f["ths_today"].iat[i] > 0
        lag_heat[i] = max(vals) if vals else 0
        lag_heat_max_other[i] = (max(vals) - (1 if self_on else 0)) if vals else 0
    f["ths_lag_heat"] = lag_heat
    f["ths_lag_heat_other"] = lag_heat_max_other
    # 市场级：当天强势股总数、最热题材的只数
    tot = np.zeros(len(g.days))
    top = np.zeros(len(g.days))
    for d_, cnt in heat.items():
        tot[d_] = int(on[:, d_].sum())
        top[d_] = max(cnt.values()) if cnt else 0
    f["ths_mkt_n"] = tot[di]
    f["ths_mkt_topheat"] = top[di]
    f["ths_lag_heat_share"] = np.where(f["ths_mkt_topheat"] > 0, f["ths_lag_heat_other"] / f["ths_mkt_topheat"], np.nan)
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：同花顺「强势股题材」（zx.10jqka.com.cn getharden），每只涨停 / 强势票当天的题材原因文本，2023-10-25..2025-12-31
（更早没有；筛选集只覆盖 2023-10-25..2024-05）。题材 = reason 按「+」切词，去掉「市场超500股涨停」「超跌反弹」等泛词。
时点：收盘后快照，T 日那行可用；最近一次上榜的题材沿用最多 20 个交易日。reason 是否事后改写未核。
覆盖：pool.parquet 从 2023-10-25 起的全部行（2023-10-25 之前这张表没有行）。
列：
  ths_today / ths_n5 / ths_n20 / ths_age   当天在强势股名单 / 近 5、20 日次数 / 距最近一次的交易日数（没有 = NaN）
  ths_theme_max / ths_theme_sum / ths_ntok  最近一次上榜那天：同题材其他强势股只数（最热的那个词 / 全部词合计）/ 词数（只在上榜日更新，其余 NaN 或旧值；配 ths_age 用）
  ths_lag_heat / ths_lag_heat_other         最近一次（<= 20 日）上榜题材在今天的热度（带该词的强势股只数，含 / 不含自己）；20 日内没上过榜 = NaN
  ths_lag_heat_share                        ths_lag_heat_other / 当天最热题材只数
  ths_mkt_n / ths_mkt_topheat               市场级：当天强势股总数 / 最热题材只数
缺失含义：NaN = 20 日内没上过强势股名单（没有题材标签）；0 = 上过但题材今天没人。
"""
    C.write_table("ths_theme", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="ths_theme（筛选段只有 2023-10-25..2024-05）")
    (HERE / "results_ths.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 ths_theme -> results_ths.txt")


if __name__ == "__main__":
    main()
