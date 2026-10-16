"""
ret20 规则（每天取 20 日涨幅最高的 10 只，剔 ST / <2 元）到底选的是什么票：
板块构成、板块内命中、剔北交所之后、新股 / 复牌占比、之后 20 根的回撤和收盘收益、逐月序列。
对照：现状模型前 10（同循环 base_s7）、候选 a158_ovh_cs4 前 10、全市场。
    PYTHONIOENCODING=utf-8 python research/rethink/rules2.py > research/rethink/rules2.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import nw_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
T0 = time.time()
ST = {str(c).zfill(6) for c in json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8")).get("codes", [])}


def say(*a):
    print(*a, flush=True)


def board_of(code: pd.Series) -> pd.Series:
    return pd.Series(np.where(code.str[:2] == "30", "chinext",
                              np.where(code.str[:2] == "68", "star",
                                       np.where(code.str[:1].isin(["9", "8", "4"]), "bj", "main"))), index=code.index)


def load_set(lab_path, px_path, px_start, lab_filters, model_path, model_filters):
    lab = pq.read_table(lab_path, columns=["code", "date", "y_open", "r20_open"], filters=lab_filters).to_pandas()
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    cols = ["code", "date", "open", "high", "low", "close"]
    px = pq.read_table(px_path, columns=cols, filters=[("date", ">=", px_start), ("date", "<=", lab_filters[1][2])]).to_pandas()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    px["ret20"] = g["close"].pct_change(20)
    px["nbar"] = g.cumcount()
    # 复牌：最近 20 根里有没有跨过 ≥ 8 个自然日的缺口（粗判）
    dd = pd.to_datetime(px["date"])
    gap = (dd - dd.groupby(px["code"]).shift(1)).dt.days
    px["gap8"] = (gap >= 8).astype(float)
    px["resume20"] = px.groupby("code")["gap8"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    # 次日开盘买入后 20 根：最大回撤（最低价）和第 20 根收盘收益
    from numpy.lib.stride_tricks import sliding_window_view as swv

    def fut(gr):
        n = len(gr)
        lo = gr["low"].to_numpy(float)
        cl = gr["close"].to_numpy(float)
        op = gr["open"].to_numpy(float)
        mn = np.full(n, np.nan)
        c20 = np.full(n, np.nan)
        if n > 20:
            mn[:n - 20] = swv(lo[1:], 20).min(axis=1)
            c20[:n - 20] = cl[20:]
        o1 = np.append(op[1:], np.nan)
        return pd.DataFrame({"dd20": mn / o1 - 1, "c20": c20 / o1 - 1}, index=gr.index)

    f = pd.concat([fut(gr) for _, gr in px.groupby("code", sort=False)])
    px = px.join(f)
    m = pq.read_table(model_path, filters=model_filters).to_pandas()
    m["code"] = m["code"].astype(str).str.zfill(6)
    r = m.merge(lab, on=["code", "date"]).merge(px[["code", "date", "close", "ret20", "nbar", "resume20", "dd20", "c20"]],
                                               on=["code", "date"], how="left")
    r = r[np.isfinite(r["y_open"])].copy()
    r["y"] = r["y_open"] > 0.5
    r["board"] = board_of(r["code"])
    return r


def top(r, score, k=10, elig=None):
    e = elig if elig is not None else pd.Series(True, index=r.index)
    s = score.where(e, -np.inf)
    rk = s.groupby(r["date"]).rank(ascending=False, method="first")
    return r[(rk <= k) & e]


def describe(tag, sub, r):
    h = sub.groupby("date")["y"].mean()
    say(f"\n[{tag}] 名额 {len(sub)}，命中 {100 * h.mean():.2f}%，零命中天 {100 * (h == 0).mean():.1f}%")
    bc = sub["board"].value_counts(normalize=True).reindex(["main", "chinext", "star", "bj"]).fillna(0)
    bh = sub.groupby("board")["y"].mean().reindex(["main", "chinext", "star", "bj"])
    say("  板块构成 main/chinext/star/bj：%s" % " / ".join("%.0f%%" % (100 * v) for v in bc.values))
    say("  板块内命中：%s" % " / ".join("%.1f%%" % (100 * v) if np.isfinite(v) else "-" for v in bh.values))
    say("  新股（表里 <60 根）占 %.1f%%（命中 %.1f%%）；近 20 根有复牌缺口的占 %.1f%%（命中 %.1f%%）" % (
        100 * (sub["nbar"] < 60).mean(), 100 * sub.loc[sub["nbar"] < 60, "y"].mean() if (sub["nbar"] < 60).any() else np.nan,
        100 * (sub["resume20"] > 0).mean(), 100 * sub.loc[sub["resume20"] > 0, "y"].mean() if (sub["resume20"] > 0).any() else np.nan))
    say("  ret20 中位 %.1f%%；之后 20 根：最高涨幅中位 %.1f%% / 均值 %.1f%%，第 20 根收盘收益中位 %.1f%% / 均值 %.1f%%，"
        "最大回撤中位 %.1f%%，回撤 ≤ −20%% 的占 %.1f%%，收盘跌 ≥ 20%% 的占 %.1f%%" % (
            100 * sub["ret20"].median(), 100 * sub["r20_open"].median(), 100 * sub["r20_open"].mean(),
            100 * sub["c20"].median(), 100 * sub["c20"].mean(), 100 * sub["dd20"].median(),
            100 * (sub["dd20"] <= -0.2).mean(), 100 * (sub["c20"] <= -0.2).mean()))


def main():
    for tag, r in [
        ("开发集 2024-06..2025-12", load_set(EXP / "leak" / "labels_open_snap.parquet", EXP / "leak" / "daily_snap.parquet", "2024-02-01",
                                          [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")],
                                          EXP / "feat_final_v1" / "all_base_s7.parquet",
                                          [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")])),
        ("筛选集 2022-01..2024-05", load_set(EXP / "screen" / "labels_screen.parquet", EXP / "screen" / "px_screen.parquet", "2021-09-01",
                                          [("date", ">=", "2022-01-01"), ("date", "<=", "2024-05-31")],
                                          EXP / "screen_base_v1" / "all_base_s7.parquet", None)),
    ]:
        say(f"\n################ {tag} ################")
        elig = ~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"])
        elig_nobj = elig & (r["board"] != "bj")
        elig_old = elig & (r["nbar"] >= 60) & (r["resume20"] == 0)
        say("全市场：基准 %.2f%%；之后 20 根最高涨幅中位 %.1f%%，第 20 根收盘收益中位 %.1f%%，最大回撤中位 %.1f%%，收盘跌 ≥20%% 占 %.1f%%" % (
            100 * r.groupby("date")["y"].mean().mean(), 100 * r["r20_open"].median(), 100 * r["c20"].median(),
            100 * r["dd20"].median(), 100 * (r["c20"] <= -0.2).mean()))
        describe("ret20 规则 前 10", top(r, r["ret20"], 10, elig), r)
        describe("ret20 规则 前 10，剔北交所", top(r, r["ret20"], 10, elig_nobj), r)
        describe("ret20 规则 前 10，再剔新股和复牌", top(r, r["ret20"], 10, elig_old), r)
        describe("ret20 规则 前 30", top(r, r["ret20"], 30, elig), r)
        describe("现状模型 前 10", top(r, r["p"], 10, elig), r)
        describe("现状模型 前 10，剔北交所", top(r, r["p"], 10, elig_nobj), r)
        # 规则 vs 模型 剔北交所 配对
        hr = top(r, r["ret20"], 10, elig_nobj).groupby("date")["y"].mean()
        hm = top(r, r["p"], 10, elig_nobj).groupby("date")["y"].mean()
        mu, se, t = nw_t((hr - hm.reindex(hr.index)).dropna().values)
        say("\n剔北交所后 规则 − 模型：%+.2fpp，NW t %.2f" % (100 * mu, t))
        hr2 = top(r, r["ret20"], 10, elig_old).groupby("date")["y"].mean()
        hm2 = top(r, r["p"], 10, elig_old).groupby("date")["y"].mean()
        mu, se, t = nw_t((hr2 - hm2.reindex(hr2.index)).dropna().values)
        say("剔新股复牌后 规则 − 模型：%+.2fpp，NW t %.2f" % (100 * mu, t))
        # 逐月
        hr = top(r, r["ret20"], 10, elig).groupby("date")["y"].mean()
        hm = top(r, r["p"], 10, elig).groupby("date")["y"].mean()
        mo = pd.DataFrame({"rule": hr, "model": hm}).groupby(hr.index.str[:7]).mean()
        mo["diff"] = mo["rule"] - mo["model"]
        say("逐月（%%）：规则 / 模型 / 差；规则赢 %d/%d 个月" % ((mo["diff"] > 0).sum(), len(mo)))
        say("  " + "  ".join("%s %.1f/%.1f" % (k, 100 * v["rule"], 100 * v["model"]) for k, v in mo.iterrows()))
        # 连续 3 月滚动最低
        roll = mo["diff"].rolling(3).mean()
        say("  差的连续 3 月滚动平均最低 %+.1fpp（%s）" % (100 * roll.min(), roll.idxmin()))

    # 候选（只有开发集）
    r = load_set(EXP / "leak" / "labels_open_snap.parquet", EXP / "leak" / "daily_snap.parquet", "2024-02-01",
                 [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")],
                 EXP / "feat_final_v1" / "all_cand_s7.parquet", [("date", ">=", "2024-06-01"), ("date", "<=", "2025-12-31")])
    elig = ~r["code"].isin(ST) & (r["close"] >= 2.0) & np.isfinite(r["ret20"])
    say("\n################ 开发集：候选 a158_ovh_cs4 ################")
    describe("候选 前 10", top(r, r["p"], 10, elig), r)
    describe("候选 前 10，剔北交所", top(r, r["p"], 10, elig & (r["board"] != "bj")), r)
    say("\n耗时 %.0f 秒" % (time.time() - T0))



if __name__ == "__main__":
    main()
