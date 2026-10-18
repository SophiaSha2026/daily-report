"""长历史特征表（research/longhist）：raw/part_XX.parquet -> feat/part_XX.parquet -> table_lh.parquet。

    PYTHONIOENCODING=utf-8 python research/longhist/build_lh.py [--parts N] [--stage feat|day|table|check|all]

口径尽量照 research/polish/build_pool.py + research/gate/build_gt.py（FEATS_PRICE 那 38 列 + 9 列日级市场状态），差别：
  - pit 价 = P_u × q_last / q（和 research/screen/prices_screen.py 同），不取整；
  - 涨停幅度按日期：主板 10%；创业板 2020-08-24 起 20%、之前 10%；科创板 20%；北交所 30%。没有历史 ST 名单，ST 的 5% 不认（噪声）；
  - 候选池 = elig0：不复权收盘 >= 2 元（不是 pit 价，早年 pit 价被复权因子压得很小）、>= 120 根、最近 20 根无 >= 5 个交易日停牌缺口、四个输入有值。
    没有 ST 剔除、没有减持 / 解禁 / 增发剔除（历史上没这些表）；
  - 只有幸存者（2026-10 仍在市的票），见 fetch_lh.py 的 probe 结论；
  - 2020-12-23 之前的负样本按 15% 抽样落表（列 w = 1/0.15 补回），正样本全留；2020-12-23 起全部行保留（给两段评估对齐 gate 表）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "breakout", ROOT / "research" / "gate", ROOT / "research" / "precision",
          ROOT / "research" / "polish", ROOT / "research" / "rulecore"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
FEAT = HERE / "feat"
FEAT.mkdir(exist_ok=True)
TABLE = HERE / "table_lh.parquet"
KEEP_ALL_FROM = "2020-12-23"
NEG_RATE = 0.15
PRICE_MIN_U = 2.0
MIN_BARS = 120
SUSP_GAP = 5


def say(*a):
    print(*a, flush=True)


def progress(text: str) -> None:
    from bjtime import now_bj
    with (ROOT / "research" / "blind" / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def board_of(code: pd.Series) -> pd.Series:
    return pd.Series(np.where(code.str[:2] == "30", "chinext",
                              np.where(code.str[:2] == "68", "star",
                                       np.where(code.str[:1].isin(["9", "8", "4"]), "bj", "main"))), index=code.index)


def lim_pct_hist(code: pd.Series, date: pd.Series) -> np.ndarray:
    b2 = code.str[:2]
    pct = np.where(b2 == "68", 20.0, np.where(code.str[:1].isin(["9", "8", "4"]), 30.0, 10.0))
    pct = np.where((b2 == "30") & (date >= "2020-08-24"), 20.0, pct)
    return pct


def calendar(parts: list[Path]) -> np.ndarray:
    ds = set()
    for f in parts:
        ds.update(pd.read_parquet(f, columns=["date"])["date"].unique())
    return np.array(sorted(ds))


def feat_part(f: Path, cal: np.ndarray) -> pd.DataFrame:
    import backfill as BF
    from datasource import limit_price_arr
    from numpy.lib.stride_tricks import sliding_window_view as swv
    u = pd.read_parquet(f)
    u["code"] = u["code"].astype(str).str.zfill(6)
    u = u.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    q_last = u.groupby("code")["qfq_factor"].last()
    bars = u.rename(columns={k + "_u": k for k in ("open", "high", "low", "close")})
    bars = BF.clean_bars(bars)
    bars = bars[np.isfinite(bars["qfq_factor"]) & (bars["qfq_factor"] > 0)].reset_index(drop=True)
    q = bars["qfq_factor"].to_numpy(float) / bars["code"].map(q_last).to_numpy(float)
    px = pd.DataFrame({"code": bars["code"], "date": bars["date"].astype(str).str[:10]})
    for c in ("open", "high", "low", "close"):
        px[c] = bars[c].to_numpy(float) / q
        px[c + "_u"] = bars[c].to_numpy(float)
    for c in ("volume", "amount", "turnover", "outstanding_share"):
        px[c] = bars[c].to_numpy(float)
    px = px.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    g = px.groupby("code", sort=False)
    c = px["close"]
    out = pd.DataFrame({"code": px["code"], "date": px["date"]})
    out["board"] = board_of(px["code"])
    for k in (1, 5, 10, 20, 40, 60):
        out["ret%d" % k] = g["close"].pct_change(k).astype(np.float32)
    out["ret20_lag5"] = (g["close"].shift(5) / g["close"].shift(20) - 1).astype(np.float32)
    lr = np.log(c).groupby(px["code"]).diff()
    out["vol20"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(20).std()).astype(np.float32)
    out["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std()).astype(np.float32)
    for w in (20, 60, 120):
        hi = g["high"].transform(lambda s, w=w: s.rolling(w).max())
        out["dist_hi%d" % w] = (c / hi - 1).astype(np.float32)
    lo20 = g["low"].transform(lambda s: s.rolling(20).min())
    hi20 = g["high"].transform(lambda s: s.rolling(20).max())
    out["pos20"] = ((c - lo20) / (hi20 - lo20 + 1e-12)).astype(np.float32)
    cmax20 = g["close"].transform(lambda s: s.rolling(20).max())
    nh = (c >= cmax20 - 1e-9).astype(float)
    out["nh20_5"] = nh.groupby(px["code"]).transform(lambda s: s.rolling(5).sum()).astype(np.float32)
    r1 = out["ret1"].astype(float)
    out["up20"] = (r1 > 0).astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["max1_20"] = r1.groupby(px["code"]).transform(lambda s: s.rolling(20).max()).astype(np.float32)
    out["amt20"] = g["amount"].transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    out["amt1"] = px["amount"].astype(np.float32)
    out["amt_ratio"] = (px["amount"] / out["amt20"]).astype(np.float32)
    out["turn1"] = px["turnover"].astype(np.float32)
    out["turn5"] = g["turnover"].transform(lambda s: s.rolling(5).mean()).astype(np.float32)
    out["turn20"] = g["turnover"].transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    amp = px["high"] / px["low"] - 1
    out["amp20"] = amp.groupby(px["code"]).transform(lambda s: s.rolling(20).mean()).astype(np.float32)
    v5 = g["volume"].transform(lambda s: s.rolling(5).mean())
    v20 = g["volume"].transform(lambda s: s.rolling(20).mean())
    out["vr5_20"] = (v5 / v20).astype(np.float32)
    out["mcap"] = (px["outstanding_share"] * px["close_u"]).astype(np.float32)
    out["price_u"] = px["close_u"].astype(np.float32)
    out["gap1"] = (px["open"] / g["close"].shift(1) - 1).astype(np.float32)
    out["cr1"] = (px["close"] / px["open"] - 1).astype(np.float32)
    pct = lim_pct_hist(px["code"], px["date"])
    out["lim_pct"] = pct.astype(np.float32)
    prev_u = g["close_u"].shift(1)
    lim = limit_price_arr(prev_u, pct, px["code"])
    lu = (px["close_u"] >= lim - 0.005) & np.isfinite(prev_u)
    out["lu_close"] = lu.astype(np.float32)
    out["yizi_today"] = ((px["low_u"] >= lim - 0.005) & lu).astype(np.float32)
    lu_i = lu.astype(np.int8).to_numpy()
    same = (px["code"].to_numpy()[1:] == px["code"].to_numpy()[:-1])
    streak = np.zeros(len(px), np.int32)
    for i in range(1, len(px)):
        streak[i] = (streak[i - 1] + 1) if (lu_i[i] and same[i - 1]) else lu_i[i]
    streak[0] = lu_i[0]
    out["lu_streak"] = streak.astype(np.float32)
    out["nlu20"] = lu.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["nlu5"] = lu.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(5).sum()).astype(np.float32)
    lim_dn = prev_u * (1 - pct / 100)
    dn = (px["close_u"] <= lim_dn + 0.005) & np.isfinite(prev_u)
    out["nld20"] = dn.astype(float).groupby(px["code"]).transform(lambda s: s.rolling(20).sum()).astype(np.float32)
    out["nbar"] = (g.cumcount() + 1).astype(np.int32)
    # 停牌缺口（全市场交易日序号）
    didx = pd.Series(np.arange(len(cal)), index=cal)
    di = px["date"].map(didx)
    assert di.notna().all()
    di = di.to_numpy(np.int64)
    gap = pd.Series(di - np.r_[di[0], di[:-1]] - 1).where(pd.Series(same, index=range(1, len(px))).reindex(range(len(px))).fillna(False).to_numpy(), 0)
    out["susp20"] = (gap.groupby(px["code"]).transform(lambda s: s.rolling(20, min_periods=1).max()) >= SUSP_GAP).to_numpy()
    out["end20"] = g["date"].shift(-20).fillna("9999-12-31")
    # 标签（buyable.compute 同口径，pit 价上算；次日一字用不复权低价对当日不复权收盘的涨停价）
    lim1 = limit_price_arr(px["close_u"], pct, px["code"])

    def lab(gr):
        n = len(gr)
        high = gr["high"].to_numpy(float)
        opn = gr["open"].to_numpy(float)
        fut = np.full(n, np.nan)
        if n > 20:
            fut[:n - 20] = swv(high[1:], 20).max(axis=1)
        return pd.DataFrame({"fut_max": fut, "open1": np.append(opn[1:], np.nan), "low1_u": np.append(gr["low_u"].to_numpy(float)[1:], np.nan)}, index=gr.index)

    lb = pd.concat([lab(gr) for _, gr in px.groupby("code", sort=False)])
    yizi1 = (lb["low1_u"] >= lim1 - 1e-9) & np.isfinite(lb["low1_u"])
    ok = np.isfinite(lb["fut_max"]) & np.isfinite(lb["open1"]) & (lb["open1"] > 0)
    out["r20_open"] = np.where(ok, lb["fut_max"] / lb["open1"] - 1.0, np.nan).astype(np.float32)
    out["y_open"] = np.where(ok, ((out["r20_open"] > 0.5) & ~yizi1).astype(float), np.nan)
    out["yizi1"] = yizi1.to_numpy()
    out["lret20"] = np.log1p(out["ret20"].astype(float).clip(lower=-0.99)).astype(np.float32)
    out["elig0"] = ((px["close_u"] >= PRICE_MIN_U) & (out["nbar"] >= MIN_BARS) & ~out["susp20"]
                    & np.isfinite(out[["lret20", "vol60", "turn20", "amp20"]]).all(axis=1)).to_numpy()
    return out


def stage_feat(parts):
    cal = calendar(parts)
    say("交易日历 %d 天 %s..%s" % (len(cal), cal[0], cal[-1]))
    t0 = time.time()
    for f in parts:
        o = FEAT / f.name
        if o.exists():
            continue
        d = feat_part(f, cal)
        d.to_parquet(o, index=False)
        say("%s -> %d 行 elig %.1f%% %.0fs" % (f.name, len(d), 100 * d["elig0"].mean(), time.time() - t0))


def stage_day(parts):
    cols = ["code", "date", "lret20", "y_open", "ret1", "lu_close", "ret20", "vol20", "elig0"]
    df = pd.concat([pd.read_parquet(FEAT / f.name, columns=cols) for f in parts], ignore_index=True)
    df = df[df["elig0"]].drop(columns=["elig0"])
    df = df.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    df["rank_ret20"] = (df.groupby("date").cumcount() + 1).astype(np.int32)
    n_day = df.groupby("date")["code"].transform("size")
    df["prank_ret20"] = (df["rank_ret20"] / n_day).astype(np.float32)
    day = df.groupby("date").agg(pool_n=("code", "size"), base=("y_open", "mean"), mret1=("ret1", "mean"),
                                 lu_share=("lu_close", "mean"), hot50=("ret20", lambda s: float((s > 0.5).mean())),
                                 mkt_vol20=("vol20", "median")).sort_index()
    day["mkt_base_t21"] = day["base"].rolling(20).mean().shift(21)
    day["mkt_ret20"] = day["mret1"].rolling(20).sum()
    day["mkt_ret5"] = day["mret1"].rolling(5).sum()
    day["lu_share20"] = day["lu_share"].rolling(20).mean()
    d2 = df[df["rank_ret20"] <= 10].groupby("date")["ret20"].agg(top1_ret20="max", top10_ret20_med="median")
    day = day.join(d2).drop(columns=["base", "mret1", "lu_share"]).astype(np.float32)
    day.reset_index().to_parquet(HERE / "day_lh.parquet", index=False)
    df[["code", "date", "rank_ret20", "prank_ret20"]].to_parquet(HERE / "rank_lh.parquet", index=False)
    say("day 表 %d 天；候选池行 %d；每天 %.0f 只（2008 %.0f / 2015 %.0f / 2023 %.0f）" % (
        len(day), len(df), day["pool_n"].mean(), day.loc[day.index.str[:4] == "2008", "pool_n"].mean(),
        day.loc[day.index.str[:4] == "2015", "pool_n"].mean(), day.loc[day.index.str[:4] == "2023", "pool_n"].mean()))


def stage_table(parts):
    import common_gt as G
    day = pd.read_parquet(HERE / "day_lh.parquet")
    rk = pd.read_parquet(HERE / "rank_lh.parquet")
    rng = np.random.default_rng(7)
    outs = []
    for f in parts:
        d = pd.read_parquet(FEAT / f.name)
        d = d[d["elig0"]].drop(columns=["elig0"]).reset_index(drop=True)
        d = d.merge(rk, on=["code", "date"], how="left").merge(day, on="date", how="left")
        d["lmcap"] = np.log(d["mcap"].clip(lower=1e6)).astype(np.float32)
        d["lamt20"] = np.log(d["amt20"].clip(lower=1e3)).astype(np.float32)
        d["lvol60"] = np.log(d["vol60"].clip(lower=1e-4)).astype(np.float32)
        d["lturn20"] = np.log(d["turn20"].clip(lower=1e-3)).astype(np.float32)
        d["lprice"] = np.log(d["price_u"].clip(lower=0.1)).astype(np.float32)
        d["lret20_lag5"] = np.log1p(d["ret20_lag5"].astype(float).clip(lower=-0.99)).astype(np.float32)
        d["lret60"] = np.log1p(d["ret60"].astype(float).clip(lower=-0.99)).astype(np.float32)
        d["board_i"] = d["board"].map({"main": 0, "chinext": 1, "star": 2, "bj": 3}).astype(np.int8)
        early = (d["date"] < KEEP_ALL_FROM).to_numpy()
        neg = (d["y_open"].to_numpy() < 0.5) & np.isfinite(d["y_open"].to_numpy())
        keep = ~early | ~neg | (rng.random(len(d)) < NEG_RATE)
        d["w"] = np.where(early & neg, 1.0 / NEG_RATE, 1.0).astype(np.float32)
        d = d[keep]
        cols = ["code", "date", "board", "y_open", "r20_open", "yizi1", "end20", "ret20", "lret20", "rank_ret20", "nbar", "w",
                "vol60", "turn20", "mcap"] + \
               [c for c in G.FEATS_PRICE if c not in ("nbar", "lret20", "rank_ret20")]
        outs.append(d[cols])
        say("%s 留 %d 行" % (f.name, len(d)))
    t = pd.concat(outs, ignore_index=True)
    t.to_parquet(TABLE, index=False)
    say("-> %s %d 行 %d 列 %s..%s；y_open 正样本率（按 w 加权，2010 前 / 2010-2019 / 2020 起）：%s" % (
        TABLE, len(t), t.shape[1], t["date"].min(), t["date"].max(),
        {k: round(float(np.average(v["y_open"].fillna(0), weights=v["w"])) * 100, 2) for k, v in
         t[np.isfinite(t["y_open"])].groupby(np.where(t.loc[np.isfinite(t["y_open"]), "date"] < "2010", "<2010", np.where(t.loc[np.isfinite(t["y_open"]), "date"] < "2020", "2010s", "2020+")))}))


def stage_check():
    import common_gt as G
    t = pd.read_parquet(TABLE, columns=["code", "date", "ret20", "vol60", "lu_close", "nlu20", "y_open", "turn20", "mcap", "rank_ret20"])
    g = G.load_table(["code", "date", "ret20", "vol60", "lu_close", "nlu20", "y_open", "turn20", "mcap", "rank_ret20", "seg"])
    g["code"] = g["code"].astype(str).str.zfill(6)
    g = g[g["seg"] != "pre"]
    m = g.merge(t, on=["code", "date"], how="left", suffixes=("_g", "_l"))
    say("gate 两段 %d 行，在长历史表里找到 %.2f%%" % (len(g), 100 * np.isfinite(m["ret20_l"]).mean()))
    mm = m[np.isfinite(m["ret20_l"])]
    for c in ("ret20", "vol60", "turn20", "mcap"):
        a, b = mm[c + "_g"].to_numpy(float), mm[c + "_l"].to_numpy(float)
        rel = np.abs(a - b) / (np.abs(a) + 1e-9)
        say("  %s：相对差中位 %.2e，> 1%% 的行 %.3f%%，相关 %.5f" % (c, np.median(rel), 100 * (rel > 0.01).mean(), np.corrcoef(a, b)[0, 1]))
    for c in ("lu_close", "nlu20", "y_open"):
        ok = np.isfinite(mm[c + "_g"]) & np.isfinite(mm[c + "_l"])
        say("  %s：一致率 %.4f%%" % (c, 100 * (mm.loc[ok, c + "_g"] == mm.loc[ok, c + "_l"]).mean()))
    # 名次：gate 前 10 的行在长历史表里的名次分布（长历史池没剔 ST / 事件，名次只会更靠后）
    top = mm[mm["rank_ret20_g"] <= 10]
    say("  gate 前 10 行在长历史池里的名次：中位 %.0f，<= 10 的 %.1f%%，<= 20 的 %.1f%%" % (
        top["rank_ret20_l"].median(), 100 * (top["rank_ret20_l"] <= 10).mean(), 100 * (top["rank_ret20_l"] <= 20).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--stage", default="all", choices=["feat", "day", "table", "check", "all"])
    a = ap.parse_args()
    parts = sorted(RAW.glob("part_*.parquet"))
    if a.parts:
        parts = parts[:a.parts]
    t0 = time.time()
    if a.stage in ("feat", "all"):
        stage_feat(parts)
    if a.stage in ("day", "all"):
        stage_day(parts)
    if a.stage in ("table", "all"):
        stage_table(parts)
    if a.stage in ("check", "all"):
        stage_check()
    say("%.0fs" % (time.time() - t0))
    if a.stage == "all" and not a.parts:
        progress("build_lh：长历史特征表 table_lh.parquet 建好（%d 个分片，%.0f 分钟）" % (len(parts), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
