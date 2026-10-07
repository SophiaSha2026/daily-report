"""
实验 16：准确率发散（2026-10-06 用户「一条一条 + 组合进行尝试」）。

六条（docs/breakout_log.md 实验 15 之后的讨论）：
    1 标签改成「买得到」：次日开盘起算，次日一字板记不可买；再加路径（到顶前最大回撤）
    2 全量负样本加权重；五种子平均
    3 新特征：涨停历史、复牌天数、长期调整突破的形态阶段
    4 按天分组的排序学习（lambdarank）
    5 板块共振特征（行业映射来自 archive/morning/cache/sector_map.parquet）
    6 历史拉到 2019（另一张表，不动生产）

协议和实验 15 一样：验证集 2025-03..12 逐月走向前，train_slice 净化线，同一份特征选择，
标准误按天聚类、臂之间按天配对。模型默认按月等权（实验 15 上线那套）。
label.py / features.py 是模型指纹的一部分，这里**不动它们**：新标签、新特征都在本脚本里
从 daily.parquet 算，按 (code, date) 并到训练表上。采纳了再搬进正式模块、走教训 36 流水。

产物 data/breakout/raw/exp_acc/（gitignore）：labels.parquet、feats.parquet、preds_*.parquet、
summary_*.json。

    python src/breakout/exp_acc.py --labels        算新标签（含和 train.parquet 的 y_up 对账）
    python src/breakout/exp_acc.py --report1       第 1 条：买不到的命中占几成
    python src/breakout/exp_acc.py --feats         算新特征
    python src/breakout/exp_acc.py --fit ARM,...   训练指定臂
    python src/breakout/exp_acc.py --analyze       汇总
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

import datasource as ds  # noqa: E402
from label import UP_THRESHOLD, UP_WINDOW  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
DATA = ROOT / "data" / "breakout"
OUT = DATA / "raw" / "exp_acc"
WF = DATA / "raw" / "wf_scores.parquet"


# ---------------------------------------------------------------
#  1. 标签
# ---------------------------------------------------------------
def _per_code_labels(g: pd.DataFrame, window: int = UP_WINDOW) -> pd.DataFrame:
    """一只票：未来 window 根的最高价 / 次日开盘 / 次日一字 / 到顶前最大回撤。"""
    n = len(g)
    close = g["close"].to_numpy(float)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    opn = g["open"].to_numpy(float)
    fut = np.full(n, np.nan)
    peak_i = np.full(n, -1)
    dd = np.full(n, np.nan)          # 买入（次日开盘）到最高点之前的最大回撤（相对次日开盘）
    if n > window:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        wh = swv(high[1:], window)            # 行 i: high[i+1 .. i+window]
        wl = swv(low[1:], window)
        m = len(wh)                           # = n - window
        fut[:m] = wh.max(axis=1)
        pk = wh.argmax(axis=1)
        peak_i[:m] = pk
        # 到最高点那根为止（含）的最低价
        cm = np.minimum.accumulate(wl, axis=1)
        lo_before = cm[np.arange(m), pk]
        o1 = opn[1:1 + m]
        with np.errstate(divide="ignore", invalid="ignore"):
            dd[:m] = lo_before / o1 - 1.0
    open1 = np.append(opn[1:], np.nan)
    low1 = np.append(low[1:], np.nan)
    return pd.DataFrame({"code": g["code"].to_numpy(), "date": g["date"].to_numpy(),
                         "close": close, "fut_max": fut, "open1": open1, "low1": low1,
                         "dd_before_peak": dd})


def build_labels() -> pd.DataFrame:
    t0 = time.time()
    px = pd.read_parquet(DATA / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close"])
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    parts = [_per_code_labels(g) for _, g in px.groupby("code", sort=False)]
    d = pd.concat(parts, ignore_index=True)
    # 次日涨停价：昨收 = 今收，按板块幅度，北交所向下取整
    pct = d["code"].map(lambda c: ds.limit_pct(c, ""))
    d["lim1"] = ds.limit_price_arr(d["close"], pct, d["code"])
    d["yizi1"] = d["low1"] >= d["lim1"] - 1e-9          # 次日全天没低于涨停价 = 一字
    ok = np.isfinite(d["fut_max"]) & (d["close"] > 0)
    d["r20"] = np.where(ok, d["fut_max"] / d["close"] - 1.0, np.nan)
    d["y_up_chk"] = np.where(ok, (d["r20"] > UP_THRESHOLD).astype(float), np.nan)
    ok1 = ok & np.isfinite(d["open1"]) & (d["open1"] > 0)
    d["r20_open"] = np.where(ok1, d["fut_max"] / d["open1"] - 1.0, np.nan)
    # 买得到的命中：次日开盘起算涨超 50%，且次日不是一字板
    d["y_open"] = np.where(ok1, ((d["r20_open"] > UP_THRESHOLD) & ~d["yizi1"]).astype(float), np.nan)
    # 路径版：到顶之前相对买入价回撤不超过 15%
    d["y_open_dd15"] = np.where(ok1, (d["y_open"] > 0) & (d["dd_before_peak"] > -0.15), np.nan)
    d["y_open_dd15"] = d["y_open_dd15"].astype(float)
    d.loc[~ok1, "y_open_dd15"] = np.nan

    # 对账：和 train.parquet 的 y_up 必须逐行一致（这里的算法就是 label_up 的向量化）
    tr = pd.read_parquet(DATA / "train.parquet", columns=["code", "date", "y_up"])
    tr["code"] = tr["code"].astype(str).str.zfill(6)
    m = tr.merge(d[["code", "date", "y_up_chk"]], on=["code", "date"], how="left")
    both = m[np.isfinite(m["y_up"]) & np.isfinite(m["y_up_chk"])]
    mism = int((both["y_up"] != both["y_up_chk"]).sum())
    print("对账：训练表 %d 行，两边都有标签 %d 行，不一致 %d 行" % (len(m), len(both), mism))
    assert mism == 0, "重算的 y_up 和 train.parquet 不一致，标签算法有问题"
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["code", "date", "y_up_chk", "y_open", "y_open_dd15", "r20", "r20_open",
            "yizi1", "dd_before_peak"]
    d[keep].to_parquet(OUT / "labels.parquet", index=False)
    print("标签 %d 行，%.0fs -> %s" % (len(d), time.time() - t0, OUT / "labels.parquet"))
    return d


def report1() -> dict:
    """第 1 条：现在的清单里，买不到的命中占几成。用 wf_scores 缓存（实验 15 模型）。"""
    import daily as D
    lab = pd.read_parquet(OUT / "labels.parquet")
    w = pd.read_parquet(WF)
    w["code"] = w["code"].astype(str).str.zfill(6)
    w = w.merge(lab, on=["code", "date"], how="left")
    prod = w[(w["score"] >= D.SCORE_MIN) & (w["rank"] <= D.CAP_A)]
    top10 = w[w["rank"] <= 10]
    out = {}
    for name, g in (("prod", prod), ("top10", top10)):
        g = g[np.isfinite(g["y_up"])]
        hit = g[g["y_up"] > 0]
        r = {"n": int(len(g)),
             "hit_close": float(g["y_up"].mean()),
             "yizi_next_day_all": float(g["yizi1"].mean()),
             "yizi_next_day_among_hits": float(hit["yizi1"].mean()) if len(hit) else None,
             "hit_open": float(g["y_open"].mean()),
             "hit_open_dd15": float(g["y_open_dd15"].mean()),
             "median_r20_open": float(g["r20_open"].median()),
             "gap_next_open_median": float((g["r20"] - g["r20_open"]).median())}
        out[name] = r
        print("[%s] n=%d  收盘口径命中 %.2f%% | 次日一字 全部 %.1f%% 命中里 %.1f%% | "
              "次日开盘起算 %.2f%% | 再要求到顶前回撤<15%% %.2f%%" % (
                  name, r["n"], 100 * r["hit_close"], 100 * r["yizi_next_day_all"],
                  100 * (r["yizi_next_day_among_hits"] or 0), 100 * r["hit_open"],
                  100 * r["hit_open_dd15"]))
    base = lab[lab["date"].between("2025-03-01", "2025-12-31")]
    out["base"] = {"close": float(base["y_up_chk"].mean()), "open": float(base["y_open"].mean()),
                   "open_dd15": float(base["y_open_dd15"].mean())}
    print("全市场基准：收盘口径 %.2f%%，次日开盘口径 %.2f%%，加回撤 %.2f%%" % (
        100 * out["base"]["close"], 100 * out["base"]["open"], 100 * out["base"]["open_dd15"]))
    (OUT / "summary_report1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------
#  3 / 5. 新特征（只从 daily.parquet 算，t 日收盘后已知）
# ---------------------------------------------------------------
def _per_code_feats(g: pd.DataFrame, open_idx: dict) -> pd.DataFrame:
    n = len(g)
    close = g["close"].to_numpy(float)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    vol = g["volume"].to_numpy(float)
    code = str(g["code"].iloc[0])
    pct = ds.limit_pct(code, "")
    prev = np.r_[np.nan, close[:-1]]
    lim = ds.limit_price_arr(pd.Series(prev), pct, pd.Series([code] * n)).to_numpy(float)
    lu = (close >= lim - 1e-9) & np.isfinite(lim)                      # 收盘封板
    zb = (high >= lim - 1e-9) & ~lu & np.isfinite(lim)                 # 摸到涨停价没封住
    cs = np.cumsum(lu).astype(float)
    lu20 = cs - np.r_[np.zeros(20), cs[:-20]] if n > 20 else cs
    cz = np.cumsum(zb).astype(float)
    zb20 = cz - np.r_[np.zeros(20), cz[:-20]] if n > 20 else cz
    streak = np.zeros(n)
    for i in range(n):
        streak[i] = streak[i - 1] + 1 if (lu[i] and i > 0) else float(lu[i])
    # 停牌：相邻两行之间漏掉的开市日数
    oi = g["date"].map(open_idx).to_numpy(float)
    gap = np.r_[0.0, np.diff(oi) - 1]
    gap = np.where(np.isfinite(gap), gap, 0.0)
    since = np.full(n, 60.0)
    last = -1
    for i in range(n):
        if gap[i] >= 1:
            last = i
        if last >= 0:
            since[i] = min(60.0, i - last)
    # 长期调整突破的形态阶段（config.yaml pullback 段的默认口径，简化版）
    big = (close / prev - 1 >= 0.10) if pct > 10 else lu
    v1 = np.r_[np.nan, vol[:-1]]
    vol_ok = vol >= 1.5 * v1
    sy = np.zeros(n, bool)
    for i in range(60, n):
        if big[i] and vol_ok[i]:
            w = vol[i - 60:i]
            c = close[i - 60:i]
            if vol[i] >= 3 * w.mean() and vol[i] > w.max() and c.max() / c.min() <= 1.30:
                sy[i] = True
    d_sy = np.full(n, 11.0)          # 距最近一次首阳几根（>10 记 11）
    vr_sy = np.full(n, np.nan)       # 今量 / 首阳量
    above_sy = np.zeros(n)           # 收盘在首阳最高价之上
    broke_sy = np.zeros(n)           # 最低价跌破首阳最低价
    last = -1
    for i in range(n):
        if sy[i]:
            last = i
        if last >= 0 and i - last <= 10:
            d_sy[i] = i - last
            vr_sy[i] = vol[i] / vol[last] if vol[last] > 0 else np.nan
            above_sy[i] = float(close[i] > high[last])
            broke_sy[i] = float(low[last + 1:i + 1].min() < low[last]) if i > last else 0.0
    return pd.DataFrame({"code": g["code"].to_numpy(), "date": g["date"].to_numpy(),
                         "x_lu20": lu20, "x_zb20": zb20, "x_lu_streak": streak,
                         "x_since_resume": since, "x_sy_days": d_sy, "x_sy_volratio": vr_sy,
                         "x_sy_above": above_sy, "x_sy_broke": broke_sy,
                         "ret1": close / prev - 1, "lu": lu.astype(float)})


def build_feats() -> pd.DataFrame:
    t0 = time.time()
    px = pd.read_parquet(DATA / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume"])
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    cnt = px.groupby("date").size()
    open_days = sorted(cnt[cnt >= 1000].index)
    open_idx = {d: i for i, d in enumerate(open_days)}
    parts = [_per_code_feats(g, open_idx) for _, g in px.groupby("code", sort=False)]
    f = pd.concat(parts, ignore_index=True)
    # 板块共振：静态行业映射（archive/morning/cache/sector_map.parquet，归档时的快照）
    sm = pd.read_parquet(ROOT / "archive" / "morning" / "cache" / "sector_map.parquet")
    sm["code"] = sm["code"].astype(str).str.zfill(6)
    f = f.merge(sm, on="code", how="left")
    f = f.sort_values(["code", "date"]).reset_index(drop=True)
    f["ret5"] = f.groupby("code")["ret1"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    g = f.groupby(["date", "sector"])
    f["x_sec_ret1"] = g["ret1"].transform("mean")
    f["x_sec_ret5"] = g["ret5"].transform("mean")
    f["x_sec_lu_share"] = g["lu"].transform("mean")
    f["x_sec_adv"] = g["ret1"].transform(lambda s: (s > 0).mean())
    f["x_sec_n"] = g["ret1"].transform("size").astype(float)
    f["x_rel_sec5"] = f["ret5"] - f["x_sec_ret5"]
    # 当日全市场排名（和其它特征同一层变换）；计数 / 标志类保留原值
    for c in ("x_sec_ret1", "x_sec_ret5", "x_sec_lu_share", "x_sec_adv", "x_rel_sec5",
              "x_sy_volratio"):
        f[c + "_pct"] = f.groupby("date")[c].rank(pct=True)
    keep = ["code", "date"] + [c for c in f.columns if c.startswith("x_")]
    OUT.mkdir(parents=True, exist_ok=True)
    f[keep].to_parquet(OUT / "feats.parquet", index=False)
    print("特征 %d 行 %d 列，%.0fs（行业映射缺失 %.1f%%）-> %s" % (
        len(f), len(keep) - 2, time.time() - t0, 100 * f["sector"].isna().mean(),
        OUT / "feats.parquet"))
    return f[keep]


F3 = ["x_lu20", "x_zb20", "x_lu_streak", "x_since_resume", "x_sy_days",
      "x_sy_volratio_pct", "x_sy_above", "x_sy_broke"]
F5 = ["x_sec_ret1_pct", "x_sec_ret5_pct", "x_sec_lu_share_pct", "x_sec_adv_pct",
      "x_rel_sec5_pct", "x_sec_n"]


# ---------------------------------------------------------------
#  训练臂
# ---------------------------------------------------------------
ARMS = {
    "base":      dict(target="y_up"),
    "open":      dict(target="y_open"),
    "open_dd":   dict(target="y_open_dd15"),
    "allneg":    dict(target="y_open", neg="all"),
    "bag5":      dict(target="y_open", seeds=5),
    "f3":        dict(target="y_open", extra=F3),
    "f5":        dict(target="y_open", extra=F5),
    "f35":       dict(target="y_open", extra=F3 + F5),
    "rank":      dict(target="y_open", model="rank"),
    "rank_f35":  dict(target="y_open", model="rank", extra=F3 + F5),
    "bag5_f35":  dict(target="y_open", seeds=5, extra=F3 + F5),
    "all":       dict(target="y_open", seeds=5, extra=F3 + F5, neg="all"),
    # 第 6 条：2019 年起的扩展历史（另一张表，raw/hist2019/train.parquet）。
    # hist_2023 是对照：同一张表但只用 2023-12 起的行（和生产表同一段），把
    # 「表不一样」和「历史更长」两件事拆开
    "hist_2023": dict(target="y_up", table="hist2019", since="2023-12-01"),
    "hist_all":  dict(target="y_up", table="hist2019", since="2019-01-01"),
}
NEG_PER_POS = 12
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                  n_jobs=12, verbose=-1)


def _grade(r: np.ndarray, y_open: np.ndarray) -> np.ndarray:
    """排序学习的分级相关性：买得到的 50% = 3，其余按 20 日最高涨幅（次日开盘起算）分档。"""
    g = np.zeros(len(r), dtype=int)
    g[r > 0.10] = 1
    g[r > 0.25] = 2
    g[y_open > 0] = 3
    return g


def _fit_one(trs: pd.DataFrame, feats: list[str], target: str, model: str, seed: int,
             w_cls: np.ndarray | None):
    """权重 = 按月等权 ×（全量负样本臂的类权重）。"""
    import lightgbm as lgb
    import model as M
    if model == "rank":
        order = np.argsort(trs["date"].to_numpy(), kind="stable")
        trs = trs.iloc[order]
        w_cls = None if w_cls is None else w_cls[order]
    w = M.month_weights(trs)
    if w_cls is not None:
        w = w * w_cls
        w = w / w.mean()
    X, y = M._xy(trs, feats, target)
    if model == "rank":
        grade = _grade(trs["r20_open"].to_numpy(float), trs[target].to_numpy(float))
        grp = trs.groupby("date", sort=False).size().to_numpy()
        assert grp.sum() == len(trs)
        m = lgb.LGBMRanker(objective="lambdarank", random_state=seed,
                           label_gain=[0, 1, 3, 7], lambdarank_truncation_level=20,
                           **LGB_PARAMS)
        m.fit(X, grade, group=grp, sample_weight=w)
        return m
    m = lgb.LGBMClassifier(objective="binary", random_state=seed, **LGB_PARAMS)
    m.fit(X, y, sample_weight=w)
    return m


def _predict(m, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    import model as M
    X, _ = M._xy(df, feats, feats[0])
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.predict(X)


def _sample(tr: pd.DataFrame, target: str, neg: str, seed: int):
    import model as M
    if neg == "all":
        # 全量负样本：每个月的负样本总权重压到正样本的 NEG_PER_POS 倍
        # （和下采样后的样本同一个正负比，只是不扔数据）
        mon = tr["date"].str[:7]
        pos = (tr[target] > 0).to_numpy()
        npos = pd.Series(pos).groupby(mon.to_numpy()).transform("sum").to_numpy(float)
        nneg = pd.Series(~pos).groupby(mon.to_numpy()).transform("sum").to_numpy(float)
        w_cls = np.where(pos, 1.0, np.minimum(1.0, NEG_PER_POS * npos / np.maximum(nneg, 1)))
        return tr, w_cls
    return M.stratified_sample(tr, target, seed=seed), None


# ---------------------------------------------------------------
#  第二轮（2026-10-07 用户：「一定有规律的」）：妖股史 / 同涨族群 / 市值价位 / 尾部二级模型
# ---------------------------------------------------------------
PEER_K = 20          # 每只票的同涨伙伴数
PEER_WIN = 60        # 相关性窗口（交易日）
PEER_MIN = 40        # 窗口里至少这么多有效收益才参与
TAIL_N = 100         # 二级模型只看每天一级分数前这么多名


def build_feats2() -> pd.DataFrame:
    """妖股史（从已关闭窗口的标签数）+ 动态同涨族群（每月按相关性重算伙伴）。"""
    import scipy.sparse as sp
    t0 = time.time()
    lab = pd.read_parquet(OUT / "labels.parquet", columns=["code", "date", "y_up_chk"])
    lab = lab.sort_values(["code", "date"]).reset_index(drop=True)
    g = lab.groupby("code", sort=False)["y_up_chk"]
    # 第 t 行能看到的只有窗口已关闭的事件：t−21 及更早（20 根窗口 + 1）
    closed = g.shift(UP_WINDOW + 1).fillna(0.0)
    cg = closed.groupby(lab["code"], sort=False)
    lab["x_ev250"] = cg.transform(lambda s: s.rolling(250, min_periods=1).sum())
    lab["x_ev500"] = cg.transform(lambda s: s.rolling(500, min_periods=1).sum())
    # 距上一次（已关闭的）起涨事件多少根；没有就 600
    idx = np.arange(len(lab))
    last_ev = pd.Series(np.where(closed.to_numpy() > 0, idx, np.nan)).groupby(
        lab["code"].to_numpy(), sort=False).ffill().to_numpy()
    lab["x_ev_since"] = np.minimum(600.0, np.where(np.isfinite(last_ev), idx - last_ev, 600.0))
    f = lab[["code", "date", "x_ev250", "x_ev500", "x_ev_since"]]

    px = pd.read_parquet(DATA / "daily.parquet", columns=["code", "date", "close"])
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.zfill(6)
    wide = px.pivot(index="date", columns="code", values="close").sort_index()
    ret = wide.pct_change(fill_method=None)
    ret5 = wide.pct_change(5, fill_method=None)
    fx = pd.read_parquet(OUT / "feats.parquet", columns=["code", "date", "x_lu20"])
    lu = (fx.pivot(index="date", columns="code", values="x_lu20").sort_index()
            .diff().clip(lower=0).reindex(index=ret.index, columns=ret.columns))
    # x_lu20 是 20 日累计，diff>0 意味着今天封板；首行没法 diff，当 0
    lu = lu.fillna(0.0)
    dates = list(ret.index)
    codes = list(ret.columns)
    n = len(codes)
    months = sorted({d[:7] for d in dates})
    rows = []
    peers_of = None
    for m in months:
        first = next((i for i, d in enumerate(dates) if d[:7] == m), None)
        if first is None or first < PEER_WIN:
            continue
        W = ret.iloc[first - PEER_WIN:first].to_numpy(float)          # 月初之前 60 天
        ok = np.isfinite(W).sum(axis=0) >= PEER_MIN
        Z = np.where(np.isfinite(W), W, np.nan)
        Z = Z - np.nanmean(Z, axis=0)
        Z = np.nan_to_num(Z / (np.nanstd(Z, axis=0) + 1e-12))
        C = (Z.T @ Z) / max(PEER_WIN - 1, 1)
        np.fill_diagonal(C, -np.inf)
        C[:, ~ok] = -np.inf
        C[~ok, :] = -np.inf
        top = np.argpartition(-C, PEER_K, axis=1)[:, :PEER_K]
        corr_mean = np.take_along_axis(C, top, axis=1)
        corr_mean = np.where(np.isfinite(corr_mean), corr_mean, np.nan).mean(axis=1)
        rows_i = np.repeat(np.arange(n), PEER_K)
        P = sp.csr_matrix((np.full(n * PEER_K, 1.0 / PEER_K), (rows_i, top.ravel())), shape=(n, n))
        P = sp.diags(ok.astype(float)) @ P
        peers_of = P
        for i in range(first, len(dates)):
            if dates[i][:7] != m:
                break
            r1 = np.nan_to_num(ret.iloc[i].to_numpy(float))
            r5 = np.nan_to_num(ret5.iloc[i].to_numpy(float))
            l1 = lu.iloc[i].to_numpy(float)
            pr1 = peers_of @ r1
            pr5 = peers_of @ r5
            plu = peers_of @ l1
            rows.append(pd.DataFrame({
                "code": codes, "date": dates[i], "x_peer_ret1": pr1, "x_peer_ret5": pr5,
                "x_peer_lu": plu, "x_rel_peer5": r5 - pr5, "x_peer_corr": corr_mean,
                "_ok": ok}))
        print("  同涨族群 %s 完成" % m, flush=True)
    peer = pd.concat(rows, ignore_index=True)
    peer = peer[peer["_ok"]].drop(columns=["_ok"])
    for c in ("x_peer_ret1", "x_peer_ret5", "x_peer_lu", "x_rel_peer5", "x_peer_corr"):
        peer[c + "_pct"] = peer.groupby("date")[c].rank(pct=True)
    out = f.merge(peer, on=["code", "date"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "feats2.parquet", index=False)
    print("第二轮特征 %d 行 %d 列，%.0fs -> %s" % (len(out), len(out.columns) - 2,
                                                 time.time() - t0, OUT / "feats2.parquet"))
    return out


G1 = ["x_mcap_pct", "x_price_pct", "x_ev250", "x_ev500", "x_ev_since"]
G2 = ["x_peer_ret1_pct", "x_peer_ret5_pct", "x_peer_lu_pct", "x_rel_peer5_pct", "x_peer_corr_pct"]
ARMS.update({
    "g1":       dict(target="y_up", extra2=G1),
    "g2":       dict(target="y_up", extra2=G2),
    "g12":      dict(target="y_up", extra2=G1 + G2),
    "tail":     dict(target="y_up", stage2=True),
    "tail_g12": dict(target="y_up", stage2=True, extra2=G1 + G2),
})


def _oof_scores(tr: pd.DataFrame, trs: pd.DataFrame, feats: list[str], target: str,
                seed: int) -> np.ndarray:
    """交叉拟合：按月份奇偶分两半，各训一个一级模型给另一半打分。tr 是全量训练行。"""
    mon_i = pd.factorize(tr["date"].str[:7], sort=True)[0]
    mon_s = pd.factorize(trs["date"].str[:7], sort=True)[0]
    mons = sorted(set(tr["date"].str[:7]))
    half = {m: (k % 2) for k, m in enumerate(mons)}
    hs = trs["date"].str[:7].map(half).to_numpy()
    ht = tr["date"].str[:7].map(half).to_numpy()
    out = np.zeros(len(tr))
    del mon_i, mon_s
    for h in (0, 1):
        m = _fit_one(trs[hs != h], feats, target, "lgb", seed, None)
        out[ht == h] = _predict(m, tr[ht == h], feats)
    return out


def fit_two_stage(tr: pd.DataFrame, trs: pd.DataFrame, te: pd.DataFrame, feats: list[str],
                  target: str, seed: int, tail_n: int = TAIL_N) -> tuple[np.ndarray, np.ndarray]:
    """一级：现状模型。二级：只在每天一级分数前 tail_n 名的行上训练，重排测试月的前 tail_n。

    返回 (pure, blend)：pure = 尾部按二级分排、尾部外按一级分排；
    blend = 尾部内一级名次和二级名次取平均。
    """
    import lightgbm as lgb
    import model as M
    oof = _oof_scores(tr, trs, feats, target, seed)
    t = tr[["date", target]].copy()
    t["_s"] = oof
    t["_r"] = t.groupby("date")["_s"].rank(ascending=False, method="first")
    tail = tr[(t["_r"] <= tail_n).to_numpy()]
    w = M.month_weights(tail)
    X, y = M._xy(tail, feats, target)
    m2 = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.03,
                            num_leaves=15, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                            random_state=seed, n_jobs=12, verbose=-1)
    m2.fit(X, y, sample_weight=w)
    m1 = _fit_one(trs, feats, target, "lgb", seed, None)
    p1 = _predict(m1, te, feats)
    d = te[["date"]].copy()
    d["_p1"] = p1
    d["_r1"] = d.groupby("date")["_p1"].rank(ascending=False, method="first")
    in_tail = (d["_r1"] <= tail_n).to_numpy()
    p2 = np.zeros(len(te))
    p2[in_tail] = _predict(m2, te[in_tail], feats)
    # 尾部整体抬到一级分之上：尾部外最大 p1 + 二级分；尾部内按 p2 排
    base_top = float(d.loc[~in_tail, "_p1"].max()) if (~in_tail).any() else 0.0
    pure = np.where(in_tail, base_top + 1.0 + p2, p1)
    d["_p2"] = p2
    d["_r2"] = d.groupby("date")["_p2"].rank(ascending=False, method="first")
    blend_rank = (d["_r1"] + d["_r2"]) / 2.0
    blend = np.where(in_tail, base_top + 1.0 + 1.0 / (1.0 + blend_rank.to_numpy()), p1)
    print("    尾部训练行 %d，正样本率 %.1f%%" % (len(tail), 100 * tail[target].mean()), flush=True)
    return pure, blend


def _read_f32(path: Path, since: str, until: str) -> pd.DataFrame:
    """分列组读 parquet，边读边转 float32、边按日期过滤：5.7 GB 的扩展表直接 read_parquet
    是 float64，峰值内存会超过这台机器的 14 GB。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = pf.schema.names
    date = pf.read(columns=["date"]).to_pandas()["date"].astype(str)
    mask = ((date >= since) & (date < until)).to_numpy()
    keep = ["code", "date", "board", "y_up"] + [c for c in names if "__" in c]
    parts = []
    for i in range(0, len(keep), 25):
        cols = keep[i:i + 25]
        d = pf.read(columns=cols).to_pandas()[mask]
        for c in cols:
            if "__" in c or c == "y_up":
                d[c] = d[c].astype("float32")
        parts.append(d.reset_index(drop=True))
    out = pd.concat(parts, axis=1)
    out["date"] = out["date"].astype(str)
    return out

def fit_arms(names: list[str]) -> None:
    import arena as A
    import fselect as FS
    import model as M
    import validate as V
    t_all = time.time()
    tables = {ARMS[n].get("table") for n in names}
    assert len(tables) == 1, "一次只能跑同一张表上的臂"
    table = tables.pop()
    if table:
        since = min(ARMS[n]["since"] for n in names)
        df = _read_f32(DATA / "raw" / table / "train.parquet", since, A.HOLDOUT_START)
        print("扩展表 %s：%d 行，%s..%s，约 %.1f GB" % (
            table, len(df), df["date"].min(), df["date"].max(),
            df.memory_usage(deep=False).sum() / 2**30), flush=True)
    else:
        df = A.load(False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    lab = pd.read_parquet(OUT / "labels.parquet")
    df = df.merge(lab[["code", "date", "y_open", "y_open_dd15", "r20_open"]],
                  on=["code", "date"], how="left")
    if any(ARMS[n].get("extra") for n in names):
        fx = pd.read_parquet(OUT / "feats.parquet")
        df = df.merge(fx, on=["code", "date"], how="left")
        for c in F3 + F5:
            df[c] = df[c].astype("float32")
    if any(ARMS[n].get("extra2") for n in names):
        fx = pd.read_parquet(OUT / "feats2.parquet")
        df = df.merge(fx, on=["code", "date"], how="left")
        # 市值、价位：训练表自带 float_mcap / close，按当日全市场排名
        df["x_mcap_pct"] = df.groupby("date")["float_mcap"].rank(pct=True).astype("float32")
        df["x_price_pct"] = df.groupby("date")["close"].rank(pct=True).astype("float32")
        for c in G1 + G2:
            df[c] = df[c].astype("float32")
    feats_all = [c for c in df.columns if "__" in c]
    feats0 = FS.run(V.train_slice(df, V.TRAIN_END[:7]), feats_all, y="y_up")["keep"]
    print("基础特征 %d 列" % len(feats0), flush=True)
    months = sorted({d[:7] for d in df["date"] if V.TRAIN_END <= d < V.VALID_END})
    st = V.st_codes()
    for name in names:
        spec = ARMS[name]
        target = spec.get("target", "y_up")
        neg = spec.get("neg", "sample")
        seeds = list(range(M.SEED, M.SEED + spec.get("seeds", 1)))
        feats = feats0 + list(spec.get("extra", [])) + list(spec.get("extra2", []))
        model = spec.get("model", "lgb")
        t0 = time.time()
        preds, qm = [], {}
        preds_blend = []
        for m in months:
            tr = V.train_slice(df, m)
            tr = tr[np.isfinite(tr[target]) & np.isfinite(tr["y_up"])]
            if model == "rank":
                tr = tr[np.isfinite(tr["r20_open"])]
            if spec.get("since"):
                tr = tr[tr["date"] >= spec["since"]]
            te = df[df["date"].str[:7] == m].copy()
            te = te[np.isfinite(te["y_up"]) & np.isfinite(te["y_open"])]
            assert str(tr["date"].max()) < V.purge_cut(df, m), "训练样本越过净化线"
            p_te, p_tr = [], []
            p_blend = None
            if spec.get("stage2"):
                trs, _ = _sample(tr, target, "sample", seeds[0])
                p, p_blend = fit_two_stage(tr, trs, te, feats, target, seeds[0])
                q = np.quantile(p, np.linspace(0, 1, 101))     # 刻度无意义，只看名次
            else:
                for sd in seeds:
                    trs, w_cls = _sample(tr, target, neg, sd)
                    mdl = _fit_one(trs, feats, target, model, sd, w_cls)
                    p_te.append(_predict(mdl, te, feats))
                    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
                    p_tr.append(_predict(mdl, ref, feats))
                p = np.mean(p_te, axis=0)
                q = np.quantile(np.mean(p_tr, axis=0), np.linspace(0, 1, 101))
            qm[m] = [float(x) for x in q]
            out = te[["date", "code", "board", "y_up", "y_open", "y_open_dd15"]].copy()
            out["p"] = p.astype(np.float32)
            out = out[~out["code"].isin(st)]
            out["score"] = np.clip(np.searchsorted(q, out["p"]), 0, 100)
            out = out.sort_values(["date", "p"], ascending=[True, False])
            out["rank"] = out.groupby("date").cumcount() + 1
            preds.append(out[out["rank"] <= 200])
            if p_blend is not None:
                ob = te[["date", "code", "board", "y_up", "y_open", "y_open_dd15"]].copy()
                ob["p"] = p_blend.astype(np.float32)
                ob = ob[~ob["code"].isin(st)]
                ob["score"] = np.clip(np.searchsorted(q, ob["p"]), 0, 100)
                ob = ob.sort_values(["date", "p"], ascending=[True, False])
                ob["rank"] = ob.groupby("date").cumcount() + 1
                preds_blend.append(ob[ob["rank"] <= 200])
            print("  [%s] %s 完成 %.0fs" % (name, m, time.time() - t0), flush=True)
        pd.concat(preds, ignore_index=True).to_parquet(OUT / f"preds_{name}.parquet", index=False)
        if preds_blend:
            pd.concat(preds_blend, ignore_index=True).to_parquet(
                OUT / f"preds_{name}_blend.parquet", index=False)
        (OUT / f"q_{name}.json").write_text(json.dumps(qm), encoding="utf-8")
        print("[%s] 完成 %.0f 分钟" % (name, (time.time() - t0) / 60), flush=True)
    print("全部 %.0f 分钟" % ((time.time() - t_all) / 60))


# ---------------------------------------------------------------
#  汇总
# ---------------------------------------------------------------
def analyze(names: list[str] | None = None) -> dict:
    import daily as D
    import exp_time as ET
    files = sorted(OUT.glob("preds_*.parquet"))
    arms = [f.stem[6:] for f in files]
    if names:
        arms = [a for a in arms if a in names]
    order = list(ARMS)
    arms.sort(key=lambda a: order.index(a) if a in order else 99)
    tabs = {a: pd.read_parquet(OUT / f"preds_{a}.parquet") for a in arms}
    lab = pd.read_parquet(OUT / "labels.parquet")
    vm = lab[lab["date"].between("2025-03-01", "2025-12-31")]
    base = {"y_up": float(vm["y_up_chk"].mean()), "y_open": float(vm["y_open"].mean()),
            "y_open_dd15": float(vm["y_open_dd15"].mean())}
    res = {"base": base, "arms": {}}
    hdr = "%-9s | %6s %6s %6s %4s | %7s %5s %6s %6s | %7s %6s" % (
        "臂", "prod n", "开盘%", "收盘%", "空天", "前10开%", "±", "配对t", "收盘%", "前30开%", "dd15%")
    print("验证集基准：收盘 %.2f%%  次日开盘 %.2f%%  加回撤 %.2f%%\n" % (
        100 * base["y_up"], 100 * base["y_open"], 100 * base["y_open_dd15"]))
    print(hdr)
    print("-" * len(hdr))
    ref = tabs.get("base")
    for a in arms:
        t = tabs[a]
        prod = t[(t["score"] >= D.SCORE_MIN) & (t["rank"] <= D.CAP_A)]
        top10 = t[t["rank"] <= 10]
        top30 = t[t["rank"] <= 30]
        days = t["date"].nunique()
        r = {"prod_open": ET.clustered(prod, "y_open"), "prod_close": ET.clustered(prod, "y_up"),
             "prod_empty": days - prod["date"].nunique(),
             "top10_open": ET.clustered(top10, "y_open"), "top10_close": ET.clustered(top10, "y_up"),
             "top10_dd15": ET.clustered(top10, "y_open_dd15"),
             "top30_open": ET.clustered(top30, "y_open")}
        if ref is not None:
            r["vs_base_top10_open"] = ET.paired_vs(top10, ref[ref["rank"] <= 10], "y_open")
        res["arms"][a] = r
        pv = r.get("vs_base_top10_open", {}).get("t", float("nan"))
        print("%-9s | %6d %5.2f%% %5.2f%% %4d | %6.2f%% %5.2f %6.2f %5.2f%% | %6.2f%% %5.2f%%" % (
            a, r["prod_open"]["n"], 100 * r["prod_open"]["hit"], 100 * r["prod_close"]["hit"],
            r["prod_empty"], 100 * r["top10_open"]["hit"], 100 * r["top10_open"]["se"], pv,
            100 * r["top10_close"]["hit"], 100 * r["top30_open"]["hit"], 100 * r["top10_dd15"]["hit"]))
    src = ref if ref is not None else tabs[arms[0]]
    months = sorted({d[:7] for d in src["date"]})
    print("\n逐月 前10 次日开盘口径命中（%）")
    print("%-9s" % "臂" + "".join("%7s" % m[2:] for m in months))
    for a in arms:
        t = tabs[a]
        row = t[t["rank"] <= 10].groupby(t["date"].str[:7])["y_open"].mean()
        print("%-9s" % a + "".join("%7.1f" % (100 * row.get(m, np.nan)) for m in months))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--report1", action="store_true")
    ap.add_argument("--feats", action="store_true")
    ap.add_argument("--feats2", action="store_true")
    ap.add_argument("--fit", default="", help="臂名，逗号分隔（见 ARMS）")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.labels:
        build_labels()
    if a.report1:
        report1()
    if a.feats:
        build_feats()
    if a.feats2:
        build_feats2()
    if a.fit:
        fit_arms([x for x in a.fit.split(",") if x])
    if a.analyze:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
