"""
强势延续回看（计划 2.3，第 3 周）。回答「已经在动的票接下来还动不动」在日线上能预测到什么程度。

候选池（t 日收盘后已知）
    当天收盘封涨停，或近 5 根 K 线内封过涨停的票；在训练表里（上市满 120 个交易日、特征齐）；剔 ST
标签
    次日开盘买入（次日一字板记不可买），之后 10 根 K 线最高价相对买入价 ≥ +15%（主）/ +20%（副）
    也记 r10_open（连续值）和 次日开盘到次日收盘的涨跌（当天就能跑掉的）
协议
    evalkit 的开发集 2024-06..2025-12 逐月走向前；训练只用候选池里的行；特征每月在候选池切片上
    对新标签重选（fselect.run，缓存键带 tag）；模型按月等权 LightGBM（和生产同一个类）；
    净化线沿用 20 根（比 10 根的标签窗口更保守）
尺子
    候选池自己的基础率；每天前 3 / 5 / 10 只的命中（按天聚类）；月度分布；
    对照一条手写规则：按「连板高度、量比」排序取前 N（不训模型能到多少）
过线（预登记）
    前 5 只命中 ≥ 候选池基础率 × 2，且 2025 起每个月都不低于基础率，且比手写规则高 t ≥ 2.5

    python src/breakout/exp_cont.py --labels     候选池 + 标签 -> raw/exp/cont/pool.parquet
    python src/breakout/exp_cont.py --fit
    python src/breakout/exp_cont.py --analyze
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import datasource as ds  # noqa: E402
import evalkit as K      # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
NAME = "cont_v1"
HYP = ("强势延续：候选池 = 当天涨停或近 5 根封过涨停；目标 = 次日开盘买得到、之后 10 根最高价 ≥ +15%。"
       "模型前 5 只命中 ≥ 池基础率 × 2，2025 起每月不低于基础率，且比手写规则（连板高度、量比排序）高 t ≥ 2.5")
ARMS = {"model": "候选池上训 LightGBM（按月等权）", "rule": "手写规则：连板高度降序、量比降序", "pool": "候选池基础率"}
D_OUT = K.EXP / NAME
WINDOW = 10
TH_MAIN, TH_ALT = 0.15, 0.20


# ---------------------------------------------------------------
#  候选池 + 标签
# ---------------------------------------------------------------
def _per_code(g: pd.DataFrame) -> pd.DataFrame:
    n = len(g)
    code = str(g["code"].iloc[0])
    close = g["close"].to_numpy(float)
    high = g["high"].to_numpy(float)
    low = g["low"].to_numpy(float)
    opn = g["open"].to_numpy(float)
    vol = g["volume"].to_numpy(float)
    pct = ds.limit_pct(code, "")
    prev = np.r_[np.nan, close[:-1]]
    lim = ds.limit_price_arr(pd.Series(prev), pct, pd.Series([code] * n)).to_numpy(float)
    lu = (close >= lim - 1e-9) & np.isfinite(lim)
    streak = np.zeros(n)
    for i in range(n):
        streak[i] = streak[i - 1] + 1 if (lu[i] and i > 0) else float(lu[i])
    cs = np.cumsum(lu).astype(float)
    lu5 = cs - np.r_[np.zeros(5), cs[:-5]] if n > 5 else cs
    v1 = np.r_[np.nan, vol[:-1]]
    vr = vol / np.where(v1 > 0, v1, np.nan)
    fut = np.full(n, np.nan)
    if n > WINDOW:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        fut[:n - WINDOW] = swv(high[1:], WINDOW).max(axis=1)
    open1 = np.append(opn[1:], np.nan)
    low1 = np.append(low[1:], np.nan)
    close1 = np.append(close[1:], np.nan)
    lim1 = ds.limit_price_arr(pd.Series(close), pct, pd.Series([code] * n)).to_numpy(float)
    yizi1 = (low1 >= lim1 - 1e-9) & np.isfinite(low1)
    ok = np.isfinite(fut) & np.isfinite(open1) & (open1 > 0)
    r10 = np.where(ok, fut / open1 - 1.0, np.nan)
    return pd.DataFrame({
        "code": g["code"].to_numpy(), "date": g["date"].to_numpy(),
        "lu": lu.astype(float), "streak": streak, "lu5": lu5, "vol_ratio": vr,
        "yizi1": yizi1.astype(float), "r10_open": r10,
        "d1_oc": np.where(np.isfinite(open1) & (open1 > 0), close1 / open1 - 1.0, np.nan),
        "y15": np.where(ok, ((r10 >= TH_MAIN) & ~yizi1).astype(float), np.nan),
        "y20": np.where(ok, ((r10 >= TH_ALT) & ~yizi1).astype(float), np.nan),
    })


def build_pool() -> pd.DataFrame:
    t0 = time.time()
    px = pd.read_parquet(K.DATA / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume"])
    px["date"] = px["date"].astype(str)
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    d = pd.concat([_per_code(g) for _, g in px.groupby("code", sort=False)], ignore_index=True)
    pool = d[(d["lu"] > 0) | (d["lu5"] > 0)].copy()
    D_OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(D_OUT / "pool.parquet", index=False)
    v = pool[np.isfinite(pool["y15"])]
    print("候选池 %d 行（每天 %.0f 只），y15 基础率 %.2f%%，y20 %.2f%%，次日一字 %.1f%%，%.0fs -> %s" % (
        len(pool), len(pool) / pool["date"].nunique(), 100 * v["y15"].mean(), 100 * v["y20"].mean(),
        100 * v["yizi1"].mean(), time.time() - t0, D_OUT / "pool.parquet"))
    return pool


# ---------------------------------------------------------------
#  走向前
# ---------------------------------------------------------------
def load_pool_frame() -> pd.DataFrame:
    import arena as A
    import validate as V
    pool = pd.read_parquet(D_OUT / "pool.parquet")
    df = A.load(with_holdout=False)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(pool, on=["code", "date"], how="inner")
    st = V.st_codes()
    df = df[~df["code"].isin(st)].reset_index(drop=True)
    return df


def feats_for(df: pd.DataFrame, month: str) -> list[str]:
    import fselect as FS
    import validate as V
    f = K.EXP / "feats_by_month.json"
    cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    key = f"{month}|cont_y15"
    if key in cache:
        return list(cache[key])
    feats_all = [c for c in df.columns if "__" in c]
    tr = V.train_slice(df, month)
    tr = tr[np.isfinite(tr["y15"])]
    rep = FS.run(tr, feats_all, y="y15")
    cache[key] = list(rep["keep"])
    f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return list(rep["keep"])


def fit_all() -> None:
    import model as M
    import validate as V
    df = load_pool_frame()
    print("候选池并特征表：%d 行，%s..%s" % (len(df), df["date"].min(), df["date"].max()), flush=True)
    months = K.months_in(K.DEV)
    tops = []
    t0 = time.time()
    for m in months:
        tr = V.train_slice(df, m)
        tr = tr[np.isfinite(tr["y15"])]
        te = df[df["date"].str[:7] == m].copy()
        te = te[np.isfinite(te["y15"])]
        if len(tr) < 2000 or not len(te):
            print("  %s 训练 %d 行，跳过" % (m, len(tr)), flush=True)
            continue
        assert str(tr["date"].max()) < V.purge_cut(df, m)
        feats = feats_for(df, m)
        # 候选池里正样本率 10% 上下，不再下采样；按月等权照用
        mdl = M.L1Lgbm(n_estimators=300, n_jobs=12, min_child_samples=50).fit(tr, feats, "y15")
        te["p"] = mdl.predict_proba(te).astype(np.float32)
        # 手写规则：连板高度降序、量比降序
        te["rule"] = te["streak"] * 1000 + te["vol_ratio"].fillna(0).clip(0, 999)
        keep = ["date", "code", "board", "y15", "y20", "r10_open", "d1_oc", "streak", "lu5",
                "vol_ratio", "p", "rule"]
        tops.append(te[keep])
        print("  %s 完成 %.0fs（训练 %d 行，测试 %d 行，特征 %d 列）" % (
            m, time.time() - t0, len(tr), len(te), len(feats)), flush=True)
    out = pd.concat(tops, ignore_index=True)
    out.to_parquet(D_OUT / "preds.parquet", index=False)
    print("-> %s" % (D_OUT / "preds.parquet"))


def ranked(t: pd.DataFrame, col: str) -> pd.DataFrame:
    t = t.sort_values(["date", col], ascending=[True, False]).copy()
    t["rank"] = t.groupby("date").cumcount() + 1
    return t


def analyze() -> dict:
    t = pd.read_parquet(D_OUT / "preds.parquet")
    res = {}
    for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
        s = t[t["date"] >= since] if since else t
        base15, base20 = float(s["y15"].mean()), float(s["y20"].mean())
        per_day = len(s) / s["date"].nunique()
        print("[%s] 候选池 %d 行、每天 %.0f 只，基础率 y15 %.2f%% y20 %.2f%%，次日开盘到收盘均值 %+.2f%%" % (
            lbl, len(s), per_day, 100 * base15, 100 * base20, 100 * s["d1_oc"].mean()))
        rm, rr = ranked(s, "p"), ranked(s, "rule")
        for k in (3, 5, 10):
            cm15, cr15 = K.clustered(rm[rm["rank"] <= k], "y15"), K.clustered(rr[rr["rank"] <= k], "y15")
            cm20 = K.clustered(rm[rm["rank"] <= k], "y20")
            cmp_ = K.compare(rm, rr, y="y15", k=k)
            mo = K.monthly(rm[rm["rank"] <= k], "y15")
            mob = s.groupby(s["date"].str[:7])["y15"].mean()
            below = [m_ for m_ in mo.index if mo[m_] < mob.get(m_, 0)]
            res[f"{lbl}|top{k}"] = {"model15": cm15, "model20": cm20, "rule15": cr15, "cmp": cmp_,
                                     "months_below_base": below, "base15": base15}
            print("  前 %2d 只：模型 y15 %5.2f%% ±%.2f（%.1f 倍）y20 %5.2f%% | 规则 y15 %5.2f%% | "
                  "模型对规则 按天 %+.2f t=%.2f 赢 %d/%d | 月中位 %.1f%% 最差 %.1f%% 低于基础率的月 %s" % (
                      k, 100 * cm15["hit"], 100 * cm15["se"], cm15["hit"] / base15, 100 * cm20["hit"],
                      100 * cr15["hit"], 100 * cmp_["diff_day"], cmp_["t_day"], cmp_["wins"],
                      cmp_["months"], 100 * mo.median(), 100 * mo.min(), below))
        # 分档：连板高度
        g = s.groupby(s["streak"].clip(0, 4))["y15"].agg(["mean", "size"])
        print("  按当天连板高度：" + "  ".join("%d板 %.1f%%(%d)" % (k_, 100 * v["mean"], v["size"])
                                           for k_, v in g.iterrows()))
        print()
    mo = sorted({d[:7] for d in t["date"]})
    rm = ranked(t, "p")
    print("逐月 前 5 只 y15（%）/ 候选池基础率")
    print("%-6s" % "" + "".join("%6s" % m_[2:] for m_ in mo))
    r5 = K.monthly(rm[rm["rank"] <= 5], "y15")
    rb = t.groupby(t["date"].str[:7])["y15"].mean()
    print("%-6s" % "模型" + "".join("%6.1f" % (100 * r5.get(m_, np.nan)) for m_ in mo))
    print("%-6s" % "池" + "".join("%6.1f" % (100 * rb.get(m_, np.nan)) for m_ in mo))
    K.record(NAME, res)
    (D_OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    if a.labels:
        build_pool()
    if a.fit:
        fit_all()
    if a.analyze:
        analyze()
    return 0


if __name__ == "__main__":
    sys.exit(main())
