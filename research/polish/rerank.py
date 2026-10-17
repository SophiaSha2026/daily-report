"""
在 ret20 前 K 名上重排：逐月走向前，训练集 = 之前 3 年里每天 ret20 前 K 的候选池行（净化：date < purge_cut 且
end20 < 测试月），标签 y_open，模型 LightGBM 或逻辑回归；测试月每天在前 K 里按预测值取前 10。

    PYTHONIOENCODING=utf-8 python research/polish/rerank.py --name rr_lgb_k50 --model lgb --k 50 [--feats ext|ext_prod] [--seed 7]
    PYTHONIOENCODING=utf-8 python research/polish/rerank.py --grid        # 筛选集调参网格（开发集算了不印）

产物 data/breakout/raw/exp/polish/rr/<name>.parquet（date code rank_ret20 p），命中序列并入 rr_hits10.parquet。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common_pl as P

RR = P.PL / "rr"
RR.mkdir(parents=True, exist_ok=True)
EXT = ["lret20", "ret1", "ret5", "ret10", "ret40", "ret60", "ret20_lag5", "vol20", "vol60", "amp20", "turn1", "turn5", "turn20",
       "amt20", "amt_ratio", "vr5_20", "mcap", "price_u", "gap1", "cr1", "dist_hi20", "dist_hi60", "dist_hi120", "pos20",
       "nh20_5", "up20", "max1_20", "lu_close", "yizi_today", "lu_streak", "nlu20", "nlu5", "nld20", "nbar", "lim_pct"]
MONTHS = {"screen": P.SEGS["screen"], "dev": P.SEGS["dev"]}


def months_all():
    import evalkit as K
    return K.months_in(("2022-01", "2025-12"))


def month_shift(m: str, k: int) -> str:
    y, mm = int(m[:4]), int(m[5:7])
    t = y * 12 + (mm - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def load_topk(k: int) -> pd.DataFrame:
    """候选池里每天 ret20 前 k 的行（全部日期，2020-12 起），带扩展特征和标签。"""
    f = RR / f"topk_{k}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = P.load_pool()
    df = df[df["pool"]].reset_index(drop=True)
    df["rank_ret20"] = df.groupby("date")["lret20"].rank(ascending=False, method="first")
    df = df[df["rank_ret20"] <= k].reset_index(drop=True)
    for c in ("mcap", "amt20", "amt1"):
        df[c] = np.log1p(df[c].astype(float)).astype(np.float32)
    df["lrank"] = np.log(df["rank_ret20"]).astype(np.float32)
    # 当天前 K 内的相对量（和当天市场状态解耦）
    for c in ("lret20", "turn5", "vol20", "ret5"):
        df["rk_" + c] = df.groupby("date")[c].rank(pct=True).astype(np.float32)
    for b in ("chinext", "star", "bj"):
        df["b_" + b] = (df["board"] == b).astype(np.float32)
    df.to_parquet(f, index=False)
    return df


def add_prod(df: pd.DataFrame) -> pd.DataFrame:
    """并上生产特征表（prod_screen，102 列）的行：按 row group 分块读、只留 (date, code) 在 df 里的行。"""
    import pyarrow.parquet as pq
    f = RR / "prod_top.parquet"
    key = df[["date", "code"]]
    if f.exists():
        pr = pd.read_parquet(f)
    else:
        pf = pq.ParquetFile(P.EXP / "screen" / "prod_screen.parquet")
        parts = []
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i).to_pandas()
            t["code"] = t["code"].astype(str).str.zfill(6)
            parts.append(t.merge(key, on=["date", "code"], how="inner"))
            P.say("  prod rg %d/%d -> %d 行" % (i + 1, pf.num_row_groups, sum(len(p) for p in parts)))
        pr = pd.concat(parts, ignore_index=True).drop(columns=["board"], errors="ignore")
        pr.to_parquet(f, index=False)
    return df.merge(pr, on=["date", "code"], how="left")


def feat_list(df: pd.DataFrame, kind: str) -> list[str]:
    base = EXT + ["lrank", "rk_lret20", "rk_turn5", "rk_vol20", "rk_ret5", "b_chinext", "b_star", "b_bj"]
    if kind == "ext":
        return base
    if kind == "ext_prod":
        return base + [c for c in df.columns if "__" in c]
    if kind == "rk":
        return ["rkk_" + c for c in EXT if c not in ("lim_pct",)] + ["lrank", "b_chinext", "b_star", "b_bj"]
    if kind == "rk_min":
        return ["rkk_" + c for c in ("lret20", "nld20", "mcap", "turn5", "vol20", "ret5", "ret40", "dist_hi20", "nbar")] + ["lrank", "b_chinext", "b_star", "b_bj"]
    if kind == "min":
        return ["lret20", "lrank", "nld20", "mcap", "turn5", "vol20", "ret5", "ret40", "dist_hi20", "nbar", "b_chinext", "b_star", "b_bj"]
    raise ValueError(kind)


def fit_predict(model: str, Xtr, ytr, wtr, Xte, seed: int, params: dict | None = None, gtr=None):
    if model == "lgb_rank":
        import lightgbm as lgb
        p = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100,
             "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
             "lambdarank_truncation_level": 20, "verbose": -1, "seed": seed, "num_threads": 8}
        if params:
            p.update({k: v for k, v in params.items() if k != "rounds"})
        rounds = (params or {}).get("rounds", 200)
        ds = lgb.Dataset(Xtr, label=ytr.astype(int), group=gtr)
        b = lgb.train(p, ds, num_boost_round=rounds)
        return b.predict(Xte)
    if model == "lgb":
        import lightgbm as lgb
        p = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100,
             "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
             "verbose": -1, "seed": seed, "num_threads": 8}
        if params:
            p.update({k: v for k, v in params.items() if k != "rounds"})
        rounds = (params or {}).get("rounds", 200)
        ds = lgb.Dataset(Xtr, label=ytr, weight=wtr)
        b = lgb.train(p, ds, num_boost_round=rounds)
        return b.predict(Xte)
    if model == "lr":
        from sklearn.linear_model import LogisticRegression
        mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0) + 1e-9
        Z = np.nan_to_num((Xtr - mu) / sd)
        Zt = np.nan_to_num((Xte - mu) / sd)
        Z, Zt = np.clip(Z, -5, 5), np.clip(Zt, -5, 5)
        m = LogisticRegression(C=(params or {}).get("C", 0.1), max_iter=500)
        m.fit(Z, ytr, sample_weight=wtr)
        return m.predict_proba(Zt)[:, 1]
    raise ValueError(model)


def run(name: str, model: str, k: int, feats: str = "ext", seed: int = 7, params: dict | None = None,
        train_years: int = 3, months: list[str] | None = None) -> pd.Series:
    import validate as V
    import model as M
    t0 = time.time()
    df = load_topk(k)
    if feats == "ext_prod":
        df = add_prod(df)
    if feats.startswith("rk"):
        for c in EXT:
            if c != "lim_pct" and "rkk_" + c not in df.columns:
                df["rkk_" + c] = df.groupby("date")[c].rank(pct=True).astype(np.float32)
    cols = feat_list(df, feats)
    df = df.sort_values(["date", "rank_ret20"], kind="mergesort").reset_index(drop=True)
    X_all = df[cols].to_numpy(np.float32)
    months = months or months_all()
    parts = []
    for m in months:
        cut = V.purge_cut(df, m)
        lo = month_shift(m, -12 * train_years) + "-01"
        tr = ((df["date"] >= lo) & (df["date"] < cut) & (df["end20"] < m + "-01") & np.isfinite(df["y_open"])).to_numpy()
        te = (df["date"].str[:7] == m).to_numpy()
        if tr.sum() < 2000 or te.sum() == 0:
            P.say("  %s 训练 %d 行，跳过" % (m, tr.sum()))
            continue
        w = M.month_weights(df[tr])
        gtr = df.loc[tr].groupby("date", sort=False).size().to_numpy() if model == "lgb_rank" else None
        p = fit_predict(model, X_all[tr], df.loc[tr, "y_open"].to_numpy(float), np.asarray(w, float), X_all[te], seed, params, gtr)
        parts.append(pd.DataFrame({"date": df.loc[te, "date"].to_numpy(), "code": df.loc[te, "code"].to_numpy(),
                                   "rank_ret20": df.loc[te, "rank_ret20"].to_numpy(), "p": p.astype(np.float32)}))
        if len(parts) % 6 == 0:
            P.say("  %s 训练 %d 行（正 %.2f%%）%.0fs" % (m, tr.sum(), 100 * df.loc[tr, "y_open"].mean(), time.time() - t0))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(RR / f"{name}.parquet", index=False)
    lab = df[["date", "code", "board", "y_open", "y_up"]]
    o = out.merge(lab, on=["date", "code"], how="left")
    t = P.rank_by(o, o["p"].to_numpy(), keep=30)
    h = P.hit_series(t, 10)
    h30 = P.hit_series(t, 30)
    _append_hits(name, h, h30)
    P.say("%s 跑完 %.0fs：%d 天" % (name, time.time() - t0, len(h)))
    return h


def _append_hits(name: str, h: pd.Series, h30: pd.Series) -> None:
    for fn, s in (("rr_hits10.parquet", h), ("rr_hits30.parquet", h30)):
        f = P.PL / fn
        H = pd.read_parquet(f) if f.exists() else pd.DataFrame(index=s.index)
        H = H.reindex(H.index.union(s.index))
        H[name] = s
        H.to_parquet(f)


def report_screen(names: list[str]) -> str:
    H = pd.read_parquet(P.PL / "rules_hits10.parquet")
    R = pd.read_parquet(P.PL / "rr_hits10.parquet")
    R30 = pd.read_parquet(P.PL / "rr_hits30.parquet")
    H30 = pd.read_parquet(P.PL / "rules_hits30.parquet")
    scr = R.index < "2024-06-01"
    b = H["base_ret20"]
    b30 = H30["base_ret20"]
    L = ["筛选集（开发集已算未看）：重排臂对 base_ret20", "%-26s 命中%%   差pp   NW t  去最好月t  前30差pp" % "臂"]
    for n in names:
        if n not in R.columns:
            continue
        d = (R.loc[scr, n] - b).dropna()
        d30 = (R30.loc[scr, n] - b30).dropna()
        dm = d.groupby(d.index.str[:7]).mean()
        best = dm.idxmax() if d.mean() >= 0 else dm.idxmin()
        L.append("%-26s %5.2f  %+6.2f  %+5.2f  %+5.2f    %+6.2f" % (
            n, 100 * R.loc[scr, n].mean(), 100 * d.mean(), P.nw_t(d), P.nw_t(d[d.index.str[:7] != best]), 100 * d30.mean()))
    return "\n".join(L)


GRID = [
    ("rr_lgb_k50", "lgb", 50, "ext", {}),
    ("rr_lgb_k100", "lgb", 100, "ext", {}),
    ("rr_lgb_k200", "lgb", 200, "ext", {}),
    ("rr_lgb_k30", "lgb", 30, "ext", {}),
    ("rr_lr_k50", "lr", 50, "ext", {}),
    ("rr_lr_k100", "lr", 100, "ext", {}),
    ("rr_lr_k200", "lr", 200, "ext", {}),
    ("rr_lgb_k100_min", "lgb", 100, "min", {}),
    ("rr_lr_k100_min", "lr", 100, "min", {}),
    ("rr_lgb_k100_prod", "lgb", 100, "ext_prod", {}),
    ("rr_lgb_k100_slow", "lgb", 100, "ext", {"learning_rate": 0.02, "rounds": 400, "num_leaves": 7}),
    # 第二轮（筛选集上）：目标改成当天内排序 / 特征改成当天内秩
    ("rr_rank_k50", "lgb_rank", 50, "ext", {}),
    ("rr_rank_k100", "lgb_rank", 100, "ext", {}),
    ("rr_rank_k100_rk", "lgb_rank", 100, "rk", {}),
    ("rr_lgb_k100_rk", "lgb", 100, "rk", {}),
    ("rr_lr_k100_rk", "lr", 100, "rk", {}),
    ("rr_lr_k50_rk", "lr", 50, "rk", {}),
    ("rr_lr_k100_rkmin", "lr", 100, "rk_min", {}),
    ("rr_lr_k50_rkmin", "lr", 50, "rk_min", {}),
    ("rr_lr_k50_min", "lr", 50, "min", {}),
    ("rr_lr_k100_min_C1", "lr", 100, "min", {"C": 1.0}),
    ("rr_lr_k100_min_C.01", "lr", 100, "min", {"C": 0.01}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--model", default="lgb")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--feats", default="ext")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.grid:
        names = []
        for name, model, k, feats, params in GRID:
            P.wait_prod_window()
            if not (RR / f"{name}.parquet").exists():
                run(name, model, k, feats, a.seed, params)
            names.append(name)
        txt = report_screen(names)
        P.say(txt)
        (P.HERE / "results_rerank_screen.txt").write_text(txt + "\n", encoding="utf-8")
    elif a.report:
        txt = report_screen([g[0] for g in GRID])
        P.say(txt)
    else:
        run(a.name, a.model, a.k, a.feats, a.seed)
        P.say(report_screen([a.name]))


if __name__ == "__main__":
    main()
