"""离线诊断（不训模型）：ret20 前 50 内每个非价格特征的条件命中（两段，按天配对、NW 聚类 t），
以及在筛选集上对前 10 的边际（剔除 / 前置 / 倾斜，换入第 11~13 名的净效果），开发集的逐日序列算出来落盘不打印。

    PYTHONIOENCODING=utf-8 python research/altdata/diag.py            -> results_diag.txt、results_screen.txt、hits10.parquet / hits30.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
import common_pl as P  # noqa: E402

SCREEN_END = "2024-06-01"
MDE_TABLE = [(0.90, 0.51), (0.80, 0.76), (0.70, 1.00), (0.60, 1.29), (0.50, 1.52), (0.40, 1.71), (0.20, 2.22), (0.0, 2.68)]


def mde_by_overlap(ov: float) -> float:
    """results_power.txt 的合成臂表：按重合度插值 MDE80(c=2.5)，单位 pp。"""
    xs = [a for a, _ in MDE_TABLE][::-1]
    ys = [b for _, b in MDE_TABLE][::-1]
    return float(np.interp(ov, xs, ys))


def seg(d: pd.Series) -> tuple[pd.Series, pd.Series]:
    return d[d.index < SCREEN_END], d[d.index >= SCREEN_END]


def tstat(d: pd.Series) -> float:
    return P.nw_t(d) if len(d) > 25 else np.nan


def fmt_seg(d: pd.Series) -> str:
    a, b = seg(d)
    return "筛选 %+.2fpp (t %.2f, %d 天)  开发 %+.2f (t %.2f, %d 天)  合并 %+.2f (t %.2f)" % (
        100 * a.mean(), tstat(a), len(a), 100 * b.mean(), tstat(b), len(b), 100 * d.mean(), tstat(d))


# ---------------------------------------------------------------
#  条件命中：前 50 内 高组 对 低组，按天配对
# ---------------------------------------------------------------
def cond_hit(df: pd.DataFrame, feat: str, mode: str, thr: float | None = None) -> dict:
    x = df[df["rank_ret20"] <= 50][["date", "y_open", feat]].copy()
    v = x[feat].astype(float)
    if mode == "tertile":
        x = x[np.isfinite(v)]
        v = x[feat].astype(float)
        r = v.groupby(x["date"]).rank(pct=True, method="average")
        hi, lo = r > 2 / 3, r <= 1 / 3
    elif mode == "flag":          # 高组 = 特征 > thr；低组 = 其余有值的
        fin = np.isfinite(v)
        hi, lo = fin & (v > thr), fin & ~(v > thr)
    elif mode == "flag_lt":       # 高组 = 特征 < thr
        fin = np.isfinite(v)
        hi, lo = fin & (v < thr), fin & ~(v < thr)
    elif mode == "missing":       # 高组 = 有值；低组 = 没值（两融标的 / 非标的）
        hi, lo = np.isfinite(v), ~np.isfinite(v)
    else:
        raise ValueError(mode)
    x["hi"], x["lo"] = hi.to_numpy(), lo.to_numpy()
    gh = x[x["hi"]].groupby("date")["y_open"].agg(["mean", "size"])
    gl = x[x["lo"]].groupby("date")["y_open"].agg(["mean", "size"])
    both = gh.join(gl, lsuffix="_h", rsuffix="_l", how="inner")
    d = (both["mean_h"] - both["mean_l"]).dropna()
    out = {"feat": feat, "mode": mode, "thr": thr, "days": int(len(d)),
           "n_hi_day": float(both["size_h"].mean()) if len(both) else np.nan,
           "n_lo_day": float(both["size_l"].mean()) if len(both) else np.nan,
           "hit_hi": float(x.loc[x["hi"], "y_open"].mean()), "hit_lo": float(x.loc[x["lo"], "y_open"].mean()),
           "share_hi": float(x["hi"].mean())}
    a, b = seg(d)
    out.update({"d_screen": float(a.mean()) if len(a) else np.nan, "t_screen": tstat(a), "n_screen": int(len(a)),
                "d_dev": float(b.mean()) if len(b) else np.nan, "t_dev": tstat(b), "n_dev": int(len(b)),
                "d_all": float(d.mean()) if len(d) else np.nan, "t_all": tstat(d)})
    out["same_sign"] = bool(np.sign(out["d_screen"]) == np.sign(out["d_dev"])) if len(a) and len(b) else False
    return out


def fmt_cond(c: dict) -> str:
    thr = "" if c["thr"] is None else " %g" % c["thr"]
    return ("%-18s %-8s%-7s 高组占 %4.1f%% (%.1f 只/天) 命中 高 %5.2f%% 低 %5.2f%% | 筛选 %+5.2fpp (t %5.2f, %3d 天) 开发 %+5.2f (t %5.2f, %3d 天) 合并 %+5.2f (t %5.2f) %s" % (
        c["feat"], c["mode"], thr, 100 * c["share_hi"], c["n_hi_day"], 100 * c["hit_hi"], 100 * c["hit_lo"],
        100 * c["d_screen"], c["t_screen"], c["n_screen"], 100 * c["d_dev"], c["t_dev"], c["n_dev"],
        100 * c["d_all"], c["t_all"], "同号" if c["same_sign"] else "异号"))


# ---------------------------------------------------------------
#  边际：在 ret20 排序上加一条规则，前 10 命中对基线的按天差
# ---------------------------------------------------------------
def rank_rule(df: pd.DataFrame, score: np.ndarray, elig: np.ndarray | None = None, keep: int = 30) -> pd.DataFrame:
    return P.rank_by(df, score, keep=keep, elig=elig)


def top_hits(t: pd.DataFrame, k: int) -> pd.Series:
    return P.hit_series(t, k)


def overlap(t: pd.DataFrame, base: pd.DataFrame, k: int = 10) -> float:
    a = t[t["rank"] <= k].groupby("date")["code"].apply(set)
    b = base[base["rank"] <= k].groupby("date")["code"].apply(set)
    j = a.to_frame("a").join(b.to_frame("b"), how="inner")
    return float(np.mean([len(x & y) / max(len(y), 1) for x, y in zip(j["a"], j["b"])]))


def main():
    t0 = time.time()
    df = pd.read_parquet(HERE / "top60.parquet")
    feats = [c for c in df.columns if c not in ("code", "date", "board", "lret20", "y_open", "y_up", "amt20", "amt1", "mcap", "turn1", "ret20", "rank_ret20")]
    P.say("top60 %d 行，特征 %s" % (len(df), feats))
    # 基线对账：按 lret20 的前 10 命中 = polish 的 base_ret20
    base = rank_rule(df, df["lret20"].to_numpy(), keep=30)
    b10 = top_hits(base, 10)
    ref = pd.read_parquet(P.PL / "rules_hits10.parquet")["base_ret20"]
    j = b10.to_frame("me").join(ref.to_frame("ref"), how="inner")
    P.say("基线对账：%d 天，最大差 %.2e，均值 %.4f / %.4f" % (len(j), (j["me"] - j["ref"]).abs().max(), j["me"].mean(), j["ref"].mean()))
    assert (j["me"] - j["ref"]).abs().max() < 1e-9, "基线对不上 polish 的 base_ret20"
    b30 = top_hits(base, 30)

    # ---- 条件命中 ----
    L = ["== 条件命中：ret20 前 50 内，高组 对 低组，按天配对差（高 - 低），NW t 滞后 20 ==",
         "（tertile = 当天前 50 内按特征三分位；flag thr = 特征 > thr 对其余；missing = 有值对没值）", ""]
    specs = []
    for f in feats:
        v = df.loc[df["rank_ret20"] <= 50, f].astype(float)
        if v.dtype == bool or f in ("rz_in",):
            specs.append((f, "flag", 0.5))
            continue
        nz = float((v.fillna(0) != 0).mean())
        if f.startswith(("lhb_n", "lhb_today", "dz_n", "sv_n", "sv_orgs")):
            specs.append((f, "flag", 0.0))
            if f in ("lhb_n20", "dz_n20", "sv_n90"):
                specs.append((f, "flag", 1.0))
        elif f.endswith("_share"):
            specs.append((f, "flag", 0.0))
            specs.append((f, "flag_lt", 0.0))
            if nz > 0.3:
                specs.append((f, "tertile", None))
        elif f.startswith("gd_chg"):
            specs.append((f, "tertile", None))
            specs.append((f, "flag", 0.0))
            specs.append((f, "flag_lt", -0.10))
        elif f == "gd_stale":
            specs.append((f, "tertile", None))
            specs.append((f, "flag_lt", 10))
        elif f.startswith("rz_"):
            specs.append((f, "tertile", None))
            if f == "rz_chg5":
                specs.append((f, "flag", 0.10))
                specs.append((f, "flag_lt", 0.0))
        else:
            specs.append((f, "tertile", None))
    if "rz_chg5" in feats:
        specs.append(("rz_chg5", "missing", None))
    conds = []
    for f, mode, thr in specs:
        c = cond_hit(df, f, mode, thr)
        conds.append(c)
        L.append(fmt_cond(c))
    L.append("")
    L.append("两段同号且两段 |t| 都 >= 1 的：")
    for c in conds:
        if c["same_sign"] and abs(c["t_screen"]) >= 1 and abs(c["t_dev"]) >= 1:
            L.append("  " + fmt_cond(c))
    txt = "\n".join(L)
    (HERE / "results_diag.txt").write_text(txt + "\n", encoding="utf-8")
    P.say(txt)
    json.dump(conds, (HERE / "diag_conds.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    # ---- 边际（筛选集打印；开发集序列落盘不看）----
    def fl(s):
        return s.astype(float).fillna(0).to_numpy()

    R: dict[str, tuple[np.ndarray, np.ndarray | None, int]] = {}   # 名: (score, elig, K)
    lr = df["lret20"].to_numpy()
    rk = df["rank_ret20"].to_numpy()

    def add_excl(name, cond):
        R[name] = (lr, ~np.asarray(cond, bool), 0)

    def add_prefer(name, cond, K):
        c = np.asarray(cond, bool) & (rk <= K)
        R[name] = (np.where(c, 1e3, 0) + lr, None, K)

    def add_tilt(name, val, w, K, sign=1):
        v = pd.Series(np.where(rk <= K, sign * fl(val), np.nan), index=df.index)
        pr = v.groupby(df["date"]).rank(pct=True).fillna(0).to_numpy()
        R[name] = (np.where(rk <= K, -rk + w * K * pr, -1e6), None, K)

    have = set(feats)
    if "lhb_today" in have:
        for K in (15, 20, 30):
            add_prefer("pf_lhb_today_%d" % K, df["lhb_today"] > 0, K)
            add_prefer("pf_lhb_n5_%d" % K, df["lhb_n5"] > 0, K)
            add_prefer("pf_lhb_net5pos_%d" % K, df["lhb_net5_share"] > 0, K)
        add_excl("ex_lhb_today", df["lhb_today"] > 0)
        add_excl("ex_lhb_n5", df["lhb_n5"] > 0)
        add_excl("ex_lhb_n20_3", df["lhb_n20"] >= 3)
        add_excl("ex_lhb_net5neg", df["lhb_net5_share"] < 0)
        add_excl("ex_lhb_net20neg", df["lhb_net20_share"] < 0)
        for K in (20, 30):
            add_tilt("tl_lhb_net20_%d" % K, df["lhb_net20_share"], 0.5, K)
    if "org_net5_share" in have:
        add_excl("ex_org_net5neg", df["org_net5_share"] < 0)
        add_excl("ex_org_net20neg", df["org_net20_share"] < 0)
        for K in (15, 20, 30):
            add_prefer("pf_org_net5pos_%d" % K, df["org_net5_share"] > 0, K)
            add_prefer("pf_org_net20pos_%d" % K, df["org_net20_share"] > 0, K)
    if "rz_chg5" in have:
        add_excl("ex_rz_in", df["rz_in"].astype(bool))
        add_excl("ex_rz_out", ~df["rz_in"].astype(bool))
        add_excl("ex_rz_chg5neg", df["rz_chg5"] < 0)
        add_excl("ex_rz_chg5_gt10", df["rz_chg5"] > 0.10)
        add_excl("ex_rz_chg20neg", df["rz_chg20"] < 0)
        for K in (20, 30):
            add_prefer("pf_rz_chg5_gt10_%d" % K, df["rz_chg5"] > 0.10, K)
            add_prefer("pf_rz_chg5_gt5_%d" % K, df["rz_chg5"] > 0.05, K)
            add_prefer("pf_rz_out_%d" % K, ~df["rz_in"].astype(bool), K)
            add_tilt("tl_rz_chg5_%d" % K, df["rz_chg5"], 0.5, K)
            add_tilt("tl_rz_chg20_%d" % K, df["rz_chg20"], 0.5, K)
            add_tilt("tl_rz_buy5_%d" % K, df["rz_buy5_share"], 0.5, K)
            add_tilt("tl_rz_level_neg_%d" % K, df["rz_level"], 0.5, K, sign=-1)
    if "dz_n20" in have:
        add_excl("ex_dz_n20", df["dz_n20"] > 0)
        add_excl("ex_dz_disc20", df["dz_disc20_share"] > 0)
        add_excl("ex_dz_amt20_1pct", df["dz_amt20_share"] > 0.01)
        for K in (20, 30):
            add_prefer("pf_dz_n20_%d" % K, df["dz_n20"] > 0, K)
    if "gd_chg1" in have:
        add_excl("ex_gd_up", df["gd_chg1"] > 0)
        add_excl("ex_gd_up10", df["gd_chg1"] > 0.10)
        add_excl("ex_gd_fresh_up", (df["gd_chg1"] > 0) & (df["gd_stale"] <= 20))
        for K in (20, 30):
            add_prefer("pf_gd_down_%d" % K, df["gd_chg1"] < 0, K)
            add_prefer("pf_gd_down10_%d" % K, df["gd_chg1"] < -0.10, K)
            add_tilt("tl_gd_chg1_neg_%d" % K, df["gd_chg1"], 0.5, K, sign=-1)
            add_tilt("tl_gd_chg3_neg_%d" % K, df["gd_chg3"], 0.5, K, sign=-1)
    if "sv_n30" in have:
        add_excl("ex_sv_n30", df["sv_n30"] > 0)
        for K in (20, 30):
            add_prefer("pf_sv_n30_%d" % K, df["sv_n30"] > 0, K)
            add_prefer("pf_sv_n90_%d" % K, df["sv_n90"] > 0, K)
            add_tilt("tl_sv_orgs90_%d" % K, df["sv_orgs90"], 0.5, K)
    if "pl_ratio" in have:
        add_excl("ex_pl_ratio30", df["pl_ratio"] > 30)
        for K in (20, 30):
            add_tilt("tl_pl_ratio_neg_%d" % K, df["pl_ratio"], 0.5, K, sign=-1)

    H10, H30 = {"base_ret20": b10}, {"base_ret20": b30}
    rows = []
    for name, (score, elig, K) in R.items():
        t = rank_rule(df, score, elig=elig, keep=30)
        h10, h30 = top_hits(t, 10), top_hits(t, 30)
        H10[name], H30[name] = h10, h30
        d = (h10 - b10).dropna()
        a, _ = seg(d)
        ov = overlap(t[t["date"] < SCREEN_END], base[base["date"] < SCREEN_END])
        se_s = P.nw_se(a.to_numpy())
        se969 = se_s * np.sqrt(len(a) / 969)
        rows.append({"arm": name, "screen_diff": float(a.mean()), "screen_t": tstat(a), "screen_days": int(len(a)),
                     "overlap": ov, "nw_se_969": se969, "mde80_self": 3.34 * se969, "mde80_table": mde_by_overlap(ov) / 100,
                     "screen_hit": float(h10[h10.index < SCREEN_END].mean())})
    pd.DataFrame(H10).to_parquet(HERE / "hits10.parquet")
    pd.DataFrame(H30).to_parquet(HERE / "hits30.parquet")
    res = pd.DataFrame(rows).sort_values("screen_diff", ascending=False)
    L = ["== 边际（筛选集 2022-01..2024-05，前 10 对 ret20 基线按天差；开发集序列落盘未看）==",
         "（ex_ = 剔除后顺延；pf_K = 前 K 名里带标志的先上；tl_K = 前 K 名里 名次 + 0.5K x 特征百分位；重合 = 和基线前 10 的平均重合比例）",
         "MDE80_自 = 3.34 x 自身 NW se（折到 969 天）；MDE80_表 = results_power 按重合度插值", ""]
    for r in res.itertuples():
        flag = "候选" if (r.screen_diff >= max(r.mde80_self, r.mde80_table)) else ""
        L.append("%-24s 筛选 %+.2fpp (t %5.2f) 命中 %.2f%% 重合 %.0f%%  se969 %.2f  MDE80 自 %.2f 表 %.2f  %s" % (
            r.arm, 100 * r.screen_diff, r.screen_t, 100 * r.screen_hit, 100 * r.overlap, 100 * r.nw_se_969,
            100 * r.mde80_self, 100 * r.mde80_table, flag))
    txt = "\n".join(L)
    (HERE / "results_screen.txt").write_text(txt + "\n", encoding="utf-8")
    res.to_json(HERE / "screen_arms.json", orient="records", force_ascii=False, indent=1)
    P.say(txt)
    P.say("%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
