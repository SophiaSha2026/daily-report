"""
多重检验：试了多少个臂、门槛该抬到多少、回测过拟合概率 PBO。

    1 试验次数：arms.REGISTRY（有逐日产物、能重算的）+ 只有日志 / summary 的（滚动校准、门控信号、
      滚动刻度、实验 20 的诊断切片）
    2 Bonferroni / Holm / BH：对重算的 NW20 p 和块自举 p 各做一遍
    3 族内最大 t 的零分布（块自举，按天整行重抽，保留臂之间的相关和时间上的自相关）：
      这是「按这个族试了这么多次，偶然冒出来的最大 t 有多大」，95% 分位就是 deflated 门槛
    4 Bailey & López de Prado 的期望最大值近似 E[max] ≈ (1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))，
      N 用臂间相关矩阵特征值算的有效试验数
    5 CSCV / PBO（Bailey, Borwein, López de Prado, Zhu 2017）：T×N 的每天前 10 命中矩阵切成 S 块，
      所有 C(S, S/2) 种对半分：样本内最好的那个臂在样本外的相对名次 ω，λ = ln(ω/(1-ω))，PBO = P(λ ≤ 0)

    python research/stats/multi.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms as AR   # noqa: E402
import hac as H     # noqa: E402

OUT = HERE / "out"
N01 = NormalDist()
GAMMA = 0.5772156649
NL = chr(10)

# 没有逐日产物、但确实在这个开发集（或它的子段）上比过的东西
EXTRA_TRIALS = [
    (15, "滚动分数线校准（base / eqm 两个）", 2),
    (15, "环境门控信号（7 个，三档 + 诚实规则）", 7),
    (17, "roll 滚动刻度", 1),
    (20, "诊断切片：名次 8 档 / 分数 7 档 / 板块 4 / 在动 3 / 连续 4 / 市值 4（HIGH_CONF 从这里挑出来）", 30),
]


# ---------------------------------------------------------------
def holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for i, j in enumerate(o):
        rank = m - i
        run = min(run, p[j] * m / rank)
        adj[j] = run
    return adj


def z_two(alpha: float) -> float:
    return N01.inv_cdf(1 - alpha / 2)


def expected_max(n: float) -> float:
    if n <= 1:
        return 0.0
    return (1 - GAMMA) * N01.inv_cdf(1 - 1 / n) + GAMMA * N01.inv_cdf(1 - 1 / (n * math.e))


def n_eff_eig(corr: np.ndarray) -> float:
    w = np.linalg.eigvalsh(np.nan_to_num(corr))
    w = np.clip(w, 0, None)
    return float(w.sum() ** 2 / (w ** 2).sum())


# ---------------------------------------------------------------
def diff_matrix(fam: set[str], span_override: str | None = None, exp: int | None = None,
                reg_y: bool = False) -> pd.DataFrame:
    """reg_y=True 用登记时的标签（实验 15 是收盘口径 y_up），否则 A/B 族统一买得到口径。"""
    cols = {}
    for r in AR.REGISTRY:
        if r["fam"] not in fam or r["span"] == "since25" or (exp is not None and r["exp"] != exp):
            continue
        span = span_override or r["span"]
        a = AR.span_filter(AR.top_of(r, r["arm"]), span)
        b = AR.span_filter(AR.top_of(r, r["ctrl"]), span)
        y = r["y"] if (reg_y or r["fam"] not in ("A", "B")) else "y_open"
        k = r["k"]
        cols[f"{r['exp']}:{r['arm']}"] = AR.daily_hit(a, k, y) - AR.daily_hit(b, k, y)
    return pd.DataFrame(cols).sort_index()


def maxt_null(D: pd.DataFrame, b: int = 5000, block: int = H.BLOCK, seed: int = H.SEED) -> dict:
    """族内最大 t 的零分布：各列去均值后按天整行块重抽，t* 用原序列的 NW20 se 标准化。

    缺值（生产规则口径某些天没名额）按 0 填在去均值之后，等于那天不贡献。"""
    X = D.to_numpy(float)
    mask = np.isfinite(X)
    mu = np.nanmean(X, axis=0)
    se = np.array([H.nw_t(D[c].dropna().to_numpy(), 20)["se"] for c in D.columns])
    Xc = np.where(mask, X - mu, 0.0)
    T = len(X)
    rng = np.random.default_rng(seed)
    idx = H._mbb_index(T, block, b, rng)
    cnt = mask.sum(axis=0)
    mx1, mx2 = np.empty(b), np.empty(b)
    for i in range(b):
        m = Xc[idx[i]].sum(axis=0) / np.maximum(mask[idx[i]].sum(axis=0), 1)
        t = m / se
        mx1[i] = np.nanmax(t)
        mx2[i] = np.nanmax(np.abs(t))
    t_obs = mu / se
    corr = pd.DataFrame(Xc, columns=D.columns).corr().to_numpy()
    return {"m": D.shape[1], "T": T, "q95_one": float(np.quantile(mx1, 0.95)), "q95_abs": float(np.quantile(mx2, 0.95)),
            "q90_one": float(np.quantile(mx1, 0.90)),
            "n_eff": n_eff_eig(corr), "mean_abs_corr": float(np.nanmean(np.abs(corr[np.triu_indices_from(corr, 1)]))),
            "t_obs": {c: float(v) for c, v in zip(D.columns, t_obs)},
            "p_fwer": {c: float((mx1 >= v).mean()) for c, v in zip(D.columns, t_obs)},
            "cnt": {c: int(v) for c, v in zip(D.columns, cnt)}}


# ---------------------------------------------------------------
def cscv_pbo(M: pd.DataFrame, S: int, metric: str = "mean") -> dict:
    """M：行 = 天，列 = 臂（每天前 10 命中均值）。按时间切 S 个连续块。"""
    M = M.dropna()
    X = M.to_numpy(float)
    T, N = X.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(S)]

    def perf(rows):
        x = X[rows]
        if metric == "mean":
            return x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        return x.mean(axis=0) / np.where(sd > 0, sd, np.nan)

    lam, is_best, oos_best, oos_rank_best = [], [], [], []
    names = list(M.columns)
    for comb in itertools.combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in comb])
        te = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        pi, po = perf(tr), perf(te)
        n_star = int(np.nanargmax(pi))
        # 样本外相对名次（1..N，越大越好），ω = rank / (N + 1)
        r = float((po < po[n_star]).sum() + 0.5 * ((po == po[n_star]).sum() - 1) + 1)
        w = r / (N + 1)
        lam.append(math.log(w / (1 - w)))
        is_best.append(pi[n_star])
        oos_best.append(po[n_star])
        oos_rank_best.append(names[n_star])
    lam = np.array(lam)
    is_best, oos_best = np.array(is_best), np.array(oos_best)
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if len(is_best) > 2 else float("nan")
    picks = pd.Series(oos_rank_best).value_counts(normalize=True).head(5).round(3).to_dict()
    return {"T": T, "N": N, "S": S, "combos": len(lam), "pbo": float((lam <= 0).mean()),
            "lambda_median": float(np.median(lam)), "degradation_slope": slope,
            "oos_mean_of_is_best": float(np.mean(oos_best)), "is_mean_of_is_best": float(np.mean(is_best)),
            "p_oos_best_below_median": float((lam <= 0).mean()), "is_best_picks": picks}


# ---------------------------------------------------------------
def main() -> int:
    OUT.mkdir(exist_ok=True)
    arms = pd.read_csv(OUT / "arms.csv")
    rep = arms[arms["tag"] == "reported"].copy()
    res = {}

    # 1 试验次数
    n_reg = len(rep)
    n_extra = sum(x[2] for x in EXTRA_TRIALS)
    fam_counts = rep.groupby("fam").size().to_dict()
    res["trials"] = {"registry_rows": n_reg, "by_family": fam_counts, "extra": EXTRA_TRIALS,
                     "extra_total": n_extra, "total_incl_diag": n_reg + n_extra,
                     "total_model_arms": int((rep["fam"].isin(["A", "B", "P"])).sum())}

    # 2 Bonferroni / Holm / BH
    m_list = sorted({res["trials"]["total_model_arms"], 40, n_reg, n_reg + n_extra})
    res["thresholds"] = {str(m): {"bonf_two_sided_z": z_two(0.05 / m), "bonf_one_sided_z": N01.inv_cdf(1 - 0.05 / m),
                                  "expected_max_iid": expected_max(m)} for m in m_list}
    fam_ab = rep[rep["fam"].isin(["A", "B", "P"])].copy()
    for col in ("p_nw20", "mbb_p"):
        p = fam_ab[col].fillna(1).to_numpy()
        fam_ab[f"{col}_holm"] = holm(p)
        fam_ab[f"{col}_bh"] = bh(p)
        fam_ab[f"{col}_bonf"] = np.minimum(p * len(p), 1)
    fam_ab[["exp", "fam", "arm", "k", "span", "t_rep", "nw20_t", "p_nw20", "p_nw20_holm", "p_nw20_bh",
            "mbb_p", "mbb_p_holm", "mbb_p_bh", "mbb_p_bonf"]].to_csv(OUT / "multitest.csv", index=False,
                                                                       encoding="utf-8-sig")
    res["multitest_survivors"] = {
        "holm_nw20_0.05": fam_ab.loc[fam_ab["p_nw20_holm"] < 0.05, "arm"].tolist(),
        "bh_nw20_0.10": fam_ab.loc[fam_ab["p_nw20_bh"] < 0.10, "arm"].tolist(),
        "holm_mbb_0.05": fam_ab.loc[fam_ab["mbb_p_holm"] < 0.05, "arm"].tolist(),
        "bh_mbb_0.10": fam_ab.loc[fam_ab["mbb_p_bh"] < 0.10, "arm"].tolist(),
    }
    print("试验：", res["trials"])
    print("门槛：", json.dumps(res["thresholds"], indent=0))
    print("幸存：", res["multitest_survivors"])

    # 3 族内最大 t 零分布
    res["maxt"] = {}
    for name, fam, span, exp, reg_y in (("A15_yup", {"A"}, None, 15, True), ("A15_yopen", {"A"}, None, 15, False),
                                        ("A16", {"A"}, None, 16, False), ("A_val207", {"A"}, None, None, False),
                                        ("B_dev387", {"B"}, None, None, False), ("BP_dev387", {"B", "P"}, None, None, False),
                                        ("AB_val207", {"A", "B"}, "val", None, False)):
        D = diff_matrix(fam, span, exp, reg_y)
        r = maxt_null(D)
        r["expected_max_neff"] = expected_max(r["n_eff"])
        r["bonf_one_sided_neff"] = N01.inv_cdf(1 - 0.05 / max(r["n_eff"], 1))
        res["maxt"][name] = r
        print("%-10s 臂 %2d 天 %3d  有效试验 %.1f  平均|相关| %.2f  最大 t 零分布 95%% 分位：单侧 %.2f / 绝对 %.2f  E[max|N_eff] %.2f" % (
            name, r["m"], r["T"], r["n_eff"], r["mean_abs_corr"], r["q95_one"], r["q95_abs"], r["expected_max_neff"]))
        top = sorted(r["t_obs"].items(), key=lambda kv: -kv[1])[:4]
        print("           最大的几个 NW20 t 和族内 FWER p：", [(k, round(v, 2), round(r["p_fwer"][k], 3)) for k, v in top])

    # 5 PBO
    hm = pd.read_parquet(OUT / "daily_hits.parquet")
    res["pbo"] = {}
    a_cols = [c for c in hm.columns if c.startswith(("15:", "16:"))]
    w_cols = [c for c in hm.columns if c.startswith("W:")]
    val = hm[(hm.index >= AR.VAL[0]) & (hm.index <= AR.VAL[1])]
    c15 = [c for c in a_cols if c.startswith("15:")]
    c16 = [c for c in a_cols if c.startswith("16:")]
    for name, M, S in (("E15_val207_S10", val[c15], 10), ("E16_val207_S10", val[c16], 10),
                       ("A_val207_S10", val[a_cols], 10), ("A_val207_S14", val[a_cols], 14),
                       ("B_dev387_S16", hm[w_cols].dropna(), 16), ("B_dev387_S10", hm[w_cols].dropna(), 10),
                       ("B_since25_S12", hm.loc[hm.index >= AR.SINCE25, w_cols], 12),
                       ("AB_val207_S10", val[a_cols + w_cols], 10), ("AB_val207_S14", val[a_cols + w_cols], 14)):
        for metric in ("mean", "sharpe"):
            r = cscv_pbo(M, S, metric)
            res["pbo"][f"{name}_{metric}"] = r
            print("PBO %-16s %-6s T=%d N=%d 组合 %d  PBO=%.3f  λ中位 %.2f  衰减斜率 %.2f  样本内最好 %.2f%% -> 样本外 %.2f%%  常被选中 %s" % (
                name, metric, r["T"], r["N"], r["combos"], r["pbo"], r["lambda_median"], r["degradation_slope"],
                100 * r["is_mean_of_is_best"] if metric == "mean" else r["is_mean_of_is_best"],
                100 * r["oos_mean_of_is_best"] if metric == "mean" else r["oos_mean_of_is_best"], r["is_best_picks"]))
    # 试验登记簿的种子：把历史上这个开发集 / 这个目标上试过的东西补登，新制度从这里接着数
    seed = []
    for r in AR.REGISTRY:
        fam = "breakout_y50_top10" if r["fam"] in ("A", "B") else (
            "breakout_y50_prodrule" if r["fam"] == "P" else "cont_y15")
        seed.append({"family": fam, "exp": r["exp"], "arm": r["arm"], "ctrl": r["ctrl"], "span": r["span"],
                     "label": r["y"], "t_iid_reported": r["t_rep"], "source": "backfill 2026-10-14"})
    for e, what, n in EXTRA_TRIALS:
        seed.append({"family": "breakout_y50_top10" if e != 20 else "breakout_y50_diag", "exp": e, "arm": what,
                     "count": n, "source": "backfill 2026-10-14（无逐日产物）"})
    (OUT / "ledger_seed.jsonl").write_text(NL.join(json.dumps(x, ensure_ascii=False) for x in seed) + NL,
                                           encoding="utf-8")
    (OUT / "multi.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
