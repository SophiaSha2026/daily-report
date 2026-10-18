"""
角度二：情绪周期（criteria 第 2 节 + 8b 先验阈值）。

    PYTHONIOENCODING=utf-8 python research/angles/regime_an.py          筛选集打印；开发集落 results_regime_dev.json 不打印
    PYTHONIOENCODING=utf-8 python research/angles/regime_an.py --dev    预登记后读开发集（judge_an.py 调）

阶段（8b）：冰点 = n_up < 20 且 n_dn > 10 且 h_max < 3；亢奋 = n_up > 80 且 blow < 0.15 且 h_max > 5；
启动 = 最近 10 个交易日内有冰点日、当天不是冰点；退潮 = 最近 5 个交易日内有亢奋日、当天不是亢奋；其余 = 其他。
优先级 冰点 > 亢奋 > 启动 > 退潮。臂（k 固定 3）：R1 冰点 + 启动 才出；R2 亢奋 + 退潮 不出。
M2 高斯 HMM（自写 EM，K=3，只在筛选集拟合，前向滤波）只作描述。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_regime.txt"
DEVJ = A.HERE / "results_regime_dev.json"
FEATS = ["ln_up", "ln_dn", "h_max", "promo1", "promo2", "blow", "y_ret", "lose", "up_share"]
PHASES = ["冰点", "启动", "亢奋", "退潮", "其他"]
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def phases(day: pd.DataFrame, relaxed: bool = False) -> pd.Series:
    if relaxed:                      # 8f：去掉跌停 / 炸板率那一条
        ice = (day["n_up"] < 20) & (day["h_max"] < 3)
        mania = (day["n_up"] > 80) & (day["h_max"] > 5)
    else:
        ice = (day["n_up"] < 20) & (day["n_dn"] > 10) & (day["h_max"] < 3)
        mania = (day["n_up"] > 80) & (day["blow"] < 0.15) & (day["h_max"] > 5)
    ice_recent = ice.shift(1).rolling(10, min_periods=1).max().fillna(0) > 0
    mania_recent = mania.shift(1).rolling(5, min_periods=1).max().fillna(0) > 0
    ph = np.where(ice, "冰点", np.where(mania, "亢奋", np.where(ice_recent, "启动", np.where(mania_recent, "退潮", "其他"))))
    return pd.Series(ph, index=day.index)


# ---------------- 自写高斯 HMM（对角协方差） ----------------
def hmm_fit(X: np.ndarray, K: int = 3, iters: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)]
    var = np.ones((K, d)) * X.var(axis=0)
    pi = np.ones(K) / K
    Am = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(Am, 0.9)
    ll_old = -np.inf
    for _ in range(iters):
        logB = -0.5 * (((X[:, None, :] - mu[None]) ** 2) / var[None] + np.log(2 * np.pi * var)[None]).sum(axis=2)
        # 前向 / 后向（带缩放）
        alpha = np.zeros((n, K))
        c = np.zeros(n)
        a = pi * np.exp(logB[0] - logB[0].max())
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, n):
            a = (alpha[t - 1] @ Am) * np.exp(logB[t] - logB[t].max())
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.zeros((n, K))
        beta[-1] = 1
        for t in range(n - 2, -1, -1):
            b = Am @ (np.exp(logB[t + 1] - logB[t + 1].max()) * beta[t + 1])
            beta[t] = b / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.zeros((K, K))
        for t in range(n - 1):
            e = np.exp(logB[t + 1] - logB[t + 1].max()) * beta[t + 1]
            m = (alpha[t][:, None] * Am) * e[None, :]
            xi += m / m.sum()
        pi = gamma[0] / gamma[0].sum()
        Am = xi / xi.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)[:, None]
        mu = (gamma.T @ X) / w
        var = (gamma.T @ (X ** 2)) / w - mu ** 2 + 1e-4
        ll = np.log(c).sum() + logB.max(axis=1).sum()
        if abs(ll - ll_old) < 1e-6:
            break
        ll_old = ll
    return {"pi": pi, "A": Am, "mu": mu, "var": var, "ll": ll}


def hmm_filter(X: np.ndarray, m: dict) -> np.ndarray:
    K = len(m["pi"])
    n = len(X)
    logB = -0.5 * (((X[:, None, :] - m["mu"][None]) ** 2) / m["var"][None] + np.log(2 * np.pi * m["var"])[None]).sum(axis=2)
    alpha = np.zeros((n, K))
    a = m["pi"] * np.exp(logB[0] - logB[0].max())
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ m["A"]) * np.exp(logB[t] - logB[t].max())
        alpha[t] = a / a.sum()
    return alpha.argmax(axis=1)


def phase_table(seg: str, ph: pd.Series, label: str, out: list):
    B = A.G.baseline(seg)
    db = A.day_base(seg)
    d = pd.DataFrame({"date": B.dates, "h1": B.hits_k(1), "h3": B.hits_k(3), "h10": B.hits_k(10)})
    d["ph"] = ph.reindex(d["date"]).to_numpy()
    d["base"] = db.reindex(d["date"]).to_numpy()
    out.append("   %s %s：%s" % (seg, label, "  ".join(
        "%s %d天(%.0f%%) P1 %.1f%% P3 %.1f%% P10 %.1f%% 基准 %.1f%%" % (
            k, len(g), 100 * len(g) / len(d), 100 * g["h1"].mean(), 100 * g["h3"].mean() / 3, 100 * g["h10"].mean() / 10, 100 * g["base"].mean())
        for k, g in d.groupby("ph", sort=False))))
    return d


def arm_rows(seg: str, good_days: set, k: int = 3) -> pd.DataFrame:
    t = A.PR.load(seg)
    t = t[(t["rank_ret20"] <= k) & t["date"].isin(good_days)]
    return t[["date", "hit"]]


def block_null(d: pd.DataFrame, good: np.ndarray, b: int = 1000, block: int = 20, seed: int = 3) -> tuple[float, float]:
    """阶段序列按 20 天整块重排，P3(G) − P3(非 G) 的零分布 95% 分位和实际值。"""
    rng = np.random.default_rng(seed)
    h3 = d["h3"].to_numpy() / 3
    n = len(h3)

    def diff(g):
        if g.sum() == 0 or (~g).sum() == 0:
            return np.nan
        return h3[g].mean() - h3[~g].mean()

    actual = diff(good)
    blocks = [good[i:i + block] for i in range(0, n, block)]
    null = []
    for _ in range(b):
        perm = rng.permutation(len(blocks))
        g = np.concatenate([blocks[i] for i in perm])[:n]
        null.append(diff(g))
    return actual, float(np.nanquantile(null, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--arms", default="R1,R2,R3,R4", help="--dev 时只打印这些臂的开发集")
    a = ap.parse_args()
    day = pd.read_parquet(A.DAY_IND).sort_values("date").reset_index(drop=True)
    day["ph"] = phases(day)
    day["ph2"] = phases(day, relaxed=True)
    day["score"] = day["n_up"] * (1 + day["h_max"]) * (1 - day["blow"].fillna(0))
    ph = pd.Series(day["ph"].to_numpy(), index=day["date"])
    ph_lag = ph.shift(1)
    ph2 = pd.Series(day["ph2"].to_numpy(), index=day["date"])
    ph2_lag = ph2.shift(1)
    say("放宽阶段（8f）占比（全期）：%s" % day["ph2"].value_counts().to_dict())
    say("角度二：情绪周期（先验阈值，criteria 8b；全市场 px_screen 日线，%d 天）" % len(day))
    say("阶段占比（全期）：%s" % day["ph"].value_counts().to_dict())
    say("情绪分 n_up x (1+h_max) x (1−blow) 按阶段中位：%s" % {k: round(float(v), 0) for k, v in day.groupby("ph")["score"].median().items()})
    say("指标中位按阶段：")
    for k, g in day.groupby("ph"):
        say("   %-3s n_up %4.0f n_dn %4.0f h_max %3.0f promo1 %.2f blow %.2f lose %.2f up_share %.2f" % (
            k, g["n_up"].median(), g["n_dn"].median(), g["h_max"].median(), g["promo1"].median(), g["blow"].median(), g["lose"].median(), g["up_share"].median()))
    say()
    # HMM 描述
    day["ln_up"] = np.log1p(day["n_up"])
    day["ln_dn"] = np.log1p(day["n_dn"])
    X = day[FEATS].fillna(0).to_numpy(float)
    scr = (day["date"] >= "2022-01-01") & (day["date"] <= "2024-05-31")
    mu, sd = X[scr].mean(axis=0), X[scr].std(axis=0) + 1e-9
    Z = (X - mu) / sd
    best = None
    for s in range(5):
        m = hmm_fit(Z[scr.to_numpy()], K=3, seed=s)
        if best is None or m["ll"] > best["ll"]:
            best = m
    st = hmm_filter(Z, best)
    order = np.argsort(best["mu"][:, 0])
    name = {int(order[0]): "HMM低", int(order[1]): "HMM中", int(order[2]): "HMM高"}
    hmm_ph = pd.Series([name[int(s)] for s in st], index=day["date"])
    say("HMM（K=3，筛选集拟合，前向滤波）状态占比：%s；状态均值 ln_up %s" % (hmm_ph.value_counts().to_dict(), np.round(best["mu"][order, 0] * sd[0] + mu[0], 2).tolist()))
    say()
    # 筛选集表
    say("== 筛选集：各阶段里名次前 1 / 3 / 10 的精确率（B10 筛选 11.3%）")
    tabs = {}
    for label, series in (("先验阶段", ph), ("先验阶段 T−1", ph_lag), ("放宽阶段", ph2), ("放宽阶段 T−1", ph2_lag), ("HMM", hmm_ph)):
        out = []
        d = phase_table("screen", series, label, out)
        say(out[0])
        tabs[label] = d
    say()
    arms = {"R1": ("冰点 + 启动 才出前 3", {"冰点", "启动"}, ph, ph_lag, "先验阶段"),
            "R2": ("亢奋 + 退潮 不出，其余出前 3", {"其他", "冰点", "启动"}, ph, ph_lag, "先验阶段"),
            "R3": ("冰点' + 启动' 才出前 3", {"冰点", "启动"}, ph2, ph2_lag, "放宽阶段"),
            "R4": ("亢奋' + 退潮' 不出，其余出前 3", {"其他", "冰点", "启动"}, ph2, ph2_lag, "放宽阶段")}
    res = {}
    for key, (desc, good, p0, p1, tabkey) in arms.items():
        res[key] = {}
        for lagged, series in (("", p0), ("（T−1 阶段）", p1)):
            good_days = set(series[series.isin(good)].index)
            es = {}
            for seg in ("screen", "dev"):
                es[seg] = A.eval_rows(arm_rows(seg, good_days), seg)
            d = tabs[tabkey if not lagged else tabkey + " T−1"]
            act, q95 = block_null(d, d["ph"].isin(good).to_numpy())
            say("   " + A.fmt("%s %s%s" % (key, desc, lagged), es["screen"], "screen"))
            say("      块重排零分布：P3(G) − P3(非G) 实际 %+.1fpp，零分布 95%% 分位 %+.1fpp -> %s" % (
                100 * act, 100 * q95, "在零分布之外" if act > q95 else "在零分布之内"))
            res[key][lagged or "T"] = {"screen": {k: v for k, v in es["screen"].items() if k not in ("u", "h", "n", "dates")},
                                       "dev": {k: v for k, v in es["dev"].items() if k not in ("u", "h", "n", "dates")},
                                       "u_screen": es["screen"]["u"].tolist(), "u_dev": es["dev"]["u"].tolist(),
                                       "null_screen": [act, q95]}
            if a.dev and key in a.arms.split(","):
                say("   " + A.fmt("%s %s%s" % (key, desc, lagged), es["dev"], "dev"))
                j = A.combined_judge(es["screen"], es["dev"])
                say("      合并 ΔP %+.1fpp NW t %+.2f 块自举 p %.3f 去%s t %.2f -> %s" % (
                    100 * j["dP"], j["t"], j["boot_p"], j["best_month"], j["t_drop_best"], A.usable(es["screen"], es["dev"], j)))
    if a.dev:
        say()
        say("== 开发集：各阶段精确率（读一次）")
        for label, series in (("先验阶段", ph), ("先验阶段 T−1", ph_lag), ("放宽阶段", ph2), ("放宽阶段 T−1", ph2_lag), ("HMM", hmm_ph)):
            out = []
            phase_table("dev", series, label, out)
            say(out[0])
        A.ledger_add({"kind": "arm", "name": "regime_R1_R2", "note": "开发集一次读（R1 / R2 + 阶段表 + HMM 描述）"})
        OUT.write_text(OUT.read_text(encoding="utf-8") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    else:
        DEVJ.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
        A.ledger_add({"kind": "screen", "name": "regime_R1_R2", "note": "筛选集阶段表 + R1 / R2；开发集落盘未读"})
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.progress("regime_an%s -> results_regime.txt" % (" --dev" if a.dev else ""))


if __name__ == "__main__":
    main()
