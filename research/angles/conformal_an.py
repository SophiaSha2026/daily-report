"""
角度一 1b（静态对照，8a 降级后只跑 V0 / V2）：conformal p 值 + 按天 BH(α = 1 − τ)，校准集 = 之前 12 个月样本外行。

    PYTHONIOENCODING=utf-8 python research/angles/conformal_an.py [--dev] [--shuffle]   -> results_conformal.txt

分数：S1 = −rank_ret20，S2 = lret20，S3 = gate/scores/lgb_price_top300.npy，S4 = lgb_alt_top300.npy（都是只用之前数据的样本外分数）。
V0 原版；V2 日内名次变换（s -> −当天候选池内名次 / 当天池大小）。另报「按月 BH」和「加支持数 >= 100 行且 >= 20 天」版本。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_conformal.txt"
DEVJ = A.HERE / "results_conformal_dev.json"
TAUS = (0.15, 0.20, 0.25, 0.30)
SCORES = {"S1_rank": None, "S2_lret20": None, "S3_lgb_price": "lgb_price_top300", "S4_lgb_alt": "lgb_alt_top300"}
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def bh_mask(p: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p)
    o = np.argsort(p, kind="mergesort")
    ps = p[o]
    k = np.arange(1, m + 1)
    ok = np.nonzero(ps <= alpha * k / m)[0]
    out = np.zeros(m, bool)
    if len(ok):
        out[o[:ok[-1] + 1]] = True
    return out


def day_rank_transform(s: np.ndarray, dates: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"d": dates, "s": s})
    fin = np.isfinite(s)
    r = df[fin].groupby("d")["s"].rank(ascending=False, method="average")
    n = df[fin].groupby("d")["s"].transform("size")
    out = np.full(len(s), np.nan)
    out[fin] = -(r / n).to_numpy()
    return out


def conformal_select(s: np.ndarray, t: pd.DataFrame, months: list[str], alphas: dict, by_month: bool = False,
                     support: bool = False) -> dict:
    """返回 {tau: 布尔选择}。校准 = 之前 12 个月、end20 < 月初、有分数有标签的行；p_j = (1 + #负样本 s_i >= s_j)/(n0 + 1)。"""
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    y = t["y_open"].to_numpy(float)
    ym = dates.astype("U7")
    sel = {tau: np.zeros(len(t), bool) for tau in alphas}
    for m in months:
        lo = A.G.month_shift(m, -12) + "-01"
        cm = (dates >= lo) & (dates < m + "-01") & (end20 < m + "-01") & np.isfinite(s) & np.isfinite(y)
        rows = np.nonzero((ym == m) & np.isfinite(s))[0]
        if cm.sum() < 1000 or len(rows) == 0 or len(np.unique(ym[cm])) < 3:
            continue
        neg = np.sort(s[cm & (y == 0)])
        n0 = len(neg)
        st = s[rows]
        cnt_ge = n0 - np.searchsorted(neg, st, side="left")
        p = (1.0 + cnt_ge) / (n0 + 1.0)
        if support:
            # 累计精确率 >= tau 且支持数 >= 100 行、>= 20 天（gate cum 口径），作为 BH 之外的并列
            sc = s[cm]
            yc = y[cm]
            dc = dates[cm]
            o = np.argsort(-sc, kind="mergesort")
            sc, yc, dc = sc[o], yc[o], dc[o]
            prec = np.cumsum(yc) / np.arange(1, len(sc) + 1)
            _, fi = np.unique(dc, return_index=True)
            seen = np.zeros(len(dc), np.int64)
            seen[fi] = 1
            nd = np.cumsum(seen)
            prec[(np.arange(1, len(sc) + 1) < 100) | (nd < 20)] = 0.0
            srt = sc[::-1]
            prec_r = prec[::-1]
            pos = np.clip(np.searchsorted(srt, st, side="left"), 0, len(srt) - 1)
            phat = prec_r[pos]
            for tau in alphas:
                sel[tau][rows] = phat >= tau
            continue
        dm = dates[rows]
        groups = [np.arange(len(rows))] if by_month else [np.nonzero(dm == d)[0] for d in np.unique(dm)]
        for tau, alpha in alphas.items():
            for g in groups:
                mk = bh_mask(p[g], alpha)
                sel[tau][rows[g[mk]]] = True
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--shuffle", action="store_true", help="1d 自检：标签按天内置换")
    a = ap.parse_args()
    t = A.G.load_table(["code", "date", "end20", "y_open", "hit", "seg", "rank_ret20", "lret20", "r20_open"])
    if a.shuffle:
        rng = np.random.default_rng(32)
        y = t["y_open"].to_numpy().copy()
        fin = np.isfinite(y)
        yy = pd.Series(y[fin])
        y[fin] = yy.groupby(t["date"].to_numpy()[fin]).transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1 << 30))).to_numpy()).to_numpy()
        t["y_open"] = y
        t["hit"] = np.nan_to_num(y)
        say("== 自检 1d：标签按天内随机置换")
    dates = t["date"].to_numpy().astype("U10")
    months = A.G.months_between("2022-01", "2025-12")
    alphas = {tau: 1 - tau for tau in TAUS}
    say("角度一 1b：conformal p 值 + BH（静态对照），校准 = 之前 12 个月样本外行；两段 B10 筛选 11.3% / 开发 16.5%")
    res = {}
    picks = {}
    for sname, npy in SCORES.items():
        if npy is None:
            s = (-t["rank_ret20"].to_numpy(float)) if sname.startswith("S1") else t["lret20"].to_numpy(float)
        else:
            s = np.load(A.G.SCORES / (npy + ".npy")).astype(float)
        variants = {"V0 按天BH": dict(by_month=False), "V2 日内名次+按天BH": dict(by_month=False, transform=True),
                    "V0 按月BH": dict(by_month=True), "支持数版(cum12 n100 d20)": dict(support=True)}
        if a.shuffle and not sname.startswith(("S1", "S3")):
            continue
        for vname, kw in variants.items():
            if a.shuffle and vname != "V0 按天BH":
                continue
            ss = day_rank_transform(s, dates) if kw.get("transform") else s
            sel = conformal_select(ss, t, months, alphas, by_month=kw.get("by_month", False), support=kw.get("support", False))
            for tau in TAUS:
                rows = t.loc[sel[tau], ["date", "hit", "r20_open"]]
                key = "%s %s τ%.2f" % (sname, vname, tau)
                es = {seg: A.eval_rows(rows, seg, boot=not a.shuffle) for seg in ("screen", "dev")}
                res[key] = {seg: {k: v for k, v in es[seg].items() if k not in ("u", "h", "n", "dates")} for seg in es}
                res[key]["u_screen"] = es["screen"]["u"].tolist()
                res[key]["u_dev"] = es["dev"]["u"].tolist()
                say("   " + A.fmt(key, es["screen"], "screen"))
                if a.shuffle:
                    say("   " + A.fmt(key, es["dev"], "dev"))
                if a.dev and key in a.arms.split("|"):
                    say("   " + A.fmt(key, es["dev"], "dev"))
                    j = A.combined_judge(es["screen"], es["dev"])
                    say("      合并 ΔP %+.1fpp NW t %+.2f 块自举 p %.3f -> %s" % (100 * j["dP"], j["t"], j["boot_p"],
                                                                          A.usable(es["screen"], es["dev"], j, tau=tau)))
                e = es["screen"]
                if e["V"] >= 150 and np.isfinite(e["P"]):
                    picks.setdefault(tau, []).append((abs(e["P"] - tau), key, e["P"], e["V"], e["dP"]))
    if not a.shuffle and not a.dev:
        say()
        say("== 1c 配置选择（筛选集：V >= 150 且 |P − τ| 最小）")
        for tau in TAUS:
            if tau in picks:
                best = sorted(picks[tau])[0]
                say("   τ %.2f -> %s  P %.1f%% V %d ΔP %+.1fpp" % (tau, best[1], 100 * best[2], best[3], 100 * best[4]))
            else:
                say("   τ %.2f -> 没有 V >= 150 的配置" % tau)
        DEVJ.write_text(json.dumps({"res": res, "picks": {str(k): sorted(v)[0] for k, v in picks.items()}}, ensure_ascii=False, default=float),
                        encoding="utf-8")
        A.ledger_add({"kind": "screen", "name": "conformal_static", "n_configs": len(res), "note": "筛选集 4 分数 x 4 变体 x 4 τ；开发集落盘未读"})
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif a.shuffle:
        A.ledger_add({"kind": "shuffle", "name": "conformal_static", "note": "标签按天内置换，S1 / S3 V0"})
        (A.HERE / "results_conformal_shuffle.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        A.ledger_add({"kind": "arm", "name": "conformal_static", "arms": a.arms.split("|"), "note": "开发集一次读"})
        OUT.write_text(OUT.read_text(encoding="utf-8") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    A.progress("conformal_an%s%s -> results_conformal.txt" % (" --dev" if a.dev else "", " --shuffle" if a.shuffle else ""))


if __name__ == "__main__":
    main()
