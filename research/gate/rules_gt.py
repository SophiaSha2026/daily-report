"""规则挖掘门槛（criteria.txt 第 2 节 F4）：<= 3 个条件的合取，逐月在拟合窗里按精确率搜，在复核窗里确认，才对测试月生效。

    PYTHONIOENCODING=utf-8 python research/gate/rules_gt.py [--nfit 100] [--ncal 30] [--maxk 3]

每个测试月 m：
    拟合窗 = m 之前 36..13 个月（净化后）；复核窗 = m 之前 12 个月且 end20 < 月初（和模型法的校准窗同一段）。
    候选条件（离散化的原值特征，见 CONDS）两两 / 三三合取；每条规则 fit_p（拟合窗精确率，支持数 >= nfit）、
    cal_p（复核窗精确率，支持数 >= ncal）。行的 p̂ = 覆盖该行的规则里 min(fit_p, cal_p) 的最大值。
    p̂ >= τ 就出。输出 scores/rules.npy（直接是 p̂，eval_gt --cal none）和每月启用的规则清单 rules_log.json。
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

import common_gt as G

CONDS = [
    ("board=bj", "board_i", "==", 3), ("board=chinext", "board_i", "==", 1), ("board=star", "board_i", "==", 2), ("board=main", "board_i", "==", 0),
    ("ret20>=0.5", "ret20", ">=", 0.5), ("ret20>=0.8", "ret20", ">=", 0.8), ("ret20>=1.0", "ret20", ">=", 1.0), ("ret20>=1.5", "ret20", ">=", 1.5),
    ("rank<=1", "rank_ret20", "<=", 1), ("rank<=2", "rank_ret20", "<=", 2), ("rank<=3", "rank_ret20", "<=", 3), ("rank<=5", "rank_ret20", "<=", 5),
    ("rank<=10", "rank_ret20", "<=", 10), ("rank<=20", "rank_ret20", "<=", 20), ("rank<=50", "rank_ret20", "<=", 50),
    ("vol60>=0.05", "vol60", ">=", 0.05), ("vol60>=0.07", "vol60", ">=", 0.07),
    ("turn1>=10", "turn1", ">=", 10), ("turn1>=20", "turn1", ">=", 20), ("turn1>=30", "turn1", ">=", 30),
    ("turn20>=5", "turn20", ">=", 5), ("turn20>=10", "turn20", ">=", 10), ("turn20>=15", "turn20", ">=", 15),
    ("mcap<20亿", "mcap", "<", 20e8), ("mcap<50亿", "mcap", "<", 50e8), ("mcap>=50亿", "mcap", ">=", 50e8),
    ("nlu20>=3", "nlu20", ">=", 3), ("nlu20>=5", "nlu20", ">=", 5), ("nlu20==0", "nlu20", "==", 0),
    ("lu_close", "lu_close", "==", 1), ("nld20==0", "nld20", "==", 0), ("nld20>=1", "nld20", ">=", 1),
    ("dist_hi60>=-0.05", "dist_hi60", ">=", -0.05), ("dist_hi60<-0.15", "dist_hi60", "<", -0.15),
    ("ret5>=0.2", "ret5", ">=", 0.2), ("ret5<=0", "ret5", "<=", 0.0), ("ret1>=0.09", "ret1", ">=", 0.09),
    ("amt_ratio>=2", "amt_ratio", ">=", 2.0), ("vr5_20>=1.5", "vr5_20", ">=", 1.5),
    ("price<10", "price_u", "<", 10), ("nbar<250", "nbar", "<", 250),
    ("mkt_base_t21<0.01", "mkt_base_t21", "<", 0.01), ("mkt_base_t21>=0.03", "mkt_base_t21", ">=", 0.03),
    ("mkt_ret20<0", "mkt_ret20", "<", 0.0), ("lu_share20>=0.02", "lu_share20", ">=", 0.02),
    ("lhb_today", "lhb_today", ">=", 1), ("lhb_n5>=1", "lhb_n5", ">=", 1), ("rz_chg20>=0.2", "rz_chg20", ">=", 0.2),
    ("rz_in", "rz_in", "==", 1), ("dz_n20==0", "dz_n20", "==", 0), ("gd_chg1<0", "gd_chg1", "<", 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfit", type=int, default=100)
    ap.add_argument("--ncal", type=int, default=30)
    ap.add_argument("--maxk", type=int, default=3)
    ap.add_argument("--name", default="rules")
    ap.add_argument("--noalt", action="store_true")
    a = ap.parse_args()
    G.PL.wait_prod_window()
    t0 = time.time()
    conds = [c for c in CONDS if not (a.noalt and c[1] in ("lhb_today", "lhb_n5", "rz_chg20", "rz_in", "dz_n20", "gd_chg1"))]
    cols = sorted({c[1] for c in conds})
    t = G.load_table(["date", "end20", "y_open"] + cols)
    n = len(t)
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    ym = dates.astype("U7")
    y = t["y_open"].to_numpy(float)
    M = np.zeros((len(conds), n), bool)
    for i, (nm, col, op, v) in enumerate(conds):
        x = t[col].to_numpy(float)
        M[i] = {"==": x == v, ">=": x >= v, "<=": x <= v, "<": x < v}[op]
    del t
    dates_u = np.unique(dates)
    months = G.months_between(*G.TEST_MONTHS)
    phat = np.full(n, np.nan, np.float32)
    combos = []
    for k in range(1, a.maxk + 1):
        combos += list(itertools.combinations(range(len(conds)), k))
    G.say("条件 %d 个，规则 %d 条，%.0fs" % (len(conds), len(combos), time.time() - t0))
    log = {}
    yb = np.nan_to_num(y, nan=0.0)
    ok = np.isfinite(y)
    for m in months:
        t1 = time.time()
        first = int(np.searchsorted(dates_u, m + "-01"))
        cut = str(dates_u[max(first - 20, 0)])
        lo = G.month_shift(m, -36) + "-01"
        cal_lo = G.month_shift(m, -12) + "-01"
        fit = (dates >= lo) & (dates < cal_lo) & ok
        cal = (dates >= cal_lo) & (dates < cut) & (end20 < m + "-01") & ok
        test = ym == m
        if test.sum() == 0 or fit.sum() < 10000 or cal.sum() < 5000:
            continue
        fi, ci, ti = np.flatnonzero(fit), np.flatnonzero(cal), np.flatnonzero(test)
        Mf, Mc, Mt = M[:, fi], M[:, ci], M[:, ti]
        yf, yc = yb[fi], yb[ci]
        pt = np.zeros(len(ti), np.float32)
        used = []
        # 束搜索：单条件 -> 二合取 -> 三合取；只从拟合窗支持数 >= nfit 且精确率 >= 0.08 的节点往下扩
        def stats(mask_f):
            nf = int(mask_f.sum())
            return nf, (float(yf[mask_f].sum() / nf) if nf else 0.0)

        def confirm(combo, rf, nf, pf):
            rc = Mc[combo[0]]
            for j in combo[1:]:
                rc = rc & Mc[j]
            nc = int(rc.sum())
            if nc < a.ncal:
                return
            pc = float(yc[rc].sum() / nc)
            q = min(pf, pc)
            if q < 0.15:
                return
            rt = Mt[combo[0]]
            for j in combo[1:]:
                rt = rt & Mt[j]
            if rt.any():
                pt[rt] = np.maximum(pt[rt], q)
                used.append({"rule": " & ".join(conds[j][0] for j in combo), "n_fit": nf, "p_fit": round(pf, 3),
                             "n_cal": nc, "p_cal": round(pc, 3), "n_test": int(rt.sum())})

        beam = []
        for i in range(len(conds)):
            nf, pf = stats(Mf[i])
            if nf < a.nfit:
                continue
            if pf >= 0.15:
                confirm((i,), Mf[i], nf, pf)
            if pf >= 0.08:
                beam.append(((i,), Mf[i]))
        for depth in range(2, a.maxk + 1):
            nxt = []
            for combo, rf in beam:
                for j in range(combo[-1] + 1, len(conds)):
                    r = rf & Mf[j]
                    nf, pf = stats(r)
                    if nf < a.nfit:
                        continue
                    if pf >= 0.15:
                        confirm(combo + (j,), r, nf, pf)
                    if pf >= 0.08 and depth < a.maxk:
                        nxt.append((combo + (j,), r))
            beam = nxt
        phat[ti] = pt
        used.sort(key=lambda r: -min(r["p_fit"], r["p_cal"]))
        log[m] = used[:40]
        top = used[0] if used else None
        G.say("%s 拟合 %d 复核 %d 测试 %d：启用规则 %d 条，最高 %s  %.0fs" % (
            m, len(fi), len(ci), len(ti), len(used), ("%s (fit %.2f n%d / cal %.2f n%d)" % (
                top["rule"], top["p_fit"], top["n_fit"], top["p_cal"], top["n_cal"])) if top else "无", time.time() - t1))
    np.save(G.SCORES / (a.name + ".npy"), phat)
    (G.HERE / ("%s_log.json" % a.name)).write_text(json.dumps({"args": vars(a), "conds": [c[0] for c in conds], "months": log},
                                                                ensure_ascii=False, indent=1), encoding="utf-8")
    G.say("-> %s  %.0f 分钟" % (G.SCORES / (a.name + ".npy"), (time.time() - t0) / 60))
    G.progress("规则挖掘 %s：%d 个条件 %d 条候选规则，nfit %d ncal %d，%d 个月 %.0f 分钟" % (
        a.name, len(conds), len(combos), a.nfit, a.ncal, len(log), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
