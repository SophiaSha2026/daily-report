"""评一份样本外分数（V2 / L2 / L3 各变体）：ret20 前 K 内按分数取前 n 对 B(k̄)，全候选池前 n 对照，同配置对「不加杠杆」的对照分数配对。

    PYTHONIOENCODING=utf-8 python research/blind/eval_scores.py --name v2_lag20 --tag V2 [--control lgb_price] [--dev]

筛选集网格全部记 kind=screen；开发集先落盘 results_<tag>_dev.json，只读筛选集最好的一个（kind=arm）。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common_bl as B

RANK_BINS = [(1, 3), (4, 10), (11, 30), (31, 50)]
QS = (0.95, 0.98, 0.99)


def diag(t, pct, title, L):
    L.append("\n%s（精确率%% / 只数）" % title)
    names = [">=.95", ".80-.95", "<.80", "缺"]
    grp = np.full(len(pct), "缺", dtype="U8")
    ok = np.isfinite(pct)
    grp[ok & (pct >= 0.95)] = ">=.95"
    grp[ok & (pct >= 0.80) & (pct < 0.95)] = ".80-.95"
    grp[ok & (pct < 0.80)] = "<.80"
    L.append("%-8s %-8s | %s" % ("段", "名次档", "  ".join("%-20s" % x for x in names)))
    for seg in ("screen", "dev"):
        for lo, hi in RANK_BINS:
            m = (t["seg"].to_numpy() == seg) & (t["rank_ret20"].to_numpy() >= lo) & (t["rank_ret20"].to_numpy() <= hi)
            cells = []
            for g in names:
                mm = m & (grp == g)
                n = int(mm.sum())
                cells.append("%5.1f%% (%6d)" % (100 * t["hit"].to_numpy()[mm].mean() if n else np.nan, n))
            L.append("%-8s %-8s | %s" % (seg, "%d-%d" % (lo, hi), "  ".join("%-20s" % c for c in cells)))
    L.append("按天配对差（>=.95 − <.80）：")
    for seg in ("screen", "dev"):
        for lo, hi in RANK_BINS:
            m = (t["seg"].to_numpy() == seg) & (t["rank_ret20"].to_numpy() >= lo) & (t["rank_ret20"].to_numpy() <= hi)
            d = pd.DataFrame({"date": t["date"].to_numpy()[m], "g": grp[m], "h": t["hit"].to_numpy()[m]})
            d = d[d["g"].isin([">=.95", "<.80"])]
            pv = d.pivot_table(index="date", columns="g", values="h", aggfunc="mean").dropna()
            if len(pv) < 30:
                L.append("  %-6s %-6s 配得上的天 %d（不足）" % (seg, "%d-%d" % (lo, hi), len(pv)))
                continue
            u = (pv[">=.95"] - pv["<.80"]).to_numpy()
            L.append("  %-6s %-6s 天 %3d  差 %+5.1fpp  NW t %+5.2f" % (seg, "%d-%d" % (lo, hi), len(pv), 100 * u.mean(), B.PR.nw_t(u)))


def paired_ctrl(t, sel_a, sel_b, seg):
    """两个同量选择的按天命中差（a − b），返回 (差 pp 对 V_a, NW t)。"""
    m = (t["seg"].to_numpy() == seg)
    d = t["date"].to_numpy()
    ha = pd.Series(t["hit"].to_numpy()[m & sel_a]).groupby(d[m & sel_a]).sum()
    hb = pd.Series(t["hit"].to_numpy()[m & sel_b]).groupby(d[m & sel_b]).sum()
    days = np.array(sorted(set(d[m])))
    u = ha.reindex(days, fill_value=0).to_numpy() - hb.reindex(days, fill_value=0).to_numpy()
    V = max(int((m & sel_a).sum()), 1)
    return 100 * u.sum() / V, B.PR.nw_t(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--control", default="")
    ap.add_argument("--nodiag", action="store_true")
    ap.add_argument("--nodev", action="store_true", help="只跑筛选集网格（开发集 json 落盘不读）；读开发集前先用 prereg_bl.py 登记")
    ap.add_argument("--dev-config", default="", help="已登记的配置名（如 'K10 n1'），只读这一个的开发集")
    a = ap.parse_args()
    t0 = time.time()
    t = B.load_keys()
    s = B.load_scores(a.name).astype(float)
    assert len(s) == len(t)
    rank = t["rank_ret20"].to_numpy()
    pct = B.day_pct(t, s)
    ctrl = B.load_scores(a.control).astype(float) if a.control else None
    L = ["%s 评分 %s  有分数行 %.1f%%  对照 %s" % (a.tag, a.name, 100 * np.isfinite(s).mean(), a.control or "无")]
    if not a.nodiag:
        diag(t, pct, "诊断：名次档 × %s 当天分位" % a.name, L)
        B.ledger_add({"kind": "diag", "what": "%s 名次档 × 分位组 两段" % a.tag, "score": a.name})
    rows, store = [], {}
    L.append("\n筛选集网格（kind=screen）：前 K 内按分数取前 n；全池前 n；分位门槛")
    cfgs = []
    for K in B.KS:
        for nn in B.NS:
            cfgs.append(("K%d n%d" % (K, nn), (rank <= K), nn))
    for nn in B.NS:
        cfgs.append(("all n%d" % nn, np.ones(len(t), bool), nn))
    for q in QS:
        cfgs.append(("K50 q%.2f n10" % q, (rank <= 50) & (pct >= q), 10))
    for cname, cand, nn in cfgs:
        sel = B.topn_by(t, cand, s, nn)
        name = "%s %s" % (a.tag, cname)
        r = B.eval_both(t, sel, boot=False)
        rs = B.screen_row(name, r["screen"])
        line = B.G.fmt(name, r["screen"], "screen")
        if ctrl is not None:
            selc = B.topn_by(t, cand if not cname.startswith("K50 q") else (rank <= 50) & (B.day_pct(t, ctrl) >= float(cname.split()[1][1:])), ctrl, nn)
            dpc, tc = paired_ctrl(t, sel, selc, "screen")
            rs["dP_ctrl"], rs["t_ctrl"] = round(dpc, 2), round(tc, 2)
            line += "  对照差 %+.2fpp (t %+.2f)" % (dpc, tc)
            dpd, td = paired_ctrl(t, sel, selc, "dev")
            store_ctrl = {"dP_ctrl": round(dpd, 2), "t_ctrl": round(td, 2)}
        else:
            store_ctrl = {}
        rows.append(rs)
        store[name] = {"screen": rs, "dev": {**B.screen_row(name, r["dev"]), **store_ctrl}}
        B.ledger_add({"kind": "screen", **rs})
        L.append(line)
    (B.HERE / ("results_%s_dev.json" % a.tag)).write_text(json.dumps(store, ensure_ascii=False, default=float, indent=0), encoding="utf-8")
    best = B.pick_best([r for r in rows if not r["name"].split()[1].startswith("all")])
    L.append("\n筛选集最好的（过量门槛，不含全池配置）：%s" % json.dumps(best, ensure_ascii=False, default=float))
    if a.nodev:
        best = None
        L.append("（--nodev：开发集没读，先登记再读）")
    if a.dev_config:
        best = {"name": "%s %s" % (a.tag, a.dev_config)}
    if best is not None:
        cname = " ".join(best["name"].split()[1:])
        cand, nn = [(c, n) for nm, c, n in cfgs if nm == cname][0]
        sel = B.topn_by(t, cand, s, nn)
        r = B.eval_both(t, sel, boot=True)
        L.append(B.fmt_both(best["name"], r, dev=True))
        L.append("判定：%s" % B.verdict(r))
        row = {"kind": "arm", "name": best["name"], "screen": B.screen_row(best["name"], r["screen"]),
               "dev": B.screen_row(best["name"], r["dev"]), "comb_t": r["comb"]["t"], "boot_p": r["comb"]["boot_p"], "verdict": B.verdict(r)}
        if ctrl is not None:
            selc = B.topn_by(t, cand, ctrl, nn)
            for seg in ("screen", "dev"):
                dpc, tc = paired_ctrl(t, sel, selc, seg)
                L.append("  对照 %s 同配置 %s：差 %+.2fpp (NW t %+.2f)" % (a.control, seg, dpc, tc))
                row["ctrl_" + seg] = {"dP": round(dpc, 2), "t": round(tc, 2)}
        B.ledger_add(row)
        for seg in ("screen", "dev"):
            e = r[seg]
            L.append("%s：出票 %d 命中 %d 每天 %.2f 只；逐年 %s" % (seg, e["V"], int(sum(e["h"])), e["vbar"],
                                                           {k: "%d/%.0f%%" % (v[0], 100 * v[1]) for k, v in e["by_year"].items()}))
    # 全池前 10（每月）对照 gate 的读法：月前 10 / 20 行精确率
    L.append("\n全候选池按分数每月事后前 N 行（gate 对照读法，乐观）：")
    for seg in ("screen", "dev"):
        m = (t["seg"].to_numpy() == seg) & np.isfinite(s)
        d = pd.DataFrame({"ym": t["date"].to_numpy()[m].astype("U7"), "s": s[m], "h": t["hit"].to_numpy()[m]})
        d["r"] = d.groupby("ym")["s"].rank(ascending=False, method="first")
        L.append("  %s 月前 10 / 20 / 40：%.1f / %.1f / %.1f%%" % (seg, *(100 * d[d["r"] <= k]["h"].mean() for k in (10, 20, 40))))
    (B.HERE / ("results_%s.txt" % a.tag)).write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.progress("eval_scores %s（%s）：网格 %d 配置 + 开发集 1 臂 -> results_%s.txt（%.0fs）" % (a.tag, a.name, len(rows), a.tag, time.time() - t0))


if __name__ == "__main__":
    main()
