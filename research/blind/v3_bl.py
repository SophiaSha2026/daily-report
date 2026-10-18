"""V3 逐票的模型可信度 + V3b 「以前走出来过」（criteria.txt 第 1 节）。

    PYTHONIOENCODING=utf-8 python research/blind/v3_bl.py [--score lgb_price]

V3：c(code, T) = Σ(y − s) / (n + 20)，只用该票 end20 < T（窗口已关）且有样本外分数的行（merge_asof 按 end20 对齐，不含等号）。
V3b：c_r = (h_r + 5 p0) / (n_r + 5)，h_r / n_r = 该票过去进 ret20 前 10 且已关窗的命中 / 次数，p0 = 全市场前 10 到 T 之前的累计精确率。
诊断（kind=diag）：名次档 × 可信度组。网格（kind=screen）只在筛选集；开发集每族一臂。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common_bl as B

RANK_BINS = [(1, 3), (4, 10), (11, 30), (31, 50)]


def asof_cum(t: pd.DataFrame, ev: pd.DataFrame, vals: list[str]) -> pd.DataFrame:
    """ev: code end20 + vals（每行一个已关窗的观察）。返回和 t 同序：各 vals 在 end20 < date 的累计和、累计个数 cnt。"""
    e = ev.groupby(["code", "end20"], sort=True)[vals].sum()
    e["cnt"] = ev.groupby(["code", "end20"], sort=True).size()
    e = e.reset_index()
    for v in vals + ["cnt"]:
        e[v] = e.groupby("code")[v].cumsum()
    e["_e"] = pd.to_datetime(e["end20"])
    q = pd.DataFrame({"code": t["code"].to_numpy(), "_d": pd.to_datetime(t["date"]).to_numpy(), "_i": np.arange(len(t))})
    m = pd.merge_asof(q.sort_values("_d"), e.sort_values("_e")[["code", "_e"] + vals + ["cnt"]], left_on="_d", right_on="_e",
                      by="code", direction="backward", allow_exact_matches=False)
    m = m.sort_values("_i")
    out = pd.DataFrame({v: m[v].fillna(0).to_numpy() for v in vals + ["cnt"]}, index=t.index)
    # 自检：对齐到的 end20 全部 < date
    ok = m["_e"].notna().to_numpy()
    assert (m["_e"].to_numpy()[ok] < m["_d"].to_numpy()[ok]).all()
    return out


def diag(t: pd.DataFrame, grp: np.ndarray, names: list[str], title: str, L: list[str]) -> None:
    L.append("\n%s（精确率%% / 只数）" % title)
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
    L.append("按天配对差（%s − %s，同名次档同一天都有的天）：" % (names[0], names[-1]))
    for seg in ("screen", "dev"):
        for lo, hi in RANK_BINS:
            m = (t["seg"].to_numpy() == seg) & (t["rank_ret20"].to_numpy() >= lo) & (t["rank_ret20"].to_numpy() <= hi)
            d = pd.DataFrame({"date": t["date"].to_numpy()[m], "g": grp[m], "h": t["hit"].to_numpy()[m]})
            d = d[d["g"].isin([names[0], names[-1]])]
            pv = d.pivot_table(index="date", columns="g", values="h", aggfunc="mean").dropna()
            if len(pv) < 30:
                L.append("  %-6s %-6s 配得上的天 %d（不足）" % (seg, "%d-%d" % (lo, hi), len(pv)))
                continue
            u = (pv[names[0]] - pv[names[-1]]).to_numpy()
            L.append("  %-6s %-6s 天 %3d  差 %+5.1fpp  NW t %+5.2f" % (seg, "%d-%d" % (lo, hi), len(pv), 100 * u.mean(), B.PR.nw_t(u)))


def run_grid(t, cands: dict, orders: dict, L, rows, store, prefix):
    for cname, cand in cands.items():
        for oname, key in orders.items():
            for nn in B.NS:
                sel = B.topn_by(t, cand, key, nn)
                name = "%s %s n%d %s" % (prefix, cname, nn, oname)
                r = B.eval_both(t, sel, boot=False)
                rs = B.screen_row(name, r["screen"])
                rows.append(rs)
                store[name] = {"screen": rs, "dev": B.screen_row(name, r["dev"]), "sel": None}
                B.ledger_add({"kind": "screen", **rs})
                L.append(B.G.fmt(name, r["screen"], "screen"))


def read_dev(t, best, build_sel, L, tag):
    sel = build_sel(best["name"])
    r = B.eval_both(t, sel, boot=True)
    L.append("\n%s 开发集一臂（筛选集最好的）：%s" % (tag, best["name"]))
    L.append(B.fmt_both(best["name"], r, dev=True))
    L.append("判定：%s" % B.verdict(r))
    B.ledger_add({"kind": "arm", "name": best["name"], "screen": B.screen_row(best["name"], r["screen"]),
                  "dev": B.screen_row(best["name"], r["dev"]), "comb_t": r["comb"]["t"], "boot_p": r["comb"]["boot_p"],
                  "verdict": B.verdict(r)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="lgb_price")
    a = ap.parse_args()
    t0 = time.time()
    t = B.load_keys()
    s = B.load_scores(a.score).astype(float)
    n = len(t)
    y = t["y_open"].to_numpy(float)
    rank = t["rank_ret20"].to_numpy()
    ret20 = t["ret20"].to_numpy(float)
    L = ["V3 逐票可信度  分数 %s  表 %d 行" % (a.score, n)]
    # ---- V3：模型残差可信度
    ok = np.isfinite(s) & np.isfinite(y)
    ev = pd.DataFrame({"code": t["code"].to_numpy()[ok], "end20": t["end20"].to_numpy()[ok], "res": (y - s)[ok]})
    cum = asof_cum(t, ev, ["res"])
    c = cum["res"].to_numpy() / (cum["cnt"].to_numpy() + 20.0)
    hist_n = cum["cnt"].to_numpy()
    L.append("V3 可信度：有历史的行 %.1f%%，历史个数中位 %d；c 分位 10/50/90 %s" % (
        100 * (hist_n > 0).mean(), np.median(hist_n[hist_n > 0]), np.round(np.quantile(c[hist_n > 0], [0.1, 0.5, 0.9]), 4)))
    cpct = B.day_pct(t, np.where(hist_n > 0, c, np.nan))
    grp = np.where(hist_n == 0, "无历史", np.where(c > 0, "c>0", "c<=0"))
    diag(t, grp, ["c>0", "c<=0", "无历史"], "诊断 D：名次档 × 模型残差可信度（c>0 = 模型历来低估这只票）", L)
    B.ledger_add({"kind": "diag", "what": "V3 名次档 × 残差可信度组 两段"})
    grp2 = np.where(~np.isfinite(cpct), "无历史", np.where(cpct >= 0.8, "c前20%", np.where(cpct <= 0.2, "c后20%", "中")))
    diag(t, grp2, ["c前20%", "中", "c后20%"], "诊断 D2：名次档 × 可信度当天分位", L)
    B.ledger_add({"kind": "diag", "what": "V3 名次档 × 可信度当天分位组 两段"})
    rows, store = [], {}
    L.append("\n筛选集网格 V3（kind=screen）")
    for K in B.KS:
        cands = {"K%d c>0" % K: (rank <= K) & (hist_n > 0) & (c > 0),
                 "K%d cp.8" % K: (rank <= K) & (cpct >= 0.8),
                 "K%d cp.9" % K: (rank <= K) & (cpct >= 0.9)}
        run_grid(t, cands, {"sT": s, "ret20": ret20, "c": c}, L, rows, store, "V3")
    best = B.pick_best(rows)
    L.append("\n筛选集最好的（过量门槛）：%s" % json.dumps(best, ensure_ascii=False))

    def build_v3(name):
        p = name.split()
        K = int(p[1][1:])
        cond = p[2]
        nn = int(p[3][1:])
        order = p[4]
        cand = (rank <= K) & ((hist_n > 0) & (c > 0) if cond == "c>0" else (cpct >= float("0" + cond[2:])))
        return B.topn_by(t, cand, {"sT": s, "ret20": ret20, "c": c}[order], nn)
    if best is not None:
        r = read_dev(t, best, build_v3, L, "V3")
        for seg in ("screen", "dev"):
            e = r[seg]
            L.append("%s：出票 %d 命中 %d 每天 %.2f 只" % (seg, e["V"], int(sum(e["h"])), e["vbar"]))
    # ---- V3b：以前进前 10 走出来过
    ok10 = (rank <= 10) & np.isfinite(y)
    ev10 = pd.DataFrame({"code": t["code"].to_numpy()[ok10], "end20": t["end20"].to_numpy()[ok10], "h": y[ok10]})
    cum10 = asof_cum(t, ev10, ["h"])
    h_r, n_r = cum10["h"].to_numpy(), cum10["cnt"].to_numpy()
    # p0：全市场前 10 到 T 之前已关窗的累计精确率（只用过去）
    g = ev10.groupby("end20")["h"].agg(["sum", "size"]).sort_index()
    g["cs"], g["cn"] = g["sum"].cumsum(), g["size"].cumsum()
    e_dates = g.index.to_numpy().astype("U10")
    pos = np.searchsorted(e_dates, t["date"].to_numpy().astype("U10"), side="left") - 1
    p0 = np.where(pos >= 0, g["cs"].to_numpy()[np.clip(pos, 0, None)] / np.maximum(g["cn"].to_numpy()[np.clip(pos, 0, None)], 1), np.nan)
    c_r = (h_r + 5 * np.nan_to_num(p0, nan=0.1)) / (n_r + 5)
    grp3 = np.where(n_r == 0, "第一次", np.where(h_r > 0, "以前中过", "以前没中"))
    L.append("\nV3b：前 10 行里 第一次 / 以前进过没中 / 以前中过 的占比（两段）：")
    for seg in ("screen", "dev"):
        m = (t["seg"].to_numpy() == seg) & (rank <= 10)
        vc = pd.Series(grp3[m]).value_counts(normalize=True).round(3).to_dict()
        L.append("  %s %s  以前进前10次数中位（有历史者）%d" % (seg, vc, np.median(n_r[m & (n_r > 0)]) if (m & (n_r > 0)).any() else -1))
    diag(t, grp3, ["以前中过", "以前没中", "第一次"], "诊断 E：名次档 × 这只票以前进 ret20 前 10 的战绩", L)
    B.ledger_add({"kind": "diag", "what": "V3b 名次档 × 以前进前 10 战绩 两段"})
    rows_b, store_b = [], {}
    L.append("\n筛选集网格 V3b（kind=screen）")
    cands = {"K10 hr>0": (rank <= 10) & (h_r > 0), "K10 nr>0": (rank <= 10) & (n_r > 0), "K10 nr=0": (rank <= 10) & (n_r == 0),
             "K20 hr>0": (rank <= 20) & (h_r > 0), "K50 hr>0": (rank <= 50) & (h_r > 0)}
    run_grid(t, cands, {"cr": c_r, "ret20": ret20, "sT": s}, L, rows_b, store_b, "V3b")
    best_b = B.pick_best(rows_b)
    L.append("\n筛选集最好的（过量门槛）：%s" % json.dumps(best_b, ensure_ascii=False))

    def build_v3b(name):
        p = name.split()
        K = int(p[1][1:])
        cond = p[2]
        nn = int(p[3][1:])
        order = p[4]
        cand = (rank <= K) & ({"hr>0": h_r > 0, "nr>0": n_r > 0, "nr=0": n_r == 0}[cond])
        return B.topn_by(t, cand, {"cr": c_r, "ret20": ret20, "sT": s}[order], nn)
    if best_b is not None:
        r = read_dev(t, best_b, build_v3b, L, "V3b")
        for seg in ("screen", "dev"):
            e = r[seg]
            L.append("%s：出票 %d 命中 %d 每天 %.2f 只" % (seg, e["V"], int(sum(e["h"])), e["vbar"]))
    store.update(store_b)
    (B.HERE / "results_v3_dev.json").write_text(json.dumps(store, ensure_ascii=False, default=float, indent=0), encoding="utf-8")
    (B.HERE / "results_v3.txt").write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.progress("v3_bl：V3 诊断 2 张 + 网格 %d + 开发集 1 臂；V3b 诊断 1 张 + 网格 %d + 开发集 1 臂 -> results_v3.txt（%.0fs）" % (
        len(rows), len(rows_b), time.time() - t0))


if __name__ == "__main__":
    main()
