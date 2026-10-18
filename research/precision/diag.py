"""离线诊断：精确率-出票量前沿。

    PYTHONIOENCODING=utf-8 python research/precision/diag.py

A 名次基线曲线 B(k)（两段，区间）
B 单维度粗分组条件精确率：固定名次档 1-3 / 4-10 / 11-30 / 31-50，两段，二值 / 三分位，不调阈值（每个维度记一条 diag look）
C 规则网格在筛选集上的前沿（出票量档 × 最高精确率，对匹配量基线 B(k̄) 的 ΔP、NW t）；开发集同一趟算出落盘 grid_hits.npz，不打印
D 嵌套逐月滚动读法（每月用前 12 个月选配置），两段都印（它是诚实的）
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import common_pr as C

BINS = [(1, 3), (4, 10), (11, 30), (31, 50)]
KS = (1, 2, 3, 5, 10, 20, 30, 50)


def ratio_diff(t: pd.DataFrame, cond: np.ndarray, dates: np.ndarray) -> tuple[float, float, float, int, int]:
    """条件行 对 非条件行 的池化精确率、差的 NW t（按天影响函数）。返回 (P_c, P_n, t, V_c, V_n)。"""
    di = pd.Series(np.arange(len(dates)), index=dates)
    D = len(dates)

    def hn(mask):
        h = np.zeros(D)
        n = np.zeros(D)
        r = t[mask]
        if len(r):
            g = r.groupby("date")["hit"].agg(["sum", "size"])
            ii = di[g.index].to_numpy()
            h[ii] = g["sum"].to_numpy()
            n[ii] = g["size"].to_numpy()
        return h, n

    hc, nc = hn(cond)
    hn_, nn = hn(~cond)
    Vc, Vn = nc.sum(), nn.sum()
    if Vc < 20 or Vn < 20:
        return (np.nan, np.nan, np.nan, int(Vc), int(Vn))
    Pc, Pn = hc.sum() / Vc, hn_.sum() / Vn
    IF = D * (hc - Pc * nc) / Vc - D * (hn_ - Pn * nn) / Vn
    se = C.nw_se(IF)
    return (float(Pc), float(Pn), float((Pc - Pn) / se) if se > 0 else np.nan, int(Vc), int(Vn))


def tercile_day(t: pd.DataFrame, col: str, top: int = 50) -> np.ndarray:
    """当天前 top 内的三分位（0/1/2），缺值 -1。"""
    s = t[col].where(t["rank_ret20"] <= top)
    q = s.groupby(t["date"]).rank(pct=True, method="first")
    return np.where(~np.isfinite(q), -1, np.where(q <= 1 / 3, 0, np.where(q <= 2 / 3, 1, 2)))


def day_tercile(t: pd.DataFrame, col: str) -> np.ndarray:
    """日级量按天三分位（各段内）。"""
    out = np.full(len(t), -1)
    for seg in ("screen", "dev"):
        m = (t["seg"] == seg).to_numpy()
        d = t.loc[m].groupby("date")[col].first()
        qs = d.quantile([1 / 3, 2 / 3]).to_numpy()
        v = t.loc[m, col].to_numpy()
        out[m] = np.where(~np.isfinite(v), -1, np.where(v <= qs[0], 0, np.where(v <= qs[1], 1, 2)))
    return out


def conditions(t: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    """名字 -> (维度, 布尔)。"""
    c = {}
    b = t["board"]
    c["板块=北交所"] = ("board", (b == "bj").to_numpy())
    c["板块=创业板"] = ("board", (b == "chinext").to_numpy())
    c["板块=科创板"] = ("board", (b == "star").to_numpy())
    c["板块=主板"] = ("board", (b == "main").to_numpy())
    c["板块≠主板"] = ("board", (b != "main").to_numpy())
    mc = t["mcap"].to_numpy()
    c["市值<20亿"] = ("mcap", mc < 2e9)
    c["市值<50亿"] = ("mcap", mc < 5e9)
    c["市值>100亿"] = ("mcap", mc > 1e10)
    c["近20根有跌停"] = ("nld20", t["nld20"].to_numpy() >= 1)
    tq = tercile_day(t, "turn1")
    c["turn1 高三分位(前50内)"] = ("turn", tq == 2)
    c["turn1 低三分位"] = ("turn", tq == 0)
    c["turn1>=15%"] = ("turn", t["turn1"].to_numpy() >= 0.15)
    r = t["ret20"].to_numpy()
    c["ret20>=0.5"] = ("ret20", r >= 0.5)
    c["ret20>=1.0"] = ("ret20", r >= 1.0)
    c["ret20>=2.0"] = ("ret20", r >= 2.0)
    c["今日涨停"] = ("lu", t["lu_close"].to_numpy() >= 1)
    c["连板>=2"] = ("lu", t["lu_streak"].to_numpy() >= 2)
    c["今日下跌"] = ("ret1", t["ret1"].to_numpy() < 0)
    s10 = t["streak10"].to_numpy()
    c["连续在榜(前10)=1天 首日"] = ("streak", s10 == 1)
    c["连续在榜 2~3天"] = ("streak", (s10 >= 2) & (s10 <= 3))
    c["连续在榜 >=4天"] = ("streak", s10 >= 4)
    c["连续在榜 >=8天"] = ("streak", s10 >= 8)
    c["昨天在前10"] = ("streak", t["in10_yday"].to_numpy())
    mr = t["model_rank"].to_numpy()
    c["影子模型前10"] = ("model", mr <= 10)
    c["影子模型前30"] = ("model", mr <= 30)
    c["影子模型前100"] = ("model", mr <= 100)
    c["影子模型>500名"] = ("model", mr > 500)
    c["龙虎榜今日上榜"] = ("alt_lhb", t["lhb_today"].to_numpy() >= 1)
    c["龙虎榜近5日上榜"] = ("alt_lhb", t["lhb_n5"].to_numpy() >= 1)
    c["机构5日净买>0"] = ("alt_org", t["org_net5_share"].to_numpy() > 0)
    c["两融标的"] = ("alt_rz", t["rz_in"].fillna(False).astype(bool).to_numpy())
    rz = t["rz_chg20"].to_numpy()
    c["融资余额20日增>=20%"] = ("alt_rz", rz >= 0.2)
    c["融资余额20日下降"] = ("alt_rz", rz < 0)
    c["近20日有大宗"] = ("alt_dz", t["dz_n20"].to_numpy() >= 1)
    c["最新户数增"] = ("alt_gd", t["gd_chg1"].to_numpy() > 0)
    c["近30日有机构调研"] = ("alt_sv", t["sv_n30"].to_numpy() >= 1)
    for col, nm in (("mkt_base_t21", "市场:T-21前基准率"), ("mkt_ret20", "市场:池等权20日收益"), ("lu_share20", "市场:20日涨停占比"),
                    ("hot50", "市场:池内ret20>50%占比"), ("top1_ret20", "市场:第1名ret20"), ("top10_ret20_med", "市场:前10 ret20中位")):
        q = day_tercile(t, col)
        c[nm + " 低三分位(天)"] = ("mkt_" + col, q == 0)
        c[nm + " 高三分位(天)"] = ("mkt_" + col, q == 2)
    return c


# ---------------------------------------------------------------
#  规则网格（筛选集上调；开发集算出落盘不印）
# ---------------------------------------------------------------
def grid(t: pd.DataFrame) -> dict[str, tuple[str, np.ndarray]]:
    g = {}
    rk = t["rank_ret20"].to_numpy()
    b = t["board"].to_numpy()
    mc = t["mcap"].to_numpy()
    nld = t["nld20"].to_numpy()
    r20 = t["ret20"].to_numpy()
    s10 = t["streak10"].to_numpy()
    mr = t["model_rank"].to_numpy()
    tq = tercile_day(t, "turn1")
    t1 = t["turn1"].to_numpy()
    lhb = t["lhb_today"].to_numpy() >= 1
    lhb5 = t["lhb_n5"].to_numpy() >= 1
    rz = t["rz_chg20"].to_numpy()
    org = t["org_net5_share"].to_numpy() > 0
    dz = t["dz_n20"].to_numpy() >= 1
    gd = t["gd_chg1"].to_numpy()
    lu = t["lu_close"].to_numpy() >= 1
    ret1 = t["ret1"].to_numpy()
    for K in KS:
        base = rk <= K
        if K <= 10:
            g["rank<=%d" % K] = ("rank", base)
        for nm, m in (("bj", b == "bj"), ("bj|chinext", np.isin(b, ["bj", "chinext"])), ("nonmain", b != "main"), ("chinext", b == "chinext")):
            g["r%d & %s" % (K, nm)] = ("board", base & m)
        for x in (2e9, 3e9, 5e9, 1e10):
            g["r%d & mcap<%d亿" % (K, x / 1e8)] = ("mcap", base & (mc < x))
        g["r%d & nld20==0" % K] = ("nld", base & (nld == 0))
        g["r%d & turn1 高三分位" % K] = ("turn", base & (tq == 2))
        g["r%d & turn1>=10%%" % K] = ("turn", base & (t1 >= 0.10))
        g["r%d & turn1>=20%%" % K] = ("turn", base & (t1 >= 0.20))
        for x in (0.5, 0.8, 1.0, 1.5):
            g["r%d & ret20>=%.1f" % (K, x)] = ("ret20", base & (r20 >= x))
        g["r%d & ret20<2.0" % K] = ("ret20", base & (r20 < 2.0))
        g["r%d & streak10==1" % K] = ("streak", base & (s10 == 1))
        g["r%d & streak10<=2" % K] = ("streak", base & (s10 <= 2) & (s10 >= 1))
        g["r%d & streak10<=3" % K] = ("streak", base & (s10 <= 3) & (s10 >= 1))
        g["r%d & streak10>=3" % K] = ("streak", base & (s10 >= 3))
        g["r%d & streak10>=5" % K] = ("streak", base & (s10 >= 5))
        for M in (5, 10, 20, 30, 50, 100, 200):
            g["r%d & model<=%d" % (K, M)] = ("model", base & (mr <= M))
        g["r%d & lhb_today" % K] = ("alt", base & lhb)
        g["r%d & lhb_n5" % K] = ("alt", base & lhb5)
        g["r%d & rz_chg20>=0.1" % K] = ("alt", base & (rz >= 0.1))
        g["r%d & rz_chg20>=0.2" % K] = ("alt", base & (rz >= 0.2))
        g["r%d & org_net5>0" % K] = ("alt", base & org)
        g["r%d & dz_n20==0" % K] = ("alt", base & ~dz)
        g["r%d & gd_chg1<0" % K] = ("alt", base & (gd < 0))
        g["r%d & lu_today" % K] = ("lu", base & lu)
        g["r%d & ret1>=0" % K] = ("ret1", base & (ret1 >= 0))
        # 日级（空榜日判据）
        for col, nm in (("mkt_base_t21", "base21"), ("mkt_ret20", "mret20"), ("lu_share20", "lu20"), ("hot50", "hot50"),
                        ("top1_ret20", "top1r"), ("top10_ret20_med", "t10med")):
            q = day_tercile(t, col)
            g["r%d & 天:%s 低" % (K, nm)] = ("mkt", base & (q == 0))
            g["r%d & 天:%s 高" % (K, nm)] = ("mkt", base & (q == 2))
            g["r%d & 天:%s 非高" % (K, nm)] = ("mkt", base & (q <= 1) & (q >= 0))
    # 组合（两条 / 三条）
    for K in (1, 2, 3, 5, 10, 20, 30):
        base = rk <= K
        g["r%d & nld0 & mcap<50亿" % K] = ("combo", base & (nld == 0) & (mc < 5e9))
        g["r%d & nld0 & nonmain" % K] = ("combo", base & (nld == 0) & (b != "main"))
        g["r%d & nld0 & ret20>=1" % K] = ("combo", base & (nld == 0) & (r20 >= 1.0))
        g["r%d & nld0 & model<=30" % K] = ("combo", base & (nld == 0) & (mr <= 30))
        g["r%d & nonmain & ret20>=1" % K] = ("combo", base & (b != "main") & (r20 >= 1.0))
        g["r%d & nonmain & model<=30" % K] = ("combo", base & (b != "main") & (mr <= 30))
        g["r%d & nonmain & mcap<50亿" % K] = ("combo", base & (b != "main") & (mc < 5e9))
        g["r%d & model<=30 & ret20>=1" % K] = ("combo", base & (mr <= 30) & (r20 >= 1.0))
        g["r%d & model<=30 & lhb_today" % K] = ("combo", base & (mr <= 30) & lhb)
        g["r%d & nld0 & mcap<50亿 & nonmain" % K] = ("combo", base & (nld == 0) & (mc < 5e9) & (b != "main"))
        g["r%d & nld0 & mcap<50亿 & model<=30" % K] = ("combo", base & (nld == 0) & (mc < 5e9) & (mr <= 30))
        g["r%d & nld0 & streak10<=3" % K] = ("combo", base & (nld == 0) & (s10 <= 3) & (s10 >= 1))
        g["r%d & nld0 & turn1 高" % K] = ("combo", base & (nld == 0) & (tq == 2))
        g["r%d & rz_chg20>=0.1 & nld0" % K] = ("combo", base & (rz >= 0.1) & (nld == 0))
        g["r%d & lhb_today & nonmain" % K] = ("combo", base & lhb & (b != "main"))
    return g


def main():
    C.P.wait_prod_window()
    t0 = time.time()
    t = C.load()
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))
    segs = {s: t[t["seg"] == s].reset_index(drop=True) for s in ("screen", "dev")}
    Bs = {s: C.Baseline(segs[s]) for s in segs}

    say("== A 名次基线曲线 B(k)：精确率%（块自举 95% 区间）、每月只数 ==")
    say("%-4s %-28s %-28s" % ("k", "筛选集 2022-01..2024-05", "开发集 2024-06..2025-12"))
    for k in range(1, 11):
        row = "%-4d" % k
        for s in ("screen", "dev"):
            B = Bs[s]
            h = B.hits_k(k)
            n = np.full(B.D, float(k))
            lo, hi = C.boot_ratio(h, n)
            row += " %5.2f [%4.1f,%4.1f] %3d只/月      " % (100 * B.precision_k(k), 100 * lo, 100 * hi, round(20 * k))
        say(row)
    say("  名次档：" + "  ".join("%d-%d 筛 %.1f / 开 %.1f" % (lo, hi, 100 * segs["screen"][(segs["screen"]["rank_ret20"] >= lo) & (segs["screen"]["rank_ret20"] <= hi)]["hit"].mean(),
                                                   100 * segs["dev"][(segs["dev"]["rank_ret20"] >= lo) & (segs["dev"]["rank_ret20"] <= hi)]["hit"].mean()) for lo, hi in BINS))

    say("\n== B 单维度粗分组条件精确率（有条件 对 无条件，池化精确率%，括号 NW t；按名次档；两段）==")
    say("   每格：有 / 无 (t)。占比 = 条件行在该档的占比。维度记 diag look（ledger），不调阈值。")
    conds = {s: conditions(segs[s]) for s in segs}
    names = list(conds["screen"])
    dims_seen = set()
    header = "%-30s" % "条件"
    for s in ("筛", "开"):
        for lo, hi in BINS:
            header += " | %s%d-%d" % (s, lo, hi)
    say(header)
    same_sign = {}
    for nm in names:
        row = "%-30s" % nm
        signs = []
        for s in ("screen", "dev"):
            dim, cond = conds[s][nm]
            dims_seen.add(dim)
            rk = segs[s]["rank_ret20"].to_numpy()
            for lo, hi in BINS:
                m = (rk >= lo) & (rk <= hi)
                sub = segs[s][m].reset_index(drop=True)
                Pc, Pn, tt, Vc, Vn = ratio_diff(sub, cond[m], Bs[s].dates)
                if np.isfinite(Pc):
                    row += " | %4.1f/%4.1f(%+4.1f)%3d%%" % (100 * Pc, 100 * Pn, tt, 100 * Vc / max(Vc + Vn, 1))
                    if (lo, hi) in ((1, 3), (4, 10)):
                        signs.append(np.sign(Pc - Pn))
                else:
                    row += " | %-18s" % ("  n=%d" % Vc)
        # 两段同号：1-3 和 4-10 两档上 筛 / 开 的符号一致
        ss = (len(signs) == 4 and signs[0] == signs[2] and signs[1] == signs[3] and signs[0] != 0)
        same_sign[nm] = bool(ss)
        say(row + ("  同号" if ss else ""))
    for dim in sorted(dims_seen):
        C.ledger_add({"kind": "diag", "dim": dim, "note": "粗分组条件精确率，两段都看（名次档 1-3/4-10/11-30/31-50）"})
    say("  两段同号（1-3 和 4-10 两档符号一致）的条件：%s" % [k for k, v in same_sign.items() if v])

    say("\n== C 规则网格在筛选集上的前沿（出票量档内按 P 排前 6；ΔP 对匹配量基线 B(k̄)；开发集算了不印）==")
    G = {s: grid(segs[s]) for s in segs}
    res = {}
    store = {}
    for s in segs:
        for nm, (fam, sel) in G[s].items():
            e = C.evaluate(segs[s], sel, Bs[s], boot=False)
            res.setdefault(nm, {})[s] = e
            store[(nm, s)] = (e.get("h", np.zeros(Bs[s].D)), e.get("n", np.zeros(Bs[s].D)))
            res[nm]["fam"] = fam
    say("  配置数 %d，%.0fs" % (len(G["screen"]), time.time() - t0))
    vb = [(0, 0.35), (0.35, 0.75), (0.75, 1.5), (1.5, 2.5), (2.5, 4), (4, 6), (6, 10.01)]
    rows = []
    for nm, r in res.items():
        e = r["screen"]
        if not np.isfinite(e.get("P", np.nan)):
            continue
        rows.append({"name": nm, "fam": r["fam"], "V": e["V"], "days": e["days_out"], "vbar": e["vbar"], "P": e["P"], "P_base": e["P_base"],
                     "dP": e["dP"], "t": e["t"], "se": e["se"], "dP_within": e["dP_within"], "empty": e["empty_share"]})
    R = pd.DataFrame(rows)
    R.to_csv(C.HERE / "grid_screen.csv", index=False, encoding="utf-8")
    floorV, floorD = C.VOL_FLOOR["screen"]
    for lo, hi in vb:
        sub = R[(R["vbar"] >= lo) & (R["vbar"] < hi) & (R["V"] >= floorV) & (R["days"] >= floorD)].sort_values("P", ascending=False)
        say("  -- 出票量 %.2f~%.2f 只/天（过量门槛 V>=%d 天>=%d 的配置 %d 个）--" % (lo, hi, floorV, floorD, len(sub)))
        for _, x in sub.head(6).iterrows():
            say("    %-36s V %5d 天 %3d k̄ %4.2f  P %5.1f%%  B(k̄) %5.1f  ΔP %+5.1fpp (t %+4.2f, se %.1f)  日配 %+5.1f  空榜 %3.0f%%" % (
                x["name"], x["V"], x["days"], x["vbar"], 100 * x["P"], 100 * x["P_base"], 100 * x["dP"], x["t"], 100 * x["se"], 100 * x["dP_within"], 100 * x["empty"]))
        bad = R[(R["vbar"] >= lo) & (R["vbar"] < hi) & ((R["V"] < floorV) | (R["days"] < floorD))]
        if len(bad):
            say("    （量不够门槛的 %d 个不印，最高 P %.1f%%）" % (len(bad), 100 * bad["P"].max()))
    say("  按族：筛选集 ΔP 最好的配置（过量门槛）")
    ok = R[(R["V"] >= floorV) & (R["days"] >= floorD)]
    for fam, sub in ok.groupby("fam"):
        x = sub.sort_values("dP", ascending=False).iloc[0]
        say("    %-7s %-36s k̄ %4.2f P %5.1f%% ΔP %+5.1fpp (t %+4.2f)  族内 ΔP>0 的 %d/%d" % (
            fam, x["name"], x["vbar"], 100 * x["P"], 100 * x["dP"], x["t"], int((sub["dP"] > 0).sum()), len(sub)))
    say("  MDE80 对照（results_power.txt，前 10 内）：v̄ 0.5 → 9.6pp，1 → 8.2，2 → 6.3，3 → 4.8，5 → 2.8；前 50 内 10.5 / 9.8 / 8.1 / 6.6 / 4.7")
    np.savez_compressed(C.HERE / "grid_hits.npz", names=np.array(list(res)), fams=np.array([res[n]["fam"] for n in res]),
                        **{"%s__%s__%s" % (s, k, i): np.stack([store[(n, s)][j] for n in res]) for s in segs for j, k in ((0, "h"), (1, "n")) for i in ("x",)})
    json.dump({"screen_dates": list(Bs["screen"].dates), "dev_dates": list(Bs["dev"].dates)}, (C.HERE / "grid_dates.json").open("w"))

    say("\n== D 嵌套逐月滚动读法（每月用前 12 个月在网格里选 ΔP 最大且 v̄>=0.5、出票天>=1/6 的配置；两段都诚实）==")
    names_all = list(res)
    for s in segs:
        B = Bs[s]
        dates = B.dates
        H = np.stack([store[(n, s)][0] for n in names_all])
        N = np.stack([store[(n, s)][1] for n in names_all])
        # 拼接前一段（筛选集的嵌套从 2022-01 起只能用 2021 年的数据 -> 没有，所以从第 13 个月起；开发集用筛选集末 12 个月）
        if s == "dev":
            Hs = np.stack([store[(n, "screen")][0] for n in names_all])
            Ns = np.stack([store[(n, "screen")][1] for n in names_all])
            H = np.concatenate([Hs, H], axis=1)
            N = np.concatenate([Ns, N], axis=1)
            dates = np.concatenate([Bs["screen"].dates, dates])
            Hb = np.concatenate([np.stack([Bs["screen"].hits_k(k) for k in range(1, 51)]), np.stack([B.hits_k(k) for k in range(1, 51)])], axis=1)
        else:
            Hb = np.stack([B.hits_k(k) for k in range(1, 51)])
        months = np.array([d[:7] for d in dates])
        um = sorted(set(months))
        picked = []
        hsum = nsum = 0.0
        hb_sum = 0.0
        for i, m in enumerate(um):
            if i < 12 or (s == "dev" and m < "2024-06"):
                continue
            win = np.isin(months, um[i - 12:i])
            Vw = N[:, win].sum(axis=1)
            Dw = win.sum()
            kb = Vw / Dw
            # 匹配量基线命中（线性插值）
            f = np.floor(kb).astype(int)
            frac = kb - f
            cumb = np.cumsum(Hb[:, win], axis=0)
            hb = np.where(f >= 1, cumb[np.maximum(f - 1, 0)].sum(axis=1) if False else 0, 0)
            # 逐配置算（配置数几百，窗口 12 个月，直接循环）
            dP = np.full(len(names_all), -np.inf)
            for j in range(len(names_all)):
                if Vw[j] < 0.5 * Dw or (N[j, win] > 0).sum() < Dw / 6:
                    continue
                k = kb[j]
                fj = int(np.floor(k))
                hbj = (cumb[fj - 1] if fj >= 1 else 0) + (k - fj) * (Hb[fj, win] if fj < 50 else 0)
                dP[j] = (H[j, win].sum() - np.sum(hbj)) / Vw[j]
            j = int(np.argmax(dP))
            cur = months == m
            hsum += H[j, cur].sum()
            nsum += N[j, cur].sum()
            kcur = N[j, win].sum() / Dw
            fj = int(np.floor(kcur))
            hb_sum += np.sum((cumb[fj - 1][0:0].sum() if False else 0))
            hb_m = (np.cumsum(Hb[:, cur], axis=0)[fj - 1] if fj >= 1 else 0) + (kcur - fj) * (Hb[fj, cur] if fj < 50 else 0)
            hb_sum += np.sum(hb_m)
            picked.append((m, names_all[j], float(dP[j]), int(N[j, cur].sum()), int(H[j, cur].sum())))
        Dn = sum(int((months == m).sum()) for m, *_ in picked)
        say("  %s：%d 个月，出票 %d 只（%.2f 只/天），精确率 %.1f%%，同量名次基线 %.1f%%，ΔP %+.1fpp" % (
            s, len(picked), int(nsum), nsum / max(Dn, 1), 100 * hsum / max(nsum, 1), 100 * hb_sum / max(nsum, 1), 100 * (hsum - hb_sum) / max(nsum, 1)))
        cnt = pd.Series([p[1] for p in picked]).value_counts()
        say("    选中最多的配置：%s" % cnt.head(5).to_dict())
    txt = "\n".join(L)
    (C.HERE / "results_diag.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 results_diag.txt 写完：A 基线曲线、B 单维度粗分组（%d 条、%d 个维度记 diag）、C 筛选集网格 %d 个配置（开发集落盘不印）、D 嵌套逐月" % (
        len(names), len(dims_seen), len(G["screen"])))


if __name__ == "__main__":
    main()
