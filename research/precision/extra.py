"""补充诊断 + 正式臂筛选（按 criteria.txt 2f / 2g 对筛选集网格机械判定）。

    PYTHONIOENCODING=utf-8 python research/precision/extra.py

1 量不够门槛的高精确率配置（只为记录「精确率最高能到多少」，不立臂）
2 每族筛选集最好配置的功效核对：表 MDE80（criteria 附录，按 v̄ 插值）和 臂自身 se 的 MDE80 两种
3 补一个没碰过的日级维度：当天 ret20 前 10 的影子模型分数均值（市场状态的另一种读法）、个股影子分数前 50 内三分位
4 正式臂筛选：两段同号（diag 第 B 节的条件 / 维度）且 筛选集 ΔP >= 表 MDE80 → 立臂；写 selection.json 和 ledger decision
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common_pr as C
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("diag_pr", C.HERE / "diag.py")   # 不能 import diag：research/polish/diag.py 同名且在 sys.path 更前
DG = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(DG)

TABLE = {10: [(0.25, 11.54), (0.5, 9.58), (1.0, 8.21), (2.0, 6.30), (3.0, 4.77), (5.0, 2.79), (10.0, 0.0)],
         50: [(0.25, 11.60), (0.5, 10.50), (1.0, 9.81), (2.0, 8.11), (3.0, 6.62), (5.0, 4.72), (10.0, 0.0)]}
# 两段同号的维度 / 条件（diag.py 第 B 节的输出，1-3 和 4-10 两档符号一致）
SAME_SIGN_TOKENS = ["nonmain", "chinext", "bj|chinext", "mcap<20亿", "nld20==0", "nld0", "turn1 高", "turn1>=", "ret20>=1", "lhb_today", "lhb_n5",
                    "org_net5", "rz_chg20", "天:base21 低", "天:hot50 低", "天:t10med 低"]


def mde_table(vbar: float, within10: bool) -> float:
    tb = TABLE[10 if within10 else 50]
    xs = [a for a, _ in tb]
    ys = [b for _, b in tb]
    return float(np.interp(vbar, xs, ys))


def main():
    C.P.wait_prod_window()
    R = pd.read_csv(C.HERE / "grid_screen.csv")
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))
    floorV, floorD = C.VOL_FLOOR["screen"]
    say("== 1 量不够门槛（V<%d 或 天<%d）里精确率最高的配置（筛选集；只记录，不立臂）==" % (floorV, floorD))
    bad = R[(R["V"] < floorV) | (R["days"] < floorD)].sort_values("P", ascending=False)
    for _, x in bad.head(10).iterrows():
        say("  %-36s V %4d 天 %3d k̄ %4.2f  P %5.1f%%  B(k̄) %5.1f  ΔP %+5.1fpp (t %+4.2f)" % (
            x["name"], x["V"], x["days"], x["vbar"], 100 * x["P"], 100 * x["P_base"], 100 * x["dP"], x["t"]))

    say("\n== 2 每族筛选集最好配置（过量门槛）的功效核对 ==")
    ok = R[(R["V"] >= floorV) & (R["days"] >= floorD) & (R["vbar"] <= 10)].copy()   # k̄ > 10 不考虑（条文 1）
    ok["within10"] = ok["name"].str.extract(r"r(\d+)")[0].fillna("10").astype(int) <= 10
    ok["mde_tab"] = [mde_table(v, w) for v, w in zip(ok["vbar"], ok["within10"])]
    ok["mde_own"] = 100 * ok["se"] * 3.34
    ok["same_sign"] = ok["name"].map(lambda n: any(tok in n for tok in SAME_SIGN_TOKENS))
    say("  %-36s %5s %6s %7s %8s %8s %6s" % ("配置", "k̄", "ΔP", "表MDE80", "自身MDE80", "两段同号", "过2g"))
    cands = []
    for fam, sub in ok.groupby("fam"):
        for _, x in sub.sort_values("dP", ascending=False).head(3).iterrows():
            pass_g = bool(100 * x["dP"] >= x["mde_tab"])
            say("  %-36s %5.2f %+5.1fpp %7.1f %8.1f %8s %6s   [%s]" % (
                x["name"], x["vbar"], 100 * x["dP"], x["mde_tab"], x["mde_own"], "是" if x["same_sign"] else "否", "过" if pass_g else "不过", fam))
            if pass_g and x["same_sign"]:
                cands.append(x["name"])
    # 全网格里有没有任何一个过 2g 的
    ok["pass_g"] = 100 * ok["dP"] >= ok["mde_tab"]
    ok["pass_own"] = 100 * ok["dP"] >= ok["mde_own"]
    say("  全网格 %d 个过量门槛的配置：过表 MDE80 的 %d 个，过自身 se MDE80 的 %d 个（%s）" % (
        len(ok), int(ok["pass_g"].sum()), int(ok["pass_own"].sum()),
        ", ".join("%s %+.1f" % (n, 100 * d) for n, d in ok[ok["pass_own"]][["name", "dP"]].to_numpy()) or "无"))

    say("\n== 3 补充维度（粗分组，记 diag）：影子模型分数的日级 / 个股三分位 ==")
    t = C.load()
    segs = {s: t[t["seg"] == s].reset_index(drop=True) for s in ("screen", "dev")}
    for s in segs:
        d = segs[s]
        m10 = d[d["rank_ret20"] <= 10].groupby("date")["model_p"].mean()
        d["day_mp"] = d["date"].map(m10)
    row_specs = []
    for s in segs:
        d = segs[s]
        q = DG.day_tercile(d.assign(seg=s), "day_mp")
        pq = DG.tercile_day(d, "model_p", 50)
        row_specs.append((s, {"天:前10影子分均值 低": q == 0, "天:前10影子分均值 高": q == 2, "个股影子分 前50内高三分位": pq == 2, "个股影子分 低三分位": pq == 0}))
    names = list(row_specs[0][1])
    for nm in names:
        row = "%-30s" % nm
        for s, conds in row_specs:
            d = segs[s]
            B = C.Baseline(d)
            rk = d["rank_ret20"].to_numpy()
            for lo, hi in DG.BINS:
                m = (rk >= lo) & (rk <= hi)
                Pc, Pn, tt, Vc, Vn = DG.ratio_diff(d[m].reset_index(drop=True), conds[nm][m], B.dates)
                row += " | %4.1f/%4.1f(%+4.1f)" % (100 * Pc, 100 * Pn, tt) if np.isfinite(Pc) else " | n=%d" % Vc
        say(row)
    C.ledger_add({"kind": "diag", "dim": "model_p_level", "note": "日级前 10 影子分均值 / 个股影子分三分位，两段都看"})

    say("\n== 4 正式臂筛选（criteria 2f + 2g + 两段同号）==")
    if cands:
        say("  候选：%s" % cands)
    else:
        say("  没有配置同时满足「两段同号」和「筛选集 ΔP >= 表 MDE80」。按条文正式臂 0 个，不登记 evalkit 新族，开发集网格序列不看。")
    sel = {"exp": "precision_v1", "family": C.FAMILY, "arms": cands,
           "note": "按 criteria.txt 2f/2g 机械筛选；空列表 = 没有臂进开发集" }
    (C.HERE / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    C.ledger_add({"kind": "decision", "exp": "precision_v1", "arms": cands, "n_grid_screen": int(len(R)), "n_pass_floor": int(len(ok)),
                  "n_pass_mde": int(ok["pass_g"].sum()), "note": "正式臂筛选结果"})
    txt = "\n".join(L)
    (C.HERE / "results_extra.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("补充诊断 + 正式臂筛选：网格 %d 个、过量门槛 %d 个、过表 MDE80 %d 个、过自身 MDE80 %d 个；正式臂 %s" % (
        len(R), len(ok), int(ok["pass_g"].sum()), int(ok["pass_own"].sum()), cands or "0 个"))


if __name__ == "__main__":
    main()
