"""
角度三：事件触发的精确率（criteria 第 3 节）。六个事件全部报、不挑、不调。

    PYTHONIOENCODING=utf-8 python research/angles/events_an.py   -> results_events.txt
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_events.txt"
KINDS = [("E1", "长期调整突破 清单 A（find_events 默认）"), ("E1L", "同上 trigger.big=loose"), ("E2", "清单 B 进榜日"),
         ("E3", "长期横盘后首个大阳线（首阳单独成立）"), ("E4", "连板后首次缩量分歧转一致"), ("E5", "突破首板（涨停 + 60 日新高 + 3 倍量）"),
         ("E6a", "2 连板"), ("E6b", ">= 3 连板")]
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def shift_dates(dates: pd.Series, k: int, cal: np.ndarray) -> pd.Series:
    idx = pd.Series(np.arange(len(cal)), index=cal)
    i = dates.map(idx).to_numpy()
    j = i + k
    ok = np.isfinite(i) & (j < len(cal))
    out = np.full(len(dates), None, dtype=object)
    out[ok] = cal[j[ok].astype(int)]
    return pd.Series(out, index=dates.index)


def main():
    ev = pd.read_parquet(A.EVENTS)
    lab = A.load_labels()
    paths = pd.read_parquet(A.PATHS, columns=["code", "date", "g1", "max20", "c20", "d50"])
    t = ev.merge(lab, on=["code", "date"], how="left").merge(paths, on=["code", "date"], how="left")
    t["hit"] = t["y_open"].astype(float)
    t["seg"] = np.where(t["date"] < "2022-01-01", "2021", A.seg_of(t["date"]))
    cal = np.array(sorted(pd.read_parquet(A.DAY_IND, columns=["date"])["date"]))
    say("角度三：事件触发（px_screen 2020-07..2025-12，事件日 T 的 y_open = 次日开盘起 20 根内最高价涨 50%，一字板记不中）")
    say("尺子：P [块自举]、B(k̄) 同量名次基线、ΔP (NW t)、日配、日基准（同日候选池基准率）；B10 = 该段现行前 10 的精确率")
    say("「全部」= 有标签的全部事件行（含 ST / 新股 / 减持等候选池外的票）；「∩池」= 候选池行（主结论）")
    say()
    rec = json.loads((A.HERE / "e1_reconcile.json").read_text(encoding="utf-8"))
    say("E1 对账（2023-08..2025-12，同一函数）：px_screen %d 次，daily.parquet %d 次，重合 %d" % (
        rec["px_screen_2023-08.."], rec["daily_2023-08.."], rec["both"]))
    say()
    summary = []
    for kind, desc in KINDS:
        e = t[t["kind"] == kind]
        nl = int(np.isfinite(e["hit"]).sum())
        say("== %s %s：事件 %d 行，有标签 %d，∩池 %d；2021 年 %d 行" % (kind, desc, len(e), nl, int((e["pool"] == True).sum()),
                                                            int((e["seg"] == "2021").sum())))
        e = e[np.isfinite(e["hit"])]
        for vname, sub in (("全部", e), ("∩池", e[e["pool"] == True])):
            es = {}
            for seg in ("screen", "dev"):
                es[seg] = A.eval_rows(sub, seg)
                say("   " + A.fmt("%s %s" % (kind, vname), es[seg], seg))
            s21 = sub[sub["seg"] == "2021"]
            if len(s21):
                say("   %-30s 2021   V %5d  P %5.1f%%（两段之外，只描述）" % ("%s %s" % (kind, vname), len(s21), 100 * s21["hit"].mean()))
            if es["screen"]["V"] > 0 and es["dev"]["V"] > 0:
                j = A.combined_judge(es["screen"], es["dev"])
                u = A.usable(es["screen"], es["dev"], j)
                say("   合并 ΔP %+.1fpp NW t %+.2f 块自举 p %.3f 去%s t %.2f 滚3 %+.1f -> %s" % (
                    100 * j["dP"], j["t"], j["boot_p"], j["best_month"], j["t_drop_best"], 100 * j["roll3"][0], u))
                summary.append((kind, vname, es, j, u))
            if vname == "∩池" and len(sub):
                g = sub.groupby("seg")
                say("   次日缺口 g1 中位 %s；一字板占比 %s；max20 中位 %s" % (
                    {k: round(100 * float(v), 1) for k, v in g["g1"].median().items()},
                    {k: round(100 * float(v), 1) for k, v in g["yizi1"].mean().items()},
                    {k: round(100 * float(v), 1) for k, v in g["max20"].median().items()}))
        say()
    # 3d 自检：事件日平移 +30 个交易日
    say("== 自检（3d）：事件日整体平移 +30 个交易日后 ∩池 的精确率（应回到日基准附近）")
    for kind in ("E1L", "E3", "E5", "E6a"):
        e = t[(t["kind"] == kind)][["code", "date"]].copy()
        e["date"] = shift_dates(e["date"], 30, cal)
        e = e.dropna(subset=["date"]).merge(lab, on=["code", "date"], how="left")
        e = e[(e["pool"] == True) & np.isfinite(e["y_open"])]
        e["hit"] = e["y_open"].astype(float)
        for seg in ("screen", "dev"):
            es = A.eval_rows(e, seg, boot=False)
            if es["V"] > 0:
                say("   %-6s %-6s V %5d  P %5.1f%% (NW se %.1f)  日基准 %4.1f%%  B(k̄) %4.1f%%" % (
                    kind, seg, es["V"], 100 * es["P"], 100 * es["P_se"], 100 * es["P_daybase"], 100 * es["P_base"]))
    say()
    say("== 汇总（∩池）")
    say("%-5s %-28s %-30s %-30s %s" % ("事件", "筛选 V / P [区间] / B(k̄) / ΔP", "开发 V / P [区间] / B(k̄) / ΔP", "合并 t / p", "判定"))
    for kind, vname, es, j, u in summary:
        if vname != "∩池":
            continue
        s, d = es["screen"], es["dev"]
        say("%-5s %4d / %4.1f [%4.1f,%4.1f] / %4.1f / %+5.1f   %4d / %4.1f [%4.1f,%4.1f] / %4.1f / %+5.1f   %+5.2f / %.3f   %s" % (
            kind, s["V"], 100 * s["P"], 100 * s.get("P_lo", np.nan), 100 * s.get("P_hi", np.nan), 100 * s["P_base"], 100 * s["dP"],
            d["V"], 100 * d["P"], 100 * d.get("P_lo", np.nan), 100 * d.get("P_hi", np.nan), 100 * d["P_base"], 100 * d["dP"],
            j["t"], j["boot_p"], u))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.ledger_add({"kind": "diag", "name": "events", "note": "六个事件两段都报（描述，不挑不调，criteria 3c）", "kinds": [k for k, _ in KINDS]})
    A.progress("events_an：六个事件两段精确率 -> results_events.txt")


if __name__ == "__main__":
    main()
