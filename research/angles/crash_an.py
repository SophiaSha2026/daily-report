"""
新臂「动量崩溃状态」（criteria 8c，Daniel & Moskowitz 2016）。

    PYTHONIOENCODING=utf-8 python research/angles/crash_an.py [--dev]   -> results_crash.txt

候选池等权指数 = 候选池行 ret1 的日均值累乘。崩溃反弹日 = 过去 252 个交易日收益 < 0 且过去 21 个交易日收益 > 0。
臂 C1 = 崩溃反弹日不出、其余出前 10；描述并列 504 日版、「其余出前 3」版。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_crash.txt"
DEVJ = A.HERE / "results_crash_dev.json"
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def rows_for(seg: str, bad_days: set, k: int) -> pd.DataFrame:
    t = A.PR.load(seg)
    t = t[(t["rank_ret20"] <= k) & ~t["date"].isin(bad_days)]
    return t[["date", "hit"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    t = A.G.load_table(["date", "ret1"])
    idx = t.groupby("date")["ret1"].mean().sort_index()
    del t
    lv = np.log1p(idx.clip(lower=-0.5)).cumsum()
    r252 = lv - lv.shift(252)
    r504 = lv - lv.shift(504)
    r21 = lv - lv.shift(21)
    crash = (r252 < 0) & (r21 > 0)
    crash504 = (r504 < 0) & (r21 > 0)
    say("动量崩溃状态（criteria 8c）：候选池等权指数 %s..%s；252 日版可判起 %s，504 日版可判起 %s" % (
        idx.index.min(), idx.index.max(), r252.dropna().index.min(), r504.dropna().index.min()))
    for seg in ("screen", "dev"):
        lo, hi = A.SEGS[seg][0] + "-01", A.SEGS[seg][1] + "-31"
        m = (crash.index >= lo) & (crash.index <= hi)
        say("   %s：崩溃反弹日 %d / %d（252 日版）；504 日版 %d（可判 %d 天）" % (
            seg, int(crash[m].sum()), int(m.sum()), int(crash504[m].sum()), int(r504[m].notna().sum())))
    bad = set(crash[crash].index)
    bad504 = set(crash504[crash504].index)
    # 筛选集：崩溃日对非崩溃日的 P10 / P3
    B = A.G.baseline("screen")
    d = pd.DataFrame({"date": B.dates, "h10": B.hits_k(10) / 10, "h3": B.hits_k(3) / 3, "h1": B.hits_k(1)})
    d["crash"] = d["date"].isin(bad)
    g = d.groupby("crash")[["h1", "h3", "h10"]].mean()
    say("   筛选集 崩溃反弹日对其余：%s" % {str(k): {c: round(100 * float(v), 1) for c, v in r.items()} for k, r in g.iterrows()})
    arms = {"C1": ("崩溃反弹日不出，其余出前 10", bad, 10), "C1_k3": ("同上，其余出前 3（描述）", bad, 3),
            "C1_504": ("504 日版，其余出前 10（描述）", bad504, 10)}
    res = {}
    for key, (desc, bd, k) in arms.items():
        es = {s: A.eval_rows(rows_for(s, bd, k), s) for s in ("screen", "dev")}
        say("   " + A.fmt("%s %s" % (key, desc), es["screen"], "screen"))
        res[key] = {s: {kk: v for kk, v in es[s].items() if kk not in ("u", "h", "n", "dates")} for s in es}
        res[key]["u_screen"] = es["screen"]["u"].tolist()
        res[key]["u_dev"] = es["dev"]["u"].tolist()
        if a.dev:
            say("   " + A.fmt("%s %s" % (key, desc), es["dev"], "dev"))
            j = A.combined_judge(es["screen"], es["dev"])
            say("      合并 ΔP %+.1fpp NW t %+.2f 块自举 p %.3f 去%s t %.2f -> %s" % (
                100 * j["dP"], j["t"], j["boot_p"], j["best_month"], j["t_drop_best"], A.usable(es["screen"], es["dev"], j)))
    if a.dev:
        B = A.G.baseline("dev")
        d = pd.DataFrame({"date": B.dates, "h10": B.hits_k(10) / 10, "h3": B.hits_k(3) / 3, "h1": B.hits_k(1)})
        d["crash"] = d["date"].isin(bad)
        g = d.groupby("crash")[["h1", "h3", "h10"]].mean()
        say("   开发集 崩溃反弹日对其余：%s" % {str(k): {c: round(100 * float(v), 1) for c, v in r.items()} for k, r in g.iterrows()})
        A.ledger_add({"kind": "arm", "name": "crash_C1", "note": "开发集一次读（C1 + 两个描述版）"})
        OUT.write_text(OUT.read_text(encoding="utf-8") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    else:
        DEVJ.write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
        A.ledger_add({"kind": "screen", "name": "crash_C1", "note": "筛选集；开发集落盘未读"})
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.progress("crash_an%s -> results_crash.txt" % (" --dev" if a.dev else ""))


if __name__ == "__main__":
    main()
