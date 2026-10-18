"""校准 + 门槛 + 评估（criteria.txt 第 1c、3 节）。

    PYTHONIOENCODING=utf-8 python research/gate/eval_gt.py --score lgb_price [--cal iso|cum] [--win 12] [--nmin 40]
        [--dev]  # 不给 --dev 只打印筛选集；开发集结果照算、落盘到 results/<名>.json 但不打印

--score ret20 / rank 不用 .npy：分数直接取 ret20 原值 / −名次。
校准 iso：过去 win 个月样本外行上的等渗回归 p̂(s)。cum：过去 win 个月里「分数 >= s 的行的累计精确率」，且该集合至少 nmin 行；
p̂(s) 取满足支持数的那条曲线。门槛 p̂ >= τ。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_gt as G

RES = G.HERE / "results"
RES.mkdir(exist_ok=True)


def cum_calibrate(scores: np.ndarray, t: pd.DataFrame, months: list[str], win: int, nmin: int, dmin: int = 0) -> np.ndarray:
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    y = t["y_open"].to_numpy(float)
    ym = dates.astype("U7")
    p = np.full(len(t), np.nan, np.float32)
    for m in months:
        lo = G.month_shift(m, -win) + "-01"
        cm = (dates >= lo) & (dates < m + "-01") & (end20 < m + "-01") & np.isfinite(scores) & np.isfinite(y)
        rows = ym == m
        if cm.sum() < 1000 or not rows.any():
            continue
        nmo = len(np.unique(ym[cm]))
        if nmo < G.CAL_MIN_MONTHS:
            continue
        sc = scores[cm].astype(float)
        yc = y[cm]
        o = np.argsort(-sc, kind="mergesort")
        sc, yc = sc[o], yc[o]
        cumh = np.cumsum(yc)
        k = np.arange(1, len(sc) + 1)
        prec = cumh / k
        prec[k < nmin] = 0.0                       # 支持数不够的不算
        if dmin > 0:                                # 支持的天数也要够（同一天的票不独立，教训 33）
            dd = dates[cm][o]
            _, first_idx = np.unique(dd, return_index=True)
            seen = np.zeros(len(dd), np.int64)
            seen[first_idx] = 1
            ndays = np.cumsum(seen)
            prec[ndays < dmin] = 0.0
        # 分数并列：同一分数的行必须整体进出 -> 取每个分数值最后一次出现处的累计精确率
        # p̂(s) = 「分数 >= s」集合的累计精确率；对测试行按分数查
        srt = sc[::-1]                              # 升序
        prec_r = prec[::-1]
        st = scores[rows].astype(float)
        pos = np.searchsorted(srt, st, side="left")  # 行数 = len(srt) - pos 是「>= s」的集合
        pos = np.clip(pos, 0, len(srt) - 1)
        p[rows] = prec_r[pos].astype(np.float32)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--cal", default="iso", choices=["iso", "cum", "none"])
    ap.add_argument("--win", type=int, default=G.CAL_MONTHS)
    ap.add_argument("--nmin", type=int, default=40)
    ap.add_argument("--dmin", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0, help="每天最多出几只（按 p̂ 取前 cap），0 不封顶")
    ap.add_argument("--uni", default="all", choices=["all", "top300", "top100", "top50"])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--look", action="store_true", help="开发集读数记 kind=look（筛选集 ΔP<0 没登记的臂，只作产品表参考）")
    ap.add_argument("--taus", default=",".join("%.2f" % x for x in G.TAUS))
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    t = G.load_table(["code", "date", "end20", "y_open", "hit", "r20_open", "seg", "rank_ret20", "ret20"])
    if a.score == "ret20":
        s = t["ret20"].to_numpy(np.float32).copy()
        s[t["date"].to_numpy() < "2021-07-01"] = np.nan
    elif a.score == "rank":
        s = (-t["rank_ret20"].to_numpy()).astype(np.float32)
        s[t["date"].to_numpy() < "2021-07-01"] = np.nan
    else:
        s = np.load(G.SCORES / (a.score + ".npy"))
    assert len(s) == len(t)
    months = G.months_between("2022-01", "2025-12")
    if a.cal == "iso":
        p = G.calibrated(s, t, months, cal_months=a.win)
    elif a.cal == "cum":
        p = cum_calibrate(s, t, months, a.win, a.nmin, a.dmin)
    else:
        p = s.astype(np.float32)
        p[t["date"].to_numpy() < "2022-01-01"] = np.nan
    if a.uni != "all":
        p[t["rank_ret20"].to_numpy() > int(a.uni[3:])] = np.nan
    t["p_hat"] = p
    name = "%s_%s%d%s" % (a.score, a.cal, a.win, ("_n%d" % a.nmin) if a.cal == "cum" else "") if a.cal != "none" else a.score
    if a.dmin:
        name += "_d%d" % a.dmin
    if a.uni != "all":
        name += "_" + a.uni
    if a.cap:
        name += "_cap%d" % a.cap + (("_" + a.tag) if a.tag else "")
    out = {"name": name, "args": vars(a), "rows": {}}
    lines = []
    cov = {seg: float(np.isfinite(p[(t["seg"] == seg).to_numpy()]).mean()) for seg in ("screen", "dev")}
    lines.append("== %s  p̂ 覆盖 筛选 %.3f 开发 %.3f ==" % (name, cov["screen"], cov["dev"]))
    for tau in taus:
        sel = np.isfinite(p) & (p >= tau)
        if a.cap and sel.any():
            q = t.loc[sel, ["date", "p_hat"]].copy()
            q["_r"] = q.groupby("date")["p_hat"].rank(method="first", ascending=False)
            drop = q.index[q["_r"] > a.cap]
            sel[drop] = False
        res = {}
        for seg in ("screen", "dev"):
            e = G.eval_gate(t, sel, seg, boot=True)
            res[seg] = e
            if seg == "screen" or a.dev:
                lines.append(G.fmt("τ=%.2f" % tau, e, seg))
        if res["screen"]["V"] > 0 and res["dev"]["V"] > 0:
            cj = G.combined_judge(res["screen"], res["dev"])
            if a.dev:
                lines.append("   合并 ΔP %+.1fpp NW t %.2f 块自举 p %.3f 去%s t %.2f 滚3 %+.1f@%s" % (
                    100 * cj["dP"], cj["t"], cj["boot_p"], cj["best_month"], cj["t_drop_best"], cj["roll3"][0], cj["roll3"][1]))
        else:
            cj = {}
        out["rows"]["%.2f" % tau] = {seg: {k: (v if not isinstance(v, (np.ndarray, pd.Series, pd.DataFrame)) else None)
                                           for k, v in res[seg].items() if k not in ("u", "h", "n", "dates", "months_out")}
                                     | {"months_out": res[seg]["months_out"].to_dict() if "months_out" in res[seg] else {}}
                                     for seg in ("screen", "dev")} | {"combined": cj}
        # 出票行落盘（开发集也落，不打印）
        rows = t.loc[sel, ["code", "date", "seg", "hit", "r20_open", "p_hat", "rank_ret20", "ret20"]]
        rows.to_parquet(G.HERE / ("rows_%s_%.2f.parquet" % (name, tau)), index=False)
    txt = "\n".join(lines)
    print(txt)
    (RES / (name + ".json")).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding="utf-8")
    with (G.HERE / ("results_%s.txt" % ("dev" if a.dev else "screen"))).open("a", encoding="utf-8") as f:
        f.write(txt + "\n")
    kind = ("look" if a.look else ("check" if a.tag == "check" else "arm")) if a.dev else "screen"
    G.ledger_add({"kind": kind, "name": name, "taus": taus,
                  "note": {"arm": "开发集一次读（预登记臂）", "look": "开发集读数：筛选集 ΔP<0 未登记，只作产品表参考",
                           "check": "打乱标签自检", "screen": "筛选集配置挑选"}[kind]})


if __name__ == "__main__":
    main()
