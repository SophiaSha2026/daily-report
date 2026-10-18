"""诊断（不是门槛）：各分数每月前 N 行的精确率，看分数的顶部在两段里有没有 20% 以上的区域。

    PYTHONIOENCODING=utf-8 python research/gate/diag_top.py lgb_price_top300 lgb_alt rank ret20 ...
按段池化：每月取分数最高的 N 行（N = 10 / 20 / 40 / 80），P = Σ命中/ΣN，和同量名次基线 B(k̄) 比。
这是「事后按月配额」的乐观读法，只用来判断方向。
"""
import sys

import numpy as np
import pandas as pd

import common_gt as G


def main():
    names = sys.argv[1:]
    t = G.load_table(["date", "hit", "seg", "rank_ret20", "ret20", "r20_open", "end20", "y_open"])
    ym = t["date"].to_numpy().astype("U7")
    out = []
    for nm in names:
        if nm == "rank":
            s = (-t["rank_ret20"].to_numpy()).astype(np.float32)
        elif nm == "ret20":
            s = t["ret20"].to_numpy(np.float32).copy()
        else:
            s = np.load(G.SCORES / (nm + ".npy"))
        s = np.where(np.isfinite(s), s, -np.inf)
        for seg in ("screen", "dev"):
            months = G.months_between(*G.SEGS[seg])
            for N in (5, 10, 20, 40, 80):
                sel = np.zeros(len(t), bool)
                for m in months:
                    mi = np.flatnonzero(ym == m)
                    o = mi[np.argsort(-s[mi], kind="mergesort")[:N]]
                    sel[o] = True
                e = G.eval_gate(t, sel, seg, boot=False)
                out.append((nm, seg, N, e["V"], e["days_out"], round(100 * e["P"], 1), round(100 * e["P_base"], 1), round(100 * e["dP"], 1), round(e["t"], 2)))
                G.say("%-22s %-6s 月前%-3d V %4d 天 %3d  P %5.1f%%  B(k̄) %5.1f%%  ΔP %+5.1f (t %+.2f)" % out[-1])
    pd.DataFrame(out, columns=["score", "seg", "N", "V", "days", "P", "B", "dP", "t"]).to_csv(G.HERE / "results_diag_top.csv", index=False)
    G.ledger_add({"kind": "diag", "name": "diag_top", "scores": names, "note": "每月前 N 行精确率，两段都看（方向诊断，不是门槛）"})


if __name__ == "__main__":
    main()
