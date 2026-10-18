"""名次曲线本身的统计（基线，不是臂）：B(k) 对 B(10) 的精确率差（影响函数 NW t）、逐年、逐月赢面、零命中月。

    PYTHONIOENCODING=utf-8 python research/precision/curve.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pr as C


def ratio_t(h1, n1, h2, n2) -> tuple[float, float]:
    D = len(h1)
    P1, P2 = h1.sum() / n1.sum(), h2.sum() / n2.sum()
    IF = D * (h1 - P1 * n1) / n1.sum() - D * (h2 - P2 * n2) / n2.sum()
    se = C.nw_se(IF)
    return float(P1 - P2), float((P1 - P2) / se) if se > 0 else np.nan


def main():
    t = C.load()
    L = ["名次曲线 B(k) 的统计（两段 + 合并；基线本身，不是臂）"]
    segs = {s: t[t["seg"] == s].reset_index(drop=True) for s in ("screen", "dev")}
    Bs = {s: C.Baseline(segs[s]) for s in segs}
    L.append("  B(k) − B(10) 精确率差（pp）和 NW t：")
    for k in (1, 2, 3, 5):
        row = "    k=%2d" % k
        hh, nn, h10, n10 = [], [], [], []
        for s in segs:
            B = Bs[s]
            h1, n1 = B.hits_k(k), np.full(B.D, float(k))
            h2, n2 = B.hits_k(10), np.full(B.D, 10.0)
            d, tt = ratio_t(h1, n1, h2, n2)
            row += "  %s %+.1fpp (t %+.2f)" % (s, 100 * d, tt)
            hh.append(h1); nn.append(n1); h10.append(h2); n10.append(n2)
        d, tt = ratio_t(np.concatenate(hh), np.concatenate(nn), np.concatenate(h10), np.concatenate(n10))
        row += "  合并 %+.1fpp (t %+.2f)" % (100 * d, tt)
        L.append(row)
    L.append("  逐年精确率%（k=1 / 2 / 3 / 5 / 10）和当年候选池基准：")
    pool = C.P.load_pool(columns=["date", "y_open", "pool"])
    pool = pool[pool["pool"] & (pool["date"] >= "2022-01-01")]
    base_y = pool.groupby(pool["date"].str[:4])["y_open"].mean()
    for s in segs:
        B = Bs[s]
        yrs = pd.Series([d[:4] for d in B.dates])
        for y in sorted(yrs.unique()):
            m = (yrs == y).to_numpy()
            vals = ["%5.1f" % (100 * B.hits_k(k)[m].sum() / (k * m.sum())) for k in (1, 2, 3, 5, 10)]
            L.append("    %s %s（%3d 天）: %s   基准 %.2f" % (s, y, m.sum(), " / ".join(vals), 100 * base_y.get(y, np.nan)))
    L.append("  逐月：B(2) 对 B(10) 赢的月数、B(2) 零命中月、B(10) 零命中月：")
    for s in segs:
        B = Bs[s]
        mo = pd.Series([d[:7] for d in B.dates])
        p2 = pd.Series(B.hits_k(2)).groupby(mo).sum() / (2 * mo.value_counts().sort_index())
        p10 = pd.Series(B.hits_k(10)).groupby(mo).sum() / (10 * mo.value_counts().sort_index())
        L.append("    %s：%d 个月，B(2) 赢 %d 平 %d 输 %d；B(2) 零命中月 %d，B(10) 零命中月 %d；B(2) 月精确率中位 %.1f%% 最差 %.1f%%" % (
            s, len(p2), int((p2 > p10).sum()), int((p2 == p10).sum()), int((p2 < p10).sum()), int((p2 == 0).sum()), int((p10 == 0).sum()),
            100 * p2.median(), 100 * p2.min()))
    L.append("  B(k) 零命中天占比（出 k 只全不中）：")
    for s in segs:
        B = Bs[s]
        L.append("    %s：%s" % (s, "  ".join("k=%d %.0f%%" % (k, 100 * (B.hits_k(k) == 0).mean()) for k in (1, 2, 3, 5, 10))))
    txt = "\n".join(L)
    C.say(txt)
    (C.HERE / "results_curve.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("名次曲线统计 results_curve.txt（B(k)−B(10) 的 t、逐年、逐月）")


if __name__ == "__main__":
    main()
