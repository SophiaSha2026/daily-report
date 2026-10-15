"""
新过线规则（提案，report.md 第 5 节）的参考实现，拿历史上每个臂回放一遍，看哪些结论要改口。

    规则 v2（全部满足才「过线」，NW t <= -门槛 记「不过线」，其余「未定」）
      1 NW t（Bartlett，滞后 20 = 标签窗口）>= 族门槛 c
        c = max(2.5, 族内最大 t 零分布 95% 分位)；拿不到逐日序列时退回单侧 Bonferroni Φ⁻¹(1 - 0.05/m)，m = 登记簿里该族的试验数
      2 块自举 p（块长 20，5000 次）< 0.05
      3 去掉最好的一个月后 NW t >= 2.0（防一个月说了算）
      4 合并命中之差和按天配对均值同号（教训 33）
      5 原有两条：最差三个月不比现状差、空榜不翻倍（这里不重算，沿用当时记录）

    python research/stats/gate_v2.py     读 out/arms.csv + out/multi.json，写 out/gate_v2.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms as AR   # noqa: E402
import hac as H     # noqa: E402

OUT = HERE / "out"
N01 = NormalDist()


def drop_best_month_t(row: dict) -> float:
    a = AR.span_filter(AR.top_of(row, row["arm"]), row["span"])
    b = AR.span_filter(AR.top_of(row, row["ctrl"]), row["span"])
    y = row["y"]
    d = (AR.daily_hit(a, row["k"], y) - AR.daily_hit(b, row["k"], y)).dropna()
    md = d.groupby(d.index.str[:7]).mean()
    dd = d[d.index.str[:7] != md.idxmax()] if md.mean() > 0 else d[d.index.str[:7] != md.idxmin()]
    return H.nw_t(dd.to_numpy(), 20)["t"]


def pooled_sign_ok(row: dict) -> bool:
    a = AR.span_filter(AR.top_of(row, row["arm"]), row["span"])
    b = AR.span_filter(AR.top_of(row, row["ctrl"]), row["span"])
    y, k = row["y"], row["k"]
    pa = a[(a["score"] >= 97) & (a["rank"] <= 10)] if k == "prod" else a[a["rank"] <= k]
    pb = b[(b["score"] >= 97) & (b["rank"] <= 10)] if k == "prod" else b[b["rank"] <= k]
    pooled = pa[y].mean() - pb[y].mean()
    paired = (AR.daily_hit(a, k, y) - AR.daily_hit(b, k, y)).dropna().mean()
    return bool(np.sign(pooled) == np.sign(paired))


def main() -> int:
    arms = pd.read_csv(OUT / "arms.csv")
    multi = json.loads((OUT / "multi.json").read_text(encoding="utf-8"))
    mx = multi["maxt"]
    m_all = multi["trials"]["total_model_arms"]
    # 三种族口径的门槛：实验自己那一族 / 这个开发集上同一目标的全部臂（逐日序列可得的） / 登记簿全部（Bonferroni）
    fam_c = {"own": {15: max(2.5, mx["A15_yup"]["q95_one"]), 16: max(2.5, mx["A16"]["q95_one"]),
                     17: max(2.5, mx["B_dev387"]["q95_one"]), 18: max(2.5, mx["B_dev387"]["q95_one"]),
                     21: max(2.5, mx["BP_dev387"]["q95_one"]), 19: 2.5},
             "all_series": max(2.5, mx["AB_val207"]["q95_one"], mx["BP_dev387"]["q95_one"]),
             "bonf_ledger": N01.inv_cdf(1 - 0.05 / multi["trials"]["total_incl_diag"])}
    rows = []
    for r in AR.REGISTRY:
        hit = arms[(arms["arm"] == r["arm"]) & (arms["k"].astype(str) == str(r["k"])) & (arms["y"] == r["y"])
                   & (arms["span"] == r["span"]) & (arms["exp"] == r["exp"])].iloc[0]
        t, p = hit["nw20_t"], hit["mbb_p"]
        tdrop = drop_best_month_t(r)
        sign_ok = pooled_sign_ok(r)
        out = {"exp": r["exp"], "arm": r["arm"], "k": r["k"], "y": r["y"], "span": r["span"],
               "t_iid_old": r["t_rep"], "old_verdict_2.5": "过线" if r["t_rep"] >= 2.5 else ("不过线" if r["t_rep"] <= -2.5 else "未定"),
               "nw20_t": t, "mbb_p": p, "nw20_t_drop_best_month": tdrop, "pooled_paired_same_sign": sign_ok}
        for name in ("own", "all_series", "bonf_ledger"):
            c = fam_c["own"][r["exp"]] if name == "own" else fam_c[name]
            ok = (t >= c) and (p < 0.05) and (tdrop >= 2.0) and sign_ok
            out[f"c_{name}"] = c
            out[f"v2_{name}"] = "过线" if ok else ("不过线" if t <= -c else "未定")
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gate_v2.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("门槛：", {k: (v if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in fam_c.items()})
    print(df[["exp", "arm", "k", "y", "span", "t_iid_old", "old_verdict_2.5", "nw20_t", "mbb_p",
              "nw20_t_drop_best_month", "pooled_paired_same_sign", "v2_own", "v2_all_series", "v2_bonf_ledger"]]
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
