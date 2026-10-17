"""
嵌套逐月滚动选配置：每个月只用之前 12 个月（trailing）的按天配对成绩在规则网格里选一个配置，用在这个月。
两段都诚实（没有用本月及之后的信息选），作为「网格在筛选集上调、开发集一次看」的并列读法。

    PYTHONIOENCODING=utf-8 python research/polish/nested.py --fam all --win 12 --seg screen   只印筛选集部分
    PYTHONIOENCODING=utf-8 python research/polish/nested.py --fam all --win 12 --seg both     登记后再跑

选择准则（看结果前定）：trailing 窗口里对 base 的按天配对 NW t 最大者；最大者 t <= 0 则用 base。
输出命中序列列名 nested_<fam>_<win> 并入 rules_hits10.parquet。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_pl as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default="all")
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--seg", default="screen")
    ap.add_argument("--src", default="rules")
    a = ap.parse_args()
    H = pd.read_parquet(P.PL / ("rules_hits10.parquet" if a.src == "rules" else "rr_hits10.parquet"))
    H30 = pd.read_parquet(P.PL / ("rules_hits30.parquet" if a.src == "rules" else "rr_hits30.parquet"))
    base = pd.read_parquet(P.PL / "rules_hits10.parquet")["base_ret20"]
    base30 = pd.read_parquet(P.PL / "rules_hits30.parquet")["base_ret20"]
    if a.src == "rules":
        grid = json.load((P.PL / "rules_grid.json").open(encoding="utf-8"))
        cols = [c for c in H.columns if a.fam == "all" or grid[c]["fam"] == a.fam]
    else:
        cols = list(H.columns)
    if "base_ret20" not in cols and a.src == "rules":
        cols.append("base_ret20")
    mon = H.index.str[:7]
    months = sorted(set(mon))
    out, out30, choice = {}, {}, {}
    for i, m in enumerate(months):
        if m < "2023-01":
            continue
        lo = months[max(i - a.win, 0)]
        tr = (mon >= lo) & (mon < m)
        best, bt = "base_ret20", 0.0
        for c in cols:
            if c == "base_ret20":
                continue
            d = (H.loc[tr, c] - base[tr]).dropna()
            if len(d) < 40:
                continue
            t = P.nw_t(d)
            if np.isfinite(t) and t > bt:
                best, bt = c, t
        choice[m] = (best, round(bt, 2))
        sel = mon == m
        out.update(H.loc[sel, best].to_dict())
        out30.update(H30.loc[sel, best].to_dict())
    name = "nested_%s_%d%s" % (a.fam, a.win, "" if a.src == "rules" else "_rr")
    s = pd.Series(out).sort_index()
    s30 = pd.Series(out30).sort_index()
    if a.seg == "screen":
        s = s[s.index < "2024-06-01"]
        s30 = s30[s30.index < "2024-06-01"]
        d = (s - base).dropna()
        d30 = (s30 - base30).dropna()
        P.say("%s 筛选集部分（%s..，%d 天）：命中 %.2f%% 对 base %.2f%%，差 %+.2fpp NW t %.2f；前 30 差 %+.2fpp" % (
            name, s.index.min(), len(d), 100 * s.mean(), 100 * base[d.index].mean(), 100 * d.mean(), P.nw_t(d), 100 * d30.mean()))
        P.say("  逐月选择：%s" % {k: v for k, v in choice.items() if k < "2024-06"})
    else:
        for fn, ser in (("rules_hits10.parquet", s), ("rules_hits30.parquet", s30)):
            f = P.PL / fn
            T = pd.read_parquet(f)
            T[name] = ser.reindex(T.index)
            T.to_parquet(f)
        d = (s - base).dropna()
        d30 = (s30 - base30).dropna()
        j = P.judge(d, d30, name)
        P.say(P.fmt_judge(j))
        P.say("  逐月选择：%s" % choice)
        (P.PL / f"{name}_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
