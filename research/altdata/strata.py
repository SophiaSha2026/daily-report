"""按名次分层看龙虎榜 / 机构 / 户数标志（为什么前 50 内条件命中 +3pp 没有转成前 10 的边际）。
    PYTHONIOENCODING=utf-8 python research/altdata/strata.py  -> results_strata.txt
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
import common_pl as P

df = pd.read_parquet(HERE / "top60.parquet")
df["seg"] = np.where(df["date"] < "2024-06-01", "screen", "dev")
df["bucket"] = pd.cut(df["rank_ret20"], [0, 3, 10, 20, 30, 50, 60], labels=["1-3", "4-10", "11-20", "21-30", "31-50", "51-60"])
flags = {"lhb_today": df["lhb_today"] > 0, "lhb_n5": df["lhb_n5"] > 0, "lhb_net5pos": df["lhb_net5_share"] > 0,
         "org_net5pos": df["org_net5_share"] > 0, "gd_down": df["gd_chg1"] < 0, "dz_n20": df["dz_n20"] > 0}
L = ["== 按 ret20 名次分层：标志占比 和 命中（池化，括号里是行数）==", ""]
for name, fl in flags.items():
    df["_f"] = fl.to_numpy()
    for seg in ("screen", "dev"):
        x = df[df["seg"] == seg]
        g = x.groupby(["bucket", "_f"], observed=True)["y_open"].agg(["mean", "size"]).unstack("_f")
        share = x.groupby("bucket", observed=True)["_f"].mean()
        row = ["%s %-6s" % (name, seg)]
        for b in g.index:
            m0 = g.loc[b, ("mean", False)] if ("mean", False) in g.columns else np.nan
            m1 = g.loc[b, ("mean", True)] if ("mean", True) in g.columns else np.nan
            row.append("%s: 标志 %2.0f%% 命中 有 %5.2f 无 %5.2f" % (b, 100 * share[b], 100 * m1, 100 * m0))
        L.append(" | ".join(row))
    L.append("")
# 关键对比：11~20 名里带标志的 对 4~10 名里不带标志的（按天配对）
L.append("== 按天配对：11~20 名带标志 - 4~10 名不带标志（换票的净效果上界）==")
for name, fl in flags.items():
    df["_f"] = fl.to_numpy()
    a = df[(df["rank_ret20"].between(11, 20)) & df["_f"]].groupby("date")["y_open"].mean()
    b = df[(df["rank_ret20"].between(4, 10)) & ~df["_f"]].groupby("date")["y_open"].mean()
    d = (a - b).dropna()
    s, v = d[d.index < "2024-06-01"], d[d.index >= "2024-06-01"]
    L.append("%-12s 筛选 %+.2fpp (t %.2f, %d 天)  开发 %+.2f (t %.2f, %d 天)" % (
        name, 100 * s.mean(), P.nw_t(s), len(s), 100 * v.mean(), P.nw_t(v), len(v)))
txt = "\n".join(L)
(HERE / "results_strata.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)
