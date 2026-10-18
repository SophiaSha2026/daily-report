"""两融倾斜的补充网格（只在筛选集上看；开发集序列落盘到 hits10/hits30 不看）。
缺值（非两融标的）填中性 0.5 百分位，不再当成最低。权重 0.25 / 0.5 / 1.0，K 15 / 20 / 30，特征 rz_chg20 / rz_chg5 / 两者均值。
    PYTHONIOENCODING=utf-8 python research/altdata/grid_rz.py -> results_screen_rz.txt
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
import common_pl as P
sys.path.insert(0, str(HERE))   # research/polish 也有 diag.py，先找本目录
import importlib.util
_sp = importlib.util.spec_from_file_location("alt_diag", HERE / "diag.py"); D = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(D)

df = pd.read_parquet(HERE / "top60.parquet")
base = D.rank_rule(df, df["lret20"].to_numpy(), keep=30)
b10, b30 = D.top_hits(base, 10), D.top_hits(base, 30)
rk = df["rank_ret20"].to_numpy()
H10 = pd.read_parquet(HERE / "hits10.parquet"); H30 = pd.read_parquet(HERE / "hits30.parquet")
rows = []
df["rz_mix"] = (df["rz_chg20"].rank(pct=True) + df["rz_chg5"].rank(pct=True)) / 2
for feat in ("rz_chg20", "rz_chg5", "rz_mix"):
    for K in (15, 20, 30):
        for w in (0.25, 0.5, 1.0):
            v = pd.Series(np.where(rk <= K, df[feat].astype(float), np.nan), index=df.index)
            pr = v.groupby(df["date"]).rank(pct=True)
            pr = pr.where(np.isfinite(v), 0.5).fillna(0.5).to_numpy()
            score = np.where(rk <= K, -rk + w * K * pr, -1e6)
            name = "tn_%s_%d_w%g" % (feat, K, w)
            t = D.rank_rule(df, score, keep=30)
            h10, h30 = D.top_hits(t, 10), D.top_hits(t, 30)
            H10[name], H30[name] = h10, h30
            d = (h10 - b10).dropna(); a, _ = D.seg(d)
            ov = D.overlap(t[t["date"] < D.SCREEN_END], base[base["date"] < D.SCREEN_END])
            se969 = P.nw_se(a.to_numpy()) * np.sqrt(len(a) / 969)
            rows.append(dict(arm=name, screen_diff=float(a.mean()), screen_t=D.tstat(a), overlap=ov, se969=se969,
                             mde_self=3.34 * se969, mde_tab=D.mde_by_overlap(ov) / 100, hit=float(h10[h10.index < D.SCREEN_END].mean())))
H10.to_parquet(HERE / "hits10.parquet"); H30.to_parquet(HERE / "hits30.parquet")
res = pd.DataFrame(rows).sort_values("screen_diff", ascending=False)
L = ["== 两融倾斜补充网格（筛选集；缺值填中性）=="]
for r in res.itertuples():
    L.append("%-24s 筛选 %+.2fpp (t %5.2f) 命中 %.2f%% 重合 %.0f%%  se969 %.2f  MDE80 自 %.2f 表 %.2f  %s" % (
        r.arm, 100 * r.screen_diff, r.screen_t, 100 * r.hit, 100 * r.overlap, 100 * r.se969, 100 * r.mde_self, 100 * r.mde_tab,
        "候选" if r.screen_diff >= max(r.mde_self, r.mde_tab) else ""))
txt = "\n".join(L); (HERE / "results_screen_rz.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)
