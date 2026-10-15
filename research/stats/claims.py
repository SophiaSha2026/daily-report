"""
复核已经进生产或进邮件的四条结论（只读开发集产物）：

    1 实验 15 按月等权 eqm（t=4.50）：前 10 两种口径 + 生产规则「合并」对「按天配对」
    2 同涨族群 g2（t=2.08 那条是 2025 起的子区间）
    3 HIGH_CONF（export.py）：98 分以上 15.3% / 连续够格 2 天 21.0% / 全部 12.7%
    4 STREAK_PERF（export.py，来自 wf_scores.parquet 的 W5 行）：各档对基准的提升

区间一律按天块自举（块长 20 个交易日，5000 次），点估计是合并命中率（和邮件印的数同一个估计量），
差值的区间也是在同一次重抽里算的（教训 33：分子分母同一条序列）。

    python research/stats/claims.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms as AR   # noqa: E402
import hac as H     # noqa: E402

OUT = HERE / "out"


def per_day(df: pd.DataFrame, dates: list[str], y: str = "y_open") -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("date")[y].agg(["sum", "size"]).reindex(dates).fillna(0)
    return g["sum"].to_numpy(float), g["size"].to_numpy(float)


def subset_vs(df_a: pd.DataFrame, df_b: pd.DataFrame | None, dates: list[str], y: str = "y_open",
              base: pd.Series | None = None) -> dict:
    ka, na = per_day(df_a, dates, y)
    out = {"n": int(na.sum()), "hit": float(ka.sum() / max(na.sum(), 1)), "days_with": int((na > 0).sum())}
    # 命中率本身的区间
    bb = H.pooled_boot(ka, na)
    out.update(hit_lo=bb["lo"], hit_hi=bb["hi"], hit_se_boot=bb["se"])
    # 按天聚类 se（evalkit.clustered 同式）
    p = out["hit"]
    out["se_clustered"] = float(np.sqrt(((ka - p * na) ** 2).sum()) / max(na.sum(), 1))
    out["se_iid"] = float(np.sqrt(p * (1 - p) / max(na.sum(), 1)))
    if df_b is not None:
        kb, nb = per_day(df_b, dates, y)
        d = H.pooled_boot(ka, na, kb, nb)
        out.update(vs_est=d["est"], vs_lo=d["lo"], vs_hi=d["hi"], vs_p=d["p"], vs_t=d["t"],
                   other_hit=float(kb.sum() / max(nb.sum(), 1)), other_n=int(nb.sum()))
        # 按天配对（两边当天都有名额）的 NW20
        both = (na > 0) & (nb > 0)
        dd = ka[both] / na[both] - kb[both] / nb[both]
        out.update(paired_days=int(both.sum()), paired_mean=float(dd.mean()) if len(dd) else np.nan,
                   paired_iid_t=H.iid_t(dd)["t"], paired_nw20_t=H.nw_t(dd, 20)["t"],
                   paired_mbb_p=H.mbb(dd)["p"])
    if base is not None:
        r = base.reindex(dates).to_numpy(float)
        d = H.pooled_boot(ka, na, ref=r)
        b_w = float((r * na).sum() / max(na.sum(), 1))
        out.update(base_w=b_w, lift=out["hit"] / b_w, excess=d["est"], ex_lo=d["lo"], ex_hi=d["hi"], ex_p=d["p"],
                   lift_lo=(d["lo"] + b_w) / b_w, lift_hi=(d["hi"] + b_w) / b_w)
    return out


def streak_of(ok: pd.DataFrame, all_dates: list[str]) -> np.ndarray:
    di = {d: i for i, d in enumerate(all_dates)}
    idx = ok["date"].map(di).to_numpy()
    sets = {c: set(ok.loc[g, "date"].map(di)) for c, g in ok.groupby("code").groups.items()}
    out = np.zeros(len(ok), int)
    for j, (c, i) in enumerate(zip(ok["code"].to_numpy(), idx)):
        s = 1
        while (i - s) in sets[c]:
            s += 1
        out[j] = s
    return out


# ---------------------------------------------------------------
def eqm() -> dict:
    row = next(r for r in AR.REGISTRY if r["arm"] == "eqm")
    a, b = AR.top_of(row, "eqm"), AR.top_of(row, "base")
    dates = sorted(a["date"].unique())
    res = {}
    for y in ("y_up", "y_open"):
        pa, pb = a[(a["score"] >= 97) & (a["rank"] <= 10)], b[(b["score"] >= 97) & (b["rank"] <= 10)]
        res[f"prod_{y}"] = subset_vs(pa, pb, dates, y)
        res[f"top10_{y}"] = subset_vs(a[a["rank"] <= 10], b[b["rank"] <= 10], dates, y)
        # 去掉最好的月
        d = (AR.daily_hit(a, 10, y) - AR.daily_hit(b, 10, y)).dropna()
        md = d.groupby(d.index.str[:7]).mean()
        best = md.idxmax()
        dd = d[d.index.str[:7] != best]
        res[f"top10_{y}_drop_{best}"] = {"iid_t": H.iid_t(dd)["t"], "nw20_t": H.nw_t(dd, 20)["t"],
                                         "mbb_p": H.mbb(dd)["p"], "mean": float(dd.mean())}
        res[f"top10_{y}_monthly"] = {k: float(v) for k, v in md.items()}
    res["prod_empty"] = {"eqm": len(dates) - a[(a["score"] >= 97) & (a["rank"] <= 10)]["date"].nunique(),
                         "base": len(dates) - b[(b["score"] >= 97) & (b["rank"] <= 10)]["date"].nunique()}
    return res


def high_conf() -> dict:
    t = AR.w_top("base")      # 实验 20 / HIGH_CONF 的来源：w1_scale_avg/top_base（开发集 387 天）
    dates = sorted(t["date"].unique())
    base = AR.market_base_open()
    res = {}
    top10 = t[t["rank"] <= 10]
    top50 = t[t["rank"] <= 50]
    A = t[(t["score"] >= 97) & (t["rank"] <= 10)].copy()
    A["streak"] = streak_of(A, dates)
    # 日志 / export.py 里写的三个数，原样复现
    res["as_logged_score98_in_top50"] = subset_vs(top50[top50["score"] == 98], None, dates, base=base)
    res["as_logged_streak2_exact"] = subset_vs(A[A["streak"] == 2], None, dates, base=base)
    res["as_logged_all_top10"] = subset_vs(top10, None, dates, base=base)
    # 邮件实际说的：清单 A 里「98 分以上」「连续够格 2 天以上」，该对比的是清单 A 本身和它的补集
    res["listA_all"] = subset_vs(A, None, dates, base=base)
    for name, m in (("listA_score98plus", A["score"] >= 98), ("listA_streak2plus", A["streak"] >= 2),
                    ("listA_highconf_union", (A["score"] >= 98) | (A["streak"] >= 2))):
        res[name] = subset_vs(A[m], A[~m], dates, base=base)
    res["top50_score98plus_vs_97"] = subset_vs(top50[top50["score"] >= 98], top50[top50["score"] == 97], dates)
    for s in (1, 2, 3, 4):
        sub = A[A["streak"] == s] if s < 4 else A[A["streak"] >= 4]
        res[f"listA_streak_{s}{'+' if s == 4 else ''}"] = subset_vs(sub, None, dates, base=base)
    # 2025 起（开发集里训练数据充足的那段）
    d25 = [d for d in dates if d >= AR.SINCE25]
    A25 = A[A["date"] >= AR.SINCE25]
    for name, m in (("since25_listA_score98plus", A25["score"] >= 98), ("since25_listA_streak2plus", A25["streak"] >= 2)):
        res[name] = subset_vs(A25[m], A25[~m], d25, base=base)
    return res


def streak_perf() -> dict:
    """STREAK_PERF 的来源：data/breakout/raw/wf_scores.parquet（旧验证集 2025-03..12，生产规则 W5）。"""
    d = AR.read(AR.RAW / "wf_scores.parquet")
    d = d[~d["code"].isin(AR.st_codes())]
    d = d.sort_values(["date", "_p"], ascending=[True, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    dates = sorted(d["date"].unique())
    ok = d[(d["score"] >= 97) & (d["rank"] <= 10)].copy()
    ok["streak"] = streak_of(ok, dates)
    base = AR.market_base_open()
    wf = json.loads((AR.RAW / "wf_base.json").read_text(encoding="utf-8"))
    res = {"base_open_logged": wf["base_open"], "days": len(dates)}
    for k in (1, 2, 3, 4, 5):
        sub = ok[ok["streak"] >= k]
        rest = ok[ok["streak"] == 1] if k > 1 else None
        r = subset_vs(sub, rest, dates, base=base)
        r["lift_logged_base"] = r["hit"] / wf["base_open"]
        res[f"ge{k}"] = r
    # 满员日 / 非满员日
    full = ok.groupby("date").size()
    fd = set(full[full >= 10].index)
    res["cap_days"] = subset_vs(ok[ok["date"].isin(fd)], ok[~ok["date"].isin(fd)], dates, base=base)
    return res


def main() -> int:
    OUT.mkdir(exist_ok=True)
    res = {"eqm": eqm(), "high_conf": high_conf(), "streak_perf": streak_perf()}
    (OUT / "claims.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    def pr(name, r):
        s = "  %-34s n=%5d 命中 %5.2f%% [%5.2f, %5.2f]  se 二项 %.2f 聚类 %.2f 自举 %.2f" % (
            name, r["n"], 100 * r["hit"], 100 * r["hit_lo"], 100 * r["hit_hi"], 100 * r["se_iid"],
            100 * r["se_clustered"], 100 * r["hit_se_boot"])
        if "lift" in r:
            s += " | 倍数 %.1f [%.1f, %.1f]" % (r["lift"], r["lift_lo"], r["lift_hi"])
        if "vs_est" in r:
            s += " | 对补集 %+.2f [%+.2f, %+.2f] p=%.3f（补集 %.2f%%）配对 %d 天 NW20 t=%.2f" % (
                100 * r["vs_est"], 100 * r["vs_lo"], 100 * r["vs_hi"], r["vs_p"], 100 * r["other_hit"],
                r["paired_days"], r["paired_nw20_t"])
        print(s)

    print("== eqm")
    for k in ("top10_y_up", "top10_y_open", "prod_y_up", "prod_y_open"):
        pr(k, res["eqm"][k])
    for k, v in res["eqm"].items():
        if "drop" in k:
            print("  ", k, {a: round(b, 3) for a, b in v.items()})
    print("  空榜", res["eqm"]["prod_empty"])
    print("== HIGH_CONF")
    for k, v in res["high_conf"].items():
        pr(k, v)
    print("== STREAK_PERF")
    for k, v in res["streak_perf"].items():
        if isinstance(v, dict):
            pr(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
