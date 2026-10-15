"""
公告类别 x 买得到命中（问题 3、4、5）。读 cache/panel.parquet、cache/events.parquet，
写 cache/results.csv、cache/model_rel.csv。只用开发集 2024-06..2025-12。

股票-日 (code, t) 被某类 c 的窗口 k 标记：存在 c 类公告的可知日 a 满足 t-k <= a <= t（交易日）。
统计（教训 33）：
  每天 hit_d = 当天被标记票的 y_open 均值，base_d = 当天全市场 y_open 均值，
  diff_d = hit_d - base_d，只在有被标记票的天上取；
  点估计 = diff_d 的均值（同一条按天序列），se_day = sd/sqrt(D)，t_day；
  t_nw = 同一序列的 Newey-West t（Bartlett，滞后 20 个观测日），标签 20 根重叠所以要它；
  hit_day / base_day 是同一批天上的按天均值，倍数 = 两者之比；_pooled 后缀是池化值，另一个问题。
  召回 = 命中里被标记的比例；覆盖 = 全部股票-日里被标记的比例。
买不买得到：
  yizi1 = 次日一字（y_open 记 0）的比例；hit_raw = 不管一字、r20_open > 50% 的比例（池化）；
  hit_ex = 去掉 t 当天一字或次日一字的行之后的按天命中均值。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
KS = (0, 1, 3, 5)
RULES = ("main", "ei")


def nw_t(x: np.ndarray, lag: int = 20) -> float:
    x = np.asarray(x, float)
    T = len(x)
    if T < 3:
        return float("nan")
    e = x - x.mean()
    L = min(lag, T - 1)
    v = e @ e / T
    for j in range(1, L + 1):
        v += 2 * (1 - j / (L + 1)) * (e[j:] @ e[:-j]) / T
    v = max(v, 1e-18)
    return float(x.mean() / math.sqrt(v / T))


def day_t(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / math.sqrt(len(x))
    return se, (x.mean() / se if se > 0 else float("nan"))


def p2(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2))) if np.isfinite(t) else 1.0


def load():
    P = pd.read_parquet(CACHE / "panel.parquet")
    cal = pd.read_parquet(CACHE / "calendar.parquet")["date"].tolist()
    idx = {d: i for i, d in enumerate(cal)}
    P["ti"] = P["date"].map(idx).astype(int)
    E = pd.read_parquet(CACHE / "events.parquet", columns=["code", "cats", "avail_main", "avail_ei"])
    E = E[E["cats"] != ""]
    E = E.assign(cat=E["cats"].str.split("|")).explode("cat")
    return P, E


def dist_col(P: pd.DataFrame, ev: pd.DataFrame, col: str) -> np.ndarray:
    """每个面板行到最近一次（不晚于 t）该类公告可知日的交易日距离，超过 max(KS) 记 99。"""
    a = ev[["code", col]].drop_duplicates()
    parts = []
    for j in range(max(KS) + 1):
        parts.append(pd.DataFrame({"code": a["code"].to_numpy(), "ti": a[col].to_numpy() + j, "d": j}))
    x = pd.concat(parts).groupby(["code", "ti"], as_index=False)["d"].min()
    m = P[["code", "ti"]].merge(x, on=["code", "ti"], how="left")
    return m["d"].fillna(99).to_numpy()


def stats(P: pd.DataFrame, flag: np.ndarray, base_d: pd.Series, n_hits: float) -> dict:
    f = P[flag]
    if not len(f):
        return {"n": 0}
    g = f.groupby("date")
    hit_d = g["y_open"].mean()
    diff = (hit_d - base_d.reindex(hit_d.index)).to_numpy()
    se, t = day_t(diff)
    ex = f[~(f["yizi0"].fillna(False) | f["yizi1"].fillna(False))]
    hit_ex = ex.groupby("date")["y_open"].mean()
    bd = base_d.reindex(hit_d.index)
    return {"n": int(len(f)), "days": int(len(hit_d)),
            "hit_day": float(hit_d.mean()), "base_day": float(bd.mean()),
            "ratio": float(hit_d.mean() / bd.mean()), "diff": float(diff.mean()),
            "se_day": se, "t_day": t, "t_nw": nw_t(diff),
            "hit_pooled": float(f["y_open"].mean()),
            "recall": float(f["y_open"].sum() / n_hits), "cover": float(len(f) / len(P)),
            "yizi1": float(f["yizi1"].mean()), "yizi0": float(f["yizi0"].fillna(False).mean()),
            "hit_raw_pooled": float(f["hit_raw"].mean()),
            "hit_ex_day": float(hit_ex.mean()) if len(hit_ex) else float("nan"),
            "diff_ex": float((hit_ex - base_d.reindex(hit_ex.index)).mean()) if len(hit_ex) else float("nan"),
            "n_ex": int(len(ex))}


def model_rel(P: pd.DataFrame, flag: np.ndarray, base_d: pd.Series, base_out_d: pd.Series) -> dict:
    f = P[flag]
    out = {"top200_share": float((f["rank_all"] <= 200).mean()),
           "pct_rank_median": float(f["pct_rank"].median()),
           "top10_n": int((f["rank"] <= 10).sum())}
    # 模型 200 名外：被标记 vs 同日 200 名外基准（互补性：模型抓不到的那块里它还有没有用）
    fo = f[f["rank_all"] > 200]
    if len(fo):
        h = fo.groupby("date")["y_open"].mean()
        d = (h - base_out_d.reindex(h.index)).to_numpy()
        se, t = day_t(d)
        out.update({"out200_n": int(len(fo)), "out200_hit_day": float(h.mean()),
                    "out200_diff": float(d.mean()), "out200_t_day": t, "out200_t_nw": nw_t(d)})
    fi = f[f["rank_all"] <= 200]
    if len(fi):
        out.update({"in200_n": int(len(fi)), "in200_hit_pooled": float(fi["y_open"].mean())})
    # 模型前 10 且有该类公告 vs 模型前 10（同一天配对）
    top = P[P["rank"] <= 10]
    tf = P[(P["rank"] <= 10) & flag]
    if len(tf):
        a = tf.groupby("date")["y_open"].mean()
        b = top.groupby("date")["y_open"].mean().reindex(a.index)
        d = (a - b).to_numpy()
        se, t = day_t(d)
        out.update({"top10f_days": int(len(a)), "top10f_hit_day": float(a.mean()),
                    "top10_hit_sameday": float(b.mean()), "top10f_diff": float(d.mean()),
                    "top10f_t_day": t, "top10f_t_nw": nw_t(d)})
    return out


def main() -> None:
    P, E = load()
    P = P.reset_index(drop=True)
    base_d = P.groupby("date")["y_open"].mean()
    base_out_d = P[P["rank_all"] > 200].groupby("date")["y_open"].mean()
    n_hits = float(P["y_open"].sum())
    print("面板 %d 行 %d 天，命中 %d，基准 %.3f%%" % (len(P), P["date"].nunique(), n_hits,
                                              100 * P["y_open"].mean()), flush=True)
    rows, mrows = [], []
    cats = sorted(E["cat"].unique())
    # 「任意公告」作为参照（不进多重检验族）
    for rule in RULES:
        col = "avail_" + rule
        dist_any = dist_col(P, E, col)
        for k in KS:
            rows.append({"rule": rule, "cat": "任意公告(参照)", "k": k,
                         **stats(P, dist_any <= k, base_d, n_hits)})
        for c in cats:
            dist = dist_col(P, E[E["cat"] == c], col)
            for k in KS:
                fl = dist <= k
                r = {"rule": rule, "cat": c, "k": k, **stats(P, fl, base_d, n_hits)}
                rows.append(r)
                if k in (0, 1, 5):
                    mrows.append({"rule": rule, "cat": c, "k": k, **model_rel(P, fl, base_d, base_out_d)})
            print(rule, c, "完成", flush=True)
    R = pd.DataFrame(rows)
    fam = R["cat"] != "任意公告(参照)"
    R["p_nw"] = R["t_nw"].map(p2)
    m = int(fam.sum())
    order = R.loc[fam, "p_nw"].sort_values()
    holm = pd.Series(np.nan, index=R.index)
    run = 0.0
    for i, (ix, p) in enumerate(order.items()):
        run = max(run, min(1.0, p * (m - i)))
        holm[ix] = run
    R["p_holm"] = holm
    R["p_bonf"] = np.where(fam, np.minimum(1.0, R["p_nw"] * m), np.nan)
    R.attrs["m"] = m
    R.to_csv(CACHE / "results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mrows).to_csv(CACHE / "model_rel.csv", index=False, encoding="utf-8-sig")
    print("检验个数（多重检验族）", m)


if __name__ == "__main__":
    sys.exit(main())
