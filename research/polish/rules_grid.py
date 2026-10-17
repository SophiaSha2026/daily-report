"""
规则网格：只在筛选集上调（打印筛选集成绩）；开发集的逐日命中序列同一趟算出来落盘但不打印，
选定后由 look_dev.py 只看选中的臂（criteria.txt 第 2 条）。

    PYTHONIOENCODING=utf-8 python research/polish/rules_grid.py

配置 = 加权当日百分位秩之和（权重 dict）+ 剔除条件列表。基线 = {"lret20": 1}，无剔除。
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd

import common_pl as P

GRID: dict[str, dict] = {}


def add(name, w, excl=(), fam=""):
    GRID[name] = {"w": w, "excl": list(excl), "fam": fam}


# ---- W 窗口 ----
add("base_ret20", {"lret20": 1}, fam="W")
for k in (5, 10, 40, 60):
    add("ret%d" % k, {"lret%d" % k: 1}, fam="W")
add("lm_10_20", {"lret10": 0.5, "lret20": 0.5}, fam="W")
add("lm_20_40", {"lret20": 0.5, "lret40": 0.5}, fam="W")
add("lm_10_20_40", {"lret10": 1 / 3, "lret20": 1 / 3, "lret40": 1 / 3}, fam="W")
add("lm_5_10_20_40", {"lret5": .25, "lret10": .25, "lret20": .25, "lret40": .25}, fam="W")
add("rs_20_40", {"r_lret20": 1, "r_lret40": 1}, fam="W")
add("rs_10_20", {"r_lret10": 1, "r_lret20": 1}, fam="W")
add("rs_5_20", {"r_lret5": 1, "r_lret20": 1}, fam="W")
add("raw_ret20", {"ret20": 1}, fam="W")          # 对数 vs 原值：排序一样，只是对账
add("ret20_norm", {"ret20_norm": 1}, fam="W")     # 按板块涨跌停幅度归一
add("lret20_vol", {"lret20_vol": 1}, fam="W")     # 波动调整
# ---- E 剔除（在 ret20 上）----
EX = {
    "nld1": [("nld20", ">=", 1)], "nld2": [("nld20", ">=", 2)],
    "mcap50": [("mcap", ">", 5e9)], "mcap100": [("mcap", ">", 1e10)], "mcap200": [("mcap", ">", 2e10)],
    "ret5neg": [("ret5", "<", 0)], "ret1neg": [("ret1", "<", 0)],
    "dh15": [("dist_hi20", "<", -0.15)], "dh10": [("dist_hi20", "<", -0.10)], "dh25": [("dist_hi20", "<", -0.25)],
    "px50": [("price_u", ">", 50)], "px100": [("price_u", ">", 100)],
    "streak3": [("lu_streak", ">=", 3)], "streak2": [("lu_streak", ">=", 2)],
    "old750": [("nbar", ">", 750)], "old1500": [("nbar", ">", 1500)],
    "r20x2": [("ret20", ">", 2.0)], "r20x3": [("ret20", ">", 3.0)],
    "yizi": [("yizi_today", ">=", 1)], "lu_today": [("lu_close", ">=", 1)],
    "turn1lo": [("turn1", "<", 0.03)], "amt20lo": [("amt20", "<", 3e7)],
    "nld1_mcap100": [("nld20", ">=", 1), ("mcap", ">", 1e10)],
    "nld1_dh15": [("nld20", ">=", 1), ("dist_hi20", "<", -0.15)],
    "nld1_mcap100_dh15": [("nld20", ">=", 1), ("mcap", ">", 1e10), ("dist_hi20", "<", -0.15)],
    "nld1_mcap100_ret5neg": [("nld20", ">=", 1), ("mcap", ">", 1e10), ("ret5", "<", 0)],
    "nld1_mcap100_dh15_ret5neg": [("nld20", ">=", 1), ("mcap", ">", 1e10), ("dist_hi20", "<", -0.15), ("ret5", "<", 0)],
}
for k, v in EX.items():
    add("ex_" + k, {"lret20": 1}, v, fam="E")
# ---- C 组合（秩和）----
for f in ("turn5", "turn1", "vol20", "vol60", "amp20", "ret5", "ret40", "amt_ratio", "max1_20"):
    for w in (0.25, 0.5, 1.0):
        add("c_%s_%g" % (f, w), {"r_lret20": 1, "r_" + f: w}, fam="C")
for w in (0.25, 0.5):
    add("c_mcap_-%g" % w, {"r_lret20": 1, "r_mcap": -w}, fam="C")
    add("c_nbar_-%g" % w, {"r_lret20": 1, "r_nbar": -w}, fam="C")
add("c_turn5_.5_mcap_-.25", {"r_lret20": 1, "r_turn5": .5, "r_mcap": -.25}, fam="C")
add("c_turn5_.5_vol20_.5", {"r_lret20": 1, "r_turn5": .5, "r_vol20": .5}, fam="C")
add("c_turn5_.5_ret5_.25", {"r_lret20": 1, "r_turn5": .5, "r_ret5": .25}, fam="C")
add("c_turn5_.5_ret40_.25", {"r_lret20": 1, "r_turn5": .5, "r_ret40": .25}, fam="C")
add("c_turn5_.5_ex_nld1", {"r_lret20": 1, "r_turn5": .5}, EX["nld1"], fam="C")
add("c_turn5_.5_ex_nld1_mcap100", {"r_lret20": 1, "r_turn5": .5}, EX["nld1_mcap100"], fam="C")


def need_cols():
    cols = set()
    for g in GRID.values():
        for k in g["w"]:
            k = k[2:] if k.startswith("r_") else k
            cols.add("ret" + k[4:] if k.startswith("lret") and k[4:].isdigit() and k != "lret20" else k)
        for f, _, _ in g["excl"]:
            cols.add(f)
    return sorted(cols)


def build(df: pd.DataFrame) -> pd.DataFrame:
    for k in (5, 10, 40, 60):
        df["lret%d" % k] = np.log1p(df["ret%d" % k].clip(lower=-0.99))
    rcols = set()
    for g in GRID.values():
        for k in g["w"]:
            if k.startswith("r_"):
                rcols.add(k)
    for k in sorted(rcols):
        df[k] = df.groupby("date")[k[2:]].rank(pct=True).astype(np.float32)
    return df


def score_of(df: pd.DataFrame, g: dict) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(len(df))
    for k, w in g["w"].items():
        s += w * df[k].to_numpy(float)
    el = np.ones(len(df), bool)
    for f, op, v in g["excl"]:
        x = df[f].to_numpy(float)
        hit = {">=": x >= v, ">": x > v, "<": x < v, "<=": x <= v}[op]
        el &= ~(hit & np.isfinite(x))
    return s, el


def main():
    t0 = time.time()
    base_cols = ["date", "code", "board", "y_open", "y_up", "pool", "lret20", "ret5", "ret10", "ret40", "ret60"]
    cols = sorted(set(base_cols) | set(need_cols()))
    df = P.load_pool(cols)
    df = df[df["pool"] & (df["date"] >= "2022-01-01")].reset_index(drop=True)
    df = build(df)
    P.say("池 %d 行 %.0fs，配置 %d 个" % (len(df), time.time() - t0, len(GRID)))
    hits, hits30, jac = {}, {}, {}
    base_top = None
    for name, g in GRID.items():
        s, el = score_of(df, g)
        t = P.rank_by(df, s, keep=30, elig=el)
        hits[name] = P.hit_series(t, 10)
        hits30[name] = P.hit_series(t, 30)
        sets = t[t["rank"] <= 10].groupby("date")["code"].apply(set)
        if base_top is None:
            base_top = sets
            jac[name] = 1.0
        else:
            jac[name] = float(np.mean([len(sets[d] & base_top[d]) / len(sets[d] | base_top[d]) for d in base_top.index if d in sets.index]))
        P.say("  %-28s %.0fs" % (name, time.time() - t0))
    H = pd.DataFrame(hits)
    H30 = pd.DataFrame(hits30)
    H.to_parquet(P.PL / "rules_hits10.parquet")
    H30.to_parquet(P.PL / "rules_hits30.parquet")
    json.dump({k: {"w": v["w"], "excl": v["excl"], "fam": v["fam"], "jac": jac[k]} for k, v in GRID.items()},
              (P.PL / "rules_grid.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 只打印筛选集
    scr = H.index < "2024-06-01"
    L = ["筛选集 2022-01..2024-05（%d 天）：对 base_ret20 按天配对；开发集已算未看（rules_hits10.parquet）" % int(scr.sum()),
         "%-28s 族  命中%%   差pp    NW t   去最好月t  前30差pp  重合" % "配置"]
    b = H.loc[scr, "base_ret20"]
    b30 = H30.loc[scr, "base_ret20"]
    rows = []
    for name in GRID:
        d = (H.loc[scr, name] - b).dropna()
        d30 = (H30.loc[scr, name] - b30).dropna()
        dm = d.groupby(d.index.str[:7]).mean()
        best = dm.idxmax() if d.mean() >= 0 else dm.idxmin()
        tdb = P.nw_t(d[d.index.str[:7] != best]) if name != "base_ret20" else 0.0
        rows.append((name, GRID[name]["fam"], 100 * H.loc[scr, name].mean(), 100 * d.mean(), P.nw_t(d) if name != "base_ret20" else 0.0,
                     tdb, 100 * d30.mean(), jac[name]))
    for r in sorted(rows, key=lambda r: (r[1], -r[4])):
        L.append("%-28s %s  %5.2f  %+6.2f  %+6.2f  %+6.2f    %+6.2f   %.2f" % r)
    txt = "\n".join(L)
    P.say(txt)
    (P.HERE / "results_rules_screen.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
