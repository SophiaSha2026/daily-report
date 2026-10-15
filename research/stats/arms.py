"""
实验 15~21 在开发集上试过的每一个臂：读产物、按 evalkit 口径重建每天的名次表。

只读 data/breakout/raw/ 下的已有产物，不重训。**确认集 2026-01..08 一行都不读**：
所有 parquet 带 filters=date<=2025-12-31 读（pool.parquet 和 labels_open.parquet 本身含 2026，
读的时候就在 pyarrow 层过滤掉），读完再断言一次。

臂的登记表 REGISTRY：每行一个「臂 vs 对照」，带实验号、族、口径（前 k、标签）、当时报的 t。
数字来源：各实验的 summary.json / prereg.json，和 docs/breakout_log.md 对过（不一致的在 note 里写）。
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "breakout" / "raw"
EXP = RAW / "exp"
DEV_END = "2025-12-31"
FILT = [("date", "<=", DEV_END)]
VAL = ("2025-03-01", "2025-12-31")      # 实验 15 / 16 用的旧验证集
SINCE25 = "2025-01-01"


def _guard(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if len(df) and str(df["date"].max()) > DEV_END:
        raise SystemExit(f"{what} 含开发集之后的日期（{df['date'].max()}），拒绝继续")
    return df


def read(path: Path, columns=None) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=columns, filters=FILT)
    d["date"] = d["date"].astype(str)
    if "code" in d:
        d["code"] = d["code"].astype(str).str.zfill(6)
    return _guard(d, path.name)


@lru_cache(maxsize=1)
def st_codes() -> frozenset:
    raw = json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8"))
    cs = raw.get("codes", raw) if isinstance(raw, dict) else raw
    return frozenset(str(c).zfill(6) for c in cs)


@lru_cache(maxsize=1)
def labels_open() -> pd.DataFrame:
    return read(RAW / "labels_open.parquet", ["code", "date", "y_open", "r20_open"])


@lru_cache(maxsize=1)
def market_base_open() -> pd.Series:
    """每天全市场（剔 ST）的买得到口径基准率。"""
    lab = labels_open()
    lab = lab[~lab["code"].isin(st_codes())]
    return lab.groupby("date")["y_open"].mean()


# ---------------------------------------------------------------
#  实验 15（exp_time，旧验证集，全市场预测值 -> 自己排名次）
# ---------------------------------------------------------------
E15_ARMS = ["base", "boost3", "hl3", "hl6", "hl12", "eqm", "roll3", "roll6", "roll12"]
E15_ENS = {"ens_win": ["base", "roll12", "roll6"], "ens_dec": ["base", "hl6", "hl12"]}


@lru_cache(maxsize=1)
def _e15_raw() -> tuple[pd.DataFrame, dict]:
    d = read(RAW / "exp_time" / "preds.parquet")
    d = d[~d["code"].isin(st_codes())]
    for name, parts in E15_ENS.items():
        d[name] = d[parts].mean(axis=1).astype(np.float32)
    q = json.loads((RAW / "exp_time" / "q.json").read_text(encoding="utf-8"))
    ref = pd.read_parquet(RAW / "exp_time" / "ref_sub.parquet")
    for name, parts in E15_ENS.items():
        for m, g in ref.groupby("month"):
            if m in q:
                q[m][name] = [float(x) for x in np.quantile(g[parts].mean(axis=1).to_numpy(), np.linspace(0, 1, 101))]
    d = d.merge(labels_open()[["code", "date", "y_open"]], on=["code", "date"], how="left")
    return d, q


def e15_top(arm: str, keep: int = 200) -> pd.DataFrame:
    d, q = _e15_raw()
    o = d[["date", "code", "board", "y_up", "y_open", arm]].rename(columns={arm: "p"})
    o = o.sort_values(["date", "p"], ascending=[True, False])
    o["rank"] = o.groupby("date").cumcount() + 1
    o = o[o["rank"] <= keep].copy()
    o["score"] = 0
    for m, idx in o.groupby(o["date"].str[:7]).groups.items():
        qq = np.asarray(q[m][arm], float)
        o.loc[idx, "score"] = np.clip(np.searchsorted(qq, o.loc[idx, "p"].to_numpy(float)), 0, 100)
    return o.reset_index(drop=True)


# ---------------------------------------------------------------
#  实验 16（exp_acc，旧验证集，每天前 200）
# ---------------------------------------------------------------
E16_ARMS = ["base", "open", "open_dd", "allneg", "bag5", "f3", "f5", "f35", "rank", "rank_f35",
            "hist_2023", "hist_all", "g1", "g2", "g12", "tail", "tail_blend", "tail_g12", "tail_g12_blend"]


def e16_top(arm: str) -> pd.DataFrame:
    return read(RAW / "exp_acc" / f"preds_{arm}.parquet")


# ---------------------------------------------------------------
#  实验 17 / 18 / 21（evalkit，开发集 19 个月，每天前 200）
# ---------------------------------------------------------------
W_SRC = {"base": "w1_scale_avg", "avg3": "w1_scale_avg",
         "g2": "w2_feat_hp", "hp_l15": "w2_feat_hp", "hp_l63": "w2_feat_hp", "hp_mc400": "w2_feat_hp",
         "hp_lr02": "w2_feat_hp", "hp_cs05": "w2_feat_hp",
         "base_re": "w3_essence", "y20": "w3_essence", "y20_blend": "w3_essence",
         "pool_small": "w3_essence", "lr_blend": "w3_essence"}


@lru_cache(maxsize=None)
def w_top(arm: str) -> pd.DataFrame:
    if arm == "nostar":   # 实验 18：现状剔科创补位（事后）
        t = w_top("base")
        t = t[t["board"] != "star"].copy()
        t["rank"] = t.groupby("date").cumcount() + 1
        return t
    if arm in ("thr98", "adj_open", "thr98_open", "adj_close"):
        return w_post(arm)
    t = read(EXP / W_SRC[arm] / f"top_{arm}.parquet")
    return t.sort_values(["date", "rank"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def w_post(arm: str) -> pd.DataFrame:
    """实验 21 的事后臂：调 exp_w3.rescore_with_adj（纯函数，只读），生产规则口径。"""
    sys.path.insert(0, str(ROOT / "src" / "breakout"))
    sys.path.insert(0, str(ROOT / "src"))
    import exp_w3  # noqa: E402
    top = w_top("base")
    if arm == "thr98":
        return top.assign(score=np.where(top["score"] >= 98, top["score"], 0))
    allp = None  # rescore_with_adj 不读 allp
    if arm in ("adj_open", "thr98_open"):
        t = exp_w3.rescore_with_adj(top, allp, "y_open")
        if arm == "thr98_open":
            t = t.assign(score=np.where(t["score"] >= 98, t["score"], 0))
        return t
    return exp_w3.rescore_with_adj(top, allp, "y_up")


# ---------------------------------------------------------------
#  实验 19（cont_v1，强势延续，另一个目标，自成一族）
# ---------------------------------------------------------------
def cont_top(arm: str) -> pd.DataFrame:
    d = read(RAW / "exp" / "cont_v1" / "preds.parquet")
    key = "p" if arm == "model" else "rule"
    d = d.sort_values(["date", key], ascending=[True, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    return d.rename(columns={"y15": "y_cont"})


# ---------------------------------------------------------------
#  登记表：每个「臂 vs 对照」一行
# ---------------------------------------------------------------
# (实验, 族, 臂, 对照, 读法, 口径 k, 标签, 区间, 当时报的 t, 当时的结论)
# 族：A = 旧验证集 2025-03..12（实验 15/16），B = 开发集 19 个月（实验 17/18/21），
#     P = 生产规则（≥97 分且前 10）的事后臂，C = 强势延续（另一目标）
REGISTRY: list[dict] = []


def _reg(exp, fam, arm, ctrl, loader, k, y, span, t_rep, verdict, note=""):
    REGISTRY.append(dict(exp=exp, fam=fam, arm=arm, ctrl=ctrl, loader=loader, k=k, y=y, span=span,
                         t_rep=t_rep, verdict=verdict, note=note))


_E15_T = {"boost3": -0.15, "hl3": 0.80, "hl6": 2.00, "hl12": 2.59, "eqm": 4.50, "roll3": -1.77,
          "roll6": -0.70, "roll12": -0.13, "ens_win": 4.10, "ens_dec": 4.03}
for a, t in _E15_T.items():
    _reg(15, "A", a, "base", "e15", 10, "y_up", "val", t, "上线" if a == "eqm" else "不上",
         "当时尺子是收盘口径 y_up；同时给买得到口径")
_E16_T = {"open": 0.00, "open_dd": -1.77, "allneg": -4.72, "bag5": -0.16, "f3": -1.43, "f5": -0.17,
          "f35": 0.05, "rank": -1.82, "rank_f35": -1.24, "hist_2023": 1.28, "hist_all": -1.21,
          "g1": -3.63, "g2": 1.05, "g12": -2.75, "tail": -3.61, "tail_blend": -0.87, "tail_g12": -4.20,
          "tail_g12_blend": -3.23}
for a, t in _E16_T.items():
    _reg(16, "A", a, "base", "e16", 10, "y_open", "val", t, "不上",
         "tail_g12_blend 日志表里没列，summary.json 里有" if a == "tail_g12_blend" else "")
_reg(17, "B", "avg3", "base", "w", 10, "y_open", "dev", 0.94, "未定")
_W2_T = {"g2": 1.14, "hp_l15": -1.49, "hp_l63": -0.65, "hp_mc400": -1.40, "hp_lr02": -0.89, "hp_cs05": -0.62,
         "nostar": 0.51}
for a, t in _W2_T.items():
    _reg(18, "B", a, "base", "w", 10, "y_open", "dev", t, "未定")
_reg(18, "B", "g2", "base", "w", 10, "y_open", "since25", 2.08, "未定（差一点）",
     "事后加看的子区间：同一个臂第二次看")
_W3_T = {"base_re": 0.73, "y20": -1.48, "y20_blend": 0.75, "lr_blend": -1.43, "pool_small": -2.98}
for a, t in _W3_T.items():
    _reg(21, "B", a, "base", "w", 10, "y_open", "dev", t, "不上")
_W3P_T = {"thr98": 1.79, "adj_open": 0.19, "thr98_open": 1.55, "adj_close": 0.61}
for a, t in _W3P_T.items():
    _reg(21, "P", a, "base", "w", "prod", "y_open", "dev", t,
         "进邮件（高置信提示）" if a == "thr98" else "不上",
         "按天配对的是两边当天的生产规则名额均值" + ("；第一次跑报 2.70，修 bug 后 0.61" if a == "adj_close" else ""))
_reg(19, "C", "model", "rule", "cont", 5, "y_cont", "cont", -1.99, "不过线", "目标 +15%/10 根，候选池涨停股")


def top_of(row: dict, arm: str) -> pd.DataFrame:
    L = row["loader"]
    if L == "e15":
        return e15_top(arm)
    if L == "e16":
        return e16_top(arm)
    if L == "w":
        return w_top(arm)
    if L == "cont":
        return cont_top(arm)
    raise KeyError(L)


def span_filter(t: pd.DataFrame, span: str) -> pd.DataFrame:
    if span == "val":
        return t[t["date"].between(*VAL)]
    if span == "since25":
        return t[t["date"] >= SINCE25]
    return t


def daily_hit(t: pd.DataFrame, k, y: str) -> pd.Series:
    """每天前 k 名（k='prod' 则生产规则 ≥97 分且前 10）的命中均值。"""
    if k == "prod":
        p = t[(t["score"] >= 97) & (t["rank"] <= 10)]
    else:
        p = t[t["rank"] <= k]
    return p.groupby("date")[y].mean()
