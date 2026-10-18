"""深挖数据（research/deepdata，2026-10-18）公用件：候选池 / ret20 名次 / 稠密网格 / 特征表落盘 / 快速诊断。

候选池、ret20、y_open 全部复用 research/polish 的 pool.parquet（common_pl.load_pool），不另算一份：
    pool 列 = 候选池（elig0 + 减持 / 解禁 / 增发剔除），lret20 = ret20 的 log，y_open = 买得到口径标签。
    ret20 名次 rank_ret20 只在 pool 行里按 lret20 降序数（和 research/altdata/features.py 同一写法）。
稠密网格 Grid 从 research/altdata/features.py 按路径加载（不复制一份；那个模块名和 src/breakout/features.py 撞名，不能 import）。

特征表：features_<源>.parquet，键 code（6 位）+ date（YYYY-MM-DD 交易日），覆盖 pool.parquet 里 date >= FEAT_START 的全部行
（含不在候选池的，gate 自己选宇宙）；每张表一段说明追加进 features_README.txt。

快速诊断 diag_table：每个特征在「全候选池」和「ret20 前 50」里 高组 对 低组 的按天配对命中差（y_open），
NW t（滞后 20，evalkit._nw_se），两段（筛选 2022-01..2024-05 / 开发 2024-06..2025-12）各算，同号标出。
连续特征按当天三分位（高 = 上三分之一、低 = 下三分之一），0/1 特征按有 / 无。不训模型、不立臂、不看 2026 标签。
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import common_pl as P  # noqa: E402

RAW = HERE / "raw"
README = HERE / "features_README.txt"
PROGRESS = HERE / "progress.txt"
FEAT_START = "2021-12-01"     # 特征表起点（标签两段从 2022-01 起，留一个月）
GRID_START = "2021-06-01"     # 稠密网格起点（给 2021-12 留 120 个交易日回看）
SCREEN_END = "2024-06-01"


def say(*a):
    print(*a, flush=True)


def progress(text: str) -> None:
    from bjtime import now_bj
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def _load_grid_cls():
    spec = importlib.util.spec_from_file_location("altdata_features", ROOT / "research" / "altdata" / "features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Grid


Grid = _load_grid_cls()


def load_pool(extra: list[str] | None = None) -> pd.DataFrame:
    """pool.parquet 从 GRID_START 起的全部行 + rank_ret20（候选池内）。"""
    t0 = time.time()
    cols = ["code", "date", "board", "pool", "elig0", "lret20", "ret20", "y_open", "y_up", "close", "mcap", "amt1", "amt20", "turn1", "lu_close"]
    for c in extra or []:
        if c not in cols:
            cols.append(c)
    df = P.load_pool(cols)
    df = df[df["date"] >= GRID_START].reset_index(drop=True)
    q = df.loc[df["pool"], ["date", "lret20"]].copy()
    q["_r"] = np.nan
    q = q.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    q["_r"] = q.groupby("date").cumcount() + 1
    df["rank_ret20"] = np.nan
    df.loc[q.index, "rank_ret20"] = q["_r"].astype(float)
    say("pool %d 行 %s..%s %d 天 %d 只 %.0fs" % (len(df), df["date"].min(), df["date"].max(), df["date"].nunique(), df["code"].nunique(), time.time() - t0))
    return df


def make_grid(df: pd.DataFrame) -> Grid:
    return Grid(sorted(df["code"].unique()), sorted(df["date"].unique()))


def dense_last_ffill(g: Grid, code: pd.Series, di: np.ndarray, val: np.ndarray):
    """事件值（按公告日对齐）前向填充成「最近一次」网格，并给出距最近一次的交易日数。"""
    A = g.dense(code, di, val, agg="last")
    V, idx = Grid.ffill(A)
    age = np.where(idx >= 0, np.arange(A.shape[1])[None, :] - idx, np.nan)
    return V, age


def write_table(name: str, feat: pd.DataFrame, readme: str) -> Path:
    """feat 必须含 code/date；float 列压成 float32。追加 README 段落和 progress 一行。"""
    out = HERE / f"features_{name}.parquet"
    f = feat.copy()
    for c in f.columns:
        if c in ("code", "date"):
            continue
        if pd.api.types.is_bool_dtype(f[c]):
            f[c] = f[c].astype(np.int8)
        elif pd.api.types.is_float_dtype(f[c]):
            f[c] = f[c].astype(np.float32)
    f.to_parquet(out, index=False)
    with README.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## features_{name}.parquet（{len(f)} 行，{f['date'].min()}..{f['date'].max()}，{len(f.columns) - 2} 列）\n{readme.strip()}\n")
    progress(f"落地 features_{name}.parquet：{len(f)} 行 {f['date'].min()}..{f['date'].max()}，列 {[c for c in f.columns if c not in ('code', 'date')]}")
    say("->", out, f.shape)
    return out


# ---------------------------------------------------------------
#  快速诊断
# ---------------------------------------------------------------
def _nw_t(d: pd.Series) -> float:
    return P.nw_t(d) if len(d) > 25 else np.nan


def paired_diff(x: pd.DataFrame, feat: str, mode: str, by: str | None = None) -> pd.Series:
    """x: date / y_open / feat [/ by]。高组对低组的按天配对差（两组当天都有人才算那天）。
    by 给了的话（名次档），三分位在 (date, by) 内切、差先在档内配对再按天取档均值 —— 控制名次后的差。"""
    v = x[feat].astype(float)
    fin = np.isfinite(v) & np.isfinite(x["y_open"].astype(float))
    x = x[fin]
    v = v[fin]
    keys = ["date"] + ([by] if by else [])
    if mode == "tertile":
        r = v.groupby([x[k] for k in keys]).rank(pct=True, method="average")
        hi, lo = r > 2 / 3, r <= 1 / 3
    else:
        hi = v > 0
        lo = ~hi
    a = x.loc[hi].groupby(keys)["y_open"].mean()
    b = x.loc[lo].groupby(keys)["y_open"].mean()
    d = (a - b).dropna()
    if by:
        d = d.groupby(level=0).mean()
    return d


def seg_stats(d: pd.Series) -> dict:
    a, b = d[d.index < SCREEN_END], d[d.index >= SCREEN_END]
    out = {"n_all": len(d), "d_all": 100 * d.mean() if len(d) else np.nan, "t_all": _nw_t(d),
           "n_s": len(a), "d_s": 100 * a.mean() if len(a) else np.nan, "t_s": _nw_t(a),
           "n_d": len(b), "d_d": 100 * b.mean() if len(b) else np.nan, "t_d": _nw_t(b)}
    out["same"] = bool(len(a) and len(b) and np.sign(a.mean()) == np.sign(b.mean()) and a.mean() != 0)
    return out


def diag_table(feat: pd.DataFrame, pool: pd.DataFrame, cols: list[str], modes: dict[str, str] | None = None,
               label: str = "") -> str:
    """feat: code/date + 特征列；pool: load_pool 的表。返回文本表，同时打印。"""
    modes = modes or {}
    base = pool.loc[pool["pool"] & (pool["date"] >= "2022-01-01") & (pool["date"] <= "2025-12-31"),
                    ["code", "date", "rank_ret20", "y_open"]]
    m = base.merge(feat[["code", "date"] + cols], on=["code", "date"], how="left")
    top = m[m["rank_ret20"] <= 50].copy()
    top["bucket"] = pd.cut(top["rank_ret20"], [0, 10, 30, 50], labels=["1-10", "11-30", "31-50"]).astype(str)
    lines = [f"# 诊断 {label}：高组对低组 按天配对命中差（pp），NW t 滞后 20；筛选 2022-01..2024-05 / 开发 2024-06..2025-12",
             "# 宇宙：全池 = 候选池全部；前50 = ret20 前 50；前50控名次 = 在 1-10 / 11-30 / 31-50 三档内各切三分位再配对（去掉和名次相关的那部分）",
             "# 分组：三分位 = 当天内上 1/3 对下 1/3；有无 = >0 对 =0（0/1 列和零占比过半的计数列）；覆盖% = 有值（有无列：>0）的行占比",
             "%-26s %-12s | %-30s | %-30s | %s" % ("特征", "宇宙", "筛选 diff (t, 天)", "开发 diff (t, 天)", "合并 diff (t) 同号 覆盖%")]
    for c in cols:
        v = m[c].astype(float)
        vf = v[np.isfinite(v)]
        uniq = np.unique(vf)
        sparse_count = len(uniq) > 0 and uniq.min() >= 0 and np.all(uniq == np.round(uniq)) and (vf == 0).mean() > 0.5
        mode = modes.get(c) or ("flag" if (set(uniq).issubset({0.0, 1.0}) or sparse_count) else "tertile")
        for uni, x, by in (("全池", m, None), ("前50", top, None), ("前50控名次", top, "bucket")):
            cov = 100 * np.isfinite(x[c].astype(float)).mean()
            if mode == "flag":
                cov = 100 * (x[c].astype(float) > 0).mean()
            d = paired_diff(x[["date", "y_open", c] + ([by] if by else [])], c, mode, by=by)
            s = seg_stats(d)
            lines.append("%-26s %-12s | %+7.2f (t %5.2f, %4d)      | %+7.2f (t %5.2f, %4d)      | %+6.2f (t %5.2f) %s %5.1f%%" % (
                c, uni + ("/三分位" if mode == "tertile" else "/有无"), s["d_s"], s["t_s"], s["n_s"], s["d_d"], s["t_d"], s["n_d"],
                s["d_all"], s["t_all"], "同号" if s["same"] else "异号", cov))
    txt = "\n".join(lines)
    say(txt)
    return txt
