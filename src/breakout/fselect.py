"""
特征选择：四道筛。只在训练集上做，验证集和 holdout 不参与。

为什么不把 80 多维全丢进去
--------------------------
在 3.38% 基础比率的数据上，噪音特征不是"没用"，是**有害**：它们给模型
提供了在训练集上降低损失的捷径，而那些捷径在验证集上一定失效。
GBDT 尤其吃这一套——它会认真地在噪音里找分裂点。

四道筛
------
    1. 单变量 IC       |Spearman(feat, y)| < 0.01 丢掉
    2. IC 时间稳定性   按月算 IC，符号翻转率 > 40% 丢掉    ← 最重要
    3. 相关剪枝        |r| > 0.85 的一对，留 IC 高的
    4. L1 正则         逻辑回归 L1 再压一轮，系数为 0 丢掉

第 2 道是关键。一个特征牛市 IC 为正、熊市为负，全样本平均下来可能还不错，
但实盘上等于抛硬币。符号翻转率直接把这类特征筛掉，而单看全样本 IC
永远发现不了。

每一道都记录**被丢掉的特征名和原因**。特征被丢的原因，和留下的特征
一样重要——下次想加特征时，这份记录能直接回答"这个试过了，没用"。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger("select")

# 2026-09-12 实验定的值。测过 0.010 / 0.005 / 0.002 三档（top10，LightGBM）：
#     0.010 -> 19 个特征, 命中 11.11%
#     0.005 -> 34 个特征, 命中 13.96%   <- 最优
#     0.002 -> 40 个特征, 命中 13.29%
# 两端都低、中间高的 U 型，说明 0.005 是真实最优点而不是碰巧。
# 0.010 砍太狠，把「单个 IC 弱但组合起来有用」的特征丢了 —— GBDT 恰恰
# 擅长用这种组合；0.002 又放进太多噪音。
IC_MIN = 0.005
FLIP_MAX = 0.40
CORR_MAX = 0.85


def single_ic(df: pd.DataFrame, cols: list[str], y: str) -> pd.Series:
    """每个特征对标签的 Spearman 相关。用秩相关不用皮尔逊：
    特征已经是横截面百分位，标签是 0/1，秩相关对这种组合更稳。"""
    out = {}
    yy = df[y].to_numpy(float)
    ok0 = np.isfinite(yy)
    for c in cols:
        x = df[c].to_numpy(float)
        ok = ok0 & np.isfinite(x)
        if ok.sum() < 500 or np.nanstd(x[ok]) == 0:
            out[c] = 0.0
            continue
        try:
            out[c] = float(spearmanr(x[ok], yy[ok]).statistic)
        except Exception:  # noqa: BLE001
            out[c] = 0.0
    return pd.Series(out).fillna(0.0)


def ic_stability(df: pd.DataFrame, cols: list[str], y: str) -> pd.DataFrame:
    """按月算 IC，返回均值、标准差和符号翻转率。

    符号翻转率 = 与总体 IC 符号相反的月份占比。这个指标回答的是
    「这个特征的方向稳不稳」，而不是「它平均有多强」。
    """
    df = df.copy()
    df["_m"] = df["date"].str[:7]
    months = sorted(df["_m"].unique())
    rec = {c: [] for c in cols}
    for m in months:
        sub = df[df["_m"] == m]
        if len(sub) < 300:
            continue
        ic = single_ic(sub, cols, y)
        for c in cols:
            rec[c].append(ic[c])
    rows = {}
    for c in cols:
        v = np.array(rec[c], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            rows[c] = (0.0, 0.0, 1.0, 0)
            continue
        mean = float(v.mean())
        sign = np.sign(mean) if mean != 0 else 1.0
        flip = float((np.sign(v) != sign).mean())
        rows[c] = (mean, float(v.std()), flip, len(v))
    return pd.DataFrame(rows, index=["ic_mean", "ic_std", "flip", "n_month"]).T


def corr_prune(df: pd.DataFrame, cols: list[str],
               ic: pd.Series, thr: float = CORR_MAX) -> tuple[list[str], list]:
    """相关剪枝：|r| > thr 的一对，留 |IC| 大的那个。"""
    sub = df[cols].astype(float)
    # 采样算相关矩阵，417 万行全量算没必要
    if len(sub) > 200_000:
        sub = sub.sample(200_000, random_state=7)
    cm = sub.corr().abs()
    order = ic.abs().sort_values(ascending=False).index.tolist()
    keep, dropped = [], []
    for c in order:
        if c not in cols:
            continue
        clash = None
        for k in keep:
            if cm.loc[c, k] > thr:
                clash = k
                break
        if clash is None:
            keep.append(c)
        else:
            dropped.append((c, f"与 {clash} 相关 {cm.loc[c, clash]:.2f}"))
    return keep, dropped


def run(df: pd.DataFrame, cols: list[str], y: str = "y_t0",
        out_dir: Path | None = None) -> dict:
    """跑完四道筛（前三道；第四道 L1 在 model.py 里跟着训练一起做）。

    返回 {"keep": [...], "log": {...}}。
    """
    report: dict = {"start": len(cols), "dropped": {}}
    cols = [c for c in cols if c in df.columns]

    # --- 1 单变量 IC ---
    ic = single_ic(df, cols, y)
    weak = [c for c in cols if abs(ic[c]) < IC_MIN]
    cols = [c for c in cols if c not in weak]
    report["dropped"]["ic_too_weak"] = {c: round(float(ic[c]), 4)
                                        for c in weak}
    log.info("筛1 单变量IC: 丢 %d，剩 %d", len(weak), len(cols))

    # --- 2 IC 时间稳定性 ---
    st = ic_stability(df, cols, y)
    unstable = st.index[st["flip"] > FLIP_MAX].tolist()
    cols = [c for c in cols if c not in unstable]
    report["dropped"]["ic_unstable"] = {
        c: {"flip": round(float(st.loc[c, "flip"]), 2),
            "ic": round(float(st.loc[c, "ic_mean"]), 4)} for c in unstable}
    log.info("筛2 IC稳定性: 丢 %d（符号翻转>%.0f%%），剩 %d",
             len(unstable), FLIP_MAX * 100, len(cols))

    # --- 3 相关剪枝 ---
    keep, dropped = corr_prune(df, cols, ic)
    report["dropped"]["collinear"] = dict(dropped)
    log.info("筛3 相关剪枝: 丢 %d，剩 %d", len(dropped), len(keep))

    report["keep"] = keep
    report["end"] = len(keep)
    report["ic_table"] = {
        c: {"ic": round(float(ic.get(c, 0)), 4),
            "flip": round(float(st.loc[c, "flip"]), 2) if c in st.index else None}
        for c in keep}

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_select.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
