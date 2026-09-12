"""
走向前验证。整个爆发线的成败由这个模块的输出判定。

训练用 y_t0，验收用 y_up
------------------------
两个标签回答不同的问题，不能混：

    y_t0   这一天是不是一段行情的**起涨点**。信号干净（一段行情只有一个），
           但基础比率只有 0.25%，适合当训练目标。
    y_up   这一天买入，未来 20 日内最高价是否涨超 50%。基础比率 3.38%，
           **这才是用户关心的结果**，所以验收指标一律按它算。

拿 y_t0 的命中率去汇报会虚高得离谱（分母小一个量级），那是自欺。

协议
----
    训练集   2023-05 .. 2025-02
    验证集   2025-03 .. 2025-12     选模型、调超参
    holdout  2026-01 .. 2026-09     锁死，全程只看一次

走向前：每个月重新拟合一次，只用该月**之前**的数据预测该月。
这是最接近实盘的评估方式——实盘里你永远只有过去的数据。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("validate")

TRAIN_END = "2025-03-01"
VALID_END = "2026-01-01"      # 之后是 holdout，锁死
TOP_N = 10                    # 清单 A 每天给几只


def daily_topn(df: pd.DataFrame, proba: np.ndarray,
               n: int = TOP_N, y: str = "y_up") -> pd.DataFrame:
    """按天取前 n 只，算命中率。

    必须**按天**取，不能全局取 top：全局 top 会集中落在少数几个大行情日
    （2024-09 那种），算出来的命中率反映的是"挑对了日子"而不是"挑对了票"。
    清单 A 是每天都要出的，评估口径必须和使用口径一致。
    """
    d = df[["date", "code", y]].copy()
    d["p"] = proba
    d = d[np.isfinite(d[y])]
    if not len(d):
        return pd.DataFrame(columns=["date", "code", "p", y])
    d = d.sort_values(["date", "p"], ascending=[True, False])
    return d.groupby("date", sort=False).head(n)


def summarize(picks: pd.DataFrame, y: str = "y_up") -> dict:
    if not len(picks):
        return {"n": 0, "hit": float("nan")}
    return {"n": int(len(picks)), "hit": float(picks[y].mean())}


def walk_forward(df: pd.DataFrame, feats: list[str], make_model,
                 y_train: str = "y_t0", y_eval: str = "y_up",
                 start: str = TRAIN_END, end: str = VALID_END,
                 top_n: int = TOP_N) -> tuple[pd.DataFrame, dict]:
    """按月走向前。返回 (每月成绩, 汇总)。

    make_model 是个工厂函数（每月要新建一个，不能复用上个月拟合过的）。
    """
    import model as M

    months = sorted({d[:7] for d in df["date"] if start <= d < end})
    rows, all_picks = [], []
    for m in months:
        tr = df[df["date"] < m + "-01"]
        te = df[df["date"].str[:7] == m]
        tr = tr[np.isfinite(tr[y_train])]
        if len(tr) < 5000 or not len(te):
            continue
        trs = M.stratified_sample(tr, y_train)
        if trs[y_train].sum() < 30:
            continue
        mdl = make_model()
        mdl.fit(trs, feats, y_train)
        p = mdl.predict_proba(te)
        picks = daily_topn(te, p, top_n, y_eval)
        s = summarize(picks, y_eval)
        base = float(te[y_eval].mean(skipna=True))
        rows.append({"month": m, "n_pick": s["n"], "hit": s["hit"],
                     "base": base,
                     "lift": s["hit"] / base if base > 0 else np.nan})
        all_picks.append(picks.assign(month=m))
        log.info("  %s  选 %3d 只  命中 %.1f%%  基础 %.1f%%  %.1f倍",
                 m, s["n"], 100 * s["hit"], 100 * base,
                 rows[-1]["lift"] if np.isfinite(rows[-1]["lift"]) else 0)

    by_month = pd.DataFrame(rows)
    picks = (pd.concat(all_picks, ignore_index=True) if all_picks
             else pd.DataFrame())
    return by_month, {"picks": picks}


def acceptance(by_month: pd.DataFrame, picks: pd.DataFrame,
               df: pd.DataFrame, y_eval: str = "y_up") -> dict:
    """对照设计文档 2.2 的验收表。每条都给实测值和是否达标。

    不达标不是"再调调参数"的信号，是按分级响应走：
    加特征 -> 换模型 -> 改事件定义 -> 判定不可行。
    """
    if not len(by_month) or not len(picks):
        return {"ok": False, "reason": "没有任何有效月份"}

    overall = float(picks[y_eval].mean())
    base = float(df[y_eval].mean(skipna=True))

    # 剔除两个极端月（924 行情和 2025-12）后重算
    ex = picks[~picks["month"].isin(["2024-09", "2025-12"])]
    hit_ex = float(ex[y_eval].mean()) if len(ex) else np.nan

    zero_months = int((by_month["hit"] == 0).sum())

    # 分板块
    board = None
    if "code" in picks.columns:
        import features as F
        b = picks.assign(board=picks["code"].map(F.board_of))
        g = b.groupby("board")[y_eval].agg(["mean", "size"])
        g = g[g["size"] >= 20]
        if len(g) >= 2:
            board = {"ratio": float(g["mean"].max() / max(g["mean"].min(), 1e-9)),
                     "detail": {k: round(float(v), 4)
                                for k, v in g["mean"].items()}}

    checks = [
        ("清单A命中率", overall, 0.15, overall >= 0.15),
        ("剔除极端月后", hit_ex, 0.10,
         bool(np.isfinite(hit_ex) and hit_ex >= 0.10)),
        ("零命中月份数", zero_months, 1, zero_months <= 1),
    ]
    if board:
        checks.append(("板块最强/最弱", board["ratio"], 3.0,
                       board["ratio"] <= 3.0))

    return {
        "ok": all(c[3] for c in checks),
        "overall_hit": overall, "base_rate": base,
        "lift": overall / base if base > 0 else np.nan,
        "hit_ex_extreme": hit_ex, "zero_months": zero_months,
        "board": board,
        "checks": [{"name": c[0], "value": (None if not np.isfinite(float(c[1]))
                                            else round(float(c[1]), 4)),
                    "threshold": c[2], "pass": bool(c[3])} for c in checks],
    }


def report(by_month: pd.DataFrame, acc: dict, out: Path,
           extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"by_month": by_month.to_dict("records"), "acceptance": acc}
    if extra:
        payload.update(extra)
    (out / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    log.info("验证报告 -> %s", out / "validation.json")
