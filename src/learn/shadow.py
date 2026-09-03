"""
影子排序器：秩归一特征上的稳健线性模型（擂台胜者 RankHuber 的转正通道）。

为什么存在（2026-09-03 的测量结论）
----------------------------------
四次点火 + 全量擂台给出了清晰的三段事实：

  1. 手写六维打分器的 9 个可调参数已近局部最优：箱内梯度 ≤0.03/σ，
     继续调旋钮可挖的改善不足 0.1%。
  2. 天花板在 2~3 倍外：同样的特征，RankHuber 样本外 IC 0.117 / 前10
     日超额 +1.40%，基线 0.022 / +0.69%。缺口在**函数形式**，不在参数。
  3. 擂台胜者恰好是**线性模型**——每个系数可打印、可归因、可用人话讨论。
     「不许黑箱排序」的红线拦的是不可解释，不拦线性回归。

于是最优架构是让老师转正——但**不直接转**：

    影子模式    每天和生产打分器并排跑，只记账不发信不排产。
    真值积累    在线快照（真 T1/T2/T3）上逐日记录双方 IC 和前10超额。
    人工切换    影子在真值上显著领先、且积累够天数后，作为**提案**
                摆到用户面前。切不切换是用户的决定，不是机器的。

确定性保证：推理 = 当日截面秩 × 固定系数向量。系数存 JSON、进 git，
给定系数文件，任何人可逐位复现当天排名。和 score.py 的可复现性同级。

训练的走向前纪律和主线一致；refit 频率低（每次 stage_learn 顺带），
系数文件带 fitted_at 和训练窗口，审计链完整。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = ROOT / "state" / "shadow_model.json"

# 和擂台同一张特征表（learn/model_select.py::FEATURES），刻意不另起炉灶：
# 影子的正当性来自「它就是擂台上赢的那个东西」，特征一换比较就失效了。


def fit(df: pd.DataFrame) -> dict | None:
    """在带 ytil 的训练表上拟合 RankHuber，落系数文件。

    失败返回 None 不抛——影子是研究性组件，任何失败都不能影响主流程。
    """
    try:
        from sklearn.linear_model import HuberRegressor
        from learn.model_select import FEATURES, prep_features, rank_norm
        d = prep_features(df)
        X = rank_norm(d[FEATURES], d["date"]).fillna(0.0)
        y = d["ytil"].to_numpy(float)
        m = HuberRegressor(alpha=1e-3, max_iter=500).fit(X, y)
        doc = {
            "kind": "RankHuber",
            "fitted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "train_days": int(d["date"].nunique()),
            "train_rows": int(len(d)),
            "features": list(FEATURES),
            "coef": {f: float(c) for f, c in zip(FEATURES, m.coef_)},
            "intercept": float(m.intercept_),
        }
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        import os
        os.replace(tmp, MODEL)
        top = sorted(doc["coef"].items(), key=lambda x: -abs(x[1]))[:5]
        log.info("影子模型已拟合（%d 天）。最大五个系数：%s",
                 doc["train_days"],
                 ", ".join(f"{k}={v:+.3f}" for k, v in top))
        return doc
    except Exception as e:  # noqa: BLE001
        log.warning("影子模型拟合失败（不影响主流程）: %s", e)
        return None


def load() -> dict | None:
    try:
        return json.loads(MODEL.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def score(df: pd.DataFrame, model: dict | None = None) -> np.ndarray | None:
    """当日截面打分。纯线性：秩归一 × 系数。没有模型文件返回 None。"""
    model = model or load()
    if not model:
        return None
    try:
        from learn.model_select import prep_features, rank_norm
        d = prep_features(df)
        feats = model["features"]
        X = rank_norm(d[feats], d["date"]).fillna(0.0)
        w = np.array([model["coef"][f] for f in feats])
        return X.to_numpy(float) @ w + model["intercept"]
    except Exception as e:  # noqa: BLE001
        log.warning("影子打分失败: %s", e)
        return None


def daily_compare(df_online: pd.DataFrame, base_scores: np.ndarray,
                  base_rej: np.ndarray, top_k: int = 10) -> list[dict]:
    """在线真值天上的双榜逐日对比。喂给面板和切换提案。

    影子也套用同一份硬性排除（准入是政策层，对两个排序器一视同仁）。
    """
    from learn.model_select import spearman
    sh = score(df_online)
    if sh is None:
        return []
    out = []
    d = df_online.assign(_b=np.where(base_rej, -np.inf, base_scores),
                         _s=np.where(base_rej, -np.inf, sh))
    for day, g in d.groupby("date"):
        ok = g[np.isfinite(g["_b"])]
        if len(ok) < 5:
            continue
        row = {"date": day}
        for tag, col in (("base", "_b"), ("shadow", "_s")):
            top = ok.nlargest(top_k, col)
            row[f"{tag}_ic"] = spearman(ok[col].to_numpy(),
                                        ok["ytil"].to_numpy())
            row[f"{tag}_top_excess"] = float(top["y"].mean())
        row["overlap"] = len(set(ok.nlargest(top_k, "_b")["code"])
                             & set(ok.nlargest(top_k, "_s")["code"])) / top_k
        out.append(row)
    return out
