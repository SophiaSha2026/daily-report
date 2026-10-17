"""
规则核（单调逻辑回归）逐月走向前 + 一行字规则 ret20 参照。两段（筛选集 2022-01..2024-05、开发集 2024-06..2025-12）。

规则核 s0 = sigmoid( b0 + w1·z(log(1+ret20)) + w2·z(vol60) + w3·z(turn20) + w4·z(amp20) + g_board )
    · 四个连续输入取原始刻度，当日内（候选池 elig0 上）减均值除标准差，不做百分位，不中性化，不截尾
    · w1..w4 >= 0（单调约束，L-BFGS-B 的 bounds），板块三个哑变量（main 为基）无约束
    · 训练标签 y_open（和决策尺子同一口径），样本权重按月等权（model.month_weights，和生产同一口径），
      L2 = 1e-3·||w||²（只为数值稳定）
    · 每个测试月重拟合：训练行 = 候选池 elig0 且有标签 且 common_rc.train_mask（validate.train_slice 的净化 + 3 年窗口）
    · 出票：每天候选池（elig0 且事件剔除后）按 s0 取前 10，没有分数门槛。当日内 z 化之后 s0 只有当天内的相对意义，
      概率门槛等价于名额，所以不设门槛（见 prereg 的说明）
参照 ret20：每天按 lret20 取前 10（同一候选池）。

产物 data/breakout/raw/exp/rulecore_v1/all_<臂>.parquet（date code p，全部候选池行）、coef_rc.json（每月系数）。
    PYTHONIOENCODING=utf-8 python research/rulecore/rulecore.py [--seg screen dev] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_rc as C  # noqa: E402

L2 = 1e-3
OUT = C.EXP / C.NAME


def design(df: pd.DataFrame) -> np.ndarray:
    Z = df[["z_" + c for c in C.INPUTS]].to_numpy(float)
    B = np.stack([(df["board"] == b).to_numpy(float) for b in C.BOARDS[1:]], axis=1)
    return np.concatenate([Z, B, np.ones((len(df), 1))], axis=1)


def fit_monotone(X: np.ndarray, y: np.ndarray, w: np.ndarray, n_mono: int = len(C.INPUTS)) -> np.ndarray:
    """带权逻辑回归，前 n_mono 个系数 >= 0。返回系数向量（含截距在最后）。"""
    y = y.astype(float)
    w = w / w.mean()
    k = X.shape[1]

    def f(beta):
        s = X @ beta
        # log(1+exp(s)) 稳定写法
        lp = np.logaddexp(0.0, s)
        loss = float((w * (lp - y * s)).sum()) + L2 * float((beta[:-1] ** 2).sum())
        p = 1.0 / (1.0 + np.exp(-s))
        grad = X.T @ (w * (p - y))
        grad[:-1] += 2 * L2 * beta[:-1]
        return loss, grad

    b0 = np.zeros(k)
    b0[-1] = np.log(max(y.mean(), 1e-4) / (1 - min(y.mean(), 0.9999)))
    bounds = [(0.0, None)] * n_mono + [(None, None)] * (k - n_mono)
    r = minimize(f, b0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    if not r.success:
        C.say("  警告：L-BFGS-B 未收敛：%s" % r.message)
    return r.x


def run_segment(df: pd.DataFrame, seg: str, smoke: bool = False) -> tuple[pd.DataFrame, dict]:
    months = C.months_in(seg)
    if smoke:
        months = months[:2]
    parts, coefs = [], {}
    t0 = time.time()
    for m in months:
        tm = C.train_mask(df, m)
        tr = df[tm & df["elig0"].to_numpy() & np.isfinite(df["y_open"]).to_numpy()]
        te = df[(df["date"].str[:7] == m).to_numpy() & df["elig0"].to_numpy()]
        if len(tr) < 5000 or not len(te):
            C.say("  %s 训练 %d 行 / 测试 %d 行，跳过" % (m, len(tr), len(te)))
            continue
        import model as M
        w = M.month_weights(tr)
        beta = fit_monotone(design(tr), tr["y_open"].to_numpy(), w)
        s = design(te) @ beta
        p = 1.0 / (1.0 + np.exp(-s))
        parts.append(pd.DataFrame({"date": te["date"].to_numpy(), "code": te["code"].to_numpy(), "p": p.astype(np.float32)}))
        coefs[m] = {"n_train": int(len(tr)), "pos": float(tr["y_open"].mean()),
                    **{k: float(v) for k, v in zip(list(C.INPUTS) + ["b_" + b for b in C.BOARDS[1:]] + ["intercept"], beta)}}
        C.say("  %s 训练 %d 行（正 %.2f%%）测试 %d 行  w=%s  %.0fs" % (
            m, len(tr), 100 * tr["y_open"].mean(), len(te), np.round(beta[:4], 3), time.time() - t0))
    return pd.concat(parts, ignore_index=True), coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="*", default=["screen", "dev"])
    ap.add_argument("--smoke", action="store_true", help="每段只跑 2 个月，不落正式产物，只印系数")
    a = ap.parse_args()
    if C.in_prod_window():
        raise SystemExit("北京 16:00~18:30 不跑")
    df = C.build_features()
    df["date"] = df["date"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    allp, allc = [], {}
    for seg in a.seg:
        C.say("== 规则核 %s %s..%s ==" % (seg, *C.SEGS[seg]))
        p, c = run_segment(df, seg, a.smoke)
        allp.append(p)
        allc.update(c)
    if a.smoke:
        C.say("smoke 完成（不落产物）")
        return
    p = pd.concat(allp, ignore_index=True)
    p.to_parquet(OUT / "all_rc.parquet", index=False)
    (OUT / "coef_rc.json").write_text(json.dumps(allc, ensure_ascii=False, indent=1), encoding="utf-8")
    # 参照：ret20 一行字（候选池内按 lret20 排）
    lo, hi = C.SEGS[a.seg[0]][0] + "-01", C.month_shift(C.SEGS[a.seg[-1]][1], 1) + "-01"
    te = df[(df["date"] >= lo) & (df["date"] < hi) & df["elig0"]]
    pd.DataFrame({"date": te["date"].to_numpy(), "code": te["code"].to_numpy(),
                  "p": te["lret20"].to_numpy(np.float32)}).to_parquet(OUT / "all_ret20.parquet", index=False)
    C.say("-> %s（规则核 %d 行，参照 %d 行）" % (OUT, len(p), len(te)))
    C.progress("规则核走向前跑完：%s，all_rc / all_ret20 落 %s" % (a.seg, OUT))


if __name__ == "__main__":
    main()
