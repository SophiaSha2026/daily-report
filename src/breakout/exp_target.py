"""
分级响应第一步：修正训练目标和评估目标的错配。

问题
----
第一版设计里训练用 y_t0（起涨第一天），评估用 y_up（当天买入未来 20 日
涨超 50%）。理由是「y_t0 信号干净，一段行情只有一个起涨点」。

擂台跑完发现这个选择可能是错的：起涨后第 2、3 天买入**同样**能涨 50%，
但训练时它们被当成负样本，模型学会了排斥它们。而评估口径恰恰把它们算作
正确答案。训练目标和评估目标不一致，模型再强也没用。

这不是调参，是修正逻辑错配，属于分级响应的正当一步。
结果（无论好坏）记入 docs/breakout_log.md。

    python src/breakout/exp_target.py
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arena as A          # noqa: E402
import fselect as FS       # noqa: E402
import model as M          # noqa: E402
import validate as V       # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main() -> int:
    df = A.load(False)
    feats_all = [c for c in df.columns if "__" in c]
    tr = df[df["date"] < V.TRAIN_END]
    print("实验：训练目标 y_t0 vs y_up（评估一律 y_up，走向前协议不变）\n")
    print(f"{'训练目标':8s} {'特征':>4s} {'命中':>7s} {'倍数':>6s} "
          f"{'剔极端月':>8s} {'零命中月':>8s} {'耗时':>6s}")
    rows = []
    for ytr in ["y_t0", "y_up"]:
        t0 = time.time()
        rep = FS.run(tr, feats_all, y=ytr)
        feats = rep["keep"]
        if len(feats) < 3:
            print(f"{ytr:8s} 特征只剩 {len(feats)} 个，跳过")
            continue
        bm, extra = V.walk_forward(
            df, feats, lambda: M.L1Lgbm(n_estimators=400),
            y_train=ytr, y_eval="y_up", start=V.TRAIN_END, end=V.VALID_END)
        acc = V.acceptance(bm, extra["picks"], df)
        print(f"{ytr:8s} {len(feats):4d} {100*acc['overall_hit']:6.2f}% "
              f"{acc['lift']:5.2f}x {100*acc['hit_ex_extreme']:7.2f}% "
              f"{acc['zero_months']:8d} {time.time()-t0:5.0f}s")
        rows.append((ytr, acc, bm))

    if len(rows) == 2:
        a, b = rows[0][1]["overall_hit"], rows[1][1]["overall_hit"]
        print(f"\n结论：y_up 训练相对 y_t0 {100*(b-a):+.2f} 个百分点")
        if b > a * 1.15:
            print("      错配确实存在，后续一律用 y_up 当训练目标")
        elif b < a * 0.85:
            print("      y_t0 反而更好，说明干净信号的价值大过口径一致")
        else:
            print("      差别不大，说明瓶颈不在这里，要往特征上找")
    return 0


if __name__ == "__main__":
    sys.exit(main())
