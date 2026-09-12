"""
分级响应第 1 级：加特征（放宽筛选阈值）+ 清单长度对命中率的影响。

两件事一起测，都只测一次（设计文档 2.2 的规矩：禁止反复微调直到达标）。

一、放宽特征筛选
    当前 IC_MIN=0.01 把 87 个特征砍到 19 个。砍得太狠的话，
    单个 IC 弱但组合起来有用的特征就丢了 —— GBDT 恰恰擅长用这种组合。
    测 IC_MIN = 0.01 / 0.005 / 0.002 三档。

二、清单长度 top_n
    当前按天取 top 10。取 5 只会更精选（命中率更高、清单更短），
    取 20 只更全（命中率更低）。这**不是调参**，是产品形态选择：
    「每天给几只」本来就该由用户定，这里只是把取舍摆出来。

    注意 top_n 变小会让命中率天然变高，所以它不能用来「达到验收线」——
    验收线是按 top 10 定的。这里出的数据是给用户做决定用的。

    python src/breakout/exp_widen.py
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

    print("=== 一、放宽特征筛选（L1 LightGBM，top10）===")
    print(f"{'IC下限':>8s} {'特征':>4s} {'命中':>7s} {'倍数':>6s} "
          f"{'剔极端月':>8s} {'耗时':>6s}")
    best = None
    for ic_min in (0.010, 0.005, 0.002):
        t0 = time.time()
        FS.IC_MIN = ic_min
        rep = FS.run(tr, feats_all, y="y_up")
        feats = rep["keep"]
        bm, extra = V.walk_forward(df, feats,
                                   lambda: M.L1Lgbm(n_estimators=400))
        acc = V.acceptance(bm, extra["picks"], df)
        print(f"{ic_min:8.3f} {len(feats):4d} {100*acc['overall_hit']:6.2f}% "
              f"{acc['lift']:5.2f}x {100*acc['hit_ex_extreme']:7.2f}% "
              f"{time.time()-t0:5.0f}s")
        if best is None or acc["overall_hit"] > best[1]["overall_hit"]:
            best = (feats, acc, ic_min)
    FS.IC_MIN = 0.010

    print(f"\n=== 二、清单长度（用上面最好的那组特征，IC下限 {best[2]}）===")
    print(f"{'每天几只':>8s} {'命中':>7s} {'倍数':>6s} {'每月期望命中':>12s}")
    for n in (5, 10, 20, 30):
        bm, extra = V.walk_forward(df, best[0],
                                   lambda: M.L1Lgbm(n_estimators=400),
                                   top_n=n)
        acc = V.acceptance(bm, extra["picks"], df)
        # 一个月约 21 个交易日
        exp = 21 * n * acc["overall_hit"]
        print(f"{n:8d} {100*acc['overall_hit']:6.2f}% {acc['lift']:5.2f}x "
              f"{exp:11.1f} 只")
    return 0


if __name__ == "__main__":
    sys.exit(main())
