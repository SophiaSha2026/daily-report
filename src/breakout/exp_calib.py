"""
回答一个问题：分数越高，涨超 50% 的可能性是不是越大？

这是用户会拿着清单直接问的问题，必须用数据回答，不能靠「模型就该是这样」。
如果分档命中率不是单调上升的，那分数就只是个排序号，「92 分比 85 分更值得
买」这句话就不成立，界面上也不该那么暗示。

做法：在验证集那 10 个月上逐月滚动（每月用之前的数据重训，预测该月），
把**全市场每只票**的分数和实际结果都记下来，再按分数分档统计。
不碰封存的那 9 个月 —— 它只许用一次，已经在最终验收时用掉了。

    python src/breakout/exp_calib.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

import arena as A       # noqa: E402
import fselect as FS    # noqa: E402
import model as M       # noqa: E402
import validate as V    # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
OUT = ROOT / "out_breakout"


def main() -> int:
    df = A.load(False)                       # 封存数据已被切掉，读不到
    feats_all = [c for c in df.columns if "__" in c]
    tr0 = df[df["date"] < V.TRAIN_END]
    feats = FS.run(tr0, feats_all, y="y_up")["keep"]
    print(f"特征 {len(feats)} 个，逐月滚动 {V.TRAIN_END} .. {V.VALID_END}\n")

    months = sorted({d[:7] for d in df["date"]
                     if V.TRAIN_END <= d < V.VALID_END})
    rows = []
    for m in months:
        tr = df[df["date"] < m + "-01"]
        te = df[df["date"].str[:7] == m].copy()
        tr = tr[np.isfinite(tr["y_up"])]
        te = te[np.isfinite(te["y_up"])]
        if len(tr) < 5000 or not len(te):
            continue
        trs = M.stratified_sample(tr, "y_up")
        mdl = M.L1Lgbm(n_estimators=400).fit(trs, feats, "y_up")
        p = mdl.predict_proba(te)
        # 分数刻度和线上一致：训练集分布的百分位
        q = np.quantile(mdl.predict_proba(trs), np.linspace(0, 1, 101))
        # 板块校正也要带上，否则算的不是线上真正用的分数
        import daily as D
        adj = te["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
        score = np.clip(np.searchsorted(q, p * adj), 0, 100)
        rows.append(pd.DataFrame({"month": m, "score": score,
                                  "y": te["y_up"].to_numpy(float)}))
        print(f"  {m} 完成（{len(te)} 只）")

    d = pd.concat(rows, ignore_index=True)
    print(f"\n合计 {len(d):,} 个 (股,日) 样本\n")

    bins = [(0, 50), (50, 70), (70, 80), (80, 90), (90, 95), (95, 99),
            (99, 101)]
    print(f"{'分数段':>10s} {'样本数':>10s} {'涨超50%的':>10s} {'占比':>8s} "
          f"{'相对随便买':>10s}")
    base = float(d["y"].mean())
    out = []
    for lo, hi in bins:
        g = d[(d["score"] >= lo) & (d["score"] < hi)]
        if not len(g):
            continue
        hit = float(g["y"].mean())
        out.append({"lo": lo, "hi": hi, "n": int(len(g)),
                    "hit": hit, "lift": hit / base})
        print(f"{lo:4d}~{hi - 1:<5d} {len(g):10,d} {int(g['y'].sum()):10,d} "
              f"{100 * hit:7.2f}% {hit / base:9.2f}x")
    print(f"{'全市场':>10s} {len(d):10,d} {int(d['y'].sum()):10,d} "
          f"{100 * base:7.2f}% {1.0:9.2f}x")

    hits = [o["hit"] for o in out]
    mono = all(hits[i] <= hits[i + 1] for i in range(len(hits) - 1))
    print(f"\n分档命中率单调上升：{'是' if mono else '否'}")
    if not mono:
        bad = [f"{out[i]['lo']}~{out[i]['hi']-1} 档({100*hits[i]:.1f}%) "
               f"高于 {out[i+1]['lo']}~{out[i+1]['hi']-1} 档({100*hits[i+1]:.1f}%)"
               for i in range(len(hits) - 1) if hits[i] > hits[i + 1]]
        for b in bad:
            print("  倒挂：" + b)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score_calibration.json").write_text(
        json.dumps({"base": base, "monotonic": mono, "bins": out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'score_calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
