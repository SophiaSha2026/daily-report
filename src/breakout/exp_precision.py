"""
用户要求：哪怕降低召回，也要把精确度提上去，一切以少出错为最优先。

现在的规则是「每天取前 10 名」，它有个毛病：不管当天多弱都要凑满 10 只。
弱势日里第 10 名可能只有 60 分，硬塞进清单纯粹是拉低精确度。

所以测两个旋钮的组合：
    排名门槛   只取前 N 名
    分数门槛   分数低于 T 的一律不要，宁可当天清单为空

「清单为空」不是故障，是这套要求下的正常输出 —— 弱势日本来就不该买。
所以下面除了精确度，还报「多少天是空的」，让人看见代价。

    python src/breakout/exp_precision.py
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
    df = A.load(False)
    feats_all = [c for c in df.columns if "__" in c]
    feats = FS.run(df[df["date"] < V.TRAIN_END], feats_all, y="y_up")["keep"]
    import daily as D

    months = sorted({d[:7] for d in df["date"]
                     if V.TRAIN_END <= d < V.VALID_END})
    keep = []
    for m in months:
        tr = df[df["date"] < m + "-01"]
        te = df[df["date"].str[:7] == m].copy()
        tr = tr[np.isfinite(tr["y_up"])]
        te = te[np.isfinite(te["y_up"])]
        if len(tr) < 5000 or not len(te):
            continue
        mdl = M.L1Lgbm(n_estimators=400).fit(
            M.stratified_sample(tr, "y_up"), feats, "y_up")
        p = mdl.predict_proba(te)
        q = np.quantile(mdl.predict_proba(M.stratified_sample(tr, "y_up")),
                        np.linspace(0, 1, 101))
        adj = te["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
        # 和线上一致：排序用连续值，分数只是显示用的整数刻度。
        # 按整数分数排序会因为大量并列而选错第 1 名，见 daily.py 的注释。
        te["_p"] = p * adj
        te["score"] = np.clip(np.searchsorted(q, te["_p"]), 0, 100)
        te = te.sort_values(["date", "_p"], ascending=[True, False])
        te["rank"] = te.groupby("date").cumcount() + 1
        keep.append(te[te["rank"] <= 20][["date", "rank", "score", "y_up"]])
        print(f"  {m} 完成")

    d = pd.concat(keep, ignore_index=True)
    days = d["date"].nunique()
    base = float(df["y_up"].mean(skipna=True))
    print(f"\n{days} 个交易日，全市场基准 {100*base:.2f}%\n")

    print(f"{'规则':>22s} {'选中':>7s} {'空仓天':>7s} {'每天':>6s} "
          f"{'精确度':>8s} {'倍数':>7s} {'每月命中':>9s}")
    rows = []
    for topn in (1, 2, 3, 5, 10):
        for thr in (0, 90, 95, 97, 99):
            g = d[(d["rank"] <= topn) & (d["score"] >= thr)]
            if not len(g):
                continue
            hit = float(g["y_up"].mean())
            nd = g["date"].nunique()
            empty = days - nd
            per_day = len(g) / days
            rows.append({"top": topn, "thr": thr, "n": int(len(g)),
                         "empty_days": int(empty), "per_day": per_day,
                         "hit": hit, "lift": hit / base,
                         "per_month": 21 * per_day * hit})
            lbl = f"前{topn}名" + (f" 且 >={thr}分" if thr else "")
            print(f"{lbl:>22s} {len(g):7d} {empty:7d} {per_day:6.1f} "
                  f"{100*hit:7.2f}% {hit/base:6.2f}x "
                  f"{21*per_day*hit:8.1f} 只")

    # 精确度最高的几个
    rows.sort(key=lambda r: -r["hit"])
    print("\n精确度最高的 5 个规则：")
    for r in rows[:5]:
        lbl = f"前{r['top']}名" + (f" 且 >={r['thr']}分" if r['thr'] else "")
        print(f"  {lbl:<20s} 精确度 {100*r['hit']:5.2f}%  {r['lift']:.2f} 倍  "
              f"平均每天 {r['per_day']:.2f} 只  "
              f"{r['empty_days']}/{days} 天空仓")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_grid.json").write_text(
        json.dumps({"days": days, "base": base, "grid": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'precision_grid.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
