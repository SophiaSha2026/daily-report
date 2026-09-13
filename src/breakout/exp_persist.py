"""
最大化准确率。两个此前没实现的机制，都来自用户最初的指令原文：

  1. 「如果当天没有符合要求的个股，清单可以为空」
     -> 清单该是**绝对门槛制**（够格才上），不是「每天固定取前 N 名」。
        之前一直用 top-N，弱势日也硬凑满 10 只，纯粹拉低准确率。

  2. 「如某个个股持续符合条件，可以连续几个交易日推荐」
     -> 连续多天都够格的票，可能比只亮一天的更可靠。这一条完全没实现过。

所以测：门槛 × 连续确认天数 的组合，看准确率能推到多高。

    python src/breakout/exp_persist.py
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
        trs = M.stratified_sample(tr, "y_up")
        mdl = M.L1Lgbm(n_estimators=400).fit(trs, feats, "y_up")
        q = np.quantile(mdl.predict_proba(trs), np.linspace(0, 1, 101))
        adj = te["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
        te["_p"] = mdl.predict_proba(te) * adj
        te["score"] = np.clip(np.searchsorted(q, te["_p"]), 0, 100)
        te = te.sort_values(["date", "_p"], ascending=[True, False])
        te["rank"] = te.groupby("date").cumcount() + 1
        keep.append(te[te["rank"] <= 30][["date", "code", "rank",
                                          "score", "y_up"]])
        print(f"  {m} 完成")

    d = pd.concat(keep, ignore_index=True).sort_values(["code", "date"])
    days = d["date"].nunique()
    base = float(df["y_up"].mean(skipna=True))
    all_dates = sorted(d["date"].unique())
    di = {x: i for i, x in enumerate(all_dates)}
    d["_i"] = d["date"].map(di)
    print(f"\n{days} 个交易日，全市场基准 {100*base:.2f}%\n")

    def with_streak(sub: pd.DataFrame) -> pd.DataFrame:
        """给每行算「到今天为止连续够格几天」。

        连续的定义是交易日相邻（_i 差 1）。中间断一天就重新计数 ——
        「持续符合条件」指的是没断过，不是「这几天里有几天符合」。
        """
        sub = sub.sort_values(["code", "_i"]).copy()
        streak = np.ones(len(sub), dtype=int)
        codes = sub["code"].to_numpy()
        idx = sub["_i"].to_numpy()
        for k in range(1, len(sub)):
            if codes[k] == codes[k - 1] and idx[k] == idx[k - 1] + 1:
                streak[k] = streak[k - 1] + 1
        sub["streak"] = streak
        return sub

    print(f"{'规则':>30s} {'选中':>6s} {'空仓天':>7s} {'每天':>5s} "
          f"{'准确率':>8s} {'倍数':>7s} {'每月命中':>9s}")
    rows = []
    for thr in (95, 97, 98, 99):
        for topn in (1, 3, 30):        # 30 = 实际上不限名次，纯看门槛
            sub = d[(d["score"] >= thr) & (d["rank"] <= topn)]
            if not len(sub):
                continue
            sub = with_streak(sub)
            for need in (1, 2, 3):
                g = sub[sub["streak"] >= need]
                if len(g) < 20:
                    continue
                hit = float(g["y_up"].mean())
                nd = g["date"].nunique()
                per_day = len(g) / days
                lbl = (f"≥{thr}分" + (f" 且前{topn}名" if topn < 30 else "")
                       + (f" 且连续{need}天" if need > 1 else ""))
                rows.append({"thr": thr, "top": topn, "need": need,
                             "n": int(len(g)), "empty": int(days - nd),
                             "per_day": per_day, "hit": hit,
                             "lift": hit / base,
                             "per_month": 21 * per_day * hit})
                print(f"{lbl:>30s} {len(g):6d} {days-nd:7d} {per_day:5.1f} "
                      f"{100*hit:7.2f}% {hit/base:6.2f}x "
                      f"{21*per_day*hit:8.1f} 只")

    rows.sort(key=lambda r: -r["hit"])
    print("\n准确率最高的 8 个（只看样本 >= 50 的，否则数字不稳）：")
    for r in [x for x in rows if x["n"] >= 50][:8]:
        lbl = (f"≥{r['thr']}分" + (f" 前{r['top']}名" if r['top'] < 30 else "")
               + (f" 连续{r['need']}天" if r['need'] > 1 else ""))
        se = float(np.sqrt(r["hit"] * (1 - r["hit"]) / r["n"]))
        print(f"  {lbl:<24s} {100*r['hit']:5.2f}% ±{100*se:.2f}  "
              f"{r['lift']:.2f}倍  样本 {r['n']:4d}  "
              f"每天 {r['per_day']:.2f} 只  每月命中 {r['per_month']:.1f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "persist_grid.json").write_text(
        json.dumps({"days": days, "base": base, "grid": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {OUT / 'persist_grid.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
