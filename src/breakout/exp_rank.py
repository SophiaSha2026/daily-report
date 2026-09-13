"""
回答：只买清单第一名，是不是命中率最高？买前几名最划算？

分数段的单调性（exp_calib.py）说明「分数越高越准」，但那是**绝对分数**。
用户实际面对的是每天的清单，问的是**当天排名**：第 1 名是不是比第 5 名强。
这两件事不一样，必须单独测。

做法和 exp_calib 一样：验证集 10 个月逐月滚动，但这次记录每天 top 20 的
**名次**，再按名次统计。不碰封存的那 9 个月。

样本量要留意：10 个月约 200 个交易日，所以「第 1 名」这一档只有 200 个
样本。命中率的波动会很大，不能看见 18% 就说第一名最准。
所以下面同时给二项分布的标准误。

    python src/breakout/exp_rank.py
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
TOP = 20


def main() -> int:
    df = A.load(False)
    feats_all = [c for c in df.columns if "__" in c]
    feats = FS.run(df[df["date"] < V.TRAIN_END], feats_all, y="y_up")["keep"]
    import daily as D

    months = sorted({d[:7] for d in df["date"]
                     if V.TRAIN_END <= d < V.VALID_END})
    picks = []
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
        adj = te["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
        te["p"] = p * adj
        te = te.sort_values(["date", "p"], ascending=[True, False])
        te["rank"] = te.groupby("date").cumcount() + 1
        picks.append(te[te["rank"] <= TOP][["date", "rank", "y_up"]])
        print(f"  {m} 完成")

    d = pd.concat(picks, ignore_index=True)
    days = d["date"].nunique()
    base = float(df["y_up"].mean(skipna=True))
    print(f"\n{days} 个交易日，每天取前 {TOP} 名\n")

    print("按名次看（每档样本数 = 交易日数）")
    print(f"{'名次':>6s} {'样本':>6s} {'命中':>8s} {'±标准误':>9s} {'相对随便买':>10s}")
    rows = []
    for r in range(1, TOP + 1):
        g = d[d["rank"] == r]
        if not len(g):
            continue
        hit = float(g["y_up"].mean())
        se = float(np.sqrt(hit * (1 - hit) / len(g)))
        rows.append({"rank": r, "n": len(g), "hit": hit, "se": se})
        if r <= 10 or r == TOP:
            print(f"{r:6d} {len(g):6d} {100*hit:7.2f}% {100*se:8.2f}% "
                  f"{hit/base:9.2f}x")

    print(f"\n买前 N 名（这才是实际决策）")
    print(f"{'买前几名':>8s} {'总样本':>7s} {'命中':>8s} {'±标准误':>9s} "
          f"{'每月大约命中':>12s}")
    cum = []
    for n in (1, 2, 3, 5, 10, 15, 20):
        g = d[d["rank"] <= n]
        hit = float(g["y_up"].mean())
        se = float(np.sqrt(hit * (1 - hit) / len(g)))
        per_month = 21 * n * hit
        cum.append({"top": n, "n": len(g), "hit": hit, "se": se,
                    "per_month": per_month})
        print(f"{n:8d} {len(g):7d} {100*hit:7.2f}% {100*se:8.2f}% "
              f"{per_month:9.1f} 只")

    # 第 1 名和第 10 名的差，够不够一个标准误
    r1 = next(x for x in rows if x["rank"] == 1)
    r10 = next(x for x in rows if x["rank"] == 10)
    diff = r1["hit"] - r10["hit"]
    pooled = float(np.sqrt(r1["se"] ** 2 + r10["se"] ** 2))
    print(f"\n第 1 名 vs 第 10 名：{100*diff:+.2f} 个百分点，"
          f"合并标准误 {100*pooled:.2f}")
    print("结论：" + ("第 1 名确实更准，差距超过了噪音"
                     if diff > 2 * pooled else
                     "差距在噪音范围内，不能说第 1 名一定比第 10 名准"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rank_analysis.json").write_text(
        json.dumps({"days": days, "base": base, "by_rank": rows,
                    "cumulative": cum}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n-> {OUT / 'rank_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
