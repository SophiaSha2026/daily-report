"""
清单 A 改成「看过去 5 个交易日（含当天）」能到多少准确率。

用户 2026-09-13 的问题。当前规则只看**当天**的分数，连续够格天数只用来
排序；这里测的是把 5 天窗口本身当成入选条件的几种口径：

    W1  当天够格 + 窗口内够格 >= k 次（不要求相邻，断了也算）
    W2  当天够格 + 连续够格 >= k 天（现规则，做对照）
    W3  不要求当天够格，只要窗口内 >= k 次
    W4  窗口内 5 天预测值的均值排名前 N（完全不用当天分数当门槛）

评估口径和实验 7 完全一致：验证集 2025-03..2025-12 逐月走向前，
y_up = 未来 20 个交易日涨超 50%，全市场基准 3.55%。

打分结果缓存到 data/breakout/raw/wf_scores.parquet（已 gitignore），
改口径不用重训模型。WF_CACHE 环境变量可换路径。

    python src/breakout/exp_window.py            用缓存（没有就先算）
    python src/breakout/exp_window.py --refit    强制重算打分
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
CACHE = Path(os.environ.get("WF_CACHE", str(ROOT / "data" / "breakout" / "raw"
                                            / "wf_scores.parquet")))
RANK_KEEP = 100     # 每天落盘前多少名。>=95 分的都在这里面，足够算窗口


def build_cache() -> tuple[pd.DataFrame, float]:
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
        keep.append(te[te["rank"] <= RANK_KEEP][
            ["date", "code", "rank", "score", "_p", "y_up"]])
        print("  " + m + " 完成", flush=True)

    d = pd.concat(keep, ignore_index=True)
    base = float(df["y_up"].mean(skipna=True))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    (CACHE.parent / "wf_base.json").write_text(
        json.dumps({"base": base}), encoding="utf-8")
    return d, base


def load_cache(refit: bool) -> tuple[pd.DataFrame, float]:
    if CACHE.exists() and not refit:
        d = pd.read_parquet(CACHE)
        base = json.loads((CACHE.parent / "wf_base.json")
                          .read_text(encoding="utf-8"))["base"]
        return d, base
    return build_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--win", type=int, default=5)
    a = ap.parse_args()
    d, base = load_cache(a.refit)

    all_dates = sorted(d["date"].unique())
    di = {x: i for i, x in enumerate(all_dates)}
    d["_i"] = d["date"].map(di)
    days = len(all_dates)
    W = a.win
    print("\n验证集 %d 个交易日（%s ~ %s），全市场基准 %.2f%%，窗口 %d 天\n"
          % (days, all_dates[0], all_dates[-1], 100 * base, W))

    rows = []

    def rec(label: str, g: pd.DataFrame, kind: str) -> None:
        if len(g) < 20:
            return
        hit = float(g["y_up"].mean())
        nd = g["date"].nunique()
        per_day = len(g) / days
        se = float(np.sqrt(hit * (1 - hit) / len(g)))
        rows.append({"kind": kind, "label": label, "n": int(len(g)),
                     "empty": int(days - nd), "per_day": per_day,
                     "hit": hit, "se": se, "lift": hit / base,
                     "per_month": 21 * per_day * hit})

    # ---- 窗口内够格次数 ----
    for thr in (95, 97, 98):
        ok = d[d["score"] >= thr]
        by_code = {c: np.sort(g["_i"].to_numpy())
                   for c, g in ok.groupby("code")}
        sub = ok.copy()
        cnt = np.zeros(len(sub), dtype=int)      # 窗口内够格次数（含当天）
        streak = np.zeros(len(sub), dtype=int)   # 连续够格天数（含当天）
        codes = sub["code"].to_numpy()
        idx = sub["_i"].to_numpy()
        for j in range(len(sub)):
            arr = by_code[codes[j]]
            i = int(idx[j])
            cnt[j] = int(((arr > i - W) & (arr <= i)).sum())
            s = 1
            aset = set(int(x) for x in arr)
            while (i - s) in aset:
                s += 1
            streak[j] = s
        sub["cnt"] = cnt
        sub["streak"] = streak

        for k in range(1, W + 1):
            rec("≥%d分 且 近%d日内够格≥%d次" % (thr, W, k),
                sub[sub["cnt"] >= k], "W1")
        for k in range(1, 6):
            rec("≥%d分 且 连续≥%d天" % (thr, k),
                sub[sub["streak"] >= k], "W2")
        for k in range(1, W + 1):
            rec("≥%d分 且 近%d日内恰好%d次" % (thr, W, k),
                sub[sub["cnt"] == k], "W1x")

    # ---- W3：不要求当天够格，窗口内 >= k 次就上 ----
    for thr in (97,):
        ok = d[d["score"] >= thr]
        by_code = {c: np.sort(g["_i"].to_numpy())
                   for c, g in ok.groupby("code")}
        base_rows = d[["code", "_i", "date", "y_up", "score"]]
        pool = set()
        for c, arr in by_code.items():
            for i in arr:
                for off in range(0, W):
                    pool.add((c, int(i) + off))
        pl = pd.DataFrame(list(pool), columns=["code", "_i"])
        m = pl.merge(base_rows, on=["code", "_i"], how="inner")
        cn = []
        for c, i in zip(m["code"].to_numpy(), m["_i"].to_numpy()):
            arr = by_code[c]
            cn.append(int(((arr > int(i) - W) & (arr <= int(i))).sum()))
        m["cnt"] = cn
        for k in range(1, W + 1):
            rec("近%d日内够格≥%d次（当天可不够格）" % (W, k),
                m[m["cnt"] >= k], "W3")

    # ---- W4：5 日平均预测值排名 ----
    dd = d.sort_values(["code", "_i"]).copy()
    dd["pm"] = (dd.groupby("code")["_p"]
                .transform(lambda s: s.rolling(W, min_periods=W).mean()))
    dd = dd[np.isfinite(dd["pm"])]
    dd = dd.sort_values(["date", "pm"], ascending=[True, False])
    dd["mrank"] = dd.groupby("date").cumcount() + 1
    for n in (1, 3, 5, 10):
        rec("近%d日均值 前%d名" % (W, n), dd[dd["mrank"] <= n], "W4")

    order = {"W1": 0, "W1x": 1, "W2": 2, "W3": 3, "W4": 4}
    rows.sort(key=lambda r: (order[r["kind"]], -r["hit"]))
    print("%-34s%6s%6s%7s%10s%7s%7s%9s"
          % ("规则", "样本", "每天", "空仓天", "准确率", "±", "倍数", "每月命中"))
    last = None
    for r in rows:
        if r["kind"] != last:
            print("-" * 86)
            last = r["kind"]
        print("%-34s%6d%6.2f%7d%8.2f%%%7.2f%6.2fx%8.1f 只"
              % (r["label"], r["n"], r["per_day"], r["empty"],
                 100 * r["hit"], 100 * r["se"], r["lift"], r["per_month"]))

    (OUT / "window_grid.json").write_text(json.dumps(
        {"days": days, "base": base, "win": W, "grid": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> " + str(OUT / "window_grid.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
