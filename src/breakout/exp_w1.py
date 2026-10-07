"""
第 1 周回看（计划 3.1 / 3.2）：分数刻度滚动化、模型平均。走 evalkit 的新协议。

    python src/breakout/exp_w1.py --fit        开发集 19 个月：现状 + 模型平均（一个循环里出）
    python src/breakout/exp_w1.py --analyze    成绩、配对、滚动刻度的事后分析

臂
--
    base   现状：按月等权 LightGBM，y_up，每月重训，分数刻度 = 训练样本内预测值分位
    avg3   模型平均：本月、上月、上上月三个模型的预测值取均值（前两个月的模型是「旧模型」，
           在本月数据上是真正的样本外）。刻度用三个参照样本预测值的均值
  事后（不重训）：
    roll   分数刻度滚动化：分数 = p 在最近 60 个交易日全市场预测值里的分位；
           门槛按开发集前半段定成和现状同样的平均够格只数，比的是够格只数的月间离散和命中
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
NAME = "w1_scale_avg"
HYP = ("3.1 分数刻度按最近 60 个交易日全市场分位定，够格只数的月间离散下降而命中不降；"
       "3.2 最近三次重训的模型取平均，最差月抬高、月间离散下降")
ARMS = {"base": "现状", "avg3": "最近三个月模型预测值平均", "roll": "滚动刻度（事后）"}
ROLL_DAYS = 60


def fit_predict(tr, te, feats, month, state):
    import model as M
    trs = M.stratified_sample(tr, "y_up")
    mdl = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, feats, "y_up")
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    hist = state.setdefault("models", [])
    hist.append((mdl, feats))
    del hist[:-3]
    p_base = mdl.predict_proba(te)
    r_base = mdl.predict_proba(ref)
    # 旧模型用它自己当时的特征列预测本月
    p_all = [m.predict_proba(te) for m, _ in hist]
    r_all = [m.predict_proba(ref) for m, _ in hist]
    return {"base": (p_base, r_base),
            "avg3": (np.mean(p_all, axis=0), np.mean(r_all, axis=0))}


def rolling_scale(all_p: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """分数 = p 在最近 ROLL_DAYS 个交易日全市场预测值里的分位（0~100，0.1 一档），不含当天。

    全市场 99 分位一天就是五十只，所以刻度要细到 0.1；门槛在 analyze 里按够格只数对齐。"""
    dates = sorted(all_p["date"].unique())
    grid = np.linspace(0, 1, 1001)
    qs = {}
    buf = []
    for d in dates:
        pool = np.concatenate(buf[-ROLL_DAYS:]) if buf else np.array([])
        qs[d] = np.quantile(pool, grid) if len(pool) > 1000 else None
        buf.append(all_p.loc[all_p["date"] == d, "p"].to_numpy(float))
    t = top.copy()
    sc = np.full(len(t), np.nan)
    for d, idx in t.groupby("date").indices.items():
        q = qs.get(d)
        if q is not None:
            sc[idx] = np.clip(np.searchsorted(q, t["p"].to_numpy(float)[idx]), 0, 1000) / 10.0
    t["score"] = sc
    return t[np.isfinite(t["score"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    months = K.months_in(K.DEV)
    if a.fit:
        df = K.load_frame(with_confirm=False)
        K.walk_forward(df, months, fit_predict, NAME)
    if a.analyze:
        import daily as D
        d = K.EXP / NAME
        tops = {arm: pd.read_parquet(d / f"top_{arm}.parquet") for arm in ("base", "avg3")}
        all_base = pd.read_parquet(d / "all_base.parquet")
        roll = rolling_scale(all_base, tops["base"])
        # 滚动刻度的门槛：开发集前半段（2024-06..2025-02）按和现状同样的平均够格只数定
        first_half = roll[roll["date"] < "2025-03-01"]
        base_first = tops["base"][tops["base"]["date"] < "2025-03-01"]
        target_per_day = (base_first["score"] >= D.SCORE_MIN).sum() / max(base_first["date"].nunique(), 1)
        best = None
        for thr in np.arange(95.0, 100.0, 0.1):
            per_day = (first_half["score"] >= thr).sum() / max(first_half["date"].nunique(), 1)
            if best is None or abs(per_day - target_per_day) < abs(best[1] - target_per_day):
                best = (thr, per_day)
        thr = best[0]
        roll2 = roll.copy()
        roll2["score"] = np.where(roll2["score"] >= thr, 100.0, 0.0)   # 够格 / 不够格
        tops["roll"] = roll2
        res = {"roll_threshold": thr, "target_per_day": float(target_per_day)}
        print("开发集 %s..%s（%d 个月）。滚动刻度门槛 %.1f（前半段够格 %.1f 只/天对现状 %.1f）\n" % (
            K.DEV[0], K.DEV[1], len(months), thr, best[1], target_per_day))
        for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
            print("[%s]" % lbl)
            for arm, t in tops.items():
                m = K.metrics(t, since)
                res[f"{arm}|{lbl}"] = m
                print(K.fmt_metrics(arm, m))
            for arm in ("avg3",):
                c = K.compare(tops[arm], tops["base"], since=since)
                res[f"cmp_{arm}|{lbl}"] = c
                print(K.fmt_compare(arm, c))
            # 够格只数的月间离散：现状 vs 滚动刻度
            for arm in ("base", "roll"):
                t = tops[arm]
                t = t[t["date"] >= since] if since else t
                q = t[t["score"] >= (D.SCORE_MIN if arm == "base" else 100)]
                per_m = q.groupby(q["date"].str[:7]).size() / t.groupby(t["date"].str[:7])["date"].nunique()
                per_m = per_m.fillna(0)
                res[f"qual_{arm}|{lbl}"] = {"mean": float(per_m.mean()), "sd": float(per_m.std()),
                                            "cv": float(per_m.std() / max(per_m.mean(), 1e-9)),
                                            "min": float(per_m.min()), "max": float(per_m.max())}
                print("  够格只数/天 %-5s 月均 %.1f 月间标准差 %.1f 变异系数 %.2f 最少 %.1f 最多 %.1f" % (
                    arm, per_m.mean(), per_m.std(), per_m.std() / max(per_m.mean(), 1e-9),
                    per_m.min(), per_m.max()))
            print()
        print("逐月 前10 买得到口径（%）")
        mo = sorted({x[:7] for x in tops["base"]["date"]})
        print("%-6s" % "臂" + "".join("%6s" % m[2:] for m in mo))
        for arm in ("base", "avg3"):
            r = K.monthly(tops[arm][tops[arm]["rank"] <= 10])
            print("%-6s" % arm + "".join("%6.1f" % (100 * r.get(m, np.nan)) for m in mo))
        K.record(NAME, res)
        (d / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
