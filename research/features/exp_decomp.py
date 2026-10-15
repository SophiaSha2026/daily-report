"""
阶段 1 第 3 步：拆原因（预登记 feat_decomp）。种子 7，同循环重跑 base 当对照。

    python research/features/exp_decomp.py --fit
    python research/features/exp_decomp.py --analyze

臂
    prod_rank   生产特征（同一批列名、同一份每月选列），但从原值复算、只做当日排名，不中性化（build_prodraw.py）
    base_pv     生产特征 + Alpha158 的价格位置族和波动趋势族（exp_robust 的 price + vol 两族）
    pv_only     只有这两族
    rank_pv     prod_rank + 两族（事后可解释性：变换和新窗口叠加）
判读：prod_rank 过了 -> 三层变换丢信息；base_pv 接近 base_a158 -> 增益来自新窗口 / 新定义，列数可减。
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402
import exp_altpath as E  # noqa: E402
import exp_robust as R  # noqa: E402

NAME = "feat_decomp"
HYP = ("base_a158 的增益从哪来：(a) prod_rank 生产特征只做当日排名、不中性化（原值复算，列名和每月选列同生产）；"
       "(b) base_pv 生产 + Alpha158 价格位置 + 波动趋势两族；(c) pv_only 只有两族；(d) rank_pv = prod_rank + 两族。"
       "coarse_a158 是偷看审计臂：OHLC 先统一取整到对数价 0.5% 网格再算 a158，若增益仍接近 base_a158 则排除取整颗粒度泄漏。"
       "对照同循环 base（种子 7）。尺子同 research/stats 新标准；本实验是诊断，结论用于决定最终候选的列集。")
ARMS = {"prod_rank": "生产特征只排名不中性化", "base_pv": "生产 + a158 价格位置/波动趋势",
        "pv_only": "只有 a158 价格位置/波动趋势", "rank_pv": "prod_rank + 两族",
        "coarse_a158": "审计：生产 + a158（OHLC 先取整到对数 0.5% 网格，抹掉前复权颗粒度）"}


def load():
    df = E.load()
    cache = __import__("json").loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
    feats = sorted({c for m in K.months_in(K.DEV) for c in cache[m]})
    rk = pd.read_parquet(C.FE / "prod_rank.parquet", columns=["code", "date"] + feats)
    rk["code"] = rk["code"].astype(str).str.zfill(6)
    rk = rk.rename(columns={c: c + "_rk" for c in feats})
    n0 = len(df)
    df = df.merge(rk, on=["code", "date"], how="left")
    assert len(df) == n0
    cc = pd.read_parquet(C.FE / "a158_coarse.parquet")
    cc["code"] = cc["code"].astype(str).str.zfill(6)
    cc = cc.rename(columns={c: "c" + c for c in cc.columns if c.startswith("a_")})
    df = df.merge(cc, on=["code", "date"], how="left")
    del cc
    assert len(df) == n0
    print("prod_rank 对上 %.2f%% 行" % (100 * df[feats[0] + "_rk"].notna().mean()), flush=True)
    return df


def fit_predict(tr, te, feats, month, state):
    import model as M
    ac = state.setdefault("ac", E.a158_cols())
    pv = state.setdefault("pv", R.fam_cols(ac, "price") + R.fam_cols(ac, "vol"))
    rkf = [c + "_rk" for c in feats]
    trs = M.stratified_sample(tr, "y_up")
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    out = {}
    for name, cols in (("base", feats), ("prod_rank", rkf), ("base_pv", feats + pv), ("pv_only", pv),
                       ("rank_pv", rkf + pv), ("coarse_a158", feats + ["c" + c for c in ac])):
        if name != "base":
            assert set(cols) != set(feats), name
        m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
        out[name] = (m.predict_proba(te), m.predict_proba(ref))
    return out


def analyze():
    T = {a: C.top(NAME, a) for a in ["base"] + list(ARMS)}
    b1 = C.top("altpath_v1", "base")
    same = np.allclose(b1.sort_values(["date", "code"])["p"].to_numpy(), T["base"].sort_values(["date", "code"])["p"].to_numpy())
    lines = ["自检：本次 base 和 altpath_v1 base 逐位相同 = %s" % same]
    T["base_a158"] = C.top("altpath_v1", "base_a158")
    res = {}
    for a in list(ARMS) + ["base_a158"]:
        c = C.full_cmp(T[a], T["base"])
        res[a] = c
        lines.append(C.fmt(a, c))
    for a in ("base_pv", "pv_only", "rank_pv", "prod_rank", "coarse_a158"):
        c = C.full_cmp(T[a], T["base_a158"])
        lines.append(C.fmt(a, c, "base_a158"))
    rows = {k: C.monthly_row(T[k]) for k in ["base", "base_a158"] + list(ARMS)}
    lines.append((100 * pd.DataFrame(rows).T).round(1).to_string())
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_decomp.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, {"res": res})
    for k, v in res.items():
        C.ledger(NAME, k, "result", t_nw=v["t_nw"], diff=v["diff_day"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    new = not (K.EXP / NAME / "prereg.json").exists()
    K.prereg(NAME, HYP, ARMS)
    if new:
        for arm in ARMS:
            C.ledger(NAME, arm, "prereg")
    if a.fit:
        K.walk_forward(load(), K.months_in(K.DEV), fit_predict, NAME)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
