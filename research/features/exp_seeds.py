"""
阶段 1 第 2 步：种子稳健性 + 标签打乱对照（预登记 feat_seeds）。

    python research/features/exp_seeds.py --fit
    python research/features/exp_seeds.py --analyze

臂（下采样种子 = model.stratified_sample 的 seed，LightGBM 自己的 random_state 照生产固定 7，
和 altpath_v2_robust 的种子 8 同一口径）：
    base_s9/s10/s11、a158_s9/s10/s11（生产 + Alpha158 142 列）
    bag3_base / bag3_a158：种子 9/10/11 三个模型的预测平均（参照样本取种子 9 的）
    shuf_a158：生产 + Alpha158，训练样本里 y_up 按月打乱（种子 7）。应当学不到东西
种子 7（altpath_v1 base / base_a158）、种子 8（altpath_v2_robust）直接读已有产物。
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

NAME = "feat_seeds"
HYP = ("altpath_v1 的 base_a158（生产 + Alpha158 142 列）对现状 +4.19pp（种子 7）、+2.89pp（种子 8）。"
       "种子 9/10/11 各跑一对 base / base_a158，连同 7、8 共五个种子：判据预先写定为五个种子配对差（每天前 10 买得到）"
       "的平均 NW t >= 2.8 且没有一个种子的差为负。另加 bag3（种子 9/10/11 三个模型预测平均，两边都做）看平均能否稳住方差；"
       "shuf_a158 是标签按月打乱的对照臂，期望前 10 命中接近全市场基准（约 4.5%），明显高于基准说明尺子或特征有问题。")
ARMS = {"base_s9": "现状 种子9", "base_s10": "现状 种子10", "base_s11": "现状 种子11",
        "a158_s9": "生产+a158 种子9", "a158_s10": "生产+a158 种子10", "a158_s11": "生产+a158 种子11",
        "bag3_base": "现状 种子9/10/11 平均", "bag3_a158": "生产+a158 种子9/10/11 平均",
        "shuf_a158": "生产+a158 标签按月打乱（对照）"}


def fit_predict(tr, te, feats, month, state):
    import model as M
    ac = state.setdefault("ac", E.a158_cols())
    out = {}
    models = {"base": [], "a158": []}
    ref_bag = None
    for seed in (9, 10, 11):
        trs = M.stratified_sample(tr, "y_up", seed=seed)
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        if ref_bag is None:
            ref_bag = ref
        for nm, cols in (("base", feats), ("a158", feats + ac)):
            m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
            out[f"{nm}_s{seed}"] = (m.predict_proba(te), m.predict_proba(ref))
            models[nm].append(m)
    for nm in ("base", "a158"):
        pt = np.mean([m.predict_proba(te) for m in models[nm]], axis=0)
        pr = np.mean([m.predict_proba(ref_bag) for m in models[nm]], axis=0)
        out[f"bag3_{nm}"] = (pt, pr)
    # 标签打乱：下采样之后在每个月内部打乱 y_up（正样本率逐月不变），特征不动
    trs = M.stratified_sample(tr, "y_up", seed=7)
    rng = np.random.default_rng(20261015)
    trs["y_up"] = trs.groupby(trs["date"].str[:7])["y_up"].transform(lambda s: rng.permutation(s.to_numpy()))
    assert trs["y_up"].sum() > 0
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, feats + ac, "y_up")
    out["shuf_a158"] = (m.predict_proba(te), m.predict_proba(ref))
    return out


def analyze():
    lines = []
    T = {a: C.top(NAME, a) for a in ARMS}
    T["base_s7"] = C.top("altpath_v1", "base")
    T["a158_s7"] = C.top("altpath_v1", "base_a158")
    T["base_s8"] = C.top("altpath_v2_robust", "base_s8")
    T["a158_s8"] = C.top("altpath_v2_robust", "base_a158_s8")
    res = {}
    diffs = {}
    for s in (7, 8, 9, 10, 11):
        a, b = T[f"a158_s{s}"], T[f"base_s{s}"]
        # 教训 32：两臂真的不一样
        ov = (a[a["rank"] <= 10].groupby("date")["code"].apply(set)
              .combine(b[b["rank"] <= 10].groupby("date")["code"].apply(set), lambda x, y: len(x & y) / 10)).mean()
        c = C.full_cmp(a, b)
        c["overlap_top10"] = float(ov)
        res[f"s{s}"] = c
        diffs[s] = C.daily_diff(a, b)
        lines.append(C.fmt(f"a158_s{s}", c, f"base_s{s}") + " 前10重合 %.1f%%" % (100 * ov))
        lines.append("    base_s%d 前10 %.2f%%" % (s, 100 * c["hit_b"]))
    D = pd.DataFrame(diffs).dropna()
    per = D.mean()
    lines.append("")
    lines.append("五个种子：配对差均值 %+.2fpp，种子间标准差 %.2fpp，最差种子 s%d %+.2fpp，最好 s%d %+.2fpp" % (
        100 * per.mean(), 100 * per.std(ddof=1), per.idxmin(), 100 * per.min(), per.idxmax(), 100 * per.max()))
    tn = [res[f"s{s}"]["t_nw"] for s in D.columns]
    lines.append("五个种子 NW t：%s，平均 %.2f；判据（平均 >= 2.8 且无负）-> %s" % (
        ", ".join("%.2f" % x for x in tn), np.mean(tn),
        "成立" if np.mean(tn) >= 2.8 and per.min() > 0 else "不成立"))
    dm = D.mean(axis=1)
    lines.append("五种子平均的逐日差序列：%+.2fpp NW t %.2f 块自举 p %.3f" % (
        100 * dm.mean(), C.nwt(dm), C.hac.mbb(dm.to_numpy())["p"]))
    # 五种子平均命中
    hb = np.mean([C.daily_hit(T[f"base_s{s}"]).mean() for s in D.columns])
    ha = np.mean([C.daily_hit(T[f"a158_s{s}"]).mean() for s in D.columns])
    lines.append("五种子平均前 10（按天均值）：base %.2f%%  base_a158 %.2f%%" % (100 * hb, 100 * ha))
    lines.append("")
    c = C.full_cmp(T["bag3_a158"], T["bag3_base"])
    res["bag3"] = c
    lines.append(C.fmt("bag3_a158", c, "bag3_base"))
    c2 = C.full_cmp(T["bag3_a158"], T["base_s7"])
    res["bag3_vs_s7"] = c2
    lines.append(C.fmt("bag3_a158", c2, "base_s7"))
    c3 = C.full_cmp(T["bag3_base"], T["base_s7"])
    lines.append(C.fmt("bag3_base", c3, "base_s7"))
    # 打乱对照
    lab = pd.read_parquet(K.DATA / "raw" / "labels_open.parquet", columns=["date", "y_open"],
                          filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    mkt = lab.groupby("date")["y_open"].mean()
    sh = C.daily_hit(T["shuf_a158"])
    lines.append("")
    lines.append("shuf_a158（标签打乱）前 10 %.2f%%，同日全市场基准 %.2f%%，差 %+.2fpp NW t %.2f；"
                 "现状 base_s7 %.2f%%" % (100 * sh.mean(), 100 * mkt.reindex(sh.index).mean(),
                                          100 * (sh - mkt.reindex(sh.index)).mean(),
                                          C.nwt((sh - mkt.reindex(sh.index)).dropna()),
                                          100 * C.daily_hit(T["base_s7"]).mean()))
    res["shuf"] = {"hit": float(sh.mean()), "mkt": float(mkt.reindex(sh.index).mean())}
    # 逐月
    lines.append("")
    rows = {k: C.monthly_row(T[k]) for k in ("base_s7", "a158_s7", "base_s9", "a158_s9", "base_s10", "a158_s10",
                                               "base_s11", "a158_s11", "bag3_base", "bag3_a158", "shuf_a158")}
    lines.append((100 * pd.DataFrame(rows).T).round(1).to_string())
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_seeds.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, {"res": res})
    for k, v in res.items():
        C.ledger(NAME, k, "result", t_nw=v.get("t_nw"), diff=v.get("diff_day"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    new = not (K.EXP / NAME / "prereg.json").exists()
    K.prereg(NAME, HYP, ARMS, metric="top10 y_open 按天配对 + NW(20)")
    if new:
        for arm in ARMS:
            C.ledger(NAME, arm, "prereg", note="种子稳健性 / 对照，不作为新候选" if arm != "shuf_a158" else "打乱对照")
    if a.fit:
        df = E.load()
        K.walk_forward(df, K.months_in(K.DEV), fit_predict, NAME)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
