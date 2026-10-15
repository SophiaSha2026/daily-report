"""
阶段 4：最终候选（预登记 feat_final_v1，名字固定）。同一循环、同一份训练表重跑现状和候选，三个下采样种子。

    python research/features/exp_final.py --fit --cand <名字> --fams <族 ...> [--cs 0.4]
    python research/features/exp_final.py --analyze

臂（每个种子 s ∈ 7/9/10）
    base_s      现状（生产特征）
    a158_s      生产 + Alpha158 142 列（参照）
    cand_s      最终候选
判定：主比较 = cand_s7 对 base_s7，用 evalkit.compare(..., exp=NAME, arm="cand_s7")（v2 六条，门槛按登记簿）；
种子 9/10 只报告，不参与判定。最差三个月两种口径（含 / 去 2024-09）都报，不替用户选。
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402
import exp_altpath as E  # noqa: E402
import exp_fam as EF  # noqa: E402

NAME = "feat_final_v1"
SEEDS = (7, 9, 10)


def make_fit(fams, cs):
    def fit_predict(tr, te, feats, month, state):
        import model as M
        ac = state.setdefault("ac", E.a158_cols())
        fc = state.setdefault("fc", [c for f in fams for c in tr.columns if c.startswith(f"f_{f}_")])
        assert len(fc) or cs != 0.7
        out = {}
        for s in SEEDS:
            trs = M.stratified_sample(tr, "y_up", seed=s)
            ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
            for name, cols, kw in (("base", feats, {}), ("a158", feats + ac, {}),
                                   ("cand", feats + ac + fc, {"colsample_bytree": cs})):
                m = M.L1Lgbm(n_estimators=400, n_jobs=12, **kw).fit(trs, cols, "y_up")
                out[f"{name}_s{s}"] = (m.predict_proba(te), m.predict_proba(ref))
        return out
    return fit_predict


def analyze():
    pr = json.loads((K.EXP / NAME / "prereg.json").read_text(encoding="utf-8"))
    T = {f"{n}_s{s}": C.top(NAME, f"{n}_s{s}") for n in ("base", "a158", "cand") for s in SEEDS}
    lines = ["候选：%s" % pr["arms"]["cand_s7"]]
    v2 = K.compare(T["cand_s7"], T["base_s7"], exp=NAME, arm="cand_s7")
    th = v2["threshold"]
    lines.append("[主判定 evalkit v2] cand_s7 对 base_s7：差 %+.2fpp NW t %.2f（10/20/30: %s）块自举 p %.3f 去最好月(%s) t %.2f "
                 "合并差 %+.2f 最差三月 %s 去924最差三月 %s 空榜 %d/%d 门槛 c=%.2f（%s，m=%d）gate=%s -> %s" % (
                     100 * v2["diff_day"], v2["t_nw"], {k: round(v, 2) for k, v in v2["t_nw_lags"].items()},
                     v2["boot"]["p"], v2["drop_best_month"], v2["t_nw_drop_best"], 100 * v2["pooled_diff"],
                     {k: round(100 * v, 1) for k, v in v2["worst3"].items()},
                     {k: round(100 * v, 1) for k, v in v2["worst3_ex_2024_09"].items()},
                     v2["empty_a"], v2["empty_b"], th["c"], th["method"], th["m"], v2["gate"], v2["verdict"]))
    res = {"v2": {k: v for k, v in v2.items() if k != "threshold"}, "threshold": th}
    for s in SEEDS:
        for a_, b_ in (("cand", "base"), ("a158", "base"), ("cand", "a158")):
            c = C.full_cmp(T[f"{a_}_s{s}"], T[f"{b_}_s{s}"])
            res[f"{a_}_vs_{b_}_s{s}"] = c
            lines.append(C.fmt(f"{a_}_s{s}", c, f"{b_}_s{s}"))
    for a_, b_ in (("cand", "base"), ("a158", "base"), ("cand", "a158")):
        D = pd.DataFrame({s: C.daily_diff(T[f"{a_}_s{s}"], T[f"{b_}_s{s}"]) for s in SEEDS}).dropna()
        per = D.mean()
        dm = D.mean(axis=1)
        lines.append("三种子 %s 对 %s：各种子 %s，均值 %+.2fpp 种子间标准差 %.2f，平均序列 NW t %.2f 块自举 p %.3f" % (
            a_, b_, ", ".join("%+.2f" % (100 * x) for x in per), 100 * per.mean(), 100 * per.std(ddof=1),
            C.nwt(dm), C.hac.mbb(dm.to_numpy())["p"]))
        res[f"seedavg_{a_}_{b_}"] = {"mean": float(per.mean()), "sd": float(per.std(ddof=1)), "t_nw": C.nwt(dm)}
    rows = {k: C.monthly_row(T[k]) for k in T}
    lines.append((100 * pd.DataFrame(rows).T).round(1).to_string())
    for k in T:
        lines.append("%s 前10 %.2f%%" % (k, 100 * C.daily_hit(T[k]).mean()))
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_final.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, {"res": res})
    C.ledger(NAME, "cand_s7", "result", t_nw=v2["t_nw"], verdict=v2["verdict"], c=th["c"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--cand", default="")
    ap.add_argument("--fams", nargs="*", default=[])
    ap.add_argument("--cs", type=float, default=0.7)
    a = ap.parse_args()
    if a.fit:
        desc = "%s：生产 + Alpha158 142 列 + 新族 %s，colsample_bytree %.1f" % (a.cand, ",".join(a.fams) or "无", a.cs)
        # 只有 cand_s7 是一次试验；base_s*/a158_s* 是同循环对照，cand_s9/10 是稳健性报告，不进登记簿的次数
        arms = {"cand_s7": desc + " 种子7"}
        hyp = ("最终候选 %s（名字固定）。主判定只有一条：种子 7 的 cand_s7 对同循环 base_s7，evalkit.compare v2 六条（NW t >= "
               "max(2.5, 族门槛)、块自举 p<0.05、去最好月 t>=2、合并差和按天差同号、最差三月不比现状差、空榜不翻倍），门槛按"
               "登记簿当时的 m。最差三月另报去掉 2024-09 的口径，不用于判定、由用户定。种子 9/10 只报告。"
               "选择过程：阶段 1~3 在同一开发集上挑出（feat_seeds、feat_decomp、feat_fam_v1、feat_comb_v1），有选择偏差。" % a.cand)
        new = not (K.EXP / NAME / "prereg.json").exists()
        K.prereg(NAME, hyp, arms)
        if new:
            C.ledger(NAME, "cand_s7", "prereg", desc=desc)
        K.walk_forward(EF.load(sorted(set(a.fams))), K.months_in(K.DEV), make_fit(a.fams, a.cs), NAME)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
