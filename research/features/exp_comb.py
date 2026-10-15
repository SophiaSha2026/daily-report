"""
阶段 3：组合和容量（预登记 feat_comb_v1）。种子 7，同循环重跑 base_a158。

    python research/features/exp_comb.py --fit --all on vc hi ext turn rm amp --sel <阶段 2 里有增量的族>
    python research/features/exp_comb.py --analyze

臂
    comb_all       base_a158 + 诊断过关的全部族
    comb_sel       base_a158 + 阶段 2 里对 base_a158 有增量的族（挑族加）
    comb_sel_cs4   comb_sel，LightGBM colsample_bytree 0.7 -> 0.4（列数从 ~50 涨到 ~250，每棵树看到的列数大致回到原来的量级；
                   容量只调这一次）
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

NAME = "feat_comb_v1"


def make_fit(all_f, sel_f):
    def fit_predict(tr, te, feats, month, state):
        import model as M
        ac = state.setdefault("ac", E.a158_cols())
        fc = state.setdefault("fc", {f: [c for c in tr.columns if c.startswith(f"f_{f}_")] for f in all_f})
        ca = feats + ac + [c for f in all_f for c in fc[f]]
        cs = feats + ac + [c for f in sel_f for c in fc[f]]
        assert len(ca) > len(feats + ac) and len(cs) > len(feats + ac)
        trs = M.stratified_sample(tr, "y_up")
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        out = {}
        for name, cols, kw in (("base_a158", feats + ac, {}), ("comb_all", ca, {}),
                               ("comb_sel", cs, {}), ("comb_sel_cs4", cs, {"colsample_bytree": 0.4})):
            m = M.L1Lgbm(n_estimators=400, n_jobs=12, **kw).fit(trs, cols, "y_up")
            out[name] = (m.predict_proba(te), m.predict_proba(ref))
        return out
    return fit_predict


def analyze():
    pr = json.loads((K.EXP / NAME / "prereg.json").read_text(encoding="utf-8"))
    arms = list(pr["arms"])
    T = {a: C.top(NAME, a) for a in ["base_a158"] + arms}
    T["base"] = C.top("altpath_v1", "base")
    lines = []
    res = {}
    for a in arms:
        res[a] = C.full_cmp(T[a], T["base_a158"])
        lines.append(C.fmt(a, res[a], "base_a158"))
        lines.append("    " + C.fmt(a, C.full_cmp(T[a], T["base"]), "base"))
    rows = {k: C.monthly_row(T[k]) for k in ["base", "base_a158"] + arms}
    lines.append((100 * pd.DataFrame(rows).T).round(1).to_string())
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_comb.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, {"res": res})
    for k, v in res.items():
        C.ledger(NAME, k, "result", t_nw=v["t_nw"], diff=v["diff_day"], ref="base_a158")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--all", nargs="*", default=[])
    ap.add_argument("--sel", nargs="*", default=[])
    a = ap.parse_args()
    if a.fit:
        arms = {"comb_all": "base_a158 + " + ",".join(a.all), "comb_sel": "base_a158 + " + ",".join(a.sel),
                "comb_sel_cs4": "comb_sel，colsample_bytree 0.4（容量只调这一次）"}
        hyp = ("阶段 3：诊断过关的族全部并进 base_a158（comb_all）对比只挑阶段 2 有增量的族（comb_sel，%s），"
               "以及列数变多后把 colsample_bytree 从 0.7 降到 0.4 一次（comb_sel_cs4）。对照 base_a158，种子 7。" % ",".join(a.sel))
        new = not (K.EXP / NAME / "prereg.json").exists()
        K.prereg(NAME, hyp, arms)
        if new:
            for arm in arms:
                C.ledger(NAME, arm, "prereg")
        K.walk_forward(EF.load(sorted(set(a.all))), K.months_in(K.DEV), make_fit(a.all, a.sel), NAME)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
