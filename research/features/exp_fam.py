"""
阶段 2：新特征族并进 base_a158 做增量（预登记 feat_fam_v1）。种子 7，同循环重跑 base_a158（base 用 altpath_v1 的种子 7，逐位可复现）。

    python research/features/exp_fam.py --fit --fams on ext ...
    python research/features/exp_fam.py --analyze

只训诊断过关的族（规则见 ledger.jsonl 的 feat_diag/rule 行，看诊断前写定）。
对照：base_a158（主）、base（附带）。
"""
import argparse
import json
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

NAME = "feat_fam_v1"


def hyp(fams):
    return ("新特征族（research/features/build_new.py，日线 OHLCV + 换手率 + 不复权均价，当日排名）各自并进 base_a158，"
            "问每一族在 Alpha158 之上还有没有增量。只训单因子诊断过关的族：%s。对照同循环 base_a158（种子 7）。"
            "尺子：每天前 10 买得到口径按天配对，research/stats 新标准。" % ",".join(fams))


def load(fams):
    import buyable as B
    cache = json.loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
    feats = sorted({c for m in K.months_in(K.DEV) for c in cache[m]})
    df = pd.read_parquet(K.DATA / "train.parquet", columns=["code", "date", "board", "y_up"] + feats,
                         filters=[("date", "<", K.CONFIRM[0] + "-01")])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df[feats + ["y_up"]] = df[feats + ["y_up"]].astype("float32")
    lab = B.open_labels()
    lab = lab[lab["date"] < K.CONFIRM[0]]
    df = df.merge(lab[["code", "date", "y_open"]], on=["code", "date"], how="left")
    a = pd.read_parquet(K.EXP / "altpath" / "a158.parquet")
    df = df.merge(a, on=["code", "date"], how="left")
    del a
    for f in fams:
        x = pd.read_parquet(C.FE / f"fam_{f}.parquet")
        x["code"] = x["code"].astype(str).str.zfill(6)
        df = df.merge(x, on=["code", "date"], how="left")
        del x
    assert df["date"].max() < K.CONFIRM[0]
    print("训练表 %d 行 %d 列 %.1f GB" % (len(df), df.shape[1], df.memory_usage().sum() / 2 ** 30), flush=True)
    return df


def make_fit(fams):
    def fit_predict(tr, te, feats, month, state):
        import model as M
        ac = state.setdefault("ac", E.a158_cols())
        if "fc" not in state:
            state["fc"] = {f: [c for c in tr.columns if c.startswith(f"f_{f}_")] for f in fams}
            assert all(len(v) for v in state["fc"].values()), state["fc"]
        trs = M.stratified_sample(tr, "y_up")
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        out = {}
        arms = [("base_a158", feats + ac)] + [(f"a158_{f}", feats + ac + state["fc"][f]) for f in fams]
        for name, cols in arms:
            m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
            out[name] = (m.predict_proba(te), m.predict_proba(ref))
        return out
    return fit_predict


def analyze():
    pr = json.loads((K.EXP / NAME / "prereg.json").read_text(encoding="utf-8"))
    arms = [a for a in pr["arms"]]
    T = {a: C.top(NAME, a) for a in ["base_a158"] + arms}
    T["base"] = C.top("altpath_v1", "base")   # 种子 7 现状，三次重跑逐位相同
    a7 = C.top("altpath_v1", "base_a158")
    same = np.allclose(a7.sort_values(["date", "code"])["p"].to_numpy(),
                       T["base_a158"].sort_values(["date", "code"])["p"].to_numpy())
    lines = ["自检：本次 base_a158 和 altpath_v1 逐位相同 = %s" % same]
    res = {}
    for a in arms:
        c = C.full_cmp(T[a], T["base_a158"])
        res[a] = c
        lines.append(C.fmt(a, c, "base_a158"))
        lines.append("    " + C.fmt(a, C.full_cmp(T[a], T["base"]), "base"))
    rows = {k: C.monthly_row(T[k]) for k in ["base", "base_a158"] + arms}
    lines.append((100 * pd.DataFrame(rows).T).round(1).to_string())
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_fam.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, {"res": res})
    for k, v in res.items():
        C.ledger(NAME, k, "result", t_nw=v["t_nw"], diff=v["diff_day"], ref="base_a158")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fams", nargs="*", default=[])
    a = ap.parse_args()
    if a.fit:
        arms = {f"a158_{f}": f"base_a158 + 新族 {f}" for f in a.fams}
        new = not (K.EXP / NAME / "prereg.json").exists()
        K.prereg(NAME, hyp(a.fams), arms)
        if new:
            for arm in arms:
                C.ledger(NAME, arm, "prereg")
        K.walk_forward(load(a.fams), K.months_in(K.DEV), make_fit(a.fams), NAME)
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
