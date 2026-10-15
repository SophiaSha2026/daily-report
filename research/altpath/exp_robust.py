"""
altpath_v1 之后的稳健性复核（预登记 altpath_v2_robust）。不是新候选，只回答 base_a158 的增益是不是种子运气、来自哪一组。

    python research/altpath/exp_robust.py --fit
    python research/altpath/exp_robust.py --analyze

臂（对照都是同循环的 base_s7，等于 altpath_v1 的 base 重跑）
    base_s8 / base_a158_s8   换下采样种子 8
    fam_kbar / fam_price / fam_vol / fam_volume   生产特征 + Alpha158 的一族
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", Path(__file__).parent, ROOT / "research" / "stats"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd  # noqa: E401,E402
import evalkit as K  # noqa: E402
import exp_altpath as E  # noqa: E402
import hac  # noqa: E402

NAME = "altpath_v2_robust"
HYP = ("altpath_v1 的 base_a158（生产特征 + Alpha158）对现状 +4.19pp、NW t 2.80：1 换下采样种子 8 后仍为正且 NW t >= 2；"
       "2 拆成四族（K 线形态 / 价格位置 / 波动与趋势拟合 / 量能）各自并进生产特征，看增益集中在哪一族。"
       "诊断性质，不作为新的过线候选，不计入 altpath_v1 的多重检验族。")
ARMS = {"base_s8": "现状，种子 8", "base_a158_s8": "生产 + a158，种子 8",
        "fam_kbar": "生产 + K 线形态 12 列", "fam_price": "生产 + 价格位置（ROC MA MAX MIN QTLU QTLD RANK RSV IMAX IMIN IMXD）",
        "fam_vol": "生产 + 波动与趋势拟合（STD BETA RSQR RESI CNTP CNTN SUMP SUMN）",
        "fam_volume": "生产 + 量能（VMA VSTD WVMA VSUMP VSUMN CORR CORD）"}
FAM = {
    "kbar": ("kmid", "klen", "kup", "klow", "ksft", "open", "high", "low"),
    "price": ("roc", "ma", "max", "min", "qtlu", "qtld", "rank", "rsv", "imax", "imin", "imxd"),
    "vol": ("std", "beta", "rsqr", "resi", "cntp", "cntn", "sump", "sumn"),
    "volume": ("vma", "vstd", "wvma", "vsump", "vsumn", "corr", "cord"),
}


def fam_cols(ac, fam):
    import re
    out = []
    for c in ac:
        stem = re.sub(r"\d+$", "", c[2:])
        if stem in FAM[fam]:
            out.append(c)
    return out


def fit_predict(tr, te, feats, month, state):
    import model as M
    ac = state.setdefault("ac", E.a158_cols())
    if "fams" not in state:
        state["fams"] = {f: fam_cols(ac, f) for f in FAM}
        assert sum(len(v) for v in state["fams"].values()) == len(ac), {f: len(v) for f, v in state["fams"].items()}
    out = {}
    for seed, arms in ((7, [("base", feats)] + [("fam_" + f, feats + c) for f, c in state["fams"].items()]),
                       (8, [("base_s8", feats), ("base_a158_s8", feats + ac)])):
        trs = M.stratified_sample(tr, "y_up", seed=seed)
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        for name, cols in arms:
            m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
            out[name] = (m.predict_proba(te), m.predict_proba(ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    if a.fit:
        df = E.load()
        K.walk_forward(df, K.months_in(K.DEV), fit_predict, NAME)
    if a.analyze:
        d = K.EXP / NAME
        v1 = K.EXP / "altpath_v1"
        T = {x: pd.read_parquet(d / f"top_{x}.parquet") for x in ["base"] + list(ARMS)}
        T["base_a158"] = pd.read_parquet(v1 / "top_base_a158.parquet")
        lines = []
        # 自检：同种子 base 和 v1 的 base 应当逐位一样（同数据同种子）
        b1 = pd.read_parquet(v1 / "top_base.parquet")
        same = np.allclose(b1.sort_values(["date", "code"])["p"].to_numpy(),
                           T["base"].sort_values(["date", "code"])["p"].to_numpy())
        lines.append("自检：本次 base 和 altpath_v1 base 逐位相同 = %s" % same)
        pairs = [("base_s8", "base"), ("base_a158_s8", "base_s8"), ("base_a158_s8", "base"),
                 ("base_a158", "base")] + [("fam_" + f, "base") for f in FAM]
        for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
            lines.append("[%s]" % lbl)
            for a_, b_ in pairs:
                dd = E.daily_diff(T[a_], T[b_], since=since)
                t, _ = E.nw_t(dd)
                s = E.summ(T[a_], since)
                lines.append("%-13s 对 %-8s 前10 开盘 %5.2f%% 差 %+.2fpp NW t=%.2f 块自举 p=%.3f" % (
                    a_, b_, 100 * s["open"]["hit"], 100 * dd.mean(), t, hac.mbb(dd)["p"]))
        txt = "\n".join(lines)
        print(txt)
        (Path(__file__).parent / "robust.txt").write_text(txt, encoding="utf-8")
        K.record(NAME, {"text": txt})
    return 0


if __name__ == "__main__":
    sys.exit(main())
