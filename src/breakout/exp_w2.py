"""
第 2 周回看（计划 3.3 / 3.4 / 2.2）：同涨族群特征复验、超参数一次性 CV、科创板处理。走 evalkit。

    python src/breakout/exp_w2.py --fit        开发集 19 个月，六个臂（约两三个小时）
    python src/breakout/exp_w2.py --analyze    对照 w1 的 base（同一套样本、特征缓存、种子）

臂
--
    g2        base 特征 + 同涨族群 5 列（实验 16 第二轮唯一每项不比现状差的）
    hp_l15    num_leaves 15        hp_l63  num_leaves 63
    hp_mc400  min_child_samples 400
    hp_lr02   learning_rate 0.02 / 800 棵
    hp_cs05   colsample_bytree 0.5
  事后：
    nostar    现状的名次里剔掉科创板再补位（科创在验证集前 10 名额里 0/27）

超参数只调这一次：六个变体一起比，过线的才换，换了就记进 model.py 不再动。
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

import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
NAME = "w2_feat_hp"
BASE_NAME = "w1_scale_avg"
HYP = ("3.3 同涨族群特征在开发集 19 个月上仍不比现状差且过线；"
       "3.4 六个超参数变体里有过线的；2.2 剔掉科创板再补位能抬买得到口径的命中")
G2 = ["x_peer_ret1_pct", "x_peer_ret5_pct", "x_peer_lu_pct", "x_rel_peer5_pct", "x_peer_corr_pct"]
HP = {
    "hp_l15": dict(num_leaves=15),
    "hp_l63": dict(num_leaves=63),
    "hp_mc400": dict(min_child_samples=400),
    "hp_lr02": dict(learning_rate=0.02, n_estimators=800),
    "hp_cs05": dict(colsample_bytree=0.5),
}
ARMS = {"g2": "base 特征 + 同涨族群 5 列", **{k: str(v) for k, v in HP.items()},
        "nostar": "现状剔科创补位（事后）"}
FEATS2 = K.DATA / "raw" / "exp_acc" / "feats2.parquet"


def fit_predict(tr, te, feats, month, state):
    import model as M
    trs = M.stratified_sample(tr, "y_up")
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    out = {}
    m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, feats + G2, "y_up")
    out["g2"] = (m.predict_proba(te), m.predict_proba(ref))
    for name, kw in HP.items():
        p = dict(n_estimators=400, n_jobs=12)
        p.update(kw)
        m = M.L1Lgbm(**p).fit(trs, feats, "y_up")
        out[name] = (m.predict_proba(te), m.predict_proba(ref))
    return out


def nostar(top: pd.DataFrame) -> pd.DataFrame:
    t = top[top["board"] != "star"].copy()
    t = t.sort_values(["date", "p"], ascending=[True, False])
    t["rank"] = t.groupby("date").cumcount() + 1
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    months = K.months_in(K.DEV)
    if a.fit:
        df = K.load_frame(with_confirm=False)
        fx = pd.read_parquet(FEATS2, columns=["code", "date"] + G2)
        df = df.merge(fx, on=["code", "date"], how="left")
        for c in G2:
            df[c] = df[c].astype("float32")
        K.walk_forward(df, months, fit_predict, NAME)
    if a.analyze:
        base = pd.read_parquet(K.EXP / BASE_NAME / "top_base.parquet")
        tops = {"base": base}
        for arm in list(HP) + ["g2"]:
            f = K.EXP / NAME / f"top_{arm}.parquet"
            if f.exists():
                tops[arm] = pd.read_parquet(f)
        tops["nostar"] = nostar(base)
        res = {}
        for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
            print("[%s]" % lbl)
            for arm, t in tops.items():
                m = K.metrics(t, since)
                res[f"{arm}|{lbl}"] = m
                print(K.fmt_metrics(arm, m))
            for arm in tops:
                if arm == "base":
                    continue
                c = K.compare(tops[arm], base, since=since)
                res[f"cmp_{arm}|{lbl}"] = c
                print(K.fmt_compare(arm, c))
            print()
        K.record(NAME, res)
        (K.EXP / NAME / "summary.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
