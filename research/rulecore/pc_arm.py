
"""
百分位压缩假说的对照臂 pc：现状 LightGBM（生产特征 + 每月 fselect 重选，缓存键 "<月>|screen_v1"，按月等权，种子 7），
只把 rs20__last（20 日涨幅的当日百分位，5 日聚合的「最后一天」）换成原值 log(1+ret20)（列 ret20_lraw，不排名不中性化）。
其余列、训练切片（每月前 3 年 + validate.train_slice 净化）、采样（stratified_sample 种子 7）和 screen_base_v1 的 base 完全一样。
顺带把现状 base 在长表上的 2025-12 整月补出来（screen_base_v1_dev 只到 2025-12-03：那张表的标签窗口没关；
这里的标签用 common_rc 补过 leak 快照的那份），开发集凑齐 387 天。

产物 data/breakout/raw/exp/rulecore_v1/parts/<月>/{all,top}_pc_s7.parquet（2025-12 另有 base_s7），拼成 all_pc_s7.parquet、
all_base_dec.parquet。可续跑；北京 16:00~18:30 自动停。
    PYTHONIOENCODING=utf-8 python research/rulecore/pc_arm.py [--seg screen dev]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_rc as C  # noqa: E402
import exp_screen as X  # noqa: E402

import evalkit as K  # noqa: E402

DROP_KEY = "screen_v1"
SEED = 7
RAW = "ret20_lraw"
OUT = C.EXP / C.NAME
PARTS = OUT / "parts"


def pc_cols(feats):
    return [c for c in feats if c != "rs20__last"] + [RAW]


def make_fit(with_base: bool):
    def fit_predict(tr, te, feats, month, state):
        import model as M
        out, rows = {}, {}
        trs = M.stratified_sample(tr, "y_up", seed=SEED)
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        arms = {"pc": pc_cols(feats)}
        if with_base:
            arms["base"] = list(feats)
        for nm, cols in arms.items():
            assert len(set(cols)) == len(cols) and all(c in tr.columns for c in cols), nm   # 教训 32
            assert (nm != "pc") or (RAW in cols and "rs20__last" not in cols)
            m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
            rows[f"{nm}_s{SEED}"] = {"n_train": int(len(trs)), "n_cols": len(cols), "has_rs20_last": "rs20__last" in feats}
            out[f"{nm}_s{SEED}"] = (m.predict_proba(te), m.predict_proba(ref))
        d = PARTS / month
        d.mkdir(parents=True, exist_ok=True)
        (d / "train_rows.json").write_text(json.dumps(rows), encoding="utf-8")
        return out
    return fit_predict


def load_seg(seg: str, feat: pd.DataFrame) -> pd.DataFrame:
    months = C.months_in(seg)
    df = X.load(months, {"base": X.ARMS["base"]})
    # 2025-12 的标签：用 common_rc 补过的那份（长表自己的在 12-04 之后是 NaN）
    pat = feat.loc[feat["patched"].astype(bool), ["code", "date", "y_up", "y_open"]]
    if len(pat):
        df = df.merge(pat.rename(columns={"y_up": "_yu", "y_open": "_yo"}), on=["code", "date"], how="left")
        m = np.isfinite(df["_yo"])
        df.loc[m, "y_up"] = df.loc[m, "_yu"].astype("float32")
        df.loc[m, "y_open"] = df.loc[m, "_yo"].astype("float32")
        df = df.drop(columns=["_yu", "_yo"])
        C.say("  2025-12 标签补 %d 行" % int(m.sum()))
    df = df.merge(feat[["code", "date", "lret20"]].rename(columns={"lret20": RAW}), on=["code", "date"], how="left")
    df[RAW] = df[RAW].astype("float32")
    C.say("  %s：%d 行，%s 非空 %.1f%%" % (seg, len(df), RAW, 100 * np.isfinite(df[RAW]).mean()))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="*", default=["screen", "dev"])
    a = ap.parse_args()
    feat = C.build_features()[["code", "date", "lret20", "y_up", "y_open", "patched"]]
    t0 = time.time()
    for seg in a.seg:
        months = C.months_in(seg)
        todo = [m for m in months if not (PARTS / m / f"all_pc_s{SEED}.parquet").exists()
                or (m == "2025-12" and not (PARTS / m / f"all_base_s{SEED}.parquet").exists())]
        if not todo:
            C.say("%s 全部已有" % seg)
            continue
        df = load_seg(seg, feat)
        for m in todo:
            if C.in_prod_window():
                C.say("北京时间在生产窗口里，停在 %s 之前；重跑同一命令续跑" % m)
                return
            lo = C.month_shift(m, -12 * C.TRAIN_YEARS) + "-01"
            hi = C.month_shift(m, 1) + "-01"
            sub = df[(df["date"] >= lo) & (df["date"] < hi)]
            K.walk_forward(sub, [m], make_fit(with_base=(m == "2025-12")), f"{C.NAME}/parts/{m}", drop=[DROP_KEY])
            del sub
            C.say("[pc] %s 完成，累计 %.1f 分钟" % (m, (time.time() - t0) / 60))
        del df
    # 拼
    alls = []
    for seg in ("screen", "dev"):
        for m in C.months_in(seg):
            f = PARTS / m / f"all_pc_s{SEED}.parquet"
            if f.exists():
                alls.append(pd.read_parquet(f))
    if alls:
        pd.concat(alls, ignore_index=True).to_parquet(OUT / "all_pc_s7.parquet", index=False)
    fb = PARTS / "2025-12" / f"all_base_s{SEED}.parquet"
    if fb.exists():
        pd.read_parquet(fb).to_parquet(OUT / "all_base_dec.parquet", index=False)
    C.say("-> %s：pc %d 个月" % (OUT, len(alls)))
    C.progress("pc 臂走向前 %d 个月跑完（含 2025-12 的 base 补算）" % len(alls))


if __name__ == "__main__":
    main()
