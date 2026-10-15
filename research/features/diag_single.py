"""
单因子诊断（不训模型）。开发集 2024-06..2025-12，每天全市场（训练表里的行，扣掉 y_open 缺失）。

    python research/features/diag_single.py prod            生产特征（训练表里的值，开发集各月选中过的列的并集）
    python research/features/diag_single.py a158            Alpha158 142 列
    python research/features/diag_single.py <名字>          data/breakout/raw/exp/features/<名字>.parquet 里 f_ 开头的列

每列（先做当日排名，缺失记 0.5）：
    ic          当日秩相关（排名 vs y_open）逐日均值；t 用 NW(20)；正的月数（19 个月）
    lift_hi/lo  当日前 5% / 后 5% 的命中 ÷ 当日全市场命中（逐日均值）
    ric_base    增量相关：y_open 减去「同月 base 预测值 50 分位档的命中率」后的残差和排名的相关，逐日均值 + NW t
    ric_a158    同上，参照换成 base_a158（种子 7）的预测
    auc_out     base 当日 200 名之外的行里，这一列区分命中 / 不命中的 AUC（逐日均值，0.5 = 没区分度）
    pos_m       ic 为正的月数
结果 results_diag_<名字>.txt 和 diag_<名字>.json。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

K = C.K


def base_frame() -> pd.DataFrame:
    lab = pd.read_parquet(K.DATA / "raw" / "labels_open.parquet", columns=["code", "date", "y_open"],
                          filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    tr = pd.read_parquet(K.DATA / "train.parquet", columns=["code", "date", "y_up"],
                         filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    tr["code"] = tr["code"].astype(str).str.zfill(6)
    x = tr.merge(lab, on=["code", "date"], how="inner")
    x = x[np.isfinite(x["y_open"]) & np.isfinite(x["y_up"])].drop(columns=["y_up"])
    for arm, nm in (("base", "pb"), ("base_a158", "pa")):
        p = pd.read_parquet(K.EXP / "altpath_v1" / f"all_{arm}.parquet")
        p["code"] = p["code"].astype(str).str.zfill(6)
        x = x.merge(p.rename(columns={"p": nm}), on=["code", "date"], how="left")
    assert x["date"].max() < "2026-01-01"
    x["m"] = x["date"].str[:7]
    for nm, out in (("pb", "res_b"), ("pa", "res_a")):
        b = x.groupby("m")[nm].rank(pct=True)
        bins = np.minimum((b * 50).astype(int), 49)
        x[out] = x["y_open"] - x.groupby([x["m"], bins])["y_open"].transform("mean")
    x["rk_b"] = x.groupby("date")["pb"].rank(ascending=False)
    return x


def load_cols(src: str, x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if src == "prod":
        cache = json.loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
        cols = sorted({c for m in K.months_in(K.DEV) for c in cache[m]})
        f = pd.read_parquet(K.DATA / "train.parquet", columns=["code", "date"] + cols,
                            filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    elif src == "a158":
        import exp_altpath as E
        cols = E.a158_cols()
        f = pd.read_parquet(K.EXP / "altpath" / "a158.parquet",
                            filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    else:
        f = pd.read_parquet(C.FE / f"{src}.parquet",
                            filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
        cols = [c for c in f.columns if c.startswith("f_")]
    f["code"] = f["code"].astype(str).str.zfill(6)
    f[cols] = f[cols].astype("float32")
    x = x.merge(f, on=["code", "date"], how="left")
    return x, cols


def corr_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Xc * Xc).sum(axis=0) * (yc * yc).sum())
    with np.errstate(all="ignore"):
        return np.where(den > 0, (Xc * yc[:, None]).sum(axis=0) / den, np.nan)


_BASE = {}


def run(src: str) -> pd.DataFrame:
    t0 = time.time()
    if "x" not in _BASE:
        _BASE["x"] = base_frame()
    x, cols = load_cols(src, _BASE["x"])
    miss = x[cols].isna().mean()
    x = x.sort_values("date").reset_index(drop=True)
    dates = x["date"].to_numpy()
    cut = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    idx = np.split(np.arange(len(x)), cut)
    k = len(cols)
    rec = {key: [] for key in ("ic", "ric_b", "ric_a", "lift_hi", "lift_lo", "auc_out")}
    days = []
    Xall = x[cols].to_numpy(np.float32)
    y_all = x["y_open"].to_numpy(float)
    rb_all, ra_all = x["res_b"].to_numpy(float), x["res_a"].to_numpy(float)
    rk_all = x["rk_b"].to_numpy(float)
    for ii in idx:
        X = pd.DataFrame(Xall[ii]).rank(pct=True).fillna(0.5).to_numpy(float)
        y = y_all[ii]
        n = len(y)
        mu = y.mean()
        days.append(dates[ii[0]])
        rec["ic"].append(corr_cols(X, y))
        rec["ric_b"].append(corr_cols(X, rb_all[ii]))
        rec["ric_a"].append(corr_cols(X, np.nan_to_num(ra_all[ii])))
        q = max(int(0.05 * n), 1)
        o = np.argsort(X, axis=0)
        hi = y[o[-q:]].mean(axis=0)
        lo = y[o[:q]].mean(axis=0)
        rec["lift_hi"].append(hi / mu if mu > 0 else np.full(k, np.nan))
        rec["lift_lo"].append(lo / mu if mu > 0 else np.full(k, np.nan))
        out = rk_all[ii] > 200
        yo = y[out]
        npos, nneg = yo.sum(), (1 - yo).sum()
        if npos >= 3 and nneg >= 3:
            R = pd.DataFrame(X[out]).rank().to_numpy()
            auc = (R[yo > 0].sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)
        else:
            auc = np.full(k, np.nan)
        rec["auc_out"].append(auc)
    D = {key: pd.DataFrame(np.vstack(v), index=days, columns=cols) for key, v in rec.items()}
    rows = []
    for c in cols:
        ic = D["ic"][c].dropna()
        mic = ic.groupby(ic.index.str[:7]).mean()
        r = {"col": c, "miss": float(miss[c]), "ic": float(ic.mean()), "ic_t": C.nwt(ic),
             "pos_m": int((np.sign(mic) == np.sign(ic.mean())).sum()), "n_m": int(len(mic)),
             "ic_2025": float(ic[ic.index >= "2025-01"].mean()),
             "ric_b": float(D["ric_b"][c].mean()), "ric_b_t": C.nwt(D["ric_b"][c].dropna()),
             "ric_a": float(D["ric_a"][c].mean()), "ric_a_t": C.nwt(D["ric_a"][c].dropna()),
             "lift_hi": float(D["lift_hi"][c].mean()), "lift_lo": float(D["lift_lo"][c].mean()),
             "auc_out": float(D["auc_out"][c].mean()), "auc_out_t": C.nwt(D["auc_out"][c].dropna() - 0.5)}
        mr = D["ric_b"][c].groupby(D["ric_b"][c].index.str[:7]).mean()
        r["ric_b_pos_m"] = int((np.sign(mr) == np.sign(r["ric_b"])).sum())
        ma_ = D["ric_a"][c].groupby(D["ric_a"][c].index.str[:7]).mean()
        r["ric_a_pos_m"] = int((np.sign(ma_) == np.sign(r["ric_a"])).sum())
        rows.append(r)
    R = pd.DataFrame(rows)
    R["abs_ric_b_t"] = R["ric_b_t"].abs()
    R = R.sort_values("abs_ric_b_t", ascending=False).drop(columns="abs_ric_b_t")
    print("%s：%d 列，%d 天，%d 行，%.0fs" % (src, len(cols), len(days), len(x), time.time() - t0), flush=True)
    return R


def fmt_table(R: pd.DataFrame, n: int = 400) -> str:
    h = ("%-28s %5s %7s %6s %5s %7s %7s %6s %7s %6s %5s %6s %6s %6s %6s" % (
        "列", "缺失", "ic", "ic_t", "正月", "ic25", "ric_b", "t", "ric_a", "t", "a同号月", "前5%", "后5%", "auc外", "t"))
    out = [h]
    for _, r in R.head(n).iterrows():
        out.append("%-28s %5.1f %+7.4f %6.2f %2d/%2d %+7.4f %+7.4f %6.2f %+7.4f %6.2f %2d/19 %6.2f %6.2f %6.3f %6.2f" % (
            r["col"], 100 * r["miss"], r["ic"], r["ic_t"], r["pos_m"], r["n_m"], r["ic_2025"], r["ric_b"], r["ric_b_t"],
            r["ric_a"], r["ric_a_t"], r["ric_a_pos_m"], r["lift_hi"], r["lift_lo"], r["auc_out"], r["auc_out_t"]))
    return "\n".join(out)


def main():
    for src in sys.argv[1:]:
        one(src)


def one(src):
    R = run(src)
    txt = fmt_table(R)
    print(txt)
    (C.HERE / f"results_diag_{src}.txt").write_text(txt, encoding="utf-8")
    R.to_json(C.HERE / f"diag_{src}.json", orient="records", force_ascii=False, indent=0)
    C.ledger("feat_diag", src, "diag", n_cols=int(len(R)))


if __name__ == "__main__":
    main()
