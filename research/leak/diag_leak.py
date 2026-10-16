"""
第 4 步：泄漏强度的直接检验（不跑走向前，诊断）。

(a) 同一行 fwd - pit 的特征差 d 和未来除权 log q 的当日秩相关（按天求、对天 NW t）；|d| 和 log q 的相关
    （取整噪声的幅度 ∝ q，模型若能读到幅度就能读到 q）。生产特征（三层变换后的 __last 列）和 Alpha158 / 三族分别看。
(b) 「特征能不能预测未来除权」：LightGBM 用某一版特征预测 log q >= log 1.03 / >= log 1.10 / 未来 20 根内有除权，
    按代码随机对半分训练 / 测试，AUC 三版对比。fwd 比 pit / ctl 高出的部分就是取整通道能传多少未来信息。
(c) 背景：y_open 按未来除权分档（反向因果通道，和泄漏无关但解释了为什么泄漏值钱）。

    python research/leak/diag_leak.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import evalkit as K  # noqa: E402

L = []
DEV_FROM = "2024-06-01"


def say(s):
    CL.say(s, L)


def day_corr(x: np.ndarray, y: np.ndarray, day: np.ndarray) -> pd.Series:
    """按天 Pearson 相关（输入已是当日秩）。day 是整数编码。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, day = x[ok], y[ok], day[ok]
    n = np.bincount(day).astype(float)
    sx, sy = np.bincount(day, x), np.bincount(day, y)
    sxx, syy, sxy = np.bincount(day, x * x), np.bincount(day, y * y), np.bincount(day, x * y)
    with np.errstate(all="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        r = cov / np.sqrt(vx * vy)
    r[n < 100] = np.nan
    return pd.Series(r)


def rank_in_day(x: np.ndarray, day: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return s.groupby(day).rank(pct=True).to_numpy()


def part_a(kind: str, fa: pd.DataFrame, fb: pd.DataFrame, cols: list[str], logq: np.ndarray, day: np.ndarray, dates):
    rq = rank_in_day(logq, day)
    out = []
    for c in cols:
        a, b = fa[c].to_numpy(float), fb[c].to_numpy(float)
        d = a - b
        ad = np.abs(d)
        r1 = day_corr(rank_in_day(d, day), rq, day)
        r2 = day_corr(rank_in_day(ad, day), rq, day)
        # 还有一种更直接的：fwd 特征本身对 log q 的相关 减 pit 特征对 log q 的相关
        r3 = day_corr(a, rq, day) - day_corr(b, rq, day)
        out.append((c, float(np.nanmean(r1)), C.nwt(r1.dropna()), float(np.nanmean(r2)), C.nwt(r2.dropna()),
                    float(np.nanmean(r3)), C.nwt(r3.dropna()), float(np.nanmean(ad)), float(np.nanmean(np.abs(a) > 0))))
    df = pd.DataFrame(out, columns=["col", "r_d", "t_d", "r_absd", "t_absd", "r_diff", "t_diff", "mean_absd", "nz"])
    say("[a %s] %d 列：corr(d=fwd-pit 的当日秩, log q 的当日秩) 按天均值 中位 %.4f，|均值| 最大 5 列：%s" % (
        kind, len(df), float(df["r_d"].abs().median()),
        "; ".join("%s %+.3f (t %.1f)" % (r.col, r.r_d, r.t_d) for _, r in df.reindex(df["r_d"].abs().sort_values(ascending=False).index).head(5).iterrows())))
    say("[a %s] corr(|d| 的当日秩, log q 秩)：中位 %.4f，最大 5 列：%s；|t| >= 3 的列 %d 个" % (
        kind, float(df["r_absd"].median()),
        "; ".join("%s %+.3f (t %.1f)" % (r.col, r.r_absd, r.t_absd) for _, r in df.sort_values("r_absd", ascending=False).head(5).iterrows()),
        int((df["t_absd"].abs() >= 3).sum())))
    say("[a %s] corr(fwd 特征, log q) - corr(pit 特征, log q)：中位 %+.4f，|均值| 最大 5 列：%s；|t| >= 3 的列 %d 个" % (
        kind, float(df["r_diff"].median()),
        "; ".join("%s %+.4f (t %.1f)" % (r.col, r.r_diff, r.t_diff) for _, r in df.reindex(df["r_diff"].abs().sort_values(ascending=False).index).head(5).iterrows()),
        int((df["t_diff"].abs() >= 3).sum())))
    say("[a %s] |fwd - pit| 的均值（特征是当日秩或中性化后的秩，量纲 [0,1] 级）中位 %.4f，最大 %s" % (
        kind, float(df["mean_absd"].median()), "; ".join("%s %.4f" % (r.col, r.mean_absd) for _, r in df.sort_values("mean_absd", ascending=False).head(3).iterrows())))
    df.to_csv(CL.HERE / f"diag_a_{kind}.csv", index=False)
    return df


def part_b(sets: dict, targets: dict, codes: np.ndarray, seeds=(1, 2)):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    res = {}
    uc = np.unique(codes)
    for s in seeds:
        rng = np.random.default_rng(s)
        tr_codes = set(rng.choice(uc, len(uc) // 2, replace=False))
        tr = np.isin(codes, list(tr_codes))
        for tname, y in targets.items():
            ok = np.isfinite(y)
            for sname, X in sets.items():
                m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.7, n_jobs=12, verbose=-1, random_state=s)
                m.fit(X[tr & ok], y[tr & ok])
                p = m.predict_proba(X[~tr & ok])[:, 1]
                res.setdefault((tname, sname), []).append(float(roc_auc_score(y[~tr & ok], p)))
    for tname in targets:
        say("[b 预测未来除权 %s] AUC（两次按代码对半分的均值）：%s" % (
            tname, "  ".join("%s %.4f" % (sname, np.mean(res[(tname, sname)])) for sname in sets)))
    return res


def main():
    t0 = time.time()
    import json
    cache = json.loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
    feats = sorted({c for m in K.months_in(K.DEV) for c in cache[m]})
    last = [c for c in feats if c.endswith("__last")]
    px = pd.read_parquet(CL.LEAK / "px.parquet", columns=["code", "date", "q", "logq"])
    px = px[px["date"] >= DEV_FROM]
    # 未来 20 根内有除权：q(s)/q(s+20) >= 1.005
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    q20 = px.groupby("code")["q"].shift(-20)
    px["adj20"] = np.where(np.isfinite(q20), (px["q"] / q20 >= 1.005).astype(float), np.nan)
    rows = CL.snap_rows()
    rows = rows[rows["date"] >= DEV_FROM]
    lab = pd.read_parquet(CL.LEAK / "labels_open_snap.parquet", columns=["code", "date", "y_open"])
    base = rows.merge(lab, on=["code", "date"], how="left").merge(px, on=["code", "date"], how="inner")
    del rows, px, lab
    base = base.sort_values(["date", "code"]).reset_index(drop=True)
    day = pd.factorize(base["date"])[0]
    dates = base["date"].unique()
    logq = base["logq"].to_numpy(float)
    say("开发集行 %d，T 日之后有除权 log q >= log1.03 的占 %.1f%%，>= log1.10 的 %.1f%%，未来 20 根内有除权 %.1f%%（%.0fs）" % (
        len(base), 100 * float((logq >= np.log(1.03)).mean()), 100 * float((logq >= np.log(1.10)).mean()),
        100 * float(np.nanmean(base["adj20"])), time.time() - t0))
    # (c) 背景
    grp = np.select([logq < np.log(1.015), logq < np.log(1.03), logq < np.log(1.10), logq >= np.log(1.10)],
                    ["干净", "1.5~3%", "3~10%", ">=10%"], "?")
    yo = base["y_open"].to_numpy(float)
    parts = []
    for g in ("干净", "1.5~3%", "3~10%", ">=10%"):
        sel = (grp == g) & np.isfinite(yo)
        parts.append("%s n=%d %.2f%%" % (g, int(sel.sum()), 100 * float(yo[sel].mean())))
    r = day_corr(rank_in_day(logq, day), rank_in_day(np.where(np.isfinite(yo), yo, np.nan), day), day)
    say("[c 背景] y_open 按 T 日之后除权幅度：%s；log q 当日秩和 y_open 的相关按天均值 %+.4f NW t %.2f" % (
        "  ".join(parts), float(np.nanmean(r)), C.nwt(r.dropna())))

    # (a) 生产特征 fwd vs pit（__last 列）
    key = base[["code", "date"]]
    PF = {}
    for v in ("fwd", "pit", "ctl"):
        p = pd.read_parquet(CL.LEAK / f"prod_{v}.parquet", columns=["code", "date"] + last)
        PF[v] = key.merge(p, on=["code", "date"], how="left")
        del p
        assert len(PF[v]) == len(base)
    part_a("prod", PF["fwd"], PF["pit"], last, logq, day, dates)
    # 取整噪声幅度分档：|d| 按 q 分档（取每列均值的均值）
    ad = np.nanmean(np.abs(PF["fwd"][last].to_numpy(float) - PF["pit"][last].to_numpy(float)), axis=1)
    s = pd.Series(ad).groupby(grp).mean()
    say("[a prod] 83 列平均 |fwd-pit| 按未来除权分档：%s" % "  ".join("%s %.4f" % kv for kv in s.items()))
    adc = np.nanmean(np.abs(PF["ctl"][last].to_numpy(float) - PF["pit"][last].to_numpy(float)), axis=1)
    say("[a prod] 对照 ctl：83 列平均 |ctl-pit| 按未来除权分档：%s（应和 q 无关）" % "  ".join(
        "%s %.4f" % kv for kv in pd.Series(adc).groupby(grp).mean().items()))
    # (b) 的抽样矩阵边装边取，alt 三版不同时驻留（内存）
    rng = np.random.default_rng(0)
    sub = rng.random(len(base)) < 0.25
    codes = base["code"].to_numpy()[sub]
    sets = {}
    for v in ("fwd", "pit", "ctl"):
        sets[f"base_{v}"] = np.nan_to_num(PF[v][last].to_numpy(np.float32)[sub])
    AF = {}
    for v in ("fwd", "pit"):
        p = pd.read_parquet(CL.LEAK / f"alt_{v}.parquet")
        AF[v] = key.merge(p, on=["code", "date"], how="left")
        del p
    acols = [c for c in AF["fwd"].columns if c.startswith("a_") or c.startswith("f_")]
    part_a("alt", AF["fwd"], AF["pit"], acols, logq, day, dates)
    for v in ("fwd", "pit"):
        sets[f"cand_{v}"] = np.hstack([sets[f"base_{v}"], np.nan_to_num(AF[v][acols].to_numpy(np.float32)[sub])])
    del AF
    p = pd.read_parquet(CL.LEAK / "alt_ctl.parquet")
    a = key.merge(p, on=["code", "date"], how="left")
    del p
    sets["cand_ctl"] = np.hstack([sets["base_ctl"], np.nan_to_num(a[acols].to_numpy(np.float32)[sub])])
    del a, PF
    targets = {"logq>=log1.03": (logq[sub] >= np.log(1.03)).astype(float),
               "logq>=log1.10": (logq[sub] >= np.log(1.10)).astype(float),
               "20根内有除权": base["adj20"].to_numpy(float)[sub]}
    say("[b] 样本 %d 行（1/4 抽样），特征 base %d 列 / cand %d 列" % (int(sub.sum()), len(last), len(last) + len(acols)))
    part_b(sets, targets, codes)
    say("用时 %.0fs" % (time.time() - t0))
    (CL.HERE / "results_diag.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
