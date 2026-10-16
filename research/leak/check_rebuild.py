"""
第 3.5 步：重建流程对不对。
    prod_fwd 对 train_snap（生产训练表快照）：行对齐、逐列 |差|<1e-4 的比例、相关；对不上就先查流程再往下
    alt_fwd 对 data/breakout/raw/exp/altpath/a158.parquet 和 features/fam_{on,vc,hi}.parquet（都从 daily.parquet 算的）
    prod_pit / prod_ctl 对 prod_fwd：差异量级（应小而非零；为零说明价格没换）

    python research/leak/check_rebuild.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

L = []


def cmp(name, m, cols, sa="", sb="_re", tol=1e-4):
    rows = []
    for c in cols:
        a, b = m[c + sa].to_numpy(float), m[c + sb].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        agree = float((np.abs(a[ok] - b[ok]) < tol).mean()) if ok.any() else float("nan")
        nanmis = float((np.isfinite(a) != np.isfinite(b)).mean())
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 and a[ok].std() > 0 else float("nan")
        rows.append((c, agree, nanmis, corr))
    df = pd.DataFrame(rows, columns=["col", "agree", "nanmis", "cc"]).sort_values("agree")
    CL.say("[%s] %d 列：一致率（|差|<%g）中位 %.4f 最低 %.4f；相关中位 %.4f 最低 %.4f；NaN 位置不一致最大 %.4f；最低 6 列：%s" % (
        name, len(df), tol, df["agree"].median(), df["agree"].min(), df["cc"].median(), df["cc"].min(), df["nanmis"].max(),
        "; ".join("%s %.4f/r%.3f" % (r.col, r.agree, r.cc) for _, r in df.head(6).iterrows())), L)
    return df


def main():
    feats = [c for c in pd.read_parquet(CL.LEAK / "prod_fwd.parquet").columns if "__" in c]
    pf = pd.read_parquet(CL.LEAK / "prod_fwd.parquet")
    snap = pd.read_parquet(CL.LEAK / "train_snap.parquet", columns=["code", "date", "board"] + feats,
                           filters=[("date", "<=", CL.KEEP_TO)])
    snap["code"] = snap["code"].astype(str).str.zfill(6)
    snap["date"] = snap["date"].astype(str).str[:10]
    m = snap.merge(pf, on=["code", "date"], suffixes=("", "_re"), how="outer", indicator=True)
    CL.say("prod_fwd %d 行，train_snap(<=%s) %d 行，对上 %d，只在快照 %d（%d 只），只在重建 %d（%d 只）" % (
        len(pf), CL.KEEP_TO, len(snap), int((m["_merge"] == "both").sum()),
        int((m["_merge"] == "left_only").sum()), m.loc[m["_merge"] == "left_only", "code"].nunique(),
        int((m["_merge"] == "right_only").sum()), m.loc[m["_merge"] == "right_only", "code"].nunique()), L)
    m = m[m["_merge"] == "both"]
    CL.say("board 一致率 %.5f；日期范围 快照 %s..%s 重建 %s..%s" % (
        float((m["board"] == m["board_re"]).mean()), snap["date"].min(), snap["date"].max(), pf["date"].min(), pf["date"].max()), L)
    d1 = cmp("prod_fwd 对 train_snap", m, feats)
    d1.to_csv(CL.HERE / "check_prod_fwd_vs_snap.csv", index=False)
    del snap, m
    for v in ("pit", "ctl"):
        p = pd.read_parquet(CL.LEAK / f"prod_{v}.parquet")
        m = pf.merge(p, on=["code", "date"], suffixes=("", "_re"))
        assert len(m) == len(pf) == len(p), (len(m), len(pf), len(p))
        cmp(f"prod_{v} 对 prod_fwd", m, feats, tol=1e-6)
        del p, m
    del pf
    # alt
    af = pd.read_parquet(CL.LEAK / "alt_fwd.parquet")
    a158 = pd.read_parquet(K.EXP / "altpath" / "a158.parquet")
    a158["code"] = a158["code"].astype(str).str.zfill(6)
    ac = [c for c in a158.columns if c.startswith("a_")]
    m = a158.merge(af[["code", "date"] + ac], on=["code", "date"], suffixes=("", "_re"), how="outer", indicator=True)
    CL.say("alt_fwd(a158) %d 行，原 a158 %d 行，对上 %d" % (len(af), len(a158), int((m["_merge"] == "both").sum())), L)
    m = m[m["_merge"] == "both"]
    cmp("alt_fwd 的 a158 对原 a158.parquet（当日排名）", m, ac, tol=1e-3)
    del a158, m
    for f_ in ("on", "vc", "hi"):
        x = pd.read_parquet(K.EXP / "features" / f"fam_{f_}.parquet")
        x["code"] = x["code"].astype(str).str.zfill(6)
        fc = [c for c in x.columns if c.startswith("f_")]
        m = x.merge(af[["code", "date"] + fc], on=["code", "date"], suffixes=("", "_re"))
        CL.say("fam_%s 原 %d 行，对上 %d" % (f_, len(x), len(m)), L)
        cmp(f"alt_fwd 的 {f_} 对原 fam_{f_}（当日排名）", m, fc, tol=1e-3)
        del x, m
    acols = [c for c in af.columns if c.startswith("a_") or c.startswith("f_")]
    for v in ("pit", "ctl"):
        p = pd.read_parquet(CL.LEAK / f"alt_{v}.parquet")
        m = af.merge(p, on=["code", "date"], suffixes=("", "_re"))
        assert len(m) == len(af) == len(p)
        cmp(f"alt_{v} 对 alt_fwd", m, acols, tol=1e-6)
        del p, m
    (CL.HERE / "results_rebuild.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
