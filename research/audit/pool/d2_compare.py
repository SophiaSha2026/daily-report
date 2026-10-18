"""审计 D 的对比步（d_perturb.py 的对比在字符串列上崩了；各表已落在 _tmp/orig、_tmp/pert，这里只读盘对比，
生产特征那一张没落盘，用同一子集、同一扰动再算一遍）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
HERE = ROOT / "research/audit/pool"
TMP = HERE / "_tmp"
TSTAR = "2025-06-30"
L = []
LABEL_COLS = {"y_up", "y_open", "r20_open", "yizi1", "dd20", "c20", "patched", "y_t0", "y_top", "hit", "yizi1_u", "y_open_u", "end20"}


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def compare(name, a, b, tstar=TSTAR, keys=("code", "date")):
    a = a[a["date"].astype(str) <= tstar].copy()
    b = b[b["date"].astype(str) <= tstar].copy()
    for t in (a, b):
        t["code"] = t["code"].astype(str).str.zfill(6)
        t["date"] = t["date"].astype(str).str[:10]
    m = a.merge(b, on=list(keys), how="outer", suffixes=("_a", "_b"), indicator=True)
    n_only = int((m["_merge"] != "both").sum())
    bad = {}
    for c in [c for c in a.columns if c not in keys]:
        xa, xb = m[c + "_a"], m[c + "_b"]
        if pd.api.types.is_numeric_dtype(xa) and pd.api.types.is_numeric_dtype(xb) and not pd.api.types.is_bool_dtype(xa):
            va, vb = xa.to_numpy(float), xb.to_numpy(float)
            diff = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
        else:
            sa, sb = xa.astype(str), xb.astype(str)
            diff = ~(sa == sb).to_numpy()
        if diff.any():
            bad[c] = (int(diff.sum()), str(m.loc[diff, "date"].min()), str(m.loc[diff, "date"].max()))
    fb = {k: v for k, v in bad.items() if k not in LABEL_COLS}
    lb = {k: v for k, v in bad.items() if k in LABEL_COLS}
    say("[%s] 行 %d（只在一边 %d）列 %d -> T* 前非标签列不同 %d 个：%s" % (name, len(m), n_only, len([c for c in a.columns if c not in keys]), len(fb), fb))
    if lb:
        say("    标签列不同（预期，窗口伸进 T* 之后）：%s" % lb)
    return fb


def prod_features(daily_sub, holders):
    import importlib.util as iu
    spec = iu.spec_from_file_location("breakout_build_mod", ROOT / "src/breakout/build.py")
    B = iu.module_from_spec(spec)
    sys.modules["breakout_build_mod"] = B
    spec.loader.exec_module(B)
    import features as F
    daily, start = B.trim_common_start(daily_sub.copy())
    daily = B.attach_turnover(daily, None)
    panel = B.assemble(daily, None)
    panel = F.holder_features(panel, holders)
    panel = F.add_market(panel)
    pdates = sorted(panel["date"].unique())
    if len(pdates) > B.WARMUP:
        panel = panel[panel["date"] >= pdates[B.WARMUP]].reset_index(drop=True)
    cols = [c for c in F.feature_columns() if c not in {n for *_, n in F.INTERACTIONS}]
    cols = [c for c in cols if c in panel.columns]
    panel = F.cross_section(panel, cols)
    panel = F.neutralize(panel, cols)
    panel = F.add_interactions(panel)
    feats = [c for c in F.feature_columns() if c in panel.columns]
    panel = panel.sort_values(["code", "date"])
    g = panel.groupby("code", sort=False)
    agg = {}
    for c in feats:
        agg[f"{c}__last"] = panel[c]
        agg[f"{c}__mean"] = g[c].transform(lambda s: s.rolling(B.WIN).mean())
        agg[f"{c}__slope"] = g[c].transform(lambda s: (s - s.shift(B.WIN - 1)) / (B.WIN - 1))
    panel = pd.concat([panel, pd.DataFrame(agg, index=panel.index)], axis=1)
    keep = ["code", "date", "board", "close", "turn_est", "float_mcap", "y_up", "y_t0", "y_top"] + [c for c in panel.columns if "__" in c]
    return panel[keep].reset_index(drop=True)


def main():
    t0 = time.time()
    tables = {"rulecore.feat": "rulecore/feat.parquet", "polish.pool": "polish/pool.parquet", "gate.table": "gate/table.parquet", "altdata.top60": "altdata/top60.parquet"}
    for f in sorted((TMP / "orig" / "deepdata").glob("features_*.parquet")):
        tables["deepdata." + f.stem] = "deepdata/" + f.name
    say("== 研究表（已落盘）==")
    res = {}
    for k, rel in tables.items():
        pa, pb = TMP / "orig" / rel, TMP / "pert" / rel
        if not (pa.exists() and pb.exists()):
            say("[%s] 缺文件 %s" % (k, pb))
            continue
        res[k] = compare(k, pd.read_parquet(pa), pd.read_parquet(pb))
    say("\n== 生产特征（同一子集、同一扰动重算）==")
    import importlib.util
    spec = importlib.util.spec_from_file_location("d_perturb_mod", HERE / "d_perturb.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    codes = sorted(pd.read_parquet(TMP / "orig/screen/px_screen.parquet", columns=["code"])["code"].astype(str).unique())
    daily = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str).str[:10]
    daily_sub = daily[daily["code"].isin(codes) & (daily["date"] <= "2025-12-31")].reset_index(drop=True)
    del daily
    hp = ROOT / "data/breakout/holders.parquet"
    holders = pd.read_parquet(hp) if hp.exists() else None
    A = prod_features(daily_sub, holders)
    daily_p = D.perturb(daily_sub, TSTAR, 8)
    chg = float((daily_p.loc[daily_p["date"] > TSTAR, "close"].to_numpy() != daily_sub.loc[daily_sub["date"] > TSTAR, "close"].to_numpy()).mean())
    say("daily 子集 %d 行 %d 只；扰动后 T* 之后 close 改变比例 %.4f" % (len(daily_sub), len(codes), chg))
    Bp = prod_features(daily_p, holders)
    res["prod.train"] = compare("prod.train", A, Bp)
    n_bad = [k for k, v in res.items() if v]
    say("\n结论：%d 张表，T* 前非标签列随未来价格改变的：%d 张 %s" % (len(res), len(n_bad), n_bad))
    say("用时 %.0fs" % (time.time() - t0))
    (HERE / "results_d_perturb.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
