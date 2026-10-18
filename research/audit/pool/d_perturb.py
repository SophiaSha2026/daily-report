"""审计 D：特征时点的扰动不变性测试。

抽 N 只票的子集，把 T* 之后所有行的价（和量）随机扰动，用**各研究脚本原来的函数**在原始 / 扰动两份子集上各建一遍表，
T* 及以前的行所有非标签列必须逐位相同。覆盖：
    rulecore  common_rc.build_features        -> feat.parquet（ret20 / vol60 / turn20 / amp20 / z_ / elig0 / susp20 / nbar）
    polish    build_pool.main                 -> pool.parquet（扩展特征 + 事件标志 + pool）
    gate      build_gt.main                   -> table.parquet（名次 / 日级市场状态 / 非价格派生）
    altdata   features.main                   -> top60.parquet
    deepdata  feat_events / feat_sw / feat_lhb_seats / feat_rating / feat_rzrq_short / feat_ths / feat_freeholders -> features_*.parquet
    生产      build.attach_turnover + assemble + holder_features + add_market + cross_section + neutralize + interactions + 5 日聚合
             （照抄 build.build 的步骤，不写 data/）
所有路径常量都改到 research/audit/pool/_tmp/ 下，不碰任何生产 / 研究产物。
    PYTHONIOENCODING=utf-8 python research/audit/pool/d_perturb.py [--n 350] [--tstar 2025-06-30] [--seed 7]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "rulecore", ROOT / "research" / "polish",
          ROOT / "research" / "precision", ROOT / "research" / "gate", ROOT / "research" / "screen", ROOT / "research" / "deepdata"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
HERE = ROOT / "research/audit/pool"
TMP = HERE / "_tmp"
L = []
LABEL_COLS = {"y_up", "y_open", "r20_open", "yizi1", "dd20", "c20", "patched", "y_t0", "y_top", "hit", "yizi1_u", "y_open_u", "end20"}
PRICE_COLS = ["open", "high", "low", "close", "close_u", "low_u"]
VOL_COLS = ["volume", "amount"]


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def perturb(px: pd.DataFrame, tstar: str, seed: int) -> pd.DataFrame:
    p = px.copy()
    m = (p["date"] > tstar).to_numpy()
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.7, 1.3, int(m.sum()))
    g = rng.uniform(0.5, 2.0, int(m.sum()))
    for c in PRICE_COLS:
        if c in p.columns:
            p.loc[m, c] = p.loc[m, c].to_numpy() * f
    for c in VOL_COLS:
        if c in p.columns:
            p.loc[m, c] = p.loc[m, c].to_numpy() * g
    if "turnover" in p.columns:
        p.loc[m, "turnover"] = p.loc[m, "turnover"].to_numpy() * g
    return p


def compare(name: str, a: pd.DataFrame, b: pd.DataFrame, tstar: str, keys=("code", "date")) -> dict:
    """T* 及以前的行逐列比。返回 {列: (不同行数, 最早不同日期)}，只列非标签列。"""
    a = a[a["date"] <= tstar]
    b = b[b["date"] <= tstar]
    m = a.merge(b, on=list(keys), how="outer", suffixes=("_a", "_b"), indicator=True)
    bad = {}
    n_only = int((m["_merge"] != "both").sum())
    cols = [c for c in a.columns if c not in keys]
    for c in cols:
        xa, xb = m[c + "_a"], m[c + "_b"]
        if xa.dtype == object or str(xa.dtype).startswith("string") or xa.dtype == bool:
            diff = ~((xa == xb) | (xa.isna() & xb.isna()))
        else:
            va, vb = xa.to_numpy(float), xb.to_numpy(float)
            diff = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
        if diff.any():
            bad[c] = (int(diff.sum()), str(m.loc[diff, "date"].min()))
    feat_bad = {k: v for k, v in bad.items() if k not in LABEL_COLS}
    lab_bad = {k: v for k, v in bad.items() if k in LABEL_COLS}
    say("[%s] 行 %d（只在一边 %d）列 %d；T*=%s 及以前 非标签列不同：%d 个 %s；标签列不同（预期）：%s" % (
        name, len(m), n_only, len(cols), tstar, len(feat_bad),
        {k: v for k, v in list(feat_bad.items())[:12]}, {k: v for k, v in list(lab_bad.items())[:8]}))
    return {"name": name, "rows": len(m), "n_only": n_only, "feat_bad": feat_bad, "label_bad": lab_bad}


def run_pipelines(tag: str, px_sub: pd.DataFrame, lab_sub: pd.DataFrame, daily_sub: pd.DataFrame, holders) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    d = TMP / tag
    scr, rc, pl, gt, alt, dd = d / "screen", d / "rulecore", d / "polish", d / "gate", d / "altdata", d / "deepdata"
    for x in (scr, rc, pl, gt, alt, dd):
        x.mkdir(parents=True, exist_ok=True)
    px_sub.to_parquet(scr / "px_screen.parquet", index=False)
    lab_sub.to_parquet(scr / "labels_screen.parquet", index=False)
    t0 = time.time()
    # ---- rulecore ----
    import common_rc as C
    C.SCR, C.RC = scr, rc
    C.PROGRESS = d / "progress_rc.txt"
    try:
        feat = C.build_features(force=True)
        out["rulecore.feat"] = feat
        say("  %s rulecore 完 %d 行 %.0fs" % (tag, len(feat), time.time() - t0))
    except Exception:
        say("  %s rulecore 失败：%s" % (tag, traceback.format_exc()[-800:]))
        return out
    # ---- polish ----
    import common_pl as P
    P.PL = pl
    P.PROGRESS = d / "progress_pl.txt"
    import build_pool as BP
    BP.SCR, BP.RC = scr, rc
    try:
        BP.main()
        out["polish.pool"] = pd.read_parquet(pl / "pool.parquet")
        say("  %s polish 完 %d 行 %.0fs" % (tag, len(out["polish.pool"]), time.time() - t0))
    except Exception:
        say("  %s polish 失败：%s" % (tag, traceback.format_exc()[-800:]))
        return out
    # ---- gate ----
    import common_gt as G
    G.TABLE = gt / "table.parquet"
    G.HERE = gt
    import build_gt as BG
    try:
        BG.main()
        out["gate.table"] = pd.read_parquet(gt / "table.parquet")
        say("  %s gate 完 %d 行 %.0fs" % (tag, len(out["gate.table"]), time.time() - t0))
    except Exception:
        say("  %s gate 失败：%s" % (tag, traceback.format_exc()[-800:]))
    # ---- altdata ----
    try:
        spec = importlib.util.spec_from_file_location("altdata_features_audit_" + tag, ROOT / "research/altdata/features.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.HERE = alt
        mod.RAW = ROOT / "research/altdata/raw"
        mod.main()
        out["altdata.top60"] = pd.read_parquet(alt / "top60.parquet")
        say("  %s altdata 完 %d 行 %.0fs" % (tag, len(out["altdata.top60"]), time.time() - t0))
    except Exception:
        say("  %s altdata 失败：%s" % (tag, traceback.format_exc()[-800:]))
    # ---- deepdata ----
    try:
        import common_dd as CD
        CD.HERE, CD.README, CD.PROGRESS = dd, dd / "features_README.txt", dd / "progress.txt"
        CD.RAW = ROOT / "research/deepdata/raw"
        for nm in ("feat_events", "feat_sw", "feat_lhb_seats", "feat_rating", "feat_rzrq_short", "feat_ths", "feat_freeholders"):
            try:
                m = importlib.import_module(nm)
                importlib.reload(m)
                m.C = CD
                m.main()
                say("  %s deepdata.%s 完 %.0fs" % (tag, nm, time.time() - t0))
            except Exception:
                say("  %s deepdata.%s 失败：%s" % (tag, nm, traceback.format_exc()[-600:]))
        for f in sorted(dd.glob("features_*.parquet")):
            out["deepdata." + f.stem] = pd.read_parquet(f)
    except Exception:
        say("  %s deepdata 失败：%s" % (tag, traceback.format_exc()[-800:]))
    # ---- 生产特征（照抄 build.build 的步骤）----
    try:
        import build as B
        import features as F
        daily = daily_sub.copy()
        daily, start = B.trim_common_start(daily)
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
        out["prod.train"] = panel[keep].reset_index(drop=True)
        say("  %s 生产特征完 %d 行（共同起点 %s）%.0fs" % (tag, len(out["prod.train"]), start, time.time() - t0))
    except Exception:
        say("  %s 生产特征失败：%s" % (tag, traceback.format_exc()[-1200:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--tstar", default="2025-06-30")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    px = pd.read_parquet(EXP / "screen/px_screen.parquet")
    px["code"] = px["code"].astype(str).str.zfill(6)
    cnt = px.groupby("code").size()
    rng = np.random.default_rng(a.seed)
    pool_codes = cnt[cnt >= 400].index.to_numpy()
    codes = set(rng.choice(pool_codes, size=min(a.n, len(pool_codes)), replace=False))
    # 保证各板块都有
    for pre in ("30", "68", "92", "60", "00"):
        extra = [c for c in pool_codes if c.startswith(pre)][:5]
        codes.update(extra)
    codes = sorted(codes)
    say("子集 %d 只（板块：%s）T*=%s" % (len(codes), pd.Series([("bj" if c[0] in "984" else "star" if c[:2] == "68" else "chinext" if c[0] == "3" else "main") for c in codes]).value_counts().to_dict(), a.tstar))
    px_sub = px[px["code"].isin(codes)].reset_index(drop=True)
    del px
    lab = pd.read_parquet(EXP / "screen/labels_screen.parquet")
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    lab_sub = lab[lab["code"].isin(codes)].reset_index(drop=True)
    del lab
    daily = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str).str[:10]
    daily_sub = daily[daily["code"].isin(codes) & (daily["date"] <= "2025-12-31")].reset_index(drop=True)
    del daily
    hp = ROOT / "data/breakout/holders.parquet"
    holders = pd.read_parquet(hp) if hp.exists() else None
    say("px 子集 %d 行，labels %d 行，daily 子集 %d 行 %.0fs" % (len(px_sub), len(lab_sub), len(daily_sub), time.time() - t0))
    # 原始 / 扰动
    A = run_pipelines("orig", px_sub, lab_sub, daily_sub, holders)
    px_p = perturb(px_sub, a.tstar, a.seed)
    daily_p = perturb(daily_sub, a.tstar, a.seed + 1)
    chk = (px_p[px_p["date"] <= a.tstar][PRICE_COLS].to_numpy() == px_sub[px_sub["date"] <= a.tstar][PRICE_COLS].to_numpy()).all()
    chk2 = (px_p[px_p["date"] > a.tstar]["close"].to_numpy() != px_sub[px_sub["date"] > a.tstar]["close"].to_numpy()).mean()
    say("扰动自检：T* 及以前价格逐位相同 %s；T* 之后 close 改变的比例 %.4f（教训 32：先断言 A、B 真的不一样）" % (chk, chk2))
    Bq = run_pipelines("pert", px_p, lab_sub, daily_p, holders)
    say("\n== 对比（T* 及以前的行，非标签列必须 0 个不同）==")
    res = []
    for k in A:
        if k in Bq:
            res.append(compare(k, A[k], Bq[k], a.tstar))
        else:
            say("[%s] 扰动版没建出来" % k)
    n_bad = sum(1 for r in res if r["feat_bad"])
    say("\n结论：%d 张表里 %d 张有 T* 前的非标签列随未来价格改变：%s" % (len(res), n_bad, [r["name"] for r in res if r["feat_bad"]]))
    # 细节：每张有问题的表，列出全部坏列
    for r in res:
        if r["feat_bad"]:
            say("  %s 坏列全表：%s" % (r["name"], r["feat_bad"]))
    say("用时 %.0f 分钟" % ((time.time() - t0) / 60))
    (HERE / "results_d_perturb.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
