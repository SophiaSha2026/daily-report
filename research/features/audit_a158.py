"""
阶段 1 第 1 步：Alpha158 偷看审计（不训模型）。

    python research/features/audit_a158.py

四项：
  1 截断复算：随机 300 只票，把日线截在 T 日再算一遍 alpha158，T 日那行的原值必须和全段算的逐位相同
    （滑窗只向后看）；同一批票在 a158.parquet 里的当日名次和原值的顺序一致（排名是当天截面）。
  2 复权泄漏：前复权价把「T 日之后的除权」按比例缩进 T 日之前的价格。比率类不受影响，但取整到分会把未来的
    缩放幅度留在价格的颗粒度里。代理：g = 不复权均价 (amount/volume) / 前复权收盘，取 5 日中位；
    log g 偏离 0 就是 T 日之后还有除权（未来信息）。看 (a) log g 自己和 y_open 的关系，(b) 每一列 a158 和 log g
    的当日秩相关，(c) 只在 |log g| < 1% 的行（之后没有除权）里重排前 10，base_a158 对 base 的增益还在不在。
  3 停牌：滑窗按该票自己的 K 线数，跨停牌。统计开发集里窗口（60 根）跨过 >= 10 个市场交易日缺口的行占比，
    以及这类行在 base_a158 前 10 里的占比和命中。
  4 合并：a158 并进训练表的行数和缺失。
结果 results_audit.txt / audit.json。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_feats as BF  # noqa: E402  (research/altpath)

OUT = {}
L = []


def say(s):
    print(s, flush=True)
    L.append(s)


def load_px():
    px = pd.read_parquet(C.ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "volume", "amount"])
    px["date"] = px["date"].astype(str).str[:10]
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px[px["date"] <= "2025-12-31"]
    return px.sort_values(["code", "date"]).reset_index(drop=True)


def a158_raw(px):
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    keep = np.ones(len(px), bool)
    return BF.alpha158(px, keep)


def check_truncation(px):
    rng = np.random.default_rng(1)
    codes = rng.choice(px["code"].unique(), 300, replace=False)
    sub = px[px["code"].isin(codes)]
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        full = a158_raw(sub)
    cols = [c for c in full.columns if c.startswith("a_")]
    worst = 0.0
    nrows = 0
    for T in ("2024-09-24", "2025-01-15", "2025-06-30", "2025-11-20"):
        with contextlib.redirect_stdout(io.StringIO()):
            tr = a158_raw(sub[sub["date"] <= T])
        a = full[full["date"] == T].set_index("code")[cols]
        b = tr[tr["date"] == T].set_index("code")[cols].reindex(a.index)
        diff = np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))
        nan_mis = int((a.isna().to_numpy() != b.isna().to_numpy()).sum())
        worst = max(worst, float(diff))
        nrows += len(a)
        say("  截断在 %s：%d 只，%d 列，最大绝对差 %.2e，NaN 位置不一致 %d 处" % (T, len(a), len(cols), diff, nan_mis))
    OUT["truncation_max_absdiff"] = worst
    # 排名是当天截面：同一批票在 a158.parquet 里的名次顺序 = 原值顺序
    T = "2025-06-30"
    st = pd.read_parquet(C.K.EXP / "altpath" / "a158.parquet", filters=[("date", "==", T)])
    st = st[st["code"].isin(codes)].set_index("code")
    raw = full[full["date"] == T].set_index("code").reindex(st.index)
    bad = []
    for c in cols:
        x, y = raw[c].to_numpy(float), st[c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10:
            r = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
            if r < 0.9999:
                bad.append((c, round(float(r), 5)))
    n_all = pd.read_parquet(C.K.EXP / "altpath" / "a158.parquet", columns=["code"], filters=[("date", "==", T)])
    say("  排名：%s 当天 a158.parquet 有 %d 只（= 当日全市场），300 只子集里原值顺序和名次顺序不一致的列 %d 个 %s" % (
        T, len(n_all), len(bad), bad[:5]))
    OUT["rank_mismatch_cols"] = bad


def adj_factor(px):
    v = px["volume"].to_numpy(float)
    a = px["amount"].to_numpy(float)
    c = px["close"].to_numpy(float)
    with np.errstate(all="ignore"):
        r = np.where((v > 0) & (c > 0), a / v / c, np.nan)
    s = pd.Series(r)
    # 单位不一（手 / 股）会差 100 倍：按票取中位后归到最近的 10 的幂上不合适，直接用 5 日中位再除以该票末段中位
    g = s.groupby(px["code"].to_numpy()).transform(lambda x: x.rolling(5, min_periods=3).median())
    # 不复权均价 / 前复权收盘：之后没有除权时约等于 1（日内均价和收盘差 1~2%），之后有除权时 > 1
    lg = np.log(g)
    return pd.DataFrame({"code": px["code"], "date": px["date"], "lg": lg.to_numpy()})


def check_adjust(px):
    t0 = time.time()
    f = adj_factor(px)
    f = f[(f["date"] >= "2024-06-01")]
    lab = pd.read_parquet(C.K.DATA / "raw" / "labels_open.parquet", columns=["code", "date", "y_open"],
                          filters=[("date", ">=", "2024-06-01"), ("date", "<", "2026-01-01")])
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    x = f.merge(lab, on=["code", "date"], how="inner")
    x = x[np.isfinite(x["y_open"]) & np.isfinite(x["lg"])]
    # 末段 20 根里有除权的票，last 本身不准；只看量级
    adj = (x["lg"] >= 0.03)
    clean = (x["lg"] < 0.015)
    say("  开发集行 %d，T 日之后还有除权（log g >= 3%%）的行占 %.1f%%，干净（< 1.5%%）%.1f%%（%.0fs）" % (len(x), 100 * adj.mean(), 100 * clean.mean(), time.time() - t0))
    say("  命中：之后有除权 %.2f%%，之后没有 %.2f%%" % (100 * x.loc[adj, "y_open"].mean(), 100 * x.loc[~adj, "y_open"].mean()))
    # log g 和 y_open 的当日秩相关（未来信息本身有多大）
    x["lg_r"] = x.groupby("date")["lg"].rank(pct=True)
    ic = x.groupby("date").apply(lambda d: d["lg_r"].corr(d["y_open"]))
    say("  log g 当日排名和 y_open 的相关：均值 %+.4f，NW t %.2f" % (ic.mean(), C.nwt(ic.dropna())))
    OUT["adj_share"] = float(adj.mean())
    OUT["lg_ic"] = float(ic.mean())
    # a158 每列和 log g 的当日秩相关（抽 40 个交易日）
    dates = sorted(x["date"].unique())[::10]
    a = pd.read_parquet(C.K.EXP / "altpath" / "a158.parquet", filters=[("date", "in", dates)])
    a["code"] = a["code"].astype(str).str.zfill(6)
    y = a.merge(x[["code", "date", "lg_r"]], on=["code", "date"], how="inner")
    cols = [c for c in a.columns if c.startswith("a_")]
    cc = {}
    for c in cols:
        cc[c] = float(y.groupby("date").apply(lambda d: d[c].corr(d["lg_r"], method="spearman")).mean())
    s = pd.Series(cc).sort_values(key=np.abs, ascending=False)
    say("  a158 各列和 log g 的当日秩相关（%d 天）：|r| 中位 %.3f，最大 5 列 %s" % (
        len(dates), s.abs().median(), ", ".join("%s %+.3f" % (k, v) for k, v in s.head(5).items())))
    OUT["a158_lg_corr_top"] = s.head(10).to_dict()
    # 生产特征同样看一眼（同样是前复权价算的）
    # 只在之后没有除权的行里重排前 10
    keep = x.loc[clean, ["code", "date"]]
    import validate as V
    st = V.st_codes()
    res = {}
    for arm, exp in (("base", "altpath_v1"), ("base_a158", "altpath_v1")):
        p = pd.read_parquet(C.K.EXP / exp / f"all_{arm}.parquet")
        p["code"] = p["code"].astype(str).str.zfill(6)
        p = p.merge(keep, on=["code", "date"]).merge(lab, on=["code", "date"])
        p = p[~p["code"].isin(st)].sort_values(["date", "p"], ascending=[True, False])
        p["rank"] = p.groupby("date").cumcount() + 1
        res[arm] = p[p["rank"] <= 200]
    d = C.daily_diff(res["base_a158"], res["base"])
    say("  只在之后没有除权的行里重排：base %.2f%% base_a158 %.2f%%，差 %+.2fpp NW t %.2f" % (
        100 * C.daily_hit(res["base"]).mean(), 100 * C.daily_hit(res["base_a158"]).mean(), 100 * d.mean(), C.nwt(d)))
    OUT["noadj_rerank"] = {"diff": float(d.mean()), "t_nw": C.nwt(d)}
    # 原样前 10 里有多少是「之后有除权」的
    for arm in ("base", "base_a158"):
        t = C.top("altpath_v1", arm)
        t = t[t["rank"] <= 10].merge(x[["code", "date", "lg"]], on=["code", "date"], how="left")
        say("  %s 前 10 里之后有除权（>=3%%）的占 %.1f%%，命中 %.2f%% / 其余 %.2f%%" % (arm, 100 * (t["lg"] >= 0.03).mean(), 100 * t.loc[t["lg"] >= 0.03, "y_open"].mean(), 100 * t.loc[~(t["lg"] >= 0.03), "y_open"].mean()))


def check_suspension(px):
    dates = np.array(sorted(px["date"].unique()))
    di = {d: i for i, d in enumerate(dates)}
    px = px[["code", "date"]].copy()
    px["i"] = px["date"].map(di).to_numpy()
    px["i60"] = px.groupby("code")["i"].shift(59)
    px["span"] = px["i"] - px["i60"]          # 60 根 K 线横跨多少个市场交易日
    x = px[px["date"] >= "2024-06-01"]
    gap = x["span"] >= 69
    say("  开发集：60 根窗口跨过 >= 10 个交易日缺口（停牌）的行占 %.2f%%" % (100 * gap.mean()))
    for arm in ("base", "base_a158"):
        t = C.top("altpath_v1", arm)
        t = t[t["rank"] <= 10].merge(x[["code", "date", "span"]], on=["code", "date"], how="left")
        g = t["span"] >= 69
        say("  %s 前 10 里这类行占 %.2f%%，命中 %.2f%%（其余 %.2f%%）" % (
            arm, 100 * g.mean(), 100 * t.loc[g, "y_open"].mean() if g.any() else float("nan"),
            100 * t.loc[~g, "y_open"].mean()))
    OUT["susp_share"] = float(gap.mean())


def main():
    t0 = time.time()
    px = load_px()
    say("[1] 截断复算 + 排名截面")
    check_truncation(px)
    say("[2] 前复权泄漏")
    check_adjust(px)
    say("[3] 停牌")
    check_suspension(px)
    say("用时 %.0fs" % (time.time() - t0))
    (C.HERE / "results_audit.txt").write_text("\n".join(L), encoding="utf-8")
    (C.HERE / "audit.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
