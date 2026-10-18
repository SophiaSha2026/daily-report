"""审计 B：候选池两份实现（生产 daily.rule_frame + stage_scan 的 elig_r 对 研究 common_rc 的 elig0 / pool）逐日对比。只读。

三组对比：
  (a) 同一天、各自的数据：生产代码跑在 daily.parquet + train.parquet 上，研究池直接读 rulecore/feat.parquet（elig0）和 polish/pool.parquet（事件剔除后 pool）。
      日期 = 2025-10-01..2025-12-31 的研究交易日（研究表到 2025-12-31，生产清单 2026-09 起才有，没有重叠日，所以用两套代码各算一遍）。
  (b) 同一天、同一份数据：把 common_rc 的候选池逻辑原样跑在 daily.parquet 上（研究代码路径），和生产代码路径比，隔离「代码差」和「数据差」。
      日期 = (a) 的日期 + 生产清单存在的 2026-09-07..2026-10-16。
  (c) 前 10：按 log1p(ret20) 排的前 10（剔事件前 / 后）的 Jaccard。
产物：results_b_pool.txt、b_diff_rows.csv（每个差异 (日, 码) 一行，带原因）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "rulecore", ROOT / "research" / "screen", ROOT / "research" / "stats"):
    sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
OUT = ROOT / "research/audit/pool"
L = []


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def jac(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def research_frame(px: pd.DataFrame) -> pd.DataFrame:
    """common_rc.build_features 的候选池部分，原样搬（不含标签 / z 化）。px 列 code date high low close turnover。"""
    import daily as D
    import validate as V
    import common_rc as C
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    px["ret20"] = g["close"].pct_change(20)
    px["lret20"] = np.log1p(px["ret20"].clip(lower=-0.99))
    lr = np.log(px["close"]).groupby(px["code"]).diff()
    px["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std())
    px["amp"] = px["high"] / px["low"] - 1
    px["amp20"] = px.groupby("code", sort=False)["amp"].transform(lambda s: s.rolling(20).mean())
    px["turn20"] = px.groupby("code", sort=False)["turnover"].transform(lambda s: s.rolling(20).mean())
    px["nbar"] = g.cumcount() + 1
    dates = np.array(sorted(px["date"].unique()))
    didx = pd.Series(np.arange(len(dates)), index=dates)
    px["_di"] = px["date"].map(didx).to_numpy()
    gap = (px["_di"] - px.groupby("code", sort=False)["_di"].shift(1) - 1).fillna(0)
    px["susp20"] = gap.groupby(px["code"]).transform(lambda s: s.rolling(20, min_periods=1).max()) >= C.SUSP_GAP
    st = V.st_codes()
    px["is_st"] = px["code"].isin(st)
    px["fin"] = np.isfinite(px[list(C.INPUTS)]).all(axis=1)
    px["elig0_nost"] = ((px["close"] >= C.PRICE_MIN) & (px["nbar"] >= D.MIN_HISTORY_DAYS) & ~px["susp20"] & px["fin"])
    px["elig0"] = px["elig0_nost"] & ~px["is_st"]
    return px


def main():
    t0 = time.time()
    import daily as D
    say("生产常量：RANK_RULE=%s RULE_MIN_BARS=%d RULE_PRICE_MIN=%.1f RULE_SUSP_GAP=%d RULE_WINDOW=%d MIN_HISTORY_DAYS=%d" % (
        D.RANK_RULE, D.RULE_MIN_BARS, D.RULE_PRICE_MIN, D.RULE_SUSP_GAP, D.RULE_WINDOW, D.MIN_HISTORY_DAYS))
    daily = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "high", "low", "close", "turnover"])
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str).str[:10]
    train = pd.read_parquet(ROOT / "data/breakout/train.parquet", columns=["code", "date", "close", "board"])
    train["code"] = train["code"].astype(str).str.zfill(6)
    train["date"] = train["date"].astype(str).str[:10]
    say("daily %d 行 %s..%s %d 只；train %d 行 %s..%s %d 只 %.0fs" % (len(daily), daily["date"].min(), daily["date"].max(), daily["code"].nunique(),
                                                               len(train), train["date"].min(), train["date"].max(), train["code"].nunique(), time.time() - t0))
    feat = pd.read_parquet(EXP / "rulecore/feat.parquet", columns=["code", "date", "close", "ret20", "nbar", "susp20", "elig0", "lret20", "vol60", "turn20", "amp20", "board"])
    feat["code"] = feat["code"].astype(str).str.zfill(6)
    feat = feat[feat["date"] >= "2025-09-01"].reset_index(drop=True)
    pool = pd.read_parquet(EXP / "polish/pool.parquet", columns=["code", "date", "pool", "ev_reduce30", "ev_release30", "ev_placement60", "ev_any", "lret20"])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool = pool[pool["date"] >= "2025-09-01"].reset_index(drop=True)
    import validate as V
    st = V.st_codes()
    say("研究表 feat %d 行（2025-09 起）pool %d 行；静态 ST 名单 %d 只" % (len(feat), len(pool), len(st)))

    dates_a = sorted(d for d in feat["date"].unique() if "2025-10-01" <= d <= "2025-12-31")
    dates_b = sorted(p.name[len("breakout_"):-len(".parquet")] for p in (ROOT / "data/breakout").glob("2026-*/breakout_*.parquet"))
    dates_b = [d for d in dates_b if d in set(daily["date"])]
    say("(a) 日期 %d 个 %s..%s；(b) 生产日期 %d 个 %s..%s" % (len(dates_a), dates_a[0], dates_a[-1], len(dates_b), dates_b[0], dates_b[-1]))

    # 研究代码路径跑在 daily.parquet 上（一次算完）
    rf = research_frame(daily[daily["date"] <= max(dates_b)].copy())
    say("研究代码路径在 daily 上算完 %.0fs" % (time.time() - t0))

    diff_rows = []
    summ = []
    for date in dates_a + dates_b:
        is_a = date in set(dates_a)
        rule = D.rule_frame(daily[["code", "date", "close"]], date)          # 生产的三样
        today = train[train["date"] == date].copy()
        today = today.merge(rule, on="code", how="left")
        elig_r = ((today["nbar"].fillna(0) >= D.RULE_MIN_BARS) & (today["close"] >= D.RULE_PRICE_MIN)
                  & ~today["susp20"].fillna(True).astype(bool) & np.isfinite(today["ret20"]))
        P = today[elig_r].copy()
        P["rank_key"] = np.log1p(P["ret20"].clip(lower=-0.99))
        Pset = set(P["code"])
        P10 = set(P.sort_values("rank_key", ascending=False).head(10)["code"])
        # 研究代码路径（同一份 daily）
        r = rf[rf["date"] == date]
        Rd_nost = set(r.loc[r["elig0_nost"], "code"])
        Rd = set(r.loc[r["elig0"], "code"])
        rec = {"date": date, "seg": "a" if is_a else "b", "n_prod": len(Pset), "n_res_daily_noST": len(Rd_nost), "n_res_daily": len(Rd),
               "prod_minus_resdaily": len(Pset - Rd_nost), "resdaily_minus_prod": len(Rd_nost - Pset),
               "prod_in_staticST": len(Pset & st)}
        # 同数据差异的原因
        for c in sorted(Pset - Rd_nost):
            x = r[r["code"] == c]
            if not len(x):
                why = "研究路径没有这只（daily 里当天没有 K 线?）"
            else:
                x = x.iloc[0]
                why = ";".join(w for w, f in (("nbar<120", x["nbar"] < D.MIN_HISTORY_DAYS), ("close<2", x["close"] < 2), ("susp20(研究法)", bool(x["susp20"])),
                                              ("输入NaN(vol60/turn20/amp20)", not bool(x["fin"]))) if f) or "?"
            diff_rows.append({"date": date, "code": c, "cmp": "同数据:生产有研究无", "why": why})
        for c in sorted(Rd_nost - Pset):
            x = today[today["code"] == c]
            if not len(x):
                why = "不在 train.parquet 当天行里"
            else:
                x = x.iloc[0]
                why = ";".join(w for w, f in (("nbar<120", x["nbar"] < D.RULE_MIN_BARS), ("close<2", x["close"] < 2), ("susp20(生产法)", bool(x["susp20"])),
                                              ("ret20 NaN", not np.isfinite(x["ret20"]))) if f) or "?"
            diff_rows.append({"date": date, "code": c, "cmp": "同数据:研究有生产无", "why": why})
        if is_a:
            f = feat[feat["date"] == date]
            R0 = set(f.loc[f["elig0"], "code"])
            pl = pool[pool["date"] == date]
            Rp = set(pl.loc[pl["pool"], "code"])
            R10 = set(pl[pl["pool"]].sort_values("lret20", ascending=False).head(10)["code"])
            R10_pre = set(f[f["elig0"]].sort_values("lret20", ascending=False).head(10)["code"])
            # 生产池 + 研究事件标志（能拿到标志的才剔）
            ev = pl.set_index("code")["ev_any"]
            Pev = P[~P["code"].map(ev).fillna(False).astype(bool)]
            P10_ev = set(Pev.sort_values("rank_key", ascending=False).head(10)["code"])
            rec.update({"n_res_px_elig0": len(R0), "n_res_pool": len(Rp), "prod_minus_res": len(Pset - R0), "res_minus_prod": len(R0 - Pset),
                        "prod_minus_res_exST": len((Pset - R0) - st), "jac_pool": jac(Pset - st, R0),
                        "jac_top10_pre": jac(P10, R10_pre), "jac_top10_ev": jac(P10_ev, R10), "n_top10_ev_common": len(P10_ev & R10)})
            # ret20 数值差（共同代码）
            m = P[["code", "ret20"]].merge(f[["code", "ret20"]], on="code", suffixes=("_prod", "_res"))
            if len(m):
                dd = (m["ret20_prod"] - m["ret20_res"]).abs()
                rec.update({"ret20_absdiff_med": float(dd.median()), "ret20_absdiff_max": float(dd.max()), "ret20_absdiff_gt1e3": int((dd > 1e-3).sum())})
            for c in sorted(Pset - R0):
                x = f[f["code"] == c]
                if not len(x):
                    why = "px_screen 没有这只 / 当天没有行"
                else:
                    x = x.iloc[0]
                    why = ";".join(w for w, fl in (("静态ST", c in st), ("nbar<120", x["nbar"] < 120), ("close<2", x["close"] < 2), ("susp20", bool(x["susp20"])),
                                                   ("输入NaN", not np.isfinite(x[["lret20", "vol60", "turn20", "amp20"]].to_numpy(float)).all())) if fl) or "?"
                diff_rows.append({"date": date, "code": c, "cmp": "各自数据:生产有研究无", "why": why})
            for c in sorted(R0 - Pset):
                x = today[today["code"] == c]
                if not len(x):
                    why = "不在 train.parquet 当天行里（daily 有 %s）" % ("是" if c in set(rule["code"]) else "否")
                else:
                    x = x.iloc[0]
                    why = ";".join(w for w, fl in (("nbar<120", x["nbar"] < D.RULE_MIN_BARS), ("close<2", x["close"] < 2), ("susp20", bool(x["susp20"])),
                                                   ("ret20 NaN", not np.isfinite(x["ret20"]))) if fl) or "?"
                diff_rows.append({"date": date, "code": c, "cmp": "各自数据:研究有生产无", "why": why})
            # 前 10 差异行
            for c in sorted(P10_ev ^ R10):
                diff_rows.append({"date": date, "code": c, "cmp": "前10(剔事件后)" + ("生产有" if c in P10_ev else "研究有"),
                                  "why": "ev_any=%s pool=%s inP=%s" % (bool(ev.get(c, False)), c in Rp, c in Pset)})
        summ.append(rec)
        print("  %s 生产池 %d 研究(同数据) %d 差 %d/%d%s" % (date, rec["n_prod"], rec["n_res_daily_noST"], rec["prod_minus_resdaily"], rec["resdaily_minus_prod"],
                                                     ("  各自数据 elig0 %d pool %d 差 %d/%d jac前10 %.2f/%.2f" % (rec["n_res_px_elig0"], rec["n_res_pool"], rec["prod_minus_res"],
                                                                                                                 rec["res_minus_prod"], rec["jac_top10_pre"], rec["jac_top10_ev"])) if is_a else ""), flush=True)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "b_daily_summary.csv", index=False)
    Dd = pd.DataFrame(diff_rows)
    Dd.to_csv(OUT / "b_diff_rows.csv", index=False)
    say("\n== 汇总 ==")
    for seg in ("a", "b"):
        s = S[S["seg"] == seg]
        if not len(s):
            continue
        say("(%s) %d 天：生产池均 %.0f 只；研究代码路径在 daily 上（不剔静态 ST）均 %.0f 只；同数据差 生产有研究无 均 %.2f / 研究有生产无 均 %.2f；生产池里在静态 ST 名单的 均 %.1f 只" % (
            seg, len(s), s["n_prod"].mean(), s["n_res_daily_noST"].mean(), s["prod_minus_resdaily"].mean(), s["resdaily_minus_prod"].mean(), s["prod_in_staticST"].mean()))
    s = S[S["seg"] == "a"]
    if len(s):
        say("(a) 各自数据：研究 elig0 均 %.0f、pool 均 %.0f；生产有研究无 均 %.1f（去掉静态 ST 后 %.1f）、研究有生产无 均 %.1f；池 Jaccard（生产池去静态 ST 对 elig0）均 %.4f" % (
            s["n_res_px_elig0"].mean(), s["n_res_pool"].mean(), s["prod_minus_res"].mean(), s["prod_minus_res_exST"].mean(), s["res_minus_prod"].mean(), s["jac_pool"].mean()))
        say("(a) 前 10：剔事件前 Jaccard 均 %.3f（=1 的天 %d/%d）；剔事件后（生产池套研究事件标志 对 研究 pool）Jaccard 均 %.3f（=1 的天 %d/%d，共同只数均 %.2f）" % (
            s["jac_top10_pre"].mean(), int((s["jac_top10_pre"] >= 0.999).sum()), len(s), s["jac_top10_ev"].mean(), int((s["jac_top10_ev"] >= 0.999).sum()), len(s), s["n_top10_ev_common"].mean()))
        say("(a) ret20 数值（生产 qfq 取整 对 研究 pit）：绝对差中位（按天中位的中位）%.2e，最大 %.4f，>1e-3 的行合计 %d" % (
            s["ret20_absdiff_med"].median(), s["ret20_absdiff_max"].max(), int(s["ret20_absdiff_gt1e3"].sum())))
    if len(Dd):
        say("\n差异原因计数：")
        say(Dd.groupby(["cmp", "why"]).size().sort_values(ascending=False).head(40).to_string())
        say("\n差异最多的代码（各自数据）：")
        say(Dd[Dd["cmp"].str.startswith("各自数据")].groupby(["code", "cmp", "why"]).size().sort_values(ascending=False).head(25).to_string())
    say("用时 %.0fs" % (time.time() - t0))
    (OUT / "results_b_pool.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
