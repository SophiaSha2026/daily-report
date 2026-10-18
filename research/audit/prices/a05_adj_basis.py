"""a05：复权基准对比。生产 daily.parquet（新浪 qfq，按各票最后一次整段重拉时的基准，两位小数）
vs 研究 px_screen（不复权 × 归一因子 q，不取整，基准 = 2026-10-15 取因子时的最后一行）。重叠期 2023-01..2025-12 逐行比。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


d = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
d["code"] = d["code"].astype(str).str.zfill(6)
d = d[d["date"] <= "2025-12-31"]
p = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/px_screen.parquet")
p["code"] = p["code"].astype(str).str.zfill(6)
p = p[p["date"] >= "2023-01-01"]
m = d.merge(p, on=["code", "date"], how="outer", suffixes=("_d", "_p"), indicator=True)
say("重叠期 2023-01..2025-12：生产 %d 行、研究 %d 行；只在生产 %d、只在研究 %d、两边都有 %d" % (
    len(d), len(p), (m["_merge"] == "left_only").sum(), (m["_merge"] == "right_only").sum(), (m["_merge"] == "both").sum()))
lo = m[m["_merge"] == "left_only"]
ro = m[m["_merge"] == "right_only"]
say("  只在生产的：只 %d，按年月 %s，例 %s" % (lo["code"].nunique(), lo["date"].str[:7].value_counts().head(5).to_dict(),
                                       lo[["code", "date"]].head(5).to_dict("records")))
say("  只在研究的：只 %d，按年月 %s，例 %s" % (ro["code"].nunique(), ro["date"].str[:7].value_counts().head(5).to_dict(),
                                       ro[["code", "date"]].head(5).to_dict("records")))
# 只在生产的行：是停牌占位行（研究表 clean_bars 清掉了）还是研究表没这只票？
ro_codes = set(ro["code"]) - set(d["code"])
say("  研究表有、生产表完全没有的只数", len(ro_codes), "；生产表有、研究表完全没有的只数", len(set(lo["code"]) - set(p["code"])))
b = m[m["_merge"] == "both"].copy()
for c in ("open", "high", "low", "close"):
    b["r_" + c] = b[c + "_d"] / b[c + "_p"] - 1
    b["a_" + c] = (b[c + "_d"] - b[c + "_p"]).abs()
# 取整颗粒：研究表不取整，生产两位小数 -> |差| <= 0.005 + 浮点
tol = 0.005 + 1e-9
ok = (b[["a_open", "a_high", "a_low", "a_close"]] <= tol).all(axis=1)
say("四个价全部在取整颗粒内（|差|<=0.005）的行 %.5f%%；超出的行 %d，只 %d" % (100 * ok.mean(), (~ok).sum(), b.loc[~ok, "code"].nunique()))
bad = b[~ok].copy()
bad["rel_close"] = bad["r_close"]
say("  超出行的 close 相对差分位", bad["rel_close"].quantile([0, .05, .5, .95, 1]).round(5).to_dict())
say("  按只 top 15", bad["code"].value_counts().head(15).to_dict())
say("  按年月", bad["date"].str[:7].value_counts().sort_index().to_dict())
# 每只的相对差是不是一个常数（基准不同 = 整段差一个比例）还是分段
g = bad.groupby("code")["rel_close"].agg(["size", "min", "max", "median"])
g["const"] = (g["max"] - g["min"]).abs() < 2e-3
say("  超出的只里，相对差整段近似常数（基准不同）的 %d 只、分段的 %d 只" % (g["const"].sum(), (~g["const"]).sum()))
say(g.sort_values("size", ascending=False).head(12).round(5).to_string())
# 分段的：找出生产表和研究表的分界日（相对差跳变处）
seg = []
for c in g.index[~g["const"]][:30]:
    x = b[b["code"] == c].sort_values("date")
    r = x["r_close"].round(4)
    jumps = x.loc[r.diff().abs() > 1e-3, "date"].tolist()
    seg.append((c, len(x), jumps[:6], float(r.iloc[0]), float(r.iloc[-1])))
say("  分段票的跳变日（前 30 只）：")
for s_ in seg:
    say("   ", s_)
bad[["code", "date", "close_d", "close_p", "q", "rel_close"]].to_parquet(OUT / "raw/a05_close_mismatch.parquet", index=False)
# 量 / 额 / 换手 / 股本（同源不同时间拉的，看新浪自己改没改）
for c in ("volume", "amount", "turnover", "outstanding_share"):
    r = (b[c + "_d"] / b[c + "_p"].replace(0, np.nan)) - 1
    say("%s 相对差 |r|>1e-6 的行 %d（%.4f%%），>1%% 的行 %d，只 %d；分位 %s" % (
        c, (r.abs() > 1e-6).sum(), 100 * (r.abs() > 1e-6).mean(), (r.abs() > 0.01).sum(), b.loc[r.abs() > 0.01, "code"].nunique(),
        r.quantile([0, .001, .5, .999, 1]).round(4).to_dict()))
    if c == "turnover":
        bt = b[r.abs() > 0.01]
        say("  turnover 差 >1%% 的行按年月 %s" % bt["date"].str[:7].value_counts().head(6).to_dict())
        say("  例", bt[["code", "date", "turnover_d", "turnover_p", "volume_d", "outstanding_share_d", "outstanding_share_p"]].head(6).to_string())
    if c == "outstanding_share":
        bo = b[r.abs() > 0.01]
        say("  股本差 >1%% 的行按年月 %s；按只 top %s" % (bo["date"].str[:7].value_counts().head(6).to_dict(), bo["code"].value_counts().head(6).to_dict()))
# 复权本身：q 的台阶（除权日）在生产表里对应的 close 比是否连续
say("== 生产表 vs 不复权：close_d / close_u 应等于 1/q（按各票常数倍）==")
b["k"] = b["close_d"] * b["q"] / b["close_u"]          # = 生产基准 / 研究基准
kk = b.groupby("code")["k"].agg(["min", "max", "median"])
say("  k 全程在 [0.995,1.005] 内的只数 %d / %d；k 中位数 != 1（±0.5%%）的只数 %d" % (
    ((kk["min"] > 0.995) & (kk["max"] < 1.005)).sum(), len(kk), ((kk["median"] - 1).abs() > 0.005).sum()))
say("  k 中位 != 1 的例子", kk[(kk["median"] - 1).abs() > 0.005].head(10).round(4).to_string())
(OUT / "results_a05.txt").write_text("\n".join(L), encoding="utf-8")
