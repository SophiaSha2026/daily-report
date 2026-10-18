"""a10：第二源逐只对账（样本 285 只，北交所只有新浪 / 通达信事件）。
  A 不复权：新浪 unadj_2020 vs 腾讯日 K raw（OHLC、量、日期集合：停牌日多 / 少一行）
  B 前复权：生产 daily.parquet（新浪 qfq）vs 腾讯 qfq（两家各自算的复权，比相对差；腾讯三位小数）
  C 生产 daily.parquet vs 现在重拉的新浪 qfq（快照行 + 历史：价、量、额、换手、股本）
  D 4 个 xdxr/新浪 台阶不一致的事件，用腾讯 raw/qfq 比值的台阶当裁判
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
sys.path.insert(0, str(ROOT / "src"))
import datasource as ds  # noqa: E402

L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


s = pd.read_csv(RAW / "sample.csv", dtype={"code": str})
s["code"] = s["code"].str.zfill(6)
codes = s["code"].tolist()
tx = pd.read_parquet(RAW / "tx_kline.parquet")
tx["code"] = tx["code"].astype(str).str.zfill(6)
txr = tx[tx["adj"] == "raw"].drop(columns=["adj", "extra"])
txq = tx[tx["adj"] == "qfq"].drop(columns=["adj", "extra"])
say("腾讯 raw %d 只 %d 行，qfq %d 只 %d 行；raw 起始日中位 %s" % (txr["code"].nunique(), len(txr), txq["code"].nunique(), len(txq), sorted(txr.groupby("code")["date"].min())[len(txr["code"].unique()) // 2]))
# 腾讯成交量单位：手（688 段是股？）用新浪核
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", filters=[("code", "in", codes)])
u["code"] = u["code"].astype(str).str.zfill(6)
u = u.rename(columns={"open_u": "open", "high_u": "high", "low_u": "low", "close_u": "close"})
u_live = u[u["volume"] > 0]

say("== A 不复权：新浪 vs 腾讯 raw ==")
lo = txr["date"].min()
a = u.merge(txr, on=["code", "date"], how="outer", suffixes=("_s", "_t"), indicator=True)
# 只比腾讯有覆盖的区间（各票 800 根起点之后）
t0 = txr.groupby("code")["date"].min()
a = a[a["date"] >= a["code"].map(t0)]
a = a[a["code"].isin(set(txr["code"]))]
only_s = a[a["_merge"] == "left_only"]
only_t = a[a["_merge"] == "right_only"]
both = a[a["_merge"] == "both"].copy()
say("腾讯覆盖区间内：两边都有 %d 行；只新浪有 %d（其中 volume=0 的停牌占位 %d）；只腾讯有 %d" % (
    len(both), len(only_s), (only_s["volume_s"] == 0).sum(), len(only_t)))
if len(only_s[only_s["volume_s"] > 0]):
    say("  只新浪有且有成交量的：%s" % only_s[only_s["volume_s"] > 0][["code", "date", "close_s", "volume_s"]].head(8).to_dict("records"))
if len(only_t):
    say("  只腾讯有的：%s" % only_t[["code", "date", "close_t", "volume_t"]].head(8).to_dict("records"))
    say("  只腾讯有的按只 %s" % only_t["code"].value_counts().head(6).to_dict())
for c in ("open", "high", "low", "close"):
    dd = (both[c + "_s"] - both[c + "_t"]).abs()
    say("  %-5s 逐位相同 %.4f%%；差 1 分 %d 行；差 >1 分 %d 行" % (c, 100 * (dd < 1e-9).mean(), ((dd - 0.01).abs() < 1e-9).sum(), (dd > 0.015).sum()))
bad = both[(both["close_s"] - both["close_t"]).abs() > 0.015]
say("  close 差 >1 分的例子：%s" % bad[["code", "date", "close_s", "close_t", "volume_s", "volume_t"]].head(8).to_dict("records"))
say("  按只 %s" % bad["code"].value_counts().head(8).to_dict())
rv = both["volume_s"] / both["volume_t"].replace(0, np.nan)
both["vr"] = rv
say("  volume 新浪/腾讯 比值分位 %s" % rv.quantile([0, .01, .5, .99, 1]).round(3).to_dict())
say("  按板块中位：%s" % both.groupby(both["code"].map(ds.board_of))["vr"].median().round(3).to_dict())
unit = np.where(both["code"].str.startswith("688"), 1.0, 100.0)
vdiff = (both["volume_s"] - both["volume_t"] * unit).abs() / both["volume_s"].replace(0, np.nan)
say("  按单位折算后 volume 相对差 <1e-6 %.3f%%，<1%% %.3f%%，>1%% %d 行；例 %s" % (
    100 * (vdiff < 1e-6).mean(), 100 * (vdiff < 0.01).mean(), (vdiff > 0.01).sum(),
    both.loc[vdiff > 0.01, ["code", "date", "volume_s", "volume_t"]].head(5).to_dict("records")))

say("== B 前复权：daily.parquet（新浪 qfq）vs 腾讯 qfq ==")
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", filters=[("code", "in", codes)])
d["code"] = d["code"].astype(str).str.zfill(6)
b = d.merge(txq, on=["code", "date"], suffixes=("_d", "_t"))
say("并上 %d 行 %d 只" % (len(b), b["code"].nunique()))
r = b["close_d"] / b["close_t"] - 1
say("  close 相对差分位 %s；|r|>0.5%% 的行 %d，只 %d" % (r.quantile([0, .01, .5, .99, 1]).round(5).to_dict(), (r.abs() > 0.005).sum(), b.loc[r.abs() > 0.005, "code"].nunique()))
kk = b.assign(r=r).groupby("code")["r"].agg(["min", "max", "median"])
say("  每只相对差整段恒定（max-min<0.4%%）的 %d 只；不恒定的 %d 只：%s" % (((kk["max"] - kk["min"]) < 0.004).sum(), ((kk["max"] - kk["min"]) >= 0.004).sum(),
                                                            kk[(kk["max"] - kk["min"]) >= 0.004].round(4).head(10).to_dict("index")))
for c in kk.index[(kk["max"] - kk["min"]) >= 0.004][:6]:
    xx = b[b["code"] == c].sort_values("date")
    rr = (xx["close_d"] / xx["close_t"]).round(4)
    jumps = xx.loc[rr.diff().abs() > 0.002, ["date", "close_d", "close_t"]]
    say("   ", c, "比值跳变日", jumps.head(5).to_dict("records"))
# 腾讯 qfq 的最后一行是否等于 raw（基准都是最新）
last = b.sort_values("date").groupby("code").tail(1)
say("  最后一行 daily 与腾讯 qfq 相差 >0.5%% 的只数 %d" % ((last["close_d"] / last["close_t"] - 1).abs() > 0.005).sum())

say("== C daily.parquet vs 现在重拉的新浪 qfq ==")
p = RAW / "sina_qfq_now.parquet"
if p.exists():
    n = pd.read_parquet(p)
    n["code"] = n["code"].astype(str).str.zfill(6)
    n = n[n["volume"] > 0]
    c_ = d.merge(n, on=["code", "date"], how="outer", suffixes=("_d", "_n"), indicator=True)
    c_ = c_[c_["date"] >= "2023-01-01"]
    say("并集 %d 行：两边都有 %d，只 daily %d，只新浪现在 %d" % (len(c_), (c_["_merge"] == "both").sum(), (c_["_merge"] == "left_only").sum(), (c_["_merge"] == "right_only").sum()))
    onl = c_[c_["_merge"] == "left_only"]
    onr = c_[c_["_merge"] == "right_only"]
    say("  只 daily 有：按年月 %s，例 %s" % (onl["date"].str[:7].value_counts().head(4).to_dict(), onl[["code", "date", "close_d", "volume_d"]].head(4).to_dict("records")))
    say("  只新浪现在有：按年月 %s，例 %s" % (onr["date"].str[:7].value_counts().head(4).to_dict(), onr[["code", "date", "close_n", "volume_n"]].head(4).to_dict("records")))
    cb = c_[c_["_merge"] == "both"].copy()
    for col in ("open", "high", "low", "close", "volume", "amount", "turnover", "outstanding_share"):
        r = cb[col + "_d"] / cb[col + "_n"].replace(0, np.nan) - 1
        say("  %-17s |r|>1e-6 %6d / %d（%.3f%%）；>1%% %d；分位 p0.1 %.4f p99.9 %.4f" % (
            col, (r.abs() > 1e-6).sum(), len(cb), 100 * (r.abs() > 1e-6).mean(), (r.abs() > 0.01).sum(), r.quantile(.001), r.quantile(.999)))
    r = cb["close_d"] / cb["close_n"] - 1
    kk = cb.assign(r=r).groupby("code")["r"].agg(["min", "max"])
    nonc = kk[(kk["max"] - kk["min"]) > 0.004]
    say("  close 比值不恒定（有断崖 / 局部基准不同）的只数 %d：%s" % (len(nonc), nonc.round(4).head(8).to_dict("index")))
    for c in nonc.index[:5]:
        xx = cb[cb["code"] == c].sort_values("date")
        rr = (xx["close_d"] / xx["close_n"]).round(4)
        say("   ", c, "跳变日", xx.loc[rr.diff().abs() > 0.002, ["date", "close_d", "close_n"]].head(4).to_dict("records"))
    const = kk[((kk["max"] - kk["min"]) <= 0.004) & (kk["min"].abs() > 0.002)]
    say("  整段常数比例 != 1（基准不同但无断崖）的只数 %d：%s" % (len(const), const.round(4).head(6).to_dict("index")))
    # 快照行（2026-09-14 起）专门看
    sn = cb[cb["date"] >= "2026-09-14"]
    for col in ("close", "volume", "amount", "turnover", "outstanding_share"):
        r = sn[col + "_d"] / sn[col + "_n"].replace(0, np.nan) - 1
        say("  [2026-09-14 起 %d 行] %-17s |r|>1e-4 %d；>1%% %d；>10%% %d" % (len(sn), col, (r.abs() > 1e-4).sum(), (r.abs() > 0.01).sum(), (r.abs() > 0.10).sum()))
    rt = sn["turnover_d"] / sn["turnover_n"] - 1
    say("    turnover 差 >1%% 的例子 %s" % sn.loc[rt.abs() > 0.01, ["code", "date", "turnover_d", "turnover_n", "outstanding_share_d", "outstanding_share_n"]].head(6).to_dict("records"))
else:
    say("sina_qfq_now.parquet 还没有")

say("== D 台阶裁判（腾讯 raw/qfq 比值）==")
for c, dt in (("000615", "2025-12-29"), ("688503", "2024-06-11"), ("300209", "2024-12-20"), ("002822", "2025-12-30"), ("301237", "2026-09-01")):
    rr = txr[txr["code"] == c].set_index("date")["close"]
    qq = txq[txq["code"] == c].set_index("date")["close"]
    idx = sorted(set(rr.index) & set(qq.index))
    k = (rr.loc[idx] / qq.loc[idx])
    i = idx.index(dt) if dt in idx else None
    if i is None or i == 0:
        say("  %s %s 腾讯没有这天" % (c, dt))
        continue
    say("  %s %s：腾讯 raw/qfq 台阶 = %.6f（前一日比 %.4f，当日比 %.4f）；raw 昨收 %.2f 今开 %.2f 今收 %.2f" % (
        c, dt, k.iloc[i - 1] / k.iloc[i], k.iloc[i - 1], k.iloc[i], rr.loc[idx[i - 1]], txr[(txr["code"] == c) & (txr["date"] == dt)]["open"].iloc[0], rr.loc[dt]))
(OUT / "results_a10.txt").write_text("\n".join(L), encoding="utf-8")
