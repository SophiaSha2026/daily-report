"""a11：腾讯快照追加的那根 K 线（raw/sina_9_*_upd*.parquet 全部行）vs 现在 daily.parquet 里同一 (code,date)。
    daily 里已被新浪整段重拉覆盖的行 = 权威值，比出快照的系统性偏差；仍等于快照的行 = 没人核对过的那部分（量多少）。
    再和 raw/sina_qfq_now.parquet（样本票现在重拉的新浪）比一遍。"""
import glob
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


shards = []
for f in sorted(glob.glob(str(ROOT / "data/breakout/raw/sina_9_*_upd*.parquet"))):
    x = pd.read_parquet(f)
    x["shard"] = Path(f).name
    shards.append(x)
s = pd.concat(shards, ignore_index=True)
s["code"] = s["code"].astype(str).str.zfill(6)
s = s.drop_duplicates(["code", "date"], keep="last")
say("快照追加行（upd/upd2 分片去重后）%d 行，%d 个交易日 %s..%s，%d 只" % (len(s), s["date"].nunique(), s["date"].min(), s["date"].max(), s["code"].nunique()))
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
d["code"] = d["code"].astype(str).str.zfill(6)
d = d[d["date"] >= s["date"].min()]
m = s.merge(d, on=["code", "date"], how="left", suffixes=("_s", "_d"), indicator=True)
say("在 daily.parquet 里找得到的 %d，找不到的 %d（被整段重拉清掉又没补回？）" % ((m["_merge"] == "both").sum(), (m["_merge"] == "left_only").sum()))
lost = m[m["_merge"] == "left_only"]
if len(lost):
    say("  找不到的按日期 %s，例 %s" % (lost["date"].value_counts().head(8).to_dict(), lost[["code", "date", "shard"]].head(6).to_dict("records")))
    # 这些票当天在 daily 里真的没有行？（可能停牌、可能被清）
m = m[m["_merge"] == "both"].copy()
cols = ["open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]
same = np.ones(len(m), bool)
for c in cols:
    same &= np.isclose(m[c + "_s"], m[c + "_d"], rtol=1e-9, atol=1e-9)
m["same"] = same
say("和 daily 逐位相同（仍是快照原样，没被新浪覆盖）%d 行（%.1f%%）；被覆盖过的 %d 行" % (same.sum(), 100 * same.mean(), (~same).sum()))
ov = m[~same]
say("被新浪覆盖的行里，快照 vs 新浪：")
for c in cols:
    r = ov[c + "_s"] / ov[c + "_d"].replace(0, np.nan) - 1
    say("  %-17s |r|>1e-4 的行 %5d / %d；分位 p1 %.4f p50 %.4f p99 %.4f；|r|>1%% %d，>10%% %d" % (
        c, (r.abs() > 1e-4).sum(), len(ov), r.quantile(.01), r.quantile(.5), r.quantile(.99), (r.abs() > 0.01).sum(), (r.abs() > 0.10).sum()))
# 复权的影响：价格差应是常数比例（除权票）而不是个别错价
rc = ov["close_s"] / ov["close_d"] - 1
big = ov[rc.abs() > 0.002]
say("  close 差 >0.2%% 的 %d 行：按日期 %s" % (len(big), big["date"].value_counts().head(6).to_dict()))
say("  例", big[["code", "date", "close_s", "close_d", "volume_s", "volume_d", "shard"]].head(8).to_string())
# 量：单位错（100 倍）的有没有
rv = ov["volume_s"] / ov["volume_d"]
say("  volume 比值在 [90,110] 或 [0.009,0.011]（单位错 100 倍）的行 %d；比值 != 1 但 |r|<1%% 的 %d" % (
    (((rv > 90) & (rv < 110)) | ((rv > 0.009) & (rv < 0.011))).sum(), ((rv != 1) & ((rv - 1).abs() < 0.01)).sum()))
# 换手率：快照两位小数百分比 vs 新浪 volume/os
rt = ov["turnover_s"] / ov["turnover_d"] - 1
say("  turnover 相对差分位 %s" % rt.quantile([.001, .01, .5, .99, .999]).round(4).to_dict())
ro = ov["outstanding_share_s"] / ov["outstanding_share_d"] - 1
say("  outstanding_share 相对差 >1%% 的行 %d，>5%% %d；例 %s" % ((ro.abs() > 0.01).sum(), (ro.abs() > 0.05).sum(),
                                                        ov.loc[ro.abs() > 0.05, ["code", "date", "outstanding_share_s", "outstanding_share_d"]].head(5).to_dict("records")))
# 样本票：现在重拉的新浪 vs daily 里仍是快照原样的行
p = OUT / "raw/sina_qfq_now.parquet"
if p.exists():
    n = pd.read_parquet(p)
    n["code"] = n["code"].astype(str).str.zfill(6)
    n = n.rename(columns={c: c + "_n" for c in cols})
    keep = m[m["same"]].merge(n, on=["code", "date"])
    say("样本票里仍是快照原样的行 %d，和现在的新浪比：" % len(keep))
    for c in cols:
        if c + "_n" not in keep:
            continue
        r = keep[c + "_s"] / keep[c + "_n"].replace(0, np.nan) - 1
        say("  %-17s |r|>1e-4 %5d / %d；p1 %.4f p50 %.4f p99 %.4f；|r|>1%% %d" % (c, (r.abs() > 1e-4).sum(), len(keep), r.quantile(.01), r.quantile(.5), r.quantile(.99), (r.abs() > 0.01).sum()))
    rc = keep["close_s"] / keep["close_n"] - 1
    say("  close 差 >0.2%% 的例子", keep.loc[rc.abs() > 0.002, ["code", "date", "close_s", "close_n", "open_s", "open_n"]].head(8).to_dict("records"))
    # 同一只票：快照行之前的历史（新浪老基准）和现在新浪的比：基准是否一致 -> 断崖
    dd = d.merge(n, on=["code", "date"], suffixes=("_d", "_n"))
    dd["k"] = dd["close"] / dd["close_n"]
    kk = dd.groupby("code")["k"].agg(["min", "max"])
    bad = kk[(kk["max"] - kk["min"]) > 0.004]
    say("样本票 2026-09 以来 daily vs 现在新浪：close 比值不恒定（有断崖 / 基准不同）的只数 %d / %d；例 %s" % (len(bad), len(kk), bad.head(6).round(4).to_dict("index")))
    for c in bad.index[:4]:
        x = dd[dd["code"] == c].sort_values("date")
        jump = x.loc[(x["k"].round(3).diff().abs() > 0.002), ["date", "close", "close_n", "k"]]
        say("   ", c, "跳变处", jump.head(4).to_dict("records"))
(OUT / "results_a11.txt").write_text("\n".join(L), encoding="utf-8")
