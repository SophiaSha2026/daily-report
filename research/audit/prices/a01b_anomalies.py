"""a01b：a01 里冒出来的几处异常追查：amount/(close*volume) 偏离、每只起始日分布、|涨跌|>45% 行、vwap 出 [low,high]。只读。"""
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
d = d.sort_values(["code", "date"]).reset_index(drop=True)
vwap = d["amount"] / d["volume"]
d["vwap_rel"] = np.where(vwap > d["high"], vwap / d["high"] - 1, np.where(vwap < d["low"], vwap / d["low"] - 1, 0.0))
out = d[np.abs(d["vwap_rel"]) > 0.02]
say("vwap 出 [low,high] 超 2%% 的行 %d（%.3f%%），只 %d" % (len(out), 100 * len(out) / len(d), out["code"].nunique()))
say("  按板块", out["code"].str[:2].value_counts().to_dict())
say("  vwap/边界 的分布", out["vwap_rel"].describe().round(3).to_dict())
say("  按年月 top", out["date"].str[:7].value_counts().head(10).to_dict())
say("  按只 top", out["code"].value_counts().head(10).to_dict())
# 这些行是不是集中在复权段（q != 1 的历史段）？用 px_screen 的 q 看
p = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/px_screen.parquet", columns=["code", "date", "q", "amount", "volume", "close_u", "close", "high", "low"])
m = out.merge(p, on=["code", "date"], how="left", suffixes=("", "_scr"))
say("  能并上 px_screen 的 %d 行；其中 q>1（历史被复权过）的 %d 行；px_screen 自己 amount/volume 出界的 %d 行" % (
    m["q"].notna().sum(), int((m["q"] > 1 + 1e-9).sum()),
    int((((m["amount_scr"] / m["volume_scr"]) > m["high_scr"] * 1.02) | ((m["amount_scr"] / m["volume_scr"]) < m["low_scr"] * 0.98)).sum())))
say("  例：", out[["code", "date", "open", "high", "low", "close", "volume", "amount", "vwap_rel"]].head(12).to_string())
# 不复权 vwap 核：amount / volume 应该落在不复权 [low_u, high_u]
pu = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "high_u", "low_u", "amount", "volume"])
pu["code"] = pu["code"].astype(str).str.zfill(6)
v = pu["amount"] / pu["volume"].replace(0, np.nan)
bad_u = pu[(v > pu["high_u"] * 1.02) | (v < pu["low_u"] * 0.98)]
say("unadj_2020：amount/volume 出不复权 [low,high] 超 2%% 的行 %d / %d，只 %d；按年 %s" % (
    len(bad_u), len(pu), bad_u["code"].nunique(), bad_u["date"].str[:4].value_counts().sort_index().to_dict()))
say("  例", bad_u.head(8).to_string())
# 所以生产表里出界的行，是「复权了价格没复权成交量 / 成交额」吗？前复权后 amount 应保持不变、volume 不变、价格缩小 => vwap/价格 = q
mm = m[m["q"].notna()]
say("  生产表出界行里 vwap/close_prod 与 q 的比：", (mm["amount"] / mm["volume"] / mm["close"] / mm["q"]).describe().round(3).to_dict())

say("== 每只起始日分布 ==")
per = d.groupby("code")["date"].agg(["min", "max", "size"])
say(per["min"].value_counts().sort_index().head(3).to_dict(), "...", per["min"].value_counts().head(6).to_dict())
say("起始 2023-01-03 的只数", int((per["min"] == "2023-01-03").sum()), "起始 2023-05-30 的", int((per["min"] == "2023-05-30").sum()),
    "2023-01-04..2023-05-29 之间起始的", int(((per["min"] > "2023-01-03") & (per["min"] < "2023-05-30")).sum()))
# 起始 2023-01-03 的票是不是 2026-09 以来被整段重拉过的票（除权 / 停牌 / 股本漂）？看 raw 分片
import glob
ref_codes = set()
for f in sorted(glob.glob(str(ROOT / "data/breakout/raw/sina_9_*_ref.parquet"))):
    ref_codes |= set(pd.read_parquet(f, columns=["code"])["code"].astype(str).str.zfill(6).unique())
early = set(per.index[per["min"] == "2023-01-03"])
say("2026-09 以来 _ref 分片重拉过的只数", len(ref_codes), "其中起始 2023-01-03 的", len(early & ref_codes),
    "；起始 2023-01-03 的票里没被重拉过的", len(early - ref_codes))
# 原始 sina_00xx 分片的起始日
s0 = pd.read_parquet(ROOT / "data/breakout/raw/sina_0000.parquet", columns=["code", "date"])
say("sina_0000 分片起始日", s0["date"].min(), "只", s0["code"].nunique(), "逐只起始 value_counts", s0.groupby("code")["date"].min().value_counts().head(3).to_dict())

say("== |涨跌|>45% 行 ==")
pc = d.groupby("code")["close"].shift(1)
ret = d["close"] / pc - 1
big = d[ret.abs() > 0.45].copy()
big["ret"] = ret[ret.abs() > 0.45]
big["prev_date"] = d.groupby("code")["date"].shift(1)[ret.abs() > 0.45]
big["nbar_before"] = d.groupby("code").cumcount()[ret.abs() > 0.45]
say(big[["code", "date", "prev_date", "close", "ret", "nbar_before"]].to_string())
x = d[(d["code"] == "301139") & (d["date"] >= "2026-09-25")]
say("301139 近期：\n" + x[["date", "open", "high", "low", "close", "volume", "amount", "outstanding_share", "turnover"]].to_string())
xu = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", filters=[("code", "==", "301139")])
say("301139 unadj 近期：\n" + xu[xu["date"] >= "2026-09-25"].to_string())
(OUT / "results_a01b.txt").write_text("\n".join(L), encoding="utf-8")
