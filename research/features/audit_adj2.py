"""
审计追加：base_a158 前 10 里「T 日之后有除权」的行命中 20.65%、其余 14.94%（base 两组一样）。
是取整颗粒度泄漏，还是「先大涨、后高送转」的反向因果？分开看：
  - 除权幅度分档：3~10%（多为现金分红）、>=10%（送转为主）
  - 板块
  - 下一次除权发生在 t 之后多少个交易日（落在标签窗口 20 根之内 / 之外）
  - 前复权价位（颗粒度 = 0.01 / 价）：泄漏假说下应集中在低价、大幅度的行
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

L = []


def say(s):
    print(s, flush=True)
    L.append(s)


px = pd.read_parquet(C.ROOT / "data" / "breakout" / "daily.parquet", columns=["code", "date", "close", "volume", "amount"])
px["date"] = px["date"].astype(str).str[:10]
px["code"] = px["code"].astype(str).str.zfill(6)
px = px.sort_values(["code", "date"]).reset_index(drop=True)
with np.errstate(all="ignore"):
    r = px["amount"] / px["volume"] / px["close"]
g = r.groupby(px["code"]).transform(lambda x: x.rolling(5, min_periods=3).median())
px["lg"] = np.log(g)
# 下一次除权：lg 序列（按日期向后）出现 >= 2% 的台阶下降的第一天
px["step"] = px.groupby("code")["lg"].diff(-1)          # lg_t - lg_{t+1}，除权日前后 qfq 因子下台阶
px["i"] = px.groupby("code").cumcount()
ev = px[px["step"] >= 0.02][["code", "i"]].rename(columns={"i": "ie"})
px = px[px["date"] < "2026-01-01"]
x = px[["code", "date", "lg", "i", "close"]]
x = x[x["date"] >= "2024-06-01"]
# 最近一次未来除权的距离
ev = ev.sort_values(["code", "ie"])
x = x.sort_values("i")
m = pd.merge_asof(x.sort_values("i"), ev.sort_values("ie"), left_on="i", right_on="ie", by="code",
                  direction="forward")
m["dist"] = m["ie"] - m["i"]
tops = {"base_s7": C.top("altpath_v1", "base"), "a158_s7": C.top("altpath_v1", "base_a158")}
for s in (9, 10, 11):
    tops[f"a158_s{s}"] = C.top("feat_seeds", f"a158_s{s}")
    tops[f"base_s{s}"] = C.top("feat_seeds", f"base_s{s}")
for k, t in tops.items():
    t = t[t["rank"] <= 10].merge(m[["code", "date", "lg", "dist", "close"]], on=["code", "date"], how="left")
    grp = np.select([t["lg"] < 0.015, t["lg"] < 0.03, t["lg"] < 0.10, t["lg"] >= 0.10], ["干净", "1.5~3%", "3~10%", ">=10%"],
                    "缺")
    s = t.groupby(grp)["y_open"].agg(["size", "mean"])
    say("%-9s 按未来除权幅度：%s" % (k, "  ".join("%s n=%d %.1f%%" % (i, r_["size"], 100 * r_["mean"]) for i, r_ in s.iterrows())))
    adj = t["lg"] >= 0.03
    d = np.select([t["dist"] <= 20, t["dist"] <= 120, t["dist"] > 120], ["<=20 根", "21~120", ">120"], "无")
    s2 = t[adj].groupby(d[adj.to_numpy()])["y_open"].agg(["size", "mean"])
    say("          有除权的行按距下次除权：%s" % "  ".join("%s n=%d %.1f%%" % (i, r_["size"], 100 * r_["mean"]) for i, r_ in s2.iterrows()))
    s3 = t.groupby([t["board"], adj])["y_open"].agg(["size", "mean"])
    say("          板块 x 有无除权：%s" % "  ".join("%s/%s n=%d %.1f%%" % (i[0], "除" if i[1] else "净", r_["size"], 100 * r_["mean"])
                                              for i, r_ in s3.iterrows()))
    lowp = t["close"] < 8
    s4 = t[adj].groupby(lowp[adj])["y_open"].agg(["size", "mean"])
    say("          有除权的行按前复权价 <8 元：%s" % "  ".join("%s n=%d %.1f%%" % ("低价" if i else "高价", r_["size"], 100 * r_["mean"]) for i, r_ in s4.iterrows()))
(C.HERE / "results_audit_adj2.txt").write_text("\n".join(L), encoding="utf-8")
