"""a06b：(1) 主板 ST 涨跌幅是 5% 还是 10%：按月看触板分布、逐只看是不是前后换挡；腾讯快照 f[47]/f[48] 当前涨跌停价直接验（几只，一次请求）。
(2) 涨停判定混淆矩阵：生产 qfq 表「容 ±1 分」 vs 不复权表「精确」，同一 (code,date) 上比。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import datasource as ds  # noqa: E402
import bjtime  # noqa: E402

L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


st = set(json.load(open(ROOT / "cache/st_codes.json", encoding="utf-8"))["codes"])
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "open_u", "high_u", "low_u", "close_u", "volume"])
u["code"] = u["code"].astype(str).str.zfill(6)
u = u[(u["volume"] > 0) & (u["close_u"] > 0)].sort_values(["code", "date"])
u["pc"] = u.groupby("code")["close_u"].shift(1)
u["ret"] = u["close_u"] / u["pc"] - 1
main_st = u[u["code"].isin(st) & (u["code"].map(ds.board_of) == "main") & (u["date"] >= "2025-01-01")].copy()
main_st["m"] = main_st["date"].str[:7]
main_st["hit5"] = (main_st["ret"].abs() - 0.05).abs() < 0.0015
main_st["hit10"] = (main_st["ret"].abs() - 0.10).abs() < 0.0015
main_st["gt52"] = main_st["ret"].abs() > 0.052
g = main_st.groupby("m").agg(n=("ret", "size"), hit5=("hit5", "sum"), hit10=("hit10", "sum"), gt52=("gt52", "sum"), codes=("code", "nunique"))
say("主板当前 ST 名单 %d 只，2025-01 起按月：" % main_st["code"].nunique())
say(g.to_string())
# 逐只：最近 60 个交易日里 |ret|>5.2% 的天数 和 触 5% 的天数
last = main_st.groupby("code").tail(60)
per = last.groupby("code").agg(gt52=("gt52", "sum"), hit5=("hit5", "sum"), hit10=("hit10", "sum"), px=("close_u", "last"))
say("逐只（最近 60 根）：|ret|>5.2%% 天数为 0 的 %d 只，>=1 的 %d 只；触 10%% 板 >=1 次的 %d 只；触 5%% 板 >=1 次的 %d 只" % (
    (per["gt52"] == 0).sum(), (per["gt52"] > 0).sum(), (per["hit10"] > 0).sum(), (per["hit5"] > 0).sum()))
say("  既触过 5%% 又触过 10%% 的 %d 只，例 %s" % (((per["hit5"] > 0) & (per["hit10"] > 0)).sum(), per[(per["hit5"] > 0) & (per["hit10"] > 0)].head(8).to_dict("index")))
# 同一只票里 5% 触板和 10% 触板的日期先后：换挡日
sw = []
for c, x in main_st.groupby("code"):
    d5 = x.loc[x["hit5"], "date"]
    d10 = x.loc[x["hit10"], "date"]
    if len(d5) and len(d10):
        sw.append((c, d5.min(), d5.max(), d10.min(), d10.max()))
sw = pd.DataFrame(sw, columns=["code", "first5", "last5", "first10", "last10"])
say("5%% 和 10%% 都触过的 %d 只：last5 < first10（先 5%% 后 10%%）%d 只；first5 > last10（先 10%% 后 5%%，像 2026 年戴帽）%d 只；交错 %d 只" % (
    len(sw), (sw["last5"] < sw["first10"]).sum(), (sw["first5"] > sw["last10"]).sum(),
    len(sw) - (sw["last5"] < sw["first10"]).sum() - (sw["first5"] > sw["last10"]).sum()))
say(sw.head(12).to_string())
# 腾讯快照：当前涨停价 / 昨收（f[47]/f[4]），验几只主板 ST + 对照
t = bjtime.now_bj()
if not (16 * 60 <= t.hour * 60 + t.minute < 18 * 60 + 30):
    try:
        pick = sorted(per.index)[:12] + ["600000", "000001", "600182"]
        q = ds.fetch_quotes([ds.to_symbol(c) for c in pick])
        for c in pick:
            v = q.get(ds.to_symbol(c))
            if v is None:
                say("  快照", c, "没回")
                continue
            raw = v.raw
            try:
                say("  快照 %s %s 昨收 %s 涨停 %s 跌停 %s -> 幅度 %.4f" % (c, raw[1], raw[4], raw[47], raw[48], float(raw[47]) / float(raw[4]) - 1))
            except Exception as e:
                say("  快照", c, "字段解析失败", repr(e)[:80], raw[:6])
    except Exception as e:
        say("快照失败", repr(e)[:200])

say("== 涨停判定混淆矩阵：qfq 容 ±1 分 vs 不复权精确 ==")
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "high", "low", "close"])
d["code"] = d["code"].astype(str).str.zfill(6)
d = d.sort_values(["code", "date"])
d["pc"] = d.groupby("code")["close"].shift(1)
pct = d["code"].map(lambda c: ds.limit_pct(c, ""))
d["lim"] = ds.limit_price_arr(d["pc"], pct, d["code"])
d["zt_exact"] = (d["close"] == d["high"]) & ((d["high"] - d["lim"]).abs() < 1e-9)
d["zt_tol"] = (d["close"] == d["high"]) & ((d["high"] - d["lim"]).abs() <= 0.01 + 1e-9)
d["yz_exact"] = d["low"] >= d["lim"] - 1e-9
d["yz_tol"] = d["low"] >= d["lim"] - 0.01 - 1e-9
pct_u = u["code"].map(lambda c: ds.limit_pct(c, ""))
u["lim"] = ds.limit_price_arr(u["pc"], pct_u, u["code"])
u["zt_u"] = (u["close_u"] == u["high_u"]) & ((u["high_u"] - u["lim"]).abs() < 1e-9)
u["yz_u"] = u["low_u"] >= u["lim"] - 1e-9
m = d.merge(u[["code", "date", "zt_u", "yz_u", "pc"]], on=["code", "date"], suffixes=("", "_u"))
m = m[m["pc"].notna() & m["pc_u"].notna()]
say("同时有昨收的行 %d" % len(m))
for a, b, tag in (("zt_exact", "zt_u", "涨停 精确(qfq) vs 真值(不复权)"), ("zt_tol", "zt_u", "涨停 容±1分(qfq) vs 真值"),
                  ("yz_exact", "yz_u", "一字 精确(qfq) vs 真值"), ("yz_tol", "yz_u", "一字 容1分(qfq) vs 真值")):
    tp = (m[a] & m[b]).sum()
    fp = (m[a] & ~m[b]).sum()
    fn = (~m[a] & m[b]).sum()
    say("  %s：真值 %d，判出 %d，对 %d，多判 %d，漏判 %d" % (tag, m[b].sum(), m[a].sum(), tp, fp, fn))
fp = m[m["zt_tol"] & ~m["zt_u"]]
say("  容 ±1 分多判的例子（不复权上不是涨停）：", fp[["code", "date", "high", "lim", "pc"]].head(6).to_dict("records"))
say("  多判的按板块 %s；其中当前 ST %d 行" % (fp["code"].map(ds.board_of).value_counts().to_dict(), fp["code"].isin(st).sum()))
fn = m[~m["zt_tol"] & m["zt_u"]]
say("  容 ±1 分仍漏判的例子：", fn[["code", "date", "high", "lim", "pc"]].head(6).to_dict("records"))
say("  漏判的按板块 %s" % fn["code"].map(ds.board_of).value_counts().to_dict())
# 真值自己：不复权表上「收在最高价、high == 涨停价」占收在最高价的比例，分板块（ST 若 5% 则主板 ST 的涨停全被漏掉）
stm = u[u["code"].isin(st) & (u["code"].map(ds.board_of) == "main") & (u["date"] >= "2025-01-01")]
h5 = (stm["high_u"] - ds.limit_price_arr(stm["pc"], pd.Series(5.0, index=stm.index), stm["code"])).abs() < 1e-9
say("  主板 ST（2025-01 起）：按 10%% 判涨停 %d 行，按 5%% 判涨停 %d 行（收在最高价 %d 行）" % (
    stm["zt_u"].sum(), (h5 & (stm["close_u"] == stm["high_u"])).sum(), (stm["close_u"] == stm["high_u"]).sum()))
(OUT / "results_a06b.txt").write_text("\n".join(L), encoding="utf-8")
