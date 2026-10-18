"""a09c：幸存者偏差量化。退市票历史（raw/tx_delist.parquet，腾讯 raw 800 根）+ 退市名单。
    (1) 2020-07..2025-12 退市只数、按年、退市前名字 ST/退 占比；腾讯能拉到的覆盖；
    (2) 这些票在市期间（剔除退市前最后 30 根，退市整理期）的 y_up / y_open 正样本率 vs 同期全市场（px_screen）；
    (3) 对全市场基准率的影响：按天把退市票并进去重算基准，差多少 pp；
    (4) 对「候选池内 20 日涨幅前 10」的影响：每天用 px_screen 的 ret20 第 10 高当门槛，数退市票超过门槛的 (票,日)，
        并看它们的 y_open（粗估：不剔 ST / 减持，只按 收盘>=2、历史>=120 根、ret20 门槛）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
L = []
W, TH = 20, 0.5


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def lab(g):
    n = len(g)
    h, o, c, lo = (g[k].to_numpy(float) for k in ("high", "open", "close", "low"))
    fut = np.full(n, np.nan)
    if n > W:
        fut[: n - W] = swv(h[1:], W).max(axis=1)
    o1 = np.append(o[1:], np.nan)
    pct = 0.2 if g["code"].iloc[0][:2] in ("30", "68") else 0.1
    lim = np.round(c * (1 + pct) + 1e-9, 2)
    l1 = np.append(lo[1:], np.nan)
    yz = np.isfinite(l1) & (l1 >= lim - 1e-9)
    r20 = fut / o1 - 1
    y_up = np.where(np.isfinite(fut), (fut / c - 1 > TH).astype(float), np.nan)
    y_open = np.where(np.isfinite(r20), ((r20 > TH) & ~yz).astype(float), np.nan)
    ret20 = pd.Series(c).pct_change(20).to_numpy()
    return pd.DataFrame({"code": g["code"].to_numpy(), "date": g["date"].to_numpy(), "close": c, "y_up": y_up, "y_open": y_open,
                         "ret20": ret20, "nbar": np.arange(n) + 1, "to_end": n - 1 - np.arange(n)})


dl = pd.read_csv(RAW / "delist_list.csv", dtype=str)
# 上交所那张表 akshare 的列名叫「暂停上市日期」，值其实是终止上市日（2024-07-05 这类），a09 探针把它映成了 suspend_date
dl["delist_date"] = dl["delist_date"].where(dl["delist_date"].notna(), dl["suspend_date"])
win = dl[(dl["delist_date"] >= "2020-07-01") & (dl["delist_date"] <= "2025-12-31") & dl["code"].str[0].isin(list("036"))]
say("2020-07..2025-12 沪深 A 股终止上市 %d 只（沪 %d / 深 %d），按年 %s" % (len(win), (win["exch"] == "sh").sum(), (win["exch"] == "sz").sum(), win["delist_date"].str[:4].value_counts().sort_index().to_dict()))
nm = win["name"].astype(str)
say("  退市时名字含 ST %d、含「退」%d、两者都不含 %d（%s）" % (nm.str.contains("ST").sum(), nm.str.contains("退").sum(), (~nm.str.contains("ST|退")).sum(),
                                                   win.loc[~nm.str.contains("ST|退"), "name"].head(12).tolist()))
t = pd.read_parquet(RAW / "tx_delist.parquet")
t["code"] = t["code"].astype(str).str.zfill(6)
t = t[(t["volume"] > 0) & (t["close"] > 0)].sort_values(["code", "date"])
cov = t.groupby("code")["date"].agg(["min", "max", "size"])
say("腾讯拉到 %d 只，每只行数中位 %d，起始日中位 %s；最后一根晚于名单退市日的 %d 只" % (
    len(cov), cov["size"].median(), sorted(cov["min"])[len(cov) // 2], (cov["max"] > cov.index.map(win.set_index("code")["delist_date"].to_dict())).sum()))
d = pd.concat([lab(g) for _, g in t.groupby("code", sort=False)], ignore_index=True)
d = d[(d["date"] >= "2020-07-01") & (d["date"] <= "2025-12-31")]
d["year"] = d["date"].str[:4]
# 在市主体：剔退市前最后 30 根（退市整理期 15 天 + 停牌）
body = d[(d["to_end"] >= 30) & np.isfinite(d["y_open"])]
say("退市票在市期间（退市前 30 根之外）有标签的行 %d：y_up %.3f%%，y_open %.3f%%；收盘>=2 元的 %d 行：y_open %.3f%%" % (
    len(body), 100 * body["y_up"].mean(), 100 * body["y_open"].mean(), (body["close"] >= 2).sum(), 100 * body.loc[body["close"] >= 2, "y_open"].mean()))
say("  按年：\n" + body.groupby("year").agg(rows=("code", "size"), codes=("code", "nunique"), y_open=("y_open", "mean"), y_up=("y_up", "mean")).round(4).to_string())
tail = d[(d["to_end"] < 30) & np.isfinite(d["y_open"])]
say("  最后 30 根 %d 行：y_open %.3f%%（退市整理期的反弹，不该进任何口径）" % (len(tail), 100 * tail["y_open"].mean()))
# 全市场同期（研究表 labels_screen + px_screen）
ls = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/labels_screen.parquet", columns=["code", "date", "y_open", "y_up"])
px = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/px_screen.parquet", columns=["code", "date", "close"])
mk = ls.merge(px, on=["code", "date"])
mk = mk[np.isfinite(mk["y_open"]) & (mk["close"] >= 2)]
mk["year"] = mk["date"].str[:4]
base_day = mk.groupby("date")["y_open"].agg(["sum", "size"])
dd = body[body["close"] >= 2].groupby("date")["y_open"].agg(["sum", "size"])
j = base_day.join(dd, rsuffix="_dl", how="left").fillna(0)
j["base"] = j["sum"] / j["size"]
j["base_all"] = (j["sum"] + j["sum_dl"]) / (j["size"] + j["size_dl"])
j["year"] = j.index.str[:4]
yy = j.groupby("year").agg(base=("base", "mean"), base_all=("base_all", "mean"), dl_per_day=("size_dl", "mean"), n_per_day=("size", "mean"))
yy["diff_pp"] = (100 * (yy["base_all"] - yy["base"])).round(4)
say("全市场按天基准（收盘>=2 元）：研究表 vs 并入退市票：\n" + yy.round(5).to_string())
# 前 10 门槛：每天 px_screen 候选（收盘>=2、>=120 根）里 ret20 第 10 高
px = px.sort_values(["code", "date"])
px["ret20"] = px.groupby("code")["close"].pct_change(20)
px["nbar"] = px.groupby("code").cumcount() + 1
el = px[(px["close"] >= 2) & (px["nbar"] >= 120) & np.isfinite(px["ret20"])]
thr = el.groupby("date")["ret20"].apply(lambda s: s.nlargest(10).iloc[-1] if len(s) >= 10 else np.nan).rename("thr10")
cand = body[(body["close"] >= 2) & (body["nbar"] >= 120)].merge(thr, left_on="date", right_index=True)
hit = cand[cand["ret20"] >= cand["thr10"]]
say("退市票里会进「ret20 前 10」门槛（不剔 ST / 减持，粗估）的 (票,日) %d，涉及 %d 只、%d 天；它们的 y_open %.2f%%（前 10 席位共 %d 个，占 %.3f%%）" % (
    len(hit), hit["code"].nunique(), hit["date"].nunique(), 100 * hit["y_open"].mean() if len(hit) else 0, 10 * thr.notna().sum(), 100 * len(hit) / max(10 * thr.notna().sum(), 1)))
say("  按年 %s" % hit.groupby("year").size().to_dict())
(OUT / "results_a09c.txt").write_text("\n".join(L), encoding="utf-8")
