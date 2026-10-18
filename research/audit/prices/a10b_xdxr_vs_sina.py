"""a10b：通达信 xdxr（category 1 除权除息）vs 新浪 qfq_factor 台阶（unadj_2020），285 只样本（含 38 只北交所）。
    台阶公式同 research/datasrc/tdx_reconcile.py：adj = (前收 − 派息/10 + 配股/10 × 配股价) / (1 + 送转/10 + 配股/10)。
    看：事件日两边对不对得上、台阶大小相对差、哪一边多出来的事件。"""
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


x = pd.read_parquet(OUT / "raw/tdx_xdxr.parquet")
x = x[x["category"] == 1].copy()
x["date"] = pd.to_datetime(dict(year=x["year"], month=x["month"], day=x["day"])).dt.strftime("%Y-%m-%d")
x = x[x["date"] >= "2020-07-02"]
s = pd.read_csv(OUT / "raw/sample.csv", dtype={"code": str})
codes = s["code"].str.zfill(6).tolist()
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "close_u", "qfq_factor"], filters=[("code", "in", codes)])
u["code"] = u["code"].astype(str).str.zfill(6)
u = u.sort_values(["code", "date"]).reset_index(drop=True)
u["step"] = u.groupby("code")["qfq_factor"].shift(1) / u["qfq_factor"]
u["pc"] = u.groupby("code")["close_u"].shift(1)
ev_s = u[(u["step"] - 1).abs() > 1e-6][["code", "date", "step", "pc"]]
say("样本 %d 只；新浪台阶事件 %d 个（2020-07 起），通达信除权除息事件 %d 个" % (len(codes), len(ev_s), len(x)))
# 通达信事件日落在哪一根 K 线：事件日当天或之后第一根
rows = []
for c, g in x.groupby("code"):
    uc = u[u["code"] == c]
    if not len(uc):
        continue
    ud = uc["date"].to_numpy()
    for _, e in g.iterrows():
        i = np.searchsorted(ud, e["date"])
        if i >= len(ud) or i == 0:
            rows.append((c, e["date"], None, np.nan, np.nan, "新浪表没有这天之后的 K 线" if i >= len(ud) else "事件在新浪表第一根之前"))
            continue
        kd = ud[i]
        pc = uc["close_u"].iloc[i - 1]
        fh = e["fenhong"] or 0.0
        sg = e["songzhuangu"] or 0.0
        pg = e["peigu"] or 0.0
        pj = e["peigujia"] or 0.0
        adj = (pc - fh / 10 + pg / 10 * pj) / (1 + sg / 10 + pg / 10)
        tdx_step = pc / adj
        sina_step = uc["step"].iloc[i]
        rows.append((c, e["date"], kd, tdx_step, sina_step, "" if kd == e["date"] else "事件日没有 K 线（停牌）"))
r = pd.DataFrame(rows, columns=["code", "ev_date", "k_date", "tdx_step", "sina_step", "note"])
ok = r[r["k_date"].notna()].copy()
ok["rel"] = (ok["tdx_step"] / ok["sina_step"] - 1).abs()
say("通达信事件能落到新浪 K 线上的 %d 个；事件日当天没 K 线的 %d 个；其中新浪当天没有台阶（|step-1|<1e-6）的 %d 个" % (
    len(ok), (ok["note"] != "").sum(), ((ok["sina_step"] - 1).abs() < 1e-6).sum()))
both = ok[(ok["sina_step"] - 1).abs() >= 1e-6]
say("两边都有台阶的 %d 个：相对差 <1e-4 %.1f%%、<1e-3 %.1f%%、>1e-2 %d 个" % (
    len(both), 100 * (both["rel"] < 1e-4).mean(), 100 * (both["rel"] < 1e-3).mean(), (both["rel"] > 1e-2).sum()))
say("  相对差 >1e-3 的例子：\n" + both[both["rel"] > 1e-3].sort_values("rel", ascending=False).head(10).to_string())
bj = both[both["code"].str[0].isin(list("849"))]
say("  北交所 %d 个：<1e-4 %.1f%%、>1e-3 %d" % (len(bj), 100 * (bj["rel"] < 1e-4).mean() if len(bj) else 0, (bj["rel"] > 1e-3).sum()))
miss_tdx = ok[(ok["sina_step"] - 1).abs() < 1e-6]
say("通达信有、新浪没有台阶的 %d 个（新浪漏除权？）：\n%s" % (len(miss_tdx), miss_tdx.head(10).to_string()))
# 新浪有台阶、通达信没有事件的
tdx_keys = set(zip(ok["code"], ok["k_date"]))
miss_sina = ev_s[~ev_s.apply(lambda q: (q["code"], q["date"]) in tdx_keys, axis=1)]
say("新浪有台阶、通达信没有除权除息事件的 %d 个：按只 %s" % (len(miss_sina), miss_sina["code"].value_counts().head(8).to_dict()))
say(miss_sina.head(10).to_string())
# 这些新浪独有台阶的大小
say("  新浪独有台阶 |step-1| 分位 %s" % (miss_sina["step"] - 1).abs().quantile([0, .5, .9, 1]).round(5).to_dict())
(OUT / "results_a10b.txt").write_text("\n".join(L), encoding="utf-8")
