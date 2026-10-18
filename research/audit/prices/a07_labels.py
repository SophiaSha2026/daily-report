"""a07：标签独立重写（不 import label.py / buyable.py），按定义自己算，和四份现成标签逐行比。

定义（任务书）：
  y_up   之后 20 根自身 K 线（不含当天）最高价 / 当天收盘 - 1 > 0.5；不足 20 根 NaN
  y_open 次日开盘买入：之后 20 根（t+1..t+20）最高价 / 次日开盘 - 1 > 0.5，且次日不是一字板；
         一字板两种定义都算：A 严格「开=收=高=低=涨停价」；B 生产口径「次日最低价 >= 涨停价」
  涨停价 = round(close_t × (1+pct), 2)（北交所向下取整），pct 按板块 10/20/30，ST 另算
比对对象：生产 train.parquet 的 y_up（raw/train_yup.parquet）、生产缓存 labels_open.parquet、
          研究 labels_screen（px_screen pit 价上）、研究 labels_open_snap（leak，2023-05-30 起）。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
L = []
W, TH = 20, 0.5


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def pct_of(c):
    if c[:2] in ("30", "68"):
        return 0.20
    if c[0] in "849":
        return 0.30
    return 0.10


def lim_of(close, c):
    x = close * (1 + pct_of(c))
    if c[0] in "849":
        return np.floor(x * 100 + 1e-6) / 100
    return np.round(x + 1e-9, 2)


def relabel(px, date_idx):
    """px：code date open high low close，按 code,date 升序。date_idx：全市场交易日序号，算窗口跨度。"""
    parts = []
    for c, g in px.groupby("code", sort=False):
        n = len(g)
        h = g["high"].to_numpy(float)
        o = g["open"].to_numpy(float)
        lo = g["low"].to_numpy(float)
        cl = g["close"].to_numpy(float)
        fut = np.full(n, np.nan)
        if n > W:
            fut[: n - W] = swv(h[1:], W).max(axis=1)
        y_up = np.where(np.isfinite(fut), (fut / cl - 1 > TH).astype(float), np.nan)
        o1 = np.append(o[1:], np.nan)
        h1 = np.append(h[1:], np.nan)
        l1 = np.append(lo[1:], np.nan)
        c1 = np.append(cl[1:], np.nan)
        lim1 = lim_of(cl, c)
        yzA = np.isfinite(o1) & (np.abs(o1 - lim1) < 1e-9) & (np.abs(c1 - lim1) < 1e-9) & (np.abs(h1 - lim1) < 1e-9) & (np.abs(l1 - lim1) < 1e-9)
        yzB = np.isfinite(l1) & (l1 >= lim1 - 1e-9)
        r20 = np.where(np.isfinite(fut) & np.isfinite(o1), fut / o1 - 1, np.nan)
        okk = np.isfinite(r20)
        y_oA = np.where(okk, ((r20 > TH) & ~yzA).astype(float), np.nan)
        y_oB = np.where(okk, ((r20 > TH) & ~yzB).astype(float), np.nan)
        di = g["date"].map(date_idx).to_numpy()
        span = np.full(n, np.nan)
        if n > W:
            span[: n - W] = di[W:] - di[: n - W]
        parts.append(pd.DataFrame({"code": c, "date": g["date"].to_numpy(), "my_y_up": y_up, "my_r20_open": r20,
                                   "my_yzA": yzA, "my_yzB": yzB, "my_y_openA": y_oA, "my_y_openB": y_oB, "span": span,
                                   "nfut": np.arange(n)[::-1]}))
    return pd.concat(parts, ignore_index=True)


def compare(mine, theirs, mcol, tcol, tag, keys=("code", "date")):
    m = mine.merge(theirs, on=list(keys), how="inner")
    both = m[np.isfinite(m[mcol]) & np.isfinite(m[tcol])]
    onlym = m[np.isfinite(m[mcol]) & ~np.isfinite(m[tcol])]
    onlyt = m[~np.isfinite(m[mcol]) & np.isfinite(m[tcol])]
    ne = both[both[mcol] != both[tcol]]
    say("[%s] 并上 %d 行；都有值 %d，不一致 %d（%.5f%%）；我有值对方 NaN %d，对方有值我 NaN %d" % (
        tag, len(m), len(both), len(ne), 100 * len(ne) / max(len(both), 1), len(onlym), len(onlyt)))
    say("   正样本率：我 %.4f%%，对方 %.4f%%" % (100 * both[mcol].mean(), 100 * both[tcol].mean()))
    if len(ne):
        say("   不一致按年月 %s；按只 top %s" % (ne["date"].str[:7].value_counts().head(6).to_dict(), ne["code"].value_counts().head(6).to_dict()))
        say("   不一致里 我=1 对方=0 %d，我=0 对方=1 %d；窗口跨 >20 个市场日的 %d" % (
            (ne[mcol] > ne[tcol]).sum(), (ne[mcol] < ne[tcol]).sum(), (ne["span"] > W).sum() if "span" in ne else -1))
        say("   例：\n" + ne.head(8).to_string())
    if len(onlyt):
        say("   对方有值我 NaN 的例子（末尾窗口不足？）：按年月 %s" % onlyt["date"].str[:7].value_counts().head(4).to_dict())
    return m, ne


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "prod"
    if which == "prod":
        d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "open", "high", "low", "close"])
        d["code"] = d["code"].astype(str).str.zfill(6)
        d = d.sort_values(["code", "date"]).reset_index(drop=True)
        dates = np.array(sorted(d["date"].unique()))
        didx = pd.Series(np.arange(len(dates)), index=dates)
        mine = relabel(d, didx)
        mine.to_parquet(OUT / "raw/a07_mine_prod.parquet", index=False)
        v = mine[np.isfinite(mine["my_y_up"])]
        say("生产日线上重算：%d 行有 y_up，正样本率 %.4f%%；y_openB 正样本率 %.4f%%；一字板 A %.3f%% B %.3f%%；窗口跨 >20 市场日的行 %d（%.3f%%），最大 %d" % (
            len(v), 100 * v["my_y_up"].mean(), 100 * v["my_y_openB"].mean(), 100 * v["my_yzA"].mean(), 100 * v["my_yzB"].mean(),
            (v["span"] > W).sum(), 100 * (v["span"] > W).mean(), v["span"].max()))
        say("  一字板 A（开=收=高=低=涨停价）和 B（最低价>=涨停价）不同的行 %d；A 真 B 假 %d（不可能），B 真 A 假 %d（次日开在涨停、盘中打开或高开后封板）" % (
            (v["my_yzA"] != v["my_yzB"]).sum(), (v["my_yzA"] & ~v["my_yzB"]).sum(), (v["my_yzB"] & ~v["my_yzA"]).sum()))
        say("  y_open 按 A / B 两种一字板定义不同的行 %d（占正样本 %.3f%%）" % (
            (v["my_y_openA"] != v["my_y_openB"]).sum(), 100 * (v["my_y_openA"] != v["my_y_openB"]).sum() / max(v["my_y_openB"].sum(), 1)))
        say("  y_up=1 但 y_openB=0 的行 %d（买不到或开盘已高）；y_up=0 但 y_openB=1 的 %d（低开后涨）" % (
            ((v["my_y_up"] == 1) & (v["my_y_openB"] == 0)).sum(), ((v["my_y_up"] == 0) & (v["my_y_openB"] == 1)).sum()))
        # 1 生产训练表 y_up
        t = pd.read_parquet(OUT / "raw/train_yup.parquet")
        t["code"] = t["code"].astype(str).str.zfill(6)
        compare(mine, t, "my_y_up", "y_up", "生产 train.parquet y_up（建表时的日线）")
        # 2 生产 labels_open.parquet
        lo = pd.read_parquet(ROOT / "data/breakout/raw/labels_open.parquet")
        lo["code"] = lo["code"].astype(str).str.zfill(6)
        m, ne = compare(mine, lo, "my_y_openB", "y_open", "生产 labels_open.parquet y_open（口径 B）")
        both = m[np.isfinite(m["my_r20_open"]) & np.isfinite(m["r20_open"])]
        say("   r20_open 逐位相同率 %.5f%%；yizi1 不一致 %d 行" % (100 * (np.abs(both["my_r20_open"] - both["r20_open"]) < 1e-9).mean(),
                                                           (both["my_yzB"] != both["yizi1"]).sum()))
        # 3 leak labels_open_snap（新浪取整价快照，2023-05-30 起）
        ls = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/labels_open_snap.parquet")
        ls["code"] = ls["code"].astype(str).str.zfill(6)
        compare(mine, ls, "my_y_openB", "y_open", "研究 labels_open_snap y_open（leak 快照）")
    else:
        p = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/px_screen.parquet", columns=["code", "date", "open", "high", "low", "close", "close_u", "low_u", "q"])
        p["code"] = p["code"].astype(str).str.zfill(6)
        p = p.sort_values(["code", "date"]).reset_index(drop=True)
        dates = np.array(sorted(p["date"].unique()))
        didx = pd.Series(np.arange(len(dates)), index=dates)
        mine = relabel(p, didx)
        mine.to_parquet(OUT / "raw/a07_mine_screen.parquet", index=False)
        ls = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/labels_screen.parquet")
        ls["code"] = ls["code"].astype(str).str.zfill(6)
        compare(mine, ls, "my_y_up", "y_up", "研究 labels_screen y_up（pit 价）")
        m, ne = compare(mine, ls, "my_y_openB", "y_open", "研究 labels_screen y_open（pit 价，口径 B）")
        both = m[np.isfinite(m["my_r20_open"]) & np.isfinite(m["r20_open"])]
        say("   r20_open |差|<1e-9 的比例 %.5f%%；yizi1 不一致 %d；我的 yzB 对 yizi1_u（不复权判）不一致 %d" % (
            100 * (np.abs(both["my_r20_open"] - both["r20_open"]) < 1e-9).mean(), (both["my_yzB"] != both["yizi1"]).sum(),
            (both["my_yzB"] != both["yizi1_u"]).sum()))
        # pit 价（不取整）上判一字板本身就有问题：涨停价是在不复权价上四舍五入的。用不复权价重判一遍当真值
        pu = p[["code", "date", "close_u", "low_u"]].copy()
        pu = pu.sort_values(["code", "date"])
        l1u = pu.groupby("code")["low_u"].shift(-1)
        limu = pu.apply(lambda r: lim_of(r["close_u"], r["code"]), axis=1) if False else None
        pct = pu["code"].map(pct_of).to_numpy()
        isbj = pu["code"].str[0].isin(list("849")).to_numpy()
        x = pu["close_u"].to_numpy() * (1 + pct)
        limu = np.where(isbj, np.floor(x * 100 + 1e-6) / 100, np.round(x + 1e-9, 2))
        yz_u = (l1u.to_numpy() >= limu - 1e-9) & np.isfinite(l1u.to_numpy())
        pu["yz_u_mine"] = yz_u
        mm = mine.merge(pu[["code", "date", "yz_u_mine"]], on=["code", "date"])
        v = mm[np.isfinite(mm["my_y_openB"])]
        say("   不复权价上重判一字板：和 pit 价判的不同 %d 行（占 %.4f%%）；翻 y_open 的 %d 行" % (
            (v["my_yzB"] != v["yz_u_mine"]).sum(), 100 * (v["my_yzB"] != v["yz_u_mine"]).mean(),
            ((v["my_r20_open"] > TH) & (v["my_yzB"] != v["yz_u_mine"])).sum()))
        cx = v[(v["code"].str[:2] == "30") & (v["date"] < "2020-08-24")]
        say("   创业板 2020-08-24 前 %d 行（涨跌停 10%%、代码按 20%% 判一字板）：y_openB 正样本 %d" % (len(cx), cx["my_y_openB"].sum()))
    (OUT / ("results_a07_%s.txt" % which)).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
