"""
同花顺题材原因（data/events/ths_harden，2023-10 起）：题材热度 / 扩散的离线诊断（不训模型、不计次数）。

    PYTHONIOENCODING=utf-8 python research/polish/theme.py

T 日文件 = T 日收盘后的涨停票及其题材（按 T 日对齐，只用 <= T 的文件）。
  theme_cnt5(theme, T)  = 近 5 个文件日里带该题材的涨停只数（去重股票）
  stock 的题材 = 它近 20 个文件日里被标过的题材集合；heat5 = max_theme theme_cnt5；tagged20 = 近 20 日出现过
在 ret20 前 30 内按 heat 三分位 / 有无标签看命中（按天聚类），两段：筛选集只有 2023-11..2024-05，开发集全。
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import common_pl as P

HARD = P.ROOT / "data" / "events" / "ths_harden"


def load_harden(end: str = "2025-12-31") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(HARD / "*" / "*.parquet"))):
        d = os.path.basename(f)[:10]
        if d > end:
            continue
        t = pd.read_parquet(f, columns=["date", "code", "reason"])
        t["date"] = d
        rows.append(t)
    h = pd.concat(rows, ignore_index=True)
    h["code"] = h["code"].astype(str).str.zfill(6)
    h["reason"] = h["reason"].fillna("")
    return h


def main():
    h = load_harden()
    days = sorted(h["date"].unique())
    P.say("harden %d 行 %d 天 %s..%s；每天中位 %d 只；北交所行 %d" % (
        len(h), len(days), days[0], days[-1], h.groupby("date").size().median(), int(h["code"].str[0].isin(list("489")).sum())))
    ex = h.assign(theme=h["reason"].str.split("+")).explode("theme")
    ex = ex[ex["theme"].str.len() > 0]
    P.say("题材 %d 种；每只平均 %.1f 个题材" % (ex["theme"].nunique(), len(ex) / len(h)))
    # 每个文件日：题材 -> 近 5 / 10 个文件日去重股票数
    didx = {d: i for i, d in enumerate(days)}
    ex["di"] = ex["date"].map(didx)
    by_theme = defaultdict(list)   # theme -> list of (di, code)
    for th, di, c in zip(ex["theme"], ex["di"], ex["code"]):
        by_theme[th].append((di, c))
    cnt5, cnt20 = {}, {}
    for th, lst in by_theme.items():
        arr = sorted(lst)
        dis = np.array([a[0] for a in arr])
        for i in range(len(days)):
            pass
        # 稀疏：只在题材出现过的日子及之后 20 天算
        active = sorted(set(dis))
        for di in range(active[0], len(days)):
            m5 = (dis > di - 5) & (dis <= di)
            if not m5.any() and not ((dis > di - 20) & (dis <= di)).any():
                continue
            cnt5[(th, di)] = len({a[1] for a, ok in zip(arr, m5) if ok})
            m20 = (dis > di - 20) & (dis <= di)
            cnt20[(th, di)] = len({a[1] for a, ok in zip(arr, m20) if ok})
    P.say("题材计数 %d 条" % len(cnt5))
    # 股票 -> 近 20 个文件日的题材集合（按文件日对齐）；再对每个交易日 T 取最近的文件日 <= T
    pool = P.load_pool(["date", "code", "board", "y_open", "y_up", "pool", "lret20", "nlu5", "nlu20"])
    pool = pool[pool["pool"] & (pool["date"] >= "2023-11-01")].reset_index(drop=True)
    t = P.rank_by(pool, pool["lret20"].to_numpy(), keep=30)
    tdays = sorted(t["date"].unique())
    last_file = {}
    j = 0
    for d in tdays:
        while j + 1 < len(days) and days[j + 1] <= d:
            j += 1
        last_file[d] = j if days[j] <= d else -1
    stock_tags = defaultdict(list)   # code -> list of (di, theme)
    for c, di, th in zip(ex["code"], ex["di"], ex["theme"]):
        stock_tags[c].append((di, th))
    heat5, heat20, tagged, ntags, own5 = [], [], [], [], []
    for d, c in zip(t["date"], t["code"]):
        di = last_file[d]
        tags = [th for (k, th) in stock_tags.get(c, []) if di - 20 < k <= di]
        own = [1 for (k, th) in stock_tags.get(c, []) if di - 5 < k <= di]
        tagged.append(1.0 if tags else 0.0)
        ntags.append(len(set(tags)))
        own5.append(1.0 if own else 0.0)
        if tags:
            heat5.append(max(cnt5.get((th, di), 0) for th in set(tags)))
            heat20.append(max(cnt20.get((th, di), 0) for th in set(tags)))
        else:
            heat5.append(np.nan)
            heat20.append(np.nan)
    t["tagged20"] = tagged
    t["ntags"] = ntags
    t["own5"] = own5
    t["heat5"] = heat5
    t["heat20"] = heat20
    t["seg"] = P.seg_of(t["date"])
    t.to_parquet(P.PL / "theme_top30.parquet", index=False)
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))

    def dc(x):
        s = x.groupby("date")["y_open"].mean()
        return 100 * s.mean(), 100 * s.std(ddof=1) / np.sqrt(max(len(s), 2)), len(s)
    for seg in ("screen", "dev"):
        s = t[(t["seg"] == seg) & (t["rank"] <= 30)]
        say("== %s（%d 天）前 30 ==" % (seg, s["date"].nunique()))
        for lab, c in (("近 20 日被同花顺标过题材", s["tagged20"] > 0), ("没标过", s["tagged20"] == 0),
                       ("近 5 日自己涨停进榜(own5)", s["own5"] > 0), ("own5=0 但 tagged20", (s["own5"] == 0) & (s["tagged20"] > 0))):
            m, se, n = dc(s[c])
            say("  %-28s 占 %4.1f%%  命中 %5.1f (se %.1f, %d 天)" % (lab, 100 * c.mean(), m, se, n))
        x = s[s["tagged20"] > 0].copy()
        for col in ("heat5", "heat20", "ntags"):
            q = x.groupby("date")[col].rank(pct=True, method="first")
            x["_b"] = np.where(q <= 1 / 3, 0, np.where(q <= 2 / 3, 1, 2))
            vals = [dc(x[x["_b"] == b]) for b in (0, 1, 2)]
            dd = (x[x["_b"] == 2].groupby("date")["y_open"].mean() - x[x["_b"] == 0].groupby("date")["y_open"].mean()).dropna()
            say("  %-8s 低/中/高 三分位命中 %5.1f / %5.1f / %5.1f  高-低 iid t %+.1f  中位值 %s" % (
                col, vals[0][0], vals[1][0], vals[2][0], dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))),
                [float(np.nanmedian(x.loc[x["_b"] == b, col])) for b in (0, 1, 2)]))
        for thr in (3, 5, 10, 20):
            c = x["heat5"] >= thr
            m1, se1, n1 = dc(x[c])
            m0, se0, n0 = dc(x[~c])
            say("  heat5>=%-3d 占(标过的) %4.1f%%  命中 %5.1f  对 <%d：%5.1f" % (thr, 100 * c.mean(), m1, thr, m0))
        # 前 10 里的同样切片
        s10 = s[s["rank"] <= 10]
        for lab, c in (("前10 tagged20", s10["tagged20"] > 0), ("前10 untagged", s10["tagged20"] == 0),
                       ("前10 heat5>=5", s10["heat5"] >= 5), ("前10 tagged & heat5<5", (s10["tagged20"] > 0) & (s10["heat5"] < 5))):
            m, se, n = dc(s10[c])
            say("  %-24s 占 %4.1f%%  命中 %5.1f (se %.1f)" % (lab, 100 * c.mean(), m, se))
    (P.HERE / "results_theme.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
