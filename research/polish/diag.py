"""
离线诊断（不训模型、不计试验次数）：ret20 名次内的条件命中率。

    PYTHONIOENCODING=utf-8 python research/polish/diag.py

1 ret20 名次档的命中（两段）
2 ret20 前 30 内，各特征三分位（当天内分位）的命中：按天聚类 se，看哪些维度在名单内部还有分辨力
3 买得到：前 10 / 30 里次日一字占比、按今日涨停 / 连板高度的次日一字概率
4 市场状态（T 日已知：T-40..T-21 候选池基准率）× 名次档
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P

FEATS = ["ret20", "ret20_norm", "lret20_vol", "ret5", "ret10", "ret40", "ret60", "ret20_lag5", "ret1", "gap1", "cr1",
         "vol20", "vol60", "amp20", "turn1", "turn5", "turn20", "amt1", "amt20", "amt_ratio", "vr5_20", "mcap", "price_u",
         "dist_hi20", "dist_hi60", "dist_hi120", "pos20", "nh20_5", "up20", "max1_20", "nlu20", "nlu5", "lu_streak", "nld20", "nbar"]


def day_cluster(x: pd.DataFrame, col: str = "y_open") -> tuple[float, float, int]:
    s = x.groupby("date")[col].mean()
    return float(s.mean()), float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan, int(len(s))


def main():
    df = P.load_pool()
    df = df[df["pool"] & (df["date"] >= "2022-01-01")].reset_index(drop=True)
    df["seg"] = P.seg_of(df["date"])
    t = P.rank_by(df, df["lret20"].to_numpy(), keep=200)
    t = t.merge(df.drop(columns=["board", "y_open", "y_up"]), on=["date", "code"], how="left")
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))

    say("== 1 ret20 名次档命中（买得到口径，按天聚类）==")
    bins = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, 200)]
    say("%-10s %-22s %-22s" % ("名次", "筛选集 命中% (se)", "开发集 命中% (se)"))
    for lo, hi in bins:
        s = t[(t["rank"] >= lo) & (t["rank"] <= hi)]
        a = day_cluster(s[s["seg"] == "screen"])
        b = day_cluster(s[s["seg"] == "dev"])
        say("%-10s %6.2f (%.2f)          %6.2f (%.2f)" % ("%d-%d" % (lo, hi), 100 * a[0], 100 * a[1], 100 * b[0], 100 * b[1]))
    for seg in ("screen", "dev"):
        s = t[(t["rank"] <= 10) & (t["seg"] == seg)]
        say("  %s 前 10：次日一字 %.1f%%，一字里 r20>50%% 的 %.1f%%（这些算没中）；y_up（收盘口径）%.2f%% 对 y_open %.2f%%" % (
            seg, 100 * s["yizi1"].mean(), 100 * (s["r20_open"][s["yizi1"] > 0] > 0.5).mean(), 100 * s["y_up"].mean(), 100 * s["y_open"].mean()))

    say("\n== 2 前 30 内各特征三分位（当天前 30 内分位）的命中，两段：低 / 中 / 高 (按天聚类 se)；高-低 的 t ==")
    top = t[t["rank"] <= 30].copy()
    for f in FEATS:
        if f not in top.columns:
            continue
        q = top.groupby("date")[f].rank(pct=True, method="first")
        top["_b"] = np.where(q <= 1 / 3, 0, np.where(q <= 2 / 3, 1, 2))
        row = "%-12s" % f
        for seg in ("screen", "dev"):
            s = top[top["seg"] == seg]
            vals = []
            for b in (0, 1, 2):
                m, se, n = day_cluster(s[s["_b"] == b])
                vals.append((m, se))
            d = s[s["_b"] == 2].groupby("date")["y_open"].mean() - s[s["_b"] == 0].groupby("date")["y_open"].mean()
            d = d.dropna()
            tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan
            row += "  | %5.1f %5.1f %5.1f  t%+5.1f" % (100 * vals[0][0], 100 * vals[1][0], 100 * vals[2][0], tt)
        say(row)
    say("  （t 是 iid 按天 t，相邻日重叠，真 t 约除以 1.5）")

    say("\n== 2b 前 30 内二值特征：有 / 无 的命中（两段）==")
    for f, cond in (("lu_close 今日涨停", top["lu_close"] > 0.5), ("yizi_today 今日一字", top["yizi_today"] > 0.5),
                    ("lu_streak>=2 连板", top["lu_streak"] >= 2), ("lu_streak>=3", top["lu_streak"] >= 3),
                    ("nlu20>=3", top["nlu20"] >= 3), ("nlu20==0", top["nlu20"] == 0), ("ret1<0 今日下跌", top["ret1"] < 0),
                    ("dist_hi20>=-0.02 贴 20 日高", top["dist_hi20"] >= -0.02), ("dist_hi20<-0.15", top["dist_hi20"] < -0.15),
                    ("ret5<0", top["ret5"] < 0), ("ret5>0.2", top["ret5"] > 0.2), ("ret20>1.0 已翻倍", top["ret20"] > 1.0),
                    ("ret20>2.0", top["ret20"] > 2.0), ("price_u<5", top["price_u"] < 5), ("price_u>50", top["price_u"] > 50),
                    ("mcap<20亿", top["mcap"] < 2e9), ("mcap>100亿", top["mcap"] > 1e10), ("amt20<5000万", top["amt20"] < 5e7),
                    ("board=bj", top["board"] == "bj"), ("board=main", top["board"] == "main"), ("board=chinext", top["board"] == "chinext"),
                    ("board=star", top["board"] == "star"), ("nld20>=1", top["nld20"] >= 1), ("susp(复牌)nbar", top["nbar"] < 250)):
        row = "%-26s" % f
        for seg in ("screen", "dev"):
            s = top[top["seg"] == seg]
            c = cond[s.index]
            a = day_cluster(s[c])
            b = day_cluster(s[~c])
            share = float(c.mean())
            row += "  | 占 %4.1f%%  有 %5.1f (n天 %3d)  无 %5.1f" % (100 * share, 100 * a[0], a[2], 100 * b[0])
        say(row)

    say("\n== 3 买得到：次日一字的概率（前 30 内）==")
    for f, cond in (("今日涨停", top["lu_close"] > 0.5), ("今日一字", top["yizi_today"] > 0.5), ("连板>=2", top["lu_streak"] >= 2),
                    ("连板>=3", top["lu_streak"] >= 3), ("今日没涨停", top["lu_close"] < 0.5)):
        row = "%-12s" % f
        for seg in ("screen", "dev"):
            s = top[top["seg"] == seg]
            c = cond[s.index]
            x = s[c]
            row += "  | 占 %4.1f%%  次日一字 %5.1f%%  命中(买得到) %5.1f%%  r20>50%%(收盘口径) %5.1f%%" % (
                100 * c.mean(), 100 * x["yizi1"].mean(), 100 * x["y_open"].mean(), 100 * (x["r20_open"] > 0.5).mean())
        say(row)
    for seg in ("screen", "dev"):
        s = t[(t["rank"] <= 10) & (t["seg"] == seg)]
        lost = s[(s["yizi1"] > 0) & (s["r20_open"] > 0.5)]
        say("  %s 前 10 里因一字丢掉的命中 %d / %d 席 = %.2fpp；若这些票换成第 11 名起的下一只，期望补回约 %.2fpp" % (
            seg, len(lost), len(s), 100 * len(lost) / len(s), 100 * (len(lost) / len(s)) * s["y_open"].mean()))

    say("\n== 4 市场状态（T-40..T-21 候选池基准率三分位，T 日已知）× ret20 名次档 ==")
    base = df.groupby("date")["y_open"].mean().sort_index()
    st = base.rolling(20).mean().shift(21)
    t["mkt"] = t["date"].map(st)
    for seg in ("screen", "dev"):
        s = t[(t["seg"] == seg) & np.isfinite(t["mkt"])].copy()
        qs = s.groupby("date")["mkt"].first().quantile([1 / 3, 2 / 3]).to_numpy()
        s["mq"] = np.where(s["mkt"] <= qs[0], "冷", np.where(s["mkt"] <= qs[1], "中", "热"))
        say("  %s：" % seg)
        for mq in ("冷", "中", "热"):
            x = s[s["mq"] == mq]
            r = "    %s(%3d 天) " % (mq, x["date"].nunique())
            for lo, hi in ((1, 5), (6, 10), (11, 20), (21, 30), (31, 50)):
                m, se, n = day_cluster(x[(x["rank"] >= lo) & (x["rank"] <= hi)])
                r += " %d-%d: %5.1f" % (lo, hi, 100 * m)
            cur = x[x["rank"] <= 10].groupby("date")["y_open"].mean().mean()
            r += "   前10 %.1f  当期候选池基准 %.2f" % (100 * cur, 100 * df[df["date"].isin(x["date"].unique())].groupby("date")["y_open"].mean().mean())
            say(r)

    (P.HERE / "results_diag.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
