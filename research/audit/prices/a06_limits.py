"""a06：涨跌停相关。
  1 ±1 分容差在多少行上起作用：生产 qfq 表 vs 不复权表各算一遍（不复权表上不该有舍入差）。
  2 ST 的涨跌幅是不是 5%：当前 ST 名单在 2026-06 之后（名单肯定生效）的不复权日涨跌分布。
  3 涨停价公式按板块对不对：收在最高价的行里，high 和 limit_price 的差的分布，分板块。
  4 新股首日 / 创业板 2020-08-24 前：labels_screen 的 yizi1 有多少落在没涨跌停或不是 20% 的日子。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import datasource as ds  # noqa: E402

L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def pct_arr(codes):
    return codes.map(lambda c: ds.limit_pct(c, ""))


def limit_stats(df, tag, pcol="close", hcol="high", lcol="low", ccol="close"):
    df = df.sort_values(["code", "date"])
    pc = df.groupby("code")[pcol].shift(1)
    pct = pct_arr(df["code"])
    lim = pd.Series(ds.limit_price_arr(pc, pct, df["code"]), index=df.index)
    ok = pc.notna()
    diff = ((df[hcol] - lim) * 100).round(2)
    at_high = (df[ccol] == df[hcol]) & ok
    say("[%s] 有昨收的行 %d；收在最高价的行 %d" % (tag, ok.sum(), at_high.sum()))
    vc = diff[at_high].value_counts()
    say("  收在最高价的行里 (high - 涨停价)*100 分布：0 分 %d、+1 分 %d、-1 分 %d、+2 %d、-2 %d、其它 %d" % (
        vc.get(0.0, 0), vc.get(1.0, 0), vc.get(-1.0, 0), vc.get(2.0, 0), vc.get(-2.0, 0),
        at_high.sum() - sum(vc.get(x, 0) for x in (0.0, 1.0, -1.0, 2.0, -2.0))))
    exact = at_high & (diff == 0)
    tol1 = at_high & (diff.abs() <= 1)
    say("  判「涨停」：精确 %d，容 ±1 分 %d（多 %d，%.2f%%）" % (exact.sum(), tol1.sum(), tol1.sum() - exact.sum(), 100 * (tol1.sum() - exact.sum()) / max(exact.sum(), 1)))
    # 分板块
    bd = df["code"].map(ds.board_of)
    for b in ("main", "chinext", "star", "bj"):
        mb = at_high & (bd == b)
        v = diff[mb].value_counts()
        say("    %s：0 分 %d、+1 %d、-1 %d、其它 %d" % (b, v.get(0.0, 0), v.get(1.0, 0), v.get(-1.0, 0), mb.sum() - v.get(0.0, 0) - v.get(1.0, 0) - v.get(-1.0, 0)))
    # 一字板（low >= 涨停价）
    yz = ok & (df[lcol] >= lim - 1e-9)
    yz1 = ok & (df[lcol] >= lim - 0.01 - 1e-9)
    say("  一字板（low>=涨停价）%d；容 1 分 %d" % (yz.sum(), yz1.sum()))
    return diff, at_high, lim


say("== 1/3 生产 qfq 表 ==")
d = pd.read_parquet(ROOT / "data/breakout/daily.parquet", columns=["code", "date", "open", "high", "low", "close"])
d["code"] = d["code"].astype(str).str.zfill(6)
limit_stats(d, "daily.parquet qfq")
say("== 1/3 不复权表 unadj_2020（2023-01 起，同期）==")
u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet")
u["code"] = u["code"].astype(str).str.zfill(6)
u = u.rename(columns={"open_u": "open", "high_u": "high", "low_u": "low", "close_u": "close"})
u = u[(u["volume"] > 0) & (u["close"] > 0)]
u23 = u[u["date"] >= "2023-01-01"]
diff_u, ah_u, lim_u = limit_stats(u23, "unadj 2023+")
# 不复权表上还差 1 分的，是什么票？
odd = u23[ah_u & (diff_u.abs() == 1)]
say("  不复权表上仍差 ±1 分的 %d 行：按板块 %s，按年 %s，例 %s" % (
    len(odd), odd["code"].map(ds.board_of).value_counts().to_dict(), odd["date"].str[:4].value_counts().to_dict(),
    odd[["code", "date", "high", "close"]].head(6).to_dict("records")))
# 不复权表上 +1 分的票是不是 ST（5% 口径下 limit 算错）
st = set(json.load(open(ROOT / "cache/st_codes.json", encoding="utf-8"))["codes"])
say("  其中当前 ST 名单内 %d 行" % odd["code"].isin(st).sum())
far = u23[ah_u & (diff_u.abs() >= 2)]
say("  不复权表上收在最高价但离涨停价 >=2 分的行 %d（正常：没涨停）；其中 |high/prev-1| 在 4.9%%~5.1%% 的 %d 行（像 5%% 板）" % (
    len(far), int(((far["high"] / u23.groupby("code")["close"].shift(1)[far.index] - 1).sub(0.05).abs() < 0.0015).sum())))

say("== 2 ST 的涨跌幅 ==")
u["ret"] = u["close"] / u.groupby("code")["close"].shift(1) - 1
rec = u[(u["date"] >= "2026-06-01") & u["code"].isin(st)]
bd = rec["code"].map(ds.board_of)
for b in ("main", "chinext", "star", "bj"):
    x = rec[bd == b]
    if not len(x):
        continue
    mx = x.groupby("code")["ret"].agg(lambda s: s.abs().max())
    say("  %s ST %d 只：2026-06 起单日 |涨跌| 最大值分位 %s；最大值 <=5.1%% 的 %d 只，>5.5%% 的 %d 只，>9.5%% 的 %d 只" % (
        b, len(mx), mx.quantile([0, .25, .5, .75, 1]).round(4).to_dict(), (mx <= 0.051).sum(), (mx > 0.055).sum(), (mx > 0.095).sum()))
    # 触板日：|ret| 在 4.9~5.1% 的天数 vs 9.9~10.1%
    say("     触 5%% 板的 (code,日) %d，触 10%% 板的 %d，触 20%% 板的 %d" % (
        ((x["ret"].abs() - 0.05).abs() < 0.0015).sum(), ((x["ret"].abs() - 0.10).abs() < 0.0015).sum(), ((x["ret"].abs() - 0.20).abs() < 0.0015).sum()))
# 腾讯快照 f[47] 当前涨停价 / 昨收 可以直接验证，这里只列几只主板 ST 最近 20 天
ex = rec[bd == "main"].groupby("code").tail(10)
say("  主板 ST 例（最近 10 天 ret）：", {c: g["ret"].round(4).tolist()[-5:] for c, g in list(ex.groupby("code"))[:4]})

say("== 4 labels_screen 的 yizi1 落在什么日子 ==")
lab = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/labels_screen.parquet", columns=["code", "date", "yizi1", "yizi1_u", "y_open", "y_open_u"])
lab["code"] = lab["code"].astype(str).str.zfill(6)
say("  yizi1 共 %d 行为真（占有标签 %.3f%%）；yizi1 != yizi1_u 的 %d 行；y_open != y_open_u 的 %d 行" % (
    lab["yizi1"].sum(), 100 * lab["yizi1"].mean(), (lab["yizi1"] != lab["yizi1_u"]).sum(),
    ((lab["y_open"] != lab["y_open_u"]) & lab["y_open"].notna()).sum()))
cx = lab[(lab["code"].str[:2] == "30") & (lab["date"] < "2020-08-24")]
say("  创业板 2020-08-24 前（涨跌停 10%%，代码按 20%% 算）：行 %d，yizi1 为真 %d（按 10%% 算应多出来的一字板没被标）" % (len(cx), cx["yizi1"].sum()))
# 用不复权表按 10% 重算创业板 2020-08-24 前的一字板
c10 = u[(u["code"].str[:2] == "30") & (u["date"] < "2020-08-25")].sort_values(["code", "date"])
pc = c10.groupby("code")["close"].shift(1)
lim10 = pd.Series(ds.limit_price_arr(pc, pd.Series(10.0, index=c10.index), c10["code"]), index=c10.index)
nxt_low = c10.groupby("code")["low"].shift(-1)
nxt_lim = c10.groupby("code")["close"].transform(lambda s: s) * 1.10
yz_true = (c10.groupby("code")["low"].shift(-1) >= (c10["close"] * 1.10).round(2) - 1e-9)
yz_code = (c10.groupby("code")["low"].shift(-1) >= (c10["close"] * 1.20).round(2) - 1e-9)
say("  创业板 2020-07-01..08-21 的 (t) 行 %d：按 10%% 次日一字 %d 行，按 20%%（代码口径）%d 行 -> 漏标 %d 行" % (
    len(c10), yz_true.sum(), yz_code.sum(), (yz_true & ~yz_code).sum()))
# 新股：上市首日 / 前 5 日没涨跌停。unadj 表每只的第一行当作上市日（2020-07 后起始的票）
first = u.groupby("code")["date"].transform("min")
nb = u.groupby("code").cumcount()
newish = u[(first > "2020-07-10") & (nb <= 5)]
say("  2020-07 后上市的票 %d 只，前 6 根 K 线 %d 行；这些行里 |ret| > 板块涨跌幅+0.5%% 的 %d 行（无涨跌停期的真实波动）" % (
    newish["code"].nunique(), len(newish), (newish["ret"].abs() > newish["code"].map(lambda c: ds.limit_pct(c, "")) / 100 + 0.005).sum()))
(OUT / "results_a06.txt").write_text("\n".join(L), encoding="utf-8")
