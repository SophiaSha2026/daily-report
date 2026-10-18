"""a01：三张价格表的结构事实（行数、只数、日期、重复、空洞、逐日只数、股本/换手单位）。只读。"""
import sys
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


def main():
    d = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
    d["code"] = d["code"].astype(str).str.zfill(6)
    say("== daily.parquet ==")
    say("行", len(d), "只", d["code"].nunique(), "日期", d["date"].min(), "..", d["date"].max())
    dup = d.duplicated(["code", "date"]).sum()
    say("重复 (code,date)", dup)
    say("缺值", d.isna().sum().to_dict())
    say("价 <=0 行", int(((d[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()),
        "high<low", int((d["high"] < d["low"]).sum()), "volume<=0", int((d["volume"] <= 0).sum()),
        "amount<=0", int((d["amount"] <= 0).sum()))
    say("close 超出 [low,high] 行", int(((d["close"] > d["high"] + 1e-9) | (d["close"] < d["low"] - 1e-9)).sum()),
        "open 超出", int(((d["open"] > d["high"] + 1e-9) | (d["open"] < d["low"] - 1e-9)).sum()))
    cnt = d.groupby("date").size()
    say("逐日只数：首 5", cnt.head(5).to_dict(), "末 8", cnt.tail(8).to_dict())
    say("逐日只数分位", cnt.quantile([0, .01, .1, .5, .9, 1]).to_dict())
    # 换手率单位：volume / outstanding_share vs turnover
    r = d["volume"] / d["outstanding_share"].replace(0, np.nan)
    ratio = (d["turnover"] / r).replace([np.inf, -np.inf], np.nan)
    say("turnover / (volume/os) 分位", ratio.quantile([.01, .1, .5, .9, .99]).round(4).to_dict())
    say("turnover 分位", d["turnover"].quantile([.01, .5, .99, .9999]).round(5).to_dict())
    # amount / (close*volume) 单位核
    av = d["amount"] / (d["close"] * d["volume"])
    say("amount/(close*volume) 分位", av.quantile([.001, .01, .5, .99, .999]).round(3).to_dict())
    bad_av = d[(av < 0.5) | (av > 2.0)]
    say("amount/(close*volume) 超出 [0.5,2] 行", len(bad_av), "只", bad_av["code"].nunique(),
        "按板块", bad_av["code"].str[:2].value_counts().head(6).to_dict(),
        "按日期 top", bad_av["date"].value_counts().head(5).to_dict())
    # 小数位：快照行 turnover 是 raw[38]/100 两位小数百分比 -> 1e-4 的整数倍
    t4 = (d["turnover"] * 1e4)
    is_snap_like = (np.abs(t4 - np.round(t4)) < 1e-6) & d["turnover"].notna()
    by_date = is_snap_like.groupby(d["date"]).mean()
    say("turnover 恰为 0.01%% 整数倍（像快照行）的比例：全表 %.4f；按日期 >0.5 的天数 %d，例 %s" % (
        is_snap_like.mean(), int((by_date > 0.5).sum()), by_date[by_date > 0.5].tail(12).round(3).to_dict()))
    # 价格小数位（前复权后新浪给两位小数？）
    for c in ("open", "high", "low", "close"):
        x = d[c] * 100
        say(c, "两位小数之外的行", int((np.abs(x - np.round(x)) > 1e-6).sum()))
    # 每只的起止、行数、最大缺口（按全市场交易日序号）
    dates = np.array(sorted(d["date"].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    d = d.sort_values(["code", "date"])
    d["_di"] = d["date"].map(di).to_numpy()
    gap = d["_di"] - d.groupby("code")["_di"].shift(1) - 1
    say("相邻行之间缺交易日 >=1 的次数", int((gap >= 1).sum()), ">=5", int((gap >= 5).sum()), ">=20", int((gap >= 20).sum()),
        "涉及只数(>=5)", d.loc[gap >= 5, "code"].nunique())
    per = d.groupby("code").agg(n=("date", "size"), d0=("date", "min"), d1=("date", "max"))
    say("每只行数分位", per["n"].quantile([0, .01, .1, .5, 1]).to_dict())
    say("最后一行不是全表最后一天的只数", int((per["d1"] < dates[-1]).sum()), "例", per[per["d1"] < dates[-1]].tail(5).to_dict("index"))
    say("第一行晚于 2023-01-04 的只数", int((per["d0"] > "2023-01-04").sum()), "晚于 2025-01-01", int((per["d0"] > "2025-01-01").sum()))
    per.to_csv(OUT / "raw/daily_per_code.csv")
    pd.DataFrame({"date": cnt.index, "n": cnt.values}).to_csv(OUT / "raw/daily_per_date.csv", index=False)
    # 涨跌停：相邻行 close/prev_close 极端
    pc = d.groupby("code")["close"].shift(1)
    ret = d["close"] / pc - 1
    say("单日涨跌幅分位", ret.quantile([0, .0001, .5, .9999, 1]).round(4).to_dict())
    say("|涨跌| > 0.45 的行", int((ret.abs() > 0.45).sum()), "例", d.loc[ret.abs() > 0.45, ["code", "date", "close"]].head(8).to_dict("records"))

    say("== unadj_2020 / px_screen ==")
    u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet")
    say("unadj_2020 列", u.columns.tolist(), "行", len(u), "只", u["code"].nunique(), u["date"].min(), "..", u["date"].max())
    say("unadj_2020 重复", int(u.duplicated(["code", "date"]).sum()), "缺值", u.isna().sum().to_dict())
    p = pd.read_parquet(ROOT / "data/breakout/raw/exp/screen/px_screen.parquet")
    say("px_screen 行", len(p), "只", p["code"].nunique(), p["date"].min(), "..", p["date"].max())
    say("px_screen q 分位", p["q"].quantile([0, .01, .5, .99, 1]).round(4).to_dict(), "q<1 行", int((p["q"] < 1 - 1e-9).sum()))
    cntp = p.groupby("date").size()
    say("px_screen 逐日只数：2020-07-01 %d, 2021-12-31 %d, 2022-12-30 %d, 2023-12-29 %d, 2025-12-31 %d" % tuple(
        cntp.get(x, 0) for x in ("2020-07-01", "2021-12-31", "2022-12-30", "2023-12-29", "2025-12-31")))
    (OUT / "results_a01.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
