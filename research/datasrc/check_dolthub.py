"""抽检：chenditc/investment_data（DoltHub）的不复权日线 vs 本机 data/breakout/daily.parquet（只读）。

本机日线是新浪前复权，最近一次除权之后的价格和不复权一致，所以抽最近 15 个交易日比收盘价；
成交量 / 成交额两边都是原始量，任何日子都能比（单位先实测再换算）。
随机抽 60 个（代码, 日期），代码从本机日线表取（教训 2）。结果写 survey/dolthub_check.csv。
"""
import random
import time

import pandas as pd
import requests

import common

API = "https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master"


def q(sql):
    r = requests.get(API, params={"q": sql}, headers={"User-Agent": common.UA}, timeout=90)
    j = r.json()
    if j.get("query_execution_status") != "Success":
        raise RuntimeError(j.get("query_execution_message"))
    return pd.DataFrame(j["rows"])


def main():
    random.seed(20261015)
    d = pd.read_parquet(common.REPO / "data/breakout/daily.parquet",
                        columns=["date", "code", "close", "volume", "amount"])
    days = sorted(d["date"].unique())
    recent = d[d["date"].isin(days[-15:])]
    old = d[d["date"].isin(days[-500:-400])]
    pick = pd.concat([recent.sample(40, random_state=1), old.sample(20, random_state=2)])
    out = []
    for _, r in pick.iterrows():
        sym = common.market(r["code"]) + r["code"]
        try:
            t = q(f"select * from final_a_stock_eod_price where symbol='{sym}' and tradedate='{r['date']}'")
        except Exception as e:  # noqa: BLE001
            out.append({**r.to_dict(), "err": repr(e)[:100]})
            continue
        time.sleep(0.5)
        rec = {**r.to_dict(), "recent": r["date"] >= days[-15]}
        if len(t):
            x = t.iloc[0]
            rec.update(d_close=float(x["close"]), d_volume=float(x["volume"]), d_amount=float(x["amount"]),
                       d_adjclose=float(x["adjclose"]))
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(common.ROOT / "survey" / "dolthub_check.csv", index=False, encoding="utf-8-sig")
    m = df.dropna(subset=["d_close"])
    m = m.assign(close_diff=(m["d_close"] / m["close"] - 1).abs(),
                 vol_ratio=m["d_volume"] / m["volume"], amt_ratio=m["d_amount"] / m["amount"])
    print("抽", len(df), "找到", len(m))
    print("最近 15 天收盘相对差 ≤0.1%:", (m[m["recent"]]["close_diff"] <= 0.001).mean(), "n=", m["recent"].sum())
    print("成交量比 中位数", m["vol_ratio"].median(), "分位", m["vol_ratio"].quantile([.05, .95]).tolist())
    print("成交额比 中位数", m["amt_ratio"].median(), "分位", m["amt_ratio"].quantile([.05, .95]).tolist())
    print(m.sort_values("close_diff").tail(5)[["code", "date", "close", "d_close", "vol_ratio", "amt_ratio"]])


if __name__ == "__main__":
    main()
