"""
「异常波动 / 风险提示」类公告是交易所按价格规则触发的。它们的信息是不是全在日线里？
用日线自己算一个代理：3 日累计涨幅 r3 = close_t / close_{t-3} - 1 达到板块阈值
（主板 20%、创业 / 科创 30%、北交 40%，交易所偏离值规则的近似，不扣指数），
和 10 日累计涨幅 r10 >= 100%（严重异常波动的近似）。
分解：代理∩公告、只有代理、只有公告；再看这些票在模型 200 名外的命中。
口径同 analyze.py（ei 口径，k=1）。写 cache/price_proxy.csv。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

ROOT = HERE.parents[1]
TH3 = {"主板": 0.20, "创业": 0.30, "科创": 0.30, "北交": 0.40}


def main() -> None:
    P, E = A.load()
    P = P.reset_index(drop=True)
    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet", columns=["code", "date", "close"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = px["date"].astype(str)
    px = px[(px["date"] >= "2024-04-01") & (px["date"] <= "2025-12-31")].sort_values(["code", "date"])
    g = px.groupby("code")["close"]
    px["r3"] = px["close"] / g.shift(3) - 1
    px["r10"] = px["close"] / g.shift(10) - 1
    P = P.merge(px[["code", "date", "r3", "r10"]], on=["code", "date"], how="left")
    th = P["board"].map(TH3).fillna(0.2)
    proxy = (P["r3"] >= th).to_numpy()
    proxy10 = (P["r10"] >= 1.0).to_numpy()
    base_d = P.groupby("date")["y_open"].mean()
    base_out = P[P["rank_all"] > 200].groupby("date")["y_open"].mean()
    nh = float(P["y_open"].sum())
    rows = []

    def add(name, fl):
        s = A.stats(P, fl, base_d, nh)
        m = A.model_rel(P, fl, base_d, base_out)
        rows.append({"组": name, **s, **{k: m.get(k) for k in ("top200_share", "out200_n", "out200_hit_day",
                                                                 "out200_diff", "out200_t_nw", "top10_n",
                                                                 "top10f_diff", "top10f_t_nw")}})

    for cat, px_flag, pname in (("异常波动", proxy, "3 日涨幅过阈值"), ("风险提示", proxy, "3 日涨幅过阈值"),
                                ("严重异常波动", proxy10, "10 日涨幅 >= 100%")):
        ann = A.dist_col(P, E[E["cat"] == cat], "avail_ei") <= 1
        add(f"{cat}（公告，k=1）", ann)
        add(f"{pname}（日线代理，当天）", px_flag)
        add(f"{cat} ∩ {pname}", ann & px_flag)
        add(f"只有 {pname}，近 2 日无 {cat}", px_flag & ~ann)
        add(f"只有 {cat}，{pname} 不成立", ann & ~px_flag)
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "cache" / "price_proxy.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = R[["组", "n", "days", "hit_day", "base_day", "ratio", "diff", "t_nw", "yizi1", "top200_share",
              "out200_n", "out200_hit_day", "out200_diff", "out200_t_nw"]].copy()
    for c in ("hit_day", "base_day", "diff", "yizi1", "top200_share", "out200_hit_day", "out200_diff"):
        show[c] = (100 * show[c]).round(2)
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
