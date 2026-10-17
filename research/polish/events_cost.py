"""
风险剔除（减持 / 解禁 / 增发）对 ret20 规则命中的代价（产品规则的诊断，两段都看，记一次 look）。

    PYTHONIOENCODING=utf-8 python research/polish/events_cost.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P


def main():
    df = P.load_pool(["date", "code", "board", "y_open", "y_up", "pool", "elig0", "lret20", "ev_reduce30", "ev_release30", "ev_placement60", "ev_any"])
    df = df[(df["date"] >= "2022-01-01") & df["elig0"]].reset_index(drop=True)
    H = pd.read_parquet(P.PL / "rules_hits10.parquet")
    base = H["base_ret20"]
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))
    for name, el in (("不剔任何事件", np.ones(len(df), bool)), ("只剔减持", ~df["ev_reduce30"].to_numpy()),
                     ("只剔解禁", ~df["ev_release30"].to_numpy()), ("只剔增发", ~df["ev_placement60"].to_numpy()),
                     ("全剔（= 基线）", ~df["ev_any"].to_numpy())):
        t = P.rank_by(df, df["lret20"].to_numpy(), keep=10, elig=el)
        h = P.hit_series(t, 10)
        d = (h - base).dropna()
        scr = d.index < "2024-06-01"
        top = t[t["rank"] <= 10].merge(df[["date", "code", "ev_reduce30", "ev_release30", "ev_placement60"]], on=["date", "code"], how="left")
        say("%-12s 筛选 %.2f%% (差 %+.2f, t %.2f)  开发 %.2f%% (差 %+.2f, t %.2f)  合并差 %+.2f (NW t %.2f)；前 10 里带减持 %.1f%% 解禁 %.1f%% 增发 %.1f%%" % (
            name, 100 * h[h.index < "2024-06-01"].mean(), 100 * d[scr].mean(), P.nw_t(d[scr]) if name != "全剔（= 基线）" else 0,
            100 * h[h.index >= "2024-06-01"].mean(), 100 * d[~scr].mean(), P.nw_t(d[~scr]) if name != "全剔（= 基线）" else 0,
            100 * d.mean(), P.nw_t(d) if name != "全剔（= 基线）" else 0,
            100 * top["ev_reduce30"].mean(), 100 * top["ev_release30"].mean(), 100 * top["ev_placement60"].mean()))
    # 带减持的票本身的命中（前 10 内）
    t = P.rank_by(df, df["lret20"].to_numpy(), keep=10)
    top = t[t["rank"] <= 10].merge(df[["date", "code", "ev_reduce30", "ev_any"]], on=["date", "code"], how="left")
    for seg, m in (("筛选", top["date"] < "2024-06-01"), ("开发", top["date"] >= "2024-06-01")):
        x = top[m]
        say("  不剔时前 10 内：带减持的票命中 %.1f%%（占 %.1f%%），不带 %.1f%%  [%s]" % (
            100 * x[x["ev_reduce30"]]["y_open"].mean(), 100 * x["ev_reduce30"].mean(), 100 * x[~x["ev_reduce30"]]["y_open"].mean(), seg))
    (P.HERE / "results_events_cost.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    P.ledger_add({"kind": "look", "exp": "events_cost", "arm": "no_event_exclusion", "note": "产品规则代价诊断，两段都看了"})


if __name__ == "__main__":
    main()
