"""
候选池口径的两件事：
  a 生产对齐：生产 build.MIN_HIST=120 根预热 + daily.MIN_HISTORY_DAYS=120 行 = 上市约 240 根才能上榜；
    研究候选池只要 120 根。ret20 规则在「>= 240 根」池上的两段成绩（生产口径核对，不是臂）
  b 放宽到 >= 60 根（只印筛选集；开发集算了不印）

    PYTHONIOENCODING=utf-8 python research/polish/pool_variants.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pl as P


def main():
    df = P.load_pool(["date", "code", "board", "y_open", "y_up", "pool", "elig0", "lret20", "nbar", "ev_any", "close", "susp20"])
    df = df[(df["date"] >= "2022-01-01")].reset_index(drop=True)
    H = pd.read_parquet(P.PL / "rules_hits10.parquet")
    H30 = pd.read_parquet(P.PL / "rules_hits30.parquet")
    base, base30 = H["base_ret20"], H30["base_ret20"]
    L = []
    say = lambda s: (L.append(s), print(s, flush=True))
    for name, el, show_dev in (("nbar>=240（生产口径）", df["pool"] & (df["nbar"] >= 240), True),
                               ("nbar>=60（放宽）", df["elig0"].to_numpy() & ~df["ev_any"].to_numpy() | (df["pool"].to_numpy()), False)):
        if name.startswith("nbar>=60"):
            # elig0 已含 nbar>=120；放宽要回到 feat.parquet 之外的行，这里的 pool 表没有 <120 根的行，只能标注做不到
            say("%s：pool.parquet 只含 nbar>=120 的行（feat.parquet elig0 口径），放宽需要重建，先跳过" % name)
            continue
        t = P.rank_by(df, df["lret20"].to_numpy(), keep=30, elig=el.to_numpy())
        h, h30 = P.hit_series(t, 10), P.hit_series(t, 30)
        H[name], H30[name] = h.reindex(H.index), h30.reindex(H30.index)
        d, d30 = (h - base).dropna(), (h30 - base30).dropna()
        scr = d.index < "2024-06-01"
        say("%s 筛选集：命中 %.2f%% 对 base %.2f%%，差 %+.2fpp NW t %.2f；前 30 差 %+.2f；每天候选 %.0f 只（base %.0f）" % (
            name, 100 * h[h.index < "2024-06-01"].mean(), 100 * base[scr].mean(), 100 * d[scr].mean(), P.nw_t(d[scr]),
            100 * d30[scr].mean(), el[df["date"] < "2024-06-01"].groupby(df["date"]).sum().mean(),
            df.loc[df["date"] < "2024-06-01", "pool"].groupby(df["date"]).sum().mean()))
        if show_dev:
            say("%s 开发集：命中 %.2f%% 对 base %.2f%%，差 %+.2fpp NW t %.2f；前 30 差 %+.2f" % (
                name, 100 * h[h.index >= "2024-06-01"].mean(), 100 * base[~scr].mean(), 100 * d[~scr].mean(), P.nw_t(d[~scr]), 100 * d30[~scr].mean()))
            j = P.judge(d, d30, name)
            say("  " + P.fmt_judge(j))
            P.ledger_add({"kind": "look", "exp": "pool_variants", "arm": name, "note": "生产口径核对，两段都看了", "diff": j["diff"], "t": j["t"]})
    H.to_parquet(P.PL / "rules_hits10.parquet")
    H30.to_parquet(P.PL / "rules_hits30.parquet")
    (P.HERE / "results_pool_variants.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
