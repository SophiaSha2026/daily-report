"""前向参考（只看一次，不进判定）：2026-09-01 之后、本机日线上标签窗口已关的天，名次基线 B(k) 的精确率，
外加三个筛选集上精确率最高、但没过功效门槛的配置（调过的，只作方向参考）。

    PYTHONIOENCODING=utf-8 python research/precision/forward_pr.py

数据和口径照 research/polish/forward.py（daily.parquet + feat_fwd.ext_features + labels_partial + 事件剔除）；2026-01..08 的行算完特征立即丢弃。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_pr as C

P = C.P
START = "2026-09-01"


def main():
    P.wait_prod_window()
    import forward as FW
    import feat_fwd as F
    import backfill as B
    import validate as V
    import common_rc as RC
    daily = pd.read_parquet(P.ROOT / "data" / "breakout" / "daily.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str).str[:10]
    daily = B.clean_bars(daily)
    f = F.ext_features(daily)
    f = f[f["date"] >= START].reset_index(drop=True)
    st = V.st_codes()
    f["elig0"] = (~f["code"].isin(st) & (f["close"] >= 2.0) & (f["nbar"] >= 120) & ~f["susp20"]
                  & np.isfinite(f[["lret20", "vol60", "turn20", "amp20"]]).all(axis=1))
    RC.EVENTS = P.EXP / "rulecore" / "events_c"
    ev = RC.event_flags(f)
    f["pool"] = f["elig0"].to_numpy() & ~ev["any"].to_numpy()
    f = f[f["pool"]].reset_index(drop=True)
    lab = FW.labels_partial(daily[daily["date"] >= "2026-06-01"][["code", "date", "open", "high", "low", "close"]])
    f = f.merge(lab, on=["code", "date"], how="left")
    f["closed"] = f["n_avail"] >= 20
    f["rank_ret20"] = f.groupby("date")["lret20"].rank(ascending=False, method="first")
    top = f[f["rank_ret20"] <= 50].copy()
    cl = top[top["closed"]]
    days = sorted(cl["date"].unique())
    L = ["前向 %s..%s（窗口已关 %d 天：%s..%s；候选池 %.0f 只/天，closed 天候选池基准 %.2f%%）" % (
        f["date"].min(), f["date"].max(), len(days), days[0] if days else "-", days[-1] if days else "-",
        f.groupby("date").size().mean(), 100 * f[f["closed"]]["hit_sofar"].mean())]
    L.append("  名次基线 B(k)：")
    for k in (1, 2, 3, 5, 10):
        s = cl[cl["rank_ret20"] <= k]
        s2 = top[top["rank_ret20"] <= k]
        L.append("    k=%2d  closed 命中 %d/%d = %5.1f%%   sofar 全部 %d 天 %d/%d = %5.1f%%（下界）" % (
            k, int(s["hit_sofar"].sum()), len(s), 100 * s["hit_sofar"].mean(), s2["date"].nunique(), int(s2["hit_sofar"].sum()), len(s2), 100 * s2["hit_sofar"].mean()))
    L.append("  筛选集上精确率最高但没过功效的配置（调过的，只作方向）：")
    b = top["board"] if "board" in top.columns else RC.board_of(top["code"])
    top["board"] = b.to_numpy()
    rules = {"r2 & nonmain & mcap<50亿": (top["rank_ret20"] <= 2) & (top["board"] != "main") & (top["mcap"] < 5e9),
             "r1 & nld20==0": (top["rank_ret20"] <= 1) & (top["nld20"] == 0),
             "r5 & nonmain & mcap<50亿": (top["rank_ret20"] <= 5) & (top["board"] != "main") & (top["mcap"] < 5e9),
             "r2 & nonmain": (top["rank_ret20"] <= 2) & (top["board"] != "main")}
    for nm, m in rules.items():
        s = top[m & top["closed"]]
        s2 = top[m]
        L.append("    %-28s closed %d 天 %d/%d = %5.1f%%   sofar %d/%d = %5.1f%%" % (
            nm, s["date"].nunique(), int(s["hit_sofar"].sum()), len(s), 100 * s["hit_sofar"].mean() if len(s) else np.nan,
            int(s2["hit_sofar"].sum()), len(s2), 100 * s2["hit_sofar"].mean() if len(s2) else np.nan))
    txt = "\n".join(L)
    C.say(txt)
    (C.HERE / "results_forward.txt").write_text(txt + "\n", encoding="utf-8")
    C.ledger_add({"kind": "forward_look", "arms": ["B(1)", "B(2)", "B(3)", "B(5)", "B(10)"] + list(rules)})
    C.progress("前向参考 results_forward.txt（只看一次）：窗口已关 %d 天" % len(days))


if __name__ == "__main__":
    main()
