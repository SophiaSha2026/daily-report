"""
对 arms.REGISTRY 里每个「臂 vs 对照」重算按天配对差值序列的稳健推断。

口径和 evalkit.compare 相同：每天前 k 名的标签均值，臂减对照，只取两边都有的天。
另给生产规则口径（≥97 分且前 10，k='prod'）的同一套数。

产物 research/stats/out/：
    arms.csv     每行一个（臂, 口径, 标签, 区间）：均值差、iid t（对账）、NW t 四个滞后、块自举 p / 区间、
                 n_eff、ACF 1/5/10/20 阶、当时报的 t
    acf.csv      每条差值序列的 ACF 前 25 阶
    daily_hits.parquet  每个臂每天前 10 的 y_open 命中均值（PBO 用）

    python research/stats/recompute.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms as AR   # noqa: E402
import hac as H     # noqa: E402

OUT = HERE / "out"
NO_SCALE = {"bag5", "rank", "rank_f35", "tail", "tail_blend", "tail_g12", "tail_g12_blend", "hist_all",
            "y20", "y20_blend", "lr_blend"}   # 日志里写明「刻度无意义」的臂，生产规则口径只作参考


def diff_series(row: dict, arm: str, ctrl: str, k, y: str, span: str) -> pd.Series:
    a = AR.span_filter(AR.top_of(row, arm), span)
    b = AR.span_filter(AR.top_of(row, ctrl), span)
    return (AR.daily_hit(a, k, y) - AR.daily_hit(b, k, y)).dropna().sort_index()


def month_t(d: pd.Series) -> tuple[float, int, int]:
    """evalkit 的 t_month 近似：按月求差值均值（这里用按天差值的月均，evalkit 用月合并命中之差）。"""
    md = d.groupby(d.index.str[:7]).mean()
    if len(md) < 3:
        return float("nan"), len(md), int((md > 0).sum())
    return float(md.mean() / (md.std(ddof=1) / np.sqrt(len(md)))), len(md), int((md > 0).sum())


def one(row: dict, k, y: str, span: str, tag: str) -> dict:
    d = diff_series(row, row["arm"], row["ctrl"], k, y, span)
    r = H.full(d.to_numpy(), f"{row['arm']}|{k}|{y}|{span}")
    tm, nm, wins = month_t(d)
    r.update(exp=row["exp"], fam=row["fam"], arm=row["arm"], ctrl=row["ctrl"], k=str(k), y=y, span=span,
             tag=tag, t_rep=row["t_rep"] if tag == "reported" else np.nan, verdict=row["verdict"],
             t_month=tm, months=nm, wins=wins, note=row["note"],
             no_scale=(k == "prod" and row["arm"] in NO_SCALE))
    r["p_nw20"] = H.norm_p(r["nw20_t"])
    r["p_iid"] = H.norm_p(r["iid_t"])
    return r


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = []
    seen = {(r["arm"], str(r["k"]), r["y"], r["span"]) for r in AR.REGISTRY}
    for row in AR.REGISTRY:
        # 当时报的那一口径
        k, y, span = row["k"], row["y"], row["span"]
        rep = one(row, k, y, span, "reported")
        rows.append(rep)
        print("%-3s %-14s %-5s %-6s %-7s 报 %5.2f  重算 iid %5.2f  NW20 %5.2f  块自举 p %.3f" % (
            row["exp"], row["arm"], k, y, span, row["t_rep"], rep["iid_t"], rep["nw20_t"], rep["mbb_p"]),
            flush=True)
        extras = []
        if row["fam"] == "A" and y == "y_up":
            extras.append((10, "y_open", span))
        if row["fam"] == "B" and span == "dev":
            extras.append((10, "y_open", "since25"))
        if row["fam"] in ("A", "B") and k != "prod":
            extras.append(("prod", "y_open", span))
        if row["fam"] == "C":
            extras += [(3, "y_cont", span), (10, "y_cont", span)]
        for k2, y2, s2 in extras:
            key = (row["arm"], str(k2), y2, s2)
            if key in seen:
                continue
            seen.add(key)
            rows.append(one(row, k2, y2, s2, "extra"))
    df = pd.DataFrame(rows)
    acf = pd.DataFrame([dict(label=r["label"], **{f"lag{i + 1}": v for i, v in enumerate(r["acf"])}) for r in rows])
    df.drop(columns=["acf"]).to_csv(OUT / "arms.csv", index=False, encoding="utf-8-sig")
    acf.to_csv(OUT / "acf.csv", index=False, encoding="utf-8-sig")

    # PBO 用的每日命中矩阵（前 10，y_open），A 族 + B 族全部臂（含对照）
    hits = {}
    for row in AR.REGISTRY:
        if row["fam"] not in ("A", "B"):
            continue
        for arm in (row["arm"], row["ctrl"]):
            key = f"{row['exp'] if row['fam'] == 'A' else 'W'}:{arm}"
            if key in hits:
                continue
            hits[key] = AR.daily_hit(AR.top_of(row, arm), 10, "y_open")
    hm = pd.DataFrame(hits).sort_index()
    hm.to_parquet(OUT / "daily_hits.parquet")
    (OUT / "registry.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in AR.REGISTRY], ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
