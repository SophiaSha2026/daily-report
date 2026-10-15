"""
特征深度研究（research/features）的公用件：路径、尺子、登记簿。

尺子（research/stats 新过线标准）：
    主：每天前 10 买得到口径（y_open）按天配对差
    NW t（Bartlett，滞后 20）、移动块自举 p（块长 20）、去掉最好一个月的 NW t、
    合并差（按名额合并命中之差）和按天差同号、最差三个月（含 / 去掉 2024-09 两种都报）、空榜不翻倍

登记簿 research/features/ledger.jsonl：本轮每一个臂（含失败的、事后加看的）一行。
只读开发集 2024-06..2025-12，确认集一行不读。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "stats", ROOT / "research" / "altpath"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evalkit as K  # noqa: E402
import hac  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
FE = K.EXP / "features"            # 本轮新特征缓存
FE.mkdir(parents=True, exist_ok=True)
NW_LAG = 20
M924 = "2024-09"


def ledger(exp: str, arm: str, kind: str, **kw) -> None:
    """kind: prereg（预登记的臂）/ posthoc（事后加的臂或子区间）/ diag（诊断，不训模型）/ result。"""
    row = {"at": dt.datetime.now().isoformat(timespec="seconds"), "exp": exp, "arm": arm, "kind": kind, **kw}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]


def n_trials() -> int:
    """本轮试验次数：prereg + posthoc 的臂（result / diag 行不另计）。"""
    seen = set()
    for r in ledger_rows():
        if r["kind"] in ("prereg", "posthoc"):
            seen.add((r["exp"], r["arm"], r.get("span", "")))
    return len(seen)


def t_bonf(m: int, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / max(m, 1)))


def top(exp: str, arm: str) -> pd.DataFrame:
    t = pd.read_parquet(K.EXP / exp / f"top_{arm}.parquet")
    assert t["date"].max() < K.CONFIRM[0], "确认集漏进来了"
    return t


def daily_hit(t: pd.DataFrame, k: int = 10, y: str = "y_open") -> pd.Series:
    return t[t["rank"] <= k].groupby("date")[y].mean()


def daily_diff(a, b, k=10, y="y_open", drop_months=()) -> pd.Series:
    d = (daily_hit(a, k, y) - daily_hit(b, k, y)).dropna().sort_index()
    if drop_months:
        d = d[~d.index.str[:7].isin(set(drop_months))]
    return d


def nwt(d) -> float:
    return float(hac.nw_t(np.asarray(d, float), NW_LAG)["t"])


def full_cmp(a: pd.DataFrame, b: pd.DataFrame, k: int = 10, y: str = "y_open") -> dict:
    """a 对 b（对照）。新过线标准需要的全部量。"""
    d = daily_diff(a, b, k, y)
    ma = a[a["rank"] <= k].groupby(a["date"].str[:7])[y].mean()
    mb = b[b["rank"] <= k].groupby(b["date"].str[:7])[y].mean()
    md = (ma - mb).dropna()
    best = md.idxmax()
    d_nobest = d[d.index.str[:7] != best]
    bb = hac.mbb(d.to_numpy())
    pa, pb = K.clustered(a[a["rank"] <= k], y), K.clustered(b[b["rank"] <= k], y)
    w3 = md.sort_values().head(3)
    w3x = md.drop(M924, errors="ignore").sort_values().head(3)
    ea = a["date"].nunique() - K.prod_picks(a)["date"].nunique() if "score" in a else 0
    eb = b["date"].nunique() - K.prod_picks(b)["date"].nunique() if "score" in b else 0
    d_x = d[d.index.str[:7] != M924]
    pra, prb = K.clustered(K.prod_picks(a), y), K.clustered(K.prod_picks(b), y)
    return {
        "hit_a": pa["hit"], "hit_b": pb["hit"], "diff_day": float(d.mean()), "pooled_diff": pa["hit"] - pb["hit"],
        "t_iid": float(hac.iid_t(d.to_numpy())["t"]), "t_nw": nwt(d), "t_nw10": float(hac.nw_t(d.to_numpy(), 10)["t"]),
        "t_nw30": float(hac.nw_t(d.to_numpy(), 30)["t"]),
        "boot_p": bb["p"], "boot_lo": bb["lo"], "boot_hi": bb["hi"],
        "best_month": best, "t_nw_drop_best": nwt(d_nobest),
        "same_sign": bool(np.sign(d.mean()) == np.sign(pa["hit"] - pb["hit"])),
        "wins": int((md > 0).sum()), "months": int(len(md)),
        "worst3": {k_: float(v) for k_, v in w3.items()},
        "worst3_ex924": {k_: float(v) for k_, v in w3x.items()},
        "worst3_ok": bool(w3.min() >= 0), "worst3_ex924_ok": bool(w3x.min() >= 0),
        "empty_a": int(ea), "empty_b": int(eb), "empty_ok": bool(ea <= 2 * max(eb, 1)),
        "diff_ex924": float(d_x.mean()), "t_nw_ex924": nwt(d_x),
        "diff_2025": float(d[d.index >= "2025-01"].mean()), "t_nw_2025": nwt(d[d.index >= "2025-01"]),
        "prod_a": pra["hit"], "prod_b": prb["hit"], "prod_na": pra["n"], "prod_nb": prb["n"],
        "monthly_diff": {k_: float(v) for k_, v in md.items()},
    }


def verdict(c: dict, thr: float) -> str:
    """新标准：NW t >= max(2.5, thr)、块自举 p<0.05、去最好月 t>=2、同号、空榜不翻倍；
    最差三月两种口径都报，不替用户选 -> 分两个判定。"""
    core = (c["t_nw"] >= max(2.5, thr) and c["boot_p"] < 0.05 and c["t_nw_drop_best"] >= 2.0
            and c["same_sign"] and c["empty_ok"])
    if c["t_nw"] <= -max(2.5, thr):
        return "不过线"
    if not core:
        return "未定"
    a = "过线" if c["worst3_ok"] else "未定(最差三月)"
    b = "过线" if c["worst3_ex924_ok"] else "未定(最差三月)"
    return f"含924:{a} / 去924:{b}"


def fmt(name: str, c: dict, ref: str = "base") -> str:
    return ("%-14s 前10 %5.2f%% 对 %s %+.2fpp 合并 %+.2f 按天t %.2f NW t %.2f (10/30: %.2f/%.2f) 块自举 p %.3f "
            "去最好月(%s) t %.2f 赢 %d/%d 最差三月 %s | 去924最差三月 %s | 去924 %+.2f t %.2f | 2025起 %+.2f t %.2f "
            "| 空榜 %d/%d 生产规则 %.2f%%/%.2f%% (n %d/%d)" % (
                name, 100 * c["hit_a"], ref, 100 * c["diff_day"], 100 * c["pooled_diff"], c["t_iid"], c["t_nw"],
                c["t_nw10"], c["t_nw30"], c["boot_p"], c["best_month"], c["t_nw_drop_best"], c["wins"], c["months"],
                ",".join("%+.1f" % (100 * v) for v in c["worst3"].values()),
                ",".join("%+.1f" % (100 * v) for v in c["worst3_ex924"].values()),
                100 * c["diff_ex924"], c["t_nw_ex924"], 100 * c["diff_2025"], c["t_nw_2025"],
                c["empty_a"], c["empty_b"], 100 * c["prod_a"], 100 * c["prod_b"], c["prod_na"], c["prod_nb"]))


def monthly_row(t: pd.DataFrame, k: int = 10) -> pd.Series:
    return t[t["rank"] <= k].groupby(t["date"].str[:7])["y_open"].mean()
