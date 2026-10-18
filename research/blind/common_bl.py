"""
盲测近期 + 底层杠杆（research/blind，2026-10-18）公用件。条文见 criteria.txt。

数据：research/gate/table.parquet（候选池全部行，2020-12-23..2025-12-31，行序固定）+ research/gate/scores/*.npy（同序的样本外分数）。
尺子：common_pr（Baseline / evaluate / boot_ratio）+ common_gt（eval_gate / combined_judge），B(10) 两段 11.34 / 16.54。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "research" / "gate", ROOT / "research" / "precision", ROOT / "research" / "polish", ROOT / "research" / "rulecore",
          ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common_gt as G  # noqa: E402
import common_pr as PR  # noqa: E402,F401  (v1_bl / v3_bl / eval_scores 经 B.PR 用)
import common_pl as PL  # noqa: E402

__all__ = ["G", "PR", "PL"]   # v1_bl / v3_bl / eval_scores 经 B.PR / B.G 用

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
SCORES.mkdir(exist_ok=True)
LEDGER = HERE / "ledger.jsonl"
SEGS = PL.SEGS
FAMILY = "breakout_blind"
EXPNAME = "blind_v1"
say = PL.say
NS = (1, 2, 3, 5, 10)
KS = (10, 20, 50)


def progress(text: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def ledger_add(row: dict) -> None:
    from bjtime import now_bj
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False, default=float) + "\n")


def ledger_count(kind: str | None = None) -> int:
    if not LEDGER.exists():
        return 0
    n = 0
    for ln in LEDGER.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if kind is None or r.get("kind") == kind:
                n += 1
    return n


# ---------------------------------------------------------------
#  基表：候选池行的键 + 名次 + 标签 + 交易日序号 + 「T−20 那一行」的指针
# ---------------------------------------------------------------
KEY_COLS = ["code", "date", "board", "y_open", "r20_open", "end20", "rank_ret20", "ret20", "seg", "hit", "nbar"]


def load_keys(extra: list[str] | None = None) -> pd.DataFrame:
    t = G.load_table(KEY_COLS + (extra or []))
    t["code"] = t["code"].astype(str).str.zfill(6)
    return t


def day_index(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    u = np.array(sorted(pd.unique(dates)))
    di = pd.Series(np.arange(len(u)), index=u)
    return dates.map(di).to_numpy(), u


def lag_pointer(t: pd.DataFrame, lag: int = 20) -> np.ndarray:
    """每行指向同一只票「往前第 lag 根 K 线」那一行在 t 里的行号；那一行不在表里（不在候选池）就 −1。
    用 nbar（该票累计根数，px 全表上数的）做键：nbar_T − lag == nbar_{T−lag}。"""
    key = pd.DataFrame({"code": t["code"].to_numpy(), "nb": t["nbar"].to_numpy(np.int64)})
    key["_i"] = np.arange(len(t))
    want = pd.DataFrame({"code": t["code"].to_numpy(), "nb": t["nbar"].to_numpy(np.int64) - lag})
    m = want.merge(key, on=["code", "nb"], how="left")
    assert len(m) == len(t)
    return m["_i"].fillna(-1).to_numpy(np.int64)


def load_scores(name: str) -> np.ndarray:
    p = G.SCORES / (name + ".npy")
    if not p.exists():
        p = SCORES / (name + ".npy")
    return np.load(p)


def day_pct(t: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    """分数在当天候选池内的分位（0~1，越大越好），NaN 分数 -> NaN。"""
    ser = pd.Series(s, index=t.index)
    return ser.groupby(t["date"].to_numpy()).rank(pct=True).to_numpy()


def topn_by(t: pd.DataFrame, cand: np.ndarray, score: np.ndarray, n: int) -> np.ndarray:
    """cand 行里按 score 降序每天取前 n，返回布尔。score NaN 的行不取。"""
    sel = np.zeros(len(t), bool)
    m = cand & np.isfinite(score)
    if not m.any():
        return sel
    idx = np.flatnonzero(m)
    sub = pd.DataFrame({"date": t["date"].to_numpy()[idx], "s": score[idx], "i": idx})
    sub = sub.sort_values(["date", "s"], ascending=[True, False], kind="mergesort")
    sub["r"] = sub.groupby("date").cumcount() + 1
    sel[sub.loc[sub["r"] <= n, "i"].to_numpy()] = True
    return sel


# ---------------------------------------------------------------
#  评估：两段 + 合并（gate 同一把尺子）
# ---------------------------------------------------------------
def eval_both(t: pd.DataFrame, sel: np.ndarray, boot: bool = True) -> dict:
    out = {}
    for seg in ("screen", "dev"):
        out[seg] = G.eval_gate(t, sel, seg, boot=boot)
    if out["screen"]["V"] > 0 and out["dev"]["V"] > 0:
        out["comb"] = G.combined_judge(out["screen"], out["dev"])
    return out


def fmt_both(name: str, r: dict, dev: bool = True) -> str:
    lines = [G.fmt(name, r["screen"], "screen")]
    if dev:
        lines.append(G.fmt(name, r["dev"], "dev"))
        if "comb" in r:
            c = r["comb"]
            lines.append("%-26s 合并 ΔP %+.2fpp NW t %.2f 块自举 p %.3f 去%s t %.2f 滚3 %+.1f@%s" % (
                "", 100 * c["dP"], c["t"], c["boot_p"], c["best_month"], c["t_drop_best"], c["roll3"][0], c["roll3"][1]))
    return "\n".join(lines)


def verdict(r: dict, tau: float | None = None) -> str:
    s, d = r["screen"], r["dev"]
    if s["V"] <= 0 or d["V"] <= 0:
        return "整段空榜"
    a = s["V"] >= 150 and s["days_out"] >= 60 and d["V"] >= 150 and d["days_out"] >= 60
    b = np.isfinite(s.get("P_lo", np.nan)) and s["P_lo"] >= s["B10"] and d["P_lo"] >= d["B10"]
    c = s["dP"] >= 0 and d["dP"] >= 0 and r["comb"]["t"] >= 2.5 and r["comb"]["boot_p"] < 0.05
    dd = True if tau is None else (s["P"] >= 0.8 * tau and d["P"] >= 0.8 * tau)
    if a and b and c and dd:
        return "可用"
    if r["comb"]["t"] <= -2.5:
        return "比名次差"
    if a and b and not c:
        return "量换精度"
    if c and not a:
        return "方向"
    return "不过线"


def screen_row(name: str, e: dict) -> dict:
    return {"name": name, "V": e["V"], "days": e["days_out"], "vbar": round(e["vbar"], 3), "P": round(100 * e["P"], 2) if e["V"] else None,
            "B": round(100 * e["P_base"], 2) if e["V"] else None, "dP": round(100 * e["dP"], 2) if e["V"] else None,
            "t": round(e["t"], 2) if e["V"] else None}


def pick_best(rows: list[dict], vmin: int = 150, dmin: int = 60) -> dict | None:
    ok = [r for r in rows if r["V"] >= vmin and r["days"] >= dmin and r["dP"] is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: r["dP"])


def dev_shuffle_within_day(t: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """hit / y_open 每天内随机置换（自检用）。"""
    rng = np.random.default_rng(seed)
    o = t.copy()
    g = o.groupby("date", sort=False).indices
    h = o["hit"].to_numpy().copy()
    for _, idx in g.items():
        h[idx] = h[rng.permutation(idx)]
    o["hit"] = h
    o["y_open"] = h
    return o
