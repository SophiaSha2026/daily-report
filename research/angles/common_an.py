"""
四个角度提精度（research/angles，2026-10-18）公用件。条文见 criteria.txt。

尺子、两段、候选池、基线全部 import research/gate 的 common_gt（它再 import precision / polish / rulecore 的三份），
不另写一份。这里只加：px_screen 读取、事件 / 日指标 / 路径表的路径、按行集合评估、进度和登记。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "research" / "gate", ROOT / "research" / "precision", ROOT / "research" / "polish",
          ROOT / "research" / "rulecore", ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common_gt as G  # noqa: E402
import common_pr as PR  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
SEGS = G.SEGS
FAMILY = "breakout_angles"
EXP_NAME = "angles_v1"
SCR = ROOT / "data" / "breakout" / "raw" / "exp" / "screen"
PX = SCR / "px_screen.parquet"
POOL = ROOT / "data" / "breakout" / "raw" / "exp" / "polish" / "pool.parquet"
DAY_IND = HERE / "day_ind.parquet"          # 角度二：全市场日级情绪指标
EVENTS = HERE / "events.parquet"            # 角度三：事件行 (kind, code, date, ...)
PATHS = HERE / "paths.parquet"              # 角度四 / 三：次日开盘起的路径量
END = "2025-12-31"
say = G.say


def progress(text: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def ledger_add(row: dict) -> None:
    from bjtime import now_bj
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False) + "\n")


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


def in_prod_window() -> bool:
    from bjtime import now_bj
    t = now_bj()
    hm = t.hour * 60 + t.minute
    return 16 * 60 <= hm < 18 * 60 + 30


def load_px(columns: list[str] | None = None) -> pd.DataFrame:
    """px_screen 2020-07..2025-12（截到 END，本来就没有 2026 年）。"""
    cols = columns or ["code", "date", "open", "high", "low", "close", "volume", "outstanding_share"]
    d = pd.read_parquet(PX, columns=cols)
    d["code"] = d["code"].astype(str).str.zfill(6)
    d["date"] = d["date"].astype(str).str[:10]
    return d[d["date"] <= END].reset_index(drop=True)


def load_labels(columns: list[str] | None = None) -> pd.DataFrame:
    """候选池表（polish pool.parquet）里的标签和 pool 列：所有行（不只候选池），用于事件 ∩ 候选池。"""
    cols = columns or ["code", "date", "y_open", "r20_open", "yizi1", "end20", "pool", "elig0", "board"]
    t = pd.read_parquet(POOL, columns=cols)
    t["code"] = t["code"].astype(str).str.zfill(6)
    return t


def seg_of(dates) -> np.ndarray:
    return G.seg_of(dates)


# ---------------------------------------------------------------
#  按行集合评估：rows 只要有 date / hit 两列（都在该段内），尺子同 precision / gate
# ---------------------------------------------------------------
def eval_rows(rows: pd.DataFrame, seg: str, boot: bool = True) -> dict:
    B = G.baseline(seg)
    lo, hi = SEGS[seg][0] + "-01", SEGS[seg][1] + "-31"
    r = rows[(rows["date"] >= lo) & (rows["date"] <= hi)]
    r = r[r["date"].isin(set(B.dates))]
    t = pd.DataFrame({"date": r["date"].to_numpy(), "hit": r["hit"].to_numpy(float)})
    e = PR.evaluate(t, np.ones(len(t), bool), B, boot=boot)
    e["B10"] = B.precision_k(10)
    e["dates"] = B.dates
    if len(r):
        months = G.months_between(*SEGS[seg])
        e["per_month"] = float(len(r) / len(months))
        if "r20_open" in r:
            hits = r[r["hit"] > 0]
            e["gain_q"] = hits["r20_open"].quantile([0.1, 0.5, 0.9]).to_dict() if len(hits) else {}
        yr = r.groupby(r["date"].str[:4])["hit"].agg(["size", "mean"])
        e["by_year"] = {k: (int(v["size"]), float(v["mean"])) for k, v in yr.iterrows()}
    else:
        e["per_month"] = 0.0
        e["gain_q"] = {}
        e["by_year"] = {}
    # 日基准：同日候选池基准率按出票数加权
    db = day_base(seg)
    if len(r):
        e["P_daybase"] = float(np.nansum(db.reindex(r["date"]).to_numpy()) / len(r))
    else:
        e["P_daybase"] = np.nan
    return e


_DB = {}


def day_base(seg: str) -> pd.Series:
    """每天候选池 y_open 基准率（gate table 的候选池行）。"""
    if seg not in _DB:
        t = G.load_table(["date", "hit", "seg"])
        for s, g in t.groupby("seg"):
            _DB[s] = g.groupby("date")["hit"].mean()
        del t
    return _DB[seg]


def fmt(name: str, e: dict, seg: str) -> str:
    if e["V"] <= 0:
        return "%-30s %-6s V 0（整段空榜）" % (name, seg)
    s = PR.fmt_eval(name, e, seg)
    s += "  月 %4.1f只 空榜 %3.0f%%  日基准 %4.1f%%  B10 %4.1f%%" % (e["per_month"], 100 * e["empty_share"],
                                                                   100 * e.get("P_daybase", np.nan), 100 * e["B10"])
    if e.get("gain_q"):
        s += "  命中涨幅 q10/50/90 %.0f/%.0f/%.0f%%" % tuple(100 * e["gain_q"][q] for q in (0.1, 0.5, 0.9))
    if e.get("by_year"):
        s += "  逐年 " + " ".join("%s %d/%.1f%%" % (k, v[0], 100 * v[1]) for k, v in sorted(e["by_year"].items()))
    return s


combined_judge = G.combined_judge


def usable(e_s: dict, e_d: dict, j: dict, tau: float | None = None) -> str:
    """criteria 第 0 节「可用」四条。"""
    a = e_s["V"] >= 150 and e_s["days_out"] >= 60 and e_d["V"] >= 150 and e_d["days_out"] >= 60
    b = (np.isfinite(e_s.get("P_lo", np.nan)) and e_s["P_lo"] >= e_s["B10"]
         and np.isfinite(e_d.get("P_lo", np.nan)) and e_d["P_lo"] >= e_d["B10"])
    c = (np.isfinite(e_s.get("dP", np.nan)) and e_s["dP"] >= 0 and np.isfinite(e_d.get("dP", np.nan)) and e_d["dP"] >= 0
         and j["t"] >= 2.5 and j["boot_p"] < 0.05)
    d = True if tau is None else (e_s.get("P", 0) >= 0.8 * tau and e_d.get("P", 0) >= 0.8 * tau)
    if a and b and c and d:
        return "可用"
    if j["t"] <= -2.5:
        return "比名次差"
    if a and b and not c:
        return "量换精度"
    if c and not a:
        return "方向"
    return "不可用"
