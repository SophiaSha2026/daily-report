"""
打磨 ret20 规则（research/polish，2026-10-17）公用件。

基线 = ret20 一行字规则（候选池内按近 20 日涨幅取前 10），候选池和 research/rulecore 同一份：
    elig0（非 ST、pit 收盘 >= 2 元、>= 120 根、最近 20 根无 >= 5 个交易日停牌缺口、四个输入有值）
    且 近 30 天无减持 / 未来 30 天解禁 < 5% / 近 60 天无增发（common_rc.event_flags）
两段：筛选集 2022-01..2024-05（582 天）、开发集 2024-06..2025-12（387 天），合并 969 天。
尺子：每天前 10 买得到命中 y_open，按天配对差，NW t（滞后 20）、块自举、去最好月、滚动三月、前 30 辅助同号。

产物 data/breakout/raw/exp/polish/：pool.parquet（候选池行 + 扩展特征 + 标签）、各臂名次表。
不改 src / tools / state / config.yaml；不读 2026-01..08 标签（表截到 2025-12-31）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "rulecore", ROOT / "research" / "screen"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
PL = EXP / "polish"
PL.mkdir(parents=True, exist_ok=True)
PROGRESS = HERE / "progress.txt"
SEGS = {"screen": ("2022-01", "2024-05"), "dev": ("2024-06", "2025-12")}
K_TOP = 10
FAMILY = "breakout_y50_top10_polish"


def say(*a):
    print(*a, flush=True)


def progress(text: str) -> None:
    from bjtime import now_bj
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def in_prod_window() -> bool:
    """北京 16:00~18:30 工作日不跑计算。周末放行。"""
    from bjtime import now_bj
    t = now_bj()
    if t.weekday() >= 5:
        return False
    hm = t.hour * 60 + t.minute
    return 16 * 60 <= hm < 18 * 60 + 30


def wait_prod_window() -> None:
    import time
    while in_prod_window():
        say("北京 16:00~18:30，暂停 5 分钟")
        time.sleep(300)


def seg_of(dates: pd.Series) -> np.ndarray:
    return np.where(dates < SEGS["dev"][0] + "-01", "screen", "dev")


def load_pool(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PL / "pool.parquet", columns=columns)


# ---------------------------------------------------------------
#  名次表 / 命中序列
# ---------------------------------------------------------------
def rank_by(df: pd.DataFrame, score: np.ndarray, keep: int = 200, elig: np.ndarray | None = None) -> pd.DataFrame:
    """按天在候选池里按 score 降序排名，留前 keep。返回 date code rank score + 标签列。"""
    o = df[["date", "code", "board", "y_open", "y_up"]].copy()
    o["score"] = np.asarray(score, np.float64)
    m = np.isfinite(o["score"].to_numpy())
    if elig is not None:
        m &= np.asarray(elig, bool)
    o = o[m].sort_values(["date", "score"], ascending=[True, False], kind="mergesort")
    o["rank"] = o.groupby("date").cumcount() + 1
    return o[o["rank"] <= keep].reset_index(drop=True)


def hit_series(t: pd.DataFrame, k: int = K_TOP, y: str = "y_open") -> pd.Series:
    """每天前 k 的命中（按天均值，缺标签的行不算）。"""
    s = t[t["rank"] <= k]
    return s.groupby("date")[y].mean()


def paired(a: pd.Series, b: pd.Series) -> pd.Series:
    d = (a - b).dropna()
    return d


def nw_se(x: np.ndarray, lag: int = 20) -> float:
    import evalkit as K
    return K._nw_se(np.asarray(x, float), lag)


def nw_t(d) -> float:
    import evalkit as K
    return K.nw_t(np.asarray(d, float))


def block_boot(d, seed: int = 20261017) -> dict:
    import evalkit as K
    return K.block_boot(np.asarray(d, float), seed=seed)


def roll3_min(d: pd.Series) -> tuple[float, str]:
    import evalkit as K
    md = d.groupby(d.index.str[:7]).mean()
    return K.roll3_min(md)


def judge(d: pd.Series, d30: pd.Series | None = None, label: str = "") -> dict:
    """按任务书条文：两段各自同号；合并 NW t；块自举 p；去最好月 t；滚动三月（含 / 去 2024-09）；前 30 同号。"""
    seg = pd.Series(seg_of(pd.Series(d.index)), index=d.index)
    out = {"label": label, "days": int(len(d)), "diff": float(d.mean()), "t": nw_t(d)}
    for s in ("screen", "dev"):
        x = d[seg == s]
        out[s] = {"days": int(len(x)), "diff": float(x.mean()) if len(x) else np.nan,
                  "t": nw_t(x) if len(x) > 25 else np.nan}
    out["same_sign_segs"] = bool(out["screen"]["diff"] >= 0 and out["dev"]["diff"] >= 0) if out["diff"] >= 0 else \
        bool(out["screen"]["diff"] <= 0 and out["dev"]["diff"] <= 0)
    out["boot_p"] = block_boot(d)["p"]
    dm = d.groupby(d.index.str[:7]).mean()
    best = dm.idxmax() if out["diff"] >= 0 else dm.idxmin()
    out["t_drop_best"] = nw_t(d[d.index.str[:7] != best]) if len(dm) > 1 else np.nan
    out["best_month"] = str(best)
    r3, r3e = roll3_min(d)
    dx = d[d.index.str[:7] != "2024-09"]
    r3x, r3xe = roll3_min(dx)
    out["roll3"] = {"min": r3, "end": r3e, "min_ex924": r3x, "end_ex924": r3xe}
    out["wins_months"] = int((dm > 0).sum())
    out["months"] = int(len(dm))
    if d30 is not None:
        d30 = d30.dropna()
        out["top30"] = {"diff": float(d30.mean()), "t": nw_t(d30),
                        "same_sign": bool(np.sign(d30.mean()) == np.sign(out["diff"])) or abs(d30.mean()) < 1e-9}
    return out


def fmt_judge(j: dict, thr: float = 2.5) -> str:
    s = ("%-28s 合并 %+.2fpp (NW t %.2f, %d 天)  筛选 %+.2f (t %.2f)  开发 %+.2f (t %.2f)  同号 %s  "
         "块自举 p %.3f  去%s %.2f  滚3 %+.1f@%s (去924 %+.1f)  赢月 %d/%d" % (
             j["label"], 100 * j["diff"], j["t"], j["days"],
             100 * j["screen"]["diff"], j["screen"]["t"], 100 * j["dev"]["diff"], j["dev"]["t"],
             "是" if j["same_sign_segs"] else "否", j["boot_p"], j["best_month"], j["t_drop_best"],
             100 * j["roll3"]["min"], j["roll3"]["end"], 100 * j["roll3"]["min_ex924"], j["wins_months"], j["months"]))
    if "top30" in j:
        s += "  前30 %+.2f (t %.2f) %s" % (100 * j["top30"]["diff"], j["top30"]["t"], "同号" if j["top30"]["same_sign"] else "异号")
    ok = (j["same_sign_segs"] and j["t"] >= thr and j["boot_p"] < 0.05 and j["t_drop_best"] >= 2.0
          and j["roll3"]["min"] >= -0.05 and ("top30" not in j or j["top30"]["same_sign"]))
    s += "  -> %s" % ("过线" if ok else ("不过线" if j["t"] <= -thr else "未定"))
    return s


def maxt_null(D: pd.DataFrame, b: int = 4000, seed: int = 7) -> np.ndarray:
    import evalkit as K
    return K._maxt_null(D, b=b, seed=seed)


def ledger_add(row: dict) -> None:
    """本地登记簿（每个臂、每次事后加看一行）。"""
    from bjtime import now_bj
    with (HERE / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False) + "\n")


def ledger_count(kind: str | None = "arm") -> int:
    p = HERE / "ledger.jsonl"
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if kind is None or r.get("kind") == kind:
                n += 1
    return n
