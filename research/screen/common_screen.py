"""
扩展筛选集（research/screen，2026-10-16，docs/design_2026-10-16.md 6.2 / 7.2 规则 7）公用件。

不改 src / tools / state / config.yaml，不改 research/leak 已有文件（只 import）。
产物都在 data/breakout/raw/exp/screen/（gitignore）。

数据
    data/breakout/raw/exp/leak/unadj_2020.parquet   新浪不复权 OHLC + volume amount outstanding_share turnover + qfq_factor，
                                                    2020-07-01 起，code 取自当前 daily.parquet（没有已退市的票：幸存者偏差）
价格口径 pit：P_u / (q_s / q_last)，不取整；q_last 取各票全表最后一行（research/leak/report.md 第 2 节）。
停牌占位行按生产的 backfill.clean_bars 清。

确认集一行不读：价格、特征、标签只算到 KEEP_TO = 2025-12-31。
筛选集 SCREEN = evalkit.SCREEN（2022-01..2024-05）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "leak", ROOT / "research" / "stats",
          ROOT / "research" / "features"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
LEAK = ROOT / "data" / "breakout" / "raw" / "exp" / "leak"
SCR = ROOT / "data" / "breakout" / "raw" / "exp" / "screen"
SCR.mkdir(parents=True, exist_ok=True)
PROGRESS = HERE / "progress.txt"
LEDGER = HERE / "ledger.jsonl"            # 筛选集本地登记簿（不进主族 breakout_y50_top10）
KEEP_TO = "2025-12-31"
DATA_START = "2020-07-01"
PX = ("open", "high", "low", "close")
EXTRA = ("volume", "amount", "turnover", "outstanding_share")
TRAIN_YEARS = 3                           # 每个测试月的训练窗口上限（生产的表本身约三年）
FAMILY = "screen_y50_top10"               # 本地登记簿的族名，和主族分开


def say(line: str, fh: list | None = None) -> None:
    print(line, flush=True)
    if fh is not None:
        fh.append(line)


def progress(text: str) -> None:
    from bjtime import now_bj
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def now_bj_str() -> str:
    from bjtime import now_bj
    return now_bj().strftime("%Y-%m-%d %H:%M")


def in_prod_window() -> bool:
    """北京 16:00~18:30 生产在算，不跑任何计算。"""
    from bjtime import now_bj
    t = now_bj()
    hm = t.hour * 60 + t.minute
    return 16 * 60 <= hm < 18 * 60 + 30


def ledger(exp: str, arm: str, kind: str, **kw) -> None:
    import datetime as dt
    import json
    row = {"at": dt.datetime.now().isoformat(timespec="seconds"), "family": FAMILY, "exp": exp, "arm": arm,
           "kind": kind, **kw}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def months_between(a: str, b: str) -> list[str]:
    import evalkit as K
    return K.months_in((a, b))


def month_shift(m: str, k: int) -> str:
    y, mm = int(m[:4]), int(m[5:7])
    t = y * 12 + (mm - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def read_px(px: str = "px_screen.parquet", cols: tuple[str, ...] = PX + EXTRA) -> pd.DataFrame:
    df = pd.read_parquet(SCR / px, columns=["code", "date"] + list(cols))
    return df


def nw_se(x, lag: int = 20) -> float:
    import evalkit as K
    return K._nw_se(np.asarray(x, float)[np.isfinite(np.asarray(x, float))], lag)
