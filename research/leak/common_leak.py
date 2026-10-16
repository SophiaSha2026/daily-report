"""
泄漏检验（research/leak）公用件：路径、价格版本、快照读取。不改 src / tools / state / config.yaml。

价格三版（同一份新浪不复权 P_u 和前复权因子 q，q 截至拉取日 2026-10-16）：
    fwd   round(P_u / q, 2)            现状（应 == daily.parquet 的四个价），取整颗粒带 T 日之后的除权信息
    pit   P_u / q 不取整                比值特征和「截至 T 日的复权价」完全一致，颗粒里没有未来信息
    ctl   round(P_u / k_code, 2) / q    k_code 是把各票 q 在 2025-01-02 的值跨票随机置换后得到的常数：
                                        同样量级的取整噪声，但和这只票自己的未来无关。fwd 对 ctl 的差才是泄漏，
                                        fwd 对 pit 的差 = 泄漏 - 取整噪声的代价

所有臂只读快照：train_snap.parquet（生产训练表 2026-10-16 06:00）、daily_snap.parquet、holders_snap.parquet。
只用开发集 2024-06..2025-12，特征只算到 2025-12-31，确认集一行不读。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "stats", ROOT / "research" / "altpath",
          ROOT / "research" / "features"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
LEAK = ROOT / "data" / "breakout" / "raw" / "exp" / "leak"
PROGRESS = HERE / "progress.txt"
KEEP_TO = "2025-12-31"
VERS = ("fwd", "pit", "ctl")
PX = ("open", "high", "low", "close")
SEED_CTL = 20261016


def say(line: str, fh=None) -> None:
    print(line, flush=True)
    if fh is not None:
        fh.append(line)


def progress(text: str) -> None:
    import datetime as dt
    sys.path.insert(0, str(ROOT / "src"))
    from bjtime import now_bj
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))
    _ = dt


def read_prices(ver: str, extra: tuple[str, ...] = ("volume", "amount", "turnover", "outstanding_share")) -> pd.DataFrame:
    """px.parquet 里取某一版价格，列名换回 open/high/low/close，带成交量等原值列。"""
    assert ver in VERS, ver
    cols = ["code", "date"] + [f"{k}_{ver}" for k in PX] + list(extra)
    df = pd.read_parquet(LEAK / "px.parquet", columns=cols)
    return df.rename(columns={f"{k}_{ver}": k for k in PX})


def snap_rows(max_date: str = KEEP_TO) -> pd.DataFrame:
    """快照训练表的 (code, date, board, y_up)，只到开发集终点。"""
    df = pd.read_parquet(LEAK / "train_snap.parquet", columns=["code", "date", "board", "y_up"],
                         filters=[("date", "<=", max_date)])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = df["date"].astype(str).str[:10]
    return df


def round2(x: np.ndarray) -> np.ndarray:
    """取整到分。prices.py 实测：numpy 的 round（银行家）和 daily.parquet 逐位相同 99.991%，半进位 99.984%，用前者。"""
    return np.round(x, 2)
