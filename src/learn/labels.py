"""
标签：当天开盘买、当天收盘卖的收益。

    r = 收盘 / 开盘 − 1

开盘价按定义就是集合竞价的撮合价，也就是我们清单的买入价，
所以这是**可执行收益**，不是纸面收益。

两条取数路径：

    live      收盘后一次 fetch_quotes 拿全池（Quote.open_ / Quote.price）。
              1160 只约 3.6 秒。
    backfill  从 cache/hist_daily.parquet 里查，一次覆盖所有历史日。

三类样本剔除（在 dataset 层做，这里只打标记）：
    停牌（无开盘价）、一字板（买不进）、开盘价与快照的 auc_price 对不上。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_DIR = ROOT / "data" / "labels"


def path_for(date: str) -> Path:
    return LABEL_DIR / date[:7] / f"label_{date}.parquet"


def from_quotes(codes: list[str]) -> pd.DataFrame:
    """live 路径：收盘后抓一次全池快照。

    收盘后 Quote.price 就是收盘价，Quote.open_ 就是当天开盘价。
    并发 5 路是 CLAUDE.md 硬约束 4 定的上限，不要往上调。
    """
    import datasource as ds
    q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
    rec = []
    for v in q.values():
        rec.append({
            "code": v.code,
            "open": float(v.open_ or 0.0),
            "close": float(v.price or 0.0),
            "prev_close": float(v.prev_close or 0.0),
        })
    return pd.DataFrame(rec)


def from_hist(date: str, hist: pd.DataFrame) -> pd.DataFrame:
    """backfill 路径：从长表日线里切出某一天。"""
    d = hist[hist["日期"].astype(str) == date]
    return pd.DataFrame({
        "code": d["code"].values,
        "open": d["开盘"].astype(float).values,
        "close": d["收盘"].astype(float).values,
    })


def build(date: str, snap: pd.DataFrame, raw: pd.DataFrame,
          max_mismatch_pct: float = 0.5) -> pd.DataFrame:
    """把开收盘拼到当天的竞价快照上，算出 r 并标出脏样本。

    `snap` 是 data/YYYY-MM/auction_<date>.parquet，`raw` 是 from_quotes /
    from_hist 的产物。
    """
    df = snap[["code", "auc_price", "one_word"]].merge(raw, on="code", how="left")
    df["date"] = date

    op, cl = df["open"], df["close"]
    df["r"] = (cl / op - 1.0).where((op > 0) & (cl > 0))

    # 开盘价应当等于我们 09:25 采到的撮合价。对不上说明两边指的不是同一天，
    # 或者数据源出了问题——这种样本进了训练集会静默污染结论。
    mis = (df["auc_price"] > 0) & (op > 0)
    df["open_mismatch_pct"] = ((op - df["auc_price"]).abs()
                               / df["auc_price"] * 100).where(mis)

    df["dirty"] = (
        df["r"].isna()
        | df["one_word"].astype(bool)                       # 一字板买不进
        | (df["open_mismatch_pct"] > max_mismatch_pct).fillna(False)
    )
    return df[["date", "code", "open", "close", "r",
               "open_mismatch_pct", "dirty"]]


def save(date: str, df: pd.DataFrame) -> Path:
    p = path_for(date)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    n_ok = int((~df["dirty"]).sum())
    log.info("标签 %s: %d 行，可用 %d，脏 %d", date, len(df), n_ok,
             len(df) - n_ok)
    return p


def load_all() -> pd.DataFrame:
    """所有已落盘的标签。没有就返回空表。"""
    fs = sorted(LABEL_DIR.glob("*/label_*.parquet"))
    if not fs:
        return pd.DataFrame(columns=["date", "code", "r", "dirty"])
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
