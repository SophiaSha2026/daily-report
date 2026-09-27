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

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_DIR = ROOT / "data" / "labels"
BJ = dt.timezone(dt.timedelta(hours=8))
# 收盘 15:00，给集合竞价撮合和快照落库留 5 分钟。和
# local_run.last_closed_trade_day / breakout.backfill.last_closed_trade_day 同一口径
CLOSE_HM = (15, 5)


def path_for(date: str) -> Path:
    return LABEL_DIR / date[:7] / f"label_{date}.parquet"


def from_quotes(codes: list[str], date: str = "",
                now: dt.datetime | None = None) -> pd.DataFrame:
    """live 路径：收盘后抓一次全池快照。

    收盘后 Quote.price 就是收盘价，Quote.open_ 就是当天开盘价。
    并发 5 路是 CLAUDE.md 硬约束 4 定的上限，不要往上调。

    date 给了就只收快照时间戳是那一天的行：隔天盘中手动重跑学习线时，
    快照里已经是新一天的开盘价和现价，拿它给昨天打标是错的（部分票会
    穿过 0.5% 的失配守卫）。这种情况返回空表，调用方按「取不到」处理。

    **还没收盘就一行都不打。** 原来只比日期不比时刻：交易日盘中直接跑
    `python src/eval_daily.py --stage label`（不带 --date，默认今天）时，
    ts 是今天、Quote.price 是**盘中现价**，会被当收盘价写成 r = 现价/开盘−1
    （09:25~09:30 期间全是 0）落进 data/labels，然后 `--stage all` 顺手写
    state/learning_status.json，计划任务当晚据此整夜跳过，错标签再也不会
    被覆盖（labels.save 的「旧的可用行更多就不覆盖」还会帮倒忙）。
    这是历史教训 28「到点已过就立即执行」的同类：时间闸要写在模块里，
    不能指望每个调用方都记得传 --date（breakout/backfill.py 就是自己算的）。
    先判再抓，还省掉一次 3.6 秒的全池请求。
    """
    now = now or dt.datetime.now(BJ)
    if date and date == now.strftime("%Y-%m-%d") and (now.hour, now.minute) < CLOSE_HM:
        log.error("%s 还没收盘（北京 %s），快照 price 是盘中现价不是收盘价，不打标",
                  date, now.strftime("%H:%M"))
        return pd.DataFrame()
    import datasource as ds
    q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
    rec = []
    key = date.replace("-", "") if date else ""
    dropped = 0
    for v in q.values():
        if key and not str(getattr(v, "ts", "")).startswith(key):
            dropped += 1
            continue
        rec.append({
            "code": v.code,
            "open": float(v.open_ or 0.0),
            "close": float(v.price or 0.0),
            "prev_close": float(v.prev_close or 0.0),
        })
    if dropped:
        log.warning("%d 只的快照时间戳不是 %s，丢弃（不是当天的价格）", dropped, date)
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


def save(date: str, df: pd.DataFrame, *,
         force: bool = False) -> tuple[Path, bool]:
    """落盘一天的标签。返回 (路径, 是否真的写了)。

    守卫「已有一份更好的（可用行更多）就不覆盖」认的是**严格少于**
    （old_ok > n_ok），可用行相等照常覆盖。它是为「重跑拿到更脏的一份」
    设计的，但口径一旦收紧（更多行判脏、失配阈值变小、新增剔除条件），
    它恰好拦住修复真正改动到的那几天：把 max_open_mismatch_pct 从 0.5
    收到 0.1 模拟一遍，08-24（1050<1059）和 08-25（1139<1140）被拦，
    其余 6 天因为新旧相等而放行。修了口径重打一遍，结果只有没差别的天
    被覆盖 —— 等于没修，而且只留一行 warning、退出码还是 0。
    所以要有 force，并且**把「写没写」告诉调用方**（教训 16：
    失败必须留下一个能被查询的对象，只写日志等于没写）。
    """
    p = path_for(date)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_ok = int((~df["dirty"]).sum())
    if p.exists() and not force:
        try:
            old = pd.read_parquet(p)
            old_ok = int((~old["dirty"]).sum())
            if old_ok > n_ok:
                log.warning("标签 %s 已有可用 %d 行的一份，本次只有 %d 行，不覆盖"
                            "（口径改了要重打就加 --force）", date, old_ok, n_ok)
                return p, False
        except Exception:  # noqa: BLE001
            pass
    df.to_parquet(p, index=False)
    log.info("标签 %s: %d 行，可用 %d，脏 %d", date, len(df), n_ok,
             len(df) - n_ok)
    return p, True


def load_all() -> pd.DataFrame:
    """所有已落盘的标签。没有就返回空表。"""
    fs = sorted(LABEL_DIR.glob("*/label_*.parquet"))
    if not fs:
        return pd.DataFrame(columns=["date", "code", "r", "dirty"])
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
