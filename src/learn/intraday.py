"""
盘中多时点采样：为「卖点」这一端提供数据。

为什么需要它
------------
学习系统的标签写死成「开盘买、收盘卖」。这隐含一个假设：竞价强度这个
信号在整个交易日里都成立。但 A 股的竞价类信号常见的形态是**开盘后
半小时见顶、尾盘回吐**。如果真是那样：

    用收盘价算 IC  ->  接近 0
    得出的结论     ->  「这套打分没用」
    而事实         ->  打分有效，只是卖点选错了

这是个能让整套评估得出相反结论的风险，必须能被测出来。

为什么不买分钟数据
------------------
Tushare 的 stk_mins（¥2000/年）首根是 09:30:00，不含竞价段，
对特征那一端毫无帮助；而卖点这一端我们自己采就行：全池 1200 只
一次批量 3.6 秒，五个时点一天 18 秒，零成本。

代价是历史测不了，只能从现在开始攒。但学习闸门本来就要 60 天才开，
两件事是同一个时间表。

产出
----
data/intraday/YYYY-MM/intraday_<date>_<HHMM>.parquet
    code, px, point, sampled_at, lag_sec
一个采样点一个文件（五个采样点是五个独立任务，写同一张表会互相覆盖）。
load_all() 拼成宽表 code × {t0935,t1000,t1130,t1400,t1500}。

下游用它算「在第 h 个时点卖出」的 IC 曲线，见 exit_curve()。
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
DIR = ROOT / "data" / "intraday"

# 采样时点。选点的理由：
#   09:35 开盘五分钟，竞价情绪最集中的窗口
#   10:00 第一波结束
#   11:30 上午收盘
#   14:00 下午开盘后
#   15:00 收盘（和现有标签对齐，用来交叉校验）
POINTS = ["09:35:00", "10:00:00", "11:30:00", "14:00:00", "15:00:00"]


def _col(t: str) -> str:
    return "t" + t.replace(":", "")[:4]


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def sample_once(codes: list[str]) -> dict[str, float]:
    import datasource as ds
    q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
    return {v.code: float(v.price or 0.0) for v in q.values()}


def collect_one(codes: list[str], date: str, label: str) -> Path:
    """一次性采一个时点，落一个独立文件。

    刻意做成「一个时点一个文件」而不是读改写同一张表：五个采样点是五个
    独立的 workflow 任务，并发写同一个 parquet 会互相覆盖。分文件之后
    每个任务只写自己那份，load_all() 再拼起来，没有竞态。

    文件里存的是**实际采样时刻**，不是计划时刻。GitHub cron 迟到是常态
    （CLAUDE.md 里记着实测迟到 97 分钟），下游据此判断这个点还能不能用。
    """
    ts = now_bj()
    px = sample_once(codes)
    df = pd.DataFrame({"code": codes})
    df["px"] = df["code"].map(px)
    df["point"] = label
    df["sampled_at"] = ts.strftime("%H:%M:%S")
    df["lag_sec"] = int((ts - dt.datetime.combine(
        dt.date.fromisoformat(date), dt.time.fromisoformat(label),
        tzinfo=dt.timezone(dt.timedelta(hours=8)))).total_seconds())
    p = DIR / date[:7] / f"intraday_{date}_{label.replace(':', '')[:4]}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    log.info("%s 采到 %d 只（实际 %s，迟 %d 秒）", label,
             int(df["px"].notna().sum()), df["sampled_at"].iloc[0],
             df["lag_sec"].iloc[0])
    return p


def nearest_point(now: dt.datetime | None = None) -> str:
    """当前时刻最接近哪个计划采样点。cron 迟到时用它对号入座。"""
    now = now or now_bj()
    cur = now.hour * 3600 + now.minute * 60 + now.second
    def secs(t):
        h, m, s = (int(x) for x in t.split(":"))
        return h * 3600 + m * 60 + s
    return min(POINTS, key=lambda t: abs(secs(t) - cur))


def load_all(max_lag_sec: int = 1800) -> pd.DataFrame:
    """所有分文件拼成宽表：一行一只票一天，各时点一列。

    迟到超过 max_lag_sec 的采样点直接丢掉。一个本该 10:00 的点实际
    11:30 才采到，它测的就不是「持有到 10:00」，留着会污染卖点曲线。
    """
    fs = sorted(DIR.glob("*/intraday_*.parquet"))
    if not fs:
        return pd.DataFrame()
    fr = []
    for f in fs:
        d = pd.read_parquet(f)
        d["date"] = f.stem.split("_")[1]
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    n0 = len(df)
    df = df[df["lag_sec"].abs() <= max_lag_sec]
    if len(df) < n0:
        log.info("丢掉迟到超过 %d 秒的采样 %d 行", max_lag_sec, n0 - len(df))
    wide = df.pivot_table(index=["date", "code"], columns="point",
                          values="px", aggfunc="last").reset_index()
    wide.columns = [_col(c) if ":" in str(c) else c for c in wide.columns]
    return wide


def exit_curve(df: pd.DataFrame, scores: np.ndarray, top_k: int = 10
               ) -> pd.DataFrame:
    """每个卖点的 IC 和前 K 超额收益。

    `df` 需要含 date / code / auc_price 以及各时点列。
    回答的问题：这个信号到底该持有多久？如果 09:35 的 IC 是 0.06、
    收盘是 0.01，那说明信号衰减很快，「收盘卖」这个标签在系统性低估打分器。
    """
    from learn.model_select import spearman
    rows = []
    for col in [c for c in df.columns if c.startswith("t")]:
        per_day = []
        for day, g in df.assign(_s=scores).groupby("date"):
            px, op = g[col].to_numpy(float), g["auc_price"].to_numpy(float)
            r = np.where((px > 0) & (op > 0), px / op - 1.0, np.nan)
            m = np.isfinite(r)
            if m.sum() < 20:
                continue
            y = r[m] - np.median(r[m])
            s = g["_s"].to_numpy()[m]
            top = np.argsort(-s)[:top_k]
            per_day.append((spearman(s, y), float(y[top].mean())))
        if not per_day:
            continue
        ic = np.array([a for a, _ in per_day if np.isfinite(a)])
        ex = np.array([b for _, b in per_day])
        rows.append({"exit": col, "days": len(per_day),
                     "ic": float(ic.mean()) if ic.size else np.nan,
                     "icir": float(ic.mean() / ic.std(ddof=1))
                             if ic.size > 1 and ic.std(ddof=1) > 0 else np.nan,
                     "top_excess": float(ex.mean())})
    return pd.DataFrame(rows)
