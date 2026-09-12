"""
爆发线的数据回填：三年日线 + 历史股本 + 股东人数。

    python src/breakout/backfill.py --stage daily     日线（最慢，25~40 分钟）
    python src/breakout/backfill.py --stage shares    历史流通股本
    python src/breakout/backfill.py --stage holders   股东人数
    python src/breakout/backfill.py --stage all

为什么要限速
------------
2026-09-12 实测：腾讯日 K 接口 5 并发连拉 200+ 只之后会限流，返回的不是
JSON 而是一段空内容，持续数分钟。CLAUDE.md 硬约束 4 说的「workers 不超过 5」
是针对快照接口的，日 K 这条更严，光靠限制并发数不够，还要**限速**。

所以这里是三并发 + 每只之间 200ms + 每 120 只休 8 秒 + 指数退避重试。
慢一点无所谓，这是一次性的回填；日常增量只拉最近 5 根。

断点续传
--------
每 200 只落一个分片 data/breakout/raw/daily_<nnn>.parquet，进度记在
data/breakout/raw/done.json。中途挂掉再跑会跳过已完成的代码，不重头来。
回填要跑半小时以上，没有断点续传等于每次网络抖动都从零开始。
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "breakout"
RAW = OUT / "raw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backfill")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 限速参数。实测出来的，不要凭感觉调大（见模块 docstring）
WORKERS = 3
SLEEP_EACH = 0.20      # 每只之间
BATCH = 120            # 每多少只休一次
SLEEP_BATCH = 8.0
RETRY = 2              # 见 COOLDOWN：限流时快速失败比反复重试划算

# 限流是**累积配额**，不是瞬时速率：一旦触发，接下来一段时间怎么重试都是白费。
# 第一版写的是每只各自退避重试 4 次（每次 25s 超时），结果一批 120 只要跑满
# 一小时还拉不到东西，因为 120 只各自把 4 次重试跑了个遍。
# 现在改成**全局熔断**：连续失败到阈值就整体停 COOLDOWN 秒等配额回来，
# 期间一个请求都不发。失败的票不计入 done，下一轮自动重试。
FAIL_STREAK = 8
COOLDOWN = 150.0

BARS = 800             # 腾讯一次最多给这么多，约 3.28 年


def prefix(code: str) -> str:
    """腾讯的市场前缀。北交所是 bj，8/4/9 开头都归它。"""
    if code[0] == "6":
        return "sh" + code
    if code[:2] in ("83", "87", "88", "43", "92") or code[:3] == "920":
        return "bj" + code
    return "sz" + code


def is_bj(code: str) -> bool:
    return code[:2] in ("83", "87", "88", "43", "92") or code[:3] == "920"


def codes(include_bj: bool = False) -> list[str]:
    """回填用的代码表。

    ST 在打分阶段才剔除，回填阶段全要：训练需要负样本，也需要
    「后来变成 ST」的票在变 ST 之前的那段历史。

    北交所默认排除：2026-09-12 实测腾讯**不提供北交所历史日线** ——
    前缀 bj 是对的（接口返回了 data dict，不是报错），但 K 线数组是空的，
    920268 只给当天 1 根。338 只北交所要单独找源，见 stage_daily_bj。
    """
    p = ROOT / "cache" / "codes.csv"
    cs = pd.read_csv(p, dtype=str)["code"].tolist()
    cs = sorted(set(cs))
    if not include_bj:
        cs = [c for c in cs if not is_bj(c)]
    return cs


def fetch_one(code: str) -> pd.DataFrame | None:
    """拉一只的日线。失败返回 None，由调用方决定重试还是放弃。"""
    url = f"{KLINE}?param={prefix(code)},day,,,{BARS},qfq"
    for attempt in range(RETRY):
        try:
            r = requests.get(url, timeout=25, headers=UA)
            data = r.json().get("data")
            if not isinstance(data, dict):
                raise ValueError("data 不是 dict")
            v = list(data.values())[0]
            ks = v.get("qfqday") or v.get("day") or []
            if len(ks) < 60:
                return None                    # 次新股，特征算不出来
            df = pd.DataFrame([k[:6] for k in ks],
                              columns=["date", "open", "close", "high",
                                       "low", "volume"])
            for c in ("open", "close", "high", "low", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["code"] = code
            return df.dropna()
        except Exception:  # noqa: BLE001
            # 指数退避 + 抖动。限流是全局的，所有线程会一起撞上，
            # 不加抖动的话它们会同步重试、同步再被限。
            if attempt < RETRY - 1:
                time.sleep((2 ** attempt) * 1.5 + random.random())
    return None


def stage_daily() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    done_f = RAW / "done.json"
    done: set[str] = set()
    if done_f.exists():
        try:
            done = set(json.loads(done_f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass

    todo = [c for c in codes() if c not in done]
    log.info("日线回填：共 %d 只，已完成 %d，本轮 %d 只",
             len(done) + len(todo), len(done), len(todo))
    if not todo:
        log.info("没有要拉的，直接合并")
        return merge_daily()

    t0 = time.time()
    buf: list[pd.DataFrame] = []
    shard = len(list(RAW.glob("daily_*.parquet")))
    failed: list[str] = []

    streak = {"n": 0}

    def work(c: str):
        time.sleep(SLEEP_EACH)
        return c, fetch_one(c)

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for c, df in ex.map(work, chunk):
                if df is None:
                    failed.append(c)
                    streak["n"] += 1
                    # 失败的**不**计入 done：可能只是限流，下一轮还要重试。
                    # 计入 done 会让临时故障变成永久缺失。
                else:
                    buf.append(df)
                    streak["n"] = 0
                    done.add(c)
        if streak["n"] >= FAIL_STREAK:
            log.warning("连续失败 %d 只，判定被限流，停 %.0f 秒等配额",
                        streak["n"], COOLDOWN)
            time.sleep(COOLDOWN)
            streak["n"] = 0
        # 落分片 + 记进度。必须在同一时刻写，否则崩溃时会丢数据或重复。
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(
                RAW / f"daily_{shard:04d}.parquet", index=False)
            shard += 1
            buf = []
        done_f.write_text(json.dumps(sorted(done), ensure_ascii=False),
                          encoding="utf-8")
        el = time.time() - t0
        log.info("进度 %d/%d  失败 %d  已用 %.0f 分钟",
                 min(i + BATCH, len(todo)), len(todo), len(failed), el / 60)
        if i + BATCH < len(todo):
            time.sleep(SLEEP_BATCH)

    log.info("拉取结束，失败 %d 只：%s", len(failed), failed[:20])
    return merge_daily()


def merge_daily(pattern: str = "daily_*.parquet") -> int:
    shards = sorted(RAW.glob(pattern))
    if not shards:
        log.error("没有任何分片")
        return 1
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df = df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "daily.parquet", index=False)
    log.info("日线合并完成：%d 行，%d 只，%s .. %s",
             len(df), df.code.nunique(), df.date.min(), df.date.max())
    return 0


# ---------------------------------------------------------------
#  新浪日线（主源）
# ---------------------------------------------------------------
#  为什么主源是新浪不是腾讯（2026-09-12 改）
#  ----------------------------------------
#  一、字段。新浪一个接口给全：OHLCV + amount(成交额) +
#      outstanding_share(逐日流通股本) + turnover(换手率)。腾讯只有
#      OHLCV 六个字段，换手率要另外拉 gbjg_em 再做阶梯填充。
#      新浪这份让整条 shares 管线都不必要了，而且是逐日精确值。
#  二、配额。腾讯日 K 有累积配额，本机短时间打 500+ 请求后限流数十分钟，
#      回填反复撞墙。新浪没观察到这个问题。
#
#  代价：新浪接口内部用 py_mini_racer 跑 JS 解密，**它不是线程安全的** ——
#  多线程并发调用会让 V8 直接崩溃（Check failed:
#  !IsConfigurablePoolInitialized()），不是抛异常是进程级 FATAL。
#  所以这里用**多进程**，每个进程一个独立 V8 实例。
SINA_WORKERS = 4
LIMIT = {"n": 0}


def _sina_one(code: str):
    """必须是模块级函数：ProcessPoolExecutor 要 pickle 它。"""
    import warnings
    warnings.filterwarnings("ignore")
    import akshare as ak
    sym = prefix(code)
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
            if d is None or len(d) < 60:
                return code, None
            d = d.tail(BARS).copy()
            d = d[["date", "open", "high", "low", "close", "volume",
                   "amount", "outstanding_share", "turnover"]]
            d["code"] = code
            d["date"] = d["date"].astype(str).str[:10]
            return code, d
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return code, None


def stage_daily_sina() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    done_f = RAW / "done_sina.json"
    done: set[str] = set()
    if done_f.exists():
        try:
            done = set(json.loads(done_f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    todo = [c for c in codes() if c not in done]
    if LIMIT["n"]:
        todo = todo[:LIMIT["n"]]
    log.info("新浪日线：共 %d 只，已完成 %d，本轮 %d 只",
             len(done) + len(todo), len(done), len(todo))
    if not todo:
        return merge_daily(pattern="sina_*.parquet")

    t0 = time.time()
    shard = len(list(RAW.glob("sina_*.parquet")))
    failed: list[str] = []
    buf = []
    STEP = 200
    for i in range(0, len(todo), STEP):
        chunk = todo[i:i + STEP]
        with ProcessPoolExecutor(max_workers=SINA_WORKERS) as ex:
            for c, d in ex.map(_sina_one, chunk, chunksize=4):
                if d is None:
                    failed.append(c)
                else:
                    buf.append(d)
                    done.add(c)
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(
                RAW / f"sina_{shard:04d}.parquet", index=False)
            shard += 1
            buf = []
        done_f.write_text(json.dumps(sorted(done), ensure_ascii=False),
                          encoding="utf-8")
        el = time.time() - t0
        rate = (i + len(chunk)) / max(el, 1e-9)
        log.info("进度 %d/%d  失败 %d  已用 %.0f 分钟  预计还需 %.0f 分钟",
                 min(i + STEP, len(todo)), len(todo), len(failed), el / 60,
                 (len(todo) - i - len(chunk)) / max(rate, 1e-9) / 60)
    log.info("新浪拉取结束，失败 %d 只", len(failed))
    return merge_daily(pattern="sina_*.parquet")


def stage_shares() -> int:
    """历史流通股本。解锁精确换手率，见设计文档 1.2。"""
    import akshare as ak
    OUT.mkdir(parents=True, exist_ok=True)
    cs = codes()
    rows, failed = [], []

    def one(code: str):
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_gbjg_em(symbol=code)
                if d is None or not len(d):
                    return None
                d = d[["变更日期", "总股本", "已上市流通A股"]].copy()
                d.columns = ["date", "total_shares", "float_shares"]
                d["code"] = code
                return d
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        return None

    t0 = time.time()
    for i in range(0, len(cs), 100):
        chunk = cs[i:i + 100]
        with ThreadPoolExecutor(max_workers=4) as ex:
            for c, d in zip(chunk, ex.map(one, chunk)):
                (rows.append(d) if d is not None else failed.append(c))
        log.info("股本 %d/%d 失败 %d 用时 %.0f 分钟",
                 min(i + 100, len(cs)), len(cs), len(failed),
                 (time.time() - t0) / 60)
        time.sleep(2)

    if not rows:
        log.error("股本一条都没拉到")
        return 1
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    df.to_parquet(OUT / "shares.parquet", index=False)
    log.info("股本完成：%d 行，%d 只（失败 %d）",
             len(df), df.code.nunique(), len(failed))
    return 0


def stage_holders() -> int:
    """股东人数。按报告期批量拉，比逐只快得多。

    披露滞后的处理不在这里，在 features.py：这里只负责把原始数据落盘，
    包含报告期本身，让下游按公告日对齐（设计文档 4.6）。
    """
    import akshare as ak
    OUT.mkdir(parents=True, exist_ok=True)
    # 2023-05 之后的所有季末 + 半月末（东财按这些日期组织数据）
    periods = []
    for y in (2023, 2024, 2025, 2026):
        for md in ("0331", "0630", "0930", "1231"):
            d = f"{y}{md}"
            if "20230501" <= d <= "20260912":
                periods.append(d)
    rows = []
    for p in periods:
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_gdhs(symbol=p)
                if d is not None and len(d):
                    d = d.copy()
                    d["报告期"] = p
                    rows.append(d)
                    log.info("股东人数 %s: %d 行", p, len(d))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("股东人数 %s 第 %d 次失败: %s", p, attempt + 1,
                            str(e)[:60])
                time.sleep(2 * (attempt + 1))
        time.sleep(1)
    if not rows:
        log.error("股东人数一条都没拉到")
        return 1
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(OUT / "holders.parquet", index=False)
    log.info("股东人数完成：%d 行，%d 期", len(df), len(periods))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只拉前 N 只，用于验证流程")
    ap.add_argument("--stage", default="all",
                    choices=["daily", "sina", "shares", "holders", "all",
                             "merge"])
    a = ap.parse_args()
    if a.stage == "merge":
        return merge_daily()
    rc = 0
    LIMIT["n"] = a.limit
    if a.stage == "sina":
        return stage_daily_sina()
    if a.stage in ("daily", "all"):
        rc |= stage_daily()
    if a.stage in ("shares", "all"):
        rc |= stage_shares()
    if a.stage in ("holders", "all"):
        rc |= stage_holders()
    return rc


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
