"""步骤 2：按抽样月份拉 ret20 前 50 的通达信历史分时（240 根 1 分钟价 / 量）+ 历史分笔（按分钟聚合成主动买 / 卖 / 中性量）。

    research/datasrc/.venv/Scripts/python.exe research/intraday/fetch.py [--workers 3]

样本：research/altdata/top60.parquet 里 rank_ret20 <= 50、月份在 MONTHS 里、板块不是北交所（pytdx 不提供北交所）。
产物：raw/parts/min_*.parquet（code date i price vol）、raw/parts/tx_*.parquet（code date hm buy sell neu n big_buy big_sell）、
      raw/parts/meta_*.parquet（code date n_min n_tx ok err）。重跑自动跳过已有的 (code, date)。
纪律：北京 15:55~18:30 不发请求（每个请求前查）；每路一条连接、socket 超时 10 秒；失败重连重试 3 次；
      全局连续失败 12 次停 5 分钟，连停 3 次放弃。并发 = workers（<= 3）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

RAW = HERE / "raw"
PARTS = RAW / "parts"
PARTS.mkdir(parents=True, exist_ok=True)
LOG = HERE / "fetch.log"
MONTHS = {"screen": ["2022-03", "2022-07", "2022-11", "2023-03", "2023-08", "2024-04"],
          "dev": ["2024-07", "2024-09", "2024-12", "2025-04", "2025-08", "2025-12"]}
HOSTS = [("218.6.170.47", 7709), ("123.125.108.14", 7709), ("180.153.18.170", 7709),
         ("60.191.117.167", 7709), ("180.153.18.172", 80), ("202.108.253.139", 80)]
BIG_YUAN = 1_000_000.0      # 单笔成交额 >= 100 万算大单（价 x 手 x 100）
FLUSH_EVERY = 150

_lock = threading.RLock()   # say() 在持锁的代码块里也会被调用，必须可重入（第一版 Lock 在第 200 个时死锁）
_state = {"fail": 0, "trips": 0, "stop": False, "done": 0, "err": 0}


def say(*a):
    s = "[%s BJ] %s" % (bjtime.now_bj().strftime("%m-%d %H:%M:%S"), " ".join(str(x) for x in a))
    print(s, flush=True)
    with _lock:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(s + "\n")


def in_blackout() -> bool:
    t = bjtime.now_bj().time()
    return dt.time(15, 55) <= t <= dt.time(18, 30)


def wait_blackout():
    while in_blackout():
        say("禁区，等 5 分钟")
        time.sleep(300)


def tdx_market(code: str) -> int:
    return 1 if (code.startswith(("6", "9")) and not code.startswith("92")) else 0


def done_keys() -> set:
    keys = set()
    for f in glob.glob(str(PARTS / "meta_*.parquet")):
        m = pd.read_parquet(f, columns=["code", "date"])
        keys |= set(zip(m["code"], m["date"]))
    return keys


def sample_items() -> pd.DataFrame:
    t = pd.read_parquet(ROOT / "research" / "altdata" / "top60.parquet",
                        columns=["date", "code", "board", "rank_ret20", "y_open"])
    t = t[t["rank_ret20"] <= 50].copy()
    t["m"] = t["date"].str[:7]
    t = t[t["m"].isin(MONTHS["screen"] + MONTHS["dev"])]
    t["bj"] = t["board"] == "bj"
    return t.sort_values(["date", "rank_ret20"]).reset_index(drop=True)


def agg_ticks(ticks: list) -> pd.DataFrame:
    df = pd.DataFrame(ticks)
    df["amt"] = df["price"] * df["vol"] * 100.0
    big = df["amt"] >= BIG_YUAN
    g = df.groupby("time", sort=True)
    out = pd.DataFrame({
        "buy": df["vol"].where(df["buyorsell"] == 0, 0).groupby(df["time"]).sum(),
        "sell": df["vol"].where(df["buyorsell"] == 1, 0).groupby(df["time"]).sum(),
        "neu": df["vol"].where(df["buyorsell"] == 2, 0).groupby(df["time"]).sum(),
        "n": g.size(),
        "big_buy": df["vol"].where(big & (df["buyorsell"] == 0), 0).groupby(df["time"]).sum(),
        "big_sell": df["vol"].where(big & (df["buyorsell"] == 1), 0).groupby(df["time"]).sum(),
    }).reset_index().rename(columns={"time": "hm"})
    return out


class Worker(threading.Thread):
    def __init__(self, wid: int, q: "queue.Queue", host):
        super().__init__(daemon=True)
        self.wid, self.q, self.host = wid, q, host
        self.api = None
        self.buf_min, self.buf_tx, self.buf_meta = [], [], []

    def connect(self):
        from pytdx.hq import TdxHq_API
        if self.api is not None:
            try:
                self.api.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=True)
        for k in range(len(HOSTS)):
            ip, port = HOSTS[(self.wid + k) % len(HOSTS)]
            try:
                if self.api.connect(ip, port, time_out=10):
                    self.host = (ip, port)
                    return
            except Exception as e:  # noqa: BLE001
                say("w%d 连 %s 失败 %r" % (self.wid, ip, e))
        raise RuntimeError("没有能连的主机")

    def fetch_one(self, code: str, date: str):
        mk = tdx_market(code)
        di = int(date.replace("-", ""))
        m = self.api.get_history_minute_time_data(mk, code, di)
        if m is None:
            raise RuntimeError("minute None")
        ticks, start, pages = [], 0, 0
        while pages < 60:
            tk = self.api.get_history_transaction_data(mk, code, start, 2000, di)
            pages += 1
            if tk is None:
                raise RuntimeError("tx None")
            if not tk:
                break
            ticks.extend(tk)
            if len(tk) < 2000:
                break
            start += len(tk)
        return m, ticks

    def flush(self):
        ts = "%s_w%d_%d" % (time.strftime("%Y%m%d%H%M%S"), self.wid, len(self.buf_meta))
        if self.buf_min:
            pd.concat(self.buf_min, ignore_index=True).to_parquet(PARTS / f"min_{ts}.parquet", index=False)
        if self.buf_tx:
            pd.concat(self.buf_tx, ignore_index=True).to_parquet(PARTS / f"tx_{ts}.parquet", index=False)
        if self.buf_meta:
            pd.DataFrame(self.buf_meta).to_parquet(PARTS / f"meta_{ts}.parquet", index=False)
        self.buf_min, self.buf_tx, self.buf_meta = [], [], []

    def run(self):
        try:
            self.connect()
        except Exception as e:  # noqa: BLE001
            say("w%d 起不来 %r" % (self.wid, e))
            return
        while not _state["stop"]:
            try:
                code, date = self.q.get_nowait()
            except queue.Empty:
                break
            if in_blackout():
                wait_blackout()
            err, m, ticks = None, None, None
            for attempt in range(3):
                try:
                    m, ticks = self.fetch_one(code, date)
                    break
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:120]
                    time.sleep(1.0 + attempt * 2)
                    try:
                        self.connect()
                    except Exception as e2:  # noqa: BLE001
                        err = repr(e2)[:120]
                        time.sleep(10)
            meta = {"code": code, "date": date, "n_min": 0, "n_tx": 0, "ok": False, "err": err}
            if m is not None:
                meta["ok"] = True
                meta["n_min"] = len(m)
                meta["n_tx"] = len(ticks)
                if len(m):
                    dm = pd.DataFrame(m)
                    dm.insert(0, "code", code)
                    dm.insert(1, "date", date)
                    dm["i"] = np.arange(len(dm), dtype=np.int16)
                    dm["price"] = dm["price"].astype(np.float32)
                    dm["vol"] = dm["vol"].astype(np.float32)
                    self.buf_min.append(dm)
                if ticks:
                    dt_ = agg_ticks(ticks)
                    dt_.insert(0, "code", code)
                    dt_.insert(1, "date", date)
                    self.buf_tx.append(dt_)
            self.buf_meta.append(meta)
            with _lock:
                if meta["ok"]:
                    _state["fail"] = 0
                    _state["trips"] = 0
                    _state["done"] += 1
                else:
                    _state["err"] += 1
                    _state["fail"] += 1
                    tripped = _state["fail"] >= 12
                    if tripped:
                        _state["trips"] += 1
                        _state["fail"] = 0
                        if _state["trips"] > 3:
                            _state["stop"] = True
                            say("熔断 3 次仍失败，放弃")
                report = _state["done"] % 200 == 0 and meta["ok"]
            if report:
                say("进度 %d 成功 / %d 失败，队列剩 %d" % (_state["done"], _state["err"], self.q.qsize()))
            if not meta["ok"] and _state["trips"] and _state["fail"] == 0 and not _state["stop"]:
                say("w%d 连续失败，全局停 5 分钟（第 %d 次）：%s" % (self.wid, _state["trips"], err))
                time.sleep(300)
            if len(self.buf_meta) >= FLUSH_EVERY:
                self.flush()
        self.flush()
        try:
            self.api.disconnect()
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.workers = max(1, min(3, a.workers))
    wait_blackout()
    items = sample_items()
    done = done_keys()
    todo = items[~items["bj"]]
    todo = todo[[(c, d) not in done for c, d in zip(todo["code"], todo["date"])]]
    say("样本 %d 个股票-日（%d 天），北交所 %d 个不拉，已完成 %d，待拉 %d" % (
        len(items), items["date"].nunique(), int(items["bj"].sum()), len(done), len(todo)))
    (HERE / "sample_plan.json").write_text(json.dumps({
        "months": MONTHS, "stock_days": int(len(items)), "days": int(items["date"].nunique()),
        "bj_skipped": int(items["bj"].sum()), "per_month": items.groupby("m").size().to_dict()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if a.limit:
        todo = todo.head(a.limit)
    q = queue.Queue()
    for c, d in zip(todo["code"], todo["date"]):
        q.put((c, d))
    t0 = time.time()
    ws = [Worker(i, q, HOSTS[i]) for i in range(a.workers)]
    for w in ws:
        w.start()
        time.sleep(0.5)
    last = -1
    while any(w.is_alive() for w in ws):
        time.sleep(60)
        if _state["done"] == last:
            say("!! 60 秒没有进展（成功 %d 失败 %d），线程 %s" % (_state["done"], _state["err"], [w.is_alive() for w in ws]))
        last = _state["done"]
    say("结束：成功 %d 失败 %d 用时 %.0f 秒" % (_state["done"], _state["err"], time.time() - t0))


if __name__ == "__main__":
    main()
