"""研究用的公共件（2026-10-15）：北京时间禁区、东财跨进程并发闸、熔断、默认超时、真实代码表。

只给 research/datasrc/ 下的脚本用，不碰生产。
- 东财：全机最多 2 路并发（两个锁文件当信号量，跨进程），每次请求后 sleep，
  连续失败 5 次全局停 10 分钟（教训 20），连停 3 次还不行就放弃。
- 每个 requests 调用都有超时（教训 43）：在 Session.request 层补默认 20 秒。
- 北京时间 16:00~18:30 生产流程在跑，这段时间不发任何行情 / 东财请求，脚本存好进度退出。
"""
from __future__ import annotations

import datetime as dt
import json
import msvcrt
import os
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ARCHIVE = ROOT / "archive"
LOCKDIR = ROOT / ".locks"
LOCKDIR.mkdir(exist_ok=True)

BJ = dt.timezone(dt.timedelta(hours=8))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def now_bj() -> dt.datetime:
    return dt.datetime.now(BJ)


class Blackout(Exception):
    pass


def check_blackout() -> None:
    """北京 16:00~18:30 不许发请求（生产流程在跑）。"""
    t = now_bj().time()
    if dt.time(15, 55) <= t <= dt.time(18, 30):
        raise Blackout(f"北京 {t:%H:%M}，在 16:00~18:30 禁区内，停下")


# ---------- 默认超时 ----------
_orig = requests.Session.request


def _req(self, method, url, *a, **kw):
    if len(a) < 6 and kw.get("timeout") is None:
        kw["timeout"] = 20
    return _orig(self, method, url, *a, **kw)


requests.Session.request = _req


# ---------- 东财跨进程 2 路信号量 ----------
_slots_cache = [0.0, 2]


def em_slots() -> int:
    """全机东财 2 路。别的研究进程（research/announce 拉东财公告）在跑时，这边只占 1 路。5 分钟查一次。"""
    if time.time() - _slots_cache[0] > 300:
        import subprocess
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\").CommandLine"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            out = "announce"
        _slots_cache[:] = [time.time(), 1 if ("announce" in out or "fetch_em" in out) else 2]
    return _slots_cache[1]


class EMGate:
    def __init__(self, gap=(0.4, 0.9)):
        self.gap = gap
        self.fh = None

    def __enter__(self):
        check_blackout()
        while True:
            for i in range(em_slots()):
                fh = open(LOCKDIR / f"em_slot{i}.lock", "a+b")
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    self.fh = fh
                    return self
                except OSError:
                    fh.close()
            time.sleep(0.05 + random.random() * 0.1)

    def __exit__(self, *exc):
        try:
            self.fh.seek(0)
            msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.fh.close()
        # 先放锁再歇：歇在锁里的话，同一进程放了马上又抢回来，别的进程永远轮不到
        time.sleep(random.uniform(*self.gap))
        return False


class Breaker:
    """连续失败 n 次就整体停 pause 秒；连停 max_trips 次没有一次成功就放弃。"""

    def __init__(self, n=5, pause=600, max_trips=3, log=print):
        self.n, self.pause, self.max_trips, self.log = n, pause, max_trips, log
        self.fail = 0
        self.trips = 0

    def ok(self):
        self.fail = 0
        self.trips = 0

    def bad(self, err) -> bool:
        """返回 False 表示该放弃了。"""
        self.fail += 1
        if self.fail >= self.n:
            self.trips += 1
            if self.trips > self.max_trips:
                self.log(f"熔断 {self.max_trips} 次仍失败，放弃：{err}")
                return False
            self.log(f"连续失败 {self.fail} 次，全局停 {self.pause}s（第 {self.trips} 次）：{err}")
            end = time.time() + self.pause
            while time.time() < end:
                check_blackout()
                time.sleep(10)
            self.fail = 0
        return True


def codes() -> list[str]:
    """真实代码表（教训 2）。"""
    import pandas as pd
    return pd.read_csv(REPO / "cache" / "codes.csv", dtype={"code": str})["code"].str.zfill(6).tolist()


def market(code: str) -> str:
    if code.startswith(("6", "9")) and not code.startswith("92"):
        return "SH"
    if code.startswith(("92", "8", "4")):
        return "BJ"
    return "SZ"


def trade_days(start: str | None = None) -> list[str]:
    """交易日：从本机日线表读日期列（只读）。"""
    import pandas as pd
    d = pd.read_parquet(REPO / "data" / "breakout" / "daily.parquet", columns=["date"])["date"]
    days = sorted(pd.to_datetime(d.unique()).strftime("%Y-%m-%d"))
    return [x for x in days if start is None or x >= start]


class Progress:
    """可续跑的进度：一个 json，记 done / empty / fail。"""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text("utf-8")) if path.exists() else {"done": {}, "fail": {}}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False), "utf-8")
        os.replace(tmp, self.path)


def logger(path: Path):
    fh = open(path, "a", encoding="utf-8")

    def log(*a):
        s = f"[{now_bj():%m-%d %H:%M:%S} BJ] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()
    return log


if __name__ == "__main__":
    print(now_bj(), len(codes()), trade_days()[-3:])
    sys.exit(0)
