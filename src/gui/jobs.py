"""
子进程任务管理：启动一条流程，把它的输出实时喂给界面。

为什么是轮询而不是 SSE
----------------------
界面每秒拉一次 /api/job/<id>?from=N 取增量。SSE 看着更时髦，但它要求
连接一直挂着，而 http.server 的每个连接占一个线程，浏览器标签页一多
或者用户合上笔记本再打开，挂死的连接就攒起来了。轮询是无状态的，
断了下次接着拉，offset 自己带着。

为什么子进程必须设 PYTHONUNBUFFERED
-----------------------------------
竞价线的 quick 阶段会自旋等到 09:14 再采样，中间几十分钟只有零星几行
输出。Python 发现 stdout 不是终端就切成块缓冲（4KB），这几行会一直卡在
缓冲区里，界面上看着就像进程死了。实际 2026-09 之前 TUI 也踩过，
表现是「跑起来之后半小时没动静」。

GUI 手动跑 vs 计划任务自动跑
----------------------------
这里管的是**手动**跑：进程是 GUI 的子进程，关掉 GUI 就断。自动跑归
计划任务（DailyReport-Local-*），和 GUI 无关，界面只读它们写的
tools/local_flow.log。两边靠 local_run.py 的 --if-needed 幂等互不打架。
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
import threading
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PY = sys.executable

# 界面上能点的动作。cmd 是相对 ROOT 的命令行，danger 决定按钮是不是红的。
#
# 「发信」那一列是给人看的，不是开关：真正决定发不发的是 SKIP_MAIL
# 环境变量和各流程自己的 dry 分支。写在这里是为了让人点之前就知道
# 这一下会不会有邮件飞出去。
ACTIONS: dict[str, dict] = {
    "morning": {
        "name": "早盘选股", "group": "早盘系统",
        "cmd": ["src/local_run.py", "--flow", "morning"],
        "mail": True, "danger": True,
        "desc": "建候选池 -> 集合竞价三次采样 -> AI 写点评 -> 09:27:30 发邮件。没到点会自己等，提前跑没关系。",
    },
    "morning_dry": {
        "name": "早盘选股（试跑）", "group": "早盘系统",
        "cmd": ["src/local_run.py", "--flow", "morning", "--dry"],
        "mail": False, "danger": False,
        "desc": "同上，但不发信、不推仓库。验证数据通路用。",
    },
    "learn": {
        "name": "参数自学", "group": "早盘系统",
        "cmd": ["src/local_run.py", "--flow", "learn"],
        "mail": False, "danger": False,
        "desc": "【早盘系统的一部分，收盘后跑】拿实际涨跌回头检验早盘选股的打分参数，六道检查全过才提出修改建议。它改不动准入条件（涨幅 2~5%、量比 2.5~10），那是你定的规则。",
    },
    "premarket": {
        "name": "早盘候选池", "group": "早盘系统",
        "cmd": ["src/premarket.py"],
        "mail": False, "danger": False,
        "desc": "筛出当天要盯的股票池，3-5 分钟。早盘选股会自己判断要不要建，一般不用手点。",
    },
    "breakout": {
        "name": "起涨预测", "group": "晚间系统",
        "cmd": ["src/breakout/daily.py", "--stage", "all"],
        "mail": True, "danger": True,
        "desc": "打分全市场 -> 剔除 ST/减持/解禁/增发 -> 出清单 A（接近起涨，"
                "10 只）和清单 B（见顶信号）-> 面板 + 邮件。约 3 分钟。",
    },
    "breakout_dry": {
        "name": "起涨预测（试跑）", "group": "晚间系统",
        "cmd": ["src/breakout/daily.py", "--stage", "scan"],
        "mail": False, "danger": False,
        "desc": "只打分出清单，不发邮件、不做面板。看看今天会选出哪些票。",
    },
    "evening": {
        "name": "回调形态", "group": "晚间系统",
        "cmd": ["src/local_run.py", "--flow", "evening"],
        "mail": True, "danger": True,
        "desc": "找「放量启动 -> 缩量回调 -> 再次启动」的形态。每日自动报告已于 "
                "2026-09-12 停用，这里是手动入口。平均每个交易日约 1 只，0 只是常态，不是故障。",
    },
    "bk_backfill": {
        "name": "补数据", "group": "晚间系统·模型维护",
        "cmd": ["src/breakout/backfill.py", "--stage", "sina"],
        "mail": False, "danger": False,
        "desc": "下载全市场三年日线。中断了可以接着跑，已下好的会跳过。首次约 70 分钟，之后只补新增的那几天。",
    },
    "bk_build": {
        "name": "建特征表", "group": "晚间系统·模型维护",
        "cmd": ["src/breakout/build.py"],
        "mail": False, "danger": False,
        "desc": "把日线算成模型能用的特征表：筹码分布、量价指标、股东人数等，再按全市场当日排名归一。约 13 分钟，产出 426 万行 x 87 个特征。",
    },
    "bk_arena": {
        "name": "模型对比", "group": "晚间系统·模型维护",
        "cmd": ["src/breakout/arena.py"],
        "mail": False, "danger": False,
        "desc": "筛特征 + 逐月滚动测试几个模型，看哪个准。**读不到封存的那 9 个月**，那段数据只许验收时用一次。约 25 分钟。",
    },
    "selftest_breakout": {
        "name": "起涨预测自检", "group": "自检",
        "cmd": ["src/selftest_breakout.py"], "mail": False, "danger": False,
        "desc": "检查筹码算法的六条数学性质、涨跌标注是否正确、有没有偷看未来数据、当日排名是否抹掉了大盘涨跌。2.6 秒。",
    },
    "selftest": {
        "name": "早盘选股自检", "group": "自检",
        "cmd": ["src/selftest.py"], "mail": False, "danger": False,
        "desc": "17 个打分用例 + 9 条打分曲线形状检查 + 4 条规则检查 + 1000 个随机样本。",
    },
    "selftest_pullback": {
        "name": "回调形态自检", "group": "自检",
        "cmd": ["src/selftest_pullback.py"], "mail": False, "danger": False,
        "desc": "13 条形态判定 + 打分排序是否单调 + 工具函数。",
    },
    "selftest_learn": {
        "name": "参数自学自检", "group": "自检",
        "cmd": ["src/selftest_learn.py"], "mail": False, "danger": False,
        "desc": "70 余条。重点检查两套打分代码结果是否完全一致、检查项有没有接错线、邮件发得出去。",
    },
    "probe": {
        "name": "数据源体检", "group": "自检",
        "cmd": ["tools/probe.py"], "mail": False, "danger": False,
        "desc": "逐个探测行情数据源通不通 + 本地依赖装齐没有。东方财富连不上不影响出榜，只是慢一点。",
    },
    "refresh_meta": {
        "name": "刷新股票代码表", "group": "维护",
        "cmd": ["src/refresh_meta.py"], "mail": False, "danger": False,
        "desc": "重新下载全市场股票代码表和行业分类。每周一次就够。",
    },
    "refresh_sector": {
        "name": "刷新行业成分", "group": "维护",
        "cmd": ["src/refresh_sector.py"], "mail": False, "danger": False,
        "desc": "开一个真浏览器去同花顺抓行业成分。它认浏览器指纹，用程序直接请求会被拒，所以必须这么绕。",
    },
    "build_site": {
        "name": "重建网页面板", "group": "维护",
        "cmd": ["src/build_site.py"], "mail": False, "danger": False,
        "desc": "把各条流程的网页面板打包发布。所有面板共用一个网站。",
    },
}

_seq = itertools.count(1)


class Job:
    """一次运行。输出攒在内存里，界面按 offset 拉增量。"""

    def __init__(self, key: str) -> None:
        act = ACTIONS[key]
        self.id = f"j{next(_seq)}"
        self.key = key
        self.name = act["name"]
        self.started = dt.datetime.now()
        self.finished: dt.datetime | None = None
        self.rc: int | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()

        env = {
            **os.environ,
            # 见模块 docstring：不设这两个，自旋等待期间界面上看着像死了
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        self.proc = subprocess.Popen(
            [PY, *act["cmd"]], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for ln in self.proc.stdout:
            with self._lock:
                self.lines.append(ln.rstrip("\n"))
                # 跑一整条竞价线大约几百行，10000 行的上限只在出错刷屏时
                # 才会碰到。砍头保尾：出问题时最后那几行才是有用的。
                if len(self.lines) > 10000:
                    del self.lines[:2000]
        self.rc = self.proc.wait()
        self.finished = dt.datetime.now()

    @property
    def running(self) -> bool:
        return self.rc is None

    def tail(self, start: int) -> tuple[int, list[str]]:
        with self._lock:
            return len(self.lines), self.lines[start:]

    def stop(self) -> None:
        """终止。先 terminate，两秒不走再 kill。

        子进程自己还会起孙进程（local_run.py 用 subprocess 跑各阶段），
        terminate 只杀得掉直接那一层。所以 Windows 上走 taskkill /T
        把整棵树端掉，否则 run_auction 会变成孤儿继续跑到 09:27:30 发信。
        """
        if not self.running:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def brief(self) -> dict:
        return {
            "id": self.id, "key": self.key, "name": self.name,
            "running": self.running, "rc": self.rc,
            "started": self.started.strftime("%H:%M:%S"),
            "secs": int(((self.finished or dt.datetime.now())
                         - self.started).total_seconds()),
            "lines": len(self.lines),
        }


class Registry:
    """所有跑过的任务。进程活着期间的历史，不落盘。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, key: str) -> tuple[Job | None, str]:
        if key not in ACTIONS:
            return None, f"未知动作 {key}"
        with self._lock:
            for j in self._jobs.values():
                if j.key == key and j.running:
                    return None, f"{ACTIONS[key]['name']}已经在跑了（{j.id}）"
            j = Job(key)
            self._jobs[j.id] = j
            return j, ""

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)

    def running_keys(self) -> set[str]:
        return {j.key for j in self._jobs.values() if j.running}
