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
tools/local_flow_<flow>.log。两边靠 local_run.py 的 --if-needed 幂等互不打架。
"""
from __future__ import annotations

import itertools
import json
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
    # no    界面上的编号。1=早盘系统 2=晚间系统 3=数据维护 4=检查
    # what  一句人话，说清楚点了会发生什么
    # mail  会不会真的往外发邮件。这一列是给人看的，不是开关
    "morning": {
        "no": "1-1", "name": "早盘选股", "group": "1 早盘系统",
        "cmd": ["src/local_run.py", "--flow", "morning"],
        "mail": True, "danger": True,
        "what": "每天早上 9:27 把当天最强的股票发到你邮箱",
        "desc": "建候选池 -> 集合竞价三次采样 -> AI 写点评 -> 9:27:30 发邮件。"
                "没到点会自己等着，提前点没关系。今天已经发过信会直接退出，"
                "不会把同一份清单再发一遍。",
    },
    "morning_dry": {
        "no": "1-1", "name": "早盘选股（试跑）", "group": "1 早盘系统",
        "cmd": ["src/local_run.py", "--flow", "morning", "--dry"],
        "mail": False, "danger": False,
        "what": "跑一遍看看会选出哪些股票，不发邮件",
        "desc": "和上面一样，但不发信、不上传。想验证数据通不通的时候用。",
    },
    "learn": {
        "no": "1-2", "name": "参数自学", "group": "1 早盘系统",
        "cmd": ["src/local_run.py", "--flow", "learn"],
        "mail": False, "danger": False,
        "what": "回头检查早盘选股打分准不准，不准就提修改建议",
        "desc": "拿实际涨跌验证昨天的打分，六道检查全过才提建议。"
                "它改不动准入条件（涨幅 2~5%、量比 2.5~10），那是你定的规则。",
    },
    "council": {
        "no": "1-3", "name": "学习会诊", "group": "1 早盘系统",
        "cmd": ["src/eval_daily.py", "--stage", "council"],
        "mail": False, "danger": False,
        "what": "让 LLM 多路并行深挖「预测和实际差在哪、为什么、怎么改」",
        "desc": "六个视角各起一个 Opus 进程（差距定位 / 波动还是真差距 / 特征体检 / "
                "市场环境 / 数据与流程 / 改进提案），主审汇总，提案自动跑实验，"
                "过闸的在「面板 -> 学习会诊」里等你批准。两条线都看。约 10~15 分钟。",
    },
    "premarket": {
        "no": "3-1", "name": "早盘候选池", "group": "3 辅助工具",
        "cmd": ["src/premarket.py"],
        "mail": False, "danger": False,
        "what": "提前筛出当天要盯的股票",
        "desc": "3~5 分钟。早盘选股会自己判断要不要建，一般不用手点。",
    },
    # 和计划任务走同一个入口（local_run.py --flow breakout）：补当天日线、
    # 重算特征、打分、发信、推仓库，一步不少。以前这里直接调 daily.py，
    # 不补数据，2026-09-14 用户点了一下，把 09-11 的清单又发了一遍。
    "breakout": {
        "no": "2-1", "name": "起涨预测", "group": "2 晚间系统",
        "cmd": ["src/local_run.py", "--flow", "breakout"],
        "mail": True, "danger": True,
        "what": "把最近一个收盘日可能要涨的股票发到你邮箱",
        "desc": "补最近一个交易日的日线 -> 重算特征（约 13 分钟）-> 打分 -> "
                "剔掉 ST、有减持、有解禁、有增发的 -> 清单 A（接近起涨）和 "
                "清单 B（可能见顶）-> 面板 + 邮件。收盘后到次日开盘前都能跑，"
                "结果一样。已经在跑、或目标日已经跑完并发过信，都直接退出。",
    },
    "breakout_dry": {
        "no": "2-1", "name": "起涨预测（试跑）", "group": "2 晚间系统",
        "cmd": ["src/local_run.py", "--flow", "breakout", "--dry"],
        "mail": False, "danger": False,
        "what": "跑一遍看看会选出哪些股票，不发邮件",
        "desc": "和上面一样，但不发信、不上传。",
    },
    "bk_refit": {
        "no": "2-2", "name": "模型自学", "group": "2 晚间系统",
        "cmd": ["src/breakout/daily.py", "--stage", "refit"],
        "mail": False, "danger": False,
        "what": "用最新数据重新训练起涨预测的模型",
        "desc": "平时不用点：每天跑的时候发现模型超过 30 天会自己重训。"
                "这里是「我现在就想让它重学一遍」的入口。约 5 分钟。",
    },
    "evening": {
        "no": "3-2", "name": "回调形态", "group": "3 辅助工具",
        "cmd": ["src/local_run.py", "--flow", "evening"],
        "mail": True, "danger": True,
        "what": "找「涨停后缩量回调、现在重新启动」的股票",
        "desc": "每日自动发送已于 2026-09-12 关掉，这里是手动入口。"
                "平均每个交易日约 1 只，0 只是常态不是故障。",
    },
    # 这个按钮以前走 --stage sina，而那是**首次回填**的断点续传：
    # done_sina.json 里已完成的票一根都不拉（2026-09-16 实测 5515/5548 已
    # done），点了退出码 0、日志「日线合并完成」，一根新 K 线都没有，
    # 用户在「日线只到 X」的时候点它纯属白等（教训 22 的同一处）。
    # 每日增量是 --stage update，和计划任务、起涨预测走的是同一条。
    "bk_backfill": {
        "no": "3-3", "name": "补数据", "group": "3 辅助工具",
        "cmd": ["src/breakout/backfill.py", "--stage", "update"],
        "mail": False, "danger": False,
        "what": "把最近一个收盘日的日线追加进去，起涨预测要用",
        "desc": "腾讯快照追加目标日那一根，秒级；昨收对不上的票整段重拉；"
                "缺不止一个交易日会自动改走全量刷新（约 70 分钟）。"
                "从没回填过请先点「全量回填」。",
    },
    "bk_refresh": {
        "no": "3-3b", "name": "全量回填", "group": "3 辅助工具",
        "cmd": ["src/breakout/backfill.py", "--stage", "refresh"],
        "mail": False, "danger": False,
        "what": "重新下载全市场三年日线",
        "desc": "首次回填或数据坏了才用，约 70 分钟。中断了再点会接着跑。",
    },
    "bk_build": {
        "no": "3-4", "name": "算特征", "group": "3 辅助工具",
        "cmd": ["src/breakout/build.py"],
        "mail": False, "danger": False,
        "what": "把日线算成模型能用的数据",
        "desc": "筹码分布、量价指标、股东人数等，再按当天全市场排名归一。"
                "约 13 分钟，产出 426 万行 x 87 个指标。",
    },
    "bk_arena": {
        "no": "3-5", "name": "比模型", "group": "3 辅助工具",
        "cmd": ["src/breakout/arena.py"],
        "mail": False, "danger": False,
        "what": "试几种模型，看哪个预测得准",
        "desc": "读不到封存的那 9 个月数据 —— 那段只许最终验收时用一次。"
                "约 25 分钟。",
    },
    "refresh_meta": {
        "no": "3-6", "name": "更新股票名单", "group": "3 辅助工具",
        "cmd": ["src/refresh_meta.py"], "mail": False, "danger": False,
        "what": "重新下载全市场股票代码和行业分类",
        "desc": "每周一次就够。",
    },
    "refresh_sector": {
        "no": "3-7", "name": "更新行业成分", "group": "3 辅助工具",
        "cmd": ["src/refresh_sector.py"], "mail": False, "danger": False,
        "what": "重新抓每个行业有哪些股票",
        "desc": "会开一个真浏览器去抓，因为对方认浏览器指纹，"
                "用程序直接请求会被拒。",
    },
    "sync": {
        "no": "3-0", "name": "同步远端产物", "group": "3 辅助工具",
        "cmd": ["src/local_run.py", "--sync"],
        "mail": False, "danger": False,
        "what": "把云端替本机跑出来的清单拉到本地面板",
        "desc": "本机没开机那天云端会代跑早盘选股，结果在 GitHub 上。"
                "计划任务每次重试都会自动拉，这里是手动入口。几秒钟。",
    },
    "build_site": {
        "no": "3-8", "name": "重建网页", "group": "3 辅助工具",
        "cmd": ["src/build_site.py"], "mail": False, "danger": False,
        "what": "把所有面板打包发到网站上",
        "desc": "手机上看的就是这个网站。",
    },
    "selftest": {
        "no": "4-1", "name": "检查早盘选股", "group": "4 检查",
        "cmd": ["src/selftest.py"], "mail": False, "danger": False,
        "what": "确认早盘选股的打分逻辑没被改坏",
        # 不写死条数：用例是一条条加的（2026-09-15 又加了「竞价额不足」那条），
        # 这里的数字改不改全靠人记得，写错了自测也不会红。真实条数看自测输出。
        "desc": "打分用例（每条准入/剔除规则各一个）+ 曲线形状检查 + 规则检查 + "
                "1000 个随机样本，各项条数在自测输出里。",
    },
    "selftest_learn": {
        "no": "4-2", "name": "检查参数自学", "group": "4 检查",
        "cmd": ["src/selftest_learn.py"], "mail": False, "danger": False,
        "what": "确认参数自学没被改坏",
        "desc": "70 余条。重点是两套打分代码结果要完全一致、检查项别接错线。",
    },
    "selftest_breakout": {
        "no": "4-3", "name": "检查起涨预测", "group": "4 检查",
        "cmd": ["src/selftest_breakout.py"], "mail": False, "danger": False,
        "what": "确认起涨预测没被改坏，特别是没偷看未来数据",
        "desc": "筹码算法的六条数学性质、涨跌标注、偷看未来检测、"
                "当天排名是否抹掉了大盘涨跌。",
    },
    "selftest_pullback": {
        "no": "4-4", "name": "检查回调形态", "group": "4 检查",
        "cmd": ["src/selftest_pullback.py"], "mail": False, "danger": False,
        "what": "确认回调形态没被改坏",
        "desc": "13 条形态判定 + 打分排序 + 工具函数。",
    },
    "probe": {
        "no": "4-5", "name": "检查数据源", "group": "4 检查",
        "cmd": ["tools/probe.py"], "mail": False, "danger": False,
        "what": "看看行情数据下载得通不通",
        "desc": "逐个探测数据源。东方财富连不上不影响出榜，只是慢一点。",
    },
}


# 按钮 -> 它会和哪条线抢文件。流程在跑时这些按钮被拒。
# evening 在这里是因为它**内嵌跑学习线**（local_run.flow_evening 末尾），
# 和计划任务 DailyReport-Local-Learn 撞上就是两份 eval_daily --stage all：
# 同日两行 theta_history、两封参数变更邮件、会诊双跑（一次约 $5）。
CONFLICTS = {"premarket": "morning", "bk_backfill": "breakout",
             "bk_refresh": "breakout",
             "bk_build": "breakout", "bk_refit": "breakout",
             "evening": "learn"}
FLOW_NAMES = {"morning": "早盘选股", "breakout": "起涨预测", "learn": "参数自学"}


def _flow_of(key: str) -> str:
    """这个按钮起的是哪条线（`local_run.py --flow X`），不是就返回空串。"""
    cmd = ACTIONS.get(key, {}).get("cmd", [])
    if len(cmd) >= 3 and cmd[0] == "src/local_run.py" and cmd[1] == "--flow":
        return cmd[2]
    return ""

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
        self._drop_lock()

    def _drop_lock(self) -> None:
        """终止之后把这条线的进程锁清掉。

        taskkill /F 是 TerminateProcess，local_run.py 里 `finally: release_lock`
        根本不会执行，state/lock/<flow>.json 留在原地。PID 一旦被系统复用
        （实测释放句柄后十几秒就可能重新分配），这条线在锁的有效期内
        （3~4 小时）会被 --if-needed 和手动入口一致跳过。
        GUI 明知是自己杀的，按 pid 核对后直接清掉。
        """
        flow = _flow_of(self.key)
        if not flow:
            return
        p = ROOT / "state" / "lock" / f"{flow}.json"
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("pid") == self.proc.pid:
                p.unlink()
        except Exception:  # noqa: BLE001
            pass

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
        # 会和正在跑的流程抢同一批文件的按钮，流程在跑时不让点。
        # 「候选池」直接起 premarket.py，不走 local_run 的锁，早盘流程正在建池
        # 时再点一下，两个进程同时写 cache/universe.parquet。
        clash = CONFLICTS.get(key)
        if clash:
            try:
                import sys
                p = str(ROOT / "src")
                if p not in sys.path:
                    sys.path.insert(0, p)
                import local_run
                other = local_run.running_instance(clash)
            except Exception:  # noqa: BLE001
                other = None
            if other:
                return None, (f"{FLOW_NAMES.get(clash, clash)}正在跑"
                              f"（pid {other.get('pid')}），它会自己做这一步，先别点")
        with self._lock:
            flow = _flow_of(key)
            for j in self._jobs.values():
                if not j.running:
                    continue
                if j.key == key:
                    return None, f"{ACTIONS[key]['name']}已经在跑了（{j.id}）"
                # 同一条线的两个按钮（「早盘选股」和「早盘选股（试跑）」）
                # 命令行都是 --flow morning，落到同一把进程锁上。以前这里
                # 只挡同 key，两个一起点就是两个实例互相看见、双双退出 0，
                # 谁都不跑。
                if flow and _flow_of(j.key) == flow:
                    return None, (f"同一条线不能同时起两个："
                                  f"{ACTIONS[j.key]['name']}正在跑（{j.id}）")
            j = Job(key)
            self._jobs[j.id] = j
            return j, ""

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)

    def running_keys(self) -> set[str]:
        return {j.key for j in self._jobs.values() if j.running}
