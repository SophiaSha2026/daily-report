"""
本地全流程编排器。控制台按钮和计划任务都走这里。

    python src/local_run.py --flow breakout   起涨预测：补日线 -> 特征 -> 打分 -> 17:00 后发信
    python src/local_run.py --flow pullback   长期调整突破：补日线 -> 扫描 -> 北京 17:58 发信
    python src/local_run.py --sync            只拉远端，不跑流程
    加 --dry 只跑不发不推（测试用）；加 --if-needed 是计划任务的入口（见下）

两条线都在晚间系统里（2026-09-27 起仓库里就这一个系统）：早盘系统
（早盘选股 + 参数自学 + 学习会诊）整体归档到 archive/morning/，
它的 flow_morning / flow_learn / flow_evening 和 LLM 文案一起搬走了。

本地为主、云端托底（2026-09-15 起）
----------------------------------
三档优先级，用户 2026-09-15 定的：

    1. 手动     控制台点按钮，随时（在开跑窗口内）
    2. 本机自动  到了「自动开跑时刻」还没手点，计划任务自己跑
    3. 云端     本机根本没跑（没开机）。两条线的数据都在本机（日线表 +
               特征表），云端算不了，北京 20:30 只发一封「本机没跑」的提醒

计划任务从窗口一开就每 15~30 分钟敲一次，但自动开跑时刻之前只拉远端、
不开跑，把时间留给手动。手动和自动是同一个入口、同一把锁，谁先起谁跑。
跑完把产物 commit + push；Pages 面板由 .github/workflows/pages.yml 在推送后发布。

    --if-needed      目标日这条线已经跑完、或不在开跑窗口，直接退出 0
                     （计划任务每 15~30 分钟重试一次，入口必须幂等，
                     否则会重复发信）

目标日不等于今天
----------------
两条线的「目标日」都是最近一个**已收盘**的交易日：北京 09-15 早上 07:00
补跑，目标日仍是 09-14，数据和 09-14 下午跑一模一样。所以开跑窗口跨过
午夜（16:00 到次日 08:30），机器整天没开、晚上（美东）才醒过来也能把当天
的清单补出来，赶在下一个交易日开盘前。

同一时刻一条线只能有一个实例
----------------------------
2026-09-14 早上两边各起了一个竞价线（用户手点 + 计划任务到点），两个进程
各采各的样、各发各的信，用户收到两封一模一样的邮件，git 还互相撞。
现在 state/lock/<flow>.json 是进程锁：后来者看到活着的锁就退出 0。
两条线共用的日线表 data/breakout/daily.parquet 另有一把
state/lock/daily_update.json（backfill.stage_update 拿），谁先到谁补，
后来的看到已经补到目标日就直接用。

云端托底协议（另一半在 tools/evening_check.py 和 .github/workflows/）
----------------------------------------------------------------
    state/claim/<线>_<date>.json   本地开跑时推送：「我来」
    state/sent/<线>_<date>.json    本地发信成功后推送：「我发了」
云端 20:30（Cloudflare Worker 20:45 再敲一次）看 origin/main 上两条线目标日的
sent 标记，缺哪条就在一封提醒里写哪条。

进度输出约定：每行 "##STEP n/m 文字" 是给 TUI 解析的进度行，
其余行原样透传。
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
# 日志时间一律北京时间（控制台「运行记录」直接印日志，界面只用北京时间）。
# 必须包 staticmethod：直接赋 lambda 会被绑成方法，每条日志都报错并丢掉
logging.Formatter.converter = staticmethod(lambda t: time.gmtime(t + 8 * 3600))
log = logging.getLogger("local")

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def today() -> str:
    return now_bj().strftime("%Y-%m-%d")


_TD: dict = {}


def trade_dates() -> set[str]:
    """交易日历，进程内缓存。拿不到就退化成「周一到周五」（fail-open：
    宁可节假日多跑一次空流程，也不能因为日历接口挂了整条线不跑）。"""
    if "s" not in _TD:
        cache = ROOT / "state" / "trade_dates.json"
        # 先看缓存。--if-needed 探针每 15~30 分钟起一次，三条线都要查日历，
        # 每次都 import akshare + 一次网络往返（而 import akshare 本身就是
        # 教训 19 里那个 V8）。缓存是 12 小时内写的、而且覆盖到今天之后，
        # 就直接用：日历一天最多变一次。
        try:
            if time.time() - cache.stat().st_mtime < 12 * 3600:
                s = set(json.loads(cache.read_text(encoding="utf-8")))
                if s and max(s) >= today():
                    _TD["s"] = s
                    return _TD["s"]
        except Exception:  # noqa: BLE001
            pass
        try:
            import datasource as ds
            # ds.trade_dates() 自己已经把 state/trade_dates.json 写好了
            # （控制台读的就是那份，它不能 import akshare，历史教训 19）。
            # 这里以前又写了一遍同样的内容：写者多一个，撞上「另一个进程正在
            # 截断重写」的窗口就多一倍，而内容逐字节相同，没有任何收益。
            _TD["s"] = set(ds.trade_dates())
        except Exception as e:  # noqa: BLE001
            try:
                _TD["s"] = set(json.loads(cache.read_text(encoding="utf-8")))
                log.warning("交易日历接口拿不到（%s），用本地缓存", e)
            except Exception:  # noqa: BLE001
                log.warning("交易日历拿不到（%s），按周一到周五算", e)
                _TD["s"] = set()
    return _TD["s"]


def last_closed_trade_day(now: dt.datetime | None = None) -> str:
    """最近一个已收盘的交易日：15:05 之后算当天，之前算上一个交易日。"""
    now = now or now_bj()
    d = now.date()
    closed = (now.hour, now.minute) >= (15, 5)
    tds = trade_dates()
    for back in range(0, 15):
        day = d - dt.timedelta(days=back)
        ok = (day.isoformat() in tds) if tds else day.weekday() < 5
        if ok and (back > 0 or closed):
            return day.isoformat()
    return d.isoformat()


def target_date(flow: str) -> str:
    """这条线这次要产出哪一天。两条线都是最近一个已收盘交易日。

    flow 参数留着：调用方（控制台、自测）一直按线问。早盘归档前竞价线的
    目标日是「今天」，以后再有目标日口径不同的线也从这里分。
    """
    return last_closed_trade_day()


# ---------------------------------------------------------------------
#  进度（控制台首页的进度条读它，2026-09-18 用户要「一个进度条」）
# ---------------------------------------------------------------------
# 写在 state/lock/ 下：那个目录不进仓库（.gitignore），进度是本机的瞬时状态，
# 推到远端只会让云端和别的机器看见一个不属于它们的「在跑」。
_FLOW = ""                  # 这个进程在跑哪条线，main 拿到锁之后设
_PROG: dict = {}
_PROG_LAST = [0.0]          # 上次落盘的时刻，细进度按 2 秒一次节流
PROGRESS_DIR = ROOT / "state" / "lock"

# 子阶段输出里认得出「做到第几个了」的几种行 -> 这一步完成了多少（0~1）。
# 起涨预测第 2 步（补日线 + 重算特征）要十几分钟，只按步数算的话进度条会
# 在同一格停十几分钟，看着像卡死。
_SUB = [
    (re.compile(r"逐只处理 (\d+)/(\d+)"),
     lambda m: 0.05 + 0.85 * int(m.group(1)) / max(1, int(m.group(2)))),
    (re.compile(r"三层变换完成"), lambda m: 0.93),
    (re.compile(r"训练表 \d+ 行"), lambda m: 0.97),
    # 长期调整突破第 4 步要等到 17:58 才发信，最长等一个多小时；pullback.wait_until
    # 每分钟打一行「等待发信 k/N 分钟」，进度条按它走，不在同一格停一个小时
    (re.compile(r"等待发信 (\d+)/(\d+) 分钟"),
     lambda m: int(m.group(1)) / max(1, int(m.group(2)))),
]


def _write_progress(force: bool = False) -> None:
    if not _FLOW:
        return
    now = time.time()
    if not force and now - _PROG_LAST[0] < 2.0:
        return
    _PROG_LAST[0] = now
    try:
        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        f = PROGRESS_DIR / f"progress_{_FLOW}.json"
        tmp = f.with_suffix(".tmp")
        _PROG["updated_at"] = now_bj().isoformat(timespec="seconds")
        tmp.write_text(json.dumps(_PROG, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)
    except Exception:  # noqa: BLE001
        pass            # 进度写不出来不许影响流程本身


def _scan_sub(tail: bytes) -> None:
    """在子阶段最近的输出里找最后一条认得出的进度行，更新这一步的完成度。"""
    txt = tail.decode("utf-8", errors="ignore")
    best_pos, best = -1, None
    for rx, fn in _SUB:
        for m in rx.finditer(txt):
            if m.start() > best_pos:
                best_pos, best = m.start(), fn(m)
    if best is not None and best > float(_PROG.get("sub") or 0):
        _PROG["sub"] = round(min(0.99, best), 3)
        _write_progress()


def step(i: int, n: int, text: str) -> None:
    print(f"##STEP {i}/{n} {text}", flush=True)
    _PROG.update({"step": i, "total": n, "text": text, "sub": None})
    _write_progress(force=True)


def keep_awake(on: bool) -> None:
    """跑流程时不让电脑因为**闲置**睡着（SetThreadExecutionState）。

    只管闲置。没插电、电量低到阈值的保护性休眠拦不住，也不该拦：
    2026-09-17 17:02 起涨预测跑到一半停了 15 小时，事件日志写的是
    Sleep Reason: Battery。那种情况只能插电，控制台首页会亮「没插电」。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | (es_system_required if on else 0))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------
#  环境
# ---------------------------------------------------------------------
def load_env() -> None:
    """tools/local.env -> os.environ（实现在 src/localenv.py，三个入口共用）。"""
    import localenv
    localenv.load()


# 这里以前还有一个通用的 sh()，flow_morning / flow_evening 用它直接跑
# `git pull --rebase`（不带 --autostash、不先清残留、退出码 check=False 吞掉）。
# 四条线四种写法，其中两条工作区一脏就静默失败。现在 git 一律走 _git()，
# 拉远端一律走 sync_repo()，所以那个口子连同 sh() 一起去掉了。


def _kill_tree(pid: int) -> None:
    """结束一个进程和它起的全部子孙进程。从不抛异常。

    Windows 上 taskkill /T 按父子关系一路杀下去（新浪那几个进程池子进程
    也在里面）；只杀父进程的话子进程成了孤儿，继续占着日志文件和锁。
    """
    if pid <= 0:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            os.kill(pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


# 子阶段超时被结束时 py() 返回的退出码（和 GNU timeout 同一个数）
TIMEOUT_RC = 124

# 每个子阶段最多跑多久：(总时长, 多久没有一行输出算卡住)，单位秒，None = 不看。
# 2026-09-29 16:30 起涨预测补日线时一个新浪请求挂住，local_run 就在 py() 里
# 陪着等了 14 小时，那天的清单次日 06:31 手动重跑才发出去（教训 43）。
# 时限按正常耗时的好几倍给：只拦「卡死」，不拦「慢」。超时就结束这一步的整棵
# 进程树，这一轮按失败退出，计划任务下一次敲门（15~30 分钟）重跑。
STEP_LIMIT = {
    # 补日线平时一两分钟。缺好几天会退化成新浪全量刷新（约 70 分钟），那条路
    # 每 200 只打一行进度，所以另看「20 分钟一行输出都没有」
    "backfill": (120 * 60, 20 * 60),
    "build": (60 * 60, None),         # 特征表，平时 13 分钟
    "scan": (45 * 60, None),          # 打分，平时一两分钟；换了特征口径会重训模型
    "send": (20 * 60, None),          # 面板 + 邮件，平时半分钟
    "pb_scan": (20 * 60, None),       # 长期调整突破扫描，平时 20 秒
    "pb_send": (20 * 60, None),       # 另加等到 17:58 的时间，见 flow_pullback
}


def py(*args: str, timeout: float | None = None, idle: float | None = None) -> int:
    """跑一个子阶段，输出原样透传，顺带认出细进度写进进度文件。

    以前是 subprocess.run 直接继承 stdout。现在读管道再原样写出去（按字节、
    不按行，tqdm 的回车刷新照样实时），同时在输出里认「做到第几个了」
    （_SUB），控制台首页的进度条读它。stderr 并进 stdout：计划任务那条路本来
    就是 `>> log 2>&1`，控制台那条路本来就 stderr=STDOUT，最终落点不变。

    timeout / idle（秒）：跑满 timeout、或者 idle 秒没有任何输出，就结束这一步
    的整棵进程树，返回 TIMEOUT_RC。流程里每一次调用都必须给 timeout
    （selftest_gui 用 AST 钉住）：子进程挂住时 local_run 不能陪着挂（教训 43）。
    读管道放在单独的线程里：孙进程没杀干净的话管道一直不关，主线程也不能等它。
    """
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    proc = subprocess.Popen([PY, *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    out = getattr(sys.stdout, "buffer", None)
    last = [time.monotonic()]

    def pump() -> None:
        tail = b""
        while True:
            try:
                chunk = (proc.stdout.read1(65536) if hasattr(proc.stdout, "read1")
                         else proc.stdout.read(4096))
            except Exception:  # noqa: BLE001
                return
            if not chunk:
                return
            last[0] = time.monotonic()
            if out is not None:
                try:
                    out.write(chunk)
                    out.flush()
                except Exception:  # noqa: BLE001
                    pass
            tail = (tail + chunk)[-8192:]
            try:
                _scan_sub(tail)
            except Exception:  # noqa: BLE001
                pass

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    t0, why = time.monotonic(), ""
    while True:
        try:
            rc = proc.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if timeout and now - t0 > timeout:
            why = f"跑了 {timeout / 60:.0f} 分钟还没结束"
        elif idle and now - last[0] > idle:
            why = f"{idle / 60:.0f} 分钟没有任何输出"
        else:
            continue
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        break
    reader.join(timeout=10)
    if why:
        log.error("子阶段 %s %s，当作卡死，已结束它和它的子进程（退出码 %d）",
                  " ".join(args[:3]), why, TIMEOUT_RC)
        return TIMEOUT_RC
    return rc


# ---------------------------------------------------------------------
#  git 标记
# ---------------------------------------------------------------------
# 一条 git 命令最多等多久（秒）。pull / push 走网络，连接半死时会一直等下去，
# 和 2026-09-29 那个新浪请求一样（教训 43）。本地命令几秒钟，给足了
GIT_TIMEOUT = 180


def _git(*args: str) -> tuple[int, str]:
    """跑一条 git，返回 (退出码, 合并后的输出)。从不抛异常。

    超过 GIT_TIMEOUT 就结束整棵进程树（git push 会再起 git-remote-https），
    按失败返回：推送失败有 push_status 和下一轮重试兜着，被打断留下的
    index.lock 由 _git_unstick 清。
    """
    try:
        p = subprocess.Popen(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return 1, f"git 起不来: {e}"
    try:
        so, se = p.communicate(timeout=GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_tree(p.pid)
        try:
            p.communicate(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        return TIMEOUT_RC, f"git {' '.join(args[:2])} 超过 {GIT_TIMEOUT} 秒没结束，已结束"
    return p.returncode, ((so or "") + (se or "")).strip()


def _git_unstick() -> None:
    """清掉上一次失败留下的 rebase / merge 中间状态。

    2026-09-07 到 09-11 连错五个交易日的根因就在这里：一次 rebase 冲突把
    仓库停在 rebase-merge 状态，没有任何人清理。此后每天的 pull --rebase
    都直接报 "already a rebase-merge directory" 秒退，三次重试全是空转，
    退出码又被 check=False 吞掉，只在日志里剩一行 warning。
    结果是本地攒了 22 个 commit、远端攒了 25 个，谁都不知道。

    推送必须是**自愈**的：一次冲突只能毁掉这一次，不能毁掉之后的每一次。
    """
    for marker, cmd in (("rebase-merge", "rebase"),
                        ("rebase-apply", "rebase"),
                        ("MERGE_HEAD", "merge"),
                        ("CHERRY_PICK_HEAD", "cherry-pick")):
        if (ROOT / ".git" / marker).exists():
            log.warning("清理残留的 %s 状态", marker)
            _git(cmd, "--abort")
    # 被 taskkill /F 打断的 git 会把 .git/index.lock 留在原地（TerminateProcess
    # 不跑 git 的清理），之后每一次 add/commit 都 rc=128，而 diff --cached 只读
    # 索引照样 rc=0 —— 配合「add 没产物就 return True」就是永久的静默成功。
    # 本项目没有任何 git 操作会持锁超过几秒，5 分钟以上的一定是残留。
    lock = ROOT / ".git" / "index.lock"
    try:
        if lock.exists():
            age = time.time() - lock.stat().st_mtime
            if age > 300:
                log.warning("清理残留的 index.lock（%d 秒前留下，没有 git 在写）",
                            int(age))
                lock.unlink()
    except OSError as e:  # noqa: BLE001  正被别的 git 占着就留着，下次再说
        log.warning("index.lock 删不掉: %s", e)


class LockBusy(RuntimeError):
    """别的流程正占着这把锁（git 工作区 / 日线表），这次放弃。"""


GitBusy = LockBusy      # 旧名字：控制台的「重试上传」和自测按它接异常

GIT_LOCK_MAX = 600      # 锁比这个老（秒）就当持锁进程死了没清


@contextlib.contextmanager
def file_lock(name: str, timeout: float = 300.0, max_age: float = GIT_LOCK_MAX,
              path: Path | None = None):
    """跨进程互斥：state/lock/<name>.json（或调用方给的 path），O_EXCL 原子创建。

    持锁进程死了、或锁比 max_age 秒还老，就当残留清掉重来（教训 15，自愈路径
    本身要能自愈）；活着就等，等过 timeout 抛 LockBusy。git_lock 和 data_lock
    都是它，一种锁只留一份实现（教训 34）。
    """
    p = path or ROOT / "state" / "lock" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = {}
            try:
                info = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass            # 半写状态，按「持锁进程不明」处理
            holder = int(info.get("pid", 0) or 0)
            try:
                age = time.time() - p.stat().st_mtime
            except OSError:
                age = 0.0
            alive = (holder and holder != os.getpid() and _pid_alive(holder)
                     and age <= max_age)
            if alive and time.time() < deadline:
                time.sleep(0.5)
                continue
            if alive:
                raise LockBusy(f"{name} 被 pid {holder} 占着超过 {timeout:.0f} 秒")
            # 持锁进程已经死了、或锁太老：清掉重来
            try:
                p.unlink()
            except OSError:
                pass
            if time.time() >= deadline:
                raise LockBusy(f"拿不到 {name} 锁")
            continue
        try:
            os.write(fd, json.dumps(
                {"pid": os.getpid(), "argv": sys.argv[1:],
                 "at": now_bj().isoformat(timespec="seconds")},
                ensure_ascii=False).encode("utf-8"))
        finally:
            os.close(fd)
        break
    try:
        yield
    finally:
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
            if int(info.get("pid", 0)) == os.getpid():
                p.unlink()
        except Exception:  # noqa: BLE001
            pass


def git_lock(timeout: float = 300.0):
    """跨流程的 git 互斥。整个仓库只有一个工作区，两条线却是并行跑的。

    2026-09-16 实测起涨预测和学习线同一秒启动，两边各自 pull --rebase
    --autostash、各自 add/commit。autostash 在对方正写产物的瞬间 stash/pop，
    pop 冲突时 git 整体放弃 stash，对方没提交的当天面板和清单就被还原成
    前一天的版本，而推送那一步只会报「没有需要提交的产物」。
    锁只串得住 git 命令本身，串不住对方的写盘，所以推送路径同时不再用
    autostash（见 git_commit_push）。
    """
    return file_lock("git", timeout, GIT_LOCK_MAX)


# 日线补数据最长的一条路是缺好几天时的全量刷新（新浪整段重拉，约 70 分钟），
# 锁的「太老就当残留」要比它长
DATA_LOCK_MAX = 3 * 3600


def data_lock(timeout: float = 1800.0, path: Path | None = None):
    """data/breakout/daily.parquet 的写锁。起涨预测和长期调整突破都会去补它
    （backfill.py --stage update），同一时刻只许一个在写。

    backfill 传 path = 它自己的 STATE/../lock/daily_update.json：生产下就是
    state/lock/daily_update.json，自测把 STATE 换成临时目录时锁也跟着进临时目录，
    不去碰（更不会去等）生产目录里那把真锁（教训 17）。"""
    return file_lock("daily_update", timeout, DATA_LOCK_MAX, path=path)


def _write_push_status(ok: bool, msg: str, detail: str = "") -> None:
    """把推送结果落盘，GUI 的「同步」卡片读它。

    只写一行 warning 是不够的：日志没人天天看，失联五天也没人发现。
    状态必须是一个能被界面查询的对象。
    """
    f = ROOT / "state" / "push_status.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        f.write_text(json.dumps({
            "ok": ok, "msg": msg, "detail": detail[:800],
            "at": now_bj().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("写推送状态失败: %s", e)


def push_main(msg: str) -> tuple[bool, str]:
    """把已经提交好的东西推上去。返回 (成不成, 给人看的命令日志)。

    push 失败就 fetch + merge -X ours 再试，最多三轮；成败都写 push_status。
    抽出来是因为控制台的「重试上传」以前自己拼了一套 git（只有 fetch +
    push），本地和远端分叉时按多少次都是 non-fast-forward —— 而它正是为
    2026-09-07 那种分叉加的。一件事只留一份实现。

    调用方负责先拿 git_lock 并 _git_unstick（两个调用方都在锁里）。
    合并方向是 -X ours = 本地优先，理由见 git_commit_push 的注释。
    """
    lines: list[str] = []
    last = ""
    for attempt in range(3):
        rc, out = _git("push", "-q", "origin", "main")
        lines.append(f"$ git push origin main\n{out}".strip())
        if rc == 0:
            _write_push_status(True, msg)
            return True, "\n\n".join(lines)
        _git("fetch", "-q", "origin", "main")
        rc, last = _git("merge", "--no-edit", "-q", "-X", "ours", "origin/main")
        lines.append(f"$ git merge -X ours origin/main\n{last}".strip())
        if rc != 0:
            log.warning("第 %d 次合并远端失败: %s", attempt + 1, last[:200])
            _git("merge", "--abort")
    _write_push_status(False, msg, last or "push 连续三次失败")
    return False, "\n\n".join(lines)


def git_commit_push(msg: str, paths: list[str], dry: bool) -> bool:
    """提交指定路径并推送到远端。返回「是否真的推上去了」。

    完全本地化之后（2026-09-12）远端不再自动跑，正常日子里 fetch 完直接
    就能 push。但手动 dispatch 应急仍可能在远端落 commit，所以合并那条路
    留着，且按**本地优先**自动解冲突：本地是唯一的执行方，它的产物就是
    权威版本。这里用 merge -X ours 而不是 rebase --autostash -X theirs：
    两者解冲突的方向一样（merge 的 ours = 本地 = rebase replay 时的 theirs），
    但 rebase 会 autostash，把另一条线正在写、还没提交的产物 stash 走，
    pop 冲突时整份放弃 —— 当天的面板和清单就悄悄回退成前一天的。
    """
    if dry:
        log.info("[dry] 跳过提交推送: %s", msg)
        return False
    try:
        with git_lock():
            _git_unstick()
            for x in paths:
                # add 的退出码是唯一能看见失败的地方，丢了就会走到下面
                # 「没有需要提交的产物 -> return True」：暂存区空的时候
                # diff --cached 照样 rc=0，于是标记和产物没推出去却报成功。
                # index.lock 是并发撞上的（控制台每 5 秒一次 git status），
                # 短重试就过去了；其余失败（被 gitignore、路径不存在）
                # 说明流程没产出预期文件，必须报出来。
                for attempt in range(4):
                    rc, out = _git("add", "-A", x)
                    if rc == 0:
                        break
                    if "index.lock" in out and attempt < 3:
                        time.sleep(0.5)
                        continue
                    # 路径压根不存在（rc=128 "did not match any files"）不算
                    # 失败：推送列表里可以有**可选产物**（早盘系统归档前，学习线
                    # 的 out_learn/council.html 只在会诊真跑过的那天才有），按失败
                    # 处理的话同一次推送里的其它产物也推不上去。
                    # 被 .gitignore 挡住的那种 rc=1 仍然要报（council.html 从
                    # 上线起一次都没进过仓库，就是它被吞掉的）。
                    if "did not match any files" in out \
                            and not (ROOT / x).exists():
                        log.info("没有 %s，这次不推它", x)
                        break
                    _write_push_status(False, msg, f"add {x} 失败: {out}")
                    log.warning("git add %s 失败: %s", x, out[:200])
                    return False
            if _git("diff", "--cached", "--quiet")[0] == 0:
                log.info("没有需要提交的产物")
                return True
            rc, out = _git("commit", "-q", "-m", msg)
            if rc != 0:
                _write_push_status(False, msg, f"commit 失败: {out}")
                log.warning("提交失败: %s", out[:200])
                return False

            ok, _ = push_main(msg)
            if ok:
                log.info("已推送: %s", msg)
            else:
                log.warning("推送失败（本地已提交，产物没丢）。GUI 的同步卡片会亮红，"
                            "点「重试上传」或手动 git push origin main")
            return ok
    except Exception as e:  # noqa: BLE001
        _write_push_status(False, msg, repr(e))
        log.warning("推送异常: %s", e)
        return False


def push_marker(kind: str, flow: str, payload: dict, dry: bool,
                date: str = "") -> None:
    """写 state/{claim,sent}/<flow>_<date>.json 并推送。

    云端 tools/evening_check.py 读 sent 标记决定要不要发「本机没跑」提醒。
    date 是目标日（两条线都传；不传就是今天，只剩兼容用途）。
    """
    date = date or today()
    p = ROOT / "state" / kind / f"{flow}_{date}.json"
    if dry:
        # 试跑连本地都不写：sent 标记是「真发过信」的唯一证据（教训 27），
        # already_done 在 run_meta 被试跑覆盖时就靠它把真跑认回来。
        # 试跑写一份下去，那个证据就作废了。
        log.info("[dry] 试跑不写 %s 标记", p.name)
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "host": socket.gethostname(),
               "at": now_bj().isoformat(timespec="seconds")}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    git_commit_push(f"{kind}: {flow} {date} [local]",
                    [p.relative_to(ROOT).as_posix()], dry)


# ---------------------------------------------------------------------
#  进程锁：同一条线同一时刻只跑一个实例
# ---------------------------------------------------------------------
def _pid_ctime(pid: int) -> int | None:
    """这个 pid 活着吗，活着的话它是哪一代进程。死了/打不开返回 None。

    只判「有进程活着」是不够的：锁里的进程被 taskkill /F 杀掉后
    finally 里的 release_lock 根本不执行，锁文件留在原地；Windows
    的 PID 空间很快就会重排（实测释放句柄后约 294 个短命进程、13.4 秒
    就重新分配到同一个号）。号被别人拿去，锁在 MAX_RUN 小时内一直
    「活着」，这条线就被 --if-needed 和手动入口一致跳过。
    进程创建时间（FILETIME，100ns）同 PID 不同世代必不同，拿它当身份，
    比每次起一个 PowerShell 查命令行便宜得多（排期页每 5 秒要查三条线）。
    """
    if pid <= 0:
        return None
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return None
            if code.value != 259:                  # STILL_ACTIVE
                return None
            c = ctypes.c_ulonglong()
            e = ctypes.c_ulonglong()
            kt = ctypes.c_ulonglong()
            ut = ctypes.c_ulonglong()
            if not k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e),
                                       ctypes.byref(kt), ctypes.byref(ut)):
                return 0                           # 活着但问不出世代
            return int(c.value)
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return 0
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    return _pid_ctime(pid) is not None


def lock_path(flow: str) -> Path:
    return ROOT / "state" / "lock" / f"{flow}.json"


# 进程表里比这个年轻（秒）的同线实例不算「在跑」。
#
# 计划任务每 15~30 分钟起一个 `--if-needed` 探针，它做完判定（几秒，
# 负载下实测 5 秒）就退出，但那几秒里它的命令行和真流程一模一样。
# 不过滤的话：控制台点击和探针互相看见 -> 双双「已经在跑」退出 0，
# 谁都不跑（实测同秒起两个，gap<=0.5s 必然双退）；sync 任务也会被探针
# 挡住不拉远端（09-16 下午连续 5 次）。
# 真流程最快也要跑几分钟，60 秒足够把两者分开。谁先写进锁文件谁赢，
# 锁是原子创建的（acquire_lock），所以漏看不会变成双跑。
PROBE_GRACE = 60


def _scan_entries(lines: list[str], flow: str, now_utc: dt.datetime,
                  own_pid: int) -> list[dict]:
    """把 PowerShell 那几行 `pid|创建时刻|命令行` 解析成这条线的全部实例。

    每项 {pid, at（北京时间，问不出来是空串）, age（秒，问不出来是 None）}。
    """
    got = []
    for line in lines:
        pid, _, rest = line.partition("|")
        created, _, cmd = rest.partition("|")
        if not pid.strip().isdigit() or int(pid) == own_pid:
            continue
        if "local_run.py" not in cmd or f"--flow {flow}" not in cmd:
            continue
        at, age = "", None
        try:
            c = dt.datetime.strptime(created.strip(), "%Y-%m-%dT%H:%M:%S")
            c = c.replace(tzinfo=dt.timezone.utc)
            age = (now_utc - c).total_seconds()
            at = c.astimezone(dt.timezone(dt.timedelta(hours=8))
                              ).isoformat(timespec="seconds")
        except ValueError:
            pass
        got.append({"pid": int(pid), "at": at, "age": age})
    return got


def _parse_scan(lines: list[str], flow: str, now_utc: dt.datetime,
                own_pid: int) -> dict | None:
    """进程表里「谁在跑」。

    抽成纯函数是为了能离线自测：真机上进程表里可能正好有一条线在跑，
    结果不可复现。
    """
    for e in _scan_entries(lines, flow, now_utc, own_pid):
        if e["age"] is not None and e["age"] < PROBE_GRACE:
            continue                          # 还在判定阶段的探针，不算
        if e["age"] is not None and e["age"] > MAX_RUN.get(flow, 3) * 3600:
            continue                          # 跑超了最长时间：卡死的，见 _parse_stuck
        # 时刻问不出来：按老规矩算它在跑（宁可多退一次，别双发）
        return {"pid": e["pid"], "flow": flow, "at": e["at"], "source": "进程表"}
    return None


def _parse_stuck(lines: list[str], flow: str, now_utc: dt.datetime,
                 own_pid: int) -> list[int]:
    """进程表里这条线跑超了 MAX_RUN 的实例：卡死了，下一个实例接管时结束它们。

    2026-09-29 16:30 起的起涨预测挂在一个新浪请求上 14 小时。它的锁三小时后
    就算过期了，可进程表那一层还一直说「在跑」，之后每一次触发都退出（教训 43）。
    流程里每一步都有时限（STEP_LIMIT），整条流程也有（main 里的看门狗），
    正常情况下活不到 MAX_RUN；活到了，就是卡在那些时限管不到的地方。
    """
    lim = MAX_RUN.get(flow, 3) * 3600
    return [e["pid"] for e in _scan_entries(lines, flow, now_utc, own_pid)
            if e["age"] is not None and e["age"] > lim]


# 进程表查询的真实实现只在这个仓库里生效：自测把 ROOT 换成临时目录时，
# 进程表里的实例属于真仓库，「接管」绝不能去结束它们（教训 17）
_REPO = Path(__file__).resolve().parent.parent


def _proc_lines() -> list[str]:
    """PowerShell 列出全部 python.exe：`pid|创建时刻(UTC)|命令行`。非 Windows 空表。"""
    if sys.platform != "win32":
        return []
    # 创建时刻问不出来时留空串，_parse_scan 会按「在跑」处理（保守那一侧）。
    # 别写成 $_.CreationDate.ToUniversalTime() 直接拼：CreationDate 偶尔是
    # $null，在 $null 上调方法会让整行报错消失，那一条就成了「没在跑」。
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { $c=''; if ($_.CreationDate) { $c = "
          "$_.CreationDate.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss') }; "
          "'' + $_.ProcessId + '|' + $c + '|' + $_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        return []
    return (r.stdout or "").splitlines()


def _scan_processes(flow: str) -> dict | None:
    """锁之外再看一眼进程表：有没有别的 `local_run.py --flow <flow>` 在跑。

    锁文件可能被手删、也可能是旧版本代码起的进程根本没写锁（2026-09-14
    晚上就是）。进程表是事实，锁只是快捷方式。只在 Windows 上做，
    别处返回 None。
    """
    return _parse_scan(_proc_lines(), flow, dt.datetime.now(dt.timezone.utc),
                       os.getpid())


def _scan_stuck(flow: str) -> list[int]:
    """进程表里这条线卡死的实例（跑超了 MAX_RUN）。自测的临时 ROOT 下一律空表。"""
    if ROOT != _REPO:
        return []
    return _parse_stuck(_proc_lines(), flow, dt.datetime.now(dt.timezone.utc),
                        os.getpid())


def _lock_holds(flow: str, info: dict) -> bool:
    """这份锁内容是否代表「别人真的在跑」。

    三个条件缺一不可，而且 running_instance 和 acquire_lock 必须用同一份
    判断：两处口径不一样的话，会出现「查着说没人跑、抢的时候又说有人跑」。
      - 不是自己写的
      - 那个 pid 现在活着，而且**是写锁时的那一代**（ctime，防 PID 复用）
      - 锁没老过 MAX_RUN（真挂住的流程不能把这条线永远堵死）
    """
    try:
        age = (now_bj() - dt.datetime.fromisoformat(info["at"])).total_seconds()
    except Exception:  # noqa: BLE001
        age = 0.0
    pid = int(info.get("pid", 0) or 0)
    if pid <= 0 or pid == os.getpid():
        return False
    ct = _pid_ctime(pid)
    if ct is None:
        return False
    # 旧格式的锁没有 ctime，退回「只看 pid 活着」的老行为
    if info.get("ctime") is not None and ct != 0 and int(info["ctime"]) != ct:
        return False
    return age <= MAX_RUN.get(flow, 3) * 3600


def running_instance(flow: str) -> dict | None:
    """这条线现在是否有别的实例在跑。返回锁内容，没有就 None。

    锁里的进程死了（崩溃、被杀）就当没锁。超过 MAX_RUN 小时的也当没锁，
    防 PID 被系统重用后误判成「还在跑」。没有锁再扫一遍进程表兜底。
    """
    p = lock_path(flow)
    info = None
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    if info and _lock_holds(flow, info):
        return info
    return _scan_processes(flow)


def _stale_lock_pid(flow: str) -> int:
    """锁里记的进程还活着、确实是写锁的那一代，锁却老过 MAX_RUN：卡死了，返回 pid。

    只认带 ctime 的锁：没有进程身份就分不清是卡死的流程还是被复用了号的
    别的进程，宁可不杀（进程表那一层还会按命令行认一遍）。
    """
    try:
        info = json.loads(lock_path(flow).read_text(encoding="utf-8"))
        pid = int(info.get("pid", 0) or 0)
        age = (now_bj() - dt.datetime.fromisoformat(info["at"])).total_seconds()
    except Exception:  # noqa: BLE001
        return 0
    if pid <= 0 or pid == os.getpid() or age <= MAX_RUN.get(flow, 3) * 3600:
        return 0
    ct = _pid_ctime(pid)
    if not ct or info.get("ctime") is None or int(info["ctime"]) != ct:
        return 0
    return pid


def take_over_stuck(flow: str) -> list[int]:
    """这条线卡死的旧实例：结束它们的整棵进程树，返回结束了哪些 pid。

    只结束跑超了 MAX_RUN 的（锁和进程表两处认）。流程自己有看门狗
    （py 的 STEP_LIMIT、main 的总时限），活到 MAX_RUN 的一定是卡在看门狗
    管不到的地方。不结束它的话，它一直占着日志文件和进程表里的「在跑」，
    之后每一次触发都白来（2026-09-29，教训 43）。
    """
    pids = set(_scan_stuck(flow))
    sp = _stale_lock_pid(flow)
    if sp:
        pids.add(sp)
    for pid in sorted(pids):
        log.warning("%s 上一个实例（pid %d）跑了 %d 小时还没结束，当作卡死："
                    "结束它和它的子进程，这次接手", FLOWS[flow][1], pid,
                    MAX_RUN.get(flow, 3))
        _kill_tree(pid)
    return sorted(pids)


def acquire_lock(flow: str) -> bool:
    """抢这条线的锁。拿到返回 True，别人在跑返回 False。

    必须原子：以前是「先 running_instance 查一遍、再 write_text」，
    同一秒起的两个实例（控制台点击撞上计划任务）双双查到「没人在跑」，
    后写的那个把前一个的锁覆盖掉。O_EXCL 保证只有一个能建出文件。
    没人在跑、但有跑超了 MAX_RUN 的卡死实例：先结束它再抢（take_over_stuck）。
    """
    other = running_instance(flow)
    if other:
        log.info("%s 已经在跑（pid %s%s），本实例退出",
                 FLOWS[flow][1], other.get("pid"),
                 f"，{other.get('at', '')[11:19]} 起" if other.get("at") else "")
        return False
    take_over_stuck(flow)
    p = lock_path(flow)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "flow": flow,
                          "ctime": _pid_ctime(os.getpid()),
                          "at": now_bj().isoformat(timespec="seconds"),
                          "argv": sys.argv[1:]}, ensure_ascii=False)
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info: dict = {}
            for _retry in range(3):
                try:
                    info = json.loads(p.read_text(encoding="utf-8"))
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(0.2)     # 可能读到半写的锁，重读
            if _lock_holds(flow, info):
                log.info("%s 已经在跑（pid %s），本实例退出",
                         FLOWS[flow][1], info.get("pid"))
                return False
            try:
                p.unlink()              # 死锁/自己的旧锁，清掉重抢
            except OSError:
                pass
            continue
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        return True
    return False


def release_lock(flow: str) -> None:
    try:
        info = json.loads(lock_path(flow).read_text(encoding="utf-8"))
        if int(info.get("pid", 0)) == os.getpid():
            lock_path(flow).unlink()
    except Exception:  # noqa: BLE001
        pass


def any_flow_running() -> str:
    for f in FLOWS:
        if running_instance(f):
            return f
    return ""


def sync_repo() -> bool:
    """把远端拉下来。云端替本地跑过的话，out/ out_breakout/ 就在远端。

    有流程在跑时不动工作区（它正在写产物、随后要 commit）。
    """
    busy = any_flow_running()
    if busy:
        log.info("%s 正在跑，这次不拉远端", FLOWS[busy][1])
        return False
    try:
        with git_lock(timeout=60):
            _git_unstick()
            before = _git("rev-parse", "HEAD")[1]
            rc, out = _git("pull", "--rebase", "--autostash", "-q",
                           "origin", "main")
            if rc != 0:
                # 只写日志等于没写（教训 16）：控制台的同步卡片读 push_status
                _write_push_status(False, "pull", out)
                log.warning("拉远端失败: %s", out[:200])
                _git_unstick()
                return False
            after = _git("rev-parse", "HEAD")[1]
            # 上一次**拉取**失败留下的红灯，这次拉成功了就清掉。以前只在失败时写，
            # 2026-09-26 一次网络重置之后每次同步都成功，控制台却一直挂着
            # 「上次推送失败」。推送失败（msg 不是 pull）不动，那要等推成功才算好
            try:
                ps = json.loads((ROOT / "state" / "push_status.json")
                                .read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                ps = {}
            if ps and not ps.get("ok") and ps.get("msg") == "pull":
                _write_push_status(True, "pull")
    except GitBusy as e:
        log.warning("这次不拉远端: %s", e)
        return False
    if after != before:
        log.info("已拉取远端 %s -> %s", before[:7], after[:7])
    else:
        log.info("远端没有新东西")
    return True


def push_all(msg: str, paths: list[str], dry: bool) -> None:
    git_commit_push(msg, paths, dry)


# ---------------------------------------------------------------------
#  「这一步真的做了吗」：退出码 0 不等于做了事（教训 22 / 27）
# ---------------------------------------------------------------------
def _run_meta_date(rel: str) -> str:
    """读某条线 run_meta 里的日期，读不到返回空串。"""
    try:
        return str(json.loads((ROOT / rel).read_text(encoding="utf-8"))
                   .get("date") or "")
    except Exception:  # noqa: BLE001
        return ""


# 日线要有这么大比例的票到了目标日，才算「补到了」。
# 正常日实测 5502/5516 = 99.7%（2026-09-16）；余下十几只是长期停牌。
DAILY_COVER_MIN = 0.9


def daily_coverage(daily, target: str) -> tuple[float, str]:
    """日线表里有多大比例的票补到了目标日，以及全表最新的一天。

    以前这里判的是 `daily["date"].max() == 目标日`，而全表 max **一只票就能
    顶起来**：2026-09-16 实测 done_sina.json 里有 33 只从没成功过的新票，
    任何一只被补齐，全表 max 就等于目标日，另外 5515 只还停在旧日期，
    build + scan 照样在旧数据上算出「目标日」的清单发出去 —— 教训 22 那种
    「拿旧数据重算、日期却是今天」，而且 run_meta 的日期还对得上，
    每 30 分钟的重试再也拦不住。
    """
    if len(daily) == 0:
        return 0.0, ""
    last = daily.groupby("code")["date"].max().astype(str)
    return float((last >= target).mean()), str(last.max())


# 目标日的行数至少要有前一个交易日的这么多。和 backfill.stage_update 里
# 那道闸同一个数：daily.parquet 898 个交易日里相邻日行数比最小 0.9706、
# 1% 分位 0.9991，0.95 留足余量，停牌（最近两次各 14 只）不会误杀。
# 编排层还要再核一次，是因为两处看的不是同一张表：stage_update 的核对发生在
# **合并之前**，而重拉分片对它覆盖的代码是「整段权威」，合并时会把 _upd 里
# 那一根清掉（腾讯快照兜底只补 pend 里有的那几只）。掉行只有在合并后的
# daily.parquet 上才看得见。
DAILY_PREV_MIN = 0.95


def _daily_covers(daily, target: str) -> tuple[bool, str]:
    """目标日在日线表里是不是一个**完整**的交易日。返回 (够不够, 人话)。

    抽成纯函数是为了能离线自测：真表 2.5GB 在本机，测不了也不该测。
    """
    cnt = daily["date"].astype(str).value_counts().sort_index()
    days = [d for d in cnt.index if d <= target]
    if not days or days[-1] != target:
        return False, f"日线表里没有 {target} 的行"
    n = int(cnt.loc[target])
    if len(days) < 2:
        return True, f"{target} {n} 行（表里没有更早的交易日可比）"
    pday, pn = days[-2], int(cnt.loc[days[-2]])
    ok = pn <= 0 or n >= DAILY_PREV_MIN * pn
    return ok, (f"{target} {n} 行，前一日 {pday} {pn} 行"
                f"（{n / pn:.1%}，要 ≥{DAILY_PREV_MIN:.0%}）" if pn > 0
                else f"{target} {n} 行")


# 补不上目标日那根 K 线的票超过这个数就不出清单。
# 少几只是常态（新上市、长期停牌、新浪当天还没生成 K 线）；成片补不上
# 说明源那边出事了，此时横截面百分位、板块中性化、市场宽度全在残缺的池子里算
# （教训 30：回测口径和生产口径差一点，成绩就不是同一件事）。
UPDATE_SHORT_MAX = 50


def _update_short(target: str) -> tuple[list[str], list[str], str]:
    """补数据那一步自己记的账：哪些票没拉到目标日。返回 (老票, 新票, 说明)。

    退出码 0 不等于做了事（教训 27）：refetch_codes 拉不到目标日的票只在
    backfill 的日志里留一行，而它写进了 state/breakout/update_status.json，
    这里把它读出来，让「今天到底缺了谁」变成一个能被判断的对象（教训 16）。

    「新票」是日线表里本来就没有历史的（新上市、首次回填漏掉的）。新浪整段
    重拉要 ≥60 根才算拉到，上市不满 60 天的新股天天都「没拉到」；它们也不在
    前一天的横截面里，缺了不会让池子变残缺。以前和除权票混在一起数，
    09-28 的 57 只里 27 只是新股，把 50 只的上限占掉一半（教训 42、43）。
    只有老票算进 UPDATE_SHORT_MAX。旧版的账没有 new_codes，全当老票（保守）。
    """
    try:
        st = json.loads((ROOT / "state" / "breakout" / "update_status.json")
                        .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [], [], "没有 update_status.json（补数据那步没写账）"
    if str(st.get("date") or "") != target:
        return [], [], f"update_status.json 是 {st.get('date')} 的，不是 {target}"
    new = {str(x) for x in (st.get("new_codes") or [])}
    allx = [str(x) for x in (st.get("short") or [])]
    short = [c for c in allx if c not in new]
    fresh = [c for c in allx if c in new]
    return short, fresh, (f"补数据记账：追加 {st.get('appended')} 只，"
                          f"整段重拉没拉到目标日 {len(allx)} 只"
                          f"（其中新股 {len(fresh)} 只），"
                          f"快照没回 {st.get('missing')} 只")


# 补跑到目标日次日这个时刻（北京）还挡在 UPDATE_SHORT_MAX 上，就不再挡：
# 没补齐的票剔出清单，其余照发。再等就过了开盘，一份少几只的清单也比没有强
# （用户 2026-09-30：「保证只要开机状态，就发送清单」）。
# 16:30 到这时候计划任务已经重拉了二三十轮，新浪还没出当天的 K 线，
# 这几只今天就是判不了。
SHORT_LATE = (7, 0)


def short_gate_late(target: str) -> bool:
    """现在是不是已经到了目标日次日（或更晚）的 SHORT_LATE 之后。"""
    now = now_bj()
    try:
        days = (now.date() - dt.date.fromisoformat(target)).days
    except ValueError:
        return False
    return days > 1 or (days == 1 and (now.hour, now.minute) >= SHORT_LATE)


def _mail_sent_date(rel: str, date: str) -> str:
    """某条线目标日的邮件真发出去了没有。发了返回发信时刻，没发返回空串。

    两条线的写法一样：breakout.daily / pullback 都在把邮件交给 SMTP
    **之后**才写这个文件，它是「发过了」的唯一证据（「没发信也 return 0」
    的分支哪条线都有，教训 27）。
    """
    try:
        ms = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        return str(ms.get("at") or "已发") if ms.get("date") == date else ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------
#  两条线的排期表
# ---------------------------------------------------------------------
# 「目标日这条线跑完了没有」读哪个文件，以及 --if-needed 允许开跑的北京时间窗口
#
# 窗口是给自动触发用的，手动跑（不带 --if-needed）不受限制。
# 上界小于下界表示跨午夜。两条线的窗口都是 16:00 到次日 08:30：收盘后数据
# 定死，目标日是最近一个已收盘交易日，机器整天没开、美东晚上才醒也能赶在
# 下一个交易日开盘前把清单补出来。
# 第五项是「自动开跑时刻」（北京）：计划任务在这之前只等手动，到点没手点
# 才自己跑。手动不受它限制，只受窗口限制。
#   breakout 16:30。收盘后半小时数据定型，17:00 前后出清单是这条线的约定；
#            美东凌晨没人手点，实际上就是自动跑，机器没醒就等醒了补。
#   pullback 17:40。用户要 17:58 收到清单（2026-09-27）。扫描几秒钟，日线通常
#            已经被起涨预测 16:30 那次补好，17:40 起跑留足余量；跑完等到
#            17:58:00 才发。过了 17:58 才开跑（没开机）就跑完立即发。
FLOWS = {
    "breakout": ("out_breakout/run_meta.json", "起涨预测", (16, 0), (8, 30), (16, 30)),
    "pullback": ("out_pullback/run_meta.json", "长期调整突破", (16, 0), (8, 30), (17, 40)),
}

# 这条线从哪个交易日起才有清单。上线之前的交易日没有「该发没发」一说：
# 2026-09-27（周日）注册计划任务时目标日还是 09-24，登录触发一敲就会把
# 09-24 当成漏发补一封。计划任务（--if-needed）不补这之前的日子；手动照跑。
START = {"pullback": "2026-09-28"}

# 一次最多跑多久（小时）。三处用它：
#   · main 的总看门狗：自己跑到 MAX_RUN 前 5 分钟还没结束，就结束自己的整棵进程树
#   · 锁比这个老就当死锁（防 PID 重用误判）
#   · 进程表里跑超了它的同线实例当卡死，下一个实例接管时结束（take_over_stuck）
# 必须盖得住各步 STEP_LIMIT 之和，否则流程内的看门狗还没来得及出手就被当成卡死
# （selftest_gui 钉住）。平时起涨预测 15 分钟、长期调整突破几分钟到两小时
# （16:00 手动起跑要等到 17:58 发信）。
#   breakout  补日线 120 + 特征 60 + 打分 45 + 真值 10 + 发信 20 = 255 分钟
#   pullback  等起涨预测补日线 45 + 自己补 120 + 扫描 20 + 等到 17:58 最多约 120
#             + 发信 20 = 325 分钟
MAX_RUN = {"breakout": 5, "pullback": 6}


def in_window(flow: str) -> bool:
    lo, hi = FLOWS[flow][2:4]
    hm = (now_bj().hour, now_bj().minute)
    if lo <= hi:
        return lo <= hm <= hi
    return hm >= lo or hm <= hi          # 跨午夜


AUTO_GRACE_MIN = 2    # 自动开跑时刻的宽限（分钟），见 auto_due


def auto_due(flow: str) -> bool:
    """到没到自动开跑时刻。窗口内且过了 FLOWS 第五项。

    跨午夜的窗口（16:00~次日 08:30）里，自动时刻在午夜前那一段：
    16:30 之后算到点，午夜后到 08:30 也算到点（那是补跑）。
    """
    if not in_window(flow):
        return False
    lo, hi, auto = FLOWS[flow][2:5]
    # 提前几分钟也算到点。计划任务的触发时刻会有秒级抖动，而且偏早：
    # 2026-09-18 早盘那次是 08:29:56 敲门，比自动时刻 08:30 早 4 秒，
    # 按「到没到 08:30」判就跳过了，要等下一次 08:45 才起 —— 每天都晚 15 分钟。
    # 触发间隔最短 15 分钟，宽限 2 分钟吃掉抖动，又不会把上一次敲门算进来。
    t = now_bj() + dt.timedelta(minutes=AUTO_GRACE_MIN)
    hm = (t.hour, t.minute)
    if lo <= hi:
        return hm >= auto
    return hm >= auto or (now_bj().hour, now_bj().minute) <= hi


# 哪条线真发信之后写哪个 sent 标记（state/sent/<名字>_<目标日>.json）
SENT_MARK = {"breakout": "breakout", "pullback": "pullback"}

# 这几条线的「跑完了」还必须有 sent 标记撑着。
#
# 两条线的 run_meta 都在扫描阶段就落盘，发信是下一个子进程（起涨预测
# daily.py --stage send，长期调整突破还要等到 17:58）。只认 run_meta 的话，
# send 挂掉的那天控制台是绿的、计划任务也判「已经跑完」整夜不再补，
# 用户直到第二天才发现没收到清单。
SENT_REQUIRED = {"breakout", "pullback"}


def done_for(flow: str, target: str) -> bool:
    """给定目标日，这条线跑完了没有。

    控制台也调它（gui/status.py），但控制台不能走 target_date ——
    那条路会 import datasource -> akshare，py_mini_racer 的 V8 在多线程
    HTTP 服务里是进程级 FATAL（教训 19）。所以目标日由调用方给。
    """
    rel = FLOWS[flow][0]
    try:
        meta = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if meta.get("date") != target:
        return False
    # 试跑（--dry）也写 run_meta。真跑完之后再点一次试跑，run_meta 就被
    # 标成 dry，以前这里直接返回 False，下一次计划任务敲门（起涨预测窗口
    # 16 小时、每 30 分钟一次）就把整条线重跑并再发一封同样的清单。
    # 试跑不写 sent 标记（push_marker 的 dry 分支），所以真跑完的证据只剩它。
    mark = SENT_MARK.get(flow)
    mark_ok = bool(mark) and (ROOT / "state" / "sent"
                              / f"{mark}_{target}.json").exists()
    if flow in SENT_REQUIRED and not mark_ok:
        return False
    return True if not meta.get("dry") else mark_ok


def already_done(flow: str) -> bool:
    """目标日这条线是否已经跑完。

    计划任务每 15~30 分钟重试一次，一直敲到窗口关（机器可能整段时间都
    不在线，2026-08-27 就因为笔记本没醒漏发过）。代价是跑完之后它还会
    继续敲，所以入口必须幂等，否则每次重试都重发一封邮件。
    """
    return done_for(flow, target_date(flow))


def weekend_skip(flow: str) -> bool:
    """周末拦截只管目标日是「今天」的线，也就是窗口不跨午夜的那几条。

    跨午夜的线（起涨预测、长期调整突破）目标日是最近一个已收盘交易日，
    北京周六 00:00~08:30（= 美东周五中午前）正是补周五清单的时段：
    计划任务按周一~周五排（08:30Z 锚定），北京周五 16:30 起跑 16 小时，
    其中 8.5 小时、17 次触发全落在北京周六。按「现在是不是周末」一刀切挡掉，周五收盘后
    机器没醒过的那一周，周五的清单就永远补不出来，而周一的连续天数
    还会因为缺这一根整榜归零。周六 10:00 之后靠 in_window 挡。
    """
    lo, hi = FLOWS[flow][2:4]
    return lo <= hi and now_bj().weekday() >= 5


# if_needed_skip 返回的原因里，这两类是「时候未到」而不是「不用跑」：
# 计划任务在这两种情况下顺手拉一次远端，把云端代跑的产物落到本地面板。
SYNC_ON = ("不在开跑窗口", "还没到自动开跑时刻")


def if_needed_skip(flow: str) -> str:
    """--if-needed 的闸门，返回跳过原因；空串表示该跑。

    抽成纯函数是为了能离线自测：这几道判断以前散在 main() 里，只能靠
    读代码核对，而它们决定的是「今天这条线到底跑不跑」。
    节假日不用单独挡：目标日是最近一个已收盘交易日，节假日那天的目标日
    就是节前最后一个交易日，早就跑完了，走「已经跑完」那一道。
    """
    if weekend_skip(flow):
        return "北京时间周末"
    since = START.get(flow, "")
    if since and target_date(flow) < since:
        return f"{FLOWS[flow][1]} {since} 起才发清单，目标日 {target_date(flow)} 不补"
    if running_instance(flow):
        return f"{FLOWS[flow][1]} 正在跑"
    if already_done(flow):
        return f"目标日 {target_date(flow)} 已经跑完"
    if not in_window(flow):
        lo, hi = FLOWS[flow][2:4]
        return (f"{SYNC_ON[0]} {lo[0]:02d}:{lo[1]:02d}-{hi[0]:02d}:{hi[1]:02d}"
                f"（北京），现在 {now_bj().strftime('%H:%M')}")
    if not auto_due(flow):
        auto = FLOWS[flow][4]
        return (f"{SYNC_ON[1]} {auto[0]:02d}:{auto[1]:02d}（北京），"
                f"先等手动；到点没手点就自动跑")
    return ""


def daily_ready(target: str) -> tuple[bool, str]:
    """日线表是不是完整地覆盖到了目标日。返回 (够不够, 人话)。两条线共用。

    两道闸（理由见 DAILY_COVER_MIN / DAILY_PREV_MIN 的注释）：
      · 九成以上的票最后一根到了目标日（全表 max 一只票就能顶起来，不能只看它）
      · 目标日的行数不少于前一个交易日的 95%（合并之后才看得见的掉行）
    """
    try:
        import pandas as pd
        dd = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                             columns=["date", "code"])
        cov, mx = daily_coverage(dd, target)
        full, why = _daily_covers(dd, target)
    except Exception as e:  # noqa: BLE001
        return False, f"读不到日线: {e}"
    if cov < DAILY_COVER_MIN:
        return False, f"日线只有 {cov * 100:.1f}% 的票到 {target}（全表最新 {mx}）"
    if not full:
        return False, f"目标日那一天的行数不够：{why}"
    return True, f"日线覆盖 {cov * 100:.1f}%，{why}"


def truth_and_regime(dry: bool) -> None:
    """市场环境指标 + 起涨预测历史清单真值。fail-open，只写 state/。"""
    if dry:
        log.info("[dry] 不重算市场环境和清单真值")
        return
    code = ("import sys; sys.path[:0]=['src','src/breakout']; "
            "import regime as R, truth as T; R.record(90); T.save(T.compute())")
    try:
        r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=600)
        if r.returncode != 0:
            log.warning("市场环境 / 清单真值没算出来（不影响发信）：%s",
                        ((r.stdout or "") + (r.stderr or ""))[-300:])
        else:
            log.info("市场环境 + 清单真值已更新")
    except Exception as e:  # noqa: BLE001
        log.warning("市场环境 / 清单真值没算出来（不影响发信）：%s", e)


# 起涨预测正在补日线时，长期调整突破最多等它这么久（分钟）。它的第 2 步
# （补日线 + 建特征表）一共十几分钟，补日线本身在最前面，通常一两分钟
PULLBACK_WAIT_DAILY_MIN = 45


def ensure_daily(target: str) -> tuple[bool, str]:
    """长期调整突破要的日线补到目标日没有；没有就等起涨预测补，或者自己补。

    谁先到谁补：backfill.stage_update 自己拿 daily_update 锁，后来的那个
    看到已经覆盖到目标日就直接退出 0（「不用追加」那条路）。
    起涨预测正在跑时先等它：它的补日线是第一步，等它比两边抢锁更省事。
    """
    deadline = time.time() + PULLBACK_WAIT_DAILY_MIN * 60
    ran = False
    while True:
        ok, why = daily_ready(target)
        if ok:
            return True, why
        if running_instance("breakout") and time.time() < deadline:
            log.info("起涨预测正在跑，等它把日线补到 %s（%s）", target, why)
            time.sleep(30)
            continue
        if ran:
            return False, why
        log.info("日线还没到 %s（%s），自己补", target, why)
        rc = py("src/breakout/backfill.py", "--stage", "update", "--target", target,
                timeout=STEP_LIMIT["backfill"][0], idle=STEP_LIMIT["backfill"][1])
        ran = True
        if rc != 0:
            return False, f"补数据失败（退出码 {rc}）"


def pullback_send_at(target: str) -> dt.datetime:
    """长期调整突破目标日的发信时刻（pullback.send_time，读 config.yaml）。

    只用来给发信那一步定时限（STEP_LIMIT 另加等待的时间）。读不到按 17:58。
    """
    try:
        import pullback as P
        return P.send_time(target, P.cfg())
    except Exception:  # noqa: BLE001
        d = dt.date.fromisoformat(target)
        return dt.datetime(d.year, d.month, d.day, 17, 58,
                           tzinfo=dt.timezone(dt.timedelta(hours=8)))


def flow_pullback(dry: bool) -> int:
    """长期调整突破：补日线 -> 扫描 -> 等到 17:58 -> 面板 + 邮件 -> 推产物。

    目标日是最近一个已收盘的交易日。日线表和起涨预测共用（它 16:30 那次
    通常已经补好），扫描本身几秒钟。17:58 之前跑完就等，之后跑完立即发。
    """
    n = 4
    d = target_date("pullback")
    # 跑完并发过信就不再跑。手动入口不带 --if-needed（手动优先），点一下
    # 不该把同一份清单再发一遍。确要重发：删 state/sent/pullback_<日>.json
    if not dry and done_for("pullback", d):
        log.info("长期调整突破 %s 已经跑完并发过信，直接退出。"
                 "确要重发先删 state/sent/pullback_%s.json", d, d)
        return 0
    step(1, n, f"同步仓库（目标日 {d}）")
    if not dry:
        sync_repo()
    push_marker("claim", "pullback", {"plan": "本地接管今日长期调整突破"}, dry, d)

    step(2, n, f"日线补到 {d}")
    ok, why = ensure_daily(d)
    if not ok:
        log.error("%s，今天不出清单", why)
        return 1
    log.info("%s", why)
    short, fresh, note = _update_short(d)
    if short or fresh:
        # 这条线逐只判，缺几只只是那几只今天判不了；不像起涨预测要在全市场
        # 横截面上排名，所以只提醒不挡
        miss = short + fresh
        log.warning("%s；没拉到 %s 的 %d 只今天判不了：%s", note, d, len(miss),
                    ",".join(miss[:20]))

    step(3, n, "扫描")
    rc = py("src/pullback.py", "--stage", "scan", "--target", d,
            timeout=STEP_LIMIT["pb_scan"][0])
    if rc != 0:
        log.error("扫描失败（退出码 %d），今天不出清单", rc)
        return rc
    # 退出码 0 不等于做了事（教训 22）：非交易日那条路 scan 也是 return 0
    if _run_meta_date("out_pullback/run_meta.json") != d:
        log.error("扫描没有产出 %s 的清单，不发信", d)
        return 1

    step(4, n, "面板 + 邮件（北京 17:58 发）")
    if dry:
        os.environ["SKIP_MAIL"] = "1"
    # 时限另加「等到 17:58」的那一段：16:00 手动起跑要等将近两小时
    wait = 0.0 if dry else max(0.0, (pullback_send_at(d) - now_bj()).total_seconds())
    rc = py("src/pullback.py", "--stage", "send", "--target", d,
            *(["--no-wait"] if dry else []),
            timeout=STEP_LIMIT["pb_send"][0] + wait)
    # 退出码 0 不等于发了信（教训 27）：SKIP_MAIL 那条路也是 0。只认
    # pullback.stage_send 在 SMTP 之后落的 out_pullback/mail_sent.json
    sent_at = _mail_sent_date("out_pullback/mail_sent.json", d)
    if rc == 0 and (sent_at or dry):
        push_marker("sent", "pullback", {"ok": True}, dry, d)
    elif rc == 0:
        log.error("send 退出码 0 但 out_pullback/mail_sent.json 不是 %s 的，"
                  "不推 sent 标记（云端 20:30 会发「本机没跑」提醒）", d)
        rc = 1
    push_all(f"长期调整突破 {d} [local]",
             [f"data/pullback/{d[:7]}", "out_pullback"], dry)
    return rc


def flow_breakout(dry: bool) -> int:
    """晚间系统：起涨预测。补当天日线 -> 特征表 -> 打分 -> 面板 + 邮件。

    目标日是最近一个已收盘的交易日。补数据走腾讯快照增量（秒级），
    追加不到目标日就**不出清单**：拿旧数据重算只会把上一个交易日的
    清单再发一遍，而且 run_meta 的日期对不上目标日，下次重试还会再发。
    """
    n = 4
    d = target_date("breakout")
    # 跑完并发过信就不再跑。手动入口不带 --if-needed（手动优先），
    # 2026-09-14 用户点了一下就把上一交易日的清单又发了一遍。补数据 +
    # 建特征要 13 分钟，末尾照样 send_mail，没有任何去重。
    if not dry and done_for("breakout", d) and \
            (ROOT / "state" / "sent" / f"breakout_{d}.json").exists():
        log.info("起涨预测 %s 已经跑完并发过信，直接退出。"
                 "确要重发先删 state/sent/breakout_%s.json", d, d)
        return 0
    step(1, n, f"同步仓库（目标日 {d}）")
    if not dry:
        sync_repo()
    push_marker("claim", "breakout", {"plan": "本地接管今日起涨预测"}, dry, d)

    step(2, n, f"补 {d} 日线 + 重算特征表")
    # 目标日由编排层算一次传下去：每个子阶段各判一次，跨午夜补跑那一段
    # （北京 00:00~08:30）两边早晚会分叉，一边按「今天」一边按「最近已收盘」。
    rc = py("src/breakout/backfill.py", "--stage", "update", "--target", d,
            timeout=STEP_LIMIT["backfill"][0], idle=STEP_LIMIT["backfill"][1])
    if rc != 0:
        log.error("补数据失败（退出码 %d），今天不出清单", rc)
        return rc
    # 两道覆盖闸抽成 daily_ready，长期调整突破用同一份（教训 34）
    ok, why = daily_ready(d)
    if not ok:
        log.error("%s，不出清单", why)
        return 1
    log.info("%s", why)
    # 补数据那一步自己记的账。短几只是常态，成片补不上就不该出清单：
    # 那天的横截面百分位、板块中性化、市场宽度都在残缺的池子里算。
    short, fresh, note = _update_short(d)
    log.info("%s", note)
    if short or fresh:
        miss = short + fresh
        log.warning("有 %d 只没拉到 %s：%s", len(miss), d, ",".join(miss[:20]))
    # 上次的遗留，这一轮不许带进去
    os.environ.pop("BREAKOUT_SHORT_EXCLUDE", None)
    if len(short) > UPDATE_SHORT_MAX:
        if not short_gate_late(d):
            log.error("补不上目标日的票 %d 只（新股不算），超过 %d 只的上限，这一轮不出清单；"
                      "%s 次日 %02d:%02d 起不再挡，把它们剔出清单照发",
                      len(short), UPDATE_SHORT_MAX, d, *SHORT_LATE)
            return 1
        log.warning("补不上目标日的票 %d 只，超过 %d 只的上限，但已经过了 %s 次日 "
                    "%02d:%02d：这 %d 只剔出清单，其余照发",
                    len(short), UPDATE_SHORT_MAX, d, *SHORT_LATE, len(short))
        os.environ["BREAKOUT_SHORT_EXCLUDE"] = ",".join(short)
    rc = py("src/breakout/build.py", timeout=STEP_LIMIT["build"][0])
    if rc != 0:
        log.error("特征表没建出来（退出码 %d），今天不出清单", rc)
        return rc

    step(3, n, "打分 + 出清单")
    rc = py("src/breakout/daily.py", "--stage", "scan",
            timeout=STEP_LIMIT["scan"][0])
    if rc != 0:
        return rc

    # 市场环境 + 历史清单真值。以前只有学习会诊（每周一次）顺手算，早盘系统
    # 2026-09-27 归档后搬到这里每天算：邮件里的「近期全市场基准」读
    # state/regime_daily.jsonl，控制台首页那张「每份清单 20 天内涨超 50%」读
    # state/breakout/truth.json。研究性步骤，失败不挡发信（fail-open）。
    truth_and_regime(dry)

    step(4, n, "面板 + 邮件")
    if dry:
        os.environ["SKIP_MAIL"] = "1"
    rc = py("src/breakout/daily.py", "--stage", "send",
            timeout=STEP_LIMIT["send"][0])
    # 退出码 0 不等于发了信（教训 27）：send 阶段的 SKIP_MAIL 分支、
    # 以及任何「面板建好了但 send_mail 没走到」的路都返回 0。只认
    # daily.py 在 send_mail 之后落的 out_breakout/mail_sent.json，
    # 和 flow_pullback 同一个写法。标记不该在没发信的日子推出去：云端 20:30
    # 的 evening_check 看 origin 上有没有 sent 标记，推了它就不再提醒。
    sent_at = _mail_sent_date("out_breakout/mail_sent.json", d)
    if rc == 0 and (sent_at or dry):
        push_marker("sent", "breakout", {"ok": True}, dry, d)
    elif rc == 0:
        log.error("send 退出码 0 但 out_breakout/mail_sent.json 不是 %s 的，"
                  "不推 sent 标记（云端 20:30 会发「本机没跑」提醒）", d)
        rc = 1
    push_all(f"起涨预测 {d} [local]",
             [f"data/breakout/{d[:7]}", "out_breakout", "state/breakout"], dry)
    return rc


def arm_deadline(flow: str) -> threading.Timer:
    """整条流程的总时限：跑到 MAX_RUN 前 5 分钟还没结束，就结束自己的整棵进程树。

    STEP_LIMIT 管的是子阶段；local_run 自己也可能卡在某个没有时限的调用里。
    这一道兜住它：先写进度、放锁，再结束自己和全部子进程，下一次触发重跑
    （2026-09-29，教训 43）。正常的流程远活不到这里（MAX_RUN 盖得住各步时限之和）。
    """
    sec = MAX_RUN.get(flow, 3) * 3600 - 300

    def fire() -> None:
        log.error("%s 跑了 %.1f 小时还没结束，当作卡死：结束自己和全部子进程，"
                  "下一次触发重跑", FLOWS[flow][1], sec / 3600)
        try:
            _PROG.update({"running": False, "rc": TIMEOUT_RC,
                          "finished_at": now_bj().isoformat(timespec="seconds")})
            _write_progress(force=True)
            release_lock(flow)
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        _kill_tree(os.getpid())
        os._exit(TIMEOUT_RC)

    t = threading.Timer(sec, fire)
    t.daemon = True
    t.start()
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=sorted(FLOWS))
    ap.add_argument("--sync", action="store_true",
                    help="只拉远端产物到本地，不跑任何流程")
    ap.add_argument("--dry", action="store_true",
                    help="只跑不发不推（测试）")
    ap.add_argument("--if-needed", action="store_true",
                    help="目标日这条线已跑完（或不在窗口、周末）就直接退出 0")
    a = ap.parse_args()

    if a.sync:
        # 有流程在跑时不动工作区，那是正常情况不是失败，退出 0；
        # 只有真的拉不下来才退出 1（计划任务的「上次结果」会显示出来）
        if any_flow_running():
            sync_repo()
            return 0
        return 0 if sync_repo() else 1
    if not a.flow:
        ap.error("--flow 或 --sync 二选一")

    name = FLOWS[a.flow][1]
    if a.if_needed:
        reason = if_needed_skip(a.flow)
        if reason:
            log.info("%s：%s，跳过", name, reason)
            # 「时候未到」的两种：顺手拉一次远端，让本地面板跟上远端。
            # 周末/在跑/已跑完都不动工作区。
            if reason.startswith(SYNC_ON):
                sync_repo()
            return 0
        # 开跑前先拉远端再核对一次：另一台机器可能已经跑完推上去了
        if sync_repo() and already_done(a.flow):
            log.info("%s 目标日 %s 远端已经有了，产物已同步到本地面板，跳过",
                     name, target_date(a.flow))
            return 0

    if not acquire_lock(a.flow):
        return 0
    global _FLOW
    _FLOW = a.flow
    _PROG.clear()
    _PROG.update({"flow": a.flow, "running": True, "pid": os.getpid(),
                  "dry": bool(a.dry), "step": 0, "total": 0, "text": "启动",
                  "started_at": now_bj().isoformat(timespec="seconds")})
    _write_progress(force=True)
    keep_awake(True)
    deadline = arm_deadline(a.flow)
    rc = 1
    try:
        load_env()
        if a.dry:
            os.environ["DRY_RUN"] = "1"      # 子阶段据此在 run_meta 里标 dry
        t0 = now_bj()
        log.info("本地全流程 %s 启动 @ %s%s", a.flow,
                 t0.strftime("%H:%M:%S"), "（dry-run）" if a.dry else "")
        rc = {"breakout": flow_breakout, "pullback": flow_pullback}[a.flow](a.dry)
        log.info("总耗时 %.0f 秒，退出码 %d",
                 (now_bj() - t0).total_seconds(), rc)
        return rc
    finally:
        deadline.cancel()
        keep_awake(False)
        _PROG.update({"running": False, "rc": rc,
                      "finished_at": now_bj().isoformat(timespec="seconds")})
        _write_progress(force=True)
        release_lock(a.flow)


if __name__ == "__main__":
    sys.exit(main())
