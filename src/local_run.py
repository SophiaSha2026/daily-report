"""
本地全流程编排器。TUI 的「一键跑」按的就是它。

    python src/local_run.py --flow morning    竞价线：候选池->采样->LLM->发信
    python src/local_run.py --flow evening    形态线+学习线：扫描->发信->评估
    加 --dry 只跑不发不推（测试用）

本地为主、云端托底（2026-09-15 起）
----------------------------------
三档优先级，用户 2026-09-15 定的：

    1. 手动     控制台点按钮，随时（在开跑窗口内）
    2. 本机自动  到了「自动开跑时刻」还没手点，计划任务自己跑
    3. 云端     本机根本没跑（没开机），GitHub cron + Cloudflare Worker 叫起云端兜底

计划任务从窗口一开就每 15~30 分钟敲一次，但自动开跑时刻之前只拉远端、
不开跑，把时间留给手动。手动和自动是同一个入口、同一把锁，谁先起谁跑。
跑完把产物 commit + push，云端看到 sent 标记就让位；Pages 面板由云端发布。

    --flow morning   竞价线：候选池 -> 采样 -> LLM -> 09:27:30 发信
    --flow breakout  晚间系统：补当天日线 -> 特征 -> 打分 -> 17:00 后发信
    --flow learn     只跑学习线，收盘后排期跑的就是它
    --flow evening   形态线 + 学习线（形态报告已停用，手动想看时才跑）
    --sync           只拉远端（云端替本地跑过的产物落到本地面板），不跑流程
    --if-needed      目标日这条线已经跑完、或不在开跑窗口，直接退出 0
                     （计划任务每 15~30 分钟重试一次，入口必须幂等，
                     否则会重复发信）

目标日不等于今天
----------------
晚间系统和学习线的「目标日」是最近一个**已收盘**的交易日：北京 09-15
早上 07:00 补跑，目标日仍是 09-14，数据和 09-14 下午 17:00 跑一模一样。
所以它们的开跑窗口跨过午夜（16:00 到次日 08:30），机器整天没开、
晚上（美东）才醒过来也能把当天的清单补出来，赶在下一个交易日开盘前。
竞价线的目标日永远是今天：09:16 之后新起进程来不及赶上 09:25 采样。

同一时刻一条线只能有一个实例
----------------------------
计划任务和控制台按钮都走这里。2026-09-14 早上两边各起了一个竞价线
（用户手点 + 计划任务到点），两个进程各采各的样、各发各的信，用户收到
两封一模一样的邮件，git 还互相撞。现在 state/lock/<flow>.json 是进程锁：
后来者看到活着的锁就退出 0，控制台上显示「已经在跑」。

云端托底协议（另一半在 tools/yield_check.py 和 .github/workflows/）
----------------------------------------------------------------
    state/claim/<flow>_<date>.json   本地开跑时推送：「我来」
    state/sent/<flow>_<date>.json    本地发信成功后推送：「我发了」
竞价线：云端 07:40 起照常采样，09:27:00 看到 claim 就等到 09:28:20 确认
sent，有 sent 只发布面板不发邮件、也不提交数据；没 claim 或没 sent 就
云端发。晚间系统云端算不了（特征表 2.5GB 在本机，新浪源云端也不通），
云端 20:30 只做一件事：看 origin/main 上有没有目标日的 out_breakout，
没有就发一封「本机今天没跑」的提醒。

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
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
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
    """这条线这次要产出哪一天。竞价线是今天，其余是最近一个已收盘交易日。"""
    return today() if flow == "morning" else last_closed_trade_day()


def step(i: int, n: int, text: str) -> None:
    print(f"##STEP {i}/{n} {text}", flush=True)


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


def py(*args: str) -> int:
    """跑一个子阶段，stdout/stderr 直接透传（TUI 靠这个显示实时进度）。"""
    return subprocess.run([PY, *args], cwd=ROOT).returncode


# ---------------------------------------------------------------------
#  git 标记
# ---------------------------------------------------------------------
def _git(*args: str) -> tuple[int, str]:
    """跑一条 git，返回 (退出码, 合并后的输出)。从不抛异常。"""
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


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


class GitBusy(RuntimeError):
    """别的流程正占着 git 工作区，这次放弃。"""


GIT_LOCK_MAX = 600      # 锁比这个老（秒）就当持锁进程死了没清


@contextlib.contextmanager
def git_lock(timeout: float = 300.0):
    """跨流程的 git 互斥。整个仓库只有一个工作区，两条线却是并行跑的。

    2026-09-16 实测起涨预测和学习线同一秒启动，两边各自 pull --rebase
    --autostash、各自 add/commit。autostash 在对方正写产物的瞬间 stash/pop，
    pop 冲突时 git 整体放弃 stash，对方没提交的当天面板和清单就被还原成
    前一天的版本，而推送那一步只会报「没有需要提交的产物」。
    锁只串得住 git 命令本身，串不住对方的写盘，所以推送路径同时不再用
    autostash（见 git_commit_push）。
    """
    p = ROOT / "state" / "lock" / "git.json"
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
                     and age <= GIT_LOCK_MAX)
            if alive and time.time() < deadline:
                time.sleep(0.5)
                continue
            if alive:
                raise GitBusy(f"git 工作区被 pid {holder} 占着超过 {timeout:.0f} 秒")
            # 持锁进程已经死了、或锁太老：清掉重来（教训 15，自愈路径本身要能自愈）
            try:
                p.unlink()
            except OSError:
                pass
            if time.time() >= deadline:
                raise GitBusy("拿不到 git 锁")
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
                    # 失败：推送列表里有几条是**可选产物**，比如学习线的
                    # out_learn/council.html 只在会诊真跑过的那天才有。
                    # 按失败处理的话，那一天连 learn.html 和 state/ 都推不上去。
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

    云端 workflow 的 tools/yield_check.py 读它们决定让不让位。
    date 是目标日，不传就是今天（竞价线）。
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


def _parse_scan(lines: list[str], flow: str, now_utc: dt.datetime,
                own_pid: int) -> dict | None:
    """把 PowerShell 那几行 `pid|创建时刻|命令行` 解析成「谁在跑」。

    抽成纯函数是为了能离线自测：真机上进程表里可能正好有一条线在跑，
    结果不可复现。
    """
    for line in lines:
        pid, _, rest = line.partition("|")
        created, _, cmd = rest.partition("|")
        if not pid.strip().isdigit() or int(pid) == own_pid:
            continue
        if "local_run.py" not in cmd or f"--flow {flow}" not in cmd:
            continue
        at = ""
        try:
            c = dt.datetime.strptime(created.strip(), "%Y-%m-%dT%H:%M:%S")
            c = c.replace(tzinfo=dt.timezone.utc)
            if (now_utc - c).total_seconds() < PROBE_GRACE:
                continue                      # 还在判定阶段的探针，不算
            at = c.astimezone(dt.timezone(dt.timedelta(hours=8))
                              ).isoformat(timespec="seconds")
        except ValueError:
            pass    # 时刻问不出来：按老规矩算它在跑（宁可多退一次，别双发）
        return {"pid": int(pid), "flow": flow, "at": at, "source": "进程表"}
    return None


def _scan_processes(flow: str) -> dict | None:
    """锁之外再看一眼进程表：有没有别的 `local_run.py --flow <flow>` 在跑。

    锁文件可能被手删、也可能是旧版本代码起的进程根本没写锁（2026-09-14
    晚上就是）。进程表是事实，锁只是快捷方式。只在 Windows 上做，
    别处返回 None。
    """
    if sys.platform != "win32":
        return None
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
        return None
    return _parse_scan((r.stdout or "").splitlines(), flow,
                       dt.datetime.now(dt.timezone.utc), os.getpid())


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


def acquire_lock(flow: str) -> bool:
    """抢这条线的锁。拿到返回 True，别人在跑返回 False。

    必须原子：以前是「先 running_instance 查一遍、再 write_text」，
    同一秒起的两个实例（控制台点击撞上计划任务）双双查到「没人在跑」，
    后写的那个把前一个的锁覆盖掉。O_EXCL 保证只有一个能建出文件。
    """
    other = running_instance(flow)
    if other:
        log.info("%s 已经在跑（pid %s%s），本实例退出",
                 FLOWS[flow][1], other.get("pid"),
                 f"，{other.get('at', '')[11:19]} 起" if other.get("at") else "")
        return False
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


def _mail_sent_today(date: str) -> str:
    """竞价清单今天发出去了没有。发了返回发信时刻，没发返回空串。

    run_auction 真把邮件交给 SMTP 之后才写 out/mail_sent.json，
    这是「发过了」的唯一证据（enrich 有四条不发信也返回 0 的分支）。
    """
    try:
        ms = json.loads((ROOT / "out" / "mail_sent.json")
                        .read_text(encoding="utf-8"))
        return str(ms.get("at") or "已发") if ms.get("date") == date else ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------
#  本地 LLM 文案（竞价线的 reason/risk 两句话）
# ---------------------------------------------------------------------
def local_commentary(timeout: int = 150) -> bool:
    """out/brief.json -> claude CLI -> out/commentary.json。

    远端这一步是 claude-code-action；本地直接调 CLI（OAuth 自动识别在
    llm_local 里）。失败就算了——enrich 拿不到 commentary 会按
    「本次无 LLM 分析」照发，这是硬约束 2。
    """
    try:
        from learn import llm_local
        brief_p = ROOT / "out" / "brief.json"
        if not brief_p.exists():
            log.info("没有 brief.json，跳过 LLM 文案")
            return False
        # brief.json 是被跟踪文件，永远存在：非交易日、超死线、候选池缺失
        # 那几条路上 quick 根本没写今天的 run_meta，这里如果照跑，就是拿
        # 上一个交易日的 brief 再调一次 Opus，把 out/commentary.json 覆盖成
        # 和已发邮件对不上的文案（每个节假日 4 次，每次约 40 秒）。
        if _run_meta_date("out/run_meta.json") != today():
            log.info("quick 没有产出今天的清单，不调 LLM")
            return False
        if not llm_local.available():
            log.info("本机没有 claude CLI，按「本次无 LLM 分析」发")
            return False
        brief = brief_p.read_text(encoding="utf-8")
        prompt = ((ROOT / "prompts" / "analyst.md").read_text(encoding="utf-8")
                  + "\n\n# brief.json\n```json\n" + brief + "\n```\n\n"
                  + "# 输出\n只输出一个 JSON 对象，不要围栏不要解释："
                  + '{"<code>": {"reason": "...", "risk": "..."}} '
                  + "覆盖 brief 里每一只。")
        envj, err = llm_local._run_cli(prompt, "claude-opus-5", timeout)
        if err:
            log.info("LLM 文案失败（照发无文案版）: %s", err[:100])
            return False
        obj = llm_local._extract_json(
            envj.get("result", "") if isinstance(envj, dict) else "")
        if not isinstance(obj, dict) or not obj:
            log.info("LLM 文案解析失败（照发无文案版）")
            return False
        clean = {str(k).zfill(6): {"reason": str(v.get("reason", ""))[:60],
                                   "risk": str(v.get("risk", ""))[:50]}
                 for k, v in obj.items() if isinstance(v, dict)}
        (ROOT / "out" / "commentary.json").write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("LLM 文案就绪：%d 只", len(clean))
        return True
    except Exception as e:  # noqa: BLE001
        log.info("LLM 文案异常（照发无文案版）: %s", e)
        return False


# ---------------------------------------------------------------------
#  两条流程
# ---------------------------------------------------------------------
def flow_morning(dry: bool) -> int:
    """竞价线。开跑时刻不限（内部各阶段自己等到点），09:26:30 前都来得及。"""
    n = 6
    d = today()
    # 发过信就不再跑。手动入口不带 --if-needed（手动优先，不该被窗口和
    # 自动时刻挡），所以发完信再点一下，09:27 前会多发一封告警 + 一封重复
    # 清单，09:27 后走抢救模式把当天真实的 09:25 快照覆盖掉再发一封
    # （学习线随后按 T1=T2=T3 占比把这一天整天丢弃）。这道门只看
    # 「今天发过没有」，锁和窗口都管不了它。
    sent_at = _mail_sent_today(d)
    if not dry and sent_at:
        log.info("今日竞价清单已于 %s 发出，不重跑不重发。"
                 "确要重发先删 out/mail_sent.json", sent_at)
        return 0
    step(1, n, "同步仓库（拿远端可能已建好的候选池）")
    if not dry:
        sync_repo()
    push_marker("claim", "auction", {"plan": "本地接管今日竞价线"}, dry)

    step(2, n, "候选池")
    meta = ROOT / "cache" / "universe_meta.json"
    need = True
    try:
        need = json.loads(meta.read_text(encoding="utf-8"))["date"] != today()
    except Exception:  # noqa: BLE001
        pass
    bj = now_bj()
    if need and (bj.hour, bj.minute) < (9, 8):
        log.info("候选池不是今天的，现在建（3-5 分钟）")
        if py("src/premarket.py") != 0:
            log.error("候选池构建失败，quick 阶段会因候选池不是今天的而放弃，远端兜底")
    elif need:
        log.warning("候选池不是今天的且时间太晚，quick 阶段会放弃（不拿旧池子发脏清单），远端兜底")
    else:
        log.info("候选池已是今天的")

    step(3, n, "采样 + 打分（自动等到 09:14 预热、09:19/09:23/09:25 采样）")
    late = (bj.hour, bj.minute) >= (9, 27)
    rc = py("src/run_auction.py", "--stage", "quick",
            *(["--late"] if late else []))
    if rc != 0:
        log.error("quick 阶段失败（退出码 %d），不推 sent 标记，远端会兜底", rc)
        return rc
    # 退出码 0 不等于做了事（教训 22）：非交易日、超死线、候选池不是今天的
    # 三条路 quick 都是 return 0 且不写 run_meta。再往下走就是拿上一个交易日
    # 的 brief 调一次 Opus、把 out/ 里的旧产物再提交推送一遍。
    if _run_meta_date("out/run_meta.json") != d:
        tds = trade_dates()
        holiday = bool(tds) and d not in tds
        log.info("quick 没有产出 %s 的清单（%s），不进 LLM、不发信", d,
                 "非交易日" if holiday else "超死线或候选池缺失")
        return 0 if holiday else 1

    step(4, n, "推送数据快照（远端 yield_check 以此判断本地活着）")
    # 候选池也提交：云端替本地跑的那天会提交它，本地留着未提交的同名
    # 文件下次 pull --autostash 就会撞上。本地跑过就以本地为准。
    push_all(f"data: {d} [local]",
             [f"data/{d[:7]}", "cache/universe.parquet",
              "cache/universe_meta.json"], dry)

    step(5, n, "LLM 文案（失败照发）")
    local_commentary()

    step(6, n, "面板 + 发信（等到 09:27:30 那一秒）")
    if dry:
        os.environ["SKIP_MAIL"] = "1"
        log.info("[dry] 发信被跳过")
    rc = py("src/run_auction.py", "--stage", "enrich")
    # 退出码 0 不等于发了信：enrich 有四条不发信的分支也返回 0。
    # 只认 run_auction 真发出去之后落的 out/mail_sent.json。
    sent_ok = bool(_mail_sent_today(d))
    if rc == 0 and (sent_ok or dry):
        push_marker("sent", "auction", {"ok": True}, dry)
        push_all(f"out: {d} [local]", ["out"], dry)
        log.info("完成。远端看到 sent 标记后只发布面板不发邮件。")
    elif rc == 0:
        log.error("enrich 退出码 0 但没有发信记录，不推 sent 标记，远端将兜底发信")
        push_all(f"out: {d} [local]", ["out"], dry)
        rc = 1
    else:
        log.error("enrich 失败（退出码 %d），远端将在 09:28:20 兜底发信", rc)
    return rc


def flow_evening(dry: bool) -> int:
    """形态线 + 学习线。17:00 后数据定型，几点跑都一样，防早不防晚。"""
    n = 5
    step(1, n, "同步仓库")
    if not dry:
        sync_repo()
    push_marker("claim", "pullback", {"plan": "本地接管今日形态线"}, dry)

    step(2, n, "形态扫描（未到 17:00 会自动等）")
    rc = py("src/pullback.py", "--stage", "scan", *(["--dry"] if dry else []))
    if rc != 0:
        log.error("扫描失败（退出码 %d），远端会兜底", rc)
        return rc

    step(3, n, "发信")
    if dry:
        log.info("[dry] 跳过发信")
    else:
        rc = py("src/pullback.py", "--stage", "send")
        if rc == 0:
            push_marker("sent", "pullback", {"ok": True}, dry)
        else:
            log.error("发信失败（退出码 %d），远端将兜底", rc)
    d = today()
    push_all(f"data: 形态 {d} [local]", [f"data/{d[:7]}", "out_pullback"], dry)

    # 学习线在这里是内嵌调用，主进程的锁是 evening 的，进程表里也只看得到
    # `--flow evening`。计划任务的 DailyReport-Local-Learn 每 30 分钟敲一次、
    # 登录也敲，两边会同时起两份 eval_daily --stage all：theta_history 同日双行、
    # 两封参数变更邮件、会诊双跑（一次约 $5）。所以嵌跑前自己把 learn 的锁拿上，
    # 让对面的 --if-needed 看得见。
    if already_done("learn"):
        log.info("学习线目标日 %s 已跑完，不嵌跑", target_date("learn"))
        return rc
    if not acquire_lock("learn"):
        return rc              # acquire_lock 已经打过「已经在跑」
    try:
        flow_learn(dry, base=3, total=n)
    finally:
        release_lock("learn")
    return rc


def flow_learn(dry: bool, base: int = 0, total: int = 3) -> int:
    """学习线：标签 -> 归因 -> 拟合与闸门 -> 推产物。

    2026-09-12 形态报告停用后，这条线单独排期跑。它调的是**竞价线**的
    参数，和形态线没有任何依赖关系，不该被一起停掉。

    base/total 让它既能独立跑（1/3、2/3、3/3），又能嵌在 evening 里
    接着前面的步号往下数，TUI 和 GUI 的进度条才不会倒退。
    """
    d = target_date("learn")
    if base == 0:
        step(1, total, "同步仓库")
        if not dry:
            sync_repo()
        base = 1

    step(base + 1, total, "学习线：标签 -> 归因 -> 拟合与闸门")
    rc = py("src/eval_daily.py", "--stage", "all", "--date", d,
            *(["--dry"] if dry else []))
    if rc != 0:
        log.warning("学习线退出码 %d（研究性步骤，不影响业务邮件）", rc)

    step(base + 2, total, "推送学习产物")
    # learn.html 也推：它进 Pages 站点，不推就没人看得到今天的裁决
    push_all(f"learn: {d} [local]",
             ["data/labels", "state", "out_learn/learn.html",
              "out_learn/council.html"], dry)
    return rc


# 「目标日这条线跑完了没有」读哪个文件，以及 --if-needed 允许开跑的北京时间窗口
#
# 窗口是给自动触发用的，手动跑（不带 --if-needed）不受限制。
# 上界小于下界表示跨午夜。各条线上界的含义：
#   morning  09:16 之后新起一个进程来不及赶上 09:25 的竞价采样，
#            硬跑只会撞上 hard_deadline 然后发一封告警邮件。计划任务带
#            「登录时触发」，不设上界的话盘后每次开机登录都发一封告警。
#   breakout 收盘后数据定死，目标日是最近一个已收盘交易日，所以可以一直
#            补到次日 08:30：机器整天没开、美东晚上才醒也能赶在开盘前出清单。
#            再晚就撞上竞价线的采样（09:14 起），不抢那几分钟。
#   learn    同上，跨午夜到 08:30
#   evening  形态线的 hard_deadline 是 22:00（已停用自动，手动不受限）
# 第五项是「自动开跑时刻」（北京）：计划任务在这之前只等手动，到点没手点
# 才自己跑。手动不受它限制，只受窗口限制。
#   morning  08:30。手动窗口 06:00~08:30（美东晚 18:00~20:30，冬令时 17:00~19:30）。
#            08:30 起跑，候选池 10 分钟，09:14 预热前有余量；再晚候选池来不及。
#   breakout 16:30。收盘后半小时数据定型，17:00 前后出清单是这条线的约定；
#            美东凌晨没人手点，实际上就是自动跑，机器没醒就等醒了补。
#   learn    16:40，同上
FLOWS = {
    "morning": ("out/run_meta.json", "竞价线", (6, 0), (9, 16), (8, 30)),
    "evening": ("out_pullback/run_meta.json", "形态线", (16, 0), (22, 0), (16, 30)),
    "learn": ("state/learning_status.json", "学习线", (16, 0), (8, 30), (16, 40)),
    "breakout": ("out_breakout/run_meta.json", "起涨预测", (16, 0), (8, 30), (16, 30)),
}

# 一次最多跑多久（小时）。锁比这个老就当死锁，防 PID 重用误判
MAX_RUN = {"morning": 4, "breakout": 3, "learn": 3, "evening": 3}


def in_window(flow: str) -> bool:
    lo, hi = FLOWS[flow][2:4]
    hm = (now_bj().hour, now_bj().minute)
    if lo <= hi:
        return lo <= hm <= hi
    return hm >= lo or hm <= hi          # 跨午夜


def auto_due(flow: str) -> bool:
    """到没到自动开跑时刻。窗口内且过了 FLOWS 第五项。

    跨午夜的窗口（16:00~次日 08:30）里，自动时刻在午夜前那一段：
    16:30 之后算到点，午夜后到 08:30 也算到点（那是补跑）。
    """
    if not in_window(flow):
        return False
    lo, hi, auto = FLOWS[flow][2:5]
    hm = (now_bj().hour, now_bj().minute)
    if lo <= hi:
        return hm >= auto
    return hm >= auto or hm <= hi


# 哪条线真发信之后写哪个 sent 标记（state/sent/<名字>_<目标日>.json）
SENT_MARK = {"morning": "auction", "evening": "pullback", "breakout": "breakout"}


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
    if not meta.get("dry"):
        return True
    # 试跑（--dry）也写 run_meta。真跑完之后再点一次试跑，run_meta 就被
    # 标成 dry，以前这里直接返回 False，下一次计划任务敲门（起涨预测窗口
    # 16 小时、每 30 分钟一次）就把整条线重跑并再发一封同样的清单。
    # 试跑不写 sent 标记（push_marker 的 dry 分支），所以真跑完的证据只剩它。
    mark = SENT_MARK.get(flow)
    return bool(mark) and (ROOT / "state" / "sent" / f"{mark}_{target}.json").exists()


def already_done(flow: str) -> bool:
    """目标日这条线是否已经跑完。

    计划任务每 15~30 分钟重试一次，一直敲到窗口关（机器可能整段时间都
    不在线，2026-08-27 就因为笔记本没醒漏发过）。代价是跑完之后它还会
    继续敲，所以入口必须幂等，否则每次重试都重发一封邮件。
    """
    return done_for(flow, target_date(flow))


def weekend_skip(flow: str) -> bool:
    """周末拦截只管目标日是「今天」的线，也就是窗口不跨午夜的那几条。

    跨午夜的线（起涨预测、学习线）目标日是最近一个已收盘交易日，
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
    """
    if weekend_skip(flow):
        return "北京时间周末"
    # 节假日（工作日但不是交易日）：竞价线的目标日永远是今天，各阶段虽然
    # 自己会 return 0，但那之前已经推了一个 claim 提交、调了一次 Opus 把
    # commentary.json 覆盖成上一个交易日的文案。08:30~09:15 每 15 分钟一轮，
    # 一个节假日 4 轮。日历拿不到时不按节假日挡（fail-open）。
    tds = trade_dates()
    if flow == "morning" and tds and today() not in tds:
        return f"{today()} 非交易日"
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
    rc = py("src/breakout/backfill.py", "--stage", "update")
    if rc != 0:
        log.error("补数据失败（退出码 %d），今天不出清单", rc)
        return rc
    try:
        import pandas as pd
        dd = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                             columns=["date", "code"])
        cov, mx = daily_coverage(dd, d)
    except Exception as e:  # noqa: BLE001
        log.error("读不到日线: %s", e)
        return 1
    if cov < DAILY_COVER_MIN:
        log.error("日线只有 %.1f%% 的票到 %s（全表最新 %s），不出清单",
                  cov * 100, d, mx)
        return 1
    rc = py("src/breakout/build.py")
    if rc != 0:
        log.error("特征表没建出来（退出码 %d），今天不出清单", rc)
        return rc

    step(3, n, "打分 + 出清单")
    rc = py("src/breakout/daily.py", "--stage", "scan")
    if rc != 0:
        return rc

    step(4, n, "面板 + 邮件")
    if dry:
        os.environ["SKIP_MAIL"] = "1"
    rc = py("src/breakout/daily.py", "--stage", "send")
    if rc == 0 and not dry:
        push_marker("sent", "breakout", {"ok": True}, dry, d)
    push_all(f"起涨预测 {d} [local]",
             [f"data/breakout/{d[:7]}", "out_breakout", "state/breakout"], dry)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", choices=["morning", "evening", "learn", "breakout"])
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
            # 「时候未到」的两种：顺手拉一次远端，云端替本地跑过的产物
            # 落到本地面板。周末/非交易日/在跑/已跑完都不动工作区。
            if reason.startswith(SYNC_ON):
                sync_repo()
            return 0
        # 开跑前先拉远端再核对一次：云端可能已经替本地跑完了
        if sync_repo() and already_done(a.flow):
            log.info("%s 目标日 %s 云端已经跑过，产物已同步到本地面板，跳过",
                     name, target_date(a.flow))
            return 0

    if not acquire_lock(a.flow):
        return 0
    try:
        load_env()
        if a.dry:
            os.environ["DRY_RUN"] = "1"      # 子阶段据此在 run_meta 里标 dry
        t0 = now_bj()
        log.info("本地全流程 %s 启动 @ %s%s", a.flow,
                 t0.strftime("%H:%M:%S"), "（dry-run）" if a.dry else "")
        rc = {"morning": flow_morning, "evening": flow_evening,
              "learn": flow_learn, "breakout": flow_breakout}[a.flow](a.dry)
        log.info("总耗时 %.0f 秒，退出码 %d",
                 (now_bj() - t0).total_seconds(), rc)
        return rc
    finally:
        release_lock(a.flow)


if __name__ == "__main__":
    sys.exit(main())
