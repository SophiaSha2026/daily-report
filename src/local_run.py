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
import datetime as dt
import json
import logging
import os
import socket
import subprocess
import sys
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
        try:
            import datasource as ds
            _TD["s"] = set(ds.trade_dates())
        except Exception as e:  # noqa: BLE001
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
    """tools/local.env -> os.environ。已存在的环境变量优先，不覆盖。"""
    p = ROOT / "tools" / "local.env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v and "FILLME" not in v and k not in os.environ:
            os.environ[k] = v


def sh(*args: str, check: bool = True, quiet: bool = False) -> int:
    r = subprocess.run(list(args), cwd=ROOT,
                       capture_output=quiet, text=True, encoding="utf-8",
                       errors="replace")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])} 退出码 {r.returncode}: "
                           f"{(r.stderr or '')[:200]}")
    return r.returncode


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


def git_commit_push(msg: str, paths: list[str], dry: bool) -> bool:
    """提交指定路径并推送到远端。返回「是否真的推上去了」。

    完全本地化之后（2026-09-12）远端不再自动跑，正常日子里 fetch 完直接
    就能 push。但手动 dispatch 应急仍可能在远端落 commit，所以 rebase
    那条路留着，且按**本地优先**自动解冲突：本地是唯一的执行方，
    它的产物就是权威版本。rebase 语义下 upstream 是 ours、正在 replay 的
    本地提交是 theirs，所以 -X theirs 才是「取本地」，别写反。
    """
    if dry:
        log.info("[dry] 跳过提交推送: %s", msg)
        return False
    try:
        _git_unstick()
        for x in paths:
            _git("add", "-A", x)
        if _git("diff", "--cached", "--quiet")[0] == 0:
            log.info("没有需要提交的产物")
            return True
        rc, out = _git("commit", "-q", "-m", msg)
        if rc != 0:
            _write_push_status(False, msg, f"commit 失败: {out}")
            log.warning("提交失败: %s", out[:200])
            return False

        last = ""
        for attempt in range(3):
            if _git("push", "-q", "origin", "main")[0] == 0:
                _write_push_status(True, msg)
                log.info("已推送: %s", msg)
                return True
            _git("fetch", "-q", "origin", "main")
            rc, last = _git("rebase", "--autostash", "-X", "theirs",
                            "origin/main")
            if rc != 0:
                log.warning("第 %d 次 rebase 失败: %s", attempt + 1,
                            last[:200])
                _git_unstick()

        _write_push_status(False, msg, last)
        log.warning("推送失败（本地已提交，产物没丢）。GUI 的同步卡片会亮红，"
                    "点「重试推送」或手动 git push origin main")
        return False
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
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "host": socket.gethostname(),
               "at": now_bj().isoformat(timespec="seconds")}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    if dry:
        log.info("[dry] 标记只写本地不推送: %s", p.name)
        return
    git_commit_push(f"{kind}: {flow} {date} [local]",
                    [p.relative_to(ROOT).as_posix()], dry)


# ---------------------------------------------------------------------
#  进程锁：同一条线同一时刻只跑一个实例
# ---------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259      # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def lock_path(flow: str) -> Path:
    return ROOT / "state" / "lock" / f"{flow}.json"


def _scan_processes(flow: str) -> dict | None:
    """锁之外再看一眼进程表：有没有别的 `local_run.py --flow <flow>` 在跑。

    锁文件可能被手删、也可能是旧版本代码起的进程根本没写锁（2026-09-14
    晚上就是）。进程表是事实，锁只是快捷方式。只在 Windows 上做，
    别处返回 None。
    """
    if sys.platform != "win32":
        return None
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { '' + $_.ProcessId + '|' + $_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:  # noqa: BLE001
        return None
    for line in (r.stdout or "").splitlines():
        pid, _, cmd = line.partition("|")
        if not pid.strip().isdigit() or int(pid) == os.getpid():
            continue
        if "local_run.py" in cmd and f"--flow {flow}" in cmd:
            return {"pid": int(pid), "flow": flow, "at": "", "source": "进程表"}
    return None


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
    if info:
        try:
            age = (now_bj() - dt.datetime.fromisoformat(info["at"])).total_seconds()
        except Exception:  # noqa: BLE001
            age = 0
        if (age <= MAX_RUN.get(flow, 3) * 3600
                and int(info.get("pid", 0)) != os.getpid()
                and _pid_alive(int(info.get("pid", 0)))):
            return info
    return _scan_processes(flow)


def acquire_lock(flow: str) -> bool:
    other = running_instance(flow)
    if other:
        log.info("%s 已经在跑（pid %s，%s 起），本实例退出",
                 FLOWS[flow][1], other.get("pid"), other.get("at", "")[11:19])
        return False
    p = lock_path(flow)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pid": os.getpid(), "flow": flow,
                             "at": now_bj().isoformat(timespec="seconds"),
                             "argv": sys.argv[1:]}, ensure_ascii=False),
                 encoding="utf-8")
    return True


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
    _git_unstick()
    before = _git("rev-parse", "HEAD")[1]
    rc, out = _git("pull", "--rebase", "--autostash", "-q", "origin", "main")
    if rc != 0:
        log.warning("拉远端失败: %s", out[:200])
        _git_unstick()
        return False
    after = _git("rev-parse", "HEAD")[1]
    if after != before:
        log.info("已拉取远端 %s -> %s", before[:7], after[:7])
    else:
        log.info("远端没有新东西")
    return True


def push_all(msg: str, paths: list[str], dry: bool) -> None:
    git_commit_push(msg, paths, dry)


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
    step(1, n, "同步仓库（拿远端可能已建好的候选池）")
    if not dry:
        sh("git", "pull", "--rebase", "-q", "origin", "main", check=False)
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
        py("src/premarket.py")
    elif need:
        log.warning("候选池不是今天的且时间太晚，quick 阶段会走缺失分支")
    else:
        log.info("候选池已是今天的")

    step(3, n, "采样 + 打分（自动等到 09:14 预热、09:19/09:23/09:25 采样）")
    late = (bj.hour, bj.minute) >= (9, 27)
    rc = py("src/run_auction.py", "--stage", "quick",
            *(["--late"] if late else []))
    if rc != 0:
        log.error("quick 阶段失败（退出码 %d），不推 sent 标记，远端会兜底", rc)
        return rc

    step(4, n, "推送数据快照（远端 yield_check 以此判断本地活着）")
    d = today()
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
    if rc == 0:
        push_marker("sent", "auction", {"ok": True}, dry)
        push_all(f"out: {d} [local]", ["out"], dry)
        log.info("完成。远端看到 sent 标记后只发布面板不发邮件。")
    else:
        log.error("enrich 失败（退出码 %d），远端将在 09:28:20 兜底发信", rc)
    return rc


def flow_evening(dry: bool) -> int:
    """形态线 + 学习线。17:00 后数据定型，几点跑都一样，防早不防晚。"""
    n = 5
    step(1, n, "同步仓库")
    if not dry:
        sh("git", "pull", "--rebase", "-q", "origin", "main", check=False)
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

    flow_learn(dry, base=3, total=n)
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
            _git_unstick()
            _git("pull", "--rebase", "--autostash", "-q", "origin", "main")
        base = 1

    step(base + 1, total, "学习线：标签 -> 归因 -> 拟合与闸门")
    rc = py("src/eval_daily.py", "--stage", "all", "--date", d,
            *(["--dry"] if dry else []))
    if rc != 0:
        log.warning("学习线退出码 %d（研究性步骤，不影响业务邮件）", rc)

    step(base + 2, total, "推送学习产物")
    # learn.html 也推：它进 Pages 站点，不推就没人看得到今天的裁决
    push_all(f"learn: {d} [local]",
             ["data/labels", "state", "out_learn/learn.html"], dry)
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


def already_done(flow: str) -> bool:
    """目标日这条线是否已经跑完。

    计划任务每 15~30 分钟重试一次，一直敲到窗口关（机器可能整段时间都
    不在线，2026-08-27 就因为笔记本没醒漏发过）。代价是跑完之后它还会
    继续敲，所以入口必须幂等，否则每次重试都重发一封邮件。
    """
    rel = FLOWS[flow][0]
    try:
        got = json.loads((ROOT / rel).read_text(encoding="utf-8"))["date"]
        return got == target_date(flow)
    except Exception:  # noqa: BLE001
        return False


def flow_breakout(dry: bool) -> int:
    """晚间系统：起涨预测。补当天日线 -> 特征表 -> 打分 -> 面板 + 邮件。

    目标日是最近一个已收盘的交易日。补数据走腾讯快照增量（秒级），
    追加不到目标日就**不出清单**：拿旧数据重算只会把上一个交易日的
    清单再发一遍，而且 run_meta 的日期对不上目标日，下次重试还会再发。
    """
    n = 4
    d = target_date("breakout")
    step(1, n, f"同步仓库（目标日 {d}）")
    if not dry:
        _git_unstick()
        _git("pull", "--rebase", "--autostash", "-q", "origin", "main")
    push_marker("claim", "breakout", {"plan": "本地接管今日起涨预测"}, dry, d)

    step(2, n, f"补 {d} 日线 + 重算特征表")
    rc = py("src/breakout/backfill.py", "--stage", "update")
    if rc != 0:
        log.error("补数据失败（退出码 %d），今天不出清单", rc)
        return rc
    try:
        import pandas as pd
        mx = str(pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                                 columns=["date"])["date"].max())
    except Exception as e:  # noqa: BLE001
        log.error("读不到日线: %s", e)
        return 1
    if mx != d:
        log.error("日线只到 %s，目标日是 %s，不出清单", mx, d)
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
        # 周末先挡掉，省得每 15 分钟就去拉一次交易日历。节假日挡不住，
        # 但各阶段内部都有 trade_dates() 判断，跑起来会自己退出 0。
        if now_bj().weekday() >= 5:
            log.info("北京时间周末，跳过")
            return 0
        if running_instance(a.flow):
            log.info("%s 正在跑，跳过", name)
            return 0
        if already_done(a.flow):
            log.info("%s 目标日 %s 已经跑完，跳过", name, target_date(a.flow))
            return 0
        if not in_window(a.flow):
            lo, hi = FLOWS[a.flow][2:4]
            log.info("%s 现在 %s 不在开跑窗口 %02d:%02d-%02d:%02d（北京），跳过",
                     name, now_bj().strftime("%H:%M"), *lo, *hi)
            # 窗口外也顺手拉一次远端：云端替本地跑过的产物落到本地面板
            sync_repo()
            return 0
        if not auto_due(a.flow):
            auto = FLOWS[a.flow][4]
            log.info("%s 现在 %s 还没到自动开跑时刻 %02d:%02d（北京），先等手动；"
                     "到点没手点就自动跑", name, now_bj().strftime("%H:%M"), *auto)
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
