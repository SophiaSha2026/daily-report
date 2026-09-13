"""
本地全流程编排器。TUI 的「一键跑」按的就是它。

    python src/local_run.py --flow morning    竞价线：候选池->采样->LLM->发信
    python src/local_run.py --flow evening    形态线+学习线：扫描->发信->评估
    加 --dry 只跑不发不推（测试用）

完全本地化（2026-09-12 起）
--------------------------
云端所有 workflow 的 cron 已停用，仓库只做版本控制。这台机器是唯一的
执行方，跑完把产物 commit + push，顺带更新 GitHub Pages 面板。

    --flow morning   竞价线：候选池 -> 采样 -> LLM -> 09:27:30 发信
    --flow evening   形态线 + 学习线（形态报告已停用，手动想看时才跑）
    --flow learn     只跑学习线，收盘后排期跑的就是它
    --if-needed      今天这条线已经跑完就直接退出 0（计划任务每 15 分钟
                     重试一次，入口必须幂等，否则会重复发信）

state/claim 和 state/sent 两个标记仍然照写照推：云端 workflow 手动
dispatch 应急时，tools/yield_check.py 还是读它们决定让不让位。

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


def push_marker(kind: str, flow: str, payload: dict, dry: bool) -> None:
    """写 state/{claim,sent}/<flow>_<date>.json 并推送。

    云端 cron 停用后这两个标记已经不再影响谁发信（远端不会自己跑了），
    但手动 dispatch 应急时 tools/yield_check.py 仍然读它们，所以保留。
    """
    p = ROOT / "state" / kind / f"{flow}_{today()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "host": socket.gethostname(),
               "at": now_bj().isoformat(timespec="seconds")}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    if dry:
        log.info("[dry] 标记只写本地不推送: %s", p.name)
        return
    git_commit_push(f"{kind}: {flow} {today()} [local]",
                    [p.relative_to(ROOT).as_posix()], dry)


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
    push_all(f"data: {d} [local]", [f"data/{d[:7]}"], dry)

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
    d = today()
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


# 「今天这条线跑完了没有」读哪个文件，以及 --if-needed 允许开跑的北京时间窗口
#
# 窗口是给自动触发用的，手动跑（不带 --if-needed）不受限制。
# 上界的含义各不相同：
#   morning  09:16 之后新起一个进程来不及赶上 09:25 的竞价采样，
#            硬跑只会撞上 hard_deadline 然后发一封告警邮件。计划任务带
#            「登录时触发」，不设上界的话盘后每次开机登录都发一封告警。
#   evening  形态线的 hard_deadline 是 22:00
#   learn    收盘后数据定死，晚一点无所谓，给到 23:30
FLOWS = {
    "morning": ("out/run_meta.json", "竞价线", (6, 0), (9, 16)),
    "evening": ("out_pullback/run_meta.json", "形态线", (16, 0), (22, 0)),
    "learn": ("state/learning_status.json", "学习线", (16, 0), (23, 30)),
    # 晚间系统：起涨预测。17:00 发信，16:30 起跑（要算全市场特征）
    "breakout": ("out_breakout/run_meta.json", "起涨预测", (16, 0), (23, 0)),
}


def in_window(flow: str) -> bool:
    _, _, lo, hi = FLOWS[flow]
    hm = (now_bj().hour, now_bj().minute)
    return lo <= hm <= hi


def already_done(flow: str) -> bool:
    """今天这条线是否已经跑完。

    计划任务从北京 07:00 起每 15 分钟重试一次，一直敲到 09:15（机器可能
    整段时间都不在线，2026-08-27 就因为笔记本没醒漏发过）。代价是跑完
    之后它还会继续敲，所以入口必须幂等，否则每 15 分钟重发一封邮件。
    """
    rel = FLOWS[flow][0]
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))["date"]             == today()
    except Exception:  # noqa: BLE001
        return False


def flow_breakout(dry: bool) -> int:
    """晚间系统：起涨预测。补当天数据 -> 打分 -> 两个清单 -> 面板 + 邮件。"""
    n = 4
    d = today()
    step(1, n, "同步仓库")
    if not dry:
        _git_unstick()
        _git("pull", "--rebase", "--autostash", "-q", "origin", "main")

    step(2, n, "补当天日线 + 重算特征表")
    if py("src/breakout/backfill.py", "--stage", "sina") != 0:
        log.warning("补数据非零退出，继续用已有数据")
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
    push_all(f"起涨预测 {d} [local]",
             [f"data/breakout/{d[:7]}", "out_breakout", "state/breakout"], dry)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True,
                    choices=["morning", "evening", "learn", "breakout"])
    ap.add_argument("--dry", action="store_true",
                    help="只跑不发不推（测试）")
    ap.add_argument("--if-needed", action="store_true",
                    help="今天这条线已跑完（或今天不是交易日）就直接退出 0")
    a = ap.parse_args()

    if a.if_needed:
        # 周末先挡掉，省得每 15 分钟就去拉一次交易日历。节假日挡不住，
        # 但各阶段内部都有 trade_dates() 判断，跑起来会自己退出 0。
        if now_bj().weekday() >= 5:
            log.info("北京时间周末，跳过")
            return 0
        name = FLOWS[a.flow][1]
        if already_done(a.flow):
            log.info("%s 今天已经跑完，跳过", name)
            return 0
        if not in_window(a.flow):
            lo, hi = FLOWS[a.flow][2:]
            log.info("%s 现在 %s 不在开跑窗口 %02d:%02d-%02d:%02d（北京），跳过",
                     name, now_bj().strftime("%H:%M"), *lo, *hi)
            return 0

    load_env()
    t0 = now_bj()
    log.info("本地全流程 %s 启动 @ %s%s", a.flow,
             t0.strftime("%H:%M:%S"), "（dry-run）" if a.dry else "")
    rc = {"morning": flow_morning, "evening": flow_evening,
          "learn": flow_learn, "breakout": flow_breakout}[a.flow](a.dry)
    log.info("总耗时 %.0f 秒，退出码 %d",
             (now_bj() - t0).total_seconds(), rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
