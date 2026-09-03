"""
本地全流程编排器。TUI 的「一键跑」按的就是它。

    python src/local_run.py --flow morning    竞价线：候选池->采样->LLM->发信
    python src/local_run.py --flow evening    形态线+学习线：扫描->发信->评估
    加 --dry 只跑不发不推（测试用）

本地为主、远端为辅的接管协议
----------------------------
远端 GitHub Actions 每天照常自动跑（Cloudflare 触发），这是兜底。
本地一键跑通过两个 git 标记和远端协商，保证**每天恰好一封邮件**：

    state/claim/<flow>_<date>.json   开跑即推送：「今天本地要接管」
    state/sent/<flow>_<date>.json    发信成功后推送：「本地已发出」

远端的 yield_check（workflow 里的一步）在自己发信前查这两个标记：
    无 claim            -> 照常 09:27:30 发（用户没开本地，99% 的日子）
    有 claim + 有 sent  -> 不发邮件，只发布面板
    有 claim + 无 sent  -> 本地挂了，远端在 send_deadline 前兜底发出

方向是**fail-open**：远端只有拿到本地成功的证据才让位。
本地开跑后崩掉，最坏结果是邮件晚 ~55 秒（远端等到 09:28:20 才确认），
绝不会没有邮件。双发只在「本地 09:27:30 发出但 50 秒内没推上标记」
这一种窄窗口里发生，方向宁重不漏。

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
def push_marker(kind: str, flow: str, payload: dict, dry: bool) -> None:
    """写 state/{claim,sent}/<flow>_<date>.json 并推送。

    推送失败重试 3 次（竞价线可能同时在推数据）。彻底失败只警告不中断：
    标记推不上去的后果是远端不让位、可能双发，方向仍是宁重不漏。
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
    try:
        rel = p.relative_to(ROOT).as_posix()
        sh("git", "add", rel, quiet=True)
        if sh("git", "diff", "--cached", "--quiet", check=False,
              quiet=True) == 0:
            return
        sh("git", "commit", "-q", "-m",
           f"{kind}: {flow} {today()} [local]", quiet=True)
        for i in range(3):
            sh("git", "pull", "--rebase", "-q", "origin", "main",
               check=False, quiet=True)
            if sh("git", "push", "-q", "origin", "main", check=False,
                  quiet=True) == 0:
                log.info("标记已推送: %s/%s", kind, p.name)
                return
        log.warning("标记推送失败（远端不会让位，可能双发，宁重不漏）")
    except Exception as e:  # noqa: BLE001
        log.warning("标记推送异常: %s", e)


def push_all(msg: str, paths: list[str], dry: bool) -> None:
    if dry:
        log.info("[dry] 跳过提交推送")
        return
    try:
        for x in paths:
            sh("git", "add", "-A", x, check=False, quiet=True)
        if sh("git", "diff", "--cached", "--quiet", check=False,
              quiet=True) == 0:
            log.info("没有需要提交的产物")
            return
        sh("git", "commit", "-q", "-m", msg, quiet=True)
        for i in range(3):
            sh("git", "pull", "--rebase", "-q", "origin", "main",
               check=False, quiet=True)
            if sh("git", "push", "-q", "origin", "main", check=False,
                  quiet=True) == 0:
                log.info("产物已推送")
                return
        log.warning("产物推送失败，本地保留，稍后可手动 git push")
    except Exception as e:  # noqa: BLE001
        log.warning("产物推送异常: %s", e)


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

    step(4, n, "学习线：标签 -> 归因 -> 拟合与闸门")
    rc2 = py("src/eval_daily.py", "--stage", "all", "--date", d,
             *(["--dry"] if dry else []))
    if rc2 != 0:
        log.warning("学习线退出码 %d（研究性步骤，不影响业务邮件）", rc2)

    step(5, n, "推送学习产物")
    push_all(f"learn: {d} [local]", ["data/labels", "state"], dry)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, choices=["morning", "evening"])
    ap.add_argument("--dry", action="store_true",
                    help="只跑不发不推（测试）")
    a = ap.parse_args()
    load_env()
    t0 = now_bj()
    log.info("本地全流程 %s 启动 @ %s%s", a.flow,
             t0.strftime("%H:%M:%S"), "（dry-run）" if a.dry else "")
    rc = flow_morning(a.dry) if a.flow == "morning" else flow_evening(a.dry)
    log.info("总耗时 %.0f 秒，退出码 %d",
             (now_bj() - t0).total_seconds(), rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
