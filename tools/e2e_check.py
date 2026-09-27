"""端到端运行时测试：真的把每条线跑一遍，看它们的行为对不对。

和三条离线自测的分工：
  自测    钉的是**接线和不变量**，全离线、秒级、不碰生产目录。
  这个    真起子进程、真联网取数、真渲染面板，检查「跑起来会发生什么」：
          退出码、产物日期、幂等、锁、防护、不该发的信没发、不该推的没推。

一律不发信（SKIP_MAIL=1）、不推送（各条线走 --dry，它跳过 commit/push）。
--dry 还会在 run_meta 里标 dry，所以跑完这个不会把当天真正的流程顶掉
（教训 27：试跑写了 run_meta，计划任务据此判「今天跑完了」整天跳过）。

用法：
    python tools/e2e_check.py                # 全套
    python tools/e2e_check.py --quick        # 跳过慢的（起涨预测要重算 13 分钟特征）
    python tools/e2e_check.py --only gui,site
报告写 tools/e2e_report.json（gitignore），最后打印一张表。

2026-09-27 早盘系统归档：参数自学、早盘候选池、早盘打分、学习会诊四步一起删掉，
加了长期调整突破整条线（试跑：不等 17:58、不发信、不推）。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
SLOW = {"breakout"}


def _env(**extra) -> dict:
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    e.setdefault("SKIP_MAIL", "1")
    e.update({k: str(v) for k, v in extra.items()})
    return e


def run(args: list[str], timeout: int = 1800, **envkw) -> tuple[int, str]:
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=str(ROOT), env=_env(**envkw),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"超时（{timeout}s，已跑 {time.time() - t0:.0f}s）"


def blocked(out: str) -> bool:
    """这次 local_run 是不是被别人的锁挡住、秒退 0 了。

    退出码 0 不等于做了事（教训 27）。另一个入口（多半是计划任务）占着锁时
    local_run 打一行「已经在跑…本实例退出」然后 return 0，这时候测到的是锁
    不是流程，必须报「没测到」而不是「通过」——这个测试自己犯过一次。
    """
    return ("已经在跑" in out) or ("本实例退出" in out)


def tail(s: str, n: int = 6) -> str:
    lines = [x for x in (s or "").splitlines() if x.strip()]
    return " / ".join(lines[-n:])[:600]


def git_status() -> set[str]:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return {ln[3:].strip().strip('"') for ln in r.stdout.splitlines() if ln.strip()}


def meta_date(rel: str) -> str:
    try:
        return str(json.loads((ROOT / rel).read_text(encoding="utf-8")).get("date", ""))
    except Exception:  # noqa: BLE001
        return ""


def meta_dry(rel: str) -> bool:
    try:
        return bool(json.loads((ROOT / rel).read_text(encoding="utf-8")).get("dry"))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------
#  每一步：返回 (通过?, 说明)
# ---------------------------------------------------------------------
def step_env() -> tuple[bool, str]:
    """本机环境：tools/local.env 能加载，发信凭证都在（只看键，不打印值）。"""
    sys.path.insert(0, str(ROOT / "src"))
    import localenv
    localenv.load()
    need = ["SMTP_HOST", "SMTP_USER", "SMTP_PASS", "MAIL_TO"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        return False, f"缺 {missing}（发信会失败）"
    return True, "发信四项齐全"


def step_skipmail() -> tuple[bool, str]:
    """SKIP_MAIL 的判定只有一份实现，九个取值都要对（教训：四处各写各的）。"""
    import mailer
    cases = {"1": True, "true": True, "TRUE": True, " true ": True,
             "0": False, "false": False, "yes": False, "no": False, "": False}
    bad = []
    old = os.environ.get("SKIP_MAIL")
    try:
        for v, want in cases.items():
            os.environ["SKIP_MAIL"] = v
            if mailer.skip_mail() != want:
                bad.append(f"{v!r}->{not want}")
        os.environ.pop("SKIP_MAIL", None)
        if mailer.skip_mail():
            bad.append("不设也跳过")
    finally:
        if old is None:
            os.environ.pop("SKIP_MAIL", None)
        else:
            os.environ["SKIP_MAIL"] = old
    return (not bad), ("全部正确（含不设变量）" if not bad else f"错的取值：{bad}")


def step_selftests() -> tuple[bool, str]:
    names = ["selftest_pullback", "selftest_gui", "selftest_breakout"]
    bad = []
    for n in names:
        rc, out = run([PY, f"src/{n}.py"], timeout=300)
        if rc != 0 or "断言失败 0 个" not in out:
            bad.append(f"{n}(rc={rc})")
    return (not bad), ("三条全绿" if not bad else f"没过：{bad}")


def step_probe() -> tuple[bool, str]:
    rc, out = run([PY, "tools/probe.py"], timeout=600)
    return rc == 0, tail(out, 4)


def step_idempotence() -> tuple[bool, str]:
    """--if-needed 的五道闸：在进程内问一遍每条线现在会不会跑、为什么。"""
    import local_run as L
    rows = []
    for f in L.FLOWS:
        reason = L.if_needed_skip(f)
        rows.append(f"{L.FLOWS[f][1]}={reason or '会跑'}")
    return True, "；".join(rows)


def step_lock() -> tuple[bool, str]:
    """进程锁：自己先拿一把，再让**另一个进程**去拿，必须拿不到（教训 23）。

    第二个入口一定要是另一个进程：`_lock_holds` 把「自己 pid 写的锁」
    当成自己的旧锁清掉再抢（防止崩溃留下的锁把这条线永远堵死），
    所以同进程调两次 acquire_lock 本来就都会成功，那不是 bug。
    """
    import local_run as L
    # 挑一条当前空闲的线来测：真有流程在跑（计划任务）时，固定用某一条会把
    # 「别人正在跑」误报成「锁坏了」。全忙就诚实说没测到，不编一个通过。
    flow = next((f for f in L.FLOWS if not L.running_instance(f)), "")
    if not flow:
        return False, "没测到：两条线此刻都有实例在跑，没有空闲的线可以测锁"
    if not L.acquire_lock(flow):
        return False, f"第一次就没拿到 {flow} 的锁（刚被别人抢走？）"
    try:
        code = ("import sys; sys.path.insert(0, r'%s'); import local_run as L; "
                "print('GOT' if L.acquire_lock('%s') else 'BLOCKED')"
                % (str(ROOT / "src"), flow))
        rc, out = run([PY, "-c", code], timeout=120)
        # 不叫 blocked：模块级有个同名函数（判 local_run 是不是被锁挡住），
        # 在这里遮住它不会报错，只会在以后有人想调它的时候变成一个怪 bug
        stopped = "BLOCKED" in out
    finally:
        L.release_lock(flow)
    still = L.lock_path(flow).exists()
    return (stopped and not still), (
        f"另一个进程{'被挡住' if stopped else '也拿到了锁（会重复发信）'}；"
        f"释放后锁文件{'还在（没清干净）' if still else '已清掉'}")


def step_breakout() -> tuple[bool, str]:
    """起涨预测整条线（--dry：补数据 -> 打分 -> 清单 -> 面板，不发不推）。"""
    before = meta_date("out_breakout/run_meta.json")
    rc, out = run([PY, "src/local_run.py", "--flow", "breakout", "--dry"],
                  timeout=3600)
    d = meta_date("out_breakout/run_meta.json")
    if blocked(out):
        # 被锁挡住时 run_meta 还是上一轮的，日期和 dry 标记都对得上，会假过
        return False, ("没测到：有另一个实例正占着起涨预测的锁，本次秒退 0 | "
                       + tail(out, 2))
    import local_run as L
    want = L.target_date("breakout")
    ok = rc == 0 and d == want and meta_dry("out_breakout/run_meta.json")
    panel = (ROOT / "out_breakout" / "panel.html")
    detail = [f"rc={rc}", f"run_meta {before or '无'} -> {d}（目标 {want}）",
              f"标了 dry={meta_dry('out_breakout/run_meta.json')}",
              f"面板 {panel.stat().st_size // 1024}KB" if panel.exists() else "面板缺失"]
    if not panel.exists():
        ok = False
    return ok, "；".join(detail) + " | " + tail(out, 3)


def step_pullback() -> tuple[bool, str]:
    """长期调整突破整条线（--dry：补日线 -> 扫描 -> 面板，不等 17:58、不发不推）。"""
    sent = ROOT / "out_pullback" / "mail_sent.json"
    before_sent = sent.read_text(encoding="utf-8") if sent.exists() else ""
    rc, out = run([PY, "src/local_run.py", "--flow", "pullback", "--dry"], timeout=1800)
    if blocked(out):
        return False, ("没测到：有另一个实例正占着长期调整突破的锁，本次秒退 0 | "
                       + tail(out, 2))
    import local_run as L
    want = L.target_date("pullback")
    d = meta_date("out_pullback/run_meta.json")
    dry = meta_dry("out_pullback/run_meta.json")
    panel = ROOT / "out_pullback" / "panel.html"
    after_sent = sent.read_text(encoding="utf-8") if sent.exists() else ""
    no_mail = after_sent == before_sent
    try:
        n = json.loads((ROOT / "out_pullback" / "run_meta.json").read_text(encoding="utf-8")).get("n")
    except Exception:  # noqa: BLE001
        n = None
    ok = rc == 0 and d == want and dry and panel.exists() and no_mail
    return ok, (f"rc={rc}；run_meta {d}（目标 {want}）dry={dry}；清单 {n} 只；"
                f"面板{'在' if panel.exists() else '缺失'}；"
                f"发信戳{'没动' if no_mail else '被改了（试跑不该发信）'} | {tail(out, 3)}")


def step_site() -> tuple[bool, str]:
    rc, out = run([PY, "src/build_site.py"], timeout=600)
    site = ROOT / "_site"
    want = ["index.html", "breakout.html", "pullback.html"]
    have = [w for w in want if (site / w).exists()]
    stale = [w for w in ("learn.html", "council.html") if (site / w).exists()]
    ok = rc == 0 and len(have) == len(want) and not stale
    return ok, (f"rc={rc}；_site 有 {have}"
                + (f"；还留着已归档的 {stale}" if stale else "") + f" | {tail(out, 2)}")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def step_gui() -> tuple[bool, str]:
    """控制台：真起服务器，验页面出得来、两道防护真的挡人。"""
    port = _free_port()
    token = "e2e-test-token"
    p = subprocess.Popen([PY, "src/gui/__main__.py", "--port", str(port),
                          "--no-open"],
                         cwd=str(ROOT), env=_env(GUI_TOKEN=token),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    base = f"http://127.0.0.1:{port}"
    checks, deadline = [], time.time() + 25

    def get(path, host=None, method="GET", data=None):
        req = urllib.request.Request(base + path, method=method,
                                     data=(data or b"") if method == "POST" else None)
        if host:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read(400).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read(200).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            return 0, str(e)[:120]

    try:
        while time.time() < deadline:
            if get("/?token=" + token)[0]:
                break
            time.sleep(0.4)
        code, _ = get("/?token=" + token)
        checks.append(("页面", code == 200, code))
        code, _ = get("/api/status?token=" + token)
        checks.append(("状态接口", code == 200, code))
        code, _ = get("/api/run", method="POST")
        checks.append(("写操作没带 token 被挡", code == 403, code))
        code, _ = get("/?token=" + token, host="evil.example.com")
        checks.append(("外部 Host 被挡", code == 403, code))
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    ok = all(c[1] for c in checks)
    return ok, "；".join(f"{n}{'✓' if good else '✗'}({c})" for n, good, c in checks)


STEPS = [
    ("env", "本机环境与凭证", step_env),
    ("skipmail", "SKIP_MAIL 判定", step_skipmail),
    ("selftests", "三条离线自测", step_selftests),
    ("probe", "数据源可达性", step_probe),
    ("idempotence", "两条线的开跑判定", step_idempotence),
    ("lock", "进程锁", step_lock),
    ("pullback", "长期调整突破整条线（dry）", step_pullback),
    ("breakout", "起涨预测整条线（dry）", step_breakout),
    ("site", "面板打包", step_site),
    ("gui", "控制台与两道防护", step_gui),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的步骤名")
    ap.add_argument("--quick", action="store_true", help="跳过慢的几步")
    a = ap.parse_args()
    only = {x.strip() for x in a.only.split(",") if x.strip()}

    before = git_status()
    rows, t_all = [], time.time()
    for key, name, fn in STEPS:
        if only and key not in only:
            continue
        if a.quick and key in SLOW:
            rows.append({"step": key, "name": name, "ok": None, "detail": "跳过（--quick）"})
            continue
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        secs = time.time() - t0
        rows.append({"step": key, "name": name, "ok": bool(ok),
                     "detail": detail, "seconds": round(secs, 1)})
        print(("  [OK] " if ok else "  [!!] ") + f"{name}（{secs:.0f}s）：{detail}",
              flush=True)

    after = git_status()
    new = sorted(after - before)
    rep = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": round(time.time() - t_all, 1),
           "steps": rows,
           "git_new_or_changed": new,
           "failed": [r["step"] for r in rows if r["ok"] is False]}
    out = ROOT / "tools" / "e2e_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n%d 步，失败 %d 步，耗时 %.0f 分钟；工作区新增/改动 %d 个路径"
          % (len(rows), len(rep["failed"]), rep["seconds"] / 60, len(new)))
    print("报告 " + str(out))
    return 1 if rep["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
