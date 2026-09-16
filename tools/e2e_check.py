"""端到端运行时测试：真的把每条线跑一遍，看它们的行为对不对。

和六条离线自测的分工：
  自测    钉的是**接线和不变量**，全离线、秒级、不碰生产目录。
  这个    真起子进程、真联网取数、真渲染面板，检查「跑起来会发生什么」：
          退出码、产物日期、幂等、锁、防护、不该发的信没发、不该推的没推。

一律不发信（SKIP_MAIL=1）、不推送（各条线走 --dry，它跳过 commit/push）。
--dry 还会在 run_meta 里标 dry，所以跑完这个不会把当天真正的流程顶掉
（教训 27：试跑写了 run_meta，计划任务据此判「今天跑完了」整天跳过）。

用法：
    python tools/e2e_check.py                # 全套
    python tools/e2e_check.py --quick        # 跳过慢的（起涨预测、候选池）
    python tools/e2e_check.py --only gui,site
报告写 out_learn/e2e_report.json，最后打印一张表。
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
SLOW = {"breakout", "morning_pool", "learn"}


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
    """本机环境：tools/local.env 能加载，发信凭证和 OAuth 都在（只看键，不打印值）。"""
    sys.path.insert(0, str(ROOT / "src"))
    import localenv
    localenv.load()
    need = ["SMTP_HOST", "SMTP_USER", "SMTP_PASS", "MAIL_TO"]
    missing = [k for k in need if not os.environ.get(k)]
    oauth = "有" if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") else "无"
    if missing:
        return False, f"缺 {missing}（发信会失败）；OAuth {oauth}"
    return True, f"发信四项齐全；OAuth token {oauth}"


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
    names = ["selftest", "selftest_train", "selftest_pullback",
             "selftest_learn", "selftest_gui", "selftest_breakout"]
    bad = []
    for n in names:
        rc, out = run([PY, f"src/{n}.py"], timeout=300)
        if rc != 0 or "断言失败 0 个" not in out and n != "selftest_pullback":
            bad.append(f"{n}(rc={rc})")
    return (not bad), ("六条全绿" if not bad else f"没过：{bad}")


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
    flow = "learn"
    if not L.acquire_lock(flow):
        return False, "第一次就没拿到锁（有流程在跑？）"
    try:
        code = ("import sys; sys.path.insert(0, r'%s'); import local_run as L; "
                "print('GOT' if L.acquire_lock('%s') else 'BLOCKED')"
                % (str(ROOT / "src"), flow))
        rc, out = run([PY, "-c", code], timeout=120)
        blocked = "BLOCKED" in out
    finally:
        L.release_lock(flow)
    still = L.lock_path(flow).exists()
    return (blocked and not still),         (f"另一个进程{'被挡住' if blocked else '也拿到了锁（会重复发信）'}；"
         f"释放后锁文件{'还在（没清干净）' if still else '已清掉'}")


def step_breakout() -> tuple[bool, str]:
    """起涨预测整条线（--dry：补数据 -> 打分 -> 清单 -> 面板，不发不推）。"""
    before = meta_date("out_breakout/run_meta.json")
    rc, out = run([PY, "src/local_run.py", "--flow", "breakout", "--dry"],
                  timeout=3600)
    d = meta_date("out_breakout/run_meta.json")
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


def step_learn() -> tuple[bool, str]:
    """参数自学（--dry：标签 -> 归因 -> 拟合与闸门；--dry 会跳过会诊，不花钱）。"""
    rc, out = run([PY, "src/local_run.py", "--flow", "learn", "--dry"],
                  timeout=3600)
    st = ROOT / "state" / "learning_status.json"
    try:
        s = json.loads(st.read_text(encoding="utf-8"))
        d, stage = s.get("date", ""), s.get("stage", "")
    except Exception:  # noqa: BLE001
        d, stage = "", ""
    spent = "会诊没跑（--dry 跳过）" if "会诊" not in out else "会诊跑了（应该跳过！）"
    ok = rc == 0 and bool(d) and "会诊跑了" not in spent
    return ok, f"rc={rc}；learning_status {d} / {stage}；{spent} | {tail(out, 3)}"


def step_pullback() -> tuple[bool, str]:
    """回调形态（自动已停，只剩手动入口，这里验手动那条路还通）。"""
    rc, out = run([PY, "src/pullback.py", "--stage", "scan", "--dry"], timeout=1800)
    return rc == 0, tail(out, 4)


def step_morning_pool() -> tuple[bool, str]:
    """早盘的候选池。09:25 那一段等不了（要等到北京 09:25），单测这一步。"""
    rc, out = run([PY, "src/premarket.py"], timeout=1800)
    try:
        m = json.loads((ROOT / "cache" / "universe_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        m = {}
    return rc == 0 and bool(m.get("date")), \
        (f"rc={rc}；候选池 {m.get('date', '?')} / {m.get('count', '?')} 只"
         f"（历史缺 {m.get('missing_hist', '?')}）| {tail(out, 2)}")


def step_morning_enrich() -> tuple[bool, str]:
    """早盘：① enrich 不许拿旧产物发信；② 打分和选榜这条路真的走一遍。

    ① 单独跑 enrich 时，out/ 里如果是上一个交易日的产物它会拒绝发信
    （正确行为），但那也意味着**这一步没有测到打分和渲染**，只测到了拒绝。
    所以 ② 拿最近一份竞价快照重放：快照里存的就是当天 score_one 打完分的行，
    用向量化孪生体 vscore 重算一遍，必须逐位复现存档的分数（差一位就说明
    config / 孪生体 / 生产打分器三者之间漂了），再按生产口径选出那张榜。
    不写 out/（教训 17）。
    """
    sent = ROOT / "out" / "mail_sent.json"
    before = sent.read_text(encoding="utf-8") if sent.exists() else ""
    rc, out = run([PY, "src/run_auction.py", "--stage", "enrich"], timeout=900)
    after = sent.read_text(encoding="utf-8") if sent.exists() else ""
    no_mail = after == before
    refused = "不发信" in out

    import glob
    import numpy as np
    import pandas as pd
    import cfg as C
    from learn import vscore
    from learn.optimize import production_order
    snaps = sorted(glob.glob(str(ROOT / "data" / "*" / "auction_*.parquet")))
    if not snaps:
        return False, "没有任何竞价快照可以重放"
    c = C.load()
    df = pd.read_parquet(snaps[-1])
    sc, rej = vscore.score_df(df, c)
    # 取整走 vscore.round1（= score.round1 的向量化版）。这里以前写 np.round，
    # 于是 1012 行里 7 行对不上 —— 查下去发现不是孪生体漂了（原始分逐位相同），
    # 是取整有两份实现：np.round 在 x.x5 上和生产的 round 结果相反。已并成一份。
    same = float(np.mean(vscore.round1(sc) == df["score"].to_numpy(float)))
    # 剔除判定也要对上：孪生体说剔、存档说没剔（或反过来）就是口径漂了
    rej_same = float(np.mean(rej == df["rejected"].notna().to_numpy()))
    sel = production_order(sc, rej, c)
    top = df.iloc[sel]
    lo, hi = c["screen"]["gap_pct_min"], c["screen"]["gap_pct_max"]
    inrange = bool(((top["gap_pct"] >= lo) & (top["gap_pct"] <= hi)).all()) if len(top) else True
    ok = (rc == 0 and no_mail and same > 0.999 and rej_same > 0.999 and inrange)
    return ok, (f"rc={rc}；发信戳{'没动' if no_mail else '被改了（不该发信）'}"
                f"{'（enrich 拒绝拿旧产物发信）' if refused else ''}；"
                f"重放 {Path(snaps[-1]).name}（{len(df)} 行）："
                f"分数复现 {100 * same:.1f}%、剔除判定复现 {100 * rej_same:.1f}%；"
                f"选出 {len(top)} 只，涨幅都在 {lo}~{hi}% 内={inrange}")


def step_site() -> tuple[bool, str]:
    rc, out = run([PY, "src/build_site.py"], timeout=600)
    site = ROOT / "_site"
    want = ["index.html", "learn.html"]
    have = [w for w in want if (site / w).exists()]
    extra = "council.html" if (site / "council.html").exists() else ""
    ok = rc == 0 and len(have) == len(want)
    return ok, f"rc={rc}；_site 有 {have}{'+' + extra if extra else ''} | {tail(out, 2)}"


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
        code, _ = get("/api/run?flow=morning", method="POST")
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


def step_council() -> tuple[bool, str]:
    """会诊不花钱的那部分：证据包 + 面板 + 台账查询（不起 LLM 进程）。"""
    import cfg
    from learn.council import evidence as EV
    import datetime as dt
    c = cfg.load()
    date = dt.date.today().isoformat()
    p = EV.build(c, date)
    ev = json.loads(Path(p).read_text(encoding="utf-8"))
    bad = []
    if (ev.get("breakout") or {}).get("error"):
        bad.append("起涨证据: " + str(ev["breakout"]["error"])[:60])
    if not (ev.get("morning") or {}).get("days"):
        bad.append("早盘证据没有真值日")
    rc, out = run([PY, "src/learn/council/run.py", "--panel"], timeout=300)
    if rc != 0:
        bad.append("面板 rc=%d" % rc)
    rc2, out2 = run([PY, "tools/council_query.py", "proposals"], timeout=120)
    if rc2 != 0:
        bad.append("台账查询 rc=%d" % rc2)
    n = len(json.loads(out2)) if rc2 == 0 and out2.strip().startswith("[") else -1
    return (not bad), (f"证据包 {Path(p).name}；早盘 {ev['morning']['days']} 个真值日；"
                       f"起涨 {(ev.get('breakout') or {}).get('n_lists', '?')} 份清单；"
                       f"台账 {n} 条" + ("" if not bad else " | 问题：" + "；".join(bad)))


STEPS = [
    ("env", "本机环境与凭证", step_env),
    ("skipmail", "SKIP_MAIL 判定", step_skipmail),
    ("selftests", "六条离线自测", step_selftests),
    ("probe", "数据源可达性", step_probe),
    ("idempotence", "四条线的开跑判定", step_idempotence),
    ("lock", "进程锁", step_lock),
    ("breakout", "起涨预测整条线（dry）", step_breakout),
    ("learn", "参数自学整条线（dry）", step_learn),
    ("pullback", "回调形态（手动 dry）", step_pullback),
    ("morning_pool", "早盘候选池", step_morning_pool),
    ("morning_enrich", "早盘打分与面板（不发信）", step_morning_enrich),
    ("site", "面板打包", step_site),
    ("gui", "控制台与两道防护", step_gui),
    ("council", "会诊证据与面板", step_council),
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
    out = ROOT / "out_learn" / "e2e_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n%d 步，失败 %d 步，耗时 %.0f 分钟；工作区新增/改动 %d 个路径"
          % (len(rows), len(rep["failed"]), rep["seconds"] / 60, len(new)))
    print("报告 " + str(out))
    return 1 if rep["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
