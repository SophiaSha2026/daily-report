"""
控制台离线自测。和另外三条一样：不联网、不碰 state/、一秒内跑完。

钉的是**接线**，不是界面好不好看
--------------------------------
这个项目栽过两次接线的跟头（历史教训 11 的闸门传参、13 的邮件传参），
共同点是「代码跑得通、输出看着合理、其实那条线根本没接上」。
控制台有三处同样的风险：

  1. ACTIONS 里每个按钮指向一个脚本路径。脚本改名之后按钮不会报错，
     点下去才发现是死的。
  2. status.LINES 和 local_run.FLOWS 各自写了一份「哪条线读哪个
     run_meta」。两边漂了，总览页会长期显示「未完成」而流程其实跑完了。
  3. 前端 tab 的 data-p 和后端 PANELS 的键必须对上，否则点了出 404。

另外把两道防护（Host 白名单、写操作 token）也钉住：它们失效是静默的，
看界面完全正常。
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

fails: list[str] = []


def ck(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ✓ {msg}")
    else:
        fails.append(msg)
        print(f"  ✗ {msg}")


def check_actions() -> None:
    print("\n[按钮接线]")
    from gui.jobs import ACTIONS
    for key, a in ACTIONS.items():
        target = ROOT / a["cmd"][0]
        ck(target.exists(), f"{key} -> {a['cmd'][0]} 存在")
    ck(all(a.get("desc") for a in ACTIONS.values()),
       "每个动作都有说明文字（界面靠它告诉人这一下会发生什么）")
    # 会发信的动作必须标出来，否则人点之前不知道有邮件飞出去
    mail_keys = {k for k, a in ACTIONS.items() if a.get("mail")}
    ck(mail_keys == {"morning", "evening"},
       f"标了会发信的正好是竞价线和形态线（实际 {sorted(mail_keys)}）")
    ck(all(ACTIONS[k]["danger"] for k in mail_keys),
       "会发信的动作都标了 danger（按钮是红边的）")


def check_flow_tables() -> None:
    """status.LINES 和 local_run.FLOWS 是同一件事的两份副本，必须一致。"""
    print("\n[流程表一致性]")
    from gui import status
    import local_run
    gui_meta = {ln["key"]: ln["meta"] for ln in status.LINES}
    run_meta = {k: v[0] for k, v in local_run.FLOWS.items()}
    ck(set(gui_meta) == set(run_meta),
       f"两边的流程键一致（gui={sorted(gui_meta)} run={sorted(run_meta)}）")
    for k in sorted(set(gui_meta) & set(run_meta)):
        ck(gui_meta[k] == run_meta[k],
           f"{k} 的完成标记文件一致：{gui_meta[k]}")


def check_windows() -> None:
    """--if-needed 的开跑窗口。上界写错会在盘后发一串告警邮件。"""
    print("\n[开跑窗口]")
    import datetime as dt
    import local_run
    orig = local_run.now_bj
    cases = [
        ("morning", 5, 59, False), ("morning", 6, 0, True),
        ("morning", 9, 16, True), ("morning", 9, 17, False),
        ("morning", 20, 0, False),
        ("learn", 15, 59, False), ("learn", 16, 0, True),
        ("learn", 23, 30, True), ("learn", 23, 31, False),
        ("evening", 22, 0, True), ("evening", 22, 1, False),
    ]
    try:
        for flow, h, m, want in cases:
            local_run.now_bj = lambda h=h, m=m: dt.datetime(2026, 9, 14, h, m)
            got = local_run.in_window(flow)
            ck(got == want, f"{flow} {h:02d}:{m:02d} -> {'可跑' if want else '跳过'}")
    finally:
        local_run.now_bj = orig


def check_panels() -> None:
    """前端 tab 的 data-p 必须和后端 PANELS 的键对得上。"""
    print("\n[面板路由]")
    import re
    from gui.server import PANELS
    from gui.ui import PAGE
    tabs = set(re.findall(r'data-p="([a-z]+)"', PAGE))
    ck(tabs == set(PANELS),
       f"前端 tab 和后端 PANELS 一致（前端 {sorted(tabs)}，后端 {sorted(PANELS)}）")
    ck("__TOKEN__" in PAGE, "页面里有 token 占位符（服务端要替换它）")


def check_http() -> None:
    """两道防护 + 读接口。失效是静默的，界面上看不出来。

    这里把 git 和 PowerShell 都换成假实现：自测必须离线（和另外三条
    一条规矩），而 /api/status 真跑起来会 git fetch 打网络、起
    powershell.exe 查计划任务，慢且结果不可复现。
    顺带这也是一次容错测试——外部命令全挂时页面照样要出得来。
    """
    print("\n[HTTP 防护]")
    from gui import server
    from gui import status as st

    orig_git, orig_tasks = st._git, st.scheduled_tasks
    st._git = lambda *_a: ""                      # 装成 git 不可用
    st.scheduled_tasks = lambda force=False: {}   # 装成查不到计划任务

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, method="GET", host=None, token=None):
        req = urllib.request.Request(
            base + path, method=method,
            data=b"{}" if method == "POST" else None)
        if host:
            req.add_header("Host", host)
        if token:
            req.add_header("X-Token", token)
        if method == "POST":
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, body = call("/")
        ck(code == 200, "GET / 返回 200")
        ck(server.TOKEN.encode() in body and b"__TOKEN__" not in body,
           "页面里的 token 占位符被真 token 替换了")

        code, body = call("/api/status")
        ck(code == 200, "GET /api/status 返回 200")
        try:
            o = json.loads(body)
        except Exception:  # noqa: BLE001
            o = {}
        ck({"lines", "sync", "tasks"} <= set(o), "状态里有 lines/sync/tasks")
        # 上面的 git 是死的，所以这同时证明了「外部命令全挂也出得来页面」
        ck(len(o.get("lines", [])) == 3, "git 不可用时三条线照样列得出来")

        ck(call("/api/status", host="evil.example.com")[0] == 403,
           "伪造 Host 被挡（防 DNS rebinding）")
        ck(call("/api/run", "POST")[0] == 403, "POST 无 token 被挡")
        ck(call("/api/run", "POST", token="wrong")[0] == 403,
           "POST 错 token 被挡")
        ck(call("/api/task", "POST", token=server.TOKEN)[0] == 403,
           "带对 token 也改不了非 DailyReport-* 的计划任务")
        ck(call("/panel/../../config.yaml")[0] in (403, 404),
           "面板路由不能拿它去读任意文件")
        ck(call("/nope")[0] == 404, "未知路径 404")
    finally:
        httpd.shutdown()
        httpd.server_close()
        st._git, st.scheduled_tasks = orig_git, orig_tasks


def main() -> int:
    import time
    t0 = time.time()
    check_actions()
    check_flow_tables()
    check_windows()
    check_panels()
    check_http()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {len(fails)} 个")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
