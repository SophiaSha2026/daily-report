"""
本地控制台的 HTTP 服务。零第三方依赖，只用标准库。

为什么不上 Flask/FastAPI
------------------------
CLAUDE.md 的规矩是依赖分线管理（学习线要 scikit-learn，另两条线不许
沾）。控制台是三条线共用的，给它引一个 Web 框架，等于给每条线都加了
依赖。标准库的 ThreadingHTTPServer 在单用户本机场景下完全够用——
这里的并发上限是「一个人开了几个标签页」。

两道防护
--------
绑 127.0.0.1 只挡住了别的机器，挡不住本机浏览器里的恶意页面：

  1. Host 头白名单。防 DNS rebinding：攻击者把自己的域名解析到
     127.0.0.1，受害者浏览器就能以同源身份访问这个服务。只认
     localhost / 127.0.0.1 就断了这条路。
  2. 写操作要 token。启动时随机生成，只出现在打印的 URL 里。
     读接口不要求，这样面板 iframe 不用到处带 token。

这不是过度设计：这个服务的写接口能发邮件、能推 git、能改计划任务。
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import simple, status
from .jobs import ACTIONS, Registry
from .ui import PAGE

ROOT = Path(__file__).resolve().parent.parent.parent
# 固定 token 便于脚本化和调试；不给就每次随机，这是默认。
TOKEN = os.environ.get("GUI_TOKEN") or secrets.token_urlsafe(16)
REG = Registry()

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "[::1]"}

# 面板文件：界面上用 iframe 嵌这三个。键要和前端对得上。
PANELS = {
    "auction": ("out/panel.html", "早盘选股面板"),
    "pullback": ("out_pullback/panel.html", "回调形态面板"),
    "learn": ("out_learn/learn.html", "参数自学面板"),
    "breakout": ("out_breakout/panel.html", "起涨预测面板"),
    "council": ("out_learn/council.html", "学习会诊面板"),
}


def _council_decide(body: dict) -> dict:
    pid = str(body.get("id") or "")
    action = str(body.get("action") or "")
    if not pid or action not in ("approve", "reject"):
        return {"ok": False, "error": "缺 id 或 action"}
    try:
        from learn.council import run as CR, experiments as EX
        if action == "reject":
            CR.write_decision(pid, "rejected", by="gui")
            res = {"ok": True, "what": "已驳回"}
        else:
            import cfg as C
            res = EX.apply(pid, C.load(), by="gui")
        try:
            from learn.council import panel as CP
            CP.build()
        except Exception as e:  # noqa: BLE001
            res["panel_error"] = str(e)
        return res
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _retry_push() -> dict:
    """「重试推送」。09-07 那次失联之后必须有一个不用开终端的出口。

    这里一条 git 命令都不自己拼，整个推送走 local_run.push_main。三个理由，
    都是这个按钮以前真的坏在上面：
      1. 它只有 fetch + push。09-07 那次是本地 22 个 commit、远端 25 个的
         **分叉**，push 每次都 non-fast-forward，按多少下结果一样 —— 按钮
         对自己的设计场景无效。push_main 失败时会 fetch + merge -X ours 再试。
      2. 它不写 state/push_status.json，推成功之后同步卡片继续红、继续说
         「上次推送失败」，红灯从此不可信（教训 16 的反面）。
      3. 有流程正在跑时它照样动工作区。sync_repo 早就有 any_flow_running
         守卫，这里没有。
    """
    import sys as _sys
    _p = str(ROOT / "src")
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    import local_run           # 顶层只 import 标准库，不会把 akshare 拉进来
    busy = local_run.any_flow_running()
    if busy:
        return {"ok": False, "busy": True,
                "error": f"{local_run.FLOWS[busy][1]} 正在跑，等它推完再试"}
    try:
        with local_run.git_lock(timeout=30):
            local_run._git_unstick()
            ok, log_txt = local_run.push_main("manual push [gui]")
        return {"ok": ok, "log": log_txt}
    except local_run.GitBusy as e:
        return {"ok": False, "busy": True,
                "error": f"有流程正在用 git（{e}），等它跑完再点"}


class Handler(BaseHTTPRequestHandler):
    server_version = "AshareConsole/1.0"

    # 默认实现会把每个请求打到 stderr，界面每秒轮询一次会把控制台刷爆
    def log_message(self, *_args) -> None:  # noqa: D102
        pass

    # ---------------- 基础设施 ----------------
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ALLOWED_HOSTS

    def _auth_ok(self, q: dict) -> bool:
        given = (q.get("token", [""])[0]
                 or self.headers.get("X-Token", ""))
        return secrets.compare_digest(given, TOKEN)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 面板是本地文件，浏览器缓存了就看不到新的一天
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # 用户切走了标签页，正常

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _html(self, text: str) -> None:
        self._send(200, text.encode("utf-8"), "text/html; charset=utf-8")

    # ---------------- 路由 ----------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._json({"error": "host not allowed"}, 403)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        # 首页是傻瓜页（2026-09-18 用户要求：一个按钮、跑了没、多久没跑、
        # 进度条，字越少越好，全部北京时间）。原来那个详细控制台挪到 /full。
        if p == "/":
            return self._html(simple.PAGE.replace("__TOKEN__", TOKEN))

        if p == "/full":
            return self._html(PAGE.replace("__TOKEN__", TOKEN))

        if p == "/api/simple":
            return self._json(simple.snapshot())

        if p == "/api/status":
            return self._json(status.overview())

        if p == "/api/actions":
            return self._json({
                "actions": [{"key": k, **{x: v for x, v in a.items()
                                          if x != "cmd"}}
                            for k, a in ACTIONS.items()],
                "running": sorted(REG.running_keys()),
                "jobs": [j.brief() for j in REG.all()[:20]],
            })

        if p.startswith("/api/job/"):
            j = REG.get(p.rsplit("/", 1)[-1])
            if not j:
                return self._json({"error": "no such job"}, 404)
            start = int(q.get("from", ["0"])[0])
            total, lines = j.tail(start)
            return self._json({**j.brief(), "next": total, "new": lines})

        # 面板 iframe 里的自动刷新脚本会轮询 stamp 文件。它用的是相对
        # 路径，而 iframe 的地址是 /panel/<key>，所以请求落在
        # /panel/stamp.txt 上 —— 没有这条路由的话每 15 秒一个 404，
        # 控制台的网络面板会被刷屏。
        if p.startswith("/panel/stamp"):
            name = p.rsplit("/", 1)[-1]
            src = {"stamp.txt": "out", "stamp-pullback.txt": "out_pullback",
                   "stamp-breakout.txt": "out_breakout",
                   }.get(name, "out")
            f = ROOT / src / "stamp.txt"
            txt = f.read_text(encoding="utf-8") if f.exists() else ""
            return self._send(200, txt.encode("utf-8"),
                              "text/plain; charset=utf-8")

        if p.startswith("/panel/"):
            key = p.rsplit("/", 1)[-1]
            if key not in PANELS:
                return self._json({"error": "no such panel"}, 404)
            rel, name = PANELS[key]
            f = ROOT / rel
            if not f.exists():
                return self._html(
                    f'<body style="background:#14161a;color:#8f9aa8;'
                    f'font:14px sans-serif;padding:40px;text-align:center">'
                    f'{name}还没有生成过<br><br>'
                    f'<span style="font-size:12px">跑一次对应的流程就会有</span>'
                    f'</body>')
            return self._html(f.read_text(encoding="utf-8"))

        if p == "/api/logfile":
            # 计划任务自动跑的输出，每条线一个文件（tools/local_flow_<flow>.log，
            # 一个文件被长流程占着时别的任务写不进去）。GUI 手动跑的日志走
            # /api/job/<id>。按最近改动排，每个只取尾巴。
            files = sorted((ROOT / "tools").glob("local_flow*.log"),
                           key=lambda f: f.stat().st_mtime, reverse=True)
            parts = []
            for f in files:
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    continue
                parts.append(f"===== {f.name} =====\n{txt[-15000:]}")
            return self._json({"text": "\n\n".join(parts)
                               or "（还没有自动运行过）"})

        if p == "/api/config":
            try:
                txt = (ROOT / "config.yaml").read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                txt = f"读不到 config.yaml: {e}"
            return self._json({"text": txt})

        return self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._json({"error": "host not allowed"}, 403)
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._auth_ok(q):
            return self._json({"error": "bad token"}, 403)

        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:  # noqa: BLE001
            body = {}
        p = u.path

        if p == "/api/run":
            j, err = REG.start(body.get("key", ""))
            if not j:
                return self._json({"error": err}, 409)
            return self._json(j.brief())

        if p == "/api/stop":
            j = REG.get(body.get("id", ""))
            if not j:
                return self._json({"error": "no such job"}, 404)
            j.stop()
            return self._json(j.brief())

        if p == "/api/task":
            return self._task(body)

        if p == "/api/council/decide":
            # 会诊提案的批准 / 驳回。批准 = 落地（早盘参数写 learned.yaml，
            # 起涨常量/特征进 overrides）。只认过闸的，其余只能驳回。
            return self._json(_council_decide(body))

        if p == "/api/push":
            r = _retry_push()
            return self._json(r, 409 if r.get("busy") else 200)

        return self._json({"error": "not found"}, 404)

    def _task(self, body: dict) -> None:
        """启用/停用一个计划任务。只认 Windows 实际报出来的 DailyReport-* 任务。

        两道，缺一不可：
          1. 正则：名字要拼进 PowerShell 的**单引号字符串**，而单引号是那种
             字符串里唯一的转义点，一个 `'` 就出来了。实测
             `DailyReport-x'; Write-Output INJECTED-$PID; '` 通过了原来的
             startswith 检查，第二条语句真的执行，而且因为最后一段
             `'' | Out-Null` 成功，退出码 0、接口回 ok:true、界面 toast「已改」，
             完全静默。
          2. 白名单：只放行 scheduled_tasks() 实际列出来的键。用它而不是
             LINES 的 task 字段，是因为本机还有 Local-Sync 和两个旧的
             Trigger 任务（共 6 个），界面给每个都画了按钮。
        """
        name = str(body.get("name", ""))
        if (not re.fullmatch(r"DailyReport-[A-Za-z0-9_-]+", name)
                or name not in status.scheduled_tasks()):
            return self._json({"error": "只允许操作 DailyReport-* 任务"}, 403)
        verb = "Enable" if body.get("enable") else "Disable"
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"{verb}-ScheduledTask -TaskName '{name}' | Out-Null"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0:
            return self._json({"error": (r.stderr or "失败")[:300]}, 500)
        status.scheduled_tasks(force=True)
        return self._json({"ok": True})


def serve(port: int = 8765, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/?token={TOKEN}"
    # flush 是必须的：从 .cmd 起或者被重定向时 stdout 是块缓冲，
    # 不 flush 的话用户盯着一个空窗口，拿不到带 token 的 URL。
    print("A股流水线 控制台", flush=True)
    print(f"  {url}", flush=True)
    print("  Ctrl+C 退出（退出会中断正在手动跑的流程，计划任务不受影响）",
          flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        httpd.shutdown()
