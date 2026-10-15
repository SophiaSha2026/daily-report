"""
claude.exe 用量记账（计划 2026-10-10 第 0 节拍板：Opus 5.5，走 CLI 登录态，不设单次上限，
累计用量超过警戒线在控制台和邮件抬头提示）。

    record(kind, code, result_json)   每次调用后把 CLI 返回的 total_cost_usd / usage 追加到
                                       state/ai_usage.jsonl（一行一次，北京时间）
    totals()                           今天 / 本月累计美元和次数，以及是否超过警戒线
    run_claude(prompt, schema, ...)    统一的 CLI 调用：Opus 5.5、JSON 输出、结构化 schema、
                                       只读工具、超时、推理等级、多轮会话、流式进度；
                                       返回 (结构化结果, 外层 JSON, 错误说明)
    models()                           本机 CLI 能用的模型和推理等级（扫本体 + --help，CLI 升级自动更新）

警戒线从 tools/local.env 读（AI_WARN_DAY_USD / AI_WARN_MONTH_USD），默认 10 / 100 美元。
只记账不拦截：拦截是用户明确不要的。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bjtime import now_bj  # noqa: E402
import localenv  # noqa: E402

# 先把 tools/local.env 读进来再取 AI_MODEL：MODEL 在导入时就定了，入口在 main() 里才 load 的话
# 已经晚了（2026-10-13 查出：ai_analyze / ai_review 只 import 了 localenv 没调 load，单独跑时
# AI_MODEL 不生效，落到别名 opus = claude-opus-5）。已存在的环境变量优先，重复 load 无害
localenv.load()

USAGE = ROOT / "state" / "ai_usage.jsonl"
# 2026-10-11 实测：本机 claude 2.1.275 不认 claude-opus-5-5（要 2.1.280 起），别名 opus 落到
# claude-opus-5。先用别名；CLI 升级后在 tools/local.env 里设 AI_MODEL=claude-opus-5-5 即可。
MODEL = os.environ.get("AI_MODEL", "opus")
WARN_DAY = float(os.environ.get("AI_WARN_DAY_USD", "10"))
WARN_MONTH = float(os.environ.get("AI_WARN_MONTH_USD", "100"))


def record(kind: str, subject: str, outer: dict | None, extra: dict | None = None) -> dict:
    u = (outer or {}).get("usage") or {}
    row = {"at": now_bj().isoformat(timespec="seconds"), "kind": kind, "subject": subject,
           "usd": float((outer or {}).get("total_cost_usd") or 0.0),
           "in": int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                 + int(u.get("cache_creation_input_tokens") or 0),
           "out": int(u.get("output_tokens") or 0),
           "web": int(((u.get("server_tool_use") or {}).get("web_search_requests") or 0))
                  + int(((u.get("server_tool_use") or {}).get("web_fetch_requests") or 0)),
           "turns": int((outer or {}).get("num_turns") or 0),
           "ms": int((outer or {}).get("duration_ms") or 0),
           "model": ",".join(sorted(((outer or {}).get("modelUsage") or {}).keys())),
           "error": bool((outer or {}).get("is_error"))}
    if extra:
        row.update(extra)
    USAGE.parent.mkdir(parents=True, exist_ok=True)
    with USAGE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def rows() -> list[dict]:
    if not USAGE.exists():
        return []
    out = []
    for line in USAGE.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def totals() -> dict:
    today = now_bj().strftime("%Y-%m-%d")
    month = today[:7]
    rs = rows()
    d = [r for r in rs if str(r.get("at", "")).startswith(today)]
    m = [r for r in rs if str(r.get("at", "")).startswith(month)]
    out = {"today_usd": round(sum(r.get("usd", 0) for r in d), 2), "today_n": len(d),
           "month_usd": round(sum(r.get("usd", 0) for r in m), 2), "month_n": len(m),
           "warn_day": WARN_DAY, "warn_month": WARN_MONTH}
    out["warn"] = (out["today_usd"] >= WARN_DAY) or (out["month_usd"] >= WARN_MONTH)
    out["note"] = ("" if not out["warn"] else
                   f"AI 用量 今天 {out['today_usd']} 美元 / 本月 {out['month_usd']} 美元，超过警戒线")
    return out


def available() -> bool:
    import shutil
    return shutil.which("claude") is not None


# ---------------------------------------------------------------
#  可用模型和推理等级（2026-10-13 AI 讨论窗口用）
#
# CLI 没有「列出模型」的子命令，试一个模型最少也要 0.3~0.9 美元（每次都带系统提示），
# 所以不逐个试。做法：
#   模型    扫 claude 本体里写死的模型 ID（CLI 升级换了本体，缓存跟着失效，新模型自动出现），
#           只留最新一代的 fable / opus / sonnet / haiku，不带日期后缀的；
#           用量账里真跑通过的标 verified
#   等级    解析 `claude --help` 里 --effort 那一行括号里的取值
# 缓存 state/ai_models.json，键是本体的路径、大小、修改时间。
# ---------------------------------------------------------------
MODELS_CACHE = ROOT / "state" / "ai_models.json"
FAMILY_ORDER = ("fable", "opus", "sonnet", "haiku")


def _cli_path() -> str:
    import shutil
    return shutil.which("claude") or ""


def _scan_models(path: str) -> list[str]:
    import re
    pat = re.compile(rb"claude-(fable|opus|sonnet|haiku)-(\d+)(?:-(\d+))?(?![-\d])")
    found: set[tuple[str, int, int]] = set()
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            buf = tail + chunk
            for m in pat.finditer(buf):
                minor = m.group(3)
                # 带 8 位日期的老 ID（claude-opus-4-1-20250805）这里 minor 是日期，丢掉
                if minor and len(minor) > 2:
                    continue
                found.add((m.group(1).decode(), int(m.group(2)), int(minor or 0)))
            tail = buf[-64:]
    if not found:
        return []
    top = max(x[1] for x in found)
    keep = sorted((x for x in found if x[1] == top),
                  key=lambda x: (FAMILY_ORDER.index(x[0]), -x[2]))
    return [f"claude-{fam}-{mj}" + (f"-{mn}" if mn else "") for fam, mj, mn in keep]


def _scan_efforts() -> list[str]:
    import re
    try:
        r = subprocess.run(["claude", "--help"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        m = re.search(r"--effort\s*<\w+>[^(]*\(([^)]*)\)", r.stdout, re.S)
        if m:
            return [x.strip() for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    except Exception:  # noqa: BLE001
        pass
    return []


def models(force: bool = False) -> dict:
    """{"models": [{"id", "verified"}], "efforts": [...], "default_model", "scanned", "cli"}。"""
    path = _cli_path()
    key = ""
    if path:
        st = os.stat(path)
        key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    cache = {}
    try:
        cache = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    if force or cache.get("key") != key or not cache.get("models"):
        ids = _scan_models(path) if path else []
        effs = _scan_efforts() if path else []
        cache = {"key": key, "models": ids, "efforts": effs,
                 "scanned": now_bj().isoformat(timespec="seconds")}
        try:
            MODELS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            MODELS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    used = set()
    for r in rows():
        if not r.get("error"):
            used.update(x for x in str(r.get("model", "")).split(",") if x)
    ids = list(cache.get("models") or [])
    if MODEL not in ids and MODEL.startswith("claude-"):
        ids.insert(0, MODEL)
    return {"models": [{"id": m, "verified": m in used} for m in ids],
            "efforts": cache.get("efforts") or [], "default_model": MODEL,
            "scanned": cache.get("scanned", ""), "cli": bool(path)}


def _parse_outer(stdout: str) -> dict | None:
    try:
        return json.loads(stdout)
    except Exception:  # noqa: BLE001
        i, j = stdout.find("{"), stdout.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(stdout[i:j + 1])
            except Exception:  # noqa: BLE001
                return None
    return None


def event_note(d: dict) -> str:
    """stream-json 的一行 -> 给人看的一句进度（在搜什么、在读哪个文件）。没有就空。"""
    if d.get("type") != "assistant":
        return ""
    out = []
    for c in (d.get("message") or {}).get("content") or []:
        if c.get("type") != "tool_use":
            continue
        name, inp = c.get("name", ""), c.get("input") or {}
        if name == "StructuredOutput":
            out.append("整理结论")
        elif name == "WebSearch":
            out.append(f"搜：{str(inp.get('query', ''))[:40]}")
        elif name == "WebFetch":
            out.append(f"读网页：{str(inp.get('url', ''))[:60]}")
        elif name == "Read":
            out.append(f"读文件：{Path(str(inp.get('file_path', ''))).name}")
        elif name in ("Grep", "Glob"):
            out.append(f"查代码：{str(inp.get('pattern', ''))[:40]}")
        else:
            out.append(name)
    return "；".join(out)


AUTH_HINT = ("claude 登录过期：在终端跑 claude setup-token，把生成的令牌写进 tools/local.env 的 "
             "CLAUDE_CODE_OAUTH_TOKEN=（长期有效）；临时也可以跑 claude 再 /login")


def explain_error(err: str) -> str:
    """CLI 的报错 -> 给人看的一句。登录类的错（2026-10-14 OAuth 过期，每日复盘静默失败）翻成怎么修。"""
    e = str(err or "")
    if any(k in e for k in ("authenticate", "OAuth", "Not logged in", "not logged in", "/login", "401")):
        return AUTH_HINT
    return e


def run_claude(prompt: str, schema: dict | None = None, model: str = MODEL,
               tools: str = "WebSearch,WebFetch", timeout: int = 900,
               system: str = "", kind: str = "ai", subject: str = "",
               effort: str = "", session_id: str = "", resume: bool = False,
               cwd: str | None = None, add_dirs: tuple[str, ...] = (),
               on_event=None) -> tuple[dict | None, dict | None, str]:
    """跑一次 `claude -p`。返回 (结构化结果, 外层 JSON, 错误说明)；无论成败都记账。

    stdin 接 DEVNULL（CLI 会等 3 秒管道输入）；--allowed-tools 只放只读工具，-p 模式下
    白名单外的工具直接拒绝。
    session_id 给了就是多轮会话（AI 讨论）：第一轮 --session-id，之后 resume=True 走
    --resume，换模型 / 推理等级照样续得上（2026-10-13 实测）；没给就 --no-session-persistence。
    on_event 给了就走 stream-json，每一行进度回调一句人话（event_note），结果取最后的 result 行
    （result 行和 json 输出同样带 structured_output / total_cost_usd，2026-10-13 实测）。
    """
    cmd = ["claude", "-p", prompt, "--model", model or MODEL]
    cmd += ["--output-format", "stream-json", "--verbose"] if on_event else ["--output-format", "json"]
    if session_id:
        cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    else:
        cmd += ["--no-session-persistence"]
    if effort:
        cmd += ["--effort", effort]
    if schema:
        cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
    if tools:
        cmd += ["--allowed-tools", tools]
    for d in add_dirs:
        cmd += ["--add-dir", d]
    if system:
        cmd += ["--append-system-prompt", system]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    extra = {"model_req": model or MODEL, "effort": effort or ""}
    nowin = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not on_event:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd,
                               stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                               timeout=timeout, creationflags=nowin)
        except subprocess.TimeoutExpired:
            record(kind, subject, None, {"timeout": timeout, **extra})
            return None, None, f"超过 {timeout} 秒没回来"
        outer = _parse_outer(r.stdout)
        err_txt = r.stderr or r.stdout or f"退出码 {r.returncode}"
    else:
        import threading
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                             cwd=cwd, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                             creationflags=nowin)
        fired = threading.Event()

        def _kill() -> None:
            fired.set()
            p.kill()
        killer = threading.Timer(timeout, _kill)
        killer.start()
        outer = None
        try:
            for line in p.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if d.get("type") == "result":
                    outer = d
                    continue
                note = event_note(d)
                if note:
                    try:
                        on_event(note)
                    except Exception:  # noqa: BLE001
                        pass
            p.wait()
            err_txt = (p.stderr.read() or "")[-600:] or f"退出码 {p.returncode}"
        finally:
            killer.cancel()
        if fired.is_set() and outer is None:
            record(kind, subject, None, {"timeout": timeout, **extra})
            return None, None, f"超过 {timeout} 秒没回来"
    record(kind, subject, outer, extra)
    if outer is None:
        return None, None, explain_error(str(err_txt))[:300]
    if outer.get("is_error"):
        return None, outer, explain_error(str(outer.get("result", "")))[:300]
    structured = outer.get("structured_output")
    if schema and not isinstance(structured, dict):
        # 没走 schema 的话从 result 文本里挖
        txt = str(outer.get("result", ""))
        i, j = txt.find("{"), txt.rfind("}")
        try:
            structured = json.loads(txt[i:j + 1]) if i >= 0 else None
        except Exception:  # noqa: BLE001
            structured = None
    return structured if schema else {"text": str(outer.get("result", ""))}, outer, ""


def main() -> int:
    """python src/ai_usage.py [--update]：打印本机能用的模型和推理等级；--update 先升级 claude 命令行。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="先跑 claude update")
    a = ap.parse_args()
    if a.update:
        r = subprocess.run(["claude", "update"], stdin=subprocess.DEVNULL, timeout=600)
        print(f"claude update 退出码 {r.returncode}", flush=True)
    m = models(force=True)
    print("模型：" + "、".join(x["id"] + ("（用过）" if x["verified"] else "") for x in m["models"]))
    print("推理等级：" + "、".join(m["efforts"]))
    t = totals()
    print(f"用量：今天 {t['today_usd']} 美元 / 本月 {t['month_usd']} 美元")
    return 0


if __name__ == "__main__":
    sys.exit(main())
