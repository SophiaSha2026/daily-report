"""
claude.exe 用量记账（计划 2026-10-10 第 0 节拍板：Opus 5.5，走 CLI 登录态，不设单次上限，
累计用量超过警戒线在控制台和邮件抬头提示）。

    record(kind, code, result_json)   每次调用后把 CLI 返回的 total_cost_usd / usage 追加到
                                       state/ai_usage.jsonl（一行一次，北京时间）
    totals()                           今天 / 本月累计美元和次数，以及是否超过警戒线
    run_claude(prompt, schema, ...)    统一的 CLI 调用：Opus 5.5、JSON 输出、结构化 schema、
                                       只读联网工具、超时；返回 (结构化结果, 外层 JSON, 错误说明)

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


def run_claude(prompt: str, schema: dict | None = None, model: str = MODEL,
               tools: str = "WebSearch,WebFetch", timeout: int = 900,
               system: str = "", kind: str = "ai", subject: str = "") -> tuple[dict | None, dict | None, str]:
    """跑一次 `claude -p`。返回 (结构化结果, 外层 JSON, 错误说明)；无论成败都记账。

    stdin 接 DEVNULL（CLI 会等 3 秒管道输入）；--no-session-persistence 不留会话；
    --allowed-tools 只放只读的联网工具，-p 模式下白名单外的工具直接拒绝。
    """
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--no-session-persistence"]
    if schema:
        cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
    if tools:
        cmd += ["--allowed-tools", tools]
    if system:
        cmd += ["--append-system-prompt", system]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        record(kind, subject, None, {"timeout": timeout})
        return None, None, f"超过 {timeout} 秒没回来"
    outer = None
    try:
        outer = json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        i, j = r.stdout.find("{"), r.stdout.rfind("}")
        if i >= 0 and j > i:
            try:
                outer = json.loads(r.stdout[i:j + 1])
            except Exception:  # noqa: BLE001
                outer = None
    record(kind, subject, outer)
    if outer is None:
        return None, None, (r.stderr or r.stdout or f"退出码 {r.returncode}")[:300]
    if outer.get("is_error"):
        return None, outer, str(outer.get("result", ""))[:300]
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
