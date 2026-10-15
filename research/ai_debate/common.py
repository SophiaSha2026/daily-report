"""研究用的 claude -p 封装（research/ai_debate，不进生产）。

和 src/ai_usage.run_claude 同一套参数，多两样：
  tools=""  时传 --tools ""，彻底不给工具（盲测要保证它读不到仓库、上不了网）
  system_replace  用 --system-prompt 换掉 Claude Code 的默认系统提示（盲测省钱，也少一层干扰）
记账照常走 ai_usage.record，kind 一律 research-* 开头。原始输出追加到 raw/<tag>.jsonl。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ai_usage  # noqa: E402

RAW = HERE / "raw"
WORK = HERE / "work"          # cwd：空目录，读不到仓库
BUDGET_USD = 30.0


def auth_ok() -> tuple[bool, str]:
    """claude 命令行登录态。过期的话所有调用秒退（2026-10-14 实测：OAuth session expired）。"""
    try:
        r = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        d = json.loads(r.stdout)
        return bool(d.get("loggedIn")), str(d.get("authMethod"))
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:100]


def spent() -> float:
    return round(sum(r.get("usd", 0) for r in ai_usage.rows()
                     if str(r.get("kind", "")).startswith("research-ai_debate")), 3)


def call(prompt: str, schema: dict | None, *, model: str, effort: str = "", tools: str = "",
         system: str = "", system_replace: bool = False, kind: str = "research-ai_debate",
         subject: str = "", timeout: int = 600, session_id: str = "", resume: bool = False,
         tag: str = "misc", meta: dict | None = None, add_dirs: tuple[str, ...] = ()) -> dict:
    if spent() >= BUDGET_USD:
        raise RuntimeError(f"研究预算 {BUDGET_USD} 美元已用完（{spent()}）")
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
    if session_id:
        cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    else:
        cmd += ["--no-session-persistence"]
    if effort:
        cmd += ["--effort", effort]
    if schema:
        cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
    if tools:
        cmd += ["--tools", tools, "--allowed-tools", tools]
    else:
        cmd += ["--tools", ""]
    for d in add_dirs:
        cmd += ["--add-dir", d]
    if system:
        cmd += (["--system-prompt", system] if system_replace else ["--append-system-prompt", system])
    WORK.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    t0 = time.time()
    outer, err = None, ""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(WORK),
                           stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        outer = ai_usage._parse_outer(r.stdout)
        if outer is None:
            err = (r.stderr or r.stdout or f"退出码 {r.returncode}")[-400:]
    except subprocess.TimeoutExpired:
        err = f"超过 {timeout} 秒"
    sec = round(time.time() - t0, 1)
    ai_usage.record(kind, subject, outer, {"model_req": model, "effort": effort, **({"timeout": timeout} if err.startswith("超过") else {})})
    res = None
    if outer is not None:
        if outer.get("is_error"):
            err = str(outer.get("result", ""))[:400]
        res = outer.get("structured_output")
        if schema and not isinstance(res, dict):
            txt = str(outer.get("result", ""))
            i, j = txt.find("{"), txt.rfind("}")
            try:
                res = json.loads(txt[i:j + 1]) if i >= 0 else None
            except Exception:  # noqa: BLE001
                res = None
        if not schema:
            res = {"text": str(outer.get("result", ""))}
    u = (outer or {}).get("usage") or {}
    st = u.get("server_tool_use") or {}
    row = {"tag": tag, "subject": subject, "model": model, "effort": effort, "sec": sec,
           "usd": round(float((outer or {}).get("total_cost_usd") or 0), 4),
           "turns": int((outer or {}).get("num_turns") or 0),
           "web_search": int(st.get("web_search_requests") or 0),
           "web_fetch": int(st.get("web_fetch_requests") or 0),
           "models_used": sorted(((outer or {}).get("modelUsage") or {}).keys()),
           "error": err, "result": res, "meta": meta or {},
           "result_text": None if res else str((outer or {}).get("result", ""))[:2000]}
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / f"{tag}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return row
