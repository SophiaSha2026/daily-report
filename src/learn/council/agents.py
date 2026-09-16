"""
起 `claude -p` 进程跑一个视角，并把它的过程记下来。

每个视角一个进程：
    claude -p <提纲+证据路径> --model <m> --effort <e> --output-format stream-json
           --verbose --json-schema <schema> --allowed-tools <白名单> --max-turns N

stream-json 每行一个事件，我们从里面抽出「说了什么 / 读了什么 / 查了什么 /
最后结论」写成 <lens>.trace.jsonl，面板按时间线展示——这就是用户要的
「LLM 的完整分析过程」。结构化结果在 result 事件的 structured_output 里。

认证复用 llm_local.find_token；失败返回 None，不抛（会诊整体 fail-open）。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from learn import llm_local
from learn.council import schemas as S

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROMPTS = ROOT / "prompts" / "council"

QUERY_TOOL = "Bash(python tools/council_query.py:*)"
TOOLS = {
    "gap_where": f"Read,{QUERY_TOOL}",
    "noise_or_real": f"Read,{QUERY_TOOL}",
    "features": f"Read,{QUERY_TOOL}",
    "regime": f"Read,{QUERY_TOOL},WebSearch,WebFetch",
    "data_quality": f"Read,{QUERY_TOOL}",
    "improve": f"Read,{QUERY_TOOL}",
    "chair": "Read",
}


def _summ(x, n: int = 400) -> str:
    if isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            s = str(x)
    return s[:n]


def run_lens(lens: str, prompt: str, out_dir: Path, model: str, effort: str,
             timeout: int, max_turns: int = 150) -> dict | None:
    """跑一个视角。返回 sanitize 过的对象；失败返回 None 并在 out_dir 留 <lens>.error。

    过程写 <lens>.trace.jsonl；结果写 <lens>.json；原始 stream 写 <lens>.raw.jsonl
    （排查用，面板不读）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = S.CHAIR_SCHEMA if lens == "chair" else S.LENS_SCHEMA
    env = dict(os.environ)
    tok, _src = llm_local.find_token()
    if tok:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    env["PYTHONIOENCODING"] = "utf-8"
    base = ["claude", "--model", model, "--effort", effort,
            "--output-format", "stream-json", "--verbose",
            "--json-schema", json.dumps(schema, ensure_ascii=False),
            "--allowed-tools", TOOLS.get(lens, "Read"),
            # 只读是硬约束：Read/Grep 默认就能用，写类工具一律禁掉
            "--disallowedTools", "Write,Edit,MultiEdit,NotebookEdit"]
    cmd = base + ["-p", prompt, "--max-turns", str(max_turns)]
    t0 = time.time()
    trace: list[dict] = []
    raw = out_dir / f"{lens}.raw.jsonl"
    result_obj, err, sid = _stream(cmd, env, raw, trace, t0, timeout)

    # 到了步数上限没来得及写结论：接着那个会话再给它一次机会，只许输出不许再查。
    # 实测（2026-09-16 冒烟）Opus max 一个视角 40 次工具调用还没停手，$5 的分析
    # 不能因为差一步全丢。
    if (result_obj or {}).get("subtype") == "error_max_turns" and sid:
        log.info("视角 %s 到步数上限（%s 轮），resume 逼出结论", lens, max_turns)
        cmd2 = base[:1] + ["--resume", sid] + base[1:] + [
            "-p", "已到步数上限。不要再调用任何工具，现在就把你已经得到的分析按 schema 输出。"
                  "查过的部分写发现，没查到的写进 questions。",
            "--max-turns", "3"]
        left = max(60, int(timeout - (time.time() - t0)))
        r2, e2, _ = _stream(cmd2, env, out_dir / f"{lens}.raw2.jsonl", trace, t0, left)
        if r2 and not r2.get("is_error"):
            result_obj, err = r2, ""
        else:
            err = err or e2 or "resume 也没拿到结论"

    dur = time.time() - t0
    (out_dir / f"{lens}.trace.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in trace), encoding="utf-8")

    obj = None
    if result_obj:
        if result_obj.get("is_error"):
            err = err or _summ(result_obj.get("result"), 200)
        so = result_obj.get("structured_output")
        if isinstance(so, dict):
            obj = so
        elif isinstance(result_obj.get("result"), str):
            obj = llm_local._extract_json(result_obj["result"])
    meta = {"lens": lens, "model": model, "effort": effort, "seconds": round(dur, 1),
            "turns": (result_obj or {}).get("num_turns"),
            "cost_usd": (result_obj or {}).get("total_cost_usd"),
            "n_tool_calls": sum(1 for x in trace if x["kind"] == "tool_use"),
            "denials": len((result_obj or {}).get("permission_denials") or []),
            "error": err or ("解析不出结构化输出" if obj is None else "")}
    if obj is None:
        (out_dir / f"{lens}.error").write_text(meta["error"], encoding="utf-8")
        (out_dir / f"{lens}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
        log.warning("视角 %s 失败：%s（%.0fs）", lens, meta["error"], dur)
        return None
    clean = S.sanitize_chair(obj) if lens == "chair" else S.sanitize_lens(obj, lens)
    clean["_meta"] = meta
    (out_dir / f"{lens}.json").write_text(json.dumps(clean, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (out_dir / f"{lens}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    log.info("视角 %s 完成：%.0fs，%d 轮，%d 次工具，$%.2f", lens, dur,
             meta["turns"] or 0, meta["n_tool_calls"], meta["cost_usd"] or 0)
    return clean


def _stream(cmd: list[str], env: dict, raw: Path, trace: list[dict], t0: float,
            timeout: int) -> tuple[dict | None, str, str]:
    """跑一次 CLI，边读 stream-json 边记过程。返回 (result 事件, 错误, session_id)。"""
    result_obj: dict | None = None
    err, sid = "", ""
    try:
        with raw.open("w", encoding="utf-8") as rf:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 stdin=subprocess.DEVNULL, cwd=str(ROOT), env=env)
            assert p.stdout is not None
            # 看门狗：CLI 卡住不吐行时 readline 会一直阻塞，只能从外面杀
            killed = {"v": False}

            def _kill():
                killed["v"] = True
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
            wd = threading.Timer(timeout, _kill)
            wd.daemon = True
            wd.start()
            for line in iter(p.stdout.readline, b""):
                s = line.decode("utf-8", "replace").rstrip("\n")
                rf.write(s + "\n")
                try:
                    ev = json.loads(s)
                except Exception:  # noqa: BLE001
                    continue
                if ev.get("type") == "system" and ev.get("subtype") == "init":
                    sid = str(ev.get("session_id") or "")
                _absorb(ev, trace, t0)
                if ev.get("type") == "result":
                    result_obj = ev
            wd.cancel()
            if killed["v"]:
                err = f"超时 {timeout}s"
            try:
                p.wait(timeout=10)
            except Exception:  # noqa: BLE001
                p.kill()
            if p.returncode not in (0, None) and not err and not result_obj:
                se = (p.stderr.read() if p.stderr else b"").decode("utf-8", "replace")
                err = f"退出码 {p.returncode} {se[:200]}"
    except FileNotFoundError:
        err = "本机没有 claude CLI"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return result_obj, err, sid


def _absorb(ev: dict, trace: list[dict], t0: float) -> None:
    """stream-json 事件 -> 过程时间线条目。"""
    t = round(time.time() - t0, 1)
    typ = ev.get("type")
    if typ == "assistant":
        for blk in (ev.get("message") or {}).get("content") or []:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text" and blk.get("text"):
                trace.append({"t": t, "kind": "text", "text": _summ(blk["text"], 2000)})
            elif blk.get("type") == "tool_use":
                inp = blk.get("input") or {}
                brief = inp.get("command") or inp.get("file_path") or inp.get("query") \
                    or inp.get("url") or _summ(inp, 200)
                trace.append({"t": t, "kind": "tool_use", "name": blk.get("name"),
                              "input": _summ(brief, 300)})
            elif blk.get("type") == "thinking":
                trace.append({"t": t, "kind": "thinking"})
    elif typ == "user":
        for blk in (ev.get("message") or {}).get("content") or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_result":
                c = blk.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                trace.append({"t": t, "kind": "tool_result", "text": _summ(c, 500),
                              "chars": len(c) if isinstance(c, str) else 0})
    elif typ == "result":
        trace.append({"t": t, "kind": "result", "ok": not ev.get("is_error"),
                      "turns": ev.get("num_turns"), "cost_usd": ev.get("total_cost_usd")})


def build_prompt(lens: str, ev_dir: Path, date: str) -> str:
    """提纲 = 公共说明 + 视角提纲 + 输入路径。文件都在 prompts/council/。"""
    common = (PROMPTS / "common.md").read_text(encoding="utf-8")
    body = (PROMPTS / f"{lens}.md").read_text(encoding="utf-8")
    paths = {"evidence": str(ev_dir / "evidence_slim.json"),
             "evidence_full": str(ev_dir / "evidence.json")}
    if lens == "chair":
        paths["opinions"] = ", ".join(str(ev_dir / f"{x}.json") for x in S.LENSES
                                      if (ev_dir / f"{x}.json").exists())
    inputs = "\n".join(f"- {k}: {v}" for k, v in paths.items())
    return (f"{common}\n\n# 本次视角：{S.LENS_NAME.get(lens, lens)}（{lens}）\n\n{body}"
            f"\n\n# 输入文件（用 Read 读；绝对路径）\n\n会诊日期 {date}\n{inputs}\n")


def run_parallel(lenses: list[str], ev_dir: Path, date: str, model: str,
                 effort: str, timeout: int, max_parallel: int,
                 max_turns: int = 150) -> dict[str, dict | None]:
    """并行跑多个视角，返回 {lens: 结果或 None}。

    max_turns 给大（一次工具调用算一轮，深挖一次 30~60 轮很正常），
    真正的上限是 timeout：到点看门狗杀进程。
    """
    out: dict[str, dict | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futs = {ln: ex.submit(run_lens, ln, build_prompt(ln, ev_dir, date), ev_dir,
                              model, effort, timeout, max_turns) for ln in lenses}
        for ln, f in futs.items():
            try:
                out[ln] = f.result()
            except Exception as e:  # noqa: BLE001
                log.warning("视角 %s 异常: %s", ln, e)
                out[ln] = None
    return out
