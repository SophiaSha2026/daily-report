"""
AI 讨论（2026-10-13 用户要求）：控制台里和 Claude 多轮讨论一只股票、一份清单、或学习状态。

    讨论对象 kind     材料（第一轮塞进提示，之后续聊靠 claude 的会话）
    stock  单只股票   tools/ai_analyze.py --materials：模型分数 / K 线事实 / 股东户数 / 减持定增 / 市场环境
    breakout 起涨预测  今天的清单 A / B、run_meta、验证集成绩、分数分档、模型和重训验收、近期清单真值、市场环境
    pullback 长期调整突破  清单 A / B、run_meta（含卡在哪一步的统计）、以前成立过的、config.yaml 的规则段
    learn  学习状态   gui/learn.summary()：模型年龄、重训验收、效果、漂移、待批提案

每一轮 AI 同时给两样东西（--json-schema 强制）：
    reply    聊天窗口里的回答
    panel    右边面板：结论、立场和置信度、重点、要点、原因（标明 数据 / 新闻 / 代码 / 推断）、没核实的
    changes  只有用户明确要求改代码或参数时才有；每条存成一个提案（tools/ai_review.py 的提案库），
             用户在窗口里点「批准」才落地：参数走白名单 + 月度重估链，代码改动跑三条自测 + pyflakes，
             不过就自动撤回（ai_review.apply_edits）。AI 自己一个字都改不了（硬约束 1：不参与排序）。

工具只给只读的：WebSearch / WebFetch 查新闻公告，Read / Grep / Glob 读仓库（--add-dir 仓库根）。
claude 的会话放在 state/ai_chat/work 下（cwd），不混进开发用的会话列表。
会话记录 state/ai_chat/<sid>.json（gitignore），用量照常记 state/ai_usage.jsonl。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bjtime import now_bj  # noqa: E402

import ai_usage  # noqa: E402

CHAT = ROOT / "state" / "ai_chat"
WORK = CHAT / "work"
KINDS = {"stock": "单只股票", "breakout": "起涨预测清单", "pullback": "长期调整突破清单", "learn": "学习状态"}
FIRST_ASK = {
    "stock": "完整分析这只股票未来 20 个交易日的走势：先查最近两周的新闻公告和题材，再结合材料，说清楚和模型分数一致还是分歧。",
    "breakout": "分析今天的起涨预测清单：哪几只最值得看、哪几只有明显风险（结合新闻公告），和模型分数有没有分歧。",
    "pullback": "分析今天的长期调整突破清单 A 和 B：每只的形态质量、风险、最近的新闻公告。",
    "learn": "现在的学习状态怎么样：模型新不新、效果如何、有没有该调整或该我拍板的地方？",
}
TOOLS = "WebSearch,WebFetch,Read,Grep,Glob"
MAX_BUSY = 3
TURN_TIMEOUT = 1500
_LOCK = threading.Lock()

REASON = {"type": "object",
          "properties": {"point": {"type": "string"},
                         "basis": {"type": "string", "enum": ["数据", "新闻", "代码", "推断"]},
                         "source": {"type": "string"}},
          "required": ["point", "basis"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "panel": {"type": "object", "properties": {
            "title": {"type": "string"},
            "conclusion": {"type": "string"},
            "stance": {"type": "string", "enum": ["看多", "看空", "观望", "不适用"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "key_points": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
            "bullets": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
            "reasons": {"type": "array", "maxItems": 8, "items": REASON},
            "unverified": {"type": "array", "items": {"type": "string"}},
        }, "required": ["conclusion", "key_points", "bullets", "reasons"]},
        "changes": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["param", "code"]},
            "title": {"type": "string"},
            "why": {"type": "string"},
            "param": {"type": "string"},
            "value": {"type": "number"},
            "edits": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {
                "file": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                "required": ["file", "old", "new"]}},
            "expected_effect": {"type": "string"},
            "risk": {"type": "string"},
            "how_to_verify": {"type": "string"},
        }, "required": ["type", "title", "why", "risk"]}},
    },
    "required": ["reply", "panel"],
}

SYSTEM = (
    "你在一个 A 股选股流水线的本地控制台里和用户讨论。仓库在 C:/home/daily-report（规矩在它的 CLAUDE.md）。"
    "中文，简洁，不用破折号，要具体事实和数字，不说「值得关注」「有望受益」这类套话，不给交易建议式的口号。"
    "只按证据说话：每条原因标明来源类型（数据 = 我给的材料或仓库产物，新闻 = 你查到的并给链接，"
    "代码 = 你读到的仓库代码并写文件名，推断 = 你自己的判断）；查不到、核实不了的放进 unverified，不要编。"
    "输出两部分：reply 是聊天窗口里对用户这一句的直接回答（短）；panel 是到目前为止整场讨论的结构化总结"
    "（每轮都给完整的最新版：conclusion 一两句结论，key_points 最重要的一到四点，bullets 要点，reasons 原因）。"
    "硬约束：清单排序 100% 由模型分数和规则决定，你只分析，不参与排序，也不能直接改任何东西。"
    "只有用户明确要求改代码或参数时，才在 changes 里给提案，用户在界面上批准后系统才执行。"
    "参数提案只能是 SCORE_MIN（90~100）、CAP_A（3~20）、MIN_STREAK（1~5），写在 param / value。"
    "代码提案写 edits：先用 Read 读原文件，old 必须是文件里逐字存在且只出现一次的片段（含缩进），new 是替换后的；"
    "改动越小越好；不许改 src/breakout/features.py、chips.py、label.py、build.py（模型指纹）、"
    "tools/local.env、state/、data/、.github/。批准后系统会跑三条自测和 pyflakes，不过就自动撤回。"
    "工具：WebSearch / WebFetch 查新闻和公告（优先交易所、公司公告、主流财经媒体）；"
    "Read / Grep / Glob 只读仓库里的代码和产物。用户要你审阅代码或算法时，先读再说，引用文件和行。"
)


# ---------------------------------------------------------------
#  材料
# ---------------------------------------------------------------
def _j(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _jsonl_tail(p: Path, n: int) -> list:
    try:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]
    except Exception:  # noqa: BLE001
        return []


def materials_stock(code: str) -> dict:
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "ai_analyze.py"), code, "--materials"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                       cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                       stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = (r.stdout or "").strip().splitlines()
    try:
        return json.loads(lines[-1])
    except Exception:  # noqa: BLE001
        return {"error": (r.stderr or r.stdout or "")[-300:]}


def materials_breakout() -> dict:
    ob, sb = ROOT / "out_breakout", ROOT / "state" / "breakout"
    grid = _j(ob / "window_grid.json", {}) or {}
    model = _j(sb / "model.json", {}) or {}
    imp = model.get("importance_gain") or {}
    truth = _j(sb / "truth.json", {}) or {}
    return {
        "run_meta": _j(ob / "run_meta.json", {}),
        "list_a": _j(ob / "list_a.json", []),
        "list_b": _j(ob / "list_b.json", []),
        "validation": {"base": grid.get("base"), "days": grid.get("days"),
                       "rows": [r for r in grid.get("grid", []) if str(r.get("kind", "")).startswith("W5")]},
        "score_bins": _j(ob / "score_calibration.json", {}),
        "board_hit": _j(ob / "board_hit.json", {}),
        "model": {k: model.get(k) for k in ("fit_date", "train_cut", "weighting")}
                 | {"top_features": sorted(imp, key=imp.get, reverse=True)[:12]},
        "refit_status": _j(sb / "refit_status.json", {}),
        "recent_lists_truth": (truth.get("lists") or [])[-12:],
        "market": _jsonl_tail(ROOT / "state" / "regime_daily.jsonl", 10),
        "notes": "清单 A：分数 ≥ SCORE_MIN 且当天前 CAP_A 名，剔除 ST / 减持 / 解禁 / 增发；"
                 "标签：次日开盘买入后 20 个交易日内最高价涨超 50%。validation 是逐月滚动的验证集成绩，"
                 "recent_lists_truth 是实发清单之后真实走势（bars 不满 20 的还没定论）。"
                 "设计和实验记录在 docs/breakout_log.md，代码在 src/breakout/。",
    }


def materials_pullback() -> dict:
    op = ROOT / "out_pullback"
    try:
        cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
        i = cfg.find("pullback:")
        cfg = cfg[i:i + 6000] if i >= 0 else cfg[:6000]
    except Exception:  # noqa: BLE001
        cfg = ""
    return {"run_meta": _j(op / "run_meta.json", {}), "list_a": _j(op / "selected.json", []),
            "list_b": _j(op / "list_b.json", []), "history_recent": (_j(op / "history.json", []) or [])[-15:],
            "config_pullback": cfg,
            "notes": "纯规则：横盘 -> 首阳 -> 缩量调整 -> 二次进攻（清单 A）；B 是走完调整、还没二次进攻的。"
                     "大多数交易日 A 是 0 只，三年 54 次。规则和阈值来历在 src/pullback.py 的模块说明。"}


def materials_learn() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from gui import learn
    out = learn.summary()
    out["latest_ai_review"] = _j(ROOT / "state" / "ai_review" / "latest_alert.json", {})
    return out


def materials(kind: str, code: str = "") -> dict:
    if kind == "stock":
        return materials_stock(code)
    return {"breakout": materials_breakout, "pullback": materials_pullback, "learn": materials_learn}[kind]()


def first_prompt(kind: str, code: str, mat: dict, ask: str) -> str:
    subj = f"{code} {mat.get('name', '')}".strip() if kind == "stock" else KINDS[kind]
    # 只写到日期：同一天同一只、材料不变时提示词逐字相同，以后加缓存才命中得了（计划 2026-10-15 第 0.5 项）
    head = (f"讨论对象：{subj}。今天北京时间 {now_bj():%Y-%m-%d}。\n"
            "下面是本机备好的材料（JSON），后面几轮讨论都基于它，需要更多就自己读仓库或联网查。\n")
    if kind == "stock":
        head += ("model 是我们的起涨预测模型：分数 0~100 是全市场排名，≥97 且当天前 10 才上清单；perf_bin 是这一档"
                 "历史上「次日开盘买入、20 个交易日内最高价涨超 50%」的命中率，base 是随便买一只的比例。"
                 "kline 是近 60 根 K 线的事实（前复权），corp_events 是减持 / 定增核查，market 是近 10 天市场环境。\n")
    return head + json.dumps(mat, ensure_ascii=False, default=str)[:120_000] + f"\n\n用户：{ask}"


# ---------------------------------------------------------------
#  会话
# ---------------------------------------------------------------
def _path(sid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f\-]{36}", sid or ""):
        raise ValueError("bad sid")
    return CHAT / f"{sid}.json"


def load(sid: str) -> dict | None:
    return _j(_path(sid), None)


def _save(s: dict) -> None:
    CHAT.mkdir(parents=True, exist_ok=True)
    p = _path(s["sid"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def sessions(kind: str = "", code: str = "", n: int = 20) -> list[dict]:
    if not CHAT.exists():
        return []
    out = []
    for p in sorted(CHAT.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        s = _j(p, None)
        if not s or (kind and s.get("kind") != kind) or (code and s.get("code") != code):
            continue
        out.append({"sid": s["sid"], "kind": s["kind"], "code": s.get("code", ""), "title": s.get("title", ""),
                    "created": s.get("created", ""), "turns": len(s.get("turns", [])),
                    "usd": round(sum(t.get("usd", 0) for t in s.get("turns", [])), 2)})
        if len(out) >= n:
            break
    return out


def _busy_count() -> int:
    n = 0
    for p in CHAT.glob("*.json") if CHAT.exists() else []:
        s = _j(p, {}) or {}
        if s.get("busy"):
            n += 1
    return n


def new_session(kind: str, code: str = "") -> dict:
    if kind not in KINDS:
        raise ValueError("kind")
    if kind == "stock" and not re.fullmatch(r"\d{6}", code or ""):
        raise ValueError("code")
    s = {"sid": str(uuid.uuid4()), "kind": kind, "code": code if kind == "stock" else "",
         "title": (code if kind == "stock" else KINDS[kind]), "created": now_bj().isoformat(timespec="seconds"),
         "claude_started": False, "busy": False, "turns": []}
    _save(s)
    return s


def _proposals_from(s: dict, turn_no: int, changes: list) -> list[dict]:
    """AI 给的改动 -> 提案库（pending）。返回给界面的简表。"""
    if not changes:
        return []
    sys.path.insert(0, str(ROOT / "tools"))
    import ai_review as AR
    out = []
    for i, c in enumerate(changes[:3], 1):
        pid = f"chat-{s['sid'][:8]}-{turn_no}-{i}"
        pr = {"type": c.get("type", "code"), "title": c.get("title", ""), "detail": c.get("why", ""),
              "expected_effect": c.get("expected_effect", ""), "risk": c.get("risk", ""),
              "how_to_verify": c.get("how_to_verify", ""), "source": "chat", "session": s["sid"],
              "subject": s.get("title", "")}
        if pr["type"] == "param":
            pr.update(param=c.get("param", ""), value=c.get("value"))
        else:
            pr["edits"] = c.get("edits") or []
        AR._save_proposal(pr, pid, now_bj().strftime("%Y-%m-%d"))
        out.append({"id": pid, **pr})
    return out


def _run_turn(sid: str, prompt: str, model: str, effort: str, resume: bool) -> None:
    def ev(note: str) -> None:
        with _LOCK:
            s = load(sid) or {}
            s.setdefault("progress", []).append(f"{now_bj():%H:%M:%S} {note}")
            s["progress"] = s["progress"][-30:]
            _save(s)

    WORK.mkdir(parents=True, exist_ok=True)
    r, outer, err = ai_usage.run_claude(
        prompt, SCHEMA, model=model, tools=TOOLS, timeout=TURN_TIMEOUT, system=SYSTEM,
        kind="chat", subject=sid[:8], effort=effort, session_id=sid, resume=resume,
        cwd=str(WORK), add_dirs=(str(ROOT),), on_event=ev)
    with _LOCK:
        s = load(sid) or {}
        turn_no = len(s.get("turns", []))
        t = {"role": "ai", "at": now_bj().isoformat(timespec="seconds"), "model": model, "effort": effort,
             "usd": round(float((outer or {}).get("total_cost_usd") or 0), 3),
             "models_used": sorted(((outer or {}).get("modelUsage") or {}).keys())}
        if err or not r:
            t["error"] = err or "没有结构化结果"
        else:
            t["reply"], t["panel"] = r.get("reply", ""), r.get("panel") or {}
            try:
                t["changes"] = _proposals_from(s, turn_no, r.get("changes") or [])
            except Exception as e:  # noqa: BLE001
                t["changes"], t["changes_error"] = [], str(e)[:200]
            s["claude_started"] = True
        s.setdefault("turns", []).append(t)
        s["busy"] = False
        s["progress"] = []
        _save(s)


def send(sid: str, message: str, model: str = "", effort: str = "") -> tuple[dict | None, str]:
    """给会话发一句（第一句可以是空串 = 用这个对象的默认开场问题）。后台线程跑，立即返回。"""
    s = load(sid)
    if not s:
        return None, "没有这个会话"
    if s.get("busy"):
        return None, "上一句还没回完"
    if _busy_count() >= MAX_BUSY:
        return None, f"同时在想的讨论已经有 {MAX_BUSY} 个，等一个回完"
    ms = ai_usage.models()
    ids = {m["id"] for m in ms["models"]} | {"opus", "sonnet", "haiku", "fable", ai_usage.MODEL}
    model = model or ai_usage.MODEL
    if model not in ids:
        return None, f"不认识的模型 {model}"
    if effort and effort not in (ms.get("efforts") or []):
        return None, f"不认识的推理等级 {effort}"
    message = (message or "").strip() or FIRST_ASK[s["kind"]]
    if len(message) > 8000:
        return None, "一句话太长（8000 字以内）"
    resume = bool(s.get("claude_started"))
    with _LOCK:
        s["turns"].append({"role": "user", "text": message, "at": now_bj().isoformat(timespec="seconds")})
        s["busy"], s["progress"] = True, [f"{now_bj():%H:%M:%S} " + ("续上讨论" if resume else "备材料")]
        _save(s)

    def work() -> None:
        try:
            if resume:
                prompt = message
            else:
                mat = materials(s["kind"], s.get("code", ""))
                with _LOCK:
                    s2 = load(sid) or s
                    if s["kind"] == "stock" and mat.get("name"):
                        s2["title"] = f"{s['code']} {mat['name']}"
                    s2["progress"].append(f"{now_bj():%H:%M:%S} 材料备好，交给 Claude")
                    _save(s2)
                prompt = first_prompt(s["kind"], s.get("code", ""), mat, message)
            _run_turn(sid, prompt, model, effort, resume)
        except Exception as e:  # noqa: BLE001
            with _LOCK:
                s3 = load(sid) or s
                s3.setdefault("turns", []).append({"role": "ai", "at": now_bj().isoformat(timespec="seconds"),
                                                   "error": f"出错：{e}"[:300], "usd": 0})
                s3["busy"] = False
                _save(s3)

    threading.Thread(target=work, daemon=True).start()
    return load(sid), ""


def recover() -> None:
    """控制台重启时，上一个进程里没回完的会话标回空闲（线程已经没了）。"""
    if not CHAT.exists():
        return
    for p in CHAT.glob("*.json"):
        s = _j(p, None)
        if s and s.get("busy"):
            s["busy"] = False
            s.setdefault("turns", []).append({"role": "ai", "at": now_bj().isoformat(timespec="seconds"),
                                              "error": "控制台重启，这一句没回完，再发一次", "usd": 0})
            _save(s)
