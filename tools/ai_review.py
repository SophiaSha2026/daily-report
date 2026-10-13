"""
Claude 深度参与现有分析（计划 2026-10-10 第 9 项）。三个时机，一个脚本：

    python tools/ai_review.py --daily [--date 2026-10-09]   每日复盘：起涨预测发信后由 local_run 调（fail-open）
    python tools/ai_review.py --weekly                      每周深评：周五发信后调，多看一周真值和漂移
    python tools/ai_review.py --diff                        改动前复核：把当前工作区的 diff 按硬约束和教训逐条审
    python tools/ai_review.py --apply <提案id>               用户在控制台批准后，落地一个提案（param / code）
    python tools/ai_review.py --list                        列出待批准的提案

代码提案带 edits（AI 讨论窗口里用户要求改代码时给，src/ai_chat.py）：批准后 apply_edits 逐条做
「原文只出现一次 -> 替换」，跑三条自测 + pyflakes，任何一条不过就把文件原样写回、提案记 failed；
都过了在 git_lock 里本地提交（只提交这几个文件），下一次流程推送时一起推上去。
不带 edits 的代码提案照旧只记待办。

边界（用户 2026-10-10 拍板）：AI 只分析、复盘、告警、提案。参数改动和代码改动**只能提案**，
用户在控制台批准后才落地；清单排序 AI 不碰。失败不阻断发信。费用记在 src/ai_usage.py。

产物：
    state/ai_review/<日>.json                 当天复盘全文（schema 见 REVIEW_SCHEMA）
    state/ai_review/latest_alert.json         最近一次复盘的严重度和一句话（邮件抬头、控制台首页读）
    state/ai_review/proposals/<id>.json       提案（pending / approved / rejected / applied / failed）
    state/ai_review/weekly_<周五>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

import ai_usage                 # noqa: E402
from bjtime import now_bj, today_bj  # noqa: E402

STATE = ROOT / "state" / "ai_review"
PROPOSALS = STATE / "proposals"
PARAM_WHITELIST = {"SCORE_MIN": (90, 100), "CAP_A": (3, 20), "MIN_STREAK": (1, 5)}

FINDING = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["正常", "注意", "严重"]},
        "where": {"type": "string"},
        "evidence": {"type": "string"},
        "why": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": ["severity", "where", "evidence", "why"],
}
PROPOSAL = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["param", "code", "process"]},
        "title": {"type": "string"},
        "detail": {"type": "string"},
        "param": {"type": "string"},
        "value": {"type": "number"},
        "expected_effect": {"type": "string"},
        "risk": {"type": "string"},
        "how_to_verify": {"type": "string"},
    },
    "required": ["type", "title", "detail", "expected_effect", "risk"],
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["正常", "注意", "严重"]},
        "summary": {"type": "string"},
        "findings": {"type": "array", "maxItems": 12, "items": FINDING},
        "proposals": {"type": "array", "maxItems": 5, "items": PROPOSAL},
        "checked": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["severity", "summary", "findings", "proposals"],
}
DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["可以提交", "先修再提交", "不要提交"]},
        "summary": {"type": "string"},
        "findings": {"type": "array", "maxItems": 15, "items": FINDING},
    },
    "required": ["verdict", "summary", "findings"],
}

SYSTEM = (
    "你是这个 A 股选股流水线的驻场审计员，中文回复，只按证据说话，要具体事实和数字，"
    "不要套话。你看到的材料都是本机算好的 JSON 和日志片段。判「严重」的标准：会导致当天清单错、"
    "发不出、发错、数字和口径对不上、数据日期对不上目标日、模型在不该重训时重训或该重训没重训。"
    "判「注意」：有异常但当天结果没错。其余「正常」。每条发现都要给 where（哪个文件 / 哪一步）和 "
    "evidence（原文或数字）。提案只许三类：param（改 overrides.json 里的 SCORE_MIN / CAP_A / MIN_STREAK）、"
    "code（描述改哪个文件改什么，不要写补丁全文）、process（流程或监控的改法）。"
    "不要提议改清单的排序逻辑、不要提议让 AI 自动改任何东西。"
)


# ---------------------------------------------------------------
#  材料
# ---------------------------------------------------------------
def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _tail_block(p: Path, marker: str, max_lines: int = 160) -> str:
    """日志里最后一次**真正开跑**的运行块（含 ##STEP 的那块，不是「已经跑完，跳过」那种
    秒退的敲门），砍到 max_lines。"""
    if not p.exists():
        return ""
    txt = p.read_text(encoding="utf-8", errors="replace")[-600_000:]
    blocks = txt.split(marker)
    chosen = ""
    for blk in reversed(blocks[1:]):
        if "##STEP" in blk or "邮件已发出" in blk or "退出码 1" in blk:
            chosen = blk
            break
    if not chosen:
        chosen = blocks[-1]
    lines = (marker + chosen).splitlines()
    return "\n".join(lines[-max_lines:])


def _jsonl_tail(p: Path, n: int) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def gather_daily(date: str) -> dict:
    sb = ROOT / "state" / "breakout"
    truth = _json(sb / "truth.json") or {}
    lists = [{k: L.get(k) for k in ("date", "n", "bars", "final", "hits_final", "n_final", "hit_rate",
                                    "hits_open", "n_open", "hit_rate_open", "base_final", "base_open_final")}
             for L in (truth.get("lists") or [])[-12:]]
    m = {
        "date": date, "now": now_bj().isoformat(timespec="seconds"),
        "breakout": {
            "run_meta": _json(ROOT / "out_breakout" / "run_meta.json"),
            "list_a": _json(ROOT / "out_breakout" / "list_a.json"),
            "list_b": _json(ROOT / "out_breakout" / "list_b.json"),
            "mail_sent": _json(ROOT / "out_breakout" / "mail_sent.json"),
            "sent_marker": (ROOT / "state" / "sent" / f"breakout_{date}.json").exists(),
            "refit_status": _json(sb / "refit_status.json"),
            "model": {k: v for k, v in (_json(sb / "model.json") or {}).items()
                      if k in ("fit_date", "train_cut", "weighting", "fingerprint")},
            "update_status": _json(sb / "update_status.json"),
            "score_stats_60d": _jsonl_tail(sb / "score_stats.jsonl", 60),
            "truth_recent_lists": lists,
            "truth_by": truth.get("by"),
            "log_tail": _tail_block(ROOT / "tools" / "local_flow_breakout.log", "=== run_local breakout ==="),
        },
        "pullback": {
            "run_meta": _json(ROOT / "out_pullback" / "run_meta.json"),
            "mail_sent": _json(ROOT / "out_pullback" / "mail_sent.json"),
            "sent_marker": (ROOT / "state" / "sent" / f"pullback_{date}.json").exists(),
            "log_tail": _tail_block(ROOT / "tools" / "local_flow_pullback.log", "=== run_local pullback ==="),
        },
        "push_status": _json(ROOT / "state" / "push_status.json"),
        "regime_10d": _jsonl_tail(ROOT / "state" / "regime_daily.jsonl", 10),
        "ai_usage": ai_usage.totals(),
        "rules": {"score_min_default": 97, "cap_a_default": 10, "model_max_age_days": 30,
                  "mail_perf_note": "邮件印的成绩按次日开盘买入口径（买得到），收盘口径另印对照"},
    }
    return m


def gather_weekly() -> dict:
    m = gather_daily(today_bj())
    sb = ROOT / "state" / "breakout"
    m["score_stats_all"] = _jsonl_tail(sb / "score_stats.jsonl", 250)
    m["ai_usage_rows"] = ai_usage.rows()[-200:]
    m["regime_60d"] = _jsonl_tail(ROOT / "state" / "regime_daily.jsonl", 60)
    log = ROOT / "docs" / "breakout_log.md"
    if log.exists():
        txt = log.read_text(encoding="utf-8")
        m["breakout_log_tail"] = txt[-12000:]
    return m


def gather_diff() -> dict:
    def git(*a: str) -> str:
        r = subprocess.run(["git", *a], cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
        return r.stdout
    diff = git("diff") + git("diff", "--cached")
    untracked = git("ls-files", "--others", "--exclude-standard").split()
    extra = ""
    for f in untracked[:20]:
        p = ROOT / f
        if p.suffix in (".py", ".md", ".yml", ".cmd", ".ps1") and p.stat().st_size < 60_000:
            extra += f"\n\n=== 新文件 {f} ===\n" + p.read_text(encoding="utf-8", errors="replace")
    cl = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    i = cl.find("## 硬约束")
    j = cl.find("### 本地为主、云端托底")
    return {"diff": diff[:120_000], "untracked": extra[:60_000],
            "constraints_and_lessons": cl[i:j][:60_000] if i >= 0 and j > i else cl[:40_000]}


# ---------------------------------------------------------------
#  提示词
# ---------------------------------------------------------------
def prompt_daily(m: dict, weekly: bool) -> str:
    scope = ("这是每周深评：除了当天，还要看近一周的真值、60 天的打分统计和用量，给出改进提案。"
             if weekly else "这是每日复盘：只看当天这次运行。")
    return (
        f"复盘 {m['date']} 的晚间流水线运行。{scope}\n"
        "逐项核对，核对过的项目写进 checked：\n"
        "1 两条线的 run_meta.date 是否等于目标日；dry 是否为 false；mail_sent 和 sent_marker 是否都在。\n"
        "2 起涨预测：n_qualified / n_ok / rejected / short_excluded 是否异常（对照 score_stats_60d 的分布）；"
        "refit_rejected / drift_note 有没有触发；model.fit_date 距今是否超过 30 天；update_status 里 short 多不多。\n"
        "3 日志尾：有没有 WARNING / ERROR / 超时 / 接手 / 推送失败；退出码 0 但没做事的痕迹"
        "（比如「跳过」后直接结束、产物日期没变）。\n"
        "4 板块构成：清单 A 是不是集中在一个板块；连续天数分布。\n"
        "5 真值：最近已定的几份清单命中和同期基准相比如何（买得到口径 hit_rate_open 为主）。\n"
        "6 推送状态、市场环境（regime_10d）有没有极端值。\n"
        "7 AI 用量是否超警戒线。\n"
        "材料（JSON）：\n" + json.dumps(m, ensure_ascii=False, default=str)[:180_000]
    )


def prompt_diff(m: dict) -> str:
    return (
        "审这次代码改动能不能提交。先读「硬约束和历史教训」，再逐条对照 diff：有没有重犯教训、"
        "有没有同一个概念写了第二份实现（教训 34）、有没有改到模型指纹里的四个文件"
        "（features.py chips.py label.py build.py）、自测产物会不会写进生产目录（教训 17 / 44）、"
        "fail-open 的分支有没有留下可查询的对象（教训 16）、路径是不是写成了导入时定死的常量（教训 44）、"
        "界面文案有没有出现代号（CLAUDE.md 术语表）。每条发现给文件和行的证据。\n\n"
        "=== 硬约束和历史教训 ===\n" + m["constraints_and_lessons"] +
        "\n\n=== diff ===\n" + m["diff"] + m["untracked"]
    )


# ---------------------------------------------------------------
#  落盘
# ---------------------------------------------------------------
def _save_review(kind: str, date: str, r: dict, usd: float) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "date": date, "at": now_bj().isoformat(timespec="seconds"), "usd": round(usd, 3), **r}
    p = STATE / (f"weekly_{date}.json" if kind == "weekly" else f"{date}.json")
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    alert = {"date": date, "kind": kind, "severity": r.get("severity", "正常"),
             "summary": r.get("summary", ""), "at": rec["at"],
             "n_findings": len(r.get("findings") or []), "n_proposals": len(r.get("proposals") or [])}
    (STATE / "latest_alert.json").write_text(json.dumps(alert, ensure_ascii=False, indent=1), encoding="utf-8")
    for i, pr in enumerate(r.get("proposals") or []):
        _save_proposal(pr, f"{date}-{kind}-{i + 1}", date)
    return p


def _save_proposal(pr: dict, pid: str, date: str) -> None:
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    rec = {"id": pid, "date": date, "status": "pending", "created": now_bj().isoformat(timespec="seconds"), **pr}
    (PROPOSALS / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def list_proposals(status: str = "") -> list[dict]:
    if not PROPOSALS.exists():
        return []
    out = []
    for p in sorted(PROPOSALS.glob("*.json")):
        j = _json(p)
        if j and (not status or j.get("status") == status):
            out.append(j)
    return out


def set_status(pid: str, status: str, note: str = "") -> dict | None:
    p = PROPOSALS / f"{pid}.json"
    j = _json(p)
    if not j:
        return None
    j["status"] = status
    j["updated"] = now_bj().isoformat(timespec="seconds")
    if note:
        j["note"] = note
    p.write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return j


def latest_alert() -> dict:
    return _json(STATE / "latest_alert.json") or {}


# ---------------------------------------------------------------
#  落地提案（只在用户批准之后）
# ---------------------------------------------------------------
# 代码改动能碰的范围。模型指纹那四个文件（教训 39）、凭证、运行态、数据、云端 workflow 一律不许
EDIT_PREFIX = ("src/", "tools/", "docs/")
EDIT_FILES = {"config.yaml", "CLAUDE.md", "OPERATIONS.md", "README.md"}
EDIT_DENY = {"src/breakout/features.py", "src/breakout/chips.py", "src/breakout/label.py",
             "src/breakout/build.py", "tools/local.env"}
EDIT_EXT = {".py", ".md", ".yaml", ".yml", ".txt", ".ps1", ".cmd", ".js", ".toml", ".vbs"}
CHECKS = [["src/selftest_pullback.py"], ["src/selftest_gui.py"], ["src/selftest_breakout.py"],
          ["-m", "pyflakes", "src", "tools"]]


def edit_target(rel: str) -> tuple[Path | None, str]:
    """AI 给的文件名 -> 仓库里的绝对路径；不在允许范围就返回 (None, 原因)。"""
    r = str(rel or "").replace("\\", "/").strip()
    if r.lower().startswith(str(ROOT).replace("\\", "/").lower() + "/"):
        r = r[len(str(ROOT)) + 1:]
    r = r.lstrip("./")
    p = (ROOT / r).resolve()
    try:
        r = p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None, f"{rel} 不在仓库里"
    if r in EDIT_DENY:
        return None, f"{r} 不许改（模型指纹 / 凭证）"
    if not (r in EDIT_FILES or r.startswith(EDIT_PREFIX)):
        return None, f"{r} 不在允许改的范围（src/ tools/ docs/ 和几份说明文档）"
    if p.suffix.lower() not in EDIT_EXT:
        return None, f"{r} 这种文件类型不改"
    return p, r


def plan_edits(edits: list) -> tuple[dict, str]:
    """把 edits 在内存里做一遍：{路径: (原文或 None, 新文)}。任何一条对不上就整体拒绝，一个字都不写。"""
    plan: dict[Path, list] = {}
    for e in edits or []:
        p, why = edit_target(e.get("file", ""))
        if p is None:
            return {}, why
        old, new = str(e.get("old", "")), str(e.get("new", ""))
        if p not in plan:
            orig = p.read_bytes().decode("utf-8") if p.exists() else None
            plan[p] = [orig, orig]
        cur = plan[p][1]
        if old == "":
            if cur is not None:
                return {}, f"{why}：old 为空只能用来新建文件，这个文件已经有了"
            plan[p][1] = new
            continue
        if cur is None:
            return {}, f"{why} 不存在"
        n = cur.count(old)
        if n != 1:
            # 本机 autocrlf：文件可能是 CRLF，AI 读到的是 LF
            if "\r\n" in cur and cur.count(old.replace("\n", "\r\n")) == 1:
                old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
            else:
                return {}, f"{why}：要替换的原文出现了 {n} 次（要正好 1 次）"
        plan[p][1] = cur.replace(old, new, 1)
    if not plan:
        return {}, "没有改动"
    return {p: (o, n) for p, (o, n) in plan.items()}, ""


def _restore(plan: dict) -> None:
    for p, (orig, _new) in plan.items():
        if orig is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(orig.encode("utf-8"))


def run_checks() -> tuple[bool, str]:
    """三条自测 + pyflakes。自测以「断言失败 N 个」那一行为准（CLAUDE.md），pyflakes 要零输出。"""
    import re
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    log = []
    for args in CHECKS:
        name = " ".join(args)
        try:
            r = subprocess.run([sys.executable, *args], cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1200, env=env,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return False, "\n".join(log + [f"{name}：超时"])
        out = (r.stdout or "") + (r.stderr or "")
        if "pyflakes" in args:
            ok = r.returncode == 0 and not out.strip()
            log.append(f"{name}：{'零输出' if ok else '有输出'}")
            if not ok:
                return False, "\n".join(log + [out[-1500:]])
            continue
        m = re.findall(r"断言失败\s*(\d+)\s*个", out)
        ok = r.returncode == 0 and bool(m) and int(m[-1]) == 0
        log.append(f"{name}：{'断言失败 ' + m[-1] + ' 个' if m else '没看到结果行'}，退出码 {r.returncode}")
        if not ok:
            return False, "\n".join(log + [out[-2000:]])
    return True, "\n".join(log)


def apply_edits(pid: str, j: dict) -> int:
    import local_run as L
    busy = L.any_flow_running()
    if busy:
        set_status(pid, "pending", f"{L.FLOWS[busy][1]}正在跑，没落地；跑完再批准")
        print("有流程正在跑，改代码会让它前后用到两版代码，先不落地，提案退回待批")
        return 1
    plan, why = plan_edits(j.get("edits") or [])
    if not plan:
        set_status(pid, "failed", f"改动对不上：{why}")
        print(f"改动对不上，一个字都没写：{why}")
        return 1
    files = [p.relative_to(ROOT).as_posix() for p in plan]
    for p, (_orig, new) in plan.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(new.encode("utf-8"))
    print(f"改了 {len(files)} 个文件：{', '.join(files)}；跑三条自测 + pyflakes …", flush=True)
    ok, log = run_checks()
    print(log, flush=True)
    if not ok:
        _restore(plan)
        set_status(pid, "failed", "自测没过，已原样撤回：" + log.splitlines()[-1][:200] if log else "自测没过，已撤回")
        print("自测没过，文件已原样写回")
        return 1
    commit = ""
    try:
        with L.git_lock(timeout=120):
            subprocess.run(["git", "add", "--", *files], cwd=str(ROOT), check=True, capture_output=True)
            msg = f"AI 改动（用户批准）：{j.get('title', '')}\n\n提案 {pid}。{j.get('detail', '')}"[:1500]
            subprocess.run(["git", "commit", "-m", msg, "--", *files], cwd=str(ROOT), check=True, capture_output=True)
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True,
                                    text=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        print(f"自测过了，但本地提交没成（{e}），改动留在工作区")
    gui = any(f.startswith("src/gui/") for f in files)
    set_status(pid, "applied", f"自测全过，{'已本地提交 ' + commit if commit else '改动在工作区'}"
               + ("；控制台自己的代码改了，关掉控制台再开才生效" if gui else ""))
    j2 = _json(PROPOSALS / f"{pid}.json") or {}
    j2["files"], j2["commit"], j2["checks"] = files, commit, log
    (PROPOSALS / f"{pid}.json").write_text(json.dumps(j2, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落地完成 {commit}")
    return 0


def apply_proposal(pid: str) -> int:
    j = _json(PROPOSALS / f"{pid}.json")
    if not j:
        print(f"没有提案 {pid}")
        return 1
    if j.get("status") not in ("approved", "pending"):
        print(f"提案 {pid} 状态是 {j.get('status')}，不落地")
        return 1
    if j.get("type") == "param":
        k, v = str(j.get("param", "")), j.get("value")
        if k not in PARAM_WHITELIST or v is None:
            set_status(pid, "failed", f"参数 {k} 不在白名单或没有值")
            print(f"参数 {k} 不在白名单 {sorted(PARAM_WHITELIST)}")
            return 1
        lo, hi = PARAM_WHITELIST[k]
        if not (lo <= float(v) <= hi):
            set_status(pid, "failed", f"{k}={v} 超出 {lo}~{hi}")
            print(f"{k}={v} 超出允许范围 {lo}~{hi}")
            return 1
        ov = ROOT / "state" / "breakout" / "overrides.json"
        o = _json(ov) or {}
        old = o.get(k)
        o[k] = int(v) if k != "BOARD_ADJ" else v
        o.setdefault("history", []).append({"proposal": pid, "at": now_bj().isoformat(timespec="seconds"),
                                            "by": "user-approved-ai-proposal", "note": f"{k}: {old} -> {o[k]}"})
        ov.write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"overrides.json: {k} {old} -> {o[k]}；接着跑月度重估链（教训 36）")
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "refit_chain.py")], cwd=str(ROOT))
        set_status(pid, "applied" if r.returncode == 0 else "failed",
                   "重估链 " + ("完成" if r.returncode == 0 else f"退出码 {r.returncode}"))
        return r.returncode
    if j.get("type") == "code" and j.get("edits"):
        return apply_edits(pid, j)
    if j.get("type") == "code":
        # 代码提案只有描述没有补丁：落地 = 开一个待办给人（或下一次 Claude 会话）做，不自动改代码
        set_status(pid, "applied", "代码提案不自动改代码；已记入 docs/breakout_log.md 待办")
        log = ROOT / "docs" / "breakout_log.md"
        with log.open("a", encoding="utf-8") as f:
            f.write(f"\n- 待办（AI 提案 {pid}，用户已批准）：{j.get('title')} —— {j.get('detail')}\n")
        print("代码提案已记入待办，不自动改代码")
        return 0
    set_status(pid, "applied", "流程提案：记录")
    print("流程提案：已记录")
    return 0


# ---------------------------------------------------------------
#  入口
# ---------------------------------------------------------------
def _print_review(r: dict) -> None:
    print(f"[{r.get('severity')}] {r.get('summary')}")
    for f in r.get("findings") or []:
        print(f"  - [{f.get('severity')}] {f.get('where')}: {f.get('evidence')}  ->  {f.get('why')}"
              + (f"  建议：{f['suggestion']}" if f.get("suggestion") else ""))
    for p in r.get("proposals") or []:
        print(f"  * 提案[{p.get('type')}] {p.get('title')}：{p.get('detail')}  预期：{p.get('expected_effect')}  风险：{p.get('risk')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--apply", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--date", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry", action="store_true", help="只备材料不调 AI")
    a = ap.parse_args()
    import importlib
    importlib.import_module("localenv").load()
    if a.list:
        for p in list_proposals():
            print(f"{p['id']}  {p['status']:<9} [{p.get('type')}] {p.get('title')}")
        return 0
    if a.apply:
        return apply_proposal(a.apply)
    if not ai_usage.available():
        print("本机没有 claude 命令行")
        return 1
    if a.diff:
        m = gather_diff()
        if not m["diff"].strip() and not m["untracked"].strip():
            print("工作区没有改动")
            return 0
        if a.dry:
            print(f"diff {len(m['diff'])} 字符，新文件 {len(m['untracked'])} 字符")
            return 0
        r, outer, err = ai_usage.run_claude(prompt_diff(m), DIFF_SCHEMA, tools="", timeout=a.timeout,
                                            system=SYSTEM, kind="review-diff", subject="worktree")
        if err or not r:
            print(f"复核失败：{err}")
            return 1
        print(f"[{r['verdict']}] {r['summary']}")
        for f in r.get("findings") or []:
            print(f"  - [{f.get('severity')}] {f.get('where')}: {f.get('evidence')}  ->  {f.get('why')}")
        print(f"本次 {float((outer or {}).get('total_cost_usd') or 0):.2f} 美元")
        return 0
    weekly = bool(a.weekly)
    date = a.date or today_bj()
    m = gather_weekly() if weekly else gather_daily(date)
    if a.dry:
        print(json.dumps({k: (type(v).__name__ if not isinstance(v, (str, int, float, bool)) else v)
                          for k, v in m["breakout"].items()}, ensure_ascii=False))
        return 0
    r, outer, err = ai_usage.run_claude(prompt_daily(m, weekly), REVIEW_SCHEMA, tools="", timeout=a.timeout,
                                        system=SYSTEM, kind="review-weekly" if weekly else "review-daily",
                                        subject=date)
    usd = float((outer or {}).get("total_cost_usd") or 0)
    if err or not r:
        print(f"复盘失败：{err}（{usd:.2f} 美元）")
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "latest_alert.json").write_text(json.dumps(
            {"date": date, "severity": "注意", "summary": f"AI 复盘没跑成：{err[:80]}",
             "at": now_bj().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
        return 1
    p = _save_review("weekly" if weekly else "daily", date, r, usd)
    _print_review(r)
    print(f"本次 {usd:.2f} 美元 -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
