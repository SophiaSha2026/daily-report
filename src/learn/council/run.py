"""
会诊编排：证据包 -> 六个视角并行 -> 主审 -> 台账 -> 自动实验 -> 面板。

    python src/learn/council/run.py --date 2026-09-16            全流程
    python src/learn/council/run.py --date 2026-09-16 --lenses gap_where,improve
    python src/learn/council/run.py --date 2026-09-16 --skip-llm  只组证据包 + 面板（离线调试）
    python src/learn/council/run.py --experiments                 只跑台账里 pending 的自动实验

产物（都在 state/council/）：
    <date>/evidence.json, evidence_slim.json
    <date>/<lens>.json, <lens>.trace.jsonl, <lens>.meta.json
    <date>/chair.json                     主审裁决
    <date>/summary.json                   本次会诊的汇总（面板和总览卡读它）
    latest.json                           指向最近一次（ok / date / 摘要）
    proposals.jsonl                       提案台账（追加写；状态在 decisions.json 里改）
    decisions.json                        {proposal_id: {status, at, by, result}}

失败不阻断：任何一步炸了，latest.json 记 ok=false 和原因，学习流程照常。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from learn.council import schemas as S     # noqa: E402

log = logging.getLogger("council")
STATE = ROOT / "state" / "council"
LEDGER = STATE / "proposals.jsonl"
DECISIONS = STATE / "decisions.json"
LATEST = STATE / "latest.json"

DEFAULTS = {"enabled": True, "model": "claude-opus-5", "effort": "max",
            "timeout_seconds": 600, "chair_timeout_seconds": 420,
            "max_parallel": 6, "lenses": list(S.LENSES),
            "history_days": 0, "auto_experiments": True, "mail": True,
            "max_turns": 60}


def cfg(c: dict | None) -> dict:
    out = dict(DEFAULTS)
    try:
        out.update((c or {}).get("learning", {}).get("council") or {})
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------
#  台账
# ---------------------------------------------------------------------
def _pid(date: str, p: dict) -> str:
    key = f"{p['kind']}|{p['line']}|{p['target']}|{p['change']}"
    return date.replace("-", "") + "-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:6]


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return rows


def read_decisions() -> dict:
    try:
        return json.loads(DECISIONS.read_text(encoding="utf-8")) if DECISIONS.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def write_decision(pid: str, status: str, by: str = "system", result: dict | None = None,
                   note: str = "") -> dict:
    """改一条提案的状态。status 见 docs/council.md 第 5 节。"""
    d = read_decisions()
    cur = d.get(pid) or {}
    cur.update({"status": status, "at": dt.datetime.now().isoformat(timespec="seconds"),
                "by": by})
    if result is not None:
        cur["result"] = result
    if note:
        cur["note"] = note
    d[pid] = cur
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = DECISIONS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(DECISIONS)
    return cur


def ledger_view() -> list[dict]:
    """台账 + 当前状态，最新在前。"""
    d = read_decisions()
    rows = []
    for p in read_ledger():
        st = d.get(p["id"]) or {}
        rows.append({**p, "status": st.get("status", p.get("status", "pending")),
                     "decided_at": st.get("at"), "by": st.get("by"),
                     "result": st.get("result"), "note": st.get("note")})
    return rows[::-1]


def append_proposals(date: str, proposals: list[dict]) -> list[dict]:
    """主审的提案进台账。同一 id 已存在就不重复追加（重跑同一天）。"""
    have = {p["id"] for p in read_ledger()}
    STATE.mkdir(parents=True, exist_ok=True)
    added = []
    with LEDGER.open("a", encoding="utf-8") as f:
        for p in proposals:
            pid = _pid(date, p)
            if pid in have:
                continue
            auto = p["kind"] in ("param", "threshold", "feature_drop")
            row = {"id": pid, "date": date, **p,
                   "status": "pending" if auto else "needs_human",
                   "auto_testable": auto}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            added.append(row)
            if not auto:
                write_decision(pid, "needs_human", note="要写代码或改流程，等人做")
    return added


def attribution_proposals(date: str) -> list[dict]:
    """归因（通道 3）的 param_proposal / feature_proposal 也进台账。

    docs/learning.md 第 8 节写了这条通道，2026-09-16 审计发现从来没人消费
    state/llm_eval/<date>.json 里的这两个字段。现在和会诊提案走同一条路。
    """
    p = ROOT / "state" / "llm_eval" / f"{date}.json"
    try:
        j = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return []
    out = []
    pp = j.get("param_proposal") or {}
    if isinstance(pp, dict) and pp:
        vals = {k: v for k, v in pp.items() if k != "rationale"
                and isinstance(v, (int, float))}
        if vals:
            out.append(S.sanitize_proposal({
                "kind": "param", "line": "morning", "target": ",".join(sorted(vals)),
                "change": "; ".join(f"{k} -> {v}" for k, v in sorted(vals.items())),
                "rationale": str(pp.get("rationale", "归因提案"))[:300],
                "expected_effect": "归因 Agent 未量化", "test_plan": "七道闸",
                "priority": "P2", "params": vals}))
    for fp in (j.get("feature_proposal") or [])[:3]:
        if isinstance(fp, str) and fp.strip():
            out.append(S.sanitize_proposal({
                "kind": "feature_add", "line": "morning", "target": fp.strip()[:60],
                "change": fp.strip(), "rationale": "归因 Agent 提出", "expected_effect": "未量化",
                "test_plan": "人工实现后走七道闸", "priority": "P2"}))
        elif isinstance(fp, dict):
            out.append(S.sanitize_proposal({
                "kind": "feature_add", "line": "morning",
                "target": str(fp.get("name") or fp.get("feature") or "")[:60],
                "change": str(fp.get("description") or fp)[:200],
                "rationale": str(fp.get("rationale") or "归因 Agent 提出")[:300],
                "expected_effect": "未量化", "test_plan": "人工实现后走七道闸",
                "priority": "P2"}))
    return out


# ---------------------------------------------------------------------
#  编排
# ---------------------------------------------------------------------
def _write_latest(obj: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def run(c: dict, date: str, lenses: list[str] | None = None, skip_llm: bool = False,
        skip_experiments: bool = False) -> dict:
    """全流程。返回 summary（也写到 <date>/summary.json 和 latest.json）。"""
    k = cfg(c)
    t0 = time.time()
    day = STATE / date
    summary: dict = {"date": date, "ok": False, "started_at":
                     dt.datetime.now().isoformat(timespec="seconds"),
                     "model": k["model"], "effort": k["effort"], "lenses": {}}
    try:
        from learn.council import evidence as EV
        EV.build(c, date, int(k.get("history_days") or 0))
        summary["evidence"] = str(day / "evidence.json")
    except Exception as e:  # noqa: BLE001
        summary["error"] = f"证据包失败: {e}"
        log.warning(summary["error"])
        _write_latest(summary)
        return summary

    if skip_llm:
        summary["ok"] = True
        summary["note"] = "只组了证据包（--skip-llm）"
        _finish(summary, day, t0)
        return summary

    from learn.council import agents as AG
    lenses = [x for x in (lenses or k["lenses"]) if x in S.LENSES]
    res = AG.run_parallel(lenses, day, date, k["model"], k["effort"],
                          int(k["timeout_seconds"]), int(k["max_parallel"]),
                          int(k.get("max_turns", 150)))
    for ln in lenses:
        meta = _meta(day, ln)
        summary["lenses"][ln] = {"ok": res.get(ln) is not None, **meta}
    n_ok = sum(1 for ln in lenses if res.get(ln) is not None)
    log.info("视角完成 %d/%d", n_ok, len(lenses))
    if n_ok == 0:
        summary["error"] = "六个视角全部失败，不开主审"
        _finish(summary, day, t0)
        return summary

    chair = AG.run_lens("chair", AG.build_prompt("chair", day, date), day, k["model"],
                        k["effort"], int(k["chair_timeout_seconds"]), max_turns=30)
    summary["lenses"]["chair"] = {"ok": chair is not None, **_meta(day, "chair")}
    if chair is None:
        summary["error"] = "主审失败"
        _finish(summary, day, t0)
        return summary

    added = append_proposals(date, chair["proposals"] + attribution_proposals(date))
    summary.update({
        "ok": True,
        "verdict": chair["noise_or_real"],
        "why": chair["why"],
        "gap": chair["gap"],
        "narrative": chair["narrative"],
        "n_proposals": len(chair["proposals"]), "n_new_proposals": len(added),
        "proposal_ids": [p["id"] for p in added],
    })
    _finish(summary, day, t0)

    if not skip_experiments and k.get("auto_experiments", True):
        try:
            from learn.council import experiments as EX
            EX.run_pending(c)
        except Exception as e:  # noqa: BLE001
            log.warning("自动实验失败（不影响会诊）: %s", e)
    _panel()
    if k.get("mail", True):
        mail(date, chair, c)
    return summary


def mail_html(date: str, chair: dict) -> str:
    """会诊邮件：主审叙述 + 差距表 + 提案及实验状态。样式复用竞价面板的。"""
    import html as _h
    from ths_export import PANEL_CSS
    e = lambda x: _h.escape("" if x is None else str(x))  # noqa: E731
    nr = chair.get("noise_or_real") or {}
    rows = "".join(
        f"<tr><td>{e(g.get('where'))}</td><td>{g.get('expected'):.2f}</td>"
        f"<td>{g.get('actual'):.2f}</td><td>{e(g.get('unit'))}</td><td>{g.get('n')}</td></tr>"
        for g in chair.get("gap") or [])
    why = "，".join(f"{e(w['cause'])} {100 * w['weight']:.0f}%" for w in chair.get("why") or [])
    dec = read_decisions()
    props = []
    for pr in ledger_view():
        if pr.get("date") != date:
            continue
        st = pr.get("status")
        res = (pr.get("result") or {}).get("detail") or pr.get("note") or ""
        props.append(f"<tr><td>{e(pr.get('kind'))}</td><td>{e(pr.get('target'))}：{e(pr.get('change'))}"
                     f"<div class='sub'>{e(pr.get('rationale'))}</div></td>"
                     f"<td>{e(st)}</td><td class='sub'>{e(res)}</td></tr>")
    body = (f"<h1>学习会诊 · {e(date)}</h1>"
            f"<div class='sub'>波动还是真差距：<b>{e(nr.get('verdict'))}</b>"
            f"（差距为真 {100 * float(nr.get('p_real') or 0):.0f}%）· 原因：{why}</div>"
            f"<div class='tip' style='white-space:pre-wrap;margin:10px 0'>{e(chair.get('narrative'))}</div>"
            + (f"<h1 style='margin-top:16px'>认定的差距</h1><table><tr><th>切片</th><th>期望</th>"
               f"<th>实际</th><th>单位</th><th>n</th></tr>{rows}</table>" if rows else "")
            + (f"<h1 style='margin-top:16px'>提案与实验</h1><table><tr><th>类型</th><th>改什么</th>"
               f"<th>状态</th><th>实验结果</th></tr>{''.join(props)}</table>"
               f"<div class='tip'>passed = 过了实验闸门，去控制台「面板 -> 学习会诊」批准才生效；"
               f"needs_human = 要写代码。</div>" if props else "<div class='tip'>本次没有提案。</div>")
            + f"<div class='tip'>{len(dec)} 条历史提案在台账里。完整过程见控制台。</div>")
    return (f"<html><head><meta charset='utf-8'><style>{PANEL_CSS}</style></head>"
            f"<body style='background:#14161a'>{body}</body></html>")


def mail(date: str, chair: dict, c: dict) -> None:
    """发会诊邮件。失败只记日志。"""
    try:
        from learn import report as R_
        nr = chair.get("noise_or_real") or {}
        R_.send(date, mail_html(date, chair), c,
                subject=f"[会诊] {date}：{nr.get('verdict', '')}，{len(chair.get('proposals') or [])} 条提案")
    except Exception as e:  # noqa: BLE001
        log.warning("会诊邮件发送失败: %s", e)


def _meta(day: Path, lens: str) -> dict:
    p = day / f"{lens}.meta.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        m = {}
    return {kk: m.get(kk) for kk in ("seconds", "turns", "cost_usd", "n_tool_calls", "error")}


def _finish(summary: dict, day: Path, t0: float) -> None:
    summary["seconds"] = round(time.time() - t0, 1)
    summary["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    summary["cost_usd"] = round(sum((v.get("cost_usd") or 0)
                                    for v in summary.get("lenses", {}).values()), 3)
    day.mkdir(parents=True, exist_ok=True)
    (day / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    _write_latest(summary)


def _panel() -> None:
    try:
        from learn.council import panel as P
        P.build()
    except Exception as e:  # noqa: BLE001
        log.warning("会诊面板生成失败: %s", e)


def latest() -> dict:
    try:
        return json.loads(LATEST.read_text(encoding="utf-8")) if LATEST.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--lenses", default="")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--experiments", action="store_true", help="只跑台账里 pending 的实验")
    ap.add_argument("--panel", action="store_true", help="只重画面板")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    import cfg as C
    c = C.load()
    if a.panel:
        _panel()
        return 0
    if a.experiments:
        from learn.council import experiments as EX
        EX.run_pending(c)
        _panel()
        return 0
    date = a.date or dt.date.today().isoformat()
    lenses = [x.strip() for x in a.lenses.split(",") if x.strip()] or None
    s = run(c, date, lenses, a.skip_llm, a.skip_experiments)
    print(json.dumps({k: v for k, v in s.items() if k in
                      ("date", "ok", "error", "seconds", "cost_usd", "n_proposals", "verdict")},
                     ensure_ascii=False))
    return 0 if s.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
