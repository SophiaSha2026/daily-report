"""
「学习」一张卡（2026-10-13 用户：简化以前「学习」那部分，让人看得懂，同时给操作选项）。

两条线里只有起涨预测会学：
    重学     模型每 30 天用最新数据重训一次（flow_breakout 打分前发现到期就自动做），
             新模型不如旧的就保留旧的（重训验收 daily.accept_model -> state/breakout/refit_status.json）
    重估     重学的同一天跑月度重估链（tools/refit_chain.py）：重算邮件里的准确率、板块系数、分数分档
    看效果   验证集成绩（out_breakout/window_grid.json）+ 实发清单之后的真实走势（state/breakout/truth.json）
    看漂移   打分分布和过去 20 天比（run_meta.drift_note）
    AI 提案  每日复盘 / AI 讨论给的改动建议，等用户批准（state/ai_review/proposals/）
长期调整突破是纯规则，不学习。

这里只读不写，把上面这些压成一句话的状态和「等你决定」的几件事；按钮走现有接口：
提案 -> /api/proposal，现在重学 -> /api/run learn_now。以前「模型自学」「月度重估」两个按钮
在详细控制台里收起来了（jobs.py hidden），入口统一到这张卡。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bjtime import now_bj  # noqa: E402

REFIT_DAYS = 30      # 和 daily.model_status 的 30 天同一个数（selftest_gui 钉住）


def _j(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _model() -> dict:
    m = _j(ROOT / "state" / "breakout" / "model.json", {}) or {}
    if not m.get("fit_date"):
        return {"exists": False}
    fit = dt.date.fromisoformat(m["fit_date"])
    age = (now_bj().date() - fit).days
    return {"exists": True, "fit_date": m["fit_date"], "age": age,
            "due_date": (fit + dt.timedelta(days=REFIT_DAYS)).isoformat(),
            "due": age >= REFIT_DAYS, "weighting": m.get("weighting", "")}


def _refit() -> dict:
    r = _j(ROOT / "state" / "breakout" / "refit_status.json", {}) or {}
    return {"accepted": r.get("accepted"), "at": str(r.get("at", ""))[:10],
            "reasons": r.get("reasons") or [], "fit_date": r.get("fit_date", "")}


def _validation() -> dict:
    g = _j(ROOT / "out_breakout" / "window_grid.json", {}) or {}
    rows = {}
    for r in g.get("grid", []):
        if r.get("kind") == "W5" and "连续≥" in str(r.get("label", "")):
            try:
                rows[int(str(r["label"]).split("连续≥")[1][0])] = r
            except Exception:  # noqa: BLE001
                pass
    if 1 not in rows:
        return {}
    return {"hit": round(100 * float(rows[1]["hit"]), 1), "n": rows[1].get("n"),
            "base": round(100 * float(g.get("base", 0)), 1), "days": g.get("days")}


def _live() -> dict:
    """实发清单之后的真实走势：满 20 天的定论 + 还在走的。"""
    t = _j(ROOT / "state" / "breakout" / "truth.json", {}) or {}
    ls = t.get("lists") or []
    fin = [x for x in ls if x.get("final")]
    run = [x for x in ls if not x.get("final")]
    out = {"n_lists": len(ls)}
    if fin:
        n = sum(int(x.get("n_final") or x.get("n") or 0) for x in fin)
        h = sum(int(x.get("hits_final") or 0) for x in fin)
        bs = [float(x["base_final"]) for x in fin if x.get("base_final") is not None]
        out["final"] = {"lists": len(fin), "n": n, "hits": h,
                        "hit": round(100 * h / n, 1) if n else None,
                        "base": round(100 * sum(bs) / len(bs), 1) if bs else None}
    if run:
        n = sum(int(x.get("n") or 0) for x in run)
        h = sum(int(x.get("hits_sofar") or 0) for x in run)
        out["running"] = {"lists": len(run), "n": n, "hits": h}
    # 模型影子清单（2026-10-19 起不进邮件）和同一天实发清单的配对，买得到口径（truth.paired）
    pr = (((t.get("shadow") or {}).get("model") or {}).get("paired") or {})
    if pr.get("days"):
        out["shadow"] = {"days": pr["days"], "prod": round(100 * pr["prod"], 1),
                         "arm": round(100 * pr["arm"], 1),
                         "t": round(pr["t"], 1) if pr.get("t") is not None else None}
    return out


def _pending() -> list[dict]:
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import ai_review as AR
        ps = AR.list_proposals("pending")
    except Exception:  # noqa: BLE001
        return []
    keep = ("id", "type", "title", "detail", "expected_effect", "risk", "param", "value", "source",
            "session", "subject", "note", "date")
    out = []
    for p in ps:
        x = {k: p.get(k) for k in keep if p.get(k) not in (None, "")}
        x["n_edits"] = len(p.get("edits") or [])
        x["files"] = sorted({str(e.get("file", "")) for e in p.get("edits") or []})
        out.append(x)
    return out


def summary() -> dict:
    model, refit, val, live = _model(), _refit(), _validation(), _live()
    meta = _j(ROOT / "out_breakout" / "run_meta.json", {}) or {}
    pend = _pending()
    pb = (_j(ROOT / "out_pullback" / "run_meta.json", {}) or {}).get("hist") or {}

    lines = []
    if model.get("exists"):
        lines.append(f"模型 {model['fit_date'][5:]} 学的，{model['age']} 天前；"
                     + (f"{model['due_date'][5:]} 到期自动重学" if not model["due"] else "已到期，下次起涨预测跑时自动重学"))
    else:
        lines.append("还没有模型，下次起涨预测跑时会先学一遍")
    if refit.get("accepted") is False:
        lines.append("上次重学没过验收（新模型不如旧的），在用旧模型")
    elif refit.get("accepted"):
        lines.append(f"上次重学 {refit['at'][5:]} 通过验收")
    if val:
        lines.append(f"验证集：清单 A 命中 {val['hit']}%，随便买 {val['base']}%")
    fin = live.get("final")
    if fin and fin.get("hit") is not None:
        lines.append(f"实发清单：{fin['lists']} 份已满 20 天，命中 {fin['hits']}/{fin['n']}（{fin['hit']}%）"
                     + (f"，同期全市场 {fin['base']}%" if fin.get("base") is not None else ""))
    run = live.get("running")
    if run:
        lines.append(f"还在走的 {run['lists']} 份清单：目前 {run['hits']}/{run['n']} 只已涨超 50%")
    sh = live.get("shadow")
    if sh:
        lines.append(f"模型影子对照：{sh['days']} 天已满 20 天，实发 {sh['prod']}% / 模型 {sh['arm']}%"
                     + (f"（差的 t {sh['t']}）" if sh.get("t") is not None else ""))
    if meta.get("drift_note"):
        lines.append(f"打分分布异常：{meta['drift_note']}")
    # 最近一次 AI 复盘判「严重 / 注意」（含「没跑成」，比如登录过期）就印一句，首页也看得见
    al = _j(ROOT / "state" / "ai_review" / "latest_alert.json", {}) or {}
    if al.get("severity") in ("严重", "注意") and al.get("summary"):
        lines.append(f"AI 复盘 {str(al.get('date', ''))[5:]} {al['severity']}：{str(al['summary'])[:120]}")

    decide = []
    for p in pend:
        what = {"param": "改参数", "code": "改代码", "process": "改流程"}.get(p.get("type"), "建议")
        src = "AI 讨论" if p.get("source") == "chat" else "AI 复盘"
        decide.append({"kind": "proposal", "id": p["id"], "type": p.get("type"),
                       "text": f"{src}建议{what}：{p.get('title', '')}",
                       "detail": p.get("detail", ""), "effect": p.get("expected_effect", ""),
                       "risk": p.get("risk", ""), "files": p.get("files", []), "note": p.get("note", ""),
                       "session": p.get("session", "")})
    if model.get("due"):
        decide.insert(0, {"kind": "relearn",
                          "text": f"模型学了 {model['age']} 天，到期了。下次起涨预测跑时会自动重学；要现在就重学吗？（约 20 分钟）"})

    if decide:
        head = f"有 {len(decide)} 件事等你决定"
    elif refit.get("accepted") is False or meta.get("drift_note"):
        head = "在学，有一处要留意"
    else:
        head = "学习正常"
    return {"head": head, "lines": lines, "decide": decide, "model": model, "refit": refit,
            "validation": val, "live": live, "drift": meta.get("drift_note", ""),
            "pullback": {"n": pb.get("n"), "per_month": pb.get("per_month"),
                         "line": "长期调整突破是固定规则，不学习"
                                 + (f"（三年成立 {pb['n']} 次，每月约 {pb.get('per_month')} 次）" if pb.get("n") else "")}}
