"""
会诊面板：out_learn/council.html。

给人看四件事：
    1. 主审结论：差距表（期望 vs 实际 vs 同期基准，带区间）、原因权重、波动还是真差距、叙述
    2. 图（内联 SVG）：起涨每份清单的命中 vs 基准；早盘逐日前 10 超额；模型重要性
    3. 过程：六个视角各一个折叠块，时间线里每一步（读了什么、查了什么、说了什么、用时）
    4. 提案台账：状态、实验结果、批准/驳回（只在控制台里生效，Pages 上只读）

和 learn/panel.py 一样：SVG 手画，不引外部库；所有输入缺失都能出页，只是块少。
按钮走控制台的 POST /api/council/decide，token 从父页面拿（同源 iframe）；
拿不到就提示「只在控制台可用」。
"""
from __future__ import annotations

import datetime as dt
import html as _h
import json
import logging
from pathlib import Path

from learn.council import run as R
from learn.council import schemas as S

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTL = ROOT / "out_learn"

_CSS = """
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;max-width:1000px;
     margin:0 auto;padding:14px;background:#111;color:#ddd}
h1{font-size:20px;margin:6px 0}h2{font-size:15px;margin:22px 0 8px;color:#aaa}
h3{font-size:14px;margin:12px 0 6px;color:#ccc}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #333;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#1a1a1a;color:#999}
.ok{color:#4c9}.no{color:#e66}.dim{color:#777;font-size:12px}.warn{color:#e9a23b}
.card{background:#181818;border:1px solid #2a2a2a;border-radius:8px;padding:10px 14px;margin:10px 0}
svg{width:100%;height:auto;background:#181818;border-radius:8px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;background:#233;color:#4c9;margin-left:6px}
.badge.no{background:#3a2020;color:#e66}.badge.old{background:#3a2a10;color:#e9a23b}
.badge.grey{background:#2a2a2a;color:#999}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}
.tile{background:#181818;border:1px solid #2a2a2a;border-left:4px solid #444;border-radius:8px;padding:8px 12px}
.tile.ok{border-left-color:#4c9}.tile.no{border-left-color:#e66}
.tile .t{font-size:13px;color:#eee}.tile .d{font-size:12px;color:#888}
.narr{font-size:14px;line-height:1.6;white-space:pre-wrap}
.why{display:flex;gap:2px;height:14px;border-radius:7px;overflow:hidden;margin:6px 0}
.why i{display:block;height:100%}
details{margin:6px 0}summary{cursor:pointer;color:#8ab4f8;font-size:13px}
.tl{font-size:12px;border-left:2px solid #2a2a2a;margin:6px 0 6px 6px;padding-left:10px}
.tl .ev{margin:4px 0}.tl .k{color:#777;display:inline-block;width:52px}
.tl .text{color:#ccc;white-space:pre-wrap}.tl .tool{color:#8ab4f8}.tl .res{color:#777}
.tl .think{color:#555}
button.act{background:#233;color:#4c9;border:1px solid #2a4;border-radius:5px;padding:2px 8px;cursor:pointer;font-size:12px}
button.act.no{background:#3a2020;color:#e66;border-color:#644}
.st-passed{color:#4c9}.st-failed{color:#e66}.st-needs_human{color:#e9a23b}.st-applied{color:#8ab4f8}
.st-pending,.st-testing{color:#999}.st-approved{color:#4c9}.st-rejected{color:#777}
"""

_JS = """
<script>
function decide(pid, action){
  var tok = "";
  try { tok = window.parent && window.parent.TOKEN ? window.parent.TOKEN : ""; } catch(e){}
  if(!tok){ alert("只在控制台里能批准/驳回。"); return; }
  if(!confirm((action==="approve"?"批准并落地":"驳回")+" "+pid+"？")) return;
  fetch("/api/council/decide", {method:"POST", headers:{"Content-Type":"application/json","X-Token":tok},
        body: JSON.stringify({id: pid, action: action})})
    .then(r=>r.json()).then(j=>{ alert(j.ok ? ("已"+(action==="approve"?"落地：":"驳回 ")+(j.what||"")) : ("失败："+(j.error||""))); location.reload(); })
    .catch(e=>alert("请求失败 "+e));
}
</script>
"""


def _e(s) -> str:
    return _h.escape("" if s is None else str(s))


def _pct(x, nd: int = 1) -> str:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return "-"
    if v != v:
        return "-"
    return f"{100 * v:.{nd}f}%"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


# ---------------------------------------------------------------------
def _head(summ: dict, day: Path) -> str:
    date = summ.get("date", "?")
    ok = summ.get("ok")
    badge = "<span class='badge'>成功</span>" if ok else \
        f"<span class='badge no'>没跑成：{_e(summ.get('error', '?'))}</span>"
    try:
        age = (dt.date.today() - dt.date.fromisoformat(date)).days
        if age >= 4:
            badge += f"<span class='badge old'>已 {age} 天未更新</span>"
    except Exception:  # noqa: BLE001
        pass
    tiles = []
    for ln in list(S.LENSES) + ["chair"]:
        m = (summ.get("lenses") or {}).get(ln) or {}
        if not m and not (day / f"{ln}.json").exists():
            continue
        good = bool(m.get("ok")) if m else (day / f"{ln}.json").exists()
        d = f"{m.get('seconds') or 0:.0f}s · {m.get('turns') or 0} 轮 · {m.get('n_tool_calls') or 0} 次查询"
        if m.get("cost_usd"):
            d += f" · ${m['cost_usd']:.2f}"
        if not good:
            d = _e(m.get("error") or "失败")
        tiles.append(f"<div class='tile {'ok' if good else 'no'}'><div class='t'>"
                     f"{_e(S.LENS_NAME.get(ln, ln))}</div><div class='d'>{d}</div></div>")
    return (f"<h1>学习会诊 · {_e(date)}{badge}</h1>"
            f"<div class='dim'>模型 {_e(summ.get('model'))} · 推理 {_e(summ.get('effort'))} · "
            f"共 {summ.get('seconds') or 0:.0f} 秒 · ${summ.get('cost_usd') or 0:.2f} · "
            f"<a href='learn.html'>← 学习面板</a></div>"
            f"<div class='grid'>{''.join(tiles)}</div>")


def _verdict_block(chair: dict) -> str:
    if not chair:
        return "<div class='card'><div class='dim'>主审没有结论。</div></div>"
    nr = chair.get("noise_or_real") or {}
    p = ["<h2>主审结论</h2><div class='card'>",
         f"<div class='narr'>{_e(chair.get('narrative'))}</div>",
         f"<h3>波动还是真差距：<span class='warn'>{_e(nr.get('verdict'))}</span>"
         f"（差距为真的概率 {_pct(nr.get('p_real'), 0)}）</h3>"
         f"<div class='dim'>{_e(nr.get('reasoning'))}</div>"]
    why = chair.get("why") or []
    if why:
        colors = ["#4c9", "#8ab4f8", "#e9a23b", "#e66", "#c9c", "#9cc", "#777"]
        p.append("<h3>原因权重</h3><div class='why'>")
        for i, w in enumerate(why):
            p.append(f"<i style='width:{100 * w['weight']:.0f}%;background:{colors[i % len(colors)]}' "
                     f"title='{_e(w['cause'])} {_pct(w['weight'], 0)}'></i>")
        p.append("</div><table><tr><th>原因</th><th>权重</th><th>证据</th></tr>")
        for i, w in enumerate(why):
            p.append(f"<tr><td><span style='color:{colors[i % len(colors)]}'>■</span> {_e(w['cause'])}</td>"
                     f"<td>{_pct(w['weight'], 0)}</td><td>{_e(w['evidence'])}</td></tr>")
        p.append("</table>")
    gaps = chair.get("gap") or []
    if gaps:
        p.append("<h3>认定的差距</h3><table><tr><th>线</th><th>切片</th><th>期望</th>"
                 "<th>实际</th><th>区间</th><th>同期基准</th><th>n</th></tr>")
        for g in gaps:
            ci = ""
            if g.get("ci_lo") == g.get("ci_lo") and g.get("ci_hi") == g.get("ci_hi"):
                ci = f"{g['ci_lo']:.2f} ~ {g['ci_hi']:.2f}"
            bl = f"{g['baseline']:.2f}" if g.get("baseline") == g.get("baseline") else "-"
            p.append(f"<tr><td>{_e(_line(g.get('line')))}</td><td>{_e(g.get('where'))}</td>"
                     f"<td>{g.get('expected'):.2f} {_e(g.get('unit'))}</td>"
                     f"<td>{g.get('actual'):.2f} {_e(g.get('unit'))}</td><td>{ci}</td>"
                     f"<td>{bl}</td><td>{g.get('n')}</td></tr>")
        p.append("</table>")
    if chair.get("dissent"):
        p.append(f"<h3>分歧</h3><div class='dim'>{_e(chair['dissent'])}</div>")
    p.append("</div>")
    return "".join(p)


def _line(x) -> str:
    return {"morning": "早盘选股", "breakout": "起涨预测", "both": "两条线"}.get(x or "", x or "")


# ---------------------------------------------------------------------
#  图
# ---------------------------------------------------------------------
def _chart_breakout(ev: dict, expected_pct: float) -> str:
    lists = (ev.get("breakout") or {}).get("lists") or []
    if not lists:
        return ""
    w, h, pad = 900, 210, 40
    n = len(lists)
    slot = (w - 2 * pad) / n
    bw = max(6.0, min(28.0, slot * 0.5))
    vals = []
    for L in lists:
        rate = L["hit_rate"] if L.get("hit_rate") is not None else (
            L["hits_sofar"] / L["n"] if L.get("n") else 0)
        vals.append(rate)
    mx = max(0.2, max(vals + [expected_pct / 100]) * 1.15)
    y0 = h - 30
    sc = (y0 - 24) / mx

    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
         f'<text x="{pad}" y="14" fill="#999" font-size="12">起涨预测 · 每份清单的命中率（20 根 K 线内涨超 50%）'
         f' vs 同期全市场基准（灰）· 虚线 = 邮件里印的期望 {expected_pct:.1f}%</text>']
    ye = y0 - expected_pct / 100 * sc
    p.append(f'<line x1="{pad}" y1="{ye:.1f}" x2="{w - pad}" y2="{ye:.1f}" stroke="#e9a23b" stroke-dasharray="4 3"/>')
    p.append(f'<line x1="{pad}" y1="{y0}" x2="{w - pad}" y2="{y0}" stroke="#333"/>')
    for i, L in enumerate(lists):
        x = pad + i * slot + (slot - bw) / 2
        rate = vals[i]
        final = bool(L.get("final"))
        cls = "#4c9" if final else "#2f6f5a"
        hh = rate * sc
        state = "（已定）" if final else f"（已走 {L.get('bars')} 根，进行中）"
        p.append(f'<rect x="{x:.1f}" y="{y0 - hh:.1f}" width="{bw:.1f}" height="{max(hh, 1):.1f}" fill="{cls}">'
                 f'<title>{_e(L["date"])} 命中 {L.get("hits_sofar")}/{L.get("n")}{state}</title></rect>')
        if final and L.get("ci_lo") is not None:
            lo, hi = y0 - L["ci_lo"] * sc, y0 - L["ci_hi"] * sc
            cx = x + bw / 2
            p.append(f'<line x1="{cx:.1f}" y1="{lo:.1f}" x2="{cx:.1f}" y2="{hi:.1f}" stroke="#9fd" stroke-width="1.5"/>')
        base = L.get("base_final") if final else L.get("base_sofar")
        if base is not None:
            bh = base * sc
            p.append(f'<rect x="{x + bw + 2:.1f}" y="{y0 - bh:.1f}" width="{bw * 0.5:.1f}" height="{max(bh, 1):.1f}" fill="#666">'
                     f'<title>同期基准 {_pct(base, 2)}</title></rect>')
        p.append(f'<text x="{x + bw / 2:.1f}" y="{h - 14}" fill="#777" font-size="10" text-anchor="middle">{_e(L["date"][5:])}</text>')
        p.append(f'<text x="{x + bw / 2:.1f}" y="{h - 3}" fill="#555" font-size="9" text-anchor="middle">{L.get("bars")}根</text>')
    p.append("</svg>")
    return "<h2>起涨预测 · 清单真值</h2>" + "".join(p) + \
        "<div class='dim'>深绿 = 20 根已走满（最终）；浅绿 = 进行中（到目前为止）；竖线 = Wilson 95% 区间；"\
        "灰柱 = 同一天全市场随便买的命中比例。命中率要和灰柱比，不是和虚线比。</div>"


def _chart_morning(ev: dict) -> str:
    m = ev.get("morning") or {}
    daily = m.get("daily") or []
    if not daily:
        return ""
    w, h, pad = 900, 200, 40
    n = len(daily)
    slot = (w - 2 * pad) / n
    bw = max(4.0, min(22.0, slot * 0.6))
    vals = [d.get("top_excess_pct") or 0 for d in daily]
    cum, s = [], 0.0
    for v in vals:
        s += v
        cum.append(s)
    mx = max(0.5, max(abs(v) for v in vals + cum))
    mid = h / 2 + 6
    sc = (mid - 30) / mx
    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
         f'<text x="{pad}" y="14" fill="#999" font-size="12">早盘选股 · 逐日前 10 超额（%/天，相对当日全池中位数）· 折线 = 累计</text>',
         f'<line x1="{pad}" y1="{mid}" x2="{w - pad}" y2="{mid}" stroke="#333"/>']
    pts = []
    for i, d in enumerate(daily):
        x = pad + i * slot + (slot - bw) / 2
        v = vals[i]
        hh = abs(v) * sc
        y = mid - hh if v >= 0 else mid
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(hh, 1):.1f}" fill="{"#4c9" if v >= 0 else "#e66"}">'
                 f'<title>{_e(d["date"])} 前10超额 {v:+.2f}% · 命中 {d.get("top_hits")}/{d.get("top_n")} · IC {d.get("ic")}'
                 f' · {_e(d.get("regime") or "")}</title></rect>')
        pts.append(f"{x + bw / 2:.1f},{mid - cum[i] * sc:.1f}")
        if n <= 40:
            p.append(f'<text x="{x + bw / 2:.1f}" y="{h - 4}" fill="#666" font-size="9" text-anchor="middle">{_e(d["date"][5:])}</text>')
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#8ab4f8" stroke-width="1.5"/>')
    p.append("</svg>")
    summ = m.get("summary") or {}
    tail = (f"<div class='dim'>{m.get('days')} 个在线真值日：前 10 日均超额 "
            f"{summ.get('top_excess_mean_pct')}%（标准误 {summ.get('top_excess_se_pct')}），"
            f"跑赢 {summ.get('top_win_days')}/{summ.get('n_days')} 天；过准入全部 "
            f"{summ.get('admitted_excess_mean_pct')}% vs 被剔除 {summ.get('rejected_excess_mean_pct')}%。</div>")
    dims = m.get("dims") or []
    if dims:
        tail += "<table><tr><th>维度</th><th>权重</th><th>高 1/3 超额</th><th>低 1/3 超额</th><th>差</th></tr>"
        for d in dims:
            sp = d.get("spread_pct")
            cls = "ok" if (sp or 0) > 0 else "no"
            tail += (f"<tr><td>{_e(d['dim'])}</td><td>{d.get('weight')}</td>"
                     f"<td>{d.get('high_third_excess_pct')}%</td><td>{d.get('low_third_excess_pct')}%</td>"
                     f"<td class='{cls}'>{sp:+.2f}</td></tr>" if sp is not None else "")
        tail += "</table><div class='dim'>差为负 = 这个维度打高分的票反而更差。</div>"
    return "<h2>早盘选股 · 逐日真值</h2>" + "".join(p) + tail


def _chart_importance(ev: dict) -> str:
    model = (ev.get("breakout") or {}).get("model") or {}
    imp = (model.get("importance") or [])[:18]
    if not imp:
        return ""
    w, rowh, pad = 900, 16, 40
    h = 30 + rowh * len(imp)
    mx = max(x["gain_share"] or 0 for x in imp) or 1
    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">',
         f'<text x="{pad}" y="14" fill="#999" font-size="12">起涨预测模型 · 特征重要性（增益占比，前 {len(imp)}）'
         f' · 训练于 {_e(model.get("fit_date"))} · {model.get("n_feats")} 个特征</text>']
    for i, x in enumerate(imp):
        y = 24 + i * rowh
        bwid = (w - 300) * (x["gain_share"] or 0) / mx
        p.append(f'<text x="{pad}" y="{y + 11}" fill="#aaa" font-size="11">{_e(x["feat"])}</text>')
        p.append(f'<rect x="240" y="{y + 2}" width="{bwid:.1f}" height="{rowh - 5}" fill="#8ab4f8"/>')
        p.append(f'<text x="{245 + bwid:.1f}" y="{y + 11}" fill="#777" font-size="10">{_pct(x["gain_share"], 1)}</text>')
    p.append("</svg>")
    grp = model.get("importance_by_group") or {}
    tail = "<div class='dim'>按组：" + "，".join(f"{k} {_pct(v, 1)}" for k, v in grp.items()) + "</div>" if grp else ""
    return "<h2>模型</h2>" + "".join(p) + tail


# ---------------------------------------------------------------------
#  过程
# ---------------------------------------------------------------------
def _process_block(day: Path) -> str:
    p = ["<h2>分析过程</h2><div class='dim'>每个视角一个进程，各自读证据包、查数据、写结论。"
         "展开看它是怎么想的。</div>"]
    any_ = False
    for ln in list(S.LENSES) + ["chair"]:
        res = _load(day / f"{ln}.json")
        trace = _jsonl(day / f"{ln}.trace.jsonl")
        meta = _load(day / f"{ln}.meta.json")
        if not res and not trace and not meta:
            continue
        any_ = True
        name = S.LENS_NAME.get(ln, ln)
        status = "<span class='badge'>完成</span>" if res else \
            f"<span class='badge no'>{_e(meta.get('error') or '失败')}</span>"
        p.append(f"<div class='card'><h3>{_e(name)} {status} "
                 f"<span class='dim'>{meta.get('seconds') or 0:.0f}s · {meta.get('turns') or 0} 轮 · "
                 f"{meta.get('n_tool_calls') or 0} 次工具</span></h3>")
        if res:
            p.append(f"<div class='narr' style='font-size:13px'>{_e(res.get('summary') or res.get('narrative'))}</div>")
            fs = res.get("findings") or []
            if fs:
                p.append("<table><tr><th>线</th><th>结论</th><th>证据</th><th>幅度</th><th>把握</th></tr>")
                for f in fs:
                    p.append(f"<tr><td>{_e(_line(f.get('line')))}</td><td>{_e(f.get('claim'))}</td>"
                             f"<td class='dim'>{_e(f.get('evidence'))}</td><td>{_e(f.get('magnitude'))}</td>"
                             f"<td>{_e(f.get('confidence'))}</td></tr>")
                p.append("</table>")
            qs = res.get("questions") or []
            if qs:
                p.append("<div class='dim'>没法从证据回答的：" + "；".join(_e(q) for q in qs) + "</div>")
        if trace:
            p.append(f"<details><summary>过程时间线（{len(trace)} 步）</summary><div class='tl'>")
            for ev in trace:
                t = f"{ev.get('t', 0):.0f}s"
                k = ev.get("kind")
                if k == "text":
                    p.append(f"<div class='ev'><span class='k'>{t}</span><span class='text'>{_e(ev.get('text'))}</span></div>")
                elif k == "tool_use":
                    p.append(f"<div class='ev'><span class='k'>{t}</span><span class='tool'>▶ {_e(ev.get('name'))} "
                             f"{_e(ev.get('input'))}</span></div>")
                elif k == "tool_result":
                    p.append(f"<div class='ev'><span class='k'>{t}</span><span class='res'>← {ev.get('chars', 0)} 字 "
                             f"{_e((ev.get('text') or '')[:160])}</span></div>")
                elif k == "thinking":
                    p.append(f"<div class='ev'><span class='k'>{t}</span><span class='think'>（思考）</span></div>")
                elif k == "result":
                    p.append(f"<div class='ev'><span class='k'>{t}</span><span class='res'>结束 · {ev.get('turns')} 轮 · "
                             f"${ev.get('cost_usd') or 0:.2f}</span></div>")
            p.append("</div></details>")
        p.append("</div>")
    if not any_:
        p.append("<div class='card'><div class='dim'>本次没有视角输出。</div></div>")
    return "".join(p)


# ---------------------------------------------------------------------
#  台账
# ---------------------------------------------------------------------
def _ledger_block(rows: list[dict]) -> str:
    p = ["<h2>提案台账</h2>"]
    if not rows:
        p.append("<div class='card'><div class='dim'>还没有提案。</div></div>")
        return "".join(p)
    p.append("<div class='dim'>过闸（passed）的等批准；批准即落地（早盘参数写 learned.yaml，起涨常量/特征进 overrides）。"
             "needs_human = 要写代码，进积压。</div>")
    p.append("<table><tr><th>日期</th><th>类型</th><th>线</th><th>改什么</th><th>依据</th><th>实验结果</th><th>状态</th><th></th></tr>")
    for r in rows[:60]:
        res = r.get("result") or {}
        det = _e(res.get("detail") or r.get("note") or "")
        st = r.get("status", "pending")
        btn = ""
        if st == "passed":
            btn = (f"<button class='act' onclick=\"decide('{_e(r['id'])}','approve')\">批准落地</button> "
                   f"<button class='act no' onclick=\"decide('{_e(r['id'])}','reject')\">驳回</button>")
        elif st in ("needs_human", "failed", "pending"):
            btn = f"<button class='act no' onclick=\"decide('{_e(r['id'])}','reject')\">驳回</button>"
        p.append(f"<tr><td>{_e(r.get('date'))}<div class='dim'>{_e(r['id'])}</div></td>"
                 f"<td>{_e(r.get('kind'))}<div class='dim'>{_e(r.get('priority'))}</div></td>"
                 f"<td>{_e(_line(r.get('line')))}</td>"
                 f"<td><b>{_e(r.get('target'))}</b><br>{_e(r.get('change'))}<div class='dim'>预期：{_e(r.get('expected_effect'))}</div></td>"
                 f"<td class='dim'>{_e(r.get('rationale'))}</td>"
                 f"<td class='dim'>{det}</td>"
                 f"<td class='st-{_e(st)}'>{_e(st)}</td><td>{btn}</td></tr>")
    p.append("</table>")
    return "".join(p)


def _history_block() -> str:
    dirs = sorted((R.STATE).glob("20*"), reverse=True)[:15]
    if len(dirs) <= 1:
        return ""
    p = ["<h2>历次会诊</h2><table><tr><th>日期</th><th>结果</th><th>判断</th><th>提案</th><th>耗时</th></tr>"]
    for d in dirs:
        s = _load(d / "summary.json")
        if not s:
            continue
        nr = s.get("verdict") or {}
        p.append(f"<tr><td>{_e(s.get('date'))}</td><td class='{'ok' if s.get('ok') else 'no'}'>"
                 f"{'成功' if s.get('ok') else _e(s.get('error'))}</td>"
                 f"<td>{_e(nr.get('verdict'))} {_pct(nr.get('p_real'), 0) if nr else ''}</td>"
                 f"<td>{s.get('n_proposals', 0)}</td><td>{s.get('seconds') or 0:.0f}s</td></tr>")
    p.append("</table>")
    return "".join(p)


# ---------------------------------------------------------------------
def build() -> Path:
    latest = R.latest()
    date = latest.get("date")
    day = R.STATE / date if date else None
    summ = _load(day / "summary.json") if day else {}
    summ = summ or latest
    chair = _load(day / "chair.json") if day else {}
    ev = _load(day / "evidence.json") if day else {}
    try:
        import export as E
        expected = float(E.STREAK_PERF[-1][1])
    except Exception:  # noqa: BLE001
        expected = 12.6

    p = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>学习会诊</title>", f"<style>{_CSS}</style></head><body>"]
    if not latest:
        p.append("<h1>学习会诊</h1><div class='dim'>还没跑过。学习流程末尾会自动跑，"
                 "或在控制台「运行」里手动点「学习会诊」。</div>")
    else:
        p.append(_head(summ, day))
        p.append(_verdict_block(chair))
        p.append(_chart_breakout(ev, expected))
        p.append(_chart_morning(ev))
        p.append(_chart_importance(ev))
        p.append(_process_block(day))
    p.append(_ledger_block(R.ledger_view()))
    p.append(_history_block())
    p.append(_JS)
    p.append("</body></html>")
    OUTL.mkdir(exist_ok=True)
    out = OUTL / "council.html"
    out.write_text("".join(p), encoding="utf-8")
    log.info("会诊面板已生成 %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(build())
