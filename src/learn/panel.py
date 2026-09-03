"""
学习面板：learn.html。

替代 Electron 的决定（2026-09-03）：TUI 已覆盖全部操作，GUI 的真实增量
是可视化，而仓库已有现成的面板管线。这一页从 state/ 和 out/ 读数，
纯静态，进 GitHub Pages，手机也能看。

2026-09-04 重做版式，目标是**一眼看明白系统现在处于哪一步**：

    1. 阶段进度条    回填训练 -> 在线真值积累 -> 影子转正提案 -> 人工切换
    2. 今日循环      抓标签 -> 归因 -> 拟合 -> 七道闸 -> 结果 -> 下一次
    3. 关键数字      在线天数、双榜超额、影子占优、参数版本
    4. 影子参考榜    今天那 10 只 + 与正式榜逐日配对对比
    5. 裁决时间线    每次裁决一格，接受绿、拦下灰红；最近一次闸门明细
    6. 归因 / 擂台 / 影子模型系数

SVG 手画，不引外部图表库——Pages 上一个 <script src> 都是隐患。
所有输入缺失都能出页面，只是块少。
"""
from __future__ import annotations

import datetime as dt
import html as _h
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
STATE = ROOT / "state"
OUTL = ROOT / "out_learn"
OUT = ROOT / "out"

_CSS = """
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
     max-width:960px;margin:0 auto;padding:14px;background:#111;color:#ddd}
h1{font-size:20px;margin:6px 0}h2{font-size:15px;margin:22px 0 8px;color:#aaa}
a{color:#8ab4f8}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #333;padding:4px 8px;text-align:left}
th{background:#1a1a1a;color:#999}
td:first-child,th:first-child{white-space:nowrap}
.ok{color:#4c9}.no{color:#e66}.dim{color:#777;font-size:12px}.warn{color:#e9a23b}
.card{background:#181818;border:1px solid #2a2a2a;border-radius:8px;
      padding:10px 14px;margin:10px 0}
svg{width:100%;height:auto;background:#181818;border-radius:8px}
.pos{fill:#4c9}.neg{fill:#e66}.b.pos{fill:#8a8a8a}.b.neg{fill:#5c5c5c}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
       background:#233;color:#4c9;margin-left:6px}
.badge.old{background:#3a2a10;color:#e9a23b}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0}
.step{background:#181818;border:1px solid #2a2a2a;border-left:4px solid #444;
      border-radius:8px;padding:10px 12px;min-height:96px}
.step.done{border-left-color:#4c9}.step.cur{border-left-color:#8ab4f8}
.step.ready{border-left-color:#e9a23b}
.step .n{font-size:11px;color:#777}.step .t{font-size:14px;color:#eee;margin:2px 0}
.step .v{font-size:22px;font-weight:600;color:#fff;margin:2px 0}
.step .d{font-size:12px;color:#888}
.bar{height:8px;background:#2a2a2a;border-radius:4px;overflow:hidden;margin:6px 0}
.bar i{display:block;height:100%;background:#8ab4f8}
.loop{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:8px 0}
.tile{background:#181818;border:1px solid #2a2a2a;border-radius:8px;padding:8px 10px;
      font-size:12px;color:#aaa;min-height:74px}
.tile b{display:block;color:#eee;font-size:13px;margin-bottom:4px}
.tile.on{border-color:#2f5}.tile.off{opacity:.55}
.dots{display:flex;gap:4px;margin:4px 0}
.dot{width:12px;height:12px;border-radius:50%;background:#e66;display:inline-block}
.dot.ok{background:#4c9}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:8px 0}
.kpi{background:#181818;border:1px solid #2a2a2a;border-radius:8px;padding:8px 10px}
.kpi .k{font-size:11px;color:#888}.kpi .v{font-size:20px;color:#fff;font-weight:600}
.kpi .s{font-size:11px;color:#777}
.tl{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0}
.sq{width:16px;height:16px;border-radius:3px;background:#3a3a3a;display:inline-block}
.sq.acc{background:#4c9}.sq.rej{background:#5a3a3a}
.cf{display:grid;grid-template-columns:150px 1fr 60px;gap:8px;align-items:center;
    font-size:12px;margin:3px 0}
.cf .bar{margin:0;background:#222;position:relative;height:10px}
.cf .bar:before{content:'';position:absolute;left:50%;top:0;bottom:0;width:1px;background:#555}
.cf .bar i{position:absolute;top:0;height:100%}
.cf .bar i.pos{left:50%;background:#4c9}.cf .bar i.neg{right:50%;background:#e66}
.legend{font-size:12px;color:#888;line-height:1.7}
.code{font-family:ui-monospace,Menlo,monospace}
@media (max-width:720px){
  .steps{grid-template-columns:repeat(2,1fr)}
  .loop{grid-template-columns:repeat(2,1fr)}
  .kpis{grid-template-columns:repeat(3,1fr)}
  .cf{grid-template-columns:110px 1fr 50px}
}
"""

# 打分器参数与特征的中文名（面板只给人看，用人话）
_PNAME = {
    "scoring.weights.gap": "权重·竞价涨幅", "scoring.weights.volume": "权重·竞价量能",
    "scoring.weights.trend": "权重·竞价斜率", "scoring.weights.position": "权重·位置形态",
    "scoring.weights.sector": "权重·板块共振", "scoring.weights.continuity": "权重·连板延续",
    "screen.gap_pct_peak": "涨幅打分峰值", "screen.auc_ratio_score_hi": "量能饱和点",
    "screen.auc_ratio_decay": "量能超饱和衰减",
}
_FNAME = {
    "gap_pct": "竞价涨幅", "gap_norm": "涨幅/涨停幅", "auc_ratio": "竞价量能",
    "t1_chg": "T1 涨幅", "t2_chg": "T2 涨幅", "t3_chg": "T3 涨幅", "slope": "竞价斜率",
    "dive": "尾盘跳水", "pos_pct_60d": "60日位置", "board_height": "连板高度",
    "sector_members": "板块成分数", "sector_prev_limitups": "板块昨涨停数",
    "limit_pct": "涨停幅度", "monotonic": "稳步抬升", "ma_bull": "均线多头",
    "breakout": "突破", "prev_limit_up": "昨日涨停", "prev_broken_board": "昨日炸板",
    "log_auc_amount": "竞价额(对数)", "log_prev_amount": "昨日额(对数)",
    "sector_gap_med": "板块竞价中位", "sector_up_frac": "板块高开比例",
    "mkt_gap_med": "全池竞价中位", "mkt_up_frac": "全池高开比例",
    "gap_x_mkt": "涨幅×市场", "aucr_x_breadth": "量能×广度",
    "cauc_ratio_prev": "昨尾盘竞价占比",
}


# ---------------------------------------------------------------------
#  小工具
# ---------------------------------------------------------------------
def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception:  # noqa: BLE001
        pass
    return out


def _e(s) -> str:
    return _h.escape(str(s if s is not None else ""))


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None       # NaN -> None


def _pct(x, digits: int = 2, sign: bool = True) -> str:
    v = _num(x)
    if v is None:
        return "–"
    return f"{v * 100:{'+' if sign else ''}.{digits}f}%"


def _f(x, digits: int = 3, sign: bool = True) -> str:
    v = _num(x)
    if v is None:
        return "–"
    return f"{v:{'+' if sign else ''}.{digits}f}"


def _add_trading_days(d0: dt.date, n: int) -> dt.date:
    d = d0
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _today_bj() -> dt.date:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).date()


def _cfg_vals() -> dict:
    """从 config 取门槛；取不到用默认值（面板永远能出）。"""
    v = {"min_days": 60, "shadow_min_days": 30, "p_req": 0.90,
         "run_at": "16:30", "top_k": 10}
    try:
        import cfg as C
        lc = C.load()["learning"]
        v["min_days"] = int(lc["gate"]["min_days"])
        sc = lc.get("shadow") or {}
        v["shadow_min_days"] = int(sc.get("min_days", 30))
        v["p_req"] = float(sc.get("p_better", 0.90))
        v["run_at"] = str(lc.get("run_at", "16:30:00"))[:5]
        v["top_k"] = int(lc["objective"]["top_k"])
    except Exception:  # noqa: BLE001
        pass
    return v


# ---------------------------------------------------------------------
#  各区块
# ---------------------------------------------------------------------
def _head(st: dict) -> str:
    date = st.get("date", "?")
    gen = str(st.get("generated_at", ""))[:16].replace("T", " ")
    badge = ""
    try:
        age = (_today_bj() - dt.date.fromisoformat(date)).days
        if age >= 4:
            badge = f"<span class='badge old'>已 {age} 天未更新</span>"
        else:
            badge = "<span class='badge'>最新</span>"
    except Exception:  # noqa: BLE001
        pass
    return ("<h1>自学习系统 · 状态与进度</h1>"
            "<div class='dim'><a href='index.html'>← 竞价面板</a> · "
            "<a href='pullback.html'>形态面板</a></div>"
            f"<div class='dim'>数据截至 {_e(date)}{badge}"
            + (f" · 生成于 {_e(gen)}" if gen else "")
            + f" · 训练源 {_e(st.get('train_source', '?'))}</div>")


def _stepper(st: dict, cv: dict, proposal: dict) -> str:
    n_train = int(st.get("n_days", 0) or 0)
    online = int(st.get("online_days", len(st.get("daily") or [])) or 0)
    stat = st.get("shadow_stat") or {}
    need = int(stat.get("min_days", cv["shadow_min_days"]))
    p_req = float(stat.get("p_req", cv["p_req"]))
    p = _num(stat.get("p_better"))
    ready = bool(stat.get("ready"))

    # 1 回填训练
    s1_cls = "done" if n_train >= cv["min_days"] else "cur"
    s1 = (f"<div class='step {s1_cls}'><div class='n'>第 1 步</div>"
          f"<div class='t'>历史回填训练</div><div class='v'>{n_train} 天</div>"
          f"<div class='d'>{'已够拟合（门槛 ' + str(cv['min_days']) + ' 天）' if n_train >= cv['min_days'] else '不足 ' + str(cv['min_days']) + ' 天，只评估不拟合'}"
          f"</div></div>")

    # 2 在线真值积累
    frac = min(1.0, online / need) if need else 1.0
    if online >= need:
        s2_cls, s2_d = "done", "天数已达标"
    else:
        eta = ""
        try:
            eta = _add_trading_days(dt.date.fromisoformat(st.get("date")),
                                    need - online).strftime("%m-%d")
        except Exception:  # noqa: BLE001
            pass
        s2_cls = "cur"
        s2_d = (f"还差 {need - online} 个交易日"
                + (f" · 约 {eta}（不含节假日）" if eta else ""))
    s2 = (f"<div class='step {s2_cls}'><div class='n'>第 2 步</div>"
          f"<div class='t'>在线真值积累</div><div class='v'>{online} / {need} 天</div>"
          f"<div class='bar'><i style='width:{frac * 100:.0f}%'></i></div>"
          f"<div class='d'>{s2_d}</div></div>")

    # 3 影子转正提案
    if ready:
        s3_cls, s3_v = "ready", f"P = {p:.0%}"
        s3_d = "证据达标，提案已发邮件"
    elif p is None:
        s3_cls, s3_v, s3_d = "", "–", f"需 ≥ {p_req:.0%} 且天数达标"
    else:
        s3_cls, s3_v = ("cur" if online >= need else ""), f"P = {p:.0%}"
        s3_d = (f"需 ≥ {p_req:.0%}（P = 按天自助「影子前10超额更高」的概率）"
                if online >= need else
                f"需 ≥ {p_req:.0%}，现在只是参考，天数够了才算数")
    s3 = (f"<div class='step {s3_cls}'><div class='n'>第 3 步</div>"
          f"<div class='t'>影子转正提案</div><div class='v'>{s3_v}</div>"
          f"<div class='d'>{s3_d}</div></div>")

    # 4 人工切换
    if proposal:
        s4_cls, s4_v = "ready", "等你决定"
        s4_d = (f"提案首发 {_e(proposal.get('date'))}，最近 "
                f"{_e(proposal.get('last_sent'))}，共 {proposal.get('times', 1)} 次")
    else:
        s4_cls, s4_v, s4_d = "", "未到", "提案邮件到了再说；切换永远是人工动作"
    s4 = (f"<div class='step {s4_cls}'><div class='n'>第 4 步</div>"
          f"<div class='t'>人工切换</div><div class='v'>{s4_v}</div>"
          f"<div class='d'>{s4_d}</div></div>")
    return "<h2>现在到哪一步了</h2><div class='steps'>" + s1 + s2 + s3 + s4 + "</div>"


def _today_loop(st: dict, verdicts: list[dict], cv: dict) -> str:
    date = st.get("date", "?")
    v = st.get("verdict") or {}
    checks = v.get("checks") or []
    regime = st.get("regime_today")
    intents = st.get("intents") or []
    moved = v.get("moved") or {}
    n_verdict = len(verdicts)
    acc_total = int(st.get("accepted_total", 0) or 0)

    t1 = (f"<div class='tile on'><b>① 抓标签</b>{_e(date)} 收盘/开盘−1，"
          f"日内中性化</div>")
    if regime:
        t2 = f"<div class='tile on'><b>② LLM 归因</b>{_e(regime)}<br>进优化器当日权重</div>"
    else:
        t2 = "<div class='tile off'><b>② LLM 归因</b>本次未归因<br>日权重按 1.0</div>"
    if intents:
        names = "、".join(_PNAME.get(k, k) for k in intents)
        t3 = f"<div class='tile on'><b>③ 拟合</b>提案动 {len(intents)} 个：{_e(names)}</div>"
    elif checks:
        t3 = "<div class='tile on'><b>③ 拟合</b>连续解无明显方向，无提案</div>"
    else:
        t3 = "<div class='tile off'><b>③ 拟合</b>天数不足，只评估</div>"
    if checks:
        dots = "".join(
            f"<span class='dot {'ok' if c.get('passed') else ''}' "
            f"title='{_e(c.get('name'))}: {_e(c.get('detail'))}'></span>"
            for c in checks)
        npass = sum(1 for c in checks if c.get("passed"))
        failed = "、".join(c.get("name", "") for c in checks if not c.get("passed"))
        t4 = (f"<div class='tile on'><b>④ 七道闸</b><span class='dots'>{dots}</span>"
              f"{npass}/{len(checks)} 过"
              + (f"，没过：{_e(failed)}" if failed else "，全过") + "</div>")
    else:
        t4 = "<div class='tile off'><b>④ 七道闸</b>本次未裁决</div>"
    if v.get("accepted"):
        chg = "、".join(f"{_PNAME.get(k, k)} {a:.3g}→{b:.3g}" for k, (a, b) in moved.items())
        t5 = (f"<div class='tile on' style='border-color:#e9a23b'><b>⑤ 结果</b>"
              f"<span class='warn'>参数已变更</span>：{_e(chg)}</div>")
    else:
        t5 = (f"<div class='tile on'><b>⑤ 结果</b>参数保持不变<br>"
              f"版本 {_e(st.get('theta_version', '基线'))} · "
              + (f"第 {n_verdict} 次裁决 · " if n_verdict else "")
              + f"累计接受 {acc_total} 次</div>")
    t6 = (f"<div class='tile'><b>⑥ 下一次</b>下个交易日 {_e(cv['run_at'])}（北京）"
          f"<br>GitHub 自动跑；本机晚上跑全流程也会顺带跑</div>")
    return ("<h2>今天这一圈做了什么</h2><div class='loop'>"
            + t1 + t2 + t3 + t4 + t5 + t6 + "</div>"
            "<div class='dim'>拦下是常态，不是故障：闸门的职责就是把噪声挡在参数外面。</div>")


def _kpis(st: dict) -> str:
    sh = st.get("shadow") or []
    n = len(sh)
    w = sum(1 for x in sh if (_num(x.get("shadow_top_excess")) or 0)
            > (_num(x.get("base_top_excess")) or 0))
    b = [(_num(x.get("base_top_excess"))) for x in sh]
    s = [(_num(x.get("shadow_top_excess"))) for x in sh]
    b = [x for x in b if x is not None]
    s = [x for x in s if x is not None]
    m = st.get("metrics") or {}
    stat = st.get("shadow_stat") or {}
    cards = [
        ("在线真值天数", str(n), "系统上线后的真实交易日"),
        ("正式榜 前10超额/日", _pct(sum(b) / len(b)) if b else "–", "在线均值"),
        ("影子榜 前10超额/日", _pct(sum(s) / len(s)) if s else "–", "在线均值"),
        ("影子占优天数", f"{w} / {n}" if n else "–",
         f"P(影子更好)={_num(stat.get('p_better')):.0%}" if _num(stat.get("p_better")) is not None else "按天自助"),
        ("回填集 IC", _f(m.get("ic_mean"), 3, False), f"{st.get('n_days', 0)} 天训练集"),
        ("参数版本", _e(st.get("theta_version", "基线")),
         f"累计接受变更 {st.get('accepted_total', 0)} 次"),
    ]
    return ("<h2>关键数字</h2><div class='kpis'>"
            + "".join(f"<div class='kpi'><div class='k'>{k}</div><div class='v'>{v}</div>"
                      f"<div class='s'>{d}</div></div>" for k, v, d in cards)
            + "</div>")


def _paired_bars(rows: list[dict], ka: str, kb: str, title: str,
                 fmt, la: str = "正式榜", lb: str = "影子榜",
                 height: int = 170) -> str:
    """每天两根柱：正式榜（灰）和影子榜（绿/红按正负）。零轴居中。"""
    if not rows:
        return ""
    w, pad = 880, 34
    n = len(rows)
    slot = (w - 2 * pad) / n
    bw = max(3.0, min(18.0, slot / 2 - 3))
    vals = [abs(_num(r.get(k)) or 0) for r in rows for k in (ka, kb)]
    mx = max(0.001, max(vals))
    mid = height / 2 + 10
    p = [f'<svg viewBox="0 0 {w} {height + 34}" xmlns="http://www.w3.org/2000/svg">',
         f'<text x="{pad}" y="14" fill="#999" font-size="12">{_e(title)}</text>',
         f'<rect x="{w - 200}" y="5" width="10" height="10" fill="#8a8a8a"/>'
         f'<text x="{w - 186}" y="14" fill="#999" font-size="11">{_e(la)}</text>'
         f'<rect x="{w - 120}" y="5" width="10" height="10" fill="#4c9"/>'
         f'<text x="{w - 106}" y="14" fill="#999" font-size="11">{_e(lb)}</text>',
         f'<line x1="{pad}" y1="{mid}" x2="{w - pad}" y2="{mid}" stroke="#333"/>']
    for i, r in enumerate(rows):
        x0 = pad + i * slot + (slot - 2 * bw - 3) / 2
        for j, (k, cls) in enumerate(((ka, "b"), (kb, "s"))):
            v = _num(r.get(k)) or 0.0
            h = abs(v) / mx * (mid - 28)
            y = mid - h if v >= 0 else mid
            sign = "pos" if v >= 0 else "neg"
            p.append(f'<rect class="{cls} {sign}" x="{x0 + j * (bw + 3):.1f}" '
                     f'y="{y:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}">'
                     f'<title>{_e(r.get("date"))} {la if j == 0 else lb} {fmt(v)}'
                     f'</title></rect>')
        if n <= 24:
            p.append(f'<text x="{x0 + bw + 1.5:.1f}" y="{height + 28}" fill="#666" '
                     f'font-size="9" text-anchor="middle">{_e(str(r.get("date", ""))[5:])}'
                     f'</text>')
    p.append("</svg>")
    return "".join(p)


def _shadow_block(st: dict, today_shadow: dict, cv: dict) -> str:
    parts = ["<h2>影子参考榜：试运行中的候任排序器</h2>"]
    rows = today_shadow.get("rows") or []
    if rows:
        base_top = set(today_shadow.get("base_top") or [])
        trs = "".join(
            f"<tr><td>{i}</td><td class='code'>{_e(r.get('code'))}</td>"
            f"<td>{_e(r.get('name'))}</td><td>{_f(r.get('gap_pct'), 2)}%</td>"
            f"<td>{_f(r.get('liangbi'), 1, False)}</td><td>{_f(r.get('sscore'), 2)}</td>"
            f"<td>{'✓ 正式榜也有' if r.get('code') in base_top else ''}</td></tr>"
            for i, r in enumerate(rows, 1))
        parts.append(
            f"<div class='card'><div class='dim'>{_e(today_shadow.get('date'))} 影子前 "
            f"{len(rows)} · 与正式榜重合 {today_shadow.get('overlap', 0)}/{cv['top_k']}"
            f"（正式榜以邮件和竞价面板为准，此榜只做参考）</div>"
            "<table><tr><th>#</th><th>代码</th><th>名称</th><th>高开</th><th>量比</th>"
            f"<th>影子分</th><th></th></tr>{trs}</table></div>")
    else:
        parts.append("<div class='dim'>今天还没有影子参考榜（09:27 竞价线跑完才有）。</div>")
    sh = st.get("shadow") or []
    if sh:
        parts.append(_paired_bars(sh, "base_top_excess", "shadow_top_excess",
                                  "逐日 前 10 超额收益（正式榜 vs 影子榜）",
                                  lambda v: f"{v * 100:+.2f}%"))
        parts.append(_paired_bars(sh, "base_ic", "shadow_ic",
                                  "逐日 IC（分数与当日实际超额的秩相关）",
                                  lambda v: f"{v:+.3f}", height=150))
        ov = [(_num(x.get("overlap")) or 0) for x in sh]
        parts.append(f"<div class='dim'>两榜日均重合 {sum(ov) / len(ov):.0%}。"
                     "影子 = 擂台胜者 RankHuber（27 特征线性模型）；它每天和正式榜"
                     "并排记账，不发信不排产。</div>")
    return "".join(parts)


def _verdict_history(verdicts: list[dict], st: dict) -> str:
    parts = ["<h2>裁决时间线</h2>"]
    if verdicts:
        sq = "".join(
            f"<span class='sq {'acc' if r.get('accepted') else 'rej'}' "
            f"title='{_e(r.get('date'))}：{'接受' if r.get('accepted') else '拦下'}"
            + (f"，没过 {_e('、'.join(r.get('failed') or []))}" if r.get("failed") else "")
            + "'></span>" for r in verdicts[-120:])
        n_acc = sum(1 for r in verdicts if r.get("accepted"))
        parts.append(f"<div class='card'><div class='tl'>{sq}</div>"
                     f"<div class='dim'>共 {len(verdicts)} 次裁决，接受 {n_acc} 次。"
                     "每格一天，绿 = 接受了参数变更，暗红 = 被闸门拦下（悬停看原因）。"
                     "</div></div>")
    else:
        parts.append("<div class='dim'>还没有裁决记录：从下一次 16:30 的学习开始记。</div>")
    v = st.get("verdict") or {}
    if v.get("checks"):
        parts.append(f"<h2>最近一次闸门明细 · {_e(st.get('date'))}</h2>"
                     "<div class='card'><table><tr><th>闸门</th><th>结论</th><th>依据</th></tr>")
        for ckd in v["checks"]:
            mark = ("<span class='ok'>过</span>" if ckd.get("passed")
                    else "<span class='no'>不过</span>")
            parts.append(f"<tr><td>{_e(ckd.get('name'))}</td><td>{mark}</td>"
                         f"<td>{_e(ckd.get('detail'))}</td></tr>")
        parts.append("</table>"
                     + ("<div class='ok'>本次已接受参数变更</div>" if v.get("accepted")
                        else "<div class='dim'>未接受变更，参数保持不变。</div>")
                     + "</div>")
    return "".join(parts)


def _regimes(st: dict) -> str:
    counts: dict[str, int] = {}
    days: dict[str, int] = {}
    for f in sorted((STATE / "llm_eval").glob("*.json")):
        j = _load(f)
        r = j.get("day_regime")
        if r:
            days[r] = days.get(r, 0) + 1
        for it in j.get("items", []):
            counts[it.get("cause", "?")] = counts.get(it.get("cause", "?"), 0) + 1
    if not counts and not days:
        return ""
    out = ["<h2>LLM 归因分布</h2><div class='card'>"]
    if days:
        out.append("<div class='dim'>日级状态（进优化器当日权重）：" + "，".join(
            f"{_e(k)} {n} 天" for k, n in sorted(days.items(), key=lambda x: -x[1]))
            + "</div>")
    if counts:
        out.append("<table><tr><th>个股偏差原因</th><th>次数</th></tr>"
                   + "".join(f"<tr><td>{_e(k)}</td><td>{n}</td></tr>"
                             for k, n in sorted(counts.items(), key=lambda x: -x[1]))
                   + "</table>")
    out.append("</div>")
    return "".join(out)


def _race_block(race: dict) -> str:
    if not race.get("table"):
        return ""
    rows = "".join(
        f"<tr><td>{_e(r['model'])}</td><td>{r['IC']:.4f}</td><td>{r['ICIR']:.3f}</td>"
        f"<td>{r['ICIR_lo']:.2f}~{r['ICIR_hi']:.2f}</td><td>{r['top_excess']:+.3%}</td>"
        f"<td>{r['hit']:.0%}</td></tr>" for r in race["table"])
    return ("<h2>模型擂台（走向前样本外）</h2><div class='card'><table>"
            "<tr><th>模型</th><th>IC</th><th>ICIR</th><th>90%区间</th>"
            f"<th>前10超额</th><th>胜率</th></tr>{rows}</table>"
            f"<div class='dim'>选中：{_e(race.get('winner', '?'))} —— {_e(race.get('why', ''))}。"
            "胜者当影子排序器试运行，不直接进生产。</div></div>")


def _coef_block(model: dict) -> str:
    coef = model.get("coef") or {}
    if not coef:
        return ""
    top = sorted(coef.items(), key=lambda x: -abs(x[1]))[:10]
    mx = max(abs(v) for _, v in top) or 1.0
    rows = "".join(
        f"<div class='cf'><span>{_e(_FNAME.get(k, k))}</span>"
        f"<span class='bar'><i class='{'pos' if v >= 0 else 'neg'}' "
        f"style='width:{abs(v) / mx * 50:.1f}%'></i></span>"
        f"<span class='code'>{v:+.2f}</span></div>" for k, v in top)
    return (f"<h2>影子模型现在最看重什么（{_e(model.get('kind', ''))}，"
            f"{model.get('train_days', '?')} 天拟合）</h2><div class='card'>{rows}"
            "<div class='dim'>秩归一特征上的线性系数，正 = 越大越好，负 = 越大越差。"
            "系数文件 state/shadow_model.json，给定它任何人可逐位复现当天排名。</div></div>")


def _legend() -> str:
    return ("<h2>怎么读这页</h2><div class='card legend'>"
            "<b>阶段</b>：回填训练已完成 → 现在在攒在线真值天 → 攒够且影子显著更好才会有提案 → 切换永远由人决定。<br>"
            "<b>前 10 超额</b>：当天榜单前 10 的开盘买、收盘卖收益，减去全池中位数。<br>"
            "<b>IC</b>：打分与当日实际超额的秩相关，单日噪声很大，看趋势别看单点。<br>"
            "<b>七道闸</b>：样本量 / 样本外改善 / 按天自助 / 步长 / 行为回放 / 冷却 / 在线否决，全过才改参数。<br>"
            "排序 100% 确定性；LLM 只做归因与提案，提案过同一道闸。完整设计 docs/learning.md"
            "</div>")


# ---------------------------------------------------------------------
def build(daily_metrics: list[dict] | None = None) -> Path:
    """生成 out_learn/learn.html。所有输入缺失都能出页面，只是块少。"""
    st = _load(STATE / "learning_status.json")
    if daily_metrics:
        st["daily"] = daily_metrics
    cv = _cfg_vals()
    race = _load(OUTL / "model_race.json")
    model = _load(STATE / "shadow_model.json")
    verdicts = _jsonl(STATE / "verdict_log.jsonl")
    proposal = _load(STATE / "shadow_proposal.json")
    today_shadow = _load(OUT / "shadow.json")

    p = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>学习面板</title>",
         f"<style>{_CSS}</style></head><body>"]
    if not st:
        p.append("<h1>自学习系统 · 状态与进度</h1>"
                 "<div class='dim'>学习系统还没跑过。第一次 16:30 的学习之后这里才有内容。</div>")
    else:
        p.append(_head(st))
        p.append(_stepper(st, cv, proposal))
        p.append(_today_loop(st, verdicts, cv))
        p.append(_kpis(st))
        p.append(_shadow_block(st, today_shadow, cv))
        p.append(_verdict_history(verdicts, st))
        p.append(_regimes(st))
        p.append(_race_block(race))
        p.append(_coef_block(model))
    p.append(_legend())
    p.append("</body></html>")

    OUTL.mkdir(exist_ok=True)
    out = OUTL / "learn.html"
    out.write_text("".join(p), encoding="utf-8")
    log.info("学习面板已生成 %s", out)
    return out
