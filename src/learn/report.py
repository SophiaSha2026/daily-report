"""
学习系统的对外产出：变更邮件 + 状态文件。

发信规则（docs/learning.md 第 10 节）：

    接受了变更   -> 当天发一封，写清楚动了哪几个旋钮、证据、行为影响、怎么回滚
    没接受       -> **不发邮件**。只写 state/learning_status.json，
                    并在竞价面板底部留一行

不发「今天也没变化」这种邮件，是刻意的。按设计这套系统前 3~5 个月
本来就不会动参数（见 docs/learning.md 5.1），天天发等于训练用户忽略它。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
STATUS = ROOT / "state" / "learning_status.json"

_CSS = """
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
     max-width:820px;margin:0 auto;padding:16px;color:#1c1c1e;font-size:14px}
h1{font-size:19px;margin:0 0 4px}h2{font-size:15px;margin:20px 0 6px}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}
th,td{border:1px solid #e3e3e6;padding:5px 8px;text-align:left}
th{background:#f6f6f8;font-weight:600}
.meta{color:#6b6b70;font-size:12px}
.up{color:#c0392b}.dn{color:#1e8449}
.ok{color:#1e8449;font-weight:600}.no{color:#c0392b;font-weight:600}
.box{background:#f8f9fa;border-left:3px solid #d0d0d5;padding:8px 12px;
     margin:10px 0;font-size:13px}
code{background:#f2f2f5;padding:1px 5px;border-radius:3px;font-size:12px}
"""

_NAME = {
    "scoring.weights.gap": "权重 · 竞价涨幅",
    "scoring.weights.volume": "权重 · 竞价量能",
    "scoring.weights.trend": "权重 · 竞价斜率",
    "scoring.weights.position": "权重 · 位置形态",
    "scoring.weights.sector": "权重 · 板块共振",
    "scoring.weights.continuity": "权重 · 连板延续",
    "screen.gap_pct_peak": "涨幅打分峰值 (%)",
    "screen.auc_ratio_score_hi": "量能打分饱和点",
    "screen.auc_ratio_decay": "量能超饱和衰减率",
}


def save_status(payload: dict) -> Path:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                 default=str), encoding="utf-8")
    return STATUS


def status_line() -> str:
    """竞价面板底部那一行。读不到就返回空串，绝不抛异常。"""
    try:
        s = json.loads(STATUS.read_text(encoding="utf-8"))
        m = s.get("metrics", {})
        return (f"学习状态：已积累 {s.get('n_days', 0)} 天 · "
                f"IC {m.get('ic_mean', float('nan')):.3f} · "
                f"前10超额 {m.get('top_excess', float('nan'))*100:+.2f}% · "
                f"参数版本 {s.get('theta_version', '基线')}")
    except Exception:  # noqa: BLE001
        return ""


def _param_table(moved: dict) -> str:
    rows = []
    for k, (old, new) in sorted(moved.items()):
        pct = (new - old) / old * 100 if old else float("nan")
        cls = "up" if new > old else "dn"
        rows.append(
            f"<tr><td>{_NAME.get(k, k)}</td><td><code>{k}</code></td>"
            f"<td>{old:.4g}</td><td class='{cls}'>{new:.4g}</td>"
            f"<td class='{cls}'>{pct:+.1f}%</td></tr>")
    return ("<table><tr><th>含义</th><th>参数</th><th>原值</th><th>新值</th>"
            "<th>变动</th></tr>" + "".join(rows) + "</table>")


def _gate_table(checks: list) -> str:
    rows = []
    for c in checks:
        c = c if isinstance(c, dict) else c.__dict__
        mark = "<span class='ok'>过</span>" if c["passed"] \
            else "<span class='no'>不过</span>"
        rows.append(f"<tr><td>{c['name']}</td><td>{mark}</td>"
                    f"<td>{c['detail']}</td></tr>")
    return ("<table><tr><th>闸门</th><th>结论</th><th>依据</th></tr>"
            + "".join(rows) + "</table>")


def _churn_table(old_top: dict, new_top: dict, n: int = 10) -> str:
    rows = []
    for day in sorted(old_top)[-n:]:
        a, b = set(old_top[day]), set(new_top.get(day, []))
        if a == b:
            continue
        rows.append(f"<tr><td>{day}</td>"
                    f"<td class='up'>{'、'.join(sorted(b - a)) or '—'}</td>"
                    f"<td class='dn'>{'、'.join(sorted(a - b)) or '—'}</td></tr>")
    if not rows:
        return "<div class='meta'>最近这些天的前 10 完全不变。</div>"
    return ("<table><tr><th>日期</th><th>会新进榜</th><th>会掉出榜</th></tr>"
            + "".join(rows) + "</table>")


def build_html(date: str, verdict, metrics_old: dict, metrics_new: dict,
               old_top: dict, new_top: dict, llm_note: str = "",
               regime_counts: dict | None = None) -> str:
    v = verdict.to_dict() if hasattr(verdict, "to_dict") else verdict
    ev = v["evidence"]
    p = [f"<style>{_CSS}</style>",
         f"<h1>筛选参数已更新 · {date}</h1>",
         f"<div class='meta'>基于 {ev['n_days']} 个交易日的开盘买入-收盘卖出"
         f"实际收益。完整设计见 docs/learning.md</div>",
         "<h2>动了什么</h2>", _param_table(v["moved"]),
         "<h2>证据</h2>",
         f"<div class='box'>样本外目标 Ĝ：<b>{ev['oos_old']:+.4f} → "
         f"{ev['oos_new']:+.4f}</b>（{ev['oos_new']-ev['oos_old']:+.4f}）<br>"
         f"按天自助 2000 次，P(新参数更好) = <b>{ev['bootstrap_p']:.1%}</b><br>"
         f"行为回放最大单日换手 {ev['worst_churn']:.0%}</div>",
         _gate_table(v["checks"]),
         "<h2>指标对比（全样本）</h2>",
         "<table><tr><th>指标</th><th>原参数</th><th>新参数</th></tr>"
         + "".join(
             f"<tr><td>{lab}</td><td>{metrics_old.get(k, float('nan')):{f}}</td>"
             f"<td>{metrics_new.get(k, float('nan')):{f}}</td></tr>"
             for lab, k, f in (
                 ("日 IC 均值", "ic_mean", ".4f"),
                 ("ICIR", "icir", ".3f"),
                 ("前 10 超额收益", "top_excess", "+.4%"),
                 ("前 10 胜率", "hit_rate", ".1%"),
                 ("日均通过数", "avg_pool", ".0f")))
         + "</table>",
         "<h2>行为影响（最近 10 天回放）</h2>", _churn_table(old_top, new_top)]
    if regime_counts:
        p.append("<h2>这段时间的归因分布</h2><table><tr><th>状态</th>"
                 "<th>天数</th></tr>"
                 + "".join(f"<tr><td>{k}</td><td>{n}</td></tr>"
                           for k, n in sorted(regime_counts.items(),
                                              key=lambda x: -x[1]))
                 + "</table>")
    if llm_note:
        p.append(f"<h2>分析</h2><div class='box'>{llm_note}</div>")
    p.append("<h2>不认可就回滚</h2><div class='box'>删掉 "
             "<code>state/learned.yaml</code> 并 push，下一次运行就回到 "
             "<code>config.yaml</code> 的人工基线。<br>"
             "或者本地跑 <code>python src/eval_daily.py --stage rollback</code>"
             "</div>")
    p.append("<div class='meta'>排序仍然 100% 由确定性代码完成。"
             "LLM 只做归因和提案，提案要过同一道闸门。</div>")
    return "".join(p)


def send(date: str, html: str, cfg: dict) -> None:
    """复用竞价线的 SMTP 底层。失败只记日志，不抛——学习系统崩了不能
    影响别的东西，这封信本身也不是关键路径。
    """
    try:
        import mailer
        c = mailer._conf()
        mailer._send(c, f"[参数更新] A股竞价筛选 {date}", html)
        log.info("变更邮件已发出")
    except Exception as e:  # noqa: BLE001
        log.warning("变更邮件发送失败（不影响参数已生效）: %s", e)
