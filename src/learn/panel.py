"""
学习面板：learn.html。

替代 Electron 的决定（2026-09-03）：TUI 已覆盖全部操作，GUI 的真实增量
是可视化，而仓库已有现成的面板管线。这一页从 state/ 和 out_learn/ 读数，
纯静态，进 GitHub Pages，手机也能看。

四块内容：
    1. 每日 IC 与前 10 超额收益（在线积累的真值天）
    2. 最近一次七道闸裁决
    3. 归因分布（LLM 通道 1 的枚举计数）
    4. 模型擂台横评（有就展示）

SVG 手画，不引外部图表库——Pages 上一个 <script src> 都是隐患。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
STATE = ROOT / "state"
OUTL = ROOT / "out_learn"

_CSS = """
body{font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
     max-width:900px;margin:0 auto;padding:14px;background:#111;color:#ddd}
h1{font-size:20px;margin:6px 0}h2{font-size:15px;margin:18px 0 6px;color:#aaa}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #333;padding:4px 8px;text-align:left}
th{background:#1a1a1a;color:#999}
.ok{color:#4c9}.no{color:#e66}.dim{color:#777;font-size:12px}
.card{background:#181818;border:1px solid #2a2a2a;border-radius:8px;
      padding:10px 14px;margin:10px 0}
svg{width:100%;height:auto;background:#181818;border-radius:8px}
.pos{fill:#4c9}.neg{fill:#e66}
"""


def _bars(vals: list[tuple[str, float]], title: str, fmt: str = "%+.2f%%",
          height: int = 160) -> str:
    """一组正负柱。零轴居中，正绿负红。"""
    if not vals:
        return ""
    w, pad = 880, 30
    n = len(vals)
    bw = max(4.0, min(40.0, (w - 2 * pad) / max(n, 1) - 4))
    mx = max(0.001, max(abs(v) for _, v in vals))
    mid = height / 2
    parts = [f'<svg viewBox="0 0 {w} {height + 30}" '
             f'xmlns="http://www.w3.org/2000/svg">',
             f'<text x="{pad}" y="14" fill="#999" font-size="12">{title}'
             f'</text>',
             f'<line x1="{pad}" y1="{mid + 15}" x2="{w - pad}" '
             f'y2="{mid + 15}" stroke="#333"/>']
    for i, (label, v) in enumerate(vals):
        x = pad + i * ((w - 2 * pad) / n) + 2
        h = abs(v) / mx * (mid - 20)
        y = mid + 15 - h if v >= 0 else mid + 15
        cls = "pos" if v >= 0 else "neg"
        parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" '
                     f'width="{bw:.1f}" height="{max(h, 1):.1f}">'
                     f'<title>{label}  {fmt % v}</title></rect>')
        if n <= 20:
            parts.append(f'<text x="{x + bw / 2:.1f}" y="{height + 26}" '
                         f'fill="#666" font-size="9" text-anchor="middle">'
                         f'{label[5:]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def build(daily_metrics: list[dict] | None = None) -> Path:
    """生成 out_learn/learn.html。所有输入缺失都能出页面，只是块少。"""
    st = _load(STATE / "learning_status.json")
    race = _load(OUTL / "model_race.json")

    # 每日指标：从 status 里带出来（stage_learn 落的），或调用方直接给
    daily = daily_metrics or st.get("daily", [])

    p = [f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>学习面板</title>",
         f"<style>{_CSS}</style></head><body>",
         "<h1>自评估 · 学习 · 迭代</h1>",
         "<div class='dim'><a href='index.html' style='color:#8ab4f8'>"
         "← 竞价面板</a> · <a href='pullback.html' style='color:#8ab4f8'>"
         "形态面板</a></div>",
         f"<div class='dim'>训练源 {st.get('train_source', '?')} · "
         f"截至 {st.get('date', '?')} · 参数版本 "
         f"{st.get('theta_version', '基线')}</div>"]

    m = st.get("metrics", {})
    if m:
        p.append("<div class='card'><table><tr>"
                 "<th>天数</th><th>日IC均值</th><th>ICIR</th>"
                 "<th>前10超额</th><th>前10胜率</th><th>日均通过</th></tr>"
                 f"<tr><td>{m.get('days', 0)}</td>"
                 f"<td>{m.get('ic_mean', float('nan')):.4f}</td>"
                 f"<td>{m.get('icir', float('nan')):.3f}</td>"
                 f"<td>{m.get('top_excess', float('nan')):+.3%}</td>"
                 f"<td>{m.get('hit_rate', float('nan')):.1%}</td>"
                 f"<td>{m.get('avg_pool', 0):.0f}</td></tr></table></div>")

    if daily:
        p.append("<h2>每日：在线真值快照</h2>")
        p.append(_bars([(d["date"], d.get("ic", 0) or 0) for d in daily],
                       "日 IC（分数与当日实际超额的秩相关）", "%+.3f"))
        p.append(_bars([(d["date"], (d.get("top_excess", 0) or 0) * 100)
                        for d in daily], "前 10 超额收益 %"))

    v = st.get("verdict", {})
    if v.get("checks"):
        p.append("<h2>最近一次闸门裁决</h2><div class='card'><table>"
                 "<tr><th>闸门</th><th>结论</th><th>依据</th></tr>")
        for ckd in v["checks"]:
            mark = "<span class='ok'>过</span>" if ckd.get("passed") \
                else "<span class='no'>不过</span>"
            p.append(f"<tr><td>{ckd.get('name')}</td><td>{mark}</td>"
                     f"<td>{ckd.get('detail')}</td></tr>")
        p.append("</table>")
        p.append(("<div class='ok'>本次已接受参数变更</div>"
                  if v.get("accepted") else
                  "<div class='dim'>未接受变更，参数保持不变（这是常态，"
                  "不是故障：闸门的职责就是拦住噪声）</div>") + "</div>")

    counts: dict[str, int] = {}
    for f in (STATE / "llm_eval").glob("*.json"):
        j = _load(f)
        for it in j.get("items", []):
            counts[it.get("cause", "?")] = counts.get(it.get("cause", "?"), 0) + 1
    if counts:
        p.append("<h2>归因分布（LLM 通道 1）</h2><div class='card'><table>"
                 "<tr><th>原因</th><th>次数</th></tr>")
        for k, n in sorted(counts.items(), key=lambda x: -x[1]):
            p.append(f"<tr><td>{k}</td><td>{n}</td></tr>")
        p.append("</table></div>")

    if race.get("table"):
        p.append("<h2>模型擂台（走向前样本外）</h2><div class='card'><table>"
                 "<tr><th>模型</th><th>IC</th><th>ICIR</th>"
                 "<th>90%区间</th><th>前10超额</th><th>胜率</th></tr>")
        for r in race["table"]:
            p.append(f"<tr><td>{r['model']}</td><td>{r['IC']:.4f}</td>"
                     f"<td>{r['ICIR']:.3f}</td>"
                     f"<td>{r['ICIR_lo']:.2f}~{r['ICIR_hi']:.2f}</td>"
                     f"<td>{r['top_excess']:+.3%}</td>"
                     f"<td>{r['hit']:.0%}</td></tr>")
        p.append(f"</table><div class='dim'>选中：{race.get('winner', '?')}"
                 f" —— {race.get('why', '')}。胜者当老师（天花板估计与形状"
                 f"建议），不当排序器。</div></div>")

    p.append("<div class='dim'>排序 100% 确定性；LLM 只做归因与提案，"
             "提案过同一道闸。完整设计 docs/learning.md</div></body></html>")

    OUTL.mkdir(exist_ok=True)
    out = OUTL / "learn.html"
    out.write_text("".join(p), encoding="utf-8")
    log.info("学习面板已生成 %s", out)
    return out
