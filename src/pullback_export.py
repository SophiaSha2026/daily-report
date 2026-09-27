"""
长期调整突破的面板和邮件。

面板的样式和自动刷新脚本和起涨预测共用（panel_style），两个面板长得一样、
行为一致。邮件复用 mailer 的 SMTP 配置和发送函数，不重复实现连接逻辑。

外部输入（行情源给的股票名称）一律在渲染点转义，同 breakout/export.py。
"""
from __future__ import annotations

import datetime as _dt
import html as _h
import json
import logging
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

from mailer import _CSS, _conf, _send
from panel_style import PANEL_CSS, REFRESH_JS

log = logging.getLogger("pullback.export")
NAME = "长期调整突破"
BOARD = {"main": "主板", "chinext": "创业板", "star": "科创板", "bj": "北交所"}


def _esc(x) -> str:
    return _h.escape(str(x if x is not None else ""))


def _pct(x, digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):+.{digits}f}%" if sign else f"{float(x):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def meta_line(meta: dict) -> str:
    return (f"{meta.get('date', '')} 收盘后扫描 · 全市场 {meta.get('n_stocks', '?')} 只 · "
            f"今日倍量大阳线 {meta.get('n_big_today', '?')} 只 · "
            f"调整中 {meta.get('n_watch', '?')} 只 · "
            f"今日二次进攻 {meta.get('n', '?')} 只")


def hist_line(meta: dict) -> str:
    h = meta.get("hist") or {}
    if not h.get("n"):
        return ""
    s = (f"同口径回看 {h.get('from', '')[:7]} ~ {h.get('to', '')[:7]}：共 {h['n']} 次，"
         f"平均每月约 {h.get('per_month')} 次，大多数交易日是空榜")
    if h.get("n_final"):
        s += (f"。走满 {h.get('bars', 20)} 个交易日的 {h['n_final']} 次里，之后最高价涨幅"
              f"中位 {_pct(h.get('max_up_median'))}，第 {h.get('bars', 20)} 天收盘中位 "
              f"{_pct(h.get('ret_median'))}（只描述历史，不是预测）")
    return s


def _bonus(r: dict, html: bool = True) -> str:
    items = [("量超首阳", r.get("bonus_vol")), ("缩到一半", r.get("bonus_half")),
             ("守开盘价", r.get("bonus_open"))]
    if not html:
        return " ".join(("✓" if ok else "✗") + nm for nm, ok in items)
    return " ".join(f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✗"}{nm}</span>'
                    for nm, ok in items)


def _cells(r: dict) -> tuple[str, str, str, str, str]:
    """一行的五个格子：名称、今日、首阳、调整、加分。邮件和面板共用。"""
    tag = ' <span class="warnt">一字板</span>' if r.get("one_word") else ""
    name = (f'{_esc(r.get("name", ""))}{tag}'
            f'<div class="dim">{BOARD.get(r.get("board"), "")}</div>')
    today = (f'收 {float(r["close"]):.2f} <span class="up">{_pct(r["gain_pct"], 2)}</span>'
             f'<div class="dim">量 {float(r["vol_ratio"]):.1f} 倍 · 首阳的 '
             f'{100 * float(r["vol_vs_launch"]):.0f}%</div>')
    launch = (f'{_esc(str(r["launch_date"])[5:])} <span class="up">'
              f'{_pct(r["launch_gain_pct"])}</span>'
              f'<div class="dim">量 {float(r["launch_vol_ratio"]):.1f} 倍 · 横盘振幅 '
              f'{float(r["base_amp_pct"]):.0f}%</div>')
    adj = (f'{int(r["adjust_days"])} 天'
           f'<div class="dim">最低量 {100 * float(r["adj_vol_min"]):.0f}% · 最深 '
           f'{_pct(r["adj_drawdown_pct"])}</div>')
    return name, today, launch, adj, _bonus(r)


# ---------------------------------------------------------------------
#  面板
# ---------------------------------------------------------------------
_PANEL = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>长期调整突破 __DATE__</title><style>
""" + PANEL_CSS + """
.dim{color:#8f9aa8;font-size:12px}.ok{color:#5fd18c}.no{color:#6b7280}
.warnt{color:#e3a23b;font-size:12px}
h2{font-size:14px;margin:22px 0 8px;color:#c9d1d9}
.rules{color:#9aa4b2;font-size:12px;line-height:1.8;margin-top:14px}
.empty{color:#9aa4b2;padding:14px 0}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
@media (max-width:640px){body{padding:10px}td,th{padding:5px 6px;font-size:12px}}
</style></head><body>
<div id="stale"></div>
<h1>长期调整突破 · __DATE__</h1>
<div class="sub">__SUB__</div>
__LATE__
<div class="bar">
  <button onclick="cp(this)">复制今日代码</button>
</div>
__TODAY__
<h2>调整中（首阳已出，还没二次进攻）· __NW__ 只</h2>
__WATCH__
<h2>最近 __NH__ 次成立（之后怎么走的）</h2>
__HIST__
<div class="rules">__RULES__</div>
<div class="tip">点代码即复制；切到同花顺，剪贴板识别框会自动弹出。</div>
<div id="toast"></div>
<script>
const D=__DATA__;
function toast(m){const t=document.getElementById('toast');t.textContent=m;
  t.className='show';setTimeout(()=>t.className='',1300);}
function put(txt,msg){
  navigator.clipboard.writeText(txt).then(()=>toast(msg))
  .catch(()=>{const a=document.createElement('textarea');a.value=txt;
    document.body.appendChild(a);a.select();document.execCommand('copy');
    a.remove();toast(msg);});
}
function cp(btn){
  if(!D.length){toast('今日没有');return;}
  put(D.join('\\n'),'已复制 '+D.length+' 个代码');btn.classList.add('on');
}
function one(c){put(c,'已复制 '+c);}
""" + REFRESH_JS + """
</script></body></html>"""


def _code_td(code: str) -> str:
    c = _esc(str(code).zfill(6))
    return f'<td class="code" onclick="one(\'{c}\')">{c}</td>'


def today_table(rows: list[dict]) -> str:
    if not rows:
        return ('<div class="empty">今天没有股票走完三步（横盘 → 首阳 → 缩量调整 → '
                '二次进攻）。这是常态，不是故障。</div>')
    tr = []
    for i, r in enumerate(rows, 1):
        name, today, launch, adj, bonus = _cells(r)
        tr.append(f"<tr><td>{i}</td>{_code_td(r['code'])}<td>{name}</td><td>{today}</td>"
                  f"<td>{launch}</td><td>{adj}</td><td>{bonus}</td></tr>")
    return ('<div class="tw"><table><thead><tr><th>#</th><th>代码</th><th>名称</th>'
            "<th>今日</th><th>首阳</th><th>调整</th><th>加分项</th></tr></thead><tbody>"
            + "".join(tr) + "</tbody></table></div>")


def watch_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="empty">没有。</div>'
    rows = sorted(rows, key=lambda r: (float(r.get("to_high_pct") or 0), str(r["code"])))
    tr = []
    for r in rows:
        L = int(r.get("adjust_days") or 0)
        state = "今日首阳" if L == 0 else f"调整第 {L} 天"
        vm = r.get("adj_vol_min")
        tr.append(f"<tr>{_code_td(r['code'])}<td>{_esc(r.get('name', ''))}"
                  f"<div class=\"dim\">{BOARD.get(r.get('board'), '')}</div></td>"
                  f"<td>{_esc(str(r['launch_date'])[5:])} "
                  f"<span class=\"up\">{_pct(r.get('launch_gain_pct'))}</span></td>"
                  f"<td>{state}</td>"
                  f"<td>{'-' if vm is None else f'{100 * float(vm):.0f}%'}</td>"
                  f"<td>{_pct(r.get('to_high_pct'), 2)}</td></tr>")
    return ('<div class="tw"><table><thead><tr><th>代码</th><th>名称</th><th>首阳</th>'
            "<th>进度</th><th>调整最低量</th><th>离首阳高点</th></tr></thead><tbody>"
            + "".join(tr) + "</tbody></table></div>")


def hist_table(rows: list[dict], bars: int = 20) -> str:
    if not rows:
        return '<div class="empty">还没有。</div>'
    tr = []
    for r in reversed(rows):
        n_after = int(r.get("n_after") or 0)
        tail = "" if n_after >= bars else f'<div class="dim">才走 {n_after} 天</div>'
        tr.append(f"<tr><td>{_esc(r['date'])}</td>{_code_td(r['code'])}"
                  f"<td>{_esc(r.get('name', ''))}</td>"
                  f"<td>{int(r['adjust_days'])} 天</td>"
                  f"<td class=\"up\">{_pct(r.get('max_up_pct'))}</td>"
                  f"<td>{_pct(r.get('ret_pct'))}{tail}</td></tr>")
    return (f'<div class="tw"><table><thead><tr><th>成立日</th><th>代码</th><th>名称</th>'
            f"<th>调整</th><th>之后 {bars} 天最高</th><th>第 {bars} 天收盘</th></tr></thead>"
            "<tbody>" + "".join(tr) + "</tbody></table></div>")


def write_panel(sel: list[dict], watch: list[dict], hist: list[dict], meta: dict,
                out_dir: Path, date: str, late_note: str = "") -> Path:
    out_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    (out_dir / "stamp.txt").write_text(stamp, encoding="utf-8")
    bars = int((meta.get("hist") or {}).get("bars", 20))
    rules = "<br>".join(_esc(x) for x in (meta.get("rules") or []))
    hl = hist_line(meta)
    if hl:
        rules += "<br>" + _esc(hl)
    html = (_PANEL.replace("__DATE__", _esc(date))
            .replace("__SUB__", _esc(meta_line(meta)))
            .replace("__LATE__", f'<div id="late" class="warnt">{_esc(late_note)}</div>'
                     if late_note else "")
            .replace("__TODAY__", today_table(sel))
            .replace("__NW__", str(len(watch)))
            .replace("__WATCH__", watch_table(watch))
            .replace("__NH__", str(len(hist)))
            .replace("__HIST__", hist_table(hist, bars))
            .replace("__RULES__", rules)
            .replace("__DATA__", json.dumps([str(r["code"]).zfill(6) for r in sel]))
            .replace("__STAMP__", stamp)
            # build_site.py 把 out_pullback/stamp.txt 发布成这个名字，
            # 避免和起涨预测的 stamp 在站点根目录撞名
            .replace("__STAMPFILE__", "stamp-pullback.txt")
            # 收盘后跑的线，面板日期本来就是最近一个已收盘交易日：关掉「日期不是
            # 今天」那条横幅（否则次日早上 100% 误报）
            .replace("__LAGOK__", "true"))
    p = out_dir / "panel.html"
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------
#  邮件
# ---------------------------------------------------------------------
_MAIL_CSS = _CSS + """
.dim{color:#888;font-size:12px}.ok{color:#2e7d32}.no{color:#aaa}
.warnt{color:#b26a00;font-size:12px}
"""


def build_html(date: str, sel: list[dict], watch: list[dict], meta: dict,
               page_url: str = "", late_note: str = "") -> str:
    parts = [f"<style>{_MAIL_CSS}</style>", f'<div class="meta">{_esc(meta_line(meta))}</div>']
    if late_note:
        parts.append(f'<div class="warn"><b>{_esc(late_note)}</b></div>')
    if sel:
        tr = []
        for r in sel:
            name, today, launch, adj, bonus = _cells(r)
            tr.append(f'<tr><td class="c">{_esc(r["code"])}</td><td>{name}</td>'
                      f"<td>{today}</td><td>{launch}</td><td>{adj}</td><td>{bonus}</td></tr>")
        parts.append("<table><thead><tr><th>代码</th><th>名称</th><th>今日</th><th>首阳</th>"
                     "<th>调整</th><th>加分项</th></tr></thead><tbody>"
                     + "".join(tr) + "</tbody></table>")
    else:
        parts.append('<div class="meta" style="font-size:14px;color:#333">今天没有股票走完三步'
                     '（横盘 → 首阳 → 缩量调整 → 二次进攻）。这是常态，不是故障。</div>')
    if watch:
        parts.append(f'<div class="meta">调整中、还没二次进攻的 {len(watch)} 只在面板里。</div>')
    if page_url:
        parts.append(f'<div class="meta">在线面板：<a href="{_esc(page_url)}">'
                     f'{_esc(page_url)}</a></div>')
    rules = "<br>".join(_esc(x) for x in (meta.get("rules") or []))
    hl = hist_line(meta)
    parts.append(f'<div class="meta" style="margin-top:14px">{rules}'
                 + (f"<br>{_esc(hl)}" if hl else "") + "</div>")
    return "".join(parts)


def subject(date: str, sel: list[dict]) -> str:
    if not sel:
        return f"[{NAME}] {date} · 今日 0 只"
    return f"[{NAME}] {date} · {len(sel)}只 · 首位 {sel[0].get('name') or sel[0]['code']}"


def send_mail(date: str, sel: list[dict], watch: list[dict], meta: dict,
              page_url: str = "", late_note: str = "",
              attachments: list[Path] | None = None) -> None:
    c = _conf()
    m = EmailMessage()
    m["Subject"] = subject(date, sel)
    m["From"] = formataddr((NAME, c["user"]))
    m["To"] = ", ".join(c["to"])
    lines = [f"{date} {NAME}：{len(sel)} 只。请用 HTML 视图查看。"]
    for r in sel:
        lines.append(f"{r['code']} {r.get('name', '')} 收 {float(r['close']):.2f} "
                     f"{float(r['gain_pct']):+.2f}% | 首阳 {r['launch_date']} | "
                     f"调整 {r['adjust_days']} 天 | {_bonus(r, html=False)}")
    if page_url:
        lines.append(f"在线面板：{page_url}")
    m.set_content("\n".join(lines))
    m.add_alternative(build_html(date, sel, watch, meta, page_url, late_note),
                      subtype="html")
    for p in (attachments or []):
        m.add_attachment(Path(p).read_bytes(), maintype="application",
                         subtype="octet-stream", filename=Path(p).name)
    _send(m, c)
    log.info("%s邮件已发送（%d 只，%d 个附件）", NAME, len(sel), len(attachments or []))
