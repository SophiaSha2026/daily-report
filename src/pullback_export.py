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
    n = meta.get("n", "?")
    tail = f"清单 A {n} 只"
    if "n_b" in meta:
        tail += f" · 清单 B {meta['n_b']} 只"
    return (f"{meta.get('date', '')} 收盘后扫描 · 全市场 {meta.get('n_stocks', '?')} 只 · "
            f"今日倍量大阳线 {meta.get('n_big_today', '?')} 只 · " + tail)


def excluded_line(meta: dict, which: str = "") -> str:
    """走完了该走的几步、但按规则剔掉的：代码 名称：原因。which="b" 是清单 B 的。"""
    ex = meta.get(f"excluded{'_' + which if which else ''}") or {}
    if not ex:
        return ""
    nm = meta.get(f"excluded{'_' + which if which else ''}_names") or {}
    return f"剔除 {len(ex)} 只：" + "；".join(
        f"{c}{' ' + nm[c] if nm.get(c) else ''}：{why}" for c, why in ex.items())


def b_rate_line(meta: dict) -> str:
    """清单 B 的历史转化率：进过 B 的后来有多少走完二次进攻。"""
    b = (meta.get("hist") or {}).get("b") or {}
    if not b.get("n"):
        return ""
    return (f"历史上进过清单 B 的 {b['n']} 次里，{b['to_a']} 次（{100 * b['rate']:.1f}%）"
            f"之后走完了二次进攻、进了清单 A；其余大多跌破首阳最低价或窗口到期"
            f"（{str(b.get('from', ''))[:7]} 起，形态口径，没剔减持 / 定增）")


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
    if h.get("n_unknown"):
        s += f"。其中 {h['n_unknown']} 次减持 / 定增数据没拉到，按不剔算"
    return s


def _bonus(r: dict, html: bool = True) -> str:
    items = [("量超首阳", r.get("bonus_vol")), ("缩到一半", r.get("bonus_half")),
             ("守开盘价", r.get("bonus_open"))]
    if not html:
        return " ".join(("✓" if ok else "✗") + nm for nm, ok in items)
    return " ".join(f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✗"}{nm}</span>'
                    for nm, ok in items)


def base_text(r: dict, html: bool = True) -> str:
    """前期调整：首阳前收盘价待在 30% 振幅里的交易日数（pullback.base_run）。"""
    n = r.get("base_run")
    if n is None:
        return "-"
    n = int(n)
    head = f"{'至少 ' if r.get('base_run_full') else ''}{n} 个交易日"
    if not html:
        return head
    return f'{head}<div class="dim">约 {n / 21:.1f} 个月</div>'


def _cells(r: dict) -> tuple[str, str, str, str, str, str]:
    """一行的六个格子：名称、今日、前期调整、首阳、缩量调整、加分。邮件和面板共用。"""
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
    return name, today, base_text(r), launch, adj, _bonus(r)


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
.excl{color:#9aa4b2;font-size:12px;margin:6px 0 0}
details{margin-top:22px}summary{cursor:pointer;color:#c9d1d9;font-size:14px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
.bar{align-items:center}.lbl{color:#888;font-size:12px}
select{background:#2a2f38;color:#e6e6e6;border:1px solid #3a4149;border-radius:5px;
       padding:5px 8px;font-size:13px;max-width:100%}
@media (max-width:640px){body{padding:10px}td,th{padding:5px 6px;font-size:12px}}
</style></head><body>
<div id="stale"></div>
<h1>长期调整突破 · <span id="hd">__DATE__</span></h1>
<div class="bar">
  __PICK__
  <button onclick="cp(this,'a','清单 A')">复制清单 A</button>
  <button onclick="cp(this,'b','清单 B')">复制清单 B</button>
</div>
__DAYS__
<div class="excl" id="brate">__BRATE__</div>
<details><summary>以前成立过的 __NH__ 次（不是今天的清单，点开看之后怎么走的）</summary>
__HIST__
</details>
<div class="rules">__RULES__</div>
<div class="tip">点代码即复制；切到同花顺，剪贴板识别框会自动弹出。</div>
<div id="toast"></div>
<script>
/* L：每一天两份清单的代码，{日期: {a: [...], b: [...]}}。复制按钮跟着下拉走 */
const L=__LISTS__, D0='__DATE__';
let cur=D0;
function toast(m){const t=document.getElementById('toast');t.textContent=m;
  t.className='show';setTimeout(()=>t.className='',1300);}
function put(txt,msg){
  navigator.clipboard.writeText(txt).then(()=>toast(msg))
  .catch(()=>{const a=document.createElement('textarea');a.value=txt;
    document.body.appendChild(a);a.select();document.execCommand('copy');
    a.remove();toast(msg);});
}
function cp(btn,k,nm){
  const D=(L[cur]||{})[k]||[], w=(cur===D0?'':cur.slice(5)+' ');
  if(!D.length){toast(nm+' '+(w||'今天 ')+'没有');return;}
  put(D.join('\\n'),'已复制 '+w+nm+' '+D.length+' 个代码');btn.classList.add('on');
}
function one(c){put(c,'已复制 '+c);}
/* 日期下拉：每天那块在 Python 里渲染好了，这里只切显示。选中的日子记在 #日期 上，
   刷新还停在那天；自动刷新跳新 stamp 时不带 #，回到最新一天 */
function pick(d){
  if(!L[d]) d=D0;
  cur=d;
  document.querySelectorAll('.day').forEach(e=>{e.hidden=(e.dataset.d!==d);});
  document.getElementById('hd').textContent=d;
  document.querySelectorAll('.bar button.on').forEach(b=>b.classList.remove('on'));
  const s=document.getElementById('daysel'); if(s) s.value=d;
  try{history.replaceState(null,'',location.pathname+location.search+(d===D0?'':'#'+d));}
  catch(e){}
}
if(location.hash.length>1 && L[location.hash.slice(1)]) pick(location.hash.slice(1));
""" + REFRESH_JS + """
</script></body></html>"""


def _b_cells(r: dict) -> tuple[str, str, str, str, str, str]:
    """清单 B 一行：名称、前期调整、首阳、缩量调整、离突破、加分（到目前为止）。"""
    name = (f'{_esc(r.get("name", ""))}'
            f'<div class="dim">{BOARD.get(r.get("board"), "")}</div>')
    launch = (f'{_esc(str(r["launch_date"])[5:])} <span class="up">'
              f'{_pct(r.get("launch_gain_pct"))}</span>'
              f'<div class="dim">量 {float(r["launch_vol_ratio"]):.1f} 倍 · 横盘振幅 '
              f'{float(r["base_amp_pct"]):.0f}%</div>')
    vm, va = r.get("adj_vol_min"), r.get("adj_vol_mean")
    adj = (f'{int(r["adjust_days"])} 天'
           f'<div class="dim">最低量 {100 * float(vm or 0):.0f}% · 均量 '
           f'{100 * float(va or 0):.0f}% · 最深 {_pct(r.get("adj_drawdown_pct"))}</div>')
    gap = (f'还差 {_pct(r.get("to_high_pct"), 2, sign=False)}'
           f'<div class="dim">收盘站上 {float(r["launch_high"]):.2f} · '
           f'还能等 {int(r.get("wait_left") or 0)} 天</div>')
    items = [("缩到一半", r.get("bonus_half")), ("守开盘价", r.get("bonus_open"))]
    bonus = " ".join(f'<span class="{"ok" if ok else "no"}">{"✓" if ok else "✗"}{nm}</span>'
                     for nm, ok in items)
    return name, base_text(r), launch, adj, gap, bonus


def b_table(rows: list[dict], html_panel: bool = True, when: str = "今天") -> str:
    if not rows:
        return f'<div class="empty">{when}没有走完前三步、还在等二次进攻的。</div>'
    tr = []
    for i, r in enumerate(rows, 1):
        name, base, launch, adj, gap, bonus = _b_cells(r)
        code = (_code_td(r["code"]) if html_panel
                else f'<td class="c">{_esc(str(r["code"]).zfill(6))}</td>')
        tr.append(f"<tr><td>{i}</td>{code}<td>{name}</td><td>{base}</td><td>{launch}</td>"
                  f"<td>{adj}</td><td>{gap}</td><td>{bonus}</td></tr>")
    return ('<div class="tw"><table><thead><tr><th>#</th><th>代码</th><th>名称</th>'
            "<th>前期调整</th><th>首阳</th><th>缩量调整</th><th>离突破</th><th>加分项</th>"
            "</tr></thead><tbody>" + "".join(tr) + "</tbody></table></div>")


def _code_td(code: str) -> str:
    c = _esc(str(code).zfill(6))
    return f'<td class="code" onclick="one(\'{c}\')">{c}</td>'


def today_table(rows: list[dict], when: str = "今天") -> str:
    if not rows:
        return (f'<div class="empty">{when}没有股票走完三步（横盘 → 首阳 → 缩量调整 → '
                '二次进攻）。这是常态，不是故障。</div>')
    tr = []
    for i, r in enumerate(rows, 1):
        name, today, base, launch, adj, bonus = _cells(r)
        tr.append(f"<tr><td>{i}</td>{_code_td(r['code'])}<td>{name}</td><td>{today}</td>"
                  f"<td>{base}</td><td>{launch}</td><td>{adj}</td><td>{bonus}</td></tr>")
    return ('<div class="tw"><table><thead><tr><th>#</th><th>代码</th><th>名称</th>'
            f"<th>{'今日' if when == '今天' else '当日'}</th><th>前期调整</th><th>首阳</th>"
            "<th>缩量调整</th><th>加分项</th></tr></thead><tbody>"
            + "".join(tr) + "</tbody></table></div>")


def day_block(date: str, sel: list[dict], blist: list[dict] | None, head: str,
              tail_a: str = "", tail_b: str = "", latest: bool = True) -> str:
    """某一天的抬头 + 清单 A + 清单 B。面板顶上的日期下拉按 data-d 切换显示哪一块。
    最新那天是 out_pullback/ 当天的产物（带剔除原因、补发说明）；以前的是
    data/pullback/ 的存档（只有两份清单本身）。blist 为 None：那天没有清单 B 的存档。"""
    when = "今天" if latest else "当天"
    b = (b_table(blist, when=when) if blist is not None
         else '<div class="empty">这天没有清单 B 的存档。</div>')
    return (f'<div class="day" data-d="{_esc(date)}"{"" if latest else " hidden"}>'
            + head +
            f'<h2>清单 A · {when}二次进攻（走完横盘 → 首阳 → 缩量调整 → 二次进攻）'
            f'· {len(sel)} 只</h2>' + today_table(sel, when) + tail_a
            + f'<h2>清单 B · 二次进攻前（走完横盘 → 首阳 → 缩量调整）'
            f'· {len(blist or [])} 只</h2>' + b + tail_b + '</div>')


def past_block(p: dict) -> str:
    """存档里的某一天。渲染失败（以后改了列、旧存档少列）只坏这一块，不拦面板和发信。"""
    d, a, b = p["date"], p.get("a") or [], p.get("b")
    head = (f'<div class="sub">当天收盘后扫描的存档 · 清单 A {len(a)} 只 · 清单 B '
            f'{"-" if b is None else len(b)} 只。数字都是那天收盘时的，不重算；'
            f'剔了谁只在最新一天列</div>')
    try:
        return day_block(d, a, b, head, latest=False)
    except Exception as e:  # noqa: BLE001
        log.warning("面板下拉：%s 的存档渲染失败（%s）", d, e)
        return (f'<div class="day" data-d="{_esc(d)}" hidden>{head}'
                f'<div class="empty">这天的存档读不出来（{_esc(e)}）。</div></div>')


def _weekday(d: str) -> str:
    try:
        return " 周" + "一二三四五六日"[_dt.date.fromisoformat(d).weekday()]
    except ValueError:
        return ""


def picker(date: str, days: list[tuple[str, int, int | None]]) -> str:
    """日期下拉。days 是 (日期, A 只数, B 只数) 新的在前，第一项是最新那天。只有一天就不出。"""
    if len(days) < 2:
        return ""
    opts = "".join(
        f'<option value="{_esc(d)}"{" selected" if d == date else ""}>{_esc(d)}{_weekday(d)}'
        f' · A {na} 只 · B {"-" if nb is None else nb} 只{"（最新）" if i == 0 else ""}</option>'
        for i, (d, na, nb) in enumerate(days))
    return (f'<span class="lbl">看哪一天</span>'
            f'<select id="daysel" onchange="pick(this.value)">{opts}</select>')


def hist_table(rows: list[dict], bars: int = 20) -> str:
    if not rows:
        return '<div class="empty">还没有。</div>'
    tr = []
    for r in reversed(rows):
        n_after = int(r.get("n_after") or 0)
        tail = "" if n_after >= bars else f'<div class="dim">才走 {n_after} 天</div>'
        tr.append(f"<tr><td>{_esc(r['date'])}</td>{_code_td(r['code'])}"
                  f"<td>{_esc(r.get('name', ''))}</td>"
                  f"<td>{base_text(r, html=False)}</td>"
                  f"<td>{int(r['adjust_days'])} 天</td>"
                  f"<td class=\"up\">{_pct(r.get('max_up_pct'))}</td>"
                  f"<td>{_pct(r.get('ret_pct'))}{tail}</td></tr>")
    return (f'<div class="tw"><table><thead><tr><th>成立日</th><th>代码</th><th>名称</th>'
            f"<th>前期调整</th><th>缩量调整</th><th>之后 {bars} 天最高</th>"
            f"<th>第 {bars} 天收盘</th></tr></thead>"
            "<tbody>" + "".join(tr) + "</tbody></table></div>")


def write_panel(sel: list[dict], blist: list[dict], hist: list[dict], meta: dict,
                out_dir: Path, date: str, late_note: str = "",
                past: list[dict] | None = None) -> Path:
    """past：以前几天的存档（pullback.recent_lists，新的在前），给了就在顶上出日期下拉。
    只进面板，不进邮件。"""
    out_dir.mkdir(exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    (out_dir / "stamp.txt").write_text(stamp, encoding="utf-8")
    bars = int((meta.get("hist") or {}).get("bars", 20))
    rules = "<br>".join(_esc(x) for x in (meta.get("rules") or []))
    hl = hist_line(meta)
    if hl:
        rules += "<br>" + _esc(hl)
    xl, xb = excluded_line(meta), excluded_line(meta, "b")
    head = f'<div class="sub">{_esc(meta_line(meta))}</div>' + (
        f'<div id="late" class="warnt">{_esc(late_note)}</div>' if late_note else "")
    past = [p for p in (past or []) if p.get("date") and p["date"] < date]
    days = day_block(date, sel, blist, head,
                     f'<div class="excl">{_esc(xl)}</div>' if xl else "",
                     f'<div class="excl">{_esc(xb)}</div>' if xb else "") \
        + "".join(past_block(p) for p in past)

    def codes(rows: list[dict] | None) -> list[str]:
        return [str(r.get("code", "")).zfill(6) for r in (rows or [])]

    lists = {date: {"a": codes(sel), "b": codes(blist)}}
    lists.update({p["date"]: {"a": codes(p.get("a")), "b": codes(p.get("b"))} for p in past})
    pick = picker(date, [(date, len(sel), len(blist))]
                  + [(p["date"], len(p.get("a") or []),
                      None if p.get("b") is None else len(p["b"])) for p in past])
    html = (_PANEL.replace("__DATE__", _esc(date))
            .replace("__BRATE__", _esc(b_rate_line(meta)))
            .replace("__NH__", str(len(hist)))
            .replace("__HIST__", hist_table(hist, bars))
            .replace("__RULES__", rules)
            .replace("__LISTS__", json.dumps(lists))
            .replace("__STAMP__", stamp)
            # build_site.py 把 out_pullback/stamp.txt 发布成这个名字，
            # 避免和起涨预测的 stamp 在站点根目录撞名
            .replace("__STAMPFILE__", "stamp-pullback.txt")
            # 收盘后跑的线，面板日期本来就是最近一个已收盘交易日：关掉「日期不是
            # 今天」那条横幅（否则次日早上 100% 误报）
            .replace("__LAGOK__", "true")
            # 带表格的两块最后塞：里面的名称来自行情源，别被后面的占位符替换碰到
            .replace("__PICK__", pick)
            .replace("__DAYS__", days))
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


def build_html(date: str, sel: list[dict], blist: list[dict], meta: dict,
               page_url: str = "", late_note: str = "") -> str:
    parts = [f"<style>{_MAIL_CSS}</style>", f'<div class="meta">{_esc(meta_line(meta))}</div>']
    if late_note:
        parts.append(f'<div class="warn"><b>{_esc(late_note)}</b></div>')
    parts.append(f'<h3>清单 A · 今天二次进攻 · {len(sel)} 只</h3>')
    if sel:
        tr = []
        for r in sel:
            name, today, base, launch, adj, bonus = _cells(r)
            tr.append(f'<tr><td class="c">{_esc(r["code"])}</td><td>{name}</td>'
                      f"<td>{today}</td><td>{base}</td><td>{launch}</td><td>{adj}</td>"
                      f"<td>{bonus}</td></tr>")
        parts.append("<table><thead><tr><th>代码</th><th>名称</th><th>今日</th><th>前期调整</th>"
                     "<th>首阳</th><th>缩量调整</th><th>加分项</th></tr></thead><tbody>"
                     + "".join(tr) + "</tbody></table>")
    else:
        parts.append('<div class="meta" style="font-size:14px;color:#333">今天没有股票走完三步'
                     '（横盘 → 首阳 → 缩量调整 → 二次进攻）。这是常态，不是故障。</div>')
    xl = excluded_line(meta)
    if xl:
        parts.append(f'<div class="meta">{_esc(xl)}</div>')
    parts.append(f'<h3>清单 B · 二次进攻前（走完横盘 → 首阳 → 缩量调整）· {len(blist)} 只</h3>')
    parts.append(b_table(blist, html_panel=False))
    for line in (excluded_line(meta, "b"), b_rate_line(meta)):
        if line:
            parts.append(f'<div class="meta">{_esc(line)}</div>')
    if page_url:
        parts.append(f'<div class="meta">在线面板：<a href="{_esc(page_url)}">'
                     f'{_esc(page_url)}</a></div>')
    rules = "<br>".join(_esc(x) for x in (meta.get("rules") or []))
    hl = hist_line(meta)
    parts.append(f'<div class="meta" style="margin-top:14px">{rules}'
                 + (f"<br>{_esc(hl)}" if hl else "") + "</div>")
    return "".join(parts)


def subject(date: str, sel: list[dict], blist: list[dict] | None = None) -> str:
    tail = f" · 清单B {len(blist)}只" if blist is not None else ""
    if not sel:
        return f"[{NAME}] {date} · 清单A 0只{tail}"
    return (f"[{NAME}] {date} · 清单A {len(sel)}只{tail} · "
            f"A 首位 {sel[0].get('name') or sel[0]['code']}")


def send_mail(date: str, sel: list[dict], blist: list[dict], meta: dict,
              page_url: str = "", late_note: str = "",
              attachments: list[Path] | None = None) -> None:
    c = _conf()
    m = EmailMessage()
    m["Subject"] = subject(date, sel, blist)
    m["From"] = formataddr((NAME, c["user"]))
    m["To"] = ", ".join(c["to"])
    lines = [f"{date} {NAME}：清单 A {len(sel)} 只，清单 B {len(blist)} 只。请用 HTML 视图查看。"]
    for r in sel:
        lines.append(f"{r['code']} {r.get('name', '')} 收 {float(r['close']):.2f} "
                     f"{float(r['gain_pct']):+.2f}% | 前期调整 {base_text(r, html=False)} | "
                     f"首阳 {r['launch_date']} | 缩量调整 {r['adjust_days']} 天 | "
                     f"{_bonus(r, html=False)}")
    if excluded_line(meta):
        lines.append(excluded_line(meta))
    for r in blist:
        lines.append(f"[B] {r['code']} {r.get('name', '')} 首阳 {r['launch_date']} | "
                     f"前期调整 {base_text(r, html=False)} | 缩量调整 {r['adjust_days']} 天 | "
                     f"离突破还差 {float(r['to_high_pct']):.2f}%（收盘站上 "
                     f"{float(r['launch_high']):.2f}）| 还能等 {r['wait_left']} 天")
    if page_url:
        lines.append(f"在线面板：{page_url}")
    m.set_content("\n".join(lines))
    m.add_alternative(build_html(date, sel, blist, meta, page_url, late_note),
                      subtype="html")
    for p in (attachments or []):
        m.add_attachment(Path(p).read_bytes(), maintype="application",
                         subtype="octet-stream", filename=Path(p).name)
    _send(m, c)
    log.info("%s邮件已发送（清单 A %d 只、B %d 只，%d 个附件）", NAME, len(sel), len(blist),
             len(attachments or []))
