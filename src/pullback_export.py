"""
形态扫描的产物：同花顺自选股 txt、HTML 面板、邮件。

面板的样式和自动刷新脚本直接复用竞价那套（`ths_export.PANEL_CSS` /
`REFRESH_JS`），两个面板长得一样、行为一致；只有表头和单元格不同。
邮件复用 `mailer` 的 SMTP 配置和发送函数，不重复实现一遍连接逻辑。
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
from pathlib import Path
from email.message import EmailMessage
from email.utils import formataddr

from ths_export import PANEL_CSS, REFRESH_JS
from mailer import _conf, _send, _CSS

log = logging.getLogger("pullback.export")


# ---------------------------------------------------------------------
def write_blocks(rows: list[dict], out_dir: Path, date: str) -> list[Path]:
    """同花顺自选股导入用的纯代码 txt。GBK + CRLF，同花顺只认这个。"""
    out_dir.mkdir(exist_ok=True)
    p = out_dir / "形态_全部.txt"
    p.write_bytes(("\r\n".join(r["code"] for r in rows) + "\r\n").encode("gbk"))
    return [p]


# ---------------------------------------------------------------------
_PANEL = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>形态榜 __DATE__</title><style>
""" + PANEL_CSS + """
.dim{color:#8f9aa8}.ok{color:#5fd18c}
</style></head><body>
<div id="stale"></div>
<h1>启动-缩量回调-再启动 · __DATE__</h1>
<div class="sub">__SUB__</div>
<div class="bar">
  <button onclick="cp('all',this)">复制全部代码</button>
</div>
<table><thead><tr>
<th>#</th><th>代码</th><th>名称</th><th>收盘</th><th>今日涨幅</th><th>今日量比</th>
<th>今日换手</th><th>启动日</th><th>启动涨幅</th><th>调整</th><th>回撤</th>
<th>分</th><th>理由 / 风险</th>
</tr></thead><tbody>__ROWS__</tbody></table>
<div class="tip">
「今日量比」「启动量比」都是<b>成交量</b>相对前一交易日的倍数，不是成交额，
也不是各家软件那个口径不一的量比。<br>
「调整」是启动日次日起到今天前一日的交易日数；「回撤」是调整期最低价相对
启动日收盘价的幅度。<br>
点任意代码即复制；复制后切到同花顺，剪贴板识别框会自动弹出。
</div>
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
function cp(k,btn){
  const s=D.map(x=>x.code);
  if(!s.length){toast('今日无标的');return;}
  put(s.join('\\n'), '已复制 '+s.length+' 个代码');
  btn.classList.add('on');
}
function one(c){put(c,'已复制 '+c);}
""" + REFRESH_JS + """
</script></body></html>"""


def write_panel(rows: list[dict], texts: dict, out_dir: Path, date: str,
                notice: str = "", stat: dict | None = None) -> Path:
    out_dir.mkdir(exist_ok=True)
    tr = []
    for i, r in enumerate(rows, 1):
        t = texts.get(r["code"], {})
        cell = (f'<div class="rn">{t.get("reason","")}</div>'
                if t.get("reason") else "")
        if t.get("risk"):
            cell += f'<div class="rz">⚠ {t["risk"]}</div>'
        brk = ' <span class="ok">破启动高</span>' if r.get("break_launch_high") else ""
        tr.append(
            f'<tr><td>{i}</td>'
            f'<td class="code" onclick="one(\'{r["code"]}\')">{r["code"]}</td>'
            f'<td>{r["name"]}{brk}</td>'
            f'<td>{r["close"]:.2f}</td>'
            f'<td class="up">+{r["gain_pct"]:.2f}%</td>'
            f'<td>{r["vol_ratio"]:.2f}x</td>'
            f'<td>{r["turnover"]:.2f}%</td>'
            f'<td class="dim">{r["launch_date"][5:]}</td>'
            f'<td class="up">+{r["launch_gain"]:.2f}%</td>'
            f'<td>{r["adjust_days"]}日 · 缩至{r["adjust_vol_mean_ratio"]*100:.0f}%</td>'
            f'<td>{r["adjust_drawdown_pct"]:.2f}%</td>'
            f'<td class="sc">{r["score"]:.0f}</td>'
            f'<td>{cell}</td></tr>')

    s = stat or {}
    sub = (f'共 {len(rows)} 只 · 收盘后扫描 · '
           f'全市场 {s.get("quotes", "?")} → 今日条件 {s.get("after_today", "?")} '
           f'→ 形态匹配 {s.get("matched", "?")}')
    if notice:
        sub += f' · <span style="color:#d0a34a">{notice}</span>'

    stamp = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime(
        "%Y%m%d-%H%M%S")
    (out_dir / "stamp.txt").write_text(stamp, encoding="utf-8")

    html = (_PANEL.replace("__DATE__", date).replace("__SUB__", sub)
            .replace("__STAMP__", stamp)
            # build_site.py 发布时把 out_pullback/stamp.txt 改名成这个，
            # 避免和竞价面板的 stamp.txt 在站点根目录撞名
            .replace("__STAMPFILE__", "stamp-pullback.txt")
            .replace("__ROWS__", "".join(tr) or
                     '<tr><td colspan="13" class="dim">今日无标的满足形态条件</td></tr>')
            .replace("__DATA__", json.dumps(
                [{"code": r["code"]} for r in rows], ensure_ascii=False)))
    p = out_dir / "panel.html"
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------
_HDR = ["代码", "名称", "收盘", "今日涨幅", "今日量比", "今日换手",
        "启动日", "启动涨幅", "调整", "回撤", "分", "理由 / 风险"]


def build_html(date: str, rows: list[dict], texts: dict, notice: str,
               page_url: str, stat: dict) -> str:
    th = "".join(f"<th>{h}</th>" for h in _HDR)
    body = []
    for r in rows:
        t = texts.get(r["code"], {})
        cell = (f'<div class="rn">{t.get("reason","")}</div>'
                if t.get("reason") else "")
        if t.get("risk"):
            cell += f'<div class="rz">⚠ {t["risk"]}</div>'
        brk = " ·破启动高" if r.get("break_launch_high") else ""
        body.append(
            f'<tr><td class="c">{r["code"]}</td><td>{r["name"]}{brk}</td>'
            f'<td>{r["close"]:.2f}</td>'
            f'<td class="up">+{r["gain_pct"]:.2f}%</td>'
            f'<td>{r["vol_ratio"]:.2f}x</td>'
            f'<td>{r["turnover"]:.2f}%</td>'
            f'<td>{r["launch_date"][5:]}</td>'
            f'<td class="up">+{r["launch_gain"]:.2f}%</td>'
            f'<td>{r["adjust_days"]}日·缩至{r["adjust_vol_mean_ratio"]*100:.0f}%</td>'
            f'<td>{r["adjust_drawdown_pct"]:.2f}%</td>'
            f'<td class="s">{r["score"]:.0f}</td>'
            f'<td>{cell}</td></tr>')

    parts = [f"<style>{_CSS}</style>",
             f'<div class="meta">{date} 收盘后形态扫描 · '
             f'全市场 {stat.get("quotes","?")} → 今日条件 {stat.get("after_today","?")} '
             f'→ 形态匹配 {stat.get("matched","?")} 只</div>']
    if notice:
        parts.append(f'<div class="notice">{notice}</div>')
    if page_url:
        parts.append(f'<div class="meta">在线面板：'
                     f'<a href="{page_url}">{page_url}</a></div>')
    if rows:
        parts.append(f"<table><thead><tr>{th}</tr></thead>"
                     f"<tbody>{''.join(body)}</tbody></table>")
    else:
        parts.append("<div class='meta'>今日没有标的同时满足三段条件。"
                     "这不是故障，形态本身就不是每天都有。</div>")
    parts.append(
        '<div class="meta" style="margin-top:14px">'
        '条件：启动日涨幅≥5%、量≥前日1.5倍、换手5%~10%；'
        '次日起调整1~6个交易日，缩量且最低价不破启动日最低价；'
        '今日涨幅≥5%、量≥前日1.5倍、换手5%~10%。'
        '「量」指成交量（手），不是成交额。</div>')
    return "".join(parts)


def send_pullback(date: str, rows: list[dict], texts: dict, notice: str = "",
                  attachments: list[Path] | None = None, page_url: str = "",
                  stat: dict | None = None) -> None:
    c = _conf()
    stat = stat or {}
    m = EmailMessage()
    top = rows[0]["name"] if rows else "无"
    m["Subject"] = f"[形态清单] {date} · {len(rows)}只 · 首位 {top}"
    m["From"] = formataddr(("形态机器人", c["user"]))
    m["To"] = ", ".join(c["to"])
    m.set_content(f"{date} 启动-缩量回调-再启动：{len(rows)} 只。请用 HTML 视图查看。"
                  + (f"\n在线面板：{page_url}" if page_url else ""))
    m.add_alternative(build_html(date, rows, texts, notice, page_url, stat),
                      subtype="html")
    for p in (attachments or []):
        m.add_attachment(Path(p).read_bytes(), maintype="application",
                         subtype="octet-stream", filename=Path(p).name)
    _send(m, c)
    log.info("形态清单邮件已发送 (%d 只, %d 附件)", len(rows), len(attachments or []))
