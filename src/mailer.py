"""邮件发送。SMTP over STARTTLS/SSL，凭证全部来自环境变量（GitHub Secrets）。"""
from __future__ import annotations

import os
import ssl
import smtplib
import logging
from pathlib import Path
from email.message import EmailMessage
from email.utils import formataddr

log = logging.getLogger(__name__)


def _conf() -> dict:
    return {
        "host": os.environ["SMTP_HOST"],
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ["SMTP_USER"],
        "pw":   os.environ["SMTP_PASS"],
        "to":   [x.strip() for x in os.environ["MAIL_TO"].split(",") if x.strip()],
    }


def _send(msg: EmailMessage, c: dict) -> None:
    ctx = ssl.create_default_context()
    if c["port"] == 465:
        with smtplib.SMTP_SSL(c["host"], 465, context=ctx, timeout=25) as s:
            s.login(c["user"], c["pw"]); s.send_message(msg)
    else:
        with smtplib.SMTP(c["host"], c["port"], timeout=25) as s:
            s.starttls(context=ctx); s.login(c["user"], c["pw"]); s.send_message(msg)


def send_alert(text: str) -> None:
    c = _conf()
    m = EmailMessage()
    m["Subject"] = "[竞价] 告警"
    m["From"] = formataddr(("竞价机器人", c["user"]))
    m["To"] = ", ".join(c["to"])
    m.set_content(text)
    _send(m, c)
    log.info("告警邮件已发送")


# ---------------------------------------------------------------------
_CSS = """
body{font:14px/1.6 -apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     color:#1a1a1a;margin:0;padding:16px;background:#fafafa}
h2{font-size:16px;margin:20px 0 8px;padding-left:8px;border-left:3px solid #c1440e}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px}
th{background:#f0f0f0;text-align:left;padding:7px 8px;font-weight:600;
   border-bottom:2px solid #ddd;white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid #eee;vertical-align:top}
.c{font-family:ui-monospace,Menlo,monospace;font-weight:600;white-space:nowrap}
.s{font-weight:700;color:#c1440e}
.up{color:#c62828}.dn{color:#2e7d32}
.rz{color:#8a6d00;font-size:12px}
.rn{color:#555;font-size:12px}
.meta{color:#888;font-size:12px;margin-bottom:14px}
.warn{background:#fff4e5;border-left:3px solid #e08600;padding:8px 10px;
      font-size:12px;margin:14px 0}
"""

_HDR = ["代码", "名称", "竞价价", "高开", "量能(量比)", "形态", "板块", "分",
        "理由 / 风险"]


def _rows_html(rows: list[dict], texts: dict) -> str:
    out = []
    for r in rows:
        t = texts.get(r["code"], {})
        shape = "抬升" if (r["monotonic"] and r["slope"] > 0) else (
                "走弱" if r["slope"] < 0 else "震荡")
        risk = " / ".join(r["risk_tags"]) if r["risk_tags"] else ""
        reason = t.get("reason", "")
        rtext = t.get("risk", "")
        cell = (f'<div class="rn">{reason}</div>' if reason else "") + \
               (f'<div class="rz">⚠ {rtext or risk}</div>'
                if (rtext or risk) else "")
        out.append(
            f'<tr><td class="c">{r["code"]}</td><td>{r["name"]}</td>'
            f'<td>{r["auc_price"]:.2f}</td>'
            f'<td class="up">+{r["gap_pct"]:.2f}%</td>'
            f'<td>{r["auc_ratio"]*100:.2f}% ({r.get("liangbi", 0):.1f})</td>'
            f'<td>{shape} {r["slope"]:+.1f}</td>'
            f'<td>{r["sector"]}'
            + (f'·{r["sector_members"]}' if r["sector_members"] >= 3 else "")
            + f'</td><td class="s">{r["score"]:.0f}</td>'
            f'<td>{cell}</td></tr>'
        )
    return "".join(out)


def build_html(date: str, result: dict, texts: dict, stage: str,
               notice: str = "", page_url: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in _HDR)
    n = len(result["A"]) + len(result["B"])
    parts = [f"<style>{_CSS}</style>",
             f'<div class="meta">{date} 集合竞价 · 采集于 09:25:10 · '
             f'共 {n} 只</div>']
    if notice:
        parts.append(f'<div class="warn"><b>{notice}</b></div>')
    if page_url:
        parts.append(f'<div class="meta">在线面板（可一键复制代码推给同花顺）：'
                     f'<a href="{page_url}">{page_url}</a></div>')
    for key, title in (("A", "竞价强弱榜"), ("B", "B组 · 低位首板预备")):
        if not result[key]:
            continue
        parts.append(f"<h2>{title}</h2><table><tr>{th}</tr>"
                     f"{_rows_html(result[key], texts)}</table>")
    if not result["A"] and not result["B"]:
        parts.append('<div class="warn">今日无标的通过筛选。'
                     '空仓也是一种仓位。</div>')
    parts.append('<div class="warn">本清单为量化筛选结果，非投资建议。'
                 '竞价数据采集于 09:25:10，开盘后走势可能与竞价背离。</div>')
    return "".join(parts)


def send_report(date: str, result: dict, texts: dict, cfg: dict,
                all_rows: list[dict] | None = None,
                attachments: list[Path] | None = None,
                stage: str = "清单", notice: str = "",
                page_url: str = "") -> None:
    c = _conf()
    m = EmailMessage()
    n = len(result["A"]) + len(result["B"])
    top = result["A"][0]["name"] if result["A"] else (
          result["B"][0]["name"] if result["B"] else "无")
    m["Subject"] = f"[竞价{stage}] {date} · {n}只 · 首位 {top}"
    m["From"] = formataddr(("竞价机器人", c["user"]))
    m["To"] = ", ".join(c["to"])

    m.set_content(f"{date} 竞价{stage}：{n} 只。请用 HTML 视图查看。"
                  + (f"\n在线面板：{page_url}" if page_url else ""))
    html = build_html(date, result, texts, stage, notice, page_url)
    m.add_alternative(html, subtype="html")

    for p in (attachments or []):
        data = Path(p).read_bytes()
        m.add_attachment(data, maintype="application", subtype="octet-stream",
                         filename=Path(p).name)

    _send(m, c)
    log.info("清单邮件已发送 (%s, %d 只, %d 附件)",
             stage, n, len(attachments or []))
