"""邮件发送。SMTP over STARTTLS/SSL，凭证来自环境变量（本机 tools/local.env，云端 GitHub Secrets）。

两条晚间线（起涨预测、长期调整突破）和云端提醒共用 _conf / _send / _CSS / send_alert。
"""
from __future__ import annotations

import os
import ssl
import smtplib
import logging
from email.message import EmailMessage
from email.utils import formataddr

log = logging.getLogger(__name__)


def skip_mail() -> bool:
    """SKIP_MAIL 的唯一判定：只认 "1" / "true"（不分大小写）。

    以前四处各写各的：三条业务线用真值判断（`if os.environ.get("SKIP_MAIL")`，
    于是 "0" 也算跳过），learn/report.py 用 `== "1"`。有人往 tools/local.env
    里写一行 SKIP_MAIL=0 想「明确开启发信」，早盘/形态/起涨预测三条线会全部
    静音，而学习和会诊邮件照发；写 true 想全部静音则正好反过来。
    workflows 里的 `== '1' && '1' || ''` 传的值本来就落在这个语义里。
    """
    return os.environ.get("SKIP_MAIL", "").strip().lower() in ("1", "true")


def _conf() -> dict:
    """凭证来自环境变量。本机跑的入口不一定加载过 tools/local.env（控制台的
    按钮就是直接跑脚本），所以在这个唯一的入口兜一次底；云端没有这个文件，
    而且已存在的环境变量优先，GitHub Secrets 不会被盖掉。"""
    try:
        import localenv
        localenv.load()
    except Exception:  # noqa: BLE001
        pass
    return {
        "host": os.environ["SMTP_HOST"],
        # `.get(k, default)` 的默认值只在**键不存在**时生效。workflow 里写的是
        # `SMTP_PORT: ${{ secrets.SMTP_PORT }}`，secret 没配时 GitHub 把变量设成
        # 空串，int("") 直接 ValueError；_conf 又是所有发信路径的唯一入口，
        # 正文信和兜底告警信会在同一个地方炸，一封都发不出去。
        # 空白串同样落回默认：local.env 里写成 `SMTP_PORT= ` 也不该炸。
        "port": int((os.environ.get("SMTP_PORT") or "").strip() or 587),
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


def send_alert(text: str, subject: str = "[A股流水线] 告警") -> None:
    c = _conf()
    m = EmailMessage()
    m["Subject"] = subject
    m["From"] = formataddr(("A股流水线", c["user"]))
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


# 竞价清单那套表格和 send_report（早盘选股专用）2026-09-27 随早盘系统归档，
# 原样在 archive/morning/src/mailer.py。
