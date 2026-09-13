"""
起涨预测的面板和邮件。样式复用 mailer 和 ths_export，和另外两条线一致。

分数是排名不是概率
------------------
每封邮件里都必须印这句：封存数据实测，每天 10 只里约 1.4 只会在未来
20 个交易日涨超 50%，是全市场平均（2.9%）的 4.96 倍。

不印的话「92 分」会被读成「92% 会涨」，那是在骗人。
"""
from __future__ import annotations

import logging
import sys
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mailer import _conf, _send          # noqa: E402
from ths_export import PANEL_CSS, REFRESH_JS  # noqa: E402

log = logging.getLogger("breakout.export")

# 封存数据（2026-01..08）上的实测成绩。改模型后要同步改这里，
# 否则邮件会拿旧成绩给新模型背书。
PERF = {"hit": 14.43, "base": 2.91, "lift": 4.96, "top": 10,
        "window": "2026-01 至 2026-08"}

# 分数段 -> 实际命中率。验证集 10 个月、111 万个样本逐月滚动测出来的，
# 严格单调上升（分数越高越准），没有一档倒挂。
# 99 分那档只有 213 个样本，数字不稳，所以合并进 95+ 一起显示。
SCORE_TABLE = [
    ("95 分以上", 8.3, 2.8),
    ("90~94 分", 6.2, 2.1),
    ("80~89 分", 4.5, 1.5),
    ("70~79 分", 3.8, 1.3),
    ("随便买", 2.9, 1.0),
]

DISCLAIMER = (
    f"分数是 0~100 的<b>排名</b>，不是上涨概率。"
    f"封存数据（{PERF['window']}，模型训练时从未见过）实测："
    f"每天 {PERF['top']} 只里约 {PERF['top'] * PERF['hit'] / 100:.1f} 只"
    f"会在未来 20 个交易日内最高价涨超 50%，"
    f"是全市场平均（{PERF['base']}%）的 {PERF['lift']} 倍。"
    f"所以这份清单的意思是「这批票里出黑马的密度比市场高 {PERF['lift']:.0f} 倍」，"
    f"不是「选出来的会涨」。"
)


def _rows_a(a: pd.DataFrame) -> str:
    if not len(a):
        return ('<tr><td colspan="5" style="color:#8f9aa8;padding:18px;'
                'text-align:center">今天没有符合条件的股票。'
                '这是正常的，不是故障。</td></tr>')
    out = []
    for i, r in enumerate(a.itertuples(), 1):
        code = str(r.code)
        out.append(
            f'<tr><td>{i}</td>'
            f'<td class="code">{code}</td>'
            f'<td>{getattr(r, "name", "") or ""}</td>'
            f'<td class="sc">{r.score:.0f}</td>'
            f'<td>{getattr(r, "close", 0):.2f}</td></tr>')
    return "".join(out)


def _rows_b(b: pd.DataFrame) -> str:
    if not len(b):
        return ('<tr><td colspan="5" style="color:#8f9aa8;padding:18px;'
                'text-align:center">没有触发见顶信号的股票。</td></tr>')
    out = []
    for i, r in enumerate(b.itertuples(), 1):
        out.append(
            f'<tr><td>{i}</td>'
            f'<td class="code">{r.code}</td>'
            f'<td>{getattr(r, "name", "") or ""}</td>'
            f'<td class="sc">{getattr(r, "best", 0):.0f}</td>'
            f'<td class="up">{getattr(r, "drop", 0):+.1f}%</td></tr>')
    return "".join(out)


def _body(date: str, a: pd.DataFrame, b: pd.DataFrame, meta: dict,
          for_panel: bool) -> str:
    head = (f'<h1>起涨预测 · {date}</h1>'
            f'<div class="sub">清单 A {len(a)} 只，清单 B {len(b)} 只 · '
            f'风险剔除 {meta.get("rejected", 0)} 只 · '
            f'模型训练于 {meta.get("model_date", "?")}</div>')
    tip = (f'<div class="tip" style="margin:0 0 14px;padding:10px 12px;'
           f'background:#1e2229;border-radius:5px">{DISCLAIMER}</div>')
    ta = (f'<h1 style="margin-top:18px">清单 A · 接近起涨</h1>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th>'
          f'<th>分数</th><th>现价</th></tr>{_rows_a(a)}</table>')
    tb = (f'<h1 style="margin-top:22px">清单 B · 见顶信号</h1>'
          f'<div class="sub">曾在清单 A 拿过 90 分以上、之后涨过一波、现在见顶回落的股票</div>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th>'
          f'<th>曾用最高分</th><th>距高点</th></tr>{_rows_b(b)}</table>')
    # 分数对照表。用户拿到清单第一个问题就是「92 分和 85 分差多少」，
    # 不给这张表的话，分数就只是个没有意义的数字。
    rows = "".join(
        f'<tr><td>{lbl}</td><td class="sc">{hit:.1f}%</td>'
        f'<td>{lift:.1f} 倍</td></tr>'
        for lbl, hit, lift in SCORE_TABLE)
    tc = (f'<h1 style="margin-top:22px">分数怎么看</h1>'
          f'<div class="sub">同一个分数段的票，历史上有多少在一个月内涨超 50%。'
          f'验证了 111 万个样本，分数越高越准，没有例外。</div>'
          f'<table><tr><th>分数</th><th>涨超 50% 的比例</th>'
          f'<th>相对随便买</th></tr>{rows}</table>'
          f'<div class="tip">清单 A 取的是当天全市场<b>前 10 名</b>，'
          f'比「90 分以上」这个绝对标准更严，所以清单的实际命中率'
          f'（{PERF["hit"]}%）高于表里 90 分那一档。</div>')

    foot = ('<div class="tip">清单 B 的三个条件：进清单 A 满 5 个交易日、'
            '进入后涨过 20%、现在从那个高点回落 8%~20% 且高点在最近 10 天内。'
            '目前是规则判定，不是训练出来的模型 —— 等清单 A 积累出足够样本'
            '后会换成模型。</div>')
    # 自动刷新脚本只放面板，不放邮件（邮件客户端会剥掉 script，放了也没用）。
    # 三件事缺一不可，缺了就会像 2026-09-13 那次一样把整段 JS 当正文印出来：
    #   1. 包在 <script> 里
    #   2. __STAMPFILE__ 换成本面板自己的 stamp 文件名
    #   3. __STAMP__ / __DATE__ 换成当前日期，否则脚本一跑就判定自己过期
    if for_panel:
        js = ("<script>" + REFRESH_JS
              .replace("__STAMPFILE__", "stamp-breakout.txt")
              .replace("__STAMP__", date)
              .replace("__DATE__", date) + "</script>")
        stale = '<div id="stale"></div>'
    else:
        js, stale = "", ""
    return stale + head + tip + ta + tb + tc + foot + js


def write_panel(a: pd.DataFrame, b: pd.DataFrame, meta: dict,
                out_dir: Path, date: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>起涨预测 {date}</title><style>{PANEL_CSS}</style></head>'
            f'<body>{_body(date, a, b, meta, True)}</body></html>')
    p = out_dir / "panel.html"
    p.write_text(html, encoding="utf-8")
    (out_dir / "stamp.txt").write_text(date, encoding="utf-8")
    log.info("面板 -> %s", p)
    return p


def send_mail(date: str, a: pd.DataFrame, b: pd.DataFrame,
              meta: dict) -> None:
    c = _conf()
    m = EmailMessage()
    m["Subject"] = f"起涨预测 {date}：A {len(a)} 只 / B {len(b)} 只"
    m["From"] = c["user"]
    m["To"] = ", ".join(c["to"])
    m.set_content("请用支持 HTML 的客户端查看。")
    body = _body(date, a, b, meta, False)
    m.add_alternative(
        f'<html><head><style>{PANEL_CSS}</style></head>'
        f'<body style="background:#14161a">{body}</body></html>',
        subtype="html")
    _send(m, c)
    log.info("邮件已发出：%s", m["Subject"])
