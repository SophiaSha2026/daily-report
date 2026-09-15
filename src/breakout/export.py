"""
起涨预测的面板和邮件。样式复用 mailer 和 ths_export，和另外两条线一致。

分数是排名不是概率
------------------
每封邮件里都必须印实测准确率（STREAK_PERF：上榜的票里有多少会在未来
20 个交易日涨超 50%，按连续上榜天数分档）和全市场基准。
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

# 实测成绩。改规则或改模型后必须同步改这里，否则邮件会拿旧成绩
# 给新规则背书。数字来源：docs/breakout_log.md 实验 9（验证集 207 个交易日），
# 就是 out_breakout/window_grid.json 里 kind=W5「生产口径」那几行，
# selftest_breakout 钉住两边一致。
#
# 清单 A 的规则是「≥97 分且当天前 10 名才上榜」，而**连续上榜的天数**是
# 清单里最强的单一信号，所以准确率按连续天数分档报，不报一个笼统的平均值。
#
# 实验 9 之前的那组（18.0 / 25.7 / 31.3）里有股东人数按报告期对齐的偷看
# （历史教训 25），修掉之后整体降了：首日 15.7%、连续 3 天 22.0%。
# 基准从 3.55% 改成 2.93%：前者混进了 2023~2024 的训练月份，和验证集
# 十个月的成绩不是同一段时间，倍数被压低了。
BASE = 2.93                       # 全市场基准：验证集 10 个月里随便买一只涨超 50% 的比例
STREAK_PERF = [
    # 连续天数下限, 准确率%, 相对随便买的倍数, 样本数
    (5, 34.0, 11.6, 47),
    (4, 25.0, 8.5, 84),
    (3, 22.0, 7.5, 150),
    (2, 18.9, 6.4, 286),
    (1, 15.7, 5.4, 918),
]
PERF = {"base": BASE, "window": "验证集 207 个交易日"}


def streak_perf(k: int) -> tuple[float, float]:
    """连续 k 天够格的历史准确率和倍数。"""
    for need, hit, lift, _n in STREAK_PERF:
        if k >= need:
            return hit, lift
    return STREAK_PERF[-1][1], STREAK_PERF[-1][2]

# 分数段 -> 实际命中率。验证集 10 个月、111 万个样本逐月滚动测出来的
# （exp_calib.py -> out_breakout/score_calibration.json），严格单调上升
# （分数越高越准），没有一档倒挂。99 分那档只有 275 个样本，数字不稳，
# 所以合并进 95+ 一起显示。
SCORE_TABLE = [
    ("95 分以上", 8.1, 2.8),
    ("90~94 分", 6.0, 2.0),
    ("80~89 分", 4.6, 1.6),
    ("70~79 分", 3.9, 1.3),
    ("随便买", 2.9, 1.0),
]

DISCLAIMER = (
    f"上榜条件是分数 ≥ 97，够不到就不上，所以<b>清单为空是正常的</b>。"
    f"「连续」指这只票连着几个交易日都够格，这是清单里最强的信号："
    f"连续 3 天的历史准确率 {STREAK_PERF[2][1]}%（是随便买的 "
    f"{STREAK_PERF[2][2]:.0f} 倍），连续 2 天 {STREAK_PERF[3][1]}%，"
    f"首日 {STREAK_PERF[4][1]}%。全市场随便买一只是 {BASE}%。"
    f"准确率的含义是「这只票未来 20 个交易日内最高价涨超 50%」。"
)


def _rows_a(a: pd.DataFrame) -> str:
    if not len(a):
        return ('<tr><td colspan="7" style="color:#8f9aa8;padding:18px;'
                'text-align:center">今天没有够格的股票（没有一只到 97 分）。'
                '这是正常的，不是故障。</td></tr>')
    out = []
    for i, r in enumerate(a.itertuples(), 1):
        k = int(getattr(r, "streak", 1) or 1)
        hit, lift = streak_perf(k)
        mark = f"{k} 天" + ("　🔥" if k >= 3 else "")
        out.append(
            f'<tr><td>{i}</td>'
            f'<td class="code">{r.code}</td>'
            f'<td>{getattr(r, "name", "") or ""}</td>'
            f'<td class="sc">{r.score:.0f}</td>'
            f'<td class="up">{mark}</td>'
            f'<td>{hit:.0f}%</td>'
            f'<td>{getattr(r, "close", 0):.2f}</td></tr>')
    return "".join(out)


def _rows_b(b: pd.DataFrame) -> str:
    """清单 B 的列和清单 A 一致：代码 / 名称 / 分数 / 上榜天数 /
    历史准确率 / 现价，末尾多两列「进榜后涨幅」和「距高点」。

    这里的准确率是它<b>当初在清单 A 上</b>那一档的成绩，不是「见顶判断
    有多准」：清单 B 现在是规则判定，还没有实测成绩。说明写在表下面。
    """
    if not len(b):
        return ('<tr><td colspan="9" style="color:#8f9aa8;padding:18px;'
                'text-align:center">没有触发见顶信号的股票。</td></tr>')
    out = []
    for i, r in enumerate(b.itertuples(), 1):
        k = int(getattr(r, "streak", 1) or 1)
        hit, _lift = streak_perf(k)
        days = int(getattr(r, "days", 0) or 0)
        out.append(
            f'<tr><td>{i}</td>'
            f'<td class="code">{r.code}</td>'
            f'<td>{getattr(r, "name", "") or ""}</td>'
            f'<td class="sc">{getattr(r, "best", 0):.0f}</td>'
            f'<td>{days} 天</td>'
            f'<td>{hit:.0f}%</td>'
            f'<td>{getattr(r, "close", 0):.2f}</td>'
            f'<td>{getattr(r, "rise", 0):+.1f}%</td>'
            f'<td class="up">{getattr(r, "drop", 0):+.1f}%</td></tr>')
    return "".join(out)


def _body(date: str, a: pd.DataFrame, b: pd.DataFrame, meta: dict,
          for_panel: bool) -> str:
    # rejected 为 None 表示不可知（补发历史清单时），这段就不印
    rej = meta.get("rejected", 0)
    rej_txt = f"风险剔除 {rej} 只 · " if rej is not None else ""
    head = (f'<h1>起涨预测 · {date}</h1>'
            f'<div class="sub">清单 A {len(a)} 只，清单 B {len(b)} 只 · '
            f'{rej_txt}'
            f'模型训练于 {meta.get("model_date", "?")}</div>')
    tip = (f'<div class="tip" style="margin:0 0 14px;padding:10px 12px;'
           f'background:#1e2229;border-radius:5px">{DISCLAIMER}</div>')
    n3 = int((a["streak"] >= 3).sum()) if len(a) and "streak" in a else 0
    ta = (f'<h1 style="margin-top:18px">清单 A · 接近起涨</h1>'
          f'<div class="sub">按连续够格天数排序，连续越久越可靠。'
          f'今天连续 3 天以上的有 {n3} 只。</div>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th><th>分数</th>'
          f'<th>连续</th><th>历史准确率</th><th>现价</th></tr>'
          f'{_rows_a(a)}</table>')
    tb = (f'<h1 style="margin-top:22px">清单 B · 见顶信号</h1>'
          f'<div class="sub">上过清单 A、之后涨过一波、现在见顶回落的股票。'
          f'分数是它在清单 A 上拿过的最高分，准确率是它当初那一档的成绩，'
          f'不是「见顶判断有多准」。</div>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th><th>分数</th>'
          f'<th>上榜天数</th><th>历史准确率</th><th>现价</th>'
          f'<th>进榜后涨幅</th><th>距高点</th></tr>{_rows_b(b)}</table>')
    # 分数对照表。用户拿到清单第一个问题就是「92 分和 85 分差多少」，
    # 不给这张表的话，分数就只是个没有意义的数字。
    rows = "".join(
        f'<tr><td>{("连续 " + str(k) + " 天及以上") if k > 1 else "全部上榜的"}'
        f'</td><td class="sc">{hit:.1f}%</td><td>{lift:.1f} 倍</td>'
        f'<td>{n} 只</td></tr>'
        for k, hit, lift, n in STREAK_PERF)
    rows += (f'<tr><td>随便买</td><td class="sc">{BASE}%</td>'
             f'<td>1.0 倍</td><td>全市场</td></tr>')
    tc = (f'<h1 style="margin-top:22px">连续天数怎么看</h1>'
          f'<div class="sub">同样是 97 分以上，连着够格的天数越多越可靠。'
          f'下面是 {PERF["window"]}的实测准确率，口径和每天发的清单完全一致'
          f'（≥97 分按预测值取前 10）。</div>'
          f'<table><tr><th>连续天数</th><th>涨超 50% 的比例</th>'
          f'<th>相对随便买</th><th>样本</th></tr>{rows}</table>'
          f'<div class="tip">光有高分没用：97 分以上的整体只有 '
          f'{STREAK_PERF[4][1]}%，连续三天的能到 {STREAK_PERF[2][1]}%。'
          f'所以清单按连续天数排序，带 🔥 的是连续 3 天以上。'
          f'连续 4 天（{STREAK_PERF[1][1]}%）和 5 天（{STREAK_PERF[0][1]}%）'
          f'样本只有 {STREAK_PERF[1][3]} 和 {STREAK_PERF[0][3]} 只，'
          f'和 3 天那档的差距在误差范围内，别当成又高一截。</div>')

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
              meta: dict, tag: str = "") -> None:
    """tag 给补发用：主题上标出来，免得和当天那封混在一起。"""
    c = _conf()
    m = EmailMessage()
    pre = f"[{tag}] " if tag else ""
    m["Subject"] = f"{pre}起涨预测 {date}：A {len(a)} 只 / B {len(b)} 只"
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
