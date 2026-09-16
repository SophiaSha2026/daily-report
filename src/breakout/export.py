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
# 同目录的 truth（按板块的命中率）要能在单独 import export 时也找得到
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    (5, 6.7, 2.3, 30),
    (4, 13.0, 4.4, 54),
    (3, 14.3, 4.9, 98),
    (2, 15.7, 5.4, 216),
    (1, 12.6, 4.3, 783),
]
PERF = {"base": BASE, "window": "验证集 207 个交易日"}
# 上限主导的日子单独报一个数。STREAK_PERF 里那 12.6% 是验证集**全部** 783 个
# 名额的平均，而 176 个非空日里 154 天是门槛在决定清单（够格不到 10 只），
# 只有 22 天满员。生产 2026-09 那 8 天**全部**满员，拿 12.6% 给它背书偏高。
# 2026-09-16 从逐月滚动打分缓存（data/breakout/raw/wf_scores.parquet，
# STREAK_PERF 的同一份）实算：满 10 只的 22 天 220 个名额命中 11.36%，
# 没满的 154 天 563 个名额 13.14%，不设门槛直接取前 10 是 11.16%。
# 注：exp_window 还没把这两行写进 window_grid.json（要加 kind="W5c"），
# 所以这两个数暂时没法像 STREAK_PERF 那样逐位对账产物。
CAP_PERF = (11.4, 220, 22)        # 准确率%, 名额数, 天数
# STREAK_PERF / BASE 是在**这条规则**下测的（exp_window.py 的 W5 那几行）。
# daily.py 的 SCORE_MIN / CAP_A 会被 state/breakout/overrides.json 覆盖
# （学习会诊批准后就会写），规则一改这张成绩表就不适用了，必须在邮件里说清楚，
# 不能拿旧规则的成绩给新规则背书。
PERF_RULE = (97, 10)
# 少于这么多样本的档次在成绩表里灰掉：连续≥5 天只有 30 个名额，
# 6.7% 和 13% 的差别完全在抽样噪音里
SMALL_N = 50
BOARD_CN = {"main": "主板", "star": "科创", "chinext": "创业", "bj": "北交"}


def streak_perf(k: int) -> tuple[float, float]:
    """连续 k 天够格的历史准确率和倍数。"""
    for need, hit, lift, _n in STREAK_PERF:
        if k >= need:
            return hit, lift
    return STREAK_PERF[-1][1], STREAK_PERF[-1][2]


def _rule(meta: dict) -> tuple[int, int]:
    """这份清单是按哪条规则出的。run_meta 没带就用当前生效值（补发/回放）。"""
    smin, cap = meta.get("score_min"), meta.get("cap_a")
    if smin is None or cap is None:
        import daily as D
        smin = D.SCORE_MIN if smin is None else smin
        cap = D.CAP_A if cap is None else cap
    return int(smin), int(cap)


def _min_days() -> int:
    """生产剔次新的那条线。写死在文案里的话，改了常量邮件还会说「120 天」。"""
    try:
        import daily as D
        return int(D.MIN_HISTORY_DAYS)
    except Exception:  # noqa: BLE001
        return 120


def _wilson(hit: float, n: int) -> tuple[float, float]:
    """百分数命中率的 Wilson 95% 区间。唯一实现在 truth.wilson。"""
    try:
        import truth as T
        lo, hi = T.wilson(int(round(hit / 100.0 * n)), int(n))
        return 100 * lo, 100 * hi
    except Exception as e:  # noqa: BLE001
        log.info("区间算不出来（%s），表里就不印", e)
        return float("nan"), float("nan")


_BOARD_HIT: dict = {}


def board_hit_table() -> dict:
    """验证集里生产口径名额按板块的命中率（truth.validation_by_board）。

    不写死：它从逐月滚动缓存算并落 out_breakout/board_hit.json，和 STREAK_PERF
    同源，重训后自动跟着变。读不到就返回空表，调用方退回整体平均。
    进程内只取一次：面板带日期下拉时要渲染 30 天，每天一块都要查它。
    """
    if "t" not in _BOARD_HIT:
        try:
            import truth as T
            _BOARD_HIT["t"] = T.validation_by_board() or {}
        except Exception as e:  # noqa: BLE001
            log.info("按板块的命中率读不到（%s），期望退回整体平均", e)
            _BOARD_HIT["t"] = {}
    return _BOARD_HIT["t"]


def expected_for(a: pd.DataFrame) -> tuple[float, str, bool]:
    """本份清单按板块构成加权的期望命中率。返回 (期望%, 构成文字, 有没有兜底)。

    2026-09-16 会诊查出来的：邮件印的 12.6% 是验证集**总平均**，而验证集里
    科创板名额命中只有 2.7%（n=150，和随便买的 2.93% 一个水平）、主板 16.5%
    （n=520）、北交 8.0%（n=113），创业板一个名额都没有。当时 8 份清单 80 个
    名额里科创占 49，按构成加权只有 7.2%，总平均高估 1.75 倍。
    板块在验证集里没有样本（创业板）就用整体平均兜底并在文案里标注。
    """
    overall = STREAK_PERF[4][1]
    if not len(a) or "board" not in a.columns:
        return overall, "", True
    cnt = a["board"].astype(str).value_counts()
    comp = " / ".join(f"{BOARD_CN.get(b, b)} {int(n)}" for b, n in cnt.items())
    boards = (board_hit_table() or {}).get("boards") or {}
    if not boards:
        return overall, comp, True
    acc, tot, fallback = 0.0, 0, False
    for b, n in cnt.items():
        row = boards.get(str(b))
        if row and int(row.get("n", 0)) > 0:
            acc += 100 * float(row["hit"]) * int(n)
        else:
            acc += overall * int(n)
            fallback = True
        tot += int(n)
    return (acc / tot if tot else overall), comp, fallback


def disclaimer(score_min: int = PERF_RULE[0], cap: int = PERF_RULE[1]) -> str:
    """免责/读法说明。规则从 run_meta 来，不写死 —— 会诊把 SCORE_MIN 改成 96
    之后，这句话还说「≥ 97 够不到就不上」就是假的。

    措辞跟着数据走。实验 8 时连续 3 天 31% 对首日 18%，写的是「最强的信号」；
    实验 9、10 把偷看和未净化去掉之后，连续 2 天 15.7% 对首日 12.6%，3 天以上
    样本不到 100 只、数字和首日差不多。现在只说「略高、样本少」。
    """
    return (
        f"上榜条件是分数 ≥ {score_min} 且当天前 {cap} 名，够不到就不上，"
        f"所以<b>清单为空是正常的</b>。"
        f"「连续」指这只票连着几个交易日都够格。验证集上全部上榜的准确率 "
        f"{STREAK_PERF[4][1]:.1f}%（是随便买的 {STREAK_PERF[4][2]:.1f} 倍），"
        f"连续 2 天以上 {STREAK_PERF[3][1]:.1f}%，再往上样本太少不作数。"
        f"全市场随便买一只是 {BASE}%。"
        f"当天够格的超过 {cap} 只、清单被上限截断时，验证集上这类日子"
        f"（{CAP_PERF[2]} 天 {CAP_PERF[1]} 个名额）的准确率是 "
        f"{CAP_PERF[0]:.1f}%，比上面那个平均值低一点。"
        f"准确率的含义是「这只票未来 20 个交易日内最高价涨超 50%」。"
    )


# 兼容旧调用方（和自测）：按实验那条规则渲染的一份
DISCLAIMER = disclaimer()


def _rows_a(a: pd.DataFrame, score_min: int = PERF_RULE[0],
            stale: bool = False) -> str:
    """stale=True 表示这份清单的规则和 PERF_RULE 不是一条，准确率那格打星号。"""
    if not len(a):
        return ('<tr><td colspan="7" style="color:#8f9aa8;padding:18px;'
                f'text-align:center">今天没有够格的股票（没有一只到 {score_min} 分）。'
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
            f'<td>{hit:.1f}%{"*" if stale else ""}</td>'
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
            f'<td>{hit:.1f}%</td>'
            f'<td>{getattr(r, "close", 0):.2f}</td>'
            f'<td>{getattr(r, "rise", 0):+.1f}%</td>'
            f'<td class="up">{getattr(r, "drop", 0):+.1f}%</td></tr>')
    return "".join(out)


def _day_block(date: str, a: pd.DataFrame, b: pd.DataFrame, meta: dict) -> str:
    """某一天的抬头 + 清单 A + 清单 B。面板按日期切换时整块替换。"""
    smin, cap = _rule(meta)
    # rejected 为 None 表示不可知（补发/回放历史清单时），这段就不印
    rej = meta.get("rejected", 0)
    rej_txt = f" · 风险剔除 {rej} 只" if rej is not None else ""
    # 够格总数：清单永远是 cap 只，不印这个数就看不出当天是门槛在决定清单
    # 还是上限在决定。2026-09 生产连着 8 天满 10 只，产物上完全看不出来（S7）。
    # 2026-09-16 之前落盘的清单没有这个键，那时候就不印
    nq = meta.get("n_qualified")
    nq_txt = ""
    if nq is not None:
        nq_txt = f" · ≥{smin} 分共 {int(nq)} 只"
        if int(nq) > cap:
            nq_txt += f"（已按前 {cap} 名截断）"
    md = meta.get("model_date")
    md_txt = f" · 模型训练于 {md}" if md else ""
    # 期望命中率要按**本份清单**的板块构成加权：清单里科创占多少、主板占多少，
    # 各板块在验证集上的成绩差 6 倍，总平均对某几种构成会高估近一倍（会诊-1）
    exp, comp, fallback = expected_for(a)
    exp_txt = ""
    if comp:
        tail = "（其中有板块在验证集里没有名额，用整体平均代入）" if fallback else ""
        exp_txt = (f'<div class="sub">本份构成 {comp}，按构成的期望命中率 '
                   f'{exp:.1f}%{tail}；验证集整体 {STREAK_PERF[4][1]:.1f}%，'
                   f'同期全市场基准 {BASE}%。</div>')
    head = (f'<h1>起涨预测 · {date}</h1>'
            f'<div class="sub">清单 A {len(a)} 只，清单 B {len(b)} 只'
            f'{nq_txt}{rej_txt}{md_txt}</div>{exp_txt}')
    n3 = int((a["streak"] >= 3).sum()) if len(a) and "streak" in a else 0
    ta = (f'<h1 style="margin-top:18px">清单 A · 接近起涨</h1>'
          f'<div class="sub">按连续够格天数排序，同样连续再按预测值。'
          f'今天连续 3 天以上的有 {n3} 只。</div>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th><th>分数</th>'
          f'<th>连续</th><th>历史准确率</th><th>现价</th></tr>'
          f'{_rows_a(a, smin, (smin, cap) != PERF_RULE)}</table>')
    tb = (f'<h1 style="margin-top:22px">清单 B · 见顶信号</h1>'
          f'<div class="sub">上过清单 A、之后涨过一波、现在见顶回落的股票。'
          f'分数是它在清单 A 上拿过的最高分，准确率是它当初那一档的成绩，'
          f'不是「见顶判断有多准」。</div>'
          f'<table><tr><th>#</th><th>代码</th><th>名称</th><th>分数</th>'
          f'<th>上榜天数</th><th>历史准确率</th><th>现价</th>'
          f'<th>进榜后涨幅</th><th>距高点</th></tr>{_rows_b(b)}</table>')
    return head + ta + tb


def _date_picker(date: str, history: list[dict],
                 today_block: str) -> tuple[str, str]:
    """面板顶部的日期下拉。返回 (下拉 HTML, 切换用的 JS，不带 script 标签)。

    历史每天那块 HTML 提前在 Python 里渲染好嵌进页里，切换时前端只换
    innerHTML，不用把 Python 和 JS 各写一套渲染。当天那块用调用方传进来的
    （它的 meta 有模型日期和风险剔除数，回放出来的没有）。只进面板，不进邮件。
    """
    import json as _j
    blocks = {h["date"]: _day_block(h["date"], h["a"], h["b"], h["meta"])
              for h in history}
    blocks[date] = today_block
    dates = sorted(blocks)
    opts = "".join(
        f'<option value="{d}"{" selected" if d == date else ""}>{d}'
        f'{"（最新）" if d == dates[-1] else ""}</option>' for d in reversed(dates))
    sel = (f'<div class="bar" style="align-items:center;gap:10px">'
           f'<span class="sub" style="margin:0">看哪一天</span>'
           f'<select id="daysel" style="background:#2a2f38;color:#e6e6e6;'
           f'border:1px solid #3a4149;border-radius:5px;padding:5px 8px;'
           f'font-size:13px">{opts}</select>'
           f'<span class="sub" style="margin:0">共 {len(dates)} 个交易日。'
           f'旧日子的清单 B 是按那天为止的价格回放的，和当天发的邮件一致</span>'
           f'</div>')
    js = ("(function(){var B=" + _j.dumps(blocks, ensure_ascii=False) + ";"
          "var s=document.getElementById('daysel');"
          "s.addEventListener('change',function(){var d=s.value;"
          "if(B[d]){document.getElementById('day').innerHTML=B[d];}});})();")
    return sel, js


def _body(date: str, a: pd.DataFrame, b: pd.DataFrame, meta: dict,
          for_panel: bool, history: list[dict] | None = None) -> str:
    smin, cap = _rule(meta)
    day = _day_block(date, a, b, meta)
    tip = (f'<div class="tip" style="margin:0 0 14px;padding:10px 12px;'
           f'background:#1e2229;border-radius:5px">{disclaimer(smin, cap)}</div>')
    # 分数对照表。用户拿到清单第一个问题就是「92 分和 85 分差多少」，
    # 不给这张表的话，分数就只是个没有意义的数字。
    # 每行带 Wilson 95% 区间、样本少于 SMALL_N 的行灰掉：连续≥5 天那档只有
    # 30 个名额，6.7% 的区间宽到 2%~22%，不加区间会被当成「连续 5 天反而差」。
    rows = ""
    for k, hit, lift, n in STREAK_PERF:
        lo, hi = _wilson(hit, n)
        ci = "—" if lo != lo else f"{lo:.1f}~{hi:.1f}%"
        dim = ' style="color:#7c8794"' if n < SMALL_N else ""
        rows += (
            f'<tr{dim}><td>{("连续 " + str(k) + " 天及以上") if k > 1 else "全部上榜的"}'
            f'</td><td class="sc">{hit:.1f}%</td><td>{ci}</td>'
            f'<td>{lift:.1f} 倍</td><td>{n} 只</td></tr>')
    rows += (f'<tr><td>随便买</td><td class="sc">{BASE}%</td><td>—</td>'
             f'<td>1.0 倍</td><td>全市场</td></tr>')
    # 规则和实验那条不一样时（会诊批准改了 SCORE_MIN / CAP_A）必须说明白：
    # 这张表是旧规则的成绩，拿它给新规则背书正是本模块开头禁止的事
    if (smin, cap) == PERF_RULE:
        rule_txt = (f'口径和每天发的清单一致：剔掉上市不足 {_min_days()} 个交易日的票后，'
                    f'≥{smin} 分按预测值取前 {cap}，连续按上榜天数算，'
                    f'训练集做了 20 日净化。风险剔除（ST / 减持 / 解禁）'
                    f'在历史上回放不了，这张表里不含。')
    else:
        rule_txt = (f'下面是按 ≥{PERF_RULE[0]} 分前 {PERF_RULE[1]} 名的<b>旧规则</b>'
                    f'实测的成绩；当前清单规则是 ≥{smin} 分前 {cap} 名，'
                    f'尚未单独实测，各行「历史准确率」只作参考。')
    tc = (f'<h1 style="margin-top:22px">连续天数怎么看</h1>'
          f'<div class="sub">下面是 {PERF["window"]}的实测准确率。{rule_txt}</div>'
          f'<table><tr><th>连续天数</th><th>涨超 50% 的比例</th>'
          f'<th>95% 区间</th><th>相对随便买</th><th>样本</th></tr>{rows}</table>'
          f'<div class="tip">连续 2 天以上（{STREAK_PERF[3][1]:.1f}%，'
          f'{STREAK_PERF[3][3]} 只）比全部上榜（{STREAK_PERF[4][1]:.1f}%）略高；'
          f'3 天以上各档样本只有 {STREAK_PERF[2][3]}、{STREAK_PERF[1][3]}、'
          f'{STREAK_PERF[0][3]} 只，区间互相盖住，数字上下抖动是样本少'
          f'，不是信号。带 🔥 的是连续 3 天以上，只是提示它已经在榜上待了几天。'
          f'</div>')

    foot = ('<div class="tip">清单 B 的三个条件：进清单 A 满 5 个交易日、'
            '进入后涨过 20%、现在从那个高点回落 8%~20% 且高点在最近 10 天内。'
            '目前是规则判定，不是训练出来的模型 —— 等清单 A 积累出足够样本'
            '后会换成模型。</div>')
    # 自动刷新脚本只放面板，不放邮件（邮件客户端会剥掉 script，放了也没用）。
    # 三件事缺一不可，缺了就会像 2026-09-13 那次一样把整段 JS 当正文印出来：
    #   1. 包在 <script> 里
    #   2. __STAMPFILE__ 换成本面板自己的 stamp 文件名
    #   3. __STAMP__ / __DATE__ 换成当前日期，否则脚本一跑就判定自己过期
    picker, pjs = "", ""
    if for_panel and history:
        picker, pjs = _date_picker(date, history, day)
        day = f'<div id="day">{day}</div>'
    if for_panel:
        js = ("<script>" + REFRESH_JS
              .replace("__STAMPFILE__", "stamp-breakout.txt")
              .replace("__STAMP__", date)
              .replace("__DATE__", date) + pjs + "</script>")
        stale = '<div id="stale"></div>'
    else:
        js, stale = "", ""
    return stale + picker + tip + day + tc + foot + js


def write_panel(a: pd.DataFrame, b: pd.DataFrame, meta: dict,
                out_dir: Path, date: str,
                history: list[dict] | None = None) -> Path:
    """history 给了就带日期下拉（最近 N 天的清单都嵌在页里，切换不发请求）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    html = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>起涨预测 {date}</title><style>{PANEL_CSS}</style></head>'
            f'<body>{_body(date, a, b, meta, True, history)}</body></html>')
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
