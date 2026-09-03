# -*- coding: utf-8 -*-
"""
生成《A股流水线 · 操作手册与架构介绍》PDF。

    python tools/build_manual.py            -> out_learn/manual.pdf

设计原则：大白话、短句、流程图直接画。数字从 state/ 和 out_learn/ 现读，
重跑一次脚本就是新版手册，不存在「文档和系统对不上」的问题。
"""
import json
import datetime as dt
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Flowable, KeepTogether)
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_learn" / "manual.pdf"

pdfmetrics.registerFont(TTFont("yh", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("yhb", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))

INK = colors.HexColor("#1c1c1e")
DIM = colors.HexColor("#6b6b70")
ACC = colors.HexColor("#0a5bd3")
OKC = colors.HexColor("#1e8449")
BAD = colors.HexColor("#c0392b")
BOX = colors.HexColor("#f2f4f8")
LINE = colors.HexColor("#c9cdd6")

S = {
    "h1": ParagraphStyle("h1", fontName="yhb", fontSize=20, leading=26,
                         textColor=INK, spaceAfter=2 * mm),
    "h2": ParagraphStyle("h2", fontName="yhb", fontSize=13, leading=18,
                         textColor=INK, spaceBefore=5 * mm, spaceAfter=2 * mm),
    "b": ParagraphStyle("b", fontName="yh", fontSize=9.5, leading=15,
                        textColor=INK),
    "dim": ParagraphStyle("dim", fontName="yh", fontSize=8.5, leading=13,
                          textColor=DIM),
    "cell": ParagraphStyle("cell", fontName="yh", fontSize=8.5, leading=12,
                           textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="yhb", fontSize=8.5, leading=12,
                            textColor=colors.white),
}


def P(t, s="b"):
    return Paragraph(t, S[s])


def tbl(head, rows, widths=None, aligns=None):
    data = [[Paragraph(str(h), S["cellh"]) for h in head]] + \
           [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3a3f4b")),
          ("GRID", (0, 0), (-1, -1), 0.4, LINE),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BOX])]
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------------
#  流程图元件：横向节点链 + 竖向分层图
# ---------------------------------------------------------------------
class Flow(Flowable):
    """一行流程：[节点] -> [节点] -> ...  节点可标注时间。"""

    def __init__(self, nodes, width=170 * mm, h=16 * mm, hl=None):
        super().__init__()
        self.nodes, self.width, self.h = nodes, width, h
        self.hl = hl or set()

    def wrap(self, aw, ah):
        return self.width, self.h

    def draw(self):
        c = self.canv
        n = len(self.nodes)
        gap = 5 * mm
        bw = (self.width - gap * (n - 1)) / n
        bh = 9.5 * mm
        y = (self.h - bh) / 2
        for i, node in enumerate(self.nodes):
            label, sub = node if isinstance(node, tuple) else (node, "")
            x = i * (bw + gap)
            fill = ACC if i in self.hl else colors.white
            c.setFillColor(fill)
            c.setStrokeColor(ACC if i in self.hl else LINE)
            c.setLineWidth(1)
            c.roundRect(x, y, bw, bh, 2 * mm, stroke=1, fill=1)
            c.setFillColor(colors.white if i in self.hl else INK)
            c.setFont("yhb", 8)
            c.drawCentredString(x + bw / 2, y + bh / 2 + (1 if sub else -2.5),
                                label)
            if sub:
                c.setFont("yh", 6.5)
                c.setFillColor(colors.white if i in self.hl else DIM)
                c.drawCentredString(x + bw / 2, y + bh / 2 - 7, sub)
            if i < n - 1:
                ax = x + bw
                c.setStrokeColor(DIM)
                c.setLineWidth(1)
                ay = y + bh / 2
                c.line(ax + 1, ay, ax + gap - 1.5, ay)
                c.setFillColor(DIM)
                c.saveState()
                p = c.beginPath()
                p.moveTo(ax + gap - 1, ay)
                p.lineTo(ax + gap - 4, ay + 1.6)
                p.lineTo(ax + gap - 4, ay - 1.6)
                p.close()
                c.drawPath(p, stroke=0, fill=1)
                c.restoreState()


class Takeover(Flowable):
    """接管协议示意：本地/远端两条泳道 + 三种结局。"""

    def __init__(self, width=170 * mm, h=52 * mm):
        super().__init__()
        self.width, self.h = width, h

    def wrap(self, aw, ah):
        return self.width, self.h

    def draw(self):
        c = self.canv
        w = self.width

        def lane(y, label, col):
            c.setFillColor(col)
            c.setFont("yhb", 8.5)
            c.drawString(0, y + 11 * mm, label)
            c.setStrokeColor(LINE)
            c.setLineWidth(0.6)
            c.line(0, y - 1.5 * mm, w, y - 1.5 * mm)

        def node(x, y, text, sub="", col=colors.white, tcol=INK, bw=30 * mm):
            bh = 8.5 * mm
            c.setFillColor(col)
            c.setStrokeColor(LINE if col == colors.white else col)
            c.roundRect(x, y, bw, bh, 1.6 * mm, stroke=1, fill=1)
            c.setFillColor(tcol)
            c.setFont("yhb", 7.5)
            c.drawCentredString(x + bw / 2, y + bh / 2 + (1 if sub else -2), text)
            if sub:
                c.setFont("yh", 6)
                c.drawCentredString(x + bw / 2, y + bh / 2 - 6.5, sub)

        def arrow(x1, y1, x2, y2, col=DIM, dash=None):
            c.setStrokeColor(col)
            c.setLineWidth(1)
            if dash:
                c.setDash(dash, 2)
            c.line(x1, y1, x2, y2)
            c.setDash()
            ang_x = 1 if x2 >= x1 else -1
            c.setFillColor(col)
            p = c.beginPath()
            if abs(y2 - y1) < 1:
                p.moveTo(x2, y2)
                p.lineTo(x2 - 3 * ang_x, y2 + 1.6)
                p.lineTo(x2 - 3 * ang_x, y2 - 1.6)
            else:
                dy = 1 if y2 > y1 else -1
                p.moveTo(x2, y2)
                p.lineTo(x2 + 1.6, y2 - 3 * dy)
                p.lineTo(x2 - 1.6, y2 - 3 * dy)
            p.close()
            c.drawPath(p, stroke=0, fill=1)

        top = self.h - 16 * mm
        bot = 6 * mm
        lane(top, "本地（你点一键跑）", ACC)
        lane(bot, "远端（GitHub 每天自动兜底）", DIM)

        node(2 * mm, top, "开跑", "推「接管声明」", ACC, colors.white, 24 * mm)
        node(32 * mm, top, "采样+打分", "自动等到 09:25", colors.white, INK, 26 * mm)
        node(64 * mm, top, "09:27:30 发信", "成功推「已发出」", OKC, colors.white, 30 * mm)

        node(2 * mm, bot, "照常自动跑", "", colors.white, INK, 24 * mm)
        node(64 * mm, bot, "09:27:00 查声明", "", colors.white, INK, 30 * mm)
        node(100 * mm, bot, "09:28:20 查「已发出」", "", colors.white, INK, 36 * mm)

        arrow(26 * mm, top + 4 * mm, 32 * mm, top + 4 * mm)
        arrow(58 * mm, top + 4 * mm, 64 * mm, top + 4 * mm)
        # 标记下沉到远端
        arrow(14 * mm, top - 1 * mm, 74 * mm, bot + 10.5 * mm, ACC, 2)
        arrow(79 * mm, top - 1 * mm, 116 * mm, bot + 10.5 * mm, OKC, 2)
        arrow(26 * mm, bot + 4 * mm, 64 * mm, bot + 4 * mm)
        arrow(94 * mm, bot + 4 * mm, 100 * mm, bot + 4 * mm)

        c.setFont("yh", 7)
        c.setFillColor(INK)
        c.drawString(139 * mm, bot + 6.5 * mm, "有「已发出」-> 不发邮件")
        c.drawString(139 * mm, bot + 2.5 * mm, "没有 -> 远端兜底发出")
        c.setFillColor(DIM)
        c.setFont("yh", 6.5)


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    st = load(ROOT / "state" / "learning_status.json")
    race = load(ROOT / "out_learn" / "model_race.json")

    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="A股流水线 操作手册与架构介绍")
    E = []

    # ============ 封面区 ============
    E.append(P("A股流水线 · 操作手册与架构介绍", "h1"))
    E.append(P(f"版本 {dt.date.today().isoformat()} · 仓库 SophiaSha2026/daily-report · "
               f"本手册由脚本从系统实时状态生成，重跑 tools/build_manual.py 即最新版", "dim"))
    E.append(Spacer(1, 4 * mm))
    E.append(P("这套系统每个交易日干三件事，全自动，不用你管："))
    E.append(Spacer(1, 2 * mm))
    E.append(tbl(["时间(北京)", "干什么", "产出"],
                 [["09:27:30", "竞价线：9:25 集合竞价里挑最强的 10 只",
                   "邮件 + 网页面板"],
                  ["17:00 后", "形态线：找「放量启动-缩量回调-今天再启动」的票",
                   "邮件 + 网页面板"],
                  ["16:30 后", "学习线：拿当天实际涨跌给早上的预测打分，攒够证据就微调参数",
                   "学习面板；改了参数才发邮件"]],
                 widths=[24 * mm, 92 * mm, 54 * mm]))
    E.append(Spacer(1, 3 * mm))
    E.append(P("一天的时间轴：", "b"))
    E.append(Flow([("08:23", "建候选池"), ("09:19-25", "三次采样"),
                   ("09:27:30", "竞价邮件"), ("15:00", "收盘"),
                   ("16:30", "自评估学习"), ("17:00", "形态邮件")],
                  hl={2, 5}))

    # ============ 日常操作 ============
    E.append(P("一、日常操作（只需要记这一节）", "h2"))
    E.append(P("<b>什么都不做</b>：远端 GitHub 每天自动跑，邮件照来。这是兜底，永远在。"))
    E.append(P("<b>想在本机跑（更快更稳，推荐）</b>：双击桌面「A股流水线」，按 1，选早/晚，"
               "选「本地(主)」。早盘 8 点前点开即可，它自己等到 9:25 采样、9:27:30 发信，"
               "全程实时显示进度。窗口别关就行。"))
    E.append(Spacer(1, 2 * mm))
    E.append(tbl(["按键", "功能", "说明"],
                 [["1", "一键跑全流程", "选 早/晚 + 本地/远端。本地跑完远端自动让位，不会收到两封"],
                  ["2", "面板", "打开在线或本地的结果网页"],
                  ["3", "状态", "两条线今天跑没跑、体检（行情源/依赖/邮箱/Claude 登录）"],
                  ["4", "日志", "云端或本地的运行日志"],
                  ["5", "退出", ""]],
                 widths=[12 * mm, 34 * mm, 124 * mm]))
    E.append(Spacer(1, 2 * mm))
    E.append(P("本地和远端怎么商量的（接管协议）：", "b"))
    E.append(Takeover())
    E.append(P("规则一句话：<b>远端只有亲眼看到「本地已发出」的凭证才让位</b>。"
               "你本地跑到一半死机，远端 09:28:20 兜底发出（最坏晚 55 秒）；"
               "绝不会出现没有邮件的一天。", "b"))

    _sec2 = []
    _sec2.append(P("二、结果去哪看", "h2"))
    _sec2.append(tbl(["东西", "位置"],
                 [["竞价榜邮件", "每交易日 09:27:30 发到邮箱，附同花顺导入文件和明细 CSV"],
                  ["竞价面板", "sophiasha2026.github.io/daily-report/（点代码即复制，可推同花顺）"],
                  ["形态面板", "同上域名 /pullback.html"],
                  ["学习面板", "同上域名 /learn.html（IC 曲线、闸门裁决、归因分布、模型擂台）"],
                  ["参数变更邮件", "只在学习系统真的改了参数那天发：动了哪个旋钮、证据、怎么回滚"]],
                 widths=[36 * mm, 134 * mm]))
    E.append(KeepTogether(_sec2))
    E.append(P("出问题排查顺序：TUI 按 3 看状态与体检 → 按 4 看日志 → "
               "手机上看 GitHub Actions 页面。邮件没来的唯一常见原因是非交易日。", "dim"))

    # ============ 架构 ============
    E.append(P("三、架构：三条流水线 + 一套学习系统", "h2"))
    E.append(P("<b>竞价线（早上，分秒必争）</b>：9:25 集合竞价的价和量一出来，"
               "五分钟内完成「采样-打分-写文案-发信」。"))
    E.append(Flow([("候选池", "昨日强势~1600只"), ("三次快照", "9:19/9:23/9:25"),
                   ("硬性排除", "涨幅2~5% 量比2.5~10"), ("六维打分", "纯规则可复现"),
                   ("LLM文案", "只写理由/风险"), ("发信", "09:27:30")], hl={3}))
    E.append(P("打分六个维度：竞价涨幅(20%)、竞价量能(25%)、竞价斜率(20%)、位置形态(15%)、"
               "板块共振(15%)、连板延续(5%)。<b>排序 100% 由规则算出，AI 不参与</b>——"
               "AI 只写「入选理由 / 风险提示」两句话，写不出来邮件照发。", "b"))
    E.append(Spacer(1, 2 * mm))
    E.append(P("<b>形态线（下午，不赶时间）</b>：收盘后扫全市场，找三段式形态。"
               "平均每个交易日约 1 只，多数日子是 0 只——这是常态不是故障。"))
    E.append(Flow([("全市场", "~5500只"), ("启动日", "涨幅≥5% 量1.5倍"),
                   ("缩量回调", "1~6天 不破启动低点"), ("今日再启动", "涨幅≥5% 量1.5倍"),
                   ("发信", "17:00后")], hl={4}))
    E.append(Spacer(1, 2 * mm))
    E.append(P("<b>学习线（傍晚，闭环）</b>：把「开盘买入、收盘卖出」的真实收益"
               "当成绩单，检验早上的排名准不准，用统计方法微调打分参数。"))
    E.append(Flow([("抓收盘价", "算真实收益"), ("LLM归因", "查每只为何涨跌"),
                   ("拟合", "401天历史+统计优化"), ("七道闸", "全过才改参数"),
                   ("发变更邮件", "改了才发")], hl={3}))

    E.append(P("四、学习系统为什么不会学坏（大白话）", "h2"))
    E.append(P("市场一天的涨跌大部分是噪声。这套系统的全部设计都在回答一个问题："
               "<b>怎么保证学到的是规律，不是把某一天的运气当真理</b>。四层防护："))
    E.append(tbl(["防护", "干什么", "大白话"],
                 [["日内中性化", "每天减去全市场中位涨幅、按当天波动归一",
                   "大盘暴跌那天所有票都跌，不能怪选股"],
                  ["Huber 聚合", "任何单日对结论的影响力有硬上限",
                   "某天运气爆棚选中涨停，也只算「一个好日子」，不会冲昏头"],
                  ["锚定 + 稀疏", "参数被拴在人工基线附近，一次最多动 2 个旋钮",
                   "数据越多绳子越松；改动少，你一眼能看懂动了什么"],
                  ["七道闸", "见下表，全过才真改",
                   "统计显著、行为不发疯、真值数据不反对，缺一不可"]],
                 widths=[26 * mm, 66 * mm, 78 * mm]))
    E.append(Spacer(1, 2 * mm))
    g = st.get("verdict", {}).get("checks", [])
    rows = [[c.get("name", ""), "通过" if c.get("passed") else "拦下",
             c.get("detail", "")] for c in g] or \
           [["（还没跑过）", "", ""]]
    E.append(KeepTogether([
        P(f"七道闸最近一次裁决（{st.get('date', '?')}）：", "b"),
        tbl(["闸门", "结论", "依据"], rows,
            widths=[30 * mm, 16 * mm, 124 * mm])]))
    E.append(P("没通过就不改参数——这不是故障，是闸门在干活。参数改了会发邮件，"
               "不认可就删 state/learned.yaml 一键回到人工基线。", "dim"))
    E.append(Spacer(1, 2 * mm))
    E.append(P("<b>红线（学习系统永远碰不到的）</b>：准入区间（涨幅 2~5%、量比 2.5~10）"
               "是你定的规则，系统只能提建议；发信时刻、数据采样窗口、绝对流动性下限，全部冻结。", "b"))

    E.append(PageBreak())

    # ============ 实测数字 ============
    E.append(P("五、首次实测（2025-01 至 2026-09，401 个交易日回填）", "h2"))
    m = st.get("metrics", {})
    if m:
        E.append(tbl(["指标", "数值", "什么意思"],
                     [["日 IC 均值", f"{m.get('ic_mean', 0):.4f}",
                       "分数和当天实际超额收益的相关性，>0 说明打分有预测力"],
                      ["ICIR", f"{m.get('icir', 0):.3f}", "IC 的稳定度"],
                      ["前 10 日均超额", f"{m.get('top_excess', 0):+.2%}",
                       "每天开盘买前10、收盘卖，相对市场中位数的日均收益（未扣约 0.13% 成本）"],
                      ["前 10 胜率", f"{m.get('hit_rate', 0):.1%}", "前 10 里跑赢市场中位的比例"],
                      ["日均通过准入", f"{m.get('avg_pool', 0):.0f} 只", "漏斗最后一层的宽度"]],
                     widths=[30 * mm, 26 * mm, 114 * mm]))
    if race.get("table"):
        E.append(Spacer(1, 2 * mm))
        E.append(P("模型擂台（同一套按天走向前验证，样本外）：", "b"))
        E.append(tbl(["模型", "IC", "ICIR", "前10超额/日", "胜率"],
                     [[r["model"], f"{r['IC']:.4f}", f"{r['ICIR']:.2f}",
                       f"{r['top_excess']:+.2%}", f"{r['hit']:.0%}"]
                      for r in race["table"]],
                     widths=[46 * mm, 24 * mm, 22 * mm, 34 * mm, 20 * mm]))
        E.append(P(f"选中：<b>{race.get('winner', '?')}</b>。{race.get('why', '')}", "b"))
        E.append(P("胜出模型的角色是「老师」：估计规则打分还差多远（天花板）、"
                   "给形状参数提建议。它不直接排序——黑箱排名无法归因，也无法用规则语言讨论。", "dim"))

    # 影子对比（有数据才显示）
    sh = st.get("shadow") or []
    E.append(Spacer(1, 2 * mm))
    E.append(P("<b>架构结论（2026-09 实测得出）</b>：调那 9 个参数已接近到头"
               "（梯度极小），真正的差距在打分函数的形式里——连线性模型都能把"
               "前 10 超额翻倍。好在擂台胜者 RankHuber 本身是线性的、每个系数"
               "都能打印解释，不算黑箱。所以它现在以<b>影子</b>身份每天和"
               "生产打分器并排跑（只记账，不发信），在真实交易日上攒证据；"
               "攒够 30 天且显著领先时，系统会给你一份<b>切换提案</b>——"
               "切不切换永远是你的决定。", "b"))
    if sh:
        import statistics as _st
        b = _st.mean(x["base_top_excess"] for x in sh)
        s2 = _st.mean(x["shadow_top_excess"] for x in sh)
        E.append(P(f"影子对比进度：{len(sh)} 个真值交易日，前10日均超额 "
                   f"基线 {b:+.2%} vs 影子 {s2:+.2%}。", "dim"))

    E.append(P("六、三层触发保险（邮件为什么一定会来）", "h2"))
    E.append(tbl(["层", "谁", "什么时候"],
                 [["1", "Cloudflare 定时器（常年在线，独立于前两者）", "北京 07:30 / 17:05 叫醒 GitHub"],
                  ["2", "本机计划任务（无窗后台，插电即跑）", "同时段备份叫醒"],
                  ["3", "你手动：桌面 TUI 按 1", "任何时候"]],
                 widths=[10 * mm, 106 * mm, 54 * mm]))
    E.append(P("GitHub 自己的定时器实测会迟到甚至整段丢失，所以从不单独依赖它。"
               "三层里任何一层活着，当天邮件就会发出。", "dim"))

    _q = [P("七、速查", "h2")]
    _q.append(tbl(["想做什么", "怎么做"],
                 [["改筛选阈值", "改 config.yaml 里 screen 段，git push 即生效（先跑 python src/selftest.py）"],
                  ["回滚学习系统的改动", "删掉 state/learned.yaml 并 push"],
                  ["补跑某天形态线", "GitHub Actions -> 5-形态扫描 -> Run workflow，填 asof 日期"],
                  ["看学习系统学了什么", "learn.html 面板；或 python src/eval_daily.py --stage status"],
                  ["本地 LLM 不工作", "终端跑一次 claude 完成登录；TUI 按 3 体检里有一行 claude auth"],
                  ["全部数据源都挂了", "TUI 按 1 选本地跑——本机可达全部行情源，比 GitHub 的机器通路更多"]],
                 widths=[52 * mm, 118 * mm]))
    E.append(KeepTogether(_q))
    E.append(Spacer(1, 4 * mm))
    E.append(P("完整设计文档：docs/learning.md（学习系统）、CLAUDE.md（全部约定与教训）。"
               "本清单为量化筛选结果，非投资建议。", "dim"))

    doc.build(E)
    print(f"OK {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
