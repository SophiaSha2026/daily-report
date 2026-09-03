# -*- coding: utf-8 -*-
"""
生成《这套系统是怎么开发出来的》PDF。

    python tools/build_devprocess.py    -> out_learn/devprocess.pdf

姊妹篇是 build_manual.py（讲怎么用）；这份讲怎么做出来的——方法、
决策、踩过的坑，给以后接手开发的人（包括未来的自己和 AI）看。
"""
import datetime as dt
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Flowable, KeepTogether)
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_learn" / "devprocess.pdf"

pdfmetrics.registerFont(TTFont("yh", "C:/Windows/Fonts/msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("yhb", "C:/Windows/Fonts/msyhbd.ttc", subfontIndex=0))

INK = colors.HexColor("#1c1c1e")
DIM = colors.HexColor("#6b6b70")
ACC = colors.HexColor("#0a5bd3")
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


def tbl(head, rows, widths=None):
    data = [[Paragraph(str(h), S["cellh"]) for h in head]] + \
           [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3a3f4b")),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BOX])]))
    return t


class Cycle(Flowable):
    """开发循环：实测 -> 设计 -> 自测 -> 上线 -> 测量 -> 校准 -> 回写文档 -> ..."""

    def __init__(self, width=170 * mm, h=34 * mm):
        super().__init__()
        self.width, self.h = width, h

    def wrap(self, aw, ah):
        return self.width, self.h

    def draw(self):
        c = self.canv
        steps = [("实测", "先量数据源\n和信号"), ("设计", "写进文档\n再动手"),
                 ("自测", "改哪条线\n跑哪条自测"), ("上线", "红线内\n小步走"),
                 ("测量", "让数字\n说话"), ("校准", "按测量值\n调，不猜"),
                 ("回写", "结论和坑\n进文档")]
        n = len(steps)
        gap = 4 * mm
        bw = (self.width - gap * (n - 1)) / n
        bh = 14 * mm
        y = (self.h - bh) / 2
        for i, (t, sub) in enumerate(steps):
            x = i * (bw + gap)
            c.setFillColor(BOX)
            c.setStrokeColor(ACC)
            c.roundRect(x, y, bw, bh, 2 * mm, stroke=1, fill=1)
            c.setFillColor(ACC)
            c.setFont("yhb", 9)
            c.drawCentredString(x + bw / 2, y + bh - 5.2 * mm, t)
            c.setFillColor(DIM)
            c.setFont("yh", 6.3)
            for k, ln in enumerate(sub.split("\n")):
                c.drawCentredString(x + bw / 2, y + bh - (8.6 + k * 2.6) * mm, ln)
            if i < n - 1:
                ax, ay = x + bw, y + bh / 2
                c.setStrokeColor(DIM)
                c.line(ax + 0.5, ay, ax + gap - 1.5, ay)
                c.setFillColor(DIM)
                p = c.beginPath()
                p.moveTo(ax + gap - 1, ay)
                p.lineTo(ax + gap - 3.5, ay + 1.5)
                p.lineTo(ax + gap - 3.5, ay - 1.5)
                p.close()
                c.drawPath(p, stroke=0, fill=1)
        # 回环箭头
        c.setStrokeColor(LINE)
        c.setDash(2, 2)
        x0, x1 = bw / 2, self.width - bw / 2
        yb = y - 2.5 * mm
        c.line(x1, y, x1, yb)
        c.line(x1, yb, x0, yb)
        c.line(x0, yb, x0, y - 0.8 * mm)
        c.setDash()
        c.setFillColor(LINE)
        p = c.beginPath()
        p.moveTo(x0, y - 0.3 * mm)
        p.lineTo(x0 - 1.5, y - 3 * mm)
        p.lineTo(x0 + 1.5, y - 3 * mm)
        p.close()
        c.drawPath(p, stroke=0, fill=1)


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="A股流水线 开发流程")
    E = []
    E.append(P("这套系统是怎么开发出来的", "h1"))
    E.append(P(f"版本 {dt.date.today().isoformat()} · 姊妹篇《操作手册与架构介绍》讲怎么用，"
               f"这份讲怎么做——方法、决策、踩过的坑。给以后接手开发的人看。", "dim"))
    E.append(Spacer(1, 3 * mm))

    E.append(P("一、一个循环转到底", "h2"))
    E.append(P("整个项目就是同一个循环转了很多圈。每圈不大，但每圈都有实测数字进、"
               "有结论出、有文档留痕："))
    E.append(Cycle())
    E.append(P("举一圈完整的例子——学习系统点火：实测（扰动探针量出信号只有每 σ "
               "0.02~0.03）→ 设计（正则强度按测量值反解，不再拍脑袋）→ 自测（新增 "
               "6 条稀疏化断言）→ 上线（第四次点火）→ 测量（样本外 −0.0007）→ "
               "裁决（闸门拦下，不改）→ 回写（全过程进 docs/learning.md 第 13 节）。"
               "<b>拦下也是产出</b>——「手调参数已是局部最优」这个结论值这一晚。", "b"))

    E.append(P("二、七条工作规矩（每条都有真实来历）", "h2"))
    E.append(tbl(["规矩", "大白话", "真实来历"],
                 [["核实优于断言", "涉及第三方接口，先实测再写代码，不凭记忆",
                   "凭记忆写过一个不存在的参数名直接报错；分钟线「应该能取」实测五个渠道全废"],
                  ["自测先行", "三条流水线各配离线自测，改哪条跑哪条，0.5 秒出结果",
                    "打分曲线曾被一次改阈值悄悄带歪，从此曲线形状用断言钉死"],
                  ["红线先立", "先写「永远不许」再写功能：LLM 不排序、失败不阻断发信、准入区间人定",
                   "红线让后面所有快速迭代都不用担心踩到底线"],
                  ["失败隔离", "每个新组件都问：它崩了会不会影响发邮件？答案必须是不会",
                   "学习线、LLM、影子模型全部 fail-open，异常吞掉照常发信"],
                  ["教训入册", "踩过的坑写进 CLAUDE.md，标日期，下次开发前先读",
                   "东财接口一天之间从可用变全灭——「结论会过期」本身就是一条教训"],
                  ["诚实裁决", "统计闸门说不改就不改，不调参数调到它「招供」为止",
                   "四次点火三次被拦，每次拦截都写明原因，最终结论反而更可信"],
                  ["接线也要测", "组件各自正确不等于连起来正确：闸门、邮件、守卫的调用处都要有断言",
                   "闸门 3 拿到的是阈值不是统计量，四次点火没人发现——输出看着完全合理"]],
                 widths=[24 * mm, 62 * mm, 84 * mm]))

    E.append(P("三、时间线：从一张表到自学习系统", "h2"))
    E.append(tbl(["阶段", "做了什么", "关键产出"],
                 [["规则落地", "把你给的竞价筛选表变成流水线：候选池、三次快照、打分、发信",
                   "每天 09:27:30 一封邮件"],
                  ["可靠性", "GitHub 定时器实测会迟到丢班 -> 三层触发（Cloudflare/本机/手动）+ 接管协议",
                   "任何一层活着邮件就会到"],
                  ["第二条线", "收盘后的形态扫描（启动-缩量回调-再启动），实测频率约每天 1 只",
                   "17:00 第二封邮件"],
                  ["数据考古", "为学习系统实测全部历史数据渠道：免费的全废，Tushare 竞价包可用",
                   "405 天 × 5400 只竞价历史"],
                  ["学习系统", "损失函数（软TopK+Huber+锚定）、七道闸、LLM 归因、影子模型",
                   "四次点火 + 每日自动评估"],
                  ["架构定型", "旋钮已到头（实测），缺口在函数形式；擂台胜者恰好透明可解释",
                   "影子转正路径 + 两份 PDF"],
                  ["扩容 v2", "AI 提的特征转正（板块竞价一致性）+ 市场状态特征 + 尾盘竞价新数据源 + Opus 审稿通道",
                   "27 特征、通道 4、TUI 学习状态"],
                  ["全仓 debug", "静态检查 + 三条自测 + 各阶段空跑 + 远端日志：抓到三处接线错误（闸门 3 读到阈值、"
                                 "变更邮件参数反了、守卫误伤回填 3 天）和一处触发时序隐患，全部加断言钉住",
                   "lint 进必跑清单；学习面板改成进度可视化"]],
                 widths=[20 * mm, 92 * mm, 58 * mm]))

    E.append(P("四、关键决策记录（为什么是现在这样）", "h2"))
    E.append(tbl(["决策", "为什么", "代价（明说）"],
                 [["LLM 永不排序", "排名必须可复现、可回测、可归因。Opus 做归因、文案、审稿——都是给人看的产出",
                   "放弃了模型即兴判断可能带来的灵活性"],
                  ["排序 100% 规则/线性", "黑箱没法用人话讨论「为什么选它」；线性模型系数可打印，够透明才许上",
                   "天花板测量显示树模型还高一截（+2.1% vs +1.4%），暂时不吃"],
                  ["本地为主远端为辅", "本机可达全部行情源且不排队；远端当兜底。两边靠 git 标记协商，宁重不漏",
                   "多维护一套 TUI 和接管协议"],
                  ["训练用回填+在线当否决", "等 60 天在线积累太慢；401 天历史当场可用，但竞价轨迹是代理值，所以真值快照有一票否决",
                   "回填有幸存者偏差和板块表漂移，已标注"],
                  ["参数一次最多动两个", "改动必须能写进一封人能看懂的邮件；九个数各挪 0.3% 等于什么都没说清",
                   "收敛更慢——这是特性不是缺陷"],
                  ["深度学习不上", "有效样本按天算只有 400；表格数据+小样本下树模型稳定更强，是公开结论",
                   "留了扩展点，天数上去可重评"]],
                 widths=[34 * mm, 84 * mm, 52 * mm]))

    E.append(P("五、质量护栏（数字）", "h2"))
    E.append(tbl(["护栏", "规模"],
                 [["离线自测", "竞价 30+ 断言 / 形态 13 条 / 学习 70 余条，合计 <2 秒跑完，改哪条跑哪条"],
                  ["静态检查", "pyflakes 全仓零输出才许提交：「赋值了没用」是接错线的信号，闸门 3 那次就是这么漏的"],
                  ["曲线不变量", "9 条钉住打分曲线形状，改阈值带歪形状会立刻红"],
                  ["等价性断言", "向量化打分器与生产打分器 2000 随机样本逐位一致，红了整个学习系统作废"],
                  ["七道闸", "样本量/样本外/自助显著/步长/行为回放/冷却/在线否决，全过才改参数"],
                  ["历史教训", "CLAUDE.md 十余条，每条标日期，新开发前必读"],
                  ["文档回写", "docs/learning.md 与代码同仓库同提交，结论全部标注测量日期"]],
                 widths=[34 * mm, 136 * mm]))

    E.append(Spacer(1, 2 * mm))
    E.append(KeepTogether([
        P("六、以后怎么继续开发", "h2"),
        tbl(["要做的事", "流程"],
            [["改竞价/形态阈值", "改 config.yaml → 跑对应自测 → push。学习系统的箱约束和曲线断言会兜住形状"],
             ["加新特征", "先进积压清单（LLM 或人提）→ 人工实现 → 加进擂台横评 → 数据说话"],
             ["动打分逻辑", "score.py 和 learn/vscore.py 是孪生体，必须同步改，等价性断言钉着"],
             ["动学习系统", "只动 learning 段超参可直接改；动闸门逻辑先加自测断言再改实现"],
             ["接手前必读", "CLAUDE.md（约定+教训）→ docs/learning.md（设计）→ 两份 PDF（全貌）"]],
            widths=[40 * mm, 130 * mm])]))
    E.append(Spacer(1, 3 * mm))
    E.append(P("最后一个花絮：写这份文档的当晚电脑强制重启过一次，恢复动作只有"
               "两步——git status 看一眼、三条自测跑一遍，全绿即继续。"
               "被打断的两个后台计算都设计成断点续跑。<b>「随时可以死、"
               "醒来接着干」不是口号，是每个组件的验收标准。</b>", "b"))
    E.append(Spacer(1, 2 * mm))
    E.append(P("一句话总结这套开发流程：<b>先实测、立红线、配自测，然后小步快跑，"
               "让统计闸门当裁判，把每个结论和每个坑都写下来。</b>", "b"))

    doc.build(E)
    print(f"OK {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
