"""
角度一 1a：数学先算清（实测之前）。

    PYTHONIOENCODING=utf-8 python research/angles/math_an.py   -> results_math.txt

(i)  FDR 对 mFDR：全局零假设下按天 BH(α) 的入选概率、入选时的精确率（模拟 + Simes 公式）。
(ii) 双正态下 PPV 达到 τ 的入选比例、每天出几只、召回：π x AUC x τ 表；以及「每天 5000 只里最高分那一只」的 PPV。
(iii) 校准负样本数和 p 值分辨率；按天聚集下的有效样本。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import common_an as A

OUT = A.HERE / "results_math.txt"
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def bh_select(p: np.ndarray, alpha: float) -> int:
    m = len(p)
    ps = np.sort(p)
    k = np.arange(1, m + 1)
    ok = np.nonzero(ps <= alpha * k / m)[0]
    return int(ok[-1] + 1) if len(ok) else 0


def main():
    rng = np.random.default_rng(1)
    say("角度一 1a：数学（2026-10-18，实测之前）")
    say()
    say("(i) FDR 不是条件精确率。全局零假设（候选池里没有一只会中，或标签和分数无关）下，按天 BH(α)：")
    say("    Simes：P(至少选一只) = α（独立均匀 p 值时恰好等于 α）。FDR = E[FDP]，空选记 0，所以 FDR = α·P(选了|...)... ≤ α 恒成立，")
    say("    但「选了的那些天」里精确率 = 基准率 π。用户要的是 Σh/Σn（池化精确率 = 1 − mFDR），不是 FDR。")
    say("    模拟：每天 m = 5000 个均匀 p 值（标签打乱 / 无信号），α = 1 − τ：")
    for tau in (0.15, 0.20, 0.25, 0.30):
        alpha = 1 - tau
        R = np.array([bh_select(rng.random(5000), alpha) for _ in range(2000)])
        say("      τ %.2f α %.2f：P(R>0) %.3f  E[R] %.1f  E[R | R>0] %.1f  R 的 50/90/99 分位 %d/%d/%d" % (
            tau, alpha, (R > 0).mean(), R.mean(), R[R > 0].mean() if (R > 0).any() else 0,
            *np.percentile(R, [50, 90, 99]).astype(int)))
    say("    读法：α ≥ 0.7 时 BH 在无信号的天里大约七到八成的天都会「选中」一批票，而且常常是一大批（step-up 一旦过线就整段放行）。")
    say("    所以 conformal + BH 在 α 接近 1 的区间没有产品意义，必须改成控制 mFDR（累计精确率 ≥ τ 且支持数够）。研究里两种都报。")
    say()
    say("(ii) 任何按分数切的选择，精确率上限 = max_s PPV(s)。双正态（负样本 N(0,1)，正样本 N(d',1)，d' = √2·Φ⁻¹(AUC)）：")
    say("     PPV(z) = π·S(z−d') / (π·S(z−d') + (1−π)·S(z))，S = 1 − Φ。给定 τ，最宽的切点 z* 使 PPV ≥ τ；入选比例 q = π·S(z*−d') + (1−π)·S(z*)。")
    say("     每天候选池 5000 只；表里「只/天」= 5000·q，「召回」= S(z*−d')。按天 AUC 实测 0.65（现状模型，rethink ceiling）。")
    say()
    say("     %4s %5s | %s" % ("π", "AUC", "  ".join("τ=%.2f: z* / 只/天 / 召回" % t for t in (0.15, 0.20, 0.25, 0.30))))
    zs = np.linspace(-3, 6, 9001)
    for pi in (0.02, 0.03, 0.04):
        for auc in (0.65, 0.70, 0.75, 0.80):
            dp = np.sqrt(2) * norm.ppf(auc)
            ppv = pi * norm.sf(zs - dp) / (pi * norm.sf(zs - dp) + (1 - pi) * norm.sf(zs))
            q = pi * norm.sf(zs - dp) + (1 - pi) * norm.sf(zs)
            cells = []
            for tau in (0.15, 0.20, 0.25, 0.30):
                ok = np.nonzero(ppv >= tau)[0]
                if len(ok) == 0:
                    cells.append("        —         ")
                    continue
                z = zs[ok[0]]
                cells.append("%5.2f / %7.3f / %5.3f" % (z, 5000 * q[ok[0]], norm.sf(z - dp)))
            say("     %4.2f %5.2f | %s" % (pi, auc, "  ".join(cells)))
    say()
    say("     「每天最高分那一只」的 PPV（q = 1/5000）和「前 3 只」（q = 3/5000）：")
    for pi in (0.02, 0.03, 0.04):
        row = []
        for auc in (0.65, 0.70, 0.75, 0.80):
            dp = np.sqrt(2) * norm.ppf(auc)
            q = pi * norm.sf(zs - dp) + (1 - pi) * norm.sf(zs)
            ppv = pi * norm.sf(zs - dp) / q
            i1 = np.argmin(np.abs(q - 1 / 5000))
            i3 = np.argmin(np.abs(q - 3 / 5000))
            row.append("AUC %.2f: %4.1f%% / %4.1f%%" % (auc, 100 * ppv[i1], 100 * ppv[i3]))
        say("     π %.2f  %s" % (pi, "   ".join(row)))
    say("     读法：实测第 1 名 14~18%（两段），对应双正态 AUC 0.70~0.75 的第 1 名水平；要让「出的票」稳定 ≥ 25%，")
    say("     AUC 0.70 下每天只能出 0.01~0.1 只（每 10~100 天一只），AUC 0.75 每天 0.1~0.5 只；≥ 30% 基本不可达。")
    say("     而且这是分布完全平稳、切点已知的理想情况；按月漂移时切点只能从过去估，上面的数字是上限。")
    say()
    say("(iii) 校准：p 值分辨率 1/(n_0+1)。按天 BH(α) 在 m = 5000 时最小门槛 α/m，α = 0.8 -> 1.6e-4，需要 n_0 ≥ 6250 个校准负样本")
    say("      才能表示这个 p 值；12 个月候选池约 100 万负样本，分辨率够。真正的约束是按天聚集：同一天的票共享市场状态，")
    say("      校准集 12 个月只有约 240 个「天」，切点之上的负样本来自几十天；可交换性要求校准天和测试天同分布，")
    say("      而第 1 名的滚动 12 月精确率在 7%~21% 之间漂（gate 报告），这正是可交换性被破坏的量级：τ = 0.20 的线在 2023 年定出来的是「不出」，")
    say("      2024-09 之后定出来的是「大量出」。时间加权 / 自适应 α 只能缩短反应时间（标签 20 根后才关窗，最快一个月），不能预知下个月。")
    say()
    say("结论（实测之前写下，实测后对照）：conformal selection 给的是「期望意义下的 FDR ≤ α」，在 α ≥ 0.7 时既不等于条件精确率，")
    say("也挡不住全局零假设下的随机入选；可达的精确率上限由分数的 PPV 曲线决定，现有分数（第 1 名 14~18%）意味着 τ = 0.20 勉强、")
    say("τ ≥ 0.25 每月不到 1 只。实测要回答的只剩：加权 / 日内名次 / 自适应 α 能不能把「校准线被热月带偏」压下去。")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.progress("math_an：角度一数学写定 -> results_math.txt")


if __name__ == "__main__":
    main()
