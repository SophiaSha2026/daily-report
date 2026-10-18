"""
审计第 4 项：research/angles 的数学（设计效应 / n_eff / 单侧下界 / Clopper-Pearson、双正态 PPV）用数值独立核一遍。

    PYTHONIOENCODING=utf-8 python research/audit/stats/math_check.py  -> results_math.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "precision", ROOT / "research" / "polish", ROOT / "research" / "audit" / "stats"):
    sys.path.insert(0, str(p))
import evalkit as K            # noqa: E402
import common_pr as C          # noqa: E402
import mc_stats as M           # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_math.txt"
lines: list[str] = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def boot_ratios(h, n, b=2000, block=20, seed=0):
    idx = K._mbb_index(len(h), block, b, np.random.default_rng(seed))
    hs, ns = h[idx].sum(axis=1), n[idx].sum(axis=1)
    r = hs / np.maximum(ns, 1)
    r[ns == 0] = np.nan
    return r


def main():
    rng = np.random.default_rng(2026)
    say("== 4a ltt_an 的单侧 95% 下界（块自举 5% 分位）和 CP(n_eff) 下界：按天聚集 + 20 天重叠 DGP 下的覆盖率（名义 95%）==")
    sig_m, sig_e, p = 0.012, 0.03, 0.15
    thr = M.calib_thr(rng, p, sig_m, sig_e)
    Pt = M.true_P_overlap(rng, thr, sig_m, sig_e)
    say("   真 P = %.4f" % Pt)
    R = 300
    for D, k in ((582, 1), (387, 1), (582, 10), (387, 10), (969, 2)):
        cov_boot = cov_cp = cov_cp_iid = 0
        des, neffs, lo5s, cps = [], [], [], []
        for r in range(R):
            n_d = np.full(D, float(k))
            h = M.gen_overlap(rng, D, n_d, p, sig_m, sig_e, thr)
            V = n_d.sum()
            P = h.sum() / V
            rr = boot_ratios(h, n_d, seed=int(rng.integers(1 << 30)))
            lo5 = float(np.nanquantile(rr, 0.05))
            var_boot = float(np.nanvar(rr))
            var_binom = P * (1 - P) / V
            de = var_boot / var_binom if var_binom > 0 else np.nan
            neff = V / de
            cp = float(beta.ppf(0.05, P * neff, neff - P * neff + 1)) if neff > 1 else np.nan
            cp_iid = float(beta.ppf(0.05, h.sum(), V - h.sum() + 1))
            cov_boot += lo5 <= Pt
            cov_cp += cp <= Pt
            cov_cp_iid += cp_iid <= Pt
            des.append(de)
            neffs.append(neff)
            lo5s.append(lo5)
            cps.append(cp)
        say("   D=%d k=%d V=%d：设计效应均值 %.2f  n_eff 均值 %.0f | 单侧下界覆盖：块自举 5%% 分位 %.3f  CP(n_eff) %.3f  CP(iid 全样本) %.3f | 下界均值 块自举 %.4f CP %.4f" % (
            D, k, D * k, np.mean(des), np.mean(neffs), cov_boot / R, cov_cp / R, cov_cp_iid / R, np.mean(lo5s), np.mean(cps)))
    say("   读法：覆盖率低于 0.95 说明「可证下界」偏高（承诺过头）；块 20 对 20 天重叠的序列只抓到部分长程相关，预期略欠覆盖。")

    say()
    say("== 4b math_an 双正态 PPV：公式 对 模拟 ==")
    for pi, auc in ((0.03, 0.70), (0.03, 0.75), (0.02, 0.65)):
        dp = np.sqrt(2) * norm.ppf(auc)
        # 每天 5000 只，正样本比例 pi，取最高分一只 / 前 3 只，模拟 20000 天
        days = 20000
        y = rng.random((days, 5000)) < pi
        s = rng.normal(0, 1, (days, 5000)) + dp * y
        top1 = y[np.arange(days), s.argmax(axis=1)].mean()
        idx3 = np.argsort(-s, axis=1)[:, :3]
        top3 = y[np.arange(days)[:, None], idx3].mean()
        # 公式：q = 1/5000 的阈值 PPV
        zs = np.linspace(-3, 6, 9001)
        q = pi * norm.sf(zs - dp) + (1 - pi) * norm.sf(zs)
        ppv = pi * norm.sf(zs - dp) / q
        i1 = np.argmin(np.abs(q - 1 / 5000))
        i3 = np.argmin(np.abs(q - 3 / 5000))
        # 模拟的按天 AUC 核对 d'
        auc_sim = np.mean([norm.cdf(dp / np.sqrt(2))])
        say("   π %.2f AUC %.2f（d' %.3f）：第 1 名 PPV 公式 %.1f%% 模拟 %.1f%%；前 3 PPV 公式 %.1f%% 模拟 %.1f%%" % (
            pi, auc, dp, 100 * ppv[i1], 100 * top1, 100 * ppv[i3], 100 * top3))
    say("   公式本身对；要点是「等方差双正态」：正样本方差更大（右尾更厚）时同一 AUC 下尾部 PPV 会高得多，表格只是量级。")
    # 不等方差示例
    pi, auc = 0.03, 0.70
    for sd_pos in (1.0, 1.5, 2.0):
        # 调 d' 使 AUC 保持 0.70：AUC = Φ(d'/sqrt(1+sd^2))
        dp = np.sqrt(1 + sd_pos ** 2) * norm.ppf(auc)
        days = 20000
        y = rng.random((days, 5000)) < pi
        s = rng.normal(0, 1, (days, 5000)) * np.where(y, sd_pos, 1.0) + dp * y
        top1 = y[np.arange(days), s.argmax(axis=1)].mean()
        say("   同 AUC 0.70、正样本 sd %.1f：第 1 名 PPV 模拟 %.1f%%" % (sd_pos, 100 * top1))

    say()
    say("== 4c LTT 固定序列的读法 ==")
    say("   ltt_an 对 k = 1,2,3,5,10 每个各跑一条「x 递增、首次不拒绝即停」的链，但一条链停了下一条 k 仍从 x=0.05 重新开始。")
    say("   固定序列法要求整个序列事先排好、任何一次不拒绝就全部停止；五条独立的链并列报告，族错误率不是 5%，")
    say("   每条链各自 5%（嵌套假设）。它只是「描述」，但 docstring 写的「LTT 的固定序列，族错误率 5%」过头。")
    say("   另：bound_row 的「双侧下界」= 2.5% 分位，「单侧 95% 下界」= 5% 分位，口径标注正确。")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
