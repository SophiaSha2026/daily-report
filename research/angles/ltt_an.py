"""
8d：按 Learn then Test 的框架算「现在能承诺的精确率下界」。

    PYTHONIOENCODING=utf-8 python research/angles/ltt_an.py   -> results_ltt.txt

对 B(k)（k = 1 2 3 5 10）：两段各自 + 合并 969 天，单侧 95% 下界 = 按天块自举（块 20，4000 次）比值的 5% 分位；
有效样本 n_eff = V / 设计效应（设计效应 = 块自举方差 / 二项方差 P(1−P)/V），Clopper-Pearson 单侧下界（n_eff，P·n_eff）。
固定序列检验：k 从 1 到 10 依次，H0: P <= x，x 从 0.10 起每 0.01 一档，p 值 = 块自举比值 <= x 的比例；
第一个在 x 上不能拒绝（p >= 0.05）的 k 之后停止（LTT 的固定序列，族错误率 5%）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

import common_an as A

OUT = A.HERE / "results_ltt.txt"
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def boot_ratios(h: np.ndarray, n: np.ndarray, b: int = 4000, block: int = 20, seed: int = 20261018) -> np.ndarray:
    import evalkit as K
    idx = K._mbb_index(len(h), block, b, np.random.default_rng(seed))
    hs = h[idx].sum(axis=1)
    ns = n[idx].sum(axis=1)
    r = hs / np.maximum(ns, 1)
    r[ns == 0] = np.nan
    return r


def bound_row(name: str, h: np.ndarray, n: np.ndarray) -> dict:
    V = n.sum()
    P = h.sum() / V
    r = boot_ratios(h, n)
    lo5 = float(np.nanquantile(r, 0.05))
    lo25 = float(np.nanquantile(r, 0.025))
    var_boot = float(np.nanvar(r))
    var_binom = P * (1 - P) / V
    de = var_boot / var_binom if var_binom > 0 else np.nan
    neff = V / de if de > 0 else np.nan
    cp = float(beta.ppf(0.05, P * neff, neff - P * neff + 1)) if np.isfinite(neff) and neff > 1 else np.nan
    say("   %-22s V %5d  P %5.1f%%  单侧95%%下界(块自举) %5.1f%%  双侧下界 %5.1f%%  设计效应 %.1f  n_eff %5.0f  CP 单侧下界 %5.1f%%" % (
        name, V, 100 * P, 100 * lo5, 100 * lo25, de, neff, 100 * cp))
    return {"V": V, "P": P, "lo5": lo5, "r": r}


def main():
    say("8d：现在能承诺的精确率下界（Learn then Test 框架，有效样本按天块算）")
    say()
    seqs = {}
    for seg in ("screen", "dev"):
        B = A.G.baseline(seg)
        seqs[seg] = {k: (B.hits_k(k), np.full(B.D, float(k))) for k in (1, 2, 3, 5, 10)}
    say("== 两段各自")
    rows = {}
    for seg in ("screen", "dev"):
        for k in (1, 2, 3, 5, 10):
            h, n = seqs[seg][k]
            rows[(seg, k)] = bound_row("%s B(%d)" % (seg, k), h, n)
    say()
    say("== 合并 969 天（筛选集 + 开发集按日期接起来）")
    for k in (1, 2, 3, 5, 10):
        h = np.concatenate([seqs["screen"][k][0], seqs["dev"][k][0]])
        n = np.concatenate([seqs["screen"][k][1], seqs["dev"][k][1]])
        rows[("all", k)] = bound_row("合并 B(%d)" % k, h, n)
    say()
    say("== 固定序列检验（LTT）：k = 1..10 依次，H0: P <= x，x 从 0.10 每 0.01 上调，p = 块自举比值 <= x 的比例，p < 0.05 拒绝")
    for scope in ("all", "screen", "dev"):
        out = []
        stop = False
        for k in (1, 2, 3, 5, 10):
            if scope == "all":
                h = np.concatenate([seqs["screen"][k][0], seqs["dev"][k][0]])
                n = np.concatenate([seqs["screen"][k][1], seqs["dev"][k][1]])
            else:
                h, n = seqs[scope][k]
            r = boot_ratios(h, n)
            best = None
            for x in np.arange(0.05, 0.40, 0.01):
                p = float(np.nanmean(r <= x))
                if p < 0.05:
                    best = x
                else:
                    break
            if stop or best is None:
                out.append("B(%d): 停（前一级没通过）" % k if stop else "B(%d): 连 5%% 都拒绝不了" % k)
                stop = True
            else:
                out.append("B(%d): 可证 P > %.0f%%" % (k, 100 * best))
        say("   %-6s %s" % (scope, "；".join(out)))
    say()
    say("读法：「标准」只能写成「合并 969 天单侧 95% 下界」那一档；两段分别的下界更低，开发集 B(1) 的下界低于筛选集 B(10) 的点估计。")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.ledger_add({"kind": "diag", "name": "ltt_bounds", "note": "B(k) 的可证下界（描述，不选）"})
    A.progress("ltt_an：可承诺下界 -> results_ltt.txt")


if __name__ == "__main__":
    main()
