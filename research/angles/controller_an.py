"""
角度一 8a：在线名额控制器。第一步：B(k) 每日命中率在滞后 21~60 的自相关（判生死）；活着才模拟控制器。

    PYTHONIOENCODING=utf-8 python research/angles/controller_an.py [--force]   -> results_controller.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_controller.txt"
KS = (1, 2, 3, 5, 10)
GRID = ((0.2, 10), (0.4, 5), (0.6, 3), (0.8, 2), (0.95, 1), (1.01, 0))
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def acf(x: np.ndarray, lags: range) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - x.mean()
    v = (x * x).sum()
    return np.array([(x[l:] * x[:-l]).sum() / v for l in lags])


def k_of(lam: float) -> int:
    for th, k in GRID:
        if lam < th:
            return k
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    say("角度一 8a：在线名额控制器（criteria 第 8a 节）")
    say()
    say("== 第一步：B(k) 每日命中率 h_d/k 的自相关，滞后 21~60（标签 20 根后才关窗，控制器只能看到 21 天前的真值）")
    alive = {}
    for seg in ("screen", "dev"):
        B = A.G.baseline(seg)
        D = B.D
        noise = 2 / np.sqrt(D)
        say("   %s %d 天，噪声水平 2/√D = %.3f" % (seg, D, noise))
        for k in KS:
            h = B.hits_k(k) / k
            r = acf(h, range(1, 61))
            ma = pd.Series(h).rolling(20).mean().dropna().to_numpy()
            rm = acf(ma, range(1, 61))
            m2160 = float(r[20:60].mean())
            say("      k=%2d 原序列: lag1 %+.3f lag5 %+.3f lag21 %+.3f lag30 %+.3f lag40 %+.3f lag60 %+.3f  平均(21~60) %+.3f  "
                "| 20 日均线: lag21 %+.3f lag30 %+.3f lag40 %+.3f lag60 %+.3f 平均(21~60) %+.3f" % (
                    k, r[0], r[4], r[20], r[29], r[39], r[59], m2160, rm[20], rm[29], rm[39], rm[59], float(rm[20:60].mean())))
            alive.setdefault(k, {})[seg] = m2160
    say()
    verdict = {}
    for k in KS:
        s, d = alive[k]["screen"], alive[k]["dev"]
        verdict[k] = (s >= 0.05 or d >= 0.05) and s > 0 and d > 0
        say("   k=%2d：筛选 %+.3f 开发 %+.3f -> %s" % (k, s, d, "活（模拟控制器）" if verdict[k] else "死（两段平均 < 0.05 或异号）"))
    any_alive = any(verdict.values())
    A.ledger_add({"kind": "diag", "name": "controller_acf", "acf2160": {str(k): alive[k] for k in KS}, "alive": any_alive})
    if not any_alive and not a.force:
        say()
        say("判死：两段 B(k) 每日命中率在滞后 21~60 的平均自相关都在噪声水平以下，21 天前的真值对今天的命中率没有信息，")
        say("按 20 天到期真值调名额的控制器只能跟踪长期平均，不能跟踪月度漂移。不模拟（criteria 8a 写死）。")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        A.progress("controller_an：自相关判死，不模拟控制器 -> results_controller.txt")
        return
    say()
    say("== 第二步：模拟控制器（%s）" % ("--force，只作参考" if not any_alive else "自相关为正"))
    t = A.G.load_table(["date", "hit", "rank_ret20", "seg", "r20_open"])
    t = t[(t["rank_ret20"] <= 10) & (t["date"] >= "2021-07-01")]
    dates = np.array(sorted(t["date"].unique()))
    di = pd.Series(np.arange(len(dates)), index=dates)
    H = np.full((len(dates), 10), np.nan)
    H[t["date"].map(di).to_numpy(), t["rank_ret20"].to_numpy() - 1] = t["hit"].to_numpy()
    H = np.nan_to_num(H)
    for pstar in (0.15, 0.20):
        for variant in ("P", "PI"):
            lam = 0.5
            ks = np.zeros(len(dates), int)
            n_sel = np.zeros(len(dates))
            miss = np.zeros(len(dates))
            cum_n = cum_h = 0.0
            for i in range(len(dates)):
                if i >= 21 and n_sel[i - 21] > 0:
                    n, m = n_sel[i - 21], miss[i - 21]
                    err = (m - (1 - pstar) * n) / max(n, 1)
                    lam += 0.1 * err
                    cum_n += n
                    cum_h += n - m
                    if variant == "PI" and cum_n > 0:
                        lam += 0.02 * (1 if cum_h / cum_n < pstar else -1)
                    lam = min(max(lam, 0.0), 1.0)
                k = k_of(lam)
                ks[i] = k
                if k > 0:
                    n_sel[i] = k
                    miss[i] = k - H[i, :k].sum()
            rows = []
            for i in range(len(dates)):
                if ks[i] > 0:
                    for r in range(ks[i]):
                        rows.append((dates[i], H[i, r]))
            rows = pd.DataFrame(rows, columns=["date", "hit"])
            es = {}
            for seg in ("screen", "dev"):
                es[seg] = A.eval_rows(rows, seg)
                say("   " + A.fmt("控制器 p*=%.2f %s" % (pstar, variant), es[seg], seg))
                m = (dates >= A.SEGS[seg][0] + "-01") & (dates <= A.SEGS[seg][1] + "-31")
                say("      k_t 分布：%s" % {int(k): int(v) for k, v in zip(*np.unique(ks[m], return_counts=True))})
            if es["screen"]["V"] > 0 and es["dev"]["V"] > 0:
                j = A.combined_judge(es["screen"], es["dev"])
                say("   合并 ΔP %+.1fpp NW t %+.2f 块自举 p %.3f -> %s" % (100 * j["dP"], j["t"], j["boot_p"],
                                                                  A.usable(es["screen"], es["dev"], j, tau=pstar)))
            A.ledger_add({"kind": "look" if not any_alive else "screen", "name": "controller_p%.2f_%s" % (pstar, variant)})
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.progress("controller_an：自相关 + 控制器模拟 -> results_controller.txt")


if __name__ == "__main__":
    main()
