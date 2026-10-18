"""
审计补充：真实序列上 NW 滞后 / 块长的敏感性。蒙特卡洛发现 20 天标签重叠下 Bartlett 滞后 20 与块长 20 把 se 低估约 17~20%，
这里在真实序列上量一下：rulecore ret20 − base 的按天差（969 天）、top50 的 B(10) / B(1) / B(2) 命中序列的影响函数 u0 = h − P·n。

    PYTHONIOENCODING=utf-8 python research/audit/stats/lag_sens.py  -> results_lag.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "precision", ROOT / "research" / "polish"):
    sys.path.insert(0, str(p))
import evalkit as K            # noqa: E402
import common_pr as C          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_lag.txt"
lines: list[str] = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def acf(x, kmax=40):
    x = np.asarray(x, float)
    xc = x - x.mean()
    g0 = (xc * xc).mean()
    return np.array([(xc[k:] * xc[:len(x) - k]).sum() / len(x) / g0 for k in range(1, kmax + 1)])


def bartlett_ratio(q):
    """MA(q) 等权相关下 Bartlett(L) 长期方差 / 真长期方差，L = 10..80。"""
    g = np.array([q + 1 - k for k in range(0, q + 1)], float)
    true = g[0] + 2 * g[1:].sum()
    out = {}
    for L in (10, 20, 30, 40, 60, 80):
        w = np.array([1 - k / (L + 1) for k in range(1, min(L, q) + 1)])
        out[L] = (g[0] + 2 * (w * g[1:len(w) + 1]).sum()) / true
    return out


def block_sd(x, block, b=4000, seed=1):
    x = np.asarray(x, float)
    idx = K._mbb_index(len(x), block, b, np.random.default_rng(seed))
    return float(x[idx].mean(axis=1).std(ddof=1))


def row(name, x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    se_iid = x.std(ddof=1) / np.sqrt(n)
    ses = {L: K._nw_se(x, L) for L in (10, 20, 30, 40, 60, 80)}
    bs = {bl: block_sd(x, bl) for bl in (20, 40, 60)}
    a = acf(x, 40)
    say("   %-34s n=%d 均值 %+.4f  iid se %.4f | NW se L=10 %.4f L=20 %.4f L=30 %.4f L=40 %.4f L=60 %.4f L=80 %.4f | 块自举 sd 块20 %.4f 块40 %.4f 块60 %.4f" % (
        name, n, m, se_iid, ses[10], ses[20], ses[30], ses[40], ses[60], ses[80], bs[20], bs[40], bs[60]))
    say("   %-34s t：iid %.2f  NW20 %.2f  NW40 %.2f  NW60 %.2f | ACF 1..5 %s  10 %.2f  15 %.2f  20 %.2f  25 %.2f  30 %.2f  40 %.2f  Σρ(1..20) %.2f Σρ(21..40) %.2f" % (
        "", m / se_iid, m / ses[20], m / ses[40], m / ses[60], " ".join("%.2f" % v for v in a[:5]), a[9], a[14], a[19], a[24], a[29], a[39], a[:20].sum(), a[20:40].sum()))


def main():
    say("== 理论：MA(q) 等权相关下 Bartlett(L) 估计的长期方差占真值的比例（se 比例取平方根）==")
    for q in (19, 10, 5):
        r = bartlett_ratio(q)
        say("   q=%d：%s" % (q, "  ".join("L=%d %.3f(se %.3f)" % (L, v, np.sqrt(v)) for L, v in r.items())))
    say()
    say("== 真实序列 ==")
    d = pd.read_parquet(EXP / "rulecore_v1" / "diff_ret20_10_y_open.parquet")
    row("rulecore ret20−base 按天差 969", d["d"].to_numpy())
    sg = d["date"].astype(str).str[:7] < "2024-06"
    row("  同上 筛选集 582", d["d"].to_numpy()[sg.to_numpy()])
    row("  同上 开发集 387", d["d"].to_numpy()[~sg.to_numpy()])
    dc = pd.read_parquet(EXP / "rulecore_confirm" / "diff_ret20_10_y_open.parquet")
    row("确认集 ret20−base 按天差 160", dc["d"].to_numpy())
    for a in ("rc", "pc"):
        dd = pd.read_parquet(EXP / "rulecore_v1" / f"diff_{a}_10_y_open.parquet")
        row("rulecore %s−base 969" % a, dd["d"].to_numpy())
    # 单臂命中序列（不是差）：ret20 前 10 每天命中数 / 10，base 同
    t = pd.read_parquet(EXP / "rulecore_v1" / "top_ret20.parquet")
    h = t[t["rank"] <= 10].groupby("date")["y_open"].mean()
    row("ret20 前 10 每天命中率（单臂）969", h.to_numpy())
    tb = pd.read_parquet(EXP / "rulecore_v1" / "top_base.parquet")
    hb = tb[tb["rank"] <= 10].groupby("date")["y_open"].mean()
    row("base 前 10 每天命中率（单臂）969", hb.to_numpy())
    # precision：B(k) 的影响函数 u0 = h − P n；B(2) − B(10) 的 u
    for seg in ("screen", "dev"):
        B = C.Baseline(C.load(seg))
        for k in (1, 2, 10):
            hk = B.hits_k(k)
            n = np.full(B.D, float(k))
            P = hk.sum() / n.sum()
            row("%s B(%d) 影响函数 u0" % (seg, k), hk - P * n)
        u = B.hits_k(2) / 2 - B.hits_k(10) / 10
        row("%s B(2)−B(10) 按天差" % seg, u)
    # 候选池基准率的日序列（全市场共同因子最强的那条）
    t50 = C.load()
    say()
    say("== 对已有结论的影响（按 L=40 / 块 40 重算的 t 和 p）==")
    res = json.loads((EXP / "rulecore_v1" / "results.json").read_text(encoding="utf-8"))
    for arm in ("ret20", "rc", "pc"):
        r = res.get(arm) or res.get("arms", {}).get(arm) or {}
        say("   rulecore results.json %s：t_nw %s  t_nw_lags %s  boot %s" % (arm, r.get("t_nw"), r.get("t_nw_lags"), r.get("boot")))
    x = d["d"].to_numpy()
    for L in (20, 30, 40, 60):
        say("   ret20−base 合并 NW t（L=%d）= %.2f；筛选集 %.2f；开发集 %.2f" % (
            L, K.nw_t(x, L), K.nw_t(x[sg.to_numpy()], L), K.nw_t(x[~sg.to_numpy()], L)))
    for bl in (20, 40, 60):
        say("   ret20−base 合并 块自举 p（块 %d）= %.4f；确认集 %.4f" % (bl, K.block_boot(x, block=bl)["p"], K.block_boot(dc["d"].to_numpy(), block=bl)["p"]))
    for L in (20, 40, 60):
        say("   确认集 ret20−base NW t（L=%d）= %.2f" % (L, K.nw_t(dc["d"].to_numpy(), L)))
    for seg in ("screen", "dev"):
        B = C.Baseline(C.load(seg))
        u = B.hits_k(2) / 2 - B.hits_k(10) / 10
        say("   %s B(2)−B(10)：NW t L=20 %.2f  L=40 %.2f  L=60 %.2f" % (seg, K.nw_t(u, 20), K.nw_t(u, 40), K.nw_t(u, 60)))
        for k in (1, 2, 10):
            hk = B.hits_k(k)
            n = np.full(B.D, float(k))
            for bl in (20, 40):
                lo, hi = C.boot_ratio(hk, n, block=bl)
                say("   %s B(%d) 块自举 95%% 区间 块%d [%.1f, %.1f]  单侧 5%% 分位 %.1f" % (
                    seg, k, bl, 100 * lo, 100 * hi, 100 * float(np.nanquantile(
                        (lambda idx: (hk[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)))(K._mbb_index(B.D, bl, 4000, np.random.default_rng(20261018))), 0.05))))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
