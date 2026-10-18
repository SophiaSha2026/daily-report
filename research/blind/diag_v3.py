"""V3 那一臂（前 20 里可信度最高的 1 只）的解剖：名单集中度、逐年、板块、机制对照（c 不用模型只用这只票自己的历史命中）、
邻域（K10 / K50，n1，order c）开发集看一次（记 look）。

    PYTHONIOENCODING=utf-8 python research/blind/diag_v3.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common_bl as B
from v3_bl import asof_cum


def cred(t, s, y, n0=20.0):
    ok = np.isfinite(s) & np.isfinite(y)
    ev = pd.DataFrame({"code": t["code"].to_numpy()[ok], "end20": t["end20"].to_numpy()[ok], "res": (y - s)[ok]})
    cum = asof_cum(t, ev, ["res"])
    return cum["res"].to_numpy() / (cum["cnt"].to_numpy() + n0), cum["cnt"].to_numpy()


def anatomy(t, sel, tag, L):
    for seg in ("screen", "dev"):
        m = sel & (t["seg"].to_numpy() == seg)
        r = t[m]
        u = r.groupby("code")["hit"].agg(["size", "sum"])
        top = u.sort_values("size", ascending=False).head(8)
        hits_u = r[r["hit"] > 0]["code"].nunique()
        L.append("  %s %s：出票 %d，不重复票 %d，命中 %d 来自 %d 只不同的票；最多的一只占 %d 次；前 8 只占 %.0f%% 的出票；板块 %s" % (
            tag, seg, len(r), len(u), int(r["hit"].sum()), hits_u, int(u["size"].max()), 100 * top["size"].sum() / len(r),
            r["board"].value_counts(normalize=True).round(2).to_dict()))
        # 连续同一只票的段：段数和段的命中（一个段算一次独立观察的粗略口径）
        rr = r.sort_values("date")
        new = (rr["code"] != rr["code"].shift(1)).to_numpy()
        seg_id = np.cumsum(new)
        g = rr.groupby(seg_id).agg(code=("code", "first"), n=("hit", "size"), h=("hit", "max"), hm=("hit", "mean"))
        L.append("    连续段 %d 个（中位长 %d 天），段内至少一次命中的段 %.1f%%；按段等权精确率 %.1f%%；逐年 %s" % (
            len(g), int(g["n"].median()), 100 * g["h"].mean(), 100 * g["hm"].mean(),
            {k: "%d/%.0f%%" % (v["size"], 100 * v["mean"]) for k, v in r.groupby(r["date"].str[:4])["hit"].agg(["size", "mean"]).iterrows()}))


def main():
    t0 = time.time()
    t = B.load_keys()
    s = B.load_scores("lgb_price").astype(float)
    y = t["y_open"].to_numpy(float)
    rank = t["rank_ret20"].to_numpy()
    L = ["V3 臂解剖  表 %d 行" % len(t)]
    c, hist_n = cred(t, s, y)
    cpct = B.day_pct(t, np.where(hist_n > 0, c, np.nan))
    base = (rank <= 20) & (cpct >= 0.9)
    sel = B.topn_by(t, base, c, 1)
    L.append("A. 正式那一臂 V3 K20 cp.9 n1 c")
    anatomy(t, sel, "V3", L)
    # 机制对照 1：c 不用模型，只用这只票自己过去的命中率（s 换成常数 = 该票之前全部行的全局基准率 0）
    c0, n0 = cred(t, np.zeros(len(t)), y)
    cp0 = B.day_pct(t, np.where(n0 > 0, c0, np.nan))
    sel0 = B.topn_by(t, (rank <= 20) & (cp0 >= 0.9), c0, 1)
    r0 = B.eval_both(t, sel0, boot=False)
    L.append("B. 机制对照：可信度只用这只票自己过去的命中频率 Σy/(n+20)（不用模型）")
    L.append(B.fmt_both("V3plain K20 n1", r0, dev=True))
    anatomy(t, sel0, "V3plain", L)
    B.ledger_add({"kind": "look", "what": "V3 机制对照：只用自身历史命中频率", "screen_dP": r0["screen"]["dP"], "dev_dP": r0["dev"]["dP"]})
    # 机制对照 2：残差只用过去 12 个月内关窗的行（近期可信度）；对照 3：n0 = 100（收缩更重）
    for tag, kw in (("c 窗口=全部 n0=100", dict(n0=100.0)),):
        c2, h2 = cred(t, s, y, **kw)
        cp2 = B.day_pct(t, np.where(h2 > 0, c2, np.nan))
        sel2 = B.topn_by(t, (rank <= 20) & (cp2 >= 0.9), c2, 1)
        r2 = B.eval_both(t, sel2, boot=False)
        L.append("C. %s" % tag)
        L.append(B.fmt_both("V3 n0=100 K20 n1", r2, dev=True))
        B.ledger_add({"kind": "look", "what": "V3 %s" % tag, "screen_dP": r2["screen"]["dP"], "dev_dP": r2["dev"]["dP"]})
    # 邻域（开发集看一次，记 look）：K10 / K50 n1 c；K20 n1 c 去掉 cp.9 过滤
    L.append("D. 邻域（开发集加看，记 look）")
    for K in (10, 50):
        selk = B.topn_by(t, (rank <= K) & (cpct >= 0.9), c, 1)
        rk = B.eval_both(t, selk, boot=True)
        L.append(B.fmt_both("V3 K%d cp.9 n1 c" % K, rk, dev=True))
        B.ledger_add({"kind": "look", "what": "V3 邻域 K%d n1 c" % K, "screen_dP": rk["screen"]["dP"], "dev_dP": rk["dev"]["dP"], "comb_t": rk["comb"]["t"]})
    # 第 2 名（n=2 的第二只单独看）：信号是不是只在第 1 只
    sel2nd = B.topn_by(t, base, c, 2) & ~sel
    r2nd = B.eval_both(t, sel2nd, boot=False)
    L.append("E. 只看可信度第 2 只（n=2 去掉第 1 只）")
    L.append(B.fmt_both("V3 K20 cp.9 第2只", r2nd, dev=True))
    B.ledger_add({"kind": "look", "what": "V3 第 2 只", "screen_dP": r2nd["screen"]["dP"], "dev_dP": r2nd["dev"]["dP"]})
    # F. 置换零分布：在前 20 里随机挑 1 只（10 个种子）的 ΔP，对照「同量名次基线」的难度
    rng = np.random.default_rng(3)
    vals = {"screen": [], "dev": []}
    for i in range(10):
        key = rng.random(len(t))
        selr = B.topn_by(t, rank <= 20, key, 1)
        rr = B.eval_both(t, selr, boot=False)
        for sg in vals:
            vals[sg].append(100 * rr[sg]["dP"])
    L.append("F. 前 20 里随机挑 1 只（10 个种子）的 ΔP 对 B(1)：筛选 %.1f±%.1fpp  开发 %.1f±%.1fpp" % (
        np.mean(vals["screen"]), np.std(vals["screen"]), np.mean(vals["dev"]), np.std(vals["dev"])))
    # G. 被挑中那只票的 ret20 名次分布、c 的大小、它当天 s(T) 的分位
    pct_T = B.day_pct(t, s)
    for seg in ("screen", "dev"):
        m = sel & (t["seg"].to_numpy() == seg)
        L.append("G. %s：被挑中票的 ret20 名次 中位 %d（1-3 名占 %.0f%%）；c 中位 %.4f；s(T) 当天分位中位 %.3f；历史行数中位 %d" % (
            seg, np.median(rank[m]), 100 * (rank[m] <= 3).mean(), np.median(c[m]), np.median(pct_T[m]), np.median(hist_n[m])))
    (B.HERE / "results_diag_v3.txt").write_text("\n".join(L), encoding="utf-8")
    B.say("\n".join(L))
    B.progress("diag_v3：V3 臂解剖 + 机制对照 + 邻域 look 4 次 -> results_diag_v3.txt（%.0fs）" % (time.time() - t0))


if __name__ == "__main__":
    main()
