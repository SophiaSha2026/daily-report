"""
审计第 1~3 项：用答案已知的合成数据验证仓库里的统计实现（不改生产代码，只调用）。

    PYTHONIOENCODING=utf-8 python research/audit/stats/mc_stats.py  -> results_mc.txt

1 精确率 P = Σh/Σn 的 NW se（按天影响函数 u_d = h_d − P n_d，Bartlett 滞后 20）和块自举比值区间的覆盖率，
  三种数据生成：独立 / AR(1) 日级相关 / 按天聚集 + 20 天标签重叠。
2 匹配量基线 B(k̄)：分数 k 线性插值 = 「每天随机多取一只」的期望；零假设「规则 = 前 K 的随机子集、命中率沿名次平坦」
  下 ΔP 以 0 为中心、t 的尺度；名次曲线不平坦时 ΔP 的中心（设计上就不是 0，记下来）。
3 evalkit 的 NW t / 块自举 p / _maxt_null 族门槛 / Bonferroni 并集上界 / 去最好月 / 滚动三月 / v2 v3 判定：
  纯噪声臂的假阳率；已知效应 = MDE80 的臂的功效。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "precision", ROOT / "research" / "polish"):
    sys.path.insert(0, str(p))
import evalkit as K            # noqa: E402
import common_pr as C          # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_mc.txt"
ND = NormalDist()
lines: list[str] = []


def say(s: str = "") -> None:
    lines.append(s)
    print(s, flush=True)


def progress(s: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), s))


# ---------------------------------------------------------------
#  数据生成
# ---------------------------------------------------------------
def gen_overlap(rng, D: int, n_d: np.ndarray, p_target: float, sig_m: float = 0.012, sig_e: float = 0.03,
                thr: float | None = None, horizon: int = 20):
    """按天聚集 + 20 天标签重叠：第 d 天第 i 只的标签 = 1[max_k Σ_{j<=k}(m_{d+j} + e_ij) > thr]，
    m 是全市场日收益（所有票共享，相邻天的标签窗口重叠 19 天），e 是个股噪声。"""
    T = D + horizon + 1
    m = rng.normal(0, sig_m, T)
    cm = np.concatenate([[0.0], np.cumsum(m)])
    h = np.zeros(D)
    for d in range(D):
        n = int(n_d[d])
        if n == 0:
            continue
        e = rng.normal(0, sig_e, (n, horizon))
        path = (cm[d + 2:d + 2 + horizon] - cm[d + 1])[None, :] + np.cumsum(e, axis=1)
        h[d] = (path.max(axis=1) > thr).sum()
    return h


def calib_thr(rng, p_target, sig_m, sig_e, horizon=20):
    e = rng.normal(0, sig_e, (200000, horizon))
    m = rng.normal(0, sig_m, (200000, horizon))
    path = np.cumsum(e + m, axis=1).max(axis=1)
    return float(np.quantile(path, 1 - p_target))


def true_P_overlap(rng, thr, sig_m, sig_e, horizon=20, n=400000):
    e = rng.normal(0, sig_e, (n, horizon))
    m = rng.normal(0, sig_m, (n, horizon))
    return float((np.cumsum(e + m, axis=1).max(axis=1) > thr).mean())


def gen_ar1(rng, D, n_d, p, rho, sd_logit=0.6):
    z = np.empty(D)
    z[0] = rng.normal(0, sd_logit)
    for t in range(1, D):
        z[t] = rho * z[t - 1] + rng.normal(0, sd_logit * np.sqrt(1 - rho ** 2))
    pd_ = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + z)))
    return rng.binomial(n_d.astype(int), pd_).astype(float), pd_


def P_and_se(h, n, D):
    V = n.sum()
    P = h.sum() / V
    u0 = h - P * n
    se = C.nw_se(u0) * D / V               # common_pr.evaluate 里 P_se 的公式
    return P, se


# ---------------------------------------------------------------
#  1 精确率与区间
# ---------------------------------------------------------------
def part1(R=400, B=1500):
    say("== 1 精确率 P 的 NW se 与块自举区间（覆盖率，名义 95%）==")
    say("   每格 R=%d 次重复，块自举 b=%d（生产 4000），区间：正态 P±1.96·NW se、块自举 2.5/97.5 分位、iid 二项 se 对照" % (R, B))
    rng = np.random.default_rng(20261018)
    p = 0.15
    sig_m, sig_e = 0.012, 0.03
    thr = calib_thr(rng, p, sig_m, sig_e)
    P_true_ov = true_P_overlap(rng, thr, sig_m, sig_e)
    say("   重叠 DGP：市场日波动 %.1f%%、个股 %.1f%%，阈值 %.3f，真 P = %.4f" % (100 * sig_m, 100 * sig_e, thr, P_true_ov))
    rows = []
    for D in (582, 387):
        for nd_kind in ("固定10", "固定1", "Binom(10,0.1)"):
            for dgp in ("独立", "AR1 rho=0.5", "重叠20天"):
                cover_n = cover_b = cover_iid = 0
                Ps, ses, los, his, ses_iid = [], [], [], [], []
                for r in range(R):
                    if nd_kind == "固定10":
                        n_d = np.full(D, 10.0)
                    elif nd_kind == "固定1":
                        n_d = np.full(D, 1.0)
                    else:
                        n_d = rng.binomial(10, 0.1, D).astype(float)
                        if n_d.sum() == 0:
                            n_d[0] = 1
                    if dgp == "独立":
                        h = rng.binomial(n_d.astype(int), p).astype(float)
                        Pt = p
                    elif dgp.startswith("AR1"):
                        h, _ = gen_ar1(rng, D, n_d, p, 0.5)
                        Pt = None
                    else:
                        h = gen_overlap(rng, D, n_d, p, sig_m, sig_e, thr)
                        Pt = P_true_ov
                    Pe, se = P_and_se(h, n_d, D)
                    lo, hi = C.boot_ratio(h, n_d, b=B, block=20, seed=int(rng.integers(1 << 30)))
                    V = n_d.sum()
                    se_iid = np.sqrt(max(Pe * (1 - Pe), 1e-9) / V)
                    Ps.append(Pe)
                    ses.append(se)
                    ses_iid.append(se_iid)
                    los.append(lo)
                    his.append(hi)
                if Pt is None:
                    # AR(1) 的真 P：E[p_d]，按同一分布大样本算
                    _, pdd = gen_ar1(rng, 200000, np.ones(200000), p, 0.5)
                    Pt = float(pdd.mean())
                Ps, ses, los, his, ses_iid = map(np.array, (Ps, ses, los, his, ses_iid))
                cover_n = np.mean(np.abs(Ps - Pt) <= 1.96 * ses)
                cover_b = np.mean((los <= Pt) & (Pt <= his))
                cover_iid = np.mean(np.abs(Ps - Pt) <= 1.96 * ses_iid)
                sd_true = Ps.std(ddof=1)
                rows.append((D, nd_kind, dgp, Pt, Ps.mean(), sd_true, ses.mean(), ses_iid.mean(), cover_n, cover_b, cover_iid,
                             np.mean(his - los) / 2))
                say("   D=%d %-14s %-12s 真P %.4f 均值 %.4f | sd(P̂) %.4f  NW se 均值 %.4f  iid se %.4f | 覆盖 正态 %.3f 块自举 %.3f iid %.3f | 自举半宽 %.4f" % rows[-1])
    progress("mc 第 1 项（覆盖率）算完")
    return rows


# ---------------------------------------------------------------
#  2 匹配量基线
# ---------------------------------------------------------------
def part2(R=600):
    say()
    say("== 2 匹配量基线 B(k̄) ==")
    rng = np.random.default_rng(7)
    D, Kp = 582, 50
    # 2a 插值 = 随机多取一只的期望
    probs = 0.18 * np.exp(-np.arange(Kp) / 25) + 0.03
    H = (rng.random((D, Kp)) < probs[None, :]).astype(float)
    t = pd.DataFrame({"date": np.repeat([f"d{i:04d}" for i in range(D)], Kp), "rank_ret20": np.tile(np.arange(1, Kp + 1), D),
                      "hit": H.ravel()})
    Bz = C.Baseline(t)
    for k in (0.3, 1.0, 1.4, 2.5, 7.9):
        f, frac = int(np.floor(k)), k - int(np.floor(k))
        exp_rand = Bz.cum[:, f - 1].copy() if f >= 1 else np.zeros(D)
        if frac > 0:
            # 每天以概率 frac 多取第 f+1 名：期望 = frac × H[:, f]；蒙特卡洛 20000 次
            draws = (rng.random((20000, D)) < frac)
            exp_mc = exp_rand[None, :] + draws * H[None, :, f]
            diff = abs(exp_mc.mean(axis=0).sum() - Bz.hits_k(k).sum()) / max(Bz.hits_k(k).sum(), 1)
        else:
            diff = abs(exp_rand.sum() - Bz.hits_k(k).sum())
        say("   2a k=%.1f  hits_k 合计 %.2f  随机多取一只的期望（MC）相对差 %.4f  P_B(k)=%.4f" % (k, Bz.hits_k(k).sum(), diff, Bz.precision_k(k)))
    # 2b 零假设：规则 = 前 K 的随机子集，命中率沿名次平坦（p 相同）-> ΔP 应以 0 为中心，t ~ N(0,1)
    for K_, kbar in ((10, 1.0), (10, 2.0), (50, 1.0), (10, 10.0)):
        for dgp in ("独立", "重叠20天"):
            dPs, ts, tws = [], [], []
            sig_m, sig_e = 0.012, 0.03
            thr = calib_thr(rng, 0.15, sig_m, sig_e) if dgp != "独立" else None
            for r in range(R):
                if dgp == "独立":
                    Hd = (rng.random((D, Kp)) < 0.15).astype(float)
                else:
                    # 全部 50 名用同一条市场路径：先造 D×50 只票的标签
                    T = D + 21
                    m = rng.normal(0, sig_m, T)
                    cm = np.concatenate([[0.0], np.cumsum(m)])
                    Hd = np.zeros((D, Kp))
                    for d in range(D):
                        e = rng.normal(0, sig_e, (Kp, 20))
                        path = (cm[d + 2:d + 22] - cm[d + 1])[None, :] + np.cumsum(e, axis=1)
                        Hd[d] = path.max(axis=1) > thr
                td = pd.DataFrame({"date": np.repeat([f"d{i:04d}" for i in range(D)], Kp), "rank_ret20": np.tile(np.arange(1, Kp + 1), D),
                                   "hit": Hd.ravel()})
                Bq = C.Baseline(td)
                n_d = rng.binomial(K_, kbar / K_, D)
                sel = np.zeros(len(td), bool)
                rk = td["rank_ret20"].to_numpy()
                for d in range(D):
                    if n_d[d]:
                        pick = rng.choice(K_, n_d[d], replace=False)
                        sel[d * Kp + pick] = True
                e = C.evaluate(td, sel, Bq, boot=False)
                dPs.append(e["dP"])
                ts.append(e["t"])
                tws.append(e["t_within"])
            dPs, ts, tws = map(np.array, (dPs, ts, tws))
            ts = ts[np.isfinite(ts)]
            say("   2b K=%d k̄=%.0f %-8s ΔP 均值 %+.4f (se %.4f)  t 均值 %+.3f sd %.3f  P(|t|>1.96) %.3f  P(t>2.5) %.3f | 日配 t 均值 %+.3f sd %.3f" % (
                K_, kbar, dgp, dPs.mean(), dPs.std(ddof=1) / np.sqrt(len(dPs)), ts.mean(), ts.std(ddof=1),
                np.mean(np.abs(ts) > 1.96), np.mean(ts > 2.5), np.nanmean(tws), np.nanstd(tws, ddof=1)))
    # 2c 名次曲线不平坦时（真实情形）：前 10 随机子集对 B(k̄) 的 ΔP 中心 = P(前10) − P_B(k̄)，是负的（设计如此）
    dPs = []
    for r in range(200):
        Hd = (rng.random((D, Kp)) < probs[None, :]).astype(float)
        td = pd.DataFrame({"date": np.repeat([f"d{i:04d}" for i in range(D)], Kp), "rank_ret20": np.tile(np.arange(1, Kp + 1), D),
                           "hit": Hd.ravel()})
        Bq = C.Baseline(td)
        n_d = rng.binomial(10, 0.1, D)
        sel = np.zeros(len(td), bool)
        for d in range(D):
            if n_d[d]:
                sel[d * Kp + rng.choice(10, n_d[d], replace=False)] = True
        dPs.append(C.evaluate(td, sel, Bq, boot=False)["dP"])
    say("   2c 名次曲线递减（1 名 %.1f%% -> 10 名 %.1f%%）：前 10 随机子集 k̄=1 对 B(1) 的 ΔP 均值 %+.4f（理论 %.4f − %.4f = %+.4f）" % (
        100 * probs[0], 100 * probs[9], np.mean(dPs), probs[:10].mean(), probs[0], probs[:10].mean() - probs[0]))
    say("      读法：零假设「随机子集」对 B(k̄) 为负是尺子的本意（随机缩名额输给按名次缩），不是 bug；平坦曲线下中心为 0。")
    progress("mc 第 2 项（B(k̄)）算完")


# ---------------------------------------------------------------
#  3 NW t / 块自举 / 族门槛 / 判定
# ---------------------------------------------------------------
def ar_series(rng, n, rho, sd=1.0, common=None, load=0.0):
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    x = x * sd
    if common is not None:
        x = np.sqrt(1 - load ** 2) * x + load * common
    return x


def ma_overlap_series(rng, n, q=19, sd=1.0):
    """MA(q)：相邻 20 天标签窗口重叠造成的相关（差值序列的近似）。"""
    e = rng.normal(0, 1, n + q)
    w = np.ones(q + 1) / np.sqrt(q + 1)
    x = np.convolve(e, w, mode="valid")
    return x * sd


def lrv_ma(q, sd=1.0):
    """MA(q) 等权核的长期方差：(Σw)^2 = (q+1)·w^2 = 1 （w = 1/sqrt(q+1)）-> 每个 w^2 乘 ...  精确算。"""
    w = np.ones(q + 1) / np.sqrt(q + 1)
    return sd ** 2 * w.sum() ** 2


def lrv_ar1(rho, sd=1.0):
    return sd ** 2 * (1 + rho) / (1 - rho)


def part3(R=400, Rfam=200):
    say()
    say("== 3 evalkit：NW t 的尺度、块自举 p 的检验水平、族门槛的族错误率、功效 ==")
    rng = np.random.default_rng(99)
    # 3a 单臂：纯噪声 d 序列下 NW t 的 sd、P(t>=2.5)、块自举 p<0.05 的频率、去最好月 t>=2 的频率
    for n in (969, 387):
        for name, mk, lrv in (("iid", lambda: rng.normal(0, 1, n), 1.0),
                              ("AR1 0.3", lambda: ar_series(rng, n, 0.3), lrv_ar1(0.3)),
                              ("AR1 0.6", lambda: ar_series(rng, n, 0.6), lrv_ar1(0.6)),
                              ("MA19 重叠", lambda: ma_overlap_series(rng, n), lrv_ma(19))):
            ts, tiid, bp, tdrop, r3ok, pass_v3 = [], [], [], [], [], []
            dates = pd.date_range("2022-01-03", periods=n, freq="B").strftime("%Y-%m-%d")
            for r in range(R):
                x = mk()
                ts.append(K.nw_t(x))
                tiid.append(x.mean() / (x.std(ddof=1) / np.sqrt(n)))
                bb = K.block_boot(x, b=800, seed=int(rng.integers(1 << 30)))
                bp.append(bb["p"])
                d = pd.Series(x, index=dates)
                dm = d.groupby(d.index.str[:7]).mean()
                best = dm.idxmax() if x.mean() >= 0 else dm.idxmin()
                td_ = K.nw_t(d[d.index.str[:7] != best].to_numpy())
                tdrop.append(td_)
                r3, _ = K.roll3_min(dm)
                r3ok.append(r3)
            ts, tiid, bp, tdrop = map(np.array, (ts, tiid, bp, tdrop))
            # 真 se = sqrt(lrv/n)；NW t 的理论尺度 1
            say("   3a n=%d %-10s NW t sd %.3f（理论 1；iid t sd %.3f） P(NW t>=2.5) %.4f（正态 0.0062） P(|t|>1.96) %.3f  "
                "块自举 p<0.05 %.3f  去最好月 t>=2.0 %.3f  去最好月 t<=-2.0 %.3f" % (
                    n, name, ts.std(ddof=1), tiid.std(ddof=1), np.mean(ts >= 2.5), np.mean(np.abs(ts) > 1.96),
                    np.mean(bp < 0.05), np.mean(tdrop >= 2.0), np.mean(tdrop <= -2.0)))
    # 3b 族门槛：s 条有序列的零臂（互相相关 + 自相关）+ u 个没序列的；FWER 是否 <= 5%
    say()
    say("   3b 族门槛（_maxt_null 联合块自举 + Bonferroni 并集上界）：s 条零臂序列（AR1 0.3，臂间相关 0.5，n=969）+ u 个无序列零臂")
    for s, u in ((8, 0), (8, 10), (20, 50), (3, 0), (1, 70)):
        n = 969
        fw_c, fw_bonf, fw_maxt, fw_floor, cs, cms, cbs = [], [], [], [], [], [], []
        for r in range(Rfam):
            common = ar_series(rng, n, 0.3)
            X = np.column_stack([ar_series(rng, n, 0.3, common=common, load=np.sqrt(0.5)) for _ in range(s)])
            D = pd.DataFrame(X, columns=[f"a{i}" for i in range(s)])
            t_obs = np.array([K.nw_t(X[:, j]) for j in range(s)])
            # 没序列的 u 个臂：独立零臂的 NW t（也用同样的 AR1 序列算，门槛那边只按正态尾算它们）
            t_u = np.array([K.nw_t(ar_series(rng, n, 0.3)) for _ in range(u)]) if u else np.array([])
            m = s + u
            mx = K._maxt_null(D, b=1000, seed=int(rng.integers(1 << 30)))
            c_bonf = ND.inv_cdf(1 - 0.05 / m)
            c_maxt = float(np.quantile(mx, 0.95))

            def excess(c):
                return float((mx >= c).mean()) + u * (1 - ND.cdf(c))
            lo, hi = 0.0, 8.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if excess(mid) <= 0.05:
                    hi = mid
                else:
                    lo = mid
            c = min(hi, c_bonf)
            c_final = max(2.5, c)
            allt = np.concatenate([t_obs, t_u])
            fw_c.append(allt.max() >= c)
            fw_floor.append(allt.max() >= c_final)
            fw_bonf.append(allt.max() >= c_bonf)
            fw_maxt.append(t_obs.max() >= c_maxt)
            cs.append(c)
            cms.append(c_maxt)
            cbs.append(c_bonf)
        say("      s=%2d u=%2d  门槛 c 均值 %.3f（c_maxt %.3f, c_bonf %.3f, 带 2.5 下限后 %.3f）  FWER：并集上界 c %.3f  Bonferroni %.3f  只看有序列臂对 c_maxt %.3f  带下限 %.3f" % (
            s, u, np.mean(cs), np.mean(cms), np.mean(cbs), np.mean(np.maximum(2.5, cs)), np.mean(fw_c), np.mean(fw_bonf), np.mean(fw_maxt), np.mean(fw_floor)))
    # 3c 功效：真效应 = MDE80 = (2.5 + 0.84) × se_true 时 P(NW t >= 2.5) 应约 80%
    say()
    say("   3c 功效：真效应 = (c + 0.84)·se_true，c = 2.5；se_true = sqrt(LRV/n)")
    for n in (969, 582, 387):
        for name, rho in (("iid", 0.0), ("AR1 0.3", 0.3), ("MA19", None)):
            if rho is None:
                lrv = lrv_ma(19)
            else:
                lrv = lrv_ar1(rho) if rho > 0 else 1.0
            se_true = np.sqrt(lrv / n)
            eff = (2.5 + 0.84) * se_true
            pw, pw_boot, pw_v3 = [], [], []
            dates = pd.date_range("2022-01-03", periods=n, freq="B").strftime("%Y-%m-%d")
            for r in range(R):
                x = (ma_overlap_series(rng, n) if rho is None else ar_series(rng, n, rho)) + eff
                t = K.nw_t(x)
                pw.append(t >= 2.5)
                bb = K.block_boot(x, b=600, seed=int(rng.integers(1 << 30)))
                d = pd.Series(x, index=dates)
                dm = d.groupby(d.index.str[:7]).mean()
                best = dm.idxmax()
                tdrop = K.nw_t(d[d.index.str[:7] != best].to_numpy())
                r3, _ = K.roll3_min(dm)
                pw_v3.append(t >= 2.5 and bb["p"] < 0.05 and tdrop >= 2.0 and r3 >= -0.05 * (eff / 0.03))  # 滚动三月按 pp 口径：这里 d 的单位是 se，不换算，只看 t 类条件
                pw_boot.append(bb["p"] < 0.05)
            say("      n=%d %-8s se_true %.4f 效应 %.4f  P(NW t>=2.5) %.3f（名义 0.80）  块自举 p<0.05 %.3f  t&boot&去最好月 %.3f" % (
                n, name, se_true, eff, np.mean(pw), np.mean(pw_boot), np.mean(pw_v3)))
    progress("mc 第 3 项（evalkit 门槛）算完")


if __name__ == "__main__":
    t0 = time.time()
    part1()
    part2()
    part3()
    say()
    say("用时 %.0fs" % (time.time() - t0))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", OUT)
