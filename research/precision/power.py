"""功效（在任何臂之前，只用基线前 50 的标签）：随机子集在各出票量下 ΔP 的 NW se、MDE80，和精确率自身的区间宽度。

    PYTHONIOENCODING=utf-8 python research/precision/power.py

随机子集：每天 n_d ~ Binomial(K, v̄/K)（K = 10 或 50），在前 K 里随机挑 n_d 只；对匹配量基线 B(k̄) 的 ΔP 的 NW se，
20 个种子取均值。另印设计效应 = (NW se / iid se)^2。
"""
from __future__ import annotations

import numpy as np

import common_pr as C


def main():
    C.P.wait_prod_window()
    t = C.load()
    L = ["功效表（criteria.txt 2f / 2g 的尺子；随机子集，20 个种子均值）"]
    for seg in ("screen", "dev"):
        s = t[t["seg"] == seg].reset_index(drop=True)
        B = C.Baseline(s)
        L.append("== %s（%d 天）==" % (seg, B.D))
        L.append("%-6s %-6s %6s %6s %8s %8s %9s %9s %10s" % ("子集", "v̄", "V", "出票天", "P%", "iid se", "NW se ΔP", "设计效应", "MDE80(2.5)"))
        for K in (10, 50):
            sub = s[s["rank_ret20"] <= K]
            for vbar in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
                ses, Ps, Vs, days, iids = [], [], [], [], []
                for seed in range(20):
                    rng = np.random.default_rng(1000 * K + seed)
                    n_d = rng.binomial(K, vbar / K, size=B.D)
                    di = dict(zip(B.dates, range(B.D)))
                    sel = np.zeros(len(sub), bool)
                    # 每天随机挑 n_d 行
                    order = rng.random(len(sub))
                    r = sub.assign(_o=order)
                    r["_rk"] = r.groupby("date")["_o"].rank(method="first")
                    need = r["date"].map(lambda d: n_d[di[d]]).to_numpy()
                    sel = (r["_rk"].to_numpy() <= need)
                    full = np.zeros(len(s), bool)
                    full[sub.index[sel]] = True
                    e = C.evaluate(s, full, B, boot=False)
                    ses.append(100 * e["se"])
                    Ps.append(e["P"])
                    Vs.append(e["V"])
                    days.append(e["days_out"])
                    p0 = e["P"]
                    iids.append(100 * np.sqrt(p0 * (1 - p0) / max(e["V"], 1)))
                se = float(np.mean(ses))
                iid = float(np.mean(iids))
                L.append("前%-3d %5.2f %6.0f %6.0f %7.1f %8.2f %9.2f %9.1f %10.2f" % (
                    K, vbar, np.mean(Vs), np.mean(days), 100 * np.mean(Ps), iid, se, (se / iid) ** 2 if iid > 0 else np.nan, C.mde80(se)))
        # 精确率自身的区间宽度（块自举）在门槛量级
        L.append("  精确率自身 95% 块自举区间半宽（随机前 10 子集，5 个种子均值）：")
        for vbar in (0.5, 1.0, 2.0):
            half = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                n_d = rng.binomial(10, vbar / 10, size=B.D)
                sub = s[s["rank_ret20"] <= 10]
                di = dict(zip(B.dates, range(B.D)))
                r = sub.assign(_o=rng.random(len(sub)))
                r["_rk"] = r.groupby("date")["_o"].rank(method="first")
                need = r["date"].map(lambda d: n_d[di[d]]).to_numpy()
                full = np.zeros(len(s), bool)
                full[sub.index[r["_rk"].to_numpy() <= need]] = True
                e = C.evaluate(s, full, B, boot=True)
                half.append(50 * (e["P_hi"] - e["P_lo"]))
            L.append("    v̄ %.1f（V≈%d）：±%.1fpp" % (vbar, int(vbar * B.D), np.mean(half)))
    txt = "\n".join(L)
    C.say(txt)
    (C.HERE / "results_power.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("功效表 results_power.txt 算完（随机子集 se / MDE80 / 设计效应 / 区间半宽）")


if __name__ == "__main__":
    main()
