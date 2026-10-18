"""嵌套逐月滚动读法（修正版：基线按「应用月实际出票量」匹配，diag.py 第 D 节那版把回看窗口的 k̄ 拿去配了应用月，基线量和实际量不等）。

    PYTHONIOENCODING=utf-8 python research/precision/nested.py

每月用之前 12 个月（跨段连续：开发集头 12 个月用筛选集末尾）在网格里选「ΔP 最大且 v̄ >= 0.5、出票天 >= 1/6」的配置，
用于下个月。两段各印：出票量、精确率、同量名次基线、ΔP、NW t（按天 u 序列）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common_pr as C


def main():
    C.P.wait_prod_window()
    z = np.load(C.HERE / "grid_hits.npz", allow_pickle=True)
    names = list(z["names"])
    dj = json.load((C.HERE / "grid_dates.json").open())
    t = C.load()
    segs = {s: t[t["seg"] == s].reset_index(drop=True) for s in ("screen", "dev")}
    Bs = {s: C.Baseline(segs[s]) for s in segs}
    H = np.concatenate([z["screen__h__x"], z["dev__h__x"]], axis=1)
    N = np.concatenate([z["screen__n__x"], z["dev__n__x"]], axis=1)
    dates = np.array(dj["screen_dates"] + dj["dev_dates"])
    Hb = np.concatenate([Bs["screen"].H, Bs["dev"].H], axis=0)          # D × 50 名次命中
    cumHb = np.cumsum(Hb, axis=1)
    months = np.array([d[:7] for d in dates])
    um = sorted(set(months))

    def base_hits(k: float, mask: np.ndarray) -> np.ndarray:
        f = int(np.floor(k))
        frac = k - f
        h = cumHb[mask, f - 1].copy() if f >= 1 else np.zeros(mask.sum())
        if frac > 1e-12 and f < 50:
            h += frac * Hb[mask, f]
        return h

    L = ["嵌套逐月滚动读法（每月用前 12 个月选配置；基线按应用月实际出票量匹配）"]
    out = {}
    for s, (m0, m1) in (("screen", ("2023-01", "2024-05")), ("dev", ("2024-06", "2025-12"))):
        u_all, h_all, n_all, picks = [], [], [], []
        for i, m in enumerate(um):
            if m < m0 or m > m1:
                continue
            win = np.isin(months, um[i - 12:i])
            Dw = int(win.sum())
            dP = np.full(len(names), -np.inf)
            for j in range(len(names)):
                Vw = N[j, win].sum()
                if Vw < 0.5 * Dw or (N[j, win] > 0).sum() < Dw / 6:
                    continue
                dP[j] = (H[j, win].sum() - base_hits(Vw / Dw, win).sum()) / Vw
            j = int(np.argmax(dP))
            cur = months == m
            Vc = N[j, cur].sum()
            Dc = int(cur.sum())
            hb = base_hits(Vc / Dc, cur) if Vc > 0 else np.zeros(Dc)
            u_all.append(H[j, cur] - hb)
            h_all.append(H[j, cur])
            n_all.append(N[j, cur])
            picks.append((m, names[j], float(dP[j]), int(Vc), int(H[j, cur].sum())))
        u = np.concatenate(u_all)
        h = np.concatenate(h_all)
        n = np.concatenate(n_all)
        V = n.sum()
        Pr = h.sum() / V
        lo, hi = C.boot_ratio(h, n)
        L.append("  %s %s..%s：%d 个月 %d 天，出票 %d 只（%.2f 只/天，空榜 %.0f%%），精确率 %.1f%% [%.1f, %.1f]，同量名次基线 %.1f%%，ΔP %+.1fpp（NW t %+.2f）" % (
            s, m0, m1, len(picks), len(n), int(V), V / len(n), 100 * (n == 0).mean(), 100 * Pr, 100 * lo, 100 * hi,
            100 * (h.sum() - u.sum()) / V, 100 * u.sum() / V, C.nw_t(u)))
        cnt = pd.Series([p[1] for p in picks]).value_counts()
        L.append("    选中最多：%s" % cnt.head(6).to_dict())
        L.append("    逐月：" + "；".join("%s %s %d/%d" % (m, nm, hh, vv) for m, nm, _, vv, hh in picks))
        out[s] = {"V": int(V), "P": Pr, "dP": float(u.sum() / V), "t": C.nw_t(u)}
    txt = "\n".join(L)
    C.say(txt)
    (C.HERE / "results_nested.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("嵌套逐月（修正基线匹配）：筛选 %+.1fpp (t %.2f)，开发 %+.1fpp (t %.2f)" % (
        100 * out["screen"]["dP"], out["screen"]["t"], 100 * out["dev"]["dP"], out["dev"]["t"]))


if __name__ == "__main__":
    main()
