"""
参数拟合 + 走向前验证 + 按天自助。

速度上的关键设计：`Problem` 一次性把整张表 prepare 成列数组，并按日期排好
存成切片边界。之后每次目标函数求值只是「整表打一次分 + 按切片取子数组」，
1400 行/天 × 250 天大约 10 毫秒，Nelder-Mead 跑几百次迭代是秒级。
逐行调 score_one 的话同样的事要 25 分钟。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import cfg as C
from learn import objective as O
from learn import vscore

log = logging.getLogger(__name__)


class Problem:
    """一批天数上的目标函数。theta -> Ĝ / loss。"""

    def __init__(self, df: pd.DataFrame, base_cfg: dict, box: dict,
                 theta0: dict, theta_prev: dict, day_w: dict[str, float] | None,
                 k: int, huber_c: float, tau_tol: float = 0.01):
        d = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        self.dates = list(pd.unique(d["date"]))
        # 每天一段连续切片，避免每次求值都 groupby
        idx = d.index.to_numpy()
        starts = d["date"].ne(d["date"].shift()).to_numpy().nonzero()[0]
        ends = np.append(starts[1:], len(d))
        self.slices = list(zip(starts, ends))
        self.arr = vscore.prepare(d)
        self.ytil = d["ytil"].to_numpy(float)
        self.y = d["y"].to_numpy(float)
        self.c = base_cfg
        self.box, self.theta0, self.theta_prev = box, theta0, theta_prev
        self.k, self.huber_c, self.tau_tol = k, huber_c, tau_tol
        self.day_w = np.array([(day_w or {}).get(x, 1.0) for x in self.dates])
        self.keys = list(box.keys())
        del idx

    # -- 打分 --------------------------------------------------------
    def _day_arrays(self, theta: dict) -> tuple[list, list]:
        c = C.apply_theta(self.c, theta)
        s = vscore.score(self.arr, c)
        rej = vscore.hard_reject(self.arr, c["screen"])
        ds, dy = [], []
        for a, b in self.slices:
            m = ~rej[a:b]
            ds.append(s[a:b][m])
            dy.append(self.ytil[a:b][m])
        return ds, dy

    def G(self, theta: dict) -> tuple[float, np.ndarray]:
        ds, dy = self._day_arrays(theta)
        return O.G_hat(ds, dy, self.day_w, self.k, self.huber_c, self.tau_tol)

    def loss(self, theta: dict, lam_a: float, lam_1: float) -> float:
        g, _ = self.G(theta)
        return -g + O.penalty(theta, self.theta0, self.theta_prev,
                              self.box, lam_a, lam_1)

    # -- 硬指标（不参与优化，只汇报）---------------------------------
    def metrics(self, theta: dict, top_k: int = 10) -> dict:
        from learn.model_select import spearman
        c = C.apply_theta(self.c, theta)
        s = vscore.score(self.arr, c)
        rej = vscore.hard_reject(self.arr, c["screen"])
        ics, tops, hits, ns = [], [], [], []
        for a, b in self.slices:
            m = ~rej[a:b]
            sc, yt, yy = s[a:b][m], self.ytil[a:b][m], self.y[a:b][m]
            ns.append(int(m.sum()))
            if m.sum() < 3:
                continue
            ics.append(spearman(sc, yt))
            o = np.argsort(-sc)[:top_k]
            tops.append(float(yy[o].mean()))
            hits.append(float((yy[o] > 0).mean()))
        ic = np.array([x for x in ics if np.isfinite(x)])
        return {
            "days": len(self.dates),
            "ic_mean": float(ic.mean()) if ic.size else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if ic.size > 1 else float("nan"),
            "icir": float(ic.mean() / ic.std(ddof=1))
                    if ic.size > 1 and ic.std(ddof=1) > 0 else float("nan"),
            "top_excess": float(np.mean(tops)) if tops else float("nan"),
            "hit_rate": float(np.mean(hits)) if hits else float("nan"),
            "avg_pool": float(np.mean(ns)) if ns else 0.0,
        }

    def top_codes(self, theta: dict, codes: np.ndarray,
                  top_k: int = 10) -> dict[str, list[str]]:
        """每天的前 K 只代码。行为回放（闸门 5）用。"""
        c = C.apply_theta(self.c, theta)
        s = vscore.score(self.arr, c)
        rej = vscore.hard_reject(self.arr, c["screen"])
        out = {}
        for (a, b), day in zip(self.slices, self.dates):
            m = ~rej[a:b]
            sc, cd = s[a:b][m], codes[a:b][m]
            out[day] = list(cd[np.argsort(-sc)[:top_k]])
        return out


# ---------------------------------------------------------------------
#  拟合
# ---------------------------------------------------------------------
def fit(prob: Problem, lam_a: float, lam_1: float,
        maxiter: int = 600, n_starts: int = 5,
        seed: int = 7) -> dict[str, float]:
    """多起点 Nelder-Mead。8 维、目标函数不可导（softmax 里有二分求根，
    L1 在 θ_prev 处还有尖点），导数无关方法是对的选择。

    多起点是首次点火学到的教训：只从 θ_prev 出发，L1 的尖点让单纯形
    一步都迈不出去——每个小移动的罚分都大于局部目标改善，看起来像
    「没有信号」，其实是被自己的正则钉死在起点。从几个抖动过的起点
    再各跑一遍，如果确实存在罚分买得起的更优点，至少有一个起点在
    尖点外侧能滑进去；如果所有起点都收回 θ_prev，那才是真的没有信号。
    """
    from scipy.optimize import minimize
    keys = prob.keys
    rng = np.random.default_rng(seed)
    sig = O.sigma_of(prob.box)

    def f(x):
        th = O.project({k: float(v) for k, v in zip(keys, x)}, prob.box)
        return prob.loss(th, lam_a, lam_1)

    starts = [np.array([prob.theta_prev[k] for k in keys], float)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(np.array(
            [prob.theta_prev[k] + rng.normal(0, 0.15) * sig[k]
             for k in keys], float))

    best_x, best_v = starts[0], f(starts[0])
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead",
                     options={"maxiter": maxiter, "xatol": 1e-4,
                              "fatol": 1e-6})
        if r.fun < best_v - 1e-9:
            best_x, best_v = r.x, r.fun
    return O.project({k: float(v) for k, v in zip(keys, best_x)}, prob.box)


def sparsify(theta_fit: dict, theta_prev: dict, box: dict,
             max_moves: int, max_step_frac: float,
             min_frac: float = 0.02) -> tuple[dict, list[str]]:
    """把优化器的连续解投影成「最多 max_moves 个意图」的稀疏提案。

    为什么需要：Nelder-Mead 的单纯形在所有维度上一起挪，不会像坐标下降
    那样给出精确零。第三次点火实测：全部 9 个参数各漂一点，被闸门 4
    按「动了 9 个」拦下——闸门没错，是提案侧欠一步稀疏化。

    规则：
      按 |Δ|/σ 排序取前 max_moves 个（小于 min_frac·σ 的不算意图，是噪声）；
      每个意图的步长截到 max_step_frac × 箱宽；
      **权重类意图**改完后其余权重等比再归一——那是「和为 1」的必然结果，
      语义上仍是一个旋钮，闸门按意图数计数（见 gate.evaluate 的 intents 参数）。
    """
    sig = O.sigma_of(box)
    delta = {k: (theta_fit[k] - theta_prev[k]) / sig[k] for k in box}
    ranked = sorted((k for k in box if abs(delta[k]) >= min_frac),
                    key=lambda k: -abs(delta[k]))[:max_moves]
    t = dict(theta_prev)
    for k in ranked:
        lo, hi = box[k]
        step = np.clip(theta_fit[k] - theta_prev[k],
                       -max_step_frac * (hi - lo), max_step_frac * (hi - lo))
        t[k] = float(np.clip(theta_prev[k] + step, lo, hi))
    # 权重意图 -> 其余权重等比压缩/放大，保持和为 1
    wk = [k for k in box if k.startswith("scoring.weights.")]
    intent_w = [k for k in ranked if k in wk]
    if intent_w:
        fixed = sum(t[k] for k in intent_w)
        others = [k for k in wk if k not in intent_w]
        rest_prev = sum(theta_prev[k] for k in others)
        target = 1.0 - fixed
        if rest_prev > 0 and target > 0:
            for k in others:
                t[k] = theta_prev[k] * target / rest_prev
    return O.project(t, box), ranked


def split_days(dates: list[str], oos_frac: float) -> tuple[list, list]:
    """尾部 oos_frac 的天数留作样本外。时间序列不能随机切。"""
    n = len(dates)
    cut = max(1, int(round(n * (1 - oos_frac))))
    return dates[:cut], dates[cut:]


def bootstrap_better(prob_oos: Problem, theta_new: dict, theta_old: dict,
                     n: int = 2000, seed: int = 7) -> float:
    """按**天**重抽样，估 P(新参数的样本外 Ĝ 更好)。

    按天不按行：同一天内的收益高度相关，按行重抽会把置信区间算窄好几倍，
    于是什么改动看起来都显著。这是量化回测最常见的自欺方式之一。
    """
    _, g_new = prob_oos.G(theta_new)
    _, g_old = prob_oos.G(theta_old)
    diff = g_new - g_old
    rng = np.random.default_rng(seed)
    m = diff.size
    if m == 0:
        return 0.0
    if float(np.abs(diff).max()) < 1e-12:
        # 两组参数产生完全相同的组合：没有证据偏向任何一边。
        # 不处理的话全零差值算出 P=0，会被误读成「新参数明显更差」。
        return 0.5
    wins = 0
    for _ in range(n):
        s = diff[rng.integers(0, m, m)]
        if O.huber_location(s, None, prob_oos.huber_c) > 0:
            wins += 1
    return wins / n
