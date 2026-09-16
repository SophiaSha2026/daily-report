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


# ---------------------------------------------------------------------
#  生产口径的「当天实发清单」
# ---------------------------------------------------------------------
def production_order(scores: np.ndarray, rej: np.ndarray, c: dict,
                     top_n: int | None = None) -> np.ndarray:
    """当天真正会发出去的那张榜的下标，按分降序。

    生产链路是 score.py::rank + run_auction.select：
        未被硬性排除 且 round(score, 1) >= output.min_score，再取前 top_n。
    学习线以前只用 `~rej` 取前 K，等于把从没发过信的低分票算进成绩和闸门 5
    的换手里：404 天里 97 天（23.5%）两张榜不是同一批票，在线 17 天里 4 天
    不同（2026-09-16 实测：09-11 池 10 只 / 实发 5 只，09-14 15 / 8，09-16 12 / 6）。
    这就是教训 30「回测口径和生产口径差一点，成绩就不是同一件事」。

    round 到 0.1 不能省：score.py 存的 `score` 字段是 round(raw, 1)，44.96 分
    在生产是过线的；排序用未取整的分（score_raw），并列时按候选池原顺序，
    所以这里用 stable 排序。
    """
    out = c["output"]
    s = np.asarray(scores, float)
    n = int(out["top_n"] if top_n is None else top_n)
    ok = (~np.asarray(rej, bool)) & (np.round(s, 1) >= float(out["min_score"]))
    idx = np.flatnonzero(ok)
    return idx[np.argsort(-s[idx], kind="stable")][:n]


def production_top(scores: np.ndarray, rej: np.ndarray, c: dict,
                   top_n: int | None = None) -> np.ndarray:
    """production_order 的布尔掩码版。"""
    m = np.zeros(len(np.asarray(scores)), bool)
    m[production_order(scores, rej, c, top_n)] = True
    return m


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
        # 汇报口径用**未缩尾**的 y_raw：缩尾（winsor_q=0.01）是给目标函数
        # 防梯度翻转的，不该出现在「前 10 超额」这种给人看的数字里。
        # 在线 17 天 168 只选中票有 10 只被截（09-11 汇报 -5.826% 实为
        # -6.057%），回填 399 天 3907 只里 199 只被截，单日最大偏差 1.78pp。
        # 旧训练表没这一列就退回 y——退回是为了不崩，不是为了对（审计 F2-10）。
        self.y_raw = (d["y_raw"].to_numpy(float) if "y_raw" in d.columns
                      else self.y)
        # cost_bp 是单边（万分之 13：印花税 10 + 佣金等），一买一卖各一次。
        # 常数，不进优化目标（不改排序），只在汇报里扣。
        self.cost = 2.0 * float(
            ((base_cfg.get("learning") or {}).get("label") or {})
            .get("cost_bp", 0.0)) / 1e4
        self.c = base_cfg
        self.box, self.theta0, self.theta_prev = box, theta0, theta_prev
        self.k, self.huber_c, self.tau_tol = k, huber_c, tau_tol
        self.day_w = np.array([(day_w or {}).get(x, 1.0) for x in self.dates])
        self.keys = list(box.keys())
        # 非有限分数/收益的行数（见 _mask）。0 以外的值要能被界面查到，
        # 只写日志等于没写（教训 16）
        self.n_nonfinite = 0
        del idx

    # -- 打分 --------------------------------------------------------
    def _mask(self, s: np.ndarray, rej: np.ndarray) -> np.ndarray:
        """参与目标函数/指标的行：过准入 **且** 分数与收益都是有限值。

        缺了 isfinite 这一半的后果是静默的：一个 NaN 分数让 solve_tau 返回
        NaN，day_G 把它当成「池子不够 k 只」退化成等权 mean(ỹ)，那一天对
        所有 θ 都一样，等于从目标函数里消失，没有异常也没有日志。
        2026-09-15 那次 auc_ratio 的 NaN 就是这条路径，404 天里 159 天的
        目标函数和参数无关，当时只修了那一列，没有加通用守卫（F1-2）。
        """
        m = (~rej) & np.isfinite(s) & np.isfinite(self.ytil)
        self.n_nonfinite = int(((~rej) & ~m).sum())
        return m

    def _day_arrays(self, theta: dict) -> tuple[list, list]:
        """目标函数看的人群：**过准入的全池**，刻意不套 output.min_score。

        硬指标（metrics / top_codes）走生产口径，这里不走，是有意的取舍：
        45 分线是 θ 的函数，把它加进掩码，样本集就随 θ 漂（箱内 10 组随机 θ
        实测日均可发只数在 17.1~27.6 之间摆，G 与「可发只数」秩相关 0.78），
        优化器会多出一根「把分数整体抬高来加长名单」的杠杆，而且池 <= K 的天
        day_G 退化成等权全池、不再衡量排序（套 45 后这种天从 39/413 涨到 100+）。
        软 TopK 是平滑代理，生产口径是硬指标，两者量的不是同一件事。
        """
        c = C.apply_theta(self.c, theta)
        s = vscore.score(self.arr, c)
        rej = vscore.hard_reject(self.arr, c["screen"])
        keep = self._mask(s, rej)
        ds, dy = [], []
        for a, b in self.slices:
            m = keep[a:b]
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
        keep = self._mask(s, rej)
        ics, tops, hits, ns, sents = [], [], [], [], []
        for a, b in self.slices:
            m = keep[a:b]
            ns.append(int(m.sum()))
            # 前 10 超额 / 胜率是「实发清单」的成绩，走生产口径（45 分线 + top_k）；
            # IC 留在过准入全池上算——套 45 分线是区间截断，会机械压低相关性
            # （实测同一份表 ic_mean 0.0100 vs 0.0180，提案的优劣还会反号）。
            # 收益用未缩尾的 y_raw 再扣双边成本，这才是可实现口径（F2-10）。
            o = production_order(s[a:b], rej[a:b], self.c, top_k)
            sents.append(int(o.size))
            if o.size:
                yy = self.y_raw[a:b][o] - self.cost
                tops.append(float(yy.mean()))
                hits.append(float((yy > 0).mean()))
            if m.sum() < 3:
                continue
            ics.append(spearman(s[a:b][m], self.ytil[a:b][m]))
        ic = np.array([x for x in ics if np.isfinite(x)])
        return {
            "days": len(self.dates),
            "ic_mean": float(ic.mean()) if ic.size else float("nan"),
            "ic_std": float(ic.std(ddof=1)) if ic.size > 1 else float("nan"),
            "icir": float(ic.mean() / ic.std(ddof=1))
                    if ic.size > 1 and ic.std(ddof=1) > 0 else float("nan"),
            "top_excess": float(np.mean(tops)) if tops else float("nan"),
            "hit_rate": float(np.mean(hits)) if hits else float("nan"),
            # avg_pool = 过准入的只数（37 左右），avg_sent = 真发出去的只数
            # （25 左右）。两个数差 48%，混成一个邮件里就在报一张不存在的榜。
            "avg_pool": float(np.mean(ns)) if ns else 0.0,
            "avg_sent": float(np.mean(sents)) if sents else 0.0,
        }

    def top_codes(self, theta: dict, codes: np.ndarray,
                  top_k: int = 10) -> dict[str, list[str]]:
        """每天**真会发出去**的那张清单。行为回放（闸门 5）和变更邮件的
        「会新进榜 / 会掉出榜」用它。

        必须是生产口径：闸门 5 量的是「这次改参数会让邮件换掉几只票」。
        用全池前 10 会把分母固定成 10，而真实清单常常只有 5~9 只
        （413 天里 24% 不足 10 只），一只换手在生产是 1/6、在回测被摊成 1/10。
        实测待批的那条 volume/trend 提案：不套 45 分线最大单日换手 20%，
        套上之后 33%（上限 40%）。
        """
        c = C.apply_theta(self.c, theta)
        s = vscore.score(self.arr, c)
        rej = vscore.hard_reject(self.arr, c["screen"])
        out = {}
        for (a, b), day in zip(self.slices, self.dates):
            o = production_order(s[a:b], rej[a:b], self.c, top_k)
            out[day] = list(codes[a:b][o])
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


def oos_blocks(te: list, k: int) -> list[list]:
    """把样本外段切成 k 个连续、不重叠、按时间顺序的块。

    闸门 2 以前只看整段的 Ĝ_oos 哪边大。那一段每天只往后挪一天（相邻两次
    裁决的 te 集合 Jaccard 0.976~0.988），所以「第 N 次裁决」不是 N 份独立
    证据；而 83 天单窗口的配对差标准误（实测 se=0.0091）和效应量（-0.0096）
    同量级，根本分辨不出 ±0.01 的 Δ。实测把同一个 te 切三块，09-16 那个提案
    是 +0.0095 / −0.0175 / −0.0134 一正两负，整段的 −0.0115 把块间分歧
    全抹平了（F1-6）。要求多数块同向改善，才算「样本外真的更好」。
    """
    d = list(te)
    if not d:
        return []
    k = max(1, int(k))
    return [list(b) for b in np.array_split(np.asarray(d, dtype=object), k)
            if len(b)]


def block_deltas(prob_oos: Problem, g_new: np.ndarray, g_old: np.ndarray,
                 blocks: list[list]) -> list[float]:
    """每个块上的 Ĝ(new) − Ĝ(old)。

    逐日 G_d 已经算好了，块只是切分，不改变逐日量，所以这里直接按块聚合，
    和 `mk(block).G(...)` 逐位相同，不用重新打分（成本几乎为零）。
    """
    pos = {d: i for i, d in enumerate(prob_oos.dates)}
    w = np.asarray(prob_oos.day_w, float)
    gn, go = np.asarray(g_new, float), np.asarray(g_old, float)
    out = []
    for b in blocks:
        idx = np.array([pos[d] for d in b if d in pos], int)
        if idx.size == 0:
            continue
        out.append(float(O.aggregate(gn[idx], w[idx], prob_oos.huber_c)
                         - O.aggregate(go[idx], w[idx], prob_oos.huber_c)))
    return out


def _paired_diff(prob_oos: Problem, theta_new: dict, theta_old: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    """样本外逐日配对差及其日权重。

    只有这一份实现：闸门 3 的点估计（paired_delta）和自助（bootstrap_better）
    必须是同一组天、同一组权重，否则邮件上会出现「P=99.95% 而配对 ΔĜ=−0.0003」
    这种被文档说成「不该有」的读数（F1-8/F2-7/F3-8 是同一处）。
    剔两类天：日权重为 0 的（LLM 判「数据异常」），和 G_d 非有限的
    （空池天，当天没有持仓，见 objective.day_G）。
    """
    _, g_new = prob_oos.G(theta_new)
    _, g_old = prob_oos.G(theta_old)
    diff = g_new - g_old
    w = np.asarray(getattr(prob_oos, "day_w", np.ones(diff.size)), float)
    keep = (w > 0) & np.isfinite(diff)
    return diff[keep], w[keep]


def paired_delta(prob_oos: Problem, theta_new: dict, theta_old: dict) -> float:
    """闸门 3 那条自助统计量的点估计。和 bootstrap_better 同一组天同一组权重。

    不能写成 `huber_location(gd_new - gd_old, None, c)`：等权且含 w=0 的天。
    也不能只把 None 换成 day_w —— huber_location 剔 w=0 是 2026-09-16 才加的，
    在那之前 median/MAD 仍用全部 x，两边差 0.0016。共用一段代码是唯一可靠的。
    """
    diff, w = _paired_diff(prob_oos, theta_new, theta_old)
    return (float(O.huber_location(diff, w, prob_oos.huber_c))
            if diff.size else 0.0)


def bootstrap_better(prob_oos: Problem, theta_new: dict, theta_old: dict,
                     n: int = 2000, seed: int = 7) -> float:
    """按**天**重抽样，估 P(新参数的样本外 Ĝ 更好)。

    按天不按行：同一天内的收益高度相关，按行重抽会把置信区间算窄好几倍，
    于是什么改动看起来都显著。这是量化回测最常见的自欺方式之一。
    """
    diff, w_all = _paired_diff(prob_oos, theta_new, theta_old)
    # 平局天不携带信息，留着有害：池子 ≤ k 的天 G_d 退化成等权 mean(ỹ)，
    # 与 θ 无关，差值恒为 0（样本外 83 天里实测 7 天）。某次重抽里这种天
    # 一旦过半，MAD=0，那次判定就由剩下的极端日决定（F1-3）。
    tie = np.abs(diff) < 1e-12
    diff, w_all = diff[~tie], w_all[~tie]
    m = diff.size
    if m < 5:
        # 有效样本不足（全被权重 0 剔光、全是平局、或非零天不到 5 天）：
        # 没有证据偏向任何一边，返回 0.5。返回 0 会被闸门 3 读成「不显著」
        # 还算对，但闸门 7 会读成「新参数在真值上明显更差」而**否决**一个
        # 本可接受的变更，邮件和 verdict_log 还写着「N 天上 P=0.00」（F1-9）。
        return 0.5
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if O.huber_location(diff[idx], w_all[idx], prob_oos.huber_c) > 0:
            wins += 1
    return wins / n
