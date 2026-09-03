"""
学习系统离线自测。不联网、不发邮件、不写 state/。

    python src/selftest_learn.py

最重要的一条是**等价性**：向量化打分器必须和 score.py 逐位一致。
那条一红，学到的参数会被生产打分器用另一套语义执行，整个系统的结论作废。

其余的钉住四件容易被悄悄改坏的事：
  软 Top-K 的极限行为、Huber 对单日的影响力上限、锚定项的衰减、
  中性化的不变量。
"""
from __future__ import annotations

import sys
import time
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import cfg as C
from score import AuctionFeature, score_one, f_gap as rf_gap, \
    f_volume as rf_vol, f_trend as rf_trend
from learn import vscore, objective as O, dataset, gate

ROOT = Path(__file__).resolve().parent.parent
BAD = 0


def ck(cond: bool, msg: str) -> None:
    global BAD
    if not cond:
        BAD += 1
    print(f"  {'✓' if cond else '✗'} {msg}")


def rnd_frame(n: int, seed: int = 11) -> pd.DataFrame:
    """随机样本。刻意覆盖区间外、边界、零值、极端量能。"""
    g = np.random.default_rng(seed)
    lp = g.choice([5.0, 10.0, 20.0, 30.0], n)
    gap = g.uniform(-11.0, 12.0, n)
    return pd.DataFrame({
        "code": [f"{600000+i}" for i in range(n)],
        "name": [f"S{i}" for i in range(n)],
        "limit_pct": lp,
        "prev_close": g.choice([0.0, 5.0, 10.0, 88.0], n, p=[.02, .32, .33, .33]),
        "auc_price": g.choice([0.0, 6.0, 11.0, 90.0], n, p=[.02, .32, .33, .33]),
        "gap_pct": gap, "gap_norm": gap / lp,
        "auc_amount": g.uniform(0, 5e8, n), "prev_amount": g.uniform(1e7, 5e9, n),
        "auc_ratio": g.choice([0.0, 1e-4], n, p=[.02, .98]) + g.uniform(0, .3, n),
        "t1_chg": g.uniform(-11, 12, n), "t2_chg": g.uniform(-11, 12, n),
        "t3_chg": gap, "slope": g.uniform(-5, 5, n),
        "monotonic": g.random(n) > .5, "dive": g.uniform(-3, 5, n),
        "pos_pct_60d": g.random(n), "ma_bull": g.random(n) > .5,
        "breakout": g.random(n) > .6, "prev_limit_up": g.random(n) > .7,
        "prev_broken_board": g.random(n) > .85,
        "board_height": g.choice([0, 1, 2, 3, 4], n),
        "sector": g.choice(["半导体", "光模块", "军工"], n),
        "sector_members": g.choice(range(0, 9), n),
        "sector_prev_limitups": g.choice(range(0, 5), n),
        "blacklisted": g.random(n) > .95, "one_word": g.random(n) > .95,
    })


def check_equivalence(c: dict) -> None:
    print("\n向量化打分器 ≡ score.py")
    df = rnd_frame(2000)
    fields = [f.name for f in dataclasses.fields(AuctionFeature)]
    rows = [score_one(AuctionFeature(**{k: r[k] for k in fields}), c)
            for _, r in df.iterrows()]

    d = vscore.prepare(df)
    pv = vscore.parts(d, c)
    sc = c["screen"]
    # 分项逐个比：这里比的是**未取整**的值，score.py 的 round 只是显示口径
    ck(np.abs(pv["gap"] - np.array([rf_gap(x, sc["gap_pct_min"],
        sc["gap_pct_max"], sc["gap_pct_peak"]) for x in df.gap_pct])).max()
        < 1e-12, "f_gap 逐位一致")
    ck(np.abs(pv["volume"] - np.array([rf_vol(x, sc["auc_ratio_min"],
        sc["auc_ratio_max"], sc["auc_ratio_score_hi"],
        sc["auc_ratio_decay"]) for x in df.auc_ratio])).max()
        < 1e-12, "f_volume 逐位一致")
    ck(np.abs(pv["trend"] - np.array([rf_trend(a, bool(b), l) for a, b, l in
        zip(df.slope, df.monotonic, df.limit_pct)])).max()
        < 1e-12, "f_trend 逐位一致")
    for k in ("position", "sector", "continuity"):
        ref = np.array([r["parts"][k] for r in rows])
        ck(np.abs(np.round(pv[k], 3) - ref).max() < 1e-9, f"f_{k} 一致")

    s_vec = vscore.score(d, c)
    ref = np.array([r["score"] for r in rows])          # score_one 内部 round(,1)
    ck(np.abs(np.round(s_vec, 1) - ref).max() <= 0.1,
       "总分一致（差异只来自 score.py 的 round 到 0.1）")
    ck(float(np.abs(np.round(s_vec, 1) - ref).mean()) < 1e-3,
       "总分平均差 < 1e-3（不是系统性偏移）")

    rej_v = vscore.hard_reject(d, sc)
    rej_r = np.array([r["rejected"] is not None for r in rows])
    ck(int((rej_v != rej_r).sum()) == 0, "硬性排除判定完全一致")


def check_objective() -> None:
    print("\n软 Top-K 与跨天聚合")
    g = np.random.default_rng(3)
    s = g.normal(60, 12, 400)
    for k in (5, 10, 30):
        tau = O.solve_tau(s, k)
        p = O._softmax(s / tau)
        perp = float(np.exp(-np.sum(p * np.log(p + 1e-300))))
        ck(abs(perp - k) < 0.5, f"τ 求解使有效持仓数 ≈ {k}（实得 {perp:.2f}）")

    y = g.normal(0, 1, 400)
    ck(abs(O.day_G(s, y - y.mean(), 10)) < 0.6,
       "分数与收益无关时 G_d 接近 0")
    aligned = np.argsort(np.argsort(s)).astype(float)
    aligned = (aligned - aligned.mean()) / aligned.std()
    ck(O.day_G(s, aligned, 10) > 1.0, "分数与收益同向时 G_d 明显为正")
    ck(O.day_G(s, -aligned, 10) < -1.0, "反向时 G_d 明显为负")

    # Huber：一天暴走不能主导
    base = np.array([0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 0.08, 0.12, 0.10, 0.11])
    m0 = O.huber_location(base)
    spike = base.copy(); spike[0] = 8.0
    m1 = O.huber_location(spike)
    mean1 = float(spike.mean())
    ck(abs(m1 - m0) < 0.05,
       f"单日 +8σ 暴走对 Huber 位置的影响 {abs(m1-m0):.4f} < 0.05")
    ck(abs(mean1 - m0) > 0.5,
       f"同一份数据算术平均被拖到 {mean1:.3f}（这就是不用均值的原因）")

    print("\n锚定项衰减")
    lam = [O.lambda_anchor(8.0, n, 120) for n in (8, 60, 120, 250, 500)]
    ck(all(a > b for a, b in zip(lam, lam[1:])), "λ_a 随天数单调下降")
    ck(abs(lam[0] / 8.0 - 0.94) < 0.01, "8 天时仍保留 94% 锚定（基本冻结）")
    ck(abs(lam[2] / 8.0 - 0.50) < 0.01, "120 天时降到 50%（先验与数据各半）")


def check_project(c: dict) -> None:
    print("\n可行域投影")
    box = c["learning"]["box"]
    g = np.random.default_rng(5)
    for _ in range(200):
        t = {k: float(g.uniform(lo - 0.3, hi + 0.3))
             for k, (lo, hi) in box.items()}
        p = O.project(t, box)
        wk = [k for k in p if k.startswith("scoring.weights.")]
        if abs(sum(p[k] for k in wk) - 1.0) > 1e-9:
            ck(False, "权重和归一到 1")
            return
        for k, (lo, hi) in box.items():
            if not (lo - 1e-9 <= p[k] <= hi + 1e-9):
                ck(False, f"{k} 落在箱内")
                return
    ck(True, "200 次随机投影：权重和恒为 1，且全部落在箱内")
    t0 = C.theta0(box)
    ck(all(abs(O.project(t0, box)[k] - v) < 1e-9 for k, v in t0.items()),
       "θ⁰ 本身是可行点，投影不动它")
    # 全部顶到上界：必须靠下压而不是等比缩放来满足和为 1
    hi = {k: box[k][1] for k in box}
    ph = O.project(hi, box)
    wk = [k for k in ph if k.startswith("scoring.weights.")]
    ck(abs(sum(ph[k] for k in wk) - 1.0) < 1e-9
       and all(box[k][0] - 1e-9 <= ph[k] <= box[k][1] + 1e-9 for k in box),
       "全部顶到上界时仍能投回可行域")


def check_neutralize() -> None:
    print("\n日内中性化")
    g = np.random.default_rng(9)
    n = 500
    df = pd.DataFrame({
        "date": ["2026-09-01"] * n + ["2026-09-02"] * n,
        # 第二天整体下移 3%（模拟大盘暴跌），离散度也翻倍
        "r": np.concatenate([g.normal(0.001, 0.02, n),
                             g.normal(-0.030, 0.04, n)]),
        "dirty": [False] * (2 * n),
    })
    out = dataset.neutralize(df, {"min_pool": 100})
    a = out[out.date == "2026-09-01"]["ytil"].to_numpy()
    b = out[out.date == "2026-09-02"]["ytil"].to_numpy()
    ck(abs(np.median(a)) < 0.05 and abs(np.median(b)) < 0.05,
       "两天的 ỹ 中位数都被拉到 0（大盘涨跌被消掉）")
    ck(abs(dataset.mad(a) - 1.0) < 0.15 and abs(dataset.mad(b) - 1.0) < 0.15,
       "两天的 ỹ 离散度都归一到 1（暴动日不再天然占更大权重）")
    ck(abs(out[out.date == "2026-09-02"]["y"].mean()
           - out[out.date == "2026-09-01"]["y"].mean()) < 0.01,
       "中心化后两天的均值可比")

    thin = pd.DataFrame({"date": ["2026-09-03"] * 50,
                         "r": g.normal(0, .02, 50), "dirty": [False] * 50})
    ck(len(dataset.neutralize(thin, {"min_pool": 200})) == 0,
       "样本不足的天被整天丢弃")
    flat = pd.DataFrame({"date": ["2026-09-04"] * 300,
                         "r": [0.01] * 300, "dirty": [False] * 300})
    ck(len(dataset.neutralize(flat, {"min_pool": 200})) == 0,
       "离散度为 0 的天被整天丢弃（尺度无意义）")


def check_gate(c: dict) -> None:
    print("\n接受门")
    box, g = c["learning"]["box"], dict(c["learning"]["gate"])
    t0 = C.theta0(box)
    t1 = dict(t0); t1["screen.gap_pct_peak"] = t0["screen.gap_pct_peak"] + 0.1
    days = [f"2026-{m:02d}-{d:02d}" for m in (6, 7, 8) for d in range(1, 26)]
    ok_args = dict(theta_new=t1, theta_old=t0, box=box, g=g, n_days=len(days),
                   all_days=days, today=days[-1], boot_p=0.95,
                   oos_new=0.05, oos_old=0.03, churn={d: 0.1 for d in days[-20:]})
    v = gate.evaluate(**ok_args)
    ck(v.accepted, "全部条件满足时接受")

    ck(not gate.evaluate(**{**ok_args, "n_days": 20}).accepted, "天数不足 -> 拒")
    ck(not gate.evaluate(**{**ok_args, "oos_new": 0.01}).accepted,
       "样本外没改善 -> 拒")
    ck(not gate.evaluate(**{**ok_args, "boot_p": 0.60}).accepted,
       "自助显著性不够 -> 拒")
    ck(not gate.evaluate(**{**ok_args, "churn": {"d": 0.9}}).accepted,
       "行为回放换手过大 -> 拒")
    big = dict(t0); big["screen.gap_pct_peak"] = 4.4
    ck(not gate.evaluate(**{**ok_args, "theta_new": big}).accepted,
       "单参数步长超上限 -> 拒")
    many = dict(t0)
    for k in list(box)[:4]:
        many[k] = t0[k] + (box[k][1] - box[k][0]) * 0.02
    ck(not gate.evaluate(**{**ok_args, "theta_new": many}).accepted,
       "一次动了 4 个参数（上限 2） -> 拒")
    ck(not gate.evaluate(**{**ok_args, "theta_new": dict(t0)}).accepted,
       "参数没有实际改动 -> 拒")

    # 第七道闸：在线稳健性只否决不要求
    g7 = {**g, "online_veto_p": 0.25, "online_min_days": 5}
    ck(gate.evaluate(**{**ok_args, "g": g7},
                     online_p=0.90, online_days=8).accepted,
       "在线 P=0.90 -> 放行")
    ck(not gate.evaluate(**{**ok_args, "g": g7},
                         online_p=0.10, online_days=8).accepted,
       "在线 P=0.10（明显更差）-> 否决")
    ck(gate.evaluate(**{**ok_args, "g": g7},
                     online_p=0.10, online_days=3).accepted,
       "在线只有 3 天 -> 记录不否决（样本不足判不出更差）")
    ck(gate.evaluate(**{**ok_args, "g": g7}).accepted,
       "无在线数据 -> 该闸不出现")

    ck(gate.churn_by_day({"d": ["a", "b", "c"]}, {"d": ["a", "b", "c"]})["d"] == 0.0,
       "前 K 完全相同时换手为 0")
    ck(abs(gate.churn_by_day({"d": ["a", "b"]}, {"d": ["a", "x"]})["d"] - 0.5)
       < 1e-9, "换手比例算法正确")


def check_sparsify(c: dict) -> None:
    print("")
    print("稀疏化投影")
    from learn import optimize as OPT
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    # 模拟 Nelder-Mead 的连续解：9 个参数全漂
    fit = {k: v + 0.03 * (i - 4) / 10 * (box[k][1] - box[k][0])
           for i, (k, v) in enumerate(t0.items())}
    fit["scoring.weights.trend"] = t0["scoring.weights.trend"] + 0.06
    fit["screen.auc_ratio_score_hi"] = t0["screen.auc_ratio_score_hi"] - 0.006
    prop, intents = OPT.sparsify(fit, t0, box, max_moves=2, max_step_frac=0.10)
    ck(len(intents) == 2, f"9 参数连续漂移 -> 意图恰好 2 个（{intents}）")
    ck("scoring.weights.trend" in intents
       and "screen.auc_ratio_score_hi" in intents,
       "选中的是 |Δ|/σ 最大的两个方向")
    wk = [k for k in prop if k.startswith("scoring.weights.")]
    ck(abs(sum(prop[k] for k in wk) - 1.0) < 1e-9, "权重和仍为 1")
    sig = {k: (hi - lo) for k, (lo, hi) in box.items()}
    non_intent = [k for k in box if k not in intents
                  and not k.startswith("scoring.weights.")]
    ck(all(abs(prop[k] - t0[k]) < 1e-12 for k in non_intent),
       "非意图的标量参数纹丝不动")
    ck(all(abs(prop[k] - t0[k]) <= 0.10 * sig[k] + 1e-9 for k in intents),
       "意图步长被截在箱宽 10% 以内")
    # 全零输入 -> 无意图
    _, none_int = OPT.sparsify(dict(t0), t0, box, 2, 0.10)
    ck(none_int == [], "无漂移 -> 无意图")


def check_cfg(c: dict) -> None:
    print("\n配置合并")
    box = c["learning"]["box"]
    ck(set(C.theta0(box)) == set(box), "θ⁰ 覆盖 box 里全部参数")
    ck(all(box[k][0] <= v <= box[k][1] for k, v in C.theta0(box).items()),
       "人工基线本身落在箱约束内")
    t = dict(C.theta0(box)); t["scoring.weights.gap"] = 0.31
    c2 = C.apply_theta(c, t)
    ck(c2["scoring"]["weights"]["gap"] == 0.31
       and c["scoring"]["weights"]["gap"] == 0.2,
       "apply_theta 写副本，不污染原配置")
    ck(abs(sum(c["scoring"]["weights"].values()) - 1.0) < 1e-9,
       "人工基线的六个权重和为 1")


def main() -> int:
    t0 = time.time()
    c = C.load()
    check_equivalence(c)
    check_objective()
    check_project(c)
    check_neutralize()
    check_gate(c)
    check_sparsify(c)
    check_cfg(c)
    print(f"\n耗时 {time.time()-t0:.2f}s | 断言失败 {BAD} 个")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
