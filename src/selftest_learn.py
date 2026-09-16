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

import ast
import os
import sys
import time
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "breakout"))

import numpy as np
import pandas as pd

import cfg as C
from score import AuctionFeature, score_one, f_gap as rf_gap, \
    f_volume as rf_vol, f_trend as rf_trend
from learn import vscore, objective as O, dataset, gate, shadow

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
        # 成交额覆盖 300 万下限两侧；量能里混 2% 的 NaN（回填表里真有 6950 行
        # NaN，score.py 剔除而向量化版曾经放行，见 2026-09-15 审计）
        "auc_amount": g.uniform(0, 5e8, n) * g.choice([0.002, 1.0], n, p=[.1, .9]),
        "prev_amount": g.uniform(1e7, 5e9, n),
        "auc_ratio": np.where(g.random(n) < .02, np.nan,
                              g.choice([0.0, 1e-4], n, p=[.02, .98])
                              + g.uniform(0, .3, n)),
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

    # --- w=0 的天不许决定 σ̂（F1-4）---
    # 以前只有 IRLS 带权，起点 median 和尺度 MAD 用的是含 w=0 天的全部 x。
    # 实测注入 10 个 w=0 的极端日，闸门 2 的 ΔĜ 漂 3.7e-3（当次差距的 32%）
    xw = np.concatenate([spike, [30.0, -30.0, 100.0]])
    ww = np.concatenate([np.ones(10), np.zeros(3)])
    ck(abs(O.huber_location(xw, ww) - m1) < 1e-9,
       "三个 w=0 的极端值不改变 Huber 位置（σ̂ 不许含它们）")
    xh = np.concatenate([spike, np.full(10, 100.0)])
    wh = np.concatenate([np.ones(10), np.zeros(10)])
    ck(abs(O.huber_location(xh, wh) - m1) < 1e-9,
       f"一半天数 w=0 且极端时也不退化成算术平均（算术平均是 {xh.mean():.3f}）")
    ck(abs(O.huber_location(xw, ww)
           - O.huber_location(xw[ww > 0], ww[ww > 0])) < 1e-12,
       "剔零 ≡「那几天不存在」：闸门 2 和闸门 3 的 σ̂ 同口径")

    # --- MAD=0 时退回加权中位数而不是加权均值（F1-3）---
    # 池子 ≤ k 的天 G_d 与 θ 无关、配对差恒为 0，某次重抽里一旦过半就走这条
    # 分支。均值在这里完全没有影响力上限，一个暴走日说了算
    ties = np.array([0.0] * 6 + [0.01, 0.02, -0.01, 5.0])
    ck(abs(float(np.mean(ties)) - 0.502) < 1e-12,
       "先确认这份数据真会走 MAD=0 分支：算术平均被 5.0 拖到 0.502")
    ck(abs(O.huber_location(ties)) < 0.05,
       f"MAD=0 时退回中位数，单个极端日不能主导（实得 {O.huber_location(ties):.4f}）")
    wmed = O.huber_location(np.array([0.0] * 6 + [1.0, 2.0, 3.0, 4.0]),
                            np.array([1.0] * 9 + [9.0]))
    ck(wmed > 0.5, f"MAD=0 的退路带日权重（权重 9 的那天把位置推到 {wmed:g}）")

    # --- 配对 ΔĜ 的口径（F1-8/F2-7/F3-8 是同一处）---
    dpair = np.array([0.01, 0.012, 0.009, 0.011, 0.010, 0.013, 0.008, 0.012,
                      0.010, -100.0])
    wpair = np.ones(10); wpair[-1] = 0.0
    mp = wpair > 0
    ck(abs(O.huber_location(dpair[mp], wpair[mp])
           - O.huber_location(dpair[:-1])) < 1e-12,
       "权重 0 的天剔除后，与「那天不存在」逐位一致")
    ck(abs(O.huber_location(dpair, None)
           - O.huber_location(dpair[mp], wpair[mp])) > 1e-6,
       "等权（旧写法传的 None）时 -100 那天仍拖动位置估计")
    ck(abs(O.huber_location(dpair, wpair)
           - O.huber_location(dpair[mp], wpair[mp])) < 1e-12,
       "传了 day_w 就等于先剔再估尺度（σ̂ 不再含 w=0 天）")

    # --- 空池天 / 小池天（F1-7）---
    ck(not np.isfinite(O.day_G(np.array([]), np.array([]), 10)),
       "硬性排除后一只不剩 -> day_G 为 NaN（当天没有持仓，不是超额收益 0）")
    small = np.array([50.0, 60.0, 70.0])
    ysm = np.array([0.4, 0.5, 0.9])
    ck(abs(O.day_G(small, ysm, 10) - float(ysm.mean())) < 1e-12,
       "池子不足 k 只 -> 等权持有那几只，G_d = 子集 mean(ỹ)（不是 0）")
    ds = [g.normal(60, 12, 400) for _ in range(6)]
    dy = [g.normal(0, 1, 400) for _ in range(6)]
    G1, _ = O.G_hat(ds, dy, np.ones(6), 10, 1.345)
    G2, g2 = O.G_hat(ds + [np.array([])], dy + [np.array([])],
                     np.append(np.ones(6), 0.35), 10, 1.345)
    ck(abs(G2 - G1) < 1e-12 and g2.size == 7 and np.isnan(g2[6]),
       "空池天不进 Ĝ（day_w 同步剔除），但 g 数组长度不变、按 dates 对齐")

    # --- 分数里混进 NaN 要炸，不许静默退化（F1-2）---
    try:
        O.solve_tau(np.array([50.0, np.nan, 70.0, 80.0]), 2)
        raised = False
    except ValueError:
        raised = True
    ck(raised, "分数含 NaN -> solve_tau 抛错（以前返回 NaN 一路静默下去）")
    try:
        O.day_G(np.array([50.0, np.nan, 70.0, 80.0, 90.0]), np.zeros(5), 2)
        raised = False
    except ValueError:
        raised = True
    ck(raised, "分数含 NaN -> day_G 抛错，不再当成「池子不够 k 只」退化成等权")

    print("\n锚定项衰减")
    lam = [O.lambda_anchor(8.0, n, 120) for n in (8, 60, 120, 250, 500)]
    ck(all(a > b for a, b in zip(lam, lam[1:])), "λ_a 随天数单调下降")
    ck(abs(lam[0] / 8.0 - 0.94) < 0.01, "8 天时仍保留 94% 锚定（基本冻结）")
    ck(abs(lam[2] / 8.0 - 0.50) < 0.01, "120 天时降到 50%（先验与数据各半）")
    # N 必须是**进入拟合**的训练天数。2026-09-16 那次：414 天全算给 0.05618，
    # 按 len(tr)=331 应是 0.06652，低了 15.5%（N→∞ 趋向 20%），F1-5
    ck(O.lambda_anchor(0.25, 331, 120) / O.lambda_anchor(0.25, 414, 120) > 1.18,
       "同一份数据按训练天数算锚定比按全部天数强 18%（样本外那 20% 不是证据）")


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

    # 抢救日守卫：55% 行 T1=T2=T3。在线快照要丢，回填表要留
    # （回填的 t1/t2/t3 是代理值，单价竞价天然三者相等，正常日就有 ~39%）
    m = 400
    t3 = g.normal(0, 1, m)
    same = np.arange(m) < int(m * 0.55)
    salv = pd.DataFrame({"date": ["2025-04-08"] * m, "r": g.normal(0, .02, m),
                         "dirty": [False] * m, "t3_chg": t3,
                         "t1_chg": np.where(same, t3, t3 - 0.5),
                         "t2_chg": np.where(same, t3, t3 - 0.2)})
    ck(len(dataset.neutralize(salv, {"min_pool": 100})) == 0,
       "在线快照：>50% 行 T1=T2=T3 判为抢救日，整天丢弃")
    ck(len(dataset.neutralize(salv, {"min_pool": 100}, salvage_guard=False)) == m,
       "回填表：同样的数据关掉守卫后整天保留（2026-09-04 前误丢 3 天）")


def check_gate(c: dict) -> None:
    print("\n接受门")
    # 自测不许依赖 state/：冷却期读 state/theta_history.jsonl，真机上一旦
    # 接受过变更，「全部条件满足时接受」就会因冷却期变红。指到临时目录。
    import tempfile
    orig_hist = gate.HISTORY
    gate.HISTORY = Path(tempfile.mkdtemp(prefix="gate_")) / "theta_history.jsonl"
    try:
        _check_gate(c)
    finally:
        gate.HISTORY = orig_hist


def _check_gate(c: dict) -> None:
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

    # 闸门 2 的块一致性（F1-6）。整段 Ĝ_oos 是一个 83 天尾窗、每天只挪一天，
    # 「连着几次都过」几乎等于「过了一次」；块间分歧会被整段一个数抹平
    # （09-16 那个提案三块是 +0.0095 / -0.0175 / -0.0134）
    v2 = gate.evaluate(**ok_args, block_delta=[0.01, 0.02, -0.01])
    ck(v2.accepted and "2/3" in next(x.detail for x in v2.checks
                                     if x.name == "样本外改善"),
       "3 块里 2 块改善 -> 闸门 2 放行，并把「分块 2/3」写进依据")
    v3 = gate.evaluate(**ok_args, block_delta=[0.01, -0.02, -0.01])
    c3 = next(x for x in v3.checks if x.name == "样本外改善")
    ck(not v3.accepted and not c3.passed and "1/3" in c3.detail,
       f"整段 Ĝ 更好但只有 1/3 块改善 -> 拒（{c3.detail[-28:]}）")
    ck(v2.evidence["block_delta"] == [0.01, 0.02, -0.01],
       "分块 ΔĜ 写进 evidence（变更邮件和面板读它）")
    v4 = gate.evaluate(**ok_args)
    ck(v4.accepted and "分块" not in next(x.detail for x in v4.checks
                                          if x.name == "样本外改善")
       and v4.evidence["block_delta"] is None,
       "block_delta=None 时退化成旧行为（依据里不提分块）")
    ck(not gate.evaluate(**{**ok_args, "oos_new": 0.01},
                         block_delta=[0.01, 0.02, 0.03]).accepted,
       "三块全改善但整段没改善 -> 仍然拒（两条都要满足）")

    ck(gate.churn_by_day({"d": ["a", "b", "c"]}, {"d": ["a", "b", "c"]})["d"] == 0.0,
       "前 K 完全相同时换手为 0")
    ck(abs(gate.churn_by_day({"d": ["a", "b"]}, {"d": ["a", "x"]})["d"] - 0.5)
       < 1e-9, "换手比例算法正确")

    # 权重和为 1。会诊的参数提案是 LLM 直接给值的，只夹箱不投影就会 Σw≠1，
    # 而 score.py 的 raw=100·Σw·v 一缩放，min_score=45 这条绝对分数线就漂了；
    # 七道闸量的全是排序量，对整体缩放完全失明（审计 F2-4）
    bad = dict(t0)
    bad["scoring.weights.trend"] = t0["scoring.weights.trend"] + 0.03
    vb = gate.evaluate(**{**ok_args, "theta_new": bad})
    ck(not vb.accepted and any(x.name == "权重和为 1" and not x.passed
                               for x in vb.checks),
       "Σw=1.03（步长 0.086 在闸门 4 之内）-> 被「权重和为 1」拦下")
    vg = gate.evaluate(**{**ok_args, "theta_new": O.project(bad, box)},
                       intents=["scoring.weights.trend"])
    ck(vg.accepted and any(x.name == "权重和为 1" and x.passed
                           for x in vg.checks),
       "投影到「箱 ∩ Σw=1」之后同一候选放行")


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

    # 会诊路：明确给的值要原样送进闸门（不截步长），但权重必须被再归一。
    # evaluate_candidate 就是这么调的，参数变了这里先红（审计 F1-10/F3-5）
    raw = dict(t0)
    raw["scoring.weights.position"] = 0.13
    prop2, int2 = OPT.sparsify(raw, t0, box, max_moves=len(box),
                               max_step_frac=1.0, min_frac=1e-12)
    wk2 = [k for k in prop2 if k.startswith("scoring.weights.")]
    ck(int2 == ["scoring.weights.position"]
       and abs(prop2["scoring.weights.position"] - 0.13) < 1e-12
       and abs(sum(prop2[k] for k in wk2) - 1.0) < 1e-9,
       "会诊口径：提案值原样保留、其余权重再归一、意图只算明确给的那一个")
    ck(all(abs(prop2[k] - t0[k]) < 1e-12 for k in box
           if not k.startswith("scoring.weights.")),
       "非权重参数不受权重再归一影响")
    # 优化器路：同样的目标值会被截到步长上限（两条路的差别只在这两个上限）
    fit2 = dict(t0)
    fit2["scoring.weights.trend"] = t0["scoring.weights.trend"] + 0.05
    prop3, int3 = OPT.sparsify(fit2, t0, box, 2, 0.10)
    ck(int3 == ["scoring.weights.trend"]
       and abs(prop3["scoring.weights.trend"]
               - (t0["scoring.weights.trend"] + 0.035)) < 1e-12,
       "优化器口径：步长截到箱宽 10%（0.35 × 0.10 = 0.035）")


def adm_frame(n: int, seed: int = 1) -> pd.DataFrame:
    """大部分能过准入的样本。

    rnd_frame 刻意取区间外的极端值，120 行里只有 0~1 行过 hard_reject，
    测不到「过准入但不足 45 分」这条线（教训 26：默认值绕开了所有准入线）。
    这一份反过来：85% 的行落在准入区间内，分数从 5 分铺到 88 分，
    每天稳定有十几二十只卡在 45 分线下面。
    """
    g = np.random.default_rng(seed)
    lp = g.choice([10.0, 20.0], n)
    gap = np.where(g.random(n) < 0.15, g.uniform(5.5, 9.0, n),
                   g.uniform(2.0, 5.0, n))     # 15% 越界，用来喂 rej
    return pd.DataFrame({
        "code": [f"{600000 + i}" for i in range(n)],
        "name": [f"S{i}" for i in range(n)],
        "limit_pct": lp, "prev_close": 10.0, "auc_price": 10.3,
        "gap_pct": gap, "gap_norm": gap / lp,
        "auc_amount": g.uniform(4e6, 5e8, n),   # 全部高于 300 万下限
        "prev_amount": g.uniform(1e8, 5e9, n),
        "auc_ratio": g.uniform(0.0104, 0.0417, n),
        "t1_chg": g.uniform(0, 3, n), "t2_chg": gap, "t3_chg": gap,
        "slope": g.uniform(0.1, 3.0, n),
        "monotonic": g.random(n) > .5, "dive": g.uniform(-3, 1.5, n),
        "pos_pct_60d": g.random(n), "ma_bull": g.random(n) > .5,
        "breakout": g.random(n) > .6, "prev_limit_up": g.random(n) > .7,
        "prev_broken_board": g.random(n) > .85,
        "board_height": g.choice([0, 1, 2, 3, 4], n),
        "sector": g.choice(["半导体", "光模块", "军工"], n),
        "sector_members": g.choice(range(0, 9), n),
        "sector_prev_limitups": g.choice(range(0, 5), n),
        "blacklisted": g.random(n) > .97, "one_word": g.random(n) > .97,
    })


def _multiday(n_days: int, n: int, seed0: int) -> pd.DataFrame:
    """多天合成表，带 date / y / ytil，够喂 OPT.Problem 和影子模型。"""
    parts = []
    for i in range(n_days):
        f = adm_frame(n, seed=seed0 + i)
        f["date"] = f"2026-01-{i + 1:02d}"
        g = np.random.default_rng(seed0 + 500 + i)
        f["y"] = 0.002 * f["gap_pct"].to_numpy() + g.normal(0, 0.01, n)
        f["ytil"] = f["y"] / 0.01
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def check_list_convention(c: dict) -> None:
    """学习线的「前 10」必须就是邮件里那张清单（生产口径）。

    2026-09-16 审计（F1-1/F8-8/F3-3）：学习线只按 `~rej` 取前 10，生产是
    `未剔除 且 round(score,1) >= 45` 再取前 10。413 天里 97 天（23.5%）
    两张榜不是同一批票，在线 17 天里 4 天不同；闸门 5 因此把一次真实换手
    33% 记成 20%（分母被固定成 10），邮件里的「日均通过数 37」实际只发 25。
    """
    print("\n实发清单口径（>=45 分再取前 10）")
    import copy
    from learn import optimize as OPT
    import score as PS

    # 1 纯函数：45 分线按 round(,1) 判，44.96 过线、44.94 出局
    sc = np.array([72, 61, 58.6, 54.9, 45.0, 44.96, 44.94, 43.5, 43.2,
                   42.3, 31.2, 29.3, 88, 50, 47], float)
    rej = np.zeros(len(sc), bool)
    o = OPT.production_order(sc, rej, c)
    ck(len(o) == 9 and float(sc[o].min()) >= 44.95
       and list(sc[o]) == sorted(sc[o], reverse=True),
       f"15 只里 9 只过线（44.96 过、44.94 不过），按分降序（实得 {len(o)} 只）")
    rej2 = rej.copy(); rej2[12] = True      # 88 分那只被硬性排除
    ck(list(sc[OPT.production_order(sc, rej2, c)])
       == [x for x in sorted(sc, reverse=True) if x >= 44.95 and x != 88.0],
       "硬性排除先生效，再套 45 分线")
    ck(len(OPT.production_order(np.linspace(45.0, 90.0, 15),
                                np.zeros(15, bool), c)) == 10,
       "15 只都过线时只取前 10")
    ck(len(OPT.production_order(np.full(8, 44.9), np.zeros(8, bool), c)) == 0,
       "全场不足 45 分 -> 当天实发清单为空（生产那天真的一只都不发）")

    # 2 口径对齐：学习线的每日清单 ≡ score.rank + run_auction.select
    df = _multiday(3, 120, 300)
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    codes = df.sort_values("date", kind="mergesort")["code"].to_numpy()
    p = OPT.Problem(df, c, box, t0, t0, None, 10, 1.345)
    learned_top = p.top_codes(t0, codes, 10)

    fields = [f.name for f in dataclasses.fields(AuctionFeature)]
    prod, below, ys = {}, 0, []
    for day, gday in df.groupby("date"):
        rows = [score_one(AuctionFeature(**{k: r[k] for k in fields}), c)
                for _, r in gday.iterrows()]
        # run_auction.select：rank() 已按 score_raw 降序，再取 top_n
        sel = sorted(PS.rank(rows, c)["all"], key=lambda r: -r["score"])
        prod[day] = [r["code"] for r in sel[: c["output"]["top_n"]]]
        below += sum(1 for r in rows if r["rejected"] is None
                     and r["score"] < c["output"]["min_score"])
        pos = {cd: i for i, cd in enumerate(gday["code"])}
        yv = gday["y"].to_numpy(float)
        if prod[day]:
            ys.append(float(np.mean([yv[pos[cd]] for cd in prod[day]])))
    # 教训 26：断言要真的碰到那条规则，不能被默认值绕开
    ck(below >= 1, f"这批数据里有 {below} 只「过准入但不足 45 分」的票（真踩到规则）")
    ck(learned_top == prod, "学习线每天的清单和 score.rank + select 逐位相同")

    m = p.metrics(t0, 10)
    # 汇报口径扣双边成本（F2-10）：cost_bp 是单边，一买一卖各一次
    cost = 2.0 * float(c["learning"]["label"]["cost_bp"]) / 1e4
    ck(abs(m["top_excess"] - (float(np.mean(ys)) - cost)) < 1e-12,
       "metrics 的前 10 超额 = 实发清单那几只的收益均值再扣双边成本")
    ck(m["avg_sent"] < m["avg_pool"],
       f"日均实发 {m['avg_sent']:.1f} < 日均过准入 {m['avg_pool']:.1f}"
       "（两个数没被混成一个）")

    # 3 只有 1 只可发的天不许被整天跳过（生产那天真发了信）
    d1 = df[df["date"] == df["date"].iloc[0]].reset_index(drop=True)
    s1, r1 = vscore.score_df(d1, c)
    top2 = np.sort(s1[~r1])[::-1][:2]
    c1 = copy.deepcopy(c)
    c1["output"]["min_score"] = float((round(top2[0], 1) + round(top2[1], 1)) / 2)
    m1 = OPT.Problem(d1, c1, box, t0, t0, None, 10, 1.345).metrics(t0, 10)
    ck(m1["avg_sent"] == 1.0 and np.isfinite(m1["top_excess"])
       and np.isfinite(m1["hit_rate"]) and np.isfinite(m1["ic_mean"]),
       "当天只有 1 只可发：前 10 超额照算，IC 仍在过准入全池上算")
    c0 = copy.deepcopy(c)
    c0["output"]["min_score"] = float(top2[0]) + 10.0
    p0 = OPT.Problem(d1, c0, box, t0, t0, None, 10, 1.345)
    m0 = p0.metrics(t0, 10)
    ck(m0["avg_sent"] == 0.0 and not np.isfinite(m0["top_excess"])
       and p0.top_codes(t0, d1["code"].to_numpy(), 10)[d1["date"].iloc[0]] == [],
       "全天没票过线 -> 清单为空、前 10 超额是 nan（不是 0）")

    # 4 闸门 5 的分母是实际榜长，且「两边都没票」不算换手
    ck(abs(gate.churn_by_day({"d": ["a", "b", "c"]},
                             {"d": ["a", "b", "x"]})["d"] - 1 / 3) < 1e-9,
       "3 只的榜换 1 只 = 1/3（不是摊成 1/10）")
    ck(gate.churn_by_day({"d": []}, {"d": []}) == {},
       "两边都发不出票的日子不产生条目（以前记成换手 100%，闸门 5 永远不过）")
    ck(abs(gate.churn_by_day({"d": ["a", "b"]}, {"d": []})["d"] - 1.0) < 1e-9,
       "一边有票一边没票 = 换手 100%")


def check_bf_dirty(c: dict) -> None:
    """回填表的可用样本口径必须和在线 labels.build 是同一条（教训 30）。"""
    print("\n回填表脏样本口径 ≡ 在线 labels.build")
    import copy
    import logging as _logging
    import eval_daily as ED
    from learn import labels as L
    _logging.getLogger().setLevel(_logging.WARNING)   # 别让 INFO 刷屏

    raw = pd.DataFrame({
        "one_word": [False, False, False, False, True, False],
        "open_mismatch_pct": [0.0, 0.2, 0.51, 1.0, 0.0, np.nan],
    })
    got = ED._bf_dirty(raw, c).tolist()
    ck(got == [False, False, True, True, True, False],
       f"偏离 >0.5% 或一字板判脏；NaN（没竞价价）不判脏（实得 {got}）")
    c2 = copy.deepcopy(c)
    c2["learning"]["label"]["max_open_mismatch_pct"] = 2.0
    ck(ED._bf_dirty(raw, c2).tolist() == [False] * 4 + [True, False],
       "阈值取自 learning.label.max_open_mismatch_pct，不是写死的 0.5")

    logs: list[str] = []
    h = _logging.Handler(); h.emit = lambda r: logs.append(r.getMessage())
    lg = _logging.getLogger("eval"); lg.addHandler(h)
    try:
        old = ED._bf_dirty(raw[["one_word"]], c).tolist()
    finally:
        lg.removeHandler(h)
    ck(old == [False] * 4 + [True, False]
       and any("open_mismatch_pct" in x for x in logs),
       "旧 parquet 缺列时不崩，但留 warning（不许静默退回旧口径）")

    # 正主：同一批行，两条线的 dirty 必须逐位相同
    snap = pd.DataFrame({"code": [f"60000{i}" for i in range(6)],
                         "auc_price": [10.0] * 6,
                         "one_word": raw["one_word"].to_numpy()})
    quotes = pd.DataFrame({"code": snap["code"],
                           "open": [10.0, 10.02, 10.051, 10.1, 10.0, 10.0],
                           "close": [10.5] * 6})
    lab = L.build("2026-09-16", snap, quotes,
                  c["learning"]["label"]["max_open_mismatch_pct"])
    bf = raw.copy()
    bf["open_mismatch_pct"] = lab["open_mismatch_pct"].to_numpy()
    ck(list(lab["dirty"]) == ED._bf_dirty(bf, c).tolist(),
       "两条线逐位相同（以前回填只判一字板，40.76 万行里差 46190 行 = 11.5%）")

    # 脏行不进当日中位数/缩尾/MAD，所以影响的不只是它自己那几行
    g = np.random.default_rng(17)
    day = pd.DataFrame({"date": ["2026-09-16"] * 400,
                        "r": g.normal(0.0, 0.02, 400),
                        "one_word": [False] * 400,
                        "open_mismatch_pct": [1.0] * 100 + [0.0] * 300})
    out = dataset.neutralize(day.assign(dirty=ED._bf_dirty(day, c)),
                             {"min_pool": 200})
    ck(len(out) == 300 and abs(float(out["day_center"].iloc[0])
                               - float(np.median(day["r"].to_numpy()[100:]))) < 1e-12,
       "400 行剔 100 行，当日中心按剩下的 300 行算")

    # AST：读回填表的地方只有 _load_train 一处，且只经 _bf_dirty 定口径。
    # 2026-09-16 起 stage_race 也走 _load_train（F3-16），所以它自己既不该
    # 调 _bf_dirty 也不该写 dirty 列 —— 这两件事同时钉住，谁再复制一份
    # 加载逻辑就会同时踩到这条和 check_wiring 里那条。
    tree = ast.parse((ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def _dirty_assigns(f):
        return [x for x in ast.walk(f) if isinstance(x, ast.Assign)
                and any(isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "dirty" for t in x.targets)
                and not (isinstance(x.value, ast.Call)
                         and getattr(x.value.func, "id", "") == "_bf_dirty")] \
            if f is not None else [None]

    for name, want in (("_load_train", 1), ("stage_race", 0)):
        f = fns.get(name)
        n_call = sum(1 for x in ast.walk(f) if isinstance(x, ast.Call)
                     and getattr(x.func, "id", "") == "_bf_dirty") if f else -1
        ck(n_call == want and not _dirty_assigns(f),
           f"{name} 里 _bf_dirty 调用 {want} 次，且不自己写 "
           "raw['one_word'].astype(bool)")


def check_learn_box(c: dict) -> None:
    """交给优化器和闸门的箱必须先按可学维度裁过（W2-2 / F8-15）。

    回填表占训练表绝大多数天，它的 trend / volume 两维是代理值或含开盘后
    成交的量：不裁的话优化器照样在这两个维度上调参数，学出来的是数据缺失
    不是结论。裁法是钉死 lo=hi=θ⁰ 而不是删键（删键 Σw 会归一到 1，
    加上仍生效的 trend/volume 两个 0.20 总权重变成 1.4）。
    """
    print("\n可学维度裁剪进优化器")
    import copy
    import eval_daily as ED
    from learn import sources as SRC

    box = c["learning"]["box"]
    t0 = C.theta0(box)
    r = ED._learn_box(c)
    ck(set(r) == set(box), "键一个不少（钉死不是删键）")
    pinned = [k for k, (lo, hi) in r.items() if hi <= lo]
    # 训练源只有回填（config 的 learnable_dims 就是这份数据的真相）：
    # trend / volume 那三个参数的上下界必须相等
    for k in ("scoring.weights.trend", "scoring.weights.volume",
              "screen.auc_ratio_score_hi", "screen.auc_ratio_decay"):
        ck(r[k][0] == r[k][1] == t0[k],
           f"{k} 上下界相等且等于人工基线（回填学不了这一维）")
    for k in ("scoring.weights.gap", "scoring.weights.sector",
              "screen.gap_pct_peak"):
        ck(r[k] == list(box[k]) and r[k][0] < r[k][1], f"{k} 仍然可学")
    ck(set(pinned) == {"scoring.weights.trend", "scoring.weights.volume",
                       "screen.auc_ratio_score_hi", "screen.auc_ratio_decay"},
       f"被钉死的恰好是那四个参数（实得 {sorted(pinned)}）")

    # 真的有后果：投影之后 trend/volume 一步都迈不出去，而 Σw 仍然是 1
    t = {k: float(v) for k, v in t0.items()}
    t["scoring.weights.trend"] = 0.40
    t["scoring.weights.volume"] = 0.40
    p_full = O.project(t, box)
    p_cut = O.project(t, r)
    ck(abs(p_cut["scoring.weights.trend"] - t0["scoring.weights.trend"]) < 1e-12
       and abs(p_cut["scoring.weights.volume"]
               - t0["scoring.weights.volume"]) < 1e-12,
       "裁过的箱：把 trend/volume 顶到 0.40 也会被拉回基线")
    ck(p_full["scoring.weights.trend"] > t0["scoring.weights.trend"] + 0.05,
       "不裁的箱：同一组值会真的把 trend 推高（说明这条规则有后果）")
    wk = [k for k in p_cut if k.startswith("scoring.weights.")]
    ck(abs(sum(p_cut[k] for k in wk) - 1.0) < 1e-9, "裁过之后 Σw 仍然是 1")

    # 配置换成「量能可学」时 volume 解锁 —— 箱跟着 learnable_dims 走，
    # 不是写死的
    c2 = copy.deepcopy(c)
    c2["learning"]["backfill"]["learnable_dims"] = ["gap", "volume"]
    r2 = ED._learn_box(c2)
    ck(r2["scoring.weights.volume"] == list(box["scoring.weights.volume"])
       and r2["scoring.weights.trend"][0] == r2["scoring.weights.trend"][1],
       "learnable_dims 加了 volume 就解锁 volume，trend 仍锁")
    ck(SRC.dims_from_cfg(c) == c["learning"]["backfill"]["learnable_dims"],
       "可学维度的唯一真相是 config.learning.backfill.learnable_dims")

    # 接线：三处取箱的地方都必须经 _learn_box，不许再裸取 lc["box"]
    src = (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("_judge", "evaluate_candidate", "stage_learn"):
        f = fns.get(name)
        n_cut = sum(1 for n in ast.walk(f) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", "") == "_learn_box") if f else 0
        raw = [n for n in ast.walk(f) if isinstance(n, ast.Subscript)
               and isinstance(n.slice, ast.Constant) and n.slice.value == "box"
               and getattr(n.value, "id", "") == "lc"] if f else [None]
        ck(n_cut == 1 and not raw,
           f"{name} 取箱经 _learn_box（没有裸 lc['box']）")


def check_proxy_cols() -> None:
    """回填行的竞价轨迹是代理值，不许当特征喂给模型和影子（W2-3）。"""
    print("\n回填行的代理列屏蔽")
    from learn import model_select as MS

    # 两段各占一天：_load_train 里在线天会覆盖同日的回填天，一天只可能
    # 来自一个源，秩归一也是按天做的
    df = pd.concat([_multiday(1, 6, 300).assign(_src="backfill",
                                                date="2026-01-01"),
                    _multiday(1, 6, 400).assign(_src="online",
                                                date="2026-01-02")],
                   ignore_index=True)
    # 回填行照 backfill.to_features 的样子摆：t1=t2=t3=gap_pct、
    # slope=dive=0、monotonic=False，也就是「五列在回填上是常数」
    m = df["_src"] == "backfill"
    for k in ("t1_chg", "t2_chg"):
        df.loc[m, k] = df.loc[m, "gap_pct"].to_numpy()
    for k in ("slope", "dive"):
        df.loc[m, k] = 0.0
    df.loc[m, "monotonic"] = False
    d = MS.prep_features(df)
    bf, on = d["_src"] == "backfill", d["_src"] == "online"

    ck(MS.PROXY_COLS == ["t1_chg", "t2_chg", "slope", "dive", "monotonic"],
       "代理列就是这五个（t3_chg 不在里面：它是撮合价涨幅，回填有真值）")
    ck(all(d.loc[bf, k].isna().all() for k in MS.PROXY_COLS),
       "回填行的五列全是 NaN（常数当特征 = 把 gap 学两遍）")
    ck(not d.loc[on, MS.PROXY_COLS].isna().any().any(),
       "在线行一个都不动（那是 09:19:40 / 09:23:30 的真采样）")
    ck(float((d.loc[bf, "t3_chg"].to_numpy()
              - df.loc[bf.to_numpy(), "gap_pct"].to_numpy()).max()) == 0.0,
       "t3_chg 不屏蔽")
    ck(all(k in MS.FEATURES for k in MS.PROXY_COLS),
       "列还在特征表里（影子模型的 features 列表不变，换了它比较就失效）")
    X = MS.rank_norm(d[MS.FEATURES], d["date"]).fillna(0.0)
    ck(float(np.abs(X.loc[bf.to_numpy(), MS.PROXY_COLS].to_numpy()).max()) == 0.0,
       "秩归一后回填那天整列是 0（截面中性），不是某个被当真的常数")

    # 没有 _src 列的调用方是实时打分（run_auction 读 out/detail.csv 给影子
    # 参考榜），那里的轨迹是真采样，抹掉就变成训练一套、生产另一套（教训 30）
    live = MS.prep_features(df[df["_src"] == "online"].drop(columns="_src"))
    ck(not live[MS.PROXY_COLS].isna().any().any(),
       "没有 _src 列的表原样保留（实时打分那路）")

    # 手写打分器基线不能吃 NaN：vscore 的 monotonic 走 astype(bool)，
    # NaN -> True，每行白送趋势分（教训 9 同型）
    ck(bool(np.asarray(np.nan).astype(bool)),
       "NaN.astype(bool) 就是 True —— 所以基线必须打在原表上")
    mtree = ast.parse((ROOT / "src" / "learn" / "model_select.py").read_text(
        encoding="utf-8"))
    wf = next(x for x in ast.walk(mtree) if isinstance(x, ast.FunctionDef)
              and x.name == "walk_forward")
    sdf = [x for x in ast.walk(wf) if isinstance(x, ast.Call)
           and getattr(x.func, "attr", "") == "score_df"]
    ck(len(sdf) == 1 and isinstance(sdf[0].args[0], ast.Name)
       and sdf[0].args[0].id == "df",
       "walk_forward 的基线打分喂的是原表 df，不是 prep_features 的产物")


def check_shadow_oos(c: dict) -> None:
    """影子对比必须是样本外的：先用上一次落盘的模型记账，再 refit。"""
    print("\n影子样本外对比与账本")
    import json as _json
    import logging as _logging
    import tempfile
    days = [f"2026-01-{i:02d}" for i in range(1, 13)]
    df = _multiday(12, 60, 100)
    df["date"] = np.repeat(days, 60)
    online = df[df["date"] >= days[-4]].reset_index(drop=True)
    # 故意绕开 hard_reject：这条钉的是 train_end 过滤和账本，不是准入
    base = online["gap_pct"].to_numpy(float) * 10.0
    rej = np.zeros(len(online), bool)

    td = Path(tempfile.mkdtemp(prefix="shadow_"))
    keep = (shadow.MODEL, shadow.LEDGER)
    shadow.MODEL = td / "shadow_model.json"
    shadow.LEDGER = td / "shadow_compare.json"
    try:
        # a 在含在线日的全表上 refit -> 对比恒为空，而且必须留下告警
        shadow.fit(df)
        logs: list[str] = []
        h = _logging.Handler(); h.emit = lambda r: logs.append(r.getMessage())
        lg = _logging.getLogger("learn.shadow"); lg.addHandler(h)
        try:
            empty = shadow.daily_compare(online, base, rej, 10)
        finally:
            lg.removeHandler(h)
        ck(empty == [], "train_end = 最新在线日 -> 对比为空（2026-09-15 起的失效模式）")
        ck(any("样本内" in x for x in logs),
           "整段被过滤时留 warning，不静默（教训 16）")

        # b 只用前 10 天拟合 -> 后 2 天才算样本外
        shadow.fit(df[df["date"] <= days[-3]])
        cmp_ = shadow.daily_compare(online, base, rej, 10)
        got = [r["date"] for r in cmp_]
        ck(got == days[-2:], f"train_end={days[-3]} -> 只比之后那 2 天（实得 {got}）")
        ck(str(_json.loads(shadow.MODEL.read_text(encoding="utf-8"))["train_end"])
           < cmp_[0]["date"], "模型训练截止日早于被比的第一天")

        # c 账本：按日期先到先得，同日重跑幂等
        l1 = shadow.record(cmp_)
        l2 = shadow.record([{**cmp_[0], "base_top_excess": 999.0}])
        l3 = shadow.record([])
        ck(len(l1) == 2 and len(l2) == 2 and l3 == l2
           and l2[0]["base_top_excess"] == cmp_[0]["base_top_excess"],
           "账本按日期去重、先到先得；空输入原样返回（refit 不能重算旧账）")
        ck(len({r["date"] for r in l2}) == 2 and not shadow.LEDGER.parent.samefile(
            Path(__file__).resolve().parent.parent / "state"),
           "账本写在临时目录，没碰 state/")

        # d 45 分线只截正式榜那一路；重合率的分母是较长那张榜
        small = online[online["date"] == days[-1]].head(6).reset_index(drop=True)
        sb = np.arange(6, dtype=float) * 10.0 + 10.0        # 10..60
        c3 = shadow.daily_compare(small, sb, np.zeros(6, bool), 10, min_score=35.0)
        ck(len(c3) == 1 and c3[0]["base_n"] == 3 and c3[0]["shadow_n"] == 6,
           "正式榜按 >=35 截成 3 只，影子榜没有这条线仍是 6 只")
        ck(abs(c3[0]["overlap"] - 0.5) < 1e-9,
           "重合率 = 3/max(3,6) = 50%（分母不是 top_k=10，否则会写成 30%）")
    finally:
        shadow.MODEL, shadow.LEDGER = keep

    # AST：必须先 daily_compare、再 record、最后 fit
    # （2026-09-16 起这段在 _status_extras 里，见 F3-7）
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "_status_extras")

    def _calls(attr):
        return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == attr
                and getattr(getattr(n.func, "value", None), "id", "") == "shadow"]
    dc, ft, rc = _calls("daily_compare"), _calls("fit"), _calls("record")
    ck(len(dc) == 1 and len(ft) == 1 and len(rc) == 1,
       "_status_extras 里 daily_compare / record / fit 各恰好一次")
    ck(dc and ft and dc[0].lineno < ft[0].lineno,
       "先 daily_compare 再 fit：顺序反了 train_end 就是今天，对比恒为空")
    ck(dc and any(k.arg == "min_score" for k in dc[0].keywords),
       "daily_compare 传了 min_score（正式榜 = 当天实发清单）")

    # 文案：影子对比天数和在线真值天数是两个数，不许混成一个
    # （2026-09-16 的 learn.html 上「在线真值天数 0」和「17 / 30 天」同页打架）
    from learn import report as R, panel as LP
    st = {"date": "2026-09-16", "n_days": 414, "online_days": 17,
          "shadow": [], "theta_version": "基线", "accepted_total": 0,
          "shadow_stat": {"days": 0, "min_days": 30, "p_req": 0.9,
                          "p_better": None, "ready": False},
          "metrics": {"ic_mean": 0.01, "top_excess": 0.006}}
    td2 = Path(tempfile.mkdtemp(prefix="status_"))
    keep_s = R.STATUS
    R.STATUS = td2 / "learning_status.json"
    try:
        R.STATUS.write_text(_json.dumps(st, ensure_ascii=False), encoding="utf-8")
        lines = "\n".join(R.status_lines())
    finally:
        R.STATUS = keep_s
    ck("17" in lines and "在线真值积累 0/30" not in lines,
       "状态摘要不再把「影子对比 0 天」写成「在线真值积累 0/30」")
    step = LP._stepper(st, {"min_days": 60, "shadow_min_days": 30, "p_req": 0.9,
                            "top_k": 10, "run_at": "16:30"}, {})
    kpi = LP._kpis(st)
    ck("0 / 30" in step and "17" in step,
       "面板第 2 步按影子对比天数走进度条，同时写明在线真值 17 天")
    ck("在线真值天数" in kpi and ">17<" in kpi and "影子对比天数" in kpi,
       "关键数字里两个天数各占一张卡（在线真值 17、影子对比 0）")


def check_wiring() -> None:
    """闸门的统计量必须来自计算，不能来自配置。

    2026-09-04 全仓 debug 抓到 stage_learn 按位置把 g["bootstrap_p"]（阈值）
    传到了 boot_p（统计量）的位置，闸门 3 变成 0.9 >= 0.9 永远通过。
    这里用 AST 钉住调用方：evaluate 的 boot_p 关键字必须是一个由
    bootstrap_better 赋值的变量。
    """
    print("\n闸门接线（AST）")
    src = (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # 2026-09-16 起裁决那段抽成 _judge（学习会诊的参数提案和 stage_learn 共用
    # 同一套闸门），gate.evaluate 只许在 _judge 里出现，两个调用方各调它一次。
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fn = fns.get("_judge")
    ck(fn is not None, "找到 _judge（闸门统计量的唯一算法）")
    if fn is None:
        return
    for caller in ("stage_learn", "evaluate_candidate"):
        cf = fns.get(caller)
        n_j = sum(1 for n in ast.walk(cf) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "_judge") if cf else 0
        n_e = sum(1 for n in ast.walk(cf) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", "") == "evaluate") if cf else 0
        ck(n_j == 1 and n_e == 0, f"{caller} 恰好调用一次 _judge，且不自己调 gate.evaluate")
    from_boot = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "attr", "") == "bootstrap_better"):
            from_boot |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "evaluate"]
    ck(len(calls) == 1, "_judge 恰好调用一次 gate.evaluate")
    kw = {k.arg: k.value for k in calls[0].keywords} if calls else {}
    bp = kw.get("boot_p")
    ck(isinstance(bp, ast.Name) and bp.id in from_boot,
       "boot_p 关键字来自 bootstrap_better 的返回值（不是配置阈值）")
    op = kw.get("online_p")
    ck(isinstance(op, ast.Name) and op.id in from_boot and op.id != getattr(bp, "id", ""),
       "online_p 是另一次 bootstrap_better 的结果，和 boot_p 不是同一个变量")
    ck(not any(isinstance(a, ast.Subscript) for a in calls[0].args) if calls else False,
       "evaluate 的位置参数里没有配置下标（阈值不许按位置混进统计量）")

    # --- 会诊候选和优化器候选走同一条稀疏化+投影路（审计 F1-10/F3-5）---
    ec = fns.get("evaluate_candidate")
    sp = [n for n in ast.walk(ec) if isinstance(n, ast.Call)
          and getattr(n.func, "attr", "") == "sparsify"] if ec else []
    self_count = [n for n in ast.walk(ec) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "intents"
                          for t in n.targets)] if ec else []
    unpack = [n for n in ast.walk(ec) if isinstance(n, ast.Assign)
              and isinstance(n.value, ast.Call)
              and getattr(n.value.func, "attr", "") == "sparsify"
              and any(isinstance(t, ast.Tuple)
                      and any(isinstance(e, ast.Name) and e.id == "intents"
                              for e in t.elts) for t in n.targets)] if ec else []
    ck(len(sp) == 1 and not self_count and len(unpack) == 1,
       "evaluate_candidate 恰好调一次 sparsify，intents 由它解包（不许自己数）")

    # --- 试跑标记：status 必须带 dry（教训 27 在学习线上的落实）---
    sl = fns.get("stage_learn")
    assigns = [n for n in ast.walk(sl) if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Dict)
               and any(isinstance(t, ast.Name) and t.id == "status"
                       for t in n.targets)] if sl else []
    ck(len(assigns) == 1, "stage_learn 里 status 只组装一次")
    dry_v = None
    if assigns:
        for k, v in zip(assigns[0].value.keys, assigns[0].value.values):
            if isinstance(k, ast.Constant) and k.value == "dry":
                dry_v = v
    ck(dry_v is not None and any(isinstance(x, ast.Name) and x.id == "dry"
                                 for x in ast.walk(dry_v)),
       "status 带 dry 且取自 dry 形参（否则试跑会被 already_done 当成「今天跑完了」）")
    saves = [n for n in ast.walk(sl) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "save_status"] if sl else []
    ck(len(saves) >= 1 and all(n.args and isinstance(n.args[0], ast.Name)
                               and n.args[0].id == "status" for n in saves),
       f"{len(saves)} 处 save_status 存的都是同一个 status（含 min_days 那条早退分支）")
    ck(bool(assigns) and bool(saves)
       and assigns[0].lineno < min(n.lineno for n in saves),
       "status 在第一次 save_status 之前就组装好")

    # --- F1-5：锚定强度的 N 必须是喂给 fit 的训练天数，不是全部天数 ---
    split_tr = None
    for node in (ast.walk(sl) if sl else []):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "attr", "") == "split_days"
                and isinstance(node.targets[0], ast.Tuple)):
            split_tr = node.targets[0].elts[0].id          # 'tr'
    la = [x for x in ast.walk(sl) if isinstance(x, ast.Call)
          and getattr(x.func, "attr", "") == "lambda_anchor"] if sl else []
    ck(len(la) == 1 and split_tr is not None,
       "stage_learn 恰好调一次 lambda_anchor，且先做了 split_days")
    arg = (la[0].args[1] if la and len(la[0].args) > 1
           else next((k.value for k in la[0].keywords if k.arg == "n_days"),
                     None) if la else None)
    ck(isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "len"
       and isinstance(arg.args[0], ast.Name) and arg.args[0].id == split_tr,
       "lambda_anchor 的 N 是 len(tr)（进入拟合的训练天数），不是全部天数 n")

    # --- F1-8/F2-7/F3-8：配对 ΔĜ 只有一份口径，来自 OPT.paired_delta ---
    from_paired, from_blocks = set(), set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            a = getattr(n.value.func, "attr", "")
            names = {t.id for t in n.targets if isinstance(t, ast.Name)}
            if a == "paired_delta":
                from_paired |= names
            elif a == "block_deltas":
                from_blocks |= names
    pdk = kw.get("paired_delta")
    ck(isinstance(pdk, ast.Name) and pdk.id in from_paired,
       "paired_delta 关键字来自 OPT.paired_delta（和闸门 3 自助同一组天同一组权重）")
    ck(not [x for x in ast.walk(fn) if isinstance(x, ast.Call)
            and getattr(x.func, "attr", "") == "huber_location"],
       "_judge 里不再手写 huber_location（那是第二份口径的来源）")

    # --- F1-6：闸门 2 的分块证据接线 ---
    bdk = kw.get("block_delta")
    ck(isinstance(bdk, ast.Name) and bdk.id in from_blocks,
       "block_delta 关键字来自 OPT.block_deltas（闸门 2 的分块证据）")
    obl = [x for x in ast.walk(fn) if isinstance(x, ast.Call)
           and getattr(x.func, "attr", "") == "oos_blocks"]
    ck(len(obl) == 1 and isinstance(obl[0].args[0], ast.Attribute)
       and obl[0].args[0].attr == "dates"
       and getattr(obl[0].args[0].value, "id", "") == "p_te",
       "分块切的就是 p_te 自己那批样本外天（不另传一份，免得两边漂）")

    # --- F2-9：擂台基线用传入的配置，不许自己 C.load() ---
    mtree = ast.parse((ROOT / "src" / "learn" / "model_select.py").read_text(
        encoding="utf-8"))
    wf = next(x for x in ast.walk(mtree) if isinstance(x, ast.FunctionDef)
              and x.name == "walk_forward")
    loads = [x for x in ast.walk(wf) if isinstance(x, ast.Call)
             and getattr(x.func, "attr", "") == "load"
             and getattr(getattr(x.func, "value", None), "id", "")
             in ("C", "cfg")]
    imps = [a.name for x in ast.walk(wf) if isinstance(x, ast.Import)
            for a in x.names]
    sdf = [x for x in ast.walk(wf) if isinstance(x, ast.Call)
           and getattr(x.func, "attr", "") == "score_df"]
    ck(not loads and "cfg" not in imps,
       "walk_forward 里没有 C.load()，也不再 import cfg")
    ck(len(sdf) == 1 and len(sdf[0].args) > 1
       and isinstance(sdf[0].args[1], ast.Name) and sdf[0].args[1].id == "c",
       "擂台基线的配置就是形参 c（否则变体配置比不出差别）")
    # 教训 26：证明这条规则真有后果 —— 基线打分对配置是敏感的，传错一份 c
    # 的结果是静默改分，不报错。跑一遍 walk_forward 要 2 秒，这里只钉打分器
    c_ = C.load()
    d0 = _multiday(2, 60, 905)
    s_a, _ = vscore.score_df(d0, c_)
    s_b, _ = vscore.score_df(d0, C.apply_theta(
        c_, {"scoring.weights.gap": 0.0, "scoring.weights.volume": 0.45}))
    ck(float(np.abs(s_a - s_b).max()) > 1.0,
       f"基线打分对配置敏感（同一批票最大差 {float(np.abs(s_a - s_b).max()):.1f} 分）")

    # --- 取前 K 一律走 production_order，不许再写裸 nlargest/argsort（F3-3）---
    # 2026-09-16 起 stage_learn 的逐日指标搬进 _status_extras（F3-7），
    # 「前 10」的算法搬进 learn/online_eval.py（F3-13），三处一起钉
    for name in ("_status_extras", "stage_brief"):
        f = fns.get(name)
        bad = [n for n in ast.walk(f) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") in ("nlargest", "argsort")
               and any(isinstance(a, ast.Constant) and a.value == "sc"
                       for a in n.args)] if f else [None]
        ck(f is not None and not bad,
           f"{name} 里没有裸 nlargest('sc')（前 10 一律走生产口径）")
    sb = fns.get("stage_brief")
    ck(sb is not None and any(isinstance(n, ast.Attribute)
                              and n.attr == "production_order"
                              for n in ast.walk(sb)),
       "stage_brief 的「前 10」经 production_order")
    otree = ast.parse((ROOT / "src" / "learn" / "optimize.py").read_text(
        encoding="utf-8"))
    ofns = {n.name: n for n in ast.walk(otree) if isinstance(n, ast.FunctionDef)}
    for name in ("metrics", "top_codes"):
        f = ofns.get(name)
        used = any(isinstance(n, ast.Name) and n.id == "production_order"
                   for n in ast.walk(f)) if f else False
        ck(used, f"optimize.{name} 用 production_order 取当天实发清单")
    etree = ast.parse((ROOT / "src" / "learn" / "online_eval.py").read_text(
        encoding="utf-8"))
    efns = {n.name: n for n in ast.walk(etree) if isinstance(n, ast.FunctionDef)}
    ck(any(isinstance(n, ast.Attribute) and n.attr == "production_order"
           for n in ast.walk(efns["daily_replay"])),
       "online_eval.daily_replay 用 production_order（回放也是生产口径）")

    # --- F3-16：模型对比和影子必须取同一张训练表 ---
    # 以前 stage_race 自己 glob 回填表，_load_train 2026-09-15 起并进在线真值天
    # 之后两边就分叉了：重叠 7 天 5843 对样本里 slope 相关 -0.033、monotonic
    # 只有 48.5% 一致，而影子最大系数就是 slope。这是 LINES/FLOWS 那种
    # 「同一张表两份手写副本」，以后 _load_train 再加源，这条拦住不跟着漂。
    fr = fns.get("stage_race")
    ck(fr is not None, "找到 stage_race")
    if fr is not None:
        n_lt = sum(1 for n in ast.walk(fr) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "_load_train")
        n_own = sum(1 for n in ast.walk(fr) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", "")
                    in ("read_parquet", "glob", "neutralize"))
        ck(n_lt == 1 and n_own == 0,
           "stage_race 只经 _load_train 取训练表，不自己 glob/read_parquet/"
           "neutralize（和 shadow.fit 同源）")
        keys = {k.value for n in ast.walk(fr) if isinstance(n, ast.Dict)
                for k in n.keys if isinstance(k, ast.Constant)}
        ck({"winner", "source", "online_days"} <= keys,
           "model_race.json 记录 source / online_days（面板能看出这张表"
           "在哪份数据上算的）")

    # --- G4：标签抓不到就 fail-closed，`--stage all` 不许往下写状态 ---
    # 盘中派发（云端 learn.yml 的 workflow_dispatch 没有时刻闸）时 from_quotes
    # 返回空表，这里一旦改成 return 0，stage_learn 就会写 learning_status(date=T)，
    # 本机 16:40 起的窗口整天判「今天跑完了」，那天的真值标签再也抓不回来。
    sl_f = fns.get("stage_label")
    empties = [n for n in ast.walk(sl_f)
               if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
               and n.test.attr == "empty"
               and getattr(n.test.value, "id", "") == "raw"] if sl_f else []
    ck(len(empties) == 1
       and any(isinstance(b, ast.Return) and isinstance(b.value, ast.Constant)
               and b.value.value == 1 for b in empties[0].body),
       "stage_label 的 raw.empty 分支 return 1（取不到收盘价就 fail-closed）")
    mainf = fns.get("main")
    allb = None
    for n in (ast.walk(mainf) if mainf else []):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.comparators[0], ast.Constant)
                and n.test.comparators[0].value == "all"):
            allb = n
    lab = [n.lineno for n in ast.walk(allb) if isinstance(n, ast.Call)
           and getattr(n.func, "id", "") == "stage_label"] if allb else []
    lrn = [n.lineno for n in ast.walk(allb) if isinstance(n, ast.Call)
           and getattr(n.func, "id", "") == "stage_learn"] if allb else []
    guard = [n for n in ast.walk(allb)
             if isinstance(n, ast.If)
             and any(isinstance(r, ast.Return) for r in ast.walk(n))
             and lab and lrn and lab[0] < n.lineno < lrn[0]] if allb else []
    ck(bool(guard),
       "--stage all 里标签失败就 return，不进 stage_learn、不写 learning_status")


def check_race_source(c: dict) -> None:
    """模型对比取的训练表 = _load_train 那张合并表，且产物自带口径标签（F3-16）。

    离线：_load_train 和 model_select 全部替换成假实现，产物写临时目录。
    """
    print("\n模型对比的训练源")
    import json as _json
    import tempfile
    import eval_daily as ED
    from learn import model_select as MS, panel as LP

    days = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-09-15", "2026-09-16"]
    df = pd.DataFrame({
        "date": days,
        "code": ["600000", "600001", "600002", "600003", "600004"],
        "_src": ["backfill"] * 3 + ["online"] * 2,
    })
    td = Path(tempfile.mkdtemp(prefix="race_"))
    seen: dict = {}
    keep = (ED._load_train, ED.OUT, MS.walk_forward, MS.summarize, MS.pick)

    def fake_wf(d, cc, **k):
        seen["rows"] = len(d)
        seen["src"] = set(d["_src"]) if "_src" in d.columns else set()
        return pd.DataFrame({"fold": [1], "model": ["RankHuber"]})

    try:
        ED._load_train = lambda cc: (df, "backfill:backfill_2026-08-28.parquet"
                                         "+online:2天")
        ED.OUT = td
        MS.walk_forward = fake_wf
        MS.summarize = lambda res, n: pd.DataFrame([{
            "model": "RankHuber", "IC": 0.118, "ICIR": 1.679, "ICIR_lo": 1.546,
            "ICIR_hi": 1.81, "top_excess": 0.012, "hit": 0.55}])
        MS.pick = lambda s: ("RankHuber", "ICIR 最高")
        ck(ED.stage_race(c) == 0, "模型对比跑通（假模型层）")
        ck(seen.get("rows") == 5 and seen.get("src") == {"backfill", "online"},
           "喂进去的就是 _load_train 那张合并表，在线真值天在里面")
        j = _json.loads((td / "model_race.json").read_text(encoding="utf-8"))
        ck(j["source"].startswith("backfill") and "online" in j["source"],
           f"产物记下训练源：{j['source']}")
        ck(j["days"] == 5 and j["online_days"] == 2 and j.get("generated_at"),
           "产物记下天数 5 / 在线 2 天 / 生成时刻（面板能核对是不是旧表）")
        html = LP._race_block(j)
        ck("真采样 2 天" in html and "backfill" in html,
           "面板把训练源和在线天数印出来")
        # 旧产物（2026-09-02 那份没有这几个键）不许让面板崩
        old = LP._race_block({"winner": "RankHuber", "why": "x",
                              "table": j["table"]})
        ck("未记录" in old and "真采样 0 天" in old,
           "旧的 model_race.json 缺键也能出页面，标成未记录")
    finally:
        (ED._load_train, ED.OUT, MS.walk_forward, MS.summarize,
         MS.pick) = keep


def check_shadow_stat() -> None:
    print("\n影子转正证据")
    good = [{"date": f"d{i}", "base_top_excess": 0.0, "shadow_top_excess": 0.01,
             "base_ic": 0.0, "shadow_ic": 0.1} for i in range(35)]
    s = shadow.promotion_stat(good, 30, 0.9, 500)
    ck(s["ready"] and s["p_better"] > 0.99 and s["wins"] == 35 and s["ic_wins"] == 35,
       "影子稳定领先 35 天 -> 就绪，P≈1，占优 35/35")
    s8 = shadow.promotion_stat(good[:8], 30, 0.9, 500)
    ck((not s8["ready"]) and s8["need_days"] == 22 and s8["p_better"] > 0.99,
       "只有 8 天 -> 未就绪（还差 22 天），即使 P≈1 也不提案")
    mixed = [{**x, "shadow_top_excess": (0.01 if i % 2 else -0.01)}
             for i, x in enumerate(good)]
    sm = shadow.promotion_stat(mixed, 30, 0.9, 500)
    ck((not sm["ready"]) and 0.2 < sm["p_better"] < 0.8,
       "一半天赢一半天输 -> P 在 0.5 附近，不就绪")
    tie = [{**x, "shadow_top_excess": 0.0} for x in good]
    ck(not shadow.promotion_stat(tie, 30, 0.9, 200)["ready"],
       "两榜完全一样 -> 无证据，不就绪")
    ck(shadow.promotion_stat([], 30, 0.9, 200)["days"] == 0, "空输入不抛异常")


def check_report_send() -> None:
    """变更/提案邮件的发送接线。用假 SMTP 层，不联网。

    2026-09-04 前 report.send 把 (conf, subject, html) 按位置塞给
    mailer._send(msg, conf)，一旦真有变更被接受，邮件永远发不出去。
    """
    print("\n学习邮件接线")
    import mailer
    from learn import report as R
    got: dict = {}
    orig = mailer._send

    def fake(msg, c):
        got["subject"] = str(msg["Subject"])
        got["to"] = str(msg["To"])
        got["html"] = msg.get_body(preferencelist=("html",)) is not None
        got["conf_is_dict"] = isinstance(c, dict)

    keep = {k: os.environ.get(k) for k in
            ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "MAIL_TO", "SKIP_MAIL")}
    os.environ.update(SMTP_HOST="smtp.test", SMTP_PORT="587", SMTP_USER="u@test",
                      SMTP_PASS="x", MAIL_TO="a@test, b@test")
    os.environ.pop("SKIP_MAIL", None)
    mailer._send = fake
    try:
        R.send("2026-09-04", "<b>ok</b>", {})
        ck(got.get("conf_is_dict") and got.get("html")
           and got.get("subject", "").startswith("[参数更新]"),
           "send() 构造 EmailMessage 并按 (msg, conf) 交给 SMTP 层")
        ck("a@test" in got.get("to", "") and "b@test" in got.get("to", ""),
           "收件人来自 MAIL_TO，多个用逗号分")
        got.clear()
        R.send("2026-09-04", "<b>ok</b>", {}, subject="[提案] 测试")
        ck(got.get("subject") == "[提案] 测试", "自定义主题（提案邮件）生效")
        got.clear()
        os.environ["SKIP_MAIL"] = "1"
        R.send("2026-09-04", "<b>ok</b>", {})
        ck(not got, "SKIP_MAIL=1 时不发（本地 dry-run / 远端让位共用）")
        # F5-14：判定只有一份（mailer.skip_mail）。以前这里写死 == "1"，
        # 而三条业务线是真值判断：写 true 学习邮件照发、写 0 那三条全静音
        for v, want_sent in (("true", False), ("TRUE", False), ("1", False),
                             ("0", True), ("false", True), ("", True)):
            got.clear()
            os.environ["SKIP_MAIL"] = v
            R.send("2026-09-04", "<b>ok</b>", {})
            ck(bool(got) is want_sent,
               f"SKIP_MAIL={v!r} -> {'照发' if want_sent else '不发'}"
               "（和早盘/形态/起涨预测同一判据）")
        rsrc = (ROOT / "src" / "learn" / "report.py").read_text(encoding="utf-8")
        ck("mailer.skip_mail()" in rsrc and 'SKIP_MAIL") == "1"' not in rsrc,
           "report.send 调 mailer.skip_mail()，不再自己比字符串")
        os.environ["SKIP_MAIL"] = "1"
        html = R.build_proposal_html("2026-09-04", shadow.promotion_stat(
            [{"date": "2026-09-01", "base_top_excess": 0.0, "shadow_top_excess": 0.01,
              "base_ic": 0.0, "shadow_ic": 0.1, "overlap": 0.5}] * 31, 30, 0.9, 100),
            [{"date": "2026-09-01", "base_top_excess": 0.0, "shadow_top_excess": 0.01,
              "base_ic": 0.0, "shadow_ic": 0.1, "overlap": 0.5}])
        ck("提案" in html and "切换意味着什么" in html, "提案邮件正文能生成")
    finally:
        mailer._send = orig
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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

    # W2-6：代码在用 g.get 取默认值的旋钮必须写进配置，否则用户改不了也看不见
    g = c["learning"]["gate"]
    ck(int(g["oos_blocks"]) == 3 and int(g["oos_min_blocks_better"]) == 2,
       "闸门 2 的分块旋钮写进 config（oos_blocks=3 / 至少 2 块同向）")
    esrc = (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")
    gsrc = (ROOT / "src" / "learn" / "gate.py").read_text(encoding="utf-8")
    ck('g.get("oos_blocks"' in esrc and 'g.get("oos_min_blocks_better"' in gsrc,
       "两处仍按同名键取（配置里的值和代码默认值是同一个旋钮）")
    # 真的碰到这条规则：三块里只有一块同向时，要求 2 块不过、要求 1 块过。
    # 只看「样本外改善」那一条 check，不看 v.accepted —— 冷却期要读
    # state/theta_history.jsonl，自测不许依赖 state/
    t0 = C.theta0(box)
    t1 = dict(t0)
    t1["screen.gap_pct_peak"] = t0["screen.gap_pct_peak"] + 0.1
    days = [f"2026-08-{d:02d}" for d in range(1, 26)]
    base = dict(theta_new=t1, theta_old=t0, box=box, n_days=len(days),
                all_days=days, today=days[-1], boot_p=0.95,
                oos_new=0.05, oos_old=0.03, churn={d: 0.1 for d in days})
    blocks = [0.0095, -0.0175, -0.0134]     # 09-16 那个提案的真实三块

    def _oos_check(need: int):
        v = gate.evaluate(g={**g, "oos_min_blocks_better": need},
                          block_delta=blocks, **base)
        return next(x for x in v.checks if x.name == "样本外改善")

    ck(not _oos_check(2).passed and _oos_check(1).passed,
       "1/3 块同向：要求 2 块不过、要求 1 块过（这个旋钮真的接上了）")

    # learnable_dims 是箱裁剪的唯一真相（restrict_box 按它执行）
    ld = c["learning"]["backfill"]["learnable_dims"]
    ck(isinstance(ld, list) and ld and "trend" not in ld and "volume" not in ld,
       f"learning.backfill.learnable_dims 存在且不含 trend/volume（{ld}）")


def check_council() -> None:
    """学习会诊（docs/council.md）。钉五件事：脏输出改写、台账幂等与状态机、
    起涨真值的窗口语义、实验过闸判定、fail-open 接线；全部在临时目录里做，不碰 state/。
    """
    print("\n学习会诊")
    import json
    import tempfile
    from learn.council import schemas as S, run as CR, experiments as EX, panel as CP

    # 1. 脏输出改写：枚举越界、超长、缺字段都能变成合法对象
    dirty = {"lens": "gap_where", "summary": "x" * 900,
             "findings": [{"line": "火星", "claim": "c" * 500, "evidence": "", "magnitude": 1,
                           "confidence": "极高"}, "不是对象"],
             "proposals": [{"kind": "改代码", "line": "morning", "target": "t", "change": "c",
                            "rationale": "r", "expected_effect": "e", "test_plan": "p",
                            "priority": "P9", "params": {"a": 1}}]}
    cl = S.sanitize_lens(dirty, "gap_where")
    ck(len(cl["summary"]) == 600 and cl["findings"][0]["line"] == "both"
       and cl["findings"][0]["confidence"] == "低" and len(cl["findings"]) == 1,
       "视角输出：越界枚举改保守值、超长截断、非对象丢弃")
    ck(cl["proposals"][0]["kind"] == "process" and cl["proposals"][0]["priority"] == "P2"
       and cl["proposals"][0]["params"] == {"a": 1}, "提案：kind/priority 越界改保守值，params 原样保留")
    ch = S.sanitize_chair({"gap": [{"line": "breakout", "where": "w", "expected": "12.6",
                                    "actual": "nan", "unit": "%", "n": "8"}],
                           "why": [{"cause": "市场环境", "weight": 3, "evidence": ""},
                                   {"cause": "外星人", "weight": 1, "evidence": ""}],
                           "noise_or_real": {"verdict": "?", "p_real": 7},
                           "proposals": [], "narrative": "n"})
    ck(abs(sum(w["weight"] for w in ch["why"]) - 1) < 1e-9 and ch["why"][1]["cause"] == "未知",
       "主审：原因权重归一，越界原因改「未知」")
    ck(ch["noise_or_real"]["verdict"] == "样本不足无法判断" and ch["noise_or_real"]["p_real"] == 1.0
       and ch["gap"][0]["actual"] == 0.0 and ch["gap"][0]["n"] == 8,
       "主审：判断枚举越界改保守值，概率裁到 [0,1]，数字字段容错")

    # 2. 台账：同一提案重跑不重复；状态机；needs_human 自动标
    with tempfile.TemporaryDirectory() as td:
        old = (CR.STATE, CR.LEDGER, CR.DECISIONS, CR.LATEST)
        CR.STATE = Path(td); CR.LEDGER = CR.STATE / "p.jsonl"
        CR.DECISIONS = CR.STATE / "d.json"; CR.LATEST = CR.STATE / "l.json"
        try:
            props = [S.sanitize_proposal({"kind": "threshold", "line": "breakout", "target": "SCORE_MIN",
                                          "change": "97->96", "rationale": "r", "expected_effect": "e",
                                          "test_plan": "t", "priority": "P1", "params": {"SCORE_MIN": 96}}),
                     S.sanitize_proposal({"kind": "feature_add", "line": "breakout", "target": "北向",
                                          "change": "加", "rationale": "r", "expected_effect": "e",
                                          "test_plan": "t", "priority": "P2"})]
            a1 = CR.append_proposals("2026-09-16", props)
            a2 = CR.append_proposals("2026-09-16", props)
            ck(len(a1) == 2 and len(a2) == 0, "台账：同一天同一提案重跑不重复追加")
            view = {r["id"]: r for r in CR.ledger_view()}
            st = sorted(r["status"] for r in view.values())
            ck(st == ["needs_human", "pending"], f"台账：能自动测的 pending，其它 needs_human（{st}）")
            pid = a1[0]["id"]
            CR.write_decision(pid, "passed", result={"detail": "ok"})
            ck(CR.ledger_view()[-1]["status"] == "passed" if CR.ledger_view()[-1]["id"] == pid
               else {r["id"]: r for r in CR.ledger_view()}[pid]["status"] == "passed",
               "台账：状态改写落盘并能读回")
            ck(a1[0]["id"] == CR._pid("2026-09-16", props[0]) and len(pid) == 15,
               "提案 id 由日期+内容哈希决定（可复现）")
        finally:
            CR.STATE, CR.LEDGER, CR.DECISIONS, CR.LATEST = old

    # 3. 起涨真值：之后 20 根、不含当天、avail 计数
    import truth as T  # noqa: E402  (src/breakout 在 sys.path 里)
    n = 30
    close = np.full(n, 10.0)
    high = np.full(n, 10.5)
    high[25] = 16.0            # 第 25 根冲高 60%
    px = pd.DataFrame({"code": "000001", "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
                       "high": high, "close": close})
    fm = T._future_max(px, 20)
    ck(fm["avail"].iloc[0] == 20 and fm["avail"].iloc[-1] == 0 and fm["avail"].iloc[-3] == 2,
       "真值：avail = min(20, 之后还有几根)")
    ck(fm["fut_max"].iloc[5] == 16.0 and fm["fut_max"].iloc[4] == 10.5 and fm["fut_max"].iloc[25] == 10.5,
       "真值：fut_max 看的是 t+1..t+20，不含当天（第 5 根看得到第 25 根，第 4 根看不到）")
    lo, hi = T.wilson(1, 10)
    ck(0.01 < lo < 0.1 < hi < 0.5, f"Wilson 区间 1/10 -> ({lo:.3f}, {hi:.3f})")

    # 4. 实验过闸判定：命中率不掉 + 样本不少 才过；连续≥2天 提升 2SE 也过
    base = {"ge1": {"n": 800, "hit": 0.126, "se": 0.0117}, "ge2": {"n": 200, "hit": 0.15, "se": 0.025}}
    ok1, _ = EX._pass_breakout(base, {"ge1": {"n": 700, "hit": 0.12, "se": 0.012},
                                       "ge2": {"n": 150, "hit": 0.14, "se": 0.028}})
    ok2, _ = EX._pass_breakout(base, {"ge1": {"n": 500, "hit": 0.13, "se": 0.015},
                                       "ge2": {"n": 150, "hit": 0.14, "se": 0.028}})
    ok3, _ = EX._pass_breakout(base, {"ge1": {"n": 500, "hit": 0.09, "se": 0.013},
                                       "ge2": {"n": 100, "hit": 0.25, "se": 0.043}})
    ck(ok1 and not ok2 and ok3, "过闸：命中不掉且样本≥80% 过；样本掉太多不过；连续档提升≥2SE 过")
    d = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3, "code": ["a", "b", "c"] * 2,
                      "rank": [1, 2, 3] * 2, "score": [99, 98, 90, 99, 97, 80],
                      "y_up": [1, 0, 1, 0, 1, 0]})
    w = EX._w5(d, 97, 10)
    ck(w["ge1"]["n"] == 4 and w["ge2"]["n"] == 2 and w["ge3"]["n"] == 0
       and abs(w["ge1"]["hit"] - 0.5) < 1e-9,
       "生产口径重算：≥97 且前 10 -> 4 行；a、b 两天都在榜 -> 连续 2 天 2 行")

    # 5. fail-open 接线 + 提纲齐全 + 面板无状态能出页
    src = (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)
               and x.name == "stage_council"), None)
    rets = [x for x in ast.walk(fn) if isinstance(x, ast.Return)] if fn else []
    ck(fn is not None and any(isinstance(x, ast.Try) for x in fn.body)
       and all(isinstance(r.value, ast.Constant) and r.value.value == 0 for r in rets),
       "stage_council 整体在 try 里且只会 return 0（会诊挂了不影响学习流程退出码）")
    from learn.council import agents as AG
    missing = [ln for ln in list(S.LENSES) + ["chair"] if not (AG.PROMPTS / f"{ln}.md").exists()]
    ck(not missing and (AG.PROMPTS / "common.md").exists(), f"每个视角都有提纲文件（缺 {missing}）")
    ck(set(AG.TOOLS) == set(S.LENSES) | {"chair"} and all(
        "Bash(" not in AG.TOOLS["chair"] and "Write" not in v and "Edit" not in v
        for v in AG.TOOLS.values()), "视角工具白名单：主审不查数据，谁都不能写文件")
    with tempfile.TemporaryDirectory() as td:
        old = (CR.STATE, CR.LEDGER, CR.DECISIONS, CR.LATEST, CP.OUTL)
        CR.STATE = Path(td) / "s"; CR.LEDGER = CR.STATE / "p.jsonl"
        CR.DECISIONS = CR.STATE / "d.json"; CR.LATEST = CR.STATE / "l.json"
        CP.OUTL = Path(td) / "o"
        try:
            out = CP.build()
            html = out.read_text(encoding="utf-8")
            ck("学习会诊" in html and "还没跑过" in html and "<script>" in html,
               "会诊面板：没有任何状态也能出页")
        finally:
            CR.STATE, CR.LEDGER, CR.DECISIONS, CR.LATEST, CP.OUTL = old
    js = json.dumps(S.LENS_SCHEMA); js2 = json.dumps(S.CHAIR_SCHEMA)
    ck('"required"' in js and '"required"' in js2 and "additionalProperties" in js,
       "两份 schema 能序列化（CLI --json-schema 用）")


def check_brief_pick() -> None:
    """归因输入的 worst/best：互斥、符号正确、rank 是当日分数名次（F3-4）。"""
    print("\n归因样本挑法")
    from learn import brief as BR, llm_local as LL
    sc = np.linspace(70.0, 30.0, 12)
    # 刻意让「前 20」里有赚钱的、「后 50%」里有亏钱的 —— 旧写法正是在这
    # 两处出错（worst 里有 ytil=+0.83、best 里有 ytil=-0.96）
    yt = np.array([+0.83, -1.20, +0.40, -0.35, +1.10, -0.05,
                   -0.02, +0.83, -0.51, -0.96, +0.60, -0.70])
    src = pd.DataFrame({"code": [f"{i:06d}" for i in range(1, 13)],
                        "sc": sc, "ytil": yt})
    shuffled = src.sample(frac=1.0, random_state=3).reset_index(drop=True)
    full, worst, best = BR.pick_worst_best(shuffled, 8, 8)

    # 教训 26：先证明这批数据真的踩到「小池子两组必然相交」那条线
    ck(set(full.head(20)["code"]) & set(full.tail(6)["code"])
       == set(full.tail(6)["code"]),
       "池 12 只：旧写法的 head(20) 完全包含 tail(6)，两组必然相交")
    ck(not (set(worst["code"]) & set(best["code"])), "worst 与 best 互斥")
    ck(len(worst) == 7 and bool((worst["ytil"] < 0).all()),
       f"worst 全是真亏了的（实得 {len(worst)} 只）")
    ck(len(best) == 2 and bool((best["ytil"] > 0).all()),
       f"best 全是真涨了的（实得 {len(best)} 只）")
    rank_of = {r.code: i + 1 for i, r in enumerate(
        src.sort_values("sc", ascending=False).itertuples(index=False))}
    ck(all(rank_of[r.code] == r.rank for r in full.itertuples(index=False)),
       "rank 是当日按分数降序的名次，不是按 ytil 排完的序号")
    ck(all(rank_of[c_] <= 20 for c_ in worst["code"]), "worst 取自分数前 20")
    ck(set(best["code"]) <= set(full.tail(6)["code"]), "best 取自后 50%")
    # 小池子不许崩：3 行、1 行，tail 至少 1 行
    for k in (3, 1):
        f2, w2, b2 = BR.pick_worst_best(src.head(k), 8, 8)
        ck(len(f2) == k and len(f2.tail(max(k // 2, 1))) >= 1,
           f"{k} 行也能挑（tail 至少 1 行），不抛异常")

    # 第二道防线：codes 去重，items 不许出现同一只票两条
    got = LL._sanitize({"day_regime": "正常", "items": []},
                       "2026-09-16", ["000001", "000001", "000002"])
    ck(len(got["items"]) == 2
       and len({x["code"] for x in got["items"]}) == 2,
       "_sanitize 去重：重复 code 只产一条 item（面板 cause 计数不再双计）")

    # AST：挑法只有一份实现，stage_brief 不许再自己写 head/nsmallest
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "stage_brief")
    used = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "pick_worst_best"]
    bad = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") in ("nsmallest", "nlargest")]
    ck(len(used) == 1 and not bad,
       "stage_brief 只调 brief.pick_worst_best，不再自己 nsmallest/nlargest")


def check_online_oos(c: dict) -> None:
    """闸门 7 只许用样本外段里的在线日（F3-6）。"""
    print("\n闸门 7 的样本外过滤")
    import eval_daily as ED
    from learn import optimize as OPT
    days = [f"d{i:03d}" for i in range(100)]
    tr, te = OPT.split_days(days, 0.20)
    ck(len(tr) == 80 and len(te) == 20, "100 天按 oos_frac=0.20 切成 80/20")

    # 40 个在线日：一半已经被拟合段吃进去了
    df = pd.DataFrame({"date": days,
                       "_src": ["backfill"] * 60 + ["online"] * 40})
    all_on = df[df["_src"] == "online"]
    got = ED._online_oos(all_on, te)
    ck(len(all_on) == 40 and len(got) == 20
       and set(got["date"]) == set(te) and not (set(got["date"]) & set(tr)),
       "40 个在线日里 20 天在拟合段（样本内），闸门 7 一个都不拿")

    # 今天的情形：在线天全在样本外段，过滤一行都不改（数字逐位不变）
    df2 = pd.DataFrame({"date": days,
                        "_src": ["backfill"] * 85 + ["online"] * 15})
    on2 = df2[df2["_src"] == "online"]
    ck(set(ED._online_oos(on2, te)["date"]) == set(on2["date"]),
       "在线 15 天全在样本外段 -> 过滤后逐行不变（今天这种情形）")
    ck(ED._online_oos(df.iloc[0:0], te).empty, "空表不抛异常")

    # 权重为 0 的天不算进闸门 7 的天数（F2-8）
    df3 = _multiday(6, 100, 700)
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    dayw = {"2026-01-04": 0.0, "2026-01-05": 0.0, "2026-01-06": 0.0}
    p = OPT.Problem(df3, c, box, t0, t0, dayw, 10, 1.345)
    ck(ED._online_days(p) == 3 and df3["date"].nunique() == 6,
       "6 个在线日里 3 天归因「数据异常」（权重 0）-> 闸门 7 只算 3 天")
    p2 = OPT.Problem(df3, c, box, t0, t0, None, 10, 1.345)
    ck(ED._online_days(p2) == 6, "全 1.0 权重时天数就是在线天数")
    # 和 bootstrap_better 的 keep 同口径：它自助的 m 就是这个数
    ck(int((np.asarray(p.day_w, float) > 0).sum())
       == ED._online_days(p), "天数口径 = bootstrap_better 的 keep")

    # AST：闸门 7 的数据来自 _online_oos，天数来自 _online_days
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "_judge")
    oos = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
           and isinstance(n.value, ast.Call)
           and getattr(n.value.func, "id", "") == "_online_oos"]
    ck(len(oos) == 1 and isinstance(oos[0].value.args[1], ast.Attribute)
       and oos[0].value.args[1].attr == "dates"
       and getattr(oos[0].value.args[1].value, "id", "") == "p_te",
       "_judge 里 dfo 由 _online_oos(..., p_te.dates) 产出")
    dfo_name = {t.id for t in oos[0].targets if isinstance(t, ast.Name)}
    prob = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and getattr(n.value.func, "attr", "") == "Problem"]
    ck(len(prob) == 1 and isinstance(prob[0].value.args[0], ast.Name)
       and prob[0].value.args[0].id in dfo_name,
       "闸门 7 的 Problem 用的就是过滤后的那张表")
    od = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == "online_days"
                  for t in n.targets) and isinstance(n.value, ast.Call)]
    ck(len(od) == 1 and getattr(od[0].value.func, "id", "") == "_online_days"
       and not any(getattr(x, "attr", "") == "nunique"
                   for x in ast.walk(od[0].value)),
       "online_days 由 _online_days 算，不再是 dfo['date'].nunique()")


def _online_frame(seed: int = 800) -> pd.DataFrame:
    """两天在线表：带快照自带的 score / score_raw / rejected + y / y_raw。"""
    df = _multiday(2, 30, seed)
    g = np.random.default_rng(seed + 7)
    n = len(df)
    df["score_raw"] = g.uniform(30.0, 90.0, n)
    df["score"] = df["score_raw"].round(1)
    df["rejected"] = pd.Series([None] * n, dtype=object)
    # 3 行在快照里被硬性排除（vscore 重放会放行它们，正是 F3-13 的漂移来源）
    df.loc[df.index[:3], "rejected"] = "竞价额低于 300 万"
    # y_raw ≠ y：证明汇报用的是未缩尾那列（F2-10）
    df["y_raw"] = df["y"] + 0.003
    return df


def check_online_daily(c: dict) -> None:
    """在线真值日的成绩必须是**当天真发出去的那张榜**（F3-13 + F2-10）。"""
    print("\n在线真值日的实发榜 vs 回放榜")
    import copy
    from learn import online_eval as OE
    df = _online_frame()
    cost = 2.0 * float(c["learning"]["label"]["cost_bp"]) / 1e4
    ms = float(c["output"]["min_score"])

    adm = OE.admitted_mask(df)
    snt = OE.sent_mask(df, ms)
    ck(int((~adm).sum()) == 3, "快照自带的 rejected 真的被认出来了（3 行）")
    ck(0 < int(snt.sum()) < int(adm.sum()),
       f"45 分线真的截掉了一批（过准入 {int(adm.sum())}、实发 {int(snt.sum())}）")
    ck(not bool(snt[~adm].any()), "被硬性排除的行一只都不在实发榜里")

    sent = OE.daily_sent(df, 10, ms, cost)
    rep = OE.daily_replay(df, c, 10, cost)
    ck(len(sent) == 2 and len(rep) == 2, "两天各一行")
    # 逐日核对：按 score_raw 降序取前 10，y_raw 均值再扣成本
    d0 = df[df["date"] == "2026-01-01"]
    top = d0[OE.sent_mask(d0, ms)].sort_values(
        "score_raw", ascending=False, kind="stable").head(10)
    ck(abs(sent[0]["top_excess"]
           - (float(top["y_raw"].mean()) - cost)) < 1e-12,
       "实发榜超额 = 快照 score_raw 前 10 的 y_raw 均值减双边成本")
    ck(sent[0]["sent_n"] == len(top) and sent[0]["pool"] == int(adm[d0.index].sum()),
       "sent_n / pool 两个数分开记")
    ck(abs(sent[0]["top_excess"] - rep[0]["top_excess"]) > 1e-9,
       "实发榜和回放榜不是同一个数（回放用当前参数重新打分）")

    # 换一套参数：实发榜逐位不变，回放榜跟着变
    c2 = copy.deepcopy(c)
    c2["output"]["min_score"] = 60.0
    ck(OE.daily_sent(df, 10, ms, cost) == sent,
       "改配置后实发榜逐位不变（它只读快照，改参数改不动历史成绩）")
    ck(OE.daily_replay(df, c2, 10, cost) != rep, "回放榜跟着配置变")

    # 没有 score_raw 的旧快照：退回 score，不抛
    old = df.drop(columns=["score_raw"])
    s_old = OE.daily_sent(old, 10, ms, cost)
    ck(len(s_old) == 2 and s_old[0]["top_excess"] is not None,
       "08-24~09-15 那种只有 round(score,1) 的旧快照照样算得出来")
    # 一只都过不了线的日子记 None，不拿 nan 冒充 0
    empty = OE.daily_sent(df, 10, 999.0, cost)
    ck(all(x["top_excess"] is None and x["sent_n"] == 0 for x in empty),
       "全场不过线 -> top_excess=None、sent_n=0")

    # AST：status["daily"] 只能来自 daily_sent
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "_status_extras")
    names = {getattr(n.func, "attr", "") for n in ast.walk(fn)
             if isinstance(n, ast.Call)}
    ck("daily_sent" in names and "daily_replay" in names,
       "_status_extras 同时算实发榜和回放榜")
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "daily"
                       for t in n.targets) and isinstance(n.value, ast.Call)]
    ck(len(assigns) == 1
       and getattr(assigns[0].value.func, "attr", "") == "daily_sent",
       "daily 由 daily_sent 赋值（不是 vscore.score_df 重放）")


def check_report_metrics(c: dict) -> None:
    """对外汇报的收益：未缩尾 + 扣双边成本（F2-10）。"""
    print("\n汇报口径（未缩尾 + 扣成本）")
    import copy
    from learn import optimize as OPT
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    df = _multiday(3, 120, 310)
    # 模拟缩尾：y 是被截过的，y_raw 是原值。差 0.003 是为了让「用错列」
    # 在断言里一眼看得出来
    df["y_raw"] = df["y"] + 0.003
    c13 = copy.deepcopy(c)
    c13["learning"]["label"]["cost_bp"] = 13
    c0 = copy.deepcopy(c)
    c0["learning"]["label"]["cost_bp"] = 0
    p13 = OPT.Problem(df, c13, box, t0, t0, None, 10, 1.345)
    p0 = OPT.Problem(df, c0, box, t0, t0, None, 10, 1.345)
    m13, m0 = p13.metrics(t0, 10), p0.metrics(t0, 10)

    # 期望值自己算一遍：逐日生产口径前 10 的 y_raw 均值
    ys = []
    for day, gday in df.groupby("date"):
        s_, rej_ = vscore.score_df(gday, c)
        o = OPT.production_order(s_, rej_, c, 10)
        if o.size:
            ys.append(float(gday.iloc[o]["y_raw"].mean()))
    want = float(np.mean(ys))
    ck(abs(m0["top_excess"] - want) < 1e-12,
       "cost_bp=0 时前 10 超额 = 实发清单的 y_raw 均值（用的是未缩尾列）")
    ck(abs(m13["top_excess"] - (want - 0.0026)) < 1e-12,
       "cost_bp=13 时正好少 26bp（双边成本真的扣了）")
    ck(abs(m0["top_excess"] - m13["top_excess"] - 0.0026) < 1e-12,
       "两个 Problem 只差一个常数成本")
    # 教训 26：断言要真的碰到那条规则 —— 这里证明用的**不是**缩尾的 y
    ys_w = []
    for day, gday in df.groupby("date"):
        s_, rej_ = vscore.score_df(gday, c)
        o = OPT.production_order(s_, rej_, c, 10)
        if o.size:
            ys_w.append(float(gday.iloc[o]["y"].mean()))
    ck(abs(m0["top_excess"] - float(np.mean(ys_w))) > 1e-6,
       f"和缩尾列算出来的 {float(np.mean(ys_w)):+.5f} 明显不同（真的换了列）")
    # 没有 y_raw 的旧表：退回 y，不抛
    p_old = OPT.Problem(df.drop(columns=["y_raw"]), c13, box, t0, t0, None,
                        10, 1.345)
    ck(abs(p_old.metrics(t0, 10)["top_excess"]
           - (want - 0.003 - 0.0026)) < 1e-12,
       "旧训练表缺 y_raw 时退回 y（仍扣成本，不崩）")


def check_status_errors(c: dict) -> None:
    """fail-open 的两个分支必须往 status 写可查询的错误键（F3-7，教训 16）。"""
    print("\n学习面板 fail-open 的失败痕迹")
    import json as _json
    import tempfile
    import eval_daily as ED
    from learn import online_eval as OE, panel as LP, report as R
    from learn import shadow as SH

    # A. AST：凡是只写 log.warning 的 except，必须同时写一个 *_error 键
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "_status_extras")
    keys = set()
    bad = []
    for h in [x for x in ast.walk(fn) if isinstance(x, ast.ExceptHandler)]:
        warns = [n for n in ast.walk(h) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "warning"]
        if not warns:
            continue
        found = [n for n in h.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Subscript)
                         and getattr(t.value, "id", "") == "status"
                         and isinstance(t.slice, ast.Constant)
                         and str(t.slice.value).endswith("_error")
                         for t in n.targets)]
        if found:
            keys |= {t.slice.value for n in found for t in n.targets
                     if isinstance(t, ast.Subscript)}
        else:
            bad.append(h.lineno)
    ck(keys == {"shadow_error", "panel_error"} and not bad,
       f"两条 fail-open 分支都往 status 写错误键（实得 {sorted(keys)}）")

    # B. 功能：三种失败各走一遍，全部写临时目录
    td = Path(tempfile.mkdtemp(prefix="extras_"))
    keep = (R.STATUS, LP.STATE, LP.OUTL, SH.MODEL, SH.LEDGER, SH.PROPOSAL,
            OE.daily_sent, SH.fit, LP.build)
    R.STATUS = td / "learning_status.json"
    LP.STATE, LP.OUTL = td, td / "out"
    SH.MODEL = td / "shadow_model.json"
    SH.LEDGER = td / "shadow_compare.json"
    SH.PROPOSAL = td / "shadow_proposal.json"
    df = _online_frame(820)
    try:
        def boom(*a, **k):
            raise RuntimeError("炸了")

        SH.fit = boom
        st: dict = {"date": "2026-01-02"}
        ED._status_extras(c, st, df, df, "2026-01-02", True)
        j = _json.loads(R.STATUS.read_text(encoding="utf-8"))
        ck(j.get("shadow_error", "").startswith("RuntimeError")
           and len(j.get("daily") or []) == 2 and "panel_error" not in j,
           "影子段挂掉：status 留 shadow_error，daily 照常写出")
        SH.fit = keep[7]

        LP.build = boom
        st = {"date": "2026-01-02"}
        ED._status_extras(c, st, df, df, "2026-01-02", True)
        ck(st.get("panel_error", "").startswith("RuntimeError"),
           "面板生成挂掉：status 留 panel_error")
        LP.build = keep[8]

        OE.daily_sent = boom
        st = {"date": "2026-01-02"}
        ED._status_extras(c, st, df, df, "2026-01-02", True)
        ck(st.get("panel_error", "").startswith("RuntimeError")
           and "online_days" not in st,
           "逐日指标就挂掉：panel_error 有值，online_days 根本没写")
        OE.daily_sent = keep[6]

        # 成功那次不留错误键（status 每次运行新建，不会粘上一次的）
        st = {"date": "2026-01-02"}
        ED._status_extras(c, st, df, df, "2026-01-02", True)
        ck("panel_error" not in st and "shadow_error" not in st
           and (LP.OUTL / "learn.html").exists(),
           "全跑通时不留错误键，learn.html 出得来")
        # 面板和状态行都要显示得出来
        head = LP._head({"date": "2026-01-02", "shadow_error": "RuntimeError: 炸了"})
        ck("异常" in head, "面板顶部把异常写成徽章（不再一直标「最新」）")
        R.STATUS.write_text(_json.dumps(
            {"date": "2026-01-02", "n_days": 1, "panel_error": "X: y",
             "verdict": {"checks": [{"name": "a", "passed": True}]}},
            ensure_ascii=False), encoding="utf-8")
        ck("学习面板异常" in R.status_line(),
           "竞价面板底部那行也带上「学习面板异常」")
    finally:
        (R.STATUS, LP.STATE, LP.OUTL, SH.MODEL, SH.LEDGER, SH.PROPOSAL,
         OE.daily_sent, SH.fit, LP.build) = keep


def check_held_wiring(c: dict) -> None:
    """Opus 审稿搁置时，裁决必须改写成「未接受」（F3-9）。"""
    print("\n审稿搁置")
    import json as _json
    import tempfile
    import eval_daily as ED
    import mailer
    from learn import panel as LP, report as R

    td = Path(tempfile.mkdtemp(prefix="held_"))
    keep = (ED.STATE, R.STATUS, gate.VERDICTS, gate.HISTORY, LP.STATE,
            LP.OUTL, mailer.send_alert)
    sent: list[str] = []
    ED.STATE = td
    R.STATUS = td / "learning_status.json"
    gate.VERDICTS = td / "verdict_log.jsonl"
    gate.HISTORY = td / "theta_history.jsonl"
    LP.STATE, LP.OUTL = td, td / "out"
    mailer.send_alert = lambda text: sent.append(text)
    try:
        v = gate.Verdict(True,
                         [gate.Check(f"闸{i}", True, "") for i in range(7)],
                         {"scoring.weights.gap": (0.20, 0.22)}, {})
        status = {"date": "2026-09-16", "n_days": 100}
        review = {"stance": "反对", "points": ["方向和归因矛盾"]}
        ED._hold_change("2026-09-16", v, status,
                        {"scoring.weights.gap": 0.22}, {}, review,
                        "backfill", 100)
        j = _json.loads(R.STATUS.read_text(encoding="utf-8"))
        ck(j["verdict"]["accepted"] is False and j.get("held") is True
           and any(x["name"] == "Opus 审稿" and not x["passed"]
                   for x in j["verdict"]["checks"]),
           "learning_status 里裁决改写成未接受，并多一条「Opus 审稿 不过」")
        row = gate.read_verdicts()[-1]
        ck(row["accepted"] is False and row.get("held") is True
           and "Opus 审稿" in row["failed"],
           "verdict_log 那一行也是未接受（不跑 apply-held 也不会留错记录）")
        h = _json.loads((td / "held_change.json").read_text(encoding="utf-8"))
        ck(h["verdict"]["accepted"] is True
           and h["theta"] == {"scoring.weights.gap": 0.22},
           "held 文件存的是**改写前**的裁决（apply-held 要按七道闸原样落地）")
        ck(not (td / "learned.yaml").exists()
           and not gate.HISTORY.exists(),
           "参数没写、theta_history 没记")
        line = R.status_line()
        ck("已变更" not in line and "搁置" in line,
           f"竞价面板状态行不写「已变更」（实得：{line[-30:]}）")
        html = (LP.OUTL / "learn.html").read_text(encoding="utf-8")
        ck("参数已变更" not in html and "被审稿搁置" in html,
           "learn.html 被重画，第 ⑤ 格写「被审稿搁置」而不是「参数已变更」")
        ck(len(sent) == 1 and "搁置" in sent[0], "发了一封搁置通知")
    finally:
        (ED.STATE, R.STATUS, gate.VERDICTS, gate.HISTORY, LP.STATE,
         LP.OUTL, mailer.send_alert) = keep

    # AST：搁置逻辑只有一份，且不碰 A.write
    fns = {n.name: n for n in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(n, ast.FunctionDef)}
    sl, hc = fns["stage_learn"], fns["_hold_change"]
    n_hold = sum(1 for n in ast.walk(sl) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_hold_change")
    n_write = sum(1 for n in ast.walk(sl) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", "") == "write"
                  and getattr(getattr(n.func, "value", None), "id", "") == "A")
    ck(n_hold == 1 and n_write == 1,
       "stage_learn 恰好调一次 _hold_change 和一次 A.write")
    inner = {getattr(n.func, "attr", "") or getattr(n.func, "id", "")
             for n in ast.walk(hc) if isinstance(n, ast.Call)}
    ck({"log_verdict", "save_status", "Check"} <= inner and "write" not in inner,
       "_hold_change 改写裁决 + 记 verdict_log + 落状态，且绝不写参数")


def check_default_date(c: dict) -> None:
    """缺省日期 = 最近一个已收盘交易日，未收盘/非交易日一律拒绝（F3-10）。"""
    print("\n缺省日期与日期闸")
    import datetime as _dt
    import tempfile
    import eval_daily as ED
    import local_run as LR
    from learn import report as R

    keep_td = LR._TD.get("s")
    keep_now = LR.now_bj
    keep = (R.STATUS, gate.VERDICTS)
    td = Path(tempfile.mkdtemp(prefix="date_"))
    R.STATUS = td / "learning_status.json"
    gate.VERDICTS = td / "verdict_log.jsonl"
    # 假日历，绝不联网也不读 state/
    LR._TD["s"] = {"2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
                   "2026-09-21"}
    bj = _dt.timezone(_dt.timedelta(hours=8))
    try:
        ck(ED.last_closed_bj(_dt.datetime(2026, 9, 17, 1, 0, tzinfo=bj))
           == "2026-09-16", "北京 01:00 的缺省日是昨天，不是今天")
        ck(ED.last_closed_bj(_dt.datetime(2026, 9, 17, 15, 4, tzinfo=bj))
           == "2026-09-16", "15:05 之前仍算昨天")
        ck(ED.last_closed_bj(_dt.datetime(2026, 9, 17, 15, 6, tzinfo=bj))
           == "2026-09-17", "15:05 之后才算当天")
        ck(ED.last_closed_bj(_dt.datetime(2026, 9, 19, 12, 0, tzinfo=bj))
           == "2026-09-18", "周六退回周五")

        LR.now_bj = lambda: _dt.datetime(2026, 9, 17, 1, 0, tzinfo=bj)
        ck(ED.stage_learn(c, "2026-09-17", False) == 2,
           "还没收盘的日子 -> 退 2，不拟合")
        ck(ED.stage_learn(c, "2026-09-13", False) == 2,
           "非交易日 -> 退 2")
        ck(ED.stage_council(c, "2026-09-17") == 0
           and not (td / "council").exists(),
           "会诊同样被拦住（控制台按钮不带 --date），但只 return 0")
        ck(not R.STATUS.exists() and not gate.VERDICTS.exists(),
           "拒绝时一个字节都不落盘")
    finally:
        R.STATUS, gate.VERDICTS = keep
        LR.now_bj = keep_now
        if keep_td is None:
            LR._TD.pop("s", None)
        else:
            LR._TD["s"] = keep_td

    # AST：today_bj() 只许出现在 intraday/backfill 那个分支里
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "main")
    branch = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.ops[0], ast.In)
                and isinstance(n.test.comparators[0], ast.Tuple)
                and {getattr(e, "value", None)
                     for e in n.test.comparators[0].elts}
                == {"intraday", "backfill"}):
            branch = n
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "today_bj"]
    inside = [n for n in ast.walk(branch) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "today_bj"] if branch else []
    ck(branch is not None and len(calls) == 1 and len(inside) == 1,
       "main 里 today_bj() 只在 intraday/backfill 分支出现，其余走 last_closed_bj")


def check_label_guard(c: dict) -> None:
    """盘中不许抓标签：先判时刻，再联网（F3-15）。"""
    print("\n标签的收盘闸")
    import datetime as _dt
    import tempfile
    import eval_daily as ED
    from learn import labels as L

    td = Path(tempfile.mkdtemp(prefix="label_"))
    keep = (L.LABEL_DIR, L.from_quotes, ED.now_bj, ED.ROOT)
    L.LABEL_DIR = td

    def never(*a, **k):
        raise AssertionError("盘中不该联网抓价")

    L.from_quotes = never
    try:
        ED.now_bj = lambda: _dt.datetime(2026, 9, 16, 14, 0,
                                         tzinfo=_dt.timezone.utc)
        ck(ED.stage_label(c, "2026-09-16", False) == 1,
           "北京 14:00 对当天抓标签 -> 退 1（现价不是收盘价）")
        ck(ED.stage_label(c, "2026-09-17", False) == 1, "未来日期同样退 1")
        ck(not L.path_for("2026-09-16").exists(), "被拦住的那次一行都没落盘")

        ED.now_bj = lambda: _dt.datetime(2026, 9, 16, 15, 10,
                                         tzinfo=_dt.timezone.utc)
        ck(ED.stage_label(c, "2000-01-03", False) == 0,
           "收盘后、快照不存在的日子仍走原来的「跳过」分支（闸门不误伤）")

        # G4：快照在、但 from_quotes 自己把这一天挡回来（模块里那道 15:05 闸，
        # 云端 learn.yml 盘中 dispatch 走的就是这条）。这时必须 fail-closed：
        # 返回 1、一行都不落盘，`--stage all` 才不会写 learning_status 把
        # 当天挡掉。快照 09:25:45 就落盘了，「没有快照」拦不住盘中派发。
        fake_root = Path(tempfile.mkdtemp(prefix="labroot_"))
        (fake_root / "data" / "2026-09").mkdir(parents=True)
        pd.DataFrame({"code": ["600000", "600001"],
                      "auc_price": [10.0, 20.0]}).to_parquet(
            fake_root / "data" / "2026-09" / "auction_2026-09-16.parquet")
        ED.ROOT = fake_root
        L.from_quotes = lambda codes, date="", **k: pd.DataFrame()
        ck(ED.stage_label(c, "2026-09-16", False) == 1,
           "快照在、from_quotes 返回空（模块闸挡回）-> 退 1，不是 0")
        ck(not L.path_for("2026-09-16").exists(),
           "这一次同样一行都没落盘（错标签进了训练表就再也覆盖不掉）")
    finally:
        L.LABEL_DIR, L.from_quotes, ED.now_bj, ED.ROOT = keep

    # AST：判时刻必须在联网之前
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "stage_label")
    now = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
           and getattr(n.func, "id", "") == "now_bj"]
    fq = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
          and getattr(n.func, "attr", "") == "from_quotes"]
    ck(now and fq and min(now) < min(fq),
       "stage_label 先 now_bj() 再 from_quotes()（先判时刻再联网）")
    saves = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "save"]
    ck(len(saves) == 1 and any(k.arg == "force" for k in saves[0].keywords),
       "labels.save 带 force（口径收紧后要能重打，见 labels.save 注释）")

    # labels.save 的覆盖规则：相等照覆盖，force 时更少也覆盖
    df1 = pd.DataFrame({"code": ["600000", "600001"], "r": [0.01, 0.02],
                        "dirty": [False, False]})
    df2 = pd.DataFrame({"code": ["600000", "600001"], "r": [0.05, 0.06],
                        "dirty": [False, False]})
    df3 = pd.DataFrame({"code": ["600000", "600001"], "r": [0.09, 0.09],
                        "dirty": [True, False]})
    keep2 = L.LABEL_DIR
    L.LABEL_DIR = Path(tempfile.mkdtemp(prefix="labsave_"))
    try:
        L.save("2026-09-16", df1)
        _, wrote = L.save("2026-09-16", df2)
        back = pd.read_parquet(L.path_for("2026-09-16"))
        ck(wrote and abs(float(back["r"].iloc[0]) - 0.05) < 1e-12,
           "可用行相等的新一份照常覆盖（守卫是严格「更少」）")
        _, wrote2 = L.save("2026-09-16", df3)
        ck(not wrote2, "可用行更少 -> 不覆盖，且把「没写」告诉调用方")
        _, wrote3 = L.save("2026-09-16", df3, force=True)
        ck(wrote3, "force=True 时更少也覆盖（口径收紧后重打一遍）")
    finally:
        L.LABEL_DIR = keep2

    check_label_rc(c)


def check_label_rc(c: dict) -> None:
    """「没覆盖」必须能从退出码看出来（W2-4，教训 27：退出码 0 不等于做了事）。

    labels.save 的守卫是「已有可用行更多就不覆盖」。口径一收紧，被拦下的
    恰好是真正改动到的那几天，而以前 stage_label 一律 return 0，批量重打
    一遍看不出哪天没打上。现在：0 写了 / 1 失败 / 2 没覆盖，`--all` 把 2 的
    那些天收集起来在结尾点名。
    """
    print("\n标签退出码：没覆盖 -> 2")
    import datetime as _dt
    import tempfile
    import logging as _logging
    import eval_daily as ED
    from learn import labels as L

    td = Path(tempfile.mkdtemp(prefix="labrc_"))
    fake_root = Path(tempfile.mkdtemp(prefix="labrcroot_"))
    (fake_root / "data" / "2026-09").mkdir(parents=True)
    codes = [f"{600000 + i:06d}" for i in range(100)]
    pd.DataFrame({"code": codes, "auc_price": [10.0] * 100,
                  "one_word": [False] * 100}).to_parquet(
        fake_root / "data" / "2026-09" / "auction_2026-09-16.parquet")

    def quotes(n_clean: int):
        # 前 n_clean 只开盘价 == 撮合价（干净），其余高开 2%（失配 > 0.5%，
        # 按 learning.label.max_open_mismatch_pct 判脏）
        return lambda cs, date="", **k: pd.DataFrame({
            "code": codes,
            "open": [10.0] * n_clean + [10.2] * (100 - n_clean),
            "close": [10.5] * 100})

    keep = (L.LABEL_DIR, L.from_quotes, ED.now_bj, ED.ROOT, ED.dataset,
            ED.stage_label)
    L.LABEL_DIR = td
    ED.ROOT = fake_root
    ED.now_bj = lambda: _dt.datetime(2026, 9, 16, 15, 10,
                                     tzinfo=_dt.timezone.utc)
    try:
        L.from_quotes = quotes(100)
        ck(ED.stage_label(c, "2026-09-16", False) == 0, "第一遍 100 行可用 -> 0")
        L.from_quotes = quotes(50)
        ck(ED.stage_label(c, "2026-09-16", False) == 2,
           "同一天再打 50 行可用 -> 2（守卫不覆盖）")
        back = pd.read_parquet(L.path_for("2026-09-16"))
        ck(int((~back["dirty"]).sum()) == 100, "盘上那份没被覆盖")
        ck(ED.stage_label(c, "2026-09-16", False, True) == 0,
           "--force 时覆盖，退 0")
        ck(int((~pd.read_parquet(L.path_for("2026-09-16"))["dirty"]).sum()) == 50,
           "force 之后盘上是新的那份（50 行可用）")

        # --all：2 不许混进失败里，但必须点名到天
        logs: list[str] = []
        h = _logging.Handler(); h.emit = lambda r: logs.append(r.getMessage())
        lg = _logging.getLogger("eval"); lg.addHandler(h)
        rcs = {"2026-09-14": 0, "2026-09-15": 2, "2026-09-16": 2}

        class _FakeDS:
            snapshot_days = staticmethod(lambda: sorted(rcs))

        ED.dataset = _FakeDS
        ED.stage_label = lambda cc, d, b, f=False: rcs[d]
        argv = sys.argv
        sys.argv = ["eval_daily.py", "--stage", "label", "--all",
                    "--date", "2026-09-16"]
        try:
            rc = ED.main()
        finally:
            sys.argv = argv
            lg.removeHandler(h)
        ck(rc == 0, "--all：只有「没覆盖」时退出码仍是 0（不是失败）")
        ck(any("2026-09-15" in x and "2026-09-16" in x and "force" in x
               for x in logs),
           "--all：没覆盖的那几天在结尾被点名，并提示 --force")
        rcs["2026-09-14"] = 1
        sys.argv = ["eval_daily.py", "--stage", "label", "--all",
                    "--date", "2026-09-16"]
        try:
            ck(ED.main() == 1, "--all：真失败的天照样让退出码非零")
        finally:
            sys.argv = argv
    finally:
        (L.LABEL_DIR, L.from_quotes, ED.now_bj, ED.ROOT, ED.dataset,
         ED.stage_label) = keep


def check_llm_brief_date() -> None:
    """brief 是别的日子的就不归因（F3-11）。"""
    print("\n归因的 brief 日期闸")
    import json as _json
    import tempfile
    from learn import llm_local as LL

    td = Path(tempfile.mkdtemp(prefix="llm_"))
    bp = td / "eval_brief.json"
    bp.write_text(_json.dumps(
        {"date": "2026-09-15", "worst": [{"code": "600000"}], "best": []},
        ensure_ascii=False), encoding="utf-8")
    keep = (LL.OUTDIR, LL.available, LL._run_cli)
    LL.OUTDIR = td / "llm_eval"
    LL.available = lambda: True
    LL._run_cli = lambda *a, **k: (
        {"result": '{"day_regime":"数据异常","items":[]}'}, "")
    try:
        ck(LL.run("2026-09-16", bp) is None
           and not (LL.OUTDIR / "2026-09-16.json").exists(),
           "brief 是 09-15 的、参数日是 09-16 -> 不归因、不落盘")
        got = LL.run("2026-09-15", bp)
        ck(isinstance(got, dict) and got["day_regime"] == "数据异常"
           and (LL.OUTDIR / "2026-09-15.json").exists(),
           "日期对得上时正常归因（闸门没把正常路径也挡掉）")
    finally:
        LL.OUTDIR, LL.available, LL._run_cli = keep


def check_attribution_idempotent(c: dict) -> None:
    """归因跑过就不再跑（F3-12：崩一次会烧 32 次 Opus，还改历史日权重）。"""
    print("\n归因的「跑过就不再跑」")
    import json as _json
    import tempfile
    import eval_daily as ED
    from learn import llm_local as LL

    td = Path(tempfile.mkdtemp(prefix="attr_"))
    (td / "state" / "llm_eval").mkdir(parents=True)
    (td / "out_learn").mkdir()
    bp = td / "out_learn" / "eval_brief.json"
    bp.write_text(_json.dumps({"date": "2026-01-05", "worst": [], "best": []},
                              ensure_ascii=False), encoding="utf-8")
    done = td / "state" / "llm_eval" / "2026-01-05.json"
    done.write_text("{}", encoding="utf-8")
    calls = []
    keep = (ED.STATE, ED.OUT, LL.run)
    ED.STATE, ED.OUT = td / "state", td / "out_learn"
    LL.run = lambda *a, **k: (calls.append(a[0]) or {"day_regime": "正常"})
    try:
        ck(ED._maybe_attribute(c, "2026-01-05") is False and not calls,
           "归因文件已在 -> 不调 CLI（重试 32 次也只烧第一次那一次）")
        done.unlink()
        ck(ED._maybe_attribute(c, "2026-01-05") is True and len(calls) == 1,
           "文件没了 -> 调一次")
        done.write_text("{}", encoding="utf-8")
        ck(ED._maybe_attribute(c, "2026-01-05", force=True) is True
           and len(calls) == 2, "force=True 强制重跑")
        done.unlink()
        bp.write_text(_json.dumps({"date": "2026-01-06"}, ensure_ascii=False),
                      encoding="utf-8")
        ck(ED._maybe_attribute(c, "2026-01-05") is False and len(calls) == 2,
           "brief 是别的日子 -> 照旧跳过（原来那道核对没被弄丢）")
    finally:
        ED.STATE, ED.OUT, LL.run = keep

    # AST：all 分支不许直接调 llm_local.run
    fn = next(x for x in ast.walk(ast.parse(
        (ROOT / "src" / "eval_daily.py").read_text(encoding="utf-8")))
        if isinstance(x, ast.FunctionDef) and x.name == "main")
    br = None
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.comparators[0], ast.Constant)
                and n.test.comparators[0].value == "all"):
            br = n
    direct = [n for n in ast.walk(br) if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "run"
              and getattr(getattr(n.func, "value", None), "id", "")
              == "llm_local"] if br else [None]
    maybe = [n for n in ast.walk(br) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_maybe_attribute"] if br else []
    ck(br is not None and not direct and len(maybe) == 1,
       "--stage all 只经 _maybe_attribute 归因，不直接调 llm_local.run")


def check_proposal_send() -> None:
    """提案邮件：先发信、按结果落盘；失败留可查询对象（F2-5）。"""
    print("\n影子转正提案的发信接线")
    import json as _json
    import tempfile
    import mailer
    from learn import panel as LP, report as R

    got: dict = {}
    orig = mailer._send

    def fake(msg, c):
        got["subject"] = str(msg["Subject"])

    def boom(msg, c):
        raise OSError("smtp down")

    keep_env = {k: os.environ.get(k) for k in
                ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
                 "MAIL_TO", "SKIP_MAIL")}
    os.environ.update(SMTP_HOST="smtp.test", SMTP_PORT="587",
                      SMTP_USER="u@test", SMTP_PASS="x", MAIL_TO="a@test")
    os.environ.pop("SKIP_MAIL", None)
    keep_p = shadow.PROPOSAL
    shadow.PROPOSAL = Path(tempfile.mkdtemp(prefix="prop_")) / "p.json"
    try:
        mailer._send = fake
        ck(R.send("2026-09-04", "<b>ok</b>", {}) is True, "send 成功返回 True")
        os.environ["SKIP_MAIL"] = "1"
        ck(R.send("2026-09-04", "<b>ok</b>", {}) is False,
           "SKIP_MAIL=1 返回 False（生成了但没发，不算已发）")
        os.environ.pop("SKIP_MAIL", None)
        mailer._send = boom
        ck(R.send("2026-09-04", "<b>ok</b>", {}) is False,
           "SMTP 抛异常时返回 False 且不往外抛")

        row = {"date": "2026-09-01", "base_top_excess": 0.0,
               "shadow_top_excess": 0.01, "base_ic": 0.0, "shadow_ic": 0.1,
               "overlap": 0.5}
        stat = shadow.promotion_stat([row] * 31, 30, 0.9, 100)
        ck(stat["ready"], "先造一个证据达标的 stat（否则根本走不到发信）")
        ck(shadow.maybe_propose("2026-09-04", stat, [row], {}, 10) is False,
           "SMTP 失败时 maybe_propose 返回 False")
        doc = _json.loads(shadow.PROPOSAL.read_text(encoding="utf-8"))
        ck("last_sent" not in doc and doc.get("send_failed") == "2026-09-04",
           "失败不写 last_sent，写 send_failed（面板据此显示红字）")
        s4 = LP._stepper({"n_days": 1}, {"shadow_min_days": 30, "p_req": 0.9,
                                         "min_days": 60, "top_k": 10,
                                         "run_at": "16:30"}, doc)
        ck("邮件没发出去" in s4 and "等你决定" not in s4,
           "面板第 4 步显示「邮件没发出去」，不再写「等你决定」")

        mailer._send = fake
        got.clear()
        ck(shadow.maybe_propose("2026-09-05", stat, [row], {}, 10) is True
           and got.get("subject", "").startswith("[提案]"),
           "失败的次日不受 remind_days 节流，立刻重发")
        doc = _json.loads(shadow.PROPOSAL.read_text(encoding="utf-8"))
        ck(doc.get("last_sent") == "2026-09-05" and doc.get("times") == 1
           and "send_failed" not in doc,
           "发成了才写 last_sent / times，send_failed 被清掉")
        got.clear()
        ck(shadow.maybe_propose("2026-09-06", stat, [row], {}, 10) is False
           and not got, "真发过之后 10 天内不再催")
    finally:
        mailer._send = orig
        shadow.PROPOSAL = keep_p
        for k, v in keep_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_split(c: dict) -> None:
    """样本外切分 + 闸门 2 的分块（F1-6）。"""
    print("\n样本外切分与分块")
    from learn import optimize as OPT
    days = [f"2026-{(i // 25) + 1:02d}-{(i % 25) + 1:02d}" for i in range(100)]
    tr, te = OPT.split_days(days, 0.2)
    blocks = OPT.oos_blocks(te, 3)
    ck(len(blocks) == 3 and all(len(b) for b in blocks), "样本外切成 3 个非空块")
    ck([d for b in blocks for d in b] == list(te),
       "三块拼起来恰好等于 te（顺序、元素都一致）")
    ck(all(blocks[i][-1] < blocks[i + 1][0] for i in range(2)),
       "块之间时间连续且不重叠")
    ck(blocks[0][0] > tr[-1], "第一块严格晚于全部训练天")
    ck(len(OPT.split_days([f"d{i:03d}" for i in range(404)], 0.2)[1]) == 81
       and len(OPT.split_days([f"d{i:03d}" for i in range(414)], 0.2)[1]) == 83,
       "真实规模 404 / 414 天 -> 样本外 81 / 83 天")
    ck(OPT.oos_blocks([], 3) == [] and len(OPT.oos_blocks(te[:2], 3)) == 2,
       "空段给空表；块数多于天数时不产生空块")

    # 块只是切分，不改变逐日量：按块聚合 ≡ 用同一批天单独建一个 Problem
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    t1 = dict(t0)
    t1["screen.gap_pct_peak"] = t0["screen.gap_pct_peak"] + 0.3
    df = _multiday(9, 60, 400)
    p = OPT.Problem(df, c, box, t0, t0, None, 10, 1.345)
    gn, go = p.G(t1)[1], p.G(t0)[1]
    bl = OPT.oos_blocks(p.dates, 3)
    got = OPT.block_deltas(p, gn, go, bl)
    want = []
    for b in bl:
        sub = OPT.Problem(df[df["date"].isin(b)], c, box, t0, t0, None,
                          10, 1.345)
        want.append(sub.G(t1)[0] - sub.G(t0)[0])
    ck(len(got) == 3 and max(abs(a - b) for a, b in zip(got, want)) < 1e-12,
       "分块 ΔĜ 和单独建 Problem 逐位相同（不用重新打分）")
    ck(any(abs(x) > 1e-9 for x in got),
       f"这组参数真的动了分数，分块 ΔĜ 不是一排 0（{[round(x, 4) for x in got]}）")


def check_paired(c: dict) -> None:
    """配对 ΔĜ 与闸门 3 自助必须同一组天、同一组权重（F1-8/F2-7/F3-8）；
    有效样本不足时返回 0.5 而不是 0（F1-9）；平局天不参与自助（F1-3）。"""
    print("\n配对 ΔĜ 与按天自助")
    from learn import optimize as OPT

    class _Stub:
        """鸭子类型的 Problem：闸门 3 只用到 G / day_w / huber_c 三样。"""
        huber_c = 1.345

        def __init__(self, bad, w=None, n=12):
            self.bad, self.n = bad, n
            self.day_w = (np.array([1.0] * 6 + [0.0] * 6) if w is None
                          else np.asarray(w, float))

        def G(self, theta):
            g = np.zeros(self.n)
            if theta == "new":
                g[:6] = 0.003
                g[6:] = self.bad
            return float(g.mean()), g

    a = OPT.paired_delta(_Stub(-0.02), "new", "old")
    b = OPT.paired_delta(_Stub(-5.0), "new", "old")
    ck(abs(a - 0.003) < 1e-9 and abs(a - b) < 1e-12,
       f"配对 ΔĜ 不受 w=0 天影响（-0.02 给 {a:+.4f}，-5.0 给 {b:+.4f}）")
    ck(OPT.bootstrap_better(_Stub(-5.0), "new", "old", n=200) == 1.0 and b > 0,
       "配对 ΔĜ 与闸门 3 自助同号（同一组天、同一组权重）")
    # 教训 26：先证明旧写法在这份数据上真的给出反号的数
    _, gn = _Stub(-5.0).G("new")
    ck(O.huber_location(gn - np.zeros(12), None) < 0,
       f"旧写法（None 权重、不剔 w=0 天）算出来是 "
       f"{O.huber_location(gn, None):+.3f}，和 P=1.00 自相矛盾")

    # 有效样本不足 -> 0.5。返回 0 会被闸门 7 读成「在真值上明显更差」而否决
    allzero = _Stub(-5.0, w=np.zeros(12))
    ck(OPT.bootstrap_better(allzero, "new", "old", n=100) == 0.5,
       "在线天全被判「数据异常」（权重全 0）-> P=0.50，不是 0.00")
    ck(OPT.paired_delta(allzero, "new", "old") == 0.0,
       "同一情形点估计给 0.0（没有样本就不冒充有结论）")
    ck(OPT.bootstrap_better(_Stub(0.0, w=np.ones(12)), "old", "old",
                            n=100) == 0.5,
       "两组参数产生同一张榜（差值全 0）-> P=0.50")

    class _Few:
        huber_c = 1.345
        day_w = np.ones(20)

        def G(self, theta):
            g = np.zeros(20)
            if theta == "new":
                g[:4] = 0.01
            return 0.0, g

    ck(OPT.bootstrap_better(_Few(), "new", "old", n=100) == 0.5,
       "20 天里只有 4 天非平局 -> 不裁决（P=0.50），不拿 16 个 0 凑显著")

    class _Ties:
        """12 个平局天（池子 ≤ k）+ 1 个暴走日 + 7 个小负日。"""
        huber_c = 1.345
        day_w = np.ones(20)

        def G(self, theta):
            g = np.zeros(20)
            if theta == "new":
                g[12] = 5.0
                g[13:] = -0.02
            return 0.0, g

    pt = OPT.bootstrap_better(_Ties(), "new", "old", n=400)
    ck(pt < 0.5,
       f"平局天过半时不许让单个 +5 暴走日说了算（实得 P={pt:.2f}）")

    class _Nan:
        """第 3 天空池（G_d 为 NaN）。"""
        huber_c = 1.345
        day_w = np.ones(12)

        def G(self, theta):
            g = np.full(12, 0.004 if theta == "new" else 0.0)
            g[2] = np.nan
            return 0.0, g

    pn = OPT.bootstrap_better(_Nan(), "new", "old", n=200)
    ck(pn == 1.0,
       f"空池天被剔除，NaN 不许把 P 静默压成 0（实得 {pn}）")
    ck(abs(OPT.paired_delta(_Nan(), "new", "old") - 0.004) < 1e-9,
       "点估计同样剔除空池天")


def check_nonfinite(c: dict) -> None:
    """分数/收益里的 NaN 必须被剔除并计数，不许静默让整天退化（F1-2）。"""
    print("\n非有限分数的守卫")
    from learn import optimize as OPT
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    df = _multiday(3, 120, 610)
    s0, rej0 = vscore.score_df(df, c)
    day0 = df["date"].iloc[0]
    cand = [i for i in df.index
            if (not rej0[i]) and df["date"].iloc[i] == day0]
    # 教训 26：必须挑一只**过了准入**的票，否则这条规则根本碰不到
    ck(len(cand) > 10, f"第一天有 {len(cand)} 只过准入的票（够触发这条规则）")
    bad = cand[3]
    # 2026-09-16 起 NaN 斜率在 score.py 和 vscore.py 两边都按 0 处理（和生产
    # 「T1 漏采就拿 T3 补、斜率 0」同口径），所以它不再是 NaN 分数的入口。
    # 先把这件事钉住（孪生体一致性的一部分），再用仍然可达的那条路径
    # ——标签算不出来——去验通用守卫。
    slope_nan = df.copy()
    slope_nan.loc[bad, "slope"] = np.nan
    s_sn, rej_sn = vscore.score_df(slope_nan, c)
    ck(not bool(rej_sn[bad]) and np.isfinite(s_sn[bad]),
       "NaN 斜率不再产生 NaN 分数（两个打分器都按 0 处理）")
    dirty = df.copy()
    dirty.loc[bad, "ytil"] = np.nan         # 标签没算出来：过了准入但收益是 NaN
    s1, rej1 = vscore.score_df(dirty, c)
    ck(not bool(rej1[bad]) and not np.isfinite(float(dirty.loc[bad, "ytil"])),
       "标签缺失的行没被硬性排除，收益确实是 NaN（守卫要抓的就是它）")

    p_ok = OPT.Problem(df.drop(index=[bad]), c, box, t0, t0, None, 10, 1.345)
    p_bad = OPT.Problem(dirty, c, box, t0, t0, None, 10, 1.345)
    g_ok, g_bad = p_ok.G(t0)[1], p_bad.G(t0)[1]
    ck(p_bad.n_nonfinite == 1 and p_ok.n_nonfinite == 0,
       f"非有限行被计数（脏表 {p_bad.n_nonfinite} 行、干净表 {p_ok.n_nonfinite} 行）")
    ck(abs(g_bad[0] - g_ok[0]) < 1e-12,
       "那一行被剔除，当天 G_d 等于「这只票不存在」")
    mean_ytil = float(dirty[dirty["date"] == day0]["ytil"].mean())
    ck(abs(g_bad[0] - mean_ytil) > 1e-6,
       f"当天没有退化成与 θ 无关的等权 mean(ỹ)={mean_ytil:+.4f}（旧实现就是它）")
    # 与 θ 无关是旧实现的症状：换一组参数，当天的 G_d 必须跟着变
    t1 = dict(t0)
    t1["screen.gap_pct_peak"] = t0["screen.gap_pct_peak"] + 0.3
    ck(abs(p_bad.G(t1)[1][0] - g_bad[0]) > 1e-9,
       "换一组 θ 之后当天 G_d 跟着变（那一天没有从目标函数里消失）")
    m_bad, m_ok = p_bad.metrics(t0, 10), p_ok.metrics(t0, 10)
    ck(np.isfinite(m_bad["ic_mean"])
       and abs(m_bad["avg_pool"] - m_ok["avg_pool"]) < 1e-12,
       f"metrics 的过准入只数按同一个掩码算（脏表 {m_bad['avg_pool']:.4f}"
       f" = 干净表 {m_ok['avg_pool']:.4f}）")
    naive = float(np.mean([int((~rej1[(dirty["date"] == d).to_numpy()]).sum())
                           for d in sorted(dirty["date"].unique())]))
    ck(abs(naive - m_bad["avg_pool"] - 1 / 3) < 1e-12,
       f"只按 ~rej 数的话日均多算 1/3 只（{naive:.4f} vs {m_bad['avg_pool']:.4f}）")


def check_apply_held_refresh(c: dict) -> None:
    """apply-held / rollback 之后状态文件、面板、变更邮件都要跟上（F3-14）。"""
    print("\n人工落地与回滚")
    import json as _json
    import tempfile
    import eval_daily as ED
    import mailer
    from learn import apply as A, panel as LP, report as R

    prod = (ROOT / "state" / "learned.yaml", ROOT / "state" / "held_change.json",
            ROOT / "out_learn" / "change.html")
    before = {p: p.exists() for p in prod}

    td = Path(tempfile.mkdtemp(prefix="held2_"))
    keep = (ED.STATE, ED.OUT, R.STATUS, gate.HISTORY, gate.VERDICTS,
            LP.STATE, LP.OUTL, A.LEARNED, C.LEARNED, mailer._send)
    keep_env = {k: os.environ.get(k) for k in
                ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
                 "MAIL_TO", "SKIP_MAIL")}
    subjects: list[str] = []
    ED.STATE, ED.OUT = td, td / "out_learn"
    R.STATUS = td / "learning_status.json"
    gate.HISTORY = td / "theta_history.jsonl"
    gate.VERDICTS = td / "verdict_log.jsonl"
    LP.STATE, LP.OUTL = td, td / "out"
    A.LEARNED = C.LEARNED = td / "learned.yaml"
    os.environ.update(SMTP_HOST="smtp.test", SMTP_PORT="587",
                      SMTP_USER="u@test", SMTP_PASS="x", MAIL_TO="a@test")
    os.environ.pop("SKIP_MAIL", None)
    mailer._send = lambda msg, conf: subjects.append(str(msg["Subject"]))
    try:
        R.save_status({"date": "2026-09-16", "theta_version": "基线",
                       "accepted_total": 0, "n_days": 414,
                       "metrics": {"ic_mean": 0.01, "top_excess": 0.006},
                       "verdict": {"accepted": False, "checks": []}})
        box = c["learning"]["box"]
        t0 = C.theta0(box)
        gapk = "scoring.weights.gap"
        theta = dict(t0); theta[gapk] = t0[gapk] + 0.02
        (td / "held_change.json").write_text(_json.dumps({
            "date": "2026-09-16", "theta": theta,
            "verdict": {
                "accepted": True,
                "checks": [{"name": f"闸{i}", "passed": True, "detail": "ok"}
                           for i in range(7)],
                "moved": {gapk: [t0[gapk], theta[gapk]]},
                "evidence": {"n_days": 414, "oos_old": 0.11, "oos_new": 0.12,
                             "bootstrap_p": 0.93, "worst_churn": 0.2,
                             "paired_delta": 0.004}},
            "metrics": {"ic_mean": 0.012, "top_excess": 0.007, "hit_rate": 0.5,
                        "icir": 0.4, "avg_pool": 37, "avg_sent": 25},
            "metrics_prev": {"ic_mean": 0.010, "top_excess": 0.006,
                             "hit_rate": 0.48, "icir": 0.35,
                             "avg_pool": 37, "avg_sent": 25},
            "old_top": {"2026-09-16": ["600000"]},
            "new_top": {"2026-09-16": ["600001"]},
            "regime_counts": {"正常": 3}}, ensure_ascii=False),
            encoding="utf-8")

        ck(ED._apply_held(c) == 0, "apply-held 正常返回")
        st = _json.loads(R.STATUS.read_text(encoding="utf-8"))
        ck(st["theta_version"] == "learned" and st["accepted_total"] == 1
           and st["verdict"]["accepted"] is True,
           f"状态刷新：版本 {st['theta_version']} · 累计接受 {st['accepted_total']} 次")
        ck((ED.OUT / "change.html").exists() and len(subjects) == 1
           and subjects[0].startswith("[参数更新]"),
           f"补发了一封完整的变更邮件并留下 change.html（{subjects}）")
        ck("600001" in (ED.OUT / "change.html").read_text(encoding="utf-8"),
           "邮件里有「会新进榜」那张表（held 文件存了两张清单才渲染得出来）")
        ck(not (td / "held_change.json").exists(), "held 文件已删")
        ck("参数已变更" in (LP.OUTL / "learn.html").read_text(encoding="utf-8"),
           "学习面板重建，第 ⑤ 格写「参数已变更」")

        ck(ED._rollback() == 0, "rollback 正常返回")
        st2 = _json.loads(R.STATUS.read_text(encoding="utf-8"))
        ck(st2["theta_version"] == "基线" and not A.LEARNED.exists(),
           "回滚后状态里的参数版本回到基线")
        ck("参数已变更" not in (LP.OUTL / "learn.html").read_text(encoding="utf-8"),
           "面板不再写「参数已变更」（裁决还是 accepted，但版本已是基线）")
        ck("已变更" not in R.status_line() and "未变更" in R.status_line(),
           f"竞价面板底部那行也跟着改（{R.status_line()[-24:]}）")
        ck(ED._rollback() == 0, "本来就是基线时再跑一次不抛")
    finally:
        (ED.STATE, ED.OUT, R.STATUS, gate.HISTORY, gate.VERDICTS,
         LP.STATE, LP.OUTL, A.LEARNED, C.LEARNED, mailer._send) = keep
        for k, v in keep_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    ck(all(p.exists() == before[p] for p in prod),
       "自测没有碰生产目录里的 learned.yaml / held_change.json / change.html（教训 17）")


def main() -> int:
    t0 = time.time()
    c = C.load()
    check_equivalence(c)
    check_objective()
    check_project(c)
    check_neutralize()
    check_gate(c)
    check_split(c)
    check_paired(c)
    check_nonfinite(c)
    check_sparsify(c)
    check_cfg(c)
    check_wiring()
    check_race_source(c)
    check_learn_box(c)
    check_proxy_cols()
    check_shadow_stat()
    check_report_send()
    check_bf_dirty(c)
    check_list_convention(c)
    check_shadow_oos(c)
    check_brief_pick()
    check_online_oos(c)
    check_online_daily(c)
    check_report_metrics(c)
    check_status_errors(c)
    check_held_wiring(c)
    check_apply_held_refresh(c)
    check_default_date(c)
    check_label_guard(c)
    check_llm_brief_date()
    check_attribution_idempotent(c)
    check_proposal_send()
    check_council()
    print(f"\n耗时 {time.time()-t0:.2f}s | 断言失败 {BAD} 个")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
