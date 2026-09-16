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
    check_wiring()
    check_shadow_stat()
    check_report_send()
    check_council()
    print(f"\n耗时 {time.time()-t0:.2f}s | 断言失败 {BAD} 个")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
