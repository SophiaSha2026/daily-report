"""
自评估 · 学习 · 迭代  的命令行入口。

    python src/eval_daily.py --stage intraday  盘中采一个时点（卖点研究用）
    python src/eval_daily.py --stage label     收盘后抓标签
    python src/eval_daily.py --stage brief     给 LLM 归因准备输入
    python src/eval_daily.py --stage llm       本地跑 LLM 归因（走本机 claude CLI）
    python src/eval_daily.py --stage learn     拟合 + 六道闸 + 落参数 + 发信
    python src/eval_daily.py --stage race      模型擂台（选最合适的传统 ML）
    python src/eval_daily.py --stage rollback  删掉学到的参数，回人工基线
    python src/eval_daily.py --stage status    打印当前状态

完整设计见 docs/learning.md。三条不可越过的边界：
  1. score.py 里永远不出现模型调用
  2. 这条线崩了不能影响早盘发信（独立 workflow + 原子写）
  3. 准入区间不自动改，只提案
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import cfg as C
from learn import (apply as A, dataset, gate, labels as L, objective as O,
                   optimize as OPT, report as R)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_learn"
STATE = ROOT / "state"


def today_bj() -> str:
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(hours=8)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------
#  stage: label
# ---------------------------------------------------------------------
def stage_label(c: dict, date: str, backfill: bool) -> int:
    """收盘后抓收盘价，和当天的竞价快照拼出标签。"""
    snap_p = ROOT / "data" / date[:7] / f"auction_{date}.parquet"
    if not snap_p.exists():
        log.warning("%s 没有竞价快照，跳过", date)
        return 0
    snap = pd.read_parquet(snap_p)

    if backfill:
        hp = ROOT / "cache" / "hist_daily.parquet"
        if not hp.exists():
            log.error("没有 cache/hist_daily.parquet，先跑 --stage backfill")
            return 1
        raw = L.from_hist(date, pd.read_parquet(hp))
    else:
        raw = L.from_quotes(list(snap["code"]))

    if raw.empty:
        log.error("%s 取不到开收盘", date)
        return 1
    L.save(date, L.build(date, snap, raw,
                         c["learning"]["label"]["max_open_mismatch_pct"]))
    return 0


# ---------------------------------------------------------------------
#  stage: brief（给 LLM 的输入）
# ---------------------------------------------------------------------
def stage_brief(c: dict, date: str) -> int:
    lc = c["learning"]
    df = dataset.build([date], lc["neutralize"])
    if df.empty:
        log.warning("%s 没有可用数据", date)
        return 0
    from learn import vscore
    s, rej = vscore.score_df(df, c)
    d = df.assign(sc=s, rej=rej)
    ok = d[~d["rej"]].sort_values("sc", ascending=False)
    if ok.empty:
        log.warning("%s 无票通过硬性排除", date)
        return 0

    n_w, n_b = lc["llm"]["n_worst"], lc["llm"]["n_best"]
    head = ok.head(20)
    tail = ok.tail(max(len(ok) // 2, 1))

    def pack(g):
        return [{
            "code": r.code, "name": r.name, "score": round(r.sc, 1),
            "rank": int(i + 1), "gap_pct": round(r.gap_pct, 2),
            "intraday_pct": round(r.r * 100, 2), "ytil": round(r.ytil, 2),
            "sector": r.sector, "risk_tags": list(r.risk_tags),
        } for i, r in enumerate(g.itertuples(index=False))]

    brief = {
        "date": date,
        "day_stats": {
            "pool": int(len(ok)),
            "market_median_intraday_pct": round(d["day_center"].iloc[0] * 100, 2),
            "top10_excess_pct": round(ok.head(10)["y"].mean() * 100, 2),
            "top10_hit": round(float((ok.head(10)["y"] > 0).mean()), 2),
        },
        "worst": pack(head.nsmallest(n_w, "ytil")),
        "best": pack(tail.nlargest(n_b, "ytil")),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "eval_brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("brief 已写出：worst %d / best %d", len(brief["worst"]),
             len(brief["best"]))
    return 0


def load_day_weights(c: dict) -> dict[str, float]:
    """LLM 归因 -> 优化器里的日权重。读不到就全 1.0。

    映射表在 config 里，所以给定那些 JSON，优化器完全确定——
    LLM 的影响是可复现的。
    """
    m = c["learning"]["llm"]["regime_weight"]
    out: dict[str, float] = {}
    for p in sorted((STATE / "llm_eval").glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            out[p.stem] = float(m.get(j.get("day_regime", "正常"), 1.0))
        except Exception as e:  # noqa: BLE001
            log.warning("归因 %s 读取失败: %s", p.name, e)
    return out


# ---------------------------------------------------------------------
#  stage: learn
# ---------------------------------------------------------------------
def _load_train(c: dict):
    """训练表：优先回填（404 天，特征齐但竞价轨迹是代理值），
    没有才用在线积累。返回 (df, source)。

    2026-09-03 用户决定不等 60 天在线积累，直接用回填历史点火学习；
    在线真值快照转为第七道闸（稳健性否决），见 gate.evaluate。
    """
    lc = c["learning"]
    bf = sorted((ROOT / "data" / "train").glob("backfill_*.parquet"))
    if bf:
        raw = pd.read_parquet(bf[-1])
        raw["dirty"] = raw["one_word"].astype(bool)
        # 抢救日守卫只认在线快照。回填表的 t1/t2/t3 是代理值，单价竞价
        # 天然三者相等，开着守卫会误伤（2026-09-04 复查丢了 3 天）。
        df = dataset.neutralize(raw, lc["neutralize"], salvage_guard=False)
        return df, f"backfill:{bf[-1].name}"
    return dataset.build(None, lc["neutralize"]), "online"


def _regime_of(date: str) -> str | None:
    """当天的 LLM 归因结论（day_regime），没有就 None。"""
    p = STATE / "llm_eval" / f"{date}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("day_regime")
    except Exception:  # noqa: BLE001
        return None


def stage_learn(c: dict, date: str, dry: bool) -> int:
    lc = c["learning"]
    box, g = lc["box"], lc["gate"]
    df, source = _load_train(c)
    if df.empty:
        log.warning("还没有任何带标签的数据")
        return 0
    days = sorted(df["date"].unique())
    n = len(days)
    theta0, theta_prev = C.theta0(box), C.theta_now(box)
    dayw = load_day_weights(c)
    log.info("训练源 %s：%d 天", source, n)

    def mk(sub_days):
        sub = df[df["date"].isin(sub_days)]
        return OPT.Problem(sub, c, box, theta0, theta_prev, dayw,
                           lc["objective"]["top_k"], lc["objective"]["huber_c"],
                           lc["objective"]["tau_perplexity_tol"])

    full = mk(days)
    m_prev = full.metrics(theta_prev, lc["objective"]["top_k"])
    log.info("当前参数：%d 天，IC %.4f，ICIR %.3f，前10超额 %+.3f%%",
             n, m_prev["ic_mean"], m_prev["icir"], m_prev["top_excess"] * 100)

    status = {"date": date, "n_days": n, "days": days,
              "theta_version": "learned" if C.diff() else "基线",
              "metrics": m_prev, "source_days_weighted": dayw}

    if n < g["min_days"]:
        # 数据不够就**根本不拟合**。不是「拟合了但不接受」——
        # 60 天以下拟合出来的数只会误导，连日志里都不该出现。
        status["verdict"] = {
            "accepted": False,
            "checks": [{"name": "最少天数", "passed": False,
                        "detail": f"{n} 天 / 要求 >= {g['min_days']}，不拟合"}],
            "moved": {}, "evidence": {}}
        R.save_status(status)
        log.info("样本 %d 天 < %d，本阶段只做评估，不动参数", n, g["min_days"])
        return 0

    tr, te = OPT.split_days(days, g["oos_frac"])
    p_tr, p_te = mk(tr), mk(te)
    lam_a = O.lambda_anchor(lc["objective"]["lambda_anchor"], n,
                            lc["objective"]["anchor_prior_days"])
    theta_fit = OPT.fit(p_tr, lam_a, lc["objective"]["lambda_l1"])
    theta_new, intents = OPT.sparsify(theta_fit, theta_prev, box,
                                      g["max_moves"], g["max_step_frac"])
    log.info("稀疏化：连续解触及 %d 个参数 -> 意图 %s",
             sum(1 for k in box
                 if abs(theta_fit[k] - theta_prev[k]) > 1e-9), intents)

    oos_new, gd_new = p_te.G(theta_new)
    oos_old, gd_old = p_te.G(theta_prev)
    bp = OPT.bootstrap_better(p_te, theta_new, theta_prev, g["bootstrap_n"])
    # 闸门 3 自助的是逐日配对差的 Huber 位置；把它的点估计也报出来，
    # 免得「P 高但 ΔĜ 为负」看起来像矛盾（两者量的不是同一件事）。
    paired = float(O.huber_location(gd_new - gd_old, None,
                                    lc["objective"]["huber_c"]))

    look = days[-g["churn_lookback"]:]
    p_look = mk(look)
    codes = df[df["date"].isin(look)].sort_values(
        "date", kind="mergesort")["code"].to_numpy()
    churn = gate.churn_by_day(p_look.top_codes(theta_prev, codes),
                              p_look.top_codes(theta_new, codes))

    # 第七道闸：在线真值快照上的稳健性。训练是回填（轨迹为代理值），
    # 这里用真采样的那几天做否决检验。抢救日已被 dataset 守卫剔除。
    online_p, online_days = None, 0
    if source.startswith("backfill"):
        dfo = dataset.build(None, lc["neutralize"])
        if not dfo.empty:
            p_on = OPT.Problem(dfo, c, box, theta0, theta_prev, dayw,
                               lc["objective"]["top_k"],
                               lc["objective"]["huber_c"],
                               lc["objective"]["tau_perplexity_tol"])
            online_days = dfo["date"].nunique()
            online_p = OPT.bootstrap_better(p_on, theta_new, theta_prev,
                                            g["bootstrap_n"])
            log.info("在线稳健性：%d 天真值快照，P(新参数更好)=%.2f",
                     online_days, online_p)

    # 统计量一律关键字传入（闸门 3 曾因位置错位拿到阈值本身，见 gate.evaluate）
    v = gate.evaluate(theta_new, theta_prev, box, g, n, days, date,
                      boot_p=bp, oos_new=oos_new, oos_old=oos_old,
                      churn=churn, online_p=online_p, online_days=online_days,
                      intents=intents, paired_delta=paired)
    status["verdict"] = v.to_dict()
    status["train_source"] = source
    status["intents"] = list(intents)
    status["lambda_anchor"] = lam_a
    status["run_at"] = lc.get("run_at", "16:30:00")
    status["accepted_total"] = gate.accepted_count()
    status["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    status["regime_today"] = _regime_of(date)
    if not dry:
        # 每次裁决都记（接受与否），面板上的「第 N 次裁决」和时间线靠它
        gate.log_verdict(date, v, {"source": source, "n_days": n})

    # 在线真值天的逐日指标 + 影子双榜对比 + 转正证据，喂给学习面板。
    # 全段 fail-open：面板是给人看的，不许拖垮裁决本身。
    try:
        from learn.model_select import spearman
        from learn import vscore, panel as LP
        dfo2 = dfo if source.startswith("backfill") else df
        daily = []
        if not dfo2.empty:
            s_, rej = vscore.score_df(dfo2, c)
            dd = dfo2.assign(sc=s_, rej=rej)
            for day, gday in dd[~dd["rej"]].groupby("date"):
                top = gday.nlargest(lc["objective"]["top_k"], "sc")
                daily.append({
                    "date": day,
                    "ic": spearman(gday["sc"].to_numpy(),
                                   gday["ytil"].to_numpy()),
                    "top_excess": float(top["y"].mean()),
                })
        status["daily"] = daily
        status["online_days"] = len(daily)
        # 影子排序器：refit + 在线双榜对比 + 转正证据。研究性组件，失败不影响任何东西。
        try:
            from learn import shadow
            shadow.fit(df)
            if not dfo2.empty:
                s2, rej2 = vscore.score_df(dfo2, c)
                cmp_ = shadow.daily_compare(dfo2, s2, rej2,
                                            lc["objective"]["top_k"])
                status["shadow"] = cmp_
                scfg = lc.get("shadow") or {}
                stat = shadow.promotion_stat(
                    cmp_, int(scfg.get("min_days", 30)),
                    float(scfg.get("p_better", 0.90)), int(g["bootstrap_n"]))
                status["shadow_stat"] = stat
                if cmp_:
                    import numpy as _np
                    b = _np.nanmean([x["base_top_excess"] for x in cmp_])
                    sh = _np.nanmean([x["shadow_top_excess"] for x in cmp_])
                    log.info("影子对比（%d 个真值日）：前10超额 基线 %+.3f%% "
                             "vs 影子 %+.3f%%，日均重合 %.0f%%",
                             len(cmp_), b * 100, sh * 100,
                             _np.mean([x["overlap"] for x in cmp_]) * 100)
                    log.info("转正证据：%d/%d 天，P(影子更好)=%.2f，%s",
                             stat["days"], stat["min_days"],
                             stat["p_better"] or 0.0,
                             "达标" if stat["ready"] else "未达标")
                if not dry and shadow.maybe_propose(
                        date, stat, cmp_, c, int(scfg.get("remind_days", 10))):
                    log.info("影子转正提案已发出（切换与否由用户决定）")
        except Exception as e:  # noqa: BLE001
            log.warning("影子对比失败（不影响流程）: %s", e)
        R.save_status(status)
        LP.build()
    except Exception as e:  # noqa: BLE001
        log.warning("学习面板生成失败（不影响流程）: %s", e)
    R.save_status(status)

    for ck in v.checks:
        log.info("闸门 %-14s %s  %s", ck.name, "过" if ck.passed else "不过",
                 ck.detail)
    if not v.accepted:
        log.info("不接受本次变更，参数保持不变")
        return 0
    if dry:
        log.info("dry-run：本可接受，但不写盘")
        return 0

    m_new = full.metrics(theta_new, lc["objective"]["top_k"])

    # 通道 4：Opus 审稿。统计闸门管数字，它管「数字和叙事对不对得上」。
    from learn import llm_review
    review = llm_review.run(date, {
        "moved": {k: list(vv) for k, vv in v.moved.items()},
        "evidence": v.evidence,
        "gates": [c_.__dict__ if hasattr(c_, "__dict__") else c_
                  for c_ in v.checks],
        "recent_regimes": _regime_counts(),
        "shadow_compare": status.get("shadow", [])[-10:],
    }, lc["llm"]["model"])
    status["review"] = review
    R.save_status(status)
    if (lc["llm"].get("review_mode", "advisory") == "veto"
            and review.get("stance") == "反对"):
        # 搁置：不写参数，把提案和证据留在 held 文件里，人工确认后落地。
        held = STATE / "held_change.json"
        held.write_text(json.dumps({
            "date": date, "theta": theta_new, "verdict": v.to_dict(),
            "review": review}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        log.warning("变更被 Opus 审稿搁置：%s。人工确认：--stage apply-held",
                    "；".join(review.get("points", [])))
        try:
            import mailer
            mailer.send_alert(
                "[学习] " + date + " 参数变更过了七道闸但被 Opus 审稿搁置。"
                + " 理由：" + "；".join(review.get("points", []))
                + " 认可就跑：python src/eval_daily.py --stage apply-held")
        except Exception as e:  # noqa: BLE001
            log.warning("搁置通知发送失败: %s", e)
        return 0

    A.write(theta_new, v.evidence, date)
    gate.record(date, theta_new, v, m_new)
    html = R.build_html(date, v, m_prev, m_new,
                        p_look.top_codes(theta_prev, codes),
                        p_look.top_codes(theta_new, codes),
                        llm_note=llm_review.as_note(review),
                        regime_counts=_regime_counts())
    OUT.mkdir(exist_ok=True)
    (OUT / "change.html").write_text(html, encoding="utf-8")
    R.send(date, html, c)
    log.info("参数已更新并发信")
    return 0


def _regime_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for p in (STATE / "llm_eval").glob("*.json"):
        try:
            k = json.loads(p.read_text(encoding="utf-8")).get("day_regime", "正常")
            out[k] = out.get(k, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------
#  stage: race（模型擂台）
# ---------------------------------------------------------------------
def stage_race(c: dict) -> int:
    from learn import model_select as MS
    lc = c["learning"]
    # 优先用回填训练表（几百个交易日），没有才退回在线积累（起步只有几天）。
    # 回填表没有 dirty 列：一字板买不进标脏，其余可执行。
    bf = sorted((ROOT / "data" / "train").glob("backfill_*.parquet"))
    if bf:
        raw = pd.read_parquet(bf[-1])
        raw["dirty"] = raw["one_word"].astype(bool)
        df = dataset.neutralize(raw, lc["neutralize"], salvage_guard=False)
        log.info("擂台用回填表 %s：%d 行 %d 天", bf[-1].name, len(df),
                 df["date"].nunique())
    else:
        df = dataset.build(None, lc["neutralize"])
    if df.empty:
        log.error("没有带标签的数据")
        return 1
    res = MS.walk_forward(df, c, n_folds=5,
                          min_train_days=max(20, len(df["date"].unique()) // 3),
                          top_k=lc["objective"]["top_k"])
    if res.empty:
        log.warning("天数不足，擂台跑不起来")
        return 0
    summ = MS.summarize(res, lc["gate"]["bootstrap_n"])
    win, why = MS.pick(summ)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n选中：{win}\n理由：{why}")
    OUT.mkdir(exist_ok=True)
    summ.to_csv(OUT / "model_race.csv", index=False)
    (OUT / "model_race.json").write_text(json.dumps(
        {"winner": win, "why": why,
         "table": summ.to_dict("records")}, ensure_ascii=False, indent=2,
        default=float), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["label", "brief", "learn", "race", "rollback",
                             "status", "backfill", "intraday", "exits",
                             "llm", "all", "apply-held"])
    ap.add_argument("--date", default=None)
    ap.add_argument("--backfill", action="store_true",
                    help="label 阶段从 cache/hist_daily.parquet 取，而不是联网")
    ap.add_argument("--all", action="store_true",
                    help="label 阶段：把所有已有快照日都补一遍")
    ap.add_argument("--point", default=None,
                    help="intraday 阶段的采样点 HH:MM:SS，留空=按当前时刻就近")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    c = C.load()
    date = a.date or today_bj()

    if a.stage == "apply-held":
        held = STATE / "held_change.json"
        if not held.exists():
            print("没有被搁置的变更")
            return 0
        j = json.loads(held.read_text(encoding="utf-8"))
        A.write(j["theta"], j["verdict"]["evidence"], j["date"])
        held.unlink()
        print(f"已落地 {j['date']} 被搁置的变更：{list(j['verdict']['moved'])}")
        return 0
    if a.stage == "rollback":
        print("已回滚" if A.rollback() else "本来就是人工基线")
        return 0
    if a.stage == "status":
        lines = R.status_lines()
        print("\n".join(lines) if lines else "还没有学习状态")
        for path, old, new in C.diff():
            print(f"  {path}: {old} -> {new}")
        return 0
    if a.stage == "backfill":
        import pandas as pd
        from learn import sources
        codes = pd.read_csv(ROOT / "cache" / "codes.csv",
                            dtype=str)["code"].tolist()
        b = c["learning"]["backfill"]
        sources.fetch_daily(codes, b["start"], date, b["workers"])
        sources.fetch_auction_ts(b["start"], date)
        log.info("回填源完整度：%s，可学维度 %s",
                 sources.completeness(), sources.learnable_dims())
        return 0
    if a.stage == "intraday":
        from learn import intraday as ID
        snap = ROOT / "data" / date[:7] / f"auction_{date}.parquet"
        if not snap.exists():
            log.warning("%s 没有竞价快照，无从采样", date)
            return 0
        codes = list(pd.read_parquet(snap)["code"])
        ID.collect_one(codes, date, a.point or ID.nearest_point())
        return 0
    if a.stage == "exits":
        from learn import intraday as ID, vscore
        wide = ID.load_all()
        if wide.empty:
            print("还没有盘中采样数据。这一项要从现在开始攒，历史补不了。")
            return 0
        frames = []
        for d in sorted(wide["date"].unique()):
            sp = ROOT / "data" / d[:7] / f"auction_{d}.parquet"
            if sp.exists():
                sn = pd.read_parquet(sp)
                sn["date"] = d
                frames.append(sn.merge(wide[wide.date == d].drop(columns="date"),
                                       on="code", how="inner"))
        if not frames:
            print("采样数据对不上任何竞价快照")
            return 0
        df = pd.concat(frames, ignore_index=True)
        s_, rej = vscore.score_df(df, c)
        df = df[~rej]
        cur = ID.exit_curve(df, s_[~rej], c["learning"]["objective"]["top_k"])
        print(cur.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("")
        print("看 ic 那列：如果 t0935 明显高于 t1500，说明信号衰减快，")
        print("「开盘买收盘卖」这个标签在系统性低估打分器。")
        return 0
    if a.stage == "label":
        if a.all:
            rc = 0
            for d in dataset.snapshot_days():
                rc |= stage_label(c, d, a.backfill)
            return rc
        return stage_label(c, date, a.backfill)
    if a.stage == "brief":
        return stage_brief(c, date)
    if a.stage == "llm":
        from learn import llm_local
        lc = c["learning"]["llm"]
        llm_local.run(date, OUT / "eval_brief.json", lc["model"],
                      lc["timeout_seconds"])
        return 0
    if a.stage == "all":
        # 本地全流程：抓标签 -> 备归因输入 -> 归因 -> 拟合与闸门。
        # 每一步失败都不阻断后面（归因尤其：它是研究性的，不是关键路径）。
        from learn import llm_local
        rc = stage_label(c, date, a.backfill)
        stage_brief(c, date)
        lc = c["learning"]["llm"]
        bp = OUT / "eval_brief.json"
        fresh = False
        try:
            fresh = json.loads(bp.read_text(encoding="utf-8"))["date"] == date
        except Exception:  # noqa: BLE001
            pass
        if lc.get("enabled", True) and fresh:
            llm_local.run(date, bp, lc["model"], lc["timeout_seconds"])
        elif not fresh:
            # brief 是别的日子留下的。拿它归因会把昨天的票安到今天头上，
            # 归因文件按日期落盘，错一天就污染那一天的日权重。
            log.info("eval_brief.json 不是 %s 的，跳过归因", date)
        return rc | stage_learn(c, date, a.dry)
    if a.stage == "race":
        return stage_race(c)
    return stage_learn(c, date, a.dry)


if __name__ == "__main__":
    sys.exit(main())
