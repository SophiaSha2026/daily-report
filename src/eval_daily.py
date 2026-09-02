"""
自评估 · 学习 · 迭代  的命令行入口。

    python src/eval_daily.py --stage intraday  盘中采一个时点（卖点研究用）
    python src/eval_daily.py --stage label     收盘后抓标签
    python src/eval_daily.py --stage brief     给 LLM 归因准备输入
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

import numpy as np
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
    d = df.assign(_s=s, _rej=rej)
    ok = d[~d["_rej"]].sort_values("_s", ascending=False)
    if ok.empty:
        log.warning("%s 无票通过硬性排除", date)
        return 0

    n_w, n_b = lc["llm"]["n_worst"], lc["llm"]["n_best"]
    head = ok.head(20)
    tail = ok.tail(max(len(ok) // 2, 1))

    def pack(g):
        return [{
            "code": r.code, "name": r["name"], "score": round(r._s, 1),
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
def stage_learn(c: dict, date: str, dry: bool) -> int:
    lc = c["learning"]
    box, g = lc["box"], lc["gate"]
    df = dataset.build(None, lc["neutralize"])
    if df.empty:
        log.warning("还没有任何带标签的数据")
        return 0
    days = sorted(df["date"].unique())
    n = len(days)
    theta0, theta_prev = C.theta0(box), C.theta_now(box)
    dayw = load_day_weights(c)

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
    theta_new = OPT.fit(p_tr, lam_a, lc["objective"]["lambda_l1"])

    oos_new = p_te.G(theta_new)[0]
    oos_old = p_te.G(theta_prev)[0]
    bp = OPT.bootstrap_better(p_te, theta_new, theta_prev, g["bootstrap_n"])

    look = days[-g["churn_lookback"]:]
    p_look = mk(look)
    codes = df[df["date"].isin(look)].sort_values(
        "date", kind="mergesort")["code"].to_numpy()
    churn = gate.churn_by_day(p_look.top_codes(theta_prev, codes),
                              p_look.top_codes(theta_new, codes))

    v = gate.evaluate(theta_new, theta_prev, box, g, n, days, date,
                      g["bootstrap_p"], oos_new, oos_old, churn)
    status["verdict"] = v.to_dict()
    status["lambda_anchor"] = lam_a
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
    A.write(theta_new, v.evidence, date)
    gate.record(date, theta_new, v, m_new)
    html = R.build_html(date, v, m_prev, m_new,
                        p_look.top_codes(theta_prev, codes),
                        p_look.top_codes(theta_new, codes),
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
                             "status", "backfill", "intraday", "exits"])
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

    if a.stage == "rollback":
        print("已回滚" if A.rollback() else "本来就是人工基线")
        return 0
    if a.stage == "status":
        print(R.status_line() or "还没有学习状态")
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
    if a.stage == "race":
        return stage_race(c)
    return stage_learn(c, date, a.dry)


if __name__ == "__main__":
    sys.exit(main())
