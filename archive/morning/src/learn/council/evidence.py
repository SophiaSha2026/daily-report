"""
证据包：两条线「预测 vs 真值」的全部数字，纯代码算，LLM 拿到的是同一份。

早盘（早盘选股 + 参数自学）
    在线真值日逐日：IC、前 10 超额、命中只数、池大小、当日中位、归因 regime
    分数分档 / 六个维度高低三分之一 / A-B 组 / 涨幅段 / 量比段 的超额
    被硬剔除 vs 过准入
    逐日 worst / best（从标签 + 快照重算）
    回填 vs 在线的口径差（同名列的均值/中位数、拒绝率）
    闸门裁决、参数历史、影子对比

起涨预测
    每份清单的真值（breakout/truth.py）：命中率 vs 邮件期望 vs 同期基准
    按连续档 / 分数段 / 板块
    模型：训练日期、重要性、今天清单的特征分位
    数据健康：每天参与横截面的代码数、NaN 比例

写到 state/council/<date>/evidence.json。数字都带样本数，没有样本数的
数字 LLM 会当成确定的事实，那是误导。
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

log = logging.getLogger(__name__)
STATE = ROOT / "state"
OUT_DIR = STATE / "council"


def _f(x, nd: int = 4):
    """浮点四舍五入，NaN/None -> None（JSON 里不许出现 NaN）。"""
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------
#  早盘
# ---------------------------------------------------------------------
def morning(c: dict, n_days: int = 0) -> dict:
    """在线真值日的预测 vs 真值。n_days=0 取全部。"""
    from learn import dataset, vscore
    from learn.model_select import spearman
    from learn.optimize import production_order
    lc = c["learning"]
    df = dataset.build(None, lc["neutralize"])
    out: dict = {"days": 0, "note": ""}
    if df.empty:
        out["note"] = "还没有在线真值日"
        return out
    days = sorted(df["date"].unique())
    if n_days:
        days = days[-n_days:]
        df = df[df["date"].isin(days)]
    sc, rej = vscore.score_df(df, c)
    df = df.assign(sc=sc, rej=rej)
    top_k = int(lc["objective"]["top_k"])
    regimes = {p.stem: _json(p).get("day_regime", "")
               for p in (STATE / "llm_eval").glob("*.json")}

    daily, worst_best = [], {}
    for d, g in df.groupby("date"):
        ok = g[~g["rej"]].sort_values("sc", ascending=False, kind="mergesort")
        # 「前 10」要按生产那张榜数：过准入 且 round(分,1) >= min_score，再取前 10。
        # 只按 ~rej 取前 K 会把从没发过信的低分票算进成绩（学习线那批修的 F1-1，
        # 在线 17 天里有 4 天两张榜不是同一批票）。
        sel = production_order(g["sc"].to_numpy(), g["rej"].to_numpy(), c, top_k)
        top = g.iloc[sel]
        ic = spearman(ok["sc"].to_numpy(), ok["ytil"].to_numpy()) \
            if len(ok) > 5 else float("nan")
        # 排序增益 = 前 10 的超额 − 过准入全部的超额。它把「准入选得好不好」和
        # 「排序排得好不好」分开：过准入的票不够 top_k 只时，前 10 就是全部，
        # 排序根本没起作用，那天的排序增益恒为 0，必须排除而不是当成 0 参与平均。
        rank_ok = len(ok) > len(top) > 0
        gain = (100 * (top["y"].mean() - ok["y"].mean())) if rank_ok else None
        daily.append({
            "date": d, "pool": int(len(g)), "admitted": int(len(ok)),
            "market_median_pct": _f(100 * g["day_center"].iloc[0], 2),
            "top_excess_pct": _f(100 * top["y"].mean(), 3) if len(top) else None,
            "top_hits": int((top["y"] > 0).sum()), "top_n": int(len(top)),
            "sent_n": int(len(top)),
            "top_mean_score": _f(top["sc"].mean(), 1) if len(top) else None,
            "admitted_excess_pct": _f(100 * ok["y"].mean(), 3),
            "rank_gain_pct": _f(gain, 3),
            "rank_evaluable": bool(rank_ok),
            "ic": _f(ic, 3),
            "regime": regimes.get(d, ""),
        })
        def pack(gg):
            return [{"code": r.code, "name": r.name, "score": _f(r.sc, 1),
                     "rank": int(getattr(r, "rank", 0) or 0),
                     "gap_pct": _f(r.gap_pct, 2), "ret_pct": _f(100 * r.r, 2),
                     "ytil": _f(r.ytil, 2), "sector": r.sector}
                    for r in gg.itertuples(index=False)]
        try:
            from learn import brief as BR
            _all, w_, b_ = BR.pick_worst_best(ok.rename(columns={"sc": "sc"}), 5, 5)
            worst_best[d] = {"worst": pack(w_), "best": pack(b_)}
        except Exception as e:  # noqa: BLE001
            log.warning("%s worst/best 挑选失败: %s", d, e)
            worst_best[d] = {"worst": [], "best": []}

    ok = df[~df["rej"]].copy()
    ok["y_pct"] = 100 * ok["y"]

    def _clustered_se(g) -> tuple[float, int]:
        """按天聚类的标准误：先按天求均值，再对天求标准误。

        同一天的票一起涨一起跌（当天的板块和大盘是共同因子），按票数算 se
        会把 17 天的数据当成几百个独立观测，什么切片都能「显著」。
        """
        per_day = g.groupby("date")["y_pct"].mean()
        nd = int(per_day.notna().sum())
        if nd < 2:
            return float("nan"), nd
        return float(per_day.std(ddof=1) / math.sqrt(nd)), nd

    def _split(frame, mask) -> dict:
        """一个子集的超额：按天均值 + 按天标准误 + 池化值，三个一起给。

        点估计（excess_pct）和 se_day_clustered 同出一条每天一个数的序列，
        比值才有意义；池化的那个另起名字，不许拿去除按天的 se。
        """
        g = frame[mask]
        n = int(len(g))
        if not n:
            return {"n": 0, "excess_pct": None, "excess_pct_pooled": None,
                    "se_day_clustered": None, "n_days": 0}
        per_day = g.groupby("date")["y_pct"].mean()
        nd = int(per_day.notna().sum())
        se = (float(per_day.std(ddof=1) / math.sqrt(nd)) if nd > 1 else float("nan"))
        return {"n": n, "excess_pct": _f(per_day.mean(), 3),
                "excess_pct_pooled": _f(g["y_pct"].mean(), 3),
                "se_day_clustered": _f(se, 3), "n_days": nd}

    def bucket(series, bins, labels):
        """分档的超额：点估计和标准误必须来自**同一条**每天一个数的序列。

        以前 excess_pct 是把全部行池化求均值、se 却是按天聚类算的，分子分母
        不是一个估计量，比值没有意义。2026-09-16 实测差别能到反号：
        gap_pct 3-4 档池化 +0.093、按天 −0.836；liangbi 2.5-4 档 +0.189 / −0.283；
        score 40-50 档 −0.036 / +0.158。六路 LLM 拿 spread/se 判显著性，
        三个维度的结论方向因此是错的（教训 33 堵了分母没堵分子）。
        池化那个数也留着（excess_pct_pooled），它回答的是另一个问题
        「这一档的票平均表现如何」，只是不能拿去除按天的 se。
        """
        cut = pd.cut(series, bins, right=False, labels=labels)
        rows = []
        for k, g in ok.groupby(cut, observed=True):
            se, nd = _clustered_se(g)
            per_day = g.groupby("date")["y_pct"].mean()
            rows.append({"bucket": str(k), "n": int(len(g)),
                         "excess_pct": _f(per_day.mean(), 3),
                         "excess_pct_pooled": _f(g["y_pct"].mean(), 3),
                         "se_day_clustered": _f(se, 3), "n_days": nd,
                         "t": _f(per_day.mean() / se, 2)
                         if se == se and se > 0 else None,
                         "hit_rate": _f((g["y"] > 0).mean(), 3)})
        return rows

    by = {
        "score": bucket(ok["sc"], [0, 40, 50, 60, 70, 101],
                        ["<40", "40-50", "50-60", "60-70", "70+"]),
        "gap_pct": bucket(ok["gap_pct"], [2, 3, 4, 5.01], ["2-3", "3-4", "4-5"]),
    }
    lb = ok["liangbi"] if "liangbi" in ok.columns else ok["auc_ratio"] * 240
    by["liangbi"] = bucket(lb, [2.5, 4, 7, 10.01], ["2.5-4", "4-7", "7-10"])

    # 六个维度：该维度得分最高 1/3 vs 最低 1/3
    dims = []
    try:
        d_ = vscore.prepare(ok)
        parts = vscore.parts(d_, c)
        w = c["scoring"]["weights"]
        for k, v in parts.items():
            v = pd.Series(np.asarray(v, float), index=ok.index)
            # 三分位**按天切**，不按全样本切。按全样本切的话，某个维度在某天
            # 可能只出高三分之一不出低三分之一（2026-09-16 实测：板块维只有
            # 12 天、连续性只有 6 天能配成对），配对差里就混进了「哪天贡献了
            # 多少 hi/lo」的日间成分，而 17 天里只有 6 天有效的配对根本不能用。
            # 按天切之后每天都是自己内部的高 1/3 对低 1/3，天数=全部有效天。
            # 并列值（板块维大量 0、连续性只有几档）会把分位撑爆，
            # 所以按 first 排名切成严格三等份而不是用 qcut。
            tmp = pd.DataFrame({"date": ok["date"].to_numpy(),
                                "v": v.to_numpy(), "y": ok["y_pct"].to_numpy()})
            rk = tmp.groupby("date")["v"].rank(method="first")
            cnt = tmp.groupby("date")["v"].transform("size")
            hi_m = (rk > 2 * cnt / 3).to_numpy()
            lo_m = (rk <= cnt / 3).to_numpy()
            tie = float((v.value_counts(normalize=True).iloc[0]) if len(v) else 0)
            # 点估计和标准误同出一条序列：每天一个「高 1/3 − 低 1/3」，
            # 对天求均值和标准误。以前点估计是池化的、se 是按天的，比值无意义。
            per_day = (tmp[hi_m].groupby("date")["y"].mean()
                       - tmp[lo_m].groupby("date")["y"].mean()).dropna()
            se = (float(per_day.std(ddof=1) / math.sqrt(len(per_day)))
                  if len(per_day) > 1 else float("nan"))
            spread = float(per_day.mean()) if len(per_day) else float("nan")
            a_d = tmp[hi_m].groupby("date")["y"].mean()
            b_d = tmp[lo_m].groupby("date")["y"].mean()
            dims.append({"dim": k, "weight": w.get(k),
                         "high_third_excess_pct": _f(a_d.mean(), 3),
                         "low_third_excess_pct": _f(b_d.mean(), 3),
                         "spread_pct": _f(spread, 3),
                         "spread_pct_pooled": _f(
                             tmp["y"][hi_m].mean() - tmp["y"][lo_m].mean(), 3),
                         "spread_se_day_clustered": _f(se, 3),
                         "spread_t": _f(spread / se, 2)
                         if se == se and se > 0 else None,
                         "spread_n_days": int(len(per_day)),
                         "n_high": int(hi_m.sum()), "n_low": int(lo_m.sum()),
                         "top_value_share": _f(tie, 3),
                         "evaluable": tie < 0.5})
        gb = vscore.assign_group_b(d_, c["screen"])
        groups = {"A": _split(ok, ~gb), "B": _split(ok, gb)}
    except Exception as e:  # noqa: BLE001
        log.warning("维度拆解失败: %s", e)
        groups = {}
    flags = {}
    for col, name in (("prev_limit_up", "昨日涨停"), ("breakout", "突破平台"),
                      ("ma_bull", "均线多头"), ("monotonic", "稳步抬升")):
        if col in ok.columns:
            m = ok[col].astype(bool).to_numpy()
            yes, no = _split(ok, m), _split(ok, ~m)
            # 差值也按天配对给标准误。以前这里只有两个池化的均值、一个 se 都没有，
            # 会诊据此提了「突破平台加分该消融」（是 −1.58 对 否 +0.17）。
            # 实际按天聚类：在线 16 天差 −1.19±0.46（t=−2.58），而同一口径在
            # 回填 400 天上是 −0.09±0.16（t=−0.54），没复现 —— 没有标准误的
            # 两个数摆在一起，看的人只能靠感觉判大小。见 tools/platform_evidence.py。
            per_day = (ok[m].groupby("date")["y_pct"].mean()
                       - ok[~m].groupby("date")["y_pct"].mean()).dropna()
            dse = (float(per_day.std(ddof=1) / math.sqrt(len(per_day)))
                   if len(per_day) > 1 else float("nan"))
            dmean = float(per_day.mean()) if len(per_day) else float("nan")
            flags[name] = {
                "yes_n": yes["n"], "yes_excess_pct": yes["excess_pct"],
                "no_n": no["n"], "no_excess_pct": no["excess_pct"],
                "diff_pct": _f(dmean, 3), "diff_se_day_clustered": _f(dse, 3),
                "diff_t": _f(dmean / dse, 2) if dse == dse and dse > 0 else None,
                "diff_n_days": int(len(per_day))}
    rj = df[df["rej"]]
    # 被剔除的按原因拆：「准入差一点」的票到底好不好，只有按原因看才知道
    by_reason = []
    if "rejected" in rj.columns:
        why = (rj["rejected"].astype(str).str.split("(").str[0]
               .str.replace(r"[-+]?\d+(\.\d+)?%?", "", regex=True)
               .str.replace(r"\s+", " ", regex=True).str.strip())
        rr = rj.assign(_why=why)
        for why, g in rr.groupby("_why"):
            if not why or why in ("None", "nan"):
                continue
            by_reason.append({"reason": why, "n": int(len(g)),
                              "excess_pct": _f(100 * g["y"].mean(), 3),
                              "hit_rate": _f((g["y"] > 0).mean(), 3)})
        by_reason.sort(key=lambda x: -x["n"])
        known = sum(x["n"] for x in by_reason)
        if len(rj) > known:
            g = rr[~rr["_why"].isin([x["reason"] for x in by_reason])]
            by_reason.append({"reason": "其他/无原因", "n": int(len(rj) - known),
                              "excess_pct": _f(100 * g["y"].mean(), 3) if len(g) else None,
                              "hit_rate": _f((g["y"] > 0).mean(), 3) if len(g) else None})
    by["rejected_by_reason"] = by_reason[:14]
    dl = [x for x in daily if x["top_excess_pct"] is not None]
    te = np.array([x["top_excess_pct"] for x in dl], float)
    rg = np.array([x["rank_gain_pct"] for x in daily
                   if x["rank_evaluable"] and x["rank_gain_pct"] is not None], float)
    out.update({
        "days": len(daily),
        "first": days[0], "last": days[-1],
        "daily": daily,
        "summary": {
            "top_excess_mean_pct": _f(te.mean(), 3) if len(te) else None,
            "top_excess_se_pct": _f(te.std(ddof=1) / math.sqrt(len(te)), 3) if len(te) > 1 else None,
            "top_win_days": int((te > 0).sum()), "n_days": int(len(te)),
            # 排序增益：噪声比前 10 超额小得多，是在线上最快能判出排序好坏的指标
            "rank_gain_mean_pct": _f(rg.mean(), 3) if len(rg) else None,
            "rank_gain_se_pct": _f(rg.std(ddof=1) / math.sqrt(len(rg)), 3) if len(rg) > 1 else None,
            "rank_gain_days": int(len(rg)),
            "rank_gain_win_days": int((rg > 0).sum()) if len(rg) else 0,
            "days_to_detect_0p5": (int(np.ceil((2.8 * rg.std(ddof=1) / 0.5) ** 2))
                                   if len(rg) > 1 and rg.std(ddof=1) > 0 else None),
            "admitted_excess_mean_pct": _f(ok["y_pct"].mean(), 3),
            "rejected_excess_mean_pct": _f(100 * rj["y"].mean(), 3),
            "n_admitted": int(len(ok)), "n_rejected": int(len(rj)),
            "ic_mean": _f(np.nanmean([x["ic"] for x in daily if x["ic"] is not None]), 3),
        },
        "by": by, "dims": dims, "groups": groups, "flags": flags,
        "worst_best": worst_best,
        "learning": _learning_state(),
        "backfill_vs_online": _backfill_vs_online(df),
        "definitions": {
            "y": "开盘买收盘卖的收益，减去当日全池中位数，按 q1/q99 缩尾（单位：%）",
            "ytil": "y 再除以当日 MAD，跨天可比",
            "top_excess_pct": "**当天真发出去的那张榜**（过准入 且 分数 >= output.min_score，"
                              "再取前 10）的 y 均值。一只都发不出去的天是 null，不是 0",
            "sent_n": "当天真发出去几只（生产口径），可能少于 10",
            "rank_gain_pct": "前 10 的 y 均值 − 过准入全部的 y 均值。过准入不足 10 只的天为 null"
                             "（那天前 10 就是全部，排序没起作用）",
            "se_day_clustered": "按天聚类的标准误：先按天求均值再对天求 se。同一天的票不独立，"
                                "按票数算的 se 会虚高",
            "excess_pct / spread_pct": "**按天**的均值（每天算一个数，再对天平均），"
                                       "和它旁边的 se_day_clustered 是同一个估计量，"
                                       "所以 t = 点估计 / se 才有意义（t 已经算好放在 t / spread_t 里）。"
                                       "带 _pooled 后缀的是把所有行池化的均值，回答的是"
                                       "「这一档的票平均表现如何」，**不要**拿它去除按天的 se —— "
                                       "2026-09-16 之前就是这么混着算的，六个维度里三个的符号是反的",
            "dims 的高/低三分之一": "**按天**各切各的（每天在自己当天的票里取该维度得分最高 1/3 和"
                                "最低 1/3），不是在全样本上切。按全样本切会出现某天只有 hi 没有 lo，"
                                "配对天数掉到 6~12 天",
            "days_to_detect_0p5": "以 80% 把握分辨 ±0.5 个百分点/天的排序增益，还需要多少个真值日",
            "ic": "当日 Spearman(分数, ytil)，只算过准入的票",
            "groups": "A 组 = 昨日涨停 / 连板 / 突破平台（接力、强势）；B 组 = 60 日位置 <= "
                      f"{c['screen'].get('pos_pct_60d_max_for_lowbase')}（低位首板预备）；其余归 A",
            "rejected_by_reason": "被硬剔除的票按原因分组的 y（只有「涨幅区间外」「量比区间外」"
                                  "才是准入差一点的近邻，其余是风险剔除）",
        },
    })
    return out


def _learning_state() -> dict:
    st = _json(STATE / "learning_status.json")
    verdicts = _jsonl(STATE / "verdict_log.jsonl")[-10:]
    theta = _jsonl(STATE / "theta_history.jsonl")[-5:]
    return {
        "status_date": st.get("date"), "n_train_days": st.get("n_days"),
        "train_source": st.get("train_source"),
        "theta_version": st.get("theta_version"),
        "metrics": st.get("metrics"),
        "last_verdict": st.get("verdict"),
        "recent_verdicts": [{"date": v.get("date"), "accepted": v.get("accepted"),
                             "failed": v.get("failed"), "moved": v.get("moved")}
                            for v in verdicts],
        "theta_history": theta,
        "shadow_stat": st.get("shadow_stat"),
        "shadow_recent": (st.get("shadow") or [])[-10:],
        "accepted_total": st.get("accepted_total"),
    }


def _backfill_vs_online(online: pd.DataFrame) -> dict:
    """同名列在回填表和在线快照里的分布差。教训 30：口径不一致不报错。"""
    files = sorted(glob.glob(str(ROOT / "data" / "train" / "backfill_*.parquet")))
    if not files:
        return {"note": "没有回填表"}
    cols = ["gap_pct", "auc_ratio", "auc_amount", "pos_pct_60d", "slope", "dive",
            "board_height", "sector_members", "t1_chg", "t3_chg"]
    flags = ["monotonic", "prev_limit_up", "breakout", "ma_bull", "one_word",
             "prev_broken_board", "blacklisted"]
    try:
        bf = pd.read_parquet(files[-1], columns=cols + flags + ["date"])
    except Exception as e:  # noqa: BLE001
        return {"note": f"回填表读取失败 {e}"}
    rows = []
    for ccol in cols:
        if ccol in online.columns and ccol in bf.columns:
            a, b = pd.to_numeric(online[ccol], errors="coerce"), pd.to_numeric(bf[ccol], errors="coerce")
            rows.append({"col": ccol, "online_median": _f(a.median()),
                         "backfill_median": _f(b.median()),
                         "online_mean": _f(a.mean()), "backfill_mean": _f(b.mean())})
    for fcol in flags:
        if fcol in online.columns and fcol in bf.columns:
            rows.append({"col": fcol,
                         "online_rate": _f(online[fcol].astype(bool).mean(), 3),
                         "backfill_rate": _f(bf[fcol].astype(bool).mean(), 3)})
    # 回填表按量比档的收益（和在线 by.liangbi 对照，看「量能反向」是不是在线才有）
    bf_by = []
    try:
        bfl = pd.read_parquet(files[-1], columns=["auc_ratio", "gap_pct", "r", "date"])
        bfl = bfl[(bfl["gap_pct"] >= 2) & (bfl["gap_pct"] <= 5)]
        med = bfl.groupby("date")["r"].transform("median")
        bfl = bfl.assign(y_pct=100 * (bfl["r"] - med), lb=bfl["auc_ratio"] * 240)
        cut = pd.cut(bfl["lb"], [2.5, 4, 7, 10.01], right=False, labels=["2.5-4", "4-7", "7-10"])
        for k, g in bfl.groupby(cut, observed=True):
            bf_by.append({"bucket": str(k), "n": int(len(g)), "excess_pct": _f(g["y_pct"].mean(), 3),
                          "hit_rate": _f((g["y_pct"] > 0).mean(), 3)})
    except Exception as e:  # noqa: BLE001
        log.warning("回填按量比档失败: %s", e)
    return {"backfill_file": Path(files[-1]).name,
            "backfill_days": int(bf["date"].nunique()),
            "online_days": int(online["date"].nunique()),
            "rows": rows,
            "backfill_by_liangbi": bf_by,
            "backfill_y_note": "回填 y = r 减当日（涨幅 2~5% 池）中位数，未缩尾，口径和在线 y 近似",
            "known_biases": "2026-09-16 那批审计之后：撮合价改取 stk_auction_o 的 open（以前取的 close 是 "
                            "09:30 之后的价，只有 44% 对得上）；竞价轨迹回填表给不出，一律按「没有证据」"
                            "（t1=t2=t3=gap_pct、slope=dive=0、monotonic=False），和线上 T1 漏采同口径，"
                            "trend 维在回填上不可学、已从可学集合里钉死；竞价额含 09:30 之后的成交"
                            "（中位高 25~33%），volume 维同样钉死；候选池两边已统一走 premarket.include_mask；"
                            "one_word / limit_pct 已统一走 datasource。仍然存在的：假涨停规则在回填表上不可观测"
                            "（在线 1.34% 触发、回填 0.001%），现金分红那种小除息判不出来。"}


# ---------------------------------------------------------------------
#  起涨预测
# ---------------------------------------------------------------------
def breakout() -> dict:
    import truth as T
    import export as E
    res = T.compute()
    T.save(res)
    exp = {"streak_perf": [{"streak_ge": k, "hit_pct": h, "lift": l, "n": n}
                           for k, h, l, n in E.STREAK_PERF],
           "base_pct": E.BASE,
           # 分数分档直接读实验产物：export.SCORE_TABLE 这个常量 2026-09-16
           # 审计时删掉了（定义了从没渲染过，属于死常量），别再从那边拿
           "score_bins": _json(ROOT / "out_breakout" / "score_calibration.json").get("bins"),
           "cap_perf": {"full_cap": E.CAP_PERF, "not_full": getattr(E, "NONCAP_PERF", None),
                        "note": "满员日（够格超过上限、被截断）vs 门槛卡得住的日子，"
                                "格式 (准确率%, 名额数, 天数)"},
           "window": E.PERF.get("window")}
    lists = []
    for L in res["lists"]:
        lists.append({k: (_f(v, 4) if isinstance(v, float) else v)
                      for k, v in L.items()})
    picks = [{k: (_f(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
             for p in res["picks"]]
    by = {}
    for k, v in res["by"].items():
        if isinstance(v, list):
            by[k] = [{kk: (_f(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in r.items()} for r in v]
        else:
            by[k] = {kk: (_f(vv, 4) if isinstance(vv, float) else vv)
                     for kk, vv in v.items()}
    return {
        "window": res["window"], "threshold": res["threshold"],
        "n_lists": len(lists), "n_final_lists": sum(1 for L in lists if L["final"]),
        "lists": lists, "picks": picks, "by": by,
        "expected": exp,
        "model": _breakout_model(),
        "data_health": _breakout_data_health(),
        "definitions": {
            "rise": "上榜日收盘到之后 min(20, 已走) 根 K 线最高价的涨幅",
            "hit": "rise > 50%（和训练标签 y_up 同口径）",
            "final": "20 根已走满，命中才算最终",
            "base_final/base_sofar": "同一天全市场随便买一只、同一窗口的命中比例（分别按已满/到目前为止）",
            "expected.streak_perf": "邮件里印的验证集成绩（207 个交易日走向前）",
        },
    }


def _breakout_model() -> dict:
    mj = _json(STATE / "breakout" / "model.json")
    out = {"fit_date": mj.get("fit_date"), "train_cut": mj.get("train_cut"),
           "n_feats": len(mj.get("feats") or []), "fingerprint": mj.get("fingerprint")}
    mt = STATE / "breakout" / "model.txt"
    try:
        if mt.exists() and mj.get("feats"):
            import lightgbm as lgb
            b = lgb.Booster(model_file=str(mt))
            gain = b.feature_importance(importance_type="gain")
            tot = float(gain.sum()) or 1.0
            imp = sorted(zip(mj["feats"], gain), key=lambda x: -x[1])
            out["importance"] = [{"feat": f, "gain_share": _f(g / tot, 4)} for f, g in imp]
            groups: dict[str, float] = {}
            try:
                import features as F
                g2f = {f: g for g, fs in F.GROUPS.items() for f in fs}
                for f, g in imp:
                    base = f.rsplit("__", 1)[0]
                    groups[g2f.get(base, "other")] = groups.get(g2f.get(base, "other"), 0) + g / tot
                out["importance_by_group"] = {k: _f(v, 4) for k, v in
                                              sorted(groups.items(), key=lambda x: -x[1])}
            except Exception as e:  # noqa: BLE001
                log.warning("按组重要性失败: %s", e)
    except Exception as e:  # noqa: BLE001
        log.warning("读模型重要性失败: %s", e)
    # 最近一份清单的特征分位（横截面百分位，直接可读）
    files = sorted((ROOT / "data" / "breakout").glob("*/breakout_*.parquet"))
    if files and mj.get("feats"):
        try:
            d = pd.read_parquet(files[-1])
            feats = [f for f in mj["feats"] if f in d.columns]
            rows = []
            for r in d.itertuples():
                rows.append({"code": str(r.code).zfill(6), "score": _f(getattr(r, "score", 0), 0),
                             "feats": {f: _f(getattr(r, f), 3) for f in feats[:20]}})
            out["latest_list_feats"] = {"date": str(d["date"].iloc[0]), "rows": rows,
                                        "note": "特征值 = 当日横截面百分位经板块×市值分箱中性化后的残差"
                                                "（正 = 高于同板块同市值档均值，量级约 ±0.5），"
                                                "__mean/__slope 是 5 日聚合；只列重要性前 20 个"}
        except Exception as e:  # noqa: BLE001
            log.warning("读清单特征失败: %s", e)
    # 生产模型旁边那份（daily.py 落的，带 fingerprint）。以前读的是
    # out_breakout/feature_select.json —— 那是实验脚本 09-12 写的，start=87
    # end=34，而生产模型是 102 进 51 出：入模的 51 列里 16 列在那份记录的
    # dropped 里。提案 20260916-9b9104 报的「87->34 与模型矛盾」就是这么来的，
    # 是读错路径造出来的假阳。指纹一起给出去，视角能自己核对是不是同一次训练。
    fs = _json(ROOT / "state" / "breakout" / "feature_select.json")
    if fs:
        out["feature_select"] = {
            "start": fs.get("start"), "end": fs.get("end"),
            "fingerprint": fs.get("fingerprint"), "fit_date": fs.get("fit_date"),
            "dropped": {k: len(v) if isinstance(v, list) else v
                        for k, v in (fs.get("dropped") or {}).items()},
            "matches_model": (str(fs.get("fingerprint") or "")[:8]
                              == str((out.get("fingerprint") or ""))[:8]) or None}
    wg = _json(ROOT / "out_breakout" / "window_grid.json")
    if wg:
        out["walk_forward_w5"] = [r for r in wg.get("grid", []) if r.get("kind") == "W5"]
    try:
        out["validation_w5"] = _validation_w5()
    except Exception as e:  # noqa: BLE001
        log.warning("验证集 W5 拆解失败: %s", e)
    return out


def _validation_w5() -> dict:
    """验证集（逐月滚动缓存 wf_scores.parquet）里生产口径名额的两个拆解：
    按板块的命中率；进度基准 = 最终命中的票在第 k 根时已经涨了多少。
    进行中的清单只能和后者比，不能和 12.6% 比。"""
    import daily as D
    import features as F
    cache = ROOT / "data" / "breakout" / "raw" / "wf_scores.parquet"
    if not cache.exists():
        return {"note": "没有逐月滚动缓存"}
    d = pd.read_parquet(cache)
    ok = d[(d["score"] >= D.SCORE_MIN) & (d["rank"] <= D.CAP_A)].copy()
    ok = ok[np.isfinite(ok["y_up"])]
    ok["board"] = ok["code"].map(F.board_of)
    by_board = []
    for b, g in ok.groupby("board"):
        h = int(g["y_up"].sum())
        by_board.append({"board": b, "n": int(len(g)), "hits": h, "hit_rate": _f(h / len(g), 4),
                         "share_of_picks": _f(len(g) / len(ok), 3)})
    # 进度基准
    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                         columns=["code", "date", "high", "close"])
    px["date"] = px["date"].astype(str)
    px = px[px["code"].isin(set(ok["code"]))].sort_values(["code", "date"])
    grp = {c: (g["date"].to_numpy(), g["high"].to_numpy(), g["close"].to_numpy())
           for c, g in px.groupby("code")}
    ks = (3, 5, 7, 10, 15)
    rec = []
    for r in ok.itertuples():
        g = grp.get(r.code)
        if g is None:
            continue
        dates, high, close = g
        i = int(np.searchsorted(dates, r.date))
        if i >= len(dates) or dates[i] != r.date or close[i] <= 0:
            continue
        row = {"hit": int(r.y_up)}
        for k in ks:
            seg = high[i + 1:i + 1 + k]
            row[f"r{k}"] = float(seg.max() / close[i] - 1) if len(seg) else np.nan
        rec.append(row)
    rf = pd.DataFrame(rec)
    pace = {}
    for k in ks:
        col = f"r{k}"
        if col not in rf:
            continue
        sub = rf[np.isfinite(rf[col])]
        hit, miss = sub[sub["hit"] == 1], sub[sub["hit"] == 0]
        p25 = sub[sub[col] >= 0.25]
        pace[f"bar{k}"] = {
            "n": int(len(sub)),
            "mean_rise_all": _f(sub[col].mean(), 4),
            "mean_rise_hits": _f(hit[col].mean(), 4) if len(hit) else None,
            "mean_rise_misses": _f(miss[col].mean(), 4) if len(miss) else None,
            "share_hits_already_ge25": _f((hit[col] >= 0.25).mean(), 3) if len(hit) else None,
            "p_hit_given_ge25": _f(p25["hit"].mean(), 3) if len(p25) else None,
            "n_ge25": int(len(p25)),
            "share_all_ge15": _f((sub[col] >= 0.15).mean(), 3),
        }
    return {"n_picks": int(len(ok)), "hit_rate": _f(ok["y_up"].mean(), 4),
            "by_board": by_board, "pace": pace,
            "note": "pace.barK：验证集名额在第 K 根时的最高涨幅；进行中的清单拿同一根数对照。"
                    "p_hit_given_ge25 = 第 K 根已涨 25% 的票最终命中的比例"}


def _breakout_data_health() -> dict:
    dp = ROOT / "data" / "breakout" / "daily.parquet"
    out: dict = {}
    try:
        px = pd.read_parquet(dp, columns=["code", "date"])
        cnt = px.groupby("date").size()
        last = cnt.tail(30)
        out["codes_per_day_recent"] = [{"date": str(k), "n": int(v)} for k, v in last.items()]
        out["codes_per_day_min_all"] = {"n": int(cnt.min()), "date": str(cnt.idxmin())}
        out["thin_days"] = int((cnt < 0.8 * cnt.median()).sum())
        out["date_range"] = [str(px["date"].min()), str(px["date"].max())]
    except Exception as e:  # noqa: BLE001
        out["note"] = f"daily.parquet 读取失败 {e}"
    files = sorted((ROOT / "data" / "breakout").glob("*/breakout_*.parquet"))
    if files:
        try:
            d = pd.read_parquet(files[-1])
            fc = [c for c in d.columns if "__" in c]
            nan = d[fc].isna().mean()
            out["latest_list_nan_feats"] = {k: _f(v, 2) for k, v in nan[nan > 0].items()}
        except Exception as e:  # noqa: BLE001
            out["note2"] = str(e)
    rm = _json(ROOT / "out_breakout" / "run_meta.json")
    out["run_meta"] = rm
    return out


# ---------------------------------------------------------------------
def build(c: dict, date: str, n_days: int = 0) -> Path:
    """两条线的证据包写到 state/council/<date>/evidence.json，返回路径。"""
    pack = {"date": date, "generated_at": dt.datetime.now().isoformat(timespec="seconds")}
    try:
        pack["morning"] = morning(c, n_days)
    except Exception as e:  # noqa: BLE001
        log.warning("早盘证据失败: %s", e)
        pack["morning"] = {"error": str(e)}
    try:
        pack["breakout"] = breakout()
    except Exception as e:  # noqa: BLE001
        log.warning("起涨预测证据失败: %s", e)
        pack["breakout"] = {"error": str(e)}
    try:
        # 市场环境：自己算的，省得 LLM 去网上查成交额还没法核对（会诊第一次就是这么干的）
        import regime as RG
        RG.record(90)
        pack["regime"] = {
            "recent": RG.load(30),
            "definitions": {
                "turnover_yi": "沪深京全市场成交额（亿元），只统计日线表里有的票",
                "adv_share": "上涨家数占比", "limit_up/limit_down": "按板块涨跌停幅度近似计数（ST 未单列）",
                "base20": "那一天全市场随便买一只，之后 20 根 K 线内最高价涨超 50% 的比例。"
                          "起涨预测的命中率要和它比，不是和验证集的 2.93% 比；"
                          "不满 20 根的日子是 null",
                "ret_median_pct": "全市场当日涨跌幅中位数",
            }}
    except Exception as e:  # noqa: BLE001
        log.warning("环境指标失败: %s", e)
        pack["regime"] = {"error": str(e)}
    d = OUT_DIR / date
    d.mkdir(parents=True, exist_ok=True)
    p = d / "evidence.json"
    p.write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8")
    # 给 LLM 读的紧凑版：去掉逐只明细（它要看可以用查询工具）
    slim = json.loads(json.dumps(pack))
    if isinstance(slim.get("breakout"), dict):
        slim["breakout"].pop("picks", None)
        m = slim["breakout"].get("model") or {}
        m.pop("latest_list_feats", None)
    if isinstance(slim.get("morning"), dict):
        wb = slim["morning"].get("worst_best") or {}
        keep = sorted(wb)[-5:]
        slim["morning"]["worst_best"] = {k: wb[k] for k in keep}
    (d / "evidence_slim.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log.info("证据包 -> %s（%d KB）", p, p.stat().st_size // 1024)
    return p


if __name__ == "__main__":
    import cfg as C
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    print(build(C.load(), d))
