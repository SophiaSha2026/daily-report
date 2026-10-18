"""evalkit 登记（新族 breakout_blind）。

    PYTHONIOENCODING=utf-8 python research/blind/prereg_bl.py --backfill
        把已经读过开发集的臂补登进 blind_v1（hypothesis 写明「事后补登」），落差值序列、记结果；
        筛选集为负、evalkit 拒登的臂（V1 / V2）和其余开发集加看按 look 记进主登记簿（族计数照实）。
    PYTHONIOENCODING=utf-8 python research/blind/prereg_bl.py --exp blind_v2 --arms S:q90:10:1,S:multi3:10:1 --hyp "..."
        新臂先登记（只算筛选集）；筛选集 < 0 的 evalkit 会拒登，那个臂就不读开发集。
    PYTHONIOENCODING=utf-8 python research/blind/prereg_bl.py --exp blind_v2 --record S:q90:10:1
        登记之后读开发集：落差值序列 + 结果（含 NW 滞后 20 / 40、块 20 / 40 的 t 和 p）。
臂名：V3 | V3b | L3reg | L2fe | S:<分数名>:<K>:<n>（ret20 前 K 内按分数取前 n）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_bl as B
import evalkit as EK
from diag_v3 import cred
from v3_bl import asof_cum

EXP = EK.EXP


def build(arm: str, t: pd.DataFrame) -> np.ndarray:
    rank = t["rank_ret20"].to_numpy()
    y = t["y_open"].to_numpy(float)
    if arm == "V3":
        s = B.load_scores("lgb_price").astype(float)
        c, hist_n = cred(t, s, y)
        cpct = B.day_pct(t, np.where(hist_n > 0, c, np.nan))
        return B.topn_by(t, (rank <= 20) & (cpct >= 0.9), c, 1)
    if arm == "V3b":
        ok10 = (rank <= 10) & np.isfinite(y)
        ev10 = pd.DataFrame({"code": t["code"].to_numpy()[ok10], "end20": t["end20"].to_numpy()[ok10], "h": y[ok10]})
        cum10 = asof_cum(t, ev10, ["h"])
        h_r, n_r = cum10["h"].to_numpy(), cum10["cnt"].to_numpy()
        g = ev10.groupby("end20")["h"].agg(["sum", "size"]).sort_index()
        g["cs"], g["cn"] = g["sum"].cumsum(), g["size"].cumsum()
        e_dates = g.index.to_numpy().astype("U10")
        pos = np.searchsorted(e_dates, t["date"].to_numpy().astype("U10"), side="left") - 1
        p0 = np.where(pos >= 0, g["cs"].to_numpy()[np.clip(pos, 0, None)] / np.maximum(g["cn"].to_numpy()[np.clip(pos, 0, None)], 1), np.nan)
        c_r = (h_r + 5 * np.nan_to_num(p0, nan=0.1)) / (n_r + 5)
        return B.topn_by(t, (rank <= 10) & (n_r == 0), c_r, 1)
    if arm == "L3reg":
        return B.topn_by(t, rank <= 10, B.load_scores("reg_l2").astype(float), 1)
    if arm == "L2fe":
        return B.topn_by(t, rank <= 10, B.load_scores("fe_nomkt").astype(float), 10)
    if arm.startswith("S:"):
        _, name, K, n = arm.split(":")
        return B.topn_by(t, rank <= int(K), B.load_scores(name).astype(float), int(n))
    raise SystemExit("不认识的臂 " + arm)


def series(r: dict) -> pd.Series:
    return pd.Series(np.concatenate([r["screen"]["u"], r["dev"]["u"]]), index=np.concatenate([r["screen"]["dates"], r["dev"]["dates"]]))


def robust(u: pd.Series) -> dict:
    x = u.to_numpy(float)
    out = {}
    for lag in (20, 40):
        out["t_nw%d" % lag] = float(np.mean(x) / EK._nw_se(x, lag))
    for blk in (20, 40):
        out["boot_p_b%d" % blk] = EK.block_boot(x, block=blk)["p"]
    return out


def record_arm(exp: str, arm: str, t: pd.DataFrame, backfill_note: str = "") -> dict:
    sel = build(arm, t)
    r = B.eval_both(t, sel, boot=True)
    u = series(r)
    d = EXP / exp
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("diff_%s.parquet" % arm.replace(":", "_"))
    pd.DataFrame({"date": u.index, "d": u.to_numpy()}).to_parquet(p, index=False)
    rb = robust(u)
    res = {"arm": arm, "verdict": B.verdict(r), "screen": B.screen_row(arm, r["screen"]), "dev": B.screen_row(arm, r["dev"]),
           "dev_P_lo_hi": [r["dev"].get("P_lo"), r["dev"].get("P_hi")], "screen_P_lo_hi": [r["screen"].get("P_lo"), r["screen"].get("P_hi")],
           "comb": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in r["comb"].items()}, **rb, "series": str(p), "note": backfill_note}
    EK.record(exp, res)
    EK.ledger_add({"family": B.FAMILY, "exp": exp, "arm": arm, "kind": "result", "series": str(p), "comb_t": r["comb"]["t"], **rb,
                   "verdict": B.verdict(r), "note": backfill_note})
    B.say("%s/%s：筛选 ΔP %+.2f (t %.2f)  开发 %+.2f (t %.2f)  合并 t20 %.2f t40 %.2f  块20 p %.3f 块40 p %.3f  -> %s" % (
        exp, arm, 100 * r["screen"]["dP"], r["screen"]["t"], 100 * r["dev"]["dP"], r["dev"]["t"], rb["t_nw20"], rb["t_nw40"],
        rb["boot_p_b20"], rb["boot_p_b40"], B.verdict(r)))
    return res


def screen_only(arm: str, t: pd.DataFrame) -> dict:
    sel = build(arm, t)
    e = B.G.eval_gate(t, sel, "screen", boot=False)
    return {"diff_day": float(np.mean(e["u"])), "days": int(len(e["u"])), "span": ["2022-01", "2024-05"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--exp", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--hyp", default="")
    ap.add_argument("--record", default="")
    a = ap.parse_args()
    t = B.load_keys()
    if a.backfill:
        exp = "blind_v1"
        note = "事后补登：这些臂的开发集已在 2026-10-18 06:30~07:05 BJ 读过（v1_bl / v3_bl / eval_scores 各一次），登记在读之后，协调者 07:10 审计指出后补登"
        arms = {"V3": "逐票可信度：前 20 里 c=Σ(y−s)/(n+20) 当天分位>=.9 且最高的 1 只（s=lgb_price 样本外），对 B(1)",
                "V3b": "前 10 里第一次进榜（以前没进过前 10）的票按 c_r 取 1 只，对 B(1)",
                "L3reg": "回归目标 clip(log1p(r20_open)) 的 LightGBM，前 10 内按分数取 1 只，对 B(1)",
                "L2fe": "init_score=logit(当天基准率) 的 LightGBM（无市场状态列），前 10 内按分数取 10 只（= B(10) 自身，差 0）"}
        sc = {k: screen_only(k, t) for k in arms}
        B.say("筛选集按天差：%s" % {k: round(v["diff_day"], 5) for k, v in sc.items()})
        EK.prereg(exp, hypothesis="研究盲测近期（research/blind/criteria.txt 第 1 节）+ 底层杠杆（第 2 节）。尺子 = 精确率对同量名次基线 B(k̄)，"
                  "按天 u_d = h_d − h_B,d。" + note, arms=arms, metric="precision vs B(kbar) 按天 u", family=B.FAMILY, screen=sc)
        for k in arms:
            record_arm(exp, k, t, backfill_note=note)
        # 筛选集为负、evalkit 拒登但已读开发集的臂，和其余开发集加看：按 look 计进族计数
        looks = [("V1 K10 q0.98 n2 sT", "V1 盲测臂，筛选 −0.48pp，06:30 读了开发集（违反条文：筛选为负不该读）"),
                 ("V2 K10 n2", "V2 结构模型臂，筛选 −0.34pp，06:55 读了开发集（同上）"),
                 ("V3plain", "diag_v3 机制对照（只用自身历史命中）开发集看一次"), ("V3 n0=100", "diag_v3 收缩 n0=100 开发集看一次"),
                 ("V3 K10 n1 c", "diag_v3 邻域 开发集看一次"), ("V3 K50 n1 c", "diag_v3 邻域 开发集看一次"), ("V3 第2只", "diag_v3 开发集看一次"),
                 ("V2 对照差", "eval_scores V2 对 lgb_price 同配置开发集配对差"), ("L2fe 对照差", "eval_scores 对照差"), ("L3reg 对照差", "eval_scores 对照差"),
                 ("look_v3", "V3 臂长滞后 / 按票自举（同臂，不算新臂但记一次看）")]
        for nm, what in looks:
            EK.ledger_add({"family": B.FAMILY, "exp": exp, "arm": nm, "kind": "look", "note": what})
        B.ledger_add({"kind": "prereg", "what": "补登 blind_v1：4 个臂 + 11 次 look 进主登记簿", "arms": list(arms)})
        th = EK.family_threshold(B.FAMILY)
        B.say("族 %s 计数 %d，门槛 %s" % (B.FAMILY, EK.family_count(B.FAMILY), json.dumps(th, ensure_ascii=False, default=float)))
        B.progress("prereg_bl --backfill：补登 4 个臂（V3 / V3b / L3reg / L2fe）进 evalkit blind_v1，11 次 look 记入；族计数 %d，门槛 c %.2f" % (
            EK.family_count(B.FAMILY), th["c"]))
        return
    if a.arms:
        arms = {k: k for k in a.arms.split(",")}
        sc = {k: screen_only(k, t) for k in arms}
        B.say("筛选集按天差：%s" % {k: round(v["diff_day"], 5) for k, v in sc.items()})
        bad = [k for k, v in sc.items() if v["diff_day"] < 0]
        if bad:
            B.say("筛选集为负、不登记也不读开发集：%s" % bad)
            for k in bad:
                EK.ledger_add({"family": B.FAMILY, "exp": a.exp, "arm": k, "kind": "look", "note": "筛选集 ΔP<0，拒登，开发集不读；screen diff_day %.5f" % sc[k]["diff_day"]})
                arms.pop(k)
                sc.pop(k)
        if arms:
            EK.prereg(a.exp, hypothesis=a.hyp, arms=arms, metric="precision vs B(kbar) 按天 u", family=B.FAMILY, screen=sc)
            B.say("已登记 %s：%s" % (a.exp, list(arms)))
        B.ledger_add({"kind": "prereg", "exp": a.exp, "arms": list(arms), "rejected": bad})
        B.progress("prereg_bl：%s 登记 %s，拒登 %s" % (a.exp, list(arms), bad))
    if a.record:
        pre = EK.prereg_info(a.exp)
        assert a.record in pre.get("arms", {}), "先登记再读开发集"
        record_arm(a.exp, a.record, t)
        B.ledger_add({"kind": "arm", "exp": a.exp, "name": a.record, "what": "登记后读开发集"})
        th = EK.family_threshold(B.FAMILY)
        B.say("族计数 %d，门槛 c %.2f" % (EK.family_count(B.FAMILY), th["c"]))


if __name__ == "__main__":
    main()
