"""
规则核预登记 + 两段判定（docs/rethink_2026-10-16.md 拍板 B / C / D）。

    PYTHONIOENCODING=utf-8 python research/rulecore/judge.py --prereg     看结果前登记（evalkit.prereg，新族，gate v3）
    PYTHONIOENCODING=utf-8 python research/rulecore/judge.py --judge      跑完 rulecore.py 和 pc_arm.py 之后判定

臂（对照 = 现状 base_s7，同循环、同候选池）
    rc     规则核：单调逻辑回归，输入 z(log(1+ret20)) z(vol60) z(turn20) z(amp20) + 板块，每月重拟合，每天前 10
    pc     百分位压缩假说对照：现状 LightGBM 把 rs20__last 换成原值 log(1+ret20)
    ret20  一行字参照：每天 20 日涨幅最高的 10 只（rethink 已在两段上看过一次，这里只作参照，照样登记计数）
主尺子  每天前 10 买得到命中（y_open）按天配对差，两段各自 >= 0，合并 969 天走 evalkit.compare v3
辅助    前 30 命中（必须和前 10 同号）、按天 AUC（只淘汰）、三句事实、板块构成、零命中天、相邻日 Jaccard
旧族    breakout_y50_top10 的 71 次试验作 Bonferroni 附加项：严格门槛 c_strict = Φ⁻¹(1 − 0.05 / (71 + 本族臂数))，
        和 evalkit 新族门槛并列印出。正式判定按拍板 D 用新族门槛（max(2.5, 族门槛)）；严格读法只作标注，
        不改 evalkit 的判定逻辑（任务书要求）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_rc as C  # noqa: E402

import evalkit as K  # noqa: E402

OUT = C.EXP / C.NAME
OLD_FAMILY, OLD_COUNT = "breakout_y50_top10", 71
ARMS = {
    "rc": "规则核：单调逻辑回归（w>=0），输入 z(log(1+ret20)) z(vol60) z(turn20) z(amp20) 当日内 z 化 + 板块哑变量，"
          "训练 y_open 按月等权，每月重拟合（前 3 年 + validate.train_slice 净化），每天候选池按 s0 取前 10，无门槛",
    "pc": "百分位压缩假说对照：现状 LightGBM（同特征选择、同切片、种子 7）把 rs20__last 换成原值 log(1+ret20)",
    "ret20": "一行字参照：每天候选池按 20 日涨幅取前 10",
}
HYP = ("日线上「20 根内最高价涨 50%」是波动延续标签，能预测的部分几乎全是 20 日涨幅、60 日波动、换手、板块（rethink 第 1 节）；"
       "现状 LightGBM 的横截面百分位 + 中性化把尾部压平。假设：规则核（<= 5 个原始刻度输入的单调逻辑回归）在筛选集 2022-01..2024-05 "
       "和开发集 2024-06..2025-12 上，每天前 10 买得到命中都不低于同循环现状 base，合并 969 天按天配对 NW t >= 2.5；"
       "对照臂 pc 回答「只把 ret20 原值喂给 LightGBM 够不够」。候选池所有臂同一份：ST / <2 元 / <120 根 / 停牌 / 近 30 天减持 / "
       "未来 30 天解禁 >= 5% / 近 60 天增发。只挑这一次：规则核的形式、输入、出票规则在看结果前定死。")
METRIC = "top10 y_open 按天配对（两段各自同号 + 合并 969 天 v3）；辅助 top30 同号、按天 AUC 只淘汰"


def c_strict(n_new: int) -> float:
    return NormalDist().inv_cdf(1 - 0.05 / (OLD_COUNT + n_new))


def do_prereg() -> None:
    assert K.family_count(OLD_FAMILY) == OLD_COUNT, K.family_count(OLD_FAMILY)
    gate = {**K.GATE_V3, "two_segments_min_diff": 0.0, "top30_same_sign": True,
            "auc_eliminate_t": -2.5, "k": C.K_TOP,
            "old_family_bonferroni": {"family": OLD_FAMILY, "count": OLD_COUNT, "c_strict": c_strict(len(ARMS)),
                                      "role": "并列标注，不进正式判定（拍板 D 主门槛 2.5）"}}
    f = K.prereg(C.NAME, HYP, ARMS, metric=METRIC, gate=gate, family=C.FAMILY)
    spec = {
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "segments": C.SEGS, "merged_days_expected": 969,
        "data": "research/screen 长表 pit 价 + labels_screen（2025-12-04..31 从 leak 快照补），现状 = screen_base_v1 / screen_base_v1_dev "
                "+ pc_arm.py 补的 2025-12 base",
        "rulecore": {"inputs": list(C.INPUTS), "transform": "当日内 z 化（候选池 elig0 上的均值 / 标准差），不做百分位、不中性化、不截尾",
                     "model": "逻辑回归，四个连续输入系数 >= 0（L-BFGS-B bounds），板块 3 个哑变量无约束，L2 1e-3",
                     "train": "y_open，按月等权（model.month_weights），每月重拟合：前 3 年 + validate.train_slice 净化，候选池行",
                     "pick": "每天候选池按 s0 取前 10，无分数门槛。理由：当日内 z 化后 s0 只有当天内相对意义，概率门槛等价于名额；"
                             "生产的 97 分是预测值在训练参照样本里的百分位，在筛选集上已退化成前 10（research/screen/report.md）"},
        "exclusions": {"st": "validate.st_codes 静态名单", "price_min": C.PRICE_MIN, "min_history_days": "daily.MIN_HISTORY_DAYS (120)",
                       "suspension": f"最近 20 根里缺 >= {C.SUSP_GAP} 个交易日", "reduce_days": 30, "release_days": 30,
                       "release_min_share": "daily.RELEASE_MIN_SHARE", "placement_days": 60,
                       "events_source": "events_rc.py（corp_events.fetch_ann_market/fetch_rec + akshare 解禁 / 增发表），按公告日 / 解禁日 / 发行日期判"},
        "decision": {
            "primary": "每天前 10 买得到命中按天配对差 对 base_s7",
            "pass": ["筛选集按天差 >= 0 且 开发集按天差 >= 0", "合并 969 天 evalkit v3：NW t >= max(2.5, 族门槛)、块自举 p < 0.05、"
                     "去最好月 NW t >= 2.0、合并与按天同号、滚动三月 >= -5pp、空榜不翻倍、筛选集同号",
                     "前 30 合并按天差和前 10 同号"],
            "auc": "按天 AUC（候选池全体）对 base 的 NW t <= -2.5 的臂：前 10 没过线就直接记淘汰（不记未定）；"
                   "前 10 过线的臂不因 AUC 被淘汰，但报告写明它是尾部规则",
            "strict": f"旧族 {OLD_COUNT} 次 + 本族 {len(ARMS)} 臂 Bonferroni：c_strict = {c_strict(len(ARMS)):.3f}，并列印出",
            "verdict_rule": "过线 = 上面三条全满足；不过线 = 合并 NW t <= -门槛 或 AUC 淘汰；其余未定",
        },
        "secondary": ["三句事实（前 10 的 20 日涨幅中位、20 根内冲高 50% 比例对照全市场、第 20 根收盘收益中位、最大回撤中位），两段分开",
                      "前 10 板块构成、前 10 ret20 中位、零命中天比例、相邻日名单 Jaccard", "规则核每月系数"],
        "controls": {"base": "screen_base_v1 base_s7（筛选集）+ screen_base_v1_dev base_s7（开发集到 12-03）+ parts/2025-12/base_s7"},
    }
    (OUT / "rulecore_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    C.say("预登记 -> %s；族 %s 门槛 %s" % (f, C.FAMILY, K.family_threshold(C.FAMILY)))
    C.progress("预登记 %s：臂 %s，族 %s，严格门槛 %.3f（旧族 %d + %d）" % (C.NAME, list(ARMS), C.FAMILY, c_strict(len(ARMS)), OLD_COUNT, len(ARMS)))


# ---------------------------------------------------------------
#  判定
# ---------------------------------------------------------------
def load_base_all() -> pd.DataFrame:
    a = pd.read_parquet(C.EXP / "screen_base_v1" / "all_base_s7.parquet")
    b = pd.read_parquet(C.EXP / "screen_base_v1_dev" / "all_base_s7.parquet")
    b = b[b["date"] < "2025-12-01"]
    c = pd.read_parquet(OUT / "all_base_dec.parquet")
    o = pd.concat([a, b, c], ignore_index=True)
    o["code"] = o["code"].astype(str).str.zfill(6)
    return o.drop_duplicates(["date", "code"])


def load_arm_all(arm: str) -> pd.DataFrame:
    f = {"rc": "all_rc.parquet", "ret20": "all_ret20.parquet", "pc": "all_pc_s7.parquet"}[arm]
    o = pd.read_parquet(OUT / f)
    o["code"] = o["code"].astype(str).str.zfill(6)
    return o


def seg_of(dates: pd.Series) -> pd.Series:
    return np.where(dates < C.SEGS["dev"][0] + "-01", "screen", "dev")


def nw(d: pd.Series) -> tuple[float, float, float]:
    x = d.to_numpy(float)
    se = K._nw_se(x)
    return float(x.mean()), float(se), float(x.mean() / se) if se > 0 else float("nan")


def jaccard_adjacent(t: pd.DataFrame, k: int = 10) -> float:
    s = t[t["rank"] <= k].groupby("date")["code"].apply(set)
    ds = list(s.index)
    return float(np.mean([len(s[ds[i]] & s[ds[i + 1]]) / len(s[ds[i]] | s[ds[i + 1]]) for i in range(len(ds) - 1)]))


def facts(sub: pd.DataFrame) -> dict:
    return {"n": int(len(sub)), "ret20_med": float(sub["ret20"].median()), "hit": float(sub["y_open"].mean()),
            "r20_med": float(sub["r20_open"].median()), "c20_med": float(sub["c20"].median()),
            "dd20_med": float(sub["dd20"].median()), "dd20_le20": float((sub["dd20"] <= -0.2).mean()),
            "c20_le20": float((sub["c20"] <= -0.2).mean())}


def auc_by_day(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame({"date": df["date"].to_numpy(), "y": df["y_open"].to_numpy(), "p": p})
    d = d[np.isfinite(d["p"]) & np.isfinite(d["y"])]
    out = {}
    for k, g in d.groupby("date"):
        if g["y"].nunique() == 2:
            out[k] = roc_auc_score(g["y"], g["p"])
    return pd.Series(out)


def do_judge() -> None:
    lines: list[str] = []

    def L(s: str = "") -> None:
        C.say(s)
        lines.append(s)

    pre = K.prereg_info(C.NAME)
    assert pre and pre.get("gate_version") == 3, "先 --prereg"
    df = C.build_features()
    lo, hi = C.SEGS["screen"][0] + "-01", C.month_shift(C.SEGS["dev"][1], 1) + "-01"
    df = df[(df["date"] >= lo) & (df["date"] < hi) & np.isfinite(df["y_open"]) & np.isfinite(df["y_up"])].reset_index(drop=True)
    ev = C.event_flags(df)
    elig = df["elig0"] & ~ev["any"]
    df["seg"] = seg_of(df["date"])
    L("候选池：行 %d，天 %d；elig0 %.1f%%；事件剔除 减持 %.2f%% 解禁 %.2f%% 增发 %.2f%%（候选池内）；最终 %.1f%%" % (
        len(df), df["date"].nunique(), 100 * df["elig0"].mean(),
        100 * ev.loc[df["elig0"], "reduce30"].mean(), 100 * ev.loc[df["elig0"], "release30"].mean(),
        100 * ev.loc[df["elig0"], "placement60"].mean(), 100 * elig.mean()))
    L("事件表状态：%s" % json.dumps(ev.attrs.get("status", {}), ensure_ascii=False)[:600])
    base_day = df[elig].groupby("date")["y_open"].mean()
    for s in ("screen", "dev"):
        sub_ = df[elig & (df["seg"] == s)]
        L("  %s 候选池基准（按天）%.3f%%，天数 %d，每天候选 %.0f 只" % (
            s, 100 * sub_.groupby("date")["y_open"].mean().mean(), sub_["date"].nunique(), len(sub_) / max(sub_["date"].nunique(), 1)))

    # 各臂的 p 并到候选池行
    allp = {"base": load_base_all(), "rc": load_arm_all("rc"), "ret20": load_arm_all("ret20")}
    if (OUT / "all_pc_s7.parquet").exists():
        allp["pc"] = load_arm_all("pc")
    tops, P = {}, {}
    for arm, ap in allp.items():
        m = df[["code", "date"]].merge(ap, on=["code", "date"], how="left")
        P[arm] = m["p"].to_numpy(float)
        cov = np.isfinite(P[arm])[elig.to_numpy()].mean()
        tops[arm] = C.rank_table(df, P[arm], elig)
        tops[arm].to_parquet(OUT / f"top_{arm}.parquet", index=False)
        L("臂 %-6s 候选池覆盖 %.2f%%，名次表 %d 行 %d 天" % (arm, 100 * cov, len(tops[arm]), tops[arm]["date"].nunique()))

    b = tops["base"]
    hb10, hb30 = C.daily_hit(b, 10), C.daily_hit(b, 30)
    auc = {arm: auc_by_day(df, P[arm] if arm != "ret20" else np.where(elig, P[arm], np.nan)) for arm in allp}
    n_new = len(ARMS)
    cs = c_strict(n_new)
    results: dict = {"c_strict": cs, "base": {}}
    L("\n== 各臂成绩（每天前 10 买得到命中；候选池同一份）==")
    for arm, t in tops.items():
        h10, h30 = C.daily_hit(t, 10), C.daily_hit(t, 30)
        row = {}
        for s in ("screen", "dev", "all"):
            idx = h10.index if s == "all" else h10.index[seg_of(pd.Series(h10.index)) == s]
            row[s] = {"hit10": float(h10.reindex(idx).mean()), "hit30": float(h30.reindex(idx).mean()),
                      "zero10": float((h10.reindex(idx) == 0).mean()), "days": int(len(idx)),
                      "auc": float(auc[arm].reindex(idx).mean())}
        sub = t[t["rank"] <= 10].merge(df[["code", "date", "ret20", "r20_open", "c20", "dd20", "seg"]], on=["code", "date"])
        row["board"] = {s: sub.loc[sub["seg"] == s, "board"].value_counts(normalize=True).reindex(list(C.BOARDS)).fillna(0).round(3).to_dict()
                        for s in ("screen", "dev")}
        row["facts"] = {s: facts(sub[sub["seg"] == s]) for s in ("screen", "dev")}
        row["jaccard"] = {s: jaccard_adjacent(t[seg_of(t["date"]) == s]) for s in ("screen", "dev")}
        results[arm] = row
        L("%-6s 筛选集 前10 %.2f%% 前30 %.2f%% 零天 %.1f%% AUC %.4f | 开发集 前10 %.2f%% 前30 %.2f%% 零天 %.1f%% AUC %.4f | 合并 前10 %.2f%%（%d 天）" % (
            arm, 100 * row["screen"]["hit10"], 100 * row["screen"]["hit30"], 100 * row["screen"]["zero10"], row["screen"]["auc"],
            100 * row["dev"]["hit10"], 100 * row["dev"]["hit30"], 100 * row["dev"]["zero10"], row["dev"]["auc"],
            100 * row["all"]["hit10"], row["all"]["days"]))
        L("       板块 筛选集 %s | 开发集 %s | 相邻日 Jaccard %.2f / %.2f" % (
            " ".join("%s %.0f%%" % (k, 100 * v) for k, v in row["board"]["screen"].items()),
            " ".join("%s %.0f%%" % (k, 100 * v) for k, v in row["board"]["dev"].items()), row["jaccard"]["screen"], row["jaccard"]["dev"]))

    # 全市场（候选池）三句事实
    mk = {s: facts(df[elig & (df["seg"] == s)]) for s in ("screen", "dev")}
    results["market"] = mk

    L("\n== 对 base 的判定（主尺子 前 10 按天配对；evalkit v3 合并 969 天；严格门槛 %.3f）==" % cs)
    verdicts = {}
    for arm in [a for a in tops if a != "base"]:
        t = tops[arm]
        h10, h30 = C.daily_hit(t, 10), C.daily_hit(t, 30)
        d10 = (h10 - hb10.reindex(h10.index)).dropna()
        d30 = (h30 - hb30.reindex(h30.index)).dropna()
        da = (auc[arm] - auc["base"].reindex(auc[arm].index)).dropna()
        segs = {}
        for s in ("screen", "dev"):
            i = d10.index[seg_of(pd.Series(d10.index)) == s]
            m10, se10, t10 = nw(d10.reindex(i))
            m30, se30, t30 = nw(d30.reindex(i))
            ma, sea, ta = nw(da.reindex(i)) if len(da.reindex(i).dropna()) > 30 else (float("nan"),) * 3
            segs[s] = {"diff10": m10, "se10": se10, "t10": t10, "diff30": m30, "t30": t30, "auc_diff": ma, "auc_t": ta, "days": int(len(i))}
        m30, se30, t30 = nw(d30)
        ma, sea, ta = nw(da)
        scr = {"diff_day": segs["screen"]["diff10"], "days": segs["screen"]["days"], "span": list(K.SCREEN)}
        cmp_ = K.compare(t, b, y="y_open", k=10, family=C.FAMILY, exp=C.NAME, arm=arm, screen=scr)
        two_seg = bool(segs["screen"]["diff10"] >= 0 and segs["dev"]["diff10"] >= 0)
        top30_ok = bool(np.sign(m30) == np.sign(cmp_["diff_day"]))
        auc_kill = bool(np.isfinite(ta) and ta <= -2.5)
        strict_ok = bool(np.isfinite(cmp_["t_nw"]) and cmp_["t_nw"] >= cs)
        v3 = cmp_["verdict_v3"]
        if v3 == "过线" and two_seg and top30_ok:
            verdict = "过线"
        elif v3 == "不过线" or (auc_kill and not (v3 == "过线")):
            verdict = "不过线"
        else:
            verdict = "未定"
        verdicts[arm] = {"verdict": verdict, "v3": v3, "two_segments": two_seg, "top30_same_sign": top30_ok, "auc_eliminated": auc_kill,
                         "strict_pass": strict_ok, "segments": segs, "top30_all": {"diff": m30, "t": t30}, "auc_all": {"diff": ma, "t": ta},
                         "compare": {k_: v for k_, v in cmp_.items() if k_ not in ("t_nw_lags",)}}
        L("\n[%s] %s" % (arm, K.fmt_compare(arm, cmp_)))
        L("  两段 前10：筛选集 %+.2fpp (NW t %.2f, %d 天)  开发集 %+.2fpp (NW t %.2f, %d 天)  -> 两段同号 %s" % (
            100 * segs["screen"]["diff10"], segs["screen"]["t10"], segs["screen"]["days"],
            100 * segs["dev"]["diff10"], segs["dev"]["t10"], segs["dev"]["days"], two_seg))
        L("  前30：筛选集 %+.2fpp (t %.2f) 开发集 %+.2fpp (t %.2f) 合并 %+.2fpp (t %.2f) -> 和前 10 同号 %s" % (
            100 * segs["screen"]["diff30"], segs["screen"]["t30"], 100 * segs["dev"]["diff30"], segs["dev"]["t30"], 100 * m30, t30, top30_ok))
        L("  按天 AUC：筛选集 %+.4f (t %.2f) 开发集 %+.4f (t %.2f) 合并 %+.4f (t %.2f) -> 淘汰 %s" % (
            segs["screen"]["auc_diff"], segs["screen"]["auc_t"], segs["dev"]["auc_diff"], segs["dev"]["auc_t"], ma, ta, auc_kill))
        L("  合并 NW t %.2f 对 新族门槛 %.2f -> %s；对 严格门槛 %.3f -> %s" % (
            cmp_["t_nw"], cmp_["threshold"]["c"], "过" if cmp_["gate_v3"]["t_nw"] else "不过", cs, "过" if strict_ok else "不过"))
        L("  >>> 判定：%s（v3 %s，两段 %s，前30 同号 %s，AUC 淘汰 %s）" % (verdict, v3, two_seg, top30_ok, auc_kill))
    # 规则核对 pc / ret20 的配对（说明用，不登记）
    extra = {}
    for a_, b_ in (("rc", "ret20"), ("rc", "pc"), ("pc", "ret20")):
        if a_ in tops and b_ in tops:
            d = (C.daily_hit(tops[a_]) - C.daily_hit(tops[b_])).dropna()
            mu, se, tt = nw(d)
            extra[f"{a_}-{b_}"] = {"diff": mu, "t": tt, "days": int(len(d))}
            L("  说明：%s − %s 前 10 按天 %+.2fpp NW t %.2f（%d 天）" % (a_, b_, 100 * mu, tt, len(d)))
    results["pairs"] = extra

    L("\n== 三句事实（前 10；买得到口径）==")
    for s in ("screen", "dev"):
        L("[%s] 候选池全市场：冲高 50%% %.2f%%，20 根最高涨幅中位 %.1f%%，第 20 根收盘中位 %+.1f%%，最大回撤中位 %.1f%%" % (
            s, 100 * mk[s]["hit"], 100 * mk[s]["r20_med"], 100 * mk[s]["c20_med"], 100 * mk[s]["dd20_med"]))
        for arm in tops:
            f = results[arm]["facts"][s]
            L("  %-6s 这批票 20 日涨幅中位 %+.1f%%；20 根内冲高 50%% 的 %.1f%%（全市场 %.1f%%）；第 20 根收盘收益中位 %+.1f%%，"
              "最大回撤中位 %.1f%%（回撤 <= -20%% 占 %.0f%%）" % (
                  arm, 100 * f["ret20_med"], 100 * f["hit"], 100 * mk[s]["hit"], 100 * f["c20_med"], 100 * f["dd20_med"], 100 * f["dd20_le20"]))

    # 逐月
    L("\n== 逐月前 10 买得到命中（%）==")
    mo = {arm: C.daily_hit(t).groupby(lambda x: x[:7]).mean() for arm, t in tops.items()}
    mo["候选池"] = base_day.groupby(lambda x: x[:7]).mean()
    L((100 * pd.DataFrame(mo)).round(1).to_string())
    coef = json.loads((OUT / "coef_rc.json").read_text(encoding="utf-8")) if (OUT / "coef_rc.json").exists() else {}
    if coef:
        cf = pd.DataFrame(coef).T
        L("\n== 规则核每月系数（z 刻度；板块相对主板）==")
        L(cf[list(C.INPUTS) + ["b_chinext", "b_star", "b_bj"]].round(3).to_string())
    results["verdicts"] = verdicts
    (C.HERE / "results.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    K.record(C.NAME, {"verdicts": {a: v["verdict"] for a, v in verdicts.items()},
                      "hit10": {a: results[a]["all"]["hit10"] for a in tops}, "c_strict": cs})
    C.progress("判定完成：%s" % {a: v["verdict"] for a, v in verdicts.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    if a.prereg:
        do_prereg()
    if a.judge:
        if C.in_prod_window():
            raise SystemExit("北京 16:00~18:30 不跑")
        do_judge()


if __name__ == "__main__":
    main()
