"""
第 2 周（设计文档第 8 节）：筛选集上标签 / 样本的七个臂，对同循环 base 配对，挑 <= 2 个准备登记到开发集。

    python research/screen/w2.py --diag                 软标签诊断（一个月：预测值和 y_open 的当日相关，低于 y_rel4 的不跑）
    python research/screen/w2.py --prereg               写预登记（看结果前）：prereg.json + 本地登记簿
    python research/screen/w2.py --analyze              结果表、板块构成、最大 t 零分布、按预登记规则选臂
    （走向前本身：python research/screen/exp_screen.py --fit --name screen_w2_v1 --arms base eqd evw y_rel4 y_rel2 y_softh y_softz y_rel4_eqd --seeds 7）

不进主登记簿（不调 evalkit.prereg / compare(exp=)）；本地登记簿 research/screen/ledger.jsonl，族 screen_y50_top10。
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import exp_screen as X  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

NAME = "screen_w2_v1"
DIAG = "screen_w2_diag"
DIAG_MONTH = "2023-03"
ARMS7 = ["eqd", "evw", "y_rel4", "y_rel2", "y_softh", "y_softz", "y_rel4_eqd"]
CTRL = "base"
SEED = 7

# ---- 预登记规则（看结果前写定；analyze 原样执行） ----
RULE = {
    "metric": "筛选集 2022-01..2024-05 每天前 10 买得到口径 y_open，臂 - 同循环 base_s7 的按天配对差；NW t（滞后 20）、块自举 p（块 20）",
    "aux": "辅助尺子：前 10 平均 log1p(r20_open)（次日一字记 0）的按天配对差和 NW t",
    "eligible": "按天差 > 0 且 NW t >= 1.0 且 月差序列任意连续 3 月滚动平均 >= -5pp（v3 规则 5 搬到筛选集）",
    "rank": "合格的臂按 NW t 降序；两个臂 NW t 差 < 0.3 视为平局，平局按辅助尺子 lr20 的 NW t",
    "dedupe": "同一想法只取一个：{y_rel4, y_rel2} 最多一个；{y_softh, y_softz} 最多一个；"
              "y_rel4_eqd 只在 eqd 和 y_rel4 的按天差都 > 0 且它的 NW t 比 y_rel4 高 >= 0.5 时才可替代 y_rel4（设计 3.2「都为正才叠加」）",
    "pick": "按上面顺序取前 2 个；不足 2 个就少登记",
    "seed9": "入选的臂加种子 9 复核：种子 9 的按天差也 > 0 才登记；翻号的降级为「未定」，换下一个合格的臂",
    "multiplicity": "七个臂同属本地族，报最大 t 零分布（七条差值序列联合块自举）下最好臂的 p；这个 p > 0.05 时"
                    "「入选」只是筛选顺序，不是证据；真正的判定在开发集 v3",
    "no_count": "本轮七个臂不进主族 breakout_y50_top10；登记到开发集时每个臂各计 1 次",
}
DIAG_RULE = ("软标签诊断（设计 2.2）：在 %s 这一个月上，臂的预测值和 y_open 的当日 Spearman 相关（全市场可打分行，按天平均）"
             "低于 y_rel4 的软标签臂不进主循环。一个月约 23 天，点估计噪声大，同时报配对 se；规则按点估计执行。" % DIAG_MONTH)
HYP = ("设计文档 2.1 / 2.2 / 3.1 / 3.2：训练目标换成当日相对口径（y_rel4 / y_rel2 / 两种软标签）或样本权重按天 / 按事件等权，"
       "在扩展筛选集 2022-01..2024-05 上对同循环 base（生产特征 + LightGBM 400 棵 + 按月等权 + 1:12 按月分层下采样）"
       "的每天前 10 买得到命中是否更高。只用来挑 <= 2 个臂登记到开发集，判定本身在开发集 v3。")


def L(lines, s):
    print(s, flush=True)
    lines.append(s)


# ---------------------------------------------------------------
#  诊断
# ---------------------------------------------------------------
def diag():
    lines = []
    arms = {k: X.ARMS[k] for k in ["base", "y_rel4", "y_softh", "y_softz", "eqd", "evw"]}
    months = [DIAG_MONTH]
    L(lines, DIAG_RULE)
    df = X.load(months, arms)
    ok = X.run(DIAG, arms, [SEED], months, df, stop_in_prod=True)
    if not ok:
        L(lines, "停在生产窗口前")
        return
    lab = pd.read_parquet(CS.SCR / "labels_screen.parquet", columns=["code", "date", "y_open", "r20_open", "yizi1"])
    lab = lab[lab["date"].str[:7] == DIAG_MONTH]
    lab["u"] = X.u_day(lab)
    rows = json.loads((K.EXP / DIAG / "parts" / DIAG_MONTH / "train_rows.json").read_text())
    L(lines, "训练行数：" + "  ".join("%s %d" % (k, v) for k, v in rows.items()))
    res = {}
    for a in arms:
        al = pd.read_parquet(K.EXP / DIAG / f"all_{a}_s{SEED}.parquet").merge(lab, on=["code", "date"], how="inner")
        al = al[np.isfinite(al["y_open"])]
        g = al.groupby("date")
        c_open = g.apply(lambda d: d["p"].corr(d["y_open"], method="spearman"))
        c_u = g.apply(lambda d: d["p"].corr(d["u"], method="spearman"))
        t = pd.read_parquet(K.EXP / DIAG / f"top_{a}_s{SEED}.parquet")
        hit = t[t["rank"] <= 10].groupby("date")["y_open"].mean()
        res[a] = {"c_open": c_open, "c_u": c_u, "hit": hit}
        L(lines, "%-8s 当日 Spearman(p, y_open) 均值 %.4f（se %.4f，%d 天）  Spearman(p, u) %.4f  前 10 命中 %.2f%%" % (
            a, c_open.mean(), c_open.std(ddof=1) / np.sqrt(len(c_open)), len(c_open), c_u.mean(), 100 * hit.mean()))
    verdict = {}
    for a in ["y_softh", "y_softz"]:
        d = (res[a]["c_open"] - res["y_rel4"]["c_open"]).dropna()
        run_it = bool(d.mean() >= 0)
        verdict[a] = run_it
        L(lines, "%s - y_rel4：Spearman(p, y_open) 差 %+.4f（配对 se %.4f，t %.2f）-> %s" % (
            a, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))),
            "进主循环" if run_it else "不进主循环（低于 y_rel4）"))
    (CS.HERE / "results_w2_diag.txt").write_text("\n".join(lines), encoding="utf-8")
    (K.EXP / DIAG / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    CS.progress("第 2 周软标签诊断（%s，一个月）：%s" % (DIAG_MONTH, "；".join(l for l in lines[-2:])))
    return verdict


# ---------------------------------------------------------------
#  预登记
# ---------------------------------------------------------------
def prereg(arms: list[str]):
    import datetime as dt
    f = K.EXP / NAME / "prereg.json"
    if f.exists():
        print("已登记：", f)
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    months = CS.months_between(*K.SCREEN)
    allarms = [CTRL] + arms
    f.write_text(json.dumps({
        "name": NAME, "registered_at": dt.datetime.now().isoformat(timespec="seconds"), "hypothesis": HYP,
        "arms": {a: X.ARMS[a]["desc"] for a in allarms}, "control": f"{CTRL}_s{SEED}", "seeds": [SEED],
        "span": list(K.SCREEN), "family": CS.FAMILY, "train_years": CS.TRAIN_YEARS, "rule": RULE,
        "diag_rule": DIAG_RULE, "gate_version": "screen（不进主族；开发集登记时用 v3）",
        "note": "不进主族 breakout_y50_top10；本地登记簿 research/screen/ledger.jsonl；确认集一行不读",
        "results": []}, ensure_ascii=False, indent=1), encoding="utf-8")
    for a in allarms:
        CS.ledger(NAME, f"{a}_s{SEED}", "prereg", desc=X.ARMS[a]["desc"], span="%s..%s" % (months[0], months[-1]),
                  rule=RULE["eligible"] + "；" + RULE["pick"])
    CS.progress("%s 预登记（看结果前）：对照 %s_s%d，臂 %s；规则：%s；%s；%s" % (
        NAME, CTRL, SEED, arms, RULE["eligible"], RULE["rank"], RULE["dedupe"]))
    print("预登记写入", f)


# ---------------------------------------------------------------
#  分析
# ---------------------------------------------------------------
def board_mix(t: pd.DataFrame, k: int = 10) -> dict:
    v = t[t["rank"] <= k]["board"].value_counts(normalize=True)
    return {b: float(v.get(b, 0.0)) for b in ("main", "chinext", "star", "bj")}


def analyze(name: str = NAME, tag: str = "", book: bool = True):
    lines = []
    have = sorted(p.stem[4:] for p in (K.EXP / name).glob("top_*.parquet"))
    T = {a: X.with_r20(X.top(name, a), tag) for a in have}
    ctrl = f"{CTRL}_s{SEED}"
    assert ctrl in T, have
    L(lines, "实验 %s，臂 %s，对照 %s" % (name, have, ctrl))
    L(lines, "预登记规则：" + json.dumps(RULE, ensure_ascii=False))
    # 全市场基准和板块基准率（和 exp_screen.analyze 同一行集合）
    months = CS.months_between(*K.SCREEN)
    rows = pd.read_parquet(CS.SCR / f"prod_screen{tag}.parquet", columns=["code", "date", "board"],
                           filters=[("date", ">=", months[0] + "-01"), ("date", "<", CS.month_shift(months[-1], 1) + "-01")])
    lab = pd.read_parquet(CS.SCR / f"labels_screen{tag}.parquet", columns=["code", "date", "y_up", "y_open", "r20_open", "yizi1"])
    rows = rows.merge(lab, on=["code", "date"], how="left")
    rows = rows[np.isfinite(rows["y_up"]) & np.isfinite(rows["y_open"])]
    rows["y_rel4"] = X.y_rel(4.0)(rows)
    rows["y_rel2"] = X.y_rel(2.0)(rows)
    bm = rows.groupby("board").agg(n=("y_open", "size"), y_open=("y_open", "mean"), y_rel4=("y_rel4", "mean"), y_rel2=("y_rel2", "mean"))
    bm["行占比"] = bm["n"] / bm["n"].sum()
    for c in ("y_open", "y_rel4", "y_rel2"):
        bm[c + "正样本占比"] = (bm[c] * bm["n"]) / (bm[c] * bm["n"]).sum()
    L(lines, "全市场按板块：基准率和各标签正样本的板块占比（相对标签会不会把正样本推向北交所）")
    L(lines, (100 * bm.drop(columns="n")).round(2).to_string())
    base_day = rows.groupby("date")["y_open"].mean()
    L(lines, "全市场基准（按天平均）%.3f%%，%d 天" % (100 * base_day.mean(), base_day.size))

    # 训练行数（每月平均）
    tr_rows = {}
    for m in months:
        f = K.EXP / name / "parts" / m / "train_rows.json"
        if f.exists():
            for k, v in json.loads(f.read_text()).items():
                tr_rows.setdefault(k, []).append(v)
    if tr_rows:
        L(lines, "训练行数（每月平均）：" + "  ".join("%s %.0f" % (k, np.mean(v)) for k, v in tr_rows.items()))

    # 各臂对照
    B = T[ctrl]
    hb = X.daily_hit(B)
    res = {ctrl: {"hit": float(hb.mean()), "board": board_mix(B), "aux": float(X.daily_hit(B, 10, "lr20").mean())}}
    D = {}
    for a in have:
        if a == ctrl:
            continue
        A = T[a]
        c = K.compare(A, B)                      # 不带 exp/arm：不写主登记簿
        d = (X.daily_hit(A) - hb).dropna().sort_index()
        dl = (X.daily_hit(A, 10, "lr20") - X.daily_hit(B, 10, "lr20")).dropna().sort_index()
        D[a] = d
        md = d.groupby(d.index.str[:7]).mean()
        r3, r3end = K.roll3_min(md)
        jac = float(np.mean([len(x & y) / len(x | y) for x, y in zip(
            A[A["rank"] <= 10].groupby("date")["code"].apply(set).reindex(hb.index).dropna(),
            B[B["rank"] <= 10].groupby("date")["code"].apply(set).reindex(hb.index).dropna())]))
        res[a] = {"hit": float(X.daily_hit(A).mean()), "diff_day": float(d.mean()), "days": int(len(d)),
                  "se_nw": CS.nw_se(d.to_numpy()), "t_nw": K.nw_t(d.to_numpy()), "boot_p": c["boot"]["p"],
                  "t_nw_drop_best": c["t_nw_drop_best"], "pooled_diff": c["pooled_diff"],
                  "roll3_min": r3, "roll3_end": r3end, "worst3": c["worst3"], "wins": c["wins"], "months": c["months"],
                  "aux": float(X.daily_hit(A, 10, "lr20").mean()), "aux_diff": float(dl.mean()),
                  "aux_t_nw": K.nw_t(dl.to_numpy()), "board": board_mix(A), "jaccard": jac,
                  "monthly_diff": {k: float(v) for k, v in md.items()}}
    # 表
    L(lines, "")
    L(lines, "%-12s %7s %8s %7s %7s %8s %8s %8s %8s %6s  %s" % (
        "臂", "前10%", "差pp", "NW t", "boot p", "去最好月t", "lr20差", "lr20 t", "滚3月min", "赢月", "板块 main/chinext/star/bj %  Jac"))
    b = res[ctrl]["board"]
    L(lines, "%-12s %7.2f %8s %7s %7s %8s %8s %8s %8s %6s  %s" % (
        ctrl, 100 * res[ctrl]["hit"], "-", "-", "-", "-", "-", "-", "-", "-",
        "%.0f/%.0f/%.0f/%.0f" % tuple(100 * b[k] for k in ("main", "chinext", "star", "bj"))))
    for a in have:
        if a == ctrl:
            continue
        r = res[a]
        b = r["board"]
        L(lines, "%-12s %7.2f %+8.2f %7.2f %7.3f %8.2f %+8.4f %8.2f %+8.1f %3d/%-2d  %s  %.2f" % (
            a, 100 * r["hit"], 100 * r["diff_day"], r["t_nw"], r["boot_p"], r["t_nw_drop_best"], r["aux_diff"], r["aux_t_nw"],
            100 * r["roll3_min"], r["wins"], r["months"],
            "%.0f/%.0f/%.0f/%.0f" % tuple(100 * b[k] for k in ("main", "chinext", "star", "bj")), r["jaccard"]))
    L(lines, "（滚 3 月 min 的窗口末月：%s）" % "  ".join("%s %s" % (a, res[a]["roll3_end"]) for a in res if a != ctrl))
    # 月度差表
    md_tab = pd.DataFrame({a: res[a]["monthly_diff"] for a in res if a != ctrl})
    L(lines, "月度配对差（pp）：")
    L(lines, (100 * md_tab).round(1).to_string())

    # 最大 t 零分布（七条序列联合块自举）
    arms_s = [a for a in have if a != ctrl]
    if len(arms_s) >= 2:
        Dm = pd.DataFrame(D)
        null = K._maxt_null(Dm)
        best = max(arms_s, key=lambda a: res[a]["t_nw"])
        p_best = float((null >= res[best]["t_nw"]).mean())
        c95 = float(np.quantile(null, 0.95))
        L(lines, "[多重检验] %d 条差值序列的最大 t 零分布：95%% 分位 %.2f；最好的臂 %s NW t %.2f 的族 p = %.3f" % (
            len(arms_s), c95, best, res[best]["t_nw"], p_best))
        for a in arms_s:
            res[a]["p_family"] = float((null >= res[a]["t_nw"]).mean())
        res["maxt_null_q95"] = c95
        L(lines, "  各臂族 p：" + "  ".join("%s %.3f" % (a, res[a]["p_family"]) for a in arms_s))

    # 按预登记规则选
    def arm_of(k):
        return k.rsplit("_s", 1)[0]
    elig = [a for a in arms_s if res[a]["diff_day"] > 0 and res[a]["t_nw"] >= 1.0 and res[a]["roll3_min"] >= -0.05]
    elig.sort(key=lambda a: (-res[a]["t_nw"], -res[a]["aux_t_nw"]))
    # 平局（t 差 < 0.3）按 lr20 t
    ordered = []
    pool = list(elig)
    while pool:
        top_t = res[pool[0]]["t_nw"]
        tie = [a for a in pool if top_t - res[a]["t_nw"] < 0.3]
        tie.sort(key=lambda a: -res[a]["aux_t_nw"])
        ordered.append(tie[0])
        pool.remove(tie[0])
    picked, used = [], set()
    g_of = {"y_rel4": "rel", "y_rel2": "rel", "y_softh": "soft", "y_softz": "soft", "y_rel4_eqd": "rel", "eqd": "eqd", "evw": "evw"}
    for a in ordered:
        nm = arm_of(a)
        if nm == "y_rel4_eqd":
            e, r4 = f"eqd_s{SEED}", f"y_rel4_s{SEED}"
            cond = (e in res and r4 in res and res[e]["diff_day"] > 0 and res[r4]["diff_day"] > 0
                    and res[a]["t_nw"] - res[r4]["t_nw"] >= 0.5)
            if not cond:
                continue
        g = g_of.get(nm, nm)
        if g in used:
            continue
        used.add(g)
        picked.append(a)
        if len(picked) == 2:
            break
    L(lines, "[按预登记规则] 合格：%s；排序：%s；入选（<= 2）：%s" % (elig, ordered, picked))
    for a in (arms_s if book else []):
        r = res[a]
        CS.ledger(name, a, "result", hit_day=r["hit"], diff_day=r["diff_day"], t_nw=r["t_nw"], boot_p=r["boot_p"],
                  aux_diff=r["aux_diff"], aux_t_nw=r["aux_t_nw"], roll3_min=r["roll3_min"], p_family=r.get("p_family"),
                  eligible=a in elig, picked=a in picked, board=r["board"])
    if book:
        CS.ledger(name, ctrl, "result", hit_day=res[ctrl]["hit"], board=res[ctrl]["board"])
        f = K.EXP / name / "prereg.json"
        j = json.loads(f.read_text(encoding="utf-8"))
        import datetime as dt
        j["results"].append({"at": dt.datetime.now().isoformat(timespec="seconds"), "eligible": elig, "ordered": ordered,
                             "picked": picked, **res})
        f.write_text(json.dumps(j, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # 给 evalkit.prereg(screen=...) 的草稿（不调用）
    draft = {arm_of(a): {"diff_day": res[a]["diff_day"], "days": res[a]["days"], "span": list(K.SCREEN)} for a in picked}
    L(lines, "evalkit.prereg(screen=...) 草稿（不调用）：" + json.dumps(draft, ensure_ascii=False))
    (CS.HERE / "results_w2.txt").write_text("\n".join(lines), encoding="utf-8")
    (K.EXP / name / "screen_draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=1), encoding="utf-8")
    return res, picked


def seed9(arms: list[str], name: str = NAME):
    """入选臂的种子 9 复核：<臂>_s9 对 base_s9 的按天差同号才登记；两种子平均的差也报。"""
    lines = []
    T = {}
    for a in arms + [CTRL]:
        for sd in (7, 9):
            k = f"{a}_s{sd}"
            if (K.EXP / name / f"top_{k}.parquet").exists():
                T[k] = X.with_r20(X.top(name, k))
    for a in arms:
        out = {}
        for sd in (7, 9):
            k, c = f"{a}_s{sd}", f"{CTRL}_s{sd}"
            if k not in T or c not in T:
                continue
            d = (X.daily_hit(T[k]) - X.daily_hit(T[c])).dropna().sort_index()
            dl = (X.daily_hit(T[k], 10, "lr20") - X.daily_hit(T[c], 10, "lr20")).dropna().sort_index()
            out[sd] = {"diff_day": float(d.mean()), "t_nw": K.nw_t(d.to_numpy()), "days": int(len(d)),
                       "aux_diff": float(dl.mean()), "aux_t_nw": K.nw_t(dl.to_numpy()), "hit": float(X.daily_hit(T[k]).mean())}
        if 7 in out and 9 in out:
            d7 = (X.daily_hit(T[f"{a}_s7"]) - X.daily_hit(T[f"{CTRL}_s7"])).dropna()
            d9 = (X.daily_hit(T[f"{a}_s9"]) - X.daily_hit(T[f"{CTRL}_s9"])).dropna()
            davg = ((d7 + d9) / 2).dropna().sort_index()
            out["avg"] = {"diff_day": float(davg.mean()), "t_nw": K.nw_t(davg.to_numpy())}
            ok = out[9]["diff_day"] > 0
            L(lines, "%-11s s7 %+.2fpp (t %.2f)  s9 %+.2fpp (t %.2f, lr20 %+.4f t %.2f)  两种子平均 %+.2fpp (t %.2f) -> 种子 9 %s" % (
                a, 100 * out[7]["diff_day"], out[7]["t_nw"], 100 * out[9]["diff_day"], out[9]["t_nw"], out[9]["aux_diff"],
                out[9]["aux_t_nw"], 100 * out["avg"]["diff_day"], out["avg"]["t_nw"], "同号，可登记" if ok else "翻号，降级为未定"))
            CS.ledger(name, f"{a}_s9", "result", **{k_: v for k_, v in out[9].items()}, same_sign=bool(ok),
                      avg_diff_day=out["avg"]["diff_day"], avg_t_nw=out["avg"]["t_nw"])
        else:
            L(lines, "%s：缺种子 9 或对照（有 %s）" % (a, [k for k in T if k.startswith(a)]))
    (CS.HERE / "results_w2_seed9.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="*", default=ARMS7)
    ap.add_argument("--debug", action="store_true", help="在 screen_dbg（伪造表）上试 analyze，不登记")
    ap.add_argument("--seed9", nargs="*", help="入选臂的种子 9 复核（先用 exp_screen --fit --seeds 9 跑 base + 这些臂）")
    a = ap.parse_args()
    if a.debug:
        analyze("screen_dbg", tag="_fake", book=False)
        return
    if a.diag:
        diag()
    if a.prereg:
        prereg(a.arms)
    if a.analyze:
        analyze()
    if a.seed9:
        seed9(a.seed9)


if __name__ == "__main__":
    main()
