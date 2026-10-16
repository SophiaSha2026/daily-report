"""
第 5 步：预登记 leak_v1，同一套走向前（evalkit.walk_forward，开发集 2024-06..2025-12）在三版价格上各跑现状和候选。

臂（<名字>_s<种子>，种子 7 主判定，9/10 只报告）
    base_snap   快照训练表的生产特征（只在 check 这一轮，核对重建流程）
    base_<v>    重建的生产特征，v = fwd / pit / ctl
    cand_<v>    base_<v> + Alpha158 142 列 + on/vc/hi 33 列（都是 v 版价格算的），colsample 0.4 = a158_ovh_cs4
    cand_mix    生产特征 fwd + Alpha158 / 三族 pit（拆泄漏在哪一套特征里；事后诊断，不进主判定）
每一轮（--pass check|fwd|pit|ctl|mix）只装那一轮要的列（内存），行集合各轮逐位相同（key.txt 钉住），
标签两臂同一份：y_up 来自 train_snap，y_open 来自 labels_open_snap；每月特征选择用生产缓存 feats_by_month.json。

预登记的判定（看结果前写定，见 prereg.json 的 hypothesis）
    Q1  候选增益里有没有泄漏：G_v = cand_v - base_v（按天前 10 y_open 配对）。Δ = G_fwd - G_ctl（主）、G_fwd - G_pit（副）
        的按天序列 NW t（滞后 20）>= 2.0 且 泄漏占比 L = Δ均值 / G_fwd均值 >= 0.3 -> 「增益有实质部分来自泄漏」；
        NW t < 2.0 且 L < 0.3 -> 「排除」；其余「未定」。候选去留另看 cand_ctl 对 base_ctl 和 cand_pit 对 base_pit 的
        evalkit.compare v2（登记为新试验 cand_ctl_s7 / cand_pit_s7）。
    Q2  生产特征有没有吃到泄漏：base_fwd - base_ctl 按天序列（登记 base_fwd_vs_ctl_s7）NW t >= 2.0 且均值 >= +0.5pp
        -> 「生产特征吃到泄漏，邮件成绩按 base_ctl / base_fwd 的比例改口」；base_fwd - base_pit 并列报（含取整噪声代价）。

    python research/leak/exp_leak.py --fit --pass check          种子 7
    python research/leak/exp_leak.py --fit --pass fwd --seeds 7 9 10
    python research/leak/exp_leak.py --analyze
"""
import argparse
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402  research/features 的尺子和登记簿
import evalkit as K  # noqa: E402

NAME = "leak_v1"
FAMS = ("on", "vc", "hi")
CS = 0.4
PASS_VER = {"check": "fwd", "fwd": "fwd", "pit": "pit", "ctl": "ctl", "mix": "fwd"}
PASS_ALT = {"check": None, "fwd": "fwd", "pit": "pit", "ctl": "ctl", "mix": "pit"}

HYP = (
    "前复权价取整到分的颗粒度带着 T 日之后的除权信息（qfq = round(P_u/q, 2)，q 依赖 T 之后全部除权）。"
    "同一份新浪不复权价造三版：fwd = round(P_u/q,2)（现状）、pit = P_u/q 不取整（无未来信息）、"
    "ctl = round(P_u/k_code,2)/q（同量级取整噪声、和自己的未来无关）。三版上各从零重建生产特征 / Alpha158 / on vc hi，"
    "标签同一份，同一走向前。"
    "Q1 候选 a158_ovh_cs4 的增益有没有泄漏：Δ = (cand_fwd-base_fwd) - (cand_ctl-base_ctl) 按天序列 NW t(20) >= 2.0 且 "
    "L = Δ/G_fwd >= 0.3 为「有实质泄漏」，NW t < 2.0 且 L < 0.3 为「排除」，其余未定；G_fwd-G_pit 并列。"
    "候选本身在干净价格上的成绩按 evalkit v2 判（cand_ctl_s7 对 base_ctl_s7、cand_pit_s7 对 base_pit_s7）。"
    "Q2 生产特征有没有吃到泄漏：base_fwd - base_ctl 按天 NW t >= 2.0 且均值 >= +0.5pp 为「吃到，邮件成绩要改口」；"
    "base_fwd - base_pit 并列（= 泄漏 - 取整噪声代价）。种子 7 判定，9/10 只报告。cand_mix 是事后诊断。"
)
ARMS = {
    "cand_ctl_s7": "候选 a158_ovh_cs4 在 ctl 价格上对同循环 base_ctl（种子 7）",
    "cand_pit_s7": "候选 a158_ovh_cs4 在 pit 价格上对同循环 base_pit（种子 7）",
    "base_fwd_vs_ctl_s7": "生产特征 fwd 对 ctl（种子 7），Q2 主判定",
}


def feats_union() -> list[str]:
    cache = json.loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
    months = K.months_in(K.DEV)
    assert all(m in cache for m in months)
    return sorted({c for m in months for c in cache[m]})


def alt_cols(ver: str) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(CL.LEAK / f"alt_{SFX['alt'] or ver}.parquet").schema.names
    return [c for c in names if c.startswith("a_")], [c for c in names if c.startswith("f_")]


SFX = {"prod": "", "alt": ""}      # --debug 时换成 _test 的小表
DBG = {"name": NAME}


def load(pas: str, seeds) -> pd.DataFrame:
    feats = feats_union()
    rows = CL.snap_rows()
    rows = rows[rows["date"] < K.CONFIRM[0] + "-01"]
    lab = pd.read_parquet(CL.LEAK / "labels_open_snap.parquet", columns=["code", "date", "y_open"])
    rows = rows.merge(lab, on=["code", "date"], how="left")
    ver = PASS_VER[pas]
    prod = pd.read_parquet(CL.LEAK / f"prod_{SFX['prod'] or ver}.parquet", columns=["code", "date"] + feats)
    df = rows.merge(prod, on=["code", "date"], how="inner")
    del prod, rows
    df["y_up"] = df["y_up"].astype("float32")
    key = hashlib.md5((df["code"] + df["date"]).str.cat(sep="|").encode()).hexdigest()
    kf = K.EXP / DBG["name"] / "key.txt"
    kf.parent.mkdir(parents=True, exist_ok=True)
    if kf.exists():
        assert kf.read_text().strip() == f"{len(df)} {key}", "这一轮的行集合和上一轮不同"
    else:
        kf.write_text(f"{len(df)} {key}")
    if pas == "check":
        snap = pd.read_parquet(CL.LEAK / "train_snap.parquet", columns=["code", "date"] + feats,
                               filters=[("date", "<", K.CONFIRM[0] + "-01")])
        snap["code"] = snap["code"].astype(str).str.zfill(6)
        snap["date"] = snap["date"].astype(str).str[:10]
        snap = snap.rename(columns={c: c + "@snap" for c in feats})
        df = df.merge(snap, on=["code", "date"], how="left")
        del snap
    else:
        av = PASS_ALT[pas]
        a = pd.read_parquet(CL.LEAK / f"alt_{SFX['alt'] or av}.parquet")
        df = df.merge(a, on=["code", "date"], how="left")
        del a
    assert df["date"].max() < K.CONFIRM[0]
    print("[%s] 训练表 %d 行 %d 列 %.1f GB" % (pas, len(df), df.shape[1], df.memory_usage().sum() / 2 ** 30), flush=True)
    return df


def make_fit(pas: str, seeds):
    ver = PASS_VER[pas]
    av = PASS_ALT[pas]

    def fit_predict(tr, te, feats, month, state):
        import model as M
        if av is not None:
            ac, fc = state.setdefault("alt", alt_cols(av))
            assert len(ac) == 142 and len(fc) == 33, (len(ac), len(fc))
        out = {}
        for s in seeds:
            trs = M.stratified_sample(tr, "y_up", seed=s)
            ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
            if pas == "check":
                arms = [("base_snap", [f + "@snap" for f in feats], {}), (f"base_{ver}", feats, {})]
            elif pas == "mix":
                arms = [("cand_mix", feats + ac + fc, {"colsample_bytree": CS})]
            else:
                arms = [(f"base_{ver}", feats, {}), (f"cand_{ver}", feats + ac + fc, {"colsample_bytree": CS})]
            for name, cols, kw in arms:
                # 教训 32：臂真的不一样
                assert len(set(cols)) == len(cols) and all(c in tr.columns for c in cols), name
                m = M.L1Lgbm(n_estimators=400, n_jobs=12, **kw).fit(trs, cols, "y_up")
                out[f"{name}_s{s}"] = (m.predict_proba(te), m.predict_proba(ref))
        return out
    return fit_predict


# ---------------------------------------------------------------
#  分析
# ---------------------------------------------------------------
def dd(a, b):
    return C.daily_diff(a, b)


def stat(d: pd.Series) -> str:
    bb = C.hac.mbb(d.to_numpy())
    return "均值 %+.2fpp NW t %.2f (10/30: %.2f/%.2f) 块自举 p %.3f [%+.2f, %+.2f] 去924 %+.2f t %.2f 2025起 %+.2f t %.2f" % (
        100 * d.mean(), C.nwt(d), float(C.hac.nw_t(d.to_numpy(), 10)["t"]), float(C.hac.nw_t(d.to_numpy(), 30)["t"]),
        bb["p"], 100 * bb["lo"], 100 * bb["hi"],
        100 * d[d.index.str[:7] != C.M924].mean(), C.nwt(d[d.index.str[:7] != C.M924]),
        100 * d[d.index >= "2025-01"].mean(), C.nwt(d[d.index >= "2025-01"]))


def analyze(seeds=(7, 9, 10), name: str = NAME, book: bool = True):
    """book=False：调试，不读预登记、不写登记簿。"""
    pr = json.loads((K.EXP / name / "prereg.json").read_text(encoding="utf-8")) if book else {"hypothesis": "(调试)"}
    have = {p.stem[4:] for p in (K.EXP / name).glob("top_*.parquet")}
    T = {a: C.top(name, a) for a in sorted(have)}
    lines = ["臂：%s" % ", ".join(sorted(T)), "假设：%s" % pr["hypothesis"]]
    res = {}
    ex = {"exp": name} if book else {}

    def ledger(*a_, **kw):
        if book:
            C.ledger(*a_, **kw)

    def L(s):
        print(s, flush=True)
        lines.append(s)

    hit = {a: 100 * C.daily_hit(t).mean() for a, t in T.items()}
    L("各臂前 10 买得到口径命中（按天平均）：" + "  ".join("%s %.2f%%" % (a, h) for a, h in sorted(hit.items())))
    pr_ = {a: K.clustered(K.prod_picks(t)) for a, t in T.items()}
    L("各臂生产规则（>=97 分且前 10）命中：" + "  ".join("%s %.2f%% (n %d)" % (a, 100 * p["hit"], p["n"]) for a, p in sorted(pr_.items())))

    # ---- 0 核对：重建流程 ----
    if "base_snap_s7" in T and "base_fwd_s7" in T:
        a, b = T["base_snap_s7"], T["base_fwd_s7"]
        d = dd(b, a)
        ta, tb = a[a["rank"] <= 10], b[b["rank"] <= 10]
        ja = ta.groupby("date")["code"].apply(set)
        jb = tb.groupby("date")["code"].apply(set)
        jac = np.mean([len(ja[k] & jb[k]) / len(ja[k] | jb[k]) for k in ja.index if k in jb.index])
        pa = pd.read_parquet(K.EXP / name / "all_base_snap_s7.parquet")
        pb = pd.read_parquet(K.EXP / name / "all_base_fwd_s7.parquet")
        m = pa.merge(pb, on=["date", "code"], suffixes=("_a", "_b"))
        L("[核对] base_fwd 对 base_snap：%s；前 10 名单日均 Jaccard %.3f；全市场预测值相关 %.4f（%d 行）" % (
            stat(d), jac, float(np.corrcoef(m["p_a"], m["p_b"])[0, 1]), len(m)))
        res["check"] = {"diff": float(d.mean()), "t_nw": C.nwt(d), "jaccard": float(jac),
                        "p_corr": float(np.corrcoef(m["p_a"], m["p_b"])[0, 1])}
        del pa, pb, m

    # ---- Q2 生产特征 ----
    for s in seeds:
        if f"base_fwd_s{s}" not in T:
            continue
        for v in ("ctl", "pit"):
            if f"base_{v}_s{s}" in T:
                d = dd(T[f"base_fwd_s{s}"], T[f"base_{v}_s{s}"])
                L("[Q2 种子 %d] base_fwd - base_%s：%s" % (s, v, stat(d)))
                res[f"q2_fwd_vs_{v}_s{s}"] = {"diff": float(d.mean()), "t_nw": C.nwt(d)}
        if f"base_pit_s{s}" in T and f"base_ctl_s{s}" in T:
            d = dd(T[f"base_pit_s{s}"], T[f"base_ctl_s{s}"])
            L("[Q2 种子 %d] base_pit - base_ctl（取整噪声的代价）：%s" % (s, stat(d)))
    if "base_fwd_s7" in T and "base_ctl_s7" in T:
        v2 = K.compare(T["base_fwd_s7"], T["base_ctl_s7"], arm="base_fwd_vs_ctl_s7" if book else "", **ex)
        res["q2_v2"] = {k: v for k, v in v2.items() if k != "threshold"}
        q2 = "吃到泄漏（要改口）" if (v2["t_nw"] >= 2.0 and v2["diff_day"] >= 0.005) else "没有证据（不改口）"
        L("[Q2 判定] base_fwd 对 base_ctl 按天 %+.2fpp NW t %.2f 块自举 p %.3f -> %s" % (
            100 * v2["diff_day"], v2["t_nw"], v2["boot"]["p"], q2))
        ledger(name, "base_fwd_vs_ctl_s7", "result", t_nw=v2["t_nw"], diff=v2["diff_day"], verdict=q2)
        for a_, b_ in (("base_fwd_s7", "base_ctl_s7"), ("base_fwd_s7", "base_pit_s7")):
            if b_ in T:
                c = C.full_cmp(T[a_], T[b_])
                L(C.fmt(a_, c, b_))

    # ---- Q1 候选 ----
    G = {}
    for s in seeds:
        for v in ("fwd", "pit", "ctl"):
            if f"cand_{v}_s{s}" in T and f"base_{v}_s{s}" in T:
                G[(v, s)] = dd(T[f"cand_{v}_s{s}"], T[f"base_{v}_s{s}"])
                L("[Q1 种子 %d] G_%s = cand_%s - base_%s：%s" % (s, v, v, v, stat(G[(v, s)])))
        for v in ("ctl", "pit"):
            if ("fwd", s) in G and (v, s) in G:
                D = pd.concat([G[("fwd", s)], G[(v, s)]], axis=1).dropna()
                delta = D.iloc[:, 0] - D.iloc[:, 1]
                share = float(delta.mean() / D.iloc[:, 0].mean()) if D.iloc[:, 0].mean() else float("nan")
                L("[Q1 种子 %d] Δ = G_fwd - G_%s：%s；泄漏占比 L = %.2f" % (s, v, stat(delta), share))
                res[f"q1_delta_{v}_s{s}"] = {"delta": float(delta.mean()), "t_nw": C.nwt(delta), "share": share,
                                             "boot_p": C.hac.mbb(delta.to_numpy())["p"]}
    if ("fwd", 7) in G and ("ctl", 7) in G:
        r = res["q1_delta_ctl_s7"]
        q1 = ("有实质泄漏" if (r["t_nw"] >= 2.0 and r["share"] >= 0.3) else
              "排除" if (r["t_nw"] < 2.0 and r["share"] < 0.3) else "未定")
        L("[Q1 判定] Δ(ctl) 均值 %+.2fpp NW t %.2f 占比 %.2f -> %s" % (100 * r["delta"], r["t_nw"], r["share"], q1))
        res["q1_verdict"] = q1
    for v in ("ctl", "pit"):
        if f"cand_{v}_s7" in T:
            v2 = K.compare(T[f"cand_{v}_s7"], T[f"base_{v}_s7"], arm=f"cand_{v}_s7" if book else "", **ex)
            th = v2["threshold"]
            L("[候选 v2 %s] cand_%s_s7 对 base_%s_s7：差 %+.2fpp NW t %.2f 块自举 p %.3f 去最好月(%s) t %.2f 合并 %+.2f "
              "最差三月 %s 去924 %s 空榜 %d/%d 门槛 %.2f（%s）gate=%s -> %s" % (
                  v, v, v, 100 * v2["diff_day"], v2["t_nw"], v2["boot"]["p"], v2["drop_best_month"], v2["t_nw_drop_best"],
                  100 * v2["pooled_diff"], {k: round(100 * x, 1) for k, x in v2["worst3"].items()},
                  {k: round(100 * x, 1) for k, x in v2["worst3_ex_2024_09"].items()}, v2["empty_a"], v2["empty_b"],
                  th["c"], th["method"], v2["gate"], v2["verdict"]))
            res[f"cand_{v}_v2"] = {k: x for k, x in v2.items() if k != "threshold"}
            ledger(name, f"cand_{v}_s7", "result", t_nw=v2["t_nw"], verdict=v2["verdict"], c=th["c"])
            c = C.full_cmp(T[f"cand_{v}_s7"], T[f"base_{v}_s7"])
            L(C.fmt(f"cand_{v}_s7", c, f"base_{v}_s7"))
    if "cand_fwd_s7" in T:
        c = C.full_cmp(T["cand_fwd_s7"], T["base_fwd_s7"])
        L(C.fmt("cand_fwd_s7", c, "base_fwd_s7"))
        res["cand_fwd_cmp"] = {k: x for k, x in c.items() if k != "monthly_diff"}
    # 候选三版互比：cand_fwd 对 cand_ctl / cand_pit；mix 拆分
    for a_, b_ in (("cand_fwd_s7", "cand_ctl_s7"), ("cand_fwd_s7", "cand_pit_s7"), ("cand_mix_s7", "cand_pit_s7"),
                   ("cand_fwd_s7", "cand_mix_s7"), ("cand_pit_s7", "cand_ctl_s7")):
        if a_ in T and b_ in T:
            d = dd(T[a_], T[b_])
            L("[互比] %s - %s：%s" % (a_, b_, stat(d)))
            res[f"{a_}_vs_{b_}"] = {"diff": float(d.mean()), "t_nw": C.nwt(d)}
    # 多种子平均序列
    for v in ("fwd", "pit", "ctl"):
        ks = [k for k in G if k[0] == v]
        if len(ks) >= 2:
            D = pd.concat([G[k] for k in ks], axis=1).dropna()
            L("[多种子] G_%s 各种子 %s，均值 %+.2fpp 种子间标准差 %.2f，平均序列 NW t %.2f" % (
                v, ", ".join("%+.2f" % (100 * x) for x in D.mean()), 100 * D.mean().mean(), 100 * D.mean().std(ddof=1),
                C.nwt(D.mean(axis=1))))
    for v in ("ctl", "pit"):
        ks = [s for s in seeds if ("fwd", s) in G and (v, s) in G]
        if len(ks) >= 2:
            D = pd.concat([(pd.concat([G[("fwd", s)], G[(v, s)]], axis=1).dropna().pipe(lambda x: x.iloc[:, 0] - x.iloc[:, 1]))
                           for s in ks], axis=1).dropna()
            L("[多种子] Δ(fwd-%s) 各种子 %s，均值 %+.2fpp，平均序列 NW t %.2f" % (
                v, ", ".join("%+.2f" % (100 * x) for x in D.mean()), 100 * D.mean().mean(), C.nwt(D.mean(axis=1))))
    rows = {k: C.monthly_row(T[k]) for k in sorted(T)}
    L("月度前 10 命中（%）：")
    L((100 * pd.DataFrame(rows).T).round(1).to_string())
    txt = "\n".join(lines)
    (CL.HERE / ("results_leak.txt" if book else "results_dbg.txt")).write_text(txt, encoding="utf-8")
    if book:
        K.record(name, {"res": res})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--pass", dest="pas", default="", choices=list(PASS_VER) + [""])
    ap.add_argument("--seeds", nargs="*", type=int, default=[7])
    ap.add_argument("--debug", action="store_true", help="150 只的小表、2 个月、不预登记、不进登记簿，只为跑通")
    a = ap.parse_args()
    if a.debug:
        SFX["prod"], SFX["alt"] = "test", "test"
        DBG["name"] = "leak_dbg"
        df = load(a.pas, a.seeds)
        K.walk_forward(df, K.months_in(K.DEV)[:2], make_fit(a.pas, a.seeds), "leak_dbg")
        print({p.name: len(pd.read_parquet(p)) for p in (K.EXP / "leak_dbg").glob("top_*.parquet")})
        if a.analyze:
            analyze(name="leak_dbg", book=False)
        return
    if a.fit:
        assert a.pas
        new = not (K.EXP / NAME / "prereg.json").exists()
        K.prereg(NAME, HYP, ARMS)
        if new:
            for arm, desc in ARMS.items():
                C.ledger(NAME, arm, "prereg", desc=desc)
            C.ledger(NAME, "cand_mix_s7", "diag", desc="生产特征 fwd + Alpha158/三族 pit，拆泄漏在哪一套；不进判定")
            CL.progress("leak_v1 预登记完成：%s" % json.dumps(ARMS, ensure_ascii=False))
        df = load(a.pas, a.seeds)
        K.walk_forward(df, K.months_in(K.DEV), make_fit(a.pas, a.seeds), NAME)
        shutil.copy(K.EXP / NAME / "q.json", K.EXP / NAME / f"q_{a.pas}.json")
        CL.progress("leak_v1 走向前 pass=%s 种子 %s 跑完" % (a.pas, a.seeds))
    if a.analyze:
        analyze()


if __name__ == "__main__":
    main()
