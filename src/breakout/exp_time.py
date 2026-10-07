"""
实验 15：时间维度。训练窗口 / 时间权重 / 多窗口集合 / 按真实结果滚动校准 / 环境门控。

用户 2026-10-06 的提议原文（节选）：「你不能永远拿所有过去的训练数据去做后续的预测，
而是拿某段时间的训练数据去预测它接下来的结果，然后根据实际上的结果来进行调整……
这个时间段的长短本身都可以成为变量或者特征。」

现状和提议的差别
----------------
走向前（validate.walk_forward）已经是逐月重拟合、只用该月之前的数据；生产每 30 天重训。
但三处和提议不同，正是这里要测的：

  1. 训练集是**扩张窗口**（能看到的全部历史），没有只用「某一段」。
  2. 没有用真实结果回头校准（板块系数 09-16 用验证期命中率估过一次，之后不动）。
  3. 窗口长短不是变量。

另外一个此前没人量化过的事实（2026-10-06 实测，2025-03 那个月的训练样本）：
按月分层下采样后 2024-09 一个月占样本 18%、占**正样本 46%**（32,010 / 69,252），
2024-09 + 2024-10 合起来占样本 35%。「按月分层」只对负样本充裕的月份把正负比压到
1:12，924 行情那两个月负样本本来就不够，整月全进，正负比 1:2。所以模型的损失函数
里将近一半的「起涨长什么样」来自同一段行情。时间权重直接针对这个。

臂（都在同一份特征、同一份下采样样本、同一条净化线上，差别只在训练窗口 / 权重）
----
    base     扩张窗口，等权（= 现状，对照）
    hl3/6/12 指数时间衰减，半衰期 63 / 126 / 252 个交易日（权重 = 0.5^(距截止日的交易日数/半衰期)）
    eqm      每个月的权重总和相等（抹平 924 的月样本量碾压，不偏向新旧）
    roll3/6/12  只用截止日前 63 / 126 / 252 个交易日（用户原话那种「某一段」）
    boost3   base 模型再在最近 63 个交易日上续训 100 棵树（lr 0.02），「一路训练」的直译
  事后不重训就能算的：
    ens_win  base + roll12 + roll6 三个模型预测值取均值 —— 「窗口长短是变量」的可操作形式
    ens_dec  base + hl6 + hl12 均值

分数刻度：每个臂每个月的 0~100 分都用**同一份参照样本**（base 的下采样训练样本）上的
预测值分位点定，和 daily.load_or_fit 对 base 的构造完全一致；对其他臂的含义是
「比参照样本里 96% 的行高」，所有臂同一把尺子。集合臂的刻度用参照样本的 5 万行子集算。

评估（验证集 2025-03..12，封存的 2026-01..09 不碰）
----
    prod   生产规则：≥97 分且当天前 10（剔 ST 后补位，validate.pick 同款，板块校正 none / shrink 两种）
    top10  每天固定前 10，不设门槛（样本 2070，噪声小一半，主要用来比臂）
    top30  每天前 30
    ap     每月全市场的平均精度（average precision），最有统计力，但评的是整条排序不是头部
  标准误一律按天聚类（教训 33）：p = Σk_d/Σn_d，var = Σ(k_d − p·n_d)² / (Σn_d)²。
  臂之间的比较按天配对：d_t = hit_t(臂) − hit_t(base) 的均值 / 标准误。

事后分析
----
  门控：把每天按信号分三档，看 prod / top10 命中率随信号怎么变。信号都是 t 日收盘后
        已知的：adv_share / limit_up / turnover_chg5 / ret_median（regime.compute 同款）、
        n_q97（当天 ≥97 分的只数，模型自己的广度）、base20_lag（t−20 那天的全市场 20 根
        50% 基准率，窗口在 t 已经关闭，可知）、prec_lag（截至 t−20 已关闭窗口的 prod 名额
        近 60 个交易日的实际命中率）。三分位的切点是全样本算的（样本内），只看单调性；
        另给一条诚实规则：信号 ≥ 它到 t−1 为止的扩张中位数才出清单。
  校准：每个月从 {95..99} 里选分数线，只用该月之前**窗口已关闭**（日期 < 净化线）的
        prod 名额：取最低的 s 使「≥s 分名额的实际命中率 ≥ 15% 且 n ≥ 30」，都不满足取 99。

产物 data/breakout/raw/exp_time/（gitignore）：preds.parquet（宽表，一列一个臂）、
q.json、ref_sub.parquet、summary.json。结论写 docs/breakout_log.md 实验 15。

结果：eqm 上线（2026-10-06）。model.L1Lgbm.fit 默认按月等权（model.DEFAULT_WEIGHTING），
这里的 WLgbm 自己接权重、绕过那个默认，所以各臂还是上面定义的样子，base 仍是等权对照。

    python src/breakout/exp_time.py --fit        训练 9 个臂 × 10 个月（约 20 分钟）
    python src/breakout/exp_time.py --analyze    读 preds.parquet 出报告
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

import arena as A       # noqa: E402
import board_adj as BA  # noqa: E402
import daily as D       # noqa: E402
import fselect as FS    # noqa: E402
import model as M       # noqa: E402
import validate as V    # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("exp_time")
OUT = ROOT / "data" / "breakout" / "raw" / "exp_time"
N_JOBS = 12
HALF_LIVES = {"hl3": 63, "hl6": 126, "hl12": 252}
ROLLS = {"roll3": 63, "roll6": 126, "roll12": 252}
BOOST_DAYS, BOOST_TREES, BOOST_LR = 63, 100, 0.02
FIT_ARMS = ["base", "boost3", "hl3", "hl6", "hl12", "eqm", "roll3", "roll6", "roll12"]
ENSEMBLES = {"ens_win": ["base", "roll12", "roll6"],
             "ens_dec": ["base", "hl6", "hl12"]}
REF_SUB = 50_000
CALIB_TARGET, CALIB_MIN_N = 0.15, 30


# ---------------------------------------------------------------
#  拟合
# ---------------------------------------------------------------
class WLgbm(M.L1Lgbm):
    """L1Lgbm 加样本权重和续训。生产的 model.py 不动，实验自己带。"""

    def fit(self, df, cols, y, w=None, init=None):
        import lightgbm as lgb
        self.cols = cols
        X, yy = M._xy(df, cols, y)
        self.m = lgb.LGBMClassifier(**self.p)
        kw = {}
        if w is not None:
            kw["sample_weight"] = np.asarray(w, dtype=np.float64)
        if init is not None:
            kw["init_model"] = init
        self.m.fit(X, yy, **kw)
        return self


def _weights_decay(age: np.ndarray, hl: int) -> np.ndarray:
    w = np.power(0.5, age / float(hl))
    return w / w.mean()


def _weights_eqm(month: pd.Series) -> np.ndarray:
    cnt = month.map(month.value_counts())
    w = 1.0 / cnt.to_numpy(float)
    return w / w.mean()


def fit_all(months: list[str] | None = None) -> None:
    t_all = time.time()
    df = A.load(False)
    feats_all = [c for c in df.columns if "__" in c]
    # 特征选择和 exp_window 同一条净化线（S21），所有臂共用这一份
    feats = FS.run(V.train_slice(df, V.TRAIN_END[:7]), feats_all, y="y_up")["keep"]
    print("特征 %d 列" % len(feats), flush=True)
    dates_all = sorted(df["date"].unique())
    didx = {d: i for i, d in enumerate(dates_all)}

    months = months or sorted({d[:7] for d in df["date"]
                               if V.TRAIN_END <= d < V.VALID_END})
    preds, qmap, refs, comp = [], {}, [], {}
    for m in months:
        t0 = time.time()
        cut = V.purge_cut(df, m)
        tr = V.train_slice(df, m)
        tr = tr[np.isfinite(tr["y_up"])]
        te = df[df["date"].str[:7] == m].copy()
        te = te[np.isfinite(te["y_up"])]
        if len(tr) < 5000 or not len(te):
            continue
        trs = M.stratified_sample(tr, "y_up")
        # --- 自检：净化线、测试月、样本 ---
        assert str(trs["date"].max()) < cut, "训练样本越过净化线"
        assert (te["date"].str[:7] == m).all(), "测试集混进别的月"
        age = (didx[cut] - trs["date"].map(didx)).to_numpy(float)
        assert (age > 0).all(), "训练样本的日龄必须为正"
        mon = trs["date"].str[:7]
        comp[m] = {"cut": cut, "n": int(len(trs)), "pos": int(trs["y_up"].sum()),
                   "share_2024_09": float((mon == "2024-09").mean()),
                   "pos_share_2024_09": float(trs.loc[mon == "2024-09", "y_up"].sum()
                                              / max(trs["y_up"].sum(), 1))}
        rng = np.random.default_rng(11)
        ref_i = rng.choice(len(trs), size=min(REF_SUB, len(trs)), replace=False)

        out = te[["date", "code", "board", "y_up"]].copy()
        qm, refm = {}, {}

        def run(name, sub=None, w=None, init=None, trees=None, lr=None):
            kw = {"n_estimators": 400, "n_jobs": N_JOBS}
            if trees:
                kw["n_estimators"] = trees
            if lr:
                kw["learning_rate"] = lr
            d_ = trs if sub is None else trs[sub]
            if d_["y_up"].sum() < 30:
                print("  %s %s 正样本不足，跳过" % (m, name), flush=True)
                return None
            mdl = WLgbm(**kw).fit(d_, feats, "y_up",
                                  w=None if w is None else (w if sub is None else w[sub]),
                                  init=init)
            pr = mdl.predict_proba(trs)          # 参照样本 = base 的完整样本
            qm[name] = [float(x) for x in np.quantile(pr, np.linspace(0, 1, 101))]
            refm[name] = pr[ref_i].astype(np.float32)
            out[name] = mdl.predict_proba(te).astype(np.float32)
            return mdl

        base = run("base")
        # 自检：全 1 权重和不传权重结果一致（权重通路没接错）
        chk = WLgbm(n_estimators=20, n_jobs=N_JOBS).fit(trs.iloc[:20000], feats, "y_up",
                                                        w=np.ones(20000))
        chk0 = WLgbm(n_estimators=20, n_jobs=N_JOBS).fit(trs.iloc[:20000], feats, "y_up")
        assert np.allclose(chk.predict_proba(te.iloc[:500]),
                           chk0.predict_proba(te.iloc[:500]), atol=1e-6), "权重通路有问题"

        recent = (age <= BOOST_DAYS)
        run("boost3", sub=recent, init=base.m.booster_, trees=BOOST_TREES, lr=BOOST_LR)
        for name, hl in HALF_LIVES.items():
            w = _weights_decay(age, hl)
            assert np.corrcoef(w, -age)[0, 1] > 0, "衰减权重必须随日龄递减"
            run(name, w=w)
        run("eqm", w=_weights_eqm(mon))
        for name, nd in ROLLS.items():
            run(name, sub=(age <= nd))

        qmap[m] = qm
        refs.append(pd.DataFrame(refm).assign(month=m))
        preds.append(out)
        print("  %s 完成 %.0fs（训练样本 %d，2024-09 占正样本 %.0f%%）"
              % (m, time.time() - t0, len(trs), 100 * comp[m]["pos_share_2024_09"]),
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "preds.parquet", index=False)
    pd.concat(refs, ignore_index=True).to_parquet(OUT / "ref_sub.parquet", index=False)
    (OUT / "q.json").write_text(json.dumps(qmap), encoding="utf-8")
    (OUT / "composition.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("全部完成 %.0f 分钟 -> %s" % ((time.time() - t_all) / 60, OUT), flush=True)


# ---------------------------------------------------------------
#  评估
# ---------------------------------------------------------------
def clustered(picks: pd.DataFrame, y: str = "y_up") -> dict:
    """按天聚类的合并命中率和标准误（教训 33）。"""
    if not len(picks):
        return {"n": 0, "hit": float("nan"), "se": float("nan"), "days": 0}
    g = picks.groupby("date")[y].agg(["sum", "size"])
    N, K = float(g["size"].sum()), float(g["sum"].sum())
    p = K / N
    var = float(((g["sum"] - p * g["size"]) ** 2).sum()) / N ** 2
    return {"n": int(N), "hit": p, "se": float(np.sqrt(var)), "days": int(len(g))}


def paired_vs(a: pd.DataFrame, b: pd.DataFrame, y: str = "y_up") -> dict:
    """按天配对：每天命中率之差的均值 / 标准误。只算两边都有名额的天。"""
    d = (a.groupby("date")[y].mean() - b.groupby("date")[y].mean()).dropna()
    if len(d) < 2:
        return {"diff": float("nan"), "se": float("nan"), "t": float("nan"), "days": int(len(d))}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"diff": float(d.mean()), "se": se,
            "t": float(d.mean() / se) if se > 0 else float("nan"), "days": int(len(d))}


def score_of(p: np.ndarray, q: list) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(q, float), p), 0, 100)


def arm_table(df: pd.DataFrame, arm: str, qmap: dict, st: set[str],
              adj_mode: str = "none", score_min: int = D.SCORE_MIN,
              cap: int = D.CAP_A) -> pd.DataFrame:
    """一个臂在验证集上的逐日名次 / 分数表（剔 ST 后按 _p 排名）。

    adj_mode=shrink 时板块系数只用本月之前该臂自己的 prod 名额估（board_adj.factors_before），
    和 exp_window.build_cache 的 shrink 臂同一套。
    """
    d = df[["date", "code", "board", "y_up", arm]].rename(columns={arm: "_p0"}).copy()
    d = d[~d["code"].astype(str).isin(st)]
    d["month"] = d["date"].str[:7]
    parts, hist = [], []
    for m, g in d.groupby("month", sort=True):
        q = qmap[m][arm]
        if adj_mode == "shrink":
            amap = BA.factors_before(pd.concat(hist, ignore_index=True), m) if hist else {}
        else:
            amap = {}
        adj = g["board"].map(amap).fillna(1.0).to_numpy(float) if amap else 1.0
        g = g.assign(_p=g["_p0"].to_numpy(float) * adj)
        g["score"] = score_of(g["_p"].to_numpy(), q)
        g = g.sort_values(["date", "_p"], ascending=[True, False])
        g["rank"] = g.groupby("date").cumcount() + 1
        parts.append(g)
        prod = g[(g["score"] >= score_min) & (g["rank"] <= cap)]
        hist.append(prod[["date", "board", "y_up"]])
    return pd.concat(parts, ignore_index=True)


def prod_picks(t: pd.DataFrame, score_min: int = D.SCORE_MIN,
               cap: int = D.CAP_A) -> pd.DataFrame:
    return t[(t["score"] >= score_min) & (t["rank"] <= cap)]


def ap_by_month(df: pd.DataFrame, arm: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for m, g in df.groupby(df["date"].str[:7]):
        y = g["y_up"].to_numpy()
        if y.sum() == 0:
            continue
        out[m] = {"ap": float(average_precision_score(y, g[arm])),
                  "auc": float(roc_auc_score(y, g[arm])),
                  "base": float(y.mean())}
    return out


def add_ensembles(df: pd.DataFrame, qmap: dict, ref: pd.DataFrame) -> list[str]:
    names = []
    for name, parts in ENSEMBLES.items():
        if not all(p in df.columns for p in parts):
            continue
        df[name] = df[parts].mean(axis=1).astype(np.float32)
        for m, g in ref.groupby("month"):
            pr = g[parts].mean(axis=1).to_numpy()
            qmap[m][name] = [float(x) for x in np.quantile(pr, np.linspace(0, 1, 101))]
        names.append(name)
    return names


def regime_signals(dates: list[str]) -> pd.DataFrame:
    """t 日收盘后已知的环境信号 + 滞后 20 根的全市场基准率。"""
    import regime as R
    px = pd.read_parquet(R.DATA / "daily.parquet", columns=["code", "date"])
    all_dates = sorted(px["date"].astype(str).unique())
    need = len(all_dates) - all_dates.index(min(dates)) + 25
    rows = R.compute(days=need)
    r = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # base20 在 s+20 根才可知；t 日能看到的最新一份是 s = t−20
    r["base20_lag"] = r["base20"].shift(R.UP_WINDOW)
    return r[["date", "adv_share", "limit_up", "turnover_chg5", "ret_median_pct",
              "base20", "base20_lag"]]


def gate_report(t: pd.DataFrame, sig: pd.DataFrame, didx: dict) -> dict:
    """按信号三分位看命中率；再给一条「信号 ≥ 截至 t−1 的扩张中位数」的诚实规则。"""
    prod = prod_picks(t)
    top10 = t[t["rank"] <= 10]
    days = pd.DataFrame({"date": sorted(t["date"].unique())})
    days["n_q97"] = days["date"].map(t[t["score"] >= D.SCORE_MIN].groupby("date").size()).fillna(0)
    days = days.merge(sig, on="date", how="left")
    # 截至 t−20 已关闭窗口的 prod 名额、近 60 个交易日的实际命中率
    pi = prod.assign(i=prod["date"].map(didx))
    di = days["date"].map(didx).to_numpy()
    pl = []
    for i in di:
        w = pi[(pi["i"] <= i - 20) & (pi["i"] > i - 20 - 60)]
        pl.append(float(w["y_up"].mean()) if len(w) >= 10 else np.nan)
    days["prec_lag"] = pl
    out = {}
    for s in ("n_q97", "adv_share", "limit_up", "turnover_chg5", "ret_median_pct",
              "base20_lag", "prec_lag"):
        v = days[["date", s]].dropna()
        if len(v) < 30:
            continue
        qs = v[s].quantile([1 / 3, 2 / 3]).to_numpy()
        v = v.assign(b=np.searchsorted(qs, v[s].to_numpy(), side="right"))
        rows = []
        for b in (0, 1, 2):
            dd = set(v.loc[v["b"] == b, "date"])
            rows.append({"bucket": ["低", "中", "高"][b], "days": len(dd),
                         "cut": [float(x) for x in qs],
                         "prod": clustered(prod[prod["date"].isin(dd)]),
                         "top10": clustered(top10[top10["date"].isin(dd)])})
        # 诚实规则：扩张中位数（只用 t−1 及以前）
        med = v[s].expanding().median().shift(1)
        ok = set(v.loc[v[s] >= med, "date"])
        out[s] = {"terciles": rows,
                  "rule_ge_expanding_median": {
                      "prod": clustered(prod[prod["date"].isin(ok)]),
                      "top10": clustered(top10[top10["date"].isin(ok)]),
                      "days_on": len(ok), "days_total": int(len(v))}}
    return out


def calib_report(t: pd.DataFrame, df_dates: list[str]) -> dict:
    """每月从 {95..99} 选分数线，只看该月之前窗口已关闭的 prod 名额。"""
    months = sorted(t["date"].str[:7].unique())
    cand = [95, 96, 97, 98, 99]
    chosen, picks = {}, []
    for m in months:
        cut = V.purge_cut(pd.DataFrame({"date": df_dates}), m)
        past = t[(t["date"] < cut) & (t["rank"] <= D.CAP_A)]
        s_pick = 99
        for s in cand:
            pp = past[past["score"] >= s]
            if len(pp) >= CALIB_MIN_N and pp["y_up"].mean() >= CALIB_TARGET:
                s_pick = s
                break
        chosen[m] = s_pick
        g = t[t["date"].str[:7] == m]
        picks.append(prod_picks(g, score_min=s_pick))
    cal = pd.concat(picks, ignore_index=True)
    fixed = {s: clustered(prod_picks(t, score_min=s)) for s in cand}
    return {"chosen": chosen, "calib": clustered(cal),
            "fixed_in_sample": {str(k): v for k, v in fixed.items()}}


def analyze() -> dict:
    df = pd.read_parquet(OUT / "preds.parquet")
    ref = pd.read_parquet(OUT / "ref_sub.parquet")
    qmap = json.loads((OUT / "q.json").read_text(encoding="utf-8"))
    comp = json.loads((OUT / "composition.json").read_text(encoding="utf-8"))
    arms = [a for a in FIT_ARMS if a in df.columns]
    arms += add_ensembles(df, qmap, ref)
    st = V.st_codes()
    base_rate = float(df[~df["code"].astype(str).isin(st)]["y_up"].mean())
    all_dates = sorted(df["date"].unique())
    didx = {d: i for i, d in enumerate(all_dates)}
    print("验证集 %d 个交易日，全市场基准 %.2f%%（剔 ST）\n" % (len(all_dates), 100 * base_rate))
    print("训练样本构成（2024-09 占正样本）: " + ", ".join(
        "%s %.0f%%" % (m[2:], 100 * c["pos_share_2024_09"]) for m, c in comp.items()))

    res = {"base_rate": base_rate, "days": len(all_dates), "arms": {}}
    tabs = {}
    for a in arms:
        t = arm_table(df, a, qmap, st, "none")
        tabs[a] = t
        r = {"prod": clustered(prod_picks(t)),
             "prod_empty_days": len(all_dates) - prod_picks(t)["date"].nunique(),
             "top10": clustered(t[t["rank"] <= 10]),
             "top30": clustered(t[t["rank"] <= 30]),
             "ap": ap_by_month(df, a)}
        r["ap_mean"] = float(np.mean([x["ap"] for x in r["ap"].values()]))
        r["auc_mean"] = float(np.mean([x["auc"] for x in r["ap"].values()]))
        ts = arm_table(df, a, qmap, st, "shrink")
        r["prod_shrink"] = clustered(prod_picks(ts))
        res["arms"][a] = r
    for a in arms:
        t, b = tabs[a], tabs["base"]
        res["arms"][a]["vs_base_top10"] = paired_vs(t[t["rank"] <= 10], b[b["rank"] <= 10])
        res["arms"][a]["vs_base_prod"] = paired_vs(prod_picks(t), prod_picks(b))

    hdr = "%-8s %5s %7s %6s %5s | %7s %6s %6s | %7s | %7s %6s | %6s %6s" % (
        "臂", "n", "命中%", "±", "空天", "前10%", "±", "配对t", "前30%", "AP", "AUC", "shr%", "n")
    print("\n" + hdr)
    print("-" * len(hdr))
    for a in arms:
        r = res["arms"][a]
        print("%-8s %5d %6.2f%% %5.2f %5d | %6.2f%% %5.2f %6.2f | %6.2f%% | %7.4f %6.4f | %5.2f%% %5d" % (
            a, r["prod"]["n"], 100 * r["prod"]["hit"], 100 * r["prod"]["se"],
            r["prod_empty_days"], 100 * r["top10"]["hit"], 100 * r["top10"]["se"],
            r["vs_base_top10"]["t"], 100 * r["top30"]["hit"], r["ap_mean"], r["auc_mean"],
            100 * r["prod_shrink"]["hit"], r["prod_shrink"]["n"]))

    print("\n逐月 前10 命中率（%）")
    months = sorted({d[:7] for d in all_dates})
    print("%-8s" % "臂" + "".join("%7s" % m[2:] for m in months))
    for a in arms:
        t = tabs[a]
        row = t[t["rank"] <= 10].groupby(t["date"].str[:7])["y_up"].mean()
        print("%-8s" % a + "".join("%7.1f" % (100 * row.get(m, np.nan)) for m in months))

    # 门控 + 校准：在 base 和（若存在）最好的臂上做
    best = max(arms, key=lambda a: res["arms"][a]["top10"]["hit"])
    sig = regime_signals(all_dates)
    res["gate"] = {}
    res["calib"] = {}
    for a in sorted({"base", best}):
        res["gate"][a] = gate_report(tabs[a], sig, didx)
        res["calib"][a] = calib_report(tabs[a], all_dates)
        print("\n[%s] 门控：信号三分位 -> prod 命中（n）/ 前10 命中" % a)
        for s, g in res["gate"][a].items():
            cells = []
            for b in g["terciles"]:
                cells.append("%s %5.1f%%(%3d) %5.1f%%" % (
                    b["bucket"], 100 * (b["prod"]["hit"] if b["prod"]["n"] else 0),
                    b["prod"]["n"], 100 * b["top10"]["hit"]))
            rr = g["rule_ge_expanding_median"]
            print("  %-15s %s | 规则≥扩张中位数: prod %5.2f%%(n=%d, %d/%d 天) 前10 %5.2f%%" % (
                s, "  ".join(cells), 100 * (rr["prod"]["hit"] if rr["prod"]["n"] else 0),
                rr["prod"]["n"], rr["days_on"], rr["days_total"], 100 * rr["top10"]["hit"]))
        c = res["calib"][a]
        print("[%s] 校准：逐月分数线 %s" % (a, {k[2:]: v for k, v in c["chosen"].items()}))
        print("     滚动校准 prod %5.2f%% ±%.2f (n=%d) | 固定(样本内) " % (
            100 * c["calib"]["hit"], 100 * c["calib"]["se"], c["calib"]["n"]) + "  ".join(
            "%s:%5.2f%%(%d)" % (s, 100 * v["hit"], v["n"]) for s, v in c["fixed_in_sample"].items()))

    res["composition"] = comp
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                      encoding="utf-8")
    print("\n-> " + str(OUT / "summary.json"))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--months", default="", help="只跑这几个月，逗号分隔（调试用）")
    a = ap.parse_args()
    if a.fit:
        fit_all([m for m in a.months.split(",") if m] or None)
    if a.analyze:
        analyze()
    if not (a.fit or a.analyze):
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
