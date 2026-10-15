"""
换条路研究（预登记 altpath_v1）。开发集 2024-06..2025-12 逐月走向前，evalkit 协议，确认集不读。

    python research/altpath/build_feats.py           先建 a158 / lu / env 三份（data/breakout/raw/exp/altpath/）
    python research/altpath/exp_altpath.py --fit     训练臂（base 对照同循环重跑，样本 / 特征缓存 / 种子同一份）
    python research/altpath/exp_altpath.py --analyze 成绩、配对、Newey-West、事后臂、环境门控

不改 src/ tools/ state/；模型指纹四个文件不碰。新特征全部在这里算、按 (code, date) 并进来。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "breakout"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import evalkit as K  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
NAME = "altpath_v1"
FE = K.EXP / "altpath"
NW_LAG = 20
HYP = (
    "换条路：现有特征 / 目标 / 模型这条路已到头（实验 16~21），试五个方向能不能真实提高每天清单 A 的命中率"
    "或稳定性。1 Alpha158 式标准量价特征：单独一个臂、和生产特征合并一个臂、两者名次平均一个臂（事后），"
    "回答特征工程是不是瓶颈。2 涨停生态：个股层（首板 / 连板 / 炸板 / 近期涨停次数，排名归一）一个臂，"
    "再加市场层（涨停家数占比、炸板率、连板高度、昨日涨停今日表现、上涨占比、成交额相对 60 日）一个臂；"
    "规则臂「首板次日」：当天首板（此前 10 根没封过）的票按现状模型分取前 10。"
    "3 族群：实验 16/18 的 g2 已做过（不重复训练），只按 Newey-West 复核；新加一个事后规则臂「族群补涨」："
    "现状前 50 名里伙伴涨停占比当日排名 >= 0.9 且相对伙伴 5 日涨幅排名 <= 0.5 的先排，其余按分。"
    "4 环境门控：开发集前半（2024-06..2025-02）在 7 个环境指标里挑和每日前 10 命中关系最强的一个（NW t 绝对值最大），"
    "后半（2025-03..12）只检验这一个：按只用过去的扩张三分位切「差」档，差档命中显著低（NW t <= -2.5）才算可做可信度指标；"
    "产品形态「差档日只出前 5」看每只命中和最差月。"
    "5 自选：a 按波动率三分组各训一个模型（赔率按组基准率校正后合并排序），理由是 50% 标签几乎只在高波动票里发生，"
    "全市场一个模型的分裂预算被低波动票的负样本占掉；b 分数三日平滑（t、t-1、t-2 的预测值均值排序），"
    "理由是实验 20 连续够格 2 天以上命中 21.0% 对 14.6%，平滑是同一信息的连续版、不减只数。"
    "对照一律是同一循环重跑的现状臂 base。主尺子：每天前 10 买得到口径按天配对；t 另报 Newey-West（滞后 20）。"
    "过线：NW t >= 2.5 且最差三个月不比现状差且空榜不翻倍；多重检验按本族 10 个臂 Bonferroni，单侧 "
    "p = 0.0062/10，对应 t >= 3.23。"
)
ARMS = {
    "a158": "只用 Alpha158 式特征（142 列，当日排名）",
    "base_a158": "生产特征 + Alpha158",
    "ens_a158": "base 和 a158 当日名次平均（事后）",
    "lu_ind": "生产特征 + 涨停生态个股层 10 列",
    "lu_all": "生产特征 + 涨停生态个股层 + 市场层 7 列",
    "fb_pool": "首板次日候选池，按现状分取前 10（事后）",
    "peer_lag": "族群补涨：现状前 50 里伙伴涨停多、自己落后的先排（事后）",
    "gate": "环境门控：前半选指标，后半检验（事后）",
    "volgrp": "按 20 日波动率三分组各训一个模型，赔率校正后合并",
    "smooth3": "现状预测值三日平滑排序（事后）",
}
N_ARMS = len(ARMS)
T_BONF = float(norm.ppf(1 - norm.sf(2.5) / N_ARMS))

LU_IND = ["l_lu", "l_zb", "l_streak", "l_first", "l_cnt5", "l_cnt20_pct", "l_cnt60_pct",
          "l_zb20_pct", "l_since_pct", "l_after_lu_ret_pct"]
LU_MKT = ["m_lu_share", "m_zb_rate", "m_max_streak", "m_prev_lu_ret", "m_prev_lb_ret",
          "m_adv_share", "m_amt_ratio60"]
ENV = ["amt_ratio60", "lu_share", "zb_rate", "adv_share", "prev_lu_ret", "max_streak", "base_closed20"]


def a158_cols() -> list[str]:
    import pyarrow.parquet as pq
    return [c for c in pq.ParquetFile(FE / "a158.parquet").schema.names if c.startswith("a_")]


# ---------------------------------------------------------------
#  数据：只读用得到的列（这台机器 14GB，另有研究在跑）
# ---------------------------------------------------------------
def load() -> pd.DataFrame:
    import buyable as B
    cache = json.loads((K.EXP / "feats_by_month.json").read_text(encoding="utf-8"))
    months = K.months_in(K.DEV)
    assert all(m in cache for m in months), "特征选择缓存不全，会触发重选（要全表），先停"
    feats = sorted({c for m in months for c in cache[m]})
    cols = ["code", "date", "board", "y_up"] + feats
    df = pd.read_parquet(K.DATA / "train.parquet", columns=cols,
                         filters=[("date", "<", K.CONFIRM[0] + "-01")])
    assert df["date"].max() < K.CONFIRM[0], "确认集漏进来了"
    df["code"] = df["code"].astype(str).str.zfill(6)
    df[feats + ["y_up"]] = df[feats + ["y_up"]].astype("float32")
    lab = B.open_labels()
    lab = lab[lab["date"] < K.CONFIRM[0]]
    df = df.merge(lab[["code", "date", "y_open"]], on=["code", "date"], how="left")
    lu = pd.read_parquet(FE / "lu.parquet", columns=["code", "date"] + LU_IND + LU_MKT)
    df = df.merge(lu, on=["code", "date"], how="left")
    del lu
    ac = a158_cols()
    a = pd.read_parquet(FE / "a158.parquet")
    df = df.merge(a, on=["code", "date"], how="left")
    del a
    print("训练表 %d 行 %d 列，a158 缺失 %.1f%%，lu 缺失 %.1f%%，内存 %.1f GB" % (
        len(df), df.shape[1], 100 * df["a_std20"].isna().mean(), 100 * df["l_lu"].isna().mean(),
        df.memory_usage().sum() / 2 ** 30), flush=True)
    assert len(ac) > 100
    return df


def fit_predict(tr, te, feats, month, state):
    import model as M
    ac = state.setdefault("ac", a158_cols())
    trs = M.stratified_sample(tr, "y_up")
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    out = {}

    def one(cols, name):
        m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
        out[name] = (m.predict_proba(te), m.predict_proba(ref))

    one(feats, "base")
    # 教训 32：先断言臂真的不一样（入模列数变了）
    for name, cols in (("a158", ac), ("base_a158", feats + ac), ("lu_ind", feats + LU_IND),
                       ("lu_all", feats + LU_IND + LU_MKT)):
        assert set(cols) != set(feats), name
        one(cols, name)
    # 波动率三分组：各训一个，赔率按组的真实基准率校正（分组采样把组间基准率抹平了）
    pt, pr = np.zeros(len(te)), np.zeros(len(ref))
    edges = (1 / 3, 2 / 3)

    def grp(x):
        v = x["a_std20"].to_numpy(float)
        v = np.where(np.isfinite(v), v, 0.5)
        return np.digitize(v, edges)
    g_tr, g_trs, g_te, g_ref = grp(tr), grp(trs), grp(te), grp(ref)
    for g in range(3):
        sub = trs[g_trs == g]
        m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(sub, feats, "y_up")
        pi = float(tr.loc[g_tr == g, "y_up"].mean())
        s = float(sub["y_up"].mean())
        k = (pi / (1 - pi)) / (s / (1 - s))
        for X, gg, dst in ((te, g_te, pt), (ref, g_ref, pr)):
            idx = gg == g
            if idx.any():
                p = np.clip(m.predict_proba(X[idx]), 1e-6, 1 - 1e-6)
                o = p / (1 - p) * k
                dst[idx] = o / (1 + o)
    out["volgrp"] = (pt, pr)
    return out


# ---------------------------------------------------------------
#  尺子：evalkit + Newey-West
# ---------------------------------------------------------------
def nw_t(d: pd.Series, lag: int = NW_LAG) -> tuple[float, float]:
    """均值的 Newey-West t（Bartlett 核）。返回 (t, se)。"""
    x = d.dropna().to_numpy(float)
    n = len(x)
    if n < lag + 5:
        return float("nan"), float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(v, 1e-18) / n))
    return float(x.mean() / se), se


def daily_diff(a: pd.DataFrame, b: pd.DataFrame, y: str = "y_open", k: int = 10, since: str = "") -> pd.Series:
    if since:
        a, b = a[a["date"] >= since], b[b["date"] >= since]
    da = a[a["rank"] <= k].groupby("date")[y].mean()
    db = b[b["rank"] <= k].groupby("date")[y].mean()
    return (da - db).dropna().sort_index()


def summ(t: pd.DataFrame, since: str = "", k: int = 10) -> dict:
    if since:
        t = t[t["date"] >= since]
    tk = t[t["rank"] <= k]
    o = {"days": int(t["date"].nunique())}
    for y, key in (("y_open", "open"), ("y_up", "close")):
        o[key] = K.clustered(tk, y)
    mo = K.monthly(tk)
    o["month_median"] = float(mo.median())
    o["month_worst"] = float(mo.min())
    o["month_sd"] = float(mo.std())
    o["monthly"] = {kk: float(v) for kk, v in mo.items()}
    if "score" in t.columns:
        pp = K.prod_picks(t)
        o["prod_open"] = K.clustered(pp, "y_open")
        o["prod_empty"] = int(o["days"] - pp["date"].nunique())
    return o


def cmp(a, b, since="", k=10):
    c = K.compare(a, b, since=since, k=k)
    d = daily_diff(a, b, since=since, k=k)
    c["t_nw"], c["se_nw"] = nw_t(d)
    sys.path.insert(0, str(ROOT / "research" / "stats"))
    import hac
    bb = hac.mbb(d)
    c["boot_p"], c["boot_lo"], c["boot_hi"] = bb["p"], bb["lo"], bb["hi"]
    c["pass_nw"] = bool(c["t_nw"] >= K.GATE["t_day"] and c["gate"]["worst3"] and c["gate"]["empty"])
    c["pass_bonf"] = bool(c["pass_nw"] and c["t_nw"] >= T_BONF)
    c["verdict_nw"] = ("过线" if c["pass_nw"] else "未定" if abs(c["diff_day"]) < 0.025 or c["t_nw"] > 0
                       else "不过线")
    return c


def fmt(name, s, c=None):
    line = "%-10s 前10 开盘 %5.2f%% ±%.2f 收盘 %5.2f%% 月中位 %4.1f 最差 %4.1f 月标准差 %4.1f" % (
        name, 100 * s["open"]["hit"], 100 * s["open"]["se"], 100 * s["close"]["hit"],
        100 * s["month_median"], 100 * s["month_worst"], 100 * s["month_sd"])
    if "prod_open" in s:
        line += " | 生产规则 %5.2f%%(n=%d,空榜 %d)" % (100 * s["prod_open"]["hit"], s["prod_open"]["n"],
                                                  s["prod_empty"])
    if c:
        line += " || 对 base %+.2f t=%.2f NW t=%.2f 块自举 p=%.3f 赢 %d/%d 最差三月 %s -> %s%s" % (
            100 * c["diff_day"], c["t_day"], c["t_nw"], c["boot_p"], c["wins"], c["months"],
            ",".join("%+.1f" % (100 * v) for v in c["worst3"].values()), c["verdict_nw"],
            "（Bonferroni 后仍过）" if c["pass_bonf"] else "")
    return line


# ---------------------------------------------------------------
#  事后臂
# ---------------------------------------------------------------
def rerank(allp: pd.DataFrame, key: str, labels: pd.DataFrame, st: set) -> pd.DataFrame:
    t = allp.merge(labels, on=["code", "date"], how="inner")
    t = t[~t["code"].isin(st)]
    t = t.sort_values(["date", key], ascending=[True, False])
    t["rank"] = t.groupby("date").cumcount() + 1
    return t[t["rank"] <= K.TOP_KEEP]


def ens(all_a, all_b, labels, st):
    x = all_a.merge(all_b, on=["code", "date"], suffixes=("_a", "_b"))
    x["ra"] = x.groupby("date")["p_a"].rank(pct=True)
    x["rb"] = x.groupby("date")["p_b"].rank(pct=True)
    x["key"] = (x["ra"] + x["rb"]) / 2 + 1e-9 * x["rb"]
    return rerank(x[["code", "date", "key"]], "key", labels, st)


def smooth3(all_b, labels, st):
    x = all_b.sort_values(["code", "date"]).copy()
    # t、t-1、t-2 三个交易日（按该票自己的行；停牌跨过去的照算，开发集里很少）
    x["key"] = x.groupby("code")["p"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return rerank(x[["code", "date", "key"]], "key", labels, st)


def fb_pool(all_b, labels, st):
    lu = pd.read_parquet(FE / "lu.parquet", columns=["code", "date", "l_first"])
    x = all_b.merge(lu, on=["code", "date"], how="left")
    pool = x[x["l_first"] == 1]
    t = rerank(pool[["code", "date", "p"]], "p", labels, st)
    return t, pool.merge(labels, on=["code", "date"])


def peer_lag(top_b):
    fx = pd.read_parquet(K.DATA / "raw" / "exp_acc" / "feats2.parquet",
                         columns=["code", "date", "x_peer_lu_pct", "x_rel_peer5_pct"])
    t = top_b[top_b["rank"] <= 50].merge(fx, on=["code", "date"], how="left")
    t["lag"] = ((t["x_peer_lu_pct"] >= 0.9) & (t["x_rel_peer5_pct"] <= 0.5)).astype(int)
    t = t.sort_values(["date", "lag", "p"], ascending=[True, False, False])
    t["rank"] = t.groupby("date").cumcount() + 1
    return t.drop(columns=["score"], errors="ignore")


# ---------------------------------------------------------------
#  环境门控
# ---------------------------------------------------------------
def gate_analysis(top_b: pd.DataFrame, base_mkt: pd.Series) -> dict:
    env = pd.read_parquet(FE / "env.parquet").set_index("date").sort_index()
    h = top_b[top_b["rank"] <= 10].groupby("date")["y_open"].mean()
    ex = (h - base_mkt.reindex(h.index))            # 超额：前 10 命中 − 当天全市场基准
    first = ("2024-06-01", "2025-03-01")
    out = {"first_half": {}, "second_half": {}}
    # 扩张三分位（只用 t-1 及以前、2024-01 起的环境历史）
    def tercile(col):
        s = env[col]
        s0 = s[s.index >= "2024-01-01"]
        lo = s0.expanding(min_periods=60).quantile(1 / 3).shift(1)
        hi = s0.expanding(min_periods=60).quantile(2 / 3).shift(1)
        z = pd.Series(np.where(s0 <= lo, 0, np.where(s0 >= hi, 2, 1)), index=s0.index)
        return z.where(lo.notna())

    def slope_t(y, x):
        d = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
        if len(d) < 40:
            return float("nan"), float("nan")
        xs = (d["x"] - d["x"].mean()) / d["x"].std()
        b = float((xs * (d["y"] - d["y"].mean())).sum() / (xs * xs).sum())
        # 斜率的 NW t：残差 × 标准化 x 的序列均值
        u = xs * (d["y"] - d["y"].mean() - b * xs)
        _, se = nw_t(pd.Series(u.to_numpy()), NW_LAG)
        se_b = se / float((xs * xs).mean())
        return b, b / se_b

    for half, (a, b) in (("first_half", first), ("second_half", ("2025-03-01", "2026-01-01"))):
        for col in ENV:
            hh = h[(h.index >= a) & (h.index < b)]
            xx = env[col].reindex(hh.index)
            bh, th = slope_t(hh, xx)
            be, te_ = slope_t(ex.reindex(hh.index), xx)
            z = tercile(col).reindex(hh.index)
            by = {int(k): (float(hh[z == k].mean()), int((z == k).sum())) for k in (0, 1, 2) if (z == k).any()}
            out[half][col] = {"slope_hit_per_sd": bh, "t_nw_hit": th, "slope_excess_per_sd": be,
                              "t_nw_excess": te_, "tercile_hit": by}
    pick = max(ENV, key=lambda c: abs(np.nan_to_num(out["first_half"][c]["t_nw_hit"])))
    sign = np.sign(out["first_half"][pick]["slope_hit_per_sd"])
    out["picked"] = pick
    out["picked_sign"] = float(sign)
    # 后半检验：「差」档 = 指标对命中不利的那一端三分位
    a, b = "2025-03-01", "2026-01-01"
    hh = h[(h.index >= a) & (h.index < b)]
    z = tercile(pick).reindex(hh.index)
    bad = z == (0 if sign > 0 else 2)
    # 差档日 vs 其余日的差：用指示变量回归 NW t
    ind = bad.astype(float)
    bb, tt = slope_t(hh, ind)
    out["test"] = {"bad_days": int(bad.sum()), "other_days": int((~bad).sum()),
                   "hit_bad": float(hh[bad].mean()), "hit_other": float(hh[~bad].mean()),
                   "t_nw_indicator": tt, "diff": float(hh[bad].mean() - hh[~bad].mean())}
    # 产品形态：差档日只出前 5
    tb = top_b[(top_b["date"] >= a) & (top_b["date"] < b)]
    badd = set(bad[bad].index)
    pol = tb[(tb["rank"] <= 10) & (~tb["date"].isin(badd) | (tb["rank"] <= 5))]
    t10 = tb[tb["rank"] <= 10]
    mo_p, mo_b = K.monthly(pol), K.monthly(t10)
    out["policy_top5_on_bad"] = {"hit": K.clustered(pol)["hit"], "n": int(len(pol)),
                                 "hit_base": K.clustered(t10)["hit"], "n_base": int(len(t10)),
                                 "worst_month": float(mo_p.min()), "worst_month_base": float(mo_b.min()),
                                 "month_sd": float(mo_p.std()), "month_sd_base": float(mo_b.std())}
    return out


def descriptive(base: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """预登记之外、只描述不判过线的两张表。"""
    out = ["", "[描述，预登记之外，不作过线依据]"]
    lu = pd.read_parquet(FE / "lu.parquet", columns=["code", "date", "l_lu", "l_zb", "l_streak", "l_first",
                                                       "l_cnt20"])
    x = labels.merge(lu, on=["code", "date"], how="inner")
    tot = x["y_open"].mean()
    cats = {"全市场": x, "当天首板": x[x["l_first"] == 1], "当天 2 连板": x[x["l_streak"] == 2],
            "当天 3 板以上": x[x["l_streak"] >= 3], "当天炸板": x[x["l_zb"] == 1],
            "近 20 天封过板、今天没封": x[(x["l_cnt20"] > 0) & (x["l_lu"] == 0)]}
    out.append("涨停生态分组的次日开盘买得到命中（开发集，按天聚类）")
    for k, v in cats.items():
        c = K.clustered(v)
        out.append("  %-18s 每天 %6.1f 只 命中 %5.2f%% ±%.2f（%.1f 倍）" % (
            k, len(v) / max(v["date"].nunique(), 1), 100 * c["hit"], 100 * c["se"], c["hit"] / tot))
    # 模型自身的「今天有多确定」：够格只数、前 10 平均预测值、和昨天前 10 的重合
    t = base.copy()
    h = t[t["rank"] <= 10].groupby("date")["y_open"].mean()
    nq = t[t["score"] >= 97].groupby("date").size().reindex(h.index).fillna(0)
    p10 = t[t["rank"] <= 10].groupby("date")["p"].mean()
    sets = t[t["rank"] <= 10].groupby("date")["code"].apply(set)
    ov = pd.Series([np.nan] + [len(sets.iloc[i] & sets.iloc[i - 1]) / 10 for i in range(1, len(sets))],
                   index=sets.index)
    out.append("模型自身信号和当天前 10 命中（每个标准差的斜率，NW t）")
    for nm, sig in (("够格只数", nq), ("前10平均预测值", p10), ("和昨天重合", ov)):
        for half, (a, b) in (("前半", ("2024-06-01", "2025-03-01")), ("后半", ("2025-03-01", "2026-01-01"))):
            hh = h[(h.index >= a) & (h.index < b)]
            xx = sig.reindex(hh.index)
            dd = pd.concat([hh, xx], axis=1, keys=["y", "x"]).dropna()
            xs = (dd["x"] - dd["x"].mean()) / dd["x"].std()
            bb = float((xs * (dd["y"] - dd["y"].mean())).sum() / (xs * xs).sum())
            u = xs * (dd["y"] - dd["y"].mean() - bb * xs)
            _, se = nw_t(pd.Series(u.to_numpy()))
            out.append("  %-12s %s 斜率 %+.2fpp t=%.2f" % (nm, half, 100 * bb, bb / (se / float((xs * xs).mean()))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS, metric="top10 y_open 按天配对 + Newey-West(20)",
             gate={**K.GATE, "t_nw": 2.5, "bonferroni_arms": N_ARMS, "t_bonf": round(T_BONF, 3)})
    months = K.months_in(K.DEV)
    if a.fit:
        df = load()
        K.walk_forward(df, months, fit_predict, NAME)
    if a.analyze:
        analyze()
    return 0


def analyze() -> None:
    import validate as V
    d = K.EXP / NAME
    st = V.st_codes()
    tops = {arm: pd.read_parquet(d / f"top_{arm}.parquet")
            for arm in ("base", "a158", "base_a158", "lu_ind", "lu_all", "volgrp")
            if (d / f"top_{arm}.parquet").exists()}
    base = tops["base"]
    all_b = pd.read_parquet(d / "all_base.parquet")
    labels = pd.read_parquet(K.DATA / "raw" / "labels_open.parquet", columns=["code", "date", "y_open"])
    labels = labels[(labels["date"] >= K.DEV[0]) & (labels["date"] < K.CONFIRM[0])]
    yup = pd.read_parquet(K.DATA / "train.parquet", columns=["code", "date", "board", "y_up"],
                          filters=[("date", ">=", K.DEV[0]), ("date", "<", K.CONFIRM[0])])
    yup["code"] = yup["code"].astype(str).str.zfill(6)
    labels = labels.merge(yup, on=["code", "date"], how="inner")
    labels = labels[np.isfinite(labels["y_open"]) & np.isfinite(labels["y_up"])]
    base_mkt = labels.groupby("date")["y_open"].mean()
    if (d / "all_a158.parquet").exists():
        tops["ens_a158"] = ens(pd.read_parquet(d / "all_a158.parquet"), all_b, labels, st)
    tops["smooth3"] = smooth3(all_b, labels, st)
    fbt, fbpool = fb_pool(all_b, labels, st)
    tops["fb_pool"] = fbt
    tops["peer_lag"] = peer_lag(base)
    res = {"n_arms": N_ARMS, "t_bonf": T_BONF, "market_base_open": float(labels["y_open"].mean())}
    # 族群 g2 复核：实验 18 的对照是 w1 的 base（同循环），这里只换 t 的算法
    g2 = K.EXP / "w2_feat_hp" / "top_g2.parquet"
    w1b = K.EXP / "w1_scale_avg" / "top_base.parquet"
    extra = {}
    if g2.exists() and w1b.exists():
        extra["g2(实验18)"] = (pd.read_parquet(g2), pd.read_parquet(w1b))
    lines = []
    for since, lbl in (("", "全部 19 个月"), ("2025-01-01", "2025 起")):
        lines.append("[%s]" % lbl)
        sb = summ(base, since)
        res[f"base|{lbl}"] = sb
        lines.append(fmt("base", sb))
        for arm, t in tops.items():
            if arm == "base":
                continue
            s = summ(t, since)
            c = cmp(t, base, since=since)
            res[f"{arm}|{lbl}"] = s
            res[f"cmp_{arm}|{lbl}"] = c
            lines.append(fmt(arm, s, c))
        for nm, (a_, b_) in extra.items():
            c = cmp(a_, b_, since=since)
            res[f"cmp_{nm}|{lbl}"] = c
            lines.append("%-10s 复核：对 w1 base %+.2f 按天 t=%.2f NW t=%.2f 赢 %d/%d" % (
                nm, 100 * c["diff_day"], c["t_day"], c["t_nw"], c["wins"], c["months"]))
        lines.append("")
    # 首板池本身
    fp = fbpool[fbpool["date"] >= K.DEV[0]]
    res["fb_pool_stats"] = {"pool_per_day": float(fp.groupby("date").size().mean()),
                            "pool_hit_open": K.clustered(fp)["hit"], "pool_days": int(fp["date"].nunique()),
                            "top10_days": int(fbt["date"].nunique())}
    lines.append("首板池：每天 %.1f 只，池内买得到命中 %.2f%%（全市场 %.2f%%），有池的天 %d" % (
        res["fb_pool_stats"]["pool_per_day"], 100 * res["fb_pool_stats"]["pool_hit_open"],
        100 * res["market_base_open"], res["fb_pool_stats"]["pool_days"]))
    g = gate_analysis(base, base_mkt)
    res["gate"] = g
    lines.append("\n环境门控（前 10 买得到命中对指标的斜率，每个标准差；NW t）")
    for half in ("first_half", "second_half"):
        lines.append("  [%s]" % half)
        for col in ENV:
            r = g[half][col]
            lines.append("  %-14s 命中斜率 %+.2fpp t=%.2f | 超额斜率 %+.2fpp t=%.2f | 三分位 %s" % (
                col, 100 * r["slope_hit_per_sd"], r["t_nw_hit"], 100 * r["slope_excess_per_sd"],
                r["t_nw_excess"], {k: "%.1f%%(%d)" % (100 * v[0], v[1]) for k, v in r["tercile_hit"].items()}))
    lines.append("  前半挑中 %s（符号 %+d），后半检验：%s" % (g["picked"], g["picked_sign"], g["test"]))
    lines.append("  差档日只出前 5：%s" % g["policy_top5_on_bad"])
    lines += descriptive(base, labels)
    # 逐月
    mo = sorted({x[:7] for x in base["date"]})
    lines.append("\n逐月 前10 买得到（%）")
    lines.append("%-10s" % "臂" + "".join("%6s" % m[2:] for m in mo))
    for arm, t in tops.items():
        r = K.monthly(t[t["rank"] <= 10])
        lines.append("%-10s" % arm + "".join("%6.1f" % (100 * r.get(m, np.nan)) for m in mo))
    txt = "\n".join(lines)
    print(txt)
    (Path(__file__).parent / "results.txt").write_text(txt, encoding="utf-8")
    K.record(NAME, res)
    (d / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                    encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
