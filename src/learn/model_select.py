"""
模型擂台：在同一套按天分块的走向前切分上，把几种传统 ML 方法和我们
手写的六维打分器放在一起比，选出最好的那个。

**选出来的模型不当生产排序器。** 它的两个用途见 docs/learning.md 第 7 节：

  1. 天花板估计——它的样本外 IC 减去我们打分器的 IC，就是「函数形式还
     丢了多少信息」。差得小就说明该停止调阈值了，这本身是个有价值的信号。
  2. 形状老师——用偏依赖曲线告诉我们 gap_pct_peak 这类参数该取多少，
     再把建议送进第 5 节的接受门。

为什么不让它直接排序：黑箱排名无法归因、无法用规则语言讨论，
而且免费源下它只能拿到 55% 的特征（竞价量能和轨迹历史上取不到）。

选择规则带**奥卡姆剃刀**：按 ICIR 排序，但如果一个更简单的模型落在
最优模型的自助置信区间里，就选简单的那个。复杂度只有在统计上买得起
的时候才值得付。
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# 喂给 ML 模型的原始特征。刻意包含 score.py 用到的全部输入，
# 这样「打分器 vs 模型」比的是函数形式，不是信息量。
FEATURES = [
    "gap_pct", "gap_norm", "auc_ratio", "t1_chg", "t2_chg", "t3_chg",
    "slope", "dive", "pos_pct_60d", "board_height", "sector_members",
    "sector_prev_limitups", "limit_pct",
    "monotonic", "ma_bull", "breakout", "prev_limit_up", "prev_broken_board",
    "log_auc_amount", "log_prev_amount",
]

# 复杂度序（用于奥卡姆剃刀）。数字越小越简单。
COMPLEXITY = {
    "Baseline(手写打分器)": 0,
    "RankRidge": 1,
    "RankElasticNet": 2,
    "RankHuber": 3,
    "ExtraTrees": 5,
    "RandomForest": 6,
    "HistGBDT": 7,
    "XGBoost": 7,          # 和 HistGBDT 同族同复杂度，公平同分
    "Ridge+GBDT残差": 8,
}


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_auc_amount"] = np.log1p(d["auc_amount"].clip(lower=0))
    d["log_prev_amount"] = np.log1p(d["prev_amount"].clip(lower=0))
    for b in ("monotonic", "ma_bull", "breakout", "prev_limit_up",
              "prev_broken_board"):
        d[b] = d[b].astype(float)
    return d


def rank_norm(x: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """按天做截面秩归一到 [-1, 1]。

    这是量化里的标准做法，不是可选项：它同时解决三件事——
    去掉日间量纲漂移、把长尾压平、让线性模型在非线性特征上也能工作。
    """
    return x.groupby(by).rank(pct=True).sub(0.5).mul(2.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _models(cfg_t: dict):
    from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor, ExtraTreesRegressor)
    t = cfg_t
    out = {
        "RankRidge": lambda: Ridge(alpha=10.0),
        "RankElasticNet": lambda: ElasticNet(alpha=0.001, l1_ratio=0.5,
                                             max_iter=5000),
        "RankHuber": lambda: HuberRegressor(alpha=1e-3, max_iter=500),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=200, max_depth=t.get("max_depth", 4) * 3,
            min_samples_leaf=t.get("min_child_samples", 200),
            n_jobs=-1, random_state=7),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=t.get("max_depth", 4) * 3,
            min_samples_leaf=t.get("min_child_samples", 200),
            n_jobs=-1, random_state=7),
        "HistGBDT": lambda: HistGradientBoostingRegressor(
            max_iter=t.get("n_estimators", 300),
            max_depth=t.get("max_depth", 4),
            learning_rate=t.get("learning_rate", 0.05),
            min_samples_leaf=t.get("min_child_samples", 200),
            l2_regularization=1.0, random_state=7),
    }
    # XGBoost 用户点名要比。装了就上场，没装不报错——擂台的意义在横评，
    # 缺一个选手不该让整场瘫掉。超参和 HistGBDT 对齐，比的是实现不是调参。
    try:
        from xgboost import XGBRegressor
        out["XGBoost"] = lambda: XGBRegressor(
            n_estimators=t.get("n_estimators", 300),
            max_depth=t.get("max_depth", 4),
            learning_rate=t.get("learning_rate", 0.05),
            min_child_weight=t.get("min_child_samples", 200),
            reg_lambda=1.0, tree_method="hist", n_jobs=-1,
            random_state=7, verbosity=0)
    except ImportError:
        pass
    return out


def walk_forward(df: pd.DataFrame, c: dict, n_folds: int = 5,
                 min_train_days: int = 40, top_k: int = 10) -> pd.DataFrame:
    """按天分块的扩展窗口走向前验证。

    按**天**切，不按行切。同一天的截面残差高度相关，按行随机切分会让
    训练集和测试集共享同一天的行情，样本外 IC 被系统性高估。
    """
    from learn import vscore
    import cfg as C

    d = prep_features(df)
    days = sorted(d["date"].unique())
    if len(days) < min_train_days + n_folds:
        log.warning("只有 %d 天，不够跑 %d 折（需要 >= %d 天）",
                    len(days), n_folds, min_train_days + n_folds)
        return pd.DataFrame()

    # 折边界：扩展窗口，测试段等长
    test_len = max(1, (len(days) - min_train_days) // n_folds)
    folds = []
    for i in range(n_folds):
        cut = min_train_days + i * test_len
        te = days[cut: cut + test_len]
        if not te:
            break
        folds.append((days[:cut], te))

    X = rank_norm(d[FEATURES], d["date"]).fillna(0.0)
    y = d["ytil"].to_numpy(float)
    date = d["date"].to_numpy()

    # 手写打分器不需要训练，直接算一遍
    base_score, base_rej = vscore.score_df(d, C.load())
    d = d.assign(_base=np.where(base_rej, -1e9, base_score))

    rows = []
    makers = _models(c.get("learning", {}).get("teacher", {}))
    for fi, (tr, te) in enumerate(folds):
        mtr, mte = np.isin(date, tr), np.isin(date, te)
        preds = {"Baseline(手写打分器)": d["_base"].to_numpy()[mte]}
        for name, make in makers.items():
            try:
                m = make()
                m.fit(X[mtr], y[mtr])
                preds[name] = m.predict(X[mte])
            except Exception as e:      # noqa: BLE001
                log.warning("%s 第 %d 折失败: %s", name, fi, e)
        # 残差堆叠：线性拿主效应，树拿残差
        try:
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import HistGradientBoostingRegressor
            lin = Ridge(alpha=10.0).fit(X[mtr], y[mtr])
            res = y[mtr] - lin.predict(X[mtr])
            gb = HistGradientBoostingRegressor(
                max_iter=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=200, random_state=7).fit(X[mtr], res)
            preds["Ridge+GBDT残差"] = lin.predict(X[mte]) + gb.predict(X[mte])
        except Exception as e:          # noqa: BLE001
            log.warning("堆叠第 %d 折失败: %s", fi, e)

        sub = d[mte]
        for name, p in preds.items():
            for day, g in sub.assign(_p=p).groupby("date"):
                ic = spearman(g["_p"].to_numpy(), g["ytil"].to_numpy())
                top = g.nlargest(top_k, "_p")
                rows.append({
                    "fold": fi, "model": name, "date": day, "ic": ic,
                    "top_excess": float(top["y"].mean()),
                    "hit": float((top["y"] > 0).mean()),
                })
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, bootstrap_n: int = 2000,
              seed: int = 7) -> pd.DataFrame:
    """每个模型的样本外指标 + 按天自助的 ICIR 置信区间。"""
    if res.empty:
        return res
    rng = np.random.default_rng(seed)
    out = []
    for name, g in res.groupby("model"):
        ic = g["ic"].dropna().to_numpy()
        if ic.size < 3:
            continue
        icir = ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0.0
        boot = []
        for _ in range(bootstrap_n):
            s = rng.choice(ic, ic.size, replace=True)
            sd = s.std(ddof=1)
            boot.append(s.mean() / sd if sd > 0 else 0.0)
        lo, hi = np.quantile(boot, [0.05, 0.95])
        out.append({
            "model": name, "days": ic.size,
            "IC": ic.mean(), "IC_std": ic.std(ddof=1), "ICIR": icir,
            "ICIR_lo": lo, "ICIR_hi": hi,
            "top_excess": g["top_excess"].mean(),
            "hit": g["hit"].mean(),
            "complexity": COMPLEXITY.get(name, 9),
        })
    return pd.DataFrame(out).sort_values("ICIR", ascending=False)


def pick(summary: pd.DataFrame) -> tuple[str, str]:
    """奥卡姆剃刀：ICIR 最高的是冠军，但落在它 90% 置信下界之上的
    更简单模型胜出。复杂度只有在统计上买得起的时候才值得付。
    """
    if summary.empty:
        return "", "样本不足，无法选型"
    best = summary.iloc[0]
    tied = summary[(summary["ICIR"] >= best["ICIR_lo"])
                   & (summary["complexity"] < best["complexity"])]
    if len(tied):
        win = tied.sort_values("complexity").iloc[0]
        return win["model"], (
            f"{best['model']} 的 ICIR {best['ICIR']:.3f} 最高，但 90% 下界是 "
            f"{best['ICIR_lo']:.3f}；{win['model']} 的 {win['ICIR']:.3f} 在这个"
            f"区间内且更简单，按奥卡姆取简单的")
    return best["model"], (
        f"ICIR {best['ICIR']:.3f}（90% 区间 {best['ICIR_lo']:.3f}~"
        f"{best['ICIR_hi']:.3f}），没有更简单的模型落进这个区间")
