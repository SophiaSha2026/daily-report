"""
第 4 步：筛选集 2022-01..2024-05 逐月走向前（evalkit.walk_forward 逐月调用，可续跑），现状模型 base，
种子 7 / 9；分析：前 10 买得到命中、全市场基准、决策门、两个种子的配对差噪声（NW se）和最小可检出效应、
辅助尺子（前 10 平均 log1p(r20_open)，次日一字记 0）。

登记：**不进主族** breakout_y50_top10。evalkit 只有 prereg() / compare(exp=, arm=) 两处写登记簿
（data/breakout/raw/exp/ledger.jsonl），walk_forward 不写；这里不调 prereg，compare 不带 exp/arm，
所以主登记簿一行不加、族门槛不变。筛选集的臂全部记在 research/screen/ledger.jsonl（族名 screen_y50_top10），
prereg.json 自己写在 data/breakout/raw/exp/<name>/。

训练窗口：每个测试月只给它前 TRAIN_YEARS=3 年（含）到净化线的行（生产的训练表本身约三年；开发集是从表头起的
扩张窗口，前几个月更短）。特征选择：每月在净化过的训练切片上 fselect.run（和生产 / 开发集同一条路），
缓存键 "<月>|screen_v1"（evalkit.feats_for 的 drop 命名空间，和 cont_y15 同一个用法），不碰开发集月份的缓存。

    python research/screen/exp_screen.py --fit                 筛选集 base，种子 7 9（可续跑，北京 16:00~18:30 自动停）
    python research/screen/exp_screen.py --analyze
    python research/screen/exp_screen.py --fit --arms base eqd --seeds 7 --name screen_w2_v1    第 2 周的用法
    python research/screen/exp_screen.py --debug               150 只伪造表、2 个月，不登记

第 2 周加一个臂：在 ARMS 里加一项 {"y": 训练标签列, "w": 样本权重函数(trs)->array 或 None, "kw": LightGBM 参数,
"cols": 列变换函数(feats)->cols 或 None, "desc": 说明}。训练标签列不在表里的，在 LABELS 里登记一个
fn(df)->Series（从 r20_open / yizi1 / y_up 现算），load() 会按需加列。一次循环可以跑多个臂（每月每个臂每个种子
各训一个模型，内存只多一份预测值）；base 必须在同一循环里（同一份特征选择、同一份训练切片）。
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402
import hac  # noqa: E402

NAME = "screen_base_v1"
DROP_KEY = "screen_v1"          # feats_for 的缓存命名空间（不是列名，什么都不丢）
SEEDS_DEFAULT = (7, 9)
HYP = ("扩展筛选集 2022-01..2024-05（2020-07 起的 pit 价格、同一套生产特征、同一条走向前）上现状模型 base 的"
       "前 10 买得到命中是否 >= 2 倍当日全市场基准（设计文档第 8 节第 1 周决策门）；两个种子的按天配对差 NW se "
       "给出筛选集能检出的最小效应。不跑新臂，不进主族登记簿。")


# ---------------------------------------------------------------
#  标签变体（第 2 周用；本周只用 y_up）。都从 r20_open / yizi1 现算，不动 label.py
# ---------------------------------------------------------------
def y_rel(q: float):
    """当日相对标签：每天在有标签的行里按 r20_open 排名（次日一字记最低），前 q% 为 1。"""
    def fn(df: pd.DataFrame) -> pd.Series:
        r = df["r20_open"].where(~df["yizi1"].fillna(False).astype(bool), -1.0)
        ok = np.isfinite(df["y_open"])
        pct = r.where(ok).groupby(df["date"]).rank(pct=True)
        return pd.Series(np.where(ok, (pct > 1 - q / 100.0).astype(float), np.nan), index=df.index, dtype="float32")
    return fn


def u_day(df: pd.DataFrame) -> pd.Series:
    """当日 r20_open 的秩百分位 u（次日一字记最低；y_open 没标签的行 NaN）。y_rel 和 y_soft 都从它来。"""
    r = df["r20_open"].where(~df["yizi1"].fillna(False).astype(bool), -1.0)
    ok = np.isfinite(df["y_open"])
    return r.where(ok).groupby(df["date"]).rank(pct=True).astype("float32")


def y_softh(df: pd.DataFrame) -> pd.Series:
    """软标签 1（设计 2.2）：尾部铰链 max(u - 0.8, 0) / 0.2，取值 [0, 1]，当天后 20% 的行才非零。"""
    u = df["u_day"] if "u_day" in df.columns else u_day(df)
    return ((u - 0.8) / 0.2).clip(0.0, 1.0).astype("float32")


def y_softz(df: pd.DataFrame) -> pd.Series:
    """软标签 2（设计 2.2）：当天 log1p(r20_open)（次日一字记当天最低），当天 1% / 99% 分位去极值，当天标准化。"""
    ok = np.isfinite(df["y_open"])
    yz = df["yizi1"].fillna(False).astype(bool)
    l = np.log1p(df["r20_open"].clip(lower=-0.99)).where(ok & ~yz)
    g = l.groupby(df["date"])
    l = l.where(~(ok & yz), g.transform("min"))
    g = l.groupby(df["date"])
    lo, hi = g.transform(lambda s: s.quantile(0.01)), g.transform(lambda s: s.quantile(0.99))
    l = l.clip(lo, hi)
    g = l.groupby(df["date"])
    return ((l - g.transform("mean")) / (g.transform("std") + 1e-9)).astype("float32")


def seg_len(df: pd.DataFrame, y: str = "y_up") -> pd.Series:
    """按事件等权（设计 3.2）要的段长：同一只票连续亮灯（y == 1 的相邻行）算一段，正样本行记所在段的长度，
    其它行 NaN。按表里的相邻行数（停牌缺行就断段）。"""
    d = df[["code", "date", y]].sort_values(["code", "date"])
    pos = (d[y].to_numpy() == 1)
    code = d["code"].to_numpy()
    prev_pos = np.r_[False, pos[:-1]]
    same = np.r_[False, code[1:] == code[:-1]]
    new = pos & ~(prev_pos & same)
    sid = np.cumsum(new)
    sid = np.where(pos, sid, 0)
    ln = np.bincount(sid[pos])
    out = np.where(pos, ln[sid].astype(float), np.nan)
    return pd.Series(out, index=d.index, dtype="float32").reindex(df.index)


LABELS = {"y_rel4": y_rel(4.0), "y_rel2": y_rel(2.0), "u_day": u_day, "y_softh": y_softh, "y_softz": y_softz,
          "seg_up": lambda df: seg_len(df, "y_up")}
# load() 按这个顺序加列（后面的用前面的）
LABEL_ORDER = ["u_day", "seg_up", "y_rel4", "y_rel2", "y_softh", "y_softz"]


# ---------------------------------------------------------------
#  样本权重（第 2 周）。L1Lgbm.fit 显式给 w 就不再算按月等权，所以这里每个函数自己把月那层算进去
# ---------------------------------------------------------------
def w_eqd(trs: pd.DataFrame) -> np.ndarray:
    """按天等权，嵌在按月等权里：w = 1 / (当天行数 x 当月天数)。每个月权重和相等（和 base 一样），
    月内每一天权重和相等（新加的那一层）。均值 1。"""
    mon = trs["date"].str[:7]
    nd = trs["date"].map(trs["date"].value_counts()).to_numpy(float)
    nm = mon.map(trs.groupby(mon)["date"].nunique()).to_numpy(float)
    w = 1.0 / (nd * nm)
    return w / w.mean()


def w_evw(trs: pd.DataFrame, y: str = "y_up", seg: str = "seg_up") -> np.ndarray:
    """按事件等权：正样本行权重 = 1 / 段长，再在每个月内把正样本权重和拉回该月正样本行数（1:12 的正负比
    在权重意义上不变，只改正样本内部的分配），负样本 1；最后乘按月等权，均值 1。"""
    import model as M
    pos = trs[y].to_numpy() > 0
    s = trs[seg].to_numpy(float)
    assert np.isfinite(s[pos]).all(), "正样本行缺段长"
    w = np.ones(len(trs))
    w[pos] = 1.0 / s[pos]
    mon = trs["date"].str[:7].to_numpy()
    for m in np.unique(mon):
        idx = pos & (mon == m)
        if idx.any():
            w[idx] *= idx.sum() / w[idx].sum()
    w *= M.month_weights(trs)
    return w / w.mean()


def soft_sample(df: pd.DataFrame, y: str, seed: int, match: str = "y_up", cold_min: float = 0.5) -> pd.DataFrame:
    """回归目标的采样（第 2 周判断 3）：按 u 分层。每月 u > 0.8 的行（铰链非零的那层，当天后 20%）全留，
    u <= 0.8 的「冷」行随机取到该月总行数 = base 的 13 x 当月 y_up 正样本数（和 base 同一个月同样多的训练行），
    冷行至少取热行的 cold_min 倍（正样本稀少的月份不至于没有冷行）。两个软标签臂共用，同种子同行集合。"""
    rng = np.random.default_rng(seed)
    df = df[np.isfinite(df[y]) & np.isfinite(df["u_day"])].copy()
    df["_m"] = df["date"].str[:7]
    out = []
    for m, g in df.groupby("_m"):
        hot = g[g["u_day"] > 0.8]
        cold = g[g["u_day"] <= 0.8]
        if len(hot) == 0:
            continue
        n_base = 13 * int((g[match] > 0).sum())
        take = min(len(cold), max(n_base - len(hot), int(len(hot) * cold_min)))
        if take > 0:
            cold = cold.iloc[rng.choice(len(cold), size=take, replace=False)]
        out.append(pd.concat([hot, cold]))
    if not out:
        return df.iloc[:0]
    r = pd.concat(out, ignore_index=True).drop(columns=["_m"])
    return r.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class L1LgbmReg:
    """L1Lgbm 的回归版（参数照抄 model.L1Lgbm，只换 objective；不改 src）。predict_proba 返回回归值，排序用。"""
    name = "L1_lightgbm_reg"

    def __init__(self, objective: str = "huber", **kw):
        import model as M
        self.p = dict(objective=objective, n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=M.SEED,
                      n_jobs=4, verbose=-1)
        self.p.update(kw)
        self.m = None
        self.cols: list[str] = []

    def fit(self, df, cols, y, w=None):
        import lightgbm as lgb
        import model as M
        self.cols = cols
        X, yy = M._xy(df, cols, y)
        self.m = lgb.LGBMRegressor(**self.p)
        w = M.month_weights(df) if w is None else np.asarray(w, float)
        self.m.fit(X, yy, sample_weight=w)
        return self

    def predict_proba(self, df):
        import model as M
        X, _ = M._xy(df, self.cols, self.cols[0])
        return self.m.predict(X)


ARMS = {
    "base": {"y": "y_up", "w": None, "kw": {}, "cols": None, "desc": "现状：生产特征 + LightGBM 400 棵，按月等权"},
    # 第 2 周（设计 2.1 / 2.2 / 3.1 / 3.2）
    "eqd": {"y": "y_up", "w": w_eqd, "kw": {}, "cols": None, "desc": "按天等权（嵌在按月等权里：每月等权，月内每天等权）"},
    "evw": {"y": "y_up", "w": w_evw, "kw": {}, "cols": None, "need": ["seg_up"],
            "desc": "按事件等权（正样本行权重 1/连续段长，每月正样本总权重不变，乘按月等权）"},
    "y_rel4": {"y": "y_rel4", "w": None, "kw": {}, "cols": None, "desc": "当日相对标签：每天按 r20_open 排名前 4%（次日一字记最低），按月等权"},
    "y_rel2": {"y": "y_rel2", "w": None, "kw": {}, "cols": None, "desc": "当日相对标签前 2%，按月等权"},
    "y_softh": {"y": "y_softh", "w": None, "kw": {}, "cols": None, "need": ["u_day"], "sampler": soft_sample, "model": "reg",
                "desc": "软标签：尾部铰链 max(u-0.8,0)/0.2，huber 回归；按 u 分层采样（u>0.8 全留 + 冷行补到 base 的月行数），按月等权"},
    "y_softz": {"y": "y_softz", "w": None, "kw": {}, "cols": None, "need": ["u_day"], "sampler": soft_sample, "model": "reg",
                "desc": "软标签：当天 log1p(r20) 去 1% 极值标准化，huber 回归；采样同 y_softh"},
    "y_rel4_eqd": {"y": "y_rel4", "w": w_eqd, "kw": {}, "cols": None, "desc": "当日相对标签前 4% + 按天等权"},
}


# ---------------------------------------------------------------
#  数据
# ---------------------------------------------------------------
def load(months: list[str], arms: dict, tag: str = "") -> pd.DataFrame:
    feats = CS.SCR / f"prod_screen{tag}.parquet"
    labs = CS.SCR / f"labels_screen{tag}.parquet"
    lo = CS.month_shift(months[0], -12 * CS.TRAIN_YEARS) + "-01"
    hi = CS.month_shift(months[-1], 1) + "-01"
    df = pd.read_parquet(feats, filters=[("date", ">=", lo), ("date", "<", hi)])
    lab = pd.read_parquet(labs, columns=["code", "date", "y_up", "y_open", "r20_open", "yizi1"])
    df = df.merge(lab, on=["code", "date"], how="left")
    df["y_up"] = df["y_up"].astype("float32")
    df["y_open"] = df["y_open"].astype("float32")
    need = set()
    for a, spec in arms.items():
        need.add(spec.get("y", "y_up"))
        need.update(spec.get("need", []))
    for c in LABEL_ORDER:
        if c in need and c not in df.columns:
            t0 = time.time()
            df[c] = LABELS[c](df)
            print("  加列 %s %.0fs（非空 %d，均值 %.4f）" % (c, time.time() - t0, int(np.isfinite(df[c]).sum()),
                                                     float(np.nanmean(df[c]))), flush=True)
    assert df["date"].max() < K.CONFIRM[0], "确认集漏进来了"
    print("训练表 %d 行 %d 列 %s..%s %.1f GB" % (len(df), df.shape[1], df["date"].min(), df["date"].max(),
                                                df.memory_usage().sum() / 2 ** 30), flush=True)
    return df


def make_fit(arms: dict, seeds, name: str = ""):
    def fit_predict(tr, te, feats, month, state):
        import model as M
        out, rows = {}, {}
        for s in seeds:
            for name_, spec in arms.items():
                y = spec.get("y", "y_up")
                sampler = spec.get("sampler") or (lambda d, yy, seed: M.stratified_sample(d, yy, seed=seed))
                trs = sampler(tr[np.isfinite(tr[y])], y, s)
                ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
                cols = spec["cols"](feats) if spec.get("cols") else list(feats)
                assert len(set(cols)) == len(cols) and all(c in tr.columns for c in cols), name_   # 教训 32
                w = spec["w"](trs) if spec.get("w") else None
                mk = L1LgbmReg if spec.get("model") == "reg" else M.L1Lgbm
                m = mk(n_estimators=400, n_jobs=12, **spec.get("kw", {})).fit(trs, cols, y, w=w)
                rows[f"{name_}_s{s}"] = int(len(trs))
                out[f"{name_}_s{s}"] = (m.predict_proba(te), m.predict_proba(ref))
        if name:
            d = K.EXP / name / "parts" / month
            d.mkdir(parents=True, exist_ok=True)
            (d / "train_rows.json").write_text(json.dumps(rows), encoding="utf-8")
        return out
    return fit_predict


# ---------------------------------------------------------------
#  可续跑的逐月循环：每个月单独调一次 evalkit.walk_forward，产物落 parts/<月>/，最后拼起来
# ---------------------------------------------------------------
def run(name: str, arms: dict, seeds, months: list[str], df: pd.DataFrame, stop_in_prod: bool = True,
        drop_key: str = DROP_KEY) -> bool:
    parts = K.EXP / name / "parts"
    keys = [f"{a}_s{s}" for a in arms for s in seeds]
    t0 = time.time()
    for m in months:
        d = parts / m
        if all((d / f"top_{k}.parquet").exists() for k in keys):
            continue
        if stop_in_prod and CS.in_prod_window():
            print("北京 %s 在生产窗口里，停在 %s 之前（已完成的月份保留，重跑同一命令续跑）" % (CS.now_bj_str(), m), flush=True)
            return False
        lo = CS.month_shift(m, -12 * CS.TRAIN_YEARS) + "-01"
        hi = CS.month_shift(m, 1) + "-01"
        sub = df[(df["date"] >= lo) & (df["date"] < hi)]
        K.walk_forward(sub, [m], make_fit(arms, seeds, name), f"{name}/parts/{m}", drop=[drop_key])
        del sub
        print("[%s] %s 完成，累计 %.1f 分钟" % (name, m, (time.time() - t0) / 60), flush=True)
    # 拼
    qmap: dict = {}
    for k in keys:
        tops, alls = [], []
        for m in months:
            d = parts / m
            if (d / f"top_{k}.parquet").exists():
                tops.append(pd.read_parquet(d / f"top_{k}.parquet"))
                alls.append(pd.read_parquet(d / f"all_{k}.parquet"))
                qmap.setdefault(k, {}).update(json.loads((d / "q.json").read_text()).get(k, {}))
        if tops:
            pd.concat(tops, ignore_index=True).to_parquet(K.EXP / name / f"top_{k}.parquet", index=False)
            pd.concat(alls, ignore_index=True).to_parquet(K.EXP / name / f"all_{k}.parquet", index=False)
    (K.EXP / name / "q.json").write_text(json.dumps(qmap), encoding="utf-8")
    return True


def write_prereg(name: str, arms: dict, seeds, months: list[str]) -> None:
    f = K.EXP / name / "prereg.json"
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    f.write_text(json.dumps({"name": name, "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "hypothesis": HYP, "arms": {a: s["desc"] for a, s in arms.items()}, "seeds": list(seeds),
                             "span": [months[0], months[-1]], "family": CS.FAMILY, "train_years": CS.TRAIN_YEARS,
                             "metric": "top10 y_open 按天配对；辅助 top10 平均 log1p(r20_open)（次日一字记 0）",
                             "gate": "base 前 10 买得到命中 >= 2 x 当日全市场基准（按天平均）才作筛选器，否则只作同号检查",
                             "note": "不进主族 breakout_y50_top10；本地登记簿 research/screen/ledger.jsonl"},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    for a in arms:
        for s in seeds:
            CS.ledger(name, f"{a}_s{s}", "prereg", desc=arms[a]["desc"], span=f"{months[0]}..{months[-1]}")
    CS.progress("%s 预登记：臂 %s 种子 %s 区间 %s..%s（本地登记簿）" % (name, list(arms), list(seeds), months[0], months[-1]))


# ---------------------------------------------------------------
#  分析
# ---------------------------------------------------------------
def top(name: str, arm: str) -> pd.DataFrame:
    t = pd.read_parquet(K.EXP / name / f"top_{arm}.parquet")
    assert t["date"].max() < K.CONFIRM[0]
    return t


def with_r20(t: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    lab = pd.read_parquet(CS.SCR / f"labels_screen{tag}.parquet", columns=["code", "date", "r20_open", "yizi1"])
    t = t.merge(lab, on=["code", "date"], how="left")
    r = t["r20_open"].where(~t["yizi1"].fillna(False).astype(bool), 0.0)
    t["lr20"] = np.log1p(r.clip(lower=-0.99))
    return t


def daily_hit(t: pd.DataFrame, k: int = 10, y: str = "y_open") -> pd.Series:
    return t[t["rank"] <= k].groupby("date")[y].mean()


def stat_line(d: pd.Series) -> str:
    bb = hac.mbb(d.to_numpy())
    se = CS.nw_se(d.to_numpy())
    return "均值 %+.3fpp NW se %.3fpp NW t %.2f (10/30: %.2f/%.2f) iid t %.2f 块自举 p %.3f [%+.2f, %+.2f] %d 天" % (
        100 * d.mean(), 100 * se, K.nw_t(d.to_numpy()), K.nw_t(d.to_numpy(), 10), K.nw_t(d.to_numpy(), 30),
        float(hac.iid_t(d.to_numpy())["t"]), bb["p"], 100 * bb["lo"], 100 * bb["hi"], len(d))


def screen_cmp(a: pd.DataFrame, b: pd.DataFrame, k: int = 10) -> dict:
    """第 2 周用：臂 a 对 b 在筛选集上的按天配对（主尺子命中 + 辅助尺子 lr20），返回可直接喂给
    evalkit.prereg(screen={臂: ...}) 的 dict（diff_day / days / span）和 status。a / b 要先过 with_r20。"""
    d = (daily_hit(a, k) - daily_hit(b, k)).dropna().sort_index()
    dl = (daily_hit(a, k, "lr20") - daily_hit(b, k, "lr20")).dropna().sort_index()
    r = K.screen_result((a, b), "y_open", k)
    r.update({"t_nw": K.nw_t(d.to_numpy()), "se_nw": CS.nw_se(d.to_numpy()), "boot": hac.mbb(d.to_numpy()),
              "aux_diff": float(dl.mean()), "aux_t_nw": K.nw_t(dl.to_numpy()), "aux_se_nw": CS.nw_se(dl.to_numpy()),
              "hit_a": float(daily_hit(a, k).mean()), "hit_b": float(daily_hit(b, k).mean())})
    return r


def analyze(name: str = NAME, seeds=SEEDS_DEFAULT, tag: str = "", book: bool = True, months: list[str] | None = None):
    lines: list[str] = []

    def L(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    months = months or CS.months_between(*K.SCREEN)
    have = sorted(p.stem[4:] for p in (K.EXP / name).glob("top_*.parquet"))
    T = {a: with_r20(top(name, a), tag) for a in have}
    L("实验 %s，臂：%s，区间 %s..%s" % (name, ", ".join(have), months[0], months[-1]))
    # 基准：和走向前同一个行集合（y_up、y_open 都有）
    feats_f = CS.SCR / f"prod_screen{tag}.parquet"
    rows = pd.read_parquet(feats_f, columns=["code", "date", "board"],
                           filters=[("date", ">=", months[0] + "-01"), ("date", "<", CS.month_shift(months[-1], 1) + "-01")])
    lab = pd.read_parquet(CS.SCR / f"labels_screen{tag}.parquet", columns=["code", "date", "y_up", "y_open", "r20_open", "yizi1"])
    rows = rows.merge(lab, on=["code", "date"], how="left")
    rows = rows[np.isfinite(rows["y_up"]) & np.isfinite(rows["y_open"])]
    rows["lr20"] = np.log1p(rows["r20_open"].where(~rows["yizi1"].fillna(False).astype(bool), 0.0).clip(lower=-0.99))
    base_day = rows.groupby("date")["y_open"].mean()
    base_pool = float(rows["y_open"].mean())
    base_close = float(rows["y_up"].mean())
    base_aux = rows.groupby("date")["lr20"].mean()
    L("全市场基准（可打分行 %d，%d 天，%d 只）：买得到口径 按天平均 %.3f%% / 合并 %.3f%%；收盘口径合并 %.3f%%；"
      "前 10 lr20 的基准（全市场按天平均 log1p(r20)）%.4f" % (
          len(rows), base_day.size, rows["code"].nunique(), 100 * base_day.mean(), 100 * base_pool, 100 * base_close,
          base_aux.mean()))
    by_board = rows.groupby("board")["y_open"].agg(["mean", "size"])
    L("  按板块：" + "  ".join("%s %.2f%% (n %d)" % (b, 100 * r.loc["mean"], r.loc["size"]) for b, r in by_board.iterrows()))
    half = rows["date"].str[:4] + "H" + ((rows["date"].str[5:7].astype(int) > 6).astype(int) + 1).astype(str)
    hb = rows.groupby(half)["y_open"].mean()
    L("  按半年：" + "  ".join("%s %.2f%%" % (k, 100 * v) for k, v in hb.items()))
    res: dict = {"base_day": float(base_day.mean()), "base_pool": base_pool, "days": int(base_day.size)}

    # 各臂成绩
    for a, t in T.items():
        dh = daily_hit(t)
        m = K.metrics(t)
        hb_a = t[t["rank"] <= 10].groupby(half.reindex(t.index).fillna("?") if False else t["date"].str[:4] + "H" + ((t["date"].str[5:7].astype(int) > 6).astype(int) + 1).astype(str))["y_open"].mean()
        L("%-10s 前 10 买得到 按天 %.2f%%（NW se %.2fpp）合并 %.2f%% 收盘 %.2f%% | 生产规则 %.2f%% (n %d, 空榜 %d/%d) | "
          "前 30 %.2f%% | 月中位 %.1f%% 最差 %.1f%% | 前 10 lr20 %.4f | 倍数（按天）%.2f" % (
              a, 100 * dh.mean(), 100 * CS.nw_se(dh.to_numpy()), 100 * m["top10_open"]["hit"], 100 * m["top10_close"]["hit"],
              100 * m["prod_open"]["hit"], m["prod_open"]["n"], m["prod_empty"], m["days"], 100 * m["top30_open"]["hit"],
              100 * m["month_median"], 100 * m["month_worst"], daily_hit(t, 10, "lr20").mean(), dh.mean() / base_day.mean()))
        L("  按半年：" + "  ".join("%s %.2f%%" % (k, 100 * v) for k, v in hb_a.items()))
        res[a] = {"hit_day": float(dh.mean()), "hit_pool": m["top10_open"]["hit"], "lift_day": float(dh.mean() / base_day.mean()),
                  "prod_hit": m["prod_open"]["hit"], "prod_n": m["prod_open"]["n"], "aux": float(daily_hit(t, 10, "lr20").mean())}
    # 决策门：base 多种子平均
    bases = [a for a in have if a.startswith("base_s")]
    if bases:
        hit_avg = float(np.mean([res[a]["hit_day"] for a in bases]))
        lift = hit_avg / float(base_day.mean())
        gate = lift >= 2.0
        L("[决策门] base 前 10 买得到命中 %.2f%%（种子平均）对 全市场基准 %.2f%% = %.2f 倍 -> %s" % (
            100 * hit_avg, 100 * base_day.mean(), lift, "过（筛选集可作筛选器）" if gate else "不过（只作同号检查）"))
        res["gate"] = {"hit": hit_avg, "base": float(base_day.mean()), "lift": lift, "pass": bool(gate)}
    # 种子噪声 -> MDE
    if len(bases) >= 2:
        a, b = bases[0], bases[1]
        d = (daily_hit(T[a]) - daily_hit(T[b])).dropna().sort_index()
        dl = (daily_hit(T[a], 10, "lr20") - daily_hit(T[b], 10, "lr20")).dropna().sort_index()
        ja = T[a][T[a]["rank"] <= 10].groupby("date")["code"].apply(set)
        jb = T[b][T[b]["rank"] <= 10].groupby("date")["code"].apply(set)
        jac = float(np.mean([len(ja[k_] & jb[k_]) / len(ja[k_] | jb[k_]) for k_ in ja.index if k_ in jb.index]))
        se, se_l = CS.nw_se(d.to_numpy()), CS.nw_se(dl.to_numpy())
        L("[种子噪声] %s - %s：%s；前 10 名单日均 Jaccard %.2f" % (a, b, stat_line(d), jac))
        L("[种子噪声] 辅助尺子 lr20：%s" % stat_line(dl))
        th = K.family_threshold()
        cs = [2.5, 2.8, round(th["c"], 2)]
        L("[MDE 换种子级（同特征同参数，重合 %.0f%%），功效 80%%，效应 >= (c+0.84) x NW se] 命中：%s；lr20：%s" % (
            100 * jac, "  ".join("c=%.2f %+.2fpp" % (c, 100 * (c + 0.84) * se) for c in cs),
            "  ".join("c=%.2f %+.4f" % (c, (c + 0.84) * se_l) for c in cs)))
        # 其它两类臂的 se 只能从开发集按天数外推（power.txt：387 天 大改 1.85pp / 增量 0.87pp / 种子 0.46pp）
        n = len(d)
        k_ = float(np.sqrt(387 / n))
        L("[外推] 开发集 387 天的 NW se（power.txt）按 sqrt(387/%d)=%.2f 缩：特征大改 %.2fpp 增量臂 %.2fpp 换种子 %.2fpp；"
          "本筛选集实测换种子 %.2fpp（外推 / 实测 = %.2f）" % (n, k_, 1.85 * k_, 0.87 * k_, 0.46 * k_, 100 * se, 0.46 * k_ / (100 * se)))
        ratio = (100 * se) / (0.46 * k_)
        L("[MDE 增量臂（按实测/外推比例 %.2f 校正开发集的 0.87pp x %.2f）] %s" % (
            ratio, k_, "  ".join("c=%.2f %+.2fpp" % (c, (c + 0.84) * 0.87 * k_ * ratio) for c in cs)))
        L("[MDE 特征大改] %s" % "  ".join("c=%.2f %+.2fpp" % (c, (c + 0.84) * 1.85 * k_ * ratio) for c in cs))
        res["seed_noise"] = {"diff": float(d.mean()), "se_nw": float(se), "t_nw": K.nw_t(d.to_numpy()), "jaccard": jac,
                             "aux_se_nw": float(se_l), "days": n,
                             "mde_seed": {str(c): float((c + 0.84) * se) for c in cs},
                             "mde_incr_est": {str(c): float((c + 0.84) * 0.0087 * k_ * ratio) for c in cs},
                             "mde_big_est": {str(c): float((c + 0.84) * 0.0185 * k_ * ratio) for c in cs}}
        # 和 compare 的口径对一遍（不带 exp/arm：不写登记簿）
        c2 = K.compare(T[a], T[b])
        L("[evalkit.compare 口径] %s 对 %s：按天 %+.2fpp NW t %.2f 块自举 p %.3f 滚动三月最低 %+.1f@%s 门槛 %.2f（%s）-> v3 %s" % (
            a, b, 100 * c2["diff_day"], c2["t_nw"], c2["boot"]["p"], 100 * c2["roll3"]["min"], c2["roll3"]["end"],
            c2["threshold"]["c"], c2["threshold"]["method"], c2["verdict_v3"]))
    # 月度
    rows_m = {a: t[t["rank"] <= 10].groupby(t["date"].str[:7])["y_open"].mean() for a, t in T.items()}
    rows_m["全市场"] = base_day.groupby(base_day.index.str[:7]).mean()
    L("月度前 10 买得到命中（%）：")
    L((100 * pd.DataFrame(rows_m)).round(1).to_string())
    txt = "\n".join(lines)
    (CS.HERE / ("results_screen.txt" if book else "results_dbg.txt")).write_text(txt, encoding="utf-8")
    if book:
        f = K.EXP / name / "prereg.json"
        j = json.loads(f.read_text(encoding="utf-8"))
        import datetime as dt
        j.setdefault("results", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"), **res})
        f.write_text(json.dumps(j, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        for a in have:
            CS.ledger(name, a, "result", hit_day=res[a]["hit_day"], lift=res[a]["lift_day"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--arms", nargs="*", default=["base"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS_DEFAULT))
    ap.add_argument("--span", nargs=2, default=list(K.SCREEN), help="测试月区间，默认筛选集")
    ap.add_argument("--debug", action="store_true", help="伪造的 150 只小表、2 个月，不登记")
    a = ap.parse_args()
    arms = {k: ARMS[k] for k in a.arms}
    if a.debug:
        months = ["2024-06", "2024-07"]
        df = load(months, arms, tag="_fake")
        # 调试用自己的缓存命名空间，别把伪造表上选出来的列表写进 screen_v1 的键
        run("screen_dbg", arms, a.seeds, months, df, stop_in_prod=False, drop_key="screen_dbg")
        analyze("screen_dbg", a.seeds, tag="_fake", book=False, months=months)
        return
    months = CS.months_between(*a.span)
    if a.fit:
        write_prereg(a.name, arms, a.seeds, months)
        df = load(months, arms)
        ok = run(a.name, arms, a.seeds, months, df)
        CS.progress("%s 走向前 %s（臂 %s 种子 %s）" % (a.name, "跑完" if ok else "停在生产窗口前，待续", list(arms), a.seeds))
        if not ok:
            return
    if a.analyze:
        analyze(a.name, a.seeds, months=months)


if __name__ == "__main__":
    main()
