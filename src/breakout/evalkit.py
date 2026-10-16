"""
评估制度（docs/breakout_plan_2026-10.md 第 1 节）。所有起涨预测实验从 2026-10-07 起走这里。

三层数据
--------
    开发集  DEV      2024-06 .. 2025-12  19 个月逐月走向前。探索都在这里
    确认集  CONFIRM  2026-01 .. 2026-08  候选在开发集过线后看一次，每季度最多一次，
                                         看了就记进 out_breakout/confirm_used.json
    实盘             state/breakout/truth.json 按月累计，最终裁判

旧验证集 2025-03..12 已被实验 8 到 16 约 30 个臂反复选择，再在它上面选出来的东西都带偏差。
开发集把起点提前到 2024-06（走向前前几个月训练数据少，两个臂同等吃亏，配对比较不受影响；
绝对数字看 2025 年起的那段）。

尺子
----
    主：买得到口径 y_open（次日开盘起算涨超 50%，次日一字板记不可买，buyable.open_labels）
    并列：收盘口径 y_up
    标准误按天聚类；臂之间按天配对（t_day）和按月配对（t_month，每月等权）
    月度分布：中位、最差、赢月数；空榜天数

过线标准（gate，2026-10-16 起 v3，docs/design_2026-10-16.md 7.2，用户拍板 A；v2 见 research/stats/report.md 第 5 节）
---------------
标签是未来 20 根的最高价，相邻交易日的结果高度重叠，按天配对的 iid t 平均被放大约 1.5 倍；
开发集上又已经试了近 90 次。所以七条全满足才「过线」，NW t <= -门槛记「不过线」，其余「未定」：
    1 NW t（Bartlett 核，滞后 20 = 标签窗口）>= c，c = max(2.5, 族门槛)。族门槛见下
    2 块自举 p < 0.05（块长 20，5000 次）
    3 去掉最好的一个月后 NW t >= 2.0（防一个月说了算）
    4 合并命中之差和按天配对均值同号（教训 33）
    5 任意连续 3 个月的滚动平均差 >= -5pp（2024-09 照算；「去掉 2024-09」并列印出）。
      v2 是「最差三个月 >= 0」：月差噪声 sd 10pp 时真 +5pp 也只有 0.1% 能过（research/design/power.txt），
      改成滚动三月后真 +5pp 约 75% 过、真 0 约 28% 误放（research/design/power_roll3.txt）
    6 空榜天数不翻倍
    7 筛选集同号：臂在扩展筛选集 SCREEN（2022-01..2024-05）上按天差 >= 0。compare / prereg 的 screen 参数
      没给就记「筛选集：无」，v3 不能过线（教训 32：没测过的不算通过）
v2（六条，第 5 条是最差三月）和 v3 两套判定每次都算；哪套算「正式」看预登记时的版本
（prereg.json 的 gate_version，2026-10-16 前登记的没有这个键 = v2）。gate / pass / verdict 是正式那套，
gate_v2 / verdict_v2 / gate_v3 / verdict_v3 并列。旧的 t_day >= 2.5 判法留在 gate_v1 / verdict_v1 里对账。

族门槛（family_threshold）
    登记簿里本族 m 次试验，其中 s 次有逐日差值序列、u = m - s 次没有。零假设下「m 次里最大的 t 超过 c」
    的概率用并集上界：P(max_s >= c) + u * (1 - Φ(c))，前一项是 s 条序列各自去均值后按天整行块自举
    （保留臂间相关和自相关）的最大 t 尾概率，后一项把没序列的试验按 Bonferroni 算。c 取让这个和 <= 0.05
    的最小值。s = 0 时就是 Bonferroni Φ⁻¹(1 - 0.05/m)，s = m 时就是族内最大 t 零分布的 95% 分位，
    中间连续：多登一条序列最多把门槛降一个 Bonferroni 项（m 对 m-1 的差，m=70 时 0.01），不再有
    「2 条用 Bonferroni 3.19、3 条用最大 t 2.50」的跳变（research/leak/report.md 第 7 节）。
    序列短于 MIN_SERIES_DAYS 的不进零分布（当成没序列）。

试验登记簿
----------
    data/breakout/raw/exp/ledger.jsonl：每次 prereg 每个臂记一行，每次事后加看子区间（compare 的
    since 不是预登记的区间）也记一行。门槛按次数抬高。第一次用时从 research/stats/out/ledger_seed.jsonl
    补登历史（实验 15~21 的 47 条 + 4 组只有计数的，合计约 87 次）。
    kind=backfill 的行（count=0）只给历史试验补上逐日差值序列，不算新试验（research/stats/backfill_ledger.py，
    2026-10-16）；kind=result 的行也不算。序列文件 diff_*.parquet 带日期列，按日期对齐；
    2026-10-16 前的 .npy 没有日期，按「和本族日历末尾对齐」读。

预登记
------
    每个实验先 prereg(name, hypothesis, arms, gate, screen) 再跑；文件已存在就拒绝改写，
    结果 append 进同一个 json。写进 docs/breakout_log.md 的条目要和它一致。
    screen = {臂: 筛选集按天差} 给了，按天差 < 0 的臂拒绝登记（规则 7）；没给照登，但 prereg.json
    记 screen 为空，compare 的 v3 就过不了线。

特征选择
--------
    和生产一样每个月在净化过的训练切片上重选（fselect.run），按月缓存在
    data/breakout/raw/exp/feats_by_month.json，多个臂共用。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "breakout"
EXP = DATA / "raw" / "exp"
CONFIRM_USED = ROOT / "out_breakout" / "confirm_used.json"
DEV = ("2024-06", "2025-12")
CONFIRM = ("2026-01", "2026-08")
GATE = {"t_day": 2.5, "worst3_min_diff": 0.0, "empty_ratio_max": 2.0}
TOP_KEEP = 200
LEDGER = EXP / "ledger.jsonl"
LEDGER_SEED = ROOT / "research" / "stats" / "out" / "ledger_seed.jsonl"
FAMILY_DEFAULT = "breakout_y50_top10"
NW_LAG, BLOCK, BOOT = 20, 20, 5000
GATE_V2 = {"t_floor": 2.5, "boot_p": 0.05, "drop_best_month_t": 2.0, "fwer": 0.05,
           "worst3_min_diff": 0.0, "empty_ratio_max": 2.0}
GATE_V3 = {"t_floor": 2.5, "boot_p": 0.05, "drop_best_month_t": 2.0, "fwer": 0.05,
           "roll3_min_diff": -0.05, "empty_ratio_max": 2.0, "screen_min_diff": 0.0}
GATE_VERSION = 3                      # 2026-10-16 起新预登记的实验用 v3
SCREEN = ("2022-01", "2024-05")       # 扩展筛选集（规则 7）
NULL_BOOT = 4000                      # 族门槛零分布的自举次数（2000 次时 95% 分位的蒙特卡洛误差约 ±0.05）
MIN_SERIES_DAYS = 100                 # 短于这个的差值序列不进零分布
log = logging.getLogger("evalkit")


# ---------------------------------------------------------------
#  数据
# ---------------------------------------------------------------
def months_in(span: tuple[str, str]) -> list[str]:
    a, b = span
    out, y, m = [], int(a[:4]), int(a[5:7])
    while f"{y:04d}-{m:02d}" <= b:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def load_frame(with_confirm: bool = False, experiment: str = "",
               candidate: str = "") -> pd.DataFrame:
    """训练表 + 买得到标签。with_confirm=False 时 2026 年整段不可见。

    with_confirm=True 要看确认集：必须说明是哪个实验的哪个候选，并且先过 confirm_guard
    （本季度看过就拒绝，看了就记账）。纪律写成代码，和 arena 的封存数据同一个思路。"""
    import arena as A
    import buyable as B
    if with_confirm:
        if not (experiment and candidate):
            raise SystemExit("看确认集要写明 experiment 和 candidate（记进 confirm_used.json）")
        confirm_guard(experiment, candidate)
    df = A.load(with_holdout=with_confirm)
    df["code"] = df["code"].astype(str).str.zfill(6)
    lab = B.open_labels()
    df = df.merge(lab[["code", "date", "y_open", "r20_open"]], on=["code", "date"], how="left")
    if with_confirm:
        # 确认集之后的月份（2026-09 起）标签窗口没关，照样切掉
        df = df[df["date"] < CONFIRM[1] + "-32"]
    return df


def feats_for(df: pd.DataFrame, month: str, drop: list[str] | None = None) -> list[str]:
    """该月走向前用的特征列：净化过的训练切片上重选，按月缓存。"""
    import fselect as FS
    import validate as V
    EXP.mkdir(parents=True, exist_ok=True)
    key = month + ("|" + ",".join(sorted(drop)) if drop else "")
    f = EXP / "feats_by_month.json"
    cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if key in cache:
        return list(cache[key])
    feats_all = [c for c in df.columns if "__" in c]
    if drop:
        ds = set(drop)
        feats_all = [c for c in feats_all if c not in ds and c.rsplit("__", 1)[0] not in ds]
    tr = V.train_slice(df, month)
    tr = tr[np.isfinite(tr["y_up"])]
    rep = FS.run(tr, feats_all, y="y_up")
    cache[key] = list(rep["keep"])
    f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return list(rep["keep"])


# ---------------------------------------------------------------
#  走向前
# ---------------------------------------------------------------
def walk_forward(df: pd.DataFrame, months: list[str], fit_predict, name: str,
                 drop: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """逐月：fit_predict(tr, te, feats, month, state) -> {臂名: (p_te, p_ref)}。

    p_ref 是该臂参照样本上的预测值，用来定 0~100 分（和 daily.load_or_fit 同构）。
    state 是跨月共享的 dict（模型平均之类的臂用它存上几个月的模型）。
    每个臂落两份：top_<臂>.parquet（每天前 TOP_KEEP，带标签）、all_<臂>.parquet（全市场 p）。
    """
    import validate as V
    out_dir = EXP / name
    out_dir.mkdir(parents=True, exist_ok=True)
    st = V.st_codes()
    tops: dict[str, list] = {}
    alls: dict[str, list] = {}
    qmap: dict[str, dict] = {}
    state: dict = {}
    t0 = time.time()
    for m in months:
        tr = V.train_slice(df, m)
        tr = tr[np.isfinite(tr["y_up"])]
        te = df[df["date"].str[:7] == m].copy()
        te = te[np.isfinite(te["y_up"]) & np.isfinite(te["y_open"])]
        if len(tr) < 5000 or not len(te):
            log.warning("%s 训练 %d 行 / 测试 %d 行，跳过", m, len(tr), len(te))
            continue
        assert str(tr["date"].max()) < V.purge_cut(df, m), "训练样本越过净化线"
        feats = feats_for(df, m, drop)
        res = fit_predict(tr, te, feats, m, state)
        for arm, (p_te, p_ref) in res.items():
            q = np.quantile(np.asarray(p_ref, float), np.linspace(0, 1, 101))
            qmap.setdefault(arm, {})[m] = [float(x) for x in q]
            o = te[["date", "code", "board", "y_up", "y_open"]].copy()
            o["p"] = np.asarray(p_te, np.float32)
            alls.setdefault(arm, []).append(o[["date", "code", "p"]])
            o = o[~o["code"].isin(st)]
            o["score"] = np.clip(np.searchsorted(q, o["p"]), 0, 100)
            o = o.sort_values(["date", "p"], ascending=[True, False])
            o["rank"] = o.groupby("date").cumcount() + 1
            tops.setdefault(arm, []).append(o[o["rank"] <= TOP_KEEP])
        print("  %s 完成 %.0fs（特征 %d 列）" % (m, time.time() - t0, len(feats)), flush=True)
    (out_dir / "q.json").write_text(json.dumps(qmap), encoding="utf-8")
    out = {}
    for arm in tops:
        t = pd.concat(tops[arm], ignore_index=True)
        t.to_parquet(out_dir / f"top_{arm}.parquet", index=False)
        pd.concat(alls[arm], ignore_index=True).to_parquet(out_dir / f"all_{arm}.parquet", index=False)
        out[arm] = t
    return out


# ---------------------------------------------------------------
#  尺子
# ---------------------------------------------------------------
def clustered(picks: pd.DataFrame, y: str = "y_open") -> dict:
    if not len(picks):
        return {"n": 0, "hit": float("nan"), "se": float("nan"), "days": 0}
    g = picks.groupby("date")[y].agg(["sum", "size"])
    N, K = float(g["size"].sum()), float(g["sum"].sum())
    p = K / N
    var = float(((g["sum"] - p * g["size"]) ** 2).sum()) / N ** 2
    return {"n": int(N), "hit": p, "se": float(np.sqrt(var)), "days": int(len(g))}


def monthly(picks: pd.DataFrame, y: str = "y_open") -> pd.Series:
    if not len(picks):
        return pd.Series(dtype=float)
    return picks.groupby(picks["date"].str[:7])[y].mean()


def prod_picks(top: pd.DataFrame, score_min: int | None = None, cap: int | None = None) -> pd.DataFrame:
    import daily as D
    sm = D.SCORE_MIN if score_min is None else score_min
    n = D.CAP_A if cap is None else cap
    return top[(top["score"] >= sm) & (top["rank"] <= n)]


def metrics(top: pd.DataFrame, since: str = "") -> dict:
    """一个臂的成绩：生产规则 / 前 10 / 前 30，两种口径，月度分布。since 给了只算之后。"""
    t = top[top["date"] >= since] if since else top
    days = t["date"].nunique()
    prod = prod_picks(t)
    out = {"days": int(days)}
    for lbl, key in (("y_open", "open"), ("y_up", "close")):
        out[f"prod_{key}"] = clustered(prod, lbl)
        out[f"top10_{key}"] = clustered(t[t["rank"] <= 10], lbl)
        out[f"top30_{key}"] = clustered(t[t["rank"] <= 30], lbl)
    out["prod_empty"] = int(days - prod["date"].nunique())
    out["prod_per_day"] = float(len(prod) / max(days, 1))
    mo = monthly(t[t["rank"] <= 10])
    out["monthly_top10_open"] = {k: float(v) for k, v in mo.items()}
    out["month_median"] = float(mo.median()) if len(mo) else float("nan")
    out["month_worst"] = float(mo.min()) if len(mo) else float("nan")
    return out


# ---------------------------------------------------------------
#  稳健推断（v2）：NW t、块自举、登记簿、族门槛。公式和 research/stats/hac.py 同一份
# ---------------------------------------------------------------
def _autocov(x: np.ndarray, k: int) -> float:
    xc = x - x.mean()
    return float((xc[k:] * xc[:len(x) - k]).sum() / len(x))


def nw_t(d, lag: int = NW_LAG) -> float:
    """Newey-West t（Bartlett 核）：长期方差 = γ0 + 2 Σ (1 - k/(L+1)) γk。样本太短返回 nan。"""
    x = np.asarray(d, float)
    x = x[np.isfinite(x)]
    se = _nw_se(x, lag)
    return float(x.mean() / se) if np.isfinite(se) else float("nan")


def _mbb_index(n: int, block: int, b: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(b, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(b, nb * block)[:, :n]


def block_boot(d, block: int = BLOCK, b: int = BOOT, seed: int = 20261014) -> dict:
    """均值的移动块自举：双侧中心化 p 和 95% 区间。"""
    x = np.asarray(d, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * block:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    ms = x[_mbb_index(n, block, b, np.random.default_rng(seed))].mean(axis=1)
    m = float(x.mean())
    lo, hi = np.quantile(ms, [0.025, 0.975])
    return {"p": max(float((np.abs(ms - m) >= abs(m)).mean()), 1.0 / b), "lo": float(lo), "hi": float(hi)}


def _ledger_rows() -> list[dict]:
    if not LEDGER.exists() and LEDGER_SEED.exists():
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(LEDGER_SEED.read_text(encoding="utf-8"), encoding="utf-8")
    if not LEDGER.exists():
        return []
    out = []
    for ln in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    return out


def ledger_add(row: dict) -> None:
    _ledger_rows()                      # 第一次用时先补登历史
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), **row},
                            ensure_ascii=False) + "\n")


NOT_TRIALS = ("result", "backfill")   # 登记簿里不算新试验的行


def family_count(family: str = FAMILY_DEFAULT) -> int:
    """登记簿里这一族试过多少次（带 count 的补登行按 count 计）。结果行、补登序列行不算新的试验。"""
    return int(sum(int(r.get("count", 1)) for r in _ledger_rows()
                   if r.get("family") == family and r.get("kind") not in NOT_TRIALS))


def _load_series(p: Path):
    """差值序列文件 -> pd.Series（parquet 带日期索引）或 np.ndarray（旧 .npy，没日期）。"""
    if p.suffix == ".parquet":
        t = pd.read_parquet(p)
        return pd.Series(t["d"].to_numpy(float), index=t["date"].astype(str).to_numpy())
    return np.load(p)


def _family_series(family: str) -> pd.DataFrame:
    """本族全部逐日差值序列，按日期对齐成一张表（行 = 日，列 = 序列；某天没值 = NaN）。

    同一个文件只算一次（同一个臂 compare 两次登记簿会有两行结果）。.npy 旧格式没有日期，
    按「和本族日历末尾对齐」放；族里一条带日期的都没有时用位置当日历。"""
    paths: list[Path] = []
    for r in _ledger_rows():
        p = r.get("series")
        if r.get("family") == family and p and Path(p) not in paths and Path(p).exists():
            paths.append(Path(p))
    dated, undated = {}, {}
    for p in paths:
        x = _load_series(p)
        if len(x) < MIN_SERIES_DAYS:
            continue
        (dated if isinstance(x, pd.Series) else undated)[str(p)] = x
    cal = sorted(set().union(*[set(s.index) for s in dated.values()])) if dated else []
    if undated:
        n = max(len(x) for x in undated.values())
        if len(cal) < n:
            cal = [f"#{i:05d}" for i in range(n - len(cal))] + list(cal)
        for k, x in undated.items():
            dated[k] = pd.Series(np.asarray(x, float), index=cal[-len(x):])
    return pd.DataFrame(dated, index=cal) if dated else pd.DataFrame()


def _maxt_null(D: pd.DataFrame, b: int = NULL_BOOT, seed: int = 7) -> np.ndarray:
    """族内最大 t 的零分布：各列去均值，按天整行块重抽，t* = 重抽均值 / 原序列的 NW se。
    缺值在去均值之后按 0 填（那天不贡献），均值按有值的天数算。返回每次重抽的最大 t。"""
    X = D.to_numpy(float)
    mask = np.isfinite(X)
    se = np.array([_nw_se(D[c].dropna().to_numpy()) for c in D.columns])
    Xc = np.where(mask, X - np.nanmean(X, axis=0), 0.0)
    n = len(X)
    idx = _mbb_index(n, BLOCK, b, np.random.default_rng(seed))
    out = np.empty(b)
    for i in range(0, b, 250):
        ii = idx[i:i + 250]
        s = Xc[ii].sum(axis=1)                                   # (块, 列)
        cnt = np.maximum(mask[ii].sum(axis=1), 1)
        out[i:i + 250] = np.nanmax(s / cnt / se, axis=1)
    return out


def _nw_se(x: np.ndarray, lag: int = NW_LAG) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < lag + 3:
        return float("nan")
    lrv = _autocov(x, 0) + sum(2 * (1 - k / (lag + 1)) * _autocov(x, k) for k in range(1, lag + 1))
    return float(np.sqrt(max(lrv, 1e-18) / n))


_TH_CACHE: dict = {}


def family_threshold(family: str = FAMILY_DEFAULT) -> dict:
    """族门槛 c。见模块说明「族门槛」一段：P(max_s >= c) + u (1 - Φ(c)) <= fwer 的最小 c，再和 2.5 取大。

    返回 c / method / m（本族试验次数）/ s（有序列的）/ u（没序列的）/ c_bonf（纯 Bonferroni）/
    c_maxt（只看有序列那部分的 95% 分位，对账用）。"""
    from statistics import NormalDist
    nd = NormalDist()
    fwer, floor = GATE_V3["fwer"], GATE_V3["t_floor"]
    m = max(family_count(family), 1)
    D = _family_series(family)
    s = int(D.shape[1]) if len(D) else 0
    u = max(m - s, 0)
    c_bonf = nd.inv_cdf(1 - fwer / m)
    if s == 0:
        return {"c": max(floor, c_bonf), "method": f"Bonferroni（m={m}）", "m": m, "s": 0, "u": u,
                "c_bonf": c_bonf, "c_maxt": float("nan")}
    key = (family, m, tuple(sorted(D.columns)), tuple(int(Path(p).stat().st_mtime_ns) for p in sorted(D.columns)))
    if key not in _TH_CACHE:
        _TH_CACHE[key] = _maxt_null(D)
    mx = _TH_CACHE[key]
    c_maxt = float(np.quantile(mx, 1 - fwer))

    def excess(c: float) -> float:
        return float((mx >= c).mean()) + u * (1 - nd.cdf(c))

    lo, hi = 0.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if excess(mid) <= fwer:
            hi = mid
        else:
            lo = mid
    # Bonferroni 本身就是 m 个正态尾的并集上界，自举尾比正态尾略胖时并集解会高出它几厘，取小者
    c = min(hi, c_bonf)
    return {"c": max(floor, c), "method": f"最大 t 并集上界（{s} 条序列 + {u} 次无序列，m={m}）",
            "m": m, "s": s, "u": u, "c_bonf": c_bonf, "c_maxt": c_maxt}


def backfill_series(family: str, exp, arm: str, ctrl: str, span: str, d: pd.Series, source: str,
                    out_dir: Path | None = None, label: str = "y_open", k=10) -> Path | None:
    """给登记簿里已经计数的历史试验补上逐日差值序列（kind=backfill，count=0，不算新试验）。

    d：按天配对差，索引是日期字符串。同一个 (exp, arm, span) 补过就不再补（返回 None）。
    文件写 EXP/backfill/diff_<exp>_<arm>_<span>_<k>_<label>.parquet。"""
    for r in _ledger_rows():
        if (r.get("kind") == "backfill" and r.get("family") == family and str(r.get("exp")) == str(exp)
                and r.get("arm") == arm and r.get("span") == span):
            return None
    out_dir = out_dir or (EXP / "backfill")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"diff_{exp}_{arm}_{span}_{k}_{label}.parquet"
    pd.DataFrame({"date": [str(i) for i in d.index], "d": d.to_numpy(float)}).to_parquet(p, index=False)
    ledger_add({"family": family, "exp": exp, "arm": arm, "ctrl": ctrl, "span": span, "label": label,
                "kind": "backfill", "count": 0, "series": str(p), "days": int(len(d)),
                "t_nw": nw_t(d.to_numpy()), "source": source})
    return p


def roll3_min(md: pd.Series) -> tuple[float, str]:
    """规则 5（v3）：月差序列上任意连续 3 个月的滚动平均的最小值，和那个窗口的末月。
    不足 3 个月就取全部的均值（窗口 = 全部）。空序列返回 nan。"""
    md = md.sort_index()
    if not len(md):
        return float("nan"), ""
    if len(md) < 3:
        return float(md.mean()), str(md.index[-1])
    r = md.rolling(3).mean().dropna()
    return float(r.min()), str(r.idxmin())


def screen_result(screen, y: str = "y_open", k: int = 10) -> dict:
    """规则 7 的输入整理。screen 可以是 None（没测）、{"diff_day": .., "days": ..} 或 (a_s, b_s) 两张筛选集上的
    名次表。返回 status = 无 / 同号 / 异号 和按天差。"""
    if screen is None:
        return {"status": "无", "diff_day": float("nan"), "days": 0, "span": list(SCREEN)}
    if isinstance(screen, dict):
        dd = float(screen.get("diff_day", float("nan")))
        days = int(screen.get("days", 0))
        span = screen.get("span", list(SCREEN))
    else:
        a_s, b_s = screen
        da = a_s[a_s["rank"] <= k].groupby("date")[y].mean()
        db = b_s[b_s["rank"] <= k].groupby("date")[y].mean()
        d = (da - db).dropna()
        dd, days = (float(d.mean()) if len(d) else float("nan")), int(len(d))
        span = [str(d.index.min())[:7], str(d.index.max())[:7]] if len(d) else list(SCREEN)
    if not np.isfinite(dd):
        return {"status": "无", "diff_day": float("nan"), "days": days, "span": span}
    return {"status": "同号" if dd >= GATE_V3["screen_min_diff"] else "异号", "diff_day": dd,
            "days": days, "span": span}


def prereg_info(exp: str) -> dict:
    """exp 的 prereg.json（没有就空 dict）。gate_version 缺省 2（2026-10-16 前登记的）。"""
    pre = EXP / exp / "prereg.json" if exp else None
    if not pre or not pre.exists():
        return {}
    j = json.loads(pre.read_text(encoding="utf-8"))
    j.setdefault("gate_version", 2)
    return j


def compare(a: pd.DataFrame, b: pd.DataFrame, y: str = "y_open", k: int = 10,
            since: str = "", family: str = FAMILY_DEFAULT, exp: str = "", arm: str = "",
            screen=None) -> dict:
    """a 对 b：按天配对、按月配对、最差三月 / 滚动三月、空榜，加稳健推断，v2 和 v3 两套判定。b 是对照（现状）。

    exp / arm 给了：逐日差值序列存 EXP/<exp>/diff_<arm>_<k>_<y>.parquet（族门槛要用），结果记进登记簿；
    since 给了而且 exp 的预登记里没有这个区间：算一次「事后加看」，登记簿加一行（子区间的序列不落盘，
    不进族门槛的零分布）。
    screen：筛选集结果（见 screen_result），没给 = 「筛选集：无」。exp 的 prereg.json 里有 screen 而这里
    没给，就用登记时那份。正式判定用哪个版本看 prereg.json 的 gate_version（没预登记按当前 GATE_VERSION）。"""
    if since:
        a, b = a[a["date"] >= since], b[b["date"] >= since]
    ta, tb = a[a["rank"] <= k], b[b["rank"] <= k]
    da = ta.groupby("date")[y].mean()
    db = tb.groupby("date")[y].mean()
    d = (da - db).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    ma, mb = monthly(ta, y), monthly(tb, y)
    md = (ma - mb).dropna().sort_index()
    mse = float(md.std(ddof=1) / np.sqrt(len(md))) if len(md) > 1 else float("nan")
    worst3 = md.sort_values().head(3)
    # 没有 score 列的表（别的目标线、只比名次的实验）就没有「生产规则」，空榜按 0
    ea = (a["date"].nunique() - prod_picks(a)["date"].nunique()) if "score" in a.columns else 0
    eb = (b["date"].nunique() - prod_picks(b)["date"].nunique()) if "score" in b.columns else 0
    res = {"diff_day": float(d.mean()), "se_day": se,
           "t_day": float(d.mean() / se) if se and se > 0 else float("nan"), "days": int(len(d)),
           "diff_month": float(md.mean()), "t_month": float(md.mean() / mse) if mse and mse > 0 else float("nan"),
           "months": int(len(md)), "wins": int((md > 0).sum()),
           "worst3": {k_: float(v) for k_, v in worst3.items()},
           "empty_a": int(ea), "empty_b": int(eb)}
    res["gate_v1"] = {
        "t_day": bool(res["t_day"] >= GATE["t_day"]),
        "worst3": bool(len(worst3) and worst3.min() >= GATE["worst3_min_diff"]),
        "empty": bool(ea <= GATE["empty_ratio_max"] * max(eb, 1)),
    }
    res["verdict_v1"] = ("过线" if all(res["gate_v1"].values()) else
                         "未定" if abs(res["diff_day"]) < 0.025 else "不过线")

    # ---- v2 / v3 ----
    pre = prereg_info(exp)
    if since and exp:
        reg_span = pre.get("dev")
        if not reg_span or since > str(reg_span[0]):
            ledger_add({"family": family, "exp": exp, "arm": arm, "kind": "second_look", "since": since})
    res["t_nw"] = nw_t(d.to_numpy())
    res["t_nw_lags"] = {L: nw_t(d.to_numpy(), L) for L in (10, 20, 30)}
    res["boot"] = block_boot(d.to_numpy())
    dm = d.groupby(d.index.str[:7]).mean()
    best = dm.idxmax() if res["diff_day"] >= 0 else dm.idxmin()
    res["t_nw_drop_best"] = nw_t(d[d.index.str[:7] != best].to_numpy()) if len(dm) > 1 else float("nan")
    res["drop_best_month"] = str(best) if len(dm) else ""
    res["pooled_diff"] = float(ta[y].mean() - tb[y].mean())
    mdx = md[md.index != "2024-09"]
    w3x = mdx.sort_values().head(3)
    res["worst3_ex_2024_09"] = {k_: float(v) for k_, v in w3x.items()}
    r3, r3_end = roll3_min(md)
    r3x, r3x_end = roll3_min(mdx)
    res["roll3"] = {"min": r3, "end": r3_end, "min_ex_2024_09": r3x, "end_ex_2024_09": r3x_end}
    if screen is None and arm and isinstance(pre.get("screen"), dict) and pre["screen"].get(arm) is not None:
        screen = pre["screen"][arm]
    res["screen"] = screen_result(screen, y, k)
    th = family_threshold(family)
    res["threshold"] = th
    common = {
        "t_nw": bool(np.isfinite(res["t_nw"]) and res["t_nw"] >= th["c"]),
        "boot_p": bool(np.isfinite(res["boot"]["p"]) and res["boot"]["p"] < GATE_V2["boot_p"]),
        "drop_best": bool(np.isfinite(res["t_nw_drop_best"]) and res["t_nw_drop_best"] >= GATE_V2["drop_best_month_t"]),
        "same_sign": bool(np.sign(res["pooled_diff"]) == np.sign(res["diff_day"])),
        "empty": res["gate_v1"]["empty"],
    }
    res["gate_v2"] = {**common, "worst3": res["gate_v1"]["worst3"]}
    res["gate_v3"] = {**common, "roll3": bool(np.isfinite(r3) and r3 >= GATE_V3["roll3_min_diff"]),
                      "screen": res["screen"]["status"] == "同号"}

    def _verdict(g: dict) -> str:
        return ("过线" if all(g.values()) else
                "不过线" if np.isfinite(res["t_nw"]) and res["t_nw"] <= -th["c"] else "未定")
    res["verdict_v2"], res["verdict_v3"] = _verdict(res["gate_v2"]), _verdict(res["gate_v3"])
    res["gate_version"] = int(pre.get("gate_version", GATE_VERSION))
    v = res["gate_version"]
    res["gate"] = res["gate_v3"] if v >= 3 else res["gate_v2"]
    res["pass"] = all(res["gate"].values())
    res["verdict"] = res["verdict_v3"] if v >= 3 else res["verdict_v2"]
    if exp and arm:
        row = {"family": family, "exp": exp, "arm": arm, "kind": "result", "t_nw": res["t_nw"],
               "gate_version": v, "verdict": res["verdict"], "verdict_v2": res["verdict_v2"],
               "verdict_v3": res["verdict_v3"], "screen": res["screen"]["status"]}
        if not since:
            p = EXP / exp / f"diff_{arm}_{k}_{y}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"date": [str(i) for i in d.index], "d": d.to_numpy(float)}).to_parquet(p, index=False)
            row["series"] = str(p)
        else:
            row["since"] = since
        ledger_add(row)
    return res


def fmt_metrics(name: str, m: dict) -> str:
    return ("%-10s prod n=%4d 开盘 %5.2f%% 收盘 %5.2f%% 空榜 %3d 每天 %.1f | 前10 开盘 %5.2f%% ±%.2f 收盘 %5.2f%% | "
            "前30 开盘 %5.2f%% | 月中位 %5.1f%% 最差 %4.1f%%" % (
                name, m["prod_open"]["n"], 100 * m["prod_open"]["hit"], 100 * m["prod_close"]["hit"],
                m["prod_empty"], m["prod_per_day"], 100 * m["top10_open"]["hit"],
                100 * m["top10_open"]["se"], 100 * m["top10_close"]["hit"],
                100 * m["top30_open"]["hit"], 100 * m["month_median"], 100 * m["month_worst"]))


def fmt_compare(name: str, c: dict) -> str:
    th = c.get("threshold") or {}
    bt = c.get("boot") or {}
    r3 = c.get("roll3") or {}
    sc = c.get("screen") or {"status": "无"}
    sc_s = "筛选集：%s" % sc["status"] + ("（%+.2f，%d 天）" % (100 * sc["diff_day"], sc["days"])
                                       if sc["status"] != "无" else "")
    return ("%-10s 对现状：按天 %+.2f（iid t=%.2f，NW t=%.2f，块自举 p=%.3f，去 %s 后 NW %.2f，%d 天） "
            "按月 %+.2f (赢 %d/%d) 最差三月 %s（去 2024-09：%s） 滚动三月最低 %+.1f@%s（去 2024-09：%+.1f@%s） "
            "空榜 %d/%d  %s  门槛 %.2f[%s] -> v%d %s（v2 %s / v3 %s）" % (
                name, 100 * c["diff_day"], c["t_day"], c.get("t_nw", float("nan")), bt.get("p", float("nan")),
                c.get("drop_best_month", ""), c.get("t_nw_drop_best", float("nan")), c["days"],
                100 * c["diff_month"], c["wins"], c["months"],
                ",".join("%s %+.1f" % (k[2:], 100 * v) for k, v in c["worst3"].items()),
                ",".join("%s %+.1f" % (k[2:], 100 * v) for k, v in (c.get("worst3_ex_2024_09") or {}).items()),
                100 * r3.get("min", float("nan")), r3.get("end", ""),
                100 * r3.get("min_ex_2024_09", float("nan")), r3.get("end_ex_2024_09", ""),
                c["empty_a"], c["empty_b"], sc_s, th.get("c", float("nan")), th.get("method", ""),
                c.get("gate_version", 2), c["verdict"], c.get("verdict_v2", "?"), c.get("verdict_v3", "?")))


# ---------------------------------------------------------------
#  纪律
# ---------------------------------------------------------------
def prereg(name: str, hypothesis: str, arms: dict[str, str], metric: str = "top10 y_open 按天配对",
           gate: dict | None = None, family: str = FAMILY_DEFAULT, screen: dict | None = None) -> Path:
    """预登记。已存在就拒绝改写（改假设 = 新实验，换名字）。每个臂在登记簿里记一次（重复登记不重复记）。

    screen = {臂: 筛选集按天差（float）或 {"diff_day", "days"}}：规则 7，按天差 < 0 的臂整个实验拒绝登记
    （先全部检查再写，不留半截）。没给的臂记 None，compare 时「筛选集：无」。"""
    d = EXP / name
    f = d / "prereg.json"
    sc: dict[str, dict | None] = {}
    for arm in arms:
        v = (screen or {}).get(arm)
        if v is None:
            sc[arm] = None
            continue
        sc[arm] = {"diff_day": float(v)} if not isinstance(v, dict) else {k_: v[k_] for k_ in v if k_ in ("diff_day", "days", "span")}
        r = screen_result(sc[arm])
        if r["status"] == "异号":
            raise SystemExit(f"{name}/{arm} 在筛选集 {SCREEN[0]}..{SCREEN[1]} 上按天差 {100 * r['diff_day']:+.2f}pp < 0，不登记（规则 7）")
    d.mkdir(parents=True, exist_ok=True)
    seen = {(r.get("exp"), r.get("arm")) for r in _ledger_rows() if r.get("kind") == "prereg"}
    for arm in arms:
        if (name, arm) not in seen:
            ledger_add({"family": family, "exp": name, "arm": arm, "kind": "prereg", "gate_version": GATE_VERSION,
                        "screen": screen_result(sc[arm])["status"]})
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("hypothesis") != hypothesis or old.get("arms") != arms:
            raise SystemExit(f"{f} 已登记且内容不同，改假设请换实验名")
        return f
    f.write_text(json.dumps({"name": name, "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "hypothesis": hypothesis, "arms": arms, "metric": metric,
                             "gate": gate or GATE_V3, "gate_version": GATE_VERSION, "family": family,
                             "screen": sc, "screen_span": list(SCREEN),
                             "threshold_at_prereg": family_threshold(family), "dev": DEV, "results": []},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return f


def record(name: str, payload: dict) -> None:
    f = EXP / name / "prereg.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    j["results"].append({"at": dt.datetime.now().isoformat(timespec="seconds"), **payload})
    f.write_text(json.dumps(j, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def confirm_guard(name: str, candidate: str) -> None:
    """确认集每季度最多看一次。看了就记，记了就挡。"""
    used = json.loads(CONFIRM_USED.read_text(encoding="utf-8")) if CONFIRM_USED.exists() else []
    now = dt.datetime.now()
    q = f"{now.year}Q{(now.month - 1) // 3 + 1}"
    for u in used:
        if u.get("quarter") == q:
            raise SystemExit(f"确认集 {q} 已经看过一次（{u}），本季度不能再看")
    used.append({"quarter": q, "at": now.isoformat(timespec="seconds"), "experiment": name,
                 "candidate": candidate})
    CONFIRM_USED.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_USED.write_text(json.dumps(used, ensure_ascii=False, indent=1), encoding="utf-8")
