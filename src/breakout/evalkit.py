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

过线标准（gate，2026-10-15 起 v2，research/stats/report.md 第 5 节，用户拍板见 docs/plan_2026-10-15.md）
---------------
标签是未来 20 根的最高价，相邻交易日的结果高度重叠，按天配对的 iid t 平均被放大约 1.5 倍；
开发集上又已经试了近 90 次。所以 v2 六条全满足才「过线」，NW t <= -门槛记「不过线」，其余「未定」：
    1 NW t（Bartlett 核，滞后 20 = 标签窗口）>= c，c = max(2.5, 族门槛)
      族门槛：登记簿里本族有逐日差值序列的臂 >= 3 个时用族内最大 t 零分布的 95% 分位（联合块自举，
      保留臂间相关）；否则用单侧 Bonferroni Φ⁻¹(1 - 0.05/m)，m = 登记簿里本族的试验次数
    2 块自举 p < 0.05（块长 20，5000 次）
    3 去掉最好的一个月后 NW t >= 2.0（防一个月说了算）
    4 合并命中之差和按天配对均值同号（教训 33）
    5 最差三个月的命中之差 >= 0（2024-09 照算；「去掉 2024-09」只并列印出，用户 2026-10-15 定）
    6 空榜天数不翻倍
旧的 t_day >= 2.5 判法留在 gate_v1 / verdict_v1 里对账。

试验登记簿
----------
    data/breakout/raw/exp/ledger.jsonl：每次 prereg 每个臂记一行，每次事后加看子区间（compare 的
    since 不是预登记的区间）也记一行。门槛按次数抬高。第一次用时从 research/stats/out/ledger_seed.jsonl
    补登历史（实验 15~21 的 47 条 + 4 组只有计数的，合计约 87 次）。

预登记
------
    每个实验先 prereg(name, hypothesis, arms, gate) 再跑；文件已存在就拒绝改写，
    结果 append 进同一个 json。写进 docs/breakout_log.md 的条目要和它一致。

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
    n = len(x)
    if n < lag + 3:
        return float("nan")
    lrv = _autocov(x, 0) + sum(2 * (1 - k / (lag + 1)) * _autocov(x, k) for k in range(1, lag + 1))
    se = float(np.sqrt(max(lrv, 1e-18) / n))
    return float(x.mean() / se)


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


def family_count(family: str = FAMILY_DEFAULT) -> int:
    """登记簿里这一族试过多少次（带 count 的补登行按 count 计）。结果行不算新的试验。"""
    return int(sum(int(r.get("count", 1)) for r in _ledger_rows()
                   if r.get("family") == family and r.get("kind") != "result"))


def _family_series(family: str) -> list[np.ndarray]:
    out = []
    for r in _ledger_rows():
        p = r.get("series")
        if r.get("family") == family and p and Path(p).exists():
            out.append(np.load(p))
    return out


def family_threshold(family: str = FAMILY_DEFAULT) -> dict:
    """族门槛 c = max(2.5, 族门槛)。见模块说明第 1 条。"""
    from statistics import NormalDist
    m = max(family_count(family), 1)
    ser = _family_series(family)
    if len(ser) >= 3:
        n = min(len(x) for x in ser)
        mat = np.stack([x[-n:] for x in ser])               # 对齐到共同的最后 n 天
        mat = mat - mat.mean(axis=1, keepdims=True)         # 零假设：中心化
        idx = _mbb_index(n, BLOCK, 2000, np.random.default_rng(7))
        mx = np.array([max(nw_t(row[ii]) for row in mat) for ii in idx[:400]])
        c = float(np.nanquantile(mx, 0.95))
        return {"c": max(GATE_V2["t_floor"], c), "method": f"族内最大 t（{len(ser)} 条序列）", "m": m}
    c = NormalDist().inv_cdf(1 - GATE_V2["fwer"] / m)
    return {"c": max(GATE_V2["t_floor"], c), "method": f"Bonferroni（m={m}）", "m": m}


def compare(a: pd.DataFrame, b: pd.DataFrame, y: str = "y_open", k: int = 10,
            since: str = "", family: str = FAMILY_DEFAULT, exp: str = "", arm: str = "") -> dict:
    """a 对 b：按天配对、按月配对、最差三个月、空榜，加 v2 的稳健推断。b 是对照（现状）。

    exp / arm 给了：逐日差值序列存 EXP/<exp>/diff_<arm>_<k>_<y>.npy（族门槛要用），结果记进登记簿；
    since 给了而且 exp 的预登记里没有这个区间：算一次「事后加看」，登记簿加一行。"""
    if since:
        a, b = a[a["date"] >= since], b[b["date"] >= since]
    ta, tb = a[a["rank"] <= k], b[b["rank"] <= k]
    da = ta.groupby("date")[y].mean()
    db = tb.groupby("date")[y].mean()
    d = (da - db).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    ma, mb = monthly(ta, y), monthly(tb, y)
    md = (ma - mb).dropna()
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

    # ---- v2 ----
    if since and exp:
        pre = EXP / exp / "prereg.json"
        reg_span = json.loads(pre.read_text(encoding="utf-8")).get("dev") if pre.exists() else None
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
    w3x = (ma - mb).dropna()
    w3x = w3x[w3x.index != "2024-09"].sort_values().head(3)
    res["worst3_ex_2024_09"] = {k_: float(v) for k_, v in w3x.items()}
    th = family_threshold(family)
    res["threshold"] = th
    res["gate"] = {
        "t_nw": bool(np.isfinite(res["t_nw"]) and res["t_nw"] >= th["c"]),
        "boot_p": bool(np.isfinite(res["boot"]["p"]) and res["boot"]["p"] < GATE_V2["boot_p"]),
        "drop_best": bool(np.isfinite(res["t_nw_drop_best"]) and res["t_nw_drop_best"] >= GATE_V2["drop_best_month_t"]),
        "same_sign": bool(np.sign(res["pooled_diff"]) == np.sign(res["diff_day"])),
        "worst3": res["gate_v1"]["worst3"],
        "empty": res["gate_v1"]["empty"],
    }
    res["pass"] = all(res["gate"].values())
    res["verdict"] = ("过线" if res["pass"] else
                      "不过线" if np.isfinite(res["t_nw"]) and res["t_nw"] <= -th["c"] else "未定")
    if exp and arm:
        p = EXP / exp / f"diff_{arm}_{k}_{y}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, d.to_numpy())
        ledger_add({"family": family, "exp": exp, "arm": arm, "kind": "result", "series": str(p),
                    "t_nw": res["t_nw"], "verdict": res["verdict"]})
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
    return ("%-10s 对现状：按天 %+.2f（iid t=%.2f，NW t=%.2f，块自举 p=%.3f，去 %s 后 NW %.2f，%d 天） "
            "按月 %+.2f (赢 %d/%d) 最差三月 %s（去 2024-09：%s） 空榜 %d/%d  门槛 %.2f[%s] -> %s" % (
                name, 100 * c["diff_day"], c["t_day"], c.get("t_nw", float("nan")), bt.get("p", float("nan")),
                c.get("drop_best_month", ""), c.get("t_nw_drop_best", float("nan")), c["days"],
                100 * c["diff_month"], c["wins"], c["months"],
                ",".join("%s %+.1f" % (k[2:], 100 * v) for k, v in c["worst3"].items()),
                ",".join("%s %+.1f" % (k[2:], 100 * v) for k, v in (c.get("worst3_ex_2024_09") or {}).items()),
                c["empty_a"], c["empty_b"], th.get("c", float("nan")), th.get("method", ""), c["verdict"]))


# ---------------------------------------------------------------
#  纪律
# ---------------------------------------------------------------
def prereg(name: str, hypothesis: str, arms: dict[str, str], metric: str = "top10 y_open 按天配对",
           gate: dict | None = None, family: str = FAMILY_DEFAULT) -> Path:
    """预登记。已存在就拒绝改写（改假设 = 新实验，换名字）。每个臂在登记簿里记一次（重复登记不重复记）。"""
    d = EXP / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "prereg.json"
    seen = {(r.get("exp"), r.get("arm")) for r in _ledger_rows() if r.get("kind") == "prereg"}
    for arm in arms:
        if (name, arm) not in seen:
            ledger_add({"family": family, "exp": name, "arm": arm, "kind": "prereg"})
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("hypothesis") != hypothesis or old.get("arms") != arms:
            raise SystemExit(f"{f} 已登记且内容不同，改假设请换实验名")
        return f
    f.write_text(json.dumps({"name": name, "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "hypothesis": hypothesis, "arms": arms, "metric": metric,
                             "gate": gate or GATE_V2, "family": family,
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
