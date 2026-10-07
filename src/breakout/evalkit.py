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

过线标准（gate）
---------------
    t_day >= 2.5  且  最差三个月的命中之差 >= 0（不比现状差）  且  空榜天数不翻倍
    小于 2.5 个百分点的差别一律「未定」

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


def load_frame(with_confirm: bool = False) -> pd.DataFrame:
    """训练表 + 买得到标签。with_confirm=False 时 2026 年整段不可见。"""
    import arena as A
    import buyable as B
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
            o = te[["date", "code", "board", "y_up", "y_open"]].copy()
            o["p"] = np.asarray(p_te, np.float32)
            alls.setdefault(arm, []).append(o[["date", "code", "p"]])
            o = o[~o["code"].isin(st)]
            o["score"] = np.clip(np.searchsorted(q, o["p"]), 0, 100)
            o = o.sort_values(["date", "p"], ascending=[True, False])
            o["rank"] = o.groupby("date").cumcount() + 1
            tops.setdefault(arm, []).append(o[o["rank"] <= TOP_KEEP])
        print("  %s 完成 %.0fs（特征 %d 列）" % (m, time.time() - t0, len(feats)), flush=True)
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


def compare(a: pd.DataFrame, b: pd.DataFrame, y: str = "y_open", k: int = 10,
            since: str = "") -> dict:
    """a 对 b：按天配对、按月配对、最差三个月、空榜。b 是对照（现状）。"""
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
    ea = a["date"].nunique() - prod_picks(a)["date"].nunique()
    eb = b["date"].nunique() - prod_picks(b)["date"].nunique()
    res = {"diff_day": float(d.mean()), "se_day": se,
           "t_day": float(d.mean() / se) if se and se > 0 else float("nan"), "days": int(len(d)),
           "diff_month": float(md.mean()), "t_month": float(md.mean() / mse) if mse and mse > 0 else float("nan"),
           "months": int(len(md)), "wins": int((md > 0).sum()),
           "worst3": {k_: float(v) for k_, v in worst3.items()},
           "empty_a": int(ea), "empty_b": int(eb)}
    res["gate"] = {
        "t_day": bool(res["t_day"] >= GATE["t_day"]),
        "worst3": bool(len(worst3) and worst3.min() >= GATE["worst3_min_diff"]),
        "empty": bool(ea <= GATE["empty_ratio_max"] * max(eb, 1)),
    }
    res["pass"] = all(res["gate"].values())
    res["verdict"] = ("过线" if res["pass"] else
                      "未定" if abs(res["diff_day"]) < 0.025 else "不过线")
    return res


def fmt_metrics(name: str, m: dict) -> str:
    return ("%-10s prod n=%4d 开盘 %5.2f%% 收盘 %5.2f%% 空榜 %3d 每天 %.1f | 前10 开盘 %5.2f%% ±%.2f 收盘 %5.2f%% | "
            "前30 开盘 %5.2f%% | 月中位 %5.1f%% 最差 %4.1f%%" % (
                name, m["prod_open"]["n"], 100 * m["prod_open"]["hit"], 100 * m["prod_close"]["hit"],
                m["prod_empty"], m["prod_per_day"], 100 * m["top10_open"]["hit"],
                100 * m["top10_open"]["se"], 100 * m["top10_close"]["hit"],
                100 * m["top30_open"]["hit"], 100 * m["month_median"], 100 * m["month_worst"]))


def fmt_compare(name: str, c: dict) -> str:
    return ("%-10s 对现状：按天 %+.2f ±%.2f (t=%.2f, %d 天) 按月 %+.2f (t=%.2f, 赢 %d/%d) "
            "最差三月 %s 空榜 %d/%d -> %s" % (
                name, 100 * c["diff_day"], 100 * c["se_day"], c["t_day"], c["days"],
                100 * c["diff_month"], c["t_month"], c["wins"], c["months"],
                ",".join("%s %+.1f" % (k[2:], 100 * v) for k, v in c["worst3"].items()),
                c["empty_a"], c["empty_b"], c["verdict"]))


# ---------------------------------------------------------------
#  纪律
# ---------------------------------------------------------------
def prereg(name: str, hypothesis: str, arms: dict[str, str], metric: str = "top10 y_open 按天配对",
           gate: dict | None = None) -> Path:
    """预登记。已存在就拒绝改写（改假设 = 新实验，换名字）。"""
    d = EXP / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "prereg.json"
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("hypothesis") != hypothesis or old.get("arms") != arms:
            raise SystemExit(f"{f} 已登记且内容不同，改假设请换实验名")
        return f
    f.write_text(json.dumps({"name": name, "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "hypothesis": hypothesis, "arms": arms, "metric": metric,
                             "gate": gate or GATE, "dev": DEV, "results": []},
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
