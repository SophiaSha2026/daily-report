"""
绝对门槛研究（research/gate，2026-10-18）公用件。条文见 criteria.txt。

数据：research/polish pool.parquet 的候选池行（pool 列）+ 本目录 build_gt.py 补的列（名次、日级市场状态、非价格数据）
      -> research/gate/table.parquet（gitignore），行序固定，各方法的样本外分数存 scores/<名>.npy 和它同序。
尺子：precision 的 common_pr（Baseline / evaluate / boot_ratio），对账 B(10) 两段 11.34 / 16.54。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "research" / "precision", ROOT / "research" / "polish", ROOT / "research" / "rulecore",
          ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common_pr as PR  # noqa: E402
import common_pl as PL  # noqa: E402
import common_rc as RC  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLE = HERE / "table.parquet"
SCORES = HERE / "scores"
SCORES.mkdir(exist_ok=True)
LEDGER = HERE / "ledger.jsonl"
SEGS = PL.SEGS
FAMILY = "breakout_gate"
TAUS = (0.20, 0.25, 0.30, 0.40, 0.50)
TEST_MONTHS = ("2021-07", "2025-12")          # 样本外分数覆盖的月份；2022-01 起才进两段
CAL_MONTHS = 12                                # 校准窗：之前 12 个月的样本外分数
CAL_MIN_MONTHS = 3
TRAIN_YEARS = 3
say = PL.say


def progress(text: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def ledger_add(row: dict) -> None:
    from bjtime import now_bj
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False) + "\n")


def ledger_count(kind: str | None = None) -> int:
    if not LEDGER.exists():
        return 0
    n = 0
    for ln in LEDGER.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            if kind is None or r.get("kind") == kind:
                n += 1
    return n


def months_between(a: str, b: str) -> list[str]:
    import evalkit as K
    return K.months_in((a, b))


month_shift = RC.month_shift


# ---------------------------------------------------------------
#  表
# ---------------------------------------------------------------
BASE_COLS = ["code", "date", "board", "y_open", "r20_open", "yizi1", "end20", "ret20", "lret20", "vol60", "turn20", "amp20",
             "ret1", "ret5", "ret10", "ret40", "ret60", "ret20_lag5", "vol20", "dist_hi20", "dist_hi60", "dist_hi120",
             "pos20", "nh20_5", "up20", "max1_20", "amt20", "amt1", "amt_ratio", "turn1", "turn5", "vr5_20", "mcap",
             "price_u", "gap1", "cr1", "lim_pct", "lu_close", "yizi_today", "lu_streak", "nlu20", "nlu5", "nld20", "nbar"]
DAY_COLS = ["pool_n", "mkt_base_t21", "mkt_ret20", "mkt_ret5", "lu_share20", "hot50", "top1_ret20", "top10_ret20_med", "mkt_vol20"]
ALT_COLS = ["lhb_today", "lhb_n5", "lhb_n20", "lhb_net5_share", "lhb_buy5_share", "org_net5_share", "org_buy5_share",
            "rz_in", "rz_chg5", "rz_chg20", "rz_buy5_share", "rz_level", "dz_n20", "dz_disc20_share", "dz_amt20_share",
            "gd_chg1", "gd_chg3", "gd_stale", "sv_n30", "sv_n90", "sv_orgs90"]
DERIVED = ["rank_ret20", "prank_ret20", "lmcap", "lamt20", "lvol60", "lturn20", "lprice", "board_i", "lret20_lag5", "lret60"]

# 特征集
FEATS_PRICE = ["lret20", "rank_ret20", "prank_ret20", "lvol60", "vol20", "lturn20", "turn1", "turn5", "amp20",
               "ret1", "ret5", "ret10", "ret40", "lret60", "lret20_lag5", "dist_hi20", "dist_hi60", "dist_hi120", "pos20",
               "nh20_5", "up20", "max1_20", "lamt20", "amt_ratio", "vr5_20", "lmcap", "lprice", "gap1", "cr1", "lim_pct",
               "lu_close", "yizi_today", "lu_streak", "nlu20", "nlu5", "nld20", "nbar", "board_i"] + DAY_COLS
FEATS_LR = ["lret20", "lvol60", "lturn20", "amp20", "lmcap", "nlu20", "dist_hi60", "board_i"]


def load_table(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(TABLE, columns=columns)


def seg_of(dates) -> np.ndarray:
    return PL.seg_of(pd.Series(dates))


def train_mask(df: pd.DataFrame, month: str) -> np.ndarray:
    """common_rc.train_mask：date < 净化线、end20 < 月初、回看 36 个月。"""
    return RC.train_mask(df, month)


# ---------------------------------------------------------------
#  校准 + 门槛
# ---------------------------------------------------------------
def calibrate_month(s: np.ndarray, y: np.ndarray, dates: np.ndarray, end20: np.ndarray, month: str,
                    cal_months: int = CAL_MONTHS, min_months: int = CAL_MIN_MONTHS):
    """给 month 月的行做等渗校准。校准行 = 之前 cal_months 个月里有样本外分数、end20 < 月初 的行。
    返回 (校准器 或 None, 校准行数, 校准月数)。"""
    from sklearn.isotonic import IsotonicRegression
    lo = month_shift(month, -cal_months) + "-01"
    m = (dates >= lo) & (dates < month + "-01") & (end20 < month + "-01") & np.isfinite(s) & np.isfinite(y)
    if m.sum() == 0:
        return None, 0, 0
    nm = len(np.unique(dates[m].astype("U7")))
    if nm < min_months:
        return None, int(m.sum()), nm
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    ir.fit(s[m].astype(float), y[m].astype(float))
    return ir, int(m.sum()), nm


def calibrated(scores: np.ndarray, t: pd.DataFrame, months: list[str], cal_months: int = CAL_MONTHS) -> np.ndarray:
    """整表的 p̂（没有校准器的月 NaN）。t 要有 date end20 y_open，和 scores 同序。"""
    dates = t["date"].to_numpy().astype("U10")
    end20 = t["end20"].to_numpy().astype("U10")
    y = t["y_open"].to_numpy(float)
    ym = dates.astype("U7")
    p = np.full(len(t), np.nan, np.float32)
    for m in months:
        ir, n, nm = calibrate_month(scores, y, dates, end20, m, cal_months=cal_months)
        rows = (ym == m) & np.isfinite(scores)
        if ir is None or not rows.any():
            continue
        p[rows] = ir.predict(scores[rows].astype(float)).astype(np.float32)
    return p


# ---------------------------------------------------------------
#  评估（precision 同一把尺子）
# ---------------------------------------------------------------
_B = {}


def baseline(seg: str) -> PR.Baseline:
    if seg not in _B:
        _B[seg] = PR.Baseline(PR.load(seg))
    return _B[seg]


def eval_gate(t: pd.DataFrame, sel: np.ndarray, seg: str, boot: bool = True) -> dict:
    """t = 整表（有 date hit seg r20_open），sel = 布尔。只算 seg 段。"""
    B = baseline(seg)
    ts = t[t["seg"] == seg]
    s = sel[(t["seg"] == seg).to_numpy()]
    e = PR.evaluate(ts, s, B, boot=boot)
    rows = ts[s]
    if len(rows):
        e["months_out"] = rows.groupby(rows["date"].str[:7]).size().reindex(
            months_between(*SEGS[seg]), fill_value=0)
        e["per_month"] = float(e["months_out"].mean())
        hits = rows[rows["hit"] > 0]
        e["gain_q"] = hits["r20_open"].quantile([0.1, 0.5, 0.9]).to_dict() if len(hits) else {}
        e["gain_all_med"] = float(rows["r20_open"].median())
        e["p_hat_mean"] = float(rows["p_hat"].mean()) if "p_hat" in rows else np.nan
        yr = rows.groupby(rows["date"].str[:4])["hit"].agg(["size", "mean"])
        e["by_year"] = {k: (int(v["size"]), float(v["mean"])) for k, v in yr.iterrows()}
        e["B10"] = B.precision_k(10)
    else:
        e["per_month"] = 0.0
        e["gain_q"] = {}
        e["by_year"] = {}
        e["p_hat_mean"] = np.nan
        e["B10"] = B.precision_k(10)
    e["dates"] = B.dates
    return e


def fmt(name: str, e: dict, seg: str) -> str:
    if e["V"] <= 0:
        return "%-26s %-6s V 0（整段空榜）" % (name, seg)
    s = PR.fmt_eval(name, e, seg)
    s += "  月 %4.1f只 空榜 %3.0f%%  p̂ %4.1f%%" % (e["per_month"], 100 * e["empty_share"], 100 * e.get("p_hat_mean", np.nan))
    if e["gain_q"]:
        s += "  命中涨幅 q10/50/90 %.0f/%.0f/%.0f%%" % tuple(100 * e["gain_q"][q] for q in (0.1, 0.5, 0.9))
    return s


def combined_judge(e_s: dict, e_d: dict) -> dict:
    """合并 969 天的 u 序列：NW t、块自举 p、去最好月、滚动三月。"""
    import evalkit as K
    u = pd.Series(np.concatenate([e_s["u"], e_d["u"]]), index=np.concatenate([e_s["dates"], e_d["dates"]]))
    V = e_s["V"] + e_d["V"]
    out = {"dP": float(u.sum() / max(V, 1)), "t": PR.nw_t(u.to_numpy())}
    out["boot_p"] = K.block_boot(u.to_numpy())["p"]
    dm = u.groupby(u.index.str[:7]).mean()
    best = dm.idxmax()
    out["t_drop_best"] = PR.nw_t(u[u.index.str[:7] != best].to_numpy())
    out["best_month"] = str(best)
    pp = 100 * u * len(u) / max(V, 1)
    md = pp.groupby(pp.index.str[:7]).mean()
    r3, r3e = K.roll3_min(md)
    out["roll3"] = (float(r3), str(r3e))
    return out
