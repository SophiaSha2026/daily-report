"""
精确率优先研究（research/precision，2026-10-18）公用件。条文见 criteria.txt。

数据：research/polish 的 pool.parquet（候选池 + 标签 + 扩展特征）；只取每天候选池内 lret20 前 50（top50.parquet，gitignore），
再并上：影子模型逐月滚动分数 p 和它在候选池内的名次（screen_base_v1 / screen_base_v1_dev / rulecore_v1 的 all_base_*）、
research/altdata/top60.parquet 的非价格派生列、连续在榜天数、日级市场状态量。
尺子：精确率 P = Σh/Σn，匹配量名次基线 B(k̄)，按天序列 u_d = h_R,d − h_B,d，NW t / 块自举 / 族最大 t 都调 evalkit 同一份实现。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "research" / "polish", ROOT / "research" / "rulecore", ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common_pl as P  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = P.EXP
OUT = EXP / "precision"
OUT.mkdir(parents=True, exist_ok=True)
TOP50 = HERE / "top50.parquet"
SEGS = P.SEGS
FAMILY = "breakout_precision"
KEEP = 50
LEDGER = HERE / "ledger.jsonl"
VOL_FLOOR = {"screen": (300, 100), "dev": (150, 60)}      # (V, 出票天) 条文 2f

say = P.say


def progress(text: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def ledger_add(row: dict) -> None:
    from bjtime import now_bj
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False) + "\n")


def ledger_count(kind: str) -> int:
    if not LEDGER.exists():
        return 0
    return sum(1 for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip() and json.loads(ln).get("kind") == kind)


def seg_of(dates) -> np.ndarray:
    return P.seg_of(pd.Series(dates))


# ---------------------------------------------------------------
#  top50 表
# ---------------------------------------------------------------
ALT_COLS = ["lhb_today", "lhb_n5", "lhb_n20", "lhb_net5_share", "org_net5_share", "rz_in", "rz_chg5", "rz_chg20", "rz_level",
            "dz_n20", "dz_disc20_share", "gd_chg1", "gd_stale", "sv_n30", "sv_n90"]


def build_top50(force: bool = False) -> pd.DataFrame:
    if TOP50.exists() and not force:
        return pd.read_parquet(TOP50)
    import time
    t0 = time.time()
    df = P.load_pool()
    df = df[df["pool"] & (df["date"] >= "2021-11-01")].reset_index(drop=True)
    say("候选池行 %d（%s..%s）%.0fs" % (len(df), df["date"].min(), df["date"].max(), time.time() - t0))
    # 日级市场状态（整个候选池上算，T 日收盘后可得）
    day = df.groupby("date").agg(pool_n=("code", "size"), base=("y_open", "mean"), mret1=("ret1", "mean"),
                                 lu_share=("lu_close", "mean"), hot50=("ret20", lambda s: float((s > 0.5).mean())))
    day = day.sort_index()
    day["mkt_base_t21"] = day["base"].rolling(20).mean().shift(21)        # T-40..T-21 候选池基准率
    day["mkt_ret20"] = day["mret1"].rolling(20).sum()                     # 候选池等权近 20 日累计收益（近似）
    day["mkt_ret5"] = day["mret1"].rolling(5).sum()
    day["lu_share20"] = day["lu_share"].rolling(20).mean()
    # 前 50 名次表
    t = P.rank_by(df, df["lret20"].to_numpy(), keep=KEEP)
    t = t.merge(df.drop(columns=["board", "y_open", "y_up"]), on=["date", "code"], how="left")
    t = t.rename(columns={"rank": "rank_ret20"})
    say("前 50 行 %d %.0fs" % (len(t), time.time() - t0))
    # 日级：第 1 名的 ret20、前 10 的 ret20 中位
    d2 = t[t["rank_ret20"] <= 10].groupby("date")["ret20"].agg(top1_ret20="max", top10_ret20_med="median")
    day = day.join(d2)
    # 连续在榜天数（前 10 / 前 50），按交易日序号数
    dates = np.array(sorted(df["date"].unique()))
    didx = pd.Series(np.arange(len(dates)), index=dates)
    t["_di"] = t["date"].map(didx).to_numpy()
    for k, col in ((10, "streak10"), (50, "streak50")):
        s = t[t["rank_ret20"] <= k][["code", "_di"]].sort_values(["code", "_di"]).copy()
        new = (s["_di"] - s.groupby("code")["_di"].shift(1)) != 1
        grp = new.cumsum()
        s[col] = s.groupby(grp).cumcount() + 1
        t = t.merge(s[["code", "_di", col]], on=["code", "_di"], how="left")
    t["streak10"] = t["streak10"].fillna(0).astype(int)
    t["streak50"] = t["streak50"].fillna(0).astype(int)
    # 昨天在不在前 10（和 streak10 的区别：今天掉到 11~50 的票）
    y10 = t[t["rank_ret20"] <= 10][["code", "_di"]].copy()
    y10["_di"] = y10["_di"] + 1
    y10["in10_yday"] = True
    t = t.merge(y10, on=["code", "_di"], how="left")
    t["in10_yday"] = t["in10_yday"].fillna(False).astype(bool)
    # 影子模型分数：全候选池内名次
    parts = []
    for p_ in (EXP / "screen_base_v1" / "all_base_s7.parquet", EXP / "screen_base_v1_dev" / "all_base_s7.parquet",
               EXP / "rulecore_v1" / "all_base_dec.parquet"):
        a = pd.read_parquet(p_)
        a["code"] = a["code"].astype(str).str.zfill(6)
        parts.append(a)
    mp = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"], keep="last")
    pk = df[["date", "code"]].merge(mp, on=["date", "code"], how="left")
    pk = pk[np.isfinite(pk["p"])]
    pk = pk.sort_values(["date", "p"], ascending=[True, False], kind="mergesort")
    pk["model_rank"] = pk.groupby("date").cumcount() + 1
    pk = pk.rename(columns={"p": "model_p"})
    t = t.merge(pk, on=["date", "code"], how="left")
    say("影子模型分数覆盖前 50 行 %.1f%%（%s..%s）" % (100 * np.isfinite(t["model_p"]).mean(), mp["date"].min(), mp["date"].max()))
    # 非价格数据
    alt = pd.read_parquet(ROOT / "research" / "altdata" / "top60.parquet")
    alt["code"] = alt["code"].astype(str).str.zfill(6)
    t = t.merge(alt[["date", "code"] + [c for c in ALT_COLS if c in alt.columns]], on=["date", "code"], how="left")
    t = t.merge(day.drop(columns=["base", "mret1"]).reset_index(), on="date", how="left")
    t["seg"] = seg_of(t["date"])
    t = t[t["date"] >= "2022-01-01"].reset_index(drop=True)
    t["hit"] = t["y_open"].fillna(0).astype(float)
    t.to_parquet(TOP50, index=False)
    say("-> %s %d 行 %d 列 %.0fs" % (TOP50, len(t), t.shape[1], time.time() - t0))
    return t


def load(seg: str | None = None) -> pd.DataFrame:
    t = pd.read_parquet(TOP50)
    return t if seg is None else t[t["seg"] == seg].reset_index(drop=True)


# ---------------------------------------------------------------
#  基线曲线
# ---------------------------------------------------------------
class Baseline:
    """每段：D 天 × 50 名的命中矩阵 H[d, r]；B(k) 的每日命中 = 前 floor(k) 之和 + 小数 × 第 ceil(k) 名。"""

    def __init__(self, t: pd.DataFrame):
        self.dates = np.array(sorted(t["date"].unique()))
        self.D = len(self.dates)
        di = pd.Series(np.arange(self.D), index=self.dates)
        H = np.zeros((self.D, KEEP))
        r = t["rank_ret20"].to_numpy() - 1
        H[t["date"].map(di).to_numpy(), r] = t["hit"].to_numpy()
        self.H = H
        self.cum = np.cumsum(H, axis=1)                     # cum[:, k-1] = 前 k 名命中数

    def hits_k(self, k: float) -> np.ndarray:
        """B(k) 每日命中（k 可为小数，0 < k <= 50）。"""
        k = float(k)
        if k <= 0:
            return np.zeros(self.D)
        f = int(np.floor(k))
        frac = k - f
        h = self.cum[:, f - 1].copy() if f >= 1 else np.zeros(self.D)
        if frac > 1e-12 and f < KEEP:
            h += frac * self.H[:, f]
        return h

    def precision_k(self, k: float) -> float:
        return float(self.hits_k(k).sum() / (k * self.D))

    def hits_nd(self, n: np.ndarray) -> np.ndarray:
        """日配基线：第 d 天取前 n_d 名。"""
        n = np.asarray(n, int)
        out = np.zeros(self.D)
        m = n > 0
        out[m] = self.cum[np.arange(self.D)[m], np.minimum(n[m], KEEP) - 1]
        return out


def nw_se(x) -> float:
    return P.nw_se(np.asarray(x, float))


def nw_t(x) -> float:
    return P.nw_t(np.asarray(x, float))


def boot_ratio(h: np.ndarray, n: np.ndarray, b: int = 4000, block: int = 20, seed: int = 20261018) -> tuple[float, float]:
    """块自举比值 Σh/Σn 的 95% 区间。"""
    import evalkit as K
    D = len(h)
    if D < 2 * block or n.sum() <= 0:
        return (np.nan, np.nan)
    idx = K._mbb_index(D, block, b, np.random.default_rng(seed))
    hs = h[idx].sum(axis=1)
    ns = n[idx].sum(axis=1)
    r = hs / np.maximum(ns, 1)
    r[ns == 0] = np.nan
    return tuple(float(x) for x in np.nanquantile(r, [0.025, 0.975]))


def evaluate(t: pd.DataFrame, sel: np.ndarray, B: Baseline, boot: bool = True) -> dict:
    """规则 = top50 行上的布尔选择 sel。返回精确率、量、匹配量基线差、按天序列 u。"""
    di = pd.Series(np.arange(B.D), index=B.dates)
    rows = t[sel]
    n = np.zeros(B.D)
    h = np.zeros(B.D)
    if len(rows):
        g = rows.groupby("date")["hit"].agg(["size", "sum"])
        ii = di[g.index].to_numpy()
        n[ii] = g["size"].to_numpy()
        h[ii] = g["sum"].to_numpy()
    V = float(n.sum())
    out = {"V": int(V), "days_out": int((n > 0).sum()), "D": int(B.D), "vbar": V / B.D, "empty_share": float((n == 0).mean())}
    if V <= 0:
        out.update({"P": np.nan, "P_base": np.nan, "dP": np.nan, "t": np.nan, "u": np.zeros(B.D), "se": np.nan})
        return out
    Pr = float(h.sum() / V)
    kbar = V / B.D
    hb = B.hits_k(min(kbar, KEEP))
    u = h - hb
    out["P"] = Pr
    out["P_base"] = float(hb.sum() / V)
    out["dP"] = float(u.sum() / V)
    out["t"] = nw_t(u)
    out["se"] = nw_se(u) * B.D / V                       # ΔP 的 NW se
    uw = h - B.hits_nd(n)
    out["dP_within"] = float(uw.sum() / V)
    out["t_within"] = nw_t(uw)
    out["u"] = u
    out["h"], out["n"] = h, n
    # 精确率自身的 NW se
    u0 = h - Pr * n
    out["P_se"] = nw_se(u0) * B.D / V
    if boot:
        out["P_lo"], out["P_hi"] = boot_ratio(h, n)
    return out


def fmt_eval(name: str, e: dict, seg: str = "") -> str:
    if not np.isfinite(e.get("P", np.nan)):
        return "%-34s %s V=0" % (name, seg)
    s = "%-34s %-6s V %5d  天 %3d/%3d  k̄ %5.2f  P %5.1f%%" % (name, seg, e["V"], e["days_out"], e["D"], e["vbar"], 100 * e["P"])
    if "P_lo" in e:
        s += " [%4.1f,%4.1f]" % (100 * e["P_lo"], 100 * e["P_hi"])
    s += "  B(k̄) %5.1f%%  ΔP %+5.1fpp (t %+5.2f)  日配 %+5.1f" % (100 * e["P_base"], 100 * e["dP"], e["t"], 100 * e["dP_within"])
    return s


def mde80(se_pp: float, c: float = 2.5) -> float:
    return (c + 0.84) * se_pp


def pp_series(e: dict) -> pd.Series:
    """u 换算成对合并精确率的贡献（pp 单位的按天序列），索引是日期。"""
    return pd.Series(100 * e["u"] * e["D"] / max(e["V"], 1), index=e["dates"] if "dates" in e else None)
