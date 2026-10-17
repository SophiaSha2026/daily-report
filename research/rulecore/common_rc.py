"""
规则核预登记判定（research/rulecore，2026-10-16，docs/rethink_2026-10-16.md 拍板 B / C / D）公用件。

不改 src / tools / state / config.yaml；research/screen、research/leak 的文件只 import。
产物都在 data/breakout/raw/exp/rulecore/（特征表、事件表）和 data/breakout/raw/exp/rulecore_v1/（各臂的 all_ / top_ 名次表、
prereg.json、diff 序列，evalkit 的约定）。

数据（两段同一口径）
    价格   research/screen 的 pit 价 px_screen.parquet（新浪不复权 / 归一前复权因子，不取整），2020-07 起，到 2025-12-31
    标签   labels_screen.parquet（y_up、y_open、r20_open、yizi1，pit 价上算）；2025-12-04..31 的标签在那张表里没关窗，
           从 research/leak 的快照（labels_open_snap / train_snap，同一份新浪价格的取整版）补上，两份在开发集重叠行
           y_open 一致 99.987%（research/screen/report.md 第 3 节）
    现状   筛选集 screen_base_v1/all_base_s7（2022-01..2024-05），开发集 screen_base_v1_dev/all_base_s7（..2025-12-03）
           + 本目录 pc_arm.py 用同一条路补的 2025-12 整月（parts/2025-12/all_base_s7）
    两段   筛选集 2022-01..2024-05（582 天）、开发集 2024-06..2025-12（387 天），合并 969 天

为什么开发集也用 screen 长表而不用 research/leak 的 pit 表
    1 一套价格、一套标签、一条训练协议（每个测试月前 3 年、validate.train_slice 的净化）贯穿 969 天，两段的差可以合并；
    2 长表上的现状 base 更强（开发集 367 天 13.30% 对 leak 表 12.04%），对照更难赢；
    3 leak 表只到 2023-05-30 起、开发集前几个月训练窗口只有一年，长表没有这个问题。

硬剔除（所有臂同一份，排名前生效，和生产 daily.stage_scan / risk_filter 同口径）
    ST（validate.st_codes 静态名单，和回测惯例一致）、pit 收盘价 < 2 元、表里累计不足 daily.MIN_HISTORY_DAYS（120）根、
    最近 20 根里有 >= SUSP_GAP 个交易日的停牌缺口、
    近 30 天股东减持（corp_events.reduce_hits 判据）、未来 30 天解禁占流通市值 >= daily.RELEASE_MIN_SHARE、
    近 60 天有增发（events_rc.py 拉的全市场表，按公告日 / 解禁日 / 发行日期判）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "screen", ROOT / "research" / "stats"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

HERE = Path(__file__).resolve().parent
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
SCR = EXP / "screen"
LEAK = EXP / "leak"
RC = EXP / "rulecore"                 # 特征表、事件表
EVENTS = RC / "events"
NAME = "rulecore_v1"                  # evalkit 实验名：EXP / rulecore_v1
FAMILY = "breakout_y50_top10_rulecore"
PROGRESS = HERE / "progress.txt"
SEGS = {"screen": ("2022-01", "2024-05"), "dev": ("2024-06", "2025-12")}
TRAIN_YEARS = 3
SUSP_GAP = 5                          # 最近 20 根里缺 >= 5 个交易日 = 停牌过，复牌不满 20 根不上榜
PRICE_MIN = 2.0
INPUTS = ("lret20", "vol60", "turn20", "amp20")          # 规则核的四个连续输入（当日内 z 化）
BOARDS = ("main", "chinext", "star", "bj")
K_TOP = 10
TOP_KEEP = 200


def say(*a):
    print(*a, flush=True)


def progress(text: str) -> None:
    from bjtime import now_bj
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def in_prod_window() -> bool:
    from bjtime import now_bj
    t = now_bj()
    hm = t.hour * 60 + t.minute
    return 16 * 60 <= hm < 18 * 60 + 30


def month_shift(m: str, k: int) -> str:
    y, mm = int(m[:4]), int(m[5:7])
    t = y * 12 + (mm - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def months_in(seg: str) -> list[str]:
    import evalkit as K
    return K.months_in(SEGS[seg])


def board_of(code: pd.Series) -> pd.Series:
    return pd.Series(np.where(code.str[:2] == "30", "chinext",
                              np.where(code.str[:2] == "68", "star",
                                       np.where(code.str[:1].isin(["9", "8", "4"]), "bj", "main"))), index=code.index)


# ---------------------------------------------------------------
#  特征表（算一次缓存）
# ---------------------------------------------------------------
def _labels() -> pd.DataFrame:
    lab = pd.read_parquet(SCR / "labels_screen.parquet", columns=["code", "date", "y_up", "y_open", "r20_open", "yizi1"])
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    # 2025-12-04..31：长表的标签窗口没关（表截到 2025-12-31），从 leak 快照补
    need = lab["date"] >= "2025-12-01"
    miss = need & ~np.isfinite(lab["y_open"])
    lo = pd.read_parquet(LEAK / "labels_open_snap.parquet", columns=["code", "date", "y_open", "r20_open", "yizi1"],
                         filters=[("date", ">=", "2025-12-01"), ("date", "<=", "2025-12-31")])
    lo["code"] = lo["code"].astype(str).str.zfill(6)
    lu = pd.read_parquet(LEAK / "train_snap.parquet", columns=["code", "date", "y_up"],
                         filters=[("date", ">=", "2025-12-01"), ("date", "<=", "2025-12-31")])
    lu["code"] = lu["code"].astype(str).str.zfill(6)
    patch = lo.merge(lu, on=["code", "date"], how="left")
    sub = lab.loc[miss, ["code", "date"]].merge(patch, on=["code", "date"], how="left")
    for c in ("y_up", "y_open", "r20_open", "yizi1"):
        lab.loc[miss, c] = sub[c].to_numpy()
    lab["patched"] = miss.to_numpy()
    say("标签 %d 行；2025-12 补 %d 行（补到 %d 行有 y_open）" % (len(lab), int(miss.sum()), int(np.isfinite(lab.loc[miss, "y_open"]).sum())))
    return lab


def build_features(force: bool = False) -> pd.DataFrame:
    f = RC / "feat.parquet"
    if f.exists() and not force:
        return pd.read_parquet(f)
    import daily as D
    import validate as V
    RC.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(SCR / "px_screen.parquet",
                         columns=["code", "date", "open", "high", "low", "close", "turnover", "outstanding_share"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    px["ret20"] = g["close"].pct_change(20)
    px["lret20"] = np.log1p(px["ret20"].clip(lower=-0.99))
    lr = np.log(px["close"]).groupby(px["code"]).diff()
    px["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std())
    px["amp"] = px["high"] / px["low"] - 1
    px["amp20"] = px.groupby("code", sort=False)["amp"].transform(lambda s: s.rolling(20).mean())
    px["turn20"] = px.groupby("code", sort=False)["turnover"].transform(lambda s: s.rolling(20).mean())
    px["nbar"] = g.cumcount() + 1
    # 停牌缺口：按全市场交易日序号数
    dates = np.array(sorted(px["date"].unique()))
    didx = pd.Series(np.arange(len(dates)), index=dates)
    px["_di"] = px["date"].map(didx).to_numpy()
    gap = (px["_di"] - px.groupby("code", sort=False)["_di"].shift(1) - 1).fillna(0)
    px["susp20"] = gap.groupby(px["code"]).transform(lambda s: s.rolling(20, min_periods=1).max()) >= SUSP_GAP
    # 第 20 根 K 线的日期（validate.train_slice 的第二个条件）
    px["end20"] = g["date"].shift(-20).fillna("9999-12-31")
    # 之后 20 根（次日开盘起算）：第 20 根收盘收益、最大回撤（三句事实）
    from numpy.lib.stride_tricks import sliding_window_view as swv

    def fut(gr):
        n = len(gr)
        lo_, cl, op = gr["low"].to_numpy(float), gr["close"].to_numpy(float), gr["open"].to_numpy(float)
        mn, c20 = np.full(n, np.nan), np.full(n, np.nan)
        if n > 20:
            mn[:n - 20] = swv(lo_[1:], 20).min(axis=1)
            c20[:n - 20] = cl[20:]
        o1 = np.append(op[1:], np.nan)
        return pd.DataFrame({"dd20": mn / o1 - 1, "c20": c20 / o1 - 1}, index=gr.index)

    px = px.join(pd.concat([fut(gr) for _, gr in px.groupby("code", sort=False)]))
    px["board"] = board_of(px["code"])
    st = V.st_codes()
    px["elig0"] = (~px["code"].isin(st) & (px["close"] >= PRICE_MIN) & (px["nbar"] >= D.MIN_HISTORY_DAYS)
                   & ~px["susp20"] & np.isfinite(px[list(INPUTS)]).all(axis=1))
    # 当日内 z 化：均值 / 标准差在当天的候选池（elig0）上算，所有行都按它变换
    for c in INPUTS:
        s = px[c].where(px["elig0"])
        mu = s.groupby(px["date"]).transform("mean")
        sd = s.groupby(px["date"]).transform("std")
        px["z_" + c] = (px[c] - mu) / (sd + 1e-12)
    lab = _labels()
    px = px.merge(lab, on=["code", "date"], how="left")
    keep = ["code", "date", "board", "close", "ret20", "nbar", "susp20", "elig0", "end20", "dd20", "c20",
            "y_up", "y_open", "r20_open", "yizi1", "patched"] + list(INPUTS) + ["z_" + c for c in INPUTS]
    px = px[keep].reset_index(drop=True)
    px.to_parquet(f, index=False)
    say("特征表 %d 行 %s..%s -> %s" % (len(px), px["date"].min(), px["date"].max(), f))
    return px


# ---------------------------------------------------------------
#  事件剔除（减持 / 解禁 / 定增），按 (代码, 日期) 向量化判定
# ---------------------------------------------------------------
def _asof_last(df: pd.DataFrame, ev: pd.DataFrame, col: str) -> pd.Series:
    """每行 (code, date) 之前（含当天）最近一条事件的日期；没有就 NaT。"""
    a = df[["code", "date"]].copy()
    a["_i"] = np.arange(len(a))
    a["_d"] = pd.to_datetime(a["date"])
    e = ev[["code", col]].copy()
    e["_e"] = pd.to_datetime(e[col])
    e = e.dropna().sort_values("_e")
    a = a.sort_values("_d")
    m = pd.merge_asof(a, e[["code", "_e"]], left_on="_d", right_on="_e", by="code", direction="backward")
    return m.sort_values("_i")["_e"].reset_index(drop=True)


def event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """返回和 df 同序的三列布尔：reduce30 / release30 / placement60，和一列 any。事件表缺哪张就记 NaN -> False 并告警。"""
    import corp_events as CE
    import daily as D
    out = pd.DataFrame(index=df.index)
    d = pd.to_datetime(df["date"])
    status = json.loads((EVENTS / "status.json").read_text(encoding="utf-8")) if (EVENTS / "status.json").exists() else {}
    # 减持：公告按 reduce_hits 的标题判据过滤后，和减持记录并成一张「减持事件日」表
    ra, rr = EVENTS / "reduce_ann.parquet", EVENTS / "reduce_rec.parquet"
    if ra.exists() and rr.exists():
        ann = pd.read_parquet(ra)
        t = ann["title"].fillna("")
        keep = (ann["node"].astype(str) == "7") & t.str.contains("减持") & ~t.str.contains("回购") & ~t.map(lambda x: bool(CE.NOT_REDUCE.search(x)))
        ev = pd.concat([ann.loc[keep, ["code", "date"]], pd.read_parquet(rr)[["code", "date"]]], ignore_index=True)
        ev["code"] = ev["code"].astype(str).str.zfill(6)
        last = _asof_last(df, ev, "date")
        out["reduce30"] = ((d - last).dt.days <= 30).fillna(False).to_numpy()
    else:
        say("!! 减持表缺，这一项不剔")
        out["reduce30"] = False
    # 解禁：未来 30 天（含当天之后）解禁占比合计 >= RELEASE_MIN_SHARE
    rl = EVENTS / "release.parquet"
    if rl.exists():
        rel = pd.read_parquet(rl)
        rel["code"] = rel["code"].astype(str).str.zfill(6)
        rel["_e"] = pd.to_datetime(rel["date"])
        rel = rel.sort_values(["code", "_e"])
        rel["_cum"] = rel.groupby("code")["share"].cumsum()
        a = df[["code"]].copy()
        a["_i"] = np.arange(len(a))
        a["_d"] = d.to_numpy()
        a["_d30"] = a["_d"] + pd.Timedelta(days=30)
        r = rel[["code", "_e", "_cum"]].sort_values("_e")
        m1 = pd.merge_asof(a.sort_values("_d30"), r, left_on="_d30", right_on="_e", by="code", direction="backward")
        m0 = pd.merge_asof(a.sort_values("_d"), r, left_on="_d", right_on="_e", by="code", direction="backward")
        c1 = m1.sort_values("_i")["_cum"].fillna(0).to_numpy()
        c0 = m0.sort_values("_i")["_cum"].fillna(0).to_numpy()
        out["release30"] = (c1 - c0) >= D.RELEASE_MIN_SHARE
    else:
        say("!! 解禁表缺，这一项不剔")
        out["release30"] = False
    pl = EVENTS / "placement.parquet"
    if pl.exists():
        last = _asof_last(df, pd.read_parquet(pl), "date")
        out["placement60"] = ((d - last).dt.days <= 60).fillna(False).to_numpy()
    else:
        say("!! 增发表缺，这一项不剔")
        out["placement60"] = False
    out["any"] = out[["reduce30", "release30", "placement60"]].any(axis=1)
    out.attrs["status"] = status
    return out


# ---------------------------------------------------------------
#  训练切片（validate.train_slice 同一条规则：日期 < 净化线 且 第 20 根 K 线落在测试月之前；再加 3 年窗口）
# ---------------------------------------------------------------
def train_mask(df: pd.DataFrame, month: str) -> np.ndarray:
    import validate as V
    cut = V.purge_cut(df, month)
    lo = month_shift(month, -12 * TRAIN_YEARS) + "-01"
    return ((df["date"] < cut) & (df["end20"] < month + "-01") & (df["date"] >= lo)).to_numpy()


# ---------------------------------------------------------------
#  名次表（evalkit 约定：date code board y_up y_open p score rank），所有臂同一份候选池
# ---------------------------------------------------------------
def rank_table(df: pd.DataFrame, p: pd.Series, elig: pd.Series, score: pd.Series | None = None,
               keep: int = TOP_KEEP) -> pd.DataFrame:
    """df 行集合 = 这一段有标签的全部行；p 和 elig 同序。排名只在 elig 行里按 p 降序。"""
    o = df[["date", "code", "board", "y_up", "y_open"]].copy()
    o["p"] = np.asarray(p, np.float32)
    o["score"] = 100.0 if score is None else np.asarray(score, float)
    o = o[elig.to_numpy(bool) & np.isfinite(o["p"])]
    o = o.sort_values(["date", "p"], ascending=[True, False])
    o["rank"] = o.groupby("date").cumcount() + 1
    return o[o["rank"] <= keep].reset_index(drop=True)


def daily_hit(t: pd.DataFrame, k: int = K_TOP, y: str = "y_open") -> pd.Series:
    return t[t["rank"] <= k].groupby("date")[y].mean()
