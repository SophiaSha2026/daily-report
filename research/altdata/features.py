"""非价格数据派生特征，只算 ret20 前 60 的 (代码, 日期)，一律按「T 日收盘后可得」截断。

    PYTHONIOENCODING=utf-8 python research/altdata/features.py

输入：research/polish 的 pool.parquet（候选池 + 标签 + amt20 等）、research/altdata/raw/<表>.parquet。
输出：research/altdata/top60.parquet（date code rank_ret20 y_open y_up amt20 board + 全部派生特征）。

可得时点（见 fetch.py TABLES.avail 和 progress.txt）：
    龙虎榜 / 大宗：TRADE_DATE 当天收盘后公布，T 日那行可用。
    两融：交易所 T+1 早上发布，T 日收盘只能用 DATE <= T-1 的行（这里整体错后一个交易日）。
    股东户数：按 HOLD_NOTICE_DATE，公告日 >= 的第一个交易日起可用（周末公告算下周一）。
    机构调研：按 NOTICE_DATE，同上（拉到的话）。
    质押：周五数据，按 +3 天（下周一）起可用（拉到的话）。
窗口按交易日数，不按日历（和 pool 的 ret20 一致）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
import common_pl as P  # noqa: E402

RAW = HERE / "raw"
KEEP_RANK = 60
START = "2021-06-01"     # 稠密网格起点（留 90 个交易日回看给 2022-01）


def say(*a):
    print(*a, flush=True)


class Grid:
    """代码 x 交易日 的稠密网格。"""

    def __init__(self, codes: list[str], days: list[str]):
        self.codes = codes
        self.days = np.array(days)
        self.ci = {c: i for i, c in enumerate(codes)}
        self.di = {d: i for i, d in enumerate(days)}

    def day_index(self, dates: pd.Series, how: str = "exact") -> np.ndarray:
        """exact：只认交易日（非交易日 -> -1）；next：>= 该日的第一个交易日（公告日对齐）。"""
        d = dates.astype(str).str[:10].to_numpy()
        pos = np.searchsorted(self.days, d, side="left")
        pos = np.clip(pos, 0, len(self.days))
        ok = pos < len(self.days)
        if how == "exact":
            ok &= self.days[np.minimum(pos, len(self.days) - 1)] == d
        return np.where(ok, pos, -1)

    def dense(self, code: pd.Series, di: np.ndarray, val: np.ndarray, agg: str = "sum") -> np.ndarray:
        ci = code.map(self.ci).to_numpy()
        m = np.isfinite(ci.astype(float)) & (di >= 0)
        ci = ci[m].astype(int)
        di = di[m]
        v = np.asarray(val, float)[m]
        A = np.zeros((len(self.codes), len(self.days)), np.float64)
        if agg == "sum":
            np.add.at(A, (ci, di), np.nan_to_num(v))
        elif agg == "last":
            A[:] = np.nan
            order = np.argsort(di, kind="mergesort")
            A[ci[order], di[order]] = v[order]
        elif agg == "max":
            A[:] = -np.inf
            np.maximum.at(A, (ci, di), v)
            A[~np.isfinite(A)] = 0
        return A

    @staticmethod
    def wsum(A: np.ndarray, n: int) -> np.ndarray:
        """窗口 [t-n+1, t] 的和。"""
        cs = np.cumsum(A, axis=1)
        out = cs.copy()
        out[:, n:] = cs[:, n:] - cs[:, :-n]
        return out

    @staticmethod
    def ffill(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """沿交易日前向填充；同时给出「最近一次有值」的列号（没有就 -1）。"""
        n = A.shape[1]
        idx = np.where(np.isfinite(A), np.arange(n)[None, :], -1)
        idx = np.maximum.accumulate(idx, axis=1)
        rows = np.arange(A.shape[0])[:, None]
        val = np.where(idx >= 0, A[rows, np.maximum(idx, 0)], np.nan)
        return val, idx

    def pick(self, A: np.ndarray, code: pd.Series, di: np.ndarray) -> np.ndarray:
        ci = code.map(self.ci).to_numpy()
        return A[ci.astype(int), di]


def main():
    t0 = time.time()
    pool = P.load_pool(["code", "date", "board", "pool", "lret20", "y_open", "y_up", "amt20", "amt1", "mcap", "turn1", "ret20"])
    pool = pool[(pool["date"] >= START)].reset_index(drop=True)
    days = sorted(pool["date"].unique())
    g = Grid(sorted(pool["code"].unique()), days)
    say("pool %d 行 %d 天 %d 只 %.0fs" % (len(pool), len(days), len(g.codes), time.time() - t0))
    # ret20 前 60（候选池内）
    q = pool[pool["pool"]].copy()
    q = q.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    q["rank_ret20"] = q.groupby("date").cumcount() + 1
    q = q[(q["rank_ret20"] <= KEEP_RANK) & (q["date"] >= "2021-12-01")].reset_index(drop=True)
    q["di"] = g.day_index(q["date"])
    assert (q["di"] >= 0).all()
    di = q["di"].to_numpy()
    code = q["code"]
    amt20 = q["amt20"].to_numpy(float)
    say("前 60 行 %d（%s..%s）" % (len(q), q["date"].min(), q["date"].max()))

    # ---- 龙虎榜明细：同一 (代码, 日) 多条上榜原因只算一次 ----
    f = RAW / "lhb_detail.parquet"
    if f.exists():
        lhb = pd.read_parquet(f, columns=["SECURITY_CODE", "TRADE_DATE", "BILLBOARD_NET_AMT", "BILLBOARD_BUY_AMT", "BILLBOARD_SELL_AMT", "ACCUM_AMOUNT"])
        lhb["code"] = lhb["SECURITY_CODE"].astype(str).str.zfill(6)
        lhb = lhb.drop_duplicates(["code", "TRADE_DATE"])
        ei = g.day_index(lhb["TRADE_DATE"])
        on = g.dense(lhb["code"], ei, np.ones(len(lhb)))
        net = g.dense(lhb["code"], ei, lhb["BILLBOARD_NET_AMT"].to_numpy(float))
        buy = g.dense(lhb["code"], ei, lhb["BILLBOARD_BUY_AMT"].to_numpy(float))
        q["lhb_today"] = g.pick(on, code, di)
        q["lhb_n5"] = g.pick(g.wsum(on, 5), code, di)
        q["lhb_n20"] = g.pick(g.wsum(on, 20), code, di)
        q["lhb_net5_share"] = g.pick(g.wsum(net, 5), code, di) / (5 * amt20)
        q["lhb_net20_share"] = g.pick(g.wsum(net, 20), code, di) / (20 * amt20)
        q["lhb_buy5_share"] = g.pick(g.wsum(buy, 5), code, di) / (5 * amt20)
        # 单位核对：龙虎榜 ACCUM_AMOUNT（当日成交额）对 pool 的 amt1
        chk = q[q["lhb_today"] > 0][["code", "date", "amt1"]].merge(
            lhb[["code", "TRADE_DATE", "ACCUM_AMOUNT"]].rename(columns={"TRADE_DATE": "date"}), on=["code", "date"])
        r = (chk["ACCUM_AMOUNT"] / chk["amt1"]).median() if len(chk) else np.nan
        say("龙虎榜 %d 行；上榜行 %d；ACCUM_AMOUNT/amt1 中位 %.3f（应≈1）" % (len(lhb), len(chk), r))
    f = RAW / "lhb_org.parquet"
    if f.exists():
        lo = pd.read_parquet(f, columns=["SECURITY_CODE", "TRADE_DATE", "NET_BUY_AMT", "BUY_AMT", "BUY_TIMES"])
        lo["code"] = lo["SECURITY_CODE"].astype(str).str.zfill(6)
        lo = lo.drop_duplicates(["code", "TRADE_DATE"])
        ei = g.day_index(lo["TRADE_DATE"])
        net = g.dense(lo["code"], ei, lo["NET_BUY_AMT"].to_numpy(float))
        buy = g.dense(lo["code"], ei, lo["BUY_AMT"].to_numpy(float))
        q["org_net5_share"] = g.pick(g.wsum(net, 5), code, di) / (5 * amt20)
        q["org_net20_share"] = g.pick(g.wsum(net, 20), code, di) / (20 * amt20)
        q["org_buy5_share"] = g.pick(g.wsum(buy, 5), code, di) / (5 * amt20)
        say("机构买卖 %d 行" % len(lo))

    # ---- 两融（T-1）----
    f = RAW / "rzrq.parquet"
    if f.exists():
        rz = pd.read_parquet(f, columns=["SCODE", "DATE", "RZYE", "RZMRE", "RZYEZB", "RZCHE", "RQYL"])
        rz["code"] = rz["SCODE"].astype(str).str.zfill(6)
        ei = g.day_index(rz["DATE"])
        bal = g.dense(rz["code"], ei, rz["RZYE"].to_numpy(float), agg="last")
        bal_f, _ = g.ffill(bal)
        bre = g.dense(rz["code"], ei, rz["RZMRE"].to_numpy(float))
        zb = g.dense(rz["code"], ei, rz["RZYEZB"].to_numpy(float), agg="last")
        zb_f, _ = g.ffill(zb)
        has = g.dense(rz["code"], ei, np.ones(len(rz)))
        dprev = np.maximum(di - 1, 0)          # T-1
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = g.pick(bal_f, code, dprev)
            b6 = g.pick(bal_f, code, np.maximum(di - 6, 0))
            b21 = g.pick(bal_f, code, np.maximum(di - 21, 0))
            q["rz_in"] = g.pick(g.wsum(has, 5), code, dprev) > 0
            q["rz_chg5"] = b1 / b6 - 1
            q["rz_chg20"] = b1 / b21 - 1
            q["rz_buy5_share"] = g.pick(g.wsum(bre, 5), code, dprev) / (5 * amt20)
            q["rz_level"] = g.pick(zb_f, code, dprev)
        say("两融 %d 行；前 60 里两融标的占 %.1f%%" % (len(rz), 100 * q["rz_in"].mean()))

    # ---- 大宗交易 ----
    f = RAW / "dzjy.parquet"
    if f.exists():
        dz = pd.read_parquet(f, columns=["SECURITY_CODE", "TRADE_DATE", "DEAL_AMT", "PREMIUM_RATIO"])
        dz["code"] = dz["SECURITY_CODE"].astype(str).str.zfill(6)
        ei = g.day_index(dz["TRADE_DATE"])
        amt = g.dense(dz["code"], ei, dz["DEAL_AMT"].to_numpy(float))
        disc = dz["PREMIUM_RATIO"].to_numpy(float) <= -0.05
        amt_d = g.dense(dz["code"], ei, np.where(disc, dz["DEAL_AMT"].to_numpy(float), 0.0))
        n = g.dense(dz["code"], ei, np.ones(len(dz)))
        q["dz_n20"] = g.pick(g.wsum(n, 20), code, di)
        q["dz_amt20_share"] = g.pick(g.wsum(amt, 20), code, di) / (20 * amt20)
        q["dz_disc20_share"] = g.pick(g.wsum(amt_d, 20), code, di) / (20 * amt20)
        q["dz_n5"] = g.pick(g.wsum(n, 5), code, di)
        say("大宗 %d 行" % len(dz))

    # ---- 股东户数（按公告日）----
    f = RAW / "gdhs.parquet"
    if f.exists():
        gd = pd.read_parquet(f, columns=["SECURITY_CODE", "HOLD_NOTICE_DATE", "END_DATE", "HOLDER_NUM", "PRE_HOLDER_NUM", "HOLDER_NUM_RATIO"])
        gd["code"] = gd["SECURITY_CODE"].astype(str).str.zfill(6)
        gd = gd[gd["HOLDER_NUM"] >= 500]       # 上市前名册几十户的行丢掉（features.py HOLDER_MIN 同一类）
        gd["chg"] = pd.to_numeric(gd["HOLDER_NUM_RATIO"], errors="coerce") / 100
        gd.loc[gd["PRE_HOLDER_NUM"] < 500, "chg"] = np.nan
        gd = gd.sort_values(["code", "HOLD_NOTICE_DATE", "END_DATE"]).drop_duplicates(["code", "HOLD_NOTICE_DATE"], keep="last")
        gd["chg3"] = gd.groupby("code")["chg"].transform(lambda s: s.rolling(3, min_periods=1).sum())
        ei = g.day_index(gd["HOLD_NOTICE_DATE"], how="next")
        c1 = g.dense(gd["code"], ei, gd["chg"].to_numpy(float), agg="last")
        c3 = g.dense(gd["code"], ei, gd["chg3"].to_numpy(float), agg="last")
        c1f, last = g.ffill(c1)
        c3f, _ = g.ffill(c3)
        q["gd_chg1"] = g.pick(c1f, code, di)
        q["gd_chg3"] = g.pick(c3f, code, di)
        li = g.pick(last, code, di)
        q["gd_stale"] = np.where(li >= 0, di - li, np.nan)
        say("股东户数 %d 行；前 60 里有户数的 %.1f%%" % (len(gd), 100 * np.isfinite(q["gd_chg1"]).mean()))

    # ---- 机构调研（按公告日）----
    f = RAW / "survey.parquet"
    if f.exists():
        sv = pd.read_parquet(f, columns=["SECURITY_CODE", "NOTICE_DATE", "RECEIVE_START_DATE"])
        sv["code"] = sv["SECURITY_CODE"].astype(str).str.zfill(6)
        ev = sv.groupby(["code", "NOTICE_DATE"]).size().rename("n").reset_index()      # 一次公告的机构数
        ei = g.day_index(ev["NOTICE_DATE"], how="next")
        cnt = g.dense(ev["code"], ei, np.ones(len(ev)))
        orgs = g.dense(ev["code"], ei, ev["n"].to_numpy(float))
        q["sv_n30"] = g.pick(g.wsum(cnt, 21), code, di)
        q["sv_n90"] = g.pick(g.wsum(cnt, 63), code, di)
        q["sv_orgs30"] = g.pick(g.wsum(orgs, 21), code, di)
        q["sv_orgs90"] = g.pick(g.wsum(orgs, 63), code, di)
        say("机构调研 %d 行 %d 次公告" % (len(sv), len(ev)))

    # ---- 质押（周度，+3 天可用）----
    f = RAW / "pledge.parquet"
    if f.exists():
        pl = pd.read_parquet(f, columns=["SECURITY_CODE", "TRADE_DATE", "PLEDGE_RATIO"])
        pl["code"] = pl["SECURITY_CODE"].astype(str).str.zfill(6)
        av = (pd.to_datetime(pl["TRADE_DATE"]) + pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d")
        ei = g.day_index(av, how="next")
        r = g.dense(pl["code"], ei, pd.to_numeric(pl["PLEDGE_RATIO"], errors="coerce").to_numpy(float), agg="last")
        rf, _ = g.ffill(r)
        q["pl_ratio"] = g.pick(rf, code, di)
        q["pl_chg13"] = q["pl_ratio"] - g.pick(rf, code, np.maximum(di - 65, 0))
        say("质押 %d 行" % len(pl))

    q = q.drop(columns=["di", "pool"])
    q.to_parquet(HERE / "top60.parquet", index=False)
    feats = [c for c in q.columns if c not in ("code", "date", "board", "lret20", "y_open", "y_up", "amt20", "amt1", "mcap", "turn1", "ret20", "rank_ret20")]
    say("-> top60.parquet %d 行，特征 %s  %.0fs" % (len(q), feats, time.time() - t0))
    P.say(q[feats].describe().T.to_string())
    with (HERE / "progress.txt").open("a", encoding="utf-8") as fh:
        from bjtime import now_bj
        fh.write("[%s BJ] 特征：top60.parquet %d 行（ret20 前 60，%s..%s），特征 %s\n" % (
            now_bj().strftime("%Y-%m-%d %H:%M"), len(q), q["date"].min(), q["date"].max(), feats))


if __name__ == "__main__":
    main()
