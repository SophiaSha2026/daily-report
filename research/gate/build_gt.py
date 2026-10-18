"""建 research/gate/table.parquet：候选池全部行 + 名次 + 日级市场状态 + 非价格数据（全候选池行）。

    PYTHONIOENCODING=utf-8 python research/gate/build_gt.py

非价格数据的可得时点照 research/altdata/features.py（按路径加载它的 Grid，不另写一份）：
龙虎榜 / 大宗 T 日收盘后可用；两融 T-1；户数 / 调研按公告日的下一个交易日起。
"""
from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd

import common_gt as G

ALT_RAW = G.ROOT / "research" / "altdata" / "raw"


def _grid_module():
    spec = importlib.util.spec_from_file_location("altdata_features_mod", G.ROOT / "research" / "altdata" / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def alt_features(t: pd.DataFrame) -> pd.DataFrame:
    """t 要有 code date amt20；返回同序的 ALT_COLS。"""
    t0 = time.time()
    mod = _grid_module()
    days = sorted(t["date"].unique())
    g = mod.Grid(sorted(t["code"].unique()), days)
    di = g.day_index(t["date"])
    assert (di >= 0).all()
    code = t["code"]
    amt20 = t["amt20"].to_numpy(float)
    out = pd.DataFrame(index=t.index)
    f = ALT_RAW / "lhb_detail.parquet"
    lhb = pd.read_parquet(f, columns=["SECURITY_CODE", "TRADE_DATE", "BILLBOARD_NET_AMT", "BILLBOARD_BUY_AMT"])
    lhb["code"] = lhb["SECURITY_CODE"].astype(str).str.zfill(6)
    lhb = lhb.drop_duplicates(["code", "TRADE_DATE"])
    ei = g.day_index(lhb["TRADE_DATE"])
    on = g.dense(lhb["code"], ei, np.ones(len(lhb)))
    net = g.dense(lhb["code"], ei, lhb["BILLBOARD_NET_AMT"].to_numpy(float))
    buy = g.dense(lhb["code"], ei, lhb["BILLBOARD_BUY_AMT"].to_numpy(float))
    out["lhb_today"] = g.pick(on, code, di)
    out["lhb_n5"] = g.pick(g.wsum(on, 5), code, di)
    out["lhb_n20"] = g.pick(g.wsum(on, 20), code, di)
    out["lhb_net5_share"] = g.pick(g.wsum(net, 5), code, di) / (5 * amt20)
    out["lhb_buy5_share"] = g.pick(g.wsum(buy, 5), code, di) / (5 * amt20)
    del on, net, buy, lhb
    lo = pd.read_parquet(ALT_RAW / "lhb_org.parquet", columns=["SECURITY_CODE", "TRADE_DATE", "NET_BUY_AMT", "BUY_AMT"])
    lo["code"] = lo["SECURITY_CODE"].astype(str).str.zfill(6)
    lo = lo.drop_duplicates(["code", "TRADE_DATE"])
    ei = g.day_index(lo["TRADE_DATE"])
    net = g.dense(lo["code"], ei, lo["NET_BUY_AMT"].to_numpy(float))
    buy = g.dense(lo["code"], ei, lo["BUY_AMT"].to_numpy(float))
    out["org_net5_share"] = g.pick(g.wsum(net, 5), code, di) / (5 * amt20)
    out["org_buy5_share"] = g.pick(g.wsum(buy, 5), code, di) / (5 * amt20)
    del net, buy, lo
    G.say("龙虎榜 %.0fs" % (time.time() - t0))
    rz = pd.read_parquet(ALT_RAW / "rzrq.parquet", columns=["SCODE", "DATE", "RZYE", "RZMRE", "RZYEZB"])
    rz["code"] = rz["SCODE"].astype(str).str.zfill(6)
    ei = g.day_index(rz["DATE"])
    bal = g.dense(rz["code"], ei, rz["RZYE"].to_numpy(float), agg="last")
    bal_f, _ = g.ffill(bal)
    del bal
    bre = g.dense(rz["code"], ei, rz["RZMRE"].to_numpy(float))
    zb = g.dense(rz["code"], ei, rz["RZYEZB"].to_numpy(float), agg="last")
    zb_f, _ = g.ffill(zb)
    del zb
    has = g.dense(rz["code"], ei, np.ones(len(rz)))
    del rz
    dprev = np.maximum(di - 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = g.pick(bal_f, code, dprev)
        b6 = g.pick(bal_f, code, np.maximum(di - 6, 0))
        b21 = g.pick(bal_f, code, np.maximum(di - 21, 0))
        out["rz_in"] = (g.pick(g.wsum(has, 5), code, dprev) > 0).astype(np.float32)
        out["rz_chg5"] = b1 / b6 - 1
        out["rz_chg20"] = b1 / b21 - 1
        out["rz_buy5_share"] = g.pick(g.wsum(bre, 5), code, dprev) / (5 * amt20)
        out["rz_level"] = g.pick(zb_f, code, dprev)
    del bal_f, bre, zb_f, has
    G.say("两融 %.0fs" % (time.time() - t0))
    dz = pd.read_parquet(ALT_RAW / "dzjy.parquet", columns=["SECURITY_CODE", "TRADE_DATE", "DEAL_AMT", "PREMIUM_RATIO"])
    dz["code"] = dz["SECURITY_CODE"].astype(str).str.zfill(6)
    ei = g.day_index(dz["TRADE_DATE"])
    amt = g.dense(dz["code"], ei, dz["DEAL_AMT"].to_numpy(float))
    disc = dz["PREMIUM_RATIO"].to_numpy(float) <= -0.05
    amt_d = g.dense(dz["code"], ei, np.where(disc, dz["DEAL_AMT"].to_numpy(float), 0.0))
    n = g.dense(dz["code"], ei, np.ones(len(dz)))
    out["dz_n20"] = g.pick(g.wsum(n, 20), code, di)
    out["dz_amt20_share"] = g.pick(g.wsum(amt, 20), code, di) / (20 * amt20)
    out["dz_disc20_share"] = g.pick(g.wsum(amt_d, 20), code, di) / (20 * amt20)
    del amt, amt_d, n, dz
    gd = pd.read_parquet(ALT_RAW / "gdhs.parquet", columns=["SECURITY_CODE", "HOLD_NOTICE_DATE", "END_DATE", "HOLDER_NUM", "PRE_HOLDER_NUM", "HOLDER_NUM_RATIO"])
    gd["code"] = gd["SECURITY_CODE"].astype(str).str.zfill(6)
    gd = gd[gd["HOLDER_NUM"] >= 500]
    gd["chg"] = pd.to_numeric(gd["HOLDER_NUM_RATIO"], errors="coerce") / 100
    gd.loc[gd["PRE_HOLDER_NUM"] < 500, "chg"] = np.nan
    gd = gd.sort_values(["code", "HOLD_NOTICE_DATE", "END_DATE"]).drop_duplicates(["code", "HOLD_NOTICE_DATE"], keep="last")
    gd["chg3"] = gd.groupby("code")["chg"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    ei = g.day_index(gd["HOLD_NOTICE_DATE"], how="next")
    c1 = g.dense(gd["code"], ei, gd["chg"].to_numpy(float), agg="last")
    c3 = g.dense(gd["code"], ei, gd["chg3"].to_numpy(float), agg="last")
    c1f, last = g.ffill(c1)
    c3f, _ = g.ffill(c3)
    out["gd_chg1"] = g.pick(c1f, code, di)
    out["gd_chg3"] = g.pick(c3f, code, di)
    li = g.pick(last, code, di)
    out["gd_stale"] = np.where(li >= 0, di - li, np.nan)
    del c1, c3, c1f, c3f, last, gd
    G.say("大宗 + 户数 %.0fs" % (time.time() - t0))
    sv = pd.read_parquet(ALT_RAW / "survey.parquet", columns=["SECURITY_CODE", "NOTICE_DATE"])
    sv["code"] = sv["SECURITY_CODE"].astype(str).str.zfill(6)
    ev = sv.groupby(["code", "NOTICE_DATE"]).size().rename("n").reset_index()
    del sv
    ei = g.day_index(ev["NOTICE_DATE"], how="next")
    cnt = g.dense(ev["code"], ei, np.ones(len(ev)))
    orgs = g.dense(ev["code"], ei, ev["n"].to_numpy(float))
    out["sv_n30"] = g.pick(g.wsum(cnt, 21), code, di)
    out["sv_n90"] = g.pick(g.wsum(cnt, 63), code, di)
    out["sv_orgs90"] = g.pick(g.wsum(orgs, 63), code, di)
    G.say("调研 %.0fs" % (time.time() - t0))
    # 2021-09 之前原始表没有数据：整段置 NaN（不是 0），模型按缺失处理
    early = (t["date"] < "2021-10-15").to_numpy()
    out.loc[early, :] = np.nan
    return out.astype(np.float32)


def main():
    t0 = time.time()
    df = PL_load()
    G.say("候选池行 %d（%s..%s）%.0fs" % (len(df), df["date"].min(), df["date"].max(), time.time() - t0))
    # 名次（候选池内 lret20 降序）
    df = df.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    df["rank_ret20"] = (df.groupby("date").cumcount() + 1).astype(np.int32)
    n_day = df.groupby("date")["code"].transform("size")
    df["prank_ret20"] = (df["rank_ret20"] / n_day).astype(np.float32)
    df["lmcap"] = np.log(df["mcap"].clip(lower=1e6)).astype(np.float32)
    df["lamt20"] = np.log(df["amt20"].clip(lower=1e3)).astype(np.float32)
    df["lvol60"] = np.log(df["vol60"].clip(lower=1e-4)).astype(np.float32)
    df["lturn20"] = np.log(df["turn20"].clip(lower=1e-3)).astype(np.float32)
    df["lprice"] = np.log(df["price_u"].clip(lower=0.1)).astype(np.float32)
    df["lret20_lag5"] = np.log1p(df["ret20_lag5"].astype(float).clip(lower=-0.99)).astype(np.float32)
    df["lret60"] = np.log1p(df["ret60"].astype(float).clip(lower=-0.99)).astype(np.float32)
    df["board_i"] = df["board"].map({"main": 0, "chinext": 1, "star": 2, "bj": 3}).astype(np.int8)
    # 日级市场状态（候选池上算，T 日收盘后可得；基准率只用窗口已关的天）
    day = df.groupby("date").agg(pool_n=("code", "size"), base=("y_open", "mean"), mret1=("ret1", "mean"),
                                 lu_share=("lu_close", "mean"), hot50=("ret20", lambda s: float((s > 0.5).mean())),
                                 mkt_vol20=("vol20", "median")).sort_index()
    day["mkt_base_t21"] = day["base"].rolling(20).mean().shift(21)
    day["mkt_ret20"] = day["mret1"].rolling(20).sum()
    day["mkt_ret5"] = day["mret1"].rolling(5).sum()
    day["lu_share20"] = day["lu_share"].rolling(20).mean()
    d2 = df[df["rank_ret20"] <= 10].groupby("date")["ret20"].agg(top1_ret20="max", top10_ret20_med="median")
    day = day.join(d2).drop(columns=["base", "mret1", "lu_share"])
    df = df.merge(day.reset_index(), on="date", how="left")
    for c in G.DAY_COLS:
        df[c] = df[c].astype(np.float32)
    G.say("名次 + 市场状态 %.0fs" % (time.time() - t0))
    alt = alt_features(df[["code", "date", "amt20"]])
    for c in G.ALT_COLS:
        df[c] = alt[c].to_numpy()
    del alt
    df["seg"] = np.where(df["date"] < "2022-01-01", "pre", G.seg_of(df["date"]))
    df["hit"] = df["y_open"].fillna(0).astype(np.float32)
    df.to_parquet(G.TABLE, index=False)
    G.say("-> %s %d 行 %d 列 %.0fs" % (G.TABLE, len(df), df.shape[1], time.time() - t0))
    # 对账：B(10) 两段精确率
    for seg in ("screen", "dev"):
        s = df[(df["seg"] == seg) & (df["rank_ret20"] <= 10)]
        G.say("%s B(10) = %.2f%%（precision 对账 11.34 / 16.54）; 行 %d 天 %d" % (seg, 100 * s["hit"].mean(), len(s), s["date"].nunique()))
        top = PR_top(seg)
        a = s[["date", "code"]].merge(top, on=["date", "code"], how="inner")
        G.say("  和 top50 前 10 的 (日,码) 交集 %d / %d" % (len(a), len(s)))
    cov = {c: float(np.isfinite(df.loc[df["seg"] != "pre", c]).mean()) for c in G.ALT_COLS}
    G.say("非价格列覆盖（两段）：%s" % {k: round(v, 3) for k, v in cov.items()})
    G.progress("table.parquet 建好：%d 行 %d 列 %s..%s；B(10) 筛选 %.2f / 开发 %.2f" % (
        len(df), df.shape[1], df["date"].min(), df["date"].max(),
        100 * df[(df["seg"] == "screen") & (df["rank_ret20"] <= 10)]["hit"].mean(),
        100 * df[(df["seg"] == "dev") & (df["rank_ret20"] <= 10)]["hit"].mean()))


def PL_load() -> pd.DataFrame:
    cols = G.BASE_COLS + ["pool"]
    df = G.PL.load_pool(cols)
    df = df[df["pool"]].drop(columns=["pool"]).reset_index(drop=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    for c in df.columns:
        if df[c].dtype == np.float64 and c not in ("y_open",):
            df[c] = df[c].astype(np.float32)
    return df


def PR_top(seg: str) -> pd.DataFrame:
    t = G.PR.load(seg)
    return t[t["rank_ret20"] <= 10][["date", "code"]]


if __name__ == "__main__":
    main()
