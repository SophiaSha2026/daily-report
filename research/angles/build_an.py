"""
建表（criteria 第 2a、3a、4b 节）：全市场日级情绪指标 day_ind.parquet、事件表 events.parquet、路径表 paths.parquet。

    PYTHONIOENCODING=utf-8 python research/angles/build_an.py

px_screen 2020-07..2025-12 -> pullback.prepare()（只读调用，和生产同一个判定：涨停 / 大阳线 / 倍量 / 停牌 / 股本跳变）
-> 指标 / 事件 / 路径。find_events 默认 config 一次、trigger.big=loose 一次（字典在内存里改，不碰 config.yaml）。
最后用 data/breakout/daily.parquet（截 2025-12-31）跑同一函数对账 E1 的事件数。
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd

import common_an as A

import pullback as PB  # noqa: E402  src/pullback.py，只读调用


def consecutive_true(flag: np.ndarray, first: np.ndarray) -> np.ndarray:
    s = pd.Series(flag.astype(np.int8))
    blocks = pd.Series((~flag) | first).cumsum()
    return s.groupby(blocks.to_numpy()).cumsum().to_numpy()


def main():
    t0 = time.time()
    pb = PB.cfg()
    d = A.load_px()
    A.say("px_screen %d 行 %s..%s %.0fs" % (len(d), d["date"].min(), d["date"].max(), time.time() - t0))
    x = PB.prepare(d, pb)
    del d
    A.say("prepare 完 %d 行 %.0fs" % (len(x), time.time() - t0))
    code = x["code"].to_numpy()
    first = np.r_[True, code[1:] != code[:-1]]
    nbar = x.groupby("code", sort=False).cumcount().to_numpy()          # 0 起
    lu = x["limit_up"].to_numpy(bool)
    gain = x["gain"].to_numpy(float)
    vol = x["volume"].to_numpy(float)
    close = x["close"].to_numpy(float)
    high = x["high"].to_numpy(float)
    low = x["low"].to_numpy(float)
    opn = x["open"].to_numpy(float)
    prev_close = x["prev_close"].to_numpy(float)
    lim = x["lim"].to_numpy(float)
    limit_px = x["limit_px"].to_numpy(float)
    streak = consecutive_true(lu, first)
    lim_dn = np.round(prev_close * (1 - lim / 100.0), 2)
    ld = (close <= low + 0.005) & (close <= lim_dn + 0.015) & (gain < 0)
    touched = (high >= limit_px - 0.015) & np.isfinite(gain)
    blow = touched & ~lu

    def shift1(a, fill):
        o = np.empty_like(a)
        o[0] = fill
        o[1:] = a[:-1]
        o[first] = fill
        return o

    prev_lu = shift1(lu, False)
    prev_streak = shift1(streak, 0)
    # ---------------- 角度二：日级指标 ----------------
    df = pd.DataFrame({"date": x["date"].to_numpy(), "lu": lu, "ld": ld, "streak": streak, "touched": touched, "blow": blow,
                       "prev_lu": prev_lu, "p1": prev_streak == 1, "p2": prev_streak >= 2, "gain": gain, "up": gain > 0,
                       "ok": np.isfinite(gain)})
    g = df.groupby("date")
    day = pd.DataFrame({
        "n_all": g["ok"].sum(), "n_up": g["lu"].sum(), "n_dn": g["ld"].sum(), "h_max": g["streak"].max(),
        "n_touch": g["touched"].sum(), "n_blow": g["blow"].sum(),
        "up_share": g["up"].sum() / g["ok"].sum(),
    })
    day["blow"] = day["n_blow"] / day["n_touch"].replace(0, np.nan)
    yl = df[df["prev_lu"]].groupby("date")
    day["y_ret"] = yl["gain"].mean()
    day["lose"] = yl["gain"].apply(lambda s: float((s < 0).mean()))
    p1 = df[df["p1"]].groupby("date")["lu"].mean()
    p2 = df[df["p2"]].groupby("date")["lu"].mean()
    day["promo1"] = p1
    day["promo2"] = p2
    day["n_p1"] = df[df["p1"]].groupby("date").size()
    day["n_p2"] = df[df["p2"]].groupby("date").size()
    day = day.sort_index().reset_index()
    day.to_parquet(A.DAY_IND, index=False)
    A.say("日指标 %d 天 -> %s %.0fs" % (len(day), A.DAY_IND, time.time() - t0))
    del df, g, yl
    # ---------------- 角度三：事件 ----------------
    ev_rows = []
    bh = []
    ev1, _, diag1 = PB.find_events(x, pb, b_hist=bh)
    A.say("E1 清单 A %d 次（diag %s）%.0fs" % (len(ev1), json.dumps(diag1, ensure_ascii=False)[:300], time.time() - t0))
    e1 = ev1[["code", "date", "launch_date", "adjust_days", "base_run", "vol_vs_launch", "gain_pct", "board"]].copy()
    e1.to_parquet(A.HERE / "events_e1.parquet", index=False)
    ev_rows.append(pd.DataFrame({"kind": "E1", "code": ev1["code"], "date": ev1["date"]}))
    bdf = pd.DataFrame(bh)
    if len(bdf):
        A.say("b_hist %d 条，列 %s" % (len(bdf), list(bdf.columns)))
        b2 = bdf.dropna(subset=["b_date"]) if "b_date" in bdf else bdf
        ev_rows.append(pd.DataFrame({"kind": "E2", "code": b2["code"].astype(str).str.zfill(6), "date": b2["b_date"].astype(str).str[:10]}))
        bdf.to_parquet(A.HERE / "events_bhist.parquet", index=False)
    pb2 = copy.deepcopy(pb)
    pb2["trigger"]["big"] = "loose"
    ev1l, _, _ = PB.find_events(x, pb2, b_hist=None)
    A.say("E1L loose %d 次 %.0fs" % (len(ev1l), time.time() - t0))
    ev_rows.append(pd.DataFrame({"kind": "E1L", "code": ev1l["code"], "date": ev1l["date"]}))
    # E3 首阳单独成立（前 60 日：收盘振幅 <= 30%，量 >= 均量 x3 且 > 最大）
    N = int(pb["base"]["days"])
    amp_max = float(pb["base"]["amp_max"])
    vmult = float(pb["base"]["vol_mean_mult"])
    lvr = float(pb["launch"]["vol_ratio_min"])
    big = x["big"].to_numpy(bool)
    vr = x["vr"].to_numpy(float)
    cs = pd.Series(close)
    vs = pd.Series(vol)
    cmax = cs.shift(1).rolling(N).max().to_numpy()
    cmin = cs.shift(1).rolling(N).min().to_numpy()
    vmean = vs.shift(1).rolling(N).mean().to_numpy()
    vmax = vs.shift(1).rolling(N).max().to_numpy()
    okN = nbar >= N
    e3 = okN & big & (vr >= lvr) & (vol >= vmult * vmean) & (vol > vmax) & (cmax / cmin - 1 <= amp_max)
    ev_rows.append(pd.DataFrame({"kind": "E3", "code": code[e3], "date": x["date"].to_numpy()[e3]}))
    # E4 N 连板后首次缩量分歧转一致
    lu1 = shift1(lu, False)
    st2 = shift1(prev_streak, 0)              # T-2 的连板数
    vol1 = shift1(vol, np.nan)
    vol2 = shift1(vol1, np.nan)
    e4 = (nbar >= 2) & lu & ~lu1 & (st2 >= 2) & (vol1 < vol2) & (vol >= 1.5 * vol1)
    ev_rows.append(pd.DataFrame({"kind": "E4", "code": code[e4], "date": x["date"].to_numpy()[e4]}))
    # E5 突破首板：涨停、昨日没涨停、收盘 = 60 日最高（含今天）、量 >= 前 20 日均量 x3
    cmax60 = cs.rolling(N).max().to_numpy()
    vmean20 = vs.shift(1).rolling(20).mean().to_numpy()
    e5 = okN & lu & ~lu1 & (close >= cmax60 - 1e-9) & (vol >= 3.0 * vmean20)
    ev_rows.append(pd.DataFrame({"kind": "E5", "code": code[e5], "date": x["date"].to_numpy()[e5]}))
    # E6 2 连板 / >= 3 连板
    ev_rows.append(pd.DataFrame({"kind": "E6a", "code": code[streak == 2], "date": x["date"].to_numpy()[streak == 2]}))
    ev_rows.append(pd.DataFrame({"kind": "E6b", "code": code[streak >= 3], "date": x["date"].to_numpy()[streak >= 3]}))
    ev = pd.concat(ev_rows, ignore_index=True)
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    ev["date"] = ev["date"].astype(str).str[:10]
    ev.to_parquet(A.EVENTS, index=False)
    A.say("事件表 %s" % ev.groupby("kind").size().to_dict())
    # ---------------- 路径表（角度四 + 事件行） ----------------
    top = pd.read_parquet(A.ROOT / "research" / "precision" / "top50.parquet", columns=["date", "code", "rank_ret20"])
    need = pd.concat([top[top["rank_ret20"] <= 10][["code", "date"]], ev[["code", "date"]]], ignore_index=True).drop_duplicates()
    pos = pd.DataFrame({"code": code, "date": x["date"].to_numpy(), "_i": np.arange(len(x))})
    need = need.merge(pos, on=["code", "date"], how="inner")
    idx = need["_i"].to_numpy()
    H = 30
    n = len(x)
    win = idx[:, None] + np.arange(1, H + 1)[None, :]
    inb = win < n
    winc = np.minimum(win, n - 1)
    same = inb & (code[winc] == code[idx][:, None])
    o1 = np.where(same[:, 0], opn[np.minimum(idx + 1, n - 1)], np.nan)
    hi = np.where(same, high[winc], np.nan) / o1[:, None] - 1
    lo_ = np.where(same, low[winc], np.nan) / o1[:, None] - 1
    cl = np.where(same, close[winc], np.nan) / o1[:, None] - 1
    with np.errstate(all="ignore"):
        out = pd.DataFrame({"code": need["code"].to_numpy(), "date": need["date"].to_numpy(),
                            "g1": o1 / close[idx] - 1,
                            "n_after20": same[:, :20].sum(axis=1),
                            "max10": np.nanmax(hi[:, :10], axis=1), "max20": np.nanmax(hi[:, :20], axis=1),
                            "max30": np.nanmax(hi, axis=1), "min20": np.nanmin(lo_[:, :20], axis=1),
                            "c20": cl[:, 19], "c10": cl[:, 9]})
        h20 = np.where(np.isnan(hi[:, :20]), -np.inf, hi[:, :20])
        out["argmax20"] = np.where(same[:, :20].any(axis=1), h20.argmax(axis=1) + 1, np.nan)
        for thr in (0.3, 0.4, 0.5):
            reach = h20 >= thr
            anyr = reach.any(axis=1)
            out["d%d" % int(thr * 100)] = np.where(anyr, reach.argmax(axis=1) + 1, np.nan)
    out.to_parquet(A.PATHS, index=False)
    A.say("路径表 %d 行 -> %s %.0fs" % (len(out), A.PATHS, time.time() - t0))
    del x, win, winc, same, hi, lo_, cl, h20
    # ---------------- 对账：daily.parquet（截 2025-12-31）上的 E1 ----------------
    try:
        dd = pd.read_parquet(A.ROOT / "data" / "breakout" / "daily.parquet",
                             columns=["code", "date", "open", "high", "low", "close", "volume", "outstanding_share"])
        dd["date"] = dd["date"].astype(str).str[:10]
        dd = dd[dd["date"] <= A.END].reset_index(drop=True)
        xd = PB.prepare(dd, pb)
        del dd
        evd, _, _ = PB.find_events(xd, pb, b_hist=None)
        del xd
        lo_m, hi_m = "2023-08-01", A.END
        a = ev1[(ev1["date"] >= lo_m) & (ev1["date"] <= hi_m)]
        b = evd[(evd["date"] >= lo_m) & (evd["date"] <= hi_m)]
        ka = set(zip(a["code"], a["date"]))
        kb = set(zip(b["code"].astype(str).str.zfill(6), b["date"].astype(str).str[:10]))
        rec = {"px_screen_2023-08..": len(ka), "daily_2023-08..": len(kb), "both": len(ka & kb),
               "only_px": sorted(ka - kb)[:20], "only_daily": sorted(kb - ka)[:20]}
        (A.HERE / "e1_reconcile.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        A.say("E1 对账 %s" % json.dumps({k: v for k, v in rec.items() if not k.startswith("only")}))
    except Exception as e:  # noqa: BLE001
        A.say("!! 对账失败 %r" % (e,))
    A.progress("build_an：日指标 %d 天、事件 %s、路径 %d 行；E1 对账见 e1_reconcile.json（%.0fs）" % (
        len(day), ev.groupby("kind").size().to_dict(), len(out), time.time() - t0))


if __name__ == "__main__":
    main()
