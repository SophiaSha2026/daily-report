"""
前向小样本（2026-09-01 起，本机日线到 2026-10-16）：只看一次、只作方向参考。

    PYTHONIOENCODING=utf-8 python research/polish/forward.py --arms forward_arms.json

数据：data/breakout/daily.parquet（前复权，比值和 pit 一致）+ feat_fwd.ext_features（和 pool 对过账的同一份实现）。
标签：次日开盘买入、之后 20 根最高价涨超 50%、次日非一字（buyable.compute 同口径）；窗口没关的行另列「到目前为止」。
2026-01..08 的行在算完特征后立即丢弃，标签只对 date >= 2026-09-01 的行算。
候选池：和研究池同口径（ST 静态名单、>= 2 元、>= 120 根、无长停牌、四个输入有值、事件剔除用 rulecore/events_c 的表）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common_pl as P
import feat_fwd as F
import rules_grid as G

START = "2026-09-01"


def labels_partial(px: pd.DataFrame, window: int = 20, thr: float = 0.5) -> pd.DataFrame:
    import datasource as ds
    px = px.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    rows = []
    for c, g in px.groupby("code", sort=False):
        n = len(g)
        high, low, opn, dates = g["high"].to_numpy(float), g["low"].to_numpy(float), g["open"].to_numpy(float), g["date"].to_numpy()
        close = g["close"].to_numpy(float)
        for i in range(n):
            if dates[i] < START:
                continue
            if i + 1 >= n:
                rows.append((c, dates[i], np.nan, np.nan, np.nan, 0, np.nan))
                continue
            hi = high[i + 1: i + 1 + window]
            navail = len(hi)
            o1 = opn[i + 1]
            pct = ds.limit_pct(c, "")
            lim1 = ds.limit_price(close[i], pct, c)
            yizi = low[i + 1] >= lim1 - 0.011
            r = hi.max() / o1 - 1 if o1 > 0 else np.nan
            rows.append((c, dates[i], r, float((r > thr) and not yizi), float(yizi), navail, o1))
    return pd.DataFrame(rows, columns=["code", "date", "r20_sofar", "hit_sofar", "yizi1", "n_avail", "open1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="forward_arms.json")
    a = ap.parse_args()
    import backfill as B
    import validate as V
    import common_rc as C
    daily = pd.read_parquet(P.ROOT / "data" / "breakout" / "daily.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str).str[:10]
    daily = B.clean_bars(daily)
    f = F.ext_features(daily)
    f = f[f["date"] >= START].reset_index(drop=True)          # 2026-01..08 的行在这里丢掉，之后不再碰
    P.say("前向特征行 %d，%s..%s，%d 天" % (len(f), f["date"].min(), f["date"].max(), f["date"].nunique()))
    st = V.st_codes()
    f["elig0"] = (~f["code"].isin(st) & (f["close"] >= 2.0) & (f["nbar"] >= 120) & ~f["susp20"]
                  & np.isfinite(f[["lret20", "vol60", "turn20", "amp20"]]).all(axis=1))
    C.EVENTS = P.EXP / "rulecore" / "events_c"
    ev = C.event_flags(f)
    f["pool"] = f["elig0"].to_numpy() & ~ev["any"].to_numpy()
    f = f[f["pool"]].reset_index(drop=True)
    lab = labels_partial(daily[daily["date"] >= "2026-06-01"][["code", "date", "open", "high", "low", "close"]])
    f = f.merge(lab, on=["code", "date"], how="left")
    f["closed"] = f["n_avail"] >= 20
    f["y_open"] = np.where(f["closed"], f["hit_sofar"], np.nan)
    f["y_up"] = f["y_open"]
    P.say("候选池每天 %.0f 只；窗口已关的日期 %d 天（%s..%s），未关的 %d 天" % (
        f.groupby("date").size().mean(), f[f["closed"]]["date"].nunique(), f[f["closed"]]["date"].min(), f[f["closed"]]["date"].max(),
        f[~f["closed"]]["date"].nunique()))
    for k in (5, 10, 40, 60):
        f["lret%d" % k] = np.log1p(f["ret%d" % k].astype(float).clip(lower=-0.99))
    arms = json.loads((P.HERE / a.arms).read_text(encoding="utf-8"))
    L = ["前向 %s..%s：每天前 10（候选池 %.0f 只/天）。closed = 20 根窗口已关；sofar = 用到目前为止的最高价（下界）" % (
        f["date"].min(), f["date"].max(), f.groupby("date").size().mean())]
    series = {}
    for name, spec in arms.items():
        if spec["type"] == "rule":
            g = {"w": spec["w"], "excl": spec.get("excl", [])}
            for key in g["w"]:
                if key.startswith("r_") and key not in f.columns:
                    f[key] = f.groupby("date")[key[2:]].rank(pct=True).astype(np.float32)
            s, el = G.score_of(f, g)
        elif spec["type"] == "rr":
            s, el = rr_scores(f, spec)
        else:
            raise ValueError(spec)
        o = f[["date", "code", "board", "y_open", "y_up", "hit_sofar", "closed", "r20_sofar"]].copy()
        t = P.rank_by(o, s, keep=10, elig=el).merge(o[["date", "code", "hit_sofar", "closed", "r20_sofar"]], on=["date", "code"], how="left")
        top = t[t["rank"] <= 10]
        series[name] = {"closed": top[top["closed"]].groupby("date")["hit_sofar"].mean(),
                        "sofar": top.groupby("date")["hit_sofar"].mean()}
        cl = top[top["closed"]]
        L.append("%-22s closed %2d 天 命中 %5.1f%% (席 %d/%d)   sofar 全部 %2d 天 %5.1f%%   前 10 的 r20 中位 %+.1f%%" % (
            name, cl["date"].nunique(), 100 * cl["hit_sofar"].mean(), int(cl["hit_sofar"].sum()), len(cl),
            top["date"].nunique(), 100 * top["hit_sofar"].mean(), 100 * top["r20_sofar"].median()))
    base = series.get("base_ret20")
    if base is not None:
        L.append("对 base_ret20 按天配对：")
        for name, s in series.items():
            if name == "base_ret20":
                continue
            for kind in ("closed", "sofar"):
                d = (s[kind] - base[kind]).dropna()
                L.append("  %-22s %-6s %2d 天 差 %+.2fpp  赢 %d / 平 %d / 输 %d 天" % (
                    name, kind, len(d), 100 * d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())))
    pool_base = f[f["closed"]].groupby("date")["hit_sofar"].mean()
    L.append("候选池基准（closed 天）%.2f%%" % (100 * pool_base.mean()))
    txt = "\n".join(L)
    P.say(txt)
    (P.HERE / "results_forward.txt").write_text(txt + "\n", encoding="utf-8")
    P.ledger_add({"kind": "forward_look", "arms": list(arms)})


def rr_scores(f: pd.DataFrame, spec: dict):
    """重排臂：用 polish 池（<= 2025-11，净化）训练，预测前向行。"""
    import rerank as R
    import model as M
    k = spec["k"]
    tr = R.load_topk(k)
    if spec["feats"].startswith("rk"):
        for c in R.EXT:
            if c != "lim_pct":
                tr["rkk_" + c] = tr.groupby("date")[c].rank(pct=True).astype(np.float32)
    cols = R.feat_list(tr, spec["feats"])
    trm = ((tr["date"] >= "2022-12-01") & (tr["date"] < "2025-11-05") & (tr["end20"] < "2025-12-01") & np.isfinite(tr["y_open"])).to_numpy()
    tr = tr[trm].sort_values(["date", "rank_ret20"]).reset_index(drop=True)
    te = f.copy()
    te["rank_ret20"] = te.groupby("date")["lret20"].rank(ascending=False, method="first")
    te = te[te["rank_ret20"] <= k].copy()
    for c in ("mcap", "amt20", "amt1"):
        te[c] = np.log1p(te[c].astype(float)).astype(np.float32)
    te["lrank"] = np.log(te["rank_ret20"]).astype(np.float32)
    for c in ("lret20", "turn5", "vol20", "ret5"):
        te["rk_" + c] = te.groupby("date")[c].rank(pct=True).astype(np.float32)
    for b in ("chinext", "star", "bj"):
        te["b_" + b] = (te["board"] == b).astype(np.float32)
    if spec["feats"].startswith("rk"):
        for c in R.EXT:
            if c != "lim_pct":
                te["rkk_" + c] = te.groupby("date")[c].rank(pct=True).astype(np.float32)
    w = M.month_weights(tr)
    gtr = tr.groupby("date", sort=False).size().to_numpy() if spec["model"] == "lgb_rank" else None
    p = R.fit_predict(spec["model"], tr[cols].to_numpy(np.float32), tr["y_open"].to_numpy(float), np.asarray(w, float),
                      te[cols].to_numpy(np.float32), 7, spec.get("params"), gtr)
    s = pd.Series(np.nan, index=f.index)
    s.loc[te.index] = p
    return s.to_numpy(), np.isfinite(s.to_numpy())


if __name__ == "__main__":
    main()
