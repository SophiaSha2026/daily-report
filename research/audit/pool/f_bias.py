"""审计 F：候选池定义里的隐性偏差（只看对名单的影响）。只读。

1 静态 ST 名单：研究用 cache/st_codes.json（2026-09-16 的名称）剔全部历史；生产按当天名称剔。
  用带日期的名称来源（龙虎榜 / 大宗 / 同花顺强势股 / 两融 / 减持公告标题前缀）还原「当时叫什么」，
  数：a 静态名单剔掉的前 10 名额里，当时真是 ST 的占多少；b 进了前 10 的票里当时是 ST（生产会剔）的占多少、命中率多少。
2 北交所 / 各板块在前 10 的占比和命中率。
3 停牌缺口 susp20、< 2 元、nbar 120~240 各挡掉多少前 10 名额，挡掉的那些命中率。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
HERE = ROOT / "research/audit/pool"
L = []


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def topk(df, elig, k=10):
    d = df[elig].sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    d["rank"] = d.groupby("date").cumcount() + 1
    return d[d["rank"] <= k]


def seg(d):
    return np.where(d < "2024-06-01", "screen", "dev")


def main():
    import validate as V
    st = V.st_codes()
    f = pd.read_parquet(EXP / "rulecore/feat.parquet", columns=["code", "date", "board", "close", "nbar", "susp20", "elig0", "lret20", "vol60", "turn20", "amp20", "y_open"])
    f["code"] = f["code"].astype(str).str.zfill(6)
    f = f[(f["date"] >= "2022-01-01") & (f["date"] <= "2025-12-31")].reset_index(drop=True)
    fin = np.isfinite(f[["lret20", "vol60", "turn20", "amp20"]].to_numpy(float)).all(axis=1)
    f["is_st"] = f["code"].isin(st)
    base = (f["close"] >= 2) & (f["nbar"] >= 120) & ~f["susp20"] & fin
    f["seg"] = seg(f["date"])
    pool = pd.read_parquet(EXP / "polish/pool.parquet", columns=["code", "date", "ev_any", "pool"])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    f = f.merge(pool, on=["code", "date"], how="left")
    f["ev_any"] = f["ev_any"].fillna(False).astype(bool)     # 非 elig0 的行没有事件标志，按无事件
    say("feat 2022-01..2025-12 %d 行；elig0 %d；静态 ST 名单 %d 只，表里命中 %d 只" % (len(f), int(f["elig0"].sum()), len(st), f.loc[f["is_st"], "code"].nunique()))

    # ---- 带日期的名称 ----
    names = []
    try:
        x = pd.read_parquet(ROOT / "research/altdata/raw/lhb_detail.parquet", columns=["SECURITY_CODE", "TRADE_DATE", "SECURITY_NAME_ABBR"])
        names.append(pd.DataFrame({"code": x["SECURITY_CODE"].astype(str).str.zfill(6), "date": x["TRADE_DATE"].astype(str).str[:10], "name": x["SECURITY_NAME_ABBR"], "src": "lhb"}))
        x = pd.read_parquet(ROOT / "research/altdata/raw/dzjy.parquet", columns=["SECURITY_CODE", "TRADE_DATE", "SECURITY_NAME_ABBR"])
        names.append(pd.DataFrame({"code": x["SECURITY_CODE"].astype(str).str.zfill(6), "date": x["TRADE_DATE"].astype(str).str[:10], "name": x["SECURITY_NAME_ABBR"], "src": "dzjy"}))
        x = pd.read_parquet(ROOT / "research/deepdata/raw/ths_harden.parquet", columns=["code", "date", "name"])
        names.append(pd.DataFrame({"code": x["code"].astype(str).str.zfill(6), "date": x["date"].astype(str).str[:10], "name": x["name"], "src": "ths"}))
        x = pd.read_parquet(EXP / "rulecore/events/reduce_ann.parquet", columns=["code", "date", "title"])
        nm = x["title"].str.extract(r"^([^:：]{2,12})[:：]")[0]
        names.append(pd.DataFrame({"code": x["code"].astype(str).str.zfill(6), "date": x["date"].astype(str).str[:10], "name": nm, "src": "ann"}).dropna())
        x = pd.read_parquet(ROOT / "research/altdata/raw/rzrq.parquet", columns=["SCODE", "DATE", "SECNAME"])
        x = x[x["DATE"].astype(str).str[8:10].isin(["01", "15"])]      # 两融每月取两天就够
        names.append(pd.DataFrame({"code": x["SCODE"].astype(str).str.zfill(6), "date": x["DATE"].astype(str).str[:10], "name": x["SECNAME"], "src": "rzrq"}))
    except Exception as e:  # noqa: BLE001
        say("名称来源读取出错：%s" % e)
    N = pd.concat(names, ignore_index=True)
    N["name"] = N["name"].astype(str)
    N = N[N["name"].str.len() >= 2]
    N["st"] = N["name"].str.upper().str.contains("ST")
    N["_d"] = pd.to_datetime(N["date"])
    N = N.sort_values("_d")
    say("带日期的名称 %d 条（来源 %s），含 ST 的 %d 条，代码 %d 只" % (len(N), N["src"].value_counts().to_dict(), int(N["st"].sum()), N["code"].nunique()))

    def st_then(rows: pd.DataFrame, tol: int = 60) -> pd.Series:
        a = rows[["code", "date"]].copy()
        a["_i"] = np.arange(len(a))
        a["_d"] = pd.to_datetime(a["date"])
        a = a.sort_values("_d")
        m = pd.merge_asof(a, N[["code", "_d", "st", "name"]], on="_d", by="code", direction="nearest", tolerance=pd.Timedelta(days=tol))
        m = m.sort_values("_i")
        return pd.Series(m["st"].to_numpy(), index=rows.index), pd.Series(m["name"].to_numpy(), index=rows.index)

    say("\n== 1 静态 ST 名单 ==")
    # a 不剔 ST 的前 10（其余条件照旧、事件照剔）里，被静态名单剔掉的名额
    t_nost = topk(f, base & ~f["ev_any"])
    rm = t_nost[t_nost["is_st"]]
    s, nmv = st_then(rm)
    rm = rm.assign(st_then=s, name_then=nmv)
    for sg in ("screen", "dev"):
        r = rm[rm["seg"] == sg]
        k = r["st_then"].notna()
        say("%s：静态名单从前 10 里剔掉 %d 个名额（%d 只），当时名称查得到 %d 个，其中当时真是 ST 的 %d 个（%.0f%%）；这些名额的命中率 %.1f%%（查得到且当时非 ST 的 %.1f%%）" % (
            sg, len(r), r["code"].nunique(), int(k.sum()), int((r["st_then"] == True).sum()), 100 * (r["st_then"] == True).sum() / max(k.sum(), 1),
            100 * r["y_open"].mean() if len(r) else np.nan, 100 * r.loc[k & (r["st_then"] == False), "y_open"].mean() if (k & (r["st_then"] == False)).any() else np.nan))
    # b 研究前 10（pool）里当时是 ST 的
    t_pool = topk(f, f["pool"].fillna(False).astype(bool))
    s, nmv = st_then(t_pool)
    t_pool = t_pool.assign(st_then=s, name_then=nmv)
    for sg in ("screen", "dev"):
        r = t_pool[t_pool["seg"] == sg]
        k = r["st_then"].notna()
        a_ = r[r["st_then"] == True]
        say("%s：研究前 10 共 %d 个名额，当时名称查得到 %d（%.0f%%），其中当时是 ST 的 %d 个（%d 只；占查得到的 %.1f%%），命中率 %.1f%%，对非 ST 名额 %.1f%%" % (
            sg, len(r), int(k.sum()), 100 * k.mean(), len(a_), a_["code"].nunique(), 100 * len(a_) / max(k.sum(), 1),
            100 * a_["y_open"].mean() if len(a_) else np.nan, 100 * r.loc[k & (r["st_then"] == False), "y_open"].mean()))
        if len(a_):
            say("   样例：%s" % a_[["date", "code", "name_then", "y_open"]].head(6).to_dict("records"))
    # 如果按「当时名称」剔 ST（近似生产），前 10 命中变多少：剔掉 st_then 的，第 11 名起补位
    t30 = topk(f, f["pool"].fillna(False).astype(bool), k=30)
    s30, _ = st_then(t30)
    t30 = t30.assign(st_then=s30)
    keep = t30[~(t30["st_then"] == True)].copy()
    keep["rank2"] = keep.groupby("date").cumcount() + 1
    for sg in ("screen", "dev"):
        h0 = t_pool[t_pool["seg"] == sg].groupby("date")["y_open"].mean()
        h1 = keep[(keep["seg"] == sg) & (keep["rank2"] <= 10)].groupby("date")["y_open"].mean()
        say("%s：前 10 按天命中 静态名单 %.2f%% -> 再剔「当时名称含 ST」补位后 %.2f%%（差 %+.2fpp）" % (sg, 100 * h0.mean(), 100 * h1.mean(), 100 * (h1.mean() - h0.mean())))

    say("\n== 2 板块 ==")
    for sg in ("screen", "dev"):
        r = t_pool[t_pool["seg"] == sg]
        g = r.groupby("board")["y_open"].agg(["size", "mean"])
        g["share"] = g["size"] / len(r)
        say("%s：%s" % (sg, {b: (int(v["size"]), "%.1f%%" % (100 * v["share"]), "命中 %.1f%%" % (100 * v["mean"])) for b, v in g.iterrows()}))
        pb = f[(f["seg"] == sg) & f["pool"].fillna(False)].groupby("board").size()
        say("   候选池板块占比：%s" % {b: "%.1f%%" % (100 * v / pb.sum()) for b, v in pb.items()})

    say("\n== 3 其它准入条件各挡掉多少前 10 名额 ==")
    ev = ~f["ev_any"]
    nost = ~f["is_st"]
    conds = {"susp20": (f["close"] >= 2) & (f["nbar"] >= 120) & fin & nost & ev, "close<2": (f["nbar"] >= 120) & ~f["susp20"] & fin & nost & ev,
             "nbar<120": (f["close"] >= 2) & ~f["susp20"] & fin & nost & ev, "输入NaN": (f["close"] >= 2) & (f["nbar"] >= 120) & ~f["susp20"] & nost & ev}
    full = base & nost & ev
    for nm, c in conds.items():
        t = topk(f, c)
        blocked = t[~full.reindex(t.index)]
        for sg in ("screen", "dev"):
            b = blocked[blocked["seg"] == sg]
            say("%s 放开 %s：前 10 里多进来 %d 个名额（%d 只），它们的命中率 %.1f%%（研究前 10 %.1f%%）" % (
                sg, nm, len(b), b["code"].nunique(), 100 * b["y_open"].mean() if len(b) else np.nan, 100 * t_pool[t_pool["seg"] == sg]["y_open"].mean()))
    # nbar 120~240（研究口径比生产旧口径多出来的那段）
    for sg in ("screen", "dev"):
        r = t_pool[t_pool["seg"] == sg]
        y = r[(r["nbar"] >= 120) & (r["nbar"] < 240)]
        say("%s：前 10 里 nbar 120~240 的名额 %d（%.1f%%），命中 %.1f%%；>= 240 的命中 %.1f%%" % (sg, len(y), 100 * len(y) / len(r), 100 * y["y_open"].mean(), 100 * r[r["nbar"] >= 240]["y_open"].mean()))
    (HERE / "results_f_bias.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
