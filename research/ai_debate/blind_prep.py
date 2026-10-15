"""盲测样本和材料（第 2 件）。不调 AI，只备料。

样本：w1_scale_avg/top_base.parquet（开发集逐月滚动、现状臂、每天前 200），取 rank <= 10、
2025-01..2025-12、非 ST，y_open 用 buyable.open_labels 重算并和产物对账；命中 / 没命中各 10 只，
按月分层、代码不重复、日期不重复，种子固定。
材料：只用 daily.parquet 里 <= 那一天的行。两份：
  named  代码、名称（腾讯现名，可能和当时不同）、真实日期、真实价格、流通市值
  blind  无代码名称，日期改成 T-19..T0，价格按 T0 收盘 = 100 归一，流通市值分档，
         市场环境去掉绝对成交额和日期
输出 research/ai_debate/blind_samples.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
import datasource as ds  # noqa: E402
import buyable  # noqa: E402
import validate  # noqa: E402

N_EACH = 10
SEED = 20261014


def pick_samples() -> pd.DataFrame:
    t = pd.read_parquet(ROOT / "data/breakout/raw/exp/w1_scale_avg/top_base.parquet")
    t["date"] = t["date"].astype(str)
    t["code"] = t["code"].astype(str).str.zfill(6)
    t = t[(t["rank"] <= 10) & (t["date"] >= "2025-01-01") & (t["date"] <= "2025-12-31")]
    st = validate.st_codes()
    t = t[~t["code"].isin(st)]
    lab = buyable.open_labels()
    lab["date"] = lab["date"].astype(str)
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    t = t.merge(lab[["code", "date", "y_open", "r20_open", "yizi1"]].rename(columns={"y_open": "y_chk"}),
                on=["code", "date"], how="left")
    both = t["y_chk"].notna() & t["y_open"].notna()
    bad = int((t.loc[both, "y_chk"] != t.loc[both, "y_open"]).sum())
    print(f"候选 {len(t)} 行，y_open 和 buyable 重算不一致 {bad} 行（共 {int(both.sum())} 行可比）")
    t = t[t["y_chk"].notna()].copy()
    rng = np.random.default_rng(SEED)
    out = []
    used_codes, used_dates = set(), set()
    t["month"] = t["date"].str[:7]
    for y in (1.0, 0.0):
        pool = t[t["y_chk"] == y]
        months = sorted(pool["month"].unique())
        rng.shuffle(months)
        got, k = 0, 0
        while got < N_EACH and k < 500:
            m = months[k % len(months)]
            k += 1
            cand = pool[(pool["month"] == m) & ~pool["code"].isin(used_codes) & ~pool["date"].isin(used_dates)]
            if cand.empty:
                continue
            r = cand.iloc[int(rng.integers(len(cand)))]
            out.append(r)
            used_codes.add(r["code"])
            used_dates.add(r["date"])
            got += 1
    s = pd.DataFrame(out).reset_index(drop=True)
    s = s.sample(frac=1, random_state=SEED % (2 ** 31)).reset_index(drop=True)
    s["sid"] = [f"S{i + 1:02d}" for i in range(len(s))]
    return s


def market_facts(daily: pd.DataFrame, dates: list[str]) -> dict:
    """每个样本日的近 10 个交易日全市场环境：上涨占比、涨停家数、中位涨幅、成交额（亿）。"""
    lo = min(dates)
    d = daily[["code", "date", "close", "amount"]]
    d = d[d["date"] >= (pd.Timestamp(lo) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")].copy()
    d = d[d["date"] <= max(dates)].sort_values(["code", "date"])
    d["prev"] = d.groupby("code")["close"].shift(1)
    d = d[d["prev"].notna()]
    pct = d["code"].map(lambda c: ds.limit_pct(c, ""))
    lim = ds.limit_price_arr(d["prev"], pct, d["code"])
    d["lu"] = (d["close"] >= lim - 1e-9)
    d["ret"] = d["close"] / d["prev"] - 1
    d["up"] = d["ret"] > 0
    g = d.groupby("date").agg(adv=("up", "mean"), lu=("lu", "sum"), med=("ret", "median"),
                              amt=("amount", "sum"))
    allds = list(g.index)
    out = {}
    for dt_ in dates:
        i = allds.index(dt_)
        w = g.iloc[max(0, i - 9): i + 1]
        out[dt_] = [{"date": idx, "adv_share": round(float(r.adv), 3), "limit_up": int(r.lu),
                     "ret_median_pct": round(100 * float(r.med), 2), "turnover_yi": round(float(r.amt) / 1e8, 0)}
                    for idx, r in w.iterrows()]
    return out


def k_facts(px: pd.DataFrame, code: str) -> dict:
    """tools/ai_analyze._k_facts 的截断版：px 已经只到样本日。"""
    px = px.sort_values("date").reset_index(drop=True)
    c = px["close"].to_numpy(float)
    h, lo, v = px["high"].to_numpy(float), px["low"].to_numpy(float), px["volume"].to_numpy(float)
    n = len(px)
    pct = ds.limit_pct(code, "")
    prev = np.r_[np.nan, c[:-1]]
    lim = ds.limit_price_arr(pd.Series(prev), pct, pd.Series([code] * n)).to_numpy(float)
    lu = (c >= lim - 1e-9) & np.isfinite(lim)
    streak = 0
    for x in lu[::-1]:
        if x:
            streak += 1
        else:
            break

    def ret(k):
        return round(100 * (c[-1] / c[-1 - k] - 1), 2) if n > k else None
    w = px.tail(60)
    tov = px["turnover"].to_numpy(float)
    osh = px["outstanding_share"].to_numpy(float)
    return {
        "close": round(float(c[-1]), 2),
        "ret_1d_pct": ret(1), "ret_5d_pct": ret(5), "ret_20d_pct": ret(20), "ret_60d_pct": ret(60),
        "vol_vs_20d_avg": round(float(v[-1] / max(v[-21:-1].mean(), 1)), 2) if n > 21 else None,
        "turnover_pct_last": round(100 * float(tov[-1]), 2) if np.isfinite(tov[-1]) else None,
        "turnover_pct_20d_avg": round(100 * float(np.nanmean(tov[-20:])), 2),
        "dist_52w_high_pct": round(100 * (c[-1] / h[-250:].max() - 1), 1),
        "dist_52w_low_pct": round(100 * (c[-1] / lo[-250:].min() - 1), 1),
        "limit_up_closes_20d": int(lu[-20:].sum()), "limit_up_streak": int(streak),
        "max_drawdown_20d_pct": round(100 * float((c[-20:] / np.maximum.accumulate(c[-20:]) - 1).min()), 1),
        "limit_pct": pct,
        "float_mcap_yi": round(float(osh[-1] * c[-1]) / 1e8, 1) if np.isfinite(osh[-1]) else None,
        "recent_20_bars": [{"d": str(r.date), "o": round(float(r.open), 2), "h": round(float(r.high), 2),
                            "l": round(float(r.low), 2), "c": round(float(r.close), 2),
                            "v_rel": round(float(r.volume / max(w["volume"].mean(), 1)), 2)}
                           for r in px.tail(20).itertuples()],
    }


def _rel(i: int, n: int) -> str:
    return "T0" if i == n - 1 else f"T-{n - 1 - i}"


def blindify(kf: dict, mkt: list[dict]) -> tuple[dict, list[dict]]:
    b = dict(kf)
    base = kf["close"]
    bars = kf["recent_20_bars"]
    b["recent_20_bars"] = [{"d": _rel(i, len(bars)),
                            **{k: round(100 * x[k] / base, 2) for k in ("o", "h", "l", "c")},
                            "v_rel": x["v_rel"]} for i, x in enumerate(bars)]
    b["close"] = 100.0
    m = kf.get("float_mcap_yi")
    b.pop("float_mcap_yi", None)
    b["float_mcap_bucket"] = (None if m is None else "<30 亿" if m < 30 else "30~100 亿" if m < 100
                              else "100~300 亿" if m < 300 else ">=300 亿")
    bm = [{"d": _rel(i, len(mkt)), **{k: r[k] for k in ("adv_share", "limit_up", "ret_median_pct")}}
          for i, r in enumerate(mkt)]
    return b, bm


def name_of(codes: list[str]) -> dict:
    try:
        q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
        out = {}
        for k, v in q.items():
            out[str(k)[-6:]] = str(getattr(v, "name", "") or "")
        return out
    except Exception as e:  # noqa: BLE001
        print("取名称失败", e)
        return {}


def main() -> None:
    s = pick_samples()
    daily = pd.read_parquet(ROOT / "data/breakout/daily.parquet")
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily["date"] = daily["date"].astype(str)
    mk = market_facts(daily, sorted(s["date"].unique()))
    names = name_of(list(s["code"]))
    out = []
    for r in s.itertuples():
        px = daily[(daily["code"] == r.code) & (daily["date"] <= r.date)]
        kf = k_facts(px, r.code)
        assert kf["recent_20_bars"][-1]["d"] == r.date, "截断不对"
        bk, bm = blindify(kf, mk[r.date])
        model = {"score": int(r.score), "rank": int(r.rank), "p": round(float(r.p), 4), "board": r.board}
        out.append({"sid": r.sid, "code": r.code, "name": names.get(r.code, ""), "date": r.date,
                    "board": r.board, "y_open": int(r.y_chk), "r20_open": round(float(r.r20_open), 4),
                    "yizi1": bool(r.yizi1),
                    "named": {"code": r.code, "name": names.get(r.code, ""), "date": r.date,
                              "model": model, "kline": kf, "market": mk[r.date]},
                    "blind": {"model": model, "kline": bk, "market": bm}})
    (HERE / "blind_samples.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame([{k: x[k] for k in ("sid", "code", "name", "date", "board", "y_open", "r20_open")}
                        for x in out]).to_string())


if __name__ == "__main__":
    main()
