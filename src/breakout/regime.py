"""
每日市场环境指标：state/regime_daily.jsonl。

2026-09-16 会诊的结论里，「市场环境」占了原因权重的 15%，但那是 LLM 去网上
查成交额和指数得出来的 —— 查到的数字没人核对，也没法做时间序列。这里把能从
自己的日线表算出来的环境量每天落一行，下次会诊直接读，不用再靠检索：

    date, n, turnover_yi, adv_share, limit_up, limit_down, max_gain_share,
    ret_median_pct, ret_mean_pct, base20, turnover_ma5_yi, turnover_chg5

其中 base20 是**全市场滚动 20 根的 50% 基准率**：那一天随便买一只，之后 20 根
K 线内最高价涨超 50% 的比例。起涨预测的期望命中率只有和它比才有意义
（验证集那十个月是 2.93%，2026-09 这几天是 0.1% 量级，差一个数量级）。

涨停家数按板块涨停幅度判（主板 10 / 科创创业 20 / 北交 30，ST 不单独区分，
所以是近似值，写在字段注释里）。指数不进来：日线表里没有指数，要联网，
而这条线的原则是能离线算的就离线算。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("breakout.regime")
DATA = ROOT / "data" / "breakout"
OUT = ROOT / "state" / "regime_daily.jsonl"
UP_WINDOW = 20
UP_THRESHOLD = 0.50


def _limit_pct(code: str) -> float:
    c = str(code).zfill(6)
    if c.startswith(("688", "300", "301")):
        return 20.0
    if c[0] in ("8", "4", "9"):
        return 30.0
    return 10.0


def compute(days: int = 60) -> list[dict]:
    """最近 days 个交易日的环境指标。只读 daily.parquet。"""
    dp = DATA / "daily.parquet"
    if not dp.exists():
        return []
    px = pd.read_parquet(dp, columns=["code", "date", "close", "high", "amount"])
    px["date"] = px["date"].astype(str)
    px = px.sort_values(["code", "date"])
    g = px.groupby("code", sort=False)
    px["prev"] = g["close"].shift(1)
    px["ret"] = px["close"] / px["prev"] - 1.0
    # 之后 20 根的最高价（和 truth / label 同一口径：t+1 起，不含当天）
    nxt = g["high"].shift(-1)
    rev = nxt[::-1].groupby(px["code"][::-1], sort=False)
    px["fut"] = rev.rolling(UP_WINDOW, min_periods=1).max().reset_index(
        level=0, drop=True)[::-1].to_numpy()
    n_bars = g["date"].transform("size").to_numpy()
    px["avail"] = np.minimum(UP_WINDOW, n_bars - g.cumcount().to_numpy() - 1)
    px["lim"] = px["code"].map(_limit_pct)

    all_dates = sorted(px["date"].unique())
    keep = set(all_dates[-days:])
    rows = []
    for d, x in px[px["date"].isin(keep)].groupby("date"):
        r = x["ret"].to_numpy(float)
        ok = np.isfinite(r)
        fin = x[x["avail"] >= UP_WINDOW]
        rows.append({
            "date": str(d),
            "n": int(len(x)),
            "turnover_yi": round(float(x["amount"].sum()) / 1e8, 1),
            "adv_share": round(float((r[ok] > 0).mean()), 4) if ok.any() else None,
            "limit_up": int(((x["ret"] >= x["lim"] / 100 - 0.004)
                             & np.isfinite(x["ret"])).sum()),
            "limit_down": int(((x["ret"] <= -x["lim"] / 100 + 0.004)
                               & np.isfinite(x["ret"])).sum()),
            "ret_median_pct": round(100 * float(np.nanmedian(r[ok])), 3) if ok.any() else None,
            "ret_mean_pct": round(100 * float(np.nanmean(r[ok])), 3) if ok.any() else None,
            "base20": (round(float((fin["fut"] / fin["close"] - 1 > UP_THRESHOLD).mean()), 5)
                       if len(fin) else None),
            "base20_final": bool(len(fin)),
        })
    rows.sort(key=lambda z: z["date"])
    ser = pd.Series([z["turnover_yi"] for z in rows], dtype=float)
    ma5 = ser.rolling(5, min_periods=1).mean()
    for i, z in enumerate(rows):
        z["turnover_ma5_yi"] = round(float(ma5.iloc[i]), 1)
        z["turnover_chg5"] = (round(float(ser.iloc[i] / ser.iloc[i - 5] - 1), 4)
                              if i >= 5 and ser.iloc[i - 5] else None)
    return rows


def save(rows: list[dict]) -> Path:
    """按日期合并写回（同一天重跑覆盖，base20 会从「未定」变成「已定」）。"""
    old = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
                old[j["date"]] = j
            except Exception:  # noqa: BLE001
                pass
    for r in rows:
        old[r["date"]] = r
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(old[k], ensure_ascii=False) + "\n"
                           for k in sorted(old)), encoding="utf-8")
    tmp.replace(OUT)
    return OUT


def load(n: int = 30) -> list[dict]:
    if not OUT.exists():
        return []
    rows = []
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return rows[-n:]


def record(days: int = 60) -> int:
    """算 + 落盘，返回写了几天。任何异常都吞掉（这条是研究性记录，不许拖垮出清单）。"""
    try:
        rows = compute(days)
        if not rows:
            return 0
        save(rows)
        last = rows[-1]
        log.info("环境指标 %s：成交额 %.0f 亿，涨停 %d 家，上涨占比 %.0f%%，"
                 "滚动20根基准 %s", last["date"], last["turnover_yi"],
                 last["limit_up"], 100 * (last["adv_share"] or 0),
                 f"{100 * last['base20']:.2f}%" if last["base20"] is not None else "未定")
        return len(rows)
    except Exception as e:  # noqa: BLE001
        log.warning("环境指标记录失败（不影响清单）: %s", e)
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = record(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    for r in load(12):
        print(f"{r['date']}  {r['turnover_yi']:>8.0f}亿  涨停 {r['limit_up']:>3}  "
              f"跌停 {r['limit_down']:>3}  上涨 {100 * (r['adv_share'] or 0):>4.0f}%  "
              f"中位 {r['ret_median_pct']:>+6.2f}%  "
              f"基准20 {('%.2f%%' % (100 * r['base20'])) if r['base20'] is not None else '未定':>7}")
    print(f"-> {OUT}（{n} 天）")
