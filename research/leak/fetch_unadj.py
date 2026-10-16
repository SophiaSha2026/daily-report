"""
泄漏检验第 0 步：全市场拉新浪不复权 OHLC + 前复权因子（研究用，不碰生产）。

2026-10-16 实测：data/breakout/daily.parquet 的四个价 == round(不复权价 / qfq_factor, 2)，
600000 / 300829 各 818 行逐位相同。所以 不复权价 / qfq_factor（不取整）就是没有取整颗粒的
同一条复权序列，比值和「截至 T 日的复权价」一致，颗粒度里没有 T 日之后的除权信息。

    python research/leak/fetch_unadj.py
    UNADJ_FULL=1 UNADJ_NAME=unadj_2020 BACKFILL_HIST_START=2020-07-01 python research/leak/fetch_unadj.py   # 扩展筛选集
产物 data/breakout/raw/exp/leak/unadj.parquet：code date open_u high_u low_u close_u qfq_factor
北京 16:00~18:30 不要跑（生产在取数）。
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "breakout"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

OUT = ROOT / "data" / "breakout" / "raw" / "exp" / "leak"
# 扩展筛选集（docs/design_2026-10-16.md 6.2）要量、额、股本、换手：设环境变量 UNADJ_FULL=1，
# 起点用 BACKFILL_HIST_START（backfill.HIST_START 读它），产物名 UNADJ_NAME
EXTRA = ["volume", "amount", "outstanding_share", "turnover"] if os.environ.get("UNADJ_FULL") == "1" else []
NAME = os.environ.get("UNADJ_NAME", "unadj")


def one(code: str):
    """模块级：进程池要 pickle。"""
    warnings.filterwarnings("ignore")
    import backfill as BF
    BF._net_guard()
    import akshare as ak
    sym = BF.prefix(code)
    for attempt in range(3):
        try:
            u = ak.stock_zh_a_daily(symbol=sym, adjust="")
            q = ak.stock_zh_a_daily(symbol=sym, adjust="qfq-factor")
            if u is None or not len(u):
                return code, None
            u = u[["date", "open", "high", "low", "close"] + EXTRA].copy()
            u["date"] = pd.to_datetime(u["date"]).astype("datetime64[ns]")
            q = q.copy()
            q["date"] = pd.to_datetime(q["date"]).astype("datetime64[ns]")
            q["qfq_factor"] = q["qfq_factor"].astype(float)
            m = pd.merge_asof(u.sort_values("date"), q.sort_values("date")[["date", "qfq_factor"]], on="date")
            m["date"] = m["date"].dt.strftime("%Y-%m-%d")
            m = m[m["date"] >= BF.HIST_START].rename(columns={k: k + "_u" for k in ("open", "high", "low", "close")})
            m["code"] = code
            return code, m
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return code, None


def main() -> int:
    import backfill as BF
    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet", columns=["code"])["code"]
                   .astype(str).str.zfill(6).unique())
    t0 = time.time()
    got, miss = [], []
    for i in range(0, len(codes), 200):
        for c, d in BF._sina_map(codes[i:i + 200], fn=one):
            (got.append(d) if d is not None and len(d) else miss.append(c))
        print("%d/%d 缺 %d  %.0fs" % (min(i + 200, len(codes)), len(codes), len(miss), time.time() - t0), flush=True)
    df = pd.concat(got, ignore_index=True)
    df.to_parquet(OUT / f"{NAME}.parquet", index=False)
    (OUT / f"{NAME}_missing.txt").write_text("\n".join(miss), encoding="utf-8")
    print("完成 %d 只 %d 行，缺 %d 只" % (df["code"].nunique(), len(df), len(miss)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
