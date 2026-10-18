"""长历史：全市场拉新浪不复权 OHLCV + 股本 / 换手 + 前复权因子，2005-01-01 起（研究用，不碰生产）。

和 research/leak/fetch_unadj.py 同一条路（backfill._sina_map 多进程 4 路、每只 3 次重试、带 http_timeout），分片落盘可续跑：
    PYTHONIOENCODING=utf-8 python research/longhist/fetch_lh.py
产物 research/longhist/raw/part_<i>.parquet（code date open_u high_u low_u close_u volume amount outstanding_share turnover qfq_factor），
missing.txt 记拉不到的代码。北京 16:00~18:30 不跑（生产在取数）。
probe_lh.py 2026-10-18 实测：在市票给上市以来全部（000001 到 1991-04），退市 / 被吸收合并的票 15/15 全部空体 -> 只有幸存者。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "breakout"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
START = "2005-01-01"
CHUNK = 400
EXTRA = ["volume", "amount", "outstanding_share", "turnover"]


def one(code: str):
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
            m = m[m["date"] >= START].rename(columns={k: k + "_u" for k in ("open", "high", "low", "close")})
            m["code"] = code
            return code, m
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return code, None


def in_prod_window() -> bool:
    from bjtime import now_bj
    t = now_bj()
    if t.weekday() >= 5:
        return False
    hm = t.hour * 60 + t.minute
    return 16 * 60 <= hm < 18 * 60 + 30


def main() -> int:
    import backfill as BF
    RAW.mkdir(parents=True, exist_ok=True)
    codes = sorted(pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"].str.zfill(6).unique())
    t0 = time.time()
    miss_f = RAW / "missing.txt"
    miss = [x for x in miss_f.read_text(encoding="utf-8").splitlines() if x.strip()] if miss_f.exists() else []
    for i in range(0, len(codes), CHUNK):
        f = RAW / ("part_%02d.parquet" % (i // CHUNK))
        if f.exists():
            continue
        while in_prod_window():
            print("北京 16:00~18:30，暂停 5 分钟", flush=True)
            time.sleep(300)
        got = []
        for c, d in BF._sina_map(codes[i:i + CHUNK], fn=one):
            (got.append(d) if d is not None and len(d) else miss.append(c))
        if got:
            pd.concat(got, ignore_index=True).to_parquet(f, index=False)
        miss_f.write_text("\n".join(sorted(set(miss))), encoding="utf-8")
        print("%d/%d 缺 %d  %.0fs" % (min(i + CHUNK, len(codes)), len(codes), len(miss), time.time() - t0), flush=True)
    print("完成，缺 %d 只，%.0f 分钟" % (len(set(miss)), (time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
