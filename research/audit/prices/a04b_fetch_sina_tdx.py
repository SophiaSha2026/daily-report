"""a04b：样本票的新浪前复权（现在重拉一份，和 daily.parquet 比）+ 通达信 xdxr 除权事件（pytdx，研究 venv）。
    新浪单进程顺序拉（教训 19），每只间隔 0.5 秒；通达信一个连接顺序查。
    用研究 venv 的 python 跑（有 pytdx）；akshare 如果 venv 里没有就只拉通达信，新浪由主 python 另跑。
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

LOG = open(OUT / "a04b.log", "a", encoding="utf-8")


def say(*a):
    s = "[%s] %s" % (bjtime.now_bj().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def guard():
    t = bjtime.now_bj()
    if 16 * 60 <= t.hour * 60 + t.minute < 18 * 60 + 30:
        say("北京 16:00~18:30，停")
        sys.exit(2)


def tdx(codes):
    from pytdx.hq import TdxHq_API
    api = TdxHq_API(raise_exception=False, auto_retry=True)
    ok = False
    for h in ("115.238.56.198", "124.71.187.122", "122.51.120.217"):
        if api.connect(h, 7709, time_out=8):
            ok = True
            say("tdx 连上", h)
            break
    if not ok:
        say("tdx 连不上")
        return
    rows = []
    for i, c in enumerate(codes):
        mkt = 1 if c[0] == "6" else (2 if c[0] in "849" else 0)
        try:
            x = api.get_xdxr_info(mkt, c)
        except Exception as e:
            say("xdxr", c, "异常", repr(e)[:80])
            x = None
        for r in (x or []):
            r = dict(r)
            r["code"] = c
            rows.append(r)
        if i % 50 == 0:
            say("xdxr 进度", i, len(rows))
        time.sleep(0.05)
    api.disconnect()
    df = pd.DataFrame(rows)
    df.to_parquet(RAW / "tdx_xdxr.parquet", index=False)
    say("xdxr 完成", len(df), "行", df["code"].nunique(), "只；category 分布", df["category"].value_counts().to_dict())


def sina(codes):
    import warnings
    warnings.filterwarnings("ignore")
    import datasource as ds
    ds.http_timeout()
    import akshare as ak
    got, fails = [], []
    for i, c in enumerate(codes):
        if i % 40 == 0:
            guard()
        sym = ("sh" if c[0] == "6" else "bj" if c[0] in "849" else "sz") + c
        d = None
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
                break
            except Exception as e:
                say("新浪", c, "第", attempt + 1, "次失败", repr(e)[:80])
                time.sleep(2 * (attempt + 1))
        if d is None or not len(d):
            fails.append(c)
        else:
            d = d.copy()
            d["date"] = d["date"].astype(str).str[:10]
            d["code"] = c
            got.append(d[d["date"] >= "2022-12-01"])
        if i % 50 == 0:
            say("新浪进度", i, "/", len(codes), "失败", len(fails))
        time.sleep(0.5)
    pd.concat(got, ignore_index=True).to_parquet(RAW / "sina_qfq_now.parquet", index=False)
    say("新浪完成", len(got), "只，失败", fails)


if __name__ == "__main__":
    guard()
    s = pd.read_csv(RAW / "sample.csv", dtype={"code": str})
    codes = s["code"].str.zfill(6).tolist()
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("tdx", "both"):
        tdx(codes)
    if what in ("sina", "both"):
        sina(codes)
