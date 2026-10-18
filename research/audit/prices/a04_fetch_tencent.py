"""a04：第二源取数（只读外部接口，写 raw/）。
    腾讯日 K 不复权 day（800 根）和 qfq（800 根）：样本里的沪深票；北交所腾讯没有历史，另试几个 URL 变体各打一次。
    单线程、每只间隔 1 秒、每 100 只歇 10 秒、连续失败 5 次全局熔断 150 秒（最多 3 次）。
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
LOG = open(OUT / "a04.log", "a", encoding="utf-8")


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


def prefix(c):
    if c[0] == "6":
        return "sh" + c
    if c[0] in "849":
        return "bj" + c
    return "sz" + c


def fetch(code, adj):
    url = f"{KLINE}?param={prefix(code)},day,,,800,{adj}"
    r = requests.get(url, timeout=20, headers=UA)
    data = r.json().get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError("data 空: %s" % r.text[:80])
    v = list(data.values())[0]
    ks = v.get("qfqday") if adj else v.get("day")
    if ks is None:
        ks = v.get("day") or v.get("qfqday") or []
    rows = []
    for k in ks:
        rows.append({"date": k[0], "open": float(k[1]), "close": float(k[2]), "high": float(k[3]), "low": float(k[4]),
                     "volume": float(k[5]), "extra": json.dumps(k[6:], ensure_ascii=False) if len(k) > 6 else ""})
    df = pd.DataFrame(rows)
    df["code"] = code
    df["adj"] = adj or "raw"
    return df


def main():
    guard()
    s = pd.read_csv(RAW / "sample.csv", dtype={"code": str})
    codes = s["code"].str.zfill(6).tolist()
    # 先试北交所几个 URL 变体，各一次
    for url in ("https://web.ifzq.gtimg.cn/appstock/app/kline/kline?param=bj920002,day,,,30,qfq",
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=bj920002,day,2025-01-01,2025-03-01,100,",
                "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get?param=bj920002,day,,,100,qfq",
                "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.920002&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=0&beg=20250101&end=20250301"):
        try:
            r = requests.get(url, timeout=12, headers=UA)
            say("BJ 变体", url[:90], r.status_code, r.text[:160].replace("\n", " "))
        except Exception as e:
            say("BJ 变体", url[:90], "失败", repr(e)[:120])
        time.sleep(1)
    jobs = [(c, "") for c in codes if c[0] not in "849"]
    jobs += [(c, "qfq") for c in codes if c[0] not in "849"][:160]
    say("任务", len(jobs), "个请求")
    got, fails, streak, cool = [], [], 0, 0
    for i, (c, adj) in enumerate(jobs):
        if i and i % 100 == 0:
            guard()
            time.sleep(10)
        try:
            df = fetch(c, adj)
            got.append(df)
            streak = 0
        except Exception as e:
            fails.append((c, adj, repr(e)[:100]))
            streak += 1
            say("失败", c, adj, repr(e)[:100])
            if streak >= 5:
                cool += 1
                if cool > 3:
                    say("熔断 3 次，停")
                    break
                say("连续失败 5 次，熔断 150 秒")
                time.sleep(150)
                streak = 0
        time.sleep(1.0)
        if i % 50 == 0:
            say("进度", i, "/", len(jobs), "成功", len(got), "失败", len(fails))
    if got:
        pd.concat(got, ignore_index=True).to_parquet(RAW / "tx_kline.parquet", index=False)
    (RAW / "tx_fails.json").write_text(json.dumps(fails, ensure_ascii=False), encoding="utf-8")
    say("完成：成功", len(got), "失败", len(fails))


if __name__ == "__main__":
    main()
