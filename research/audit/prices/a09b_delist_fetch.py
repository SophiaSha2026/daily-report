"""a09b：退市票的历史从腾讯日 K raw 拉（新浪 / BaoStock 都没有），2020-07..2025-12 终止上市的沪深票 + 2026 年的。
    单线程、每只 1.5 秒、每 60 只歇 15 秒、连续失败 5 次熔断 180 秒。写 raw/tx_delist.parquet。"""
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
LOG = open(OUT / "a09b.log", "a", encoding="utf-8")


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


def fetch(code, adj=""):
    sym = ("sh" if code[0] == "6" else "sz") + code
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,800,{adj}"
    r = requests.get(url, timeout=20, headers=UA)
    data = r.json().get("data")
    if not isinstance(data, dict) or not data:
        raise ValueError("data 空: %s" % r.text[:80])
    v = list(data.values())[0]
    ks = (v.get("qfqday") if adj else v.get("day")) or v.get("day") or []
    rows = [{"date": k[0], "open": float(k[1]), "close": float(k[2]), "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])} for k in ks]
    df = pd.DataFrame(rows)
    df["code"] = code
    df["adj"] = adj or "raw"
    return df


def main():
    guard()
    dl = pd.read_csv(RAW / "delist_list.csv", dtype=str)
    dl["delist_date"] = dl["delist_date"].where(dl["delist_date"].notna(), dl["suspend_date"])
    if len(sys.argv) > 1 and sys.argv[1] == "sh":      # 第二趟只补上交所（第一趟漏了，见 a09c 注释）
        dl = dl[dl["exch"] == "sh"]
    dl = dl[(dl["delist_date"] >= "2020-07-01") & (dl["delist_date"] <= "2026-12-31")]
    dl = dl[dl["code"].str[0].isin(list("036"))]       # 去掉 B 股 (2/9 开头)
    codes = dl["code"].tolist()
    say("退市票", len(codes), "只")
    got, fails, streak, cool = [], [], 0, 0
    for i, c in enumerate(codes):
        if i and i % 60 == 0:
            guard()
            time.sleep(15)
        try:
            got.append(fetch(c))
            streak = 0
        except Exception as e:
            fails.append((c, repr(e)[:100]))
            streak += 1
            say("失败", c, repr(e)[:100])
            if streak >= 5:
                cool += 1
                if cool > 3:
                    say("熔断 3 次，停")
                    break
                say("连续失败 5 次，熔断 180 秒")
                time.sleep(180)
                streak = 0
        time.sleep(1.5)
        if i % 40 == 0:
            say("进度", i, "/", len(codes), "成功", len(got), "失败", len(fails))
    if got:
        outp = RAW / "tx_delist.parquet"
        new = pd.concat(got, ignore_index=True)
        if outp.exists():
            old = pd.read_parquet(outp)
            new = pd.concat([old[~old["code"].isin(set(new["code"]))], new], ignore_index=True)
        new.to_parquet(outp, index=False)
    (RAW / "tx_delist_fails.json").write_text(json.dumps(fails, ensure_ascii=False), encoding="utf-8")
    say("完成：成功", len(got), "失败", len(fails))


if __name__ == "__main__":
    main()
