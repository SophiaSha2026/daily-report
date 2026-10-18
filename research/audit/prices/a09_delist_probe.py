"""a09：幸存者偏差。(1) 沪深两所终止上市名单（akshare stock_info_sh_delist / stock_info_sz_delist）；
(2) 退市票的历史能不能从新浪 / 腾讯拉到（各试 5 只）；(3) 北交所退市名单试 akshare。只读外部接口，写 raw/。"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

t = bjtime.now_bj()
if 16 * 60 <= t.hour * 60 + t.minute < 18 * 60 + 30:
    sys.exit(2)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import datasource as ds  # noqa: E402

ds.http_timeout()
import akshare as ak  # noqa: E402

L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


frames = []
try:
    sh = ak.stock_info_sh_delist()
    say("上交所终止上市", len(sh), sh.columns.tolist())
    sh = sh.rename(columns={"公司代码": "code", "公司简称": "name", "暂停上市日期": "suspend_date", "终止上市日期": "delist_date"})
    sh["exch"] = "sh"
    frames.append(sh)
except Exception as e:
    say("上交所名单失败", repr(e)[:200])
for sym in ("终止上市公司", "暂停上市公司"):
    try:
        sz = ak.stock_info_sz_delist(symbol=sym)
        say("深交所", sym, len(sz), sz.columns.tolist())
        sz = sz.rename(columns={"证券代码": "code", "证券简称": "name", "上市日期": "list_date", "终止上市日期": "delist_date", "暂停上市日期": "suspend_date"})
        sz["exch"] = "sz"
        frames.append(sz)
    except Exception as e:
        say("深交所", sym, "失败", repr(e)[:200])
    time.sleep(1)
if frames:
    dl = pd.concat(frames, ignore_index=True)
    dl["code"] = dl["code"].astype(str).str.zfill(6)
    dl["delist_date"] = pd.to_datetime(dl["delist_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dl.to_csv(RAW / "delist_list.csv", index=False)
    win = dl[(dl["delist_date"] >= "2020-07-01") & (dl["delist_date"] <= "2025-12-31")]
    say("2020-07..2025-12 终止上市 %d 只（沪 %d 深 %d）；按年 %s" % (len(win), (win["exch"] == "sh").sum(), (win["exch"] == "sz").sum(),
                                                       win["delist_date"].str[:4].value_counts().sort_index().to_dict()))
    say("  按代码段 %s" % win["code"].str[:2].value_counts().to_dict())
    # 其中名字带 ST / 退 的
    say("  名字含 ST 的 %d，含「退」的 %d" % (win["name"].astype(str).str.contains("ST").sum(), win["name"].astype(str).str.contains("退").sum()))
    codes = pd.read_csv(ROOT / "cache/codes.csv", dtype=str)["code"].str.zfill(6)
    say("  这些退市代码还在 cache/codes.csv 里的 %d" % win["code"].isin(set(codes)).sum())
    # 也在 2026 年退市的（研究表 2026-10 取的，2026 年退市的也不在）
    w26 = dl[(dl["delist_date"] >= "2026-01-01")]
    say("2026 年退市 %d 只" % len(w26))
    # 试拉历史
    test = win.sample(min(6, len(win)), random_state=1)
    for _, r in test.iterrows():
        c = r["code"]
        sym = ("sh" if c[0] == "6" else "sz") + c
        try:
            d = ak.stock_zh_a_daily(symbol=sym, adjust="")
            say("  新浪 %s %s 退市 %s：%d 行 %s..%s" % (c, r["name"], r["delist_date"], len(d), d["date"].min(), d["date"].max()))
        except Exception as e:
            say("  新浪 %s %s：失败 %s" % (c, r["name"], repr(e)[:80]))
        time.sleep(0.8)
        try:
            import requests
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,,,800," % sym
            j = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
            v = list(j["data"].values())[0] if j.get("data") else {}
            k = v.get("day") or []
            say("  腾讯 %s：%d 行 %s" % (c, len(k), (k[0][0], k[-1][0]) if k else ""))
        except Exception as e:
            say("  腾讯 %s：失败 %s" % (c, repr(e)[:80]))
        time.sleep(0.8)
(OUT / "results_a09_probe.txt").write_text("\n".join(L), encoding="utf-8")
