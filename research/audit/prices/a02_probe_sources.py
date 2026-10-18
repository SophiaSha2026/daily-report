"""a02：第二数据源可达性实测（pytdx 通达信、腾讯日 K、新浪不复权），各打 3 只真实代码。只读，请求带 timeout。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

t = bjtime.now_bj()
hm = t.hour * 60 + t.minute
if 16 * 60 <= hm < 18 * 60 + 30:
    print("北京 16:00~18:30，不发请求", t)
    sys.exit(2)
print("BJ", t)

CODES = ["600000", "300750", "688008", "920002", "000001"]

# 1) pytdx（研究 venv）
try:
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
    api = TdxHq_API(raise_exception=False, auto_retry=True)
    hosts = [("119.147.212.81", 7709), ("115.238.56.198", 7709), ("180.153.18.170", 7709), ("119.147.164.60", 7709), ("124.71.187.122", 7709)]
    ok = None
    for h, p in hosts:
        t0 = time.time()
        try:
            if api.connect(h, p, time_out=8):
                print("pytdx 连上", h, round(time.time() - t0, 2), "s")
                ok = (h, p)
                break
        except Exception as e:
            print("pytdx", h, "失败", e)
    if ok:
        for c in CODES:
            mkt = 1 if c[0] == "6" else (2 if c[0] in "849" else 0)
            try:
                bars = api.get_security_bars(9, mkt, c, 0, 10)
                print(c, "bars", None if bars is None else len(bars), (bars or [None])[:2])
            except Exception as e:
                print(c, "get_security_bars 异常", repr(e)[:200])
        # 日线不足时试 get_k_data / 指数
        try:
            x = api.get_security_bars(9, 1, "600000", 0, 800)
            print("600000 800 根", None if x is None else (len(x), x[0]["datetime"], x[-1]["datetime"]))
        except Exception as e:
            print("800 根异常", repr(e)[:200])
        # 复权信息
        try:
            xdxr = api.get_xdxr_info(1, "600000")
            print("600000 xdxr", None if xdxr is None else (len(xdxr), xdxr[-2:]))
        except Exception as e:
            print("xdxr 异常", repr(e)[:200])
        api.disconnect()
except Exception as e:
    print("pytdx 不可用", repr(e)[:300])

# 2) 腾讯日 K（qfq 和 不复权）
import requests  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
for adj in ("qfq", ""):
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600000,day,,,5,%s" % adj
    try:
        r = requests.get(url, timeout=15, headers=UA)
        data = r.json()["data"]["sh600000"]
        k = data.get("qfqday") or data.get("day")
        print("腾讯 日K", adj or "raw", len(k), k[-1])
        print("  键", list(data.keys()))
    except Exception as e:
        print("腾讯 日K", adj, "失败", repr(e)[:200])
    time.sleep(0.5)
for sym in ("bj920002", "sh688008"):
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,,,3,qfq" % sym
    try:
        r = requests.get(url, timeout=15, headers=UA)
        print("腾讯", sym, str(r.json())[:300])
    except Exception as e:
        print("腾讯", sym, "失败", repr(e)[:200])
    time.sleep(0.5)

# 3) 新浪不复权（akshare）
try:
    import akshare as ak
    d = ak.stock_zh_a_daily(symbol="sh600000", adjust="")
    print("新浪 raw", len(d), d.columns.tolist(), d.tail(2).to_dict("records"))
    f = ak.stock_zh_a_daily(symbol="sh600000", adjust="qfq-factor")
    print("新浪 qfq-factor", len(f), f.columns.tolist(), f.tail(3).to_dict("records"))
    h = ak.stock_zh_a_daily(symbol="sh600000", adjust="hfq-factor")
    print("新浪 hfq-factor", len(h), h.tail(3).to_dict("records"))
except Exception as e:
    print("新浪 失败", repr(e)[:300])
