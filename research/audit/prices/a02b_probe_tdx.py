"""a02b：pytdx get_security_bars 为什么返回 None；换 mootdx / 多主机 / raise_exception 看原因。"""
import sys
import time
import traceback

sys.path.insert(0, r"C:\home\daily-report\src")
import bjtime  # noqa: E402

t = bjtime.now_bj()
if 16 * 60 <= t.hour * 60 + t.minute < 18 * 60 + 30:
    sys.exit(2)

import pytdx  # noqa: E402
print("pytdx", getattr(pytdx, "__version__", "?"))
from pytdx.hq import TdxHq_API  # noqa: E402

hosts = [("115.238.56.198", 7709), ("119.147.212.81", 7709), ("124.71.187.122", 7709), ("122.51.120.217", 7709),
         ("111.229.247.189", 7709), ("124.71.9.153", 7709), ("119.97.185.59", 7709), ("47.107.75.159", 7709)]
for h, p in hosts:
    api = TdxHq_API(raise_exception=True, auto_retry=False)
    try:
        if not api.connect(h, p, time_out=6):
            print(h, "连不上")
            continue
        t0 = time.time()
        q = api.get_security_quotes([(1, "600000")])
        print(h, "quotes", None if q is None else q[0].get("price"), round(time.time() - t0, 2))
        bars = api.get_security_bars(9, 1, "600000", 0, 5)
        print(h, "bars", bars)
        cnt = api.get_security_count(1)
        print(h, "count sh", cnt)
    except Exception:
        print(h, "异常:", traceback.format_exc()[-600:])
    finally:
        try:
            api.disconnect()
        except Exception:
            pass

print("---- mootdx ----")
try:
    from mootdx.quotes import Quotes
    client = Quotes.factory(market="std", timeout=10)
    df = client.bars(symbol="600000", frequency=9, offset=5)
    print("mootdx bars", None if df is None else df.tail(3))
    df = client.bars(symbol="920002", frequency=9, offset=5)
    print("mootdx bj", None if df is None else df.tail(3))
    x = client.xdxr(symbol="600000")
    print("mootdx xdxr", None if x is None else len(x))
except Exception:
    print("mootdx 异常:", traceback.format_exc()[-800:])
