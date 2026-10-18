"""新浪 stock_zh_a_daily 能拉到多早、退市票给不给（真实代码，教训 2）。单进程顺序跑，十几个请求。"""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
import backfill as BF
BF._net_guard()
import akshare as ak
import pandas as pd

LIVE = ["600000", "000001", "000002", "600519", "002001", "300001", "688001", "920002"]
# 已退市 / 被吸收合并（真实代码）：
DEAD = ["600005", "000024", "600485", "300104", "000693", "002070", "600001", "600087", "002680", "300028", "600401", "600240", "600747", "000979", "600806", "000995"]
rows = []
for c in LIVE + DEAD:
    t0 = time.time()
    try:
        u = ak.stock_zh_a_daily(symbol=BF.prefix(c), adjust="")
        q = ak.stock_zh_a_daily(symbol=BF.prefix(c), adjust="qfq-factor")
        rows.append((c, "live" if c in LIVE else "dead", len(u), str(u["date"].min())[:10], str(u["date"].max())[:10],
                     len(q) if q is not None else -1, list(u.columns)[:12], round(time.time() - t0, 1)))
    except Exception as e:
        rows.append((c, "live" if c in LIVE else "dead", -1, "", "", -1, repr(e)[:80], round(time.time() - t0, 1)))
    print(rows[-1], flush=True)
pd.DataFrame(rows, columns=["code", "kind", "n", "first", "last", "nq", "cols", "sec"]).to_csv(Path(__file__).parent / "probe.csv", index=False)
