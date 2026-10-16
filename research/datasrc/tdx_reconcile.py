"""
通达信 gbbq（股本变迁）算出的复权台阶 vs 新浪前复权因子台阶，20 只真实代码对账（docs/rethink_2026-10-16.md 拍板 G）。

    gbbq.zip 来自 www.tdx.com.cn/products/data/data/dbf/gbbq.zip，用 research/datasrc/.venv 里的 pytdx GbbqReader 解密，
    category 1（除权除息）存成 CSV：research/datasrc/.venv/Scripts/python.exe -c "from pytdx.reader import GbbqReader; ..."
    python research/datasrc/tdx_reconcile.py <gbbq_cat1.csv>

台阶公式（交易所除权参考价）：adj = (前收 − 每10股派息/10 + 每10股配股/10 × 配股价) / (1 + 每10股送转/10 + 每10股配股/10)，
台阶 = 前收 / adj；新浪台阶 = q_{t−1} / q_t（unadj_2020.parquet 的 qfq_factor）。代码取 2020-08 起事件最多的 10 只 + 随机 10 只。
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    g = pd.read_csv(sys.argv[1], dtype={"code": str})
    g["code"] = g["code"].str.zfill(6)
    g["date"] = pd.to_datetime(g["datetime"].astype(str)).dt.strftime("%Y-%m-%d")
    g = g[g["date"] >= "2020-08-01"]
    u = pd.read_parquet(ROOT / "data/breakout/raw/exp/leak/unadj_2020.parquet", columns=["code", "date", "close_u", "qfq_factor"])
    codes = set(pd.read_csv(ROOT / "cache/codes.csv", dtype=str)["code"].str.zfill(6))
    cnt = g[g["code"].isin(codes)].groupby("code").size().sort_values(ascending=False)
    pick = list(cnt.index[:10]) + list(cnt.sample(10, random_state=1).index)
    rows, steps, miss = [], 0, []
    for c in pick:
        x = u[u["code"] == c].sort_values("date").reset_index(drop=True)
        for _, e in g[g["code"] == c].iterrows():
            i = x.index[x["date"] >= e["date"]]
            if not len(i) or i[0] == 0 or x["date"][i[0]] != e["date"]:
                continue
            i = i[0]
            pc = x["close_u"][i - 1]
            adj = ((pc - e["hongli_panqianliutong"] / 10 + e["peigu_houzongguben"] / 10 * e["peigujia_qianzongguben"])
                   / (1 + e["songgu_qianzongguben"] / 10 + e["peigu_houzongguben"] / 10))
            tdx, sina = pc / adj, x["qfq_factor"][i - 1] / x["qfq_factor"][i]
            rows.append((c, e["date"], tdx, sina, abs(tdx / sina - 1)))
        st = x["qfq_factor"].shift(1) / x["qfq_factor"]
        ev = x[((st - 1).abs() > 1e-6) & (x["date"] >= "2020-08-01")]
        steps += len(ev)
        miss += [(c, d) for d in ev["date"] if d not in set(g[g["code"] == c]["date"])]
    r = pd.DataFrame(rows, columns=["code", "date", "tdx", "sina", "rel"])
    print("代码 %d，除权事件 %d；台阶相对差 <1e-4 %.1f%%、<1e-3 %.1f%%，最大 %.4f" % (
        len(pick), len(r), 100 * (r["rel"] < 1e-4).mean(), 100 * (r["rel"] < 1e-3).mean(), r["rel"].max()))
    print("新浪台阶 %d 个，通达信没有对应事件的 %d 个 %s" % (steps, len(miss), miss[:10]))
    print(r.sort_values("rel", ascending=False).head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
