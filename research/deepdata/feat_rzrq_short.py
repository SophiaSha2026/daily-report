"""融券侧特征（research/altdata 只用了融资余额那一侧）-> features_rzrq_short.parquet + 诊断 results_rzrq_short.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_rzrq_short.py

输入 research/altdata/raw/rzrq.parquet（RPTA_WEB_RZRQ_GGMX，2021-09..2025-12，不重复拉）。
时点：交易所 T+1 早上发布，T 日收盘只能用 DATE <= T-1 的行 —— 全部列按「上一交易日」取值（整体错后一个交易日）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common_dd as C  # noqa: E402


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    z = pd.read_parquet(ROOT / "research" / "altdata" / "raw" / "rzrq.parquet", columns=["DATE", "SCODE", "RQYL", "RQMCL", "RQYE", "RQCHL", "RZYE", "SZ"])
    z["code"] = z["SCODE"].astype(str).str.zfill(6)
    z = z[z["code"].isin(set(g.codes))]
    di = g.day_index(z["DATE"].astype(str).str[:10])
    z = z[di >= 0]
    di = di[di >= 0] + 1          # 错后一个交易日：DATE 那行在下一交易日 T 的收盘前才可得
    keep = di < len(g.days)
    z = z[keep]
    di = di[keep]
    C.say("两融 %d 行 %d 只 %.0fs" % (len(z), z["code"].nunique(), time.time() - t0))
    num = lambda c: pd.to_numeric(z[c], errors="coerce").to_numpy(float)  # noqa: E731
    YL = g.dense(z["code"], di, num("RQYL"), agg="last")
    MCL = g.dense(z["code"], di, num("RQMCL"), agg="last")
    YE = g.dense(z["code"], di, num("RQYE"), agg="last")
    RZ = g.dense(z["code"], di, num("RZYE"), agg="last")
    SZ = g.dense(z["code"], di, num("SZ"), agg="last")
    ON = np.isfinite(YL).astype(float)
    rows = pool[pool["date"] >= C.FEAT_START]
    rdi = g.day_index(rows["date"])
    code = rows["code"]
    f = rows[["code", "date"]].copy().reset_index(drop=True)
    f["rq_target"] = g.pick(ON, code, rdi)

    def shift(A, k):
        B = np.full_like(A, np.nan)
        B[:, k:] = A[:, :-k]
        return B

    def wsum_nan(A, n):
        return C.Grid.wsum(np.nan_to_num(A), n)

    yl = g.pick(YL, code, rdi)
    f["rq_yl_chg5"] = np.where(g.pick(shift(YL, 5), code, rdi) > 0, yl / g.pick(shift(YL, 5), code, rdi) - 1, np.nan)
    f["rq_yl_chg20"] = np.where(g.pick(shift(YL, 20), code, rdi) > 0, yl / g.pick(shift(YL, 20), code, rdi) - 1, np.nan)
    ye, rz, sz = g.pick(YE, code, rdi), g.pick(RZ, code, rdi), g.pick(SZ, code, rdi)
    f["rq_share"] = np.where((ye + rz) > 0, ye / (ye + rz), np.nan)
    f["rq_ye_mcap"] = np.where(sz > 0, ye / sz, np.nan)
    m5 = g.pick(wsum_nan(MCL, 5), code, rdi)
    f["rq_mcl5_rel"] = np.where(yl > 0, m5 / yl, np.nan)
    m20 = g.pick(wsum_nan(MCL, 20), code, rdi)
    f["rq_mcl5_vs20"] = np.where(m20 > 0, m5 / (m20 / 4), np.nan)
    f["rq_zero_yl"] = np.where(np.isfinite(yl), (yl <= 0).astype(float), np.nan)
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：东财两融个股明细（research/altdata/raw/rzrq.parquet，不重复拉）融券那一侧。
时点：交易所 T+1 早上发布，所有列按 DATE = T 的上一个交易日取值（整体错后一天），T 日收盘可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；非两融标的全部 NaN（rq_target = 0）。
列：
  rq_target        是否两融标的（上一交易日有两融行）
  rq_yl_chg5 / 20  融券余量 5 / 20 个交易日变化率
  rq_share         融券余额 / (融资余额 + 融券余额)
  rq_ye_mcap       融券余额 / 流通市值（SZ 列）
  rq_mcl5_rel      近 5 日融券卖出量合计 / 当前融券余量
  rq_mcl5_vs20     近 5 日融券卖出量 / (近 20 日融券卖出量 / 4)，融券卖出突增倍数
  rq_zero_yl       融券余量为 0（标的但无券）
缺失含义：NaN = 非标的或分母为 0。
"""
    C.write_table("rzrq_short", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="rzrq_short（T-1 可得）")
    (HERE / "results_rzrq_short.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 rzrq_short -> results_rzrq_short.txt")


if __name__ == "__main__":
    main()
