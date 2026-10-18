"""十大流通股东（定期报告，按公告日）-> features_freeholders.parquet + 诊断 results_freeholders.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_freeholders.py

输入 raw/freeholders.parquet（RPT_F10_EH_FREEHOLDERS，按 NOTICE_DATE 月块拉 2021-09..2025-12）。
时点：按 NOTICE_DATE（定期报告公告日），非交易日算下一交易日，公告日收盘起可用；END_DATE（报告期）只用来排序，不用来对齐（教训 25）。
每份报告合成一行，然后沿交易日前向沿用到下一份报告。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_dd as C  # noqa: E402


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    cols = ["SECURITY_CODE", "END_DATE", "NOTICE_DATE", "HOLDER_NAME", "FREE_HOLDNUM_RATIO", "HOLD_NUM_CHANGE", "CHANGE_RATIO",
            "IS_HOLDORG", "HOLDER_RANK", "HOLDER_STATE", "HOLDER_NEWTYPE", "HOLDER_TYPE", "SHARES_TYPE"]
    h = pd.read_parquet(C.RAW / "freeholders.parquet")
    h = h[[c for c in cols if c in h.columns]]
    h["code"] = h["SECURITY_CODE"].astype(str).str.zfill(6)
    h = h[h["code"].isin(set(g.codes))]
    h["d"] = h["NOTICE_DATE"].astype(str).str[:10]
    h["end"] = h["END_DATE"].astype(str).str[:10]
    st = h["HOLDER_STATE"].astype(str) if "HOLDER_STATE" in h else pd.Series("", index=h.index)
    h["new"] = st.str.contains("新进").astype(float)
    h["up"] = st.str.contains("增持|增加").astype(float)
    h["down"] = st.str.contains("减持|减少").astype(float)
    h["org"] = (pd.to_numeric(h.get("IS_HOLDORG"), errors="coerce").fillna(0) > 0).astype(float)
    ht = h["HOLDER_NEWTYPE"].astype(str) if "HOLDER_NEWTYPE" in h else pd.Series("", index=h.index)
    h["person"] = ht.str.contains("个人|自然人").astype(float)
    h["fund"] = ht.str.contains("基金|社保|保险|QFII|券商|信托|年金").astype(float)
    h["ratio"] = pd.to_numeric(h["FREE_HOLDNUM_RATIO"], errors="coerce")
    h["chg"] = pd.to_numeric(h["CHANGE_RATIO"], errors="coerce")
    # 同一公告日可能有多份（年报 + 一季报同日）：取报告期最新的那份
    h = h.sort_values(["code", "d", "end"])
    last_end = h.groupby(["code", "d"])["end"].transform("max")
    h = h[h["end"] == last_end]
    a = h.groupby(["code", "d"], sort=False).agg(n=("HOLDER_NAME", "size"), ratio=("ratio", "sum"), new=("new", "sum"), up=("up", "sum"),
                                                  down=("down", "sum"), org=("org", "sum"), person=("person", "sum"), fund=("fund", "sum"),
                                                  chg=("chg", "sum"), end=("end", "max")).reset_index()
    a = a[a["n"] >= 5]
    ei = g.day_index(a["d"], how="next")
    a = a[ei >= 0]
    ei = ei[ei >= 0]
    C.say("十大流通股东 %d 行 -> 报告 %d 份 %s..%s；HOLDER_STATE 取值 %s；HOLDER_NEWTYPE 取值 %s" % (
        len(h), len(a), a["d"].min(), a["d"].max(), st.value_counts().head(6).to_dict(), ht.value_counts().head(8).to_dict()))
    rows = pool[pool["date"] >= C.FEAT_START]
    di = g.day_index(rows["date"])
    code = rows["code"]
    f = rows[["code", "date"]].copy().reset_index(drop=True)
    for c, name in (("ratio", "fh_top10_ratio"), ("new", "fh_new_n"), ("up", "fh_up_n"), ("down", "fh_down_n"), ("org", "fh_org_n"),
                    ("person", "fh_person_n"), ("fund", "fh_fund_n"), ("chg", "fh_chg_sum")):
        V, age = C.dense_last_ffill(g, a["code"], ei, a[c].to_numpy(float))
        f[name] = g.pick(V, code, di)
        if c == "ratio":
            f["fh_age"] = g.pick(age, code, di)
    # 上一份报告的前十合计，算变化
    a2 = a.sort_values(["code", "d"])
    a2["prev_ratio"] = a2.groupby("code")["ratio"].shift(1)
    V, _ = C.dense_last_ffill(g, a2["code"], g.day_index(a2["d"], how="next"), (a2["ratio"] - a2["prev_ratio"]).to_numpy(float))
    f["fh_top10_dratio"] = g.pick(V, code, di)
    f["fh_fresh20"] = (f["fh_age"] <= 20).astype(float)
    f.loc[~np.isfinite(f["fh_age"]), "fh_fresh20"] = np.nan
    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    readme = """
来源：东财十大流通股东（RPT_F10_EH_FREEHOLDERS），每份定期报告一组，2021-09..2025-12 按公告日拉。
时点：NOTICE_DATE 公告日收盘起可用，之后沿用到下一份报告；报告期不用于对齐。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；2021-09 之前没有报告的票在首份公告前为 NaN。
列（都是「最近一份已公告报告」的值）：
  fh_age            距最近一份报告公告日的交易日数
  fh_fresh20        最近一份报告在 20 个交易日内公告（刚出季报 / 年报）
  fh_top10_ratio    前十大流通股东持股占流通股比例合计（%）
  fh_top10_dratio   对上一份报告的变化（百分点）
  fh_new_n / fh_up_n / fh_down_n   前十里新进 / 增持 / 减持 的席位数
  fh_org_n / fh_fund_n / fh_person_n   机构席位数 / 基金社保保险等席位数 / 自然人席位数
  fh_chg_sum        前十持股变动比例合计（东财 CHANGE_RATIO 之和）
缺失含义：NaN = 2021-09 以来没有可用报告。
"""
    C.write_table("freeholders", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="freeholders")
    (HERE / "results_freeholders.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 freeholders -> results_freeholders.txt")


if __name__ == "__main__":
    main()
