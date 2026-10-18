"""公司事件（业绩预告 / 业绩快报 / 回购 / 分红送转 / 股权激励 / 股东增持）-> features_events.parquet + 诊断 results_events.txt。

    PYTHONIOENCODING=utf-8 python research/deepdata/feat_events.py

全部按公告日对齐（教训 25）：公告日 s 落在非交易日就算下一个交易日；s 日收盘起可用（沪深晚间公告东财记成次日，天然保守）。
「age」= 距最近一次公告的交易日数（当天 = 0），没有过公告 = NaN；窗口 flag / 计数按交易日数。
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

POS_TYPES = {"预增", "扭亏", "略增", "续盈"}
NEG_TYPES = {"预减", "首亏", "续亏", "略减"}


def _ev(g: C.Grid, ev: pd.DataFrame, datecol: str) -> np.ndarray:
    return g.day_index(ev[datecol].astype(str).str[:10], how="next")


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(float)


def main():
    t0 = time.time()
    pool = C.load_pool()
    g = C.make_grid(pool)
    rows = pool[pool["date"] >= C.FEAT_START]
    di = g.day_index(rows["date"])
    code = rows["code"]
    close = rows["close"].to_numpy(float)
    f = rows[["code", "date"]].copy().reset_index(drop=True)
    notes = []

    def put(name, A):
        f[name] = g.pick(A, code, di)

    def emit_age_flags(prefix, on, windows=(20, 60)):
        """on: 0/1 事件网格。age + 窗口内次数。"""
        cnt = on.copy()
        cnt[cnt == 0] = np.nan
        _, idx = C.Grid.ffill(cnt)
        age = np.where(idx >= 0, np.arange(on.shape[1])[None, :] - idx, np.nan)
        put(prefix + "_age", age)
        for w in windows:
            put(prefix + f"_n{w}", C.Grid.wsum(on, w))

    # ---- 业绩预告 ----
    p = C.RAW / "yjyg.parquet"
    if p.exists():
        y = pd.read_parquet(p, columns=["SECURITY_CODE", "NOTICE_DATE", "PREDICT_TYPE", "ADD_AMP_LOWER", "ADD_AMP_UPPER", "PREDICT_FINANCE_CODE", "REPORT_DATE"])
        y["code"] = y["SECURITY_CODE"].astype(str).str.zfill(6)
        y["pos"] = y["PREDICT_TYPE"].isin(POS_TYPES).astype(float)
        y["neg"] = y["PREDICT_TYPE"].isin(NEG_TYPES).astype(float)
        y["amp"] = np.nanmax(np.c_[_num(y["ADD_AMP_LOWER"]), _num(y["ADD_AMP_UPPER"])], axis=1)
        # 同一公告多行（净利润 / 扣非 / 营收）：按 (code, 公告日) 合成一条，正负取多数、幅度取最大
        a = y.groupby(["code", "NOTICE_DATE"], sort=False).agg(pos=("pos", "max"), neg=("neg", "max"), amp=("amp", "max")).reset_index()
        ei = _ev(g, a, "NOTICE_DATE")
        on = g.dense(a["code"], ei, np.ones(len(a)))
        on = (on > 0).astype(float)
        emit_age_flags("yjyg", on)
        pos = (g.dense(a["code"], ei, a["pos"].to_numpy(), agg="max") > 0).astype(float)
        neg = (g.dense(a["code"], ei, a["neg"].to_numpy(), agg="max") > 0).astype(float)
        put("yjyg_pos_n20", C.Grid.wsum(pos, 20))
        put("yjyg_pos_n60", C.Grid.wsum(pos, 60))
        put("yjyg_neg_n60", C.Grid.wsum(neg, 60))
        amp, _ = C.dense_last_ffill(g, a["code"], ei, a["amp"].to_numpy())
        put("yjyg_amp_last", amp)
        lastpos, _ = C.dense_last_ffill(g, a["code"], ei, a["pos"].to_numpy())
        put("yjyg_last_pos", lastpos)
        notes.append("业绩预告 %d 条公告（%s..%s）；类型取值 %s" % (len(a), a["NOTICE_DATE"].min(), a["NOTICE_DATE"].max(), y["PREDICT_TYPE"].value_counts().head(10).to_dict()))
    # ---- 业绩快报 ----
    p = C.RAW / "yjkb.parquet"
    if p.exists():
        k = pd.read_parquet(p, columns=["SECURITY_CODE", "NOTICE_DATE", "UPDATE_DATE", "YSTZ", "JLRTBZCL"])
        k["code"] = k["SECURITY_CODE"].astype(str).str.zfill(6)
        k["d"] = k["NOTICE_DATE"].astype(str).str[:10]
        k = k.sort_values(["code", "d"]).drop_duplicates(["code", "d"], keep="last")
        ei = _ev(g, k, "d")
        on = (g.dense(k["code"], ei, np.ones(len(k))) > 0).astype(float)
        emit_age_flags("yjkb", on)
        v, _ = C.dense_last_ffill(g, k["code"], ei, _num(k["JLRTBZCL"]))
        put("yjkb_np_yoy_last", v)
        v, _ = C.dense_last_ffill(g, k["code"], ei, _num(k["YSTZ"]))
        put("yjkb_rev_yoy_last", v)
        notes.append("业绩快报 %d 条（%s..%s）" % (len(k), k["d"].min(), k["d"].max()))
    # ---- 回购 ----
    p = C.RAW / "repurchase.parquet"
    if p.exists():
        r = pd.read_parquet(p)
        r["code"] = r["DIM_SCODE"].astype(str).str.zfill(6)
        r["d"] = r["NOTICEDATE"].astype(str).str[:10]
        r["share"] = _num(r["ZSZSX"])          # 回购上限占总市值 %
        r["amt"] = _num(r["REPURAMOUNTLIMIT"]) / 1e8
        r["is_plan"] = (r["DIM_DATE"].astype(str).str[:10] == r["d"]).astype(float)   # 公告日 == 首次公告日 -> 预案
        a = r.groupby(["code", "d"], sort=False).agg(share=("share", "max"), amt=("amt", "max"), is_plan=("is_plan", "max")).reset_index()
        ei = _ev(g, a, "d")
        on = (g.dense(a["code"], ei, np.ones(len(a))) > 0).astype(float)
        emit_age_flags("rep", on)
        plan = (g.dense(a["code"], ei, a["is_plan"].to_numpy(), agg="max") > 0).astype(float)
        put("rep_plan_n60", C.Grid.wsum(plan, 60))
        v, _ = C.dense_last_ffill(g, a["code"], ei, a["share"].to_numpy())
        put("rep_share_last", v)
        v, _ = C.dense_last_ffill(g, a["code"], ei, a["amt"].to_numpy())
        put("rep_amt_last", v)
        notes.append("回购 %d 条公告（%s..%s），REPURPROGRESS 取值 %s" % (len(a), a["d"].min(), a["d"].max(), r["REPURPROGRESS"].value_counts().head(8).to_dict()))
    # ---- 分红送转 ----
    p = C.RAW / "bonus.parquet"
    if p.exists():
        b = pd.read_parquet(p, columns=["SECURITY_CODE", "PLAN_NOTICE_DATE", "BONUS_IT_RATIO", "BONUS_RATIO", "IT_RATIO", "PRETAX_BONUS_RMB"])
        b["code"] = b["SECURITY_CODE"].astype(str).str.zfill(6)
        b["d"] = b["PLAN_NOTICE_DATE"].astype(str).str[:10]
        b["it"] = _num(b["BONUS_IT_RATIO"])
        b["it"] = np.where(np.isfinite(b["it"]), b["it"], 0.0)
        b["cash"] = _num(b["PRETAX_BONUS_RMB"])
        a = b.groupby(["code", "d"], sort=False).agg(it=("it", "max"), cash=("cash", "max")).reset_index()
        ei = _ev(g, a, "d")
        on = (g.dense(a["code"], ei, np.ones(len(a))) > 0).astype(float)
        emit_age_flags("bon", on, windows=(60,))
        hi = (g.dense(a["code"], ei, a["it"].to_numpy(), agg="max") >= 5).astype(float)      # 10 送转 >= 5
        put("bon_high_n60", C.Grid.wsum(hi, 60))
        v, _ = C.dense_last_ffill(g, a["code"], ei, a["it"].to_numpy())
        put("bon_it_last", v)
        notes.append("分红送转预案 %d 条（%s..%s），送转 >= 5 的 %d 条" % (len(a), a["d"].min(), a["d"].max(), int((a["it"] >= 5).sum())))
    # ---- 股权激励 ----
    p = C.RAW / "incentive.parquet"
    if p.exists():
        i = pd.read_parquet(p, columns=["SECURITY_CODE", "NOTICE_DATE", "PLAN_NOTICE_DATE", "PLAN_PROCESS_CH", "INITIAL_EXERCISE_PRICE", "EI_TARGET_CH"])
        i["code"] = i["SECURITY_CODE"].astype(str).str.zfill(6)
        i["d"] = i["NOTICE_DATE"].astype(str).str[:10]
        i["pd"] = i["PLAN_NOTICE_DATE"].astype(str).str[:10]
        i["px"] = _num(i["INITIAL_EXERCISE_PRICE"])
        a = i.sort_values(["code", "d"]).drop_duplicates(["code", "d"], keep="last")
        ei = _ev(g, a, "d")
        on = (g.dense(a["code"], ei, np.ones(len(a))) > 0).astype(float)
        emit_age_flags("inc", on, windows=(60,))
        pl = i.drop_duplicates(["code", "pd"])
        pl = pl[pl["pd"] >= "2021-01-01"]
        pi = _ev(g, pl, "pd")
        plan = (g.dense(pl["code"], pi, np.ones(len(pl))) > 0).astype(float)
        put("inc_plan_n60", C.Grid.wsum(plan, 60))
        put("inc_plan_n250", C.Grid.wsum(plan, 250))
        v, _ = C.dense_last_ffill(g, a["code"], ei, a["px"].to_numpy())
        px = g.pick(v, code, di)
        f["inc_px_prem"] = np.where(np.isfinite(px) & (px > 0), close / px - 1, np.nan)     # 现价对授予价的溢价
        notes.append("股权激励 %d 条公告 / %d 个预案（%s..%s），进度取值 %s" % (len(a), len(pl), a["d"].min(), a["d"].max(), i["PLAN_PROCESS_CH"].value_counts().head(8).to_dict()))
    # ---- 股东增持 ----
    p = C.RAW / "holder_inc.parquet"
    if p.exists():
        h = pd.read_parquet(p, columns=["SECURITY_CODE", "NOTICE_DATE", "CHANGE_FREE_RATIO", "CHANGE_NUM", "HOLDER_NAME"])
        h["code"] = h["SECURITY_CODE"].astype(str).str.zfill(6)
        h["d"] = h["NOTICE_DATE"].astype(str).str[:10]
        h["ratio"] = _num(h["CHANGE_FREE_RATIO"])
        a = h.groupby(["code", "d"], sort=False).agg(ratio=("ratio", "sum"), n=("HOLDER_NAME", "size")).reset_index()
        ei = _ev(g, a, "d")
        on = (g.dense(a["code"], ei, np.ones(len(a))) > 0).astype(float)
        emit_age_flags("hinc", on)
        rt = g.dense(a["code"], ei, a["ratio"].fillna(0).to_numpy())
        put("hinc_ratio60", C.Grid.wsum(rt, 60))
        notes.append("股东增持 %d 条公告日（%s..%s）" % (len(a), a["d"].min(), a["d"].max()))

    C.say("特征 %d 行 %.0fs" % (len(f), time.time() - t0))
    for n in notes:
        C.say(" ", n)
    readme = """
来源：东财 datacenter 六张事件表（raw/yjyg RPT_PUBLIC_OP_NEWPREDICT、yjkb RPT_FCI_PERFORMANCEE、repurchase RPTA_WEB_GETHGLIST_NEW、
bonus RPT_SHAREBONUS_DET、incentive RPT_EQUITYINCENTIVE、holder_inc RPT_SHARE_HOLDER_INCREASE(DIRECTION=增持)），2021-09..2025-12。
时点：一律按公告日（NOTICE_DATE / NOTICEDATE / PLAN_NOTICE_DATE），落在非交易日算下一交易日，公告日收盘起可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；事件表从 2021-09 起，所以 2021-12 的 60 日窗口完整、250 日窗口不完整。
列：
  yjyg_age / yjyg_n20 / yjyg_n60        距最近一次业绩预告的交易日数（无 = NaN）/ 近 20、60 个交易日公告次数
  yjyg_pos_n20 / yjyg_pos_n60 / yjyg_neg_n60   正向（预增 扭亏 略增 续盈）/ 负向（预减 首亏 续亏 略减）预告次数
  yjyg_amp_last / yjyg_last_pos         最近一次预告的同比变动幅度上限（%）/ 是否正向
  yjkb_age / yjkb_n20 / yjkb_n60 / yjkb_np_yoy_last / yjkb_rev_yoy_last   业绩快报：距上次 / 次数 / 最近一次净利润同比 % / 营收同比 %
  rep_age / rep_n20 / rep_n60 / rep_plan_n60 / rep_share_last / rep_amt_last   回购：距上次公告 / 次数 / 60 日内首次公告（预案）次数 /
                                        最近一次回购上限占总市值 % / 金额上限（亿元）
  bon_age / bon_n60 / bon_high_n60 / bon_it_last   分红送转预案：距上次 / 60 日内次数 / 60 日内 10 送转 >= 5 的次数 / 最近一次每 10 股送转数
  inc_age / inc_n60 / inc_plan_n60 / inc_plan_n250 / inc_px_prem   股权激励：距上次公告 / 60 日内公告数 / 60、250 日内新预案数 /
                                        现价对最近一次授予(行权)价的溢价（close/价 − 1）
  hinc_age / hinc_n20 / hinc_n60 / hinc_ratio60   股东增持：距上次公告 / 次数 / 60 日内增持占流通股比例合计（%）
缺失含义：*_age NaN = 表里 2021-09 以来没有过公告；计数列 0 = 真 0；*_last NaN = 没有过公告或该次没给数值。
"""
    C.write_table("events", f, readme)
    cols = [c for c in f.columns if c not in ("code", "date")]
    txt = C.diag_table(f, pool, cols, label="events")
    txt += "\n\n" + "\n".join(notes)
    (HERE / "results_events.txt").write_text(txt + "\n", encoding="utf-8")
    C.progress("诊断 events -> results_events.txt")


if __name__ == "__main__":
    main()
