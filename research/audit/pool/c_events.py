"""审计 C：事件剔除（减持 / 解禁 / 增发）的时点。

离线：
  1 东财公告 notice_date 对 art_code 里的发布日：晚间公告是否记成次日（决定「T 日收盘后才出的公告，研究在 T 日看不看得到」）。
  2 减持记录 NOTICE_DATE 对 减持区间 END_DATE。
  3 增发：研究 / 生产都按「发行日期」剔，发行结果公告一般晚几天才出 -> 用长期调整突破的公告缓存（data/pullback/corp，按票拉的融资公告）
    量「发行日期 -> 第一份发行结果 / 发行情况报告书 / 上市公告书 的公告日」的滞后分布。qbzf 原表的 发行方式 / 发行日期 / 增发上市日期。
  4 解禁：东财「占解禁前流通市值比例」用的是**实际**解禁数量；T 日只知道计划数量。过去月 实际 对 计划 的差异率；未来月这一列有没有值。
联网（巨潮全文检索，顺序、每请求 1 秒、timeout 30）：
  5 50 个真实例子：从研究候选池前 10 里抽 被剔 / 没被剔 的 (票, 日)，查原始公告的真实发布时间（announcementTime），
    对照研究用的事件日，判「T 日收盘时知不知道」。
产物：results_c_events.txt、c_examples.csv
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "rulecore", ROOT / "research" / "screen", ROOT / "research" / "stats"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
EV = EXP / "rulecore/events"
HERE = ROOT / "research/audit/pool"
RAW = HERE / "raw"
L = []
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def in_window() -> bool:
    import bjtime
    t = bjtime.now_bj()
    hm = t.hour * 60 + t.minute
    return t.weekday() < 5 and 16 * 60 <= hm < 18 * 60 + 30


def cninfo(code: str, key: str, sdate: str, edate: str) -> list[dict]:
    import requests
    for i in range(2):
        try:
            r = requests.post("http://www.cninfo.com.cn/new/fulltextSearch/full",
                              params={"searchkey": f"{code} {key}", "sdate": sdate, "edate": edate, "isfulltext": "false",
                                      "sortName": "pubdate", "sortType": "asc", "pageNum": "1", "pageSize": "30"}, headers=UA, timeout=30)
            j = r.json()
            out = []
            for a in j.get("announcements") or []:
                if str(a.get("secCode")) != code:
                    continue
                out.append({"title": re.sub(r"</?em>", "", a.get("announcementTitle", "")), "time": pd.to_datetime(a.get("announcementTime"), unit="ms") + pd.Timedelta(hours=8)})
            return out
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return []


def main():
    t0 = time.time()
    say("== 1 东财公告 notice_date 对发布日（art_code 前 8 位）==")
    ann = pd.read_parquet(EV / "reduce_ann.parquet")
    ann["art_d"] = pd.to_datetime(ann["art"].str[2:10], format="%Y%m%d", errors="coerce")
    ann["nd"] = pd.to_datetime(ann["date"])
    lag = (ann["nd"] - ann["art_d"]).dt.days
    say("持股变动公告 %d 条；notice_date − 发布日：%s" % (len(ann), lag.value_counts().sort_index().loc[-3:5].to_dict()))
    say("notice_date 星期分布 %s；发布日星期分布 %s" % (ann["nd"].dt.weekday.value_counts().sort_index().to_dict(), ann["art_d"].dt.weekday.value_counts().sort_index().to_dict()))
    say("解读：发布日落在周五 / 周日的公告 notice_date 记成周六 / 周一，说明东财 notice_date = 公告正文日期（次一自然日），不是上传时刻；"
        "研究按 notice_date <= T 判，所以 T 日晚间发布（notice_date = T+1）的减持公告在 T 日看不到，不偷看。")
    rec = pd.read_parquet(EV / "reduce_rec.parquet")
    d = (pd.to_datetime(rec["date"]) - pd.to_datetime(rec["end"], errors="coerce")).dt.days
    say("\n== 2 减持记录：NOTICE_DATE − 减持区间结束日 分位 %s；NOTICE < END 的 %d 条 / %d" % (d.quantile([0.05, 0.5, 0.95]).to_dict(), int((d < 0).sum()), len(rec)))

    say("\n== 3 增发：发行日期 对 结果公告日 ==")
    q = pd.read_parquet(RAW / "qbzf.parquet") if (RAW / "qbzf.parquet").exists() else None
    if q is not None:
        q["fd"] = pd.to_datetime(q["发行日期"], errors="coerce")
        q["ld"] = pd.to_datetime(q["增发上市日期"], errors="coerce")
        say("qbzf %d 行；发行方式分布 %s" % (len(q), q["发行方式"].value_counts().to_dict()))
        lag2 = (q["ld"] - q["fd"]).dt.days
        say("增发上市日期 − 发行日期：分位 %s；上市日期缺 %d" % (lag2.quantile([0.1, 0.5, 0.9]).to_dict(), int(q["ld"].isna().sum())))
        say("发行日期 > 今天（还没发、已登表）的行 %d：%s" % (int((q["fd"] > pd.Timestamp.today()).sum()), q.loc[q["fd"] > pd.Timestamp.today(), ["股票代码", "发行日期", "发行方式"]].head(5).to_dict("records")))
    import corp_events as CE
    pl = pd.read_parquet(EV / "placement.parquet")
    pl["code"] = pl["code"].astype(str).str.zfill(6)
    rows = []
    for fp in glob.glob(str(ROOT / "data/pullback/corp/*.json")):
        code = Path(fp).stem
        if not re.match(r"^\d{6}$", code):
            continue
        c = json.load(open(fp, encoding="utf-8"))
        if not isinstance(c, dict) or "ann" not in c:
            continue
        a2 = [x for x in c.get("ann", []) if x.get("node") == "2"]
        if not a2:
            continue
        for _, r in pl[(pl["code"] == code) & (pl["date"] >= c["from"]) & (pl["date"] <= c["through"])].iterrows():
            fd = r["date"]
            done = sorted(x["date"] for x in a2 if (CE.DONE_TITLE.search(x.get("title", "")) or set(x.get("cats", [])) & CE.DONE) and fd <= x["date"] <= (pd.Timestamp(fd) + pd.Timedelta(days=90)).strftime("%Y-%m-%d"))
            plan = sorted(x["date"] for x in a2 if x["date"] <= fd and CE.PLAN_DOC.search(x.get("title", "")))
            rows.append({"code": code, "issue_date": fd, "first_done_ann": done[0] if done else None,
                         "lag_days": (pd.Timestamp(done[0]) - pd.Timestamp(fd)).days if done else None, "has_plan_before": bool(plan)})
    P = pd.DataFrame(rows)
    if len(P):
        say("公告缓存里能对上的增发 %d 笔（%d 只）；找到结果公告的 %d 笔；发行日期 -> 结果公告 滞后天数分位 %s；滞后 >= 1 天的占 %.0f%%；发行日期前有预案的 %.0f%%" % (
            len(P), P["code"].nunique(), int(P["lag_days"].notna().sum()), P["lag_days"].dropna().quantile([0.1, 0.5, 0.9]).to_dict(),
            100 * (P["lag_days"].dropna() >= 1).mean(), 100 * P["has_plan_before"].mean()))
        say("  样例：%s" % P.dropna().head(6).to_dict("records"))
        P.to_csv(HERE / "c_placement_lag.csv", index=False)
    # 研究表里 placement60 在候选池前 10 剔掉的名额里，有多少是「发行日期在 T 之前 7 天内」（结果公告很可能还没出）
    pool = pd.read_parquet(EXP / "polish/pool.parquet", columns=["code", "date", "elig0", "pool", "lret20", "ev_reduce30", "ev_release30", "ev_placement60", "ev_any", "y_open"])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool = pool[(pool["date"] >= "2022-01-01")].copy()
    pool = pool.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    pool["rank0"] = pool.groupby("date").cumcount() + 1        # elig0 内名次（剔事件前）
    top0 = pool[pool["rank0"] <= 10]
    say("\n剔事件前的前 10 名额 %d：被减持剔 %d、解禁剔 %d、增发剔 %d（任一 %d，%.1f%%）；被剔名额命中 %.1f%%，未剔 %.1f%%" % (
        len(top0), int(top0["ev_reduce30"].sum()), int(top0["ev_release30"].sum()), int(top0["ev_placement60"].sum()), int(top0["ev_any"].sum()),
        100 * top0["ev_any"].mean(), 100 * top0.loc[top0["ev_any"], "y_open"].mean(), 100 * top0.loc[~top0["ev_any"], "y_open"].mean()))
    plc = top0[top0["ev_placement60"]].merge(pl.rename(columns={"date": "issue_date"}), on="code")
    plc = plc[(plc["issue_date"] <= plc["date"])]
    plc["age"] = (pd.to_datetime(plc["date"]) - pd.to_datetime(plc["issue_date"])).dt.days
    plc = plc[plc["age"] <= 60].sort_values("age").drop_duplicates(["code", "date"])
    say("增发剔掉的前 10 名额 %d：发行日期距 T 的天数分位 %s；<= 7 天（结果公告多半还没出，研究提前知道）的 %d 个" % (
        len(plc), plc["age"].quantile([0.1, 0.5, 0.9]).to_dict() if len(plc) else {}, int((plc["age"] <= 7).sum())))

    say("\n== 4 解禁：实际 对 计划 ==")
    for nm in ("release_2025-03.parquet", "release_2026-11.parquet"):
        p = RAW / nm
        if not p.exists():
            continue
        r = pd.read_parquet(p)
        a_, b_ = pd.to_numeric(r["解禁数量"], errors="coerce"), pd.to_numeric(r["实际解禁数量"], errors="coerce")
        s = pd.to_numeric(r["占解禁前流通市值比例"], errors="coerce")
        say("%s：%d 行；实际 != 计划 的 %d 行（%.0f%%），实际/计划 分位 %s；占比列有值 %d，>= 5%% 的 %d；实际解禁数量缺 %d" % (
            nm, len(r), int((a_ != b_).sum()), 100 * (a_ != b_).mean(), (b_ / a_).quantile([0.05, 0.25, 0.5]).round(3).to_dict(),
            int(s.notna().sum()), int((s >= 0.05).sum()), int(b_.isna().sum())))
    rel = pd.read_parquet(EV / "release.parquet")
    say("研究 release.parquet：%d 行，share 中位 %.3f，>= 5%% 的 %.0f%%" % (len(rel), rel["share"].median(), 100 * (rel["share"] >= 0.05).mean()))

    # ---------------- 5 联网核 50 个例子 ----------------
    say("\n== 5 50 个例子：原始公告时间（巨潮 announcementTime，北京时间）==")
    if in_window():
        say("禁区内，不联网，跳过")
    else:
        ann_all = ann.copy()
        rec["code"] = rec["code"].astype(str).str.zfill(6)
        rng = np.random.default_rng(20261018)
        ex = []
        # 被减持剔的前 10 名额：找研究用的事件日（T 往前 30 天内最近一条）
        red = top0[top0["ev_reduce30"] & (top0["date"] >= "2023-01-01")]
        red = red.iloc[rng.choice(len(red), size=min(18, len(red)), replace=False)]
        for _, r in red.iterrows():
            T = r["date"]
            lo = (pd.Timestamp(T) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            a1 = ann_all[(ann_all["code"] == r["code"]) & (ann_all["date"] >= lo) & (ann_all["date"] <= T) & ann_all["title"].str.contains("减持") & ~ann_all["title"].str.contains("回购")]
            if len(a1):
                a1 = a1[~a1["title"].map(lambda x: bool(CE.NOT_REDUCE.search(x))).to_numpy(bool)]
            r1 = rec[(rec["code"] == r["code"]) & (rec["date"] >= lo) & (rec["date"] <= T)]
            evd = max([*a1["date"].tolist(), *r1["date"].tolist()] or [None])
            src = "公告" if (len(a1) and evd in set(a1["date"])) else "减持记录"
            ex.append({"kind": "减持-被剔", "code": r["code"], "T": T, "event_date_used": evd, "src": src, "title": a1["title"].iloc[-1] if len(a1) else "", "y_open": r["y_open"]})
        # 没被剔、但 31~45 天前有减持公告（刚出窗口）的前 10 名额
        near = []
        cand = top0[~top0["ev_any"] & (top0["date"] >= "2023-01-01")]
        cand = cand.iloc[rng.choice(len(cand), size=min(4000, len(cand)), replace=False)]
        for _, r in cand.iterrows():
            T = pd.Timestamp(r["date"])
            a1 = ann_all[(ann_all["code"] == r["code"]) & (ann_all["nd"] >= T - pd.Timedelta(days=45)) & (ann_all["nd"] <= T - pd.Timedelta(days=31)) & ann_all["title"].str.contains("减持") & ~ann_all["title"].str.contains("回购")]
            if len(a1):
                near.append({"kind": "减持-刚出窗", "code": r["code"], "T": r["date"], "event_date_used": a1["date"].max(), "src": "公告", "title": a1["title"].iloc[-1], "y_open": r["y_open"]})
            if len(near) >= 8:
                break
        ex += near
        # 解禁剔的：事件日 = T 之后 30 天内的解禁日
        rel["code"] = rel["code"].astype(str).str.zfill(6)
        rl = top0[top0["ev_release30"] & (top0["date"] >= "2023-01-01")]
        rl = rl.iloc[rng.choice(len(rl), size=min(12, len(rl)), replace=False)]
        for _, r in rl.iterrows():
            T = pd.Timestamp(r["date"])
            x = rel[(rel["code"] == r["code"]) & (pd.to_datetime(rel["date"]) > T) & (pd.to_datetime(rel["date"]) <= T + pd.Timedelta(days=30))]
            ex.append({"kind": "解禁-被剔", "code": r["code"], "T": r["date"], "event_date_used": x["date"].min() if len(x) else None, "src": "解禁日", "title": "", "y_open": r["y_open"], "share": float(x["share"].sum()) if len(x) else np.nan})
        # 增发剔的
        pc = plc.iloc[rng.choice(len(plc), size=min(12, len(plc)), replace=False)] if len(plc) else plc
        for _, r in pc.iterrows():
            ex.append({"kind": "增发-被剔", "code": r["code"], "T": r["date"], "event_date_used": r["issue_date"], "src": "发行日期", "title": "", "y_open": r["y_open"]})
        say("抽到 %d 个例子（%s），开始查巨潮，约 %d 个请求" % (len(ex), pd.Series([e["kind"] for e in ex]).value_counts().to_dict(), len(ex)))
        out = []
        for e in ex:
            evd = e["event_date_used"]
            T = e["T"]
            if evd is None:
                e["verdict"] = "研究表里找不到事件日（?）"
                out.append(e)
                continue
            if e["kind"].startswith("减持"):
                hits = cninfo(e["code"], "减持", (pd.Timestamp(evd) - pd.Timedelta(days=4)).strftime("%Y-%m-%d"), (pd.Timestamp(evd) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
                hits = [h for h in hits if "减持" in h["title"] and "回购" not in h["title"]]
                # 最接近事件日的那份
                if hits:
                    h = min(hits, key=lambda h: abs((h["time"].normalize() - pd.Timestamp(evd)).days))
                    pub = h["time"]
                    # T 日收盘（15:00）时知不知道
                    known_at_T_close = pub <= pd.Timestamp(T) + pd.Timedelta(hours=15)
                    e.update({"cninfo_title": h["title"][:40], "cninfo_time": pub.strftime("%Y-%m-%d %H:%M"),
                              "pub_vs_used_days": (pub.normalize() - pd.Timestamp(evd)).days, "known_at_T_close": bool(known_at_T_close)})
                    e["verdict"] = "T 日收盘前已公开" if known_at_T_close else ("T 日收盘后才发布（研究在 T 日已剔 = 偷看一晚）" if e["kind"] == "减持-被剔" else "T 日后发布")
                else:
                    e["verdict"] = "巨潮没查到对应公告"
            elif e["kind"].startswith("解禁"):
                hits = cninfo(e["code"], "上市流通", (pd.Timestamp(evd) - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), evd)
                hits = [h for h in hits if ("上市流通" in h["title"] or "解除限售" in h["title"] or "解禁" in h["title"])]
                if hits:
                    h = hits[0]
                    lead = (pd.Timestamp(evd) - h["time"].normalize()).days
                    e.update({"cninfo_title": h["title"][:40], "cninfo_time": h["time"].strftime("%Y-%m-%d %H:%M"), "lead_days": lead,
                              "known_at_T_close": bool(h["time"] <= pd.Timestamp(T) + pd.Timedelta(hours=15))})
                    e["verdict"] = "提示性公告在 T 收盘前已出（解禁日 T 日已知）" if e["known_at_T_close"] else "提示性公告在 T 之后才出（但首发 / 定增解禁日在招股书 / 上市公告里早已确定）"
                else:
                    e["verdict"] = "巨潮 40 天内没查到提示性公告（解禁日来自更早的招股书 / 上市公告）"
            else:
                hits = cninfo(e["code"], "发行", evd, (pd.Timestamp(evd) + pd.Timedelta(days=60)).strftime("%Y-%m-%d"))
                hits = [h for h in hits if CE.DONE_TITLE.search(h["title"]) or "发行结果" in h["title"]]
                if hits:
                    h = hits[0]
                    lag_ = (h["time"].normalize() - pd.Timestamp(evd)).days
                    e.update({"cninfo_title": h["title"][:40], "cninfo_time": h["time"].strftime("%Y-%m-%d %H:%M"), "lag_days": lag_,
                              "known_at_T_close": bool(h["time"] <= pd.Timestamp(T) + pd.Timedelta(hours=15))})
                    e["verdict"] = "T 日收盘前结果已公告" if e["known_at_T_close"] else "T 日收盘时结果公告还没出（研究按发行日期已剔 = 提前知道）"
                else:
                    e["verdict"] = "巨潮 60 天内没查到结果公告"
            out.append(e)
            time.sleep(1.0)
        X = pd.DataFrame(out)
        X.to_csv(HERE / "c_examples.csv", index=False, encoding="utf-8-sig")
        say(X.groupby(["kind", "verdict"]).size().to_string())
        if "known_at_T_close" in X:
            for k, g in X.groupby("kind"):
                kk = g["known_at_T_close"].dropna()
                say("  %s：查到 %d 个，T 日收盘前已知 %d" % (k, len(kk), int(kk.sum())))
        say("\n逐条：")
        cols = [c for c in ["kind", "code", "T", "event_date_used", "src", "cninfo_time", "pub_vs_used_days", "lead_days", "lag_days", "verdict", "y_open"] if c in X.columns]
        say(X[cols].to_string(index=False))
    say("用时 %.0fs" % (time.time() - t0))
    (HERE / "results_c_events.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
