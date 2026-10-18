"""
审计第 5~6 项：报告里反复引用的数，从原始表用最小实现独立重算；生产侧印出来的常量对账。

    PYTHONIOENCODING=utf-8 python research/audit/stats/reconcile.py  -> results_reconcile.txt
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data" / "breakout" / "raw" / "exp"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_reconcile.txt"
lines: list[str] = []
SEGS = {"screen": ("2022-01", "2024-05"), "dev": ("2024-06", "2025-12")}


def say(s: str = "") -> None:
    lines.append(s)
    print(s, flush=True)


def progress(s: str) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), s))


def seg(dates: pd.Series) -> np.ndarray:
    return np.where(dates.str[:7] < "2024-06", "screen", "dev")


def pooled(df, y="y_open"):
    v = df[y].to_numpy(float)
    v = v[np.isfinite(v)]
    return 100 * v.mean(), len(v)


def daymean(df, y="y_open"):
    return 100 * df.groupby("date")[y].mean().mean()


def main():
    t0 = time.time()
    say("== 5.1 rulecore_v1 名次表（RULE_PERF 前两行、报告第 0 节的来源）==")
    tops = {a: pd.read_parquet(EXP / "rulecore_v1" / f"top_{a}.parquet") for a in ("ret20", "base", "rc", "pc")}
    for a, t in tops.items():
        t["seg"] = seg(t["date"])
        t10 = t[t["rank"] <= 10]
        parts = []
        for s in ("screen", "dev"):
            x = t10[t10["seg"] == s]
            pm, n = pooled(x)
            parts.append("%s 池化 %.2f%%（n=%d，天 %d，按天均值 %.2f%%）" % (s, pm, n, x["date"].nunique(), daymean(x)))
        pm, n = pooled(t10)
        say("   %-6s %s | 合并 池化 %.2f%% 按天均值 %.2f%%（n=%d，天 %d）" % (a, "  ".join(parts), pm, daymean(t10), n, t10["date"].nunique()))
        # 每天前 10 是否都满 10 只、标签是否都有
        cnt = t10.groupby("date").size()
        say("          每天前 10 行数：min %d max %d；y_open 缺失 %d 行" % (cnt.min(), cnt.max(), int((~np.isfinite(t10["y_open"])).sum())))
    # 按天配对差（ret20 − base）与 NW t，对报告第 0 节 +3.76 (4.14) / +3.90 (2.41) / 合并 4.50
    sys.path.insert(0, str(ROOT / "src" / "breakout"))
    import evalkit as K
    da = tops["ret20"][tops["ret20"]["rank"] <= 10].groupby("date")["y_open"].mean()
    db = tops["base"][tops["base"]["rank"] <= 10].groupby("date")["y_open"].mean()
    d = (da - db).dropna()
    sg = seg(pd.Series(d.index))
    for s in ("screen", "dev"):
        x = d[sg == s]
        say("   ret20 − base 按天差 %s：%+.2fpp（NW t %.2f，%d 天）" % (s, 100 * x.mean(), K.nw_t(x.to_numpy()), len(x)))
    say("   ret20 − base 按天差 合并：%+.2fpp（NW t %.2f，块自举 p %.3f，%d 天）" % (100 * d.mean(), K.nw_t(d.to_numpy()), K.block_boot(d.to_numpy())["p"], len(d)))
    # 和登记簿里落的 diff 序列对一下
    dd = pd.read_parquet(EXP / "rulecore_v1" / "diff_ret20_10_y_open.parquet")
    dd = pd.Series(dd["d"].to_numpy(), index=dd["date"].astype(str))
    both = d.index.intersection(dd.index)
    say("   登记簿 diff_ret20 序列：%d 天，和重算逐日最大绝对差 %.2e" % (len(dd), float(np.abs(d.reindex(both) - dd.reindex(both)).max())))

    say()
    say("== 5.2 rulecore_confirm（RULE_PERF 第三行、RULE_FACTS 的来源）==")
    ct = {a: pd.read_parquet(EXP / "rulecore_confirm" / f"top_{a}.parquet") for a in ("ret20", "base")}
    for a, t in ct.items():
        t10 = t[t["rank"] <= 10]
        pm, n = pooled(t10)
        say("   %-6s 前 10 池化 %.2f%% 按天均值 %.2f%%（n=%d，天 %d，%s..%s）" % (a, pm, daymean(t10), n, t10["date"].nunique(), t10["date"].min(), t10["date"].max()))
    res = json.loads((EXP / "rulecore_confirm" / "results.json").read_text(encoding="utf-8"))
    say("   results.json：pool_base %.4f  ret20 hit10 %.4f diff10 %.4f t10 %.3f  facts %s" % (
        res["pool_base"], res["ret20"]["hit10"], res["ret20"]["diff10"], res["ret20"]["t10"], json.dumps(res["ret20"]["facts"])))
    # 三句事实独立重算：feat_c.parquet 里有没有 ret20 / r20_open / c20 / dd20
    fc_path = EXP / "rulecore" / "feat_c.parquet"
    if fc_path.exists():
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(fc_path).schema_arrow.names
        say("   feat_c.parquet 列：%s" % ", ".join(cols[:40]))
        need = [c for c in ("date", "code", "ret20", "r20_open", "c20", "dd20", "y_open", "elig0", "close", "nbar") if c in cols]
        fc = pd.read_parquet(fc_path, columns=need)
        fc["code"] = fc["code"].astype(str).str.zfill(6)
        t10 = ct["ret20"][ct["ret20"]["rank"] <= 10].copy()
        t10["code"] = t10["code"].astype(str).str.zfill(6)
        m = t10.merge(fc, on=["date", "code"], how="left", suffixes=("", "_f"))
        say("   确认集 ret20 前 10 合并特征表：%d 行，ret20 缺 %d" % (len(m), int(m["ret20"].isna().sum()) if "ret20" in m else -1))
        f = {}
        if "ret20" in m:
            f["ret20_med"] = 100 * m["ret20"].median()
        if "r20_open" in m:
            f["r20_med"] = 100 * m["r20_open"].median()
        if "c20" in m:
            f["c20_med"] = 100 * m["c20"].median()
        if "dd20" in m:
            f["dd20_med"] = 100 * m["dd20"].median()
            f["dd20_le20"] = 100 * (m["dd20"] <= -0.2).mean()
        f["hit"] = 100 * m["y_open"].mean()
        say("   重算三句事实：%s" % "  ".join("%s %.1f" % kv for kv in f.items()))
        # 全市场（候选池）对照：c20 中位 −1.8、基准 2.6
        if "elig0" in fc.columns:
            # 候选池口径（elig0 + 事件剔除）在 confirm 里用的是 all_ret20 的覆盖行；这里用 all_ret20 的 (date, code) 当候选池
            al = pd.read_parquet(EXP / "rulecore_confirm" / "all_ret20.parquet", columns=["date", "code"])
            al["code"] = al["code"].astype(str).str.zfill(6)
            pool = al.merge(fc, on=["date", "code"], how="left")
            say("   候选池（all_ret20 覆盖行 %d）：y_open 均值 %.2f%%  c20 中位 %.1f%%  r20_open 中位 %.1f%%  dd20 中位 %.1f%%" % (
                len(pool), 100 * pool["y_open"].mean(), 100 * pool["c20"].median() if "c20" in pool else float("nan"),
                100 * pool["r20_open"].median() if "r20_open" in pool else float("nan"), 100 * pool["dd20"].median() if "dd20" in pool else float("nan")))
    else:
        say("   feat_c.parquet 不存在，三句事实只能对 results.json")

    say()
    say("== 5.3 precision top50.parquet（B(1) / B(2) / B(10) 的来源）==")
    t50 = pd.read_parquet(ROOT / "research" / "precision" / "top50.parquet", columns=["date", "code", "rank_ret20", "hit", "y_open", "seg", "lret20", "ret20"])
    for s in ("screen", "dev"):
        x = t50[t50["seg"] == s]
        D = x["date"].nunique()
        row = []
        for k in (1, 2, 3, 5, 10):
            xk = x[x["rank_ret20"] <= k]
            row.append("B(%d) %.2f%%" % (k, 100 * xk["hit"].sum() / (k * D)))
        say("   %-6s D=%d  %s  | hit 与 y_open 不一致行 %d（y_open NaN %d）" % (
            s, D, "  ".join(row), int((x["hit"] != x["y_open"].fillna(0)).sum()), int(x["y_open"].isna().sum())))
    # ltt_an 的双侧下界 / 单侧下界对 results_ltt.txt：用同一块自举复算 B(1)/B(2) 开发集
    sys.path.insert(0, str(ROOT / "research" / "precision"))
    sys.path.insert(0, str(ROOT / "research" / "polish"))
    import common_pr as C
    for s in ("screen", "dev"):
        Bz = C.Baseline(C.load(s))
        for k in (1, 2, 10):
            h = Bz.hits_k(k)
            n = np.full(Bz.D, float(k))
            lo, hi = C.boot_ratio(h, n)
            P = h.sum() / n.sum()
            say("   %-6s B(%d) P %.2f%% 块自举 95%% [%.1f, %.1f]（报告 precision 表）" % (s, k, 100 * P, 100 * lo, 100 * hi))

    say()
    say("== 5.4 pool.parquet：候选池基准 1.62 / 4.32、ret20 前 10 名次和 top50 的一致性 ==")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(EXP / "polish" / "pool.parquet")
    say("   pool.parquet %d 行 %d 列" % (pf.metadata.num_rows, pf.metadata.num_columns))
    pool = pd.read_parquet(EXP / "polish" / "pool.parquet", columns=["date", "code", "y_open", "pool", "lret20", "elig0"])
    pool = pool[pool["pool"] & (pool["date"] >= "2022-01-01") & (pool["date"] <= "2025-12-31")]
    pool["seg"] = seg(pool["date"])
    for s in ("screen", "dev"):
        x = pool[pool["seg"] == s]
        yo = x["y_open"].to_numpy(float)
        say("   %-6s 候选池行 %d（天 %d，每天 %.0f 只）  y_open 均值（池化）%.2f%%  按天均值再平均 %.2f%%  y_open NaN %d" % (
            s, len(x), x["date"].nunique(), len(x) / x["date"].nunique(), 100 * np.nanmean(yo), daymean(x), int(np.isnan(yo).sum())))
    # 独立重排 ret20 前 10（按 lret20 降序，mergesort 稳定）和 top50 的 rank_ret20 对一下
    pool = pool.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    pool["rk"] = pool.groupby("date").cumcount() + 1
    p10 = pool[pool["rk"] <= 10]
    m = p10.merge(t50[["date", "code", "rank_ret20"]], on=["date", "code"], how="left")
    say("   独立重排 ret20 前 10：%d 行，和 top50 名次一致 %.4f，top50 里缺 %d" % (len(m), (m["rk"] == m["rank_ret20"]).mean(), int(m["rank_ret20"].isna().sum())))
    for s in ("screen", "dev"):
        x = p10[p10["seg"] == s]
        say("   %-6s pool 重排前 10 池化 %.2f%%（rulecore 报告 11.34 / 16.54 用的是 rulecore 自己的表，候选池同一定义但价格 / 标签来源不同）" % (s, 100 * np.nanmean(x["y_open"].to_numpy(float))))
    # 并列：lret20 并列 / 第 10 名并列时的名次是否稳定
    ties = pool[pool["rk"].between(9, 11)].groupby("date")["lret20"].nunique()
    say("   第 9~11 名 lret20 有并列的天数 %d / %d" % (int((ties < 3).sum()), len(ties)))
    progress("reconcile 第 5 项（原始表重算）算完")

    say()
    say("== 6 生产侧印出来的数 ==")
    src = (ROOT / "src" / "breakout" / "export.py").read_text(encoding="utf-8")
    rp = re.search(r"RULE_PERF = \[(.*?)\]\n", src, re.S).group(1)
    say("   export.RULE_PERF：%s" % " ".join(rp.split()))
    rf = re.search(r"RULE_FACTS = \{(.*?)\}", src, re.S).group(1)
    say("   export.RULE_FACTS：%s" % " ".join(rf.split()))
    for k_, v_ in (("BASE", r"^BASE = ([0-9.]+)"), ("CLOSE_PERF", r"CLOSE_PERF = \((.*?)\)"), ("CAP_PERF", r"CAP_PERF = \((.*?)\)"), ("NONCAP_PERF", r"NONCAP_PERF = \((.*?)\)")):
        mm = re.search(v_, src, re.M)
        say("   export.%s = %s" % (k_, mm.group(1) if mm else "?"))
    sp = re.search(r"STREAK_PERF = \[(.*?)\]\n", src, re.S).group(1)
    say("   export.STREAK_PERF：%s" % " ".join(l.strip() for l in sp.splitlines() if l.strip() and not l.strip().startswith("#")))
    # window_grid.json 的 W5 行（学习卡片 _validation 读的）
    wg = ROOT / "out_breakout" / "window_grid.json"
    if wg.exists():
        g = json.loads(wg.read_text(encoding="utf-8"))
        say("   window_grid.json：base %s days %s board_adj %s" % (g.get("base"), g.get("days"), g.get("board_adj")))
        for r in g.get("grid", []):
            if r.get("kind") in ("W5", "W5close"):
                say("      %s %-14s hit %.4f n %s lift %s" % (r.get("kind"), r.get("label"), float(r.get("hit", float("nan"))), r.get("n"), r.get("lift")))
    else:
        say("   window_grid.json 不存在")
    # truth.json：Wilson 区间独立复算、paired 的 se 口径
    tj = ROOT / "state" / "breakout" / "truth.json"
    if tj.exists():
        T = json.loads(tj.read_text(encoding="utf-8"))

        def wilson_ind(k, n, z=1.96):
            if n == 0:
                return (0.0, 1.0)
            p = k / n
            den = 1 + z * z / n
            c = (p + z * z / (2 * n)) / den
            h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            return (max(0, c - h), min(1, c + h))
        bad = 0
        fin = [L for L in T.get("lists", []) if L.get("n_final")]
        for L in fin:
            lo, hi = wilson_ind(L["hits_final"], L["n_final"])
            if abs(lo - L["ci_lo"]) > 1e-9 or abs(hi - L["ci_hi"]) > 1e-9:
                bad += 1
        say("   truth.json：%d 份清单，%d 份已有定论，Wilson 区间独立复算不一致 %d 份" % (len(T.get("lists", [])), len(fin), bad))
        n = sum(L["n_final"] for L in fin)
        h = sum(L["hits_final"] for L in fin)
        no = sum(L.get("n_open") or 0 for L in fin)
        ho = sum(L.get("hits_open") or 0 for L in fin)
        bs = [L["base_final"] for L in fin if L.get("base_final") is not None]
        bo = [L["base_open_final"] for L in fin if L.get("base_open_final") is not None]
        say("   实发已定论：收盘口径 %d/%d = %.1f%%（同期全市场 %.2f%%）  买得到口径 %d/%d = %.1f%%（全市场 %.2f%%）" % (
            h, n, 100 * h / max(n, 1), 100 * np.mean(bs) if bs else float("nan"), ho, no, 100 * ho / max(no, 1), 100 * np.mean(bo) if bo else float("nan")))
        say("   学习卡片 _live 印的是 hits_final / n_final（收盘口径）+ base_final 均值；邮件 2026-10-07 起成绩常量是买得到口径")
        sh = (T.get("shadow") or {}).get("model") or {}
        say("   影子配对：%s" % json.dumps(sh.get("paired"), ensure_ascii=False))
        by = T.get("by", {})
        if by.get("all"):
            say("   by.all：%s（用的是 hit = 收盘口径）" % json.dumps(by["all"]))
    else:
        say("   truth.json 不存在")
    bh = ROOT / "out_breakout" / "board_hit.json"
    if bh.exists():
        b = json.loads(bh.read_text(encoding="utf-8"))
        say("   board_hit.json：overall %s label %s arm_matches %s" % (b.get("overall"), b.get("label"), b.get("arm_matches")))
    say()
    say("用时 %.0fs" % (time.time() - t0))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress("reconcile 第 6 项（生产常量 / truth）算完")
    print("->", OUT)


if __name__ == "__main__":
    main()
