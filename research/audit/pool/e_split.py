"""审计 E：验证集切分 / 净化 / 标签窗口的数值核对。只读。

1 common_rc.train_mask（rulecore / gate / blind / polish rerank 共用的净化规则）逐月核：训练行最大日期 < 净化线、最大 end20 < 测试月初；
  第二个条件单独挡掉多少行、这些行的正样本率（教训 30 / train_slice 注释那类停牌票）。
2 gate / blind 自己写的 purge_cut 和 validate.purge_cut 是否逐月相同。
3 测试集里 y_open 为 NaN 的行（次日没有开盘价：停牌 / 表尾）在各研究里怎么处理：rulecore daily_hit 跳过、precision/gate hit=fillna(0)、
  evalkit.walk_forward 直接不进测试集。数一数前 10 里有多少这样的行，两种口径的命中差多少。
4 feats_by_month.json 缓存键（特征选择用的是哪段数据选的）。
5 2026 年数据的隔离：研究表最大日期、leak/train_snap 里 2026 年行（谁在读）、确认集表只被 confirm.py / forward 读。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src", ROOT / "research" / "rulecore", ROOT / "research" / "screen", ROOT / "research" / "stats",
          ROOT / "research" / "gate", ROOT / "research" / "blind", ROOT / "research" / "polish", ROOT / "research" / "precision"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
HERE = ROOT / "research/audit/pool"
L = []


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def main():
    import common_rc as C
    import validate as V
    import evalkit as K
    df = pd.read_parquet(EXP / "rulecore/feat.parquet", columns=["code", "date", "end20", "elig0", "y_open", "y_up", "nbar", "lret20", "board"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    say("== 1 train_mask 逐月核（feat.parquet，%d 行）==" % len(df))
    months = K.months_in(C.SEGS["screen"]) + K.months_in(C.SEGS["dev"])
    rows = []
    end = df["end20"].to_numpy().astype(str)
    dts = df["date"].to_numpy().astype(str)
    for m in months[::3] + [months[-1]]:
        cut = V.purge_cut(df, m)
        tm = C.train_mask(df, m)
        lo = C.month_shift(m, -12 * C.TRAIN_YEARS) + "-01"
        tr_d, tr_e = dts[tm], end[tm]
        only_date = (dts < cut) & (dts >= lo)
        leak_rows = only_date & ~(end < m + "-01")
        y = df["y_open"].to_numpy(float)
        rows.append({"month": m, "cut": cut, "train_rows": int(tm.sum()), "max_date": pd.Series(tr_d).max() if tm.any() else "-", "max_end20": pd.Series(tr_e).max() if tm.any() else "-",
                     "ok_date": bool((tr_d < cut).all()), "ok_end20": bool((tr_e < m + "-01").all()),
                     "purged_by_end20": int(leak_rows.sum()), "pos_rate_purged": float(np.nanmean(y[leak_rows])) if leak_rows.any() else np.nan,
                     "pos_rate_train": float(np.nanmean(y[tm])) if tm.any() else np.nan,
                     "max_end20_purged": pd.Series(end[leak_rows]).max() if leak_rows.any() else "-"})
    R = pd.DataFrame(rows)
    say(R.to_string(index=False))
    say("全部月份两个条件都成立：%s / %s" % (R["ok_date"].all(), R["ok_end20"].all()))
    # end20 的定义核：= 同一只票往后第 20 根的日期
    g = df.sort_values(["code", "date"]).groupby("code")["date"].shift(-20)
    chk = (g.fillna("9999-12-31").to_numpy().astype(str) == df.sort_values(["code", "date"])["end20"].to_numpy().astype(str)).mean()
    say("end20 == 该票往后第 20 根日期（末尾哨兵 9999-12-31）：一致率 %.6f" % chk)

    say("\n== 2 gate / blind 自写 purge_cut 对 validate.purge_cut ==")
    try:
        import walk_gt as WG
        import walk_bl as WB
        dates_u = np.array(sorted(df["date"].unique()))
        bad = []
        for m in months:
            a = V.purge_cut(df, m)
            b = WG.purge_cut(dates_u, m)
            c = WB.purge_cut(dates_u, m)
            if not (a == b == c):
                bad.append((m, a, b, c))
        say("%d 个月全部相同：%s；不同的 %s" % (len(months), not bad, bad[:5]))
    except Exception as e:  # noqa: BLE001
        say("导入 walk_gt / walk_bl 失败：%s" % e)

    say("\n== 3 测试集里 y_open 为 NaN 的行 ==")
    pool = pd.read_parquet(EXP / "polish/pool.parquet", columns=["code", "date", "pool", "lret20", "y_open", "y_up", "yizi1"])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool = pool[pool["pool"] & (pool["date"] >= "2022-01-01")].copy()
    pool = pool.sort_values(["date", "lret20"], ascending=[True, False], kind="mergesort")
    pool["rank"] = pool.groupby("date").cumcount() + 1
    top = pool[pool["rank"] <= 10]
    nan_rows = top[~np.isfinite(top["y_open"])]
    say("候选池 pool 行 %d，y_open NaN %d（%.4f%%）；前 10 行 %d，y_open NaN %d" % (len(pool), int((~np.isfinite(pool["y_open"])).sum()),
                                                                     100 * (~np.isfinite(pool["y_open"])).mean(), len(top), len(nan_rows)))
    if len(nan_rows):
        say("  NaN 行样例：%s" % nan_rows[["code", "date", "y_up"]].head(8).to_dict("records"))
    for seg, m in (("screen", top["date"] < "2024-06-01"), ("dev", top["date"] >= "2024-06-01")):
        t = top[m]
        h_skip = t.groupby("date")["y_open"].mean().mean()
        h_zero = t.assign(h=t["y_open"].fillna(0)).groupby("date")["h"].mean().mean()
        say("  %s：前 10 按天均值 跳过 NaN %.4f%%，NaN 记 0 %.4f%%（差 %.4fpp）" % (seg, 100 * h_skip, 100 * h_zero, 100 * (h_skip - h_zero)))
    # evalkit.walk_forward 的测试集过滤：isfinite(y_up) & isfinite(y_open)。候选池里 y_up 有 y_open 没有的行
    say("  y_up 有值而 y_open 没有：%d 行（evalkit 测试集不含它们，rulecore rank_table 含）" % int((np.isfinite(pool["y_up"]) & ~np.isfinite(pool["y_open"])).sum()))

    say("\n== 4 feats_by_month.json 缓存键 ==")
    f = EXP / "feats_by_month.json"
    if f.exists():
        cache = json.loads(f.read_text(encoding="utf-8"))
        keys = sorted(cache)
        plain = [k for k in keys if "|" not in k]
        say("键 %d 个；无后缀 %d（%s..%s）；后缀种类 %s" % (len(keys), len(plain), plain[0] if plain else "-", plain[-1] if plain else "-",
                                                sorted({k.split("|", 1)[1] for k in keys if "|" in k})[:10]))
        say("  列数中位 %d" % int(np.median([len(v) for v in cache.values()])))

    say("\n== 5 2026 年数据的隔离 ==")
    import subprocess
    out = subprocess.run(["git", "grep", "-l", "-E", "train_snap|daily_snap", "--", "research/*.py", "src/*.py"], cwd=ROOT, capture_output=True, text=True).stdout
    say("读 leak/train_snap 或 daily_snap（含 2026 年行的快照）的脚本：%s" % out.split())
    for nm in ("train_snap.parquet", "daily_snap.parquet"):
        p = EXP / "leak" / nm
        if p.exists():
            t = pd.read_parquet(p, columns=["date"])
            say("  %s 最大日期 %s，2026 年行 %d" % (nm, t["date"].max(), int((t["date"].astype(str) >= "2026-01-01").sum())))
    (HERE / "results_e_split.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
