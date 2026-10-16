"""
第 1 步：unadj_2020.parquet -> pit 价格表 px_screen.parquet。

    code date open high low close（pit = P_u / (q_s/q_last)，不取整）volume amount turnover outstanding_share
    q（归一后的因子）close_u low_u（不复权，只给一字板判定核对用）

处理：
    code 补零、去重；按 backfill.clean_bars 清停牌占位行 / 零价行（生产同一个函数）；
    只留 DATA_START..KEEP_TO；q_last 取各票全表（含 2026 年）的最后一行，只读 q 不读价。
核对：
    和 leak 的 px.parquet（2023-01 起、q 同口径）在重叠行上 close_pit 逐位相同率；覆盖率；幸存者偏差的只数

    python research/screen/prices_screen.py            正式
    python research/screen/prices_screen.py --fake     拿 leak 的 px.parquet 150 只伪造（只为跑通下游）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

L: list[str] = []


def fake():
    cols = ["code", "date"] + [f"{p}_pit" for p in CS.PX] + list(CS.EXTRA) + ["q"]
    p = pd.read_parquet(CS.LEAK / "px.parquet", columns=cols)
    keep = sorted(p["code"].unique())[:150]
    p = p[p["code"].isin(keep)].rename(columns={f"{k}_pit": k for k in CS.PX})
    p["close_u"] = p["close"] * p["q"]
    p["low_u"] = p["low"] * p["q"]
    p.to_parquet(CS.SCR / "px_screen_fake.parquet", index=False)
    print("伪造版 ->", CS.SCR / "px_screen_fake.parquet", len(p), "行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.fake:
        return fake()
    import backfill as BF
    u = pd.read_parquet(CS.LEAK / "unadj_2020.parquet")
    u["code"] = u["code"].astype(str).str.zfill(6)
    u["date"] = u["date"].astype(str).str[:10]
    n0 = len(u)
    u = u.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    CS.say("unadj_2020 %d 行（去重后 %d）%d 只 %s..%s" % (n0, len(u), u["code"].nunique(), u["date"].min(), u["date"].max()), L)
    miss_f = CS.LEAK / "unadj_2020_missing.txt"
    miss = [x for x in miss_f.read_text(encoding="utf-8").splitlines() if x.strip()] if miss_f.exists() else []
    CS.say("fetch 报缺 %d 只：%s" % (len(miss), ", ".join(miss[:10])), L)
    # q_last 取全表最后一行（新浪因子表登了尚未到来的除权，最后一行 q 不一定是 1）
    q_last = u.groupby("code")["qfq_factor"].last()
    CS.say("各票最后一行 q != 1 的 %d 只；q 全表分位 50/90/99 %s" % (
        int((np.abs(q_last - 1) > 1e-9).sum()), np.round(u["qfq_factor"].quantile([0.5, 0.9, 0.99]).to_numpy(), 3)), L)
    u = u[(u["date"] >= CS.DATA_START) & (u["date"] <= CS.KEEP_TO)]
    # 停牌占位行 / 零价行：和生产 backfill 同一个函数（按 open/high/low/close/volume 判）
    bars = u.rename(columns={f"{k}_u": k for k in CS.PX})
    n1 = len(bars)
    bars = BF.clean_bars(bars)
    CS.say("clean_bars 去掉 %d 行（%.3f%%）：停牌占位 / 零价 / 零量" % (n1 - len(bars), 100 * (n1 - len(bars)) / max(n1, 1)), L)
    q = bars["qfq_factor"].to_numpy(float) / bars["code"].map(q_last).to_numpy(float)
    bad_q = int((q < 1 - 1e-9).sum())
    CS.say("归一后 q < 1 的行 %d" % bad_q, L)
    out = bars[["code", "date"]].copy()
    for p in CS.PX:
        out[p] = bars[p].to_numpy(float) / q
    for c in CS.EXTRA:
        out[c] = bars[c].to_numpy(float)
    out["q"] = q
    out["close_u"] = bars["close"].to_numpy(float)
    out["low_u"] = bars["low"].to_numpy(float)
    out = out.reset_index(drop=True)
    bad = out[(out["low"] <= 0) | ~np.isfinite(out[list(CS.PX)]).all(axis=1)]
    CS.say("pit 价 <= 0 或非有限的行 %d" % len(bad), L)
    # 换手率 / 股本口径核对：turnover ≈ volume / outstanding_share（新浪的 turnover 是百分数还是小数？）
    r = (out["volume"] / out["outstanding_share"].replace(0, np.nan))
    ratio = (out["turnover"] / r.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    CS.say("turnover / (volume/outstanding_share) 中位 %.4f（daily.parquet 是小数口径，应约 1；若约 100 是百分数）" % float(ratio.median()), L)
    if 50 < float(ratio.median()) < 200:
        out["turnover"] = out["turnover"] / 100.0
        CS.say("  -> turnover 是百分数，已除以 100", L)
    # 和 leak 的 px.parquet（同一口径 pit）在重叠行核对
    lk = pd.read_parquet(CS.LEAK / "px.parquet", columns=["code", "date", "close_pit", "volume", "turnover"])
    m = out.merge(lk, on=["code", "date"], how="inner", suffixes=("", "_lk"))
    rel = np.abs(m["close"] - m["close_pit"]) / m["close_pit"]
    CS.say("和 leak px.parquet 重叠 %d 行（leak 共 %d）：close_pit 相对差 <1e-6 的 %.4f%%，中位 %.2e 最大 %.2e；"
           "volume 相同 %.4f%%；turnover 相同(1e-6) %.4f%%" % (
               len(m), len(lk), 100 * float((rel < 1e-6).mean()), float(rel.median()), float(rel.max()),
               100 * float((m["volume"] == m["volume_lk"]).mean()),
               100 * float((np.abs(m["turnover"] - m["turnover_lk"]) < 1e-6).mean())), L)
    only_lk = lk.merge(out[["code", "date"]], on=["code", "date"], how="left", indicator=True)
    CS.say("leak 有而本表没有的行 %d（%d 只；clean_bars 或 fetch 缺）" % (
        int((only_lk["_merge"] == "left_only").sum()), only_lk.loc[only_lk["_merge"] == "left_only", "code"].nunique()), L)
    del lk, m, only_lk
    # 幸存者偏差：按日期数在表里的只数（都是今天还在的票）
    cnt = out.groupby("date")["code"].nunique()
    for d in ("2020-07-01", "2020-12-31", "2021-06-30", "2021-12-31", "2022-06-30", "2022-12-30", "2023-06-30",
              "2023-12-29", "2024-05-31", "2024-12-31", "2025-12-31"):
        c = cnt[cnt.index <= d]
        if len(c):
            CS.say("  %s 表里有 K 线的票 %d 只（只含 2026-10 仍在市的）" % (c.index[-1], int(c.iloc[-1])), L)
    out.to_parquet(CS.SCR / "px_screen.parquet", index=False)
    CS.say("-> %s %d 行 %d 只 %d 列" % (CS.SCR / "px_screen.parquet", len(out), out["code"].nunique(), out.shape[1]), L)
    (CS.HERE / "results_prices.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
