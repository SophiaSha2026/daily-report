"""
第 1 步：把新浪不复权价 + 前复权因子（fetch_unadj.py 的 unadj.parquet）和 daily_snap.parquet 对齐，
造三版价格（见 common_leak 的说明），落 data/breakout/raw/exp/leak/px.parquet：
    code date open_fwd .. close_fwd open_pit .. close_pit open_ctl .. close_ctl volume amount turnover outstanding_share
    q（截至拉取日的前复权因子，q>=1）logq = log q  k_ctl（ctl 版的常数）
并做核对：
    fwd == daily_snap 四个价的逐位一致率（两种四舍五入）；缺的票 / 行；q 的分布；最后一行 q 是否 = 1

    python research/leak/prices.py            正式
    python research/leak/prices.py --fake     没有 unadj.parquet 时拿 daily_snap 伪造三版（只为跑通下游脚本）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

L = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    daily = pd.read_parquet(CL.LEAK / "daily_snap.parquet")
    daily["date"] = daily["date"].astype(str).str[:10]
    daily["code"] = daily["code"].astype(str).str.zfill(6)
    daily = daily[daily["date"] <= CL.KEEP_TO].sort_values(["code", "date"]).reset_index(drop=True)
    CL.say("daily_snap（<= %s）%d 行 %d 只" % (CL.KEEP_TO, len(daily), daily["code"].nunique()), L)
    if a.fake:
        out = daily[["code", "date"]].copy()
        for v in CL.VERS:
            for p in CL.PX:
                out[f"{p}_{v}"] = daily[p].to_numpy(float)
        for c in ("volume", "amount", "turnover", "outstanding_share"):
            out[c] = daily[c].to_numpy(float)
        out["q"] = 1.0
        out["logq"] = 0.0
        out["k_ctl"] = 1.0
        out.to_parquet(CL.LEAK / "px_fake.parquet", index=False)
        print("伪造版 ->", CL.LEAK / "px_fake.parquet")
        return

    u = pd.read_parquet(CL.LEAK / "unadj.parquet")
    u["code"] = u["code"].astype(str).str.zfill(6)
    u["date"] = u["date"].astype(str).str[:10]
    u = u.drop_duplicates(["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    # q_last 要取全表（含 2026 年）的最后一行：daily.parquet 的复权是相对拉取当天的。只读 q，不读价
    q_last_all = u.groupby("code")["qfq_factor"].last()
    u = u[u["date"] <= CL.KEEP_TO]
    miss_file = CL.LEAK / "missing.txt"
    miss = [x for x in miss_file.read_text(encoding="utf-8").splitlines() if x.strip()] if miss_file.exists() else []
    CL.say("unadj %d 行 %d 只；fetch 报缺 %d 只" % (len(u), u["code"].nunique(), len(miss)), L)
    q_last = u.groupby("code")["qfq_factor"].last()
    CL.say("各票最后一行 q：=1 的占 %.1f%%，中位 %.4f，最大 %.3f；q 全表分位 50/90/99：%s" % (
        100 * float((np.abs(q_last - 1) < 1e-9).mean()), float(q_last.median()), float(q_last.max()),
        np.round(u["qfq_factor"].quantile([0.5, 0.9, 0.99]).to_numpy(), 3)), L)
    bad_q = u[~(u["qfq_factor"] >= 1 - 1e-9)]
    CL.say("q < 1 的行 %d（%d 只）" % (len(bad_q), bad_q["code"].nunique()), L)

    # ---- 覆盖率：按票看 daily_snap 的行在 unadj 里有多少 ----
    key_d = daily[["code", "date"]].assign(_d=1)
    key_u = u[["code", "date"]].assign(_u=1)
    kk = key_d.merge(key_u, on=["code", "date"], how="outer")
    cov = kk.groupby("code").agg(nd=("_d", "sum"), nu=("_u", "sum"), both=("_d", lambda s: 0))
    both = kk[kk["_d"].notna() & kk["_u"].notna()].groupby("code").size()
    cov["both"] = both.reindex(cov.index).fillna(0)
    cov["nd"] = cov["nd"].fillna(0)
    cov["cover"] = cov["both"] / cov["nd"].replace(0, np.nan)
    full = cov[(cov["nd"] > 0) & (cov["cover"] >= 0.99)].index
    partial = cov[(cov["nd"] > 0) & (cov["cover"] < 0.99)]
    CL.say("daily_snap 的 %d 只里，unadj 覆盖 >= 99%% 行的 %d 只；不足的 %d 只（其中完全没有 %d 只）：%s" % (
        cov["nd"].gt(0).sum(), len(full), len(partial), int((partial["both"] == 0).sum()),
        ", ".join("%s %.0f%%" % (c, 100 * r) for c, r in partial["cover"].fillna(0).head(15).items())), L)
    only_u = int((kk["_d"].isna() & kk["_u"].notna()).sum())
    CL.say("unadj 有而 daily_snap 没有的行 %d（不用）" % only_u, L)

    # ---- 对齐，只留覆盖齐的票、两边都有的行 ----
    m = daily[daily["code"].isin(full)].merge(u, on=["code", "date"], how="inner")
    m = m.sort_values(["code", "date"]).reset_index(drop=True)
    dropped_rows = int(cov.loc[full, "nd"].sum() - len(m))
    CL.say("对齐后 %d 行 %d 只（覆盖齐的票里丢了 %d 行 = daily 有 unadj 没有的日子）" % (len(m), m["code"].nunique(), dropped_rows), L)
    # 新浪因子表已登了尚未到来的除权日，最新一行的 q 不一定是 1（主会话实测 000726 1.017）；daily.parquet 的
    # 复权是相对拉取当天的，所以 daily = round(P_u / (q_s / q_last), 2)。q 一律按 q_s / q_last 用，比值特征无影响
    q = m["qfq_factor"].to_numpy(float) / m["code"].map(q_last_all).to_numpy(float)
    CL.say("q 归一到各票最后一行：q_last != 1 的票 %d 只，归一后 q<1 的行 %d" % (
        int((np.abs(q_last_all - 1) > 1e-9).sum()), int((q < 1 - 1e-9).sum())), L)
    # ---- 核对 fwd == daily_snap ----
    for name, rnd in (("半进位", CL.round2), ("numpy 银行家", lambda x: np.round(x, 2))):
        parts = []
        for p in CL.PX:
            f = rnd(m[f"{p}_u"].to_numpy(float) / q)
            d = m[p].to_numpy(float)
            parts.append("%s %.4f%%" % (p, 100 * float((np.abs(f - d) < 1e-6).mean())))
        CL.say("fwd=round(P_u/q) 和 daily_snap 逐位相同率（%s）：%s" % (name, "  ".join(parts)), L)
    f = CL.round2(m["close_u"].to_numpy(float) / q)
    d = m["close"].to_numpy(float)
    diff = ~(np.abs(f - d) < 1e-6)
    if diff.any():
        bad = m.loc[diff, ["code", "date"]]
        by_code = bad.groupby("code").size().sort_values(ascending=False)
        CL.say("close 对不上的 %d 行，涉及 %d 只；最多的 8 只：%s；差 1 分以内的占 %.1f%%" % (
            int(diff.sum()), len(by_code), ", ".join("%s %d" % x for x in by_code.head(8).items()),
            100 * float((np.abs(f[diff] - d[diff]) <= 0.0100001).mean())), L)
        # 对不上的行是整段（q 过期，daily 没重拉）还是零星
        CL.say("  对不上的票里，对不上的行占该票行数的中位 %.1f%%" % (
            100 * float((by_code / m.groupby("code").size().reindex(by_code.index)).median())), L)

    # ---- 三版价格 ----
    out = m[["code", "date"]].copy()
    for p in CL.PX:
        pu = m[f"{p}_u"].to_numpy(float)
        out[f"{p}_fwd"] = CL.round2(pu / q)
        out[f"{p}_pit"] = pu / q
    # ctl：每只票一个常数 k，取各票 2025-01-02 前后最近一行的 q 跨票随机置换
    mid = pd.Series(q, index=m.index).where(m["date"] <= "2025-01-02").groupby(m["code"]).last()
    mid = mid.reindex(sorted(m["code"].unique())).fillna(1.0)
    rng = np.random.default_rng(CL.SEED_CTL)
    k = pd.Series(rng.permutation(mid.to_numpy()), index=mid.index)
    kk_ = m["code"].map(k).to_numpy(float)
    for p in CL.PX:
        pu = m[f"{p}_u"].to_numpy(float)
        out[f"{p}_ctl"] = CL.round2(pu / kk_) / q
    for c in ("volume", "amount", "turnover", "outstanding_share"):
        out[c] = m[c].to_numpy(float)
    out["q"] = q
    out["logq"] = np.log(q)
    out["k_ctl"] = kk_
    CL.say("ctl 的 k：置换前各票 2025-01-02 的 q 分位 50/75/90/99：%s（=1 的占 %.1f%%）" % (
        np.round(mid.quantile([0.5, 0.75, 0.9, 0.99]).to_numpy(), 3), 100 * float((np.abs(mid - 1) < 1e-9).mean())), L)
    # 取整颗粒：fwd 相对 pit 的相对误差量级，按 q 分档
    rel = np.abs(out["close_fwd"] - out["close_pit"]) / out["close_pit"]
    grp = np.select([q < 1.015, q < 1.03, q < 1.10, q >= 1.10], ["q<1.015", "1.015~1.03", "1.03~1.10", ">=1.10"], "?")
    s = pd.Series(rel).groupby(grp).agg(["size", "mean", "max"])
    CL.say("close 取整相对误差 |fwd-pit|/pit 按 q 分档：%s" % "  ".join(
        "%s n=%d 均 %.2e 最大 %.2e" % (i, r_["size"], r_["mean"], r_["max"]) for i, r_ in s.iterrows()), L)
    bad = out[(out["low_pit"] <= 0) | ~np.isfinite(out[[f"{p}_pit" for p in CL.PX]]).all(axis=1)]
    CL.say("pit 价 <= 0 或非有限的行 %d" % len(bad), L)
    out.to_parquet(CL.LEAK / "px.parquet", index=False)
    CL.say("-> %s %d 行 %d 列" % (CL.LEAK / "px.parquet", len(out), out.shape[1]), L)
    (CL.HERE / "results_prices.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
