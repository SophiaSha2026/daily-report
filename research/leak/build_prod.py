"""
第 2 步：在某一版价格上从零重建生产特征（照抄 src/breakout/build.build 的流程，只换输入输出，不改 src）。

    共同起点（按 daily_snap 全表的日期覆盖算，和生产同一个起点）-> attach_turnover -> 逐只 per_stock + 筹码
    -> 砍前 MIN_HIST 行 -> 丢横截面不足半数的日期 -> board -> 股东户数（holders_snap，按公告日）-> 市场层
    -> 砍市场预热 -> ① 当日排名 ② 中性化 ③ 交互 -> 5 日聚合 -> __last/__mean/__slope

标签不在这里算（两臂同用快照的 y_up 和 buyable 的 y_open）。逐只那步用 4 个进程。
产物 data/breakout/raw/exp/leak/prod_<ver>.parquet（code date + 87 个特征列，float32）。

    python research/leak/build_prod.py --ver fwd|pit|ctl [--px px_fake.parquet] [--sample 200] [--workers 4]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_leak as CL  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build as B  # noqa: E402
import features as F  # noqa: E402

RAW_COLS = [c for c in F.feature_columns() if c not in {n for *_, n in F.INTERACTIONS}]


def one(g: pd.DataFrame) -> pd.DataFrame | None:
    """模块级，进程池要 pickle。一只票：时序特征 + 筹码，砍预热。"""
    import warnings
    warnings.filterwarnings("ignore")
    g = g.sort_values("date").reset_index(drop=True)
    if len(g) < B.MIN_HIST:
        return None
    g = F.per_stock(g)
    g = F.add_chip_block(g)
    g = g.iloc[B.MIN_HIST:]
    keep = ["code", "date", "close", "outstanding_share", "float_mcap", "ret20", "ret60"] + \
        [c for c in RAW_COLS if c in g.columns]
    g = g[keep].copy()
    num = [c for c in g.columns if c not in ("code", "date")]
    g[num] = g[num].astype("float32")
    return g


def common_start() -> str:
    """共同起点按 daily_snap 全表（含 2026 年的 code/date 计数，不读价格）算，和生产 build 同一条规则。"""
    d = pd.read_parquet(CL.LEAK / "daily_snap.parquet", columns=["code", "date"])
    d["date"] = d["date"].astype(str).str[:10]
    _, start = B.trim_common_start(d)
    return start


def agg5(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    g = panel.groupby("code", sort=False)
    out = {"code": panel["code"].to_numpy(), "date": panel["date"].to_numpy(), "board": panel["board"].to_numpy()}
    for c in feats:
        s = panel[c]
        out[f"{c}__last"] = s.to_numpy(np.float32)
        out[f"{c}__mean"] = g[c].transform(lambda x: x.rolling(B.WIN).mean()).to_numpy(np.float32)
        out[f"{c}__slope"] = g[c].transform(lambda x: (x - x.shift(B.WIN - 1)) / (B.WIN - 1)).to_numpy(np.float32)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", required=True, choices=CL.VERS)
    ap.add_argument("--px", default="px.parquet")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    start = common_start()
    cols = ["code", "date"] + [f"{k}_{a.ver}" for k in CL.PX] + ["volume", "amount", "turnover", "outstanding_share"]
    daily = pd.read_parquet(CL.LEAK / a.px, columns=cols).rename(columns={f"{k}_{a.ver}": k for k in CL.PX})
    daily = daily[(daily["date"] >= start) & (daily["date"] <= CL.KEEP_TO)]
    if a.sample:
        keep = sorted(daily["code"].unique())[:a.sample]
        daily = daily[daily["code"].isin(keep)]
    daily = B.attach_turnover(daily, None)
    print("[%s] 日线 %d 行 %d 只，共同起点 %s，%.0fs" % (a.ver, len(daily), daily["code"].nunique(), start, time.time() - t0), flush=True)
    groups = [g for _, g in daily.groupby("code", sort=False)]
    del daily
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, groups, chunksize=20)):
            if r is not None:
                out.append(r)
            if (i + 1) % 1000 == 0:
                print("  逐只 %d/%d，%.1f 分钟" % (i + 1, len(groups), (time.time() - t0) / 60), flush=True)
    del groups
    panel = pd.concat(out, ignore_index=True)
    del out
    n_by_date = panel.groupby("date")["code"].nunique()
    thin = n_by_date[n_by_date < 0.5 * n_by_date.median()].index
    if len(thin):
        print("  丢掉 %d 个横截面不足半数的日期 %s..%s" % (len(thin), min(thin), max(thin)), flush=True)
        panel = panel[~panel["date"].isin(set(thin))].reset_index(drop=True)
    panel["board"] = panel["code"].map(F.board_of)
    panel = F.holder_features(panel, pd.read_parquet(CL.LEAK / "holders_snap.parquet"))
    panel = F.add_market(panel)
    pdates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= pdates[B.WARMUP]].reset_index(drop=True)
    cols = [c for c in RAW_COLS if c in panel.columns]
    panel = F.cross_section(panel, cols)
    panel = F.neutralize(panel, cols)
    panel = F.add_interactions(panel)
    for c in F.feature_columns():
        panel[c] = panel[c].astype("float32")
    print("  三层变换完成 %d 行，%.1f 分钟" % (len(panel), (time.time() - t0) / 60), flush=True)
    feats = list(F.feature_columns())
    r = agg5(panel[["code", "date", "board"] + feats], feats)
    del panel
    dst = CL.LEAK / (a.out or f"prod_{a.ver}.parquet")
    r.to_parquet(dst, index=False)
    print("[%s] -> %s %d 行 %d 列，%.1f 分钟" % (a.ver, dst, len(r), r.shape[1] - 3, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
