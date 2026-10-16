"""
第 2 步：在 pit 价格表上从零重建生产特征（复用 research/leak/build_prod.py 的 one / agg5，不改它）。

和 build_prod 的差别只有输入输出和两条起点规则：
    · 共同起点不用 build.trim_common_start（COVER_MIN=0.90 是为「回填没对齐」的薄片段写的：生产表 2023-01~05
      只有分过红的几十只）。本表每只票从上市或 2020-07-01 起都有 K 线，2020-07 的只数少是真的 IPO 增长
      （2020-07 约 3900 只、2025-12 约 5400 只，比值 0.72 < 0.90，照抄那条规则会把起点砍到 2023 年）。
      所以起点 = DATA_START，横截面在当日全部在市（且今天仍在）的票里排。
    · 其余照抄：attach_turnover -> 逐只 per_stock + 筹码 -> 砍前 MIN_HIST 行 -> 丢横截面不足半数的日期
      -> board -> 股东户数（holders.parquet，公告日 <= KEEP_TO）-> 市场层 -> 砍 WARMUP -> ① ② ③ -> 5 日聚合
产物 data/breakout/raw/exp/screen/prod_screen.parquet（code date board + 102 列，float32）。

    python research/screen/build_screen.py [--px px_screen_fake.parquet --out prod_screen_fake.parquet] [--workers 4]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import pandas as pd  # noqa: E402

import build as B  # noqa: E402
import build_prod as BP  # noqa: E402  research/leak：one（逐只特征+筹码）、agg5、RAW_COLS
import features as F  # noqa: E402


def holders_upto(keep_to: str) -> pd.DataFrame:
    """holders.parquet 只留公告日 <= keep_to 的行（确认集一行不读）。公告日缺的按报告期 + 120 天（features 同口径）。"""
    h = pd.read_parquet(CS.ROOT / "data" / "breakout" / "holders.parquet")
    ann = pd.to_datetime(h["公告日期"], errors="coerce")
    per = pd.to_datetime(h["报告期"].astype(str), format="%Y%m%d", errors="coerce")
    avail = ann.where(ann.notna() & (ann > per), per + pd.Timedelta(days=120))
    return h[avail <= pd.Timestamp(keep_to)].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", default="px_screen.parquet")
    ap.add_argument("--out", default="prod_screen.parquet")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    daily = CS.read_px(a.px)
    daily = daily[(daily["date"] >= CS.DATA_START) & (daily["date"] <= CS.KEEP_TO)]
    if a.sample:
        keep = sorted(daily["code"].unique())[:a.sample]
        daily = daily[daily["code"].isin(keep)]
    daily = B.attach_turnover(daily, None)
    print("日线 %d 行 %d 只 %s..%s，%.0fs" % (len(daily), daily["code"].nunique(), daily["date"].min(), daily["date"].max(),
                                           time.time() - t0), flush=True)
    groups = [g for _, g in daily.groupby("code", sort=False)]
    del daily
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(BP.one, groups, chunksize=20)):
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
    print("  面板 %d 行，首日 %s（%d 只）末日 %s（%d 只），%.1f 分钟" % (
        len(panel), n_by_date.index.min(), int(n_by_date.iloc[0]), n_by_date.index.max(), int(n_by_date.iloc[-1]),
        (time.time() - t0) / 60), flush=True)
    panel["board"] = panel["code"].map(F.board_of)
    hold = holders_upto(CS.KEEP_TO)
    print("  股东户数 %d 行（公告日 <= %s），最早报告期 %s" % (len(hold), CS.KEEP_TO, hold["报告期"].min()), flush=True)
    panel = F.holder_features(panel, hold)
    del hold
    panel = F.add_market(panel)
    pdates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= pdates[B.WARMUP]].reset_index(drop=True)
    cols = [c for c in BP.RAW_COLS if c in panel.columns]
    panel = F.cross_section(panel, cols)
    panel = F.neutralize(panel, cols)
    panel = F.add_interactions(panel)
    for c in F.feature_columns():
        panel[c] = panel[c].astype("float32")
    print("  三层变换完成 %d 行，%.1f 分钟" % (len(panel), (time.time() - t0) / 60), flush=True)
    feats = list(F.feature_columns())
    r = BP.agg5(panel[["code", "date", "board"] + feats], feats)
    del panel
    dst = CS.SCR / a.out
    r.to_parquet(dst, index=False)
    print("-> %s %d 行 %d 列，%.1f 分钟" % (dst, len(r), r.shape[1] - 3, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
