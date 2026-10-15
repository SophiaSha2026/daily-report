"""
阶段 1 第 3 步 (a)：生产特征「只做当日排名、不中性化」的版本，从 daily.parquet 原值复算。

只 import src/breakout 的 features / build / chips（读，不改）。步骤照抄 build.build：
共同起点 -> 换手率 -> 逐只 per_stock + 筹码 -> 砍预热 -> 薄日 -> 股东户数 -> 市场层 -> 砍市场预热 ->
① 当日排名，然后分叉：
    neut：② 中性化 -> ③ 交互 -> 5 日聚合          只用来核对复算和 train.parquet 一致
    rank：不中性化，交互项用（排名 - 0.5）代入同一公式 -> 5 日聚合   臂 prod_rank 用
日线只读到 2025-12-31（确认集不读），所以共同起点 / 当日排名的分母可能和 train.parquet 有细微差别，
核对时报一致率。

    python research/features/build_prodraw.py
产物 data/breakout/raw/exp/features/prod_rank.parquet（列名和 train.parquet 一样，带 __last/__mean/__slope）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build as B  # noqa: E402
import features as F  # noqa: E402


def agg5(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    g = panel.groupby("code", sort=False)
    out = {"code": panel["code"].to_numpy(), "date": panel["date"].to_numpy()}
    for c in feats:
        s = panel[c]
        out[f"{c}__last"] = s.to_numpy(np.float32)
        out[f"{c}__mean"] = g[c].transform(lambda x: x.rolling(B.WIN).mean()).to_numpy(np.float32)
        out[f"{c}__slope"] = g[c].transform(lambda x: (x - x.shift(B.WIN - 1)) / (B.WIN - 1)).to_numpy(np.float32)
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    daily = pd.read_parquet(C.ROOT / "data" / "breakout" / "daily.parquet")
    daily["date"] = daily["date"].astype(str).str[:10]
    daily = daily[daily["date"] <= "2025-12-31"]
    daily, start = B.trim_common_start(daily)
    daily = B.attach_turnover(daily, None)
    print("日线 %d 行，共同起点 %s" % (len(daily), start), flush=True)
    cols = [c for c in F.feature_columns() if c not in {n for *_, n in F.INTERACTIONS}]
    out = []
    for i, (code, g) in enumerate(daily.groupby("code", sort=False)):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < B.MIN_HIST:
            continue
        g = F.per_stock(g)
        g = F.add_chip_block(g)
        g = g.iloc[B.MIN_HIST:]
        keep = ["code", "date", "close", "outstanding_share", "float_mcap", "ret20", "ret60"] + \
            [c for c in cols if c in g.columns]
        g = g[[c for c in keep if c in g.columns]].copy()
        num = [c for c in g.columns if c not in ("code", "date")]
        g[num] = g[num].astype("float32")
        out.append(g)
        if (i + 1) % 1000 == 0:
            print("  逐只 %d 用时 %.1f 分钟" % (i + 1, (time.time() - t0) / 60), flush=True)
    del daily
    panel = pd.concat(out, ignore_index=True)
    del out
    n_by_date = panel.groupby("date")["code"].nunique()
    thin = n_by_date[n_by_date < 0.5 * n_by_date.median()].index
    panel = panel[~panel["date"].isin(set(thin))].reset_index(drop=True)
    panel["board"] = panel["code"].map(F.board_of)
    hp = C.ROOT / "data" / "breakout" / "holders.parquet"
    panel = F.holder_features(panel, pd.read_parquet(hp))
    panel = F.add_market(panel)
    pdates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= pdates[B.WARMUP]].reset_index(drop=True)
    cols = [c for c in cols if c in panel.columns]
    panel = F.cross_section(panel, cols)
    for c in cols:
        panel[c] = panel[c].astype("float32")
    print("排名完成 %d 行，%.1f 分钟" % (len(panel), (time.time() - t0) / 60), flush=True)
    feats = [c for c in F.feature_columns()]
    # ---- rank 版 ----
    rk = panel[["code", "date"] + cols].copy()
    cen = rk.copy()
    tgt = [c for c in cols if c not in F.MARKET_WIDE]
    cen[tgt] = cen[tgt] - 0.5
    for a, sa, b, sb, name in F.INTERACTIONS:
        rk[name] = np.minimum(sa * cen[a], sb * cen[b]).astype("float32")
    del cen
    r = agg5(rk, feats)
    del rk
    r.to_parquet(C.FE / "prod_rank.parquet", index=False)
    print("rank 版 %d 行 %d 列 %.1f 分钟" % (len(r), r.shape[1] - 2, (time.time() - t0) / 60), flush=True)
    del r
    # ---- neut 版（核对用） ----
    panel = F.neutralize(panel, cols)
    panel = F.add_interactions(panel)
    nt = agg5(panel[["code", "date"] + feats], feats)
    del panel
    tr = pd.read_parquet(C.K.DATA / "train.parquet", columns=["code", "date"] + list(nt.columns[2:]),
                         filters=[("date", "<", "2026-01-01")])
    tr["code"] = tr["code"].astype(str).str.zfill(6)
    m = tr.merge(nt, on=["code", "date"], suffixes=("", "_re"))
    lines = ["核对：train.parquet %d 行（< 2026），复算 %d 行，对上 %d 行" % (len(tr), len(nt), len(m))]
    bad = []
    for c in nt.columns[2:]:
        a, b = m[c].to_numpy(float), m[c + "_re"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        close = np.abs(a[ok] - b[ok]) < 1e-4
        nanmis = float((np.isfinite(a) != np.isfinite(b)).mean())
        bad.append((c, float(close.mean()), nanmis))
    bad.sort(key=lambda x: x[1])
    lines.append("逐列一致率（|差| < 1e-4）最低 8 列：" + "; ".join("%s %.4f(NaN不一致 %.4f)" % x for x in bad[:8]))
    lines.append("一致率中位 %.4f" % float(np.median([x[1] for x in bad])))
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_prodraw_check.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
