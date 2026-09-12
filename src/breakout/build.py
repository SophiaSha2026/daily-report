"""
把回填下来的三份原始数据组装成训练表。

    python src/breakout/build.py            全量
    python src/breakout/build.py --sample 300   先拿 300 只跑通流程

顺序（不能换）
--------------
    日线 + 股本 -> 精确换手率
                -> 逐只时序特征 + 筹码（路径依赖，必须逐只）
                -> 股东人数按公告日对齐
                -> 全市场层（相对强度、市场环境）
                -> ① 横截面百分位  -> ② 中性化  -> ③ 交互项
                -> 5 日窗口聚合
                -> 拼标签（来自 label.py，物理隔离）
                -> data/breakout/train.parquet

标签是**最后**才拼上去的，而且来自另一个模块。这样特征侧的任何一步
都不可能碰到未来信息 —— 它压根拿不到标签列。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = ROOT / "data" / "breakout"

import features as F          # noqa: E402
from label import label_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build")

WIN = 5          # 用户要的「起涨前 5 个交易日」


def attach_turnover(daily: pd.DataFrame,
                    shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """准备 turnover（小数形式）和 float_mcap。

    2026-09-12 换源之后这里大幅简化：新浪的日线接口**自带**
    turnover（= volume / outstanding_share，小数）和 outstanding_share
    （逐日流通股本）。原先那套「拉 gbjg_em 的股本变更记录 + 按变更日期
    阶梯前向填充」整个不需要了，而且新浪这份是逐日精确值，比阶梯填充更准。

    腾讯那条路（只有 OHLCV、没有换手率）保留为兜底：它的 volume 单位是
    **手**，新浪是**股**，差 100 倍。所以判断依据是列在不在，不是值的大小。
    """
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)

    if "turnover" in daily.columns and daily["turnover"].notna().any():
        daily["turn_est"] = ~np.isfinite(daily["turnover"])
        if "outstanding_share" in daily.columns:
            daily["float_mcap"] = daily["outstanding_share"] * daily["close"]
        else:
            daily["float_mcap"] = np.nan
    else:
        # --- 兜底：腾讯源，没有换手率，用成交量相对自身中位数估算 ---
        daily["turn_est"] = True
        med = daily.groupby("code")["volume"].transform(
            lambda s: s.rolling(250, min_periods=60).median())
        daily["turnover"] = (daily["volume"] / med.replace(0, np.nan)) * 0.015
        daily["float_mcap"] = np.nan

    # 成交额：新浪直接有 amount，腾讯要估
    if "amount" not in daily.columns:
        daily["amount"] = (daily["volume"] * 100
                           * (daily["high"] + daily["low"] + daily["close"]) / 3)
    daily["float_mcap"] = daily["float_mcap"].fillna(
        daily["amount"] * 50)          # 只用于市值分档，量级够用

    # 一字板 / 数据异常会给出离谱的换手，截断到 60%（A股单日极限量级）
    daily["turnover"] = daily["turnover"].clip(0, 0.60)
    return daily


def build(sample: int = 0) -> int:
    t0 = time.time()
    dp = DATA / "daily.parquet"
    if not dp.exists():
        log.error("缺 %s，先跑 backfill --stage daily", dp)
        return 1
    daily = pd.read_parquet(dp)
    log.info("日线 %d 行 %d 只 %s..%s", len(daily), daily.code.nunique(),
             daily.date.min(), daily.date.max())

    if sample:
        keep = sorted(daily.code.unique())[:sample]
        daily = daily[daily.code.isin(keep)]
        log.info("抽样模式：只用 %d 只", daily.code.nunique())

    daily = attach_turnover(daily, None)
    est = daily.groupby("code")["turn_est"].first()
    log.info("换手率：真值 %d 只，估算 %d 只（%.1f%%）",
             int((~est).sum()), int(est.sum()), 100 * float(est.mean()))

    # ---- 逐只：时序特征 + 筹码 + 标签 ----
    # 筹码是路径依赖的（今天的分布取决于昨天），没法跨股票向量化。
    out = []
    codes = daily.code.unique()
    for i, code in enumerate(codes):
        g = daily[daily.code == code].sort_values("date").reset_index(drop=True)
        if len(g) < 120:
            continue
        g = F.per_stock(g)
        g = F.add_chip_block(g)
        g = pd.concat([g, label_frame(g)], axis=1)   # 标签最后拼，来自另一模块
        out.append(g)
        if (i + 1) % 500 == 0:
            log.info("  逐只处理 %d/%d  用时 %.1f 分钟",
                     i + 1, len(codes), (time.time() - t0) / 60)
    panel = pd.concat(out, ignore_index=True)
    log.info("逐只完成 %d 行，用时 %.1f 分钟", len(panel), (time.time()-t0)/60)

    panel["board"] = panel["code"].map(F.board_of)

    # ---- 股东人数（按公告日对齐） ----
    hp = DATA / "holders.parquet"
    holders = pd.read_parquet(hp) if hp.exists() else None
    if holders is None:
        log.warning("没有 holders.parquet，股东人数特征全 NaN")
    panel = F.holder_features(panel, holders)

    # ---- 全市场层 ----
    panel = F.add_market(panel)

    # ---- ① 横截面 ② 中性化 ③ 交互 ----
    cols = [c for c in F.feature_columns()
            if c not in {n for _, _, n in F.INTERACTIONS}]
    cols = [c for c in cols if c in panel.columns]
    panel = F.cross_section(panel, cols)
    panel = F.neutralize(panel, cols)
    panel = F.add_interactions(panel)
    log.info("三层变换完成，用时 %.1f 分钟", (time.time() - t0) / 60)

    # ---- 5 日窗口聚合 ----
    feats = [c for c in F.feature_columns() if c in panel.columns]
    panel = panel.sort_values(["code", "date"])
    g = panel.groupby("code", sort=False)
    agg = {}
    for c in feats:
        agg[f"{c}__last"] = panel[c]
        agg[f"{c}__mean"] = g[c].transform(lambda s: s.rolling(WIN).mean())
        agg[f"{c}__slope"] = g[c].transform(
            lambda s: (s - s.shift(WIN - 1)) / (WIN - 1))
    panel = pd.concat([panel, pd.DataFrame(agg, index=panel.index)], axis=1)

    keep = (["code", "date", "board", "close", "turn_est", "float_mcap",
             "y_up", "y_t0", "y_top"]
            + [c for c in panel.columns if "__" in c])
    train = panel[keep].copy()
    DATA.mkdir(parents=True, exist_ok=True)
    train.to_parquet(DATA / "train.parquet", index=False)

    n_feat = len([c for c in train.columns if "__" in c])
    log.info("训练表 %d 行 × %d 特征 -> %s", len(train), n_feat,
             DATA / "train.parquet")
    v = train.dropna(subset=["y_t0"])
    log.info("标签：y_up 正样本率 %.3f%%，起涨点 y_t0 %d 个，见顶 y_top %d 个",
             100 * train["y_up"].mean(skipna=True),
             int(np.nansum(train["y_t0"])), int(np.nansum(train["y_top"])))
    log.info("有效样本 %d 行，总用时 %.1f 分钟", len(v), (time.time()-t0)/60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="只用前 N 只，先跑通流程")
    return build(ap.parse_args().sample)


if __name__ == "__main__":
    sys.exit(main())
