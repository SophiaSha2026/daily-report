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
# 生产（daily.stage_scan）只给历史 ≥120 根的票打分，训练表必须同口径。
# 以前训练表把每只票的前 120 行也喂进去（65.7 万行 = 15.3%，正样本 8.7%），
# 那些行的滚动窗口大半是 NaN→0，筹码是「几天成本冒充三年累积」
# （实测 2023-06-15 那批的 chip_conc90 百分位偏 +0.13），生产从不给这种票打分。
# 注意：train.parquet 已经按这条规则砍过预热行，所以**下游不能再砍一次**。
# daily.stage_scan 现在还在对 train.parquet 数 `cnt >= 120`，那等于要求
# 240 根日线，规则悄悄严了一倍（2026-09-16 审计记下，改在 daily.py 那边）。
MIN_HIST = 120
# 共同起点：某一天全市场的票不齐，当天的横截面百分位、板块中性化、rs20/rs60
# 就不是在同一个池子里算的。首次回填按根数截、除权重拉按日历截，两套起点没
# 对齐，daily.parquet 里 2023-01-03~05-29 那 96 天只有 37~889 只（全是分过红
# 的），而且**每次除权重拉都会再往这段薄片里塞一只** —— 训练表在没有新信息的
# 情况下天天自己漂，同一份表重建一次结果就不同。
COVER_MIN = 0.90
# 市场层的预热：mkt_ret5 是 5 日滚动和，面板首日每只票都没有前一根，
# 当天的 mkt_breadth 没有分母。个股层的预热已经在 MIN_HIST 那一步砍掉了
# （ret20/ret60/dist_52w_* 都是在**裁剪前**的整段历史上算的），
# 所以这里不需要核实员建议的 65 天，5 天就够。
WARMUP = WIN


def eligible(daily: pd.DataFrame, date: str,
             min_hist: int = MIN_HIST) -> pd.Index:
    """截至 date（含）历史根数够的代码 —— 「够不够老」这条规则的唯一定义。

    前 min_hist 根只用来预热滚动窗和筹码，第 min_hist+1 根起才够格，
    所以判据是 `> min_hist` 而不是 `>=`（assemble 的 `iloc[MIN_HIST:]`
    是同一条规则的逐只写法，selftest 钉住两者等价）。

    生产 daily.stage_scan 打分前剔次新、训练表建表时砍预热行、回测排名，
    三处必须是同一条规则：2026-09-16 实测旧缓存的 W5 首日 783 个样本里
    61 个是历史不足 120 根的次新（命中 3.28%，非次新 13.43%），它们还挤掉了
    18 个本该进前 10 的票 —— 邮件里印的成绩描述的不是线上在跑的规则。
    """
    h = daily[daily["date"] <= date].groupby("code")["date"].size()
    return h[h > min_hist].index


def trim_common_start(daily: pd.DataFrame,
                      cover: float = COVER_MIN) -> tuple[pd.DataFrame, str]:
    """砍掉全市场不齐的那段前缀，返回 (表, 共同起点)。见 COVER_MIN。

    参照系是全期最大只数，而新股是一直在上市的：2026-09 实测 2023-05-30
    有 5039 只、全期最大 5516 只，比值 0.913，**离 0.90 只差 1.3 个百分点**。
    等以后新股把分母撑大，这条线会突然往后跳、白扔几个月历史，所以砍掉超过
    一个季度时要出声 —— 无声地少一段训练数据是查不出来的。
    """
    cnt = daily.groupby("date")["code"].nunique()
    if not len(cnt):
        return daily, ""
    full = cnt.index[cnt >= cover * cnt.max()]
    if not len(full):
        return daily, str(cnt.index.min())
    start = str(full.min())
    cut = int((cnt.index.astype(str) < start).sum())
    if cut > 60:
        log.warning("共同起点 %s 砍掉了 %d 个交易日（全期最大 %d 只，"
                    "起点当天 %d 只）。砍得这么多要核一下是不是该跑 "
                    "backfill --stage refresh 把起点对齐",
                    start, cut, int(cnt.max()), int(cnt.loc[full.min()]))
    return daily[daily["date"].astype(str) >= start], start


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
            # close 是**前复权**价（backfill.py 取 adjust="qfq"，akshare 只除
            # 价格四列，volume/amount/outstanding_share 全是原值），股本是
            # 当时的真实股本，直接相乘等于把除权日之前的市值按复权因子压小。
            # 实测 daily.parquet 里 amount/(close×volume) >1.2 的行占 10.2%、
            # >1.5 占 2.7%，涉及 1289/5516 只；300260 2023-06-15 算出 5.59e9，
            # 真值 1.02e10。float_mcap 唯一的用途是 features.neutralize 的
            # 市值五分档，抽样四天实测 7.8% 的行因此分错档，它们的**全部**
            # 中性化特征减的是邻档均值（turn_pct 百分位偏 0.22σ 中位）。
            # 更糟的是这个偏差和标签同向（ratio>1.2 的行 y_up 正样本率 4.62%
            # vs 3.28%）：复权因子由 t **之后**的分红送转决定，等于把未来
            # 塞进了 t 日的分组键，属教训 25 那一类的时间错位。
            # 打分日那一行几乎不动（末日因子分位 0.964~1.012）。
            px = daily["close"]
            if "amount" in daily.columns:
                vwap = daily["amount"] / daily["volume"].where(
                    daily["volume"] > 0)
                fac = vwap / daily["close"].where(daily["close"] > 0)
                # 因子只可能 ≥1（qfq），落在区间外的是停牌行或 stage_update
                # 那条路上的成交量单位判错（688 的股/手差 100 倍），退回 close
                px = daily["close"] * fac.where(fac.between(0.5, 60)).fillna(1.0)
            daily["float_mcap"] = daily["outstanding_share"] * px
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


def assemble(daily: pd.DataFrame, t0: float | None = None) -> pd.DataFrame:
    """逐只算时序特征 + 筹码 + 标签，然后砍掉两类和生产口径不一致的行。

    筹码是路径依赖的（今天的分布取决于昨天），没法跨股票向量化。

    两道过滤（2026-09-16 加）：
      · 每只票前 MIN_HIST 行只用来预热滚动窗和筹码，不进训练表。先算完
        特征和标签再切，标签向前看，不受影响。这里的 `iloc[MIN_HIST:]`
        和 eligible() 是同一条规则的两种写法（逐只 / 按日），
        selftest 的 check_history_alignment 钉住两者逐票一致。
      · 横截面不足当日中位数一半的日期整天丢掉。首次回填按根数截、除权
        重拉按日历截，两套起点没对齐，2023-01-03~05-29 那 96 天只有
        37~889 只票（全是分过红的），当日全市场排名是在 37 只里排的，
        而且每次除权重拉都会再往这段薄片里塞一只 —— 训练表在没有新信息的
        情况下天天自己漂。
    """
    out = []
    codes = daily.code.unique()
    for i, code in enumerate(codes):
        g = daily[daily.code == code].sort_values("date").reset_index(drop=True)
        if len(g) < MIN_HIST:
            continue
        g = F.per_stock(g)
        g = F.add_chip_block(g)
        # 标签最后拼，来自另一模块，而且**只把价格递过去**：label_frame 会
        # 断言自己没收到别的列。以前整个 g（36 列特征+筹码）都传进去，
        # 结果虽然逐位相同，但 label.py 开头「不共享 DataFrame」那句
        # 就只有注释在守（S11）。
        g = pd.concat([g, label_frame(g[["close", "high"]])], axis=1)
        out.append(g.iloc[MIN_HIST:])
        if t0 is not None and (i + 1) % 500 == 0:
            log.info("  逐只处理 %d/%d  用时 %.1f 分钟",
                     i + 1, len(codes), (time.time() - t0) / 60)
    panel = pd.concat(out, ignore_index=True)
    if t0 is not None:
        log.info("逐只完成 %d 行，用时 %.1f 分钟", len(panel),
                 (time.time() - t0) / 60)
    n_by_date = panel.groupby("date")["code"].nunique()
    thin = n_by_date[n_by_date < 0.5 * n_by_date.median()].index
    if len(thin):
        log.warning("丢掉 %d 个横截面不足半数的日期 %s..%s",
                    len(thin), min(thin), max(thin))
        panel = panel[~panel["date"].isin(set(thin))].reset_index(drop=True)
    panel["board"] = panel["code"].map(F.board_of)
    return panel


def build(sample: int = 0) -> int:
    t0 = time.time()
    dp = DATA / "daily.parquet"
    if not dp.exists():
        # 首次回填走的是新浪（--stage sina），腾讯那条 stage 只是兜底源，
        # 照它跑会拿到没有换手率 / 股本的表（S27）
        log.error("缺 %s，先跑 backfill --stage sina（首次回填）"
                  "或 --stage update（日常增量）", dp)
        return 1
    daily = pd.read_parquet(dp)
    log.info("日线 %d 行 %d 只 %s..%s", len(daily), daily.code.nunique(),
             daily.date.min(), daily.date.max())
    daily, start = trim_common_start(daily)
    log.info("共同起点 %s（之前全市场不齐，横截面不可比，见 COVER_MIN）", start)

    if sample:
        keep = sorted(daily.code.unique())[:sample]
        daily = daily[daily.code.isin(keep)]
        log.info("抽样模式：只用 %d 只", daily.code.nunique())

    daily = attach_turnover(daily, None)
    est = daily.groupby("code")["turn_est"].first()
    log.info("换手率：真值 %d 只，估算 %d 只（%.1f%%）",
             int((~est).sum()), int(est.sum()), 100 * float(est.mean()))

    panel = assemble(daily, t0)

    # ---- 股东人数（按公告日对齐） ----
    hp = DATA / "holders.parquet"
    holders = pd.read_parquet(hp) if hp.exists() else None
    if holders is None:
        log.warning("没有 holders.parquet，股东人数特征全 NaN")
    panel = F.holder_features(panel, holders)

    # ---- 全市场层 ----
    panel = F.add_market(panel)
    # 市场层自己的预热期（见 WARMUP）：面板首日没有昨收，mkt_breadth 是 NaN、
    # mkt_ret5 要攒满 5 天。留着这几天等于给全市场几千行喂一个 NaN 市场环境。
    pdates = sorted(panel["date"].unique())
    if len(pdates) > WARMUP:
        panel = panel[panel["date"] >= pdates[WARMUP]].reset_index(drop=True)
        log.info("砍掉市场层预热的前 %d 个交易日（到 %s）", WARMUP, pdates[WARMUP])

    # ---- ① 横截面 ② 中性化 ③ 交互 ----
    cols = [c for c in F.feature_columns()
            if c not in {n for *_, n in F.INTERACTIONS}]
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
