"""
形态历史频率验证。回答一个问题：这个形态到底多久出一次票。

为什么必须有这个
----------------
三段条件叠起来很紧。如果实际上每天都扫不出票，那条流水线每天准点发一封
「今日无标的」，看着一切正常，其实毫无用处，而且**这种失败是看不出来的**：
「今天 0 只」本来就是形态类策略的常态，跟条件写错了长得一模一样。

怎么算
------
抽样估计，不做全市场全历史（那要 5548 只 x 90 天的日线，太重）：

1. 从代码表里随机抽 `--sample` 只（默认 800）
2. 拉 90 个自然日日线
3. 把最近 `--days` 个交易日逐个当成「执行日 T」跑一遍完整三段判定
4. 命中数按 5548/sample 放大，得到全市场每日期望只数

换手率的锚
----------
历史日线（腾讯那路）不带换手率，所以仍用等比反推，但锚点换成**这只票在
窗口内的今日快照**：先用批量行情取一次今日换手率和今日成交量，再把窗口里
每一天按成交量等比折算。和线上跑的是同一个恒等式，误差同源，不会因为
换算方式不同而产生系统性偏差。

抽样误差
--------
800 只样本、命中率若为 p，放大后的标准误约 sqrt(p*(1-p)/800) * 5548。
p 很小时估计值波动大，所以输出里同时给原始命中数，别只看放大后的数字。
"""
from __future__ import annotations

import sys
import random
import logging
import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import datasource as ds                                    # noqa: E402
from datasource import _turnover as turnover_pct           # noqa: E402
from pullback import norm_hist, find_pattern               # noqa: E402

BJ = ZoneInfo("Asia/Shanghai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backtest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--days", type=int, default=20, help="回看几个交易日")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    c = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pb, tg = c["pullback"], c["pullback"]["trigger"]

    codes = ds.load_code_list()
    random.seed(a.seed)
    pick = random.sample(codes, min(a.sample, len(codes)))
    log.info("全市场 %d 只，抽样 %d 只", len(codes), len(pick))

    # 换手率的锚点：今日快照
    q = ds.fetch_quotes([ds.to_symbol(x) for x in pick])
    anchor = {v.code: (turnover_pct(v), float(v.volume_hand))
              for v in q.values() if float(v.volume_hand) > 0}
    log.info("换手率锚点拿到 %d 只", len(anchor))

    today = dt.datetime.now(BJ).strftime("%Y-%m-%d")
    start = (dt.datetime.now(BJ) - dt.timedelta(days=pb["lookback_days"])
             ).strftime("%Y-%m-%d")
    hs = ds.daily_hist_many(pick, start, today, workers=4)
    log.info("日线 %d 只成功，来源 %s",
             sum(1 for v in hs.values() if v is not None and len(v)),
             ds.hist_source_stats())

    per_day: dict[str, int] = {}
    hits: list[tuple[str, str, str, int]] = []
    scanned = 0
    diag: dict[str, int] = {}

    for code, raw in hs.items():
        h = norm_hist(raw)
        h = h[h["日期"] < today].reset_index(drop=True)   # 今日未收盘，不参与
        need = pb["adjust"]["max_days"] + 3
        if len(h) < need + a.days:
            continue
        to_a, vol_a = anchor.get(code, (0.0, 0.0))
        if vol_a <= 0 or to_a <= 0:
            continue
        scanned += 1
        for k in range(a.days):
            j = len(h) - 1 - k                    # 把 h[j] 当成执行日 T
            if j < need:
                break
            T, prev = h.iloc[j], h.iloc[j - 1]
            day = str(T["日期"])
            per_day.setdefault(day, 0)
            if float(T["涨跌幅"]) < tg["gain_pct_min"]:
                continue
            if float(prev["成交量"]) <= 0:
                continue
            if float(T["成交量"]) / float(prev["成交量"]) < tg["vol_ratio_min"]:
                continue
            t_to = to_a * float(T["成交量"]) / vol_a
            if not (tg["turnover_min"] <= t_to <= tg["turnover_max"]):
                continue
            # 用执行日自己的换手/量做锚，和线上完全一致
            pat = find_pattern(h.iloc[:j].reset_index(drop=True),
                               t_to, float(T["成交量"]), pb, diag)
            if pat:
                per_day[day] += 1
                hits.append((day, code, pat["launch_date"], pat["adjust_days"]))

    if not scanned:
        log.error("没有一只票凑齐足够的历史，无法估计")
        return 1

    scale = len(codes) / scanned
    days = sorted(per_day)
    total = sum(per_day.values())
    print(f"\n样本内可用 {scanned} 只，覆盖 {len(days)} 个交易日，"
          f"放大系数 {scale:.1f}x")
    print(f"{'交易日':<12}{'样本命中':>8}{'全市场估计':>12}")
    for d in days:
        print(f"{d:<12}{per_day[d]:>8}{per_day[d] * scale:>12.0f}")
    print(f"{'合计':<12}{total:>8}{total * scale:>12.0f}")
    if days:
        print(f"\n平均每个交易日：样本 {total / len(days):.2f} 只，"
              f"全市场约 {total * scale / len(days):.0f} 只")
    print(f"\n落选归因（样本内）：{dict(sorted(diag.items(), key=lambda kv: -kv[1]))}")
    if hits:
        print("\n命中样例（最多 15 条）")
        for d, code, ld, ad in hits[:15]:
            print(f"  {d}  {code}  启动日 {ld}  调整 {ad} 日")
    return 0


if __name__ == "__main__":
    sys.exit(main())
