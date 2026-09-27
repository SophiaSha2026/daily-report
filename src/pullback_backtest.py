"""
长期调整突破的历史回看：在本机三年全市场日线上把形态扫一遍，看多久出一次、
卡在哪一步、之后怎么走的。只读，不落任何产物。

    python src/pullback_backtest.py              按 config.yaml 当前口径
    python src/pullback_backtest.py --grid       再把几个关键旋钮各拧一格对比频率

判定用的是 pullback.prepare / find_events 本身，和每天发的清单是同一个函数
（教训 34：回测另写一份判据，迟早和生产分叉）。selftest_pullback 用 AST 钉住这件事。

注意三点再读数字：
  · 这是在已知历史上数出来的，阈值也是看过这段历史之后定的（样本内），
    频率可信，「之后怎么走」只能当描述，不能当预期收益
  · ST 按今天的名单剔（cache/st_codes.json），历史上当时是不是 ST 拿不到
  · 减持 / 定增按每次的事件日核，只用那天之前已经公告的（corp_events，和生产同一份）。
    第一次跑要联网把这几十只票的公告拉下来（缓存在 data/pullback/corp/）
  · 前复权价：最近一次除权之前的价格被缩放过，涨停判定有 ±1 分的舍入容差
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import pullback as P          # noqa: E402


def run(x: pd.DataFrame, c: dict, st: set[str]) -> tuple[pd.DataFrame, dict]:
    """和每天发的清单同一套：find_events -> forward -> 按事件日核减持 / 定增 -> 剔 ST。"""
    ev, _, diag = P.find_events(x, c)
    evf = P.mark_risk(P.forward(x, ev, int(c.get("output", {}).get("forward_bars", 20))), c)
    if len(evf):
        # 被减持 / 定增剔掉的也记进各环节淘汰，回看时看得见卡在哪
        for why in evf.loc[evf["risk"] == "out", "risk_why"]:
            k = "剔除：股东减持" if "减持" in why else "剔除：定增价格没定"
            diag[k] = diag.get(k, 0) + 1
        n_unk = int((evf["risk"] == "unknown").sum())
        if n_unk:
            diag["减持 / 定增数据没拉到（按不剔算）"] = n_unk
    return P.history_pool(evf, st).reset_index(drop=True), diag


def report(ev: pd.DataFrame, x: pd.DataFrame, diag: dict, bars: int, base_days: int) -> None:
    st = P.history_stats(ev, x, bars, base_days)
    # 下面的分项和频率用同一批事件：全市场覆盖齐了之后那一段（P.coverage_start）
    ev = ev[ev["date"] >= st["from"]] if len(ev) else ev
    print(f"区间 {st['from']} ~ {st['to']}，{st['days']} 个交易日"
          f"（日线表更早的那段只有几百只票有数据，不算）")
    print(f"成立 {st['n']} 次，平均每月 {st['per_month']} 次，"
          f"有成立的交易日 {ev['date'].nunique() if len(ev) else 0} 天")
    if not len(ev):
        return
    print("\n按年：", ev.groupby(ev["date"].str[:4]).size().to_dict())
    print("按板块：", ev.groupby("board").size().rename(P.BOARD_NAME).to_dict())
    print("调整天数：", ev.groupby("adjust_days").size().to_dict())
    print("加分项满足个数：",
          ev[["bonus_vol", "bonus_half", "bonus_open"]].astype(int).sum(axis=1)
          .value_counts().sort_index().to_dict())
    if st.get("n_final"):
        print(f"\n走满 {bars} 根的 {st['n_final']} 次：之后最高价涨幅中位 "
              f"{st['max_up_median']:+.1f}%，第 {bars} 根收盘中位 {st['ret_median']:+.1f}%，"
              f"收涨的占 {100 * st['up_share']:.0f}%（样本内描述，不是预期收益）")
    print("\n各环节淘汰（全表所有倍量大阳线）：")
    for k, v in sorted(diag.items(), key=lambda kv: -kv[1]):
        print(f"  {v:7d}  {k}")
    print("\n前期调整天数（首阳前收盘价待在振幅上限里的交易日数）：",
          ev["base_run"].describe()[["min", "25%", "50%", "75%", "max"]].round(0).to_dict())
    print("\n最近 15 次：")
    cols = ["date", "code", "board", "launch_date", "base_run", "adjust_days", "gain_pct",
            "vol_vs_launch", "adj_vol_min", "max_up_pct", "ret_pct"]
    print(ev.sort_values("date").tail(15)[cols].to_string(index=False))


GRID = [
    ("横盘振幅 ≤25%", {"base.amp_max": 0.25}),
    ("横盘振幅 ≤35%", {"base.amp_max": 0.35}),
    ("首阳 ≥ 横盘均量 2.5 倍", {"base.vol_mean_mult": 2.5}),
    ("不要求首阳是 3 个月最大量", {"base.vol_above_max": False}),
    ("调整最多 5 天", {"adjust.max_days": 5}),
    ("调整最多 15 天", {"adjust.max_days": 15}),
    ("不破首阳开盘价（更严）", {"adjust.floor": "open"}),
    ("第二根放宽成 ≥5% 阳线", {"trigger.big": "loose"}),
]


def with_over(c: dict, over: dict) -> dict:
    c2 = copy.deepcopy(c)
    for path, v in over.items():
        sec, key = path.split(".")
        c2[sec][key] = v
    return c2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="几个关键旋钮各拧一格对比频率")
    a = ap.parse_args()
    t0 = time.time()
    c = P.cfg()
    raw = pd.read_parquet(P.DAILY, columns=["code", "date", "open", "high", "low", "close",
                                             "volume", "outstanding_share"])
    x = P.prepare(raw, c)
    st = P.st_cache()
    bars = int(c.get("output", {}).get("forward_bars", 20))
    base = int(c["base"]["days"])
    ev, diag = run(x, c, st)
    report(ev, x, diag, bars, base)
    if a.grid:
        # 频率只有 history_stats 一份算法（以前这里另算 len/全表天数）
        s0 = P.history_stats(ev, x, bars, base)
        print(f"\n旋钮对比（当前口径 {s0['n']} 次，每月 {s0['per_month']:.2f} 次）：")
        for name, over in GRID:
            e2, _ = run(x, with_over(c, over), st)
            s2 = P.history_stats(e2, x, bars, base)
            tail = (f"，之后最高中位 {s2['max_up_median']:+.1f}%"
                    if s2.get("n_final") else "")
            print(f"  {name:<22} {s2['n']:4d} 次  每月 {s2['per_month']:.2f} 次{tail}")
    print(f"\n用时 {time.time() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
