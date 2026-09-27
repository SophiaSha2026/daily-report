"""
长期调整突破的离线自测。不联网、不碰 state/ 和生产目录，产物一律写临时目录。

    python src/selftest_pullback.py

钉的东西：
  · 每条硬规则都有一个用例真的碰到它（教训 26：写在配置里、印在邮件上，
    却没有一处代码执行它，是这个项目最常见的一类 bug）
  · 板块口径：主板要涨停，创业板/科创板/北交所要 ≥10%；复权舍入的涨停判定
  · 「调整后第一次站上首阳高点」就是考卷，考砸了这段形态作废
  · 股本跳变（送转）那几天的量不可比
  · 产物往返：代码前导零（教训 29）、空榜写空文件、面板和邮件的转义
  · 发信接线：日期对不上不发、SKIP_MAIL 不发不落戳、真发之后才落 mail_sent.json
  · 17:58 的等待：过点立即放行
  · 回测脚本和生产是同一个判定函数（AST，教训 34）
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import corp_events as CE     # noqa: E402
import datasource as ds      # noqa: E402
import pullback as P         # noqa: E402
import pullback_export as E  # noqa: E402

fails: list[str] = []


def ck(cond: bool, msg: str) -> None:
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)


# 测试代码取自真实代码表（教训 2：凭记忆写的代码失败时分不清是逻辑错还是代码不存在）
MAIN, CYB, STAR, BJ = "600650", "300461", "688693", "920964"


def pb(**over) -> dict:
    c = P.cfg()
    for path, v in over.items():
        sec, key = path.split(".")
        c[sec][key] = v
    return c


# ---------------------------------------------------------------------
#  造数据
# ---------------------------------------------------------------------
def _dates(n: int, start: str = "2025-01-02") -> list[str]:
    d = pd.bdate_range(start, periods=n)
    return [x.strftime("%Y-%m-%d") for x in d]


def build(code: str, *, base_n: int = 70, base_amp: float = 0.08, base_vol: float = 1e6,
          spike: float = 0.0, s_gain: float | None = None, s_vol: float = 5.0,
          adj: list[tuple[float, float, float]] | None = None,
          t_gain: float | None = None, t_vr: float = 1.8, t_close_over: bool = True,
          tail: list[tuple[float, float]] | None = None, os_jump_at: int | None = None,
          limit_exact: bool = True) -> pd.DataFrame:
    """一只票：横盘 base_n 根 -> 首阳 S -> 调整 adj -> 二次进攻 T -> 之后 tail。

    adj 每项 (收盘相对首阳收盘, 最低相对首阳最低, 量相对首阳量)
    tail 每项 (收盘相对前一根, 量相对前一根)，在 T 之后
    s_gain / t_gain = None 表示「按板块封涨停」（主板）或 +12%（其余）
    """
    lim = ds.limit_pct(code, "")
    main = P.board_of(code) == "main"
    rows = []
    px = 10.0
    for i in range(base_n):
        # 横盘：收盘在 10 × (1 ± base_amp/2) 之间来回，量围着 base_vol 小幅波动
        c = px * (1 + (base_amp / 2) * (1 if i % 4 < 2 else -1) * (0.5 + 0.5 * ((i * 7) % 5) / 4))
        v = base_vol * (1 + 0.15 * ((i * 3) % 5 - 2) / 2)
        if spike and i == base_n - 10:
            v = base_vol * spike
        rows.append([c * 0.998, c * 1.006, c * 0.994, round(c, 2), v])
    prev = rows[-1][3]

    def up_close(p: float, gain: float | None) -> float:
        if gain is None:
            if main:
                lp = ds.limit_price(p, lim, code)
                return lp if limit_exact else round(lp - 0.02, 2)
            return round(p * 1.12, 2)
        return round(p * (1 + gain), 2)

    # 首阳
    sc = up_close(prev, s_gain)
    so, sl = round(prev * 1.01, 2), round(prev * 1.005, 2)
    sv = s_vol * rows[-1][4] if s_vol else rows[-1][4]
    rows.append([so, sc, sl, sc, sv])
    s_close, s_low, s_vol_abs = sc, sl, sv
    # 调整
    for cr, lr, vr in (adj if adj is not None else [(0.985, 1.01, 0.6), (0.975, 1.005, 0.45),
                                                   (0.98, 1.008, 0.4)]):
        c = round(s_close * cr, 2)
        lo = round(s_low * lr, 2)
        rows.append([c, max(c, lo) * 1.004, lo, c, s_vol_abs * vr])
    # 二次进攻
    if t_gain is not False:
        p = rows[-1][3]
        tc = up_close(p, t_gain)
        if not t_close_over:
            tc = round(min(tc, s_close - 0.01), 2)
        rows.append([round(p * 1.01, 2), tc, round(p * 1.005, 2), tc, rows[-1][4] * t_vr])
    for cr, vr in (tail or []):
        p = rows[-1][3]
        c = round(p * cr, 2)
        rows.append([c, c * 1.01, c * 0.99, c, rows[-1][4] * vr])
    n = len(rows)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    # 最高价至少盖住开收，最低价至少托住开收
    df["high"] = df[["open", "high", "close"]].max(axis=1).round(2)
    df["low"] = df[["open", "low", "close"]].min(axis=1).round(2)
    df["open"] = df["open"].round(2)
    df["code"] = code
    df["date"] = _dates(n)
    df["outstanding_share"] = 1e8
    if os_jump_at is not None:
        df.loc[os_jump_at:, "outstanding_share"] = 2e8
    return df


def events_of(df: pd.DataFrame, c: dict | None = None):
    c = c or pb()
    x = P.prepare(df, c)
    return P.find_events(x, c)


# ---------------------------------------------------------------------
def check_board() -> None:
    print("[板块口径]")
    ck(P.board_of("600650") == "main" and P.board_of("000001") == "main"
       and P.board_of("001289") == "main", "60/00/001 开头是沪深主板")
    ck(P.board_of("300461") == "chinext" and P.board_of("301369") == "chinext", "30 开头是创业板")
    ck(P.board_of("688693") == "star" and P.board_of("689009") == "star", "68 开头是科创板")
    ck(all(P.board_of(c) == "bj" for c in ("920964", "830799", "430047")), "8/4/9 开头是北交所")
    codes = set(pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"])
    ck({MAIN, CYB, STAR, BJ} <= codes, "自测用的四只代码都在 cache/codes.csv 里（教训 2）")


def check_positive() -> None:
    print("[标准形态]")
    for code in (MAIN, CYB, STAR, BJ):
        ev, op, diag = events_of(build(code))
        ok = len(ev) == 1 and ev.iloc[0]["adjust_days"] == 3
        ck(ok, f"{P.BOARD_NAME[P.board_of(code)]} {code}：横盘 -> 首阳 -> 缩量 3 天 -> 二次进攻，成立 1 次")
    ev, _, _ = events_of(build(MAIN))
    r = ev.iloc[0]
    ck(bool(r["launch_limit_up"]), "主板首阳是涨停")
    ck(r["date"] == _dates(75)[-1] and r["launch_date"] == _dates(75)[-5],
       "事件日是二次进攻那天，首阳日是 4 根之前")
    ck(abs(r["vol_ratio"] - 1.8) < 0.01, "今日量比 = 今日量 / 昨日量")


def check_launch_rules() -> None:
    print("[首阳]")
    ev, _, d = events_of(build(MAIN, s_gain=0.08))
    ck(len(ev) == 0, "主板首阳 +8% 不是涨停：不算首阳")
    ev, _, _ = events_of(build(MAIN, limit_exact=False))
    ck(len(ev) == 0, "主板收在涨停价下方 2 分：不算涨停")
    ev, _, _ = events_of(build(CYB, s_gain=0.095))
    ck(len(ev) == 0, "创业板首阳 +9.5%：不到 10%")
    # 「正好 10%」要造一根真能正好 10% 的：昨收 10.00 -> 11.00（两位小数舍入后
    # 多数价格凑不出正好 10%，比如 9.63 -> 10.59 是 +9.97%）
    x = pd.DataFrame({"code": [CYB] * 3, "date": _dates(3),
                      "open": [10.0, 10.1, 11.0], "high": [10.0, 11.0, 12.0],
                      "low": [10.0, 10.0, 10.9], "close": [10.0, 11.0, 11.99],
                      "volume": [1e6, 3e6, 5e6], "outstanding_share": [1e8] * 3})
    y = P.prepare(x, pb())
    ck(bool(y["big"].iloc[1]), "创业板 10.00 -> 11.00 正好 +10%：算（「10% 以上」含 10%）")
    ck(not bool(y["big"].iloc[2]), "创业板 11.00 -> 11.99 是 +9.0%：不算")
    ev, _, _ = events_of(build(MAIN, s_vol=1.4))
    ck(len(ev) == 0, "首阳量只有前一日 1.4 倍：不算倍量")
    # 复权舍入：收在最高价、比按昨收算的涨停价高 1 分 / 低 1 分都算涨停
    x = pd.DataFrame({"code": [MAIN] * 3, "date": _dates(3),
                      "open": [10.0, 10.1, 11.0], "high": [10.0, 11.01, 12.10],
                      "low": [10.0, 10.1, 11.0], "close": [10.0, 11.01, 12.10],
                      "volume": [1e6, 3e6, 5e6], "outstanding_share": [1e8] * 3})
    y = P.prepare(x, pb())
    ck(bool(y["limit_up"].iloc[1]) and bool(y["limit_up"].iloc[2]),
       "收在最高价、离涨停价 ±1 分（复权段舍入）按涨停算")
    x2 = x.copy()
    x2.loc[1, ["high", "close"]] = [11.0, 10.98]
    y2 = P.prepare(x2, pb())
    ck(not bool(y2["limit_up"].iloc[1]), "没收在最高价（冲高回落）不算涨停")


def check_base_rules() -> None:
    print("[横盘]")
    # build 的 base_amp=a 造出来的收盘振幅是 (1+a/2)/(1-a/2)-1：0.40 -> 50%
    ev, _, d = events_of(build(MAIN, base_amp=0.40))
    ck(len(ev) == 0 and d.get("横盘振幅超标", 0) >= 1, "横盘 60 日振幅 50%：不合格（上限 30%）")
    ev, _, d = events_of(build(MAIN, spike=6.0))
    ck(len(ev) == 0 and d.get("横盘期有比首阳更大的量", 0) >= 1,
       "横盘期有一天量比首阳还大：不合格（首阳必须是 3 个月来最大量）")
    ev, _, d = events_of(build(MAIN, s_vol=2.0))
    ck(len(ev) == 0 and d.get("首阳量不到横盘均量倍数", 0) >= 1,
       "首阳量只有横盘均量 2 倍：不合格（要 3 倍）")
    ev, _, _ = events_of(build(MAIN, base_n=50))
    ck(len(ev) == 0, "首阳前不满 60 根：不判（「3 个月以上」）")
    c = pb(**{"base.amp_max": 0.55})
    ev, _, _ = events_of(build(MAIN, base_amp=0.40), c)
    ck(len(ev) == 1, "阈值从 config 读：amp_max 调到 55% 后同一段就成立")


def check_adjust_rules() -> None:
    print("[调整]")
    ev, _, d = events_of(build(MAIN, adj=[(0.98, 1.01, 0.5)]))
    ck(len(ev) == 0, "只调整 1 天就站上首阳高点：不算（至少 2 天）")
    long_adj = [(0.97, 1.01, 0.5)] * 11
    ev, op, d = events_of(build(MAIN, adj=long_adj))
    ck(len(ev) == 0 and d.get("调整超过10天没突破", 0) >= 1, "调整 11 天：超过 10 天上限")
    ev, _, _ = events_of(build(MAIN, adj=[(0.97, 1.01, 0.5)] * 10))
    ck(len(ev) == 1, "调整正好 10 天：还算")
    ev, _, d = events_of(build(MAIN, adj=[(0.98, 1.01, 0.6), (0.97, 1.01, 1.05), (0.98, 1.01, 0.4)]))
    ck(len(ev) == 0 and d.get("调整期有一天没缩量", 0) >= 1, "调整期有一天量超过首阳：不算缩量")
    ev, _, d = events_of(build(MAIN, adj=[(0.98, 1.01, 0.95), (0.97, 1.01, 0.9), (0.98, 1.01, 0.85)]))
    ck(len(ev) == 0 and any("调整期均量太大" in k for k in d),
       "调整期每天都略低于首阳但均量 90%：不算（均量上限 80%）")
    ev, _, d = events_of(build(MAIN, adj=[(0.98, 1.01, 0.5), (0.96, 0.995, 0.4), (0.98, 1.01, 0.4)]))
    ck(len(ev) == 0 and d.get("跌破首阳最低价", 0) >= 1, "调整期最低价跌破首阳最低价：作废")
    ev, _, _ = events_of(build(MAIN, adj=[(0.98, 1.0, 0.5), (0.97, 1.0, 0.4), (0.98, 1.0, 0.4)]))
    ck(len(ev) == 1, "调整期最低价正好等于首阳最低价：不算跌破")
    c = pb(**{"adjust.floor": "open"})
    ev, _, _ = events_of(build(MAIN, adj=[(0.98, 1.0, 0.5), (0.97, 1.0, 0.4), (0.98, 1.0, 0.4)]), c)
    ck(len(ev) == 0, "floor 改成 open：跌到首阳最低价（低于开盘价）就作废")


def check_trigger_rules() -> None:
    print("[二次进攻]")
    ev, _, d = events_of(build(MAIN, t_gain=0.06))
    ck(len(ev) == 0 and any("不是大阳线" in k for k in d),
       "主板二次进攻 +6% 站上首阳高点但没涨停：不算（和首阳同一个大阳线定义）")
    ev, _, _ = events_of(build(MAIN, t_gain=0.06), pb(**{"trigger.big": "loose"}))
    ck(len(ev) == 1, "trigger.big=loose：+6% 的阳线站上首阳高点就算")
    ev, _, d = events_of(build(MAIN, t_vr=1.4))
    ck(len(ev) == 0 and any("量不够前日1.5倍" in k for k in d), "二次进攻量只有前一日 1.4 倍：不算")
    ev, op, d = events_of(build(MAIN, t_close_over=False))
    ck(len(ev) == 0, "收盘没站上首阳最高价：不是二次进攻")
    # 第一次站上首阳高点是考卷：量不够就作废，后面再来一根也不算
    ev, _, d = events_of(build(MAIN, t_vr=1.2, tail=[(1.10, 2.0)]))
    ck(len(ev) == 0, "第一次站上首阳高点时量不够，第二天再放量涨停也不补算")
    ev, _, _ = events_of(build(CYB, t_gain=0.105))
    ck(len(ev) == 1, "创业板二次进攻 +10.5%：算")
    ev, _, _ = events_of(build(CYB, t_gain=0.09))
    ck(len(ev) == 0, "创业板二次进攻 +9%：不到 10%")


def check_bonus_and_rank() -> None:
    print("[加分项 / 排序]")
    # 调整期最低价压在首阳开盘价（昨收 × 1.01）和首阳最低价（昨收 × 1.005）之间
    ev, _, _ = events_of(build(MAIN, t_vr=3.0, adj=[(0.985, 1.001, 0.6), (0.975, 1.002, 0.45),
                                                     (0.98, 1.003, 0.4)]))
    r = ev.iloc[0]
    # 调整最后一天量 = 首阳 × 0.4，T = 其 3 倍 = 1.2 × 首阳
    ck(bool(r["bonus_vol"]), "二次进攻量 1.2 倍首阳：量超首阳 ✓")
    ck(bool(r["bonus_half"]), "调整最低量 40%：缩到一半以下 ✓")
    ck(not bool(r["bonus_open"]), "调整最低价低于首阳开盘价：守开盘价 ✗")
    ev2, _, _ = events_of(build(MAIN, t_vr=1.6))
    ck(not bool(ev2.iloc[0]["bonus_vol"]), "二次进攻量 0.64 倍首阳：量超首阳 ✗")
    ck(bool(ev2.iloc[0]["bonus_open"]), "调整期最低价一直在首阳开盘价之上：守开盘价 ✓")
    a = pd.DataFrame([
        {"code": "000001", "bonus_vol": False, "bonus_half": True, "bonus_open": False,
         "vol_vs_launch": 2.0, "gain_pct": 10.0},
        {"code": "000002", "bonus_vol": True, "bonus_half": True, "bonus_open": False,
         "vol_vs_launch": 1.1, "gain_pct": 10.0},
        {"code": "000004", "bonus_vol": True, "bonus_half": True, "bonus_open": False,
         "vol_vs_launch": 1.5, "gain_pct": 10.0},
    ])
    ck(list(P.rank(a)["code"]) == ["000004", "000002", "000001"],
       "排序：加分项个数 -> 今日量/首阳量 -> 涨幅")
    ck(len(P.rank(pd.DataFrame())) == 0, "空表排序不炸")


def check_share_jump() -> None:
    print("[股本跳变]")
    # 送转那天流通股本翻倍：首阳在第 70 根，股本在调整期第 2 天翻倍
    ev, _, d = events_of(build(MAIN, os_jump_at=72))
    ck(len(ev) == 0 and d.get("股本跳变", 0) >= 1, "调整期里流通股本翻倍（送转）：前后成交量不可比，作废")
    ev, _, _ = events_of(build(MAIN, os_jump_at=70))
    ck(len(ev) == 0, "首阳当天股本跳变：不算首阳")
    ev, _, _ = events_of(build(MAIN, os_jump_at=30))
    ck(len(ev) == 1, "横盘期里的股本跳变不管（只会让横盘均量偏大，保守的一侧）")


def check_suspension() -> None:
    print("[停牌]")
    # 两只票同一段日期：A 走标准形态，B 是陪跑的（让「全市场开市日」有参照）
    a = build(MAIN)
    b = build("600000", t_gain=False, adj=[(0.99, 1.0, 0.5)] * 4)
    both = pd.concat([a, b], ignore_index=True)
    ev, _, _ = events_of(both)
    ck(len(ev[ev["code"] == MAIN]) == 1, "两只票都在：A 的标准形态成立（对照组）")
    adj_day = a["date"].iloc[72]                 # 首阳在第 70 根，调整第 2 天
    ev, _, d = events_of(both[~((both["code"] == MAIN) & (both["date"] == adj_day))])
    ck(len(ev[ev["code"] == MAIN]) == 0 and d.get("形态段里停过牌", 0) >= 1,
       "调整期里 A 停了一天（B 那天照常交易）：停牌不是缩量调整，这一段作废")
    pre = a["date"].iloc[69]                     # 首阳前一天
    ev, _, d = events_of(both[~((both["code"] == MAIN) & (both["date"] == pre))])
    ck(len(ev[ev["code"] == MAIN]) == 0 and d.get("首阳是复牌第一天", 0) >= 1,
       "首阳前一天 A 停牌：复牌第一天的「倍量」比的是停牌前那根，不算首阳")
    ev, _, _ = events_of(both[both["date"] != adj_day])
    r = ev[ev["code"] == MAIN]
    ck(len(r) == 1 and int(r.iloc[0]["adjust_days"]) == 2,
       "那一天两只都没有（节假日全市场休市）：不算停牌，形态照常成立，调整按交易日数 2 天")
    # 日线表早年只有少数票有数据（真实表 2023 年初两三百只、之后五千多只）：
    # 拿全表行数的中位数定开市日，早年那段全被当成休市，停牌识别不出来
    late = []
    for k, code in enumerate(["600004", "600006", "600007", "600008",
                              "600009", "600010", "600011", "600012"]):
        dd = _dates(260)[90:]
        late.append(pd.DataFrame({"code": code, "date": dd, "open": 10.0, "high": 10.1,
                                  "low": 9.9, "close": 10.0 + 0.01 * (k % 3),
                                  "volume": 1e6, "outstanding_share": 1e8}))
    sparse = pd.concat([both] + late, ignore_index=True)
    ev, _, _ = events_of(sparse)
    ck(len(ev[ev["code"] == MAIN]) == 1, "再加 8 只后来才有数据的票：A 的标准形态照常成立")
    ev, _, d = events_of(sparse[~((sparse["code"] == MAIN) & (sparse["date"] == adj_day))])
    ck(len(ev[ev["code"] == MAIN]) == 0 and d.get("形态段里停过牌", 0) >= 1,
       "早年只有 2 只票有数据时 A 停一天，照样识别成停牌（开市日按在市票数算，不按全表中位数）")


def check_open_patterns() -> None:
    print("[进行中]")
    ev, op, _ = events_of(build(MAIN, t_gain=False))
    ck(len(ev) == 0 and len(op) == 1 and int(op.iloc[0]["adjust_days"]) == 3,
       "首阳后调整 3 天、数据到头：进行中，调整天数 3")
    ev, op, _ = events_of(build(MAIN, adj=[], t_gain=False))
    ck(len(op) == 1 and int(op.iloc[0]["adjust_days"]) == 0, "今天刚出首阳：进行中，调整 0 天")
    ev, op, _ = events_of(build(MAIN, adj=[(0.97, 1.01, 0.5)] * 11, t_gain=False))
    ck(len(op) == 0, "调整已经 11 天、数据到头：过期，不算进行中")

    print("[清单 B：二次进攻前]")
    ev, op, _ = events_of(build(MAIN, t_gain=False))
    r = op.iloc[0]
    ck(bool(r["b_ok"]) and int(r["wait_left"]) == pb()["adjust"]["max_days"] - 3 + 1,
       "缩量调整 3 天、均量 48%：进清单 B，还能等 max_days - 3 + 1 天")
    ck(r["launch_high"] > r["close"] and r["to_high_pct"] > 0 and r["base_run"] >= 60
       and r["adj_vol_mean"] < 0.8, "清单 B 带突破价（首阳最高价）、离突破多少、前期调整天数、均量")
    ev, op, _ = events_of(build(MAIN, adj=[(0.985, 1.01, 0.6)], t_gain=False))
    ck(not bool(op.iloc[0]["b_ok"]), "只调整了 1 天：还没走完缩量调整，不进清单 B")
    ev, op, _ = events_of(build(MAIN, adj=[(0.99, 1.01, 0.9), (0.99, 1.01, 0.85)], t_gain=False))
    ck(not bool(op.iloc[0]["b_ok"]), "调整 2 天但均量 88% > 80%：不进清单 B")
    bh: list = []
    a = build(MAIN)
    P.find_events(P.prepare(a, pb()), pb(), b_hist=bh)
    ck(len(bh) == 1 and bh[0]["status"] == "成立" and bh[0]["b_date"] == a["date"].iloc[72],
       "走完二次进攻的形态在 b_hist 里：第 2 个合格调整日进的 B，结局「成立」")
    bh2: list = []
    x2 = P.prepare(pd.concat([build(MAIN), build("600000", t_gain=False),
                              build(CYB, adj=[(0.985, 1.01, 0.6), (0.97, 0.99, 0.4)],
                                    t_gain=False)], ignore_index=True), pb())
    P.find_events(x2, pb(), b_hist=bh2)
    st_ = {r["code"]: r["status"] for r in bh2}
    ck(st_.get(MAIN) == "成立" and st_.get("600000") == "进行中" and CYB not in st_,
       "b_hist：成立 / 还在调整的都记；第 2 天就跌破首阳最低价的没进过 B，不记")
    bs = P.b_stats(bh2, x2, set(), 60)
    ck(bs["n"] == 1 and bs["to_a"] == 1 and bs["rate"] == 1.0,
       "转化率只数已经有结果的（还在调整的不算分母）")


def check_forward() -> None:
    print("[之后怎么走的]")
    a = build(MAIN, tail=[(1.05, 1.0), (0.97, 1.0), (1.02, 1.0)])
    b = build("600000")
    x = P.prepare(pd.concat([a, b], ignore_index=True), pb())
    ev, _, _ = P.find_events(x, pb())
    f = P.forward(x, ev, bars=20)
    r = f[f["code"] == MAIN].iloc[0]
    ck(int(r["n_after"]) == 3, "之后只有 3 根就只算 3 根，不串到下一只票")
    ck(r["max_up_pct"] > 5 and abs(r["ret_pct"] - round(100 * (1.05 * 0.97 * 1.02 - 1), 1)) < 0.3,
       "之后最高涨幅、最后一根收盘涨幅按事件日收盘算")
    st = P.history_stats(f, x, bars=20, base_days=60)
    ck(st["n"] == len(f) and st["n_final"] == 0, "走不满 20 根的不进「之后」统计")
    # 频率的分母只数全市场覆盖齐了之后：早年只有 A、B 两只，后来 6 只才有数据
    late = [pd.DataFrame({"code": code, "date": _dates(300)[100:], "open": 10.0,
                          "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e6})
            for code in ["600004", "600006", "600007", "600008", "600009", "600010"]]
    x2 = P.prepare(pd.concat([a, b] + late, ignore_index=True), pb())
    st2 = P.history_stats(f, x2, bars=20, base_days=60)
    ck(st2["from"] == _dates(300)[160] and st2["days"] == 140 and st2["n"] == 0,
       "早年只有两只票有数据：频率从后来 6 只都有 60 根横盘那天起算（140 天），之前的事件不数")


def check_base_run() -> None:
    print("[前期调整天数]")
    a = build(MAIN)                              # 横盘 70 根就是数据的开头
    x = P.prepare(a, pb())
    ev, _, _ = P.find_events(x, pb())
    r = ev.iloc[0]
    ck(int(r["base_run"]) == 70 and bool(r["base_run_full"]),
       "横盘从数据第一根就开始：70 个交易日，标「至少」")
    # 前面再加 30 根高一倍的价：横盘从出界那根的下一根算起
    pre = a.iloc[:30].copy()
    pre[["open", "high", "low", "close"]] *= 2.0
    pre["date"] = pd.bdate_range(end=pd.Timestamp(a["date"].iloc[0]) - pd.Timedelta(days=1),
                                 periods=30).strftime("%Y-%m-%d")
    x2 = P.prepare(pd.concat([pre, a], ignore_index=True), pb())
    ev2, _, _ = P.find_events(x2, pb())
    r2 = ev2.iloc[0]
    ck(int(r2["base_run"]) == 70 and not bool(r2["base_run_full"]),
       "前面有一段出了振幅上限：只数到出界为止，不标「至少」")
    ck(E.base_text({"base_run": 70, "base_run_full": True}, html=False) == "至少 70 个交易日"
       and "约 3.3 个月" in E.base_text({"base_run": 70}), "显示「至少 N 个交易日 / 约 M 个月」")
    ck(E.base_text({}) == "-", "老产物没有这一列：显示 -")


def check_corp_events() -> None:
    """减持 / 定增判定：只读缓存的纯函数，造公告测。不联网。"""
    print("[股东减持 / 定增]")
    RED = "001002007004003"
    asof = "2026-09-24"

    def ann(date, node, cats, title, art=None):
        return {"art": art or f"{date}{title[:6]}", "date": date, "node": node,
                "cats": cats, "title": "公司:" + title}

    c = {"ann": [ann("2026-06-26", "7", [RED], "关于股东减持股份的预披露公告")], "rec": []}
    ck(len(CE.reduce_hits(c, asof, 90)) == 1, "第 90 天那条减持公告算（含当天往前 90 天）")
    c = {"ann": [ann("2026-06-25", "7", [RED], "关于股东减持股份的预披露公告")], "rec": []}
    ck(not CE.reduce_hits(c, asof, 90), "第 91 天的不算")
    c = {"ann": [ann("2026-09-25", "7", [RED], "关于股东减持股份的预披露公告")], "rec": []}
    ck(not CE.reduce_hits(c, asof, 90), "事件日之后才公告的不算（回看历史不偷看）")
    c = {"ann": [ann("2026-09-20", "7", [RED], "关于回购股份集中竞价减持的进展公告"),
                 ann("2026-09-20", "7", [RED], "关于股东减持计划期限届满未实施减持的公告"),
                 ann("2026-09-20", "7", [], "关于控股股东承诺不减持公司股份的公告")], "rec": []}
    ck(not CE.reduce_hits(c, asof, 90), "公司卖回购股、「未实施减持」「不减持」都不算")
    c = {"ann": [], "rec": [{"date": "2026-09-01", "holder": "某投资", "free_ratio": 1.1,
                             "start": "2026-08-25", "end": "2026-08-29"}]}
    ck(len(CE.reduce_hits(c, asof, 90)) == 1, "交易所减持记录（没有单独公告的）也算")

    plan = ann("2026-05-30", "2", [CE.PLAN], "2026年度向特定对象发行A股股票预案", "P1")
    ck(CE.pending_plan({"ann": [plan]}, asof, 730)["art"] == "P1", "有预案、没发完没终止：还在走流程")
    for extra, why in [
            (ann("2026-08-01", "2", [CE.SEO + "005001"], "向特定对象发行股票发行结果暨股本变动公告"),
             "之后有发行结果：发完了"),
            (ann("2026-08-01", "2", [CE.SEO + "006001"], "向特定对象发行股票上市公告书"),
             "之后有上市公告书：发完了"),
            (ann("2026-08-01", "2", [CE.SEO + "007001"], "以简易程序向特定对象发行股票竞价结果的公告"),
             "简易程序出了竞价结果：价格定了"),
            (ann("2026-08-01", "2", [CE.STOP], "关于终止2026年度向特定对象发行A股股票事项的公告"),
             "之后终止了"),
            (ann("2026-08-01", "2", [CE.SEO + "007002"], "关于撤回向特定对象发行股票申请文件的公告"),
             "之后撤回了")]:
        ck(CE.pending_plan({"ann": [plan, extra]}, asof, 730) is None, why + " -> 不算在走流程")
    ck(CE.pending_plan({"ann": [ann("2026-05-30", "2", [CE.PLAN], "2026年第二次临时股东会决议公告")]},
                       asof, 730) is None, "东财也给股东会决议打「增发预案」：没有方案文件就不算")
    ck(CE.pending_plan({"ann": [ann("2026-04-28", "2", [CE.PLAN],
                                    "关于提请股东大会授权董事会办理以简易程序向特定对象发行股票的公告")]},
                       asof, 730) is None, "年度股东会例行的「授权简易程序定增」不是一笔定增")
    ck(CE.pending_plan({"ann": [ann("2026-01-05", "2", [CE.REVISE],
                                    "关于调整公司向不特定合格投资者公开发行股票并在北交所上市方案的公告")]},
                       asof, 730) is None, "北交所上市的公开发行方案不是定增")
    ck(CE.pending_plan({"ann": [dict(plan, date="2024-09-23")]}, asof, 730) is None,
       "730 天前的预案不看")
    ck(CE.pending_plan({"ann": [dict(plan, date="2026-09-25")]}, asof, 730) is None,
       "事件日之后才出的预案不看（不偷看）")
    old = dict(plan, date="2025-03-01", art="P0")
    done = ann("2025-06-01", "2", [CE.SEO + "006001"], "向特定对象发行股票上市公告书")
    ck(CE.pending_plan({"ann": [old, done, plan]}, asof, 730)["art"] == "P1",
       "上一轮发完了、又出了新预案：按新的一轮算")
    rev = ann("2026-07-10", "2", [CE.REVISE], "2026年度向特定对象发行A股股票预案(修订稿)", "P2")
    brief = ann("2026-07-10", "2", [CE.REVISE], "2026年度向特定对象发行A股股票预案(修订稿)(摘要)", "P3")
    ck(CE.pending_plan({"ann": [plan, brief, rev]}, asof, 730)["art"] == "P2",
       "定价看最新一份修订稿，同一天优先全文不看摘要")

    # 定价方式：造几页正文，走 fetch_pricing 的真实判定
    pages = {
        "bid": ["一、本次向特定对象发行股票的定价基准日为发行期首日，发行价格不低于"],
        "lock": ["本次发行的定价基准日为公司第三届董事会第十次会议决议公告日，"],
        "fixed": ["本次发行股票数量按照募集资金总额除以发行价格34.12元/股确定"],
        "table": ["定价基准日      指   上市公司第十届董事会\n第十八次会议决议公告日"],
        "both": ["购买资产的定价基准日为公司第十届董事会第十八次会议决议公告日", "……",
                 "募集配套资金的定价基准日为本次向特定对象发行股票的发行期首日"],
        "none": ["发行价格由股东大会授权董事会与主承销商协商确定"],
    }
    real_get = CE._get

    def fake_get(url, params, tries=3):
        pg = pages[params["art_code"]]
        i = int(params["page_index"])
        return {"data": {"notice_content": pg[i - 1] if i <= len(pg) else "", "page_size": len(pg)}}

    try:
        CE._get = fake_get
        got = {k: CE.fetch_pricing(k) for k in pages}
    finally:
        CE._get = real_get
    ck(got == {"bid": "bid", "lock": "lock", "fixed": "lock", "table": "lock", "both": "bid",
               "none": "unknown"},
       f"定价方式：竞价 / 锁价 / 写死价格 / 表格里的锁价 / 购买资产+配套募资竞价 / 认不出（{got}）")

    # 起涨预测用的全市场版：拉一次全市场，判据还是 reduce_hits
    mk = [dict(ann("2026-09-10", "7", [RED], "关于持股5%以上股东减持股份结果的公告"), code="600004"),
          dict(ann("2026-09-11", "7", [RED], "关于回购股份集中竞价减持的进展公告"), code="600006"),
          dict(ann("2026-09-12", "7", [RED], "关于股东减持股份的预披露公告"), code="600007")]
    recs = [{"code": "600008", "date": "2026-09-20", "holder": "某基金", "free_ratio": 0.5,
             "start": "2026-09-15", "end": "2026-09-18"},
            {"code": "600009", "date": "2026-09-26", "holder": "某基金", "free_ratio": 0.5,
             "start": "2026-09-22", "end": "2026-09-25"}]
    keep_m = (CE.fetch_ann_market, CE.fetch_rec)
    try:
        CE.fetch_ann_market = lambda node, b, e, max_pages=200: [a for a in mk if b <= a["date"] <= e]
        CE.fetch_rec = lambda code, since: [r for r in recs if r["date"] >= since]
        got = CE.recent_reduction_reasons(["600004", "600006", "600008", "600009"], asof, 30)
    finally:
        CE.fetch_ann_market, CE.fetch_rec = keep_m
    ck(set(got) == {"600004", "600008"},
       "全市场版：减持公告、减持记录都认；回购股不算、不在候选里的不管、事件日之后的不看"
       f"（{sorted(got)}）")

    with tempfile.TemporaryDirectory() as td:
        keep = CE.CACHE
        try:
            CE.CACHE = Path(td)
            ck(CE.verdict("600004", asof)[0] == "unknown", "没有缓存：unknown，不当成「没问题」")
            (Path(td) / "600004.json").write_text(json.dumps(
                {"code": "600004", "from": "2000-01-01", "through": asof, "ann": [plan], "rec": []}),
                encoding="utf-8")
            ck(CE.verdict("600004", asof)[0] == "unknown", "定增还在走、定价方式没拉到：unknown")
            for kind, want, word in [("bid", "out", "竞价"), ("unknown", "out", "认不出"),
                                     ("lock", "ok", "")]:
                (Path(td) / "_pricing.json").write_text(json.dumps({"P1": {"kind": kind}}),
                                                        encoding="utf-8")
                v = CE.verdict("600004", asof)
                ck(v[0] == want and word in v[1],
                   f"定价 {kind} -> {want}" + ("（价格已定是用户说的例外）" if kind == "lock" else ""))
            (Path(td) / "600004.json").write_text(json.dumps(
                {"code": "600004", "from": "2025-01-01", "through": asof, "ann": [], "rec": []}),
                encoding="utf-8")
            ck(CE.verdict("600004", asof)[0] == "unknown", "缓存没覆盖到往前 730 天：unknown")
        finally:
            CE.CACHE = keep


def check_rules_text() -> None:
    print("[规则文案]")
    t = "\n".join(P.rules_text(pb()))
    ck("60 个交易日" in t and "≤ 1.30" in t and "3 倍" in t, "横盘三项阈值从 config 现拼")
    ck("2~10 个交易日" in t and "80%" in t, "调整天数和均量上限从 config 现拼")
    t2 = "\n".join(P.rules_text(pb(**{"adjust.max_days": 7, "base.amp_max": 0.25})))
    ck("2~7 个交易日" in t2 and "≤ 1.25" in t2, "改阈值文案跟着变，不用改代码")
    t3 = "\n".join(P.rules_text(pb(**{"trigger.big": "loose"})))
    ck("≥5% 的阳线" in t3, "trigger.big=loose 时文案说清楚放宽了")
    ck("剔除（两份清单都剔）" in t and "90 天内有股东减持" in t and "价格没定的定增" in t,
       "剔除规则（减持 / 定增）印在规则里，天数从 config 现拼，两份清单都剔")
    ck("清单 B（二次进攻前）" in t and "2 天以上" in t, "清单 B 的口径印在规则里")


def check_config() -> None:
    print("[配置不变量]")
    c = pb()
    a = c["adjust"]
    ck(1 <= a["min_days"] <= a["max_days"], "调整天数 1 ≤ min ≤ max")
    ck(0 < c["base"]["amp_max"] < 1, "横盘振幅上限在 (0, 1)")
    ck(c["launch"]["vol_ratio_min"] >= 1 and c["trigger"]["vol_ratio_min"] >= 1, "倍量阈值 ≥ 1")
    ck(0 < a["vol_mean_max"] <= a["vol_day_max"], "均量上限 ≤ 单日上限（均值不可能大于最大值）")
    ck(str(a["floor"]) in ("low", "open") and str(c["trigger"]["big"]) in ("same", "loose"),
       "floor / trigger.big 只认那两个取值")
    hh, mm, ss = (int(v) for v in str(c["send_at"]).split(":"))
    ck((hh, mm) >= (15, 5), "发信时刻在收盘之后")
    ex = c.get("exclude") or {}
    ck(int(ex.get("reduce_days", 0)) > 0 and int(ex.get("placement_days", 0)) >= 365,
       "剔除窗口：减持天数 > 0，定增预案往前至少看一年")


def check_products() -> None:
    """stage_scan / stage_send 全走一遍，产物写临时目录。"""
    print("[产物 + 发信接线]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 002652 这种前导零的代码：read_json 会读成 2652（教训 29）
        zero = "002652"
        a = build(zero)
        # 进行中：调整 4 天、没有二次进攻，总根数和 a 一样，最后一根正好是目标日
        b = build(CYB, t_gain=False, adj=[(0.985, 1.01, 0.6), (0.975, 1.005, 0.45),
                                          (0.98, 1.008, 0.4), (0.97, 1.01, 0.35)])
        st = build("000010")                    # ST 那只：名字带 ST 要剔掉
        # 另外三只形态也成立，但按用户 2026-09-28 的规则要剔：近期减持 / 价格没定的定增 /
        # 数据没拉到（宁可为空）
        red, seo, unk = build("600004"), build("600006"), build("600007")
        # 清单 B 的另两只：调整 4 天但近期有减持（要剔）；只调整了 1 天（不够格）。
        # 两只的最后一根都要落在目标日，才是「当天还在等二次进攻」的候选
        four = [(0.985, 1.01, 0.6), (0.975, 1.005, 0.45), (0.98, 1.008, 0.4), (0.97, 1.01, 0.35)]
        b_red = build("600008", t_gain=False, adj=four)
        b_one = build("600009", t_gain=False, adj=[(0.985, 1.01, 0.6)])
        b_one["date"] = a["date"].iloc[-len(b_one):].to_numpy()
        allx = pd.concat([a, b, st, red, seo, unk, b_red, b_one], ignore_index=True)
        dp = td / "daily.parquet"
        allx.to_parquet(dp, index=False)
        target = a["date"].iloc[-1]
        day = lambda k: (dt.date.fromisoformat(target) + dt.timedelta(days=k)).isoformat()  # noqa: E731
        corp = td / "corp"
        corp.mkdir()

        def fake(code, ann, rec=()):
            (corp / f"{code}.json").write_text(json.dumps(
                {"code": code, "from": "2000-01-01", "through": target, "ts": 0,
                 "ann": list(ann), "rec": list(rec)}, ensure_ascii=False), encoding="utf-8")

        RED = "001002007004003"
        # 标准形态那只：有减持字样但都不该算 + 一笔已经发完的定增
        fake(zero, [
            {"art": "z1", "date": day(-5), "node": "7", "cats": [RED],
             "title": "公司:关于回购股份集中竞价减持的进展公告"},
            {"art": "z2", "date": day(-20), "node": "7", "cats": [RED],
             "title": "公司:关于股东减持计划期限届满未实施减持的公告"},
            {"art": "z3", "date": day(-120), "node": "7", "cats": [RED],
             "title": "公司:关于持股5%以上股东减持股份的预披露公告"},
            {"art": "z4", "date": day(-200), "node": "2", "cats": [CE.PLAN],
             "title": "公司:2025年度向特定对象发行A股股票预案"},
            {"art": "z5", "date": day(-30), "node": "2", "cats": [CE.SEO + "005001"],
             "title": "公司:向特定对象发行股票发行情况报告书"}])
        fake("000010", [])
        fake("600004", [{"art": "r1", "date": day(-10), "node": "7", "cats": [RED],
                         "title": "公司:关于持股5%以上股东减持股份的预披露公告"}])
        fake("600006", [{"art": "s1", "date": day(-60), "node": "2", "cats": [CE.PLAN],
                         "title": "公司:2026年度向特定对象发行A股股票预案"}])
        fake(CYB, [])
        fake("600008", [{"art": "r2", "date": day(-3), "node": "7", "cats": [RED],
                         "title": "公司:关于控股股东减持股份计划的公告"}])
        (corp / "_pricing.json").write_text(json.dumps({"s1": {"kind": "bid"}}), encoding="utf-8")
        keep = (P.OUT, P.DATA_DIR, P.DAILY, P.names_for, ds.trade_dates, E._send, E._conf,
                CE.CACHE, CE.prefetch)
        sent: list = []
        try:
            P.OUT, P.DATA_DIR, P.DAILY = td / "out", td / "data", dp
            CE.CACHE, CE.prefetch = corp, (lambda *a, **k: {})   # 不联网：只读假缓存
            P.names_for = lambda cs: {c: ("*ST测试" if c == "000010" else f"名<b>{c}")
                                      for c in cs}
            ds.trade_dates = lambda: set(allx["date"])
            os.environ.pop("DRY_RUN", None)
            rc = P.stage_scan(target)
            ck(rc == 0, "stage_scan 退出码 0")
            meta = json.loads((P.OUT / "run_meta.json").read_text(encoding="utf-8"))
            sel = json.loads((P.OUT / "selected.json").read_text(encoding="utf-8"))
            ck(meta["date"] == target and meta["n"] == 1 and not meta["dry"], "run_meta 日期是目标日、非试跑")
            ck([r["code"] for r in sel] == [zero], "清单只有标准形态那只，ST 被剔掉")
            ex = meta["excluded"]
            ck("000010" in ex, "剔除原因落进 run_meta")
            ck("减持" in ex.get("600004", "") and "定增" in ex.get("600006", "")
               and "竞价" in ex.get("600006", "") and "没法核" in ex.get("600007", ""),
               "近 90 天减持剔、竞价中的定增剔、数据没拉到也剔（宁可为空），原因写明")
            ck(zero not in ex, "回购股减持、「未实施减持」、90 天前的减持、已发完的定增都不剔")
            ck(meta["n_pattern"] == 5 and meta["n"] == 1,
               "形态成立 5 只（ST 1 + 规则剔 3 + 入选 1），清单 1 只")
            ck(not (P.OUT / "watch.json").exists(), "旧的 watch.json 不再出（清单 B 用 list_b.json）")
            bl = json.loads((P.OUT / "list_b.json").read_text(encoding="utf-8"))
            ck([r["code"] for r in bl] == [CYB] and meta["n_b"] == 1 and meta["n_b_pattern"] == 2,
               "清单 B 只有走完前三步那只；调整 1 天的不够格，近期减持的剔掉")
            ck("减持" in meta["excluded_b"].get("600008", ""), "清单 B 也按同样的规则剔，原因写明")
            ck((P.OUT / f"{P.NAME}B.txt").read_bytes() == (CYB + "\r\n").encode("gbk")
               and list(pd.read_parquet(P.DATA_DIR / target[:7] / f"pullback_b_{target}.parquet")
                        ["code"]) == [CYB], "清单 B 的同花顺 txt 和每日清单也落盘")
            ck(sel[0].get("base_run", 0) >= 60, "入选的票带前期调整天数（>= 横盘检查的 60 根）")
            pq = pd.read_parquet(P.DATA_DIR / target[:7] / f"pullback_{target}.parquet")
            ck(list(pq["code"]) == [zero], "每日清单入库 data/pullback/YYYY-MM/")
            txt = (P.OUT / f"{P.NAME}.txt").read_bytes()
            ck(txt == (zero + "\r\n").encode("gbk"), "同花顺 txt 是 GBK + CRLF")

            # 日期对不上不发
            meta2 = dict(meta, date="2000-01-01")
            (P.OUT / "run_meta.json").write_text(json.dumps(meta2), encoding="utf-8")
            ck(P.stage_send(target, wait=False) == 1, "run_meta 不是目标日的：不发信、退出码 1")
            (P.OUT / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

            # SKIP_MAIL：面板照出，不发不落戳
            os.environ["SKIP_MAIL"] = "1"
            E._send = lambda m, c: sent.append(m)
            E._conf = lambda: {"user": "bot@example.com", "to": ["me@example.com"]}
            ck(P.stage_send(target, wait=False) == 0 and not sent
               and not (P.OUT / "mail_sent.json").exists(), "SKIP_MAIL=1：不发信、不落 mail_sent.json")
            html = (P.OUT / "panel.html").read_text(encoding="utf-8")
            ck(zero in html and "名&lt;b&gt;" in html and "名<b>" not in html,
               "面板带前导零的代码，名称转义")
            ck("stamp-pullback.txt" in html and (P.OUT / "stamp.txt").exists(), "面板自刷新接到自己的 stamp")
            ck(not re.search(r"__[A-Z]+__", html) and "LAGOK=('true'" in html,
               "面板里没有没替换的占位符；LAGOK=true（面板日期本来就落后今天，不报过期）")
            ck("调整中" not in html and "<details>" in html and "前期调整" in html
               and "缩量调整" in html and "剔除 4 只：" in html and "600004" in html,
               "面板：没有观察名单，历史收起来，有前期调整 / 缩量调整两列，剔了谁写一行")
            ck("清单 B · 二次进攻前" in html and "复制清单 B" in html and CYB in html
               and "离突破" in html and "历史上进过清单 B" in html,
               "面板有清单 B：离突破多少、还能等几天、进过 B 的后来走完多少")
            os.environ.pop("SKIP_MAIL", None)

            ck(P.stage_send(target, wait=False) == 0 and len(sent) == 1, "真发信走 _send 一次")
            ms = json.loads((P.OUT / "mail_sent.json").read_text(encoding="utf-8"))
            ck(ms["date"] == target and ms["n"] == 1, "发出去之后才落 mail_sent.json，日期对得上")
            m = sent[0]
            ck(m["Subject"].startswith(f"[{P.NAME}] {target} · 清单A 1只 · 清单B 1只"),
               "邮件标题带线名、日期、两份清单的只数")
            body = m.get_body(("html",)).get_content()
            ck(zero in body and "名<b>" not in body, "邮件正文前导零、名称转义")
            ck(not re.search(r"__[A-Z]+__", body) and "<script" not in body,
               "邮件里没有占位符、没有 script（邮件客户端会剥掉）")
            ck("前期调整" in body and "个交易日" in body and "剔除 4 只：" in body
               and "调整中" not in body, "邮件：带前期调整天数、剔了谁，不提观察名单")
            plain = m.get_body(("plain",)).get_content()
            ck("前期调整" in plain and "缩量调整" in plain, "纯文本那份也带两个调整天数")
            ck("清单 B" in body and CYB in body and f"[B] {CYB}" in plain,
               "邮件（HTML 和纯文本）都有清单 B")
            names = {p.get_filename() for p in m.iter_attachments()}
            ck({f"{P.NAME}.txt", f"{P.NAME}B.txt"} <= names, "两份非空清单各附一个同花顺 txt")
            # 补发说明：发信晚了 15 分钟以上邮件顶部写明
            sent.clear()
            real_now = P.now_bj
            P.now_bj = lambda: P.send_time(target, P.cfg()) + dt.timedelta(hours=3)
            P.stage_send(target, wait=True)
            P.now_bj = real_now
            b2 = sent[0].get_body(("html",)).get_content()
            ck("没开机，补发于" in b2, "晚于 17:58 十五分钟以上：邮件顶部写「补发于」")
            ck("补发于" in (P.OUT / "panel.html").read_text(encoding="utf-8"),
               "真补发时面板也写明")
            os.environ["SKIP_MAIL"] = "1"
            P.now_bj = lambda: P.send_time(target, P.cfg()) + dt.timedelta(days=3)
            P.stage_send(target, wait=True)
            P.now_bj = real_now
            os.environ.pop("SKIP_MAIL", None)
            ck("补发" not in (P.OUT / "panel.html").read_text(encoding="utf-8"),
               "SKIP_MAIL 事后重出面板：没发信就不写「补发于」")
        finally:
            (P.OUT, P.DATA_DIR, P.DAILY, P.names_for, ds.trade_dates, E._send, E._conf,
             CE.CACHE, CE.prefetch) = keep
            os.environ.pop("SKIP_MAIL", None)

    # 空榜：空文件、空榜文案
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        x = build(MAIN, t_gain=False)
        dp = td / "daily.parquet"
        x.to_parquet(dp, index=False)
        target = x["date"].iloc[-1]
        keep = (P.OUT, P.DATA_DIR, P.DAILY, P.names_for, ds.trade_dates, CE.CACHE, CE.prefetch)
        try:
            P.OUT, P.DATA_DIR, P.DAILY = td / "out", td / "data", dp
            # 这只调整了 3 天，是清单 B 的候选：也要不联网（空缓存 -> 没法核 -> 剔）
            CE.CACHE, CE.prefetch = td / "corp", (lambda *a, **k: {})
            P.names_for = lambda cs: {}
            ds.trade_dates = lambda: set(x["date"])
            ck(P.stage_scan(target) == 0, "空榜也退出码 0")
            ck((P.OUT / f"{P.NAME}.txt").read_bytes() == b"", "空榜 txt 是 0 字节，不留上次的代码")
            m0 = json.loads((P.OUT / "run_meta.json").read_text(encoding="utf-8"))
            ck("没法核" in m0["excluded_b"].get(MAIN, "") and m0["n_b"] == 0
               and (P.OUT / f"{P.NAME}B.txt").read_bytes() == b"",
               "清单 B 的候选数据没拉到也剔（宁可为空），B 的 txt 也是 0 字节")
            pq = pd.read_parquet(P.DATA_DIR / target[:7] / f"pullback_{target}.parquet")
            ck(len(pq) == 0, "空榜也落一个空的每日清单")
            html = E.build_html(target, [], [], json.loads(
                (P.OUT / "run_meta.json").read_text(encoding="utf-8")))
            ck("这是常态，不是故障" in html and "今天没有走完前三步" in html,
               "空榜邮件写明「这是常态，不是故障」，清单 B 空也写一句")
            ck(E.subject(target, [], []).endswith("清单A 0只 · 清单B 0只"), "空榜标题写两份都是 0 只")
            ck(P.stage_scan("2000-01-03") == 0 and json.loads(
                (P.OUT / "run_meta.json").read_text(encoding="utf-8"))["date"] == target,
               "非交易日 scan 直接退出 0，不动产物")
        finally:
            (P.OUT, P.DATA_DIR, P.DAILY, P.names_for, ds.trade_dates, CE.CACHE, CE.prefetch) = keep


def check_wait() -> None:
    print("[17:58]")
    c = pb()
    t = P.send_time("2026-09-28", c)
    ck(t.strftime("%Y-%m-%d %H:%M:%S") == "2026-09-28 17:58:00" and t.utcoffset() == dt.timedelta(hours=8),
       "发信时刻 = 目标日北京 17:58:00")
    t0 = dt.datetime.now()
    P.wait_until(P.now_bj() - dt.timedelta(minutes=5))
    ck((dt.datetime.now() - t0).total_seconds() < 1, "已经过点：立即放行，不等")


def check_single_impl() -> None:
    print("[单一实现]")
    src = (ROOT / "src" / "pullback_backtest.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {f"{n.func.value.id}.{n.func.attr}" for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)}
    ck("P.find_events" in calls and "P.prepare" in calls,
       "回测脚本调 pullback.prepare / find_events，不自己另写判据（教训 34）")
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    ck(not ({"find_events", "prepare", "is_launch"} & defs), "回测脚本里没有同名的第二份判定函数")
    ck("P.mark_risk" in calls and "P.history_pool" in calls,
       "回测脚本剔减持 / 定增走 pullback.mark_risk / history_pool，和每天的清单同一份")
    ps = (ROOT / "src" / "pullback.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(ps)) if isinstance(n, ast.FunctionDef) and n.name == "scan")
    used = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    ck({"mark_risk", "history_pool"} <= used, "生产扫描也走 mark_risk / history_pool")
    lr = (ROOT / "src" / "local_run.py").read_text(encoding="utf-8")
    t2 = ast.parse(lr)
    fp = next(n for n in ast.walk(t2) if isinstance(n, ast.FunctionDef) and n.name == "ensure_daily")
    names = {n.func.id for n in ast.walk(fp) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    ck("daily_ready" in names, "长期调整突破核日线用 daily_ready，和起涨预测同一份闸")


def main() -> int:
    import time
    t0 = time.time()
    check_board()
    check_positive()
    check_launch_rules()
    check_base_rules()
    check_adjust_rules()
    check_trigger_rules()
    check_bonus_and_rank()
    check_share_jump()
    check_suspension()
    check_open_patterns()
    check_forward()
    check_base_run()
    check_corp_events()
    check_rules_text()
    check_config()
    check_products()
    check_wait()
    check_single_impl()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {len(fails)} 个")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
