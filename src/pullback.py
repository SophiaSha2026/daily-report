"""
长期调整突破（内部代号 pullback / 形态线）。每个交易日收盘后扫描，北京 17:58 发清单。

    python src/pullback.py --stage scan                 扫描目标日，落产物
    python src/pullback.py --stage send                 等到 17:58 发信（过了就立即发）
    python src/pullback.py --stage scan --target 2026-09-24 --dry   扫历史某天，只打印

用户 2026-09-27 给的规则（原话要点，量化口径见下）
-----------------------------------------------
  1. 3 个月以上低成交量、窄幅震荡，某天突然倍量（≥ 前一交易日 1.5 倍）涨停
     （沪深主板）；或涨幅 ≥10% 且 1.5 倍量（科创板、创业板、北交所）。
     这一根是基准：第一根大阳线（下文「首阳」）。
  2. 随后两到三个交易日开始缩量调整（量能逐渐缩小，最好缩至首阳一半以下），
     股价不跌破首阳的最低点或开盘价。
  3. 调整一段时间后某天，再来一根倍量大阳线突破前高（量能最好超过首阳，
     并且当天收盘价超过首阳最高价），确认二次进攻 —— 这一天推荐。
  没有符合的，清单为空。

量化口径（全部阈值在 config.yaml 的 pullback 段）
------------------------------------------------
  横盘   首阳前 base.days=60 个交易日（约 3 个月）：
           收盘价 最高/最低 − 1 ≤ base.amp_max（30%）             「窄幅震荡」
           首阳量 ≥ 这 60 日均量 × base.vol_mean_mult（3 倍）      「低成交量」
           且首阳量大于这 60 日里任何一天（首阳是 3 个月来最大量）
         「低成交量」为什么这么量化：和「此前 120 日」比会漏掉长期地量的票
         （2026-09-27 实测 601326 横盘日均换手 0.5% 却因为更早更冷清被判不合格），
         和全市场比又会系统性排除小盘。用户的原意是「冷清很久、突然放量」，
         对首阳量说话最直接。
  首阳   主板（60/00 开头）涨停；创业板 30、科创板 68、北交所 8/4/9 涨幅 ≥10%。
         成交量 ≥ 前一交易日 × launch.vol_ratio_min（1.5）。
  调整   首阳后 adjust.min_days~max_days（2~10）个交易日，逐日：
           成交量 < 首阳量（缩量）；最低价 ≥ 首阳最低价（adjust.floor=low）；
           收盘 ≤ 首阳最高价（否则突破已经提前发生，不算调整）。
         整段均量 ≤ 首阳量 × adjust.vol_mean_max（0.8），在突破那天核。
  二次进攻  调整之后**第一个**收盘站上首阳最高价的交易日，必须同时：
           和首阳同一个大阳线定义（trigger.big=same：主板涨停 / 其余 ≥10%）；
           成交量 ≥ 前一交易日 × trigger.vol_ratio_min（1.5）。
         第一次站上但条件不够（量不够、涨幅不够），这一段形态就算用掉了，
         不会等下一次。
  加分   用户说「最好」的三项只排序、不决定入选：量超首阳、调整最低量 ≤ 首阳一半、
         调整期最低价不破首阳开盘价。

「最低点或开盘价」取最低点当硬门槛：开盘价 ≥ 最低价恒成立，「或」取宽的那个；
不破开盘价是更强的情形，放进加分项。

「第二根大阳线」取和首阳同一个定义（用户原话「以此为基准：第一根大阳线」
「再次来一根倍量大阳线」）。放宽成「涨幅 ≥5% 的阳线」是 trigger.big=loose，
2026-09-27 回测频率从每月约 1.9 次变成约 4.1 次（2023-08 起全市场覆盖齐了的 749 天）。

「量」一律是成交量（股），不是成交额：成交额受价格影响，涨停日天然更大，
拿它判「放大 1.5 倍」会系统性偏松（形态线 2026-08 就定的口径）。

数据从哪来
----------
data/breakout/daily.parquet：起涨预测那条线维护的全市场三年日线（新浪前复权 +
腾讯快照每日追加），本机收盘后 `backfill.py --stage update` 补到目标日。
这里只读。复权价的两个坑都在 prepare() 里处理：
  · 涨停判定：复权段（最近一次除权之前）的价格被整体缩放后再四舍五入到分，
    按昨收重算的涨停价会差 ±1 分。2026-09-27 全表实测：收在最高价、离涨停价
    「高 1 分」2634 次、「低 1 分」2524 次，几乎对称 —— 说明几乎全是舍入，
    真正「差一分没封住」的情形可以忽略。所以判据是「收在最高价且不低于
    涨停价 1 分以上」。
  · 停牌：首阳前一天到推荐日之间，这只票只要缺了一个全市场开市日（停牌），这一段
    就作废，首阳是复牌第一天也不算。停牌不是缩量调整，复牌那天的量和停牌前也不可比
    （2026-09-27 回看：688693 首阳 03-10，03-16~03-27 停牌 10 个交易日，第一版把它算成了
    「调整 3 天后二次进攻」）。
    节假日全市场都休市，不受影响；横盘期里停过牌不管（60 根 K 线跨得更久，照样是
    「3 个月以上」）。
  · 送转 / 大额解禁：流通股本一天变 20% 以上时，那天前后的成交量不可比
    （10 送 10 之后每天的量天然翻倍，会伪造「倍量」）。首阳、调整期、二次进攻
    任何一天股本跳变就整段作废（guard.os_jump_max）。横盘期里的跳变不管：
    送转只会让横盘均量偏大（首阳更难显得放量，保守的一侧），解禁不改变成交量
    本身。2026-09-27 第一版连横盘期一起挡，300461（2026-01-23 标准形态）被误杀。

回测和生产是同一个函数
----------------------
find_events() 在整张日线表上扫出所有历史事件；生产只取「二次进攻日 == 目标日」
的那几行，面板和邮件里的「过去三年同口径 N 次」也从同一次调用里数出来。
不另写一份回测判据（教训 34：同一个名词两份实现，迟早分叉）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import datasource as ds                                    # noqa: E402

OUT = ROOT / "out_pullback"
DATA_DIR = ROOT / "data" / "pullback"
DAILY = ROOT / "data" / "breakout" / "daily.parquet"
BJ = ZoneInfo("Asia/Shanghai")
NAME = "长期调整突破"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
# 日志时间一律北京时间（控制台「运行记录」直接印日志，界面只用北京时间）。
# 必须包 staticmethod：直接赋 lambda 会被绑成方法，每条日志都报错并丢掉
logging.Formatter.converter = staticmethod(lambda t: time.gmtime(t + 8 * 3600))
log = logging.getLogger("pullback")


def now_bj() -> dt.datetime:
    return dt.datetime.now(BJ)


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["pullback"]


# ---------------------------------------------------------------------
#  板块
# ---------------------------------------------------------------------
def board_of(code: str) -> str:
    """沪深主板 main / 创业板 chinext / 科创板 star / 北交所 bj。

    用户的规则按这个分：主板要涨停，另外三个要涨幅 ≥10%。代码段和
    datasource.limit_pct 同一口径（北交所 8/4/9 三段，2026-09-15 漏过 9）。
    """
    c = str(code).zfill(6)
    if c.startswith("30"):
        return "chinext"
    if c.startswith("68"):
        return "star"
    if c[0] in ("8", "4", "9"):
        return "bj"
    return "main"


BOARD_NAME = {"main": "主板", "chinext": "创业板", "star": "科创板", "bj": "北交所"}


# ---------------------------------------------------------------------
#  逐行派生量（向量化）
# ---------------------------------------------------------------------
def prepare(d: pd.DataFrame, pb: dict) -> pd.DataFrame:
    """按 (code, date) 排好序，补上逐行判定要用的列。输入不改。

    big     这一根算不算「大阳线」：主板涨停，其余板块涨幅 ≥ launch.gain_min_other
    vr      成交量 / 前一交易日成交量
    os_jump 当日流通股本相对前一日变动超过 guard.os_jump_max（送转 / 大额解禁）
    missed  离这只票上一根 K 线之间，全市场开了几天市而它没有 K 线（停牌天数）。
            节假日全市场都休市，不算；「开市日」取有 K 线的票不少于在市票数一半的日子
    """
    need = ["code", "date", "open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise ValueError(f"日线缺列 {miss}")
    x = d.sort_values(["code", "date"]).reset_index(drop=True).copy()
    x["code"] = x["code"].astype(str).str.zfill(6)
    x["date"] = x["date"].astype(str).str[:10]
    first = x["code"].ne(x["code"].shift())
    x["prev_close"] = x["close"].shift().where(~first)
    x["prev_vol"] = x["volume"].shift().where(~first)
    x["gain"] = x["close"] / x["prev_close"] - 1.0
    x["vr"] = x["volume"] / x["prev_vol"].where(x["prev_vol"] > 0)

    codes = pd.Series(x["code"].unique())
    brd = dict(zip(codes, codes.map(board_of)))
    lim = dict(zip(codes, codes.map(lambda c: ds.limit_pct(c, ""))))
    x["board"] = x["code"].map(brd)
    x["lim"] = x["code"].map(lim).astype(float)
    lp = ds.limit_price_arr(x["prev_close"], x["lim"], x["code"])
    x["limit_px"] = np.asarray(lp, dtype=float)
    # 收在最高价，且不低于涨停价 1 分以上（复权舍入，见模块 docstring）
    at_high = x["close"] >= x["high"] - 0.005
    limit_up = at_high & (x["close"] >= x["limit_px"] - 0.015) & (x["gain"] > 0)
    gmin = float(pb["launch"]["gain_min_other"]) / 100.0
    x["limit_up"] = limit_up.fillna(False)
    x["big"] = np.where(x["board"].eq("main"), x["limit_up"],
                        (x["gain"] >= gmin - 1e-9).fillna(False))
    x["big"] = x["big"].astype(bool)

    # 开市日：当天有 K 线的票不少于「在市」票数的一半。在市 = 这只票的第一根到
    # 最后一根之间。不能拿全表行数的中位数比：日线表 2023 年初只有两三百只票有
    # 数据（中位 5304），按中位数那段 96 天全被当成休市，停牌就识别不出来
    cnt = x["date"].value_counts()
    u = np.array(sorted(cnt.index))
    span = x.groupby("code")["date"].agg(["min", "max"])
    alive = (np.searchsorted(np.sort(span["min"].to_numpy()), u, side="right")
             - np.searchsorted(np.sort(span["max"].to_numpy()), u, side="left"))
    mdays = u[cnt.reindex(u).to_numpy() >= 0.5 * alive]
    mi = x["date"].map({dd: i for i, dd in enumerate(mdays)}).astype(float)
    x["missed"] = (mi - mi.shift() - 1).where(~first).fillna(0).clip(lower=0)

    jmax = float((pb.get("guard") or {}).get("os_jump_max", 0.20))
    if "outstanding_share" in x.columns:
        osh = x["outstanding_share"].astype(float)
        prev_os = osh.shift().where(~first)
        chg = (osh / prev_os - 1.0).abs()
        x["os_jump"] = (chg > jmax).fillna(False)
    else:
        x["os_jump"] = False
    return x


# ---------------------------------------------------------------------
#  形态判定：唯一实现，回测和生产共用
# ---------------------------------------------------------------------
def _loose_big(gain: float, c: float, o: float, pb: dict) -> bool:
    return (gain >= float(pb["trigger"].get("gain_min_loose", 5.0)) / 100.0 - 1e-9
            and c > o)


EVENT_COLS = ["code", "board", "date", "launch_date", "adjust_days", "close", "gain_pct",
              "vol_ratio", "vol_vs_launch", "one_word", "launch_open", "launch_high",
              "launch_low", "launch_close", "launch_gain_pct", "launch_vol_ratio",
              "launch_limit_up", "base_amp_pct", "launch_vs_base", "adj_vol_min",
              "adj_vol_mean", "adj_low", "adj_drawdown_pct", "bonus_vol", "bonus_half",
              "bonus_open", "_s", "_t"]
OPEN_COLS = ["code", "board", "last_date", "launch_date", "adjust_days", "close",
             "launch_high", "launch_low", "launch_open", "launch_gain_pct",
             "launch_vol_ratio", "base_amp_pct", "adj_vol_min", "adj_vol_mean", "to_high_pct"]


def find_events(x: pd.DataFrame, pb: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """在 prepare() 过的日线上扫出全部「二次进攻」事件和仍在进行中的形态。

    返回 (events, opens, diag)：
      events  每行一个完整三段（首阳 S、调整 S+1..T-1、二次进攻 T）
      opens   每只票最后一根 K 线时还没走完的形态（首阳已出、调整中），
              生产拿「最后一根 == 目标日」的那些当观察名单
      diag    各环节淘汰计数，空榜时用来说清楚「是没有首阳还是调整不合格」
    """
    bc, lc, ac, tc = pb["base"], pb["launch"], pb["adjust"], pb["trigger"]
    N = int(bc["days"])
    amp_max = float(bc["amp_max"])
    vmult = float(bc["vol_mean_mult"])
    above_max = bool(bc.get("vol_above_max", True))
    lvr = float(lc["vol_ratio_min"])
    lmin, lmax = int(ac["min_days"]), int(ac["max_days"])
    vday = float(ac["vol_day_max"])
    vmean_max = float(ac["vol_mean_max"])
    floor_kind = str(ac.get("floor", "low"))
    close_cap = bool(ac.get("close_cap", True))
    tvr = float(tc["vol_ratio_min"])
    same_big = str(tc.get("big", "same")) == "same"
    half = float((pb.get("bonus") or {}).get("shrink_half", 0.5))

    code = x["code"].to_numpy()
    date = x["date"].to_numpy()
    o = x["open"].to_numpy(float)
    h = x["high"].to_numpy(float)
    lo = x["low"].to_numpy(float)
    c = x["close"].to_numpy(float)
    v = x["volume"].to_numpy(float)
    gain = x["gain"].to_numpy(float)
    vr = x["vr"].to_numpy(float)
    big = x["big"].to_numpy(bool)
    osj = x["os_jump"].to_numpy(bool)
    miss = x["missed"].to_numpy(float) if "missed" in x.columns else np.zeros(len(x))
    brd = x["board"].to_numpy()
    lup = x["limit_up"].to_numpy(bool)
    n = len(x)
    if not n:
        return pd.DataFrame(columns=EVENT_COLS), pd.DataFrame(columns=OPEN_COLS), {}
    starts = np.r_[0, np.flatnonzero(code[1:] != code[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    cid = np.repeat(np.arange(len(starts)), ends - starts)
    pos = np.arange(n) - starts[cid]           # 这只票的第几根（0 起）
    end_of = ends[cid]

    diag: dict[str, int] = {}

    def bump(k: str) -> None:
        diag[k] = diag.get(k, 0) + 1

    cand = np.flatnonzero(big & (np.nan_to_num(vr) >= lvr) & ~osj & (pos >= N))
    ev: list[dict] = []
    op: list[dict] = []
    for s in cand:
        bump("倍量大阳线")
        base_c = c[s - N:s]
        base_v = v[s - N:s]
        if not (np.all(base_c > 0) and np.all(base_v > 0)):
            bump("横盘期有零价零量")
            continue
        amp = float(base_c.max() / base_c.min() - 1.0)
        if amp > amp_max:
            bump("横盘振幅超标")
            continue
        vmean = float(base_v.mean())
        if v[s] < vmult * vmean:
            bump("首阳量不到横盘均量倍数")
            continue
        if above_max and v[s] <= float(base_v.max()):
            bump("横盘期有比首阳更大的量")
            continue
        if miss[s] > 0:
            # 复牌第一天：「比前一天放量」比的是停牌前那根，不是用户说的「突然倍量」
            bump("首阳是复牌第一天")
            continue
        bump("首阳成立")
        vs, hs, ls, os_ = v[s], h[s], lo[s], o[s]
        floor = os_ if floor_kind == "open" else ls
        e = end_of[s]
        t = s + 1
        status = ""
        while t < e and t <= s + lmax + 1:
            L = t - s - 1                      # t 之前已经调整了几天
            if osj[t]:
                status = "股本跳变"
                break
            if miss[t] > 0:
                # 调整期或推荐日之前停过牌：停牌不是缩量调整，复牌那天的量和
                # 停牌前也不可比（2026-09-27 回看：688693 首阳后停牌 10 个交易日，
                # 复牌就被算成「调整 3 天后二次进攻」）
                status = "形态段里停过牌"
                break
            if L >= lmin and c[t] > hs:
                # 调整够天数之后第一次收盘站上首阳最高价：这一天就是考卷
                adj = slice(s + 1, t)
                vm = float(v[adj].mean()) / vs
                okbig = big[t] if same_big else _loose_big(gain[t], c[t], o[t], pb)
                why = []
                if not okbig:
                    why.append("不是大阳线")
                if not (vr[t] >= tvr):
                    why.append("量不够前日1.5倍")
                if vm > vmean_max:
                    why.append("调整期均量太大")
                if why:
                    status = "突破日不合格：" + "、".join(why)
                    break
                amin = float(v[adj].min())
                alow = float(lo[adj].min())
                ev.append({
                    "code": code[s], "board": brd[s],
                    "date": date[t], "launch_date": date[s],
                    "adjust_days": int(L),
                    "close": round(float(c[t]), 2), "gain_pct": round(100 * float(gain[t]), 2),
                    "vol_ratio": round(float(vr[t]), 2),
                    "vol_vs_launch": round(float(v[t] / vs), 2),
                    "one_word": bool(o[t] == h[t] == lo[t] == c[t]),
                    "launch_open": round(float(os_), 2), "launch_high": round(float(hs), 2),
                    "launch_low": round(float(ls), 2), "launch_close": round(float(c[s]), 2),
                    "launch_gain_pct": round(100 * float(gain[s]), 2),
                    "launch_vol_ratio": round(float(vr[s]), 2),
                    "launch_limit_up": bool(lup[s]),
                    "base_amp_pct": round(100 * amp, 1),
                    "launch_vs_base": round(float(vs / vmean), 1),
                    "adj_vol_min": round(amin / vs, 3),
                    "adj_vol_mean": round(vm, 3),
                    "adj_low": round(alow, 2),
                    "adj_drawdown_pct": round(100 * (alow / float(c[s]) - 1), 2),
                    "bonus_vol": bool(v[t] > vs),
                    "bonus_half": bool(amin <= half * vs),
                    "bonus_open": bool(alow >= os_),
                    "_s": int(s), "_t": int(t),
                })
                status = "成立"
                break
            # t 是调整日
            if v[t] >= vday * vs:
                status = "调整期有一天没缩量"
                break
            if lo[t] < floor:
                status = "跌破首阳最低价" if floor_kind == "low" else "跌破首阳开盘价"
                break
            if close_cap and c[t] > hs:
                status = f"调整不到{lmin}天就站上首阳高点"
                break
            t += 1
        if not status:
            L = t - s - 1
            if L > lmax:
                status = f"调整超过{lmax}天没突破"
            else:
                # 数据到头了还在调整：进行中的形态（明天那根还可能是二次进攻）
                adj = slice(s + 1, t)
                op.append({
                    "code": code[s], "board": brd[s],
                    "last_date": date[e - 1], "launch_date": date[s],
                    "adjust_days": int(L),
                    "close": round(float(c[e - 1]), 2),
                    "launch_high": round(float(hs), 2), "launch_low": round(float(ls), 2),
                    "launch_open": round(float(os_), 2),
                    "launch_gain_pct": round(100 * float(gain[s]), 2),
                    "launch_vol_ratio": round(float(vr[s]), 2),
                    "base_amp_pct": round(100 * amp, 1),
                    "adj_vol_min": round(float(v[adj].min()) / vs, 3) if L else None,
                    "adj_vol_mean": round(float(v[adj].mean()) / vs, 3) if L else None,
                    "to_high_pct": round(100 * (hs / float(c[e - 1]) - 1), 2),
                })
                status = "进行中"
        bump(status)
    # 空表也要带列：历史上一次都没成立时（换一段数据、自测），下游按列名取值不能炸
    evd = pd.DataFrame(ev, columns=EVENT_COLS) if not ev else pd.DataFrame(ev)
    if len(evd):
        # 同一只票同一天只留一行：两个首阳都成立时取最近的那个（形态讲的是
        # 「上一次启动之后的这一段回调」，和旧版形态线同一个取法）
        evd = (evd.sort_values(["code", "date", "launch_date"])
                  .drop_duplicates(["code", "date"], keep="last")
                  .reset_index(drop=True))
    opd = pd.DataFrame(op, columns=OPEN_COLS) if not op else pd.DataFrame(op)
    if len(opd):
        opd = (opd.sort_values(["code", "launch_date"])
                  .drop_duplicates(["code"], keep="last").reset_index(drop=True))
    return evd, opd, diag


def rank(ev: pd.DataFrame) -> pd.DataFrame:
    """排序：「最好」三项满足几项 -> 今日量 / 首阳量 -> 今日涨幅。纯确定性。"""
    if not len(ev):
        return ev
    e = ev.copy()
    e["bonus_n"] = (e[["bonus_vol", "bonus_half", "bonus_open"]]
                    .astype(int).sum(axis=1))
    return e.sort_values(["bonus_n", "vol_vs_launch", "gain_pct", "code"],
                         ascending=[False, False, False, True]).reset_index(drop=True)


def forward(x: pd.DataFrame, ev: pd.DataFrame, bars: int = 20) -> pd.DataFrame:
    """每个历史事件之后 bars 根 K 线：最高价涨幅、第 bars 根收盘涨幅（相对事件日收盘）。

    只描述历史，不是预测；走不满 bars 根的标 n_after < bars。
    """
    if not len(ev):
        return ev.assign(n_after=pd.Series(dtype=int), max_up_pct=pd.Series(dtype=float),
                         ret_pct=pd.Series(dtype=float))
    c = x["close"].to_numpy(float)
    h = x["high"].to_numpy(float)
    code = x["code"].to_numpy()
    out = []
    for _, r in ev.iterrows():
        t = int(r["_t"])
        j = t + 1
        k = min(len(x), t + 1 + bars)
        while k > j and code[k - 1] != code[t]:
            k -= 1
        n_after = k - j
        if n_after <= 0:
            out.append((0, None, None))
            continue
        out.append((n_after,
                    round(100 * (float(h[j:k].max()) / c[t] - 1), 1),
                    round(100 * (c[k - 1] / c[t] - 1), 1)))
    e = ev.copy()
    e["n_after"] = [a for a, _, _ in out]
    e["max_up_pct"] = [b for _, b, _ in out]
    e["ret_pct"] = [r for _, _, r in out]
    return e


def coverage_start(x: pd.DataFrame, base_days: int) -> str:
    """全市场都能被扫到的第一天：前面已有 base_days 根 K 线的票数第一次到中位数九成。

    一只票要先有 base_days 根横盘才可能出首阳。日线表 2023 年初只有两三百只票有
    数据（2023-05 之后才五千多只），拿全表天数当分母，「每月几次」被摊薄两成，
    「回看 2023-01 起」也名不副实（2026-09-27 实测：全表 905 天算每月 1.55 次，
    覆盖齐了的 749 天算 1.9 次）。
    """
    if not len(x):
        return ""
    pos = x.groupby("code").cumcount()
    n = x[pos >= base_days].groupby("date").size()
    if not len(n):
        return str(x["date"].min())
    return str(n.index[n >= 0.9 * n.median()].min())


def history_stats(evf: pd.DataFrame, x: pd.DataFrame, bars: int, base_days: int) -> dict:
    """同口径的历史频率，外加走满 bars 根的那些事件之后怎么样了。

    只数全市场覆盖齐了之后的那一段（coverage_start），频率和「之后」用同一批事件。
    """
    start = coverage_start(x, base_days)
    last = str(x["date"].max()) if len(x) else ""
    days = int(x.loc[x["date"] >= start, "date"].nunique()) if len(x) else 0
    evw = evf[evf["date"] >= start] if len(evf) else evf
    n = int(len(evw))
    done = evw[evw["n_after"] >= bars] if n else evw
    per_month = round(n / days * 21, 2) if days else None
    st = {"n": n, "days": days, "from": start, "to": last, "per_month": per_month,
          "bars": bars, "n_final": int(len(done))}
    if len(done):
        st.update({
            "max_up_median": round(float(done["max_up_pct"].median()), 1),
            "ret_median": round(float(done["ret_pct"].median()), 1),
            "up_share": round(float((done["ret_pct"] > 0).mean()), 3),
        })
    return st


# ---------------------------------------------------------------------
#  名称 / ST
# ---------------------------------------------------------------------
def names_for(codes: list[str]) -> dict[str, str]:
    """腾讯快照拿当前名称。拿不到就空，调用方按 ST 缓存兜底。"""
    if not codes:
        return {}
    try:
        q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
        return {c: getattr(q.get(ds.to_symbol(c)), "name", "") or "" for c in codes}
    except Exception as e:  # noqa: BLE001
        log.warning("名称拿不到（%s），ST 按缓存名单剔", e)
        return {}


def st_cache() -> set[str]:
    p = ROOT / "cache" / "st_codes.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        cs = raw.get("codes", raw) if isinstance(raw, dict) else raw
        return {str(c).zfill(6) for c in cs}
    except Exception:  # noqa: BLE001
        return set()


def is_excluded(code: str, name: str, st: set[str]) -> str:
    """剔除原因，空串 = 不剔。ST / 退市整理不选。"""
    if name:
        if ds.is_st_name(name):
            return f"ST（{name}）"
        return ""
    return "ST（缓存名单）" if code in st else ""


# ---------------------------------------------------------------------
#  扫描
# ---------------------------------------------------------------------
def load_daily(target: str) -> pd.DataFrame:
    cols = ["code", "date", "open", "high", "low", "close", "volume",
            "outstanding_share"]
    d = pd.read_parquet(DAILY, columns=cols)
    d["date"] = d["date"].astype(str).str[:10]
    return d[d["date"] <= target]


def last_closed_trade_day() -> str:
    """最近一个已收盘（15:05 后）的交易日。和 local_run.last_closed_trade_day 同口径。"""
    now = now_bj()
    closed = (now.hour, now.minute) >= (15, 5)
    try:
        tds = set(ds.trade_dates())
    except Exception:  # noqa: BLE001
        tds = set()
    for back in range(15):
        day = now.date() - dt.timedelta(days=back)
        ok = (day.isoformat() in tds) if tds else day.weekday() < 5
        if ok and (back > 0 or closed):
            return day.isoformat()
    return now.date().isoformat()


def scan(target: str, pb: dict, d: pd.DataFrame | None = None) -> dict:
    """扫描目标日。返回一个 dict，stage_scan 负责落盘。d 给了就不读文件（自测用）。"""
    t0 = time.time()
    raw = load_daily(target) if d is None else d[d["date"].astype(str) <= target]
    x = prepare(raw, pb)
    if not len(x) or str(x["date"].max()) != target:
        raise RuntimeError(f"日线表里没有 {target} 的行（最新 {x['date'].max() if len(x) else '无'}）")
    ev, opn, diag = find_events(x, pb)
    evf = forward(x, ev, int(pb.get("output", {}).get("forward_bars", 20)))
    today = x[x["date"] == target]
    todays = evf[evf["date"] == target] if len(evf) else evf
    watch = opn[opn["last_date"] == target] if len(opn) else opn

    need = sorted(set(todays["code"] if len(todays) else [])
                  | set(watch["code"] if len(watch) else []))
    hist_n = int(pb.get("output", {}).get("history_n", 30))
    # 多取一些再剔 ST，剔完才截到 hist_n 条（先截后剔，面板上会少几条）
    hist = (evf[evf["date"] < target].sort_values(["date", "code"])
            .tail(hist_n + 20)) if len(evf) else evf
    need_all = sorted(set(need) | set(hist["code"] if len(hist) else []))
    nm = names_for(need_all)
    st = st_cache()

    def attach(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        if not len(df):
            return df.assign(name=pd.Series(dtype=str)), {}
        df = df.copy()
        df["name"] = df["code"].map(lambda c: nm.get(c, ""))
        bad = {c: why for c in df["code"] if (why := is_excluded(c, nm.get(c, ""), st))}
        return df[~df["code"].isin(bad)].reset_index(drop=True), bad

    todays, bad_t = attach(todays)
    watch, bad_w = attach(watch)
    if len(watch) and len(todays):
        # 今天的二次进攻那一根往往又是一根新的「首阳」（量更大、横盘振幅也够），
        # 同一只票不能既在清单上又在观察名单里
        watch = watch[~watch["code"].isin(set(todays["code"]))].reset_index(drop=True)
    hist, _ = attach(hist)
    hist = hist.tail(hist_n).reset_index(drop=True)
    todays = rank(todays)
    # 历史频率按今天的 ST 名单剔（历史上当时是不是 ST 拿不到），和
    # pullback_backtest 同一个口径，邮件里的「三年 N 次」和回看脚本对得上
    evs = evf[~evf["code"].isin(st)] if len(evf) else evf
    stats = history_stats(evs, x, int(pb.get("output", {}).get("forward_bars", 20)),
                          int(pb["base"]["days"]))
    # 目标日当天的分项计数（邮件抬头那一行）
    big_today = int((today["big"] & (today["vr"] >= float(pb["launch"]["vol_ratio_min"]))).sum())
    info = {
        "date": target, "n": int(len(todays)), "n_watch": int(len(watch)),
        "n_stocks": int(len(today)), "n_big_today": big_today,
        "excluded": {**bad_t, **bad_w}, "diag_all": diag, "hist": stats,
        "seconds": round(time.time() - t0, 1),
        "rules": rules_text(pb),
    }
    log.info("%s：全市场 %d 只，今日倍量大阳线 %d 只，调整中 %d 只，今日二次进攻 %d 只"
             "（剔除 %d 只）| 三年同口径 %d 次，约每月 %s 次 | %.1fs",
             target, info["n_stocks"], big_today, info["n_watch"], info["n"],
             len(info["excluded"]), stats["n"], stats["per_month"], info["seconds"])
    return {"info": info, "today": todays, "watch": watch, "hist": hist,
            "all_events": evf}


def rules_text(pb: dict) -> list[str]:
    """邮件和面板里印的规则，从 config 现拼，改阈值不用改文案。"""
    bc, lc, ac, tc = pb["base"], pb["launch"], pb["adjust"], pb["trigger"]
    big = f"主板涨停，创业板/科创板/北交所涨幅 ≥{lc['gain_min_other']:g}%"
    tbig = (f"同样的大阳线（{big}）" if str(tc.get("big", "same")) == "same"
            else f"涨幅 ≥{tc.get('gain_min_loose', 5):g}% 的阳线")
    floor = "最低价" if str(ac.get("floor", "low")) == "low" else "开盘价"
    return [
        f"横盘：首阳前 {bc['days']} 个交易日，收盘价最高/最低 ≤ {1 + bc['amp_max']:.2f}；"
        f"首阳量 ≥ 这段日均量 {bc['vol_mean_mult']:g} 倍"
        + ("，且比其中任何一天都大" if bc.get("vol_above_max", True) else ""),
        f"首阳：{big}；成交量 ≥ 前一日 {lc['vol_ratio_min']:g} 倍",
        f"调整：首阳后 {ac['min_days']}~{ac['max_days']} 个交易日，每天量都低于首阳、"
        f"均量 ≤ 首阳 {ac['vol_mean_max'] * 100:.0f}%，不破首阳{floor}，收盘不超过首阳最高价",
        f"二次进攻（今天）：{tbig}，量 ≥ 前一日 {tc['vol_ratio_min']:g} 倍，"
        f"收盘 > 首阳最高价（调整后第一次站上）",
        "加分项只排序不决定入选：量超首阳、调整最低量缩到首阳一半以下、不破首阳开盘价",
        "「量」是成交量（股），不是成交额。ST、形态中途停过牌的不选",
    ]


def _records(df: pd.DataFrame) -> list[dict]:
    if not len(df):
        return []
    keep = [c for c in df.columns if not c.startswith("_")]
    return json.loads(df[keep].to_json(orient="records", force_ascii=False))


def stage_scan(target: str, dry: bool = False) -> int:
    pb = cfg()
    try:
        tds = set(ds.trade_dates())
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历拿不到（%s），按工作日兜底", e)
        tds = set()
    if tds and target not in tds:
        log.info("%s 不是交易日，不扫描", target)
        return 0
    try:
        r = scan(target, pb)
    except Exception as e:  # noqa: BLE001
        log.error("扫描失败：%s", e)
        return 1
    info = r["info"]
    if dry:
        # 命令行试扫：只打印，不落盘。local_run --dry 不走这条（它不传 --dry，
        # 靠环境变量 DRY_RUN 让 run_meta 标 dry，产物照写，面板照出）
        for _, e in r["today"].iterrows():
            log.info("  成立 %s %s 收%.2f %+.2f%% 量%.2f倍(首阳%.0f%%) | 首阳 %s 调整%d天 "
                     "最低量%.0f%% | 加分 %s%s%s", e["code"], e["name"], e["close"],
                     e["gain_pct"], e["vol_ratio"], 100 * e["vol_vs_launch"],
                     e["launch_date"], e["adjust_days"], 100 * e["adj_vol_min"],
                     "量" if e["bonus_vol"] else "-", "半" if e["bonus_half"] else "-",
                     "开" if e["bonus_open"] else "-")
        for _, w in r["watch"].iterrows():
            log.info("  调整中 %s %s 首阳 %s 已调整 %d 天，离首阳高点 %+.2f%%",
                     w["code"], w["name"], w["launch_date"], w["adjust_days"],
                     w["to_high_pct"])
        log.info("命令行试扫：不落盘")
        return 0
    OUT.mkdir(exist_ok=True)
    (OUT / "selected.json").write_text(json.dumps(
        _records(r["today"]), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "watch.json").write_text(json.dumps(
        _records(r["watch"]), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(
        _records(r["hist"]), ensure_ascii=False, indent=1), encoding="utf-8")
    # 同花顺自选股导入用的纯代码 txt。GBK + CRLF，同花顺只认这个。
    # 空榜写 0 字节：这个文件每天覆盖，「今天没有」必须是空的
    (OUT / f"{NAME}.txt").write_bytes(("\r\n".join(r["today"]["code"])
                                       + ("\r\n" if len(r["today"]) else "")
                                       ).encode("gbk"))
    info = {**info, "dry": bool(os.environ.get("DRY_RUN"))}
    (OUT / "run_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    # 每日清单入库（空榜也落一个空文件）：面板按日期回看、以后核对都靠它
    dd = DATA_DIR / target[:7]
    dd.mkdir(parents=True, exist_ok=True)
    t = r["today"]
    t = t[[c for c in t.columns if not c.startswith("_")]] if len(t) else \
        pd.DataFrame(columns=["code"])
    t.to_parquet(dd / f"pullback_{target}.parquet", index=False)
    log.info("产物已写：%s，清单 %d 只", OUT, info["n"])
    return 0


# ---------------------------------------------------------------------
#  发信
# ---------------------------------------------------------------------
def send_time(target: str, pb: dict) -> dt.datetime:
    hh, mm, ss = (int(v) for v in str(pb.get("send_at", "17:58:00")).split(":"))
    d = dt.date.fromisoformat(target)
    return dt.datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=BJ)


def wait_until(when: dt.datetime) -> None:
    """等到发信时刻。按墙钟轮询，不是 sleep 一大段：合盖睡着再醒来已经过点，
    下一轮就直接放行（这条线晚发没有害处，不像竞价那样过了窗口数据就脏）。"""
    left = (when - now_bj()).total_seconds()
    if left <= 0:
        return
    total = max(1, math.ceil(left / 60))
    log.info("等到 %s 发信（还有 %.0f 分钟）", when.strftime("%H:%M:%S"), left / 60)
    said = -1
    while (left := (when - now_bj()).total_seconds()) > 0:
        # 每分钟一行，local_run 的进度条认这一行（_SUB），不然要在同一格停很久
        k = total - math.ceil(left / 60)
        if k != said:
            print(f"等待发信 {k}/{total} 分钟", flush=True)
            said = k
        time.sleep(min(left, 20))


def stage_send(target: str, wait: bool = True) -> int:
    import pullback_export as E
    from mailer import skip_mail
    pb = cfg()
    try:
        meta = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        meta = {}
    # out_pullback/ 提交进仓库，checkout 会带着上一交易日的产物。日期对不上
    # 就说明这次扫描没跑完，绝不拿旧清单发信
    if meta.get("date") != target:
        log.warning("out_pullback/ 里是 %s 的产物（目标日 %s），不发信",
                    meta.get("date", "未知"), target)
        return 1

    def load(nm: str) -> list[dict]:
        try:
            rows = json.loads((OUT / nm).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        for r in rows:
            if "code" in r:
                r["code"] = str(r["code"]).zfill(6)   # 教训 29：前导零
        return rows

    sel, watch, hist = load("selected.json"), load("watch.json"), load("history.json")
    due = send_time(target, pb)
    if wait:
        wait_until(due)
    late = now_bj() - due
    muted = skip_mail()
    # 「补发于」只在真要发信时写。SKIP_MAIL（试跑、手动重出面板）不发信，
    # 写上去面板就在说一件没发生的事（2026-09-27 首发那份 09-24 面板就是这样）
    late_note = (f"本机 {due.strftime('%H:%M')} 没开机，补发于 "
                 f"{now_bj().strftime('%m-%d %H:%M')}"
                 if late.total_seconds() > 15 * 60 and not muted else "")
    owner = os.environ.get("GH_OWNER", "")
    repo = os.environ.get("GH_REPO", "")
    page = f"https://{owner.lower()}.github.io/{repo}/pullback.html" if owner else ""
    E.write_panel(sel, watch, hist, meta, OUT, target, late_note)
    if muted:
        log.info("SKIP_MAIL 已设：面板已生成，邮件不发")
        return 0
    att = [p for p in [OUT / f"{NAME}.txt"] if p.exists() and p.stat().st_size > 0]
    E.send_mail(target, sel, watch, meta, page_url=page, late_note=late_note,
                attachments=att)
    # 真发出去才落这个戳（教训 27：退出码 0 不等于做了事）。local_run 只认它
    p = OUT / "mail_sent.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"date": target, "n": len(sel),
                               "at": now_bj().isoformat(timespec="seconds")},
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scan", "send"], required=True)
    ap.add_argument("--target", default="", help="目标交易日，默认最近一个已收盘的交易日")
    ap.add_argument("--dry", action="store_true",
                    help="scan：只打印不落盘（命令行试扫用；local_run --dry 不传它）")
    ap.add_argument("--no-wait", action="store_true", help="send：不等 17:58，立即发")
    a = ap.parse_args()
    try:
        import localenv
        localenv.load()
    except Exception:  # noqa: BLE001
        pass
    target = a.target or last_closed_trade_day()
    if a.stage == "scan":
        return stage_scan(target, dry=a.dry)
    return stage_send(target, wait=not a.no_wait)


if __name__ == "__main__":
    sys.exit(main())
