"""
指标计算 + 打分排序。

原则：排序 100% 由确定性代码完成，LLM 不参与。
      这样结果可复现、可回测、可归因。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


# ---------------------------------------------------------------------
#  单只票的竞价特征
# ---------------------------------------------------------------------
@dataclass
class AuctionFeature:
    code: str
    name: str
    limit_pct: float          # 当日涨停幅度 10/20/30/5
    prev_close: float
    auc_price: float          # 9:25 竞价成交价
    gap_pct: float            # 高开幅度 %
    gap_norm: float           # 高开幅度 / 涨停幅度
    auc_amount: float         # 竞价成交额（元）
    prev_amount: float        # 昨日全天成交额（元）
    auc_ratio: float          # 竞价额 / 昨日额
    t1_chg: float             # 9:19:40 虚拟涨幅 %
    t2_chg: float             # 9:23:30 涨幅 %
    t3_chg: float             # 9:25:10 涨幅 % (= gap_pct)
    slope: float              # t3 - t1
    monotonic: bool           # t1 <= t2 <= t3
    dive: float               # t2 - t3（尾盘跳水幅度）
    pos_pct_60d: float        # 60日价格分位 0~1
    ma_bull: bool             # 5>10>20 多头排列
    breakout: bool            # 突破 N 日平台高点
    prev_limit_up: bool
    prev_broken_board: bool   # 昨日炸板
    board_height: int         # 连板高度
    sector: str
    sector_members: int       # 候选池内同板块入选数
    sector_prev_limitups: int
    blacklisted: bool
    one_word: bool            # 一字板（竞价即封涨停且无量差）


# ---------------------------------------------------------------------
#  分项打分：全部映射到 [0, 1]
# ---------------------------------------------------------------------
def f_gap(gap: float, lo: float, hi: float, peak: float) -> float:
    """钟形：peak 处得 1，两个边界处都归零。入参是竞价涨幅百分点。

    左右两臂各自按自己的跨度归一。上一版写的是
    `half = max(peak - lo, hi - peak)`，只有长的那一臂能在边界上归零，
    短的那一臂到边界还剩一截，被区间外的 return 0 硬切掉。
    旧参数(0.20/0.35/0.55)恰好是上臂更长，没暴露；下限降到 0 之后上臂只剩
    1.5，于是 +4.99% 得 0.57、+5.00%（仍在允许范围内）却得 0。
    分臂归一同时也让「越接近 5% 的剔除线，扣分越快」成立，这正是用户设这条
    上限的理由（高开过多容易高开低走）。

    2026-09-02 退回 2%~5% 之后两臂又恰好等长（各 1.5），这个 bug 在当前配置下
    看不出来 —— 正因为看不出来才不能把分臂归一改回去，selftest 的断言留着。
    """
    if gap <= lo or gap >= hi:
        return 0.0
    half = (peak - lo) if gap < peak else (hi - peak)
    return max(0.0, 1.0 - abs(gap - peak) / half)


def f_volume(ratio: float, lo: float, hi: float, sat: float,
             decay: float = 0.40) -> float:
    """对数刻度：lo -> 0，sat -> 1，超过 sat 后继续按 log 衰减，超 hi 为 0。

    衰减速率由 decay 单独给，**不能**由 hi 推导。上一版写成
    `1 - 0.4*(ratio-sat)/(hi-sat)`，把 auc_ratio_max 从 8% 放宽到 20.8% 时
    衰减被摊到 2.8 倍宽的区间上：量比 19 的得分从 0.61 变成 0.89，量比 25/35
    从「直接出局」变成 0.83/0.74，「越极端越警惕」这层意思被悄悄抹掉了。
    上限只说明「还能接受」，不说明「一样好」，两者必须解耦。
    """
    if ratio < lo or ratio > hi:
        return 0.0
    if ratio <= sat:
        return math.log(ratio / lo) / math.log(sat / lo)
    return max(0.0, 1.0 - decay * math.log(ratio / sat))


def f_trend(slope: float, monotonic: bool, limit: float) -> float:
    """竞价斜率归一到涨停幅度；单调抬升额外加分。"""
    s = max(-1.0, min(1.0, slope / (limit * 0.3)))
    base = (s + 1.0) / 2.0
    return min(1.0, base + (0.15 if monotonic else 0.0))


def f_position(feat: AuctionFeature, group: str) -> float:
    if group == "B":                      # 低位启动
        p = max(0.0, 1.0 - feat.pos_pct_60d / 0.40)
        return min(1.0, p + (0.2 if feat.ma_bull else 0.0))
    s = 0.0                               # A 组：趋势中放量突破
    if feat.breakout:
        s += 0.5
    if feat.ma_bull:
        s += 0.3
    if 0.55 <= feat.pos_pct_60d <= 0.95:
        s += 0.2
    return min(1.0, s)


def f_sector(feat: AuctionFeature, min_members: int, min_prev: int) -> float:
    if feat.sector_members <= 1:
        return 0.0
    s = min(1.0, (feat.sector_members - 1) / (min_members + 1))
    if feat.sector_prev_limitups >= min_prev:
        s = min(1.0, s + 0.3)
    return s


def f_continuity(feat: AuctionFeature) -> float:
    if feat.board_height >= 3:
        return 0.6                        # 高位连板，风险 > 收益，不给高分
    if feat.board_height == 2:
        return 1.0
    if feat.prev_limit_up:
        return 0.85
    return 0.3


# ---------------------------------------------------------------------
#  硬性排除
# ---------------------------------------------------------------------
def _liangbi(auc_ratio: float, sc: dict[str, Any]) -> float:
    """把自定义的 AUC_RATIO 换算成用户熟悉的「量比」口径，只用于显示。

    换算系数在 config 里（默认 240，即设昨日量 ≈ 5 日均量）。筛选和打分
    一律用 AUC_RATIO 本身，不用这个值——量比各家口径不一致、不可复现。
    """
    return auc_ratio * sc.get("liangbi_per_auc_ratio", 240)


def hard_reject(feat: AuctionFeature, sc: dict[str, Any]) -> str | None:
    if feat.blacklisted:
        return "隔夜公告黑名单"
    if feat.one_word:
        return "一字板（买不进）"
    if feat.prev_close <= 0 or feat.auc_price <= 0:
        return "停牌或数据缺失"
    if not (sc["gap_pct_min"] <= feat.gap_pct <= sc["gap_pct_max"]):
        return f"竞价涨幅 {feat.gap_pct:.2f}% 超出区间"
    if not (sc["auc_ratio_min"] <= feat.auc_ratio <= sc["auc_ratio_max"]):
        return (f"竞价量能 {feat.auc_ratio*100:.2f}%"
                f"(量比≈{_liangbi(feat.auc_ratio, sc):.1f}) 超出区间")
    # 先判假涨停：它同时也会触发跳水条件，先判才能给出准确的拒绝原因
    if (feat.t1_chg >= feat.limit_pct * sc["fake_limit_t1_frac"]
            and feat.t3_chg < feat.limit_pct * sc["fake_limit_t3_frac"]):
        return "假涨停撤单"
    if feat.dive >= sc["last_min_dive_max"]:
        return f"尾盘跳水 {feat.dive:.1f}pct"
    if sc["require_positive_slope"] and feat.slope <= 0:
        return "竞价走势走弱"
    return None


def assign_group(feat: AuctionFeature, sc: dict[str, Any]) -> str:
    """A = 接力/强势延续；B = 低位首板预备。"""
    if feat.prev_limit_up or feat.board_height >= 1 or feat.breakout:
        return "A"
    if feat.pos_pct_60d <= sc["pos_pct_60d_max_for_lowbase"]:
        return "B"
    return "A"


def score_one(feat: AuctionFeature, cfg: dict[str, Any]) -> dict[str, Any]:
    sc, w, pen = cfg["screen"], cfg["scoring"]["weights"], cfg["scoring"]["penalties"]

    reject = hard_reject(feat, sc)
    group = assign_group(feat, sc)

    parts = {
        "gap":        f_gap(feat.gap_pct, sc["gap_pct_min"], sc["gap_pct_max"],
                            sc["gap_pct_peak"]),
        "volume":     f_volume(feat.auc_ratio, sc["auc_ratio_min"],
                               sc["auc_ratio_max"], sc["auc_ratio_score_hi"],
                               sc.get("auc_ratio_decay", 0.40)),
        "trend":      f_trend(feat.slope, feat.monotonic, feat.limit_pct),
        "position":   f_position(feat, group),
        "sector":     f_sector(feat, sc["sector_min_members"],
                               sc["concept_prev_limitup_min"]),
        "continuity": f_continuity(feat),
    }
    raw = 100.0 * sum(w[k] * v for k, v in parts.items())

    penalty = 0.0
    tags: list[str] = []
    if feat.dive >= sc["last_min_dive_max"] * 0.6:
        penalty += pen["last_min_dive"] * 0.5
        tags.append("竞价尾段走弱")
    # 触发线用绝对量比，不从 auc_ratio_score_hi 推导。原来写的是 sat*2 = 6%
    # （量比 14.4），量比上限收回 10 之后那个值落在准入区间之外，永远触发不了。
    # 量比 >10 本身已由 auc_ratio_max 硬剔除，这里管的是「高位 + 逼近上限」。
    if (feat.pos_pct_60d > 0.9
            and _liangbi(feat.auc_ratio, sc) >= sc.get("high_pos_liangbi_min", 8.0)):
        penalty += pen["high_pos_extreme_volume"]
        tags.append("高位极端放量")
    if feat.prev_broken_board:
        penalty += pen["yesterday_broken_board"]
        tags.append("昨日炸板")
    if feat.board_height >= 3:
        tags.append(f"{feat.board_height}连板 高位")

    return {
        **asdict(feat),
        "group": group,
        "liangbi": round(_liangbi(feat.auc_ratio, sc), 1),
        "score": round(max(0.0, raw - penalty), 1),
        "parts": {k: round(v, 3) for k, v in parts.items()},
        "penalty": round(penalty, 1),
        "risk_tags": tags,
        "rejected": reject,
    }


def rank(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list]:
    out = cfg["output"]
    ok = [r for r in rows if r["rejected"] is None and r["score"] >= out["min_score"]]
    ok.sort(key=lambda r: r["score"], reverse=True)
    return {
        "A": [r for r in ok if r["group"] == "A"][: out["top_n_a"]],
        "B": [r for r in ok if r["group"] == "B"][: out["top_n_b"]],
        "all": ok,
    }
