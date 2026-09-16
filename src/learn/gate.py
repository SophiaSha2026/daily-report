"""
接受门：六道闸，全过才允许改参数。

这是整套系统里**最重要的部分**。优化器永远能找到一个「更好」的参数，
闸门决定那个更好是不是真的。设计原则：宁可几个月不动，
也不要为一次噪声改一次口径。

一条不过 -> 不改，把不过的原因写进 state/learning_status.json。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY = ROOT / "state" / "theta_history.jsonl"     # 只记接受了的变更
VERDICTS = ROOT / "state" / "verdict_log.jsonl"     # 每次裁决都记，面板画进度用


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Verdict:
    accepted: bool
    checks: list
    moved: dict          # 参数路径 -> (旧, 新)
    evidence: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) if not isinstance(c, dict) else c
                       for c in self.checks]
        return d


def last_accept_date() -> str | None:
    if not HISTORY.exists():
        return None
    last = None
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                last = json.loads(line).get("date")
            except Exception:  # noqa: BLE001
                pass
    return last


def _trading_days_between(a: str, b: str, all_days: list[str]) -> int:
    """(a, b] 之间的交易日数。优先用真实交易日历（state/trade_dates.json，
    由 datasource.trade_dates 维护）；拿不到才退回训练表的日期。

    训练表的日期是回填表的日期，止于回填截止日：一旦在那之后接受过变更，
    按它数 gap 永远是 0，冷却期闸门就再也过不了。
    """
    try:
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parent.parent.parent / "state" / "trade_dates.json"
        cal = json.loads(f.read_text(encoding="utf-8"))
        if cal:
            return len([d for d in cal if a < d <= b])
    except Exception:  # noqa: BLE001
        pass
    return len([d for d in all_days if a < d <= b])


def evaluate(theta_new: dict, theta_old: dict, box: dict, g: dict,
             n_days: int, all_days: list[str], today: str, *,
             boot_p: float, oos_new: float, oos_old: float,
             churn: dict[str, float],
             online_p: float | None = None,
             online_days: int = 0,
             intents: list[str] | None = None,
             paired_delta: float | None = None,
             block_delta: list[float] | None = None) -> Verdict:
    """跑完六道闸（外加可选的第七道），返回裁决。

    统计量一律**关键字传入**。2026-09-04 全仓 debug 抓到：调用方按位置把
    配置阈值 g["bootstrap_p"] 传到了 boot_p 的位置，闸门 3 变成 0.9 >= 0.9
    永远通过，而真正算出来的 P 被丢掉。关键字参数让这种错位在代码里一眼可见，
    selftest_learn.check_wiring 再用 AST 钉住调用方。

    参数说明：
      g            config 里的 learning.gate 段
      boot_p       按天自助算出的 P(新参数样本外更好)
      paired_delta 样本外逐日差 G_d(new) − G_d(old) 的 Huber 位置。闸门 2 比的是
                   两个位置估计之差，闸门 3 自助的是配对差的位置，两者可以
                   一正一负（各自都对，量的不是同一件事）；把配对量也写进
                   裁决，读的人不用猜为什么 P 高而 ΔĜ 为负。
                   由 optimize.paired_delta 算：**已剔除日权重 0 的天、按 day_w
                   加权**，和闸门 3 的自助同一组天同一组权重（审计 F1-8）
      block_delta  样本外段切成若干连续块，每块各自的 ΔĜ。闸门 2 除了整段
                   改善，还要求多数块同向改善（审计 F1-6）。None = 不检查
      churn        日期 -> 前 K 变动比例（闸门 5 的输入）
      online_p     P(新参数在**在线真值快照**上更好)。训练主体是回填表，
                   竞价轨迹是代理值；这道闸保证学到的东西搬到真值上
                   至少不明显更差。None = 无在线数据，跳过。
                   只用**样本外段**里的在线日：在线天 2026-09-15 起并进训练表，
                   落进走向前拟合段的那些天是样本内，拿它们做否决检验是自证
                   （eval_daily._online_oos，审计 F3-6）。
      online_days  参与在线检验的天数，按**日权重 > 0** 计，和 bootstrap_better
                   的 keep 同口径：归因为「数据异常」的天权重是 0，自助里
                   已经剔掉了，天数却曾照数（eval_daily._online_days，审计 F2-8）。
                   少于 g["online_min_days"] 只记录不否决——几天的样本连
                   「明显更差」都判不出来。
    """
    checks: list[Check] = []
    sig = {k: (hi - lo) for k, (lo, hi) in box.items()}
    moved = {k: (theta_old[k], theta_new[k]) for k in box
             if abs(theta_new[k] - theta_old[k]) > 1e-9}

    # 0 权重和为 1。优化器路径靠 sparsify/O.project 保证，但学习会诊的参数
    # 提案是 LLM 直接给值的，2026-09-16 审计（F2-4）实测：只把 trend
    # 0.20 -> 0.23 夹进箱送进来，Σw=1.03 一路过闸落地。score.py 的
    # raw = 100·Σw·v 于是满分变 103，而 min_score=45 和各项扣分都是绝对值，
    # 等于同时放松准入分数线、相对削弱扣分（18 天真实快照里 >=45 分的行
    # 280 -> 299）。七道闸量的全是排序量，对整体缩放完全失明，所以要单列一条。
    wk = [k for k in box if k.startswith("scoring.weights.")]
    if wk:
        wsum = sum(float(theta_new[k]) for k in wk)
        checks.append(Check("权重和为 1", abs(wsum - 1.0) < 1e-6,
                            f"Σw={wsum:.4f}（{len(wk)} 个权重）"))

    # 1 最少天数
    ok = n_days >= g["min_days"]
    checks.append(Check("最少天数", ok,
                        f"{n_days} 天 / 要求 >= {g['min_days']}"))

    # 2 走向前样本外改善 + 块一致性。
    # 整段的 Ĝ_oos 是一个 83 天尾窗，每天只往后挪一天（相邻两次裁决的 te
    # 重叠 98%），所以「连着几次都过」几乎等于「过了一次」；而整段一个数
    # 会把块间分歧抹平（09-16 那个提案三块是 +0.0095/−0.0175/−0.0134）。
    # 要求多数块同向，一个只在某一段时间里有效的提案就过不去（审计 F1-6）。
    need_blocks = int(g.get("oos_min_blocks_better", 2))
    n_up = sum(1 for d in (block_delta or []) if d > 0)
    ok = oos_new > oos_old and (not block_delta or n_up >= need_blocks)
    checks.append(Check("样本外改善", ok,
                        f"Ĝ_oos {oos_old:+.4f} -> {oos_new:+.4f} "
                        f"（{oos_new - oos_old:+.4f}）"
                        + (f"；分块 {n_up}/{len(block_delta)} 改善"
                           f"（要求 >= {need_blocks}）" if block_delta else "")))

    # 3 自助显著性
    ok = boot_p >= g["bootstrap_p"]
    checks.append(Check("按天自助显著", ok,
                        f"P(更好)={boot_p:.3f} / 要求 >= {g['bootstrap_p']}"
                        + (f"；配对 ΔĜ={paired_delta:+.4f}"
                           if paired_delta is not None else "")))

    # 4 步长上限 + 改动个数。
    # intents 给出时按**意图**计数：动一个权重必然带出其余权重的等比再归一
    # （和为 1 是硬约束），那些是结果不是决定，不占改动名额；
    # 但步长上限对**所有**实际变化生效，包括再归一的残差。
    over = {k: abs(v[1] - v[0]) / sig[k] for k, v in moved.items()
            if abs(v[1] - v[0]) / sig[k] > g["max_step_frac"] + 1e-12}
    if intents is not None:
        ok = not over and len(intents) <= g["max_moves"]
        checks.append(Check("步长与改动个数", ok,
                            f"意图 {len(intents)} 个: {intents}"
                            f"（上限 {g['max_moves']}，实际触及 {len(moved)} 个"
                            f"含权重再归一）"
                            + (f"；超步长: {list(over)}" if over else "")))
    else:
        ok = not over and len(moved) <= g["max_moves"]
        checks.append(Check("步长与改动个数", ok,
                            f"动了 {len(moved)} 个（上限 {g['max_moves']}）"
                            + (f"；超步长: {list(over)}" if over else "")))

    # 5 行为回放
    worst_day = max(churn, key=lambda d: churn[d]) if churn else ""
    worst = churn[worst_day] if worst_day else 0.0
    ok = worst <= g["max_churn"]
    checks.append(Check("行为回放换手", ok,
                        f"最大单日前 K 变动 {worst:.0%}"
                        + (f"（{worst_day}）" if worst_day else "")
                        + f" / 上限 {g['max_churn']:.0%}（回放 {len(churn)} 天）"))

    # 6 冷却期
    la = last_accept_date()
    if la is None:
        ok, detail = True, "从未接受过变更"
    else:
        gap = _trading_days_between(la, today, all_days)
        ok = gap >= g["cooldown_days"]
        detail = f"距上次接受 {gap} 个交易日 / 要求 >= {g['cooldown_days']}"
    checks.append(Check("冷却期", ok, detail))

    # 7 在线稳健性否决（只否决不要求，样本不足时放行但记录）
    if online_p is not None:
        if online_days >= g.get("online_min_days", 5):
            ok = online_p >= g.get("online_veto_p", 0.25)
            checks.append(Check("在线稳健性", ok,
                                f"真值快照 {online_days} 天上 P(更好)="
                                f"{online_p:.2f} / 否决线 "
                                f"{g.get('online_veto_p', 0.25)}"))
        else:
            checks.append(Check("在线稳健性", True,
                                f"仅 {online_days} 天（< "
                                f"{g.get('online_min_days', 5)}），"
                                f"记录 P={online_p:.2f} 不否决"))

    accepted = all(c.passed for c in checks) and bool(moved)
    if not moved:
        checks.append(Check("有实际改动", False, "优化器给出的参数与当前一致"))
        accepted = False

    return Verdict(accepted, checks, moved,
                   {"oos_old": oos_old, "oos_new": oos_new,
                    "bootstrap_p": boot_p, "paired_delta": paired_delta,
                    "block_delta": list(block_delta) if block_delta else None,
                    "n_days": n_days, "worst_churn": worst,
                    "online_p": online_p, "online_days": online_days,
                    "intents": list(intents or [])})


def churn_by_day(old_top: dict[str, list], new_top: dict[str, list]
                 ) -> dict[str, float]:
    """每天清单的变动比例 = 1 − 交集 / 较长那张榜的长度。

    分母是**实际榜长**不是 top_k：生产清单常常不足 10 只（413 天里 24%），
    一只换手在 6 只的榜上是 1/6，摊成 1/10 就把闸门 5 量小了。

    两边都是空清单的日子不产生条目。以前 `max(..., 1)` 兜底成 1，
    「当天一只都发不出去」被算成换手 100%（2026-08-28 就是这么来的），
    生产口径下这种天有 3/413，留着会让闸门 5 永远不过。
    清单只有 1~2 只的日子照常计入且不打折：那种天换一只，用户收到的邮件
    就整封换了，闸门 5 判它「换手大」是对的。
    """
    out = {}
    for day, a in old_top.items():
        b = new_top.get(day, [])
        k = max(len(a), len(b))
        if k == 0:
            continue
        out[day] = 1.0 - len(set(a) & set(b)) / k
    return out


def log_verdict(today: str, verdict: Verdict, extra: dict | None = None
                ) -> None:
    """每次裁决记一行（接受与否都记）。同一天重复跑（本地一次、远端一次）
    只保留最后一次，面板上的「第 N 次裁决」才不会虚高。
    """
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    rows = read_verdicts()
    rows = [r for r in rows if r.get("date") != today]
    rows.append({
        "date": today,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "accepted": bool(verdict.accepted),
        "passed": [c.name for c in verdict.checks if c.passed],
        "failed": [c.name for c in verdict.checks if not c.passed],
        "moved": {k: list(v) for k, v in verdict.moved.items()},
        "evidence": verdict.evidence,
        **(extra or {}),
    })
    tmp = VERDICTS.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                           for r in rows), encoding="utf-8")
    import os
    os.replace(tmp, VERDICTS)


def read_verdicts() -> list[dict]:
    if not VERDICTS.exists():
        return []
    out = []
    for line in VERDICTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def accepted_count() -> int:
    if not HISTORY.exists():
        return 0
    return sum(1 for ln in HISTORY.read_text(encoding="utf-8").splitlines()
               if ln.strip())


def record(today: str, theta: dict, verdict: Verdict, metrics: dict) -> None:
    """只在接受时追加一行。这份文件是冷却期和审计的依据。"""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "date": today,
            "ts": dt.datetime.now().isoformat(timespec="seconds"),
            "theta": theta,
            "moved": {k: list(v) for k, v in verdict.moved.items()},
            "evidence": verdict.evidence,
            "metrics": metrics,
        }, ensure_ascii=False) + "\n")
