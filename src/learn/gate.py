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

import numpy as np

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY = ROOT / "state" / "theta_history.jsonl"


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
    return len([d for d in all_days if a < d <= b])


def evaluate(theta_new: dict, theta_old: dict, box: dict, g: dict,
             n_days: int, all_days: list[str], today: str,
             boot_p: float, oos_new: float, oos_old: float,
             churn: dict[str, float], *,
             online_p: float | None = None,
             online_days: int = 0) -> Verdict:
    """跑完六道闸（外加可选的第七道），返回裁决。

    参数说明：
      g            config 里的 learning.gate 段
      boot_p       按天自助算出的 P(新参数样本外更好)
      churn        日期 -> 前 K 变动比例（闸门 5 的输入）
      online_p     P(新参数在**在线真值快照**上更好)。训练数据是回填表，
                   竞价轨迹是代理值；这道闸保证学到的东西搬到真值上
                   至少不明显更差。None = 无在线数据，跳过。
      online_days  参与在线检验的天数。少于 g["online_min_days"] 只记录不否决
                   ——几天的样本连「明显更差」都判不出来。
    """
    checks: list[Check] = []
    sig = {k: (hi - lo) for k, (lo, hi) in box.items()}
    moved = {k: (theta_old[k], theta_new[k]) for k in box
             if abs(theta_new[k] - theta_old[k]) > 1e-9}

    # 1 最少天数
    ok = n_days >= g["min_days"]
    checks.append(Check("最少天数", ok,
                        f"{n_days} 天 / 要求 >= {g['min_days']}"))

    # 2 走向前样本外改善
    ok = oos_new > oos_old
    checks.append(Check("样本外改善", ok,
                        f"Ĝ_oos {oos_old:+.4f} -> {oos_new:+.4f} "
                        f"（{oos_new - oos_old:+.4f}）"))

    # 3 自助显著性
    ok = boot_p >= g["bootstrap_p"]
    checks.append(Check("按天自助显著", ok,
                        f"P(更好)={boot_p:.3f} / 要求 >= {g['bootstrap_p']}"))

    # 4 步长上限 + 改动个数
    over = {k: abs(v[1] - v[0]) / sig[k] for k, v in moved.items()
            if abs(v[1] - v[0]) / sig[k] > g["max_step_frac"] + 1e-12}
    ok = not over and len(moved) <= g["max_moves"]
    checks.append(Check("步长与改动个数", ok,
                        f"动了 {len(moved)} 个（上限 {g['max_moves']}）"
                        + (f"；超步长: {list(over)}" if over else "")))

    # 5 行为回放
    worst = max(churn.values()) if churn else 0.0
    ok = worst <= g["max_churn"]
    checks.append(Check("行为回放换手", ok,
                        f"最大单日前 K 变动 {worst:.0%} / 上限 "
                        f"{g['max_churn']:.0%}（回放 {len(churn)} 天）"))

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
                    "bootstrap_p": boot_p, "n_days": n_days,
                    "worst_churn": worst,
                    "online_p": online_p, "online_days": online_days})


def churn_by_day(old_top: dict[str, list], new_top: dict[str, list]
                 ) -> dict[str, float]:
    """每天前 K 的变动比例 = 1 − 交集/K。"""
    out = {}
    for day, a in old_top.items():
        b = new_top.get(day, [])
        k = max(len(a), len(b), 1)
        out[day] = 1.0 - len(set(a) & set(b)) / k
    return out


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
