"""
配置加载器：人工基线 + 机器学到的增量。

两个文件的分工：

    config.yaml           人工书写，永远是 θ⁰。git diff 它，看到的只有人的意图。
    state/learned.yaml    学习系统写的当前 θ。删掉它 = 一键回到人工基线。

`load()` 做浅层合并（只覆盖叶子标量），返回合并后的 dict。
所有流水线都应该走这个函数，不要再各自 yaml.safe_load(config.yaml)。

失败隔离：learned.yaml 读不了、格式不对、含有不允许的键，一律记 warning
然后**返回纯 config.yaml**。学习系统崩掉不能让早盘发不出信。
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "config.yaml"
LEARNED = ROOT / "state" / "learned.yaml"

# learned.yaml 只允许改这些前缀下的叶子。学习系统再怎么出错，
# 也改不动快照时刻、发信阈值、准入区间。
_ALLOWED_PREFIX = (
    "scoring.weights.",
    "screen.gap_pct_peak",
    "screen.auc_ratio_score_hi",
    "screen.auc_ratio_decay",
)


def _flat(d: dict, prefix: str = "") -> dict[str, Any]:
    """把嵌套 dict 摊平成 'a.b.c' -> value。只摊 dict，list 当叶子。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _set_path(d: dict, path: str, value: Any) -> None:
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _allowed(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _ALLOWED_PREFIX)


def base() -> dict:
    """只读人工基线 θ⁰。回归测试和锚定项用。"""
    return yaml.safe_load(BASE.read_text(encoding="utf-8"))


def learned() -> dict[str, Any]:
    """当前生效的学到值，摊平成 'a.b.c' -> value。读不到就是空。"""
    if not LEARNED.exists():
        return {}
    try:
        raw = yaml.safe_load(LEARNED.read_text(encoding="utf-8")) or {}
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise TypeError("learned.yaml 的 params 不是 dict")
        bad = [k for k in params if not _allowed(k)]
        if bad:
            log.warning("learned.yaml 含不允许的键，整份忽略: %s", bad)
            return {}
        return params
    except Exception as e:  # noqa: BLE001
        log.warning("learned.yaml 读取失败，退回人工基线: %s", e)
        return {}


def load() -> dict:
    """合并后的配置。所有流水线的唯一入口。"""
    c = base()
    for path, val in learned().items():
        _set_path(c, path, val)
    return c


def diff() -> list[tuple[str, Any, Any]]:
    """(参数路径, 人工基线值, 学到值)，只列真的变了的。变更邮件用。"""
    flat0 = _flat(base())
    out = []
    for path, val in learned().items():
        old = flat0.get(path)
        if old != val:
            out.append((path, old, val))
    return sorted(out)


def theta0(box: dict[str, list]) -> dict[str, float]:
    """按 box 里列的参数路径，从人工基线取出 θ⁰ 向量。"""
    flat = _flat(base())
    return {k: float(flat[k]) for k in box}


def theta_now(box: dict[str, list]) -> dict[str, float]:
    """当前生效的 θ（基线 + 学到的）。"""
    flat = _flat(load())
    return {k: float(flat[k]) for k in box}


def apply_theta(c: dict, theta: dict[str, float]) -> dict:
    """把一个 θ 覆盖进配置的副本，不改原对象。打分回放用。"""
    out = copy.deepcopy(c)
    for path, val in theta.items():
        _set_path(out, path, val)
    return out
