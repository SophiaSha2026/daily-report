"""
把接受了的 θ 落到 state/learned.yaml。

写法是「临时文件 + os.replace」，因为这个文件在早上 07:30 会被
竞价流水线读。半个 YAML 会让 cfg.load() 走进异常分支（那里会退回人工
基线，不至于发不出信，但会静默丢掉学到的参数）。原子替换根除这种情况。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
LEARNED = ROOT / "state" / "learned.yaml"

_HEADER = """# 这个文件由学习系统自动生成，不要手改。
# 人工基线在 config.yaml，那份永远是 θ⁰，git diff 它只会看到人的意图。
# 删掉本文件 = 一键回到人工基线。
#
# 完整设计见 docs/learning.md；变更证据见 state/theta_history.jsonl。
"""


def write(theta: dict[str, float], evidence: dict, date: str) -> Path:
    LEARNED.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "as_of": date,
        "evidence": evidence,
        "params": {k: float(v) for k, v in theta.items()},
    }
    tmp = LEARNED.with_suffix(".yaml.tmp")
    tmp.write_text(_HEADER + yaml.safe_dump(doc, allow_unicode=True,
                                            sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, LEARNED)
    log.info("已写入 %s", LEARNED)
    return LEARNED


def rollback() -> bool:
    """删掉学到的参数，回到人工基线。变更邮件里给的就是这条路。"""
    if LEARNED.exists():
        LEARNED.unlink()
        log.info("已回滚到人工基线")
        return True
    return False
