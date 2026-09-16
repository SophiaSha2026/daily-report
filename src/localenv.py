"""
tools/local.env -> os.environ。本机跑的每个入口都要先调一次。

为什么单独一个模块：SMTP 凭证和 OAuth token 只在这个文件里（gitignore），
谁忘了加载谁就静默发不出信 —— 2026-09-16 控制台的「学习会诊」按钮直接跑
`eval_daily.py --stage council`，不经过 local_run，会诊邮件报 'SMTP_HOST'
被 fail-open 吞成一行日志。以前 local_run 和 breakout/daily 各抄了一份，
第三份就该抽出来。

已存在的环境变量优先（云端用 GitHub Secrets 注入，不该被本地文件盖掉）。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "tools" / "local.env"


def load(path: Path | None = None) -> int:
    """读进环境变量，返回新设了几个。文件不在就返回 0，不抛。"""
    p = path or ENV
    n = 0
    try:
        if not p.exists():
            return 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k and v and "FILLME" not in v and k not in os.environ:
                os.environ[k] = v
                n += 1
    except Exception:  # noqa: BLE001
        return n
    return n
