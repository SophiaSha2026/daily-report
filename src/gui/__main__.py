"""控制台入口。见 gui/__init__.py。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 直接 `python src/gui/__main__.py` 跑时包还没在路径上
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.server import serve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(prog="gui", description="A股流水线 控制台")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()
    try:
        serve(port=a.port, open_browser=not a.no_open)
    except OSError as e:
        # 最常见的是端口被上一个还没退干净的实例占着
        print(f"起不来（端口 {a.port}）：{e}")
        print(f"换个端口：python -m gui --port {a.port + 1}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
