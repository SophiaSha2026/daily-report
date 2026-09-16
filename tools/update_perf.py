"""
把 export.py 里印给用户的成绩常量，按实验产物重写一遍。

改了口径就必须重跑实验、重写常量，否则邮件里是旧口径的成绩给新模型背书
（历史教训 30）。selftest_breakout 会逐位对账两边，所以这两件事只能一起做。

    python tools/update_perf.py --show          只打印产物里的数，不改文件
    python tools/update_perf.py --from rolling  按滚动校正臂写（默认）
    python tools/update_perf.py --from fixed    按固定板块系数那一臂写

为什么默认用滚动校正臂：生产的 BOARD_ADJ 是在整个验证集上按板块命中率算出来
的，再拿同一个验证集评估等于自己给自己打分（审计 F8-11）。滚动臂每个月只用
之前已结束的月份估板块系数，是这条线上能给出的最诚实的数。两臂的数都写进
docs/breakout_log.md，邮件里印哪一个在这里选。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_breakout"
EXPORT = ROOT / "src" / "breakout" / "export.py"


def w5(grid: dict) -> dict[int, dict]:
    out = {}
    for r in grid.get("grid", []):
        if r.get("kind") != "W5":
            continue
        m = re.search(r"连续≥(\d)天", r.get("label", ""))
        if m:
            out[int(m.group(1))] = r
    return out


def w5c(grid: dict) -> dict | None:
    """满员日那一行（kind=W5c）。exp_window 还没写就返回 None。"""
    for r in grid.get("grid", []):
        if r.get("kind") == "W5c" and "满" in r.get("label", ""):
            return r
    return None


def load(arm: str) -> tuple[dict, dict, dict]:
    f = OUT / ("window_grid.json" if arm == "fixed" else "window_grid_rolling.json")
    if not f.exists():
        raise SystemExit(f"没有 {f.name}，先跑 exp_window.py"
                         + ("" if arm == "fixed" else " --adj rolling"))
    grid = json.loads(f.read_text(encoding="utf-8"))
    cal = json.loads((OUT / "score_calibration.json").read_text(encoding="utf-8"))
    board = {}
    bf = OUT / "board_hit.json"
    if bf.exists():
        board = json.loads(bf.read_text(encoding="utf-8"))
    return grid, cal, board


def show(arm: str) -> None:
    grid, cal, board = load(arm)
    rows = w5(grid)
    base = 100 * float(cal["base"])
    print(f"[{arm}] {grid.get('days')} 个交易日，基准 {base:.2f}%，"
          f"板块系数 {grid.get('board_adj_mode', '?')}")
    for k in sorted(rows, reverse=True):
        r = rows[k]
        print(f"  连续≥{k}天  n={r['n']:<5} 命中 {100 * r['hit']:.2f}%  "
              f"±{100 * r['se']:.2f}  倍数 {100 * r['hit'] / base:.2f}")
    c = w5c(grid)
    if c:
        print(f"  满员日   n={c['n']:<5} 命中 {100 * c['hit']:.2f}%  "
              f"天数 {c.get('empty', '?')}")
    if board.get("boards"):
        print("  按板块：", "，".join(
            f"{k} {100 * v['hit']:.1f}%（{v['n']}）"
            for k, v in sorted(board["boards"].items(), key=lambda x: -x[1]["n"])))
    for b in cal.get("bins", [])[::-1][:5]:
        print(f"  分数 {b['lo']}~{b['hi']}  n={b['n']:<7} 命中 {100 * b['hit']:.2f}%")


def rewrite(arm: str) -> None:
    grid, cal, board = load(arm)
    rows = w5(grid)
    if len(rows) < 5:
        raise SystemExit(f"W5 行只有 {len(rows)} 条，不够写 STREAK_PERF")
    base = round(100 * float(cal["base"]), 2)
    lines = []
    for k in (5, 4, 3, 2, 1):
        r = rows[k]
        hit = round(100 * r["hit"], 1)
        lines.append(f"    ({k}, {hit}, {round(hit / base, 1)}, {int(r['n'])}),")
    s = EXPORT.read_text(encoding="utf-8")
    s2 = re.sub(r"BASE = [\d.]+", f"BASE = {base}", s, count=1)
    s2 = re.sub(r"STREAK_PERF = \[\n(?:.*\n)*?\]",
                "STREAK_PERF = [\n    # 连续天数下限, 准确率%, 相对随便买的倍数, 样本数\n"
                + "\n".join(lines) + "\n]", s2, count=1)
    s2 = re.sub(r'PERF = \{"base": BASE, "window": "[^"]*"\}',
                f'PERF = {{"base": BASE, "window": "验证集 {grid.get("days")} 个交易日"}}', s2, count=1)
    c = w5c(grid)
    if c:
        s2 = re.sub(r"CAP_PERF = \([\d., ]+\)",
                    f"CAP_PERF = ({round(100 * c['hit'], 1)}, {int(c['n'])}, "
                    f"{int(c.get('days') or c.get('empty') or 0)})", s2, count=1)
    if s2 == s:
        raise SystemExit("一个常量都没改到，正则和文件对不上了")
    EXPORT.write_text(s2, encoding="utf-8")
    print(f"已按 [{arm}] 更新 export.py：BASE={base}，STREAK_PERF={lines}")
    print("接着跑 python src/selftest_breakout.py，它会逐位对账两边")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="arm", choices=["fixed", "rolling"], default="rolling")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show:
        for arm in ("fixed", "rolling"):
            try:
                show(arm)
            except SystemExit as e:
                print(e)
            print()
        return 0
    rewrite(a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
