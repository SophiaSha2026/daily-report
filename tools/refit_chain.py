"""
月度重估链（计划 3.6）：教训 36 那条手跑的流水，做成一条命令。

    python tools/refit_chain.py           全部跑：成绩表 -> 提升网格 -> 分档 -> 邮件常量 -> 板块命中率 -> 自测
    python tools/refit_chain.py --dry     只打印要做什么

local_run.flow_breakout 在模型到期那天、打分之前调它（model_due），所以成绩表、板块系数、
邮件常量和当天重训的模型同一天刷新。手改 overrides.json 之后也该手跑一次。

六步，哪一步失败就停在哪一步、退出码非 0；前面几步写下的产物不回滚（都是可重跑的派生物，
邮件的失效保护 export._same_rule 会把不一致标出来，不会静默）。每步有时限。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_breakout"
PY = sys.executable


def bj() -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).strftime("%H:%M:%S")


def run(label: str, args: list[str], timeout: int) -> None:
    print(f"{bj()} [{label}] {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY, *args], cwd=str(ROOT), timeout=timeout,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                            "PYTHONUTF8": "1"})
    if r.returncode != 0:
        raise SystemExit(f"[{label}] 退出码 {r.returncode}，重估链停在这一步")
    print(f"{bj()} [{label}] 完成 {time.time() - t0:.0f}s", flush=True)


def promote_grid() -> None:
    """window_grid_shrink.json -> window_grid.json：板块系数四舍五入到 4 位和 board_adj.json 一致，
    否则 export._same_rule 比出 1e-5 的差把当天清单标成「旧规则」。"""
    src, dst = OUT / "window_grid_shrink.json", OUT / "window_grid.json"
    g = json.loads(src.read_text(encoding="utf-8"))
    g["board_adj"] = {k: round(float(v), 4) for k, v in g["board_adj"].items()}
    g["note_adj"] = ("板块系数是经验贝叶斯收缩、逐月只用过去的月份估的（收盘口径的名额命中）；"
                     "这里的因子是最后一个验证月之后该用的那组，即生产此刻在用的"
                     "（state/breakout/board_adj.json）。W5 行是买得到口径，W5close 是收盘口径。"
                     f"由 tools/refit_chain.py 于 {dt.date.today()} 提升。")
    dst.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{bj()} [提升网格] {g['board_adj']}", flush=True)


def board_hit() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src" / "breakout"))
    import importlib
    importlib.import_module("localenv")
    import truth as T
    r = T.validation_by_board(refresh=True)
    print(f"{bj()} [板块命中率] " + json.dumps(
        {k: (v["n"], round(100 * v["hit"], 1)) for k, v in r.get("boards", {}).items()},
        ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    steps = [
        ("成绩表", lambda: run("成绩表", ["src/breakout/exp_window.py", "--refit", "--adj", "shrink",
                                           "--save-adj"], 20 * 60)),
        ("提升网格", promote_grid),
        ("分档", lambda: run("分档", ["src/breakout/exp_calib.py"], 15 * 60)),
        ("邮件常量", lambda: run("邮件常量", ["tools/update_perf.py"], 5 * 60)),
        ("板块命中率", board_hit),
        ("自测", lambda: run("自测", ["src/selftest_breakout.py"], 5 * 60)),
    ]
    if a.dry:
        for name, _ in steps:
            print("会跑：", name)
        return 0
    t0 = time.time()
    for name, fn in steps:
        fn()
    print(f"{bj()} 重估链全部完成，{(time.time() - t0) / 60:.0f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
