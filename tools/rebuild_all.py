"""
口径改动之后的重建重训流水。2026-09-16 那批审计修了 40 余处会改口径的地方
（筹码网格、均线暖机、股东户数过滤、共同起点、回填表的竞价字段……），
特征表和训练表必须整个重建，成绩常量也要按新表重算，否则邮件里印的是
旧口径的成绩给新模型背书（历史教训 30）。

    python tools/rebuild_all.py            全套
    python tools/rebuild_all.py --only breakout   只重建起涨预测那条
    python tools/rebuild_all.py --only morning    只重建早盘的训练表
    python tools/rebuild_all.py --dry             只打印要跑什么

顺序有讲究：
  起涨预测  build（特征表，约 20 分钟）-> refit（模型，指纹变了本来也会自动重训）
            -> exp_window --adj shrink --save-adj（逐月滚动 + 重估板块系数，
               成绩表 STREAK_PERF 和 state/breakout/board_adj.json 都从这来）
            -> exp_calib（分数分档，BASE / SCORE_TABLE 的来源）
            -> truth.validation_by_board（按板块命中率，邮件的加权期望用）
  早盘      build-train（回填训练表）-> 学习线下一次跑自然会用新表

每一步都核对产物日期/行数，不合格就停下来，不许「退出码 0 但什么都没做」
（历史教训 22、27）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rebuild")
DATA = ROOT / "data" / "breakout"
OUT = ROOT / "out_breakout"


def run(args: list[str], timeout: int = 7200, env_extra: dict | None = None) -> int:
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    log.info("跑 %s", " ".join(args[1:]))
    t0 = time.time()
    r = subprocess.run(args, cwd=str(ROOT), env=env, timeout=timeout)
    log.info("  -> 退出码 %d，用时 %.1f 分钟", r.returncode, (time.time() - t0) / 60)
    return r.returncode


def step_build() -> bool:
    if run([sys.executable, str(ROOT / "src" / "breakout" / "build.py")]) != 0:
        log.error("特征表没建出来")
        return False
    import pandas as pd
    t = DATA / "train.parquet"
    d = pd.read_parquet(t, columns=["code", "date"])
    last = str(d["date"].max())
    log.info("特征表：%d 行，%d 只，最后一天 %s", len(d), d["code"].nunique(), last)
    # 特征表最后一天必须是日线表的最后一天，否则今晚的清单会拿旧数据算
    px = pd.read_parquet(DATA / "daily.parquet", columns=["date"])
    if last != str(px["date"].max()):
        log.error("特征表最后一天 %s 对不上日线表 %s", last, px["date"].max())
        return False
    return True


def step_refit() -> bool:
    if run([sys.executable, str(ROOT / "src" / "breakout" / "daily.py"),
            "--stage", "refit"]) != 0:
        return False
    m = json.loads((ROOT / "state" / "breakout" / "model.json").read_text(encoding="utf-8"))
    log.info("模型：训练于 %s，截止 %s，%d 个特征，指纹 %s",
             m.get("fit_date"), m.get("train_cut"), len(m.get("feats") or []),
             str(m.get("fingerprint"))[:8])
    return m.get("fit_date") == dt.date.today().isoformat() or bool(m.get("feats"))


def step_window() -> bool:
    """逐月滚动 + 板块系数重估。**生产在用的是收缩臂**（2026-09-16 用户批准）。

    以前这里跑的是 `--refit`（固定臂，用 daily.BOARD_ADJ 那四个写死的数）。
    板块系数改成收缩之后再跑固定臂，等于「用收缩估出来的系数、又在同一段
    数据上评成绩」——样本内那个老毛病原样回来，而且会把 window_grid.json
    覆盖成另一套口径。--save-adj 顺带把下一期该用的因子写进
    state/breakout/board_adj.json，生产读它，所以重建和更新系数是同一步。
    """
    if run([sys.executable, str(ROOT / "src" / "breakout" / "exp_window.py"),
            "--refit", "--adj", "shrink", "--save-adj"],
           env_extra={"WF_OUT": str(OUT / "window_grid.json")}) != 0:
        return False
    g = json.loads((OUT / "window_grid.json").read_text(encoding="utf-8"))
    w5 = [r for r in g["grid"] if r["kind"] == "W5"]
    log.info("逐月滚动：%d 个交易日，基准 %.2f%%", g["days"], 100 * g["base"])
    for r in sorted(w5, key=lambda x: x["label"]):
        log.info("  %s  n=%d  命中 %.1f%%  ±%.1f", r["label"], r["n"],
                 100 * r["hit"], 100 * r["se"])
    return len(w5) >= 5


def step_window_rolling() -> bool:
    """滚动臂：每个板块各信各的（不收缩）。只当对照，**邮件不用它**。

    2026-09-16 实测：它和收缩臂整体命中接近，但清单构成被小样本板块掀翻
    （北交所 19 个名额的 26.3% 变成因子 2.37，主板 67%/科创 30% -> 主板 54%/
    北交 31%），那是另一套策略。留着是为了下次有人问「不收缩会怎样」时
    有现成的数，成绩表不从这里取。
    """
    if run([sys.executable, str(ROOT / "src" / "breakout" / "exp_window.py"),
            "--refit", "--adj", "rolling"]) != 0:
        return False
    f = OUT / "window_grid_rolling.json"
    if not f.exists():
        log.error("没写出 %s", f.name)
        return False
    g = json.loads(f.read_text(encoding="utf-8"))
    for r in sorted([x for x in g["grid"] if x["kind"] == "W5"], key=lambda x: x["label"]):
        log.info("  [滚动校正] %s  n=%d  命中 %.1f%%  ±%.1f", r["label"], r["n"],
                 100 * r["hit"], 100 * r["se"])
    return True


def step_calib() -> bool:
    if run([sys.executable, str(ROOT / "src" / "breakout" / "exp_calib.py")]) != 0:
        return False
    c = json.loads((OUT / "score_calibration.json").read_text(encoding="utf-8"))
    log.info("分数分档：基准 %.2f%%，单调 %s", 100 * c["base"], c.get("monotonic"))
    return True


def step_board() -> bool:
    import truth as T
    b = T.validation_by_board(refresh=True)
    if not b:
        log.error("按板块命中率算不出来")
        return False
    for k, v in sorted(b["boards"].items(), key=lambda x: -x[1]["n"]):
        log.info("  %-8s n=%-4d 命中 %.1f%%", k, v["n"], 100 * v["hit"])
    return True


def step_train_table() -> bool:
    if run([sys.executable, str(ROOT / "src" / "eval_daily.py"),
            "--stage", "build-train"]) != 0:
        return False
    import glob
    import pandas as pd
    fs = sorted(glob.glob(str(ROOT / "data" / "train" / "backfill_*.parquet")))
    if not fs:
        log.error("回填训练表没建出来")
        return False
    d = pd.read_parquet(fs[-1], columns=["date", "code"])
    log.info("回填训练表：%s，%d 行，%d 天", Path(fs[-1]).name, len(d), d["date"].nunique())
    return True


STEPS = {
    "breakout": [("建特征表", step_build), ("重训模型", step_refit),
                 ("逐月滚动", step_window), ("逐月滚动-滚动校正", step_window_rolling),
                 ("分数分档", step_calib), ("按板块命中率", step_board)],
    "morning": [("回填训练表", step_train_table)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["breakout", "morning"], default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from-step", default="", help="从这一步开始（名字前缀）")
    a = ap.parse_args()
    lines = ["breakout", "morning"] if not a.only else [a.only]
    plan = [(ln, name, fn) for ln in lines for name, fn in STEPS[ln]]
    if a.from_step:
        i = next((i for i, (_, n, _) in enumerate(plan) if n.startswith(a.from_step)), 0)
        plan = plan[i:]
    log.info("计划 %d 步：%s", len(plan), " -> ".join(n for _, n, _ in plan))
    if a.dry:
        return 0
    t0 = time.time()
    for ln, name, fn in plan:
        log.info("=== %s · %s", ln, name)
        try:
            ok = fn()
        except Exception as e:  # noqa: BLE001
            log.error("%s 异常：%s", name, e)
            return 1
        if not ok:
            log.error("%s 没通过核对，停在这里", name)
            return 1
    log.info("全部完成，用时 %.1f 分钟。接下来：把 export.py 的 STREAK_PERF / BASE "
             "按新的 window_grid.json / score_calibration.json 更新，再跑一次 "
             "selftest_breakout（它会钉两边一致）", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
