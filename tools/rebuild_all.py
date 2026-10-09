"""
口径改动之后的重建重训流水。2026-09-16 那批审计修了 40 余处会改口径的地方
（筹码网格、均线暖机、股东户数过滤、共同起点……），
特征表和训练表必须整个重建，成绩常量也要按新表重算，否则邮件里印的是
旧口径的成绩给新模型背书（历史教训 30）。

    python tools/rebuild_all.py            全套
    python tools/rebuild_all.py --only breakout   只重建起涨预测那条
    python tools/rebuild_all.py --dry             只打印要跑什么

顺序有讲究：
  起涨预测  build（特征表，约 20 分钟）-> refit（模型，指纹变了本来也会自动重训，
            重训验收 daily.accept_model 在这一步里）
            -> tools/refit_chain.py（逐月滚动 + 收缩板块系数 + 分档 + 邮件常量 +
               按板块命中率 + 自测；和每月模型到期那天自动跑的是同一条）
  （早盘那条 build-train 2026-09-27 随早盘系统归档；长期调整突破没有训练表，
   规则改了跑 python src/pullback_backtest.py 看频率就行）

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
# 日志时间一律北京时间（控制台「运行记录」直接印日志，界面只用北京时间）。
# 必须包 staticmethod：直接赋 lambda 会被绑成方法，每条日志都报错并丢掉
logging.Formatter.converter = staticmethod(lambda t: time.gmtime(t + 8 * 3600))
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


def step_chain() -> bool:
    """成绩表 / 板块系数 / 分档 / 邮件常量 / 按板块命中率 / 自测：一律交给 tools/refit_chain.py。

    以前这几步在这里另写了一份（逐月滚动直接写 window_grid.json、板块系数没四舍五入、
    不改写邮件常量、还多跑一个早就不用的滚动臂），和每月自动跑的那条链各演各的（教训 34）。
    """
    return run([sys.executable, str(ROOT / "tools" / "refit_chain.py")], timeout=3600) == 0


STEPS = {
    "breakout": [("建特征表", step_build), ("重训模型", step_refit),
                 ("月度重估链", step_chain)],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["breakout"], default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--from-step", default="", help="从这一步开始（名字前缀）")
    a = ap.parse_args()
    lines = ["breakout"] if not a.only else [a.only]
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
