"""
模型擂台主脚本：特征选择 -> 四层模型 -> 走向前验证 -> 验收表。

    python src/breakout/arena.py                  在验证集上比，选模型
    python src/breakout/arena.py --holdout        动 holdout，全程只许一次

holdout 纪律用代码强制
----------------------
2026-01..09 这九个月是 holdout，默认**任何路径都读不到它**：
load() 直接把这段切掉。要碰它必须显式加 --holdout，而且脚本会：

    1. 先检查 out_breakout/holdout_used.json 在不在
    2. 在 -> 拒绝运行，打印上次动它的时间和当时的模型配置
    3. 不在 -> 跑，然后写下这个文件

为什么要做到这个程度：擂台有 4 层模型、每层若干超参，反复在验证集上比较
本身就是一种搜索，搜到最后验证集会被间接拟合。holdout 是唯一能给出诚实
估计的东西，而"只看一次"这条纪律靠自觉是守不住的——每次都觉得"这次
改动很小，再看一眼没关系"。所以写成代码。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = ROOT / "data" / "breakout"
OUT = ROOT / "out_breakout"

import fselect as FS          # noqa: E402
import model as M             # noqa: E402
import validate as V          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("arena")

HOLDOUT_START = "2026-01-01"
HOLDOUT_MARK = OUT / "holdout_used.json"


def load(with_holdout: bool = False) -> pd.DataFrame:
    """读训练表。特征列一律降到 float32。

    426 万行 x 87 特征，float64 是 3.3GB，加上走向前每月切片的复制，
    在 14GB 的机器上会很紧。float32 直接砍一半，而特征本身是横截面
    百分位（值域 [0,1]），float32 的精度远远够用。
    """
    df = pd.read_parquet(DATA / "train.parquet")
    fc = [c for c in df.columns if "__" in c]
    df[fc] = df[fc].astype("float32")
    for c in ("y_up", "y_t0", "y_top", "close", "float_mcap"):
        if c in df.columns:
            df[c] = df[c].astype("float32")
    if not with_holdout:
        n0 = len(df)
        df = df[df["date"] < HOLDOUT_START].copy()
        log.info("已切掉 holdout：%d -> %d 行（%s 之后不可见）",
                 n0, len(df), HOLDOUT_START)
    log.info("内存占用约 %.1f GB", df.memory_usage(deep=False).sum() / 2**30)
    return df


def guard_holdout(config: dict) -> None:
    if HOLDOUT_MARK.exists():
        prev = json.loads(HOLDOUT_MARK.read_text(encoding="utf-8"))
        log.error("holdout 已经用过了（%s），不能再用。", prev.get("at"))
        log.error("上次的配置：%s", json.dumps(prev.get("config"),
                                             ensure_ascii=False))
        log.error("再看一次就不再是 holdout，而是第二个验证集。"
                  "要重新验收，只能换一段全新的、从未参与过的时间。")
        raise SystemExit(2)
    OUT.mkdir(parents=True, exist_ok=True)
    HOLDOUT_MARK.write_text(json.dumps(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "config": config},
        ensure_ascii=False, indent=2), encoding="utf-8")


def run(with_holdout: bool = False, top_n: int = 10,
        quick: bool = False) -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(with_holdout)
    feats_all = [c for c in df.columns if "__" in c]
    log.info("训练表 %d 行 × %d 特征，%s .. %s",
             len(df), len(feats_all), df.date.min(), df.date.max())

    # ---- 特征选择：只在训练段上做 ----
    tr = df[df["date"] < V.TRAIN_END]
    log.info("特征选择（只用 %s 之前的 %d 行）", V.TRAIN_END, len(tr))
    rep = FS.run(tr, feats_all, y="y_up", out_dir=OUT)
    feats = rep["keep"]
    log.info("特征 %d -> %d", rep["start"], rep["end"])
    if len(feats) < 3:
        log.error("筛完只剩 %d 个特征，没法继续。检查数据质量或放宽阈值。",
                  len(feats))
        return 1

    # ---- 擂台 ----
    zoo = [("L0_logistic", lambda: M.L0Logistic()),
           ("L1_lightgbm", lambda: M.L1Lgbm(
               n_estimators=150 if quick else 400))]
    if M.L2Gru.available():
        zoo.append(("L2_gru", lambda: M.L2Gru(epochs=3 if quick else 6)))
    else:
        log.warning("torch 没装，L2 序列模型跳过")

    end = "2026-10-01" if with_holdout else V.VALID_END
    results = {}
    for name, factory in zoo:
        log.info("--- %s 走向前 ---", name)
        by_month, extra = V.walk_forward(
            df, feats, factory, start=V.TRAIN_END, end=end, top_n=top_n)
        if not len(by_month):
            log.warning("%s 没有有效月份，跳过", name)
            continue
        acc = V.acceptance(by_month, extra["picks"], df)
        results[name] = {"by_month": by_month, "acc": acc}
        log.info("%s 总命中 %.2f%%（基础 %.2f%%，%.1f倍）验收 %s",
                 name, 100 * acc["overall_hit"], 100 * acc["base_rate"],
                 acc["lift"], "通过" if acc["ok"] else "不通过")

    if not results:
        log.error("没有任何模型跑出结果")
        return 1

    best = max(results, key=lambda k: results[k]["acc"]["overall_hit"])
    log.info("=== 擂台冠军：%s ===", best)

    payload = {
        "with_holdout": with_holdout,
        "n_features": len(feats), "features": feats,
        "best": best,
        "models": {k: {"acceptance": v["acc"],
                       "by_month": v["by_month"].to_dict("records")}
                   for k, v in results.items()},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    fn = "arena_holdout.json" if with_holdout else "arena.json"
    (OUT / fn).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     default=float), encoding="utf-8")
    log.info("报告 -> %s  用时 %.1f 分钟", OUT / fn, (time.time() - t0) / 60)

    # L0 是尺子：看看 L1/L2 相对它提升了多少
    if "L0_logistic" in results and len(results) > 1:
        base = results["L0_logistic"]["acc"]["overall_hit"]
        for k, v in results.items():
            if k == "L0_logistic":
                continue
            gain = v["acc"]["overall_hit"] - base
            log.info("%s 相对 L0 尺子 %+.2f 个百分点%s", k, 100 * gain,
                     "（提升有限，说明瓶颈在特征不在模型）"
                     if abs(gain) < 0.01 else "")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="动 holdout。全程只许一次，脚本会拦第二次")
    ap.add_argument("--top-n", type=int, default=10, help="清单 A 每天几只")
    ap.add_argument("--quick", action="store_true", help="减少迭代，快速冒烟")
    a = ap.parse_args()
    if a.holdout:
        guard_holdout({"top_n": a.top_n, "quick": a.quick})
    return run(a.holdout, a.top_n, a.quick)


if __name__ == "__main__":
    sys.exit(main())
