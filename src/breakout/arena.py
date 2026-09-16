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

import daily as D             # noqa: E402
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


def _git_head() -> str:
    """当前提交。拿不到就空串：记录不全好过因为没装 git 就跑不了验收。"""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def guard_holdout(config: dict) -> None:
    if HOLDOUT_MARK.exists():
        prev = json.loads(HOLDOUT_MARK.read_text(encoding="utf-8"))
        log.error("holdout 已经用过了（%s），不能再用。", prev.get("at"))
        log.error("上次的配置：%s", json.dumps(prev.get("config"),
                                             ensure_ascii=False))
        log.error("上次是哪个模型：指纹 %s，%d 个特征，提交 %s",
                  prev.get("fingerprint", "未记录"),
                  len(prev.get("features") or []),
                  (prev.get("commit") or "未记录")[:8])
        log.error("再看一次就不再是 holdout，而是第二个验证集。"
                  "要重新验收，只能换一段全新的、从未参与过的时间。")
        raise SystemExit(2)
    OUT.mkdir(parents=True, exist_ok=True)
    # 先写、后跑：run() 中途炸了这段 holdout 也算用掉了，纪律不能靠「跑成功」
    HOLDOUT_MARK.write_text(json.dumps(
        {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "config": config,
         "commit": _git_head()},
        ensure_ascii=False, indent=2), encoding="utf-8")


def stamp_holdout(feats: list[str], df: pd.DataFrame) -> None:
    """把「用掉 holdout 的是哪个模型」补进标记。

    2026-09-16 审计：旧标记只有时间和 {top_n, quick} 两个开关，下次守卫
    报错时说不清当时验收的是哪份特征、哪个训练表。而 09-12 那次 14.43%
    的成绩后来发现协议本身有前视偏差，想追认是哪个模型都追认不了。
    """
    try:
        payload = json.loads(HOLDOUT_MARK.read_text(encoding="utf-8"))
        payload.update({
            "features": list(feats), "n_features": len(feats),
            "fingerprint": D.feature_fingerprint(df),
            "rows": int(len(df)),
            "date_range": [str(df["date"].min()), str(df["date"].max())]})
        HOLDOUT_MARK.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        # 补记失败不能阻断验收本身，但别静默：标记文件已经写下了
        log.warning("holdout 标记补记失败：%s", e)


def run(with_holdout: bool = False, top_n: int | None = None,
        quick: bool = False, with_l2: bool = False) -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(with_holdout)
    feats_all = [c for c in df.columns if "__" in c]
    log.info("训练表 %d 行 × %d 特征，%s .. %s",
             len(df), len(feats_all), df.date.min(), df.date.max())

    # ---- 特征选择：只在训练段上做，而且和走向前第一个月同一条净化线 ----
    # 以前切的是 date < TRAIN_END（2025-03-01），而第一个测试月就是 2025-03：
    # 紧挨着的那 20 个交易日（106,779 行、4,473 个正样本，占筛选集正样本的
    # 5.19%）的 y_up 全部或部分由 2025-03 的最高价算出来，等于让「已知 3 月
    # 谁涨了」去投特征的去留票。2026-09-16 实测 2025-02 整月对四个临界特征
    # 投的都是「留」（range_compress__mean 月度 IC +0.0423，而 2025-01 是负的），
    # 净化后总 IC 0.0071 -> 0.0053，离 IC_MIN=0.005 只剩 0.0003。
    # train_slice 的截止日就是 walk_forward 第一个月的训练截止日（S21 + S10）。
    tr = V.train_slice(df, V.TRAIN_END[:7])
    log.info("特征选择（只用 %s 之前、标签窗口不碰测试月的 %d 行）",
             V.purge_cut(df, V.TRAIN_END[:7]), len(tr))
    rep = FS.run(tr, feats_all, y="y_up", out_dir=OUT)
    feats = rep["keep"]
    log.info("特征 %d -> %d", rep["start"], rep["end"])
    if len(feats) < 3:
        log.error("筛完只剩 %d 个特征，没法继续。检查数据质量或放宽阈值。",
                  len(feats))
        return 1
    if with_holdout:
        stamp_holdout(feats, df)

    # ---- 擂台 ----
    zoo = [("L0_logistic", lambda: M.L0Logistic()),
           ("L1_lightgbm", lambda: M.L1Lgbm(
               n_estimators=150 if quick else 400))]
    # L2 默认不进擂台：2026-09-12 在验证集上实测它比 L0 线性尺子还差
    # 4.06 个百分点（4.40% vs 8.45%）。原因是 model.py 的 _seq 从三个
    # 统计量线性重建 5 步序列，这个近似不但丢信息还引入了不存在的平滑
    # 结构。要救它得在 build 阶段落逐日 [5x26] 真序列（训练表膨胀 5 倍）。
    # 用 --with-l2 可以强行拉它进来复现这个结论。
    if with_l2:
        if M.L2Gru.available():
            zoo.append(("L2_gru", lambda: M.L2Gru(epochs=3 if quick else 6)))
        else:
            log.warning("torch 没装，L2 序列模型跳过")

    end = "2026-10-01" if with_holdout else V.VALID_END
    results = {}
    for name, factory in zoo:
        log.info("--- %s 走向前 ---", name)
        # 验收必须按**生产规则**评：≥SCORE_MIN 分、板块校正、CAP_A 只。
        # 2026-09-12 那次唯一的 holdout 验收评的是「每天固定 10 只、无门槛、
        # 无板块校正」这条已经不存在的规则，主指标假阴（14.16% 未达标，
        # 生产口径 16~18% 达标）、板块检查假阳（1.63 通过，生产口径 6.20
        # 重度不通过），两个方向相反，所以那次验收对上线规则没有证明力。
        by_month, extra = V.walk_forward(
            df, feats, factory, start=V.TRAIN_END, end=end,
            top_n=top_n, score_min=D.SCORE_MIN, board_adj=D.BOARD_ADJ)
        if not len(by_month):
            log.warning("%s 没有有效月份，跳过", name)
            continue
        # holdout 模式下走向前跨了验证集段和 holdout 段两截，
        # **验收只能看 holdout 那一截**，混在一起算等于让已经看过的
        # 验证集月份去稀释（或美化）最终成绩。
        if with_holdout:
            hp = extra["picks"][extra["picks"]["month"] >= HOLDOUT_START[:7]]
            hb = by_month[by_month["month"] >= HOLDOUT_START[:7]]
            hd = df[df["date"] >= HOLDOUT_START]
            acc = V.acceptance(hb, hp, hd)
            acc_valid = V.acceptance(
                by_month[by_month["month"] < HOLDOUT_START[:7]],
                extra["picks"][extra["picks"]["month"] < HOLDOUT_START[:7]],
                df[df["date"] < HOLDOUT_START])
            log.info("%s 验证集段 %.2f%%（%.1f倍）",
                     name, 100 * acc_valid["overall_hit"], acc_valid["lift"])
        else:
            acc = V.acceptance(by_month, extra["picks"], df)
        results[name] = {"by_month": by_month, "acc": acc}
        log.info("%s %s %.2f%%（基础 %.2f%%，%.1f倍）验收 %s",
                 name, "HOLDOUT 命中" if with_holdout else "总命中",
                 100 * acc["overall_hit"], 100 * acc["base_rate"],
                 acc["lift"], "通过" if acc["ok"] else "不通过")

    if not results:
        log.error("没有任何模型跑出结果")
        return 1

    best = max(results, key=lambda k: results[k]["acc"]["overall_hit"])
    log.info("=== 擂台冠军：%s ===", best)

    payload = {
        "with_holdout": with_holdout,
        "rule": V.rule_stamp(cap=top_n),
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
    ap.add_argument("--top-n", type=int, default=D.CAP_A,
                    help="清单 A 每天几只（默认取生产的 CAP_A）")
    ap.add_argument("--quick", action="store_true", help="减少迭代，快速冒烟")
    ap.add_argument("--with-l2", action="store_true",
                    help="把 L2 序列模型拉回擂台（已证明是负贡献）")
    a = ap.parse_args()
    if a.holdout:
        guard_holdout({"top_n": a.top_n, "quick": a.quick})
    return run(a.holdout, a.top_n, a.quick, a.with_l2)


if __name__ == "__main__":
    sys.exit(main())
