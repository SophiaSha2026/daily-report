"""
补发起涨预测的历史清单。用户 2026-09-13 要「过去 5 个交易日的 A/B 清单」。

不重跑 scan：那会拿**今天**的 ST / 减持 / 解禁数据去剔除历史那天的票，
剔出来的东西和当时不一样，而且会覆盖已入库的每日清单
（data/breakout/YYYY-MM/breakout_*.parquet，连续天数靠它数）。
这里直接读那些已落盘的清单，只重新渲染成新格式再发。

清单 B 按当天为止的 A 池现算，价格只看到那一天（build_list_b 的 asof），
不许用之后的价格 —— 补发不是马后炮。

    python tools/resend_breakout.py --days 5              发信
    python tools/resend_breakout.py --days 5 --dry-run    只生成面板，不发信
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

import daily as D           # noqa: E402
import export as E          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("resend")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    D.load_env()
    files = sorted(glob.glob(str(D.DATA / "*" / "breakout_*.parquet")))
    files = files[-a.days:]
    if not files:
        log.error("没有已落盘的每日清单，先跑「起涨预测」")
        return 1
    log.info("补发 %d 天：%s", len(files),
             ", ".join(Path(f).stem[-10:] for f in files))

    px = pd.read_parquet(D.DATA / "daily.parquet",
                         columns=["code", "date", "close"])

    # 模型日期要在改 STATE 之前读：model_status() 找的是 STATE/model.json，
    # 换成临时目录后就找不到了，2026-09-13 那批补发邮件头上因此印着
    # 「模型训练于 ?」。
    model_date = D.model_status().get("fit_date", "?")

    # A 池在临时目录里重建，不碰 state/breakout/a_pool.json。
    # 补发要还原「当天为止」的池，拿生产那份（已经含 5 天全部）会让
    # 09-07 那天的清单 B 看到 09-11 才进池的票。
    tmp = Path(tempfile.mkdtemp(prefix="resend_"))
    real_state = D.STATE
    D.STATE = tmp
    # 但「当天为止」不等于「只有这 5 天」：从空池起算的话，更早上过清单 A 的票
    # 永远进不了补发的清单 B，first 也会被改晚（S17）。窗口之前的池按已落盘的
    # 清单折叠出来做种，只有窗口内这几天是现算的
    first_date = Path(files[0]).stem[-10:]
    seed = D.replay_pool(before=first_date)
    (tmp / "a_pool.json").write_text(
        json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("A 池做种：%s 之前折叠出 %d 条", first_date, len(seed))
    try:
        for f in files:
            picks = pd.read_parquet(f)
            date = str(picks["date"].iloc[0])
            pool = D.update_pool(picks, date)
            blist = D.build_list_b(px, pool, asof=date)
            # 落盘的清单只有入选的票，当天剔除了几只已经不可知，
            # 给 None 让邮件头不印这一段，不要印个假的 0。
            meta = {"date": date, "n_a": len(picks), "n_b": len(blist),
                    "score_min": D.SCORE_MIN, "cap_a": D.CAP_A,
                    "n_streak3": int((picks["streak"] >= 3).sum()),
                    "pool": len(pool),
                    "model_date": model_date,
                    "rejected": None}
            # 够格总数和剔除数不一样：它**落在清单的 parquet 里**（2026-09-16
            # 起每行都带），有就照印，没有那天就不印
            if "n_qualified" in picks.columns and len(picks):
                nq = picks["n_qualified"].iloc[0]
                if pd.notna(nq):
                    meta["n_qualified"] = int(nq)
            E.write_panel(picks, blist, meta, tmp / date, date)
            log.info("%s  A %d 只（连续3天以上 %d）  B %d 只  池 %d",
                     date, len(picks), meta["n_streak3"], len(blist),
                     len(pool))
            if a.dry_run:
                continue
            E.send_mail(date, picks, blist, meta, tag="补发")
        (tmp / "meta.json").write_text(
            json.dumps({"files": files}, ensure_ascii=False),
            encoding="utf-8")
    finally:
        D.STATE = real_state
    log.info("产物在 %s（临时目录，不进仓库）", tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
