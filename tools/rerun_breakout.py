"""
用**当前模型**重算最近 N 个交易日的起涨预测清单 A/B，并逐日发信。

和 resend_breakout.py 的区别：resend 只把已落盘的清单重新渲染发一遍；
这里是重新打分。用途是模型或特征改了之后（2026-09-15 修掉三处回测漏洞、
换了筹码网格），把最近几天按新模型重出一遍。

    python tools/rerun_breakout.py --days 7            重算 + 发信 + 落盘
    python tools/rerun_breakout.py --days 7 --dry-run  只重算，不发信不落盘

口径和每日流程一致：≥ SCORE_MIN 分按预测值取前 CAP_A，连续天数按上榜天数，
清单 B 按当天为止的 A 池和价格算（不许看之后的）。两处不一样，邮件里写明：
  1. 风险剔除用的是**今天**的 ST / 减持 / 解禁 / 增发名单（历史那天的拿不到）
  2. 连续天数从重算的第一天起算（之前的清单是旧模型出的，不接着数）

落盘会**覆盖** data/breakout/ 里这几天的清单和 A 池：它们是旧模型出的，
留着会让后面的连续计数接在旧口径上。最后一天的产物写进 out_breakout，
当天的计划任务看到 run_meta 日期就不再跑。
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

import daily as D           # noqa: E402
import export as E          # noqa: E402
import local_run as LR      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rerun")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-build", action="store_true",
                    help="特征表已经是最新的，不重算")
    a = ap.parse_args()

    D.load_env()
    if not LR.acquire_lock("breakout"):
        log.error("起涨预测正在跑，等它结束再来")
        return 1
    try:
        return run(a)
    finally:
        LR.release_lock("breakout")


def run(a) -> int:
    if not a.skip_build:
        log.info("重算特征表（约 20 分钟）")
        if LR.py("src/breakout/build.py") != 0:
            log.error("特征表没建出来")
            return 1

    df = pd.read_parquet(D.DATA / "train.parquet")
    fc = [c for c in df.columns if "__" in c]
    df[fc] = df[fc].astype("float32")
    dates = sorted(df["date"].unique())[-a.days:]
    log.info("重算 %d 天：%s .. %s", len(dates), dates[0], dates[-1])

    obj = D.load_or_fit(df)
    cnt = df.groupby("code")["date"].size()
    px = pd.read_parquet(D.DATA / "daily.parquet", columns=["code", "date", "close"])

    # 风险剔除只查一次（今天的名单），七天共用。历史那天的名单拿不到。
    import datasource as ds

    tmp = Path(tempfile.mkdtemp(prefix="rerun_"))
    real_state = D.STATE
    D.STATE = tmp                                   # A 池在临时目录里重建
    streak_prev: dict[str, tuple[str, int]] = {}    # code -> (上次上榜日, 连续天数)
    bad_cache: dict[str, str] = {}
    outputs = []
    try:
        for i, date in enumerate(dates):
            today = df[df["date"] == date].copy()
            X = np.nan_to_num(today[obj["feats"]].to_numpy(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
            adj = today["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
            today["_p"] = obj["booster"].predict(X) * adj
            today["score"] = D.to_score(today["_p"].to_numpy(), obj["quantiles"])
            today = today.sort_values("_p", ascending=False)
            today = today[today["code"].map(cnt).fillna(0) >= 120]
            cand = today.head(60)["code"].tolist()
            need = [c for c in cand if c not in bad_cache and c not in _checked]
            if need:
                got = D.risk_filter(need)
                bad_cache.update(got)
                _checked.update(need)
            today["reject"] = today["code"].map(bad_cache).fillna("")
            ok = today[(today["reject"] == "") & (today["score"] >= D.SCORE_MIN)]
            picks = ok.head(D.CAP_A).copy()
            # 连续天数：上一个重算日也在榜就 +1（交易日相邻）
            prev_date = dates[i - 1] if i else None
            st = []
            for c in picks["code"]:
                last, k = streak_prev.get(c, ("", 0))
                st.append(k + 1 if last == prev_date else 1)
            picks["streak"] = st
            picks = picks.sort_values(["streak", "_p"], ascending=[False, False])
            for c, k in zip(picks["code"], picks["streak"]):
                streak_prev[c] = (date, int(k))
            try:
                q = ds.fetch_quotes([ds.to_symbol(c) for c in picks["code"]])
                picks["name"] = picks["code"].map(
                    lambda c: getattr(q.get(ds.to_symbol(c)), "name", ""))
            except Exception:  # noqa: BLE001
                picks["name"] = ""
            pool = D.update_pool(picks, date)
            blist = D.build_list_b(px, pool, asof=date)
            n_rej = sum(1 for c in cand if c in bad_cache)
            meta = {"date": date, "n_a": len(picks), "n_b": len(blist),
                    "score_min": D.SCORE_MIN,
                    "n_streak3": int((picks["streak"] >= 3).sum()),
                    "pool": len(pool), "model_date": obj["fit_date"],
                    "rejected": n_rej, "dry": bool(a.dry_run)}
            log.info("%s  A %d 只（连续3天以上 %d）  B %d 只  剔除 %d  池 %d",
                     date, len(picks), meta["n_streak3"], len(blist), n_rej,
                     len(pool))
            for r in picks.itertuples():
                log.info("    %s %-6s %3.0f 分 连续 %d 天 %.2f", r.code, r.name,
                         r.score, r.streak, r.close)
            outputs.append((date, picks, blist, meta))
            E.write_panel(picks, blist, meta, tmp / date, date)
            if not a.dry_run:
                E.send_mail(date, picks, blist, meta, tag="重算")

        if a.dry_run:
            log.info("dry-run：产物在 %s，不落盘", tmp)
            return 0

        # 落盘：覆盖这几天的清单（旧模型出的），A 池换成重算的，最后一天进 out_breakout
        for date, picks, blist, meta in outputs:
            (D.DATA / date[:7]).mkdir(parents=True, exist_ok=True)
            picks.assign(model_date=meta["model_date"],
                         rejected=meta["rejected"]).to_parquet(
                D.DATA / date[:7] / f"breakout_{date}.parquet", index=False)
        shutil.copy2(tmp / "a_pool.json", real_state / "a_pool.json")
        date, picks, blist, meta = outputs[-1]
        D.OUT.mkdir(parents=True, exist_ok=True)
        cols = ["code", "name", "score", "streak", "close", "board"]
        picks[cols].to_json(D.OUT / "list_a.json", orient="records",
                            force_ascii=False, indent=2)
        blist.to_json(D.OUT / "list_b.json", orient="records",
                      force_ascii=False, indent=2)
        (D.OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                             encoding="utf-8")
        # 清单刚落盘，回放历史给面板的日期下拉用
        E.write_panel(picks, blist, meta, D.OUT, date,
                      history=D.recent_history(30))
        LR.push_marker("sent", "breakout", {"ok": True, "rerun": True}, False, date)
        LR.push_all(f"起涨预测 重算 {dates[0]}..{dates[-1]} [local]",
                    [f"data/breakout/{dates[0][:7]}", f"data/breakout/{dates[-1][:7]}",
                     "out_breakout", "state/breakout"], False)
        return 0
    finally:
        D.STATE = real_state


_checked: set[str] = set()


if __name__ == "__main__":
    sys.exit(main())
