"""
竞价主流程。两个阶段由 --stage 区分，同一个 job 内先后执行。

  --stage quick   等待 -> T1/T2/T3/T4 快照 -> 打分排序 -> 写 out/brief.json + prompt.md
  --stage enrich  读 out/commentary.json（Claude 产出）-> 发邮件 + 生成同花顺文件

时间线（北京时间）
  08:47  cron 触发（实际可能 08:47-09:20 之间任意时刻，GH 排队抖动）
  09:14  预热连接
  09:19:40  T1  撤单前虚拟撮合价
  09:23:30  T2
  09:25:10  T3  最终竞价结果
  09:27:20  T4  补采 T3 漏掉的票（价格 9:25-9:30 固定不变，此步只补全）
  09:27:40  Claude 分析（硬超时 150s，失败不阻断）
  09:28:30  发信
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import datetime as dt
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import datasource as ds
from datasource import Quote, limit_pct, limit_price
from score import AuctionFeature, score_one, rank
from ths_export import write_ths_blocks, write_ths_panel
from tdx_export import write_tdx_custom
from mailer import send_report, send_alert

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TZ = dt.timezone(dt.timedelta(hours=8))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("auction")


def now_bj() -> dt.datetime:
    return dt.datetime.now(TZ)


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def target(hms: str) -> dt.datetime:
    h, m, s = (int(x) for x in hms.split(":"))
    return now_bj().replace(hour=h, minute=m, second=s, microsecond=0)


def sleep_until(hms: str, label: str) -> bool:
    tgt = target(hms)
    d = (tgt - now_bj()).total_seconds()
    if d <= 0:
        log.warning("%s 时点已过 %.0fs，立即执行", label, -d)
        return False
    log.info("等待 %s (%s)，%.0fs", label, hms, d)
    while (rem := (tgt - now_bj()).total_seconds()) > 0:
        time.sleep(min(rem, 20))
    return True


# ---------------------------------------------------------------------
def build_features(uni: pd.DataFrame, snaps: dict[str, dict[str, Quote]],
                   c: dict) -> list[AuctionFeature]:
    q1, q2, q3, q4 = (snaps[k] for k in ("T1", "T2", "T3", "T4"))
    feats = []
    for row in uni.itertuples():
        sym = ds.to_symbol(row.code)
        a3 = q3.get(sym) or q4.get(sym)          # T4 补采
        if a3 is None or a3.prev_close <= 0:
            continue
        a1, a2 = q1.get(sym), q2.get(sym)
        lp = limit_pct(row.code, a3.name)
        t3 = a3.chg_pct
        t1 = a1.chg_pct if a1 else t3
        t2 = a2.chg_pct if a2 else t3
        prev_amt = float(row.prev_amount) or 1.0
        up = limit_price(a3.prev_close, lp)

        feats.append(AuctionFeature(
            code=row.code, name=a3.name, limit_pct=lp,
            prev_close=a3.prev_close, auc_price=a3.price,
            gap_pct=round(t3, 2), gap_norm=round(t3 / lp, 3) if lp else 0.0,
            auc_amount=a3.amount_yuan, prev_amount=prev_amt,
            auc_ratio=round(a3.amount_yuan / prev_amt, 5),
            t1_chg=round(t1, 2), t2_chg=round(t2, 2), t3_chg=round(t3, 2),
            slope=round(t3 - t1, 2),
            monotonic=(t1 <= t2 + 0.05 <= t3 + 0.10),
            dive=round(t2 - t3, 2),
            pos_pct_60d=float(row.pos_pct_60d), ma_bull=bool(row.ma_bull),
            breakout=(a3.price > float(row.platform_high)),
            prev_limit_up=bool(row.prev_limit_up),
            prev_broken_board=bool(row.prev_broken_board),
            board_height=int(row.board_height), sector=str(row.sector),
            sector_members=0,
            sector_prev_limitups=int(row.sector_prev_limitups),
            blacklisted=bool(row.blacklisted),
            one_word=(abs(a3.price - up) < 0.005),
        ))

    sc = c["screen"]
    floor = sc["min_auc_amount_wan"] * 1e4
    prelim = [f for f in feats
              if sc["gap_norm_min"] <= f.gap_norm <= sc["gap_norm_max"]
              and sc["auc_ratio_min"] <= f.auc_ratio <= sc["auc_ratio_max"]
              and f.auc_amount >= floor]
    cnt: dict[str, int] = {}
    for f in prelim:
        cnt[f.sector] = cnt.get(f.sector, 0) + 1
    # 「未分类」不是板块，是板块缓存缺失时的占位。不剔掉的话它会变成一个
    # 装着几百只票的巨型「板块」，f_sector 给每只都打满分，白送 15 分权重。
    cnt.pop("未分类", None)
    for f in feats:
        f.sector_members = cnt.get(f.sector, 0)
    log.info("特征 %d 只，初筛通过 %d 只", len(feats), len(prelim))
    return feats


def select(rows: list[dict], c: dict) -> list[dict]:
    o = c["output"]
    res = rank(rows, c)
    if o.get("merge_groups"):
        return sorted(res["all"], key=lambda r: -r["score"])[: o["top_n"]]
    return res["A"] + res["B"]


def write_prompt(sel: list[dict], date: str) -> None:
    brief = [{k: r[k] for k in
              ("code", "name", "gap_pct", "auc_ratio", "slope", "monotonic",
               "sector", "sector_members", "board_height", "prev_limit_up",
               "score", "risk_tags")} for r in sel]
    OUT.mkdir(exist_ok=True)
    (OUT / "brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = (ROOT / "prompts" / "analyst.md").read_text(encoding="utf-8")
    (OUT / "prompt.md").write_text(f"# 日期：{date}\n\n{tpl}", encoding="utf-8")
    (OUT / "commentary.json").write_text("{}", encoding="utf-8")
    # 日期戳。out/ 每天会被提交进仓库，下一次 checkout 就带着上一交易日的
    # selected.json。采集环节但凡提前退出（非交易日/超死线/候选池缺失），
    # enrich 读到的就是旧清单，会把昨天的票当成今天的发出去。
    # 所以发信前必须比对这个戳。
    (OUT / "run_meta.json").write_text(
        json.dumps({"date": date, "n": len(sel)}, ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------------
def stage_quick(c: dict) -> int:
    rt = c["runtime"]
    today = now_bj().strftime("%Y-%m-%d")
    try:
        if today not in ds.trade_dates():
            log.info("%s 非交易日", today); return 0
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历失败(%s)，按工作日兜底", e)
        if now_bj().weekday() >= 5:
            return 0

    if now_bj() > target(rt["hard_deadline"]):
        send_alert(f"{today} 竞价任务启动过晚（{now_bj():%H:%M:%S}），已跳过。\n"
                   f"原因通常是 GitHub Actions 排队延迟，非代码故障。")
        return 0

    p = ROOT / "cache" / "universe.parquet"
    if not p.exists():
        send_alert(f"{today} 候选池缺失，盘前任务可能失败，今日无清单。")
        return 1
    uni = pd.read_parquet(p)
    syms = [ds.to_symbol(x) for x in uni["code"]]
    log.info("候选池 %d 只", len(syms))

    sleep_until("09:14:00", "预热")
    ds.fetch_quotes(syms[:60])

    snaps = {}
    for tag, key in (("T1", "snapshot_t1"), ("T2", "snapshot_t2"),
                     ("T3", "snapshot_t3"), ("T4", "snapshot_t4")):
        sleep_until(rt[key], tag)
        t0 = time.time()
        snaps[tag] = ds.fetch_quotes(syms)
        log.info("%s: %d/%d 只, %.1fs", tag, len(snaps[tag]), len(syms),
                 time.time() - t0)

    got = len(snaps["T3"] | snaps["T4"])
    if got < len(syms) * 0.5:
        send_alert(f"{today} 竞价数据严重缺失（{got}/{len(syms)}），未生成清单。")
        return 1

    feats = build_features(uni, snaps, c)
    rows = [score_one(f, c) for f in feats]

    d = ROOT / "data" / today[:7]
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"auction_{today}.parquet", index=False)

    sel = select(rows, c)
    OUT.mkdir(exist_ok=True)
    (OUT / "selected.json").write_text(
        json.dumps(sel, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "detail.csv", index=False,
                              encoding="utf-8-sig")
    write_prompt(sel, today)
    log.info("入选 %d 只，等待 Claude 分析", len(sel))
    return 0


def stage_enrich(c: dict) -> int:
    today = now_bj().strftime("%Y-%m-%d")
    f = OUT / "selected.json"
    if not f.exists():
        log.info("无 selected.json（今日未运行或非交易日），跳过")
        return 0

    # out/ 是提交进仓库的，checkout 下来就带着上一交易日的产物。
    # 没有今天的日期戳就说明本次采集没跑完，绝不能拿旧清单发信。
    try:
        stamp = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        stamp = {}
    if stamp.get("date") != today:
        log.warning("out/ 里是 %s 的产物（今天 %s），本次不发信",
                    stamp.get("date", "未知"), today)
        return 0
    sel = json.loads(f.read_text(encoding="utf-8"))

    texts, notice = {}, ""
    try:
        texts = json.loads((OUT / "commentary.json").read_text(encoding="utf-8"))
        if not isinstance(texts, dict):
            raise ValueError("格式错误")
    except Exception as e:  # noqa: BLE001
        log.warning("Claude 输出不可用: %s", e)
    if not texts:
        notice = ("本次无 LLM 分析（模型调用失败或额度耗尽），"
                  "以下为纯量化结果")
        log.warning(notice)

    o = c["output"]
    if not sel and not o.get("send_when_empty", True):
        log.info("空榜且配置为不发信"); return 0

    tiers = o["ths_tiers"]
    OUT.mkdir(exist_ok=True)
    blocks = write_ths_blocks(sel, OUT, tiers, today) if sel else []
    write_ths_panel(sel, texts, OUT, tiers, today, notice)
    if sel:
        write_tdx_custom(sel, OUT)          # 可选：给通达信用

    owner = os.environ.get("GH_OWNER", "")
    repo = os.environ.get("GH_REPO", "")
    page = f"https://{owner.lower()}.github.io/{repo}/" if owner else ""

    att = list(blocks)
    if o.get("attach_csv") and (OUT / "detail.csv").exists():
        att.append(OUT / "detail.csv")

    sleep_until(c["runtime"]["send_at"], "发信")
    send_report(today, {"A": sel, "B": []}, texts, c,
                attachments=att, stage="清单", notice=notice, page_url=page)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["quick", "enrich"], required=True)
    a = ap.parse_args()
    c = cfg()
    return stage_quick(c) if a.stage == "quick" else stage_enrich(c)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        log.exception("流程异常")
        try:
            send_alert(f"竞价任务异常终止：{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
