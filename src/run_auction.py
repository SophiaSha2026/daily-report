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
  09:25:40  T4  补采 T3 漏掉的票（价格 9:25-9:30 固定不变，此步只补全）
  09:25:45  Claude 分析（硬超时 2 分钟，失败不阻断）
  09:27:30  发信
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
              if sc["gap_pct_min"] <= f.gap_pct <= sc["gap_pct_max"]
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


def write_prompt(sel: list[dict], date: str, late: bool = False) -> None:
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
        json.dumps({"date": date, "n": len(sel), "late": bool(late)},
                   ensure_ascii=False),
        encoding="utf-8")


# ---------------------------------------------------------------------
def stage_quick(c: dict, late: bool = False) -> int:
    """
    late=True 是**抢救模式**：cron 和本机触发器都没在点上跑，等发现时
    09:25-09:30 那个数据窗口已经过了。

    还能救回来的：竞价成交价 = 今开（Quote.open_）。这个值**精确**——
    开盘价按定义就是集合竞价的撮合价，盘中不会再变，所以 gap_pct 是真值。
    救不干净的：竞价成交额只能用当前累计额，混进了开盘后连续竞价的量，
    **偏大**；跑得越晚污染越重。
    救不回来的：T1/T2/T3 三次快照没采到，斜率、稳步抬升、假涨停撤单、
    尾盘跳水四个维度全部失效，这里置成中性值。

    邮件顶部会写明这是抢救结果，不会假装成一次正常的竞价扫描。
    """
    rt = c["runtime"]
    today = now_bj().strftime("%Y-%m-%d")
    try:
        if today not in ds.trade_dates():
            log.info("%s 非交易日", today); return 0
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历失败(%s)，按工作日兜底", e)
        if now_bj().weekday() >= 5:
            return 0

    if not late and now_bj() > target(rt["hard_deadline"]):
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

    if late:
        # 量能维度在抢救模式下是**不可用**的，不是「偏大一点」而已：
        # 当前累计成交额里混着开盘后的连续竞价，跑得越晚混得越多。
        # 2026-08-28 首次实测，09:51 跑的那次 auc_ratio 全线超出当时 20.8% 的
        # 上限，635 只被量能条件一刀切光，发出去一封空邮件。
        # 2026-09-02 上限收回 4.17%（量比 10）之后只会切得更狠，这段更不能省。
        # 所以这里直接把量能的准入区间放开、权重清零并按比例分给其余维度，
        # 而不是拿一个已知污染的数去做筛选和打分。
        sc = c["screen"]
        sc["auc_ratio_min"], sc["auc_ratio_max"] = 0.0, 1e9
        w = c["scoring"]["weights"]
        vw = w.pop("volume", 0.0)
        rest = sum(w.values())
        if rest > 0:
            for k in w:
                w[k] = w[k] / rest * (rest + vw)
        w["volume"] = 0.0
        log.warning("抢救模式：量能维度不可用（累计额已混入连续竞价），"
                    "准入区间放开、权重 %.2f 已分摊给其余维度", vw)

    snaps = {}
    if late:
        # 只取一次快照，四个 T 全指向它。price 换成 open_（=竞价撮合价），
        # gap_pct 因此是真值。T1=T2=T3 意味着斜率 0、单调成立、跳水 0，
        # 这些维度事实上已失效，置中性比编造一个数诚实。
        t0 = time.time()
        q = ds.fetch_quotes(syms)
        for v in q.values():
            if v.open_ and v.open_ > 0:
                v.price = v.open_
        log.warning("抢救模式：单次快照 %d/%d 只, %.1fs。竞价价用今开（精确），"
                    "量能用当前累计额（偏大），斜率/形态维度失效",
                    len(q), len(syms), time.time() - t0)
        snaps = {k: q for k in ("T1", "T2", "T3", "T4")}
    else:
        sleep_until("09:14:00", "预热")
        ds.fetch_quotes(syms[:60])
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
    write_prompt(sel, today, late=late)
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
    if stamp.get("late"):
        notice = ("⚠ 抢救结果，非正常竞价扫描：cron 与本机触发器均未按时启动，"
                  "09:25-09:30 数据窗口已过。竞价价取今开（精确值），"
                  "量能维度**已停用**（累计额混入连续竞价，无法还原竞价量），"
                  "斜率/稳步抬升/假涨停/尾盘跳水四个维度同样失效。"
                  "本榜实质是按高开幅度+位置+板块共振排序。")
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

    # 影子参考榜：用影子模型（RankHuber 线性系数）给同一个池子打分，取前 10。
    # 用户要求两个榜都给。它是**参考**不是正式榜：正式榜仍由 score.py 排。
    # 整段 fail-open——影子是研究组件，任何失败都不许影响发信（硬约束 2）。
    shadow_rows = []
    try:
        det = OUT / "detail.csv"
        if det.exists() and (ROOT / "state" / "shadow_model.json").exists():
            import pandas as pd
            sys.path.insert(0, str(ROOT / "src"))
            from learn import shadow as _sh
            dfa = pd.read_csv(det, dtype={"code": str})
            dfa["date"] = today
            sc = _sh.score(dfa)
            if sc is not None:
                dfa = dfa.assign(ss=sc)
                ok = dfa[dfa["rejected"].isna() | (dfa["rejected"] == "")]
                top = ok.nlargest(10, "ss")
                shadow_rows = [
                    {"code": r.code, "name": r.name,
                     "gap_pct": float(r.gap_pct),
                     "liangbi": float(getattr(r, "liangbi", 0) or 0),
                     "sscore": round(float(r.ss), 3)}
                    for r in top.itertuples(index=False)]
                base_top = {x["code"] for x in sel[:10]}
                ov = len(base_top & {x["code"] for x in shadow_rows})
                log.info("影子参考榜 %d 只，与正式榜重合 %d/10",
                         len(shadow_rows), ov)
                # 落盘给 TUI 状态页和学习面板读。带日期，读的一方要核对。
                (OUT / "shadow.json").write_text(json.dumps({
                    "date": today, "rows": shadow_rows, "overlap": ov,
                    "base_top": [x["code"] for x in sel[:10]],
                }, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("影子参考榜生成失败（不影响发信）: %s", e)

    tiers = o["ths_tiers"]
    OUT.mkdir(exist_ok=True)
    blocks = write_ths_blocks(sel, OUT, tiers, today) if sel else []
    write_ths_panel(sel, texts, OUT, tiers, today, notice, c["screen"],
                    shadow_rows)
    if sel:
        write_tdx_custom(sel, OUT)          # 可选：给通达信用

    owner = os.environ.get("GH_OWNER", "")
    repo = os.environ.get("GH_REPO", "")
    page = f"https://{owner.lower()}.github.io/{repo}/" if owner else ""

    att = list(blocks)
    if o.get("attach_csv") and (OUT / "detail.csv").exists():
        att.append(OUT / "detail.csv")

    # 双阈值：软时点到了就发；Claude 拖过软时点也照发，但越过硬上限要留痕。
    rt = c["runtime"]
    soft = rt["send_at"]
    hard = rt.get("send_deadline", soft)
    if not sleep_until(soft, "发信"):
        late = (now_bj() - target(soft)).total_seconds()
        if now_bj() > target(hard):
            msg = f"本次发信晚于硬上限 {hard}（迟 {late:.0f} 秒）"
            log.error(msg)
            notice = f"{notice} · {msg}" if notice else msg
        else:
            log.warning("晚于软时点 %s %.0f 秒，仍在硬上限 %s 之内",
                        soft, late, hard)
    if os.environ.get("SKIP_MAIL"):
        # 两个来源：本地 dry-run；远端 yield_check 确认本地已发信。
        # 面板、txt、csv 全部照常生成，只有邮件不发。
        log.info("SKIP_MAIL=1：面板已生成，邮件不发（本地已接管或 dry-run）")
        return 0
    send_report(today, {"A": sel, "B": []}, texts, c,
                attachments=att, stage="清单", notice=notice, page_url=page,
                shadow_rows=shadow_rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["quick", "enrich"], required=True)
    ap.add_argument("--late", action="store_true",
                    help="抢救模式：窗口已过，用今开当竞价价补出一份清单")
    a = ap.parse_args()
    c = cfg()
    return stage_quick(c, late=a.late) if a.stage == "quick" else stage_enrich(c)


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
