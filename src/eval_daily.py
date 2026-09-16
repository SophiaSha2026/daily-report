"""
自评估 · 学习 · 迭代  的命令行入口。

    python src/eval_daily.py --stage intraday  盘中采一个时点（卖点研究用）
    python src/eval_daily.py --stage label     收盘后抓标签
    python src/eval_daily.py --stage brief     给 LLM 归因准备输入
    python src/eval_daily.py --stage llm       本地跑 LLM 归因（走本机 claude CLI）
    python src/eval_daily.py --stage learn     拟合 + 六道闸 + 落参数 + 发信
    python src/eval_daily.py --stage race      模型擂台（选最合适的传统 ML）
    python src/eval_daily.py --stage rollback  删掉学到的参数，回人工基线
    python src/eval_daily.py --stage status    打印当前状态

不带 `--date` 时缺省是**最近一个已收盘交易日**（北京 15:05 之后才算当天），
和 local_run 判「今天跑完了」用的是同一个口径；只有 intraday / backfill
两个阶段的缺省是自然日今天。

完整设计见 docs/learning.md。三条不可越过的边界：
  1. score.py 里永远不出现模型调用
  2. 这条线崩了不能影响早盘发信（独立 workflow + 原子写）
  3. 准入区间不自动改，只提案
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import cfg as C
import localenv
from learn import (apply as A, dataset, gate, labels as L, objective as O,
                   optimize as OPT, report as R, sources)

# SMTP 凭证和 OAuth token 在 tools/local.env 里。控制台的按钮直接跑这个脚本
# （不经过 local_run），不加载就发不出信，而发信失败是 fail-open 的（教训 16）。
localenv.load()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_learn"
STATE = ROOT / "state"


def now_bj() -> dt.datetime:
    """北京时间的当前时刻。单独一层是为了让自测能替换它（审计 F3-15）。"""
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)


def today_bj() -> str:
    return now_bj().strftime("%Y-%m-%d")


def last_closed_bj(now: dt.datetime | None = None) -> str:
    """最近一个已收盘交易日（北京 15:05 之后算当天）。

    label / brief / llm / learn / all / council 的缺省日必须是这个，不是自然日：
    计划任务判「今天跑完了」用的是 local_run.target_date("learn")，也就是
    last_closed_trade_day。午夜后手敲一次不带 --date 的学习线会把**明天**
    写进 learning_status.json 和 verdict_log，当天 16:40 起的 16 小时窗口
    因日期匹配整天跳过，那个交易日的在线真值标签再也抓不回来
    （from_quotes 只能抓当天实时价）。所以缺省值和调度器共用同一个实现，
    不另写一份（审计 F3-10）。
    """
    from local_run import last_closed_trade_day
    return last_closed_trade_day(now)


def _reject_bad_date(date: str, what: str) -> bool:
    """未收盘 / 非交易日就别往下走。True = 应当拒绝（审计 F3-10）。"""
    lim = last_closed_bj()
    if date > lim:
        log.error("%s 还没收盘（最近已收盘交易日 %s），%s", date, lim, what)
        return True
    try:
        from local_run import trade_dates
        tds = trade_dates()
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历拿不到（%s），跳过交易日核对", e)
        tds = set()
    if tds and date not in tds:
        log.error("%s 不是交易日，%s", date, what)
        return True
    return False


def _cost(c: dict) -> float:
    """汇报口径的双边成本。`learning.label.cost_bp` 是单边（万分之 13），
    一买一卖各一次所以乘 2。只在汇报里扣，不进优化目标——它是常数，
    不改排序（config.yaml 那句注释在 2026-09-16 之前一直没有代码兑现）。
    """
    return 2.0 * float(((c.get("learning") or {}).get("label") or {})
                       .get("cost_bp", 0.0)) / 1e4


# ---------------------------------------------------------------------
#  stage: label
# ---------------------------------------------------------------------
def stage_label(c: dict, date: str, backfill: bool,
                force: bool = False) -> int:
    """收盘后抓收盘价，和当天的竞价快照拼出标签。

    退出码：0 写了（或那天根本没有快照，无事可做）、1 失败、
    **2 没覆盖**（已有一份可用行更多的标签，见 labels.save 的守卫）。
    2 和 0 必须分开：口径收紧后重打一遍，恰好是真正改动到的那几天会被守卫
    拦下，只留一行 warning、退出码还是 0 —— 等于没修而且看不出来（教训 27：
    退出码 0 不等于做了事）。要强行覆盖加 --force。
    """
    # 先判时刻**再联网**：labels.from_quotes 的过滤只看时间戳是不是今天，
    # 盘中现价一样以今天开头，会被当成收盘价写进标签；而 labels.save 的
    # 「可用行更多才覆盖」还可能让收盘后正确的那份被拒，错标签永久留下
    # （审计 F3-15）。快照存在性检查在后面，它拦不住这件事：快照 09:25:45
    # 就落盘了。
    now = now_bj()
    today = now.strftime("%Y-%m-%d")
    if date > today or (date == today and (now.hour, now.minute) < (15, 5)):
        log.error("%s 还没收盘（北京 %s），盘中现价不是收盘价，不抓标签",
                  date, now.strftime("%H:%M"))
        return 1
    snap_p = ROOT / "data" / date[:7] / f"auction_{date}.parquet"
    if not snap_p.exists():
        log.warning("%s 没有竞价快照，跳过", date)
        return 0
    snap = pd.read_parquet(snap_p)

    if backfill:
        hp = ROOT / "cache" / "hist_daily.parquet"
        if not hp.exists():
            log.error("没有 cache/hist_daily.parquet，先跑 --stage backfill")
            return 1
        raw = L.from_hist(date, pd.read_parquet(hp))
    else:
        raw = L.from_quotes(list(snap["code"]), date)

    if raw.empty:
        log.error("%s 取不到开收盘", date)
        return 1
    # force 给「口径改了要重打一遍」用：save 的守卫是「可用行更少就不覆盖」，
    # 收紧口径时它恰好拦住真正改动到的那几天（见 labels.save 的注释）
    _p, wrote = L.save(date, L.build(
        date, snap, raw, c["learning"]["label"]["max_open_mismatch_pct"]),
        force=force)
    return 0 if wrote else 2


# ---------------------------------------------------------------------
#  stage: brief（给 LLM 的输入）
# ---------------------------------------------------------------------
def stage_brief(c: dict, date: str) -> int:
    lc = c["learning"]
    df = dataset.build([date], lc["neutralize"])
    if df.empty:
        log.warning("%s 没有可用数据", date)
        return 0
    from learn import vscore
    s, rej = vscore.score_df(df, c)
    d = df.assign(sc=s, rej=rej)
    ok = d[~d["rej"]].sort_values("sc", ascending=False)
    if ok.empty:
        log.warning("%s 无票通过硬性排除", date)
        return 0

    n_w, n_b = lc["llm"]["n_worst"], lc["llm"]["n_best"]
    # worst/best 的挑法只有一份实现（learn/brief.py）：两组互斥、符号正确、
    # rank 是当日分数名次。以前这里是 head(20)/tail(50%) 各自取极值，池子
    # 不足 40 只时两组必然相交，17 个在线日有 8 天中招（审计 F3-4）。
    from learn import brief as BR
    ok, worst, best = BR.pick_worst_best(ok, n_w, n_b)
    # 「前 10」必须是当天真发出去的那张清单（生产口径：>=45 分再取前 10）。
    # 以前按过准入全池的前 10 算，09-16 那天 brief 里写「池 12、前 10 命中
    # 30%」，实际邮件只发了 6 只（审计 F3-3）。这份 JSON 是喂给 LLM 归因的。
    sent = ok.iloc[OPT.production_order(ok["sc"].to_numpy(float),
                                        np.zeros(len(ok), bool), c)]
    # 汇报口径：未缩尾的 y_raw 再扣双边成本（审计 F2-10）
    from learn import online_eval as OE
    cost = _cost(c)
    sent_ex = OE.excess(sent, cost)

    def pack(g):
        return [{
            "code": r.code, "name": r.name, "score": round(r.sc, 1),
            "rank": int(r.rank), "gap_pct": round(r.gap_pct, 2),
            "intraday_pct": round(r.r * 100, 2), "ytil": round(r.ytil, 2),
            "sector": r.sector, "risk_tags": list(r.risk_tags),
        } for r in g.itertuples(index=False)]

    brief = {
        "date": date,
        "day_stats": {
            "pool": int(len(ok)),
            "sent_n": int(len(sent)),
            "market_median_intraday_pct": round(d["day_center"].iloc[0] * 100, 2),
            "top10_excess_pct": (round(float(sent_ex.mean()) * 100, 2)
                                 if len(sent) else None),
            "top10_hit": (round(float((sent_ex > 0).mean()), 2)
                          if len(sent) else None),
        },
        "worst": pack(worst),
        "best": pack(best),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "eval_brief.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("brief 已写出：worst %d / best %d", len(brief["worst"]),
             len(brief["best"]))
    return 0


def load_day_weights(c: dict) -> dict[str, float]:
    """LLM 归因 -> 优化器里的日权重。读不到就全 1.0。

    映射表在 config 里，所以给定那些 JSON，优化器完全确定——
    LLM 的影响是可复现的。
    """
    m = c["learning"]["llm"]["regime_weight"]
    out: dict[str, float] = {}
    for p in sorted((STATE / "llm_eval").glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            out[p.stem] = float(m.get(j.get("day_regime", "正常"), 1.0))
        except Exception as e:  # noqa: BLE001
            log.warning("归因 %s 读取失败: %s", p.name, e)
    return out


# ---------------------------------------------------------------------
#  stage: learn
# ---------------------------------------------------------------------
def _learn_box(c: dict) -> dict[str, list]:
    """交给优化器和闸门之前，先把箱裁到这份训练数据**真能学**的维度。

    训练表的绝大多数天是回填（404 天回填 vs 17 天在线），而回填的
    t1/t2/slope/dive/monotonic 是代理值、竞价额含开盘后成交（learn/backfill.py
    已知偏差 5、learn/sources.py 模块注释）：trend / volume 两维在那上面是
    常数。不裁的话优化器照样在这两个维度上调参数，把权重推到边界去给别的
    维度腾预算——那不是学到的结论，是数据缺失，而且它一路过闸就会落地。
    只打日志不裁是没用的（restrict_box 的注释）。

    裁法是钉死 lo=hi=θ⁰ 而不是删键：删键会让剩下四个权重被 O.project 归一到
    1，加上仍在生效的 trend/volume 两个 0.20，总权重变成 1.4。
    可学维度以 config.learning.backfill.learnable_dims 为唯一真相
    （sources.dims_from_cfg），探测只当兜底——probe_ts 要联网。
    """
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    out = sources.restrict_box(box, t0, sources.dims_from_cfg(c))
    # 被钉死的维度上如果 learned.yaml 已经学到了别的值，θ_prev 就落在箱外：
    # 锚定项会被 σ=1e-12 放大，闸门 4 的步长分母还是 0（gate.evaluate 的
    # sig = hi − lo）。这种状态只能人来解，先 --stage rollback 回基线。
    now = C.theta_now(box)
    bad = [k for k, (lo, hi) in out.items()
           if hi <= lo and abs(now[k] - t0[k]) > 1e-9]
    if bad:
        log.warning("这些维度已转为不可学，但当前生效值不是人工基线：%s。"
                    "先 --stage rollback 回基线，否则闸门算不出步长", bad)
    return out


def _bf_dirty(raw: pd.DataFrame, c: dict) -> pd.Series:
    """回填表的「脏样本」口径，必须和在线 labels.build 是同一条（教训 30）。

    在线那边（learn/labels.py:89-97）判两件事：一字板买不进，以及日线开盘价
    与快照 auc_price 偏离超过 `learning.label.max_open_mismatch_pct`（0.5%）——
    买入价和特征描述的价不是同一个价，那一行的标签就是错的。
    回填表以前只判一字板。2026-09-16 审计实测：40.76 万行里 |open-auc_price|
    /auc_price > 0.5% 的有 46190 行（11.53%），而准入区间（涨幅 2~5%、
    量比 2.5~10）内占 21.4%，正是软 TopK 看的那一批；这批行的标签被机械带偏
    约 1pp（open 比 auc_price 低 0.5% 以上的 2888 行 r 均值 +1.54%，两者一致的
    12594 行 +0.20%），而中性化后的真实信号只有 0.2pp 量级。
    同一条规则在线 18 个标签文件里只打掉 18 行（0.08%），差异出在回填源
    （Tushare stk_auction_o）而不是规则本身。

    脏行是真生效的：dataset.neutralize 把它们排除在当日中位数/缩尾分位/MAD
    之外，所以受影响的不只是被污染的那几行，而是每一天的 ỹ 尺度。
    """
    thr = float(c["learning"]["label"]["max_open_mismatch_pct"])
    d = raw["one_word"].astype(bool)
    if "open_mismatch_pct" in raw.columns:
        d = d | (pd.to_numeric(raw["open_mismatch_pct"], errors="coerce")
                 > thr).fillna(False)
    else:
        # 旧 parquet 没这一列，不能 KeyError 崩掉，但也不许静默退回旧口径
        log.warning("回填表没有 open_mismatch_pct 列（旧表），可用样本口径"
                    "与在线不一致；重建训练表：--stage build-train")
    return d


def _load_train(c: dict):
    """训练表 = 回填历史 + 在线真值，返回 (df, source)。

    2026-09-03 用户决定不等 60 天在线积累，直接用回填历史点火学习。
    2026-09-15 起在线真值天**并进训练表**（用户决定）：回填表只是先验，
    每天真采到的快照 + 真值标签才是这套系统真正要学的东西。以前在线天
    只做第七道闸，训练表冻结在回填截止日，每天在同一份数据上重复拟合，
    七次裁决的数字逐位相同。

    合并规则：同一天两边都有的（回填截止前的那几个在线日）以在线为准，
    它的竞价轨迹是真采样，回填的是代理值。`_src` 列标来源，第七道闸和
    面板只看在线那部分。在线天在时间轴最末端，走向前切分时天然落在
    样本外那一段，等攒过 oos_frac 的比例才逐步进入拟合。
    """
    lc = c["learning"]
    nz = lc["neutralize"]
    parts, srcs = [], []
    bf = sorted((ROOT / "data" / "train").glob("backfill_*.parquet"))
    if bf:
        raw = pd.read_parquet(bf[-1])
        raw["dirty"] = _bf_dirty(raw, c)
        # 抢救日守卫只认在线快照。回填表的 t1/t2/t3 是代理值，单价竞价
        # 天然三者相等，开着守卫会误伤（2026-09-04 复查丢了 3 天）。
        d_bf = dataset.neutralize(raw, nz, salvage_guard=False)
        parts.append(d_bf.assign(_src="backfill"))
        srcs.append(f"backfill:{bf[-1].name}")
    d_on = dataset.build(None, nz)
    if not d_on.empty:
        on_days = set(d_on["date"].unique())
        parts = [x[~x["date"].isin(on_days)] for x in parts]
        parts.append(d_on.assign(_src="online"))
        srcs.append(f"online:{len(on_days)}天")
    if not parts:
        return pd.DataFrame(), "none"
    df = pd.concat(parts, ignore_index=True).sort_values(
        "date", kind="mergesort").reset_index(drop=True)
    return df, "+".join(srcs)


def _regime_of(date: str) -> str | None:
    """当天的 LLM 归因结论（day_regime），没有就 None。"""
    p = STATE / "llm_eval" / f"{date}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("day_regime")
    except Exception:  # noqa: BLE001
        return None


def _online_oos(dfo_all: pd.DataFrame, te_days) -> pd.DataFrame:
    """闸门 7 只许用**样本外段**里的在线日（进了拟合段的是样本内）。

    2026-09-15 起在线天并进训练表，走向前的训练段 tr 会一天天把最早的在线日
    吃进去（split_days 是纯尾部切分）。dfo 照旧取全部在线天的话，那些天的
    g_new−g_old 系统性偏正——theta_new 就是在这些天上最大化 G 得到的——
    P(新参数更好) 被抬高，0.25 的否决线名存实亡。今天 17 个在线日全在 te，
    这个过滤一行都不改；按每交易日各 +1 推演，约 83 个交易日后首个在线日
    进 tr，渐近样本内占比趋向 1−oos_frac=80%（审计 F3-6）。
    影子线早就为同一件事加了 train_end 过滤（learn/shadow.py），闸门 7 缺这层。
    """
    return dfo_all[dfo_all["date"].isin(set(te_days))]


def _online_days(p_on) -> int:
    """第七道闸的天数 = 自助**真正用到**的天数（日权重 > 0）。

    bootstrap_better 按 `keep = day_w > 0` 过滤，归因为「数据异常」的天权重
    是 0，已经被剔出统计；而 online_days 以前数的是 dfo 里所有天。
    6 个在线日里 3 天异常时 online_days=6 过了 min_days=5，P 却只由 3 天
    自助得出（离线复现 P=0.0，那 3 天真参与的话是 0.658），过了前六道闸的
    候选被 3 天的证据打回，Check.detail 还写着「真值快照 6 天」（审计 F2-8）。
    """
    return int((np.asarray(p_on.day_w, float) > 0).sum())


def _judge(c: dict, df, mk, p_te, theta_new: dict, theta_prev: dict,
           days: list[str], date: str, *, intents: list[str] | None = None):
    """算一遍闸门要的全部统计量并裁决。stage_learn 和会诊的参数提案共用这一段，
    两边评的是同一套闸门（教训 11：统计量一律关键字传入）。

    返回 (Verdict, {"online_days", "online_p", "dfo", "dfo_gate", "paired",
    "churn", "boot_p"})。
    """
    lc = c["learning"]
    box, g = _learn_box(c), lc["gate"]
    theta0 = C.theta0(box)
    dayw = load_day_weights(c)
    n = len(days)
    oos_new, gd_new = p_te.G(theta_new)
    oos_old, gd_old = p_te.G(theta_prev)
    bp = OPT.bootstrap_better(p_te, theta_new, theta_prev, g["bootstrap_n"])
    # 闸门 3 自助的是逐日配对差的 Huber 位置；把它的点估计也报出来，
    # 免得「P 高但 ΔĜ 为负」看起来像矛盾（两者量的不是同一件事）。
    # 必须走 OPT.paired_delta：它和 bootstrap_better 共用 _paired_diff，
    # 同一组天、同一组权重。这里以前手写 huber_location(..., None, ...)，
    # 等权且含 w=0 的天，实测能出「P=99.95% 而 ΔĜ=−0.0003」（审计 F1-8）。
    paired = OPT.paired_delta(p_te, theta_new, theta_prev)
    # 闸门 2 的块一致性：样本外段切成连续块各自比一次（审计 F1-6）。
    # p_te.dates 就是切分出来的 te，不另传一份免得两边漂。
    blocks = OPT.oos_blocks(p_te.dates, int(g.get("oos_blocks", 3)))
    block_delta = OPT.block_deltas(p_te, gd_new, gd_old, blocks)

    look = days[-g["churn_lookback"]:]
    p_look = mk(look)
    codes = df[df["date"].isin(look)].sort_values(
        "date", kind="mergesort")["code"].to_numpy()
    churn = gate.churn_by_day(p_look.top_codes(theta_prev, codes),
                              p_look.top_codes(theta_new, codes))

    # 第七道闸：在线真值快照上的稳健性。训练主体是回填（轨迹为代理值），
    # 这里用真采样的那几天做否决检验。抢救日已被 dataset 守卫剔除。
    # 只取样本外段里的在线日：在线天自 2026-09-15 起并进训练表，落进拟合段
    # 的那些天是样本内，拿它们做否决检验是自证（见 _online_oos）。
    online_p, online_days = None, 0
    dfo_all = (df[df["_src"] == "online"] if "_src" in df.columns
               else df.iloc[0:0])
    dfo = _online_oos(dfo_all, p_te.dates)
    if not dfo.empty:
        p_on = OPT.Problem(dfo, c, box, theta0, theta_prev, dayw,
                           lc["objective"]["top_k"],
                           lc["objective"]["huber_c"],
                           lc["objective"]["tau_perplexity_tol"])
        online_days = _online_days(p_on)
        online_p = OPT.bootstrap_better(p_on, theta_new, theta_prev,
                                        g["bootstrap_n"])
        log.info("在线稳健性：%d 天权重>0（样本外段 %d 天 / 在线共 %d 天），"
                 "P(新参数更好)=%.2f", online_days, dfo["date"].nunique(),
                 dfo_all["date"].nunique(), online_p)

    # 统计量一律关键字传入（闸门 3 曾因位置错位拿到阈值本身，见 gate.evaluate）
    v = gate.evaluate(theta_new, theta_prev, box, g, n, days, date,
                      boot_p=bp, oos_new=oos_new, oos_old=oos_old,
                      churn=churn, online_p=online_p, online_days=online_days,
                      intents=list(intents or []), paired_delta=paired,
                      block_delta=block_delta)
    # dfo = 全部在线天（面板逐日指标、影子对比、status["online_days"] 看它）；
    # dfo_gate = 闸门 7 实际用的那一段，两者不许混（审计 F3-6）
    return v, {"online_days": online_days, "online_p": online_p, "dfo": dfo_all,
               "dfo_gate": dfo, "paired": paired, "churn": churn, "boot_p": bp,
               "block_delta": block_delta, "p_look": p_look, "codes": codes}


def evaluate_candidate(c: dict, theta_new: dict, date: str) -> dict | None:
    """给学习会诊用：一组参数值走和优化器候选同一套七道闸，**不写盘不记裁决**。

    返回 {"verdict": dict, "metrics": {"prev":..., "new":...}}；数据不够返回 None。
    """
    lc = c["learning"]
    box, g = _learn_box(c), lc["gate"]
    df, _source = _load_train(c)
    if df.empty:
        return None
    days = sorted(df["date"].unique())
    if len(days) < g["min_days"]:
        return None
    theta0, theta_prev = C.theta0(box), C.theta_now(box)
    dayw = load_day_weights(c)
    raw_theta = {k: float(theta_new.get(k, theta_prev[k])) for k in box}
    # 会诊候选和优化器候选必须走**同一条**稀疏化+投影路（审计 F1-10/F3-5）：
    # 只 clip 进箱的话，提案改一个权重就让 Σw≠1，而闸门量的全是排序量，
    # 对整体缩放完全失明；落地后 score.py 的 raw=100·Σw·v 满分变 103，
    # min_score=45 这条绝对分数线被静默放松（18 天真实快照 >=45 的行 280→299）。
    # 意图计数也要同一口径：动一个权重必然带出其余权重的等比再归一，
    # 那是「和为 1」的结果不是决定，不占 max_moves 的名额。
    # 这里刻意放开 sparsify 的两个上限（max_moves=箱维数、步长不截、
    # 噪声门 ~0）：会诊明确给的值要原样送进闸门去判，被 sparsify 截一刀
    # 等于闸门 4 评的不是提案本身。min_frac 用一个极小正数而不是 0，
    # 否则 |Δ|=0 的参数也会被算成意图。
    theta_new, intents = OPT.sparsify(raw_theta, theta_prev, box,
                                      max_moves=len(box), max_step_frac=1.0,
                                      min_frac=1e-12)

    def mk(sub_days):
        sub = df[df["date"].isin(sub_days)]
        return OPT.Problem(sub, c, box, theta0, theta_prev, dayw,
                           lc["objective"]["top_k"], lc["objective"]["huber_c"],
                           lc["objective"]["tau_perplexity_tol"])

    tr, te = OPT.split_days(days, g["oos_frac"])
    p_te = mk(te)
    v, _ = _judge(c, df, mk, p_te, theta_new, theta_prev, days, date,
                  intents=intents)
    full = mk(days)
    return {"verdict": v.to_dict(),
            "metrics": {"prev": full.metrics(theta_prev, lc["objective"]["top_k"]),
                        "new": full.metrics(theta_new, lc["objective"]["top_k"])},
            "theta": theta_new, "n_days": len(days)}


def _hold_change(date: str, v, status: dict, theta_new: dict, m_new: dict,
                 review: dict, source: str, n: int, *,
                 metrics_prev: dict | None = None,
                 old_top: dict | None = None,
                 new_top: dict | None = None) -> None:
    """Opus 审稿否决：不写参数，把提案和证据留在 held 文件里，人工确认后落地。

    裁决必须**改写成「未接受」**。verdict_log.jsonl、learning_status.json、
    learn.html 的第 ⑤ 格和时间线、竞价面板底部那行状态、会诊喂给 LLM 的
    裁决记录，全都只看 accepted 这一个字段；搁置时它们会一致地显示
    「参数已变更」，而 learned.yaml 根本没写、theta_history 没记、变更邮件
    没发。不跑 apply-held 的话 verdict_log 里那行 accepted=True 永远不会
    被纠正，「共 N 次裁决、接受 M 次」的 M 从此和 theta_history 对不上
    （审计 F3-9）。

    held 文件里存的是**改写前**的裁决：apply-held 事后落地时要按七道闸原样
    记进 theta_history，不能带着「Opus 审稿 不过」那条 check 进去。

    指标对比和两张前 10 清单也一起存下来：apply-held 事后要渲染一封和闸门
    直接接受时**一样完整**的变更邮件（动了什么 / 证据 / 行为影响 / 怎么回滚），
    而那时 Problem 早就没了，重算要几十秒还得重读训练表（审计 F3-14）。
    """
    held_verdict = v.to_dict()
    (STATE / "held_change.json").write_text(json.dumps({
        "date": date, "theta": theta_new, "verdict": held_verdict,
        "metrics": m_new, "review": review,
        "metrics_prev": metrics_prev or {},
        "old_top": old_top or {}, "new_top": new_top or {},
        "regime_counts": _regime_counts()},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    v.checks.append(gate.Check("Opus 审稿", False,
                               "反对：" + "；".join(review.get("points", []))))
    v.accepted = False
    status["verdict"] = v.to_dict()
    status["held"] = True
    R.save_status(status)
    gate.log_verdict(date, v, {"source": source, "n_days": n, "held": True})
    try:
        from learn import panel as LP
        LP.build()   # 上一次 build 是按 accepted=True 画的，必须重画
    except Exception as e:  # noqa: BLE001
        log.warning("学习面板重建失败（不影响流程）: %s", e)
    log.warning("变更被 Opus 审稿搁置：%s。人工确认：--stage apply-held",
                "；".join(review.get("points", [])))
    try:
        import mailer
        mailer.send_alert(
            "[学习] " + date + " 参数变更过了七道闸但被 Opus 审稿搁置。"
            + " 理由：" + "；".join(review.get("points", []))
            + " 认可就跑：python src/eval_daily.py --stage apply-held")
    except Exception as e:  # noqa: BLE001
        log.warning("搁置通知发送失败: %s", e)


def _status_extras(c: dict, status: dict, df, dfo, date: str,
                   dry: bool) -> None:
    """在线真值天的逐日指标 + 影子双榜对比 + 转正证据，喂给学习面板。

    全段 fail-open：面板是给人看的，不许拖垮裁决本身。但**失败必须往 status
    里写一个能被界面查询的键**（教训 16）。以前这两个 except 只有一行
    log.warning：vscore.score_df 挂了的话，learn.html 停在上一版还被推上
    Pages、面板徽章照标「最新」、GUI 的参数自学照标「完成」、GUI 的
    「真实榜单」那一行因为 days=0 整行消失、竞价面板底部写「在线真值 0 天」
    ——三条失败路径在界面上一处都看不出来（审计 F3-7）。
    """
    try:
        from learn import online_eval as OE, panel as LP
        lc = c["learning"]
        cost = _cost(c)
        top_k = lc["objective"]["top_k"]
        daily, daily_now = [], []
        if not dfo.empty:
            # daily = 当天**真发出去**的那张榜，只读快照自带的 score/
            # score_raw/rejected，后续改参数改规则都改不动它；
            # daily_now = 按当前参数回放。以前只有回放一个数，界面上却叫
            # 「实盘口径」，17 天里 12 天前 10 不是同一批票（审计 F3-13）。
            daily = OE.daily_sent(dfo, top_k,
                                  float(c["output"]["min_score"]), cost)
            daily_now = OE.daily_replay(dfo, c, top_k, cost)
        status["daily"] = daily
        status["daily_now"] = daily_now
        status["online_days"] = len(daily)
        # 影子排序器：在线双榜对比 + 记账 + refit + 转正证据。
        # 研究性组件，失败不影响任何东西。
        try:
            from learn import shadow, vscore
            fresh = []
            if not dfo.empty:
                s2, rej2 = vscore.score_df(dfo, c)
                # 顺序是硬要求：先用**上一次落盘的**模型给新到的在线日记账，
                # 再 refit。反过来 train_end 就等于今天，daily_compare 的
                # `date > train_end` 过滤恒为空（2026-09-15 起两次学习都是
                # 0 天，面板另一处却写 17/30）。record 按日期先到先得，
                # 同日重跑幂等（审计 G1/F2-2/F3-1）。
                fresh = shadow.daily_compare(
                    dfo, s2, rej2, top_k,
                    min_score=float(c["output"]["min_score"]), cost=cost)
            cmp_ = shadow.record(fresh)
            shadow.fit(df)
            status["shadow"] = cmp_
            scfg = lc.get("shadow") or {}
            stat = shadow.promotion_stat(
                cmp_, int(scfg.get("min_days", 30)),
                float(scfg.get("p_better", 0.90)),
                int(lc["gate"]["bootstrap_n"]))
            status["shadow_stat"] = stat
            if cmp_:
                b = [x["base_top_excess"] for x in cmp_
                     if x.get("base_top_excess") is not None]
                sh = [x["shadow_top_excess"] for x in cmp_
                      if x.get("shadow_top_excess") is not None]
                log.info("影子对比（%d 个真值日）：前10超额 基线 %+.3f%% "
                         "vs 影子 %+.3f%%，日均重合 %.0f%%",
                         len(cmp_),
                         (sum(b) / len(b) * 100) if b else float("nan"),
                         (sum(sh) / len(sh) * 100) if sh else float("nan"),
                         float(np.mean([x["overlap"] for x in cmp_])) * 100)
                log.info("转正证据：%d/%d 天，P(影子更好)=%.2f，%s",
                         stat["days"], stat["min_days"],
                         stat["p_better"] or 0.0,
                         "达标" if stat["ready"] else "未达标")
            if not dry:
                if shadow.maybe_propose(date, stat, cmp_, c,
                                        int(scfg.get("remind_days", 10))):
                    log.info("影子转正提案已发出（切换与否由用户决定）")
                elif (shadow.load_proposal() or {}).get("send_failed"):
                    log.warning("影子转正提案邮件没发出去，见 "
                                "state/shadow_proposal.json")
        except Exception as e:  # noqa: BLE001
            status["shadow_error"] = f"{type(e).__name__}: {e}"[:200]
            log.warning("影子对比失败（不影响流程）: %s", e)
        R.save_status(status)
        LP.build()
    except Exception as e:  # noqa: BLE001
        status["panel_error"] = f"{type(e).__name__}: {e}"[:200]
        log.warning("学习面板生成失败（不影响流程）: %s", e)


def stage_learn(c: dict, date: str, dry: bool) -> int:
    # 日期闸在最前面：拿「明天」跑出来的裁决和今天一模一样，只是日期盖错，
    # 而 learning_status.json 一旦写成明天，当天的计划任务就整天跳过
    # （审计 F3-10）。退 2 表示「什么都没做」，和 0（做完了）区分开。
    if _reject_bad_date(date, "不拟合、不写状态"):
        return 2
    lc = c["learning"]
    box, g = _learn_box(c), lc["gate"]
    df, source = _load_train(c)
    if df.empty:
        log.warning("还没有任何带标签的数据")
        return 0
    days = sorted(df["date"].unique())
    n = len(days)
    theta0, theta_prev = C.theta0(box), C.theta_now(box)
    dayw = load_day_weights(c)
    log.info("训练源 %s：%d 天", source, n)

    def mk(sub_days):
        sub = df[df["date"].isin(sub_days)]
        return OPT.Problem(sub, c, box, theta0, theta_prev, dayw,
                           lc["objective"]["top_k"], lc["objective"]["huber_c"],
                           lc["objective"]["tau_perplexity_tol"])

    full = mk(days)
    m_prev = full.metrics(theta_prev, lc["objective"]["top_k"])
    log.info("当前参数：%d 天，IC %.4f，ICIR %.3f，前10超额 %+.3f%%",
             n, m_prev["ic_mean"], m_prev["icir"], m_prev["top_excess"] * 100)

    # dry 必须写进状态文件：local_run.already_done 先看 dry 再比日期，
    # 没有这个键的试跑会被计划任务当成「今天跑完了」整天跳过，当天的裁决、
    # 变更、会诊、影子提案全丢，次日目标日前移后永久补不回来（教训 27）。
    # 三条线口径一致：run_auction.py 和 breakout/daily.py 的 run_meta 认
    # DRY_RUN 环境变量（local_run 设），这里两个都认。
    status = {"date": date, "dry": bool(dry) or bool(os.environ.get("DRY_RUN")),
              "n_days": n, "days": days,
              "theta_version": "learned" if C.diff() else "基线",
              "metrics": m_prev, "source_days_weighted": dayw}

    if n < g["min_days"]:
        # 数据不够就**根本不拟合**。不是「拟合了但不接受」——
        # 60 天以下拟合出来的数只会误导，连日志里都不该出现。
        status["verdict"] = {
            "accepted": False,
            "checks": [{"name": "最少天数", "passed": False,
                        "detail": f"{n} 天 / 要求 >= {g['min_days']}，不拟合"}],
            "moved": {}, "evidence": {}}
        R.save_status(status)
        log.info("样本 %d 天 < %d，本阶段只做评估，不动参数", n, g["min_days"])
        return 0

    tr, te = OPT.split_days(days, g["oos_frac"])
    p_tr, p_te = mk(tr), mk(te)
    # 锚定强度 λ_a(N) = λ₀·n₀/(n₀+N) 里的 N 是「喂给优化器的证据量」，
    # 也就是**进入拟合的训练天数**，不是全部天数：样本外那 20% 一行都没
    # 参与 fit。以前传 n=414，λ_a 比按 331 天算低 15.5%（N→∞ 时趋向 20%），
    # 连续解离 θ⁰ 远了 19%（0.756σ vs 0.615σ），第二个意图是 position 还是
    # trend 的排序余量从 0.053σ 缩到 0.003σ（审计 F1-5）。
    lam_a = O.lambda_anchor(lc["objective"]["lambda_anchor"], len(tr),
                            lc["objective"]["anchor_prior_days"])
    theta_fit = OPT.fit(p_tr, lam_a, lc["objective"]["lambda_l1"])
    theta_new, intents = OPT.sparsify(theta_fit, theta_prev, box,
                                      g["max_moves"], g["max_step_frac"])
    log.info("稀疏化：连续解触及 %d 个参数 -> 意图 %s",
             sum(1 for k in box
                 if abs(theta_fit[k] - theta_prev[k]) > 1e-9), intents)

    v, judged = _judge(c, df, mk, p_te, theta_new, theta_prev, days, date,
                       intents=intents)
    dfo, p_look, codes = judged["dfo"], judged["p_look"], judged["codes"]
    status["verdict"] = v.to_dict()
    status["train_source"] = source
    status["intents"] = list(intents)
    status["lambda_anchor"] = lam_a
    status["run_at"] = lc.get("run_at", "16:30:00")
    # 非有限分数/收益的行数。0 以外的值说明有一列在回填或在线链路上缺了，
    # 那些行已被剔出目标函数，但必须留一个能被界面查询的数（教训 16、F1-2）
    status["n_nonfinite"] = int(getattr(full, "n_nonfinite", 0))
    status["accepted_total"] = gate.accepted_count()
    status["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    status["regime_today"] = _regime_of(date)
    if not dry:
        # 每次裁决都记（接受与否），面板上的「第 N 次裁决」和时间线靠它
        gate.log_verdict(date, v, {"source": source, "n_days": n})

    _status_extras(c, status, df, dfo, date, dry)
    R.save_status(status)

    for ck in v.checks:
        log.info("闸门 %-14s %s  %s", ck.name, "过" if ck.passed else "不过",
                 ck.detail)
    if not v.accepted:
        log.info("不接受本次变更，参数保持不变")
        return 0
    if dry:
        log.info("dry-run：本可接受，但不写盘")
        return 0

    m_new = full.metrics(theta_new, lc["objective"]["top_k"])

    # 通道 4：Opus 审稿。统计闸门管数字，它管「数字和叙事对不对得上」。
    from learn import llm_review
    review = llm_review.run(date, {
        "moved": {k: list(vv) for k, vv in v.moved.items()},
        "evidence": v.evidence,
        "gates": [c_.__dict__ if hasattr(c_, "__dict__") else c_
                  for c_ in v.checks],
        "recent_regimes": _regime_counts(),
        "shadow_compare": status.get("shadow", [])[-10:],
    }, lc["llm"]["model"])
    status["review"] = review
    R.save_status(status)
    if (lc["llm"].get("review_mode", "advisory") == "veto"
            and review.get("stance") == "反对"):
        _hold_change(date, v, status, theta_new, m_new, review, source, n,
                     metrics_prev=m_prev,
                     old_top=p_look.top_codes(theta_prev, codes),
                     new_top=p_look.top_codes(theta_new, codes))
        return 0

    A.write(theta_new, v.evidence, date)
    # theta_version 是在 A.write 之前算的（那时 learned.yaml 还不存在），
    # 不刷新的话首次接受变更当天面板会写「参数已变更 · 版本 基线」自相矛盾
    status["theta_version"] = "learned" if C.diff() else "基线"
    gate.record(date, theta_new, v, m_new)
    html = R.build_html(date, v, m_prev, m_new,
                        p_look.top_codes(theta_prev, codes),
                        p_look.top_codes(theta_new, codes),
                        llm_note=llm_review.as_note(review),
                        regime_counts=_regime_counts())
    OUT.mkdir(exist_ok=True)
    (OUT / "change.html").write_text(html, encoding="utf-8")
    # 发没发出去要留在状态里：R.send 吞掉所有异常，只看它不抛等于没看
    # （教训 13/16，和影子提案那条是同一个形状，见 shadow.maybe_propose）
    status["change_mail_sent"] = bool(R.send(date, html, c))
    R.save_status(status)
    log.info("参数已更新并发信" if status["change_mail_sent"]
             else "参数已更新，但变更邮件没发出去（见 state/learning_status.json）")
    return 0


def _refresh_manual_status(verdict: dict | None) -> None:
    """人工落地/回滚之后，把状态文件和学习面板刷一遍（审计 F3-14）。

    apply-held 和 rollback 以前只动 learned.yaml / theta_history，不碰
    learning_status.json，也不重建 learn.html：到下一次学习跑之前（最长跨
    周末），面板和 `--stage status` 一直显示旧的「参数版本 / 累计接受 N 次」，
    而参数其实已经变了或已经回滚了。用户按变更邮件的指引跑完 rollback，
    回头看面板会以为没回滚成。
    """
    try:
        st = json.loads(R.STATUS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        st = {}
    st["theta_version"] = "learned" if C.diff() else "基线"
    st["accepted_total"] = gate.accepted_count()
    if verdict is not None:
        st["verdict"] = verdict
        st.pop("held", None)
    st["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    R.save_status(st)
    try:
        from learn import panel as LP
        LP.build()
    except Exception as e:  # noqa: BLE001
        log.warning("学习面板生成失败（不影响流程）: %s", e)


def _apply_held(c: dict) -> int:
    """人工确认落地被 Opus 审稿搁置的那次变更。"""
    held = STATE / "held_change.json"
    if not held.exists():
        print("没有被搁置的变更")
        return 0
    j = json.loads(held.read_text(encoding="utf-8"))
    A.write(j["theta"], j["verdict"]["evidence"], j["date"])
    # 和闸门直接接受的路径一样记一行：冷却期、accepted_total、审计都读它。
    # 以前这里不记，人工落地的变更在系统眼里等于「从未接受过」。
    vd = j["verdict"]
    v = gate.Verdict(accepted=True, checks=vd.get("checks", []),
                     moved={k: tuple(x) for k, x in vd.get("moved", {}).items()},
                     evidence=vd.get("evidence", {}))
    gate.record(j["date"], j["theta"], v, j.get("metrics", {}))
    _refresh_manual_status({**vd, "accepted": True})
    # 变更邮件也要补发：report.py 的规则是「接受了变更 -> 当天发一封」，
    # 搁置告警邮件里只有 Opus 的反对理由，没有参数表和回滚方法。
    try:
        from learn import llm_review
        review = j.get("review") or {}
        note = (llm_review.as_note(review) if review.get("stance") else "")
        html = R.build_html(
            j["date"], v, j.get("metrics_prev", {}), j.get("metrics", {}),
            j.get("old_top", {}), j.get("new_top", {}),
            llm_note=note + "（Opus 审稿曾搁置，人工确认落地）",
            regime_counts=j.get("regime_counts"))
        OUT.mkdir(exist_ok=True)
        (OUT / "change.html").write_text(html, encoding="utf-8")
        R.send(j["date"], html, c)      # SKIP_MAIL=1 照样生效
    except Exception as e:  # noqa: BLE001
        log.warning("变更邮件补发失败（参数已生效）: %s", e)
    held.unlink()
    print(f"已落地 {j['date']} 被搁置的变更：{list(j['verdict']['moved'])}")
    return 0


def _rollback() -> int:
    if A.rollback():
        _refresh_manual_status(None)
        print("已回滚")
    else:
        print("本来就是人工基线")
    return 0


def _regime_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for p in (STATE / "llm_eval").glob("*.json"):
        try:
            k = json.loads(p.read_text(encoding="utf-8")).get("day_regime", "正常")
            out[k] = out.get(k, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    return out


def _maybe_attribute(c: dict, date: str, force: bool = False) -> bool:
    """LLM 归因：**跑过就不再跑**（state/llm_eval/<date>.json 在就跳过）。

    两个理由（审计 F3-12）：
      1. 成本。stage_learn 任何未捕获异常都会让 DailyReport-Local-Learn 在
         16 小时窗口里每 30 分钟重跑一次 `--stage all`，每次真调一次 Opus
         （上限 240 秒），最多 32 次，而且每次都把新的归因文件 push 进 git。
      2. 可复现。历史日权重本该是冻结的训练输入。CLI 带 WebSearch，输出
         天然不确定；同一天的 day_regime 一翻转，那天在优化器里的权重就在
         1.0 / 0.70 / 0.35 / 0.0 之间变，后续所有拟合、闸门统计量、面板计数
         和会诊证据跟着变，而 load_day_weights 的文档承诺「给定那些 JSON
         优化器完全确定」。

    强制重跑走 `--stage llm`。返回是否真调了一次。
    """
    lc = c["learning"]["llm"]
    if not lc.get("enabled", True):
        return False
    done = STATE / "llm_eval" / f"{date}.json"
    if done.exists() and not force:
        log.info("归因 %s 已有，跳过（强制重跑：--stage llm）", done.name)
        return False
    bp = OUT / "eval_brief.json"
    try:
        fresh = json.loads(bp.read_text(encoding="utf-8"))["date"] == date
    except Exception:  # noqa: BLE001
        fresh = False
    if not fresh:
        # brief 是别的日子留下的。拿它归因会把昨天的票安到今天头上，
        # 归因文件按日期落盘，错一天就污染那一天的日权重。
        log.info("eval_brief.json 不是 %s 的，跳过归因", date)
        return False
    from learn import llm_local
    return llm_local.run(date, bp, lc["model"],
                         lc["timeout_seconds"]) is not None


# ---------------------------------------------------------------------
#  stage: race（模型擂台）
# ---------------------------------------------------------------------
def stage_race(c: dict) -> int:
    from learn import model_select as MS
    lc = c["learning"]
    # 训练表只经 _load_train 拿，和 stage_learn / shadow.fit 逐行同源（审计
    # F3-16）。原来这里自己 glob 回填表 + neutralize，是第二份加载逻辑：
    # 2026-09-15 起 _load_train 把在线真值天并进来并用在线版覆盖重叠日，
    # 擂台却还是纯回填。重叠 7 天 5843 对样本实测，同一只票两份数据里
    # slope 相关 -0.033、dive 0.008、monotonic 只有 48.5% 一致（回填的竞价
    # 轨迹是代理值，在线是真采样），而影子最大系数就是 slope。那时
    # 「胜者当影子试运行」说的已经不是同一份数据上的胜者了。
    df, source = _load_train(c)
    if df.empty:
        log.error("没有带标签的数据")
        return 1
    n_on = (int(df.loc[df["_src"] == "online", "date"].nunique())
            if "_src" in df.columns else 0)
    log.info("擂台训练源 %s：%d 行 %d 天（在线 %d 天）", source, len(df),
             df["date"].nunique(), n_on)
    res = MS.walk_forward(df, c, n_folds=5,
                          min_train_days=max(20, len(df["date"].unique()) // 3),
                          top_k=lc["objective"]["top_k"])
    if res.empty:
        log.warning("天数不足，擂台跑不起来")
        return 0
    summ = MS.summarize(res, lc["gate"]["bootstrap_n"])
    win, why = MS.pick(summ)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n选中：{win}\n理由：{why}")
    OUT.mkdir(exist_ok=True)
    summ.to_csv(OUT / "model_race.csv", index=False)
    # 记下这张表是在哪份数据、什么时候算出来的：擂台是手动阶段，盘上那份
    # 2026-09-02 产的结果在面板上挂了两周，没人看得出它比影子少了 17 个
    # 在线日（审计 F3-16）。
    (OUT / "model_race.json").write_text(json.dumps(
        {"winner": win, "why": why, "source": source,
         "days": int(df["date"].nunique()), "online_days": n_on,
         "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
         "table": summ.to_dict("records")}, ensure_ascii=False, indent=2,
        default=float), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["label", "brief", "learn", "race", "rollback",
                             "status", "backfill", "intraday", "exits",
                             "llm", "all", "apply-held", "build-train",
                             "council"])
    ap.add_argument("--date", default=None)
    ap.add_argument("--backfill", action="store_true",
                    help="label 阶段从 cache/hist_daily.parquet 取，而不是联网")
    ap.add_argument("--all", action="store_true",
                    help="label 阶段：把所有已有快照日都补一遍")
    ap.add_argument("--point", default=None,
                    help="intraday 阶段的采样点 HH:MM:SS，留空=按当前时刻就近")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force-llm", action="store_true",
                    help="all 阶段：归因已有也重跑一次（默认跑过就不再跑）")
    ap.add_argument("--force", action="store_true",
                    help="label 阶段：口径改了，可用行更少也覆盖旧标签")
    a = ap.parse_args()

    c = C.load()
    # 缺省日必须和调度器同一口径：intraday / backfill 确实要「今天」，
    # 其余一律最近一个**已收盘交易日**。今天之前这里是自然日 today_bj()，
    # 北京 00:00~15:05 之间手敲一次不带 --date 的学习线就会把「明天」
    # 盖进 learning_status.json，当天的计划任务整天跳过（审计 F3-10）。
    if a.stage in ("intraday", "backfill"):
        date = a.date or today_bj()
    else:
        date = a.date or last_closed_bj()

    if a.stage == "apply-held":
        return _apply_held(c)
    if a.stage == "rollback":
        return _rollback()
    if a.stage == "status":
        lines = R.status_lines()
        print("\n".join(lines) if lines else "还没有学习状态")
        for path, old, new in C.diff():
            print(f"  {path}: {old} -> {new}")
        return 0
    if a.stage == "build-train":
        # 用已缓存的回填源重建训练表。以前 learn.backfill.build 没有任何入口，
        # 表一旦落盘就再也生不出来（改了口径也换不掉）。
        from learn import backfill as BF
        out = BF.build(c)
        print(f"训练表 -> {out}")
        return 0
    if a.stage == "backfill":
        import pandas as pd
        from learn import sources
        codes = pd.read_csv(ROOT / "cache" / "codes.csv",
                            dtype=str)["code"].tolist()
        b = c["learning"]["backfill"]
        sources.fetch_daily(codes, b["start"], date, b["workers"])
        sources.fetch_auction_ts(b["start"], date)
        log.info("回填源完整度：%s，可学维度 %s",
                 sources.completeness(), sources.learnable_dims())
        return 0
    if a.stage == "intraday":
        from learn import intraday as ID
        snap = ROOT / "data" / date[:7] / f"auction_{date}.parquet"
        if not snap.exists():
            log.warning("%s 没有竞价快照，无从采样", date)
            return 0
        codes = list(pd.read_parquet(snap)["code"])
        ID.collect_one(codes, date, a.point or ID.nearest_point())
        return 0
    if a.stage == "exits":
        from learn import intraday as ID, vscore
        wide = ID.load_all()
        if wide.empty:
            print("还没有盘中采样数据。这一项要从现在开始攒，历史补不了。")
            return 0
        frames = []
        for d in sorted(wide["date"].unique()):
            sp = ROOT / "data" / d[:7] / f"auction_{d}.parquet"
            if sp.exists():
                sn = pd.read_parquet(sp)
                sn["date"] = d
                frames.append(sn.merge(wide[wide.date == d].drop(columns="date"),
                                       on="code", how="inner"))
        if not frames:
            print("采样数据对不上任何竞价快照")
            return 0
        df = pd.concat(frames, ignore_index=True)
        s_, rej = vscore.score_df(df, c)
        df = df[~rej]
        cur = ID.exit_curve(df, s_[~rej], c["learning"]["objective"]["top_k"])
        print(cur.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("")
        print("看 ic 那列：如果 t0935 明显高于 t1500，说明信号衰减快，")
        print("「开盘买收盘卖」这个标签在系统性低估打分器。")
        return 0
    if a.stage == "label":
        if a.all:
            # 2（没覆盖）不能 `rc |=` 混进失败里，也不能就这么算了：批量重打
            # 一遍时被守卫拦下的正是「口径改了、行数变少」的那几天，而那是唯一
            # 需要人看一眼的信号。逐天收集，末尾一次性列出来（教训 16：
            # 失败只写日志等于没写，这里至少要点名到天）。
            rc, kept = 0, []
            for d in dataset.snapshot_days():
                r = stage_label(c, d, a.backfill, a.force)
                if r == 2:
                    kept.append(d)
                else:
                    rc |= r
            if kept:
                log.warning("%d 天保留了旧标签没覆盖（新的可用行更少）：%s。"
                            "口径改了要重打就加 --force", len(kept),
                            " ".join(kept))
            return rc
        return stage_label(c, date, a.backfill, a.force)
    if a.stage == "brief":
        return stage_brief(c, date)
    if a.stage == "llm":
        # 手工阶段要能从退出码看出「没做事」：以前无论跳没跳过都 return 0
        from learn import llm_local
        lc = c["learning"]["llm"]
        res = llm_local.run(date, OUT / "eval_brief.json", lc["model"],
                            lc["timeout_seconds"])
        return 0 if res else 1
    if a.stage == "all":
        # 本地全流程：抓标签 -> 备归因输入 -> 归因 -> 拟合与闸门。
        # 每一步失败都不阻断后面（归因尤其：它是研究性的，不是关键路径）。
        rc = stage_label(c, date, a.backfill, a.force)
        if rc == 1:
            # 标签没拿到（网络不通、快照时间戳不是当天）就别往下走：往下走
            # 会写 learning_status，计划任务据此判「今天跑完了」不再重试，
            # 这一天的真值就永久丢了。退出非零，下一次敲门再抓一次（很快）。
            log.error("%s 标签没拿到，本轮不拟合、不写状态，等下一次重试", date)
            return rc
        if rc == 2:
            # 2 = 已有一份可用行更多的标签，没覆盖。那天的真值本来就在盘上，
            # 拟合照跑；当成失败 return 2 的话，计划任务每半小时重试一次、
            # 一整天都不会拟合（labels.save 的守卫每次都拦），学习线停摆。
            log.warning("%s 保留了旧标签（新的可用行更少），学习照常往下跑", date)
            rc = 0
        stage_brief(c, date)
        _maybe_attribute(c, date, force=a.force_llm)
        rc |= stage_learn(c, date, a.dry)
        # 学习会诊：研究性组件，任何失败都不影响上面的裁决和退出码
        if not a.dry:
            stage_council(c, date)
        return rc
    if a.stage == "council":
        return stage_council(c, date)
    if a.stage == "race":
        return stage_race(c)
    return stage_learn(c, date, a.dry)


def stage_council(c: dict, date: str) -> int:
    """学习会诊（docs/council.md）。fail-open：失败只写 state/council/latest.json。"""
    lc = (c.get("learning") or {}).get("council") or {}
    if not lc.get("enabled", True):
        log.info("学习会诊已关闭（learning.council.enabled=false）")
        return 0
    # 控制台的「学习会诊」按钮不带 --date（gui/jobs.py），午夜后按一次就会
    # 把会诊归档到 state/council/<明天>/，提案 id 前缀和邮件主题全错一天
    # （审计 F3-10）。会诊是 fail-open 的，这里也只能 return 0。
    if _reject_bad_date(date, "不做会诊"):
        return 0
    try:
        from learn.council import run as CR
        s = CR.run(c, date)
        log.info("学习会诊 %s：%s，%d 条提案，%.0f 秒", date,
                 "成功" if s.get("ok") else s.get("error", "失败"),
                 s.get("n_proposals", 0), s.get("seconds", 0))
    except Exception as e:  # noqa: BLE001
        log.warning("学习会诊失败（不影响学习流程）: %s", e)
        try:
            from learn.council import run as CR
            CR._write_latest({"date": date, "ok": False, "error": str(e)[:200]})
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
