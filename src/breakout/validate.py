"""
走向前验证。整个爆发线的成败由这个模块的输出判定。

训练和验收都用 y_up（2026-09-12 实验修正）
------------------------------------------
第一版设计是「训练用 y_t0（起涨第一天，信号干净），验收用 y_up」。
实验证明这是错的：

    训练 y_t0  ->  命中 7.39%  （2.08 倍）
    训练 y_up  ->  命中 11.11% （3.13 倍）   +3.72 个百分点

原因是**训练目标和评估目标错配**。起涨后第 2、3 天买入同样能涨 50%，
y_up 把它们算作正确答案，而 y_t0 把它们当负样本 —— 模型被教会了排斥
本该选中的票。口径一致比信号干净重要得多。

y_t0 仍然保留，用于统计「一段行情算一次」的场景（比如报告里说
三年出了多少段行情），不再当训练目标。

验收指标一律按 y_up 算：拿 y_t0 的命中率去汇报会虚高得离谱
（分母小一个量级），那是自欺。

协议
----
    训练集   2023-05 .. 2025-02
    验证集   2025-03 .. 2025-12     选模型、调超参
    holdout  2026-01 .. 2026-09     锁死，全程只看一次

走向前：每个月重新拟合一次，只用该月**之前**的数据预测该月。
这是最接近实盘的评估方式——实盘里你永远只有过去的数据。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# 净化长度必须等于标签窗口，两个 20 各写各的迟早会漂（见 purge_cut / train_slice）
from label import UP_WINDOW as PURGE_DAYS

log = logging.getLogger("validate")

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_END = "2025-03-01"
VALID_END = "2026-01-01"      # 之后是 holdout，锁死
TOP_N = 10                    # 清单 A 每天给几只
ST_CACHE = ROOT / "cache" / "st_codes.json"
_ST: dict = {}


def st_codes(path: Path | None = None) -> set[str]:
    """ST 名单。回测和生产必须同一份来源。

    生产（daily.risk_filter）按当天腾讯快照的名称判 ST，回测没有历史名称，
    只能读这份缓存。2026-09-16 实测：W5 的 783 个样本里 98 个（12.5%，29 只
    代码）名称含 ST，而生产永远不会选它们；剔掉并按预测值补位之后，
    连续≥1..5 天的命中率从 12.64/15.74/14.29/12.96/6.67% 变成
    11.76/13.41/11.90/10.64/3.70% —— 邮件里印给用户的成绩高报了 7%~80%。

    拿不到缓存就返回空集合：宁可不剔，也不在回测里凭空造一个名单。
    注意静态名单只能把**数量**做对、成员做不对（上面 98 行里有一半是现已
    摘帽、当时不是 ST 的票），真正的做法是训练表带一列 is_st，见 breakout_log。
    """
    p = path or ST_CACHE
    key = str(p)
    if key not in _ST:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            cs = raw.get("codes", raw) if isinstance(raw, dict) else raw
            _ST[key] = {str(c).zfill(6) for c in cs}
        except Exception as e:  # noqa: BLE001
            log.info("没有 ST 名单缓存（%s），回测不剔 ST", e)
            _ST[key] = set()
    return _ST[key]


def daily_topn(df: pd.DataFrame, proba: np.ndarray,
               n: int = TOP_N, y: str = "y_up",
               st: set[str] | None = None) -> pd.DataFrame:
    """按天取前 n 只，算命中率。

    必须**按天**取，不能全局取 top：全局 top 会集中落在少数几个大行情日
    （2024-09 那种），算出来的命中率反映的是"挑对了日子"而不是"挑对了票"。
    清单 A 是每天都要出的，评估口径必须和使用口径一致。
    """
    d = df[["date", "code", y]].copy()
    d["p"] = proba
    d = d[np.isfinite(d[y])]
    if not len(d):
        return pd.DataFrame(columns=["date", "code", "p", y])
    d = d.sort_values(["date", "p"], ascending=[True, False])
    # ST 在**取前 n 名之前**剔：生产是剔完再从 11 名之后补位的，
    # 先取 10 再删等于每天少给几只，两条路的样本集不一样
    bad = st_codes() if st is None else st
    if bad:
        d = d[~d["code"].astype(str).isin(bad)]
    return d.groupby("date", sort=False).head(n)


def pick(day: pd.DataFrame, proba: np.ndarray, q: np.ndarray | None,
         score_min: int | None = None, cap: int | None = None,
         board_adj: dict | None = None,
         eligible: pd.Series | None = None,
         st: set[str] | None = None) -> pd.DataFrame:
    """**一天**的清单 A。生产和验收唯一的一份实现。

    规则：预测值 × 板块系数 -> 分数刻度 -> 剔除 -> 够格才上 -> 截到 cap。
    以前验收走 daily_topn（每天固定前 10、无门槛、无板块校正、无风险剔除），
    生产走 daily.stage_scan（够 SCORE_MIN 分、板块校正、截 CAP_A），两条路径
    没有任何共用代码。同一份逐月滚动打分实测：前 10 无门槛 11.16%（n=2070），
    生产口径 12.64%（n=783，和 out_breakout/board_hit.json 逐位相同），
    板块最强/最弱从 1.63「通过」变成 6.20「重度不通过」—— 一个假阴一个假阳，
    验收对上线规则没有证明力。

    参数给 None 表示「用生产值」（daily.SCORE_MIN / CAP_A / BOARD_ADJ）。
    board_adj={} 且 score_min=0 时逐行退化成 daily_topn，给旧实验复算用。
    """
    import daily as D
    d = day.copy()
    adj_map = D.BOARD_ADJ if board_adj is None else board_adj
    if adj_map and "board" in d.columns:
        adj = d["board"].map(adj_map).fillna(1.0).to_numpy(float)
    else:
        adj = np.ones(len(d))
    # 排序必须用**连续**的预测值：分数取整后前几名大量并列，
    # 按整数排会把第 1 名的命中率从 25.12% 打到 21.74%（daily.py 同一条注释）
    d["_p"] = np.asarray(proba, dtype=float) * adj
    d["score"] = (D.to_score(d["_p"].to_numpy(), q) if q is not None
                  else np.full(len(d), 100.0))
    d = d.sort_values("_p", ascending=False)
    if eligible is not None:
        d = d[eligible.reindex(d.index).fillna(True).to_numpy(bool)]
    # ST 和风险剔除一样，在截 cap 之前生效，让 11 名之后的票补位上来
    bad = st_codes() if st is None else st
    if bad and "code" in d.columns:
        d = d[~d["code"].astype(str).isin(bad)]
    sm = D.SCORE_MIN if score_min is None else score_min
    n = D.CAP_A if cap is None else cap
    return d[d["score"] >= sm].head(n)


def pick_days(df: pd.DataFrame, proba: np.ndarray, q: np.ndarray | None,
              score_min: int | None = None, cap: int | None = None,
              board_adj: dict | None = None) -> pd.DataFrame:
    """对多天逐日调 pick。必须**按天**取，理由见 daily_topn。"""
    d = df.copy()
    d["_proba"] = np.asarray(proba, dtype=float)
    out = [pick(g, g["_proba"].to_numpy(), q, score_min=score_min, cap=cap,
                board_adj=board_adj)
           for _, g in d.groupby("date", sort=False)]
    out = [x for x in out if len(x)]
    if not out:
        return pd.DataFrame(columns=list(d.columns) + ["_p", "score"])
    return pd.concat(out, ignore_index=True)


def summarize(picks: pd.DataFrame, y: str = "y_up") -> dict:
    if not len(picks):
        return {"n": 0, "hit": float("nan")}
    return {"n": int(len(picks)), "hit": float(picks[y].mean())}


def purge_cut(df: pd.DataFrame, month: str, gap: int = PURGE_DAYS) -> str:
    """训练集截止日：测试月首个交易日往前数 gap 个**市场**交易日。

    紧挨测试月前 20 天的行，y_up 是用测试月里的最高价算出来的：把它们放进
    训练集，模型就带着「测试月初谁会涨」的信息进测试月。生产的 load_or_fit
    用 TRAIN_END_GAP=25 截断是净化过的，走向前以前没做，成绩偏乐观
    （2026-09-15 审计，实验 10 重算）。

    只按全市场日期数是不够的，停牌票会从这条线底下钻过去，见 train_slice。
    """
    dates = sorted(df["date"].unique())
    first = next((i for i, d in enumerate(dates) if d >= month + "-01"), len(dates))
    return dates[max(first - gap, 0)]


def train_slice(df: pd.DataFrame, month: str, gap: int = PURGE_DAYS) -> pd.DataFrame:
    """走向前的训练集：行日期在净化线之前，**且**该行标签窗口的第 gap 根
    K 线也落在测试月之前。

    为什么两个条件都要：label.label_up 的窗口数的是这只票**自己**的 K 线
    （`high[i+1:i+1+window]`），而 purge_cut 数的是全市场日期。停牌日在
    新浪的日线表里没有行，于是停牌票的第 20 根 K 线会伸到净化线之后 ——
    2026-09-16 实测走向前十个月里有 1,884 行是这么漏进训练集的，其中
    415 个正样本（正样本率 22.0%，全表 3.42% 的 6.4 倍：停牌复牌常伴大涨），
    最深伸进测试月及之后 44 个交易日。例：600777 的 2025-04-29 行日期
    < 净化线 2025-04-30，但它停牌到 2025-07-08，第 20 根 K 线落在
    2025-08-01，标签用的是整个 6 月和 7 月的最高价。

    生产侧 daily.load_or_fit 不受影响：那些行的标签是 NaN，被
    model.stratified_sample / fselect 的 isfinite 过滤掉，只是丢行，
    不会喂错标签。
    """
    cut = purge_cut(df, month, gap)
    # shift(-gap) 给的是该票第 gap 根 K 线的日期；末尾不足 gap 根的填一个
    # 永远落在未来的哨兵，这些行的标签本来就是 NaN
    end = (df.sort_values(["code", "date"]).groupby("code")["date"]
             .shift(-gap).reindex(df.index).fillna("9999-12-31"))
    return df[(df["date"] < cut) & (end < month + "-01")]


def walk_forward(df: pd.DataFrame, feats: list[str], make_model,
                 y_train: str = "y_up", y_eval: str = "y_up",
                 start: str = TRAIN_END, end: str = VALID_END,
                 top_n: int | None = None, score_min: int | None = None,
                 board_adj: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """按月走向前。返回 (每月成绩, 汇总)。

    make_model 是个工厂函数（每月要新建一个，不能复用上个月拟合过的）。

    选票走 pick()，默认就是**生产规则**（≥SCORE_MIN 分、板块校正、CAP_A）：
    验收评的必须是真正要上线的那条规则。要复算旧口径就显式传
    score_min=0, board_adj={}, top_n=TOP_N。
    """
    import model as M

    months = sorted({d[:7] for d in df["date"] if start <= d < end})
    rows, all_picks = [], []
    for m in months:
        tr = train_slice(df, m)
        te = df[df["date"].str[:7] == m]
        tr = tr[np.isfinite(tr[y_train])]
        if len(tr) < 5000 or not len(te):
            continue
        trs = M.stratified_sample(tr, y_train)
        if trs[y_train].sum() < 30:
            continue
        mdl = make_model()
        mdl.fit(trs, feats, y_train)
        # 分数刻度和生产一致：用训练集自己的预测值分布定分位点
        q = np.quantile(mdl.predict_proba(trs), np.linspace(0, 1, 101))
        te = te[np.isfinite(te[y_eval])]     # 生产不需要标签，丢弃留在 pick 之外
        if not len(te):
            continue
        p = mdl.predict_proba(te)
        picks = pick_days(te, p, q, score_min=score_min, cap=top_n,
                          board_adj=board_adj)
        # 只留要用的几列：87 个特征跟着 picks 走十个月会白占几百 MB
        keep = [c for c in ("date", "code", "board", "_p", "score", y_eval)
                if c in picks.columns]
        picks = picks[keep]
        s = summarize(picks, y_eval)
        base = float(te[y_eval].mean(skipna=True))
        rows.append({"month": m, "n_pick": s["n"], "hit": s["hit"],
                     "base": base,
                     "lift": s["hit"] / base if base > 0 else np.nan})
        all_picks.append(picks.assign(month=m))
        log.info("  %s  选 %3d 只  命中 %.1f%%  基础 %.1f%%  %.1f倍",
                 m, s["n"], 100 * s["hit"], 100 * base,
                 rows[-1]["lift"] if np.isfinite(rows[-1]["lift"]) else 0)

    by_month = pd.DataFrame(rows)
    picks = (pd.concat(all_picks, ignore_index=True) if all_picks
             else pd.DataFrame())
    return by_month, {"picks": picks}


def acceptance(by_month: pd.DataFrame, picks: pd.DataFrame,
               df: pd.DataFrame, y_eval: str = "y_up") -> dict:
    """对照设计文档 2.2 的验收表。每条都给实测值和是否达标。

    不达标不是"再调调参数"的信号，是按分级响应走：
    加特征 -> 换模型 -> 改事件定义 -> 判定不可行。
    """
    if not len(by_month) or not len(picks):
        return {"ok": False, "reason": "没有任何有效月份"}

    overall = float(picks[y_eval].mean())
    # 基准必须和成绩**同一段时间**。调用方给的 df 往往是整张训练表（arena.py
    # 只切掉 holdout，2023-01..2025-12 的 y_up 均值 3.55%），而评估只发生在
    # picks 覆盖的那几个月（验证集 2025-03..12 是 2.93%）。混用把倍数压低 21%
    # （L1 记成 3.13 倍，真值 3.79 倍），和 export.BASE / exp_window 的 2.93%
    # 对不上，同一份 arena.json 里逐月 lift 和汇总 lift 也互相矛盾（教训 30）。
    src = picks if "month" in picks.columns else by_month
    mo = set(src["month"].astype(str))
    seg = df[df["date"].astype(str).str[:7].isin(mo)]
    base = float(seg[y_eval].mean(skipna=True)) if len(seg) else float("nan")
    if not len(seg):
        log.warning("传进来的 df 一行都不落在评估月份 %s 里，基准算不出来",
                    sorted(mo)[:3])

    # 剔除两个极端月（924 行情和 2025-12）后重算
    ex = picks[~picks["month"].isin(["2024-09", "2025-12"])]
    hit_ex = float(ex[y_eval].mean()) if len(ex) else np.nan

    # 门槛制下有的月份一只票都不够格，n_pick=0 时 hit 是 NaN，那是**空仓**
    # 不是「零命中」。不区分的话闸门会把「这个月没出手」记成「这个月全错」。
    have = by_month[by_month["n_pick"] > 0]
    zero_months = int((have["hit"] == 0).sum())
    empty_months = int((by_month["n_pick"] == 0).sum())

    # 分板块
    board = None
    if "code" in picks.columns:
        import features as F
        b = picks.assign(board=picks["code"].map(F.board_of))
        g = b.groupby("board")[y_eval].agg(["mean", "size"])
        g = g[g["size"] >= 20]
        if len(g) >= 2:
            board = {"ratio": float(g["mean"].max() / max(g["mean"].min(), 1e-9)),
                     "detail": {k: round(float(v), 4)
                                for k, v in g["mean"].items()}}

    checks = [
        ("清单A命中率", overall, 0.15, overall >= 0.15),
        ("剔除极端月后", hit_ex, 0.10,
         bool(np.isfinite(hit_ex) and hit_ex >= 0.10)),
        ("零命中月份数", zero_months, 1, zero_months <= 1),
    ]
    if board:
        checks.append(("板块最强/最弱", board["ratio"], 3.0,
                       board["ratio"] <= 3.0))

    return {
        "ok": all(c[3] for c in checks),
        "overall_hit": overall, "base_rate": base,
        # 分母是哪几个月、多少行也写进产物：以后再和别处对不上，一眼能看出
        # 是不是同一段时间
        "base_months": sorted(mo), "base_n": int(len(seg)),
        "lift": overall / base if base > 0 else np.nan,
        "hit_ex_extreme": hit_ex, "zero_months": zero_months,
        "empty_months": empty_months,
        "board": board,
        "checks": [{"name": c[0], "value": (None if not np.isfinite(float(c[1]))
                                            else round(float(c[1]), 4)),
                    "threshold": c[2], "pass": bool(c[3])} for c in checks],
    }


def rule_stamp(score_min: int | None = None, cap: int | None = None,
               board_adj: dict | None = None) -> dict:
    """报告自己声明评的是哪条规则。

    同一套规则以前在 daily / truth / exp_precision / exp_window 各抄了一份，
    改 SCORE_MIN 或 BOARD_ADJ 其中一处忘了同步，面板、邮件、验收会各报各的数
    而且不报错（教训 30）。把它写进产物，对不上时自测能看见。
    """
    import daily as D
    return {"score_min": D.SCORE_MIN if score_min is None else score_min,
            "cap": D.CAP_A if cap is None else cap,
            "board_adj": dict(D.BOARD_ADJ if board_adj is None else board_adj),
            # 减持 / 解禁 / 增发三项没有历史数据，回测里复现不了，只复现 ST。
            # 产物写明剔了多少只，拿含 ST 的旧成绩给邮件背书时看得出来。
            "risk_filter": False, "st_excluded": len(st_codes())}


def report(by_month: pd.DataFrame, acc: dict, out: Path,
           extra: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"by_month": by_month.to_dict("records"), "acceptance": acc,
               "rule": rule_stamp()}
    if extra:
        payload.update(extra)
    (out / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    log.info("验证报告 -> %s", out / "validation.json")
