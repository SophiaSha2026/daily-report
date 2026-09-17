"""
清单 A 改成「看过去 5 个交易日（含当天）」能到多少准确率。

用户 2026-09-13 的问题。当前规则只看**当天**的分数，连续够格天数只用来
排序；这里测的是把 5 天窗口本身当成入选条件的几种口径：

    W1  当天够格 + 窗口内够格 >= k 次（不要求相邻，断了也算）
    W2  当天够格 + 连续够格 >= k 天（现规则，做对照）
    W3  不要求当天够格，只要窗口内 >= k 次
    W4  窗口内 5 天预测值的均值排名前 N（完全不用当天分数当门槛）

评估口径和实验 7 完全一致：验证集 2025-03..2025-12 逐月走向前，
y_up = 未来 20 个交易日涨超 50%。基准只算这十个月（2.93%，见 build_cache），
不是整张训练表的 3.55%（那里面混着 2023~2024 的训练月）。

打分结果缓存到 data/breakout/raw/wf_scores.parquet（已 gitignore），
改口径不用重训模型。WF_CACHE 环境变量可换路径。

    python src/breakout/exp_window.py            用缓存（没有就先算）
    python src/breakout/exp_window.py --refit    强制重算打分
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402

import arena as A       # noqa: E402
import build as BD      # noqa: E402
import board_adj as BA
import daily as D       # noqa: E402
import fselect as FS    # noqa: E402
import model as M       # noqa: E402
import validate as V    # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
OUT = ROOT / "out_breakout"
CACHE = Path(os.environ.get("WF_CACHE", str(ROOT / "data" / "breakout" / "raw"
                                            / "wf_scores.parquet")))
RANK_KEEP = 100     # 每天落盘前多少名。>=95 分的都在这里面，足够算窗口
FINAL_ADJ: dict = {}   # rolling/shrink 臂跑完后，下一个月该用的那组因子
# BOARD_ADJ（daily.py）是按验证集 2025-03..12 各板块的命中率除以整体命中率
# 算出来的：arena.json 的 {main .1326, star .1410, bj .0998, chinext .0660}
# 除以 .1111 正好是 {1.19, 1.27, 0.90, 0.59}。而这个脚本又在**同一段**数据上
# 乘着它评 W5，邮件里印的成绩就是板块校正的样本内成绩。
ADJ_FIT_WINDOW = ["2025-03", "2025-12"]


def board_factors(picks: pd.DataFrame, before_month: str,
                  min_n: int = 100, min_board: int = 20) -> dict:
    """用 before_month **之前**已结束月份的名额算板块校正因子。

    因子 = 该板块命中率 / 整体命中率，和 daily.BOARD_ADJ 同一个定义，
    区别只在**拟合窗口和评估窗口不重叠**。用缓存做的反事实（2026-09-16）：
    带固定因子 n=783 命中 12.64%、连续≥2 天 15.74%；不带 n=588 命中 11.73%、
    连续≥2 天 11.83% —— 邮件里的「连续 2 天 15.7%」含约 3.9 个百分点
    （相对高 33%）的样本内增益，而创业板被缓存截断，这还是个下界。

    样本不足就返回空 dict（= 不校正）。宁可不校正也不用一个噪音因子：
    同一批数据劈两半重算，bj 因子 1.52 -> 0.62、main 0.69 -> 1.53，
    段间抖动远大于因子之间的差距。
    """
    past = picks[picks["date"].astype(str).str[:7] < before_month]
    if len(past) < min_n:
        return {}
    overall = float(past["y_up"].mean())
    if not overall > 0:
        return {}
    out = {}
    for b, g in past.groupby("board"):
        if len(g) >= min_board:
            out[str(b)] = float(g["y_up"].mean()) / overall
    return out


def production_rank(d: pd.DataFrame, st: set[str] | None = None) -> pd.DataFrame:
    """按生产口径重排每天的名次：先剔 ST，再按预测值补位。

    daily.stage_scan 是「剔完风险再从 11 名之后补位到 CAP_A 只」，而这里的
    rank 是在**没剔过**的全市场上排的（build_cache 第 128 行）。两者选的不是
    同一批票：2026-09-16 实测 W5 里 12.5% 的样本名称含 ST，剔掉并补位之后
    首日命中 12.64% -> 11.76%，连续≥5 天 6.67% -> 3.70%。
    次新（历史不足 build.MIN_HIST 根）不在这里剔：train.parquet 建表时
    已经按同一条规则砍掉了预热行，缓存里压根没有这种行。
    """
    st = V.st_codes() if st is None else st
    d = d.copy()
    if st:
        d = d[~d["code"].astype(str).isin(st)]
    d = d.sort_values(["date", "_p"], ascending=[True, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    return d


def build_cache(adj_mode: str = "fixed") -> tuple[pd.DataFrame, float]:
    df = A.load(False)
    feats_all = [c for c in df.columns if "__" in c]
    # 消融：WF_DROP=vol_ratio20,amt_ma20 这样列出要排除的特征名（不带 __ 后缀），
    # 配合 WF_CACHE 指到另一个缓存文件，就能在同一份训练表上比「有没有这几个
    # 特征」的差别。实验 9 用它把「公告日对齐」和「加成交量组」的影响拆开。
    drop = [x.strip() for x in os.environ.get("WF_DROP", "").split(",") if x.strip()]
    if drop:
        # 基础名（vol_ratio20）去掉它的三个变换；带 __ 的（mkt_ret5__mean）只去那一列
        ds = set(drop)
        feats_all = [c for c in feats_all
                     if c not in ds and c.rsplit("__", 1)[0] not in ds]
        print("消融：排除 %s，剩 %d 列" % (drop, len(feats_all)), flush=True)
    # 特征选择和走向前第一个月同一条净化线：切在 TRAIN_END 上的话，紧挨
    # 2025-03 的那 20 个交易日的标签由 3 月的最高价决定（S21）
    feats = FS.run(V.train_slice(df, V.TRAIN_END[:7]), feats_all,
                   y="y_up")["keep"]

    months = sorted({d[:7] for d in df["date"]
                     if V.TRAIN_END <= d < V.VALID_END})
    keep, qmap, hist = [], {}, []
    for m in months:
        tr = V.train_slice(df, m)   # 20 日净化 + 停牌票的标签窗口，见 validate
        te = df[df["date"].str[:7] == m].copy()
        tr = tr[np.isfinite(tr["y_up"])]
        te = te[np.isfinite(te["y_up"])]
        if len(tr) < 5000 or not len(te):
            continue
        trs = M.stratified_sample(tr, "y_up")
        mdl = M.L1Lgbm(n_estimators=400).fit(trs, feats, "y_up")
        q = np.quantile(mdl.predict_proba(trs), np.linspace(0, 1, 101))
        qmap[m] = [float(x) for x in q]
        if adj_mode == "none":
            amap = {}
        elif adj_mode == "rolling":
            # 只用本月之前的名额估因子，拟合窗口和评估窗口不重叠
            amap = board_factors(pd.concat(hist, ignore_index=True), m) if hist else {}
        elif adj_mode == "shrink":
            # 同上，但各板块的命中率先按经验贝叶斯收缩回全市场（board_adj.py）：
            # 北交所 19 个名额的 26.3% 不会变成因子 2.37 把清单构成掀翻，
            # 板块之间看不出真实差异时因子恒为 1（自动退化成不校正）。
            amap = BA.factors_before(pd.concat(hist, ignore_index=True), m) if hist else {}
        else:
            amap = D.BOARD_ADJ
        adj = (te["board"].map(amap).fillna(1.0).to_numpy(float) if amap
               else np.ones(len(te)))
        p0 = mdl.predict_proba(te)
        # _p0 是**没乘板块系数**的原始预测值。落盘它和 board，才能不重训就
        # 精确重算「不带校正」的分数和名次（以前只能靠分位边界反推）
        te["_p0"] = p0
        te["_p"] = p0 * adj
        te["score"] = np.clip(np.searchsorted(q, te["_p"]), 0, 100)
        te = production_rank(te)
        keep.append(te[te["rank"] <= RANK_KEEP][
            ["date", "code", "board", "rank", "score", "_p", "_p0", "y_up"]])
        prod = te[(te["score"] >= D.SCORE_MIN) & (te["rank"] <= D.CAP_A)]
        hist.append(prod[["date", "board", "y_up"]])
        print("  " + m + " 完成", flush=True)

    # 最后一个测试月之后该用的那组因子（= 用全部验证月的名额估）。
    # 生产要用的就是它，和验收臂同源。
    global FINAL_ADJ
    if adj_mode in ("rolling", "shrink") and hist:
        h = pd.concat(hist, ignore_index=True)
        nxt = "9999-99"
        FINAL_ADJ = (BA.factors_before(h, nxt) if adj_mode == "shrink"
                     else board_factors(h, nxt))
    d = pd.concat(keep, ignore_index=True)
    # 基准只算验证集那十个月：成绩是在这段上测的，基准混进 2023~2024 的
    # 训练月份（3.56%）就和成绩不是同一段时间，倍数被压低。
    # export.BASE / score_calibration 用的都是这个口径（2.93%）。
    vm = df[(df["date"] >= V.TRAIN_END) & (df["date"] < V.VALID_END)]
    st = V.st_codes()
    if st:
        # 生产的候选池不含 ST，基准也不能含：分子剔了分母不剔，倍数是虚的
        vm = vm[~vm["code"].astype(str).isin(st)]
    base = float(vm["y_up"].mean(skipna=True))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    (CACHE.parent / "wf_base.json").write_text(
        json.dumps({"base": base, "st_excluded": len(st),
                    "min_hist": BD.MIN_HIST}), encoding="utf-8")
    # 每月的分位点：重算「不带校正」的分数要用同一把尺子
    (CACHE.parent / "wf_q.json").write_text(
        json.dumps(qmap), encoding="utf-8")
    return d, base


def load_cache(refit: bool, adj_mode: str = "fixed") -> tuple[pd.DataFrame, float]:
    if CACHE.exists() and not refit:
        d = pd.read_parquet(CACHE)
        base = json.loads((CACHE.parent / "wf_base.json")
                          .read_text(encoding="utf-8"))["base"]
        return d, base
    return build_cache(adj_mode)


def reprice(d: pd.DataFrame, qmap: dict) -> pd.DataFrame:
    """用 _p0 + 每月的分位点重算「不带板块校正」的分数和名次。

    缓存只留每天校正后的前 RANK_KEEP 名，所以重算出来的名次是近似
    （创业板被系数 0.59 压出榜的那些票本来就不在缓存里），偏差方向是
    **高估**不带校正的成绩。
    """
    if "_p0" not in d.columns:
        raise SystemExit("缓存里没有 _p0 列，先跑一次 --refit")
    d = d.copy()
    mon = d["date"].astype(str).str[:7]
    sc = pd.Series(index=d.index, dtype=float)
    for m, g in d.groupby(mon):
        q = np.asarray(qmap[m], dtype=float)
        sc.loc[g.index] = np.clip(np.searchsorted(q, g["_p0"]), 0, 100)
    d["score"] = sc
    d["_p"] = d["_p0"]
    d = d.sort_values(["date", "_p"], ascending=[True, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    return d


def grid_payload(days: int, base: float, win: int, rows: list,
                 all_dates: list, adj_mode: str, adj: dict,
                 n_st: int = 0) -> dict:
    """产物自己声明板块校正是怎么来的。

    只写成绩不写口径，下一个人（或下一个我）就会拿一份样本内成绩当证据用。
    adj_inconsistent=True 表示因子的拟合区间和这次评估的区间**重叠**，
    即这份成绩里含板块校正的样本内增益。
    """
    ev = [str(all_dates[0])[:7], str(all_dates[-1])[:7]] if all_dates else ["", ""]
    fit = ADJ_FIT_WINDOW if adj_mode == "fixed" else None
    applied = adj_mode != "none"
    overlap = bool(fit and ev[0] and fit[0] <= ev[1] and ev[0] <= fit[1])
    return {"days": days, "base": base, "win": win, "grid": rows,
            "drop": os.environ.get("WF_DROP", ""),
            "board_adj": dict(adj), "board_adj_applied": applied,
            "board_adj_mode": adj_mode, "adj_fit_window": fit,
            "eval_window": ev, "adj_inconsistent": overlap,
            # 产物自己声明剔没剔 ST、次新的门槛是多少。邮件里的 STREAK_PERF
            # 就取这份 W5，拿含 ST 的旧网格给它背书时看得出来
            "st_excluded": bool(n_st), "n_st": int(n_st),
            "min_hist": int(BD.MIN_HIST),
            "rule": {"score_min": int(D.SCORE_MIN), "cap": int(D.CAP_A)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--adj", default="fixed",
                    choices=["fixed", "none", "rolling", "shrink"],
                    help="板块校正：fixed=daily.BOARD_ADJ（样本内）、"
                         "none=不校正、rolling=逐月只用之前的月份估、"
                         "shrink=同 rolling 但先做经验贝叶斯收缩（推荐）")
    ap.add_argument("--save-adj", action="store_true",
                    help="--adj shrink 专用：把最后一个测试月之后该用的那组因子"
                         "写进 state/breakout/board_adj.json 给生产用")
    a = ap.parse_args()
    d, base = load_cache(a.refit, a.adj)
    if a.adj == "none" and not a.refit:
        qf = CACHE.parent / "wf_q.json"
        if not qf.exists():
            raise SystemExit("缺 wf_q.json（每月分位点），先跑一次 --refit")
        d = reprice(d, json.loads(qf.read_text(encoding="utf-8")))
    # rolling / shrink 臂也要把因子记进成绩表：export._same_rule 拿它和生产
    # 此刻生效的 BOARD_ADJ 比，记成空的话那道防呆就退化成只比 (分数线, 上限)。
    adj_used = ({} if a.adj == "none"
                else dict(D.BOARD_ADJ) if a.adj == "fixed"
                else dict(FINAL_ADJ))
    # 缓存可能是含 ST 的旧版（rank 在没剔过的全市场上排），这里按生产口径
    # 重排一次；缓存本来就是剔过的话这一步是恒等变换
    st = V.st_codes()
    d = production_rank(d, st)

    all_dates = sorted(d["date"].unique())
    di = {x: i for i, x in enumerate(all_dates)}
    d["_i"] = d["date"].map(di)
    days = len(all_dates)
    W = a.win
    print("\n验证集 %d 个交易日（%s ~ %s），全市场基准 %.2f%%，窗口 %d 天\n"
          % (days, all_dates[0], all_dates[-1], 100 * base, W))

    rows = []

    def rec(label: str, g: pd.DataFrame, kind: str) -> None:
        if len(g) < 20:
            return
        hit = float(g["y_up"].mean())
        nd = g["date"].nunique()
        per_day = len(g) / days
        se = float(np.sqrt(hit * (1 - hit) / len(g)))
        rows.append({"kind": kind, "label": label, "n": int(len(g)),
                     "empty": int(days - nd), "per_day": per_day,
                     "hit": hit, "se": se, "lift": hit / base,
                     "per_month": 21 * per_day * hit})

    # ---- 窗口内够格次数 ----
    for thr in (95, 97, 98):
        ok = d[d["score"] >= thr]
        by_code = {c: np.sort(g["_i"].to_numpy())
                   for c, g in ok.groupby("code")}
        sub = ok.copy()
        cnt = np.zeros(len(sub), dtype=int)      # 窗口内够格次数（含当天）
        streak = np.zeros(len(sub), dtype=int)   # 连续够格天数（含当天）
        codes = sub["code"].to_numpy()
        idx = sub["_i"].to_numpy()
        for j in range(len(sub)):
            arr = by_code[codes[j]]
            i = int(idx[j])
            cnt[j] = int(((arr > i - W) & (arr <= i)).sum())
            s = 1
            aset = set(int(x) for x in arr)
            while (i - s) in aset:
                s += 1
            streak[j] = s
        sub["cnt"] = cnt
        sub["streak"] = streak

        for k in range(1, W + 1):
            rec("≥%d分 且 近%d日内够格≥%d次" % (thr, W, k),
                sub[sub["cnt"] >= k], "W1")
        for k in range(1, 6):
            rec("≥%d分 且 连续≥%d天" % (thr, k),
                sub[sub["streak"] >= k], "W2")
        for k in range(1, W + 1):
            rec("≥%d分 且 近%d日内恰好%d次" % (thr, W, k),
                sub[sub["cnt"] == k], "W1x")

    # ---- W5：生产口径。daily.py 的清单 A 是「≥97 分且当天按预测值前 10 名」，
    # 连续天数按**上榜**天数算（中间一天没上榜就归零）。名次已由
    # production_rank 剔 ST 后重排、11 名之后补位，和 daily.stage_scan 一致；
    # 减持 / 解禁 / 增发没有历史数据，回测里复现不了（tools/rerun_breakout.py
    # 也承认拿不到历史名单）。邮件里印的 STREAK_PERF 就取这组。 ----
    for thr in (D.SCORE_MIN,):
        ok = d[(d["score"] >= thr) & (d["rank"] <= D.CAP_A)].copy()
        by_code = {c: np.sort(g["_i"].to_numpy())
                   for c, g in ok.groupby("code")}
        codes = ok["code"].to_numpy()
        idx = ok["_i"].to_numpy()
        streak = np.zeros(len(ok), dtype=int)
        for j in range(len(ok)):
            aset = set(int(x) for x in by_code[codes[j]])
            i, s_ = int(idx[j]), 1
            while (i - s_) in aset:
                s_ += 1
            streak[j] = s_
        ok["streak"] = streak
        for k in range(1, 6):
            rec("生产口径 ≥%d分且前%d名 连续≥%d天" % (thr, D.CAP_A, k),
                ok[ok["streak"] >= k], "W5")

    # ---- W3：不要求当天够格，窗口内 >= k 次就上 ----
    for thr in (97,):
        ok = d[d["score"] >= thr]
        by_code = {c: np.sort(g["_i"].to_numpy())
                   for c, g in ok.groupby("code")}
        base_rows = d[["code", "_i", "date", "y_up", "score"]]
        pool = set()
        for c, arr in by_code.items():
            for i in arr:
                for off in range(0, W):
                    pool.add((c, int(i) + off))
        pl = pd.DataFrame(list(pool), columns=["code", "_i"])
        m = pl.merge(base_rows, on=["code", "_i"], how="inner")
        cn = []
        for c, i in zip(m["code"].to_numpy(), m["_i"].to_numpy()):
            arr = by_code[c]
            cn.append(int(((arr > int(i) - W) & (arr <= int(i))).sum()))
        m["cnt"] = cn
        for k in range(1, W + 1):
            rec("近%d日内够格≥%d次（当天可不够格）" % (W, k),
                m[m["cnt"] >= k], "W3")

    # ---- W4：5 日平均预测值排名 ----
    dd = d.sort_values(["code", "_i"]).copy()
    dd["pm"] = (dd.groupby("code")["_p"]
                .transform(lambda s: s.rolling(W, min_periods=W).mean()))
    dd = dd[np.isfinite(dd["pm"])]
    dd = dd.sort_values(["date", "pm"], ascending=[True, False])
    dd["mrank"] = dd.groupby("date").cumcount() + 1
    for n in (1, 3, 5, 10):
        rec("近%d日均值 前%d名" % (W, n), dd[dd["mrank"] <= n], "W4")

    order = {"W5": 0, "W1": 1, "W1x": 2, "W2": 3, "W3": 4, "W4": 5}
    rows.sort(key=lambda r: (order[r["kind"]], -r["hit"]))
    print("%-34s%6s%6s%7s%10s%7s%7s%9s"
          % ("规则", "样本", "每天", "空仓天", "准确率", "±", "倍数", "每月命中"))
    last = None
    for r in rows:
        if r["kind"] != last:
            print("-" * 86)
            last = r["kind"]
        print("%-34s%6d%6.2f%7d%8.2f%%%7.2f%6.2fx%8.1f 只"
              % (r["label"], r["n"], r["per_day"], r["empty"],
                 100 * r["hit"], 100 * r["se"], r["lift"], r["per_month"]))

    # 消融跑完别覆盖生产那份（selftest 钉着 STREAK_PERF 和它一致），
    # WF_OUT 指到别的文件名
    dflt = OUT / ("window_grid.json" if a.adj == "fixed"
                  else f"window_grid_{a.adj}.json")
    out = Path(os.environ.get("WF_OUT", str(dflt)))
    payload = grid_payload(days, base, W, rows, all_dates, a.adj, adj_used,
                           n_st=len(st))
    if payload["adj_inconsistent"]:
        print("\n注意：板块校正的拟合区间 %s 和这次评估的区间 %s 重叠，"
              "下面的成绩含样本内增益（反事实见 --adj none）"
              % (payload["adj_fit_window"], payload["eval_window"]))
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n-> " + str(out))
    if a.save_adj:
        if not FINAL_ADJ:
            print("[!] 没有估出因子（--adj 不是 rolling/shrink，或者没有历史名额），"
                  "不落盘")
        else:
            f = ROOT / "state" / "breakout" / "board_adj.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(
                {"factors": {k: round(float(v), 4) for k, v in FINAL_ADJ.items()},
                 "mode": a.adj, "fit_months": ADJ_FIT_WINDOW,
                 "n_picks": int(sum(1 for _ in [])),
                 "made_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "note": ("用截至最后一个验证月的全部生产名额估的板块因子，"
                          "和 --adj %s 臂逐月用的是同一个估计量。"
                          "daily.py 读它；删掉这个文件就回到代码里写死的那组。"
                          % a.adj)},
                ensure_ascii=False, indent=1), encoding="utf-8")
            print("-> " + str(f) + "  " + json.dumps(
                {k: round(v, 3) for k, v in FINAL_ADJ.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
