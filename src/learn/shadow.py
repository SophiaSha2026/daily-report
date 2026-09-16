"""
影子排序器：秩归一特征上的稳健线性模型（擂台胜者 RankHuber 的转正通道）。

为什么存在（2026-09-03 的测量结论）
----------------------------------
四次点火 + 全量擂台给出了清晰的三段事实：

  1. 手写六维打分器的 9 个可调参数已近局部最优：箱内梯度 ≤0.03/σ，
     继续调旋钮可挖的改善不足 0.1%。
  2. 天花板在 2~3 倍外：同样的特征，RankHuber 样本外 IC 0.117 / 前10
     日超额 +1.40%，基线 0.022 / +0.69%。缺口在**函数形式**，不在参数。
  3. 擂台胜者恰好是**线性模型**——每个系数可打印、可归因、可用人话讨论。
     「不许黑箱排序」的红线拦的是不可解释，不拦线性回归。

于是最优架构是让老师转正——但**不直接转**：

    影子模式    每天和生产打分器并排跑，只记账不发信不排产。
    真值积累    在线快照（真 T1/T2/T3）上逐日记录双方 IC 和前10超额。
    人工切换    影子在真值上显著领先、且积累够天数后，作为**提案**
                摆到用户面前。切不切换是用户的决定，不是机器的。

确定性保证：推理 = 当日截面秩 × 固定系数向量。系数存 JSON、进 git，
给定系数文件，任何人可逐位复现当天排名。和 score.py 的可复现性同级。

记账纪律（2026-09-16 修正，审计 G1/F2-2/F3-1）
--------------------------------------------
调用顺序必须是「先用**上一次落盘的**模型给新到的在线日记账，再 refit」：
每个被比的日子用的正是那天早上产出 out/shadow.json 的那份系数，
和生产口径一致（教训 30）。顺序反了 train_end 就等于今天，
`daily_compare` 的 `date > train_end` 过滤恒为空集——2026-09-15 起
两次学习的影子对比都是 0 天，而面板另一处按 online_days 写着「17/30」，
同一页两个数互相矛盾，用户看到的是「还在积累」而不是「坏了」（教训 16）。

对比结果落在 `state/shadow_compare.json` 这本账上，按日期先到先得：
已经记过的天不许被后来 refit 的模型重算覆盖，否则账目随每次训练漂移。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = ROOT / "state" / "shadow_model.json"
PROPOSAL = ROOT / "state" / "shadow_proposal.json"
LEDGER = ROOT / "state" / "shadow_compare.json"   # 逐日对比账本，先到先得

# 和擂台同一张特征表（learn/model_select.py::FEATURES），刻意不另起炉灶：
# 影子的正当性来自「它就是擂台上赢的那个东西」，特征一换比较就失效了。


def fit(df: pd.DataFrame) -> dict | None:
    """在带 ytil 的训练表上拟合 RankHuber，落系数文件。

    失败返回 None 不抛——影子是研究性组件，任何失败都不能影响主流程。
    """
    try:
        from sklearn.linear_model import HuberRegressor
        from learn.model_select import FEATURES, prep_features, rank_norm
        d = prep_features(df)
        X = rank_norm(d[FEATURES], d["date"]).fillna(0.0)
        y = d["ytil"].to_numpy(float)
        m = HuberRegressor(alpha=1e-3, max_iter=500).fit(X, y)
        doc = {
            "kind": "RankHuber",
            "fitted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "train_days": int(d["date"].nunique()),
            "train_rows": int(len(d)),
            "train_end": str(d["date"].max()),   # 对比只许用这之后的在线日
            "features": list(FEATURES),
            "coef": {f: float(c) for f, c in zip(FEATURES, m.coef_)},
            "intercept": float(m.intercept_),
        }
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        import os
        os.replace(tmp, MODEL)
        top = sorted(doc["coef"].items(), key=lambda x: -abs(x[1]))[:5]
        log.info("影子模型已拟合（%d 天）。最大五个系数：%s",
                 doc["train_days"],
                 ", ".join(f"{k}={v:+.3f}" for k, v in top))
        return doc
    except Exception as e:  # noqa: BLE001
        log.warning("影子模型拟合失败（不影响主流程）: %s", e)
        return None


def load() -> dict | None:
    try:
        return json.loads(MODEL.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def score(df: pd.DataFrame, model: dict | None = None) -> np.ndarray | None:
    """当日截面打分。纯线性：秩归一 × 系数。没有模型文件返回 None。"""
    model = model or load()
    if not model:
        return None
    try:
        from learn.model_select import prep_features, rank_norm
        d = prep_features(df)
        feats = model["features"]
        X = rank_norm(d[feats], d["date"]).fillna(0.0)
        w = np.array([model["coef"][f] for f in feats])
        return X.to_numpy(float) @ w + model["intercept"]
    except Exception as e:  # noqa: BLE001
        log.warning("影子打分失败: %s", e)
        return None


def daily_compare(df_online: pd.DataFrame, base_scores: np.ndarray,
                  base_rej: np.ndarray, top_k: int = 10,
                  min_score: float | None = None,
                  cost: float = 0.0) -> list[dict]:
    """在线真值天上的双榜逐日对比。喂给面板和切换提案。

    影子也套用同一份硬性排除（准入是政策层，对两个排序器一视同仁）。

    正式榜那一路还要再套 `min_score`（生产的 45 分线，score.py::rank），
    因为它就是**当天真发出去的那张清单**；影子分数不在 45 分刻度上，没有
    对应语义，所以保持前 top_k。两张榜长度可能不同，重合率的分母因此用
    较长那张，不能固定成 top_k（否则正式榜只有 5 只时重合率天然 ≤50%）。

    两边的超额都走汇报口径（未缩尾的 y_raw 再扣双边 cost），转正统计取的是
    配对差，常数成本本来就抵消，但逐日数字要能和面板别处并排看（F2-10）。
    注意这里的 base 是**按当前参数回放**的分，不是当天真发的那张榜
    （审计 F3-13）；面板上的标签要写清楚。
    """
    from learn.model_select import spearman
    from learn import online_eval as OE
    sh = score(df_online)
    if sh is None:
        return []
    # 影子在含在线日的表上拟合过，那些天是它的样本内，不能拿来给它算成绩。
    # 只比 train_end 之后的在线日。以前 15 个真值日里 7 个是样本内，
    # P(影子更好)=0.97 偏高。调用方必须**先比后 refit**，否则 train_end
    # 等于今天，这个过滤会把所有天都吃掉（见模块注释）。
    train_end = ""
    try:
        train_end = str(json.loads(MODEL.read_text(encoding="utf-8"))
                        .get("train_end", ""))
    except Exception:  # noqa: BLE001
        pass
    out = []
    d = df_online.assign(_b=np.where(base_rej, -np.inf, base_scores),
                         _s=np.where(base_rej, -np.inf, sh),
                         _ex=OE.excess(df_online, cost).to_numpy(float))
    if train_end:
        n0 = int(d["date"].nunique())
        d = d[d["date"] > train_end]
        if n0 and d.empty:
            # fail-open 的分支必须留下能被查询的信号（教训 16）：静默返回空
            # 表面上是「还在积累」，实际是 stage_learn 的调用顺序反了。
            log.warning("影子对比：%d 个在线日全部 <= train_end=%s（样本内），"
                        "对比为空；stage_learn 必须先 daily_compare 再 fit",
                        n0, train_end)
    for day, g in d.groupby("date"):
        ok = g[np.isfinite(g["_b"])]
        if len(ok) < 5:
            continue
        base_top = ok.nlargest(top_k, "_b")
        if min_score is not None:
            # round 到 0.1：和 score.py 存的 score 字段同口径，44.96 过线
            base_top = base_top[base_top["_b"].round(1) >= float(min_score)]
        shadow_top = ok.nlargest(top_k, "_s")
        row = {"date": day, "base_n": int(len(base_top)),
               "shadow_n": int(len(shadow_top))}
        for tag, col, top in (("base", "_b", base_top),
                              ("shadow", "_s", shadow_top)):
            row[f"{tag}_ic"] = spearman(ok[col].to_numpy(),
                                        ok["ytil"].to_numpy())
            row[f"{tag}_top_excess"] = (float(top["_ex"].mean())
                                        if len(top) else None)
        row["overlap"] = (len(set(base_top["code"]) & set(shadow_top["code"]))
                          / max(len(base_top), len(shadow_top), 1))
        out.append(row)
    return out


def record(rows: list[dict], ledger: Path | None = None) -> list[dict]:
    """把新出炉的样本外对比行并进账本，按日期去重，**先到先得**。

    已经记过的天不许被后来 refit 的模型重算覆盖：那一天的成绩是当天早上
    那份系数产出的，事后用见过这一天的模型重算就不是样本外了（教训 30）。
    同日重跑（--if-needed 反复重试、dry）第二次 daily_compare 返回空，
    record([]) 原样返回账本，幂等。
    """
    p = Path(ledger or LEDGER)
    try:
        old = json.loads(p.read_text(encoding="utf-8"))
        old = [r for r in old if isinstance(r, dict) and r.get("date")]
    except Exception:  # noqa: BLE001
        old = []
    seen = {r["date"] for r in old}
    merged = sorted(old + [r for r in rows
                           if isinstance(r, dict) and r.get("date")
                           and r["date"] not in seen],
                    key=lambda r: str(r["date"]))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1,
                                  default=str), encoding="utf-8")
        import os
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        log.warning("影子对比账本写入失败（不影响流程）: %s", e)
    return merged


def promotion_stat(cmp_: list[dict], min_days: int, p_req: float,
                   n_boot: int = 2000, seed: int = 11) -> dict:
    """转正证据：在线真值天上，(影子 − 正式) 前 10 超额的按天自助。

    按天不按票，和主线闸门 3 同一纪律。统计量用配对差的均值——
    「平均每天多赚多少」是能写进提案邮件、能用人话讨论的量。
    ready = 天数 ≥ min_days 且 P(均值 > 0) ≥ p_req。两个条件缺一不可：
    天数不够时 P 再高也只是运气。
    """
    pairs = [(x.get("shadow_top_excess"), x.get("base_top_excess"),
              x.get("shadow_ic"), x.get("base_ic")) for x in cmp_]
    d = np.array([s - b for s, b, _, _ in pairs
                  if s is not None and b is not None
                  and np.isfinite(s) and np.isfinite(b)], float)
    n = int(d.size)
    out = {"days": n, "min_days": int(min_days), "p_req": float(p_req),
           "need_days": max(0, int(min_days) - n),
           "p_better": None, "mean_diff": None, "wins": 0, "ic_wins": 0,
           "ready": False}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (int(n_boot), n))
    p = float((d[idx].mean(axis=1) > 0).mean())
    ic_wins = sum(1 for _, _, si, bi in pairs
                  if si is not None and bi is not None
                  and np.isfinite(si) and np.isfinite(bi) and si > bi)
    out.update(p_better=p, mean_diff=float(d.mean()),
               wins=int((d > 0).sum()), ic_wins=int(ic_wins),
               ready=bool(n >= int(min_days) and p >= float(p_req)))
    return out


def load_proposal() -> dict | None:
    try:
        return json.loads(PROPOSAL.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def maybe_propose(date: str, stat: dict, cmp_: list[dict], cfg: dict,
                  remind_days: int = 10) -> bool:
    """证据达标就发「切换提案」邮件并落 state/shadow_proposal.json。

    已发过的按 remind_days 间隔再提醒，不天天催。切换动作本身永远是
    人工的：这里不写任何参数，不动 score.py，只把证据摆到用户面前。

    **先发信、按结果落盘**（2026-09-16 审计 F2-5）。以前是先写
    last_sent=今天、times+1 再调 R.send，而 R.send 吞掉所有异常也不返回成败：
    SMTP 一挂，提案就被记成「已发」，接下来 remind_days=10 天内 :202 直接
    return False 不重发，日志里 eval_daily 还打「影子转正提案已发出」，
    面板第 4 步显示「等你决定 · 共 1 次」，用户去邮箱找一封不存在的信。
    和教训 13（参数变更邮件永远发不出去、四次点火没发现）同一个形状。
    失败时**不写 last_sent**、只写 send_failed，面板据此显示红字，
    次日照常重试（节流只认 last_sent）。
    """
    if not stat.get("ready"):
        return False
    prev = load_proposal()
    if prev and prev.get("last_sent"):
        # 只有真发出去过才节流。不能退回 prev["date"] 兜底：失败那次写了
        # date 没写 last_sent，拿 date 顶上去照样锁 10 天，等于没修。
        try:
            last = dt.date.fromisoformat(str(prev["last_sent"]))
            if (dt.date.fromisoformat(date) - last).days < int(remind_days):
                return False
        except Exception:  # noqa: BLE001
            pass
    from learn import report as R
    html = R.build_proposal_html(date, stat, cmp_)
    ok = R.send(date, html, cfg, subject=f"[提案] 影子排序器转正 · {date}")
    if ok:
        doc = {"date": (prev or {}).get("date") or date, "last_sent": date,
               "times": int((prev or {}).get("times", 0)) + 1, "stat": stat}
    else:
        # 失败必须留下一个能被界面查询的对象（教训 16），只写日志等于没写
        doc = {k: prev[k] for k in ("date", "last_sent", "times")
               if prev and k in prev}
        doc.update(send_failed=date, stat=stat)
    PROPOSAL.parent.mkdir(parents=True, exist_ok=True)
    PROPOSAL.write_text(json.dumps(doc, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    return bool(ok)
