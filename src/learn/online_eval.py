"""
在线真值天的成绩：**当天真发出去的那张榜** vs **按当前参数回放的榜**。

为什么要分成两个数（2026-09-16 审计 F3-13）
-------------------------------------------
以前 learning_status.json 的 `daily` 是用**当前生效的 θ** 把历史快照重新
打一遍分算出来的，界面上却写着「真实榜单（实盘口径）」。两者对不上，
而且不用等接受第一次参数变更就已经对不上：

  * 08-24 ~ 09-02 的快照是老规则打的分（涨幅无下限、量比 2.5~50），
    重放用现行规则，每天有 21~97 行剔除结论不一致；
  * 09-15 审计把 `min_auc_amount_wan` 补进 hard_reject 之后，重放会追溯
    剔掉 09-14 的 603082（竞价额 243 万、当天真实榜第 3）等 6 只；
  * 重放不看 `output.min_score`，09-14 实际只发 9 只，重放照取 10 只。

17 个在线日里 12 天前 10 不是同一批票，日均超额重放 -0.681%、
真实 -1.029%。GUI 上那句「前 10 每天平均比大盘少赚 0.68%」低估了三分之一。

所以口径拆开：`daily_sent` 只读快照自带的 `score` / `score_raw` /
`rejected`（那是当天早上真写进邮件的那份），任何后续的参数或规则改动都
改不动它；`daily_replay` 是「按今天的参数重新打一遍」，它有它的用处
（看一次提案会把历史榜改成什么样），但它不叫实盘成绩。

汇报口径（审计 F2-10）
----------------------
对外的「前 10 超额」用未缩尾的 `y_raw` 再扣双边成本。缩尾（winsor_q=0.01）
是给目标函数防梯度翻转用的，不该出现在给人看的数字里：在线 17 天 168 只
选中票有 10 只被截，09-11 汇报 -5.826% 实为 -6.057%。
`cost_bp: 13` 是单边，config 注释写着「只在汇报里扣」，在这之前全仓
`grep cost_bp` 零命中，那句注释是个未兑现的承诺。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def excess(df: pd.DataFrame, cost: float = 0.0) -> pd.Series:
    """汇报口径的超额收益列：未缩尾的 y_raw 减双边成本。

    旧训练表没有 y_raw 列时退回 y（缩尾值），结果仍然可用，只是那一段
    汇报仍是缩尾口径——退回是为了不崩，不是为了对。
    """
    col = df["y_raw"] if "y_raw" in df.columns else df["y"]
    return col.astype(float) - float(cost)


def admitted_mask(df: pd.DataFrame) -> pd.Series:
    """快照自带的硬性排除结论。`rejected` 空 = 当天真的过了准入。"""
    if "rejected" not in df.columns:
        return pd.Series(True, index=df.index)
    r = df["rejected"]
    return r.isna() | (r.astype("string").fillna("").str.strip() == "")


def sent_score(df: pd.DataFrame) -> pd.Series:
    """排序用的分。score_raw 是未取整的，2026-09-16 起的快照才有；
    更早的只有 round(raw, 1)，并列的票只能靠候选池原顺序定序。"""
    if "score_raw" in df.columns:
        return df["score_raw"].astype(float).fillna(df["score"].astype(float))
    return df["score"].astype(float)


def sent_mask(df: pd.DataFrame, min_score: float) -> pd.Series:
    """当天真会进邮件的行：过准入 且 round(score,1) >= output.min_score。

    round 到 0.1 不能省：score.py 存的就是取整后的分，44.96 在生产是过线的。
    """
    s = df["score"].astype(float) if "score" in df.columns else sent_score(df)
    return admitted_mask(df) & (s.round(1) >= float(min_score))


def _row(day, pool: pd.DataFrame, top: pd.DataFrame, sc: str) -> dict:
    from learn.model_select import spearman
    return {
        "date": day,
        "ic": (spearman(pool[sc].to_numpy(float), pool["ytil"].to_numpy(float))
               if len(pool) >= 3 else float("nan")),
        # 一只都发不出去的日子记 None，别拿 nan 冒充 0
        # （gui/status 和学习面板都按 None 过滤）
        "top_excess": float(top["_ex"].mean()) if len(top) else None,
        "pool": int(len(pool)),
        "sent_n": int(len(top)),
    }


def daily_sent(df: pd.DataFrame, top_k: int, min_score: float,
               cost: float = 0.0) -> list[dict]:
    """逐日「当天真发出去的那张榜」的成绩。只读快照自带的列。"""
    d = df.assign(_sc=sent_score(df).to_numpy(float),
                  _ex=excess(df, cost).to_numpy(float),
                  _adm=admitted_mask(df).to_numpy(bool),
                  _snt=sent_mask(df, min_score).to_numpy(bool))
    out = []
    for day, g in d.groupby("date", sort=True):
        # IC 在过准入全池上算，45 分线是区间截断，套上去会机械压低相关性
        pool = g[g["_adm"]]
        top = g[g["_snt"]].sort_values("_sc", ascending=False,
                                       kind="stable").head(int(top_k))
        out.append(_row(day, pool, top, "_sc"))
    return out


def daily_replay(df: pd.DataFrame, c: dict, top_k: int,
                 cost: float = 0.0) -> list[dict]:
    """逐日「按当前参数重新打分」的榜。和 daily_sent 并排看才有意义。"""
    from learn import optimize as OPT, vscore
    s, rej = vscore.score_df(df, c)
    d = df.assign(_sc=np.asarray(s, float), _rej=np.asarray(rej, bool),
                  _ex=excess(df, cost).to_numpy(float))
    out = []
    for day, g in d.groupby("date", sort=True):
        o = OPT.production_order(g["_sc"].to_numpy(float),
                                 g["_rej"].to_numpy(bool), c, int(top_k))
        out.append(_row(day, g[~g["_rej"]], g.iloc[o], "_sc"))
    return out
