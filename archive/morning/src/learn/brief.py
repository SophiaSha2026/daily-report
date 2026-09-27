"""
归因输入里「最差 / 最好」两组样本的挑法。

为什么单独一个模块（2026-09-16 审计 F3-4）
------------------------------------------
eval_daily.stage_brief 和学习会诊的证据包各写过一份 head/tail，两份都错，
而且错法不同：

  * 两组**没有互斥**，池子小于 40 只时 `head(20)` 和 `tail(len//2)` 必然
    相交。09-16 那天池 12 只，003006 / 002084 / 601919 / 300313 同时出现在
    worst 和 best 里；17 个在线日有 8 天两组有交集。
  * 两组**没有收益符号约束**。worst 里躺着 ytil=+0.83 的票，best 里躺着
    ytil=-0.96 的票，而提纲告诉模型 worst 是「预测对了但没涨」。
    11 天的 best 里有 ytil<0 的票。
  * `rank` 写的是 `enumerate` 出来的序号，也就是**按 ytil 排完之后**的位次，
    不是当日分数名次。09-16 的 worst 里 301520 标 rank=1（真实分数名次 6），
    688499 标 rank=6（真实名次 2）。

这三件事凑在一起，LLM 拿到的是自相矛盾的输入，而它的结论经 day_regime ->
regime_weight -> 优化器日权重，是会改变拟合结果的。所以挑法只留一份，
两个调用方共用；`selftest_learn.check_brief_pick` 用 AST 钉住没人再自己写。
"""
from __future__ import annotations

import pandas as pd


def pick_worst_best(ok: pd.DataFrame, n_w: int, n_b: int,
                    head_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame]:
    """(带 rank 的全表, worst, best)。

    worst = 分数前 head_n 里**真的亏了**的那几只，按亏得最多排；
    best  = 分数后 50% 里**真的涨了**的那几只，按涨得最多排，且不与 worst 重复。
    rank 一律是当日按分数降序的位次，和 worst/best 的内部顺序无关。

    ok 必须是已经过硬性排除的那一批。两组都可能为空（小池子、或者当天
    全池同向），空是正确答案，不要用「凑够 n 只」去补——补出来的样本
    和提纲说的事实不符，比没有更糟。
    """
    ok = ok.sort_values("sc", ascending=False).reset_index(drop=True)
    ok = ok.assign(rank=range(1, len(ok) + 1))
    head = ok.head(head_n)
    # tail 至少一行：池子只有 1~3 只的日子照样要有「后 50%」这个概念
    tail = ok.tail(max(len(ok) // 2, 1))
    worst = head[head["ytil"] < 0].nsmallest(n_w, "ytil")
    best = tail[(tail["ytil"] > 0)
                & ~tail["code"].isin(worst["code"])].nlargest(n_b, "ytil")
    return ok, worst, best
