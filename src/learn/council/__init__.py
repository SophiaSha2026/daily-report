"""
学习会诊（council）：每次学习更新时，多路 LLM 并行深挖「预测和实际差在哪」。

设计见 docs/council.md。包结构：

    schemas.py      枚举 + JSON schema（CLI --json-schema 用）+ 脏输出改写
    evidence.py     证据包：两条线的预测 vs 真值，全部由代码算
    agents.py       起 `claude -p` 进程：并行、超时、stream-json 过程记录
    experiments.py  提案的自动实验（确定性代码），结果写回台账
    run.py          编排 + 台账 + latest.json
    panel.py        out_learn/council.html

三条边界（和 llm_local 一样）：不排序不改参数；失败不阻断；枚举固定。
"""
