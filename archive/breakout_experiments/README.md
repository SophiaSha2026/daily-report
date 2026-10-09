# 起涨预测的早期实验（归档，不执行）

2026-10-08 从 `src/breakout/` 和 `out_breakout/` 搬过来。结论都在 `docs/breakout_log.md`，
这里只留可复查的原件，没有任何代码 import 或读它们。

| 文件 | 实验 | 结论去向 |
|---|---|---|
| exp_target.py | 实验 2：训练目标 y_t0 改 y_up | 已上线（label / validate） |
| exp_widen.py | 实验 4：放宽特征筛选阈值 | 已上线（fselect.IC_MIN） |
| exp_precision.py / out/precision_grid.json | 实验 5、7：门槛制、清单长度 | 已上线（SCORE_MIN / CAP_A） |
| exp_rank.py / out/rank_analysis.json | 实验 6：分数和名次的含义 | 写进 export 文案 |
| exp_persist.py / out/persist_grid.json | 实验 7：连续确认天数 | 只排序，不当门槛 |
| out/window_grid_exp9_ablation.json、exp10_*.json | 实验 9、10：股东户数偷看修复、成交量组消融 | 已上线 |
| out/window_grid_nomkt.json | 实验 14：全市场共同列消融 | 留着那 4 列 |
| out/window_grid_rolling.json | 实验 13：板块系数滚动臂（各板块各信各的） | 被收缩估计取代，update_perf 不再认它 |

这些脚本用的是 2025-03..12 那段旧验证集的协议。2026-10-07 起新实验一律走
`src/breakout/evalkit.py`（开发集 2024-06..2025-12、确认集每季度只看一次、预登记）。
直接运行它们需要先把文件放回 `src/breakout/`（它们 import 同目录的 arena / validate / model）。
