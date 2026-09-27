# 回调形态 v1（2026-08-26 ~ 2026-09-27）

形态线（内部代号 pullback）的第一版规则，2026-09-27 被「长期调整突破」替换。

| 段 | v1 条件 |
|---|---|
| 启动日 S | 涨幅 ≥5%；成交量 ≥ 前一日 1.5 倍；换手 5%~10% |
| 调整期 | S+1 到 T-1，1~6 个交易日；缩量；期间最低价 ≥ S 日最低价 |
| 执行日 T | 涨幅 ≥5%；成交量 ≥ 前一日 1.5 倍；换手 5%~10% |

v1 的每日报告 2026-09-12 就取消了（只剩控制台手动入口）。数据来源是扫描当时
拉的腾讯批量行情 + 逐只日线，和新版不同（新版读本机三年日线表，逐只不联网）。

| 这里 | 是什么 |
|---|---|
| `src/pullback.py` `pullback_export.py` `pullback_backtest.py` `selftest_pullback.py` | v1 代码原样 |
| `out_pullback/` | v1 最后一次的产物（2026-09-14） |
| `data/2026-0x/pullback_*.parquet` | v1 每日结果（2026-08-26 ~ 2026-09-14） |
| `workflows/pullback.yml` | 云端手动扫描入口（新版的日线在本机，云端跑不了） |
| `tools/trigger_pullback.cmd` | 更早的本机派发云端的脚本 |
| `prompts/pullback_analyst.md` | v1 的 LLM 文案指令（新版不用 LLM） |

v1 的领域知识（「涨停或涨幅≥5%」只判后者、历史换手率反推、多个启动日取最近、
每天约 1 只的抽样实测）在 `archive/morning/CLAUDE_morning.md` 末尾。
