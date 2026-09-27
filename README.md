# A股晚间流水线

每个交易日收盘后两封邮件（北京时间）：

| 清单 | 时间 | 是什么 |
|---|---|---|
| 起涨预测 | 17:00 后 | 模型打分，技术形态接近起涨的股票（清单 A）+ 可能见顶的（清单 B） |
| 长期调整突破 | 17:58 | 清单 A：横盘 3 个月以上 -> 倍量大阳线（首阳）-> 缩量调整 -> 再一根倍量大阳线站上首阳最高价，这一天推荐；清单 B：走完前三步、还在等二次进攻的。两份都剔除近期股东减持、价格没定的定增，没有就发空榜 |

在线面板：https://sophiasha2026.github.io/daily-report/

> 量化筛选工具，非投资建议。

- **日常怎么用、没收到邮件怎么办**：`OPERATIONS.md`
- **改代码之前**：`CLAUDE.md`（架构、文件地图、硬约束、历史教训）
- **长期调整突破的规则和每个阈值的来历**：`src/pullback.py` 开头，阈值在 `config.yaml`
- **早盘系统（09:27:30 竞价强弱榜 + 参数自学）**：2026-09-27 整体归档，见 `archive/morning/RESTORE.md`

## 跑在哪

本机为主：控制台 `tools/gui.cmd`（桌面「A股流水线」）是唯一入口，自动跑靠
Windows 计划任务 `DailyReport-Local-*`（定义在 `tools/setup_tasks.ps1`）。
两条线的数据（三年全市场日线、特征表）都在本机，云端算不了；本机那天没跑，
北京 20:30 云端发一封提醒，开机后自动补，最晚到次日 08:30。

GitHub 上只做三件事：存代码和产物、推送后发布 Pages 面板（`pages.yml`）、
20:30 的晚间提醒（`evening_check.yml`，Cloudflare Worker 20:45 再敲一次）。

## 第一次在新机器上装

```bash
pip install -r requirements.txt -r requirements-breakout.txt
```

1. 复制 `tools/local.env.example` 为 `tools/local.env`，填 Gmail 应用专用密码等发信项
2. 首次回填三年日线（约 70 分钟）：控制台「辅助工具 -> 全量回填」
3. 注册计划任务：`powershell -ExecutionPolicy Bypass -File tools\setup_tasks.ps1`
4. 跑一遍自测：`python src/selftest_pullback.py`、`python src/selftest_gui.py`、
   `python src/selftest_breakout.py`，再 `python tools/e2e_check.py`
