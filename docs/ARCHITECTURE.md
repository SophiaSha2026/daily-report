# 仓库架构总览（2026-10-10）

一页说清楚：入口 → 编排 → 两条线 → 数据 → 产物 → 发布。改代码前读 `CLAUDE.md`（硬约束、教训），
日常操作读 `OPERATIONS.md`，这页只讲结构。时间一律北京时间。

## 1. 它是什么

A 股晚间选股流水线。每个交易日收盘后在**本机**跑两条互不相干的线，各发一封邮件，面板推到
GitHub Pages。云端只做托底提醒和面板发布，不算数据。

| 线 | 性质 | 发信 | 清单 |
|---|---|---|---|
| 起涨预测 | LightGBM 模型打分 + 规则 | 17:00 后，最晚次日 08:30 | A：分数 ≥97 且当天前 10（可以为空）；B：A 池里涨过一波后见顶 |
| 长期调整突破 | 纯规则 | 17:58 准点，没开机就开机后补 | A：横盘 → 首阳 → 缩量调整 → 二次进攻当天；B：走完前三步的 |

## 2. 入口（人从哪里进来）

| 入口 | 是什么 | 调谁 |
|---|---|---|
| 桌面「A股流水线」 | `tools/gui.cmd` → `python src/gui/__main__.py`，端口 8765 | 控制台 |
| 控制台首页 `/` | 一屏：两条线各一张卡片（状态词 / 上次跑通 / 目标日 / 清单只数 / 一个按钮）、查一只股票、一行系统状态（下次开机、插电、推送）、一张清单命中小图 | `gui/simple.py` |
| 控制台详细页 `/full` | 左栏按钮分「跑 / 查 / 修」三组（`gui/jobs.ACTIONS` 的 group），右边一个「状态」页：当前任务输出 + 折叠的排期 / 自动运行日志 / 配置；两个面板是顶栏链接 | `gui/ui.py` |
| 命令行 | `python src/local_run.py --flow breakout|pullback [--dry]`、`tools/predict_one.py <代码>`、`tools/refit_chain.py`、`tools/e2e_check.py`、`tools/audit_repo.py` | 同控制台按钮 |
| 计划任务 ×3 | `DailyReport-Local-Evening / -Pullback / -Sync`，`tools/setup_tasks.ps1` 唯一定义，按 UTC 锚定 | `tools/headless.vbs` → `tools/run_local.cmd` → `local_run.py --if-needed` |
| 云端 | `pages.yml`（推送后发布面板）、`evening_check.yml`（20:30 查没发就提醒）、Cloudflare Worker（20:45 再敲一次） | 只读 origin 上的标记 |

控制台的 HTTP 接口（`gui/server.py`，标准库，零依赖）：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/`、`/full` | 两个页面（HTML/CSS/JS 都在 Python 字符串里，没有构建步骤） |
| GET | `/api/simple`、`/api/status`、`/api/actions`、`/api/job/<id>`、`/api/logfile`、`/api/config` | 首页快照、详细状态、按钮表、任务输出、自动运行日志、config.yaml |
| GET | `/panel/<key>`、`/panel/stamp*` | 两个面板和刷新戳 |
| GET | `/api/predict?code=` | 单只股票预测（起子进程跑 `tools/predict_one.py --json`） |
| POST（要 token） | `/api/run`、`/api/stop`、`/api/task`、`/api/push` | 起 / 停一个按钮任务、启停计划任务、重试推送 |

两道防护：Host 白名单（防 DNS rebinding）、写操作 token（页面 URL 里带）。

## 3. 编排层 `src/local_run.py`

所有入口（按钮、计划任务、命令行）都走它。一条线一次运行：

1. 判定：目标日 = 最近一个已收盘交易日（`bjtime.last_closed`，15:05 后算当天）；
   `--if-needed` 时过五道闸（周末 / 上线日之前 / 已在跑 / 目标日跑完且发过信 / 不在开跑窗口）。
2. 进程锁 `state/lock/<线>.json`；跑超 `MAX_RUN` 的旧实例被接手。
3. 每步子进程有时限（`STEP_LIMIT`），整条有总时限；输出透传到 `tools/local_flow_<线>.log`，
   细进度写 `state/lock/progress_<线>.json`（控制台进度条读它）。
4. git：pull / commit / push 全在 `git_lock` 里；推送失败自愈，状态落 `state/push_status.json`。
5. 标记：`state/claim/`（本机接管）、`state/sent/`（真发了信才写，认 `mail_sent.json`）。

### 起涨预测一次运行（`flow_breakout`）

```
同步仓库 → 补当天日线（backfill --stage update，腾讯快照追加一根，除权票新浪重拉）
        → daily_ready 闸（补到目标日才继续；补不齐超 50 只且没到次日 07:00 就不出清单）
        → 重算特征表（build.py，约 13 分钟）
        → [模型到期那天] 月度重估链 tools/refit_chain.py
        → 打分出清单（daily.py --stage scan：加载或重训模型 → 验收 → 打分 → 风险剔除 → A/B 清单）
        → 真值和市场环境（truth / regime，fail-open）
        → 面板 + 邮件（daily.py --stage send）→ 推送
```

### 长期调整突破一次运行（`flow_pullback`）

```
同步 → 等日线补到目标日（起涨预测在补就等它，没人补就自己补）
     → 扫描（pullback.py --stage scan，几秒）→ 等到 17:58 → 面板 + 邮件 → 推送
```

## 4. 起涨预测内部（`src/breakout/`）

| 阶段 | 模块 | 输入 → 输出 |
|---|---|---|
| 日线 | `backfill.py` | 新浪（主）/ 腾讯（兜底）→ `data/breakout/daily.parquet`（三年全市场，gitignore） |
| 筹码 | `chips.py` | 日线 → 自算筹码分布（六项一致性检验） |
| 标签 | `label.py` | 20 根 K 线内最高价涨超 50%（和特征物理隔离） |
| 特征 | `features.py` | 六组指标 → 当日全市场排名 → 中性化 → 组内正交 |
| 组装 | `build.py` | → `data/breakout/train.parquet`（2.6 GB，gitignore） |
| 筛选 | `fselect.py` | 四道筛（相关性、稳定性、相关剪枝、L1 只记录） |
| 模型 | `model.py` | LightGBM，按月等权；L0 线性 / L2 序列只做对比 |
| 选票 | `validate.pick` | 预测值 × 板块系数 → 0~100 分 → 剔 ST / 减持 / 解禁 / 增发 → ≥97 分取前 10（**唯一实现**，生产和回测同一份） |
| 每日 | `daily.py` | 重训（30 天 / 指纹变 / 权重口径变）+ 验收 `accept_model` + 打分 + 漂移告警 `drift_note` |
| 真值 | `truth.py`、`buyable.py` | 历史清单之后涨没涨（收盘口径 + 买得到口径） |
| 评估 | `evalkit.py` | 新实验的协议：开发集 2024-06..2025-12、确认集每季度一次、预登记、过线标准 |
| 成绩链 | `exp_window.py`、`exp_calib.py` → `tools/update_perf.py` | 邮件里印的常量从这里来（`tools/refit_chain.py` 串起来） |

上面四个文件 `features / chips / label / build` 的源码进**模型指纹**，改一行就重训；更正只用 `#` 注释。

## 5. 长期调整突破内部

`pullback.py`（规则和每个阈值的来历在 docstring，阈值在 `config.yaml`）→ `find_events`
（回测和生产同一个函数）→ `rank`；`corp_events.py` 查减持 / 定增（两条线共用减持判据）；
`pullback_export.py` 面板和邮件；`pullback_backtest.py` 三年回看。

## 6. 数据和产物（谁写、谁读）

| 路径 | 写者 | 读者 | 入库 |
|---|---|---|---|
| `data/breakout/daily.parquet` | backfill（两条线都会去补，写锁 `state/lock/daily_update.json`） | 两条线、truth、regime、buyable | 否 |
| `data/breakout/train.parquet`、`holders.parquet` | build / backfill | daily、实验 | 否 |
| `data/breakout/raw/` | 回填分片、实验缓存（6.6 GB，其中 5.7 GB 是 2019 扩展表，可删） | 实验 | 否 |
| `data/breakout/YYYY-MM/breakout_<日>.parquet` | daily（每日清单 A） | truth、连续天数 | 是 |
| `data/pullback/YYYY-MM/pullback[_b]_<日>.parquet` | pullback | 面板日期下拉 | 是 |
| `out_breakout/`、`out_pullback/` | 两条线当日产物（panel.html、run_meta、清单、mail_sent、同花顺 txt） | 控制台、build_site、云端 | 是 |
| `out_breakout/window_grid.json`、`score_calibration.json`、`board_hit.json` | refit_chain | export（邮件常量防呆）、selftest | 是 |
| `state/breakout/` | 模型（model.txt / model.json）、板块系数、overrides、验收状态、打分统计、A 池、日线补数账 | daily、控制台 | 是 |
| `state/sent/ claim/ alert/` | local_run | 云端 evening_check、控制台 | 是 |
| `state/regime_daily.jsonl`、`state/breakout/truth.json` | local_run 每天发信前 | 邮件「近期基准」、控制台小图 | 前者是，后者否 |
| `state/lock/`、`state/push_status.json`、`state/trade_dates.json` | 运行态 | 控制台 | 否 |
| `cache/codes.csv`、`st_codes.json` | 手动刷新 | 回填、回测剔 ST | 是 |
| `_site/` | `build_site.py` | pages.yml 发布 | 否 |

## 7. 发布

本机推送 `out_breakout/ out_pullback/` 的面板 → `pages.yml` 跑 `build_site.py` 打包 `_site/`
（入口 index + 两个面板 + 两个刷新戳）→ GitHub Pages。面板里的脚本轮询刷新戳绕过 CDN 缓存。

## 8. 质量门

| 门 | 命令 | 管什么 |
|---|---|---|
| 三条离线自测 | `src/selftest_breakout.py`、`selftest_gui.py`、`selftest_pullback.py` | 不变量和接线（含 AST 钉子），二十秒级，产物只写临时目录 |
| 静态 | `python -m pyflakes src tools` | 零输出 |
| 审计 | `tools/audit_repo.py` | 死代码、重复实现、文档和接线漂移，零输出 |
| 端到端 | `tools/e2e_check.py` | 真起子进程把两条线试跑一遍（28 分钟，不发信不推送） |
| 数据源 | `tools/probe.py` | 五个主机 + 依赖 + SMTP |

## 9. 归档（不执行）

`archive/morning/`（早盘系统，2026-09-27）、`archive/pullback_v1/`（回调形态 v1）、
`archive/breakout_experiments/`（起涨预测实验 2~14 的脚本和产物）。

## 10. 文档

`CLAUDE.md`（硬约束、文件地图、教训 1~44）、`OPERATIONS.md`（运维）、`README.md`（新机器装）、
`docs/breakout_log.md`（实验 1~19 全记录）、`docs/breakout_plan_2026-10.md`（稳定化计划）、
`docs/plan_2026-10-10.md`（本轮九项计划）、`docs/breakout_design.md`（最初设计，历史）。
