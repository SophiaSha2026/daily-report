# 早盘系统归档（2026-09-27）

用户 2026-09-27 要求：北京时间每天早上 9 点半那部分整个取消，代码、报告、邮件、
前端全部归档，以后不再需要。「早盘系统」= 早盘选股（09:27:30 竞价强弱榜）+
参数自学（收盘后调早盘打分的参数）+ 学习会诊（参数自学末尾的 LLM 多路分析，
它也看起涨预测，但跑在参数自学里，一起归档）。

这里的东西**不会被执行**：`.github/workflows/` 之外的 yml GitHub 不认，
`src/` 之外的脚本没有任何入口调用。全部用 `git mv` 搬过来，历史都在。

## 停了什么（三层触发）

| 层 | 停法 | 在哪 |
|---|---|---|
| 本机计划任务 | `DailyReport-Local-Morning`、`DailyReport-Local-Learn` 注销；早已停用的 `DailyReport-TriggerAuction` / `TriggerPullback` 一起注销 | 定义在 `tools/setup_tasks.ps1`（本目录那份是归档前的原样）；注销前导出的 XML 在 `tools/scheduled_tasks/`（只在本机，gitignore：带用户 SID） |
| GitHub Actions | `auction.yml` `premarket.yml` `learn.yml` `intraday.yml` `refresh_sector.yml` 先在 GitHub 上 disable，再移进 `workflows/` | `workflows/` |
| Cloudflare Worker | 北京 07:30 派发 auction.yml 的 `30 23 * * *` 删掉，已 `wrangler deploy`，只剩 20:45 的晚间提醒 | `tools/external-trigger/`（在用的那份） |

## 搬了什么

| 类 | 原位置 -> 这里 |
|---|---|
| 早盘选股 | `src/premarket.py run_auction.py score.py ths_export.py tdx_export.py collect_llm.py selftest.py cfg.py refresh_sector.py`、`prompts/analyst.md`、`mailer.py` 里的竞价邮件那一半（本目录的 `src/mailer.py` 是归档前原样） |
| 参数自学 / 会诊 | `src/eval_daily.py`、`src/learn/`（整个包）、`selftest_learn.py selftest_train.py`、`prompts/eval_analyst.md prompts/council/`、`tools/council_query.py platform_evidence.py`、`docs/learning.md docs/council.md` |
| 产物和数据 | `out/`（竞价面板）、`out_learn/`（学习 / 会诊面板）、`data/2026-0x/auction_*.parquet`（竞价快照）、`data/labels/`、`data/train/`、`state/` 里的 learning_status / verdict_log / shadow_* / llm_eval / council、`state/claim|sent/auction_*`、`cache/universe*`、`cache/sector_map.parquet`、`cache/sina_industries.json` |
| 其它 | `config.yaml`（归档前的完整版，含 runtime / universe / screen / scoring / output / llm / learning 各段）、`tools/yield_check.py trigger_auction.cmd build_manual.py build_devprocess.py`、`src/refresh_meta.py`（归档前那版还刷行业板块表）、旧 `README.md` `HANDOVER.md`、`CLAUDE_morning.md`（CLAUDE.md 里早盘专属的硬约束和领域知识） |

`cache/hist_daily.parquet` / `hist_auction*.parquet`（学习线回填用的原始缓存，
gitignore、共 143MB，只在本机）当晚第二轮清理时也搬进了本目录的 `cache/`，
`state/lock/progress_learn|morning.json` 搬进 `state/lock/`，都没删。

### 第二轮清理（2026-09-27 晚）

- `src/datasource.py` 里只给早盘用的函数删了：全市场快照 `spot_all` / `_turnover`，
  三路日线 `daily_hist` / `daily_hist_many` / `daily_hist_em|tx|sina` 和东财熔断
  （`_em_*`、`hist_source_stats`），次新判定 `is_new_listing`。盘前候选池
  （`premarket.py`）要它们，恢复时从提交 `20fe1ba` 的 `src/datasource.py` 拷回去。
- 冒烟测试 `src/smoke_test.py` + `.github/workflows/smoke_test.yml` 搬到本目录
  `src/` 和 `workflows/`：它测的是早盘在 runner 上要用的源（全市场快照、三路日线、
  竞价窗口吞吐）。晚间两条线云端算不了，本机的数据源体检是控制台「检查数据源」
  （`tools/probe.py`，这次改成测晚间线实际连的主机）。

## 留下的共用部分（没归档）

`datasource.py`（删掉了早盘专用的几个函数，见上）、`mailer.py`（发信底层 + 告警）、
`localenv.py`、`local_run.py`、控制台 `gui/`、`refresh_meta.py`（只刷代码表了）、`tools/probe.py`、
Pages 发布（改由 `.github/workflows/pages.yml` 推送触发，以前是 auction.yml 顺手发）。

早盘归档时顺手接过来的两件事（以前是早盘系统顺带在做、别的线在用）：
- Pages 发布：以前只有 auction.yml 部署 Pages，晚间面板要等第二天早上才上线。
- `state/regime_daily.jsonl` 和 `state/breakout/truth.json`：以前只有学习会诊在写，
  起涨预测邮件里的「近期基准」和控制台的清单图读它们。现在 `local_run.flow_breakout`
  每天写（`truth_and_regime`）。

`state/breakout/overrides.json`（会诊批准过的起涨常量）照常生效，只是不再有
「批准」这条路，要改就手改。

## 真要恢复

1. `git mv` 回原位置（上表），`archive/morning/config.yaml` 里早盘那几段并回 `config.yaml`，
   `mailer.py` 的竞价邮件那一半从本目录的 `src/mailer.py` 拷回去，`ths_export.py`
   现在从 `panel_style` import 样式（抽出去的那两段），不用再拷。
2. `local_run.py` 的 `flow_morning / flow_learn / local_commentary` 和 FLOWS 里那两行
   从 git 历史（2026-09-27 这次提交之前）找回来；控制台 `gui/` 同理。
3. 计划任务：`archive/morning/tools/setup_tasks.ps1` 里的 Morning / Learn 定义抄回去重跑。
4. workflows 移回 `.github/workflows/` 后在 GitHub 上 `gh workflow enable`；
   Worker 加回 `30 23 * * *` 并 deploy。三层都要回来，缺一层就是教训 24 反过来。
5. `datasource.py` 的早盘函数和冒烟测试按上面「第二轮清理」取回。
6. 跑 `selftest.py selftest_learn.py selftest_train.py`（也在本目录）。
