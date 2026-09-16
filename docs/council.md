# 学习会诊（council）：每次学习更新时 LLM 深度参与的设计

2026-09-16 起。用户要求：每次学习系统更新/升级时，LLM（Opus 5，max 推理）
像审问一样全面分析「模型预测和实际结果差在哪、差多大、为什么、是波动还是
真差距、怎么改」，多路并行深挖，过程和可视化进控制台，输出走固定 schema，
结论能落地到模型。用户拍板（2026-09-16）：**半自动** —— LLM 出结构化提案，
系统自动跑对照实验，过闸门的进邮件和控制台等用户一键批准；学习白名单内的
参数沿用现有七道闸自动生效。规模：**多路并行**，5~6 个视角各一个进程，
再由一个主审汇总。

这一页是设计和接线说明。运行看 OPERATIONS.md，学习系统本身看 learning.md。

## 1. 边界（不许越过的）

- **LLM 不排序、不打分、不直接改参数。** 它的输出是 JSON，进三条可审计的路：
  日权重查表（已有）、提案 -> 实验 -> 闸门 -> 人批准、面板文案。
  给定落盘的 JSON，模型和清单完全确定。
- **失败不阻断。** 会诊挂了学习流程照常，面板显示「本次没跑成」。
- **枚举和 schema 固定。** CLI 用 `--json-schema` 强制结构，越界枚举改写成
  保守值，脏 JSON 丢弃不落盘。
- **实验是确定性代码跑的。** LLM 只提出要做什么实验，`experiments.py` 用
  现有走向前/消融/闸门去做，数字由代码给。
- **准入区间、score.py、特征代码不由机器改。** 特征增删这类提案标
  `needs_human`，附带实验结果等人做。

## 2. 每次会诊做什么

```
eval_daily --stage all
   label -> brief -> llm(归因) -> learn(七道闸) -> council(会诊, fail-open)

council：
   1. evidence.py    组证据包（纯代码，两条线）-> state/council/<date>/evidence.json
   2. agents.py      6 个视角并行，各起一个 `claude -p` 进程（Opus 5, --effort max,
                     --json-schema, --allowed-tools 白名单）-> <lens>.json + <lens>.trace.jsonl
   3. chair          主审读 6 份意见 + 证据包 -> verdict.json（差距在哪/多大/为什么/
                     波动还是真差距/改进清单）
   4. experiments.py 提案里能自动做的（阈值、消融、参数）排队跑，结果写回台账
   5. panel.py       out_learn/council.html：过程时间线 + 结论 + 图 + 提案台账
   6. 台账           state/council/proposals.jsonl；过闸门的进邮件；控制台一键批准
```

六个视角（每个一个进程，同一份证据包，各自的提纲和 schema）：

| 视角 | 问什么 | 允许的工具 |
|---|---|---|
| `gap_where` 差距定位 | 差距集中在哪：哪些天、板块、分数段、连续档、板块（科创/创业）、A/B 组 | Read, Bash(query) |
| `noise_or_real` 波动还是真差距 | 用二项/自助区间、同期全市场基准、多重比较，判断每个差距是抽样波动还是真实 | Read, Bash(query) |
| `features` 特征体检 | 重要性 vs 实际贡献、漂移、符号翻转、冗余；该增/删/改哪些 | Read, Bash(query) |
| `regime` 市场环境 | 这段时间市场发生了什么，环境能解释多少差距 | Read, WebSearch, WebFetch |
| `data_quality` 数据与流程 | 证据包里的异常：单位、缺失、时间对齐、停牌、口径不一致 | Read, Bash(query) |
| `improve` 改进提案 | 独立提出可检验的改进实验，按预期收益排序 | Read, Bash(query) |

主审 `chair`：只读 6 份意见和证据包，不再查数据，出最终裁决和提案合并去重。

`Bash(query)` = `Bash(python tools/council_query.py:*)`，一个只读查询工具，
LLM 想看更多切片时自己调，能做的事全在那个脚本里列着，改不了任何东西。

## 3. 证据包（evidence.json）

两条线各一段，全部由代码算，LLM 拿到的是同一份。

早盘（早盘选股 + 参数自学）：
- 最近 N 个在线真值日（默认全部）：每天 IC、前 10 超额、命中只数、池大小、
  当日中位涨幅、day_regime（归因）、闸门裁决
- 分数分档 vs 实际超额；六个打分维度高/低三分之一的超额差；A/B 组；
  涨幅段/量比段；被硬剔除的票 vs 过准入的票
- 逐日 worst/best（brief 的历史版本，从标签+快照重算）
- 回填 vs 在线的特征分布差（教训 30 的口径表）
- 影子榜对比、参数版本、theta 历史

起涨预测：
- 每一份历史清单 A 的**真值**（`src/breakout/truth.py`）：每只票之后 20 个交易日
  最高涨幅、是否 ≥ 50%、目前走了几天、当前回撤；满 20 天的算最终，没满的算
  「进行中」并标明
- 清单 A 实际命中率 vs 邮件里印的期望（STREAK_PERF/分数分档）vs **同期全市场
  基准**（同一起点、同一窗口，全市场随便买涨超 50% 的比例）；按连续档、
  分数段、板块拆
- 模型：训练日期、45 个特征的重要性、今天清单的特征分位 vs 训练正样本
- 清单 B 的命中（见顶之后是否真跌）
- 数据：特征表行数/日期、每天参与横截面的代码数、NaN 比例

## 4. 输出 schema（要点）

每个视角：`{lens, summary(≤200字), findings[{claim, evidence, magnitude, confidence}],
proposals[...]}`。

主审 verdict：
```
gap:        [{where, size, unit, expected, actual, ci_lo, ci_hi, n}]
why:        [{cause_enum, weight(0~1), evidence}]
noise_or_real: {verdict: 噪声|真差距|混合, p_real, reasoning}
proposals:  [{id, kind, target, change, rationale, expected_effect, test_plan,
              priority, auto_testable, needs_human}]
narrative:  ≤ 600 字给人读的结论
```

`kind` 枚举：`param`（学习白名单参数）、`threshold`（SCORE_MIN/CAP_A/BOARD_ADJ 等
起涨预测常量）、`feature_drop`、`feature_add`、`feature_modify`、`data_fix`、
`scope`（扩大/缩小候选范围）、`process`（流程/时间）。

`cause_enum`：`模型偏差`、`特征失效`、`市场环境`、`数据质量`、`样本不足`、
`口径不一致`、`未知`。

## 5. 提案怎么落地（半自动）

| kind | 自动实验 | 落地 |
|---|---|---|
| `param`（早盘白名单） | 走现有 `gate.evaluate` 七道闸（和优化器候选同一道） | 过闸自动写 learned.yaml（已有） |
| `threshold`（起涨常量） | 用缓存的走向前分数（`wf_scores.parquet`）重算 W5 表 | 过闸进「待批准」，控制台一键批准写 `state/breakout/overrides.json`，daily.py 读 |
| `feature_drop` | `exp_window.py` 的 `WF_DROP` 消融（约 10 分钟） | 同上，批准后进 overrides 的 drop 列表，重训 |
| `feature_add/modify` | 不能自动（要写代码） | `needs_human`，进台账等人做 |
| `data_fix/scope/process` | 不能自动 | `needs_human` |

「过闸」对起涨预测的定义：全部上榜准确率不降（差 ≥ −1 个标准误）且
样本数不少于原来的 80%，或连续 ≥2 天档提升 ≥ 2 个标准误。写在
`experiments.py`，数字由代码给。

台账 `state/council/proposals.jsonl` 每条：`{id, date, kind, ..., status}`，
status ∈ `pending / testing / passed / failed / approved / rejected / needs_human / applied`。
控制台的批准/驳回按钮改 status；`applied` 由应用那一步写。

## 6. 面板（out_learn/council.html）

1. 抬头：会诊日期、耗时、6 个视角各自状态（成功/超时/解析失败）
2. 主审结论：差距表（期望 vs 实际 vs 同期基准，带区间）、原因权重、
   「波动还是真差距」及概率、叙述
3. 图（内联 SVG，不引外部库）：
   - 起涨：每份清单的实际命中率 vs 期望 vs 同期基准（带二项区间）
   - 早盘：逐日前 10 超额与累计；分数分档 vs 实际
   - 特征：重要性 vs 实际贡献散点
4. 过程：每个视角一个折叠块，时间线里每一步（读了什么、查了什么、说了什么、
   用时），最后一条是它的结论。给人看「LLM 是怎么想的」
5. 提案台账：状态、实验结果、批准/驳回按钮（只在控制台里可用，Pages 上只读）

## 7. 接线

- `src/eval_daily.py`：`--stage council`，`all` 末尾调用，try/except，
  失败只写 `state/council/latest.json` 的 `ok=false`
- `src/gui`：面板 tab `council`（PANELS + data-p）、动作 `council`（手动跑）、
  `POST /api/council/decide`（批准/驳回）、总览页一张「最近会诊」卡
- `src/local_run.flow_learn`：推送 `state/council`、`out_learn/council.html`
- `src/build_site.py`：council.html 进 Pages
- `config.yaml` `learning.council`：enabled、model、effort、timeout、lenses、
  max_parallel、min_new_truth_days
- 自测 `selftest_learn.py`：证据包在合成数据上能建、schema 校验、脏输出改写、
  fail-open 接线（AST：council 调用在 try 里、不进 gate）、面板无状态能出页；
  `selftest_gui.py`：tab/PANELS/动作/端点 403

## 8. 成本与时间

Opus 5 max 六路并行，每路 3~8 分钟，主审 3~5 分钟，整体 10~15 分钟，
在学习流程末尾跑（16:40 北京起）。超时各 10 分钟，总 25 分钟。

## 8.5 跑的节奏（2026-09-16 用户定）

一次六个视角加主审实测 **20.23 美元**（Opus 5 max 推理，七个进程）。
每个交易日都跑约 440 美元/月，而 17 天的样本里每天的增量信息很小，
六个视角会反复说同一件事。所以：

- **自动**（学习线末尾）过 `run.due()` 的闸：距上次不足 `cadence_days`（7 天）
  就不跑，除非有加跑触发 —— 参数真的变过（`state/learned.yaml` 或
  `state/breakout/overrides.json` 的修改日期晚于上次会诊，按**北京日期**比），
  或者自上次以来又有 `trigger_new_truth`（10）个名额的真值到期。
- **手动**（控制台按钮、`--stage council`）不受节奏限制，随时可以问。
- 关掉整条线仍然是 `learning.council.enabled: false`。

`summary.json` / `latest.json` 里记 `n_settled`（起涨清单里已结算的名额总数），
下一轮就是拿它比「又到期了多少」。

## 9. 第一次真跑（2026-09-16）记下来的

六个视角并行 + 主审，实际数字：每个视角 11~13 分钟、8~21 次工具调用、$2.5~3.6，
主审 10 分钟 $2.3，一次会诊合计约 25 分钟、$20。证据包 100 KB（给 LLM 的精简版 46 KB）。

踩到的四个坑，都已经修进代码：

1. **步数上限会把十分钟的分析整个丢掉。** 第一次冒烟 Opus max 跑到 41 轮
   还在查数据，`--max-turns` 一到就 `error_max_turns`，$5.2 的分析没有任何输出。
   现在：提纲里写死「最多 25 次工具调用，第 20 次后开始写结论」，代码里步数到顶
   或看门狗超时都会 `--resume` 同一个会话再问一次「不许再查，现在就按 schema 输出」。
2. **主审的超时要按「想多久」算，不是按「查多久」算。** 主审只读 6 份意见不查数据，
   420 秒还是被杀（max 推理一次能想十分钟）。现在 1200 秒，且超时也会 resume。
3. **消融空转。** `_known_features()` 里 `import features` 失败时返回空集，
   校验被 `if known and ...` 跳过，于是拿一个根本不存在的特征名跑了十分钟，
   还因为「和基线逐位相同」被判成过闸。现在：读不到特征名单就不跑；候选与基线
   逐位相同判「空转」并写明入模列数；成绩表比特征表新就直接复用，不再重复跑十分钟。
4. **发信在控制台那条路上是哑的。** 控制台的按钮直接跑 `eval_daily.py`，
   不经过 `local_run`，`tools/local.env` 没加载，会诊邮件报 `'SMTP_HOST'` 被
   fail-open 吞成一行日志（历史教训 16 的同类）。现在三个入口共用
   `src/localenv.py`，`mailer._conf()` 再兜一次底。

**会诊进程要能从会话中断里活下来**：它是 `Start-Process` 起的独立进程，
父会话断了照样跑完、照样写台账和面板。计划任务那条路本来就是这样。

第一次的结论（存档）：早盘「排序≈0」在回填 414 天和在线 17 天两个样本上一致
（p≈0.85），「跑输」未证实（p≈0.4，17 天区间 [−1.75, +0.39]）；起涨预测结果
未到期（0 份满 20 根），但清单结构的问题确定（科创占 61%，而验证集科创命中
2.7%、主板 16.5%，按构成加权的期望是 7.2% 而不是邮件里印的 12.6%）。
10 条提案里自动实验判出 1 条过闸（板块系数 star 1.27→1.0，14.4% vs 12.6%）。

**当天晚上的对齐检查推翻了这一跑的一部分量化输入，读存档结论时要知道：**

- 证据包里 `spread_pct` / `excess_pct` 是把所有行池化求的均值，而旁边的
  `se_day_clustered` 是按天聚类的，两个不同的估计量相除。六个维度里
  gap / sector / continuity 三个的**符号**在两种算法下相反，gap_where 和
  features 两路是照着池化的符号下的结论。已修（教训 33 的第二句）。
- `feature_select` 读的是 2026-09-12 的实验产物（87 进 34 出），不是生产模型
  旁边那份（102 进 51 出）。提案 20260916-9b9104 报的「筛选记录与模型矛盾」
  是读错路径造出来的假阳。已修。
- 提案 20260916-75d86e（「突破平台」加分该消融）依据的两个数没有标准误。
  补测之后：在线 16 天差 −1.19±0.46（t=−2.58），但同一口径在回填 400 天上是
  −0.09±0.16（t=−0.54），没复现；而真正要问的「掐掉会不会更好」两段数据都是
  +0.03 个百分点、t<1。用户定「先出证据再决定」，证据不支持改动。
  见 `tools/platform_evidence.py`。

「过闸的那条」（star 1.27→1.0）不受影响：它走的是缓存分数即时重算那条路，
和证据包的统计量无关。用户当晚批准，随后发现成绩常量的失效保护看不见板块系数
这一维（教训 35）。
