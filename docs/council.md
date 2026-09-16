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
