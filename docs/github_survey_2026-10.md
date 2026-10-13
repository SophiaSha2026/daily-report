# GitHub 开源量化 / 股市分析仓库调研（2026-10-13）

调研日期 2026-10-13。星数和最近推送时间用 `gh api repos/<仓库>` 当天实查；功能描述来自各仓库 README 和关键源码
（逐个读过，没读到的写「未核实」）。几条数据接口在本机用 akshare 1.18.94 实测过，结果在第 3 节。

对照的现状（docs/breakout_log.md 实验 15~21）：起涨预测前 10 买得到口径命中 12.7%（开发集），全市场 2.9%；
实验 16~21 在特征、目标、候选池、融合、排序学习、更长历史、调参上没有一个过线；验证期 90.3% 的命中在模型
名次 200 名之外，起涨前一天和全市场长得一样（实验 16 第二轮那张表）。结论是「规律不在日线里」。

## 1. 逐个仓库

| 仓库 | 星数 / 最近推送 | 是什么 | 对我们可借鉴的点 | 不值得的理由 | 成本 |
|---|---|---|---|---|---|
| [UFund-Me/Qbot](https://github.com/UFund-Me/Qbot) | 18,637 / 2026-03-11 | 本地部署的量化平台大合集：backtrader 回测、qlib 模型表、同花顺 / 通达信 / 东财下单、邮件飞书推送 | 无实质新东西。因子是 MACD / KDJ / RSRS 等技术指标，模型表直接引用 qlib，资金流类指标标为付费 | 没有逐月滚动测试、封存数据、漂移处理；没有新闻 / 事件模块；Python 3.8/3.9；部分功能付费 | 不做 |
| [microsoft/qlib](https://github.com/microsoft/qlib) | 49,347 / 2026-10-08 | 微软的 AI 量化研究平台：Alpha158 / Alpha360 特征、20 多个模型、滚动重训、DDG-DA 漂移适配、PIT 数据库、RD-Agent | ① DDG-DA（`examples/benchmarks_dynamic`）：标签 20 天、每 20 天滚动，LightGBM 相关性 0.0816 -> 0.0878，本质是「学出来的样本加权」，和我们实验 15 的按月等权同一类事。② HIST 模型用概念板块成员关系共享信息，思路和实验 16/18 的同涨族群（g2）一致。③ Alpha158 可以当一个「标准特征集对照臂」 | 官方 A 股数据集停用（README：数据安全政策，临时停用），社区替代是 chenditc/investment_data；DDG-DA 增益小、要引平台、还要 `trusted=True` 读 pickle；平台整体比我们的 evalkit 宽松（没有封存数据纪律） | DDG-DA 大；Alpha158 对照中 |
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 111,019 / 2026-10-03 | LangGraph 多智能体：四个分析师并行 -> 多空研究员辩论 -> 研究经理裁决 -> 交易员 -> 激进 / 保守 / 中性三方风控辩论 -> 组合经理 | 见第 4 节（AI 讨论窗口）。实读源码：`conditional_logic.py` 辩论轮数 `max_debate_rounds` 默认 1（多空各 1 次），风控 `max_risk_discuss_rounds` 默认 1（三方各 1 次）；`rating.py` 解析不出评级时记 `REVIEW` 而不是默认 Hold；研究经理提示词写明「有分歧本身不是 Hold 的理由」；决策日志到期后自动结算真实收益并写一段反思（v0.6.0，2026-10） | 交易执行、仓位、风控辩论对我们无用（我们只出清单）；没有测过 LLM 判断的命中率；A 股只认代码后缀，没有 A 股专属数据 | 借设计小 |
| [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | 63,974 / 2026-10-02 | 「AI 对冲基金」：巴菲特 / 芒格等人格 LLM 代理和 PEAD 等量化模型同一接口，输出 [-1,1] 观点 + 理由，回测 / 模拟盘一条代码路径 | ① `signals/llm_agent.py` 的 **blind 模式**：回测时提示词去掉代码、行业、日期、金额，防止 LLM 凭记忆「知道」结果（对我们评估 AI 分析很关键，见第 4 节）。② 每次 LLM 调用落盘完整提示词 + 回复，快照没变就不重复付费（PromptCache）。③ LLM 失败一律「弃权」而不是中性，数据层错误直接抛。④ `event_study/` 市场模型异常收益 + t 检验 + bootstrap | 只有美股（Financial Datasets API）；人格代理对 A 股小票没意义 | 借设计小 |
| [ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) | 66,295 / 2026-10-13 | LLM 每日个股分析 + 大盘复盘推送，GitHub Actions 北京 18:00 跑，多源行情 + 搜索引擎新闻 | ① `docs/decision-signals.md`：AI 建议落成结构化信号，按 1/3/5/10 日事后结算命中率、最大不利波动，**每档满 30 个结算样本才显示比率**，不满写「仅供观察」。②`docs/data-source-stability.md`：同一源连续失败 3 次冷却 5 分钟后半开探测（熔断，和我们教训 20 同一思路）。③ `docs/intelligence-sources.md`：每条情报存发布时间和抓取时间、按 URL 去重。④ 个股资金流用 akshare `stock_individual_fund_flow`、行业资金流 `stock_sector_fund_flow_rank` | 没有任何「LLM 判断 vs 之后价格」的准确率报告（`agent-trajectory-eval.md` 只评工具调用轨迹，明确不评预测对错）；新闻源是财联社热门 / 雪球 / 华尔街见闻快讯，**没有公告、龙虎榜、板块**；LLM 直接给买卖评级，违反我们硬约束 1 | 借设计小 |
| [myhhub/stock](https://github.com/myhhub/stock)（InStock） | 14,786 / 2026-04-02 | A 股选股系统：东财抓数、32 个指标（TA-Lib）、61 种 K 线形态、11 个策略、选股后回测成功率 | 每日抓的数据清单可以抄：资金流、龙虎榜、大宗交易、**涨停原因**、早盘 / 尾盘抢筹、行业 / 概念资金流 | 策略（停机坪、突破平台、回踩年线）是单条件规则，没有我们长期调整突破那套逐阈值回看；回测方法 README 没写（持有期、口径都不明）；只有东财一个源，要配 cookie | 小（只抄抓数清单） |
| [AI4Finance/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 21,399 / 2026-09-23 | 金融 LLM 微调：情感分类（v3.3 FPB 0.882 F1）、Forecaster（道指 30）、RAG | 无 | 新闻条目停在 2023-11；情感模型英文为主，Forecaster 只有道指 30；LLM 输出进模型违反硬约束 1 | 不做 |
| [AI4Finance/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 16,638 / 2026-10-13 | 强化学习交易（A2C/PPO/SAC 等），教学定位，生产推荐 FinRL-X | 无 | 我们不做仓位和交易；README 不谈过拟合 | 不做 |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 45,849 / 2026-10-06 | 交易框架，4.x 加了 `vnpy.alpha`（Alpha158 + Lasso / LightGBM / MLP） | 无（Alpha158 直接看 qlib） | 价值在券商接口和实盘，我们不下单 | 不做 |

自己搜到的：

| 仓库 | 星数 / 最近推送 | 是什么 | 可借鉴 | 不值得的理由 | 成本 |
|---|---|---|---|---|---|
| [akfamily/akshare](https://github.com/akfamily/akshare) | 22,923 / 2026-10-13 | 我们已经在用的数据库 | 事件 / 情绪类接口，本机实测见第 3 节：龙虎榜历史到 2023、人气榜每只一年、涨停池只有近 30 天 | 东财接口的可达性会漂（硬约束 5） | 小 |
| [hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) | 32,276 / 2026-09-22 | TradingAgents 中文增强版，接 Tushare / AKShare / BaoStock | `tradingagents/dataflows/` 有 A 股数据和新闻适配，可参考 A 股新闻怎么喂给代理 | **混合授权**（LICENSE 实读）：`app/ frontend/ core/` 源码可见但禁止修改后分发和商用，只有 `tradingagents/` 等是 Apache-2.0；架构和原版一样 | 小（只看） |
| [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | 35,414 / 2026-10-13 | 个人交易代理（技能 + 回测 + 多代理） | ① `skills/event-driven/SKILL.md`：事件 CSV `date,event_type,score,source,summary`，**date 是可知日，盘后发布记次一交易日**，按类型给衰减期（业绩 1~5 天、政策 20~60 天）。② `skills/ashare-pre-st-filter`：用 tushare `namechange`（带 `change_reason`、起止日、公告日）拿历史 ST 改名 | 事件打分由 LLM 给 -1~1，不能进我们的排序 | 小 |
| [ArvinLovegood/go-stock](https://github.com/ArvinLovegood/go-stock) | 7,833 / 2026-10-11 | Go 写的 A 股 AI 分析桌面工具 | 数据面全：通达信协议的竞价和资金流、龙虎榜席位（含游资席位映射）、涨停梯队、题材轮动、两融；AI 推荐按 3/5/10/20/30 日对沪深 300 结算 | 无公开准确率；README 自称「娱乐」 | 小（只看） |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 14,921 / 2026-10-11 | LLM 自动挖因子、改模型（接 qlib） | 产出的是确定性代码，形式上不违反硬约束 1 | 我们的瓶颈不是特征不够（实验 16/18/21），自动挖几百个因子在同一开发集上比，选择偏差只会更重 | 不做 |
| [chenditc/investment_data](https://github.com/chenditc/investment_data) | 1,522 / 2026-10-13 | qlib 社区 A 股数据（DoltHub），qlib 官方推荐的替代源 | 如果要跑 Alpha158 对照臂，从这里拿 qlib 格式数据最省事；也可以拿它抽检我们的日线 | 我们自己的日线更全（换手率、流通股本） | 小 |
| [esvhd/pypbo](https://github.com/esvhd/pypbo) | 143 / 2026-07-06 | 回测过拟合概率（PBO / CSCV）、Deflated Sharpe | 给 evalkit 加「试了多少个臂」的多重检验修正，见第 5 节 | 小库，照公式自己写十几行就够 | 小 |
| [hugo2046/QuantsPlaybook](https://github.com/hugo2046/QuantsPlaybook) | 6,543 / 2026-10-03 | 券商金工研报复现 | `B-因子构建类/筹码因子`、行业轮动、北向资金能力 | 没有涨停 / 事件驱动；多是月频选股和择时，和 20 天翻半倍的标签不是一类问题 | 不做 |

## 2. 三个问题的回答

### A. 能不能突破「新闻驱动、模型抓不到」

**没有一个仓库拿出过「事件 / 情绪信号在 A 股上样本外有增量」的证据。** TradingAgents、daily_stock_analysis、
go-stock 都是让 LLM 读新闻给结论，都没公布准确率；FinGPT 只做情感分类 F1。能借的是**数据和事件口径**，
不是模型。按「数据在不在、历史够不够做 evalkit 开发集（2024-06..2025-12）」分：

| 数据 | 来源 | 历史够不够 | 判断 |
|---|---|---|---|
| **全市场公告，带东财细类标签** | `np-anotice-stock`（corp_events.py 第 150 行已有按日期区间全市场查的路径） | 够，按公告日 | **最值得试**。控制权变更、协议转让、重大资产重组、中标 / 重大合同、业绩预增、异动公告这几类是「突然点燃」的典型来源，而且是规则可判的类别，不用 LLM，不碰硬约束 1 |
| 龙虎榜明细 | akshare `stock_lhb_detail_em`（datacenter-web） | 够，实测 2023-10 可取 | 只覆盖已经异动的票，帮的是前 200 名那一块，不是 90% 那块 |
| 东财人气榜个股历史 | `stock_hot_rank_detail_em`（emappdata） | 每只只有一年（实测 366 行，2025-10-14 起） | 只能测 2025-10 之后，落在确认集和实盘段，先存档 |
| 涨停池（首次封板时间、炸板次数、连板数、所属行业） | `stock_zt_pool_em`（push2ex） | **只有近 30 天**（实测 2026-09-30 有 52 行，2026-09-01 返回 0 行；炸板池源码写死「只能获取最近 30 个交易日」） | 要从今天起每天存档，否则永远没有历史 |
| 个股资金流 | `stock_individual_fund_flow`（push2his fflow） | 120 天 | 同上，只能存档 |
| 互动易问答 | `stock_irm_cninfo` | 每只只返回近几条（300829 实测 3 条） | 不够 |
| 新闻（东财个股新闻） | `stock_news_em` | 最近 100 条 | 不够，且只能 LLM 读，不能进排序 |

要说清楚的风险：公告大多盘后发布，次日一字板买不进。实验 16 已经量过命中里 14.6% 次日一字板，
事件驱动的命中这个比例只会更高。所以第一步先做诊断（第 5 节第 1 条），看命中前几天有没有可判类别的公告、
以及这些命中**买不买得到**，再决定要不要建特征。

### B. 评估、漂移、在线学习有没有我们没做的

我们的 evalkit（逐月滚动、净化线、确认集按季度、预登记、按天聚类）已经比这些仓库都严：qlib 只有滚动回测，
Qbot / InStock 是单段回测，daily_stock_analysis 不评预测。还缺三处：

1. **标签重叠造成的自相关没修正。** 标签是 20 根之后的最高价，相邻交易日的前 10 名单高度重合、结果共享同一段行情，
   按天配对的差值不独立，`GATE["t_day"] = 2.5` 的 t 偏大。应该用 Newey-West（滞后 20）或按月分块 bootstrap 再算一遍。
   这不提高命中，但会改「什么算过线」：g2 的 t = 2.08、thr98 的 t = 1.8 会更低，eqm 的 4.50 要复核还在不在 2.5 以上。
2. **没有多重检验修正。** 实验 15~21 在同一个开发集上比了 40 个左右的臂，2.5 是固定门槛。pypbo 的 CSCV / PBO
   或者最简单的「t 门槛随本族臂数抬高」（Bonferroni 式）可以写进 prereg，每个预登记族记臂数。
3. **漂移：DDG-DA 不必上。** 它是学出来的样本加权，qlib 自己的表上 LightGBM 相关性只多 0.006；我们的按月等权
   已经拿到了这一类改动里最大的一块（实验 15，t = 4.50），滚动窗口、时间衰减、滚动校准、环境门控也都测过。
   漂移告警（`drift_note`）已上线。

另外两件小事：ai-hedge-fund 的「回测和实盘一条代码路径」我们已经做到（`validate.pick` 唯一实现）；
TradingAgents v0.5.0 的 point-in-time 纪律（过去日期的回测不给内部人交易数据）和我们教训 25 是同一件事。

### C. 多智能体讨论哪些适合「AI 讨论」窗口（AI 只分析不排序）

适合的：

| 设计 | 出处 | 落到哪里 |
|---|---|---|
| **正方 / 反方 / 裁判，固定轮数**：多空各说一次，裁判按「哪边证据更硬」下结论，提示词写明「有分歧本身不是观望的理由；证据太薄才观望」 | TradingAgents `conditional_logic.py`、`research_manager.py` | AI 讨论窗口的一种模式：同一个 `run_claude` 会话三次调用，轮数写死（教训 31：不给上限它会一直查） |
| **解析不出结论就记「待复核」，不默认中性** | TradingAgents `rating.py` 的 `REVIEW` | `tools/ai_analyze.py` 的 schema 结果校验 |
| **到期结算 + 一段反思写进记忆** | TradingAgents 决策日志（v0.6.0） | AI 分析每次落盘立场和置信度，20 根后按 `buyable.py` 的 y_open 结算；下一次分析同一只票时把上次结论和结果带进提示词 |
| **满 30 个结算样本才显示命中率** | daily_stock_analysis `decision-signals.md` | 控制台显示 AI 分析的历史命中时同样设门槛，不满写「仅供观察」 |
| **blind 模式** | ai-hedge-fund `llm_agent.py` | 任何**回头**评估 AI 分析的实验（用历史日期重放）都必须去掉代码、名称、日期。训练截止日之前的行情 Opus 可能「记得」，重放 2025 年的票等于开卷。只有向前结算的成绩算数 |
| 提示词 + 回复完整落盘，输入没变不重复调用 | ai-hedge-fund PromptCache | `src/ai_usage.py` 记账旁边加一份按输入哈希的缓存，同一天同一只票重复点不再花 1.5~1.8 美元 |
| 证据包带发布时间和抓取时间 | daily_stock_analysis `intelligence-sources.md` | AI 讨论里引用的新闻 / 公告逐条带时间，裁判只认目标日收盘前能看到的 |
| 快慢两档模型：分析师和辩手用便宜档，裁判用深档 | TradingAgents `quick_think_llm` / `deep_think_llm` | 正反方用低推理等级，裁判用 Opus 高等级，控成本 |

不适合的：三方风控辩论和组合经理（我们不管仓位）；人格代理（巴菲特们对小盘题材股没有意义）；
让 LLM 输出买卖评级再进清单（硬约束 1）；Jev 这类外部打分服务（多一个外部依赖，结果也只能看）。

## 3. 本机实测（2026-10-13，美东家宽，akshare 1.18.94）

| 接口 | 主机 | 结果 |
|---|---|---|
| `stock_lhb_detail_em` 2025-09-01..05 | datacenter-web.eastmoney.com | 423 行，4.8 秒 |
| `stock_lhb_detail_em` 2023-10-09..13 | 同上 | 302 行，历史可取 |
| `stock_hot_rank_detail_em` SZ300829 | emappdata.eastmoney.com | 366 行，2025-10-14 ~ 2026-10-14 |
| `stock_individual_fund_flow` 300829 | push2his（fflow） | 120 行，2026-04-15 起 |
| `stock_board_concept_name_em` | push2（clist） | 506 个概念，**能通**（CLAUDE.md 5b 记的 push2 不通是 08-24 的结论，又漂了） |
| `stock_irm_cninfo` 300829 | irm.cninfo.com.cn | 3 行 |
| `stock_zt_pool_em` 20261013 / 20260930 / 20260901 / 20250910 | push2ex | 67 / 52 / 0 / 0 行：只有近 30 天 |
| `stock_board_concept_hist_em` | 91.push2his | 120 秒内没返回，未核实 |

tushare `namechange` 文档实读：返回 `ts_code, name, start_date, end_date, ann_date, change_reason`，示例里有
「撤销ST」等原因，600848 记录回到 1994；需要的积分文档没写，沪深是否都全未核实。

## 4. 按性价比排序的建议

过线一律按 evalkit：预登记、开发集 2024-06..2025-12 逐月滚动、买得到口径为主、按天 t ≥ 2.5 且最差三个月
不比现状差且空榜不翻倍。

| # | 做什么 | 预期影响 | 怎么验证 | 成本 |
|---|---|---|---|---|
| 1 | **公告诊断**：用 `np-anotice-stock` 按日期区间拉 2024-06 起全市场公告（带细类标签、公告时间），对实验 16 那 31,988 个买得到的命中，统计起涨日前 5 个交易日内各细类公告出现率，和同期全市场比；同时量这些命中次日一字板的比例 | 回答「90% 那块到底是不是公告驱动、买不买得到」。是的话才有第 2 条；不是的话 A 这条路关掉，省后面所有成本 | 纯描述统计，不进模型，不占确认集。判据预先写：某类公告的命中前出现率 ≥ 全市场 2 倍、且这批命中买得到的比例 ≥ 50%，才进第 2 条 | 中（拉数几小时，缓存进 data/pullback/corp/ 同类目录） |
| 2 | evalkit 加 **Newey-West（滞后 20）或按月分块 bootstrap 的 t**，并记录每个预登记族的臂数，门槛按臂数抬高 | 不改命中，改「什么算过线」：减少把噪音当改进的概率。复核 eqm（4.50）、g2（2.08）、thr98（1.8） | 在实验 15/17/18/21 已缓存的臂上重算，旧结论变不变写进 breakout_log；selftest_breakout 钉一个「已知无自相关的序列，两种 t 接近」用例 | 小 |
| 3 | **每天存档事件数据**：涨停池（首封时间、炸板、连板、行业）、龙虎榜、人气榜、个股资金流，起涨预测发信后 fail-open 跑一次，落 `data/events/YYYY-MM/` | 今天不改成绩；涨停池和资金流的历史只有 30 / 120 天，不从现在存，半年后想测也没得测 | 存档本身只验接线（selftest_gui：fail-open、不挡发信、有步骤时限，教训 43）。满 6 个月后按第 1 条的方式先做诊断 | 小 |
| 4 | **公告事件特征**（第 1 条过了才做）：按细类做规则特征，「近 N 天有某类公告」+ 当日全市场排名，不用 LLM | 目标是 200 名外那 90%。实话：上限受「次日一字」限制，买得到口径的增量可能远小于收盘口径 | 预登记一个臂（只加公告列）对现状，按 evalkit 过线；要求同时报收盘口径和买得到口径，两者差距写进日志 | 中（features.py 在模型指纹里，采纳要走教训 36 流水） |
| 5 | **AI 分析向前结算**：`ai_analyze.py` 每次落盘立场、置信度、目标日；到期按 y_open 结算；满 30 个样本前控制台只显示「仅供观察」 | 知道 AI 分析到底有没有用，避免用户被一段写得很好的理由带着走。不进排序 | 30 个样本后比较「AI 看多」和「AI 看空 / 观望」两组的命中，按天聚类；同期和模型分数同档的票对照 | 小 |
| 6 | **AI 讨论窗口加「正方 / 反方 / 裁判」模式**：固定一轮、裁判提示词借 TradingAgents 那两句、解析不出记「待复核」、输入哈希缓存、证据带时间 | 讨论质量和成本可控；不影响清单 | 接线自测（固定轮数、REVIEW 分支、缓存命中不调用 CLI）；成本看 `state/ai_usage.jsonl` | 小到中 |
| 7 | **tushare `namechange` + 退市名单**修回测的 ST 静态名单和幸存者偏差（计划第 4 节 4.1 / 4.2 两条没修的） | 数字更诚实，不是更高（4.1 量过上界不到 1 个百分点） | 先核 tushare 积分要求和沪深覆盖；修完重跑现状臂，差值写进日志 | 小（要 tushare token） |
| 8 | **龙虎榜特征**（历史到 2023，实测可取）：上榜、机构净买、游资席位 | 帮的是已经在动的那一块；实验 19 说那块的天花板是连板高度，增量估计很小 | 预登记一个臂，同第 4 条 | 中 |

不建议做：DDG-DA（增益小、成本大）、RD-Agent 自动挖因子（加重选择偏差）、FinGPT 情感模型、FinRL、vnpy、
Qbot 整体迁移。Alpha158 对照臂可以放在第 8 条之后，只用来回答「我们的 40 列是不是比标准特征集差」，
不期望它提高命中。
