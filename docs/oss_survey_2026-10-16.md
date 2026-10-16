# 开源 / 文献 / 数据源增量调研（2026-10-16）

（调研子代理写成，主会话落盘。只调研，没改代码。实测在本机（美东家宽，北京 20:55 后，akshare 1.18.94），每个接口单次请求或二分探测，不超过 10 次。和 docs/github_survey_2026-10.md、research/datasrc/report.md 重复的不再列，只写增量。）

## 1. 摘要（按「可能增量 × 获取成本」排序）

| # | 项 | 理由 |
|---|---|---|
| 1 | **同花顺涨停揭秘 `data.10jqka.com.cn/dataapi/limit_up/limit_up_pool`，现在就存档** | 已实测：首封时间、炸板次数、板型、几天几板、涨停原因，历史到 **2025-10-09**，比东财涨停池（30 天）多约一年。窗口应是滚动的（推断），每天丢一天；约 250 个请求能存下 |
| 2 | **同花顺强势股题材 `zx.10jqka.com.cn/event/api/getharden/date/<日>/`** | 已实测：每只涨停票带当天题材原因（如「业绩预增+高端PCB」），历史约从 **2023-10 下旬**起，覆盖整个开发集。设计 4.4 因「没有时点一致的行业表」关掉的族群方向可以重开「题材热度」 |
| 3 | **申万行业变迁史 `StockClassifyUse_stock.xls`** | 已实测：1.1 MB，12929 行、5934 只（含 920 段），每次调整有计入日期，point-in-time 行业表 |
| 4 | **通达信盘后包 `g4day/<日>.zip` + 全历史 `hsjday.zip` + 股本变迁 `gbbq.zip`** | 已实测可下载：每日包 2021-08-13 起、当天收盘后发布、不复权，北交所 2022-05-06 起。配 gbbq 本机算后复权（设计 6.1），不用整表重拉新浪；逐日包可能含退市票，能修筛选集幸存者偏差（未核实） |
| 5 | **`arch` 库的 StepM / SPA / MCS** | 查了源码：多重比较和「哪些臂和最好的分不开」（MCS）有现成实现，能替换手写最大 t。修尺子，不提命中 |
| 6 | 开盘啦历史首板 `apphis.longhuvip.com`（`HisHomeDingPan`） | 已实测 2020-07-01 有数据，带原因和板块代码。无文档、App 内部接口、条款风险；924 只有 557 条（本机算约 914 只涨停）；一列概念标签是当前值（偷看）。只作第 2 条的后备 |
| 7 | 通达信分时包 `g4tic/<日>.zip` | 已实测 2024-01-02 起，每天约 100 MB。解析要通达信 Linux datatool，只适合抽样诊断 |
| 8 | qlib `DoubleEnsemble` | 只看了文档：Alpha158 / CSI300 基准相关性 0.0448 → 0.0521（对 LightGBM）。本质是样本重加权 + 特征重采样，同类改动在我们这里都在噪声里，只配当诊断臂 |

**总判断**：没找到任何公开项目、论文或竞赛在「A 股 20 天涨 50%、每天前 10」这类问题上给出过可信的样本外成绩。真正的增量在**数据**：带时点的题材标签、带时点的行业、更长的涨停微结构、不复权原始日线。

## 2. 分类清单

### 2.1 暴涨 / 涨停 / 右尾预测

| 名字 | 链接 | 和我们的关系 | 核实 |
|---|---|---|---|
| GitHub「涨停预测」类仓库（hacker4257/astock_analyzer、bluesli2005/a-share-limit-up-predictor、william-smith/zhangting-guilv-self-learning 等） | gh search「涨停 预测」「妖股」「打板」「连板 预测」「起涨点」 | 全是 5 星以下，没有逐月测试和命中率报告，无可借鉴 | 查了仓库列表 |
| simonlin1212/a-stock-data（10,895 星，2026-10-11，Apache-2.0） | https://github.com/simonlin1212/a-stock-data | A 股数据 Skill，87 个端点；第 1、2、4、7 条的接口线索来自它。只抄端点不引入 | 读了 SKILL.md，3 个端点已实测 |
| Liang & Hu 2025，Journal of Forecasting 44(2) | https://ideas.repec.org/a/wly/jforec/v44y2025i2p297-319.html | 逐股建模，平衡准确率 50% → 约 66%，评估远比我们松 | 只看摘要 |
| Conrad, Kapadia, Xing 2014「Death and jackpot」（JFE 113） | https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html | 最接近的问题定义：logit 预测 jackpot 概率。**jackpot 概率高的票平均收益反而低**：命中率不等于收益，邮件只印命中率是对的 | 只看摘要 |
| A 股 crash / jackpot 概率异象（CFRI 16(1)） | https://www.emerald.com/cfri/article/16/1/145/1250726/Crash-and-jackpot-probability-anomalies-in-the | A 股 jackpot logit，加市销率、区分市场状态 | 只看摘要（付费墙） |
| Bali, Cakici, Whitelaw 2011「Maxing Out」 | https://nber.org/papers/w14804 | 彩票类特征，Alpha158 的 STD / MAX 已覆盖 | 只看摘要 |
| Baruník 等 2024，分位数神经网络 | https://arxiv.org/abs/2408.07497 | 越往尾部越弱，印证「分位数回归 0.95 不做」 | 只看摘要 |
| 跳跃预测（订单簿注意力网络；深交所日内跳跃随机森林） | https://arxiv.org/pdf/1810.10845 、https://hpedsi.uh.edu/publications/predicting-intraday-jumps-stock-prices-using-liquidity-measures-and-technical | 要 Level-2 / 订单簿，没有免费源 | 只看摘要 |
| Kaggle / 天池 / DataFountain | 搜了没找到 | 没有同类题（JPX、Ubiquant、Jane Street、Optiver 都是均值或短期收益） | 只搜索 |

### 2.2 数据：见第 3 节

### 2.3 方法论

| 名字 | 链接 | 对我们 | 核实 |
|---|---|---|---|
| **arch**：StepM、SPA、RealityCheck、MCS，平稳 / 循环 / 移动块自举（1,598 星，NCSA） | https://github.com/bashtage/arch `arch/bootstrap/multiple_comparison.py` | StepM 能替换手写最大 t；MCS 回答「哪些臂和最好的分不开」，用于选臂。纯 CPU 秒级。本机未装 | 查了源码 |
| skfolio `CombinatorialPurgedCV`（2,507 星，BSD-3） | https://github.com/skfolio/skfolio | 能用但帮助小，见第 4 节 | 查了源码 |
| mlfinpy（mlfinlab 的 MIT 替代，89 星） | https://github.com/baobach/mlfinpy | 和 skfolio 重叠 | 查了目录 |
| pypbo | 上一轮已列 | **许可证 AGPL-3.0**（上一轮没写）；research/stats 已算过 PBO，不引入 | 查了元数据 |
| focal loss 用于 GBDT | https://arxiv.org/html/2407.14381v1 | 表格数据有实证，没有选股；对前 k 排序和下采样 / 重加权同类 | 只看摘要 |
| qlib DoubleEnsemble / TRA | https://github.com/microsoft/qlib `examples/benchmarks/README.md` | DoubleEnsemble 见摘要 8；TRA 要 GPU，不做 | 只看基准表 |
| qlib `limit_threshold` | `qlib/backtest/exchange.py` | 和我们 buyable.py 的 y_open 同一件事，已做 | 查了源码 |

### 2.4 qlib / RD-Agent / FinRL 增量

只补：DoubleEnsemble、TRA，Alpha191 / Alpha101 实现（popbo/alphas，2023 后不更新，和 Alpha158 同族，不值得另开一臂）。tdx2db（https://github.com/jing2uo/tdx2db ，432 星，MIT）可参考盘后包和 gbbq 解析、后复权算法（README 写明与 QUANTAXIS / 通达信一致）。

### 2.5 社区经验（有具体证据的很少）

- 雪球、知乎带数字的复盘帖没找到（只搜索）。
- Buczynski 等 2021 综述：多数已发表的股票 ML 结果来自「训几百个模型报最好的」，实盘兑现不了（https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8019690/ ）。和我们的登记簿门槛同一出发点。
- Kybernetes 2023（上海 A 股）：博彩型股票识别当月收益高，之后 1~6 个月显著回落（https://www.doi.org/10.1108/k-05-2023-0802 ）。
- Lin、Qiu、Zheng 2023（JBF）：涨跌停暴露大的票吸引关注，之后收益更低（https://ideas.repec.org/a/eee/jbfina/v150y2023ics0378426623000432.html ）。
- 系统管理学报 2023：A 股量价趋势与未来收益反向，小盘高波动更明显（https://xtglxb.sjtu.edu.cn/CN/Y2023/V32/I4/774 ）。
- 合起来：「右尾概率高」和「平均收益高」在 A 股是两件事甚至相反。我们的标签只问前者，邮件不该出现任何收益暗示，现在的口径是对的。

## 3. 数据源表（只列 research/datasrc/report.md 没覆盖或写成「未核实」的）

| 数据项 | 接口 | 历史起点 | 本机实测（2026-10-16） | point-in-time 问题 |
|---|---|---|---|---|
| 涨停池 + 原因（首封时间、炸板次数、板型、几天几板、封单、回封） | 同花顺 `data.10jqka.com.cn/dataapi/limit_up/limit_up_pool?date=` | **2025-10-09**（09-30 及更早 0 行） | 2026-07-01 149 行，本机算非北交所涨停 151 只。`filter=HS,GEM2STAR` 不含北交所。requests 直连能通 | 收盘后快照；原因是否事后改写未核；窗口很可能滚动一年（推断） |
| 涨停票当日题材原因 | 同花顺 `zx.10jqka.com.cn/event/api/getharden/date/YYYY-MM-DD/...` | **约 2023-10 下旬**（10-09 0 行，11-01 43 行） | 2026-10-15 36 行（本机算 36 只）；2024-09-30 693 行（本机约 914）：**极端日不全** | 原因文本是否事后编辑未核；只给涨了的票 |
| 首板列表 + 原因、板块代码 | 开盘啦 `apphis.longhuvip.com/w1/api/index.php?a=DailyLimitPerformance&c=HisHomeDingPan&Day=` | 2020-07-01 已有 | 2020-07-01 41 行，2024-09-30 557 行；应只有首板 | **第 12 列是当前概念标签，会偷看**；第 5 列原因像当时值（推断）；无官方文档 |
| 申万行业归属变迁 | `www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls`（akshare `stock_industry_clf_hist_sw` 同一地址，查了源码） | 1990 起；2021-07-30 全量重分 | 12929 行、5934 只、3914 只调整过；300829 在 2021-07-30 由 110504 改到 220315。requests 偶发 DNS 失败，curl 正常 | 按「计入日期」对齐即可；只有代码没有中文名 |
| 全市场不复权日线（含成交额，沪深北） | 通达信 `www.tdx.com.cn/products/data/data/g4day/YYYYMMDD.zip` | **2021-08-13** | HEAD 200，每天 1.5~2.3 MB，当天下午发布 | 天然时点值；2022 年前深市很多代码当日无价（a-stock-data 注明） |
| 全历史日线完整包 | `data.tdx.com.cn/vipdoc/hsjday.zip` | 全历史 | HEAD 200，549 MB | 不复权；是否含退市票未核 |
| 股本变迁 / 除权（gbbq） | `www.tdx.com.cn/products/data/data/dbf/gbbq.zip` | 全历史 | HEAD 200，6.2 MB | 按除权日生效，算后复权不偷看 |
| 全市场分时 | `www.tdx.com.cn/products/data/data/g4tic/YYYYMMDD.zip` | **2024-01-02** | HEAD 200，每天 85~103 MB | 要通达信 datatool 转档，Windows 不能直接用 |
| 北向个股持股 | 东财 `RPT_MUTUAL_HOLD_DET`（akshare `stock_hsgt_individual_detail_em`） | **实际已断** | 002475：akshare 报 TypeError；最新 2024-09-30，HOLD_SHARES 为空 | 不可用 |
| 互动易问答 | 巨潮 `irm.cninfo.com.cn/newircs/company/question` | 近几个月 | 300829 3 条、002475 77 条 | 不够做历史 |

沿用 10-15 结论没重测：龙虎榜（2004 起）、两融（2010 起）、大宗交易、解禁、质押（2014 起）、机构调研（2011 起）、研报评级、baostock 5 分钟线（2020-04 起）、pytdx 历史分笔 / 分时（2005 起）。

**冲突提醒**：a-stock-data V3.10.0（2026-09-22）写通达信公开服务器 K 线 / 盘口 / 逐笔 TCP 命令已失效，而 10-15 普查说 pytdx 历史分笔能连。用前先复测。

**5b 表补一个细节**：同花顺「只认真浏览器」只对 `q.10jqka.com.cn` 成立；`data.10jqka.com.cn`、`zx.10jqka.com.cn` 今天 requests 直接 200。

## 4. 对我们没用的常见建议

| 建议 | 为什么没用 |
|---|---|
| CPCV | 瓶颈是独立观测少（387 天约 44 个有效天），重新组合同一批天不会缩小 MDE；还要拿未来段训练，924 会在折间串味。用途是调参，调参已关（实验 18） |
| 样本唯一性权重 | 截面上标签重叠都是 20 根，权重几乎常数；evw 已证有害（筛选集 NW t −3.48） |
| focal loss、类别平衡损失 | 没有选股实证；和下采样 / 重加权同类，第 2 周七个臂都在噪声里 |
| 分位数 / 极值理论 / 预测分布 | 尾部分位数最难预测（Baruník 2024），已判不做 |
| 序列模型、截面 Transformer、TRA、图像 CNN | 要 GPU；qlib 基准打不过树模型 + Alpha158；MDE 7.5pp 测不出 |
| Alpha191 / Alpha101 / 更多量价因子库 | 和 Alpha158 同族，base_pv 已吸收 |
| 北向资金个股特征 | 2024-08 后数据断了 |
| 互动易、个股新闻 | 历史只有几个月；只能交给 LLM 读，进不了排序（硬约束 1） |
| 逐股建模 | 每只票正样本个位数，评估口径也不是截面前 k |
| 「彩票股 / MAX 效应」当选股信号 | 这类票平均收益低；它解释右尾概率（已在用），当「好票」信号方向就反了 |
| 整套搬 a-stock-data、Qbot、InStock | 只需要 3~4 个端点；整套引入多一层依赖和东财封禁风险 |

**时效提醒**：第 1 条（同花顺涨停揭秘）的历史窗口每天往前丢，要存档越早越好。tools/archive_events.py 现在只存东财六池，没有这个源。
