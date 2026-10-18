# 深挖数据源 x 起涨精确率（research/deepdata，2026-10-18）

（Fable 5.1 high 子代理写成；.md 由主会话落盘。实测北京 2026-10-18 03:38 ~ 08:58，本机美东家宽，akshare 1.18.94。
全程没进 16:00~18:30 禁区；没改 src / state / data / out_* / cache；没碰 research/gate、research/intraday；2026-01..08 标签一行没读；
候选池 / ret20 / y_open 全部复用 research/polish 的 pool.parquet（common_pl），没另算一份。逐步记录 progress.txt。）

## 0. 一屏结论

1. 新拉了 **11 张原始表 + 1 份行业变迁表**（约 300 MB，raw/，gitignore）、落地 **7 张特征表**（features_*.parquet，键 code + date，覆盖 pool.parquet 2021-12-01..2025-12-31 全部行，
   不只候选池；说明在 features_README.txt）：申万时点行业 x 板块共振、龙虎榜席位（时点自算的活跃度 / 战绩）、研报评级（巨潮按日全市场）、
   公司事件六类（业绩预告 / 快报 / 回购 / 送转 / 股权激励 / 增持）、融券侧、同花顺题材（只有 2023-10-25 起）、十大流通股东（按公告日）。
2. **快速诊断（不训模型）里，两段同号、在「ret20 前 50 且控制名次档」后仍 >= 2pp 的只有五类，方向都是「越没人管越好」**：
   - 研报覆盖少：一年没研报 rat_nocov250 +2.6 / +2.0pp；一年研报篇数 rat_n250 三分位 −2.7 / −2.6；60 日买入级篇数 rat_strong_n60 −2.3 / −2.6。
   - 两融标的里融券余量为 0（rq_zero_yl）+3.8 / +2.4；融券占两融比例高（rq_share）−1.5 / −4.4。
   - 回购金额上限大（rep_amt_last）−3.5 / −4.8（覆盖 20%）。
   - 近期上过龙虎榜：lhbs_today +2.0 / +1.3，距上次上榜天数 lhbs_age −2.3 / −1.7（和 altdata 的「今日上榜」同一件事，这里多了席位维度，席位维度本身没加出东西）。
   - 题材：最近一次上榜题材今天的热度 ths_lag_heat +1.7 / +2.2（t 1.2 / 0.9，筛选段只有 145 天）。
   这些 t 大多在 1~3，和 precision 报告的 MDE（出 1 只/天要 +8pp）差一个量级；它们是门槛模型的候选输入，不是单独能用的规则。
3. **板块共振（时点申万行业）在全候选池里非常显著、到前 50 里就没了**：同二级行业当天涨停占比 ind2_lu_share 全池 +0.84 / +1.06（t 8.2 / 4.5），
   前 50 控名次 +1.9 / −0.5（异号）。它分辨的是「谁会进前 50」，ret20 已经把它吸收了。唯一前 50 内两段同号的是 ind2_ret20_excess（自己 ret20 减行业中位，+2.7 / +0.9）
   和一级行业涨停只数 ind1_lu_n（+1.7 / +1.3）。
4. 业绩预告 / 快报 / 送转 / 股权激励 / 增持：前 50 内基本没有两段同号的量级，有的还是反向（正向预告 yjyg_pos_n20 −0.6 / −6.5；负向预告 yjyg_neg_n60 +4.4 / +1.3）。
   和 announce 报告一致：基本面利好不是 20 天涨 50% 的触发。
5. 拉不到 / 不值得的：人气榜历史只有一年（2025-10 起）、涨停池（东财 30 天、同花顺一年滚动）、个股资金流（120 天）、北向个股（2024-08 断）、
   概念板块成分无时点历史（东财 / 同花顺都没有进出记录，回填必偷看）、东财 reportapi 全市场研报按日只给 1/5（28 对巨潮 145）、
   基金持仓只有报告期无公告日、通达信财务包无公告日。付费选项：Tushare limit_list_d（5000 积分）/ moneyflow（2000 积分），没买。

## 1. 数据源排序表（按 对 20 天涨 50% 的先验价值 x 可得性）

列：实测（2026-10-18 03:40 北京，probe_dd.py -> probe_result.jsonl）、历史起点、时点、拉了没有。

| # | 数据 | 接口 | 实测 | 历史起点 | 时点（T 日收盘后可得的口径） | 拉了 | 价值 x 可得 |
|---|---|---|---|---|---|---|---|
| 1 | 申万行业归属变迁（时点行业）+ 板块共振 | swsresearch StockClassifyUse_stock.xls（requests SSL 失败，curl 可） | 12929 行 5934 只，计入日期 1990..2026-10-14，候选池覆盖 100% | 全历史（2021-07-30 整体换 2021 版） | 计入日期 <= T 的最后一条 | 是，features_sw_industry 30 列 | 高 x 高（但前 50 内被 ret20 吸收） |
| 2 | 龙虎榜买 / 卖席位 | datacenter-web RPT_BILLBOARD_DAILYDETAILSBUY / SELL | 每月约 7600 行；带 RISE_PROBABILITY_3DAY / TOTAL_BUYER_SALESTIMES_3DAY 后验列（删） | 2004 | 交易所盘后公布，T 日可用；席位战绩只用 s <= T−21 的标签 | 是，39.3 万 + 39.3 万行，features_lhb_seats 37 列 | 中 x 高 |
| 3 | 研报评级（全市场按日） | 巨潮 webapi p_sysapi1089（akshare stock_rank_forecast_cninfo） | 2022-01-05 123 行、2024-03-05 145 行；东财 reportapi 同日只 28 行（不全，弃用） | 至少 2022-01（更早没测，表从 2021-09 拉） | 发布日期，非交易日算下一交易日 | 是，1583 天 37.9 万篇，features_rating 20 列 | 中 x 高 |
| 4 | 融券侧 | research/altdata/raw/rzrq.parquet（RPTA_WEB_RZRQ_GGMX，没重复拉） | 365 万行 | 2010 | T+1 早上发布，按 T−1 取值 | 派生，features_rzrq_short 8 列 | 中 x 高 |
| 5 | 业绩预告 | datacenter RPT_PUBLIC_OP_NEWPREDICT | 2024-01 7681 行（一公告多行：净利润 / 扣非 / 营收） | 2003 | NOTICE_DATE | 是，6.3 万行 | 中 x 高（诊断：没用 / 反向） |
| 6 | 业绩快报 | RPT_FCI_PERFORMANCEE | 有 NOTICE_DATE + EITIME | 1996 | NOTICE_DATE | 是，9553 行 | 低 x 高 |
| 7 | 回购 | RPTA_WEB_GETHGLIST_NEW | 每月约 116 条；REPURPROGRESS 几乎全是 006；RCHANGE* 后验列（删） | 2006 | NOTICEDATE（最新公告）/ DIM_DATE（首次） | 是，3330 行 | 低 x 高 |
| 8 | 分红送转预案 | RPT_SHAREBONUS_DET | 按 PLAN_NOTICE_DATE 查；D10/BD10/D30 后验列（删） | 全历史 | PLAN_NOTICE_DATE | 是，1.7 万行，10 送转 >= 5 只有 283 条 | 低 x 高 |
| 9 | 股权激励 | RPT_EQUITYINCENTIVE | 全表 4184 行，**只从 2023-08-01 起**（只覆盖开发集） | 2023-08 | NOTICE_DATE / PLAN_NOTICE_DATE | 是 | 低 x 中 |
| 10 | 股东增持 | RPT_SHARE_HOLDER_INCREASE(DIRECTION=增持) | 每月约 140 行 | 1994 | NOTICE_DATE | 是，6646 行 | 低 x 高（announce 已判负向） |
| 11 | 同花顺强势股题材 | zx.10jqka getharden/date/<日> | 2023-10-25 72 行、2022-06-01 0 行；2024-09-30 693 行（极端日不全） | **2023-10-17** | 收盘后快照；reason 是否事后改写未核 | 是，545 天 4.1 万行，features_ths_theme 13 列 | 中 x 低（不覆盖筛选集） |
| 12 | 十大流通股东 | RPT_F10_EH_FREEHOLDERS | 有 NOTICE_DATE；每季约 5.4 万行；**整月深翻页在报告季月报「服务器繁忙」，要按日拉** | 全历史 | NOTICE_DATE | 是，92.8 万行，features_freeholders 11 列 | 低 x 中 |
| 13 | 人气榜历史 | emappdata getHisList（datasrc 已抢救） | 每只一年 | 2025-10-15 | 当天那行是盘中值 | 否：不覆盖 2022..2025 | 中 x 无 |
| 14 | 涨停池（首封 / 炸板 / 连板） | 东财 push2ex 六池 / 同花顺 limit_up_pool | 东财 30 天；同花顺 2025-10-10 有、2025-09-01 0 行（一年滚动） | 2025-10 | 收盘后 | 否：不覆盖；Tushare limit_list_d 2020 起要 5000 积分 | 高 x 无 |
| 15 | 概念板块成分（带时点） | 东财 push2 clist / 同花顺 | 只有当前成分，无进出历史 | 无 | 无法时点化 -> 回填必偷看 | 否 | 中 x 无 |
| 16 | 开盘啦历史首板 + 原因 + 板块代码 | apphis.longhuvip HisHomeDingPan | oss 调研：2020-07 起，第 12 列概念是当前值（偷看），原因列像当时值（推断） | 2020-07 | 不可核 | 否：无文档、偷看风险，只作题材第二源备选 | 中 x 低 |
| 17 | 基金 / 机构持仓明细 | RPT_MAINDATA_MAIN_POSITIONDETAILS | 每季 9.9 万行，**只有 REPORT_DATE 没有公告日** | | 按报告期对齐就是偷看（教训 25） | 否 | 低 x 低 |
| 18 | 北向个股持股 | RPT_MUTUAL_HOLD_DET | oss 调研：2024-09-30 后为空 | | | 否 | 无 |
| 19 | 个股资金流 | push2his fflow/daykline | 120 天；Tushare moneyflow 2010 起付费 | | | 否 | 中 x 无 |
| 20 | 股权质押 | RPT_CSDC_LIST | 周度约 1000 页（altdata 评低价值没拉） | 2014 | 周五 +3 天 | 否 | 低 x 中 |
| 21 | 可转债 / ETF 申赎 / 个股新闻数 | | 可转债只能给「有没有转债」静态标志；ETF 无个股粒度；新闻无历史 | | | 否 | 低 |

## 2. 特征表（features_<源>.parquet，键 code + date，都是 T 日收盘后可得；详见 features_README.txt）

诊断口径（common_dd.diag_table）：每个特征在三个宇宙里 高组 对 低组 的按天配对 y_open 命中差（pp），NW t 滞后 20，两段各算：
全池 = 候选池全部；前 50 = ret20 前 50；前 50 控名次 = 在 1-10 / 11-30 / 31-50 三档内各切三分位再配对（去掉和 ret20 名次相关的部分）。
三分位 = 当天内上 1/3 对下 1/3；0/1 列和零占比过半的计数列按 >0 对 =0。全部数字在 results_*.txt，汇总在 results_summary.txt（A 段 = 控名次后两段同号 >= 0.8pp）。

### 2.1 features_sw_industry（467 万行，30 列）
- 全池里板块共振全部显著且两段同号：ind2_lu_share +0.84 / +1.06（t 8.2 / 4.5）、ind2_lu_share5 +1.02 / +1.23（t 8.5 / 4.7）、ind2_nh20_share +0.74 / +0.71、ind2_in50_other +0.89 / +1.26。
- 前 50 控名次后：ind2_lu_share +1.9 / −0.5 异号；ind2_lu_share20 +0.6 / −2.2 异号。两段同号的只有 ind2_ret20_excess +2.7 / +0.9（t 3.4 / 0.7）、ind1_lu_n +1.7 / +1.3（t 2.6 / 1.2）、ind2_in50_other 有无 +1.5 / +0.8。
- 读法：行业涨停 / 新高密度决定谁会涨进前 50，ret20 本身已经是这个信息的结果；在前 50 里「板块里只有它一个在涨」（excess 高）比「全板块一起涨」略好。

### 2.2 features_lhb_seats（467 万行，37 列）
- 席位层面：8603 个买方席位，freq250 中位 176、hit_n 中位 106。席位类型按名称（机构专用 / 沪深股通 / 拉萨）。
- 前 50 控名次两段同号：lhbs_today +2.0 / +1.3（t 2.9 / 1.2）、lhbs_age −2.3 / −1.7（t −3.2 / −1.6）、lhbs_hot_buy_n5 有无 +1.7 / +0.9、lhbs_hot_buy_n20 +1.6 / +1.0。
- 席位「质量」没加出东西：famous_max 异号、hit_max 控名次后 −0.5 / +2.7 异号、hit_wavg 异号、机构买入 inst_buy +1.9 / +0.5（t 1.0 / 0.3）、拉萨席位 lasa_buy 控名次后 0。
- 结论：上榜本身（altdata 已有）是全部信息，谁买的分不出来。

### 2.3 features_rating（467 万行，20 列）
- 评级分布：买入 23.2 万、增持 10.4 万、不评级 3.4 万、中性 1777；评级变化「调高」3321、「调低」2387，「首次评级」稀少，调高 / 调低 / 首次三列在前 50 里天数不够。
- 两段同号、控名次后仍在：rat_n250 三分位 −2.7 / −2.6（t −2.4 / −1.3）、rat_strong_n60 有无 −2.3 / −2.6、rat_buy_n60 −2.0 / −2.4、rat_n60 −2.0 / −2.4、rat_n20 −2.7 / −1.6、rat_nocov250 +2.6 / +2.0（t 2.0 / 1.1）。
- 读法：机构覆盖越多越不像能 20 天涨 50% 的票；这是新信息（和市值 / 换手有关联但 pool 里没有这列）。rat_tgt_prem60（目标价溢价）−1.7 / −0.2，没有用。

### 2.4 features_events（467 万行，39 列）
- 前 50 控名次两段同号 >= 0.8pp 的只有：rep_amt_last −3.5 / −4.8（回购上限金额大的差；覆盖 20%）、bon_age −1.6 / −2.2（刚出过送转预案略好）、yjyg_neg_n60 +4.4 / +1.3（60 日内有负向预告反而好，t 3.2 / 0.4）。
- 反向或异号：yjyg_pos_n20 −0.6 / −6.5、yjyg_pos_n60 −0.2 / −2.5、yjkb_np_yoy_last +0.4 / −4.1、inc_plan_n60 −0.5 / −5.0、hinc_n20 −1.8 / +12.8（开发集只 138 天）。
- 股权激励表只从 2023-08 起（覆盖开发集 + 筛选集最后 10 个月）。

### 2.5 features_rzrq_short（467 万行，8 列；T−1 可得）
- rq_zero_yl（两融标的但融券余量为 0）前 50 +3.5 / +3.1（t 2.5 / 1.9），控名次 +3.8 / +2.4；rq_share −1.7 / −3.6，控名次 −1.5 / −4.4；rq_ye_mcap −1.4 / −1.6。
- 融券卖出突增 rq_mcl5_rel +0.9 / −3.6 异号；余量变化 rq_yl_chg5 / 20 异号。
- 读法：有券可借、被融券盯上的票不像；小票无券的像。大概率是规模 / 流动性的代理，但 pool 里的 mcap 三分位在 precision 报告里 1-3 名内是反的，这个更干净，值得给门槛模型。

### 2.6 features_ths_theme（268 万行，13 列；只有 2023-10-17 起，筛选段约 145 天）
- ths_today 全池 +2.9 / +3.8（t 6.6 / 4.8），前 50 控名次 +0.2 / +0.3：题材标签只在「涨停」那一刻有信息，和涨停本身重合。
- ths_lag_heat（最近一次上榜题材今天的热度）前 50 +2.1 / +2.3、控名次 +1.7 / +2.2（t 1.2 / 0.9）；ths_n5 / n20（近 5 / 20 日上榜次数）控名次后 −2.8 / −2.1、−2.3 / −3.1（上得越多越不好）。
- 数据只覆盖开发集，开发集的数不能当确认；只能作为「题材热度」这类特征的形状参考。

### 2.7 features_freeholders（467 万行，11 列）
- 取数坑：整月深翻页在年报 / 季报月（04 / 08 / 10 月，每月 5~10 万行、100~200 页）第 3 次重试仍「服务器繁忙」，11 个月全丢；改按自然日拉（每天 <= 20 页）全部补齐，
  最后 92.8 万行 / 52 块、7.7 万份报告。HOLDER_STATE 只有「新进 / 加仓」两种标签（没有减仓标签，fh_down_n 恒 0）。
- 全池里机构结构非常显著：前十里自然人席位多 fh_person_n +1.0 / +1.9（t 9.7 / 6.7）、机构席位多 fh_org_n −1.2 / −2.2（t −5.8 / −5.7）、基金 fh_fund_n −0.8 / −1.7、
  前十合计占比高 fh_top10_ratio −0.6 / −1.9。前 50 不控名次：fh_org_n −1.8 / −2.5（t −1.6 / −1.7）；控名次后只剩 fh_age +0.9 / +2.1（t 1.1 / 1.4），其余 < 0.8pp 或异号。
- 读法：和研报覆盖同一个故事（机构不在的票像），到前 50 里和 ret20 名次重叠大半。

## 3. 哪些最值得进门槛模型（给 research/gate/）

按「控名次后两段同号、量级、覆盖、不是已有列的改写」排：

1. **研报覆盖**：rat_nocov250、rat_n250、rat_n60（或 rat_strong_n60）。三列高度相关，取一两列。
2. **融券结构**：rq_zero_yl、rq_share（非标的填 NaN，模型当缺失处理）。
3. **龙虎榜时效**：lhbs_age（距上次上榜交易日数）比 0/1 的 lhb_today 信息更连续；lhbs_hot_buy_n5 / n20 次之。
4. **行业内相对强度**：ind2_ret20_excess、ind1_lu_n、ind2_in50_other。全池显著的 ind2_lu_share5 也可进（模型会学到它主要解释名次）。
5. **回购金额** rep_amt_last（负向，覆盖 20%）、送转 bon_age。
6. **股东结构** fh_org_n / fh_person_n / fh_top10_ratio：全池 t 5~10，前 50 内 −1.8 / −2.5，和研报覆盖同源，二选一或都给让模型挑。
7. 题材热度 ths_lag_heat 只能作开发集内的辅助特征，不能用来过线。

不建议进：席位质量三列（famous / hit）、业绩预告正负、快报同比、股权激励、增持、研报目标价溢价、融券卖出突增、前十股东变动（fh_new_n / fh_chg_sum），
两段异号或量级 < 1pp。

提醒：以上都是单变量快速诊断，做了 123 个特征 x 3 个宇宙 = 369 条比较，没有做多重比较校正，「两段同号 + t 2~3」在这个数量下会自然出现几条。
门槛模型按 evalkit 的过线 v3 走时，这里的任何一条都不能算预登记证据。

## 4. 没拉 / 不值得的原因

- 人气榜、涨停池、资金流、千股千评：免费源只有近期（已由 tools/archive_events.py 每日存，一年后才够一段）。
- 概念板块成分：无时点历史，回填必偷看；替代是申万时点行业（已做）和同花顺题材文本（2023-10 起）。
- 开盘啦首板历史：概念列是当前值，原因列真假没法核，无文档。
- 基金持仓明细 / 通达信财务包：只有报告期，按报告期对齐就是教训 25。
- 东财 reportapi 全市场研报：按日只给巨潮的 1/5，疑似只给白名单机构。
- 质押：周度 1000 页，先验低，altdata 没拉，这里也没拉。
- 付费：Tushare limit_list_d（涨跌停列表 2020 起）、moneyflow（2010 起）、新闻权限，记为选项。

## 5. 文件（research/deepdata/）

probe_dd.py（单次实测，probe_result.jsonl）、fetch_dd.py（东财 datacenter 分块拉取，可续跑；merge 子命令）、fetch_fh_days.py（十大流通股东缺月逐日补）、
fetch_cninfo.py（巨潮评级逐日）、fetch_ths.py（同花顺题材逐日）、common_dd.py（候选池 / 网格 / 落盘 / 诊断）、feat_sw.py / feat_lhb_seats.py / feat_rating.py / feat_events.py /
feat_rzrq_short.py / feat_ths.py / feat_freeholders.py、rediag.py（只重跑诊断）、summary_dd.py（results_summary.txt）、
features_*.parquet + features_README.txt、results_*.txt、progress.txt、fetch*.log、raw/（gitignore）。
纪律：所有请求 20~30 秒超时、重试 3 次、连续失败 5 次全局停 10 分钟、连停 3 次放弃；东财并发 2、巨潮 / 同花顺单线程；
每个请求前查北京时间禁区（research/datasrc/common.check_blackout）。
