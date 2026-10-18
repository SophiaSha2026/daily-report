# research/deepdata 特征表说明（键 code 6 位 + date 交易日；全部 T 日收盘后可得；覆盖 pool.parquet 2021-12-01..2025-12-31 全部行，不只候选池）
# 诊断见 results_<源>.txt / results_summary.txt，取数纪律见 report.txt
## features_sw_industry.parquet（4669348 行，2021-12-01..2025-12-31，28 列）
来源：申万行业归属变迁表（带计入日期，时点可知）+ pool.parquet 的 T 日收盘价量。全部为 T 日收盘后可得。
覆盖：pool.parquet 全部行（2021-12-01..2025-12-31）；行业缺失（新股 / 表里没有）的行所有行业列为 NaN。
列（ind2_ = 申万二级，ind1_ = 申万一级；每个行业当天在 pool.parquet 里的成分）：
  sw_ind2 / sw_ind1        T 日生效的申万二级 / 一级代码（字符串；2021-07-30 前为 2014 版口径）
  *_n                      同行业当天只数
  *_lu_n / *_lu_share      同行业当天收盘涨停只数 / 占比（lu_close，含自己）
  *_lu_share5 / 20         同行业涨停占比近 5 / 20 个交易日累计（板块持续活跃度）
  *_nh20_share             同行业收在 20 日新高的占比
  *_ret20_med / *_ret5_med 同行业 ret20 / ret5 中位数（行业动量）
  *_ret20_pct / *_ret5_pct / *_lu_pct   该行业的 ret20 中位 / ret5 中位 / 涨停占比 在当天全部行业（成分 >= 5 只）里的百分位
  ind2_in50_other          同二级行业里其他在 ret20 前 50 的只数（候选池内名次）
  ind2_lu_other            同二级行业里其他涨停的只数
  ind2_ret20_rank_in       自己 ret20 在同二级行业内的百分位（1 = 行业龙头）
  ind2_ret20_excess        自己 ret20 减行业中位
缺失含义：NaN = 没有行业归属或行业内无人有值；计数列 0 = 真 0。

## features_rzrq_short.parquet（4669348 行，2021-12-01..2025-12-31，8 列）
来源：东财两融个股明细（research/altdata/raw/rzrq.parquet，不重复拉）融券那一侧。
时点：交易所 T+1 早上发布，所有列按 DATE = T 的上一个交易日取值（整体错后一天），T 日收盘可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；非两融标的全部 NaN（rq_target = 0）。
列：
  rq_target        是否两融标的（上一交易日有两融行）
  rq_yl_chg5 / 20  融券余量 5 / 20 个交易日变化率
  rq_share         融券余额 / (融资余额 + 融券余额)
  rq_ye_mcap       融券余额 / 流通市值（SZ 列）
  rq_mcl5_rel      近 5 日融券卖出量合计 / 当前融券余量
  rq_mcl5_vs20     近 5 日融券卖出量 / (近 20 日融券卖出量 / 4)，融券卖出突增倍数
  rq_zero_yl       融券余量为 0（标的但无券）
缺失含义：NaN = 非标的或分母为 0。

## features_ths_theme.parquet（2684779 行，2023-10-17..2025-12-31，12 列）
来源：同花顺「强势股题材」（zx.10jqka.com.cn getharden），每只涨停 / 强势票当天的题材原因文本，2023-10-25..2025-12-31
（更早没有；筛选集只覆盖 2023-10-25..2024-05）。题材 = reason 按「+」切词，去掉「市场超500股涨停」「超跌反弹」等泛词。
时点：收盘后快照，T 日那行可用；最近一次上榜的题材沿用最多 20 个交易日。reason 是否事后改写未核。
覆盖：pool.parquet 从 2023-10-25 起的全部行（2023-10-25 之前这张表没有行）。
列：
  ths_today / ths_n5 / ths_n20 / ths_age   当天在强势股名单 / 近 5、20 日次数 / 距最近一次的交易日数（没有 = NaN）
  ths_theme_max / ths_theme_sum / ths_ntok  最近一次上榜那天：同题材其他强势股只数（最热的那个词 / 全部词合计）/ 词数（只在上榜日更新，其余 NaN 或旧值；配 ths_age 用）
  ths_lag_heat / ths_lag_heat_other         最近一次（<= 20 日）上榜题材在今天的热度（带该词的强势股只数，含 / 不含自己）；20 日内没上过榜 = NaN
  ths_lag_heat_share                        ths_lag_heat_other / 当天最热题材只数
  ths_mkt_n / ths_mkt_topheat               市场级：当天强势股总数 / 最热题材只数
缺失含义：NaN = 20 日内没上过强势股名单（没有题材标签）；0 = 上过但题材今天没人。

## features_lhb_seats.parquet（4669348 行，2021-12-01..2025-12-31，28 列）
来源：东财龙虎榜买方 / 卖方席位明细（RPT_BILLBOARD_DAILYDETAILSBUY / SELL），2021-09..2025-12，同一 (票, 日, 席位) 多条原因只算一次。
时点：T 日收盘后交易所公布，T 日那行可用；席位战绩只用 s <= T-21 的历史标签（窗口已关）。
席位统计全部按时点自算（freq250 = 该席位此前 250 个交易日买方上榜次数；hit_rate = 此前 250 日买方上榜票的 y_open 命中率，
收缩 (hits + 0.8) / (n + 10)；hot = freq250 >= 60）。席位类型按名称：机构专用 / 沪深股通 / 拉萨。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31。不上榜的日子 lhbs_today = 0，当日席位列 NaN，窗口计数列 0。
列：
  lhbs_today / lhbs_age                 当天是否上榜（买方或卖方席位表里有）/ 距最近一次上榜的交易日数（没有 = NaN）
  lhbs_n_buy / inst_buy / north_buy / lasa_buy / hot_buy   当天买方席位数 / 机构专用数 / 沪深股通数 / 拉萨席位数 / 活跃游资席位数
  lhbs_famous_max / famous_sum          买方席位 freq250 最大 / 合计
  lhbs_hit_max / hit_wavg               买方席位收缩后历史命中率的最大 / 按买入额加权均值
  lhbs_top_ratio / buy_share            第一大买方占当日成交额比例 / 买方五席合计占比
  lhbs_inst_sell / lasa_sell / famous_sell / sell_share   卖方：机构专用数 / 拉萨数 / freq250 合计 / 五席合计占比
  lhbs_net_share                        买方合计占比 − 卖方合计占比
  lhbs_{inst_buy,hot_buy,north_buy}_n5 / _n20   近 5 / 20 个交易日累计（不上榜算 0）
  lhbs_famous_max_5 / lhbs_hit_max_5    近 5 个交易日上榜日里的最大值（5 日内没上榜 = NaN）
  lhbs_net_share_5 / lhbs_inst_sell_n5  近 5 日净占比合计 / 机构卖出席位数合计
缺失含义：当日席位列 NaN = 当天不在龙虎榜；lhbs_age NaN = 2021-09 以来没上过榜。

## features_rating.parquet（4669348 行，2021-12-01..2025-12-31，17 列）
来源：巨潮资讯「投资评级」按日全市场（webapi.cninfo p_sysapi1089，akshare stock_rank_forecast_cninfo），2021-09-01..2025-12-31 逐自然日；
东财 reportapi 全市场按日只给约 1/5（2024-03-05：28 对 145），没用。
时点：发布日期，非交易日算下一交易日，发布日收盘起可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；250 日窗口在 2022-08 之前不完整（表从 2021-09 起）。
列：
  rat_today / rat_age                   当天有研报 / 距最近一篇研报的交易日数（表里从没有过 = NaN）
  rat_n5 / n20 / n60 / n250             近 5 / 20 / 60 / 250 个交易日研报篇数（同一天多篇都算）
  rat_first_n20 / n60                   「首次评级」篇数（巨潮字段「是否首次评级」）
  rat_up_n20 / n60 / rat_down_n60       评级变化含「调高 / 上调」/「调低 / 下调」的篇数
  rat_buy_n60 / rat_strong_n60 / rat_buy_share60   60 日内正面评级篇数（买入 增持 推荐 优于大市 等）/ 买入级篇数 / 正面占比
  rat_nocov250                          一年没有任何研报（无人覆盖）
  rat_surge                             近 20 日篇数 / (近 250 日篇数 / 12.5)，研报关注度突增倍数（一年没研报 = NaN）
  rat_tgt_prem60                        60 日内最近一篇给了目标价的：目标价上限 / 收盘 − 1（没有 = NaN）
缺失含义：计数列 0 = 真 0；rat_age NaN = 2021-09 以来无研报。

## features_events.parquet（4669348 行，2021-12-01..2025-12-31，32 列）
来源：东财 datacenter 六张事件表（raw/yjyg RPT_PUBLIC_OP_NEWPREDICT、yjkb RPT_FCI_PERFORMANCEE、repurchase RPTA_WEB_GETHGLIST_NEW、
bonus RPT_SHAREBONUS_DET、incentive RPT_EQUITYINCENTIVE、holder_inc RPT_SHARE_HOLDER_INCREASE(DIRECTION=增持)），2021-09..2025-12。
时点：一律按公告日（NOTICE_DATE / NOTICEDATE / PLAN_NOTICE_DATE），落在非交易日算下一交易日，公告日收盘起可用。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；事件表从 2021-09 起，所以 2021-12 的 60 日窗口完整、250 日窗口不完整。
列：
  yjyg_age / yjyg_n20 / yjyg_n60        距最近一次业绩预告的交易日数（无 = NaN）/ 近 20、60 个交易日公告次数
  yjyg_pos_n20 / yjyg_pos_n60 / yjyg_neg_n60   正向（预增 扭亏 略增 续盈）/ 负向（预减 首亏 续亏 略减）预告次数
  yjyg_amp_last / yjyg_last_pos         最近一次预告的同比变动幅度上限（%）/ 是否正向
  yjkb_age / yjkb_n20 / yjkb_n60 / yjkb_np_yoy_last / yjkb_rev_yoy_last   业绩快报：距上次 / 次数 / 最近一次净利润同比 % / 营收同比 %
  rep_age / rep_n20 / rep_n60 / rep_plan_n60 / rep_share_last / rep_amt_last   回购：距上次公告 / 次数 / 60 日内首次公告（预案）次数 /
                                        最近一次回购上限占总市值 % / 金额上限（亿元）
  bon_age / bon_n60 / bon_high_n60 / bon_it_last   分红送转预案：距上次 / 60 日内次数 / 60 日内 10 送转 >= 5 的次数 / 最近一次每 10 股送转数
  inc_age / inc_n60 / inc_plan_n60 / inc_plan_n250 / inc_px_prem   股权激励：距上次公告 / 60 日内公告数 / 60、250 日内新预案数 /
                                        现价对最近一次授予(行权)价的溢价（close/价 − 1）
  hinc_age / hinc_n20 / hinc_n60 / hinc_ratio60   股东增持：距上次公告 / 次数 / 60 日内增持占流通股比例合计（%）
缺失含义：*_age NaN = 表里 2021-09 以来没有过公告；计数列 0 = 真 0；*_last NaN = 没有过公告或该次没给数值。

## features_freeholders.parquet（4669348 行，2021-12-01..2025-12-31，11 列）
来源：东财十大流通股东（RPT_F10_EH_FREEHOLDERS），每份定期报告一组，2021-09..2025-12 按公告日拉。
时点：NOTICE_DATE 公告日收盘起可用，之后沿用到下一份报告；报告期不用于对齐。
覆盖：pool.parquet 全部行 2021-12-01..2025-12-31；2021-09 之前没有报告的票在首份公告前为 NaN。
列（都是「最近一份已公告报告」的值）：
  fh_age            距最近一份报告公告日的交易日数
  fh_fresh20        最近一份报告在 20 个交易日内公告（刚出季报 / 年报）
  fh_top10_ratio    前十大流通股东持股占流通股比例合计（%）
  fh_top10_dratio   对上一份报告的变化（百分点）
  fh_new_n / fh_up_n / fh_down_n   前十里新进 / 增持 / 减持 的席位数
  fh_org_n / fh_fund_n / fh_person_n   机构席位数 / 基金社保保险等席位数 / 自然人席位数
  fh_chg_sum        前十持股变动比例合计（东财 CHANGE_RATIO 之和）
缺失含义：NaN = 2021-09 以来没有可用报告。
