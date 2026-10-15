# 特征文献目录：说明（2026-10-15）

（研究子代理写成，子代理不能写 .md，由主会话落盘。结构化数据在 catalog.json，纯文本说明在 catalog.txt。）

## 摘要

- catalog.json 55 条（F001~F055），前 41 条按「和目标的相关性 × 不重叠 × 可算」排序，分 10 个族；14 条 priority 3 是和 Alpha158 / 生产特征高度重叠的，列出来只为说明不必再做；另列 13 项需要新数据才能算的。
- 出处核到哪一层每条都标了：25 条读了公式表或复现代码，19 条只核到摘要，8 条自拟，3 条凭记忆（Hou-Moskowitz 价格延迟 D1、股价同步性、预期特质偏度），实现前必须核对原文。
- **最重要的判断：目标是右尾概率，不是收益排序。** 文献里对均值收益为负的彩票类因子（MAX、偏度、特质波动、小盘、低价），对「20 天涨 50%」应当是正向的。符号交给模型学，不能照搬文献。

## 1. 范围和方法

- 能用的数据：daily.parquet 的前复权 open/high/low/close、原值 volume/amount、turnover、outstanding_share，另加股东户数、自算筹码和板块。
- 实读：qlib loader.py（Alpha158 / Alpha360 全部定义）；WQ101 的 101 条（yli188 复现代码里逐条抄录的原式）；国君 191（wukan1986/alpha_examples 公式表）；QuantsPlaybook 里球队硬币、凸显理论 STR、均线收敛（开源 91）、理想振幅（开源 7）、振幅切割动量（开源 3）、W 式切割（开源 1）、上下影线（东吴）、CGO、特质波动、高质量动量、APM、筹码四阶矩、APB 的复现。
- 只核到摘要：Bali-Cakici-Whitelaw 2011、Conrad-Kapadia-Xing 2014、Lou-Polk-Skouras 2019、George-Hwang 2004、Blitz-Huij-Martens 2011、Gervais-Kaniel-Mingelgrin 2001，以及几篇 A 股 MAX 效应研究。

## 2. 三个因子库的结论

- **WQ101**：平均持有期 0.6~6.4 天，超短周期的均值回归和量价相关，和 20 天右尾关系弱；48 条用到 VWAP 或行业中性化（18 条要行业，我们没有时点一致的行业）；大量 rank(correlation) 和 Alpha158 的 CORR / CORD 同类，而量能族已证无增益。只收 #101、#55 做对照。
- **国君 191**：4 条要基准指数，#30 要 FF 三因子，12 条算子没实现；剩 174 条里 36 条带 VWAP。收了 #23、#11/#60、#109、#188、#118、#110、#144、#166，其余和 Alpha158 只差平滑方式。
- **Alpha158**：altpath 已全做。它缺：隔夜和日内收益拆分、三四阶矩、日收益极值、对市场的残差、市值和价位、换手在自身历史里的分位、60 日以上窗口、按价位或振幅切割的统计量。目录主要补这几块。
- **Alpha360**：60 天原始序列，只适合序列模型，priority 3。

## 3. 三条实现约束

1. 量能一律用换手率，不用成交股数（原值，送转日整段跳升，伪造放量）。
2. 不用前复权的绝对价位，也不用前复权收盘价乘流通股本当市值（前复权价会被之后的分红送转整体改写，等于偷看）。价位和市值用当天不复权均价 AMT/V：`nominal_price = log(AMT/V)`，`size_raw = log(FS × AMT/V)`。
3. VWAP 是不复权价，不能和前复权收盘价直接比；带 VWAP 的公式改用典型价 (H+L+C)/3，或只在窗口不跨除权日时用 VWAP 自身的比值。

## 4. 总排序

**priority 1（前 10）**：

1. F018 past_jackpot（极端收益族）：过去 250 天已关闭的窗口内是否出现过「20 天涨 50%」（Conrad 2014 jackpot 持续性）。偷看风险最高，窗口终点必须不晚于 t，要用自测钉住边界。
2. F016 skew_ret：20 / 60 / 120 日偏度（国君 #166、Conrad）。
3. F015 max_ret：20 日最大日收益、前 k 大日收益均值（Bali 等）。主板被 10% 封顶，要和板块一起看。
4. F032 idio_vol：剔除市场和板块后的特质波动。
5. F001 ovn_mom：隔夜收益均值（Lou-Polk-Skouras）。
6. F048 size_raw：不复权流通市值（生产把市值中性化掉了，模型看不到「它有多小」）。
7. F031 resid_mom：残差动量（Blitz-Huij-Martens）。
8. F040 ideal_amplitude：理想振幅（开源 7）。
9. F002 intra_mom：日内收益均值。
10. F008 pcf_conv：多根均线离散度（开源 91），除以收盘价归一。

**priority 2（11~41）**：隔夜距离、换手自身一年分位、流通股本变化、缩量新高、上行半方差占比、APM 隔夜 / 日内版、非流动性、价格延迟、球队硬币、凸显理论、不复权价位、上市年龄、未回补跳空、低振幅日动量、偏度 × 远离高点、去趋势换手、换手率均线收敛、平台突破、NR7 / 连续收缩、振幅冲击、距 120 / 250 日高点天数、120 / 500 日高点距离、筹码三四阶矩、上下影线、开盘后上攻力度、CLV 加权累计、股价同步性、特质偏度、隔夜日内拔河、长期换手水平、买卖压力。编号以 catalog.json 为准。

**priority 3（不建议先做）**：kurt_ret、min_ret、hv_premium、corr_ret_dtr、range_accel、cgo、wq_101、hq_momentum、info_discreteness、tr_stability、true_range_up、wq_rsv_vol_corr、a360_seq；skew_ret 和 idio_skew 二选一。

## 5. 和 altpath 结论的衔接

altpath 的增益在价格位置族和波动趋势族，量能族没贡献，所以量能类压到 priority 3；放在前面的是 Alpha158 结构上没有的：高阶矩、日收益极值、残差化、隔夜和日内分离、市值、按价位或振幅切割。建议按族拆臂，每臂并进 base_a158 做增量，不要一次全并（altpath 换种子时增益已缩三成）。

## 6. 需要新数据、值得以后存档

APM 原版、理想反转 / 大单日、CPV 等分钟线因子、聪明钱、尾盘 / 开盘成交占比（都要分钟线）；t 日集合竞价量价；涨停封板质量（涨停池只有 30 天，正在抢救）；龙虎榜游资席位；个股资金流（只能回溯 120 天）；两融余额；时点一致的概念成分；限售解禁日程；人气榜、千股千评、搜索热度。

## 7. 还没核实的

- 价格延迟 D1 的公式凭记忆写。
- Conrad jackpot logit 的去趋势换手窗口没读到原文 Table 3。
- STR 的复现代码是截面排序，原文是个股时间序列，实现前要核对。
- Wang-Wang-Wu 2023（博彩偏好只出现在远离高点的票里）只看了检索摘要。

主要出处：qlib loader.py、101 Formulaic Alphas（arXiv 1601.00991）、yli188/WorldQuant_alpha101_code、wukan1986/alpha_examples、hugo2046/QuantsPlaybook、Tug of War（LSE）、Death and Jackpot（JFE 2014）、Residual Momentum（EUR）、High-Volume Return Premium（JF 2001）。
