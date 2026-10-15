"""生成 research/features/catalog.json。因子列表在 FACTORS 里，按族写。"""
import json
import sys

OUT = sys.argv[1]

WQ = "https://arxiv.org/abs/1601.00991"
WQ_CODE = "https://github.com/yli188/WorldQuant_alpha101_code/blob/master/101Alpha_code_1.py"
GTJA = "https://github.com/wukan1986/alpha_examples/blob/main/transformer/alpha191.txt"
QLIB = "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py"
QP = "https://github.com/hugo2046/QuantsPlaybook/tree/master/B-因子构建类/"

NOTATION = {
    "O,H,L,C": "前复权开高低收（daily.parquet open/high/low/close）",
    "V": "成交股数 volume（原值，不复权，送转日跳升；跨除权比较一律改用 TR）",
    "AMT": "成交额 amount（原值）",
    "TR": "换手率 turnover（%），送转前后可比，量能默认用它",
    "FS": "流通股本 outstanding_share（原值，逐日）",
    "r": "r_t = C_t/C_{t-1} - 1",
    "ovn": "隔夜收益 ovn_t = O_t/C_{t-1} - 1（同一前复权序列内比值，除权不影响）",
    "intra": "日内收益 intra_t = C_t/O_t - 1",
    "mkt": "当日全市场等权平均（cs_mean），不用指数，避免另取数据",
    "brd": "当日同板块（主板/创业/科创/北交）等权平均",
    "rank_cs": "当日全市场排名归一到 [0,1]（和 altpath 的 Alpha158 处理一致）",
    "SMA(x,n,m)": "国君 191 的递归平滑 y_t = (m*x_t + (n-m)*y_{t-1})/n",
    "DECAYLINEAR(x,d)": "线性衰减加权平均，权重 d..1",
    "TSRANK(x,d)": "x 在自己过去 d 天里的分位",
    "VWAP": "AMT/V 是不复权均价；前复权 C 和它不可比（altpath 因此删了 VWAP0）。可用的只有窗口内不跨除权日的 VWAP 自身比值，或用典型价 TP=(H+L+C)/3 代理",
}

F = []


def add(family, name, formula, windows, source_url, rationale, overlap, cost,
        lookahead_risk, priority, origin, notes=""):
    F.append(dict(id=f"F{len(F)+1:03d}", family=family, name=name, formula=formula,
                  windows=windows, source_url=source_url, origin=origin,
                  rationale=rationale, overlap=overlap, cost=cost,
                  lookahead_risk=lookahead_risk, priority=priority, notes=notes))


LOW = "低：只用 t 日收盘及以前的 OHLCV"
LOW_ADJ = "低：只用比值，前复权的整体缩放不影响；不要用绝对价位"

# ---------------- A 隔夜 / 日内分离 ----------------
FA = "A_隔夜日内分离"
add(FA, "ovn_mom", "MEAN(ovn, d)，ovn_t = O_t/C_{t-1}-1", [5, 20, 60],
    "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
    "Lou, Polk, Skouras (2019 JFE)：隔夜和日内收益各自多年持续、互相反转；动量利润几乎全在隔夜，"
    "隔夜由散户 / 开盘集合竞价推动。妖股启动前常见连续高开（散户追价在竞价里体现），"
    "隔夜累计是「散户关注度」的日线代理。",
    "Alpha158 只有当天 OPEN0=O/C，没有隔夜累计；生产特征没有。不重叠。",
    "低（一次滚动均值）", LOW, 1, "学术")
add(FA, "intra_mom", "MEAN(intra, d)，intra_t = C_t/O_t-1", [5, 20, 60],
    "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
    "同上，日内段由机构主导。和隔夜同时入模，模型可学两者的组合（ROC = 隔夜 + 日内的复合）。",
    "Alpha158 KMID 只有当天；ROC 是两者之和。中度重叠（ROC 已知时只多一个自由度）。",
    "低", LOW, 1, "学术")
add(FA, "ovn_intra_tug", "rank_cs(SUM(ovn,20)) - rank_cs(SUM(intra,20))", [20],
    "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
    "「拔河」：隔夜涨、日内跌说明散户追、机构卖；对 20 天涨 50% 这种散户驱动行情，"
    "方向可能和美股相反（隔夜强的票更可能被炒作），让模型自己学符号。",
    "和 ovn_mom / intra_mom 线性相关，二选一或都给树模型。",
    "低", LOW, 2, "学术")
add(FA, "ovn_dist", "MEAN(|ovn_t - mkt(ovn_t)|, 20)（方正「球队硬币」的隔夜距离）", [20],
    QP + "个股动量效应的识别及球队硬币因子",
    "隔夜涨跌幅全市场最平静，个股偏离市场均值越多，说明隔夜有个股层面的信息 / 关注度冲击。"
    "极端上涨前的「异动」常先出现在开盘。",
    "不重叠（Alpha158 没有隔夜，生产没有）。",
    "低", LOW, 2, "券商（方正金工，QuantsPlaybook 复现）")
add(FA, "coin_flip_ret",
    "MEAN(r_t * sign(ΔTR_t - mkt(ΔTR_t)), 20)，ΔTR_t = TR_t - TR_{t-1}；"
    "同法可换 ovn / intra（方正「球队硬币」换手率翻转版）", [20],
    QP + "个股动量效应的识别及球队硬币因子/FactorZoo/SportBetting.py",
    "方正金工：换手率变化低于市场均值的日子视为「硬币」（无信息、该反转），高于的视为「球队」（有信息、该延续）。"
    "把收益按是否放量加签名后再平均，区分「放量涨」和「缩量涨」。",
    "和 Alpha158 CORD（收益和量变化的相关）概念近，但这里是加签名均值；生产 up_vol_share20 部分重叠。中度。",
    "低", LOW, 2, "券商（方正金工）",
    "复现代码 get_coins_team：factor * sign(baseline - cs_mean)，opr='lt'。原报告是月频选股，日频滚动 20 天是我们的改法。")
add(FA, "ovn_vol_share", "STD(ovn,20) / (STD(ovn,20)+STD(intra,20))", [20],
    "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf",
    "波动集中在隔夜说明信息 / 情绪在盘后到达（消息、竞价抢筹），集中在日内说明盘中博弈。",
    "Alpha158 STD 是收盘价波动，不拆段。不重叠。",
    "低", LOW, 3, "自拟（基于 Lou-Polk-Skouras 拆分）")
add(FA, "gap_up_cnt", "COUNT(L_t > H_{t-1}, 20)（未回补的跳空缺口天数）；另一列 COUNT(O_t/C_{t-1}-1 > 0.03, 20)", [20, 60],
    "自拟（经典技术分析：突破缺口）",
    "强势股启动时出现不回补的跳空，是抢筹的痕迹。和 20 天涨 50% 的连板行情直接相关。",
    "涨停生态 lu 列（近 20 天封板次数）部分重叠：一字连板必然跳空。中度。",
    "低", LOW, 2, "技术分析")

# ---------------- B 波动收缩与突破 ----------------
FB = "B_波动收缩与突破"
add(FB, "pcf_conv", "-LOG(1 + STD([C, MA5, MA10, MA20, MA60, MA120]) / C)（多根均线的离散度，按 C 归一）", [120],
    QP + "开源证券-开源量化评论（91）：形态识别，均线的收敛与发散/FactorArithmetic/convergence_factor.py",
    "开源证券（91）均线收敛：均线粘合越紧，形态越「收」，后续突破概率越大。妖股启动前多是长期横盘、均线粘合。",
    "生产 ma_align（排列方向）、boll_width（20 日）概念相邻但不同：这里是 6 根均线的离散度。Alpha158 MA 是单根乖离。低到中度。",
    "低", LOW_ADJ, 1, "券商（开源证券，QuantsPlaybook 复现）",
    "复现代码直接对后复权价求 std，没有按价格归一，价位高的票天然大；我们必须除以 C，否则是价位因子。")
add(FB, "trcf_conv", "-LOG(1 + STD([TR, MA5(TR), MA10(TR), MA20(TR), MA60(TR), MA120(TR)]) / MA120(TR))", [120],
    QP + "开源证券-开源量化评论（91）：形态识别，均线的收敛与发散/README.md",
    "换手率均线收敛（TRCF）：成交长期平稳、没有异动，是「冷清很久」的连续版，和长期调整突破的横盘段同一个意思。",
    "生产 vol_compress、turn_std20 部分重叠（20 日窗口）；这里 120 日多均线。中度。",
    "低", LOW, 2, "券商（开源证券）")
add(FB, "range_accel", "SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2)（国君 Alpha109，用 (H-L)/C 代替 H-L 去价位）", [10],
    GTJA, "振幅相对自己均值的加速：收缩后第一次扩张。",
    "生产 atr_ratio（ATR 短 / 长）高度相近。中到高度。",
    "低", LOW_ADJ, 3, "国君 191 #109")
add(FB, "range_shock", "((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2)（国君 Alpha188）", [11],
    GTJA, "今天振幅相对近 11 天平滑振幅的冲击。突破日通常振幅放大。",
    "Alpha158 KLEN（当天振幅/开盘）只看当天水平，这里是相对自己。低到中度。",
    "低", LOW_ADJ, 2, "国君 191 #188")
add(FB, "nr_rank", "TSRANK((H-L)/C_{t-1}, 7)，以及「近 d 天振幅单调收缩的次数」 SUM(rng_t < rng_{t-1}, d)", [7, 20],
    "自拟（Crabel NR7 / Minervini VCP 收缩次数）",
    "VCP（波动收缩形态）：每次回调幅度比上一次小。NR7 是今天振幅为 7 日最小。都在描述「弹簧压紧」。",
    "生产 range_compress / atr_ratio 是比值，没有「连续收缩次数」。低到中度。",
    "低", LOW_ADJ, 2, "技术分析")
add(FB, "box_breakout",
    "box_w = MAX(H,60)_{t-1}/MIN(L,60)_{t-1} - 1；brk = C_t / MAX(H,60)_{t-1} - 1；交互 brk × (1/box_w)", [60, 120],
    "自拟（平台突破，和长期调整突破同源）",
    "窄平台 + 收盘突破平台上沿。宽平台的突破和窄平台的突破含义不同，交互项让树更容易学。",
    "Alpha158 MAX60 / RSV60 和生产 range_compress 各覆盖一半，交互项没有。中度。",
    "低", LOW_ADJ, 2, "技术分析")

# ---------------- C 极端收益与偏度 ----------------
FC = "C_极端收益与偏度"
add(FC, "max_ret", "MAX(r, 20)；以及前 k 大日收益的均值 MAXk = MEAN(top_k(r, 20))，k=3,5", [20, 60],
    "https://alphaarchitect.com/hot-off-the-jfe-press-maxing-out-your-returns/",
    "Bali, Cakici, Whitelaw (2011 JFE)：MAX 高的票未来均值收益低（彩票偏好），但我们的目标是「极端上涨的概率」"
    "而不是均值：MAX 刻画的正是右尾厚度，对 P(20 天涨 50%) 应当是正向。A 股按板块涨跌停截断，"
    "MAX 在主板被 10% 封顶，要和板块一起看（或用 MAX / 板块涨停幅度）。",
    "Alpha158 没有日收益的最大值（MAX 是最高价/收盘）；lu 列有涨停次数。中度。",
    "低", LOW, 1, "学术")
add(FC, "skew_ret", "SKEW(r, d)；国君 Alpha166 是 20 日偏度的一个写法", [20, 60, 120],
    GTJA,
    "历史偏度是 Conrad-Kapadia-Xing (2014) 预测 jackpot（一年内对数收益 > 100%）的主变量之一（另几个是过去一年收益、波动、销售增长）。"
    "右偏的票更容易再来一次极端上涨。",
    "Alpha158、生产都没有三阶矩。不重叠。",
    "低", LOW, 1, "学术 + 国君 191 #166")
add(FC, "kurt_ret", "KURT(r, d)", [60, 120], "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "厚尾程度，和偏度互补：对称厚尾 vs 右偏。",
    "不重叠。", "低", LOW, 3, "学术")
add(FC, "upvol_share", "SMA(r>0 ? STD(C,20) : 0, 20, 1) / (同式 + SMA(r<=0 ? STD(C,20):0, 20, 1))（国君 Alpha23，相对波动指数 RVI）；"
    "等价的简洁版：上行半方差 / 总方差 = SUM(r^2·[r>0],20)/SUM(r^2,20)", [20, 60],
    GTJA, "上行半方差占比：波动主要来自上涨日，说明右尾在发力。",
    "Alpha158 SUMP 是涨幅绝对值之和占比（一阶），这里是二阶。中度。", "低", LOW, 2, "国君 191 #23")
add(FC, "past_jackpot",
    "过去 250 天里「已关闭」的窗口内是否出现过 20 天涨 50%：MAX_{s ≤ t-20} [MAX(H_{s+1..s+20})/O_{s+1} - 1]，"
    "取 s 在 [t-250, t-20]；另列次数 COUNT", [250],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "Conrad et al. 2014：jackpot 概率高的股票过去偏度高、过去收益高、波动高。A 股的「妖股基因」：同一只票反复成为游资标的。"
    "这是把目标标签本身的历史（只用已关闭窗口）当特征。",
    "lu 列 l_cnt60（涨停次数）部分重叠。中度。",
    "中（每只票滚动 max，O(n·20)）",
    "高：s+20 必须 ≤ t，否则用到未来最高价。写成 shift(20) 之后再取 rolling max，自测钉住边界", 1, "学术（自拟落地）")
add(FC, "min_ret", "MIN(r, 20)（最大单日跌幅，Conrad 的 death 一侧）", [20],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "Conrad et al.：death 和 jackpot 概率正相关（同一批高风险票）。跌停过的票也是游资票。",
    "Alpha158 MIN 是最低价/收盘（价格位置），不是日收益。低。", "低", LOW, 3, "学术")

# ---------------- D 价格延迟与反应不足 ----------------
FD = "D_价格延迟与反应不足"
add(FD, "price_delay_d1",
    "日线改写：60 天内回归 r_t = a + b0·mkt_t + Σ_{k=1..5} b_k·mkt_{t-k} + e（无约束）和只含 b0 的约束回归；"
    "D1 = 1 - R²_约束 / R²_无约束", [60, 120],
    "https://arxiv.org/abs/1903.01655",
    "Hou & Moskowitz (2005 RFS)：价格对市场信息反应越慢（延迟越大）的票，未来收益越高，且多是小、冷门、少人关注的票。"
    "冷门票一旦被关注，补涨幅度大，正是妖股的起点。",
    "生产 rs20/rs60（相对强弱）、Alpha158 都没有。不重叠。",
    "中（每只票滚动 OLS 两次，可用 numpy 累积和向量化）", LOW, 2, "学术",
    "原文用周收益、4 阶滞后；日频改写的滞后阶数要预登记。D1 定义待读原文核实（检索结果只确认了思路）。")
add(FD, "amihud", "MEAN(|r_t| / AMT_t, 20)，取对数；国君 Alpha144 是只算下跌日的版本", [20, 60],
    GTJA,
    "Amihud (2002) 非流动性：单位成交额推动的价格变化。小资金就能推动的票，被游资拉 50% 的成本低。",
    "生产 amt_ma20（成交额水平）、市值相关；中度。",
    "低", LOW, 2, "学术 + 国君 191 #144")
add(FD, "synchronicity", "60 天回归 r = a + b·mkt + c·brd + e 的 R²，取 log(R²/(1-R²))", [60],
    "自拟（Morck-Yeung-Yu 2000 的股价同步性）",
    "R² 低 = 股价由个股信息驱动、跟大盘不同步，独立行情的前提。",
    "Alpha158 RSQR 是对时间趋势的 R²，不是对市场。不重叠。",
    "中", LOW, 2, "学术")

# ---------------- E 换手率结构 ----------------
FE = "E_换手率结构"
add(FE, "float_chg", "FS_t / FS_{t-d} - 1（流通股本变化：解禁、增发上市）", [20, 60, 120],
    "自拟（数据特有）",
    "流通盘突然扩大 = 供给冲击，压制炒作；流通盘长期不变的小盘票是游资偏好。",
    "生产没有；Alpha158 没有。不重叠。注意送转也会让 FS 跳升（价格已前复权），要用 os_jump 同样的门槛区分。",
    "低", "低：FS 是当日已知量", 2, "数据特有")
add(FE, "tr_long_level", "LOG(MEAN(TR, 250))，以及 MEAN(TR,20)/MEAN(TR,250)", [250],
    "自拟",
    "长期换手水平是「平时有没有人玩」的底色，短/长比是关注度的突变。",
    "生产 vol_ratio20 是 20 日；turn_pct 是当天换手。250 日底色不重叠。",
    "低", LOW, 2, "自拟")
add(FE, "hv_premium",
    "今天 TR 在过去 50 天里是否为最高（Gervais-Kaniel-Mingelgrin 的高成交量冲击）：TR_t >= MAX(TR,50)_{t-1}；"
    "连续版 TSRANK(TR, 50)", [50],
    "https://ideas.repec.org/a/bla/jfinan/v56y2001i3p877-919.html",
    "Gervais, Kaniel, Mingelgrin (2001 JF)：成交量异常放大的票此后一个月跑赢（关注度上升）。",
    "生产 vol_ratio20、Alpha158 VMA；altpath 量能族无增益。高度重叠，低优先。",
    "低", LOW, 3, "学术", "原文的触发是「一天或一周的成交量异常高」（摘要），50 天参照窗口是我们的口径")

# ---------------- F 量价背离 / 相关 ----------------
FF = "F_量价结构"
add(FF, "clv_vol_sum", "SUM(((C-L)-(H-C))/(H-L) * TR, d)（国君 Alpha11/60 的换手率版，Chaikin 累积/派发）", [6, 20],
    GTJA, "收在高位且放量的日子累计：主动买盘的日线代理。",
    "Alpha158 KSFT2 是当天的 CLV，不加权不累计；生产 obv_slope20 是收盘涨跌签名量。中度。",
    "低", LOW, 2, "国君 191 #11 / #60")
add(FF, "str_salience",
    "σ_t = |r_t - mkt_t| / (|r_t| + |mkt_t| + 0.1)；在个股自己的近 20 天里按 σ 降序排名 k_t，w_t = 0.7^{k_t} / MEAN(0.7^k)；"
    "STR = COV(w_t, r_t) over 20 天", [20],
    QP + "凸显理论STR因子/scr/core.py",
    "凸显理论（Bordalo-Gennaioli-Shleifer；Cosemans-Frehen 2021 落地）：投资者高估「显眼」的收益日。"
    "显眼日是大涨的票被过度追捧。对极端上涨是直接的注意力度量。",
    "不重叠。",
    "低", LOW, 2, "券商复现（QuantsPlaybook 凸显理论 STR）",
    "复现代码 calc_weight 用 rank(axis=1)，是按截面排；原文是在个股自己的时间序列里排，实现前要核对。")
add(FF, "corr_ret_dtr", "CORR(r_t, ΔTR_t, 20)", [20, 60],
    "自拟（Alpha158 CORD 的换手率版）",
    "量价同向性：涨时放量、跌时缩量。",
    "Alpha158 CORD 用的是 log 成交量变化（受送转影响）；altpath 里量能族无增益。高度重叠，低优先。",
    "低", LOW, 3, "自拟")

# ---------------- G 高点距离多周期 ----------------
FG = "G_高点距离多周期"
add(FG, "days_since_high", "(t - argmax(H, d)) / d，d = 120, 250（Alpha158 IMAX 只到 60）", [120, 250],
    "https://alphaarchitect.com/52-week-high-q-factor-investment-model/",
    "George & Hwang (2004 JF)：离 52 周高点越近，未来越强（锚定）。距上次高点的时间刻画「套牢盘有多老」，"
    "越老的高点越容易被突破。",
    "生产 dist_52w_high 就是 George-Hwang 的 PTH（重叠）；时间维度没有。低到中度。",
    "低", LOW_ADJ, 2, "学术")
add(FG, "pth_multi", "C / MAX(H, d) 与 C / MIN(L, d)，d = 120, 500（数据三年，500 日可算到 2025 年以后）", [120, 500],
    "https://alphaarchitect.com/52-week-high-q-factor-investment-model/",
    "多周期高点距离，Alpha158 只到 60，生产 250。120 填中间，500 看两年历史高点（突破两年高点的「无套牢盘」状态）。",
    "250 那根已在生产；120/500 不重叠。", "低", LOW_ADJ, 2, "学术")

# ---------------- H 残差动量 / 特质波动 ----------------
FH = "H_残差动量与特质波动"
add(FH, "resid_mom",
    "120 天回归 r = a + b·mkt + c·brd + e（窗口不含最近 20 天）；用系数算最近 20 天残差 e，"
    "resid_mom = SUM(e,20) / STD(e,20)", [20, 120],
    "https://repub.eur.nl/pub/22252",
    "Blitz, Huij, Martens (2011)：剔除共同因子暴露后的动量，风险调整后利润约为普通动量两倍。"
    "对我们：剔除「今天大盘 / 板块好」的部分后的个股独立强度。",
    "生产 rs20 / rs60 是相对强弱（减市场），没有按 beta 剔除和按残差波动标准化。中度。",
    "中（滚动 OLS）", LOW, 1, "学术")
add(FH, "idio_vol", "STD(e, 20)，e 为上式残差（Ang-Hodrick-Xing-Zhang 2006 的特质波动）", [20, 60],
    "https://alphaarchitect.com/hot-off-the-jfe-press-maxing-out-your-returns/",
    "特质波动高的票均值收益低（AHXZ），但右尾大；BCW 说加入 MAX 后特质波动之谜反转。对极端上涨概率应当是正向。",
    "Alpha158 STD 是总波动（含市场）；生产 atr_ratio 是比值。中度。",
    "中", LOW, 1, "学术")
add(FH, "idio_skew", "SKEW(e, 60)（剔除市场后的偏度）", [60, 120],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "Boyer-Mitton-Vorkink (2010) 的预期特质偏度和 Conrad 的 jackpot 都指向特质右尾。",
    "和 skew_ret 高度相关，二选一。", "中", LOW, 2, "学术")

# ---------------- I 筹码 / 成本 ----------------
FI = "I_筹码与成本"
add(FI, "chip_moments",
    "换手率衰减加权的成本分布（权重 w_s = TR_s · Π_{u>s}(1-TR_u)，窗口 60）上，x_s = 1 - C_s/C_t："
    "ARC = Σw·x（平均浮盈），VRC = 方差，SRC = 偏度，KRC = 峰度", [60],
    QP + "筹码因子/scr/turnover_coefficient_ops.py",
    "筹码分布的四阶矩。SRC / KRC 描述套牢盘是集中在上方一块还是分散，突破时上方抛压的形状。",
    "生产 chip_win（获利盘）、chip_conc90、chip_dev 覆盖一、二阶；三、四阶不重叠。",
    "中（每只票滚动 60 天加权）", LOW_ADJ, 2, "券商（QuantsPlaybook 筹码因子复现）")
add(FI, "cgo", "CGO = (C - RP)/C，RP = Σ_{k≥1} V_{t-k}·Π_{j<k}(1-TR_{t-j})·P_{t-k} / 归一，P 用 TP=(H+L+C)/3，窗口 100",
    [100, 250], QP + "处置效应因子",
    "Grinblatt-Han (2005) 处置效应 / 资本利得悬置：未实现浮盈大的票，持有者惜售，上涨阻力小。",
    "生产 chip_win、chip_dev 几乎是同一件事（自算筹码）。高度重叠，只在想换衰减口径时再做。",
    "中", LOW_ADJ, 3, "学术 + 券商")
add(FI, "apb", "APB_d = MEAN(VWAP, d) / (Σ VWAP·V / Σ V)，d = 5, 30（窗口内跨除权日则置空）", [5, 30],
    QP + "基于量价关系度量股票的买卖压力/py/CalFunc.py",
    "量价关系度量买卖压力：成交集中在低价位（加权均价低于算术均价）说明低位吸筹。",
    "不重叠（生产没有用 VWAP）。",
    "低", "低；VWAP 不复权，跨除权日比值会错，必须按除权日切断（流通股本跳变或 C/C_{t-1} 与 AMT/V 比值跳变识别）",
    2, "券商（QuantsPlaybook 复现）")

# ---------------- J K 线与 A 股形态 ----------------
FJ = "J_K线与影线"
add(FJ, "shadow_std",
    "蜡烛上影线 up = H - MAX(O,C)；威廉下影线 wlo = C - L；每日标准化 x_t / MEAN(x_{t-5..t-1})；"
    "Upper_shadow_std = STD(标准化 up, 20)，Williams_lower_shadow_mean = MEAN(标准化 wlo, 20)；"
    "UBL = 两者各自截面标准化后相加（东吴「上下影线，蜡烛好还是威廉好」）", [20],
    QP + "上下影线因子",
    "长上影线密集 = 上方抛压试盘；下影线密集 = 下方承接。妖股启动前常见反复长下影。",
    "Alpha158 KUP / KLOW 只有当天。低。", "低", LOW_ADJ, 2, "券商（东吴金工，QuantsPlaybook 复现）",
    "标准化分母用前 5 天（shift(1)），不含当天。原报告月频、做市值中性化；我们日频滚动、只做当日排名")
add(FJ, "wq_101", "(C - O) / ((H - L) + 0.001)（WQ Alpha#101），取 MEAN(.,5)", [1, 5],
    WQ_CODE, "日内收盘强度的简单版。",
    "Alpha158 KMID2 当天就是它。高度重叠；5 日均值算新增。", "低", LOW_ADJ, 3, "WQ 101 #101")

# ================= 第二批（2026-10-15 补全） =================
add(FA, "apm_ovn_intra",
    "APM 日线改写：取近 20 天，隔夜段 ovn_t 和日内段 intra_t 共 40 个观测，对同段的 mkt 收益做一次时间序列回归 "
    "r = α + β·R + ε；δ_t = ε^ovn_t - ε^intra_t；stat = MEAN(δ)/(STD(δ)/√20)；再对当日 rank_cs(ret20) 做截面回归取残差", [20],
    QP + "APM因子模型",
    "APM（开源证券）原版比较上午和下午的残差收益：知情交易集中在上午。日线只能拆成隔夜 / 日内；"
    "stat 高 = 个股的超额收益主要在盘前到达（消息 / 竞价抢筹），对散户驱动的极端上涨是正向线索。",
    "和 ovn_mom / ovn_intra_tug 同源，但去掉了市场部分并做成 t 统计量。中度。",
    "中（每只票 20 天一次 OLS）", LOW, 2, "券商（开源证券 APM，QuantsPlaybook 复现）",
    "原版要上午 / 下午分时收益（见需要新数据一节）。复现代码 issue #25 指出它按日期分组回归退化成截面去均值；实现时按研报口径每只票合并 40 个观测回归")
add(FB, "ideal_amplitude",
    "近 20 天每日振幅 amp = H/L - 1；按收盘价在这 20 天里的分位切：收盘价最高的 λ 那些天振幅均值 V_high，最低的 λ 那些天 V_low；"
    "V = V_high - V_low，λ = 0.25（研报扫 0.2~1.0）。剔除停牌和一字涨跌停后的日子，有效天数 < 10 置空", [20],
    QP + "振幅因子的隐藏结构",
    "开源证券「理想振幅」：高价位振幅大 = 高位多空分歧（派发），低价位振幅大 = 低位换手（吸筹）。研报 IC −0.075（λ=20%），"
    "中证 500 比沪深 300 强。对我们是「高位放波动」的反向信号，λ 越小越强。",
    "生产 range_compress、Alpha158 STD 只看振幅水平，不按价位切。不重叠。",
    "低（20 天窗口内排序）", LOW_ADJ, 1, "券商（开源证券市场微观结构 7，QuantsPlaybook 复现）")
add(FB, "amp_cut_mom",
    "近 N=120 天，按每日振幅 H/L-1 排序，取振幅最低的 λ=30% 那些天的日收益之和（A 因子，动量）；"
    "振幅最高的那些天的收益之和是 B 因子（反转）", [120],
    QP + "A股市场中如何构造动量因子？",
    "开源证券（开源量化评论 3）：A 股反转强、动量弱，但低振幅日的涨跌是「动量」，高振幅日的涨跌是「反转」。"
    "低振幅日在稳步上行，说明有资金在不显眼地推，对后续加速是正向；研报 IC 0.021（λ=30%，N=120）。",
    "Alpha158 ROC 是全部天数的和，不切；生产 rs60 同。低。",
    "低", LOW, 2, "券商（开源证券）")
add(FC, "skew_x_far_high",
    "skew_ret_60 × (1 - rank_cs(C / MAX(H,250)))（右偏 × 离一年高点远）", [60, 250],
    "https://ideas.repec.org/a/eee/pacfin/v80y2023ics0927538x23001208.html",
    "Wang, Wang, Wu (2023，检索摘要)：A 股的博彩偏好只出现在远离 52 周高点的票里。右偏 × 深套，正是「超跌妖股」的组合。",
    "交互项，树模型能自己学，显式给出能省分裂。和 skew_ret、dist_52w_high 重叠。",
    "低", LOW, 2, "学术（摘要级）",
    "出处链接是检索结果给出的 Pacific-Basin Finance Journal 2023 条目，标题未逐字核对")
add(FC, "detrended_turnover",
    "MEAN(TR,120) - MEAN(TR, t-480..t-121)（近 6 个月换手减去更早 18 个月；数据从 2023-01 起，2025 年后才满窗）", [120, 480],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "Conrad-Kapadia-Xing (2014) 的 jackpot logit 用去趋势换手率（和过去收益、波动、偏度、市值、上市年龄等并列）。换手抬升 = 关注度上升。",
    "生产 vol_ratio20 是 20 日，tr_long_level 是 250 日；去趋势口径不重叠，但信息相近。中度。",
    "低", LOW, 2, "学术",
    "Conrad 原文的去趋势窗口没读到全文，6/18 个月是我们的口径，需预登记")
add(FC, "listing_age",
    "LOG(本票在日线表里的交易日数)，截断 750；日线表起点之前上市的票统一记 750", [750],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "Conrad et al.：jackpot 公司更年轻。A 股次新股是游资主战场。",
    "生产无（features.py 注释里记过次新偏差：短历史票 y_up 率是成熟票 1.7 倍）。不重叠。",
    "低", "低；日线表第一根 K 线可能晚于真实上市日（2023 年初覆盖不全），要按 coverage_start 处理", 2, "学术")
add(FC, "hq_momentum", "ret60 - 3000 · STD(r,60)^2（光大「再论动量」 / QuantsPlaybook 高质量动量）", [60],
    QP + "高质量动量因子选股",
    "风险调整后的动量：大波动堆出来的涨幅被扣掉。对极端上涨可能是反向的（妖股恰恰靠大波动），做对照用。",
    "Alpha158 ROC60 和 STD60 的组合，树模型已能表达。高度重叠。",
    "低", LOW, 3, "券商（光大金工）")
add(FD, "info_discreteness",
    "ID = sign(SUM(r, 60)) × (下跌日占比 - 上涨日占比)（Da, Gurun, Warachka 2014 frog-in-the-pan）", [60],
    QP + "高质量动量因子选股",
    "信息离散度：同样的涨幅由很多小涨组成（ID 低）还是少数大涨组成（ID 高）。投资者对连续小变化反应不足。",
    "Alpha158 CNTD × sign(ROC) 就是它。高度重叠。复现的研报也说 A 股上 ID 作用不大，改用 60 日换手标准差。",
    "低", LOW, 3, "学术")
add(FE, "size_raw",
    "LOG(FS_t × AMT_t / V_t)：用不复权均价 AMT/V 乘流通股本，得到不含未来除权信息的流通市值", [1],
    "https://ideas.repec.org/a/eee/jfinec/v113y2014i3p455-475.html",
    "jackpot / 彩票股都偏小盘（Conrad 2014；清华彩票股研究：高 MAX、高换手、低价）。生产把市值中性化掉了（同日同板块同市值档去均值），"
    "Alpha158 臂也没有市值，模型看不到「它有多小」。给树模型做条件分裂用。",
    "生产 float_mcap 只用于中性化，不入模。不重叠。",
    "低", "低：AMT/V 是当天不复权均价；不要用前复权 C × FS（前复权价会被以后的分红改写，含未来信息）", 1, "学术 + 数据特有")
add(FE, "nominal_price", "LOG(AMT_t / V_t)（不复权均价）", [1],
    "https://cdn.shanghai.nyu.edu/sites/default/files/honorsthesis2025_zijin_su.pdf",
    "低价股效应：A 股彩票型股票常见定义是高 MAX、高换手、低价（清华研究，检索摘要）。散户对「便宜」的偏好。",
    "和 size_raw 相关但不同（价格 vs 市值）。不重叠。",
    "低", "低：同上，必须用不复权价", 2, "学术（摘要级）")
add(FE, "tr_own_rank", "TSRANK(TR, 250)：今天换手在自己过去一年里的分位", [250],
    "自拟",
    "截面排名只说「今天全市场谁换手高」，自身分位说「对它自己来说今天是不是罕见地热」。冷门票第一次放量是异动的定义。",
    "生产 vol_ratio20 是对 20 日均值的比，不是一年分位。低到中度。",
    "低", LOW, 2, "自拟")
add(FE, "tr_stability", "STD(TR,20)/MEAN(TR,20)（东吴「量稳换手率」的日线近似）", [20],
    "https://www.fxbaogao.com/detail/4657447",
    "换手稳定 = 筹码交换平稳；东吴月报持续跟踪的因子。",
    "生产 turn_std20 就是它。完全重叠，列出只为说明不要重复做。",
    "低", LOW, 3, "券商（东吴金工，月报标题级）")
add(FF, "ctrl_newhigh",
    "rank_cs(C/MAX(C,60)) - rank_cs(MEAN(TR,5)/MEAN(TR,60))；离散版：C_t = MAX(C,60) 且 MEAN(TR,5) < MEAN(TR,60)（缩量新高）", [5, 60],
    "自拟（控盘度，经典日线特征）",
    "价新高而换手没放大，说明筹码锁定、上方没有抛压，是高控盘票的典型形态；妖股主升段常见缩量连板。",
    "Alpha158 有 MAX60 和 VMA5，单独都有；差 / 交互项没有。中度。",
    "低", LOW_ADJ, 2, "技术分析")
add(FF, "open_push", "SUM(H-O,20)/SUM(O-L,20)（国君 Alpha118）", [20],
    GTJA, "开盘后向上的空间相对向下的空间：盘中资金往上推的力度。和隔夜族配合看开盘之后的接力。",
    "Alpha158 KUP/KLOW 以 max(O,C) 为基准、单日；这里以开盘为基准、累计 20 天。低。",
    "低", LOW_ADJ, 2, "国君 191 #118")
add(FF, "true_range_up", "SUM(MAX(0, H - C_{t-1}), 20) / SUM(MAX(0, C_{t-1} - L), 20)（国君 Alpha110，类 BR 指标）", [20],
    GTJA, "相对昨收的上攻空间 / 下探空间。",
    "和 open_push、Alpha158 SUMP 相近。中到高度。", "低", LOW_ADJ, 3, "国君 191 #110")
add(FF, "wq_rsv_vol_corr", "-CORR(RSV12, TR, 6)，RSV12 = (C - MIN(L,12))/(MAX(H,12) - MIN(L,12))（WQ Alpha#55 / 国君 #176 的换手率版，原式对两边先做截面 rank）", [6, 12],
    WQ_CODE, "价格在区间里的位置和换手同涨同跌：高位放量还是低位放量。",
    "Alpha158 CORR（收盘价和 log 量）近似。高度。", "低", LOW, 3, "WQ 101 #55")
add(FJ, "a360_seq",
    "Alpha360：过去 60 天 C/O/H/L/VWAP 各除以今日 C、量除以今日量，共 360 列原始序列；我们换成 TR、去掉 VWAP，共 300 列", [60],
    QLIB,
    "给序列模型（GRU / Transformer）的原始输入，让模型自己找形态。和我们 L2 序列模型的输入是同一类。",
    "和 Alpha158 是同一信息的两种表示。对树模型增益通常不如 Alpha158；只在重做序列模型时用。",
    "高（300 列，特征表多约 3 GB float32）", "低；VWAP 不复权，必须去掉或按除权日切断", 3, "qlib Alpha360")

VERIFIED = {
    "code": {"pcf_conv", "trcf_conv", "range_accel", "range_shock", "upvol_share", "skew_ret", "coin_flip_ret",
             "ovn_dist", "str_salience", "chip_moments", "cgo", "apb", "shadow_std", "wq_101", "clv_vol_sum",
             "ideal_amplitude", "amp_cut_mom", "apm_ovn_intra", "hq_momentum", "info_discreteness", "open_push",
             "true_range_up", "wq_rsv_vol_corr", "a360_seq", "amihud"},
    "abstract": {"ovn_mom", "intra_mom", "ovn_intra_tug", "ovn_vol_share", "max_ret", "past_jackpot", "min_ret", "kurt_ret",
                 "resid_mom", "idio_vol", "days_since_high", "pth_multi", "hv_premium", "skew_x_far_high",
                 "detrended_turnover", "listing_age", "size_raw", "nominal_price", "tr_stability"},
    "memory": {"price_delay_d1", "synchronicity", "idio_skew"},
}
for _f in F:
    _v = "self"
    for _k, _names in VERIFIED.items():
        if _f["name"] in _names:
            _v = _k
    _f["verified"] = {"code": "公式表 / 复现代码已读", "abstract": "摘要级（概念和方向已核，公式按定义写）",
                      "memory": "据记忆，实现前核对原文", "self": "自拟，无外部出处"}[_v]

NEW_DATA = [
    dict(name="APM 原版（上午 / 下午残差）", data="分钟线（至少上午收盘价）", source_url=QP + "APM因子模型",
         why="知情交易集中在上午；日线只能用隔夜 / 日内代替（apm_ovn_intra）"),
    dict(name="理想反转（W 式切割）/ 大单日", data="每日成交笔数（平均单笔成交金额 = 成交额 / 笔数）或逐笔",
         source_url=QP + "开源证券-市场微观结构研究系列（1）：A股反转之力的微观来源",
         why="开源证券：按平均单笔成交金额切 20 天，大单日涨跌是反转之力（高分位切割效果最好）；日线没有成交笔数"),
    dict(name="价量相关性 CPV、换手率 / 信息分布均匀度 UTD / UID、适度冒险、潮汐、勇攀高峰", data="分钟线",
         source_url="https://fxbaogao.com/detail/4688163",
         why="东吴 / 方正金工的高频降频因子，都要日内分钟价量；UID 月报写明用分钟数据算每日高频波动"),
    dict(name="聪明钱因子 2.0", data="分钟线（按每分钟 |涨幅|/√成交量 找聪明钱时段）",
         source_url=QP + "聪明钱因子模型的2.0版本", why="分钟级排序，日线无法近似"),
    dict(name="尾盘 / 开盘成交占比", data="分钟线", source_url="",
         why="尾盘抢筹（14:30 后成交占比、尾盘涨幅）和次日开盘买入口径最近"),
    dict(name="t 日集合竞价量价（竞价涨幅、竞价额 / 全天成交额）", data="09:25 竞价快照；历史回填见 archive/morning",
         source_url="archive/morning/RESTORE.md", why="t 日的竞价可用；t+1 竞价是偷看（买点就在 t+1 开盘）"),
    dict(name="涨停封板质量（首次封板时间、开板次数、封单额 / 流通市值）", data="涨停池 stock_zt_pool_em（只有近 30 天，抢救中）",
         source_url="docs/github_survey_2026-10.md", why="连板生态里封板强弱是游资接力的直接信号；日线只能判断封没封"),
    dict(name="龙虎榜（游资席位、机构净买）", data="交易所龙虎榜日频明细，需逐日存档",
         source_url="", why="妖股推手是游资席位，同一席位反复出现是连板最直接的标记；上榜本身是价格规则触发，增量在席位身份"),
    dict(name="个股资金流（大单 / 超大单净流入）", data="东财 fflow/daykline（本机可达，只回溯 120 天，见 CLAUDE.md 5b）",
         source_url="CLAUDE.md 5b", why="从现在起每天存，攒够两年才能进开发集"),
    dict(name="融资融券余额 / 融资买入占比", data="交易所两融明细（日频）", source_url="", why="杠杆资金追涨强度；只覆盖两融标的"),
    dict(name="概念板块成分历史（时点一致）", data="每日概念成分快照，需自存", source_url="docs/github_survey_2026-10.md",
         why="题材炒作的同涨族群；现在只有静态行业表，g2 用收益相关性代替"),
    dict(name="限售解禁日程", data="解禁日历（公告日已知）", source_url="",
         why="未来 20 天内的解禁量是事前公布的，按公告日对齐不算偷看；供给冲击压制炒作"),
    dict(name="人气榜 / 千股千评 / 搜索热度", data="已在抢救（人气榜一年、千股千评 37 天）", source_url="",
         why="注意力的直接测量，历史太短不能进开发集，先存档"),
]

# 总排序：和目标的相关性 × 不重叠 × 可算。priority 3 的不进排序（rank = 42）
RANKED = [
    "past_jackpot", "skew_ret", "max_ret", "idio_vol", "ovn_mom", "size_raw", "resid_mom",
    "ideal_amplitude", "intra_mom", "pcf_conv",
    "ovn_dist", "tr_own_rank", "float_chg", "ctrl_newhigh", "upvol_share", "apm_ovn_intra", "amihud",
    "price_delay_d1", "coin_flip_ret", "str_salience", "nominal_price", "listing_age", "gap_up_cnt",
    "amp_cut_mom", "skew_x_far_high", "detrended_turnover", "trcf_conv", "box_breakout", "nr_rank",
    "range_shock", "days_since_high", "pth_multi", "chip_moments", "shadow_std", "open_push",
    "clv_vol_sum", "synchronicity", "idio_skew", "ovn_intra_tug", "tr_long_level", "apb",
]


def main():
    F.sort(key=lambda f: (f["family"], f["priority"]))
    order = {n: i for i, n in enumerate(RANKED)}
    assert set(RANKED) <= {f["name"] for f in F}, set(RANKED) - {f["name"] for f in F}
    for f in F:   # id 保持初版插入顺序（F001~F038 不变，补的从 F039 起），外部脚本按 id 引用
        f["rank"] = order.get(f["name"], len(RANKED)) + 1
    out = dict(
        version="v1",
        created="2026-10-14",
        target="次日开盘买入后 20 个交易日内最高价涨超 50%（买得到口径，次日一字不可买）",
        notation=NOTATION,
        priority_scale="1 = 先做（和目标相关、和 Alpha158/生产不重叠、可算）；2 = 第二批；3 = 重叠高或理由弱，只在需要时补",
        factors=F,
        needs_new_data=NEW_DATA,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(len(F), "factors ->", OUT)


main()
