# 扩展筛选集：建表 + 现状模型 base（research/screen，2026-10-16）

（Fable 5.1 high 子代理写成，主会话落盘。阶段结论在 progress.txt，数字可从 results_*.txt / *.log 复核。）

src / tools / state / config.yaml 没动，research/leak 已有文件没动（只 import build_prod 的 one / agg5）。
代码在 research/screen/，产物在 data/breakout/raw/exp/screen/（2.6 GB）、exp/screen_base_v1/、exp/screen_base_v1_dev/。
确认集一行没读：价格、特征、标签、股东户数都截到 2025-12-31。北京 16:00~18:30 没跑任何东西（全部 14:50 跑完）。
主登记簿 data/breakout/raw/exp/ledger.jsonl 138 行没动；筛选集的臂记在 research/screen/ledger.jsonl（族 screen_y50_top10）。

## 0. 结论

| 项 | 数字 |
|---|---|
| 筛选集 2022-01..2024-05 | 582 天、可打分行 2,771,429、5210 只 |
| base 前 10 买得到命中 | 种子 7 8.33%（NW se 0.86pp）、种子 9 8.45%；收盘口径 10.05% |
| 全市场基准 | 按天 1.669%（合并 1.675%）；bj 6.37 / chinext 2.07 / star 1.66 / main 1.33 |
| 决策门 >= 2 倍 | 5.03 倍，过：筛选集可作筛选器 |
| 种子噪声 s7 − s9 | −0.12pp，NW se 0.456pp，前 10 名单日均 Jaccard 0.46 |
| MDE（功效 80%，c = 2.5 / 2.8 / 3.07） | 换种子级 +1.52 / +1.66 / +1.78pp；增量臂约 +2.9 / +3.1 / +3.4pp；特征大改约 +6.1 / +6.7 / +7.2pp |
| 只作同号检查（同号概率 80%） | 增量臂效应 >= 0.73pp，大改 >= 1.55pp |

- 筛选集 NW se 没按 1/sqrt(n) 缩（582 天 0.456pp，开发集 387 天 0.46pp）：按天差离散更大、两种子重合更低（0.46 对 0.58）。
  筛选集 + 开发集一起看，增量臂 se 约 0.62pp，c=3.07 下 MDE 约 2.4pp。
- 「生产规则 >= 97 分」在筛选集上退化成前 10（空榜 0/582）：分数是预测值在训练参照样本里的百分位，筛选集训练集里没有
  2024-09 那种正样本占三成的月份，参照样本 97 分位 0.23（开发集 0.41）。跨期只比前 10。
- 按半年 base / 基准（%）：2022H1 10.6~10.8 / 1.94，2022H2 6.4~6.6 / 1.41，2023H1 8.4~9.1 / 1.35，2023H2 7.7~10.0 / 1.64，2024H1 5.9~8.3 / 2.12。
  最差月 2024-01 0.9%（基准 0.7%）、2022-09 1.0~1.4%（1.2%）。前 10 板块构成 main 62 / chinext 28 / bj 6 / star 4（开发集 35 / 33 / 25 / 7）。
- 附加（screen_base_v1_dev）：同一张长表上开发集 2024-06..2025-12 base_s7 13.30%（367 天）对 leak base_pit_s7 12.04%：+1.28pp，
  NW se 0.83，NW t 1.55，块自举 p 0.13，v3 未定；两份名单 Jaccard 0.17。开发集基准 4.49%，倍数 2.96（筛选集 1.67%，5.0 倍）。

## 1. 数据和口径

- unadj_2020.parquet：新浪不复权 OHLC + volume / amount / outstanding_share / turnover + qfq_factor，2020-07-01 起，5526 只 7,480,098 行，缺 0。
- pit 价 = P_u / (q_s / q_last)，不取整，q_last 取各票全表（含 2026）最后一行，46 只 q_last ≠ 1。停牌占位行按生产 backfill.clean_bars 清（去 17 行）。
  和 leak px.parquet 重叠 3,381,947 行 close 逐位相同 100%，volume 100%，turnover 99.9999%。
- 幸存者偏差：表里只有 2026-10 仍在市的票。有 K 线的只数：2020-07-01 3664、2021-12-31 4480、2022-12-30 4913、2024-05-31 5267、2025-12-31 5436。
  对照公开的上市总数（2021 年底约 4,700、2022 年底约 5,070，子代理凭记忆的大约数，**未拉源核实**），2022 年初候选池缺约 200 只（约 4%），
  全是之后退市的票。方向：它们大多进不了前 10，缺席主要抬高全市场基准，base 绝对命中和倍数略偏乐观，量级一个百分点以内；两臂配对差不受影响。
- 股东户数 holders.parquet 最早报告期 2023-06-30（公告约 2023-08 起）。2020-12..2023-07 五个 gdhs_* 和 x_holder_chip 全 NaN -> 模型里 0（组均值）：
  筛选集 29 个月里 20 个月没有这组特征，base 偏保守，配对比较不受影响。

## 2. 五个决定

1. 训练窗口：每个测试月前 3 年（含）到净化线（validate.train_slice 同一条净化规则）。行从 2020-12-31 起，2022-01 的窗口约 12 个月，
   比开发集前几个月（2024-06 约 5 个月）宽；3 年上限只在 2024-01 起生效。不从更晚起：筛选集的意义就是多出的天数。
2. 特征选择：每月在净化过的训练切片上重跑 fselect.run（42~48 列，每月 1.5~3 分钟），缓存键 "<月>|screen_v1"；
   不用开发集缓存（测试月之后的数据上选的），不固定列表（生产每月重选）。
3. 标签：y_up 用 label.label_frame，y_open 用 buyable.compute，都在 pit 价上。一字板用 pit 价判和用不复权价判，翻 y_open 的 565 行 = 0.0089%，忽略。
4. 共同起点不用 build.trim_common_start：COVER_MIN=0.90 是为回填没对齐的薄片段写的；本表 2020-07 只数少是真的 IPO 增长，照抄会把起点砍到 2023 年。
   其余流程照抄 build.build。
5. 登记：不调 evalkit.prereg / compare(exp=)，主登记簿和族门槛不变；本地登记簿 research/screen/ledger.jsonl。

## 3. 一致性核对（results_check.txt）

- 特征表 prod_screen 5,786,512 行 102 列，2020-12-31..2025-12-31。
- 对 leak prod_pit（同 pit 口径、起点 2023-05-30）：重叠 2,660,819 行全部对上，本表多 647 行（26 只，停牌占位行差异）；board 一致 100%。
  开发集 102 列相关中位 1.0000、最低 0.847（chip_conc_chg20__slope）。差全在筹码：chips 是换手率加权的指数记忆，本表从 2020-07 累积，
  低换手票收敛慢，chip_conc90 相关 2023H2 0.66 -> 2024H1 0.87 -> 2025H2 0.95；dist_52w_* 2023H2 0.79、2024H2 起 0.999。
- y_up 对 train_snap 开发集 99.994%；正样本率 4.540% 对 4.539%。
- y_open 对 labels_open_snap 开发集 99.987%（不一致 261 行，都是取整颗粒）。

## 4. 走向前和分析（results_screen.txt / results_dev.txt）

- screen_base_v1：evalkit.walk_forward 逐月（parts/<月>/ 可续跑），base = 生产特征 + L1Lgbm(400 棵, 按月等权)，种子 7 / 9。62 分钟。
- 月度前 10 买得到（%，s7 / s9 / 基准）：2022-01 7.4/10.5/1.4，02 6.2/7.5/1.4，03 6.1/4.8/1.5，04 13.7/14.2/2.2，05 16.3/15.3/3.0，06 13.8/12.9/2.2，
  07 10.0/11.9/2.4，08 3.5/4.8/0.8，09 1.0/1.4/1.2，10 10.0/7.5/2.4，11 11.4/12.3/1.2，12 3.6/2.3/0.8；2023-01 5.6/4.4/1.4，02 7.0/7.0/0.6，
  03 10.0/13.0/1.8，04 12.6/10.0/1.4，05 9.0/11.5/1.6，06 5.5/7.0/1.2，07 3.8/1.9/0.7，08 5.7/3.0/0.9，09 13.0/11.5/1.1，10 20.0/15.9/2.3，
  11 11.8/10.9/3.4，12 8.1/4.8/1.5；2024-01 0.9/0.9/0.7，02 16.7/19.3/6.9，03 2.9/3.3/1.5，04 4.5/12.0/1.9，05 8.0/9.5/0.9。
- 种子噪声：s7 − s9 −0.12pp，NW se 0.456pp，NW t −0.26，块自举 p 0.80；辅助尺子 lr20 的差 −0.0009，se 0.0024。按天命中序列 sd 0.095、ACF1 0.45（开发集 0.150 / 0.70）。
- MDE 外推：开发集 NW se 大改 1.85 / 增量 0.87 / 换种子 0.46pp，按 sqrt(387/582) 缩后换种子应为 0.38，实测 0.46，比例 1.22；增量和大改按同比例校正后仍为 0.87 / 1.85pp。
- screen_base_v1_dev（长表上开发集，base s7，64 分钟）：13.30%（367 天），基准 4.49%，倍数 2.96；对 leak base_pit_s7 +1.28pp NW t 1.55 p 0.13，名单 Jaccard 0.17。

## 5. 第 2 周接口

exp_screen.py：ARMS 加一项 {"y", "w", "kw", "cols", "desc"}；标签变体在 LABELS 登记 fn(df)->Series（已有 y_rel4 / y_rel2，day_weights 已写）；
y_soft 要给 spec["sampler"]（现在所有臂用 stratified_sample(y)，只认 0/1 标签）；evw 的权重函数要写。

    python research/screen/exp_screen.py --fit --name screen_w2_v1 --arms base eqd evw y_rel4 y_rel2 --seeds 7

screen_cmp(with_r20(top(name, 臂)), with_r20(top(name, "base_s7"))) 给 diff_day / days / span / status / NW t / p / lr20，可直接作
evalkit.prereg(screen={臂: r}) 的输入。fselect 缓存 29 个月已全部命中，每个模型每月 40~90 秒，7 臂 + base 单种子约 4 小时，分两段跑（北京 16:00~18:30 自动停）。

## 文件

research/screen/：common_screen.py、prices_screen.py、build_screen.py、labels_screen.py、check_screen.py、exp_screen.py、compare_dev.py；
results_*.txt、*.log、ledger.jsonl、progress.txt。
data/breakout/raw/exp/screen/：px_screen.parquet、prod_screen.parquet、labels_screen.parquet（2.6 GB）。
data/breakout/raw/exp/screen_base_v1/、screen_base_v1_dev/：prereg.json、top_ / all_ 名次表、parts/。
data/breakout/raw/exp/feats_by_month.json 多 48 个 "<月>|screen_v1" 键。
