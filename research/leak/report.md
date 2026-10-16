# 前复权取整颗粒泄漏检验（research/leak，2026-10-16）

（Fable 5.1 high 研究子代理写成，主会话落盘。阶段结论在 progress.txt，所有数字可从 results_*.txt 复核。）

src / tools / state / config.yaml 一个字没改；代码全在 research/leak/，产物在 data/breakout/raw/exp/leak/ 和 exp/leak_v1/。开发集 2024-06..2025-12 逐月走向前（evalkit.walk_forward），确认集一行没读（价格和特征只算到 2025-12-31）。所有臂只读快照：train_snap.parquet、daily_snap.parquet、holders_snap.parquet（10-16 06:00 拷贝）。北京 16:00~18:30 没跑任何东西。

## 0. 结论

| 问题 | 答案 | 关键数字（每天前 10 买得到口径，按天配对） |
|---|---|---|
| Q1 候选 a158_ovh_cs4 的 +5.22pp 有没有泄漏成分 | **排除** | G_fwd +5.25、G_pit +6.04、G_ctl +5.33（三种子均值）；Δ(fwd−pit) 均值 −0.79pp，NW t −2.43；Δ(fwd−ctl) −0.09，NW −0.25 |
| Q2 生产特征有没有吃到泄漏 | **没有证据，邮件 13.8% 不改口** | 种子 7：fwd−ctl +0.13pp NW 0.33；三种子 fwd−pit 均 +0.34pp NW 1.72 p 0.09；生产规则口径 NW < 0.9 |
| 候选在干净价格上的 v2 判定 | 未定（同 feat_final） | cand_pit +5.50 NW 2.64 p 0.007；cand_ctl +4.73 NW 2.66 p 0.007；最差三月都不过 |

## 1. 机制和设计

新浪前复权价 qfq_s = round(P_u,s / q_s, 2)，q_s 是 s 之后全部除权的累计因子。取整到分之后，颗粒度 0.01·q_s/P_u 带着 q_s，也就是「T 日之后有没有除权」。开发集里 T 日之后还有除权（q ≥ 1.03）的行占 28.7%；log q 当日秩和 y_open 相关 −0.048，NW t −6.8。

三版价格，同一份新浪不复权价和因子（fetch_unadj.py，5525 只，缺 601991 一只，两臂都不含它）：

| 版本 | 定义 | 含义 |
|---|---|---|
| fwd | round(P_u / q, 2) | 现状。和 daily_snap 四个价逐位相同 99.991%（对不上的 541 行全在 1 分以内） |
| pit | P_u / q 不取整 | 比值特征和「截至 T 日的复权价」一致，颗粒里没有未来信息 |
| ctl | round(P_u / k_code, 2) / q | k_code = 各票 2025-01-02 的 q 跨票随机置换。同量级取整噪声，但和这只票自己的未来无关 |

为什么要 ctl：fwd − pit = 泄漏 − 取整噪声的代价，两项符号相反；fwd − ctl 单独量泄漏，pit − ctl 单独量噪声代价。

三版上各从零重建三套特征：生产特征（照抄 build.build 全流程，4 进程 5.4 分钟一版）、Alpha158（altpath.build_feats.alpha158）、on/vc/hi（features.build_new.FAMS）。标签两臂同一份：y_up 取快照，y_open 用 buyable.compute(daily_snap)。每月特征选择用生产缓存 feats_by_month.json。

预登记 leak_v1（prereg.json）：Q1 主判定 = 种子 7 的 Δ = (cand_fwd−base_fwd) − (cand_ctl−base_ctl) 按天序列，NW t ≥ 2.0 且占比 ≥ 0.3 为「有实质泄漏」，NW < 2.0 且占比 < 0.3 为「排除」；Q2 主判定 = base_fwd − base_ctl，NW ≥ 2.0 且 ≥ +0.5pp 为「吃到，要改口」。登记簿加 3 个臂（cand_ctl_s7、cand_pit_s7、base_fwd_vs_ctl_s7），cand_mix 登记为诊断。种子 9/10 只报告。

## 2. 前置核对

- **缩放不变性**（results_invariance.txt）：四个价整体乘 3.7 或 1/2.3，生产特征 28 列、Alpha158 142 列、三族 33 列、三个标签全部逐位不变（相对差 < 1e-7），只有浮点平局翻转。chip_avg 是绝对价位但不进特征；float_mcap 用 amount/volume 折回不复权价。
- **q 的用法**：新浪因子表登了尚未到来的除权，要按 q_s / q_last（全表最后一行）用，之后 99.991% 逐位相同。
- **重建流程**（results_rebuild.txt）：prod_fwd 和 train_snap 2660819 行全部对上，102 列相关中位 1.0000、最低 0.9965。alt_fwd 对原 a158 / fam 相关 ≥ 0.9999。pit / ctl 对 fwd 逐位相同率几乎 0（教训 32）。
- **微扰敏感度**（results_jaccard.txt，要记住）：base_snap 对 base_fwd（只少一只票、当日排名挪 2e-4）命中 12.64% 对 12.14%，−0.49pp NW −1.5，前 10 名单日均 Jaccard 0.53。fwd / pit / ctl 任意两版之间 Jaccard 0.52~0.53。**价格版本换掉的是「谁上榜」，不是「榜的质量」；单种子的任何 0.5pp 差都不能当真。**

## 3. 直接诊断（results_diag.txt）

- 取整噪声幅度 |fwd − pit| 和 log q 的当日秩相关：生产列中位 0.10（range_compress 0.40），alt 列中位 0.34（a_klow 0.60）。幅度通道存在。
- 特征值本身和 log q 的相关，fwd 减 pit：中位 0.0000；最大是「上涨天数占比」一类（f_on_on_pos20 −0.024、a_cntp60 −0.017、up_vol_share20 −0.007）：q > 1 时一个 tick 的变动取整后成零收益。量级很小。
- **模型能不能从 fwd 特征读到 q**：LightGBM 预测「T 日之后有除权 ≥ 3%」，按代码对半分，AUC base fwd/pit/ctl = 0.7253 / 0.7256 / 0.7257，cand 0.7685 / 0.7713 / 0.7715；≥ 10% 与 20 根内有除权同样无差别。合法特征本身就能以 AUC 0.72~0.81 预测未来除权（动量 → 高送转的反向因果，这是 feat_audit 里「之后有除权的行命中高」的来源），取整通道没有模型读得到的增量。

## 4. 走向前（results_leak.txt、analyze_final.log、results_q2_seeds.txt）

前 10 命中（%），生产规则 = ≥ 97 分且前 10：

| 臂 | 种子 7 | 种子 9 | 种子 10 | 三种子均 | 生产规则三种子均 |
|---|---|---|---|---|---|
| base_snap | 12.64 | | | | 16.29 |
| base_fwd | 12.14 | 12.17 | 12.04 | 12.12 | 16.11 |
| base_pit | 12.04 | 11.68 | 11.60 | 11.77 | 15.66 |
| base_ctl | 12.02 | 11.63 | 11.03 | 11.56 | 15.14 |
| cand_fwd | 17.62 | 17.86 | 16.61 | 17.36 | 19.76 |
| cand_pit | 17.55 | 18.17 | 17.73 | 17.81 | 19.98 |
| cand_ctl | 16.74 | 16.80 | 17.13 | 16.89 | 19.70 |
| cand_mix（生产 fwd + alt pit） | 17.78 | | | | 19.98 |

**Q1**（增益 G = cand − base）：

| | 种子 7 | 种子 9 | 种子 10 | 均值 / 平均序列 NW t |
|---|---|---|---|---|
| G_fwd | +5.48（NW 2.68） | +5.68（2.71） | +4.57（2.69） | +5.25 / 2.78 |
| G_pit | +5.50（2.64） | +6.49（3.07） | +6.12（3.24） | +6.04 / 3.04 |
| G_ctl | +4.73（2.66） | +5.17（2.68） | +6.10（3.54） | +5.33 / 2.99 |
| Δ = G_fwd − G_ctl（主判定） | +0.75（NW 1.12，p 0.27，占比 0.14） | +0.52（0.95） | −1.52（−2.06） | −0.09 / −0.25 |
| Δ = G_fwd − G_pit | −0.03（−0.04） | −0.80（−1.72） | −1.55（−2.38） | −0.79 / −2.43 |

预登记判定：NW 1.12 < 2.0 且占比 0.14 < 0.3 → **排除**。三种子更干脆：去掉取整候选反而更好。cand_mix − cand_pit +0.23（NW 0.65）、cand_fwd − cand_mix −0.16（NW −0.40）。

**Q2**（生产特征）：

| | 种子 7（主判定） | 种子 9 | 种子 10 | 三种子均 / 平均序列 NW t / 块自举 p |
|---|---|---|---|---|
| base_fwd − base_ctl | +0.13（NW 0.33，p 0.75） | +0.54（1.74） | +1.01（1.87） | +0.56 / 2.97 / 0.003 |
| base_fwd − base_pit | +0.10（0.28） | +0.49（1.33） | +0.44（1.17） | +0.34 / 1.72 / 0.093 |
| base_pit − base_ctl（ctl 的噪声代价） | +0.03（0.07） | +0.05（0.13） | +0.57（1.22） | +0.22 |
| 生产规则口径 fwd − pit | +0.65（0.83） | +0.37（0.36） | +0.17（0.16） | |
| 生产规则口径 fwd − ctl | −0.28（−0.27） | −0.23（−0.25） | +0.79（0.80） | |

预登记判定（种子 7）：+0.13pp，NW 0.33 → **没有证据，不改口**。诚实说明：三种子平均 fwd − ctl 到了 NW 2.97，但 ctl 比 fwd 噪声更大（74% 的票 k > 1），pit − ctl 同样为正，所以干净的度量是 fwd 对 pit：+0.34pp、NW 1.72、p 0.09，三种子都正但没有一个到 2。两个旁证：候选在同样比较里方向相反（cand_pit > cand_fwd）；模型从 fwd 特征读不到 q。结论：前 10 口径至多 0.3~0.5pp 偏高，生产规则口径看不出差别，13.8% 不改。

**候选 v2（干净价格上）**：cand_pit_s7 对 base_pit_s7 +5.50pp，NW 2.64（10/30 滞后 3.10/2.56），块自举 p 0.007，去最好月（2024-10）t 2.39，同号，空榜 30/70，最差三月 2025-11 −5.5 / 2025-08 −3.3 / 2025-10 −2.9（去 2024-09 同）；cand_ctl_s7 +4.73，NW 2.66，p 0.007，去最好月 2.43，最差三月 −3.7/−3.5/−1.9。门槛：第一次 compare 时 Bonferroni m=71 → c 3.19（不过），之后族内刚够 3 条序列换成最大 t 法 → c 2.50（过）；最差三月两种口径都不过 → 「未定」。

## 5. 对主会话初步思路的核对

对的：三套特征和标签全是价格比值（实测）；pit 的颗粒无未来信息；两臂同标签只换特征；尺子；A/B 先断言臂不一样。

要改的：
- fwd − pit 不是泄漏的干净度量（混着取整噪声代价），加了 ctl；Q1 两种度量同向，Q2 不同向，正是这个原因。
- 「fwd 重建应和快照几乎相同」只在特征值层面成立；前 10 名单层面 Jaccard 0.53、命中 −0.49pp。顶部名单对微扰极敏感（支持生产多种子平均）。
- q 要用 q_s / q_last。
- 用真实 log(q_s/q_last) 代替 log g 做了诊断（第 3 节）。

## 6. 下一步建议

1. 候选 a158_ovh_cs4 留。九次重跑全在 +4.6~+6.5pp，泄漏排除；不建议再加臂。base_pv（95 列）瘦身版若要上，当作同一候选的工程简化，不算另一次试验。
2. 最差三月（pit 口径 −5.5/−3.3/−2.9）是真问题，要用户定。
3. Q2 的 0.3pp 上限不值得为它动数据源；后复权 / 不取整改造顺手就干净了，优先级低。
4. 产物 12 GB（三版特征表）可删，留 px.parquet / unadj.parquet / 三个快照即可复算。

## 7. 主会话补记（2026-10-16）

- **族门槛会跳**：同一个臂，登记簿里族内序列从 2 条变 3 条，门槛就从 Bonferroni 3.19 掉到最大 t 法的 2.50。v3（docs/design_2026-10-16.md 7.2，用户已同意）补登 47 条历史序列，门槛约 2.8，正好堵这个口子。
- 用户 2026-10-16 已拍板：确认集那一次给第 5 周的组合候选，不单独给本候选（设计文档拍板 D）；无论泄漏结论，日线都换后复权（拍板 E）。本报告的「建议确认集用在本候选上」按拍板 D 作废，本候选作为组合候选的特征部分。

## 文件

research/leak/：fetch_unadj.py（主会话）、common_leak.py、check_invariance.py、prices.py、build_prod.py、build_alt.py、make_labels.py、check_rebuild.py、diag_leak.py、exp_leak.py；results_invariance.txt、results_prices.txt、results_rebuild.txt（+ check_prod_fwd_vs_snap.csv）、results_diag.txt（+ diag_a_prod.csv / diag_a_alt.csv）、results_leak.txt、results_jaccard.txt、results_q2_seeds.txt、各 .log、progress.txt。
data/breakout/raw/exp/leak/：unadj.parquet、missing.txt、px.parquet、train_snap / daily_snap / holders_snap.parquet、labels_open_snap.parquet、prod_{fwd,pit,ctl}.parquet、alt_{fwd,pit,ctl}.parquet。
data/breakout/raw/exp/leak_v1/：prereg.json、key.txt、top_*/all_*.parquet（20 个臂）、diff_*.npy、q_*.json。
登记簿：research/features/ledger.jsonl +10 行，data/breakout/raw/exp/ledger.jsonl 68 → 77 行。
