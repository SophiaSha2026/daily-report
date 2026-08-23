# A股集合竞价选股 · 自动化流水线

每交易日 **09:28:30（北京时间）** 把竞价强弱榜前 10 发到邮箱，
同时更新在线面板。全程跑在 GitHub Actions，本地不需要装任何东西。

> 量化筛选工具，非投资建议。

---

## 目录

- [部署（全程网页操作）](#部署全程网页操作)
- [Claude 认证的三条路](#claude-认证的三条路)
- [每天怎么用](#每天怎么用)
- [调参](#调参)
- [已知风险](#已知风险)

---

## 部署（全程网页操作）

### 1. 建仓库

打开 https://github.com/new

| 字段 | 值 |
|---|---|
| Repository name | `daily-report` |
| 可见性 | **Public** |
| Add README | **不勾** |

选 Public 是因为 Actions 分钟数无限制；Private 免费额度 2000 分钟/月，
本流程约需 900 分钟/月。Secrets 在 Public 仓库里同样是加密的，读不出来。

### 2. 上传文件

新仓库页面 → **uploading an existing file** →
把解压后的**所有内容**（包括 `.github` 文件夹）拖进去 → Commit changes。

> 如果拖拽后看不到 `.github` 文件夹：Windows 资源管理器默认隐藏以点开头的
> 文件夹。查看 → 勾选「隐藏的项目」，再拖一次。
> 或者直接把整个解压目录拖进去，GitHub 会保留目录结构。

上传完确认仓库里有这几项：`.github/workflows/`、`src/`、`prompts/`、
`config.yaml`、`requirements.txt`。

### 3. 配 Secrets

仓库 → **Settings** → 左栏 **Secrets and variables** → **Actions**
→ **New repository secret**，逐个添加：

| Name | Secret |
|---|---|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | 你的 Gmail 地址 |
| `SMTP_PASS` | Gmail 应用专用密码（16 位，**去掉空格**） |
| `MAIL_TO` | 收件地址，多个用英文逗号分隔 |

**Gmail 应用专用密码怎么来**：
1. https://myaccount.google.com/security → 开启两步验证（必须先开）
2. https://myaccount.google.com/apppasswords → 起个名字 → 生成
3. 复制那 16 位，去掉空格填进 `SMTP_PASS`

Claude 的认证 secret 见下一节。

### 4. 开 Actions 和 Pages

- **Actions**：仓库 → Actions 标签 → 点 "I understand my workflows, go ahead and enable them"
- **Pages**：Settings → Pages → Source 选 **GitHub Actions**（不是 Deploy from a branch）

Pages 开好后，你的面板地址是
`https://sophiasha2026.github.io/daily-report/`

### 5. 跑冒烟测试

Actions → 左栏 **0-冒烟测试（首次必跑）** → 右侧 **Run workflow** → 绿色按钮。

它验证 6 项：腾讯批量行情、1600 只吞吐耗时、东财全市场快照、交易日历、
日线历史、打分逻辑自测，最后发一封测试邮件。

**必须全绿。** 如果行情类失败，说明 GitHub runner 的出口 IP 被国内接口限流，
需要换部署位置（见「已知风险」）。

### 6. 初始化板块缓存

Actions → **3-刷新板块缓存** → Run workflow。跑 2-4 分钟。
不跑这步，板块共振维度会失效（不影响其他维度）。

### 7. 试跑一次盘前

Actions → **1-盘前候选池** → Run workflow。确认绿灯。

完成。次日 08:23 起自动运行。

---

## Claude 认证的三条路

**你要的「对本地无要求」和 `claude setup-token` 是冲突的**——那条命令必须
在本机装 Node.js 和 Claude Code CLI 才能跑。三个选项：

### 路 A：API Key（真正零本地，推荐先用这个）

1. https://console.anthropic.com → API keys → Create Key
2. 复制，存为 Secret `ANTHROPIC_API_KEY`

工作流会自动走备用分支。纯网页操作，零安装。
代价是按量计费，与订阅分开结算。本流程每天约 1 万 token 量级，
成本很低，但不是零。

### 路 B：OAuth（用订阅额度，需要本地跑一条命令）

CLAUDE_CODE_OAUTH_TOKEN 认证你的 Claude 订阅，Pro / Max / Team / Enterprise
都支持，由 `claude setup-token` 生成。

```
# 需要 Node.js 18+
npm install -g @anthropic-ai/claude-code
claude setup-token
```

浏览器走 OAuth，终端打印 `sk-ant-oat01-...`，**只显示一次**。
存为 Secret `CLAUDE_CODE_OAUTH_TOKEN`。

两个注意点：它消耗你的订阅额度，跑重了会挤占本地 Claude Code 的用量；
2026 年初社区有过这类 token 被拒的反复报告，所以工作流里它是
`continue-on-error`，挂了照发邮件并在顶部声明「本次无 LLM 分析」。

### 路 C：完全不配

两个 Secret 都不填也能跑。邮件照发，只是没有题材理由和风险提示，
其余量化字段（评分、高开、量能、形态、板块共振）完全不受影响。

**先按路 A 上线，跑通之后想省钱再换路 B。**

---

## 每天怎么用

| 时间(BJT) | 发生什么 |
|---|---|
| 08:23 | 构建候选池（全市场筛到 400-1600 只，扫隔夜公告） |
| 08:47 | 竞价任务启动，自旋等待 |
| 09:19:40 | T1 快照 —— 撤单前虚拟撮合价 |
| 09:23:30 | T2 快照 |
| 09:25:10 | T3 快照 —— 最终竞价结果 |
| 09:27:20 | T4 补采 —— 只补 T3 漏掉的票（9:25-9:30 价格固定不变） |
| 09:27:40 | Claude 查证题材与风险（硬超时 150 秒） |
| 09:28:30 | **发邮件 + 更新在线面板** |

你在美东时区：北京 09:28 = 美东 21:28（夏令时）/ 20:28（冬令时）。

**收到之后**：

1. 邮件里就是完整表格，直接看
2. 要推进同花顺 → 打开在线面板 `https://sophiasha2026.github.io/daily-report/`
   → 点「复制全部代码」或「仅强」→ 切到同花顺，剪贴板识别框会自动弹出
   → 点「加入自选股/板块股」
3. 或者用邮件附件里的 `竞价_强.txt` / `竞价_中.txt`：
   同花顺 → 自选股版块设置 → 导入 → 文件类型选 TXT

### 关于同花顺

同花顺**没有**通达信那种自定义数据管理器，外部算好的数值导不进去做成
可排序的一列。社区工具链全是「同花顺 → 通达信」方向的，反向不存在。
所以排名只能用**分层板块**（强 / 中 / 观察）表达，理由和风险放在面板里。

仓库里保留了 `src/tdx_export.py`。如果你哪天愿意额外装个通达信
（免费、体积小、可与同花顺共存）当评分看板，那边能做到真正的可排序评分列。

---

## 调参

全部在 `config.yaml`，网页上直接改，Commit 后下一个交易日生效。

### 两个自定义指标

各家软件 9:25 的「量比」口径不一致、不可复现，所以自己定义：

```
GAP_NORM  = 高开幅度 / 当日涨停幅度       # 10/20/30cm 统一可比
AUC_RATIO = 竞价成交额 / 昨日全天成交额    # 竞价量能
```

`AUC_RATIO` 与传统量比的换算（设昨日量 ≈ 5日均量）：

| AUC_RATIO | ≈ 量比 | 含义 |
|---|---|---|
| 0.8% | 1.9 | 常态 |
| **1.5%** | 3.6 | 放量下限（默认） |
| **3.0%** | 7.2 | 明显异动（打分饱和点） |
| 5.0% | 12 | 强异动 |
| **8%** | 19 | 上限，超出多为一字板 |

> 你原始规则里的「竞价量能 ≥ 昨日全天 10%」等价于量比约 24，
> 与「量比 ≤ 10」直接冲突，两条同时用交集接近空集，已改为上表口径。

### 常改的几项

| 想要 | 改哪里 |
|---|---|
| 高开区间改成主板 3%-5% | `gap_norm_min: 0.30` / `gap_norm_max: 0.50` |
| 出 15 只 | `output.top_n: 15` |
| 更严 | `min_score: 60`、`require_positive_slope: true` |
| 放宽量能 | `auc_ratio_min: 0.010` |
| 分层阈值 | `ths_tiers: [80, 65]` |
| 加大板块权重 | `scoring.weights.sector: 0.25`（其余等比缩减，和为 1.0） |
| 恢复 A/B 双榜 | `merge_groups: false` |

### 为什么有 `min_auc_amount_wan`

你选了不限市值。一只 5 亿市值的票竞价成交 20 万也能满足
`AUC_RATIO ≥ 1.5%`，但那个盘口买不进去——比例指标在小盘票上会失真。
所以加了绝对流动性下限 300 万。这个数可以调，但不建议去掉。

---

## 已知风险

| 风险 | 影响 | 处理 |
|---|---|---|
| GH Actions cron 延迟 30 分钟以上 | 错过 9:25 窗口 | 三个错峰入口 + 硬截止 09:26:30 + 告警邮件 |
| runner IP 被国内行情接口限流 | 拿不到数据 | 冒烟测试先验；不通则改用香港轻量服务器（约 24 元/月）跑同一份代码 |
| akshare 随数据源改版失效 | 盘前任务失败 | 时间宽裕、带重试；失败发告警邮件 |
| 仓库 60 天无提交 cron 被停 | 静默失效 | 每日 commit 快照数据自动 keepalive |
| Claude 撞额度 | 无理由文案 | 照发邮件，顶部声明「本次无 LLM 分析」 |
| 腾讯字段索引变动 | 解析异常 | 只用低位稳定字段，涨停价自行推算；冒烟测试会先发现 |

**尚未在真实行情上验证的两点**（冒烟测试会告诉你）：
GitHub runner 能否访问腾讯行情接口；腾讯在 9:15-9:25 返回的「当前价」
是否为虚拟撮合参考价。后者若不符，改成从东财盘前分时接口取，代码里留了位置。

---

## 数据积累

流程每天把全候选池的竞价快照落到 `data/YYYY-MM/`。市面上买不到便宜的
历史竞价数据，只能自己攒。攒够 40 个交易日后，这份数据可以用来标定
`config.yaml` 里所有阈值的真实最优区间。
