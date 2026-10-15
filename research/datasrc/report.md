# 免费数据源普查 + 抢救存档 + 每日存档器设计（2026-10-15）

（研究子代理写成，子代理不能写 .md，由主会话原样落盘。）

实测时间是北京 2026-10-15 04:20 ~ 09:30，本机美东家宽，akshare 1.18.94。新包装在 `research/datasrc/.venv`（`--system-site-packages`，生产环境没动），版本：baostock 0.9.4、mootdx 0.11.7、pytdx 1.72、efinance 0.5.9、adata 2.9.5。

> **时间口径**：一律按 `bjtime.now_bj()` 的真实北京时间。`common.check_blackout()` 在每个请求前和熔断等待中都查一次，北京 15:55~18:30 不发请求，脚本存好进度退出。

> **东财并发**：另一个研究进程 `research/announce/fetch_em.py` 同期在拉东财公告。`common.EMGate` 是跨进程的锁文件信号量：它在跑时这边只占 1 路，它停了才用 2 路，全机东财不超过 2 路。

## 0. 结论先说

1. **大部分事件数据有完整历史、带公告日，不用抢救。**
   - 包括：龙虎榜（明细 / 席位 / 机构 / 营业部，2004 起）、大宗交易（2000 起）、两融个股（2010 起）、股东增减持、解禁（含未来日程）、质押（2014 起）、回购、机构调研（2011 起，接待日 + 公告日）、业绩预告 / 快报 / 报表、研报、东财全市场公告（2015 能按日查）、巨潮全市场公告（2005 起，时间到分钟）。
   - 这批可以直接回填。
2. **只给近期的是情绪和资金类。**
   - 包括：涨停五池 + 次新池（近 30 个自然日）、个股资金流（120 天）、人气榜（每只一年）、千股千评四张表（5~40 个交易日）、人气关键词 / 快讯（只有当前）。
   - 这批已抢救（第 2 节），以后靠每日存档器（第 3 节）。
3. **有便宜的付费替代。**
   - Tushare `limit_list_d`（涨跌停列表，2020 起，含首封 / 末封、炸板、连板）要 5000 积分，约 500 元 / 年。
   - Tushare `moneyflow`（资金流，2010 起）要 2000 积分，约 200 元 / 年。
   - 所以这两样不是「不存就永远没有」，只是免费源没有历史。
4. **新发现：通达信历史分笔和分时到 2005 年。**
   - pytdx 的 `get_history_transaction_data` / `get_history_minute_time_data` 按日期取，分笔带主动买卖标志。
   - 能自算主动买入占比、尾盘抢筹，当天收盘就有，不偷看。
   - 代价是量大：全市场两年是百万级请求。
   - K 线接口 `get_security_bars` 对现在 7 台能连的服务器全部解析失败（struct.error），不能用。
5. **东财 push2 / push2his 今天抖得厉害，资金流基本没抢到。**
   - `push2his .../fflow/daykline` 04:28 起连 8 次就断（RemoteDisconnected），到 09:30 仍断。
   - 同主机 kline 路径 04:3x 能通，04:50 起也断；`push2 clist` 大页 502。
   - datacenter-web、emappdata、np-anotice、push2ex 一直正常。
   - 生产不走这几条路径，不受影响。

## 1. 普查表（按「对起涨标签的价值 × 可得性」排序）

- **价值**：高 = 可能直接解释「20 天涨 50%」的触发或资金；中 = 背景 / 过滤；低 = 关系弱。
- **可得性**：历史够不够做开发集（2024-06..2025-12）、有没有可知时间、本机能不能通。
- **首条**：本机实测能拿到的最早一条。
- 原始记录在 `survey/results.jsonl`，样本在 `survey/samples/`。

| # | 数据 | 接口 / 主机 | 首条（实测） | 可知时间 | 速度 / 限流 | 对标签的用处 | 风险 | 价值 × 可得性 |
|---|---|---|---|---|---|---|---|---|
| 1 | 全市场公告（东财细类） | np-anotice-stock `api/security/ann` 按日期 | 2015-03-02 当日 199 条 | `notice_date` + `display_time` | 2 秒 / 页 | 控制权变更、重组、中标、预增等点火事件 | 另一研究线在用，别同时狂拉 | 高 × 高 |
| 2 | 巨潮全市场公告（分类码 + 时间到分钟） | cninfo `hisAnnouncement/query` column=szse | 2005-03-01 当日 289 条；2001-03 整月 8 条 | `announcementTime` 毫秒 | 0.4 秒 / 页，30 条一页，一天约 45 页 | 同上，第二源，可判盘中 / 盘后 | 沪深都在 szse 列；北交所未核 | 高 × 高 |
| 3 | 业绩预告 | datacenter.eastmoney.com/securities `RPT_PUBLIC_OP_NEWPREDICT` | 2003-04-12，19.1 万条 | `NOTICE_DATE` | 4 秒 | 预增 / 扭亏 | 盘后发布，次日一字多 | 高 × 高 |
| 4 | 龙虎榜明细 | datacenter-web `RPT_DAILYBILLBOARD_DETAILSNEW` | 2004-06-25，26.9 万条 | `TRADE_DATE` 收盘后 | 3~7 秒 | 异动票资金性质 | **含 D1~D30 后验涨跌列（偷看），入库前删** | 中 × 高 |
| 5 | 龙虎榜席位 / 活跃营业部 | `RPT_BILLBOARD_DAILYDETAILSBUY`（/SELL）、`RPT_OPERATEDEPT_ACTIVE` | 2004-06-25；买方 132 万条，营业部 137 万条 | 收盘后 | 同上 | 游资席位 | `RISE_PROBABILITY_3DAY` 是后验 | 中 × 高 |
| 6 | 龙虎榜机构买卖 | `RPT_ORGANIZATION_TRADE_DETAILS` | 2006-07-03，10.9 万条 | 收盘后 | | 机构净买 | | 中 × 高 |
| 7 | 业绩快报 / 业绩报表 | `RPT_FCI_PERFORMANCEE` / `RPT_LICO_FN_CPD` | 1996 / 1990 起 | `UPDATE_DATE` / `NOTICE_DATE` | | 同 3 | | 中 × 高 |
| 8 | 股东增减持 | `RPT_SHARE_HOLDER_INCREASE` | 1994-08，14.7 万条 | `NOTICE_DATE` | | 增持是正向事件 | 生产 corp_events 已用 | 中 × 高 |
| 9 | 机构调研明细 | `RPT_ORG_SURVEY` | 2011-01-15，334.7 万条（一机构一行） | `NOTICE_DATE` + `RECEIVE_START_DATE` | 11 秒 | 调研密度上升 | 必须按公告日对齐，接待日更早（教训 25 同类） | 中 × 高 |
| 10 | 研报 + 评级 + 盈利预测 | reportapi `stock_research_report_em` 逐只 | 600519 到 2017-08（772 篇）；300829 只 14 篇 | 发布日 | 1 秒 / 页 | 首次覆盖、上调评级 | 小票覆盖稀，要逐只 5500 次 | 中 × 中 |
| 11 | 巨潮投资评级（按日全市场） | webapi.cninfo `stock_rank_forecast_cninfo` | 2023-08-17 当日 391 条（更早未测） | 发布日期 | 3 秒 | 同上 | | 中 × 中 |
| 12 | 两融个股（东财） | `RPTA_WEB_RZRQ_GGMX` | 2010-03-31，689.6 万条 | `DATE`，T+1 | 11 秒 | 融资买入异动 | 只覆盖两融标的 | 中 × 高 |
| 13 | 两融明细（交易所） | `stock_margin_detail_sse/szse/bse` | 沪深 2010-04-01 都有；10-13 沪 2000 / 深 2109 / 北 346 行 | **T+1 早上才发** | 1~35 秒 / 天 | 同上 | | 中 × 高 |
| 14 | 千股千评-主力控盘 | `RPT_DMSK_TS_STOCKEVALUATE` 全市场分页 | 约 37 个交易日（08-17 起） | `TRADE_DATE` | 500 行 / 页 | 资金流替代（东财口径） | 只有滚动窗口 | 中 × 低（已抢救） |
| 15 | 千股千评-关注指数 + 全市场名次 | `RPT_STOCK_MARKETFOCUS` | 30 个交易日（08-26 起） | `TRADE_DATE` | | 关注度突增 | 同上 | 中 × 低（已抢救） |
| 16 | 千股千评当日快照 | `RPT_DMSK_TS_STOCKNEW` | 只有当天，5200 只 | 当天 | 11 页 | 同 14 | 只有快照 | 中 × 低（已存，每日存） |
| 17 | 东财人气榜个股历史 | emappdata `getHisList` | 每只一年（2025-10-15 起，366 行） | `calcTime` 日期；当天那行是盘中值 | 约 1.2 秒 / 只 | 散户关注度 | **北交所要写 `SZ920xxx`** | 中 × 低（已抢救） |
| 18 | 人气关键词 | emappdata `getHotStockRankList` | 只有当前 | 整点 | 2 秒 | 题材标签的唯一免费近似 | 只有快照 | 中 × 低（每日存） |
| 19 | 涨停 / 昨日涨停 / 强势 / 炸板 / 跌停 / 次新 六池 | push2ex `getTopic*Pool` | **只有近 30 个自然日** | 收盘后（首封 / 末封到秒） | 1 秒 / 天 | 首封时间、炸板、连板、封单额 | 只有快照；Tushare 2020 起付费 | 高 × 低（已抢救） |
| 20 | 个股资金流 | push2his `fflow/daykline` | 120 个交易日 | 收盘后 | **04:28 起被掐** | 主力净流入 | Tushare `moneyflow` 2010 起（付费） | 中 × 低（没抢到） |
| 21 | 概念板块列表 / 成分 / 日 K | push2 clist、push2his kline | 列表 506 个、BK0815 成分 67 只；日 K 没拉到 | | 04:50 后 502 / 断 | 题材轮动 | **成分没有历史进出**，回填有幸存者偏差 | 中 × 低（每日存成分） |
| 22 | 同花顺概念列表 | q.10jqka `stock_board_concept_name_ths` | 375 个 | | 35 秒 | 同上 | 成分要真浏览器（5b） | 中 × 低 |
| 23 | 通达信历史分笔（带买卖方向） | pytdx `get_history_transaction_data` | 2005-01-04 有 688 笔；2000 年空 | 当天收盘 | 0.4 秒 / 页 | 主动买卖占比、尾盘抢筹 | 量大；38 台服务器里约 7 台能连 | 中 × 中 |
| 24 | 通达信历史分时 | pytdx `get_history_minute_time_data` | 2005 起 | 当天收盘 | 0.4 秒 / 只·天 | 日内形态 | 同上 | 中 × 中 |
| 25 | BaoStock 日 K（含逐日 `isST`） | baostock.com | 600000 到 1999-11-10 | 收盘后 | **65 秒 / 只** | 历史 ST 逐日，修回测 ST 静态名单 | 单连接，慢 | 中 × 中 |
| 26 | 千股千评-综合评分 / 参与意愿 | `RPT_STOCK_HISTORYMARK` / `RPT_STOCK_PARTICIPATION` | 30 / 8 个交易日 | | | 弱 | | 低 × 低（已抢救） |
| 27 | 人气榜粉丝结构 | emappdata `getHisProfileList` | 每只一年 | `calcTime` 带时刻（23:00） | 1 秒 / 只 | 新粉涌入 | | 低 × 低（已抢救） |
| 28 | 大宗交易 | `RPT_DATA_BLOCKTRADE` | 2000-08-29，68.3 万条 | 收盘后 | | 折价大宗多是减持通道 | | 低 × 高 |
| 29 | 限售解禁 | `RPT_LIFT_STAGE` | 2010-01-04 ~ 2035-10（含未来） | 日程上市时就公开 | | 解禁压力 | | 低 × 高 |
| 30 | 股权质押比例 | `RPT_CSDC_LIST` | 2014-03-07，周度 | 中登周五 | | 风险 | 周频 | 低 × 高 |
| 31 | 回购 | `RPTA_WEB_GETHGLIST_NEW` | 2006-03-02 | `NOTICEDATE` | | 回购预案 | | 低 × 高 |
| 32 | 董监高持股变动 | `RPT_EXECUTIVE_HOLD_DETAILS` | 2004-11-29 | **变动日，不是公告日** | | 弱 | 要另找公告日 | 低 × 中 |
| 33 | 新浪龙虎榜 | `stock_lhb_detail_daily_sina` | 2015-01-05 | | 4 秒 | 东财龙虎榜兜底 | | 低 × 高 |
| 34 | 新闻联播文字稿 | `news_cctv` | 2018-01-05 能取；2016 报错 | 当晚 | 10~40 秒 / 天 | 政策题材 | | 低 × 中 |
| 35 | 东财全球快讯 / 财联社电报 | np-weblist / cls | 只有最近 200 / 20 条 | 到秒 | 2 秒 | 催化 | 只能 LLM 读（硬约束 1） | 低 × 低（每日存） |
| 36 | 个股新闻（东财搜索） | search-api-web | akshare 在 pandas 3 下报错，**未核实** | | | | | 低 × 低 |
| 37 | 互动易 / 上证 e 互动 | irm.cninfo / sns.sseinfo | 只给最新一页 | 提问时间 | 慢 | 弱 | | 低 × 低 |
| 38 | 北向持股个股 | `RPT_MUTUAL_STOCK_NORTHSTA` | 服务器繁忙 | | | 2024-08 起不按日披露 | | 低 × 低 |
| 39 | 通达信 K 线 / 财务 / F10 / 除权 | pytdx / mootdx | K 线解析失败；财务只有最新一期 | | | 除权表可当复权校验 | 协议变了，库没跟上 | 低 |
| 40 | 通达信历史财务包 | mootdx `Affair.files()` | 148 个季度包 | 报告期，**无公告日** | | 弱 | 按报告期对齐就是偷看（教训 25） | 低 |
| 41 | BaoStock 5 分钟线 / 业绩预告 / 季频财务 | baostock.com | 5 分钟到 2020-04-22；预告和季频带公告日 | 公告日 | 慢 | 和 3 重复 | | 低 × 中 |
| 42 | chenditc/investment_data | DoltHub SQL API；release 568 MB | 到 2026-10-14 | | 点查 1 秒 | 日线抽检 / Alpha158 对照臂 | | 低 × 高 |
| 43 | adata | 多源封装 | 没新数据；百度资金流 403 + 验证码（不绕） | | | | | 低 |
| 44 | efinance | 东财封装 | 和 akshare 同一批路径 | | | | 没新数据 | 低 |
| 45 | 上交所交易公开信息 | query.sse.com.cn | 参数不对，**未核实** | | | | | 低 |
| 46 | 雪球关注 / 讨论榜 | xueqiu | NetworkError，疑似要 cookie | | | | | 低 |

**Tushare 门槛**（2026-10-15 读官方文档，没注册）：

| 积分 | 年费 | 每分钟 | 总量 | 能用的接口 |
|---|---|---|---|---|
| 120 | 0 | 50 次 | 每天 8000 次 | 只有不复权日线 |
| 2000 | 约 200 元 | 200 次 | 每接口每天 10 万次 | 含 `moneyflow`（个股资金流，2010 起） |
| 5000 | 约 500 元 | 500 次 | 常规数据无上限 | 含 `limit_list_d`（2020 起） |

新闻 / 快讯 / 新闻联播是单独权限，1000 元。

**chenditc 抽检**（`check_dolthub.py`）：60 个随机（代码，日期）都找到；最近 15 个交易日收盘价 39/40 和本机一致，不一致的 603120 疑似之后除权。成交量单位是本机 / 100，成交额 / 1000。可以当日线第二源。

## 2. 抢救存档（`research/datasrc/archive/`，共约 75 MB）

| 数据 | 文件 | 覆盖 | 规模 |
|---|---|---|---|
| 涨停池 | `ztpool/zt/<日>.parquet` | 09-16 ~ 10-14，15 天 | 886 行 |
| 昨日涨停池 | `ztpool/zt_prev/` | 同上 | 881 行 |
| 强势股池 | `ztpool/strong/` | 同上 | 2903 行 |
| 炸板池 | `ztpool/zb/` | 同上 | 280 行 |
| 跌停池 | `ztpool/dt/` | 09-15 ~ 10-14 | 197 行 |
| 次新股池 | `ztpool/subnew/` | 09-16 ~ 10-14 | 2430 行 |
| 人气榜排名 | `hotrank.parquet` | 2025-10-15 ~ 2026-10-15，5573 只 | 200.9 万行 |
| 人气榜粉丝结构 | `hotprofile.parquet` | 同上 | 200.9 万行 |
| 千股千评-主力控盘 | `dc/RPT_DMSK_TS_STOCKEVALUATE.parquet` | 08-17 ~ 10-14，37 天 | 19.2 万行 |
| 千股千评-关注指数 | `dc/RPT_STOCK_MARKETFOCUS.parquet` | 08-26 ~ 10-14，30 天 | 15.6 万行 |
| 千股千评-综合评分 | `dc/RPT_STOCK_HISTORYMARK.parquet` | 08-26 ~ 10-14，30 天 | 15.6 万行 |
| 千股千评-参与意愿 | `dc/RPT_STOCK_PARTICIPATION.parquet` | 09-28 ~ 10-14，8 天 | 2.6 万行 |
| 千股千评当日快照 | `dc/RPT_DMSK_TS_STOCKNEW.parquet` | 10-14 | 5200 行 |
| 个股资金流 120 天 | `fflow/` | **截至 09:30 落盘 0 只**，后台进程北京 15:55 自停，18:30 后可续跑 | |

- 涨停池：首封 `fbt`、末封 `lbt` 到秒，`lbc` 连板数，`zbc` 炸板次数，`fund` 封单额；价格保持「元 × 1000」整数。
- 人气榜：当天那一行是盘中值，用时去掉；排名推断是收盘后的值，只能用于次日（推断，没逐条核）。
- 主力控盘表可以部分替代没抢到的资金流（37 天对 120 天）。

续跑：`rescue_ztpool.py`（跑过的日子跳过）、`rescue_perstock.py fflow|hotrank|hotprofile`（进度文件续跑）、`rescue_perstock.py merge`、`rescue_dc.py`。

## 3. 每日存档器设计（只设计，不改生产）

### 存什么

只存「只给近期、以后拿不到」的：六池（每天 6 个请求）、千股千评当日快照 + 主力控盘 + 关注指数（每天约 33 页）、人气关键词（约 100）、概念列表和当前成分（约 500）、快讯（2）、人气榜全量（每周一次）、个股资金流全量（每月一次，不通时用主力控盘表顶）。合计每天约 2 MB，全年约 1 GB 以内。

### 存哪里

`data/events/<种类>/YYYY-MM/<日>.parquet`，**gitignore，不入库**（一年近 1 GB 会撑大仓库，同步和推送都要搬）。每周打包备份到仓库外。状态记 `state/events_status.json`，控制台读它（教训 16）。

### 放进流程的哪一步

单独一个计划任务 `DailyReport-Local-Events`：北京 20:00 起每 30 分钟重试到次日 08:00，带 `--if-needed`。理由：16:00~18:30 是两条线取数发信时段，不能抢东财配额；涨停池、千股千评 17:00 后才稳定；单独任务坏了不影响发信。

### 伪代码

```python
# tools/archive_events.py（设计稿）
ITEMS = [("ztpool", fetch_ztpools, "daily", 120), ("comment_snap", ..., "daily", 300),
         ("evaluate", ..., "daily", 300), ("focus", ..., "daily", 300),
         ("hot_keyword", ..., "daily", 300), ("concept_cons", ..., "daily", 1200),
         ("flash", ..., "daily", 60), ("hotrank_full", ..., "weekly", 3 * 3600),
         ("fflow_full", ..., "monthly", 4 * 3600)]

def main(if_needed=True):
    localenv.load(); ds.http_timeout()                 # 每个请求都有超时（教训 43）
    target = bjtime.last_closed()
    with file_lock("events"):                           # 进程锁（教训 23）
        arm_deadline(MAX_RUN)
        for name, fn, freq, limit in ITEMS:
            if not due(...) or (if_needed and done(name, target)): continue
            if in_blackout(): break                     # 北京 15:55~18:30 不发请求
            try:
                with step_deadline(limit), em_gate(slots=1), breaker(n=5, pause=600, max_trips=3):
                    df = fn(target)
                check_date(df, target)                  # 产物日期对得上目标日才算完成（教训 22 / 27）
                drop_lookahead_cols(df)                 # 龙虎榜 D1~D30 这类后验列不落盘
                write_atomic(path, df)
                status[name] = {"date": str(target), "rows": len(df), "ok": True}
            except Exception as e:                      # fail-open，记下来
                status[name] = {"date": str(target), "ok": False, "err": repr(e)[:200]}
            save_status(status)
```

### 和生产抢配额

时间错开（20:00 后跑、15:55~18:30 硬禁区）；东财只开 1 路、熔断 3 次放弃当天；易掐路径排最后、每月一次；不碰腾讯、新浪；selftest_gui 钉每项带时限、异常不抛、禁区不发请求。

## 4. 下一步建议（按性价比）

1. 公告诊断（已由另一路做完，见 research/announce/report.md：结论是公告类别没有增量）。
2. **修 ST 名单**：用 BaoStock 逐日 `isST` 补回测的 ST 历史，替代静态 `cache/st_codes.json`。只拉候选池里出现过的票。
3. **存档器先小后大**：先只上六池和千股千评三张表（一天约 40 个请求），跑一个月看稳定，再加人气榜和资金流。
4. **Tushare 先验证再买**：用抢救的涨停池和主力控盘表看起涨前后这些字段长什么样，有信号再买。
5. **通达信分笔只拉样本**：只拉命中 + 同日随机对照那几万个（代码，日期），算主动买入占比。

## 5. 文件（`research/datasrc/`）

`common.py`（禁区、跨进程东财闸、熔断、超时）、`survey.py`（普查探针，结果 `survey/results.jsonl`）、`rescue_ztpool.py`、`rescue_perstock.py`、`rescue_dc.py`、`check_dolthub.py`。`chain.sh` 是临时串跑脚本，可以删。
