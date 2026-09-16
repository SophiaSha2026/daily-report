"""
盘前候选池构建（北京时间 08:35 运行，时间宽裕）。

两阶段收缩，避免对全市场 5000+ 只票逐个拉日线：
  阶段1  仅用一次全市场快照做廉价过滤  -> 约 400-800 只
  阶段2  只对存活者拉 90 日日线算形态  -> 每只 ~0.35s

产出 cache/universe.parquet，供 09:25 的竞价任务直接读取。
"""
from __future__ import annotations

import sys
import os
import time
import logging
import datetime as dt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datasource as ds
from datasource import limit_pct, limit_price

ROOT = Path(__file__).resolve().parent.parent
TZ = dt.timezone(dt.timedelta(hours=8))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("premarket")


def cfg() -> dict:
    """合并 config.yaml + state/learned.yaml，和竞价主流程同一份配置。"""
    import cfg as _cfg
    return _cfg.load()


# ---------------------------------------------------------------------
def assert_pre_open(now: dt.datetime | None = None) -> None:
    """建池的时刻闸：09:10 之后一律不建。

    09:15 集合竞价一开，腾讯快照就从「上一交易日收盘态」翻成「今日竞价态」：
    成交额变成竞价额、涨跌幅变成竞价涨幅。按它建的池子里 prev_amount 是
    竞价额，量比整体放大两个数量级，09:25 那一跑会被量能条件把全榜剔光，
    发一封没有任何解释的空清单；如果只有价翻篇、量还没清零，昨日涨停会
    整批丢成 False，炸板误标，清单分组全错却毫无异常迹象。
    留 5 分钟余量给 stage2（拉日线要 35~55 秒）。抛出去由 __main__ 的
    except 发告警邮件，竞价线随后按「候选池不是今天的」放弃，不发脏清单。
    """
    now = now or dt.datetime.now(TZ)
    if (now.hour, now.minute) >= (9, 10):
        raise RuntimeError(
            f"已过 09:10（现在 {now:%H:%M:%S}），盘前快照可能已翻成竞价态，"
            f"本次不建池")


def last_closed_day(tds: set[str], now: dt.datetime | None = None) -> str:
    """最近一个已收盘的交易日（15:05 之后算当天）。

    盘前跑时就是昨天那个交易日 —— 腾讯此刻给的最新价/成交额正是它的
    收盘价和全天成交额，spot_all 拿它核对 f[30] 时间戳。
    15:05 这条分界和 local_run.last_closed_trade_day 一致。
    日历拿不到（tds 空）就返回空串，调用方据此跳过校验而不是拿错日期去比。
    """
    if not tds:
        return ""
    now = now or dt.datetime.now(TZ)
    d = now.date()
    if (now.hour, now.minute) < (15, 5):
        d -= dt.timedelta(days=1)
    for _ in range(15):
        if d.isoformat() in tds:
            return d.isoformat()
        d -= dt.timedelta(days=1)
    return ""


def include_mask(prev_limit_up: pd.Series, prev_gain: pd.Series,
                 prev_turnover_pct: pd.Series, amount_rank: pd.Series,
                 inc: dict) -> pd.Series:
    """入池条件（满足任一）。生产 stage1 和回填 build_pool **共用这一份**。

    以前两边各写各的：生产是「昨日涨停 / 昨涨>=5% / 换手>=5% / 成交额前600」，
    回填是「昨日涨停 / 昨涨>=5% / amount_ratio_5d>=1.5 / 成交额前600」。
    结果 8 个重叠日里两个池子日均只重合 70%，差异的九成以上出自这一条：
    回填 24.0% 的行靠 amount_ratio_5d 进池（其中 14.5% 生产四条规则一条都
    不满足），而生产每天约 21% 的准入候选（重叠日 36/168）靠换手率进池、
    回填里根本没有同类样本（2026-09-16 审计）。优化器的「当日竞争对手集合」
    和生产不是一个集合，学出来的权重就不是给生产用的。

    阈值一律从 config 的 include_if 取，函数里不许出现字面量：写死过的
    5.0 / 600 正是 config 里 6/11 个键是死配置的原因。
    """
    keep = pd.Series(False, index=prev_limit_up.index)
    if inc.get("yesterday_limit_up"):
        keep |= prev_limit_up.fillna(False).astype(bool)
    if inc.get("yesterday_gain_pct_gte") is not None:
        keep |= prev_gain >= inc["yesterday_gain_pct_gte"]
    if inc.get("turnover_pct_gte") is not None:
        keep |= prev_turnover_pct.fillna(0.0) >= inc["turnover_pct_gte"]
    if inc.get("amount_rank_top"):
        keep |= amount_rank <= inc["amount_rank_top"]
    return keep


def stage1(spot: pd.DataFrame, c: dict) -> pd.DataFrame:
    """只用全市场快照做过滤。此时快照反映的是上一交易日收盘状态。"""
    u = c["universe"]
    inc = u["include_if"]
    df = spot.rename(columns={"代码": "code", "名称": "name"}).copy()
    for col in ("最新价", "涨跌幅", "成交额", "换手率", "总市值"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    if u["exclude_st"]:
        df = df[~df["name"].astype(str).map(ds.is_st_name)]
    if u["min_listed_days"]:
        # 名字带 N/C 前缀 = 上市第 1~5 天，零成本的兜底；真正的
        # 「日线不足 min_listed_days 根」判据在 stage2（那里才有日线）
        df = df[~df["name"].astype(str).map(ds.is_new_listing)]
    df = df[df["成交额"].fillna(0) > 0]                                # 停牌
    if u["exclude_bj"]:
        # 北交所三个代码段：8/4 开头的老段和 920xxx 新段
        df = df[~df["code"].astype(str).str[0].isin(["8", "4", "9"])]

    if u["max_mktcap_yi"] or u["min_mktcap_yi"]:
        # 腾讯快照给不出总市值（低位字段里没有，硬约束 6 不许读高位字段），
        # 这两个键一旦非 0 会把整池筛空、当天发一封「今日无标的」的空榜邮件，
        # 用户分不出是市场没票还是过滤器坏了。宁可报错发告警邮件。
        cap = (pd.to_numeric(df["总市值"], errors="coerce")
               if "总市值" in df.columns else None)
        if cap is None or not cap.notna().any():
            raise RuntimeError(
                "配置要求按总市值筛选，但快照没有可用的「总市值」列；"
                "先补数据源，否则候选池会被筛空")
        # 有市值的行才按市值判，NaN 的行留下：以后补上市值源时多半是
        # 部分覆盖，NaN 与任何数比较都是 False，直接比会把没覆盖到的票全剔掉
        if u["max_mktcap_yi"]:
            df = df[cap.isna() | (cap <= u["max_mktcap_yi"] * 1e8)]
            cap = cap.loc[df.index]
        if u["min_mktcap_yi"]:
            df = df[cap.isna() | (cap >= u["min_mktcap_yi"] * 1e8)]

    df["prev_amount"] = df["成交额"]
    df["prev_gain"] = df["涨跌幅"]
    df["lim"] = [limit_pct(r.code, r.name) for r in df.itertuples()]
    df["prev_limit_up"] = df["prev_gain"] >= df["lim"] - 0.4

    # 换手率靠腾讯 index 38，取不到时 _turnover 返回 0 —— 那会把「换手>=5%」
    # 这条规则静默关掉（生产池里约三分之一的票只靠它进来）。成片为 0
    # 就报错，让 main 的 except 发告警邮件（教训 26：规则要真的被执行）。
    turn = (df["换手率"] if "换手率" in df.columns
            else pd.Series(0.0, index=df.index))
    if (len(df) and inc.get("turnover_pct_gte") is not None
            and float((turn.fillna(0) > 0).mean()) < 0.5):
        raise RuntimeError("快照换手率字段缺失，换手入池规则无法执行")

    keep = include_mask(df["prev_limit_up"], df["prev_gain"], turn,
                        df["成交额"].rank(ascending=False), inc)
    df = df[keep].copy()
    if u["exclude_yesterday_limit_down"]:
        # NaN 留下（`~le` 而不是 `gt`）：回填那边同一条规则也必须是这个
        # NaN 语义，否则除权次日那批行两边一个留一个丢
        df = df[~df["prev_gain"].le(u["yesterday_drop_pct_max"])]
    df = df.nlargest(min(len(df), u["max_candidates"]), "成交额")
    # 空池不是一个合法结果：真实市场每天 1000~1400 只满足入池条件。
    # 空了必然是快照或配置出了问题，而下游全是静默的 —— universe_meta
    # 的日期是对的、退出码 0、竞价线照跑，最后发一封「今日无标的」的空榜
    # 邮件（config.send_when_empty=true）。让它在这里就响（教训 16/26）。
    if len(df) == 0:
        raise RuntimeError(f"阶段1 过滤后候选池为空（入池前 {n0} 只），"
                           f"快照或 universe 配置有问题")
    log.info("阶段1: %d -> %d", n0, len(df))
    return df.reset_index(drop=True)


# stage2 产出的六个形态列。空池时要靠它给出完整列（见函数末尾的 concat）
_SHAPE_COLS = ("pos_pct_60d", "ma_bull", "platform_high", "board_height",
               "prev_broken_board", "amount_ratio_5d")


def stage2(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """
    对存活者拉 90 日日线，算位置/形态/连板高度。

    日线是并发拉的（datasource.daily_hist_many）：1600 只串行要 9 分钟以上，
    盘前到竞价只有 50 分钟，串行没有余量。源的优先级和熔断在 datasource 里。
    """
    today = dt.datetime.now(TZ).strftime("%Y-%m-%d")
    end = today.replace("-", "")
    start = (dt.datetime.now(TZ) - dt.timedelta(days=140)).strftime("%Y%m%d")
    look = c["screen"]["breakout_lookback"]
    min_days = int(c["universe"]["min_listed_days"] or 0)

    t0 = time.time()
    hists = ds.daily_hist_many([r.code for r in df.itertuples()], start, end)
    log.info("日线拉取完成 %d 只，耗时 %.0fs，来源 %s",
             len(hists), time.time() - t0, ds.hist_source_stats())

    rec = []
    empty = 0
    young: list[int] = []
    for i, r in enumerate(df.itertuples()):
        h = hists.get(r.code)
        d = dict(pos_pct_60d=0.5, ma_bull=False, platform_high=1e9,
                 board_height=0, prev_broken_board=False,
                 amount_ratio_5d=1.0)
        # 「拉不到日线」和「拿到了但根数不足」必须分开：前者是源降级，
        # 混为一谈的话源一挂整池都被当次新剔光（140 天窗口约 95 根 K，
        # 够判 60）。min_listed_days 以前是死配置，上市第 2~5 天的票
        # 照进候选池，而面板上写着已剔次新（2026-09-16 审计）。
        if h is not None and len(h) and "日期" in h.columns:
            # 盘中/盘后跑时日线尾根就是**今天**那根未定型 K 线（三路实测：
            # 2026-09-16 收盘后东财/腾讯/新浪的最后一根都是 09-16）。
            # 所有形态字段的语义是「截至昨天」，混进今天那根等于错位一天：
            # 用 09-16 真实池子（1012 只取 98 只）复现，保留今天那根后
            # 32 只昨日涨停里 20 只 board_height 归零、炸板标记翻转 33/98、
            # platform_high 含当日最高价后「突破」永远为 False。
            # 和 learn/backfill 的 shift(1) 口径、形态线 pullback.py 的
            # `h[h["日期"] < today]` 都是同一条规矩。盘前跑时这里是空操作。
            h = h[h["日期"].astype(str).str[:10] < today].reset_index(drop=True)
        if h is not None and min_days and len(h) < min_days:
            young.append(i)
        if h is None or len(h) < 25:
            empty += 1
            rec.append(d)
            continue

        cl = pd.to_numeric(h["收盘"], errors="coerce")
        hi = pd.to_numeric(h["最高"], errors="coerce")
        am = pd.to_numeric(h["成交额"], errors="coerce")
        pc = pd.to_numeric(h["涨跌幅"], errors="coerce")
        w = cl.tail(60)
        rng = w.max() - w.min()
        d["pos_pct_60d"] = float((cl.iloc[-1] - w.min()) / rng) if rng > 0 else 0.5
        if len(cl) >= 20:
            m5, m10, m20 = (cl.rolling(k).mean().iloc[-1] for k in (5, 10, 20))
            d["ma_bull"] = bool(m5 > m10 > m20)
        d["platform_high"] = float(hi.tail(look).max())
        if len(am) >= 6:
            # 分母是**不含昨日**的前 5 日均额。只写进 universe.parquet，
            # 生产里目前没有消费者（不入 AuctionFeature、不进 detail.csv）：
            # 两阶段收缩下这个比值要全市场日线才算得出来，做不了入池条件。
            # 判 `> 0` 而不是 `or 1.0`：bool(nan) 是 True，`float(nan) or 1.0`
            # 得到的还是 nan，而且分子那一侧的 NaN 根本盖不住。
            # NaN 落进 parquet 不报错，将来谁把它接进打分就会静默判成「未放量」。
            base = float(am.tail(6).iloc[:-1].mean())
            last = float(am.iloc[-1])
            if base > 0 and last == last:          # NaN != NaN，两边都落回默认 1.0
                d["amount_ratio_5d"] = last / base
        lim = r.lim - 0.4
        # 连板高度按「日线涨跌幅」回数，而日线三路里只有东财那路是复权真值
        # （chg_adj），腾讯/新浪是相邻不复权收盘价之比，分红除权日会错。
        # 最近的那一根换成快照给的 prev_gain（腾讯昨收已含除权调整），
        # 至少让「昨天涨没涨停」这一根是对的；更早几根等复权源到位再统一。
        seq = pc.tolist()
        if seq:
            seq[-1] = float(r.prev_gain)
        n = 0
        for v in reversed(seq):
            if v >= lim:
                n += 1
            else:
                break
        d["board_height"] = n
        # 昨日炸板：盘中触及涨停但收盘未封。涨停价用 limit_price（四舍五入
        # 到分，交易所口径），不要写 `昨收×(1+lim/100) - 0.01`：那个 −0.01
        # 把阈值整体放宽一分，428 万行实测 15.5% 的「炸板」是最高价恰好
        # = 涨停价 − 0.01 的误判，白扣 15 分（详见 datasource.limit_price_arr）
        if len(hi) and r.prev_gain < lim:
            prev_close = float(cl.iloc[-2]) if len(cl) >= 2 else 0
            # 传 code：北交所封板价是向下取整，四舍五入会高 0.01，
            # 恰好封在涨停价的 920 段票会漏判炸板（datasource.limit_price）
            if prev_close and float(hi.iloc[-1]) >= limit_price(prev_close,
                                                                r.lim, r.code):
                d["prev_broken_board"] = True
        rec.append(d)

    df.attrs["missing_hist"] = int(empty)
    if empty:
        log.warning("阶段2: %d/%d 只没拿到日线，形态字段用默认值",
                    empty, len(df))
    # 显式给列名：池子为 0 只时 pd.DataFrame([]) 一列都没有，main() 的
    # df[cols] 会抛 KeyError，告警邮件里只剩一句看不懂的
    # 「['prev_broken_board', ...] not in index」，指不到真正的原因
    out = pd.concat([df.reset_index(drop=True),
                     pd.DataFrame(rec, columns=list(_SHAPE_COLS))], axis=1)
    if young:
        frac = len(young) / max(len(out), 1)
        if frac > 0.05:
            # 次新一天不会有几十只。超过 5% 只可能是日线源降级（返回的根数
            # 普遍偏少），这时剔除等于误杀整池，只告警不动手
            log.warning("阶段2: 日线不足 %d 根的有 %d 只（%.1f%%），"
                        "疑似日线源降级，本次不剔次新",
                        min_days, len(young), frac * 100)
        else:
            log.info("阶段2: 次新剔除 %d 只（日线不足 %d 根）",
                     len(young), min_days)
            out = out.drop(index=young).reset_index(drop=True)
    out.attrs["missing_hist"] = int(empty)
    log.info("阶段2 完成 %d 只，耗时 %.0fs", len(out), time.time() - t0)
    return out


def attach_sector(df: pd.DataFrame) -> pd.DataFrame:
    """行业归属 + 板块昨日涨停家数。缺缓存则降级为「未分类」。"""
    p = ROOT / "cache" / "sector_map.parquet"
    if not p.exists():
        log.warning("板块缓存缺失，本次板块共振失效（跑一次 refresh_meta 即可）")
        df["sector"] = "未分类"
    else:
        m = pd.read_parquet(p).set_index("code")["sector"].to_dict()
        df["sector"] = df["code"].map(m).fillna("未分类")
    cnt = (df[df["prev_limit_up"]].groupby("sector").size().to_dict())
    cnt.pop("未分类", None)          # 占位符不参与板块共振，理由见 run_auction
    df["sector_prev_limitups"] = df["sector"].map(cnt).fillna(0).astype(int)
    return df


def attach_blacklist(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """隔夜公告关键词命中。接口不稳，失败则全部置 False 并告警。"""
    df["blacklisted"] = False
    kws = c["announcement_blacklist"]
    try:
        import akshare as ak
        # 接口要 YYYYMMDD：它按 date[:4]/date[4:6]/date[6:] 切。以前传的是
        # 带横杠的 2026-09-15，切出 "2026--0-9-15"，服务端忽略过滤返回全库
        # 500 页，每天白跑 6~15 分钟，然后匹配的列还是「名称」不是「公告标题」，
        # 从上线起一次都没命中过。
        # 隔夜公告多数带的是当天日期，也有前一晚就挂出来的，两天都查。
        now = dt.datetime.now(TZ)
        frames = []
        for day in (now, now - dt.timedelta(days=1)):
            try:
                x = ak.stock_notice_report(symbol="全部",
                                           date=day.strftime("%Y%m%d"))
                if x is not None and len(x):
                    frames.append(x)
            except Exception as e:  # noqa: BLE001
                log.warning("公告接口 %s 失败: %s", day.strftime("%m-%d"), e)
        if frames:
            nt = pd.concat(frames, ignore_index=True)
            col_c = next(x for x in nt.columns if "代码" in x)
            col_t = ("公告标题" if "公告标题" in nt.columns
                     else next(x for x in nt.columns if "标题" in x))
            hit = {str(r[col_c]).zfill(6) for _, r in nt.iterrows()
                   if any(k in str(r[col_t]) for k in kws)}
            df["blacklisted"] = df["code"].isin(hit)
            log.info("公告黑名单：%d 条公告，命中 %d 只",
                     len(nt), int(df["blacklisted"].sum()))
    except Exception as e:  # noqa: BLE001
        log.warning("公告接口不可用(%s)，本次不做公告过滤", e)
    return df


def main() -> int:
    c = cfg()
    today = dt.datetime.now(TZ).strftime("%Y-%m-%d")
    tds: set[str] = set()
    try:
        tds = set(ds.trade_dates())
        if today not in tds:
            log.info("%s 非交易日", today)
            return 0
    except Exception:  # noqa: BLE001
        if dt.datetime.now(TZ).weekday() >= 5:
            return 0

    assert_pre_open()
    df = stage1(ds.spot_all(expect_date=last_closed_day(tds) or None), c)
    df = stage2(df, c)
    missing_hist = int(df.attrs.get("missing_hist", 0))   # 后面的 merge 可能丢 attrs
    df = attach_sector(df)
    df = attach_blacklist(df, c)

    cols = ["code", "name", "lim", "prev_amount", "prev_gain", "prev_limit_up",
            "prev_broken_board", "board_height", "pos_pct_60d", "ma_bull",
            "platform_high", "amount_ratio_5d", "sector",
            "sector_prev_limitups", "blacklisted"]
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    # 先写临时文件再改名：控制台「候选池」按钮和竞价流程可能同时写这个文件
    tmp = out / "universe.parquet.tmp"
    df[cols].to_parquet(tmp, index=False)
    os.replace(tmp, out / "universe.parquet")
    # 给竞价任务判新鲜度用：日期对不上就视同缺失（不发脏清单）。
    # missing_hist 是阶段 2 没拿到日线、形态字段用了默认值的只数。
    (out / "universe_meta.json").write_text(
        __import__("json").dumps({"date": today, "count": int(len(df)),
                                  "missing_hist": missing_hist},
                                 ensure_ascii=False),
        encoding="utf-8")
    log.info("候选池已写入 %d 只", len(df))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        log.exception("盘前任务失败")
        try:
            from mailer import send_alert
            send_alert(f"盘前候选池构建失败：{type(e).__name__}: {e}\n"
                       f"今日竞价任务将无法运行。")
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
