"""
爆发线的数据回填：三年日线 + 历史股本 + 股东人数。

    python src/breakout/backfill.py --stage update    每日增量：追加最近一个交易日（秒级）
    python src/breakout/backfill.py --stage refresh   全量重拉新浪日线 + 股东人数（约 70 分钟）
    python src/breakout/backfill.py --stage sina      新浪三年日线，断点续传（首次回填）
    python src/breakout/backfill.py --stage daily     腾讯日线（兜底源，最慢，25~40 分钟）
    python src/breakout/backfill.py --stage shares    历史流通股本
    python src/breakout/backfill.py --stage holders   股东人数
    python src/breakout/backfill.py --stage all

每日增量为什么走腾讯快照而不是新浪
----------------------------------
新浪的日线接口没有「只要最近几根」的参数，每只都回整段历史，5515 只
四进程要 70 分钟，每天下午这么拉一遍太慢。腾讯批量快照一次给全市场
5548 只当天的 OHLCV + 成交额 + 换手率，4 秒。收盘后快照就是当天的日线，
`--stage update` 把它追加进 daily.parquet。

两个坑（都在 stage_update 里处理）：
  · 腾讯快照的成交量单位**按板块不同**：主板/创业板/北交所是手，科创板是股。
    不按代码段猜，按「换手率 × 流通股本」核对，哪个单位对得上用哪个。
  · 除权除息日新浪的前复权历史会整体重算，只追加当天一根会留下断崖。
    腾讯的昨收是复权后的，和历史最后一根收盘对不上就说明发生了除权，
    这只票整段从新浪重拉。

为什么要限速
------------
2026-09-12 实测：腾讯日 K 接口 5 并发连拉 200+ 只之后会限流，返回的不是
JSON 而是一段空内容，持续数分钟。CLAUDE.md 硬约束 4 说的「workers 不超过 5」
是针对快照接口的，日 K 这条更严，光靠限制并发数不够，还要**限速**。

所以这里是三并发 + 每只之间 200ms + 每 120 只休 8 秒 + 指数退避重试。
慢一点无所谓，这是一次性的回填；日常增量只拉最近 5 根。

断点续传
--------
每 200 只落一个分片 data/breakout/raw/daily_<nnn>.parquet，进度记在
data/breakout/raw/done.json。中途挂掉再跑会跳过已完成的代码，不重头来。
回填要跑半小时以上，没有断点续传等于每次网络抖动都从零开始。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "breakout"
RAW = OUT / "raw"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backfill")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

# 限速参数。实测出来的，不要凭感觉调大（见模块 docstring）
WORKERS = 3
SLEEP_EACH = 0.20      # 每只之间
BATCH = 120            # 每多少只休一次
SLEEP_BATCH = 8.0
RETRY = 2              # 见 COOLDOWN：限流时快速失败比反复重试划算

# 限流是**累积配额**，不是瞬时速率：一旦触发，接下来一段时间怎么重试都是白费。
# 第一版写的是每只各自退避重试 4 次（每次 25s 超时），结果一批 120 只要跑满
# 一小时还拉不到东西，因为 120 只各自把 4 次重试跑了个遍。
# 现在改成**全局熔断**：连续失败到阈值就整体停 COOLDOWN 秒等配额回来，
# 期间一个请求都不发。失败的票不计入 done，下一轮自动重试。
FAIL_STREAK = 8
COOLDOWN = 150.0

BARS = 800             # 腾讯一次最多给这么多，约 3.28 年
HIST_START = "2023-01-01"   # 日线表只保留这之后的（特征最长窗口 250 天，三年够用）


def prefix(code: str) -> str:
    """腾讯的市场前缀。北交所是 bj，8/4/9 开头都归它。"""
    if code[0] == "6":
        return "sh" + code
    if code[:2] in ("83", "87", "88", "43", "92") or code[:3] == "920":
        return "bj" + code
    return "sz" + code


def is_bj(code: str) -> bool:
    return code[:2] in ("83", "87", "88", "43", "92") or code[:3] == "920"


def codes(include_bj: bool = False) -> list[str]:
    """回填用的代码表。

    ST 在打分阶段才剔除，回填阶段全要：训练需要负样本，也需要
    「后来变成 ST」的票在变 ST 之前的那段历史。

    北交所（代码表里是 920xxx 段，2024 年启用的北交所专属代码段，
    338 只稀疏分布在 920000~920992）：
      · **腾讯完全不支持**，实测 v_pv_none_match，新旧代码段都不认
      · **新浪支持**，实测 9/9 全通，历史 77~1388 行
    所以走新浪的 stage_daily_sina 默认**包含**北交所，走腾讯的
    stage_daily 默认排除。

    踩过的坑：一开始拿 832735 / 430139 / 873169 这些老代码段测，
    全部失败就以为北交所拿不到 —— 而它们根本不在 codes.csv 里。
    CLAUDE.md 教训 2「测试用编造的代码」的翻版：**测试数据必须取自
    真实的代码表**，不能凭记忆写。
    """
    p = ROOT / "cache" / "codes.csv"
    cs = pd.read_csv(p, dtype=str)["code"].tolist()
    cs = sorted(set(cs))
    if not include_bj:
        cs = [c for c in cs if not is_bj(c)]
    return cs


def fetch_one(code: str) -> pd.DataFrame | None:
    """拉一只的日线。失败返回 None，由调用方决定重试还是放弃。"""
    url = f"{KLINE}?param={prefix(code)},day,,,{BARS},qfq"
    for attempt in range(RETRY):
        try:
            r = requests.get(url, timeout=25, headers=UA)
            data = r.json().get("data")
            if not isinstance(data, dict):
                raise ValueError("data 不是 dict")
            v = list(data.values())[0]
            ks = v.get("qfqday") or v.get("day") or []
            if len(ks) < 60:
                return None                    # 次新股，特征算不出来
            df = pd.DataFrame([k[:6] for k in ks],
                              columns=["date", "open", "close", "high",
                                       "low", "volume"])
            for c in ("open", "close", "high", "low", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["code"] = code
            return df.dropna()
        except Exception:  # noqa: BLE001
            # 指数退避 + 抖动。限流是全局的，所有线程会一起撞上，
            # 不加抖动的话它们会同步重试、同步再被限。
            if attempt < RETRY - 1:
                time.sleep((2 ** attempt) * 1.5 + random.random())
    return None


def stage_daily() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    done_f = RAW / "done.json"
    done: set[str] = set()
    if done_f.exists():
        try:
            done = set(json.loads(done_f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass

    todo = [c for c in codes() if c not in done]
    log.info("日线回填：共 %d 只，已完成 %d，本轮 %d 只",
             len(done) + len(todo), len(done), len(todo))
    if not todo:
        log.info("没有要拉的，直接合并")
        return merge_daily()

    t0 = time.time()
    buf: list[pd.DataFrame] = []
    shard = len(list(RAW.glob("daily_*.parquet")))
    failed: list[str] = []

    streak = {"n": 0}

    def work(c: str):
        time.sleep(SLEEP_EACH)
        return c, fetch_one(c)

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for c, df in ex.map(work, chunk):
                if df is None:
                    failed.append(c)
                    streak["n"] += 1
                    # 失败的**不**计入 done：可能只是限流，下一轮还要重试。
                    # 计入 done 会让临时故障变成永久缺失。
                else:
                    buf.append(df)
                    streak["n"] = 0
                    done.add(c)
        if streak["n"] >= FAIL_STREAK:
            log.warning("连续失败 %d 只，判定被限流，停 %.0f 秒等配额",
                        streak["n"], COOLDOWN)
            time.sleep(COOLDOWN)
            streak["n"] = 0
        # 落分片 + 记进度。必须在同一时刻写，否则崩溃时会丢数据或重复。
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(
                RAW / f"daily_{shard:04d}.parquet", index=False)
            shard += 1
            buf = []
        done_f.write_text(json.dumps(sorted(done), ensure_ascii=False),
                          encoding="utf-8")
        el = time.time() - t0
        log.info("进度 %d/%d  失败 %d  已用 %.0f 分钟",
                 min(i + BATCH, len(todo)), len(todo), len(failed), el / 60)
        if i + BATCH < len(todo):
            time.sleep(SLEEP_BATCH)

    log.info("拉取结束，失败 %d 只：%s", len(failed), failed[:20])
    return merge_daily()


def merge_daily(pattern: str = "daily_*.parquet") -> int:
    shards = sorted(RAW.glob(pattern))
    if not shards:
        log.error("没有任何分片")
        return 1
    # 腾讯分片（daily_*）是兜底源，量的单位是手、没有换手率和成交额，
    # 不许覆盖新浪主表：写到 daily_tx.parquet
    target = OUT / ("daily.parquet" if pattern.startswith("sina") else "daily_tx.parquet")
    # 分片按文件名排序，增量分片 sina_9_<时间戳>_* 排在初始回填 sina_0000..
    # 之后，同一 (code, date) 以**后拉到的**为准。
    # 重拉分片（_ref / _full）对它包含的代码是**整段权威**的：先把这些代码
    # 在更早分片里的行全部丢掉再拼。只按 (code, date) 覆盖的话，重拉窗口
    # （最近 800 根）之前的旧行仍是除权前的价格，在窗口起点留一个假跳空。
    parts: list[pd.DataFrame] = []
    for p in shards:
        d = pd.read_parquet(p)
        if "_ref" in p.name or "_full" in p.name:
            codes_new = set(d["code"].unique())
            parts = [x[~x["code"].isin(codes_new)] for x in parts]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["date"].astype(str) >= HIST_START]
    df = (df.drop_duplicates(["code", "date"], keep="last")
            .sort_values(["code", "date"]))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    log.info("日线合并完成：%d 行，%d 只，%s .. %s -> %s",
             len(df), df.code.nunique(), df.date.min(), df.date.max(),
             target.name)
    return 0


# ---------------------------------------------------------------
#  新浪日线（主源）
# ---------------------------------------------------------------
#  为什么主源是新浪不是腾讯（2026-09-12 改）
#  ----------------------------------------
#  一、字段。新浪一个接口给全：OHLCV + amount(成交额) +
#      outstanding_share(逐日流通股本) + turnover(换手率)。腾讯只有
#      OHLCV 六个字段，换手率要另外拉 gbjg_em 再做阶梯填充。
#      新浪这份让整条 shares 管线都不必要了，而且是逐日精确值。
#  二、配额。腾讯日 K 有累积配额，本机短时间打 500+ 请求后限流数十分钟，
#      回填反复撞墙。新浪没观察到这个问题。
#
#  代价：新浪接口内部用 py_mini_racer 跑 JS 解密，**它不是线程安全的** ——
#  多线程并发调用会让 V8 直接崩溃（Check failed:
#  !IsConfigurablePoolInitialized()），不是抛异常是进程级 FATAL。
#  所以这里用**多进程**，每个进程一个独立 V8 实例。
SINA_WORKERS = 4
LIMIT = {"n": 0}


def _sina_one(code: str):
    """必须是模块级函数：ProcessPoolExecutor 要 pickle 它。"""
    import warnings
    warnings.filterwarnings("ignore")
    import akshare as ak
    sym = prefix(code)
    for attempt in range(3):
        try:
            d = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
            if d is None or len(d) < 60:
                return code, None
            d = d.copy()
            d["date"] = d["date"].astype(str).str[:10]
            # 只保留 HIST_START 之后：重拉分片要整段替换旧历史，但新浪会把
            # 上市以来全部给回来（老票到 1994 年），特征只用三年，多出来的
            # 只会拖慢筹码计算、还让对数网格锚在几十年前的价格上（2026-09-16
            # 实测 33 只新补的票把 daily.parquet 拉到 1994 年）。
            d = d[d["date"] >= HIST_START]
            d = d[["date", "open", "high", "low", "close", "volume",
                   "amount", "outstanding_share", "turnover"]]
            d["code"] = code
            if len(d) < 60:
                return code, None
            return code, d
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return code, None


def stage_daily_sina() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    done_f = RAW / "done_sina.json"
    done: set[str] = set()
    if done_f.exists():
        try:
            done = set(json.loads(done_f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    # 新浪支持北交所，这里要 include_bj=True（腾讯那条路不支持，见 codes）
    todo = [c for c in codes(include_bj=True) if c not in done]
    if LIMIT["n"]:
        todo = todo[:LIMIT["n"]]
    log.info("新浪日线：共 %d 只，已完成 %d，本轮 %d 只",
             len(done) + len(todo), len(done), len(todo))
    if not todo:
        return merge_daily(pattern="sina_*.parquet")

    t0 = time.time()
    shard = len(list(RAW.glob("sina_*.parquet")))
    stamp = now_bj().strftime("%Y%m%d%H%M%S")
    # 只有目录里一个分片都没有时才用 sina_0000.. 编号（真正的首次回填）。
    # 以前按「done 为空」判断：refresh 先删 done 再进来，新分片沿用旧编号，
    # 第二次 refresh 时同名覆盖、随后又被当旧分片删掉，daily.parquet 只剩
    # 几百只（2026-09-15 审计）。
    first = shard == 0
    failed: list[str] = []
    buf = []
    STEP = 200
    for i in range(0, len(todo), STEP):
        chunk = todo[i:i + STEP]
        with ProcessPoolExecutor(max_workers=SINA_WORKERS) as ex:
            for c, d in ex.map(_sina_one, chunk, chunksize=4):
                if d is None:
                    failed.append(c)
                else:
                    buf.append(d)
                    done.add(c)
        if buf:
            # 重拉的分片带时间戳：merge 按文件名排序、后者覆盖前者，
            # 它们必须排在原始分片 sina_0000.. 之后
            name = (f"sina_{shard:04d}.parquet" if first
                    else f"sina_9_{stamp}_full{shard:04d}.parquet")
            pd.concat(buf, ignore_index=True).to_parquet(RAW / name, index=False)
            shard += 1
            buf = []
        done_f.write_text(json.dumps(sorted(done), ensure_ascii=False),
                          encoding="utf-8")
        el = time.time() - t0
        rate = (i + len(chunk)) / max(el, 1e-9)
        log.info("进度 %d/%d  失败 %d  已用 %.0f 分钟  预计还需 %.0f 分钟",
                 min(i + STEP, len(todo)), len(todo), len(failed), el / 60,
                 (len(todo) - i - len(chunk)) / max(rate, 1e-9) / 60)
    log.info("新浪拉取结束，失败 %d 只", len(failed))
    return merge_daily(pattern="sina_*.parquet")


# ---------------------------------------------------------------
#  每日增量（腾讯快照）
# ---------------------------------------------------------------
def now_bj():
    import datetime as dt
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def last_closed_trade_day(now=None) -> str:
    """最近一个**已收盘**的交易日。15:05 之后算当天，之前算上一个交易日。

    晚间系统的「目标日」就是它：北京 09-15 早上 07:00 跑，目标日仍是 09-14，
    数据和 09-14 下午 17:00 跑一模一样。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import datasource as ds
    now = now or now_bj()
    tds = sorted(ds.trade_dates())
    today = now.strftime("%Y-%m-%d")
    closed = (now.hour, now.minute) >= (15, 5)
    cands = [d for d in tds if d < today or (d == today and closed)]
    return cands[-1]


def _shard_name(kind: str) -> Path:
    ts = now_bj().strftime("%Y%m%d%H%M%S")
    return RAW / f"sina_9_{ts}_{kind}.parquet"


def refetch_codes(cs: list[str]) -> int:
    """整段重拉指定代码（新浪，多进程）。除权那天用。"""
    if not cs:
        return 0
    buf = []
    with ProcessPoolExecutor(max_workers=SINA_WORKERS) as ex:
        for c, d in ex.map(_sina_one, cs, chunksize=2):
            if d is not None:
                buf.append(d)
    if buf:
        pd.concat(buf, ignore_index=True).to_parquet(_shard_name("ref"),
                                                     index=False)
    log.info("重拉 %d 只，成功 %d 只", len(cs), len(buf))
    return len(buf)


def stage_update() -> int:
    """把最近一个已收盘交易日追加进 daily.parquet。秒级，每天收盘后跑。

    返回 0 表示 daily.parquet 已经覆盖到目标日（不管是刚追加的还是本来
    就有）。中间缺了不止一个交易日（机器几天没开）就退化成全量刷新。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import datasource as ds
    dp = OUT / "daily.parquet"
    if not dp.exists():
        log.error("缺 %s，先跑 --stage sina 做首次回填", dp)
        return 1
    target = last_closed_trade_day()
    daily = pd.read_parquet(dp)
    hist_max = str(daily["date"].max())
    if hist_max >= target:
        log.info("日线已覆盖到 %s（目标日 %s），不用追加", hist_max, target)
        return 0
    tds = sorted(ds.trade_dates())
    missing = [d for d in tds if hist_max < d <= target]
    if len(missing) > 1:
        log.warning("日线停在 %s，到 %s 缺 %d 个交易日，改走全量刷新（约 70 分钟）",
                    hist_max, target, len(missing))
        return stage_refresh()

    last = (daily.sort_values("date").groupby("code").tail(1)
            .set_index("code"))
    cs = codes(include_bj=True)
    q = ds.fetch_quotes([ds.to_symbol(c) for c in cs])
    log.info("快照 %d 只，目标日 %s", len(q), target)
    tkey = target.replace("-", "")
    rows, refetch, newcodes, stale = [], [], [], 0
    for c in cs:
        v = q.get(ds.to_symbol(c))
        if v is None:
            continue
        if not str(v.ts).startswith(tkey):
            stale += 1                       # 停牌或还没更新，今天没有这根
            continue
        if c not in last.index:
            newcodes.append(c)               # 没有历史：下面整段拉
            continue
        h = last.loc[c]
        if abs(float(v.prev_close) - float(h["close"])) > 0.006:
            refetch.append(c)                # 除权/除息，或历史已经不对，整段重拉
            continue
        try:
            high, low = float(v.raw[33]), float(v.raw[34])
            turn_q = float(v.raw[38] or 0) / 100.0
        except Exception:  # noqa: BLE001
            continue
        if not (v.price > 0 and high > 0 and low > 0 and v.open_ > 0):
            stale += 1
            continue
        os_ = (float(h["outstanding_share"])
               if pd.notna(h["outstanding_share"]) else 0.0)
        # 科创板快照的成交量是股不是手。有换手率就按「换手率 × 流通股本」核对，
        # 没有就按板块给默认单位（以前没有兜底，换手率偶发为空时科创板放大 100 倍）
        vol = v.volume_hand if c.startswith("688") else v.volume_hand * 100.0
        if turn_q > 0 and os_ > 0:
            exp = turn_q * os_
            vol = min((v.volume_hand * 100.0, v.volume_hand),
                      key=lambda x: abs(math.log((x + 1.0) / (exp + 1.0))))
        turn = vol / os_ if os_ > 0 else turn_q
        rows.append({"date": target, "open": v.open_, "high": high,
                     "low": low, "close": v.price, "volume": vol,
                     "amount": v.amount_wan * 1e4,
                     "outstanding_share": os_ if os_ > 0 else float("nan"),
                     "turnover": turn, "code": c})
    log.info("追加 %d 只；无当日数据 %d；无历史 %d；需整段重拉 %d",
             len(rows), stale, len(newcodes), len(refetch))
    if len(rows) < 1000:
        log.error("只拼出 %d 只，快照不像是收盘后的完整数据，本次不追加",
                  len(rows))
        return 1
    pd.DataFrame(rows).to_parquet(_shard_name("upd"), index=False)
    # 没有历史的票（新上市、首次回填漏掉的）整段拉，不然永远进不来
    if refetch or newcodes:
        refetch_codes(refetch + newcodes[:200])
    rc = merge_daily(pattern="sina_*.parquet")
    # 股东人数按季披露，日常流程以前从不刷新：公告日对齐修完了，生产侧
    # 却一直拿着旧表。超过 7 天就刷一次（13 次请求，秒级）。
    hp = OUT / "holders.parquet"
    if rc == 0 and (not hp.exists()
                    or time.time() - hp.stat().st_mtime > 7 * 86400):
        try:
            stage_holders()
        except Exception as e:  # noqa: BLE001
            log.warning("股东人数刷新失败（%s），继续用旧表", e)
    return rc


def consolidate() -> int:
    """把 daily.parquet 落成唯一一个分片 sina_0000.parquet，其余分片删掉。

    增量分片和重拉分片会一直攒；每次全量刷新后收拢一次，之后的 merge
    从这一个分片起算。拉失败的票的旧历史也在 daily.parquet 里，不会丢。
    """
    dp = OUT / "daily.parquet"
    if not dp.exists():
        return 1
    df = pd.read_parquet(dp)
    tmp = RAW / "sina_consolidate.tmp"
    df.to_parquet(tmp, index=False)
    for p in RAW.glob("sina_*.parquet"):
        try:
            p.unlink()
        except OSError:
            pass
    tmp.replace(RAW / "sina_0000.parquet")
    log.info("分片已收拢为一个：%d 行", len(df))
    return 0


def stage_refresh() -> int:
    """全量重拉新浪日线 + 股东人数，然后把分片收拢。

    重拉的分片带时间戳、排在旧分片之后，merge 时新数据覆盖旧数据；
    拉失败的票保留旧历史（不像第一版那样连旧分片一起删）。
    """
    done_f = RAW / "done_sina.json"
    if done_f.exists():
        done_f.unlink()
    rc = stage_daily_sina()
    if rc == 0:
        rc = merge_daily(pattern="sina_*.parquet")
    if rc == 0:
        rc = consolidate()
    rc |= stage_holders()
    return rc


def stage_shares() -> int:
    """历史流通股本。解锁精确换手率，见设计文档 1.2。"""
    import akshare as ak
    OUT.mkdir(parents=True, exist_ok=True)
    cs = codes()
    rows, failed = [], []

    def one(code: str):
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_gbjg_em(symbol=code)
                if d is None or not len(d):
                    return None
                d = d[["变更日期", "总股本", "已上市流通A股"]].copy()
                d.columns = ["date", "total_shares", "float_shares"]
                d["code"] = code
                return d
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        return None

    t0 = time.time()
    for i in range(0, len(cs), 100):
        chunk = cs[i:i + 100]
        with ThreadPoolExecutor(max_workers=4) as ex:
            for c, d in zip(chunk, ex.map(one, chunk)):
                (rows.append(d) if d is not None else failed.append(c))
        log.info("股本 %d/%d 失败 %d 用时 %.0f 分钟",
                 min(i + 100, len(cs)), len(cs), len(failed),
                 (time.time() - t0) / 60)
        time.sleep(2)

    if not rows:
        log.error("股本一条都没拉到")
        return 1
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).sort_values(["code", "date"])
    df.to_parquet(OUT / "shares.parquet", index=False)
    log.info("股本完成：%d 行，%d 只（失败 %d）",
             len(df), df.code.nunique(), len(failed))
    return 0


def stage_holders() -> int:
    """股东人数。按报告期批量拉，比逐只快得多。

    披露滞后的处理不在这里，在 features.py：这里只负责把原始数据落盘，
    包含报告期本身，让下游按公告日对齐（设计文档 4.6）。
    """
    import akshare as ak
    OUT.mkdir(parents=True, exist_ok=True)
    # 2023-05 之后的所有季末 + 半月末（东财按这些日期组织数据）
    periods = []
    today = now_bj().strftime("%Y%m%d")
    for y in range(2023, int(today[:4]) + 1):
        for md in ("0331", "0630", "0930", "1231"):
            d = f"{y}{md}"
            if "20230501" <= d <= today:
                periods.append(d)
    rows = []
    for p in periods:
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_gdhs(symbol=p)
                if d is not None and len(d):
                    d = d.copy()
                    d["报告期"] = p
                    rows.append(d)
                    log.info("股东人数 %s: %d 行", p, len(d))
                break
            except Exception as e:  # noqa: BLE001
                log.warning("股东人数 %s 第 %d 次失败: %s", p, attempt + 1,
                            str(e)[:60])
                time.sleep(2 * (attempt + 1))
        time.sleep(1)
    if not rows:
        log.error("股东人数一条都没拉到")
        return 1
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(OUT / "holders.parquet", index=False)
    log.info("股东人数完成：%d 行，%d 期", len(df), len(periods))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只拉前 N 只，用于验证流程")
    ap.add_argument("--stage", default="all",
                    choices=["daily", "sina", "shares", "holders", "all",
                             "merge", "update", "refresh"])
    a = ap.parse_args()
    if a.stage == "merge":
        return merge_daily(pattern="sina_*.parquet")
    if a.stage == "update":
        return stage_update()
    if a.stage == "refresh":
        return stage_refresh()
    rc = 0
    LIMIT["n"] = a.limit
    if a.stage == "sina":
        return stage_daily_sina()
    if a.stage in ("daily", "all"):
        rc |= stage_daily()
    if a.stage in ("shares", "all"):
        rc |= stage_shares()
    if a.stage in ("holders", "all"):
        rc |= stage_holders()
    return rc


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
