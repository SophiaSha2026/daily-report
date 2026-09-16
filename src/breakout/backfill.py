"""
爆发线的数据回填：三年日线 + 历史股本 + 股东人数。

    python src/breakout/backfill.py --stage update    每日增量：追加最近一个交易日（秒级）
    python src/breakout/backfill.py --stage refresh   全量重拉新浪日线 + 股东人数（约 70 分钟）
    python src/breakout/backfill.py --stage sina      新浪三年日线，断点续传（首次回填）
    python src/breakout/backfill.py --stage daily     腾讯日线（兜底源，最慢，25~40 分钟）
    python src/breakout/backfill.py --stage shares    历史流通股本（build 不读，已不用）
    python src/breakout/backfill.py --stage holders   股东人数

`--stage` 必须显式给，没有 all：默认 all 跑的是「腾讯日线 + 股本 + 股东人数」，
而首次回填该走 sina，股本那份 build.attach_turnover 压根不读
（新浪日线自带逐日 outstanding_share）。裸跑会安静地耗掉 25~40 分钟。

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
STATE = ROOT / "state" / "breakout"

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
# 腾讯换手率反推出来的流通股本 vs 昨天那根新浪 K 线的股本，超过这个就整段重拉。
# 解禁 / 增发 / 回购注销都不动价格，昨收核对一个都接不住（daily.parquet 实测
# 股本跳升 ≥1.25 倍的 3649 次里 3624 次价格不变）。见 _snap_row。
OS_DRIFT_TOL = 0.03


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


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """去掉停牌占位行和零价行。

    新浪会给停牌日塞一根占位 K 线：价格照抄前一日、成交量和成交额为 0；
    偶尔还会给出 open=high=low=0 的行（2026-09-16 实测 daily.parquet 里
    16 行，其中 3 行价格为 0，全部 volume=0，且每行之后都跟着 1~14 个
    自然日的日期空洞，即那天本来就不该有 K 线）。

    留着它们不会报错，只会静默污染特征：零价行当天的 TR 被算成
    |0 - 昨收|（实测放大 32 倍），atr_ratio 之后 14 天被抬高、再之后 46 天
    被压低；零量行让 vol_ratio5/vol_ratio20/vol_compress 在随后 5~20 天全部失真。
    chips.py 早就自己挡了零价（它有 low>0 & high>0 的守卫），这里是把同一道
    闸补到数据层，所有下游共用。

    按「列在不在」逐列判，是为了不误伤只有 code/date/close 的分片。
    """
    if not len(df):
        return df
    m = pd.Series(True, index=df.index)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            m &= df[c] > 0
    if {"high", "low"} <= set(df.columns):
        m &= df["high"] >= df["low"]
    if "volume" in df.columns:
        m &= df["volume"] > 0
    return df[m]


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
            return clean_bars(df.dropna())
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
    # 已经落盘的老分片里也有停牌占位行（16 行，2026-09-16 实测），在这里清掉
    # 就不用为它们重拉一遍
    df = clean_bars(df)
    df = df[df["date"].astype(str) >= HIST_START]
    df = (df.drop_duplicates(["code", "date"], keep="last")
            .sort_values(["code", "date"]))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    log.info("日线合并完成：%d 行，%d 只，%s .. %s -> %s",
             len(df), df.code.nunique(), df.date.min(), df.date.max(),
             target.name)
    # 各票起点是否对齐。首次回填那版按根数截（tail(800)，起于 2023-05-30），
    # 现在按日历截（HIST_START），除权重拉的票会一路补到 2023-01-03，于是
    # 2023-01~05 这段只有「分过红的那几十只」的横截面，它们的当日全市场排名
    # 是在 37 只里排的（2026-09-16 实测：2023-01-03 只有 37 只，05-30 跳到 5039）。
    # 改 HIST_START 或截尾规则都必须配一次 --stage refresh 把所有票对齐。
    if len(df):
        n_first = int(df[df["date"] == df["date"].min()]["code"].nunique())
        n_last = int(df[df["date"] == df["date"].max()]["code"].nunique())
        if n_first < 0.5 * n_last:
            log.warning("各票起点不一致：最早日 %s 只有 %d 只，最末日 %s 有 %d 只。"
                        "跑 --stage refresh 对齐，否则早期横截面排名失真",
                        df["date"].min(), n_first, df["date"].max(), n_last)
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
            d = clean_bars(d)          # 停牌占位行不能进表，见 clean_bars
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

    直接借 local_run 那一份，不再自己算（S20）。本来这里是
    `sorted(ds.trade_dates())[-1]` 加一句筛选，日历接口挂**且**
    state/trade_dates.json 不存在（新克隆 / state 被清）时 ds.trade_dates()
    会 raise，stage_update 和 main 都不接，整条晚间线以 traceback 退出 —— 而
    local_run.trade_dates 的文档明写「宁可节假日多跑一次空流程，也不能因为
    日历接口挂了整条线不跑」。同一件事仓库里有四份实现
    （local_run / evening_check / gui.status / 这里），只有这一份是 raise。
    日历或缓存任一可用时两份逐日一致（2024-01-01~2026-12-31 每天四个时点
    共 4384 个用例，0 处不一致）。

    编排层（local_run.flow_breakout）应当把算好的目标日用 --target 传进来，
    这里只是手动裸跑时的兜底。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import local_run
    return local_run.last_closed_trade_day(now or now_bj())


def _shard_name(kind: str) -> Path:
    ts = now_bj().strftime("%Y%m%d%H%M%S")
    return RAW / f"sina_9_{ts}_{kind}.parquet"


def refetch_codes(cs: list[str],
                  target: str | None = None) -> tuple[list[str], list[str]]:
    """整段重拉指定代码（新浪，多进程）。除权那天用。

    返回 (拉到目标日的, 没拉到目标日的)。以前只返回成功只数而且没人接：
    「成功」只意味着拿到 ≥60 行，不意味着含目标日。新浪收盘后当天 K 线
    有时还没生成，这只票于是既不在 _upd 分片（除权分支 continue 掉了）
    也不在 _ref 分片的目标日，而 _ref 对它的代码是**整段权威**的，
    当天这一行就彻底消失，清单、面板、run_meta 全都看不出少了谁。
    """
    if not cs:
        return [], []
    buf, got = [], {}
    with ProcessPoolExecutor(max_workers=SINA_WORKERS) as ex:
        for c, d in ex.map(_sina_one, cs, chunksize=2):
            if d is not None:
                buf.append(d)
                got[c] = str(d["date"].max())
    if buf:
        pd.concat(buf, ignore_index=True).to_parquet(_shard_name("ref"),
                                                     index=False)
    short = [c for c in cs if target and got.get(c, "") < target]
    log.info("重拉 %d 只，成功 %d 只，没拉到目标日 %s 的 %d 只：%s",
             len(cs), len(buf), target, len(short), ",".join(short[:20]))
    return [c for c in cs if c not in short], short


def _snap_row(c: str, v, os_: float, target: str) -> tuple[dict | None, bool]:
    """腾讯快照 -> 一根日线。返回 (行, 流通股本是否漂了)。

    os_ 是 daily.parquet 里该票最后一根**新浪** K 线的流通股本。
    """
    try:
        high, low = float(v.raw[33]), float(v.raw[34])
        turn_q = float(v.raw[38] or 0) / 100.0
    except Exception:  # noqa: BLE001
        return None, False
    if not (v.price > 0 and high > 0 and low > 0 and v.open_ > 0):
        return None, False
    # Quote.volume_hand 一律是「手」：腾讯对科创板原始给「股」，2026-09-16 起
    # 由 datasource.tx_vol_hand 在解析时就折算好了（三路数据源统一口径）。
    # 这里只做一次 手 -> 股。**别再按板块分叉**：datasource 折过之后再分叉，
    # 科创板会被当成已经是股、少乘 100，当天的成交量和换手率整段差两个数量级。
    # 有换手率时仍按「换手率 × 流通股本」核对，防的是股本漂了而不是单位。
    board_vol = v.volume_hand * 100.0
    vol, drift = board_vol, False
    if turn_q > 0 and os_ > 0:
        exp = turn_q * os_
        cand = min((v.volume_hand * 100.0, v.volume_hand),
                   key=lambda x: abs(math.log((x + 1.0) / (exp + 1.0))))
        # 股本漂 10 倍以上时「错单位」的候选反而更贴近 exp，单位会被判翻
        # （实测 001289 那种 14.59 倍的 IPO 解禁，成交量会差 100 倍）。
        # 两个候选都对不上 exp 就不信 exp，退回板块默认单位。
        vol = (cand if abs(math.log((cand + 1.0) / (exp + 1.0))) < math.log(3.0)
               else board_vol)
        # raw[38] 是交易所按**当日**流通股本算的换手率，os_ 是上一根新浪 K 线的。
        # 解禁/增发/回购注销不伴随除权，昨收判据接不住：daily.parquet 实测
        # 股本跳升 ≥1.25 倍的 3649 次里 3624 次价格不变，按「股本变化 >5%」
        # 算平均每天有 4.8 只票的股本变更是昨收判据看不见的。用换手率反推股本
        # 纠正，并让新浪整段重拉拿回精确值，否则 turnover 一直按旧股本算
        # （2026-09-16 实测 13 只偏离 >5%，最大 2.28 倍），
        # 而 turnover 又喂给 chips.py 的每日换手衰减和 turn_pct/turn_std20。
        os_q = vol / turn_q
        # raw[38] 只有两位小数，量化误差 = 0.00005/turn_q。死写一个绝对
        # 百分比的话，今天全市场换手 <0.1% 的 10 只票会天天被误判成股本变更，
        # 所以容差 = 固定项 + 舍入项（4 倍余量）。固定项 3% 的依据：腾讯和
        # 新浪两套换手率口径的差 p1~p99 在 ±1.4% 以内，3% 有两倍余量，
        # 而真正的股本变动一天约 10~16 只（>5% 的约 8 只），重拉几秒。
        tol = OS_DRIFT_TOL + 4.0 * 0.00005 / turn_q
        if abs(os_q / os_ - 1.0) > tol:
            os_, drift = os_q, True
    # 交易所的换手率按定义就是对的，两位小数在 1.5% 的中位换手上只有 0.7%
    # 相对误差，远好于拿陈旧股本除出来的成倍偏差
    turn = turn_q if turn_q > 0 else (vol / os_ if os_ > 0 else float("nan"))
    return {"date": target, "open": v.open_, "high": high, "low": low,
            "close": v.price, "volume": vol, "amount": v.amount_wan * 1e4,
            "outstanding_share": os_ if os_ > 0 else float("nan"),
            "turnover": turn, "code": c}, drift


def _scan_snapshot(cs: list[str], q: dict, last: pd.DataFrame, target: str,
                   prev_td: str | None) -> dict:
    """按快照逐只拼当日行。每一条 continue 都要计数（见返回值）。

    四个计数加起来必须等于请求只数：2026-09-15/16 两轮实测
    5487+14+33+14 = 5548、5472+14+32+30 = 5548，残差恒为 0，
    所以残差是个零误报的检测器 —— 少一批 60 只立刻看得见。
    """
    import datasource as ds
    out = {"rows": [], "refetch": [], "newcodes": [], "pend": {},
           "stale": 0, "noq": []}
    tkey = target.replace("-", "")
    for c in cs:
        v = q.get(ds.to_symbol(c))
        if v is None:
            # 快照没回。datasource._one_batch 一批 60 只重试两次后整批返回 {}，
            # 只留一行 warning；以前这里连计数都没有，日志里看不出少了谁
            out["noq"].append(c)
            continue
        if not str(v.ts).startswith(tkey):
            out["stale"] += 1                # 停牌或还没更新，今天没有这根
            continue
        if c not in last.index:
            out["newcodes"].append(c)        # 没有历史：下面整段拉
            continue
        h = last.loc[c]
        os_ = (float(h["outstanding_share"])
               if pd.notna(h["outstanding_share"]) else 0.0)
        row, drift = _snap_row(c, v, os_, target)
        if abs(float(v.prev_close) - float(h["close"])) > 0.006:
            out["refetch"].append(c)         # 除权/除息，或历史已经不对，整段重拉
            if row is not None:
                out["pend"][c] = row         # 新浪还没出当日 K 线时的兜底行
            continue
        if prev_td and str(h["date"]) < prev_td and v.open_ > 0:
            # 上一根不是前一个交易日：要么停过牌（新浪本来就没那几天），
            # 要么某天漏采。快照分不出来，一律整段重拉。只靠昨收比对补不上：
            # 缺的那天收平（全表 2.63% 的行）差值就是 0，缺口会永久留下。
            # 这里**不 continue**：重拉失败时当天这根还在，不会又挖一个新洞。
            out["refetch"].append(c)
        if row is None:
            out["stale"] += 1                # 价格异常或字段解析失败
            continue
        if drift:
            out["refetch"].append(c)         # 股本漂了，整段重拉拿回精确值
        out["rows"].append(row)
    return out


def holders_stale(hp: Path, max_age_h: float = 20) -> bool:
    """股东人数表该不该刷。

    原来是 7 天门槛，于是 build 时刻的表龄均匀落在 0~7 天，而模型是在
    「公告当天即知」的行上训练和验收的（features.holder_features）。
    公告是**成堆**来的：68208 条里，2025-04-30 前 7 天有 7123 条（占全市场
    131%，年报加一季报每只两条）、2026-04-30 有 7081 条；696 个工作日里
    19.5% 的日子近 7 天公告数 ≥5% 全市场、10.5% ≥20%。那些天生产用的是
    0~7 天前的旧户数，训练行用的是当天的，教训 30 那类口径差。
    20 小时：同一晚每 30 分钟重试一次不会重复拉（13 期 × 11 页 ≈ 143 个请求、
    约 1 分钟），但每个交易日至少刷一次。
    """
    return (not hp.exists()) or time.time() - hp.stat().st_mtime > max_age_h * 3600


def _write_status(payload: dict) -> None:
    """把这一轮的核对结果落成一个**能被界面查询的对象**（历史教训 16）。

    失败只写日志等于没写：2026-09-07~09-11 同一件事连错五个交易日没人发现。
    """
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "update_status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("update_status.json 写不出来：%s", e)


def stage_update(target: str = "") -> int:
    """把最近一个已收盘交易日追加进 daily.parquet。秒级，每天收盘后跑。

    返回 0 表示 daily.parquet 已经覆盖到目标日（不管是刚追加的还是本来
    就有）。中间缺了不止一个交易日（机器几天没开）就退化成全量刷新。

    target 由编排层（local_run.flow_breakout）传进来：目标日只该算一次，
    每个子阶段各判一次早晚会分叉（S20，学习线的 `--date` 是同一做法）。
    没传就自己算，手动裸跑时用。
    """
    sys.path.insert(0, str(ROOT / "src"))
    import datasource as ds
    dp = OUT / "daily.parquet"
    if not dp.exists():
        log.error("缺 %s，先跑 --stage sina 做首次回填", dp)
        return 1
    target = target or last_closed_trade_day()
    daily = pd.read_parquet(dp)
    # 「覆盖到哪天」要按**覆盖完整的那天**算，不是全表最大日期：某几批快照
    # 失败时那 60×N 只票当天没有行，而全表 max 被其余几千只托着照样等于目标日，
    # 下一轮直接「不用追加」，缺的票再也补不回来（stage_update 只追加目标日，
    # 中间的洞没人扫）。全量 refresh 跨过新浪更新时点时更狠：只要 20 只有当日行
    # 就会判定已覆盖，然后拿 20 只当全市场出清单。
    cnt = daily.groupby("date")["code"].size().sort_index()
    ref = int(cnt.iloc[-6:-1].median()) if len(cnt) > 5 else int(cnt.max())
    full = cnt[cnt >= 0.97 * ref]
    hist_max = str(full.index[-1]) if len(full) else str(cnt.index[-1])
    if hist_max >= target:
        log.info("日线已覆盖到 %s（%d 只，目标日 %s），不用追加",
                 hist_max, int(cnt.loc[hist_max]), target)
        # 这条早退路也要写账：local_run.flow_breakout 读 update_status.json 判
        # 「今天缺了谁」，最常见的一条路不写的话，那边只能记一行「没写账」。
        try:
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / "update_status.json").write_text(json.dumps(
                {"date": target, "covered": hist_max, "rows": int(cnt.loc[hist_max]),
                 "appended": 0, "stale": 0, "newcodes": 0, "refetch_only": 0,
                 "missing": 0, "short": [], "filled": [],
                 "note": "已覆盖，本轮没有追加"}, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("update_status.json 写不出来：%s", e)
        return 0
    tds = sorted(ds.trade_dates())
    missing_days = [d for d in tds if hist_max < d <= target]
    if len(missing_days) > 1:
        log.warning("日线停在 %s，到 %s 缺 %d 个交易日，改走全量刷新（约 70 分钟）",
                    hist_max, target, len(missing_days))
        return stage_refresh()
    prev_td = tds[tds.index(target) - 1] if target in tds[1:] else None

    # last 只能取 ≤ hist_max 的行：目标日已经有行的那几只，最后一根就是目标日
    # 收盘价，和快照的 prev_close 必然对不上，会被判成除权而整表重拉上千只。
    last = (daily[daily["date"].astype(str) <= hist_max]
            .sort_values("date").groupby("code").tail(1).set_index("code"))
    cs = codes(include_bj=True)
    q = ds.fetch_quotes([ds.to_symbol(c) for c in cs])
    log.info("快照 %d 只，目标日 %s", len(q), target)
    r = _scan_snapshot(cs, q, last, target, prev_td)
    if r["noq"]:
        # 只重打没回的那一两批，秒级
        log.warning("快照没回 %d 只，定向重打一次", len(r["noq"]))
        q2 = ds.fetch_quotes([ds.to_symbol(c) for c in r["noq"]])
        r2 = _scan_snapshot(r["noq"], q2, last, target, prev_td)
        for k in ("rows", "refetch", "newcodes"):
            r[k] += r2[k]
        r["pend"].update(r2["pend"])
        r["stale"] += r2["stale"]
        r["noq"] = r2["noq"]        # 只剩第二次还没回的，不是两次相加

    rows, refetch, newcodes = r["rows"], r["refetch"], r["newcodes"]
    have = {x["code"] for x in rows}
    ref_only = [c for c in refetch if c not in have]
    missing = len(r["noq"])
    # 覆盖率要和**前一个完整交易日**比，不能只跟请求只数比。腾讯一批 60 只，
    # 限流是累积配额（教训 20），一旦中途限流后面会连片失败，而丢掉的常常是
    # 整个板块：cache/codes.csv 排序后前 1500 只里创业板只有 6 只、科创板和
    # 北交所一只都没有。那天的横截面百分位、板块中性化、rs20/rs60、市场宽度
    # 就全在残缺的池子里算（实测用前 1500 只算 mkt_breadth 0.708 vs 全市场
    # 0.752），清单 A 也只能从这部分票里选，而且**当天补不回来**
    # （hist_max 已到目标日，下一轮直接「不用追加」）。
    # daily.parquet 898 个交易日里相邻日行数比最小 0.9706、1% 分位 0.9991，
    # 0.95 留足余量，停牌（stale，最近两次各 14 只）不会误杀。
    prev_n = int((daily["date"].astype(str) == hist_max).sum())
    covered = len(rows) + len(ref_only)      # 追加的 + 靠重拉拿当日那根的
    log.info("追加 %d 只；无当日数据 %d；无历史 %d；需整段重拉 %d；快照没回 %d",
             len(rows), r["stale"], len(newcodes), len(refetch), missing)
    status = {"date": target, "requested": len(cs), "appended": len(rows),
              "stale": r["stale"], "newcodes": len(newcodes),
              "refetch": len(refetch), "refetch_only": len(ref_only),
              "missing": missing, "missing_codes": r["noq"][:20],
              "prev_n": prev_n, "covered": covered,
              "closed": (len(rows) + r["stale"] + len(newcodes)
                         + len(ref_only) + missing == len(cs))}
    if missing > 20 or len(rows) < len(cs) * 0.80 or covered < 0.95 * prev_n:
        # 一批 = 60 只 = 1.08%，所以百分比阈值拦不住单批失败，残差才拦得住。
        # 什么都不写是安全的：local_run 会「今天不出清单」，30 分钟后干净重跑
        log.error("快照缺 %d 只（例 %s），当日可覆盖 %d 只、前一日 %d 只，"
                  "本次不追加，等下一轮重试",
                  missing, r["noq"][:5], covered, prev_n)
        _write_status({**status, "ok": False, "short": [], "filled": 0})
        return 1
    pd.DataFrame(rows).to_parquet(_shard_name("upd"), index=False)
    # 没有历史的票（新上市、首次回填漏掉的）整段拉，不然永远进不来
    short: list[str] = []
    if refetch or newcodes:
        _ok, short = refetch_codes(refetch + newcodes[:200], target=target)
    # 新浪还没出当日 K 线的除权票：_ref 分片对它们是整段权威，会把 _upd 里
    # 那一根也清掉，于是这只票当天从候选池静默消失。用快照兜一根。
    # 股本已经由 _snap_row 按换手率反推过，送转当天也不会偏。
    fill = [r["pend"][c] for c in short if c in r["pend"]]
    if fill:
        # 分片名带时间戳，必须排在刚写的 _ref 之后，否则被「整段权威」清掉
        pd.DataFrame(fill).to_parquet(_shard_name("upd2"), index=False)
        log.warning("%d 只除权票新浪还没出当日 K 线，先用腾讯快照补当天一根",
                    len(fill))
    status.update({"short": short, "filled": len(fill), "ok": True})
    _write_status(status)
    rc = merge_daily(pattern="sina_*.parquet")
    hp = OUT / "holders.parquet"
    if rc == 0 and holders_stale(hp):
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
    # 股东人数失败**不并进 rc**：日线此时已经补到目标日了，把它拖成非零
    # 会让 local_run 判「补数据失败，今天不出清单」，白扔一个重试周期。
    # 和 stage_update 里那处「继续用旧表」同口径。
    try:
        if stage_holders() != 0:
            log.warning("股东人数没刷新成功，继续用旧表")
    except Exception as e:  # noqa: BLE001
        log.warning("股东人数刷新失败（%s），继续用旧表", e)
    return rc


def stage_shares() -> int:
    """历史流通股本。解锁精确换手率，见设计文档 1.2。

    **已不用**：2026-09-12 换新浪源之后日线自带逐日 outstanding_share，
    build.attach_turnover 只看日线表里的列，shares.parquet 一个消费者都没有
    （grep 核实）。保留是因为腾讯兜底那条路没有股本，真掉回去时还得靠它。
    """
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
    rows, got, failed = [], [], []
    for p in periods:
        ok = False
        for attempt in range(3):
            try:
                d = ak.stock_zh_a_gdhs(symbol=p)
                if d is not None and len(d):
                    d = d.copy()
                    d["报告期"] = p
                    rows.append(d)
                    got.append(p)
                    log.info("股东人数 %s: %d 行", p, len(d))
                ok = True
                break
            except Exception as e:  # noqa: BLE001
                log.warning("股东人数 %s 第 %d 次失败: %s", p, attempt + 1,
                            str(e)[:60])
                time.sleep(2 * (attempt + 1))
        if not ok:
            failed.append(p)
        time.sleep(1)
    if not rows:
        log.error("股东人数一条都没拉到")
        return 1
    df = pd.concat(rows, ignore_index=True)
    hp = OUT / "holders.parquet"
    # 掉一期就会用残缺表覆盖完整表，而且**不会自愈**（下次全量刷新前一直用它）。
    # 后果：features.py 的 merge_asof 把取数期整体回退一期，2026-09-16 在真实
    # 68208 行上复现「丢最新一期」：5429 只里 5385 只（99.2%）回退，
    # gdhs_chg1 变了 5365 只（|差| 中位 9.4 个百分点），
    # gdhs_stale_days 中位 21 天涨到 141 天，而这组特征重要性排第二。
    # 判据必须是「期集合是旧表的超集」，不能只看行数：丢中间一期只掉约 7.5% 行，
    # 任何「行数掉 10%」式的阈值都拦不住。
    # 拉到的期用新数据，没拉到的期**沿用旧表那几期的行**：整表保留旧表等于把
    # 这次拿到的新一期也扔了，而每天刷一次（holders_stale）之后「某一期偶发
    # 失败」会是常态，不能一失败就一整天用不上新公告。
    lost: list[str] = []
    if hp.exists():
        try:
            old = pd.read_parquet(hp)
        except Exception as e:  # noqa: BLE001
            log.warning("旧的股东人数表读不出来（%s），只写这次拿到的", e)
            old = None
        if old is not None and "报告期" in old.columns:
            keep = old[~old["报告期"].astype(str).isin(set(got))]
            lost = sorted(set(keep["报告期"].astype(str)))
            if lost:
                log.warning("股东人数 %s 这次没拉到，沿用旧表里的 %d 行",
                            lost, len(keep))
                df = pd.concat([keep, df], ignore_index=True)
    df.to_parquet(hp, index=False)
    log.info("股东人数完成：%d 行，%d/%d 期（失败 %s）",
             len(df), len(got), len(periods), failed or "无")
    # 非零让 stage_refresh 的日志看得见；stage_update / stage_refresh 那两处
    # 本来就 try 包住，不会因为这个不出清单
    return 0 if not lost and not failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只拉前 N 只，用于验证流程")
    # 必须显式给 stage，没有 all：以前默认 all = 腾讯日线 + 股本 + 股东人数，
    # 而首次回填走的是新浪（sina），股本那份 build 根本不读。裸跑 backfill.py
    # 会安静地跑 25~40 分钟腾讯兜底源，跑完 build 还是缺列（S27）
    ap.add_argument("--stage", required=True,
                    choices=["daily", "sina", "shares", "holders",
                             "merge", "update", "refresh"])
    ap.add_argument("--target", default="",
                    help="目标日 YYYY-MM-DD，由编排层传入（只 --stage update 用）")
    a = ap.parse_args()
    if a.stage == "merge":
        return merge_daily(pattern="sina_*.parquet")
    if a.stage == "update":
        return stage_update(target=a.target)
    if a.stage == "refresh":
        return stage_refresh()
    rc = 0
    LIMIT["n"] = a.limit
    if a.stage == "sina":
        return stage_daily_sina()
    if a.stage == "daily":
        rc |= stage_daily()
    if a.stage == "shares":
        rc |= stage_shares()
    if a.stage == "holders":
        rc |= stage_holders()
    return rc


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    sys.exit(main())
