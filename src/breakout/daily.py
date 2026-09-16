"""
起涨预测的每日流程。晚间系统的主脚本。

    python src/breakout/daily.py --stage scan    补数据 -> 建特征 -> 打分 -> 出清单
    python src/breakout/daily.py --stage send    面板 + 邮件
    python src/breakout/daily.py --stage all

两个清单
--------
    清单 A  当天技术形态接近起涨的股票，按分数从强到弱，每天 10 只
    清单 B  曾经在清单 A 上拿过 90 分以上、现在出现见顶特征的股票

清单 A 的分数怎么读
-------------------
分数是 0~100 的**排名**，不是概率。封存数据上实测：每天 10 只里约 1.4 只
会在未来 20 个交易日内涨超 50%，是全市场平均水平（2.9%）的 4.96 倍。

所以清单的正确用法是「这批票里出黑马的密度比市场高 5 倍」，
不是「选出来的会涨」。这句话必须印在每封邮件里。

模型什么时候重训
----------------
存在 state/breakout/model.pkl 里，超过 30 天自动重训。重训只用截止到
昨天的数据，不会碰到未来。封存的那 9 个月已经在验收时用掉了，
之后的模型改动要换一段新的时间验收 —— 见 arena.py 的说明。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "breakout"
OUT = ROOT / "out_breakout"
STATE = ROOT / "state" / "breakout"

import model as M            # noqa: E402
import fselect as FS         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("breakout")

# 清单 A 的规则（2026-09-13 按用户「最大化准确率」的要求定，实测见
# docs/breakout_log.md 实验 7）。
#
# 核心是**门槛制 + 持续性**，不是「每天取前 N 名」：
#   · 用户原话「如果当天没有符合要求的个股，清单可以为空」——
#     所以够格才上，弱势日就空着，不凑数
#   · 用户原话「如某个个股持续符合条件，可以连续几个交易日推荐」——
#     连续够格的天数是最强的单一信号。实验 8 用**生产口径**（≥97 分按
#     预测值取前 10，就是下面这段代码在做的事）重测：
#         全部上榜的       18.04%（5.1 倍，样本 815）
#         连续 2 天         25.65%（7.2 倍，样本 230）
#         连续 3 天         31.31%（8.8 倍，样本  99）
#         连续 4 天         32.61%（9.2 倍，样本  46）
#         连续 5 天         35.00%（9.9 倍，样本  20）
#     光有分数门槛没用，门槛 + 持续性才有效。
#     实验 7 报的 15.43 / 25.81 / 32.04 是「每天只看前 30 名」的口径，
#     和生产不一致，已作废；邮件和面板一律用上面这组。
#
# 入选条件只看**当天**（2026-09-13 用户确认）。实验 8 另测了「过去 5 个
# 交易日内够格 ≥k 次」当入选条件：≥3 次 28.85%、≥4 次 30.14%，和同 k 的
# 连续版在误差内持平，但每天出票从 3.94 只掉到 0.75 只。用户决定不改
# 入选规则，连续天数继续只用来排序和展示。
SCORE_MIN = 97      # 够不到这个分数就不上清单，当天可以为空
CAP_A = 10          # 上限。防止极端强势日几百只同时够格，清单没法看


def load_overrides() -> dict:
    """学习会诊批准过的常量覆盖（state/breakout/overrides.json）。

    只认 SCORE_MIN / CAP_A / MIN_STREAK / drop_features 四个键，其余忽略。
    文件由 learn/council/experiments.apply 写，人在控制台批准才会有；
    删掉文件 = 回到代码里的默认值。读失败按没有处理，不许影响出清单。
    """
    p = ROOT / "state" / "breakout" / "overrides.json"
    try:
        if not p.exists():
            return {}
        o = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for k in ("SCORE_MIN", "CAP_A", "MIN_STREAK"):
            if k in o:
                out[k] = int(o[k])
        if isinstance(o.get("drop_features"), list):
            out["drop_features"] = sorted(str(x) for x in o["drop_features"])
        if isinstance(o.get("BOARD_ADJ"), dict):
            out["BOARD_ADJ"] = {str(k): float(v) for k, v in o["BOARD_ADJ"].items()
                                if k in ("main", "star", "bj", "chinext")}
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("overrides.json 读取失败，按默认值: %s", e)
        return {}


_OVR = load_overrides()
SCORE_MIN = int(_OVR.get("SCORE_MIN", SCORE_MIN))
CAP_A = int(_OVR.get("CAP_A", CAP_A))
DROP_FEATURES = list(_OVR.get("drop_features", []))   # 会诊批准去掉的基础特征名
if _OVR:
    log.info("起涨预测常量覆盖生效：%s", _OVR)
TOP_A = CAP_A       # 兼容旧名字
SCORE_B = SCORE_MIN  # 进 B 池的门槛。清单 A 本身就是 ≥97 分，两者一致
POOL_DAYS = 60      # A 池里的股票保留多少个交易日
MODEL_MAX_AGE = 30  # 模型多少天重训一次
RELEASE_MIN_SHARE = 0.05   # 解禁市值占流通市值达到这个比例才剔除（设计文档 5.7）
TRAIN_END_GAP = 25  # 训练只用到 N 个交易日之前（标签要 20 天才能确定）
MIN_HOLD_DAYS = 5   # 进 A 池后至少过几个交易日才可能进 B

# 板块校正。模型对北交所**严重超配**：北交所占全市场 5.5%，却占清单的 50%，
# 而它的命中率是四个板块里最低的之一。两段独立数据指向同一个方向：
#
#              验证集(2025-03..12)   封存数据(2026-01..08)
#   主板              13.3%                16.1%
#   科创板            14.1%                16.7%
#   北交所             10.0%                11.3%
#   创业板              6.6%                10.3%
#
# 因子 = 该板块命中率 / 整体命中率，取**验证集**的数字算
# （封存数据只许用一次，已经在验收时用掉了，不能拿它来调模型）。
# 这不是拍脑袋的偏好，是用历史证据纠正模型的系统性偏差。
BOARD_ADJ = {"main": 1.19, "star": 1.27, "bj": 0.90, "chinext": 0.59}
BOARD_ADJ.update(_OVR.get("BOARD_ADJ", {}))   # 会诊批准过的板块系数覆盖
RISE_MIN = 0.20     # 进池后至少涨过这么多，才谈得上「波段结束」


def now_bj() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=8)


# ---------------------------------------------------------------
#  模型持久化
# ---------------------------------------------------------------
def model_path() -> Path:
    """模型存两个文件：

        model.txt   LightGBM 的**原生**格式
        model.json  特征列表 + 分数刻度 + 训练日期

    刻意不用 pickle 存整个包装类：pickle 把类的模块路径也存进去了，
    `src/breakout/model.py` 一改名或挪位置，存下来的模型就再也读不出来
    （而且报的是看不懂的 ModuleNotFoundError）。原生格式只认自己，
    和我这边的代码组织完全解耦。
    """
    return STATE / "model.txt"


def feature_fingerprint(df: pd.DataFrame) -> str:
    """训练表特征列的指纹（列名集合 + 筹码算法版本）。"""
    import hashlib
    import chips as CH
    cols = sorted(c for c in df.columns if "__" in c
                  if c.rsplit("__", 1)[0] not in set(DROP_FEATURES))
    key = "|".join(cols) + f"|chips={CH.N_BINS}|drop={','.join(DROP_FEATURES)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_or_fit(df: pd.DataFrame, force: bool = False):
    """加载模型；没有或太旧就重训。

    重训只用到 TRAIN_END_GAP 个交易日之前的数据：更近的数据标签还没定
    （y_up 要看未来 20 个交易日），拿进去训练等于喂了一堆假的负样本。
    """
    p = model_path()
    meta_p = p.with_suffix(".json")
    fp = feature_fingerprint(df)
    if p.exists() and meta_p.exists() and not force:
        try:
            import lightgbm as lgb
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            age = (now_bj().date()
                   - dt.date.fromisoformat(meta["fit_date"])).days
            if meta.get("fingerprint") not in (None, fp):
                # 特征表的列变了（加了特征、改了定义）而模型没重训：
                # 旧模型在新语义的列上打分，列被删还会 KeyError。自动重训。
                log.info("特征表指纹变了（%s -> %s），重训",
                         str(meta.get("fingerprint"))[:8], fp[:8])
            elif age <= MODEL_MAX_AGE:
                log.info("用已有模型（%s 训练，%d 天前，%d 个特征）",
                         meta["fit_date"], age, len(meta["feats"]))
                return {"booster": lgb.Booster(model_file=str(p)),
                        "feats": meta["feats"],
                        "quantiles": np.array(meta["quantiles"]),
                        "fit_date": meta["fit_date"],
                        "train_cut": meta["train_cut"]}
            log.info("模型已 %d 天，超过 %d 天上限，重训", age, MODEL_MAX_AGE)
        except Exception as e:  # noqa: BLE001
            log.warning("模型读不出来（%s），重训", e)

    dates = sorted(df["date"].unique())
    cut = dates[-TRAIN_END_GAP] if len(dates) > TRAIN_END_GAP else dates[0]
    tr = df[df["date"] < cut]
    log.info("重训模型：用 %s 之前的 %d 行", cut, len(tr))
    feats_all = [c for c in df.columns if "__" in c
                 and c.rsplit("__", 1)[0] not in set(DROP_FEATURES)]
    if DROP_FEATURES:
        log.info("会诊批准去掉的特征：%s（剩 %d 列）", DROP_FEATURES, len(feats_all))
    rep = FS.run(tr, feats_all, y="y_up")
    feats = rep["keep"]
    trs = M.stratified_sample(tr, "y_up")
    mdl = M.L1Lgbm(n_estimators=400).fit(trs, feats, "y_up")

    # 分数刻度：把预测值映射成 0~100 的排名。用训练集自己的分布定分位点，
    # 这样「90 分」在任何一天都表示「比训练集里 90% 的样本更像起涨」。
    q = np.quantile(mdl.predict_proba(trs), np.linspace(0, 1, 101))
    STATE.mkdir(parents=True, exist_ok=True)
    mdl.m.booster_.save_model(str(p))
    meta_p.write_text(json.dumps(
        {"feats": feats, "quantiles": [float(x) for x in q],
         "fit_date": now_bj().strftime("%Y-%m-%d"), "train_cut": cut,
         "fingerprint": fp},
        ensure_ascii=False), encoding="utf-8")
    log.info("模型已保存：%d 个特征 -> %s", len(feats), p)
    return {"booster": mdl.m.booster_, "feats": feats, "quantiles": q,
            "fit_date": now_bj().strftime("%Y-%m-%d"), "train_cut": cut}


def to_score(proba: np.ndarray, q: np.ndarray) -> np.ndarray:
    """预测值 -> 0~100 分。见模块 docstring：是排名不是概率。"""
    return np.clip(np.searchsorted(q, proba), 0, 100).astype(float)


# ---------------------------------------------------------------
#  风险剔除
# ---------------------------------------------------------------
def risk_filter(codes: list[str]) -> dict[str, str]:
    """返回 {代码: 剔除原因}。拿不到数据就跳过那一项，不阻断流程。"""
    import warnings
    warnings.filterwarnings("ignore")
    bad: dict[str, str] = {}
    today = now_bj().date()

    # --- ST：腾讯快照的名称带 ST ---
    try:
        import datasource as ds
        # fetch_quotes 收**带市场前缀**的代码（sh600000），返回的键也带前缀。
        # 传裸代码进去腾讯一个都不认，静默返回 0 只，ST 就全漏了。
        # 项目里其它调用方一律走 ds.to_symbol，这里也必须。
        q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
        for sym, v in q.items():
            nm = str(getattr(v, "name", ""))
            if "ST" in nm.upper():
                bad[sym[-6:]] = f"ST（{nm}）"
    except Exception as e:  # noqa: BLE001
        log.warning("ST 检查跳过：%s", e)

    # --- 近 30 天减持（巨潮，带公告日）---
    try:
        import akshare as ak
        d = ak.stock_hold_change_cninfo(symbol="全部")
        col = next((c for c in d.columns if "公告日期" in c), None)
        cc = next((c for c in d.columns if "证券代码" in c), None)
        rc = next((c for c in d.columns if "变动原因" in c), None)
        if col and cc:
            d = d.copy()
            d["_d"] = pd.to_datetime(d[col], errors="coerce").dt.date
            recent = d[d["_d"] >= today - dt.timedelta(days=30)]
            for _, r in recent.iterrows():
                c = str(r[cc]).zfill(6)
                if c in codes and c not in bad:
                    why = str(r[rc]) if rc else ""
                    if "减" in why or not why:
                        bad[c] = "近 30 天有减持公告"
    except Exception as e:  # noqa: BLE001
        log.warning("减持检查跳过：%s", e)

    # --- 未来 30 天解禁：只剔「解禁市值占流通市值 >= RELEASE_MIN_SHARE」的 ---
    # 设计文档 5.7 定的门槛。以前任何一笔（含股权激励零头）都整只剔除，
    # 会把够格的票挤出前 10（2026-09-15 用户决定按 5% 门槛）。
    try:
        import akshare as ak
        d = ak.stock_restricted_release_detail_em(
            start_date=today.strftime("%Y%m%d"),
            end_date=(today + dt.timedelta(days=30)).strftime("%Y%m%d"))
        cc = next((c for c in d.columns if "代码" in c), None)
        sc_ = next((c for c in d.columns if "占解禁前流通市值比例" in c), None)
        if cc and sc_:
            share = pd.to_numeric(d[sc_], errors="coerce")
            if share.max() > 1.0:          # 万一哪天改成百分数
                share = share / 100.0
            # 同一只票 30 天内多笔解禁按票累计
            tot = (d.assign(_c=d[cc].astype(str).str.zfill(6), _s=share.fillna(0))
                    .groupby("_c")["_s"].sum())
            for c, v in tot.items():
                if c in codes and c not in bad and v >= RELEASE_MIN_SHARE:
                    bad[c] = f"未来 30 天解禁占流通市值 {100 * v:.1f}%"
        elif cc:
            log.warning("解禁表没有占比列，退回「任何解禁都剔」")
            for c in d[cc].astype(str).str.zfill(6):
                if c in codes and c not in bad:
                    bad[c] = "未来 30 天有解禁"
    except Exception as e:  # noqa: BLE001
        log.warning("解禁检查跳过：%s", e)

    # --- 近期定增 ---
    try:
        import akshare as ak
        d = ak.stock_qbzf_em()
        cc = next((c for c in d.columns if "股票代码" in c), None)
        dc = next((c for c in d.columns if "发行日期" in c), None)
        if cc and dc:
            d = d.copy()
            d["_d"] = pd.to_datetime(d[dc], errors="coerce").dt.date
            recent = d[d["_d"] >= today - dt.timedelta(days=60)]
            for c in recent[cc].astype(str).str.zfill(6):
                if c in codes and c not in bad:
                    bad[c] = "近 60 天有增发"
    except Exception as e:  # noqa: BLE001
        log.warning("定增检查跳过：%s", e)

    return bad


# ---------------------------------------------------------------
#  A 池（清单 B 要用）
# ---------------------------------------------------------------
def pool_path() -> Path:
    return STATE / "a_pool.json"


def update_pool(picks: pd.DataFrame, date: str) -> dict:
    """把当天 90 分以上的票记进 A 池，顺便清掉过期的。

    除了最高分，还累计两个给清单 B 用的数字：

        days    一共上过几天清单 A（不要求连续）
        streak  上榜期间最长的连续天数

    同一天重复跑（补跑、手点控制台）不能把 days 加两次，所以按
    `last == date` 判重。
    """
    p = pool_path()
    pool = {}
    if p.exists():
        try:
            pool = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pool = {}
    pool = pool_step(pool, picks, date)
    STATE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pool, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return pool


def pool_step(pool: dict, picks: pd.DataFrame, date: str) -> dict:
    """A 池推进一天（纯函数，不碰文件）。update_pool 和历史回放共用。"""
    pool = {k: dict(v) for k, v in pool.items()}
    for _, r in picks[picks["score"] >= SCORE_B].iterrows():
        c = r["code"]
        e = pool.get(c, {"first": date, "best": 0.0, "days": 0})
        if e.get("last") != date:
            e["days"] = int(e.get("days", 0)) + 1
        e["last"] = date
        e["best"] = max(float(e.get("best", 0)), float(r["score"]))
        e["streak"] = max(int(e.get("streak", 0)),
                          int(r.get("streak", 1) or 1))
        e["name"] = r.get("name", "")
        pool[c] = e
    # 过期清理：POOL_DAYS 个交易日按 1.47 折算成自然日
    cut = (dt.date.fromisoformat(date)
           - dt.timedelta(days=int(POOL_DAYS * 1.47))).isoformat()
    return {k: v for k, v in pool.items() if v.get("last", "") >= cut}


def recent_history(n: int = 30) -> list[dict]:
    """最近 n 个交易日的清单 A/B，给面板的日期下拉用。

    清单 A 读每天落盘的 breakout_<date>.parquet；清单 B 当时没有落盘，
    这里从最早一份清单起把 A 池逐日回放出来，再按「那天为止」的价格算
    （和补发工具 tools/resend_breakout.py 同一套逻辑）。
    """
    files = sorted(DATA.glob("*/breakout_*.parquet"))
    if not files:
        return []
    dp = DATA / "daily.parquet"
    px = (pd.read_parquet(dp, columns=["code", "date", "close"])
          if dp.exists() else pd.DataFrame(columns=["code", "date", "close"]))
    pool: dict = {}
    out = []
    for f in files:
        picks = pd.read_parquet(f)
        if not len(picks):
            continue
        date = str(picks["date"].iloc[0])
        pool = pool_step(pool, picks, date)
        blist = build_list_b(px, pool, asof=date) if len(px) else pd.DataFrame()
        # 2026-09-15 之前落盘的清单没有这两列，抬头就不印
        first = picks.iloc[0]
        rej = first.get("rejected")
        out.append({"date": date, "a": picks, "b": blist,
                    "meta": {"date": date, "n_a": len(picks), "n_b": len(blist),
                             "n_streak3": int((picks["streak"] >= 3).sum())
                             if "streak" in picks else 0,
                             "pool": len(pool),
                             "model_date": first.get("model_date"),
                             "rejected": None if pd.isna(rej) else int(rej)}})
    return out[-n:]


# ---------------------------------------------------------------
def build_list_b(df: pd.DataFrame, pool: dict,
                 asof: str = "") -> pd.DataFrame:
    """清单 B：A 池里的票**涨上去之后**见顶。

    用户的原意是「A 清单的票涨了一波，现在到顶了」。所以三个条件缺一不可：

      1. 进池后至少过了 MIN_HOLD_DAYS 个交易日（当天进池当天见顶是荒谬的）
      2. 进池之后确实涨过 RISE_MIN（没涨过就谈不上「波段结束」）
      3. 现在从那个高点回落 8%~20%，且高点就在最近 10 天内

    第一版漏了前两条，结果 A 池刚建立当天，5 只票同时出现在 A 和 B 上，
    一边说「接近起涨」一边说「见顶」。

    `df` 只需要 code / date / close 三列，所以补发历史清单时可以喂
    `daily.parquet`（122MB）而不是 `train.parquet`（2.5GB）。
    `asof` 给补发用：只看到那一天为止的价格，不许用之后的。
    """
    if asof:
        df = df[df["date"] <= asof]
    bl = []
    for c, info in pool.items():
        g = df[df["code"] == c].sort_values("date")
        first = info.get("first", "")
        after = g[g["date"] >= first]
        # after 含进池当天那一行，「进池后满 MIN_HOLD_DAYS 个交易日」要 +1
        if len(after) < MIN_HOLD_DAYS + 1:
            continue
        px = after["close"].to_numpy(float)
        if len(px) < 3 or not np.isfinite(px).all() or px[0] <= 0:
            continue
        peak_i = int(np.nanargmax(px))
        peak = float(px[peak_i])
        rise = peak / px[0] - 1                 # 进池之后涨了多少
        drop = px[-1] / peak - 1                # 从高点回落多少
        recent_peak = (len(px) - 1 - peak_i) <= 10
        if rise >= RISE_MIN and -0.20 < drop < -0.08 and recent_peak:
            # 清单 B 的列和清单 A 对齐：代码 / 名称 / 分数 / 上榜天数 /
            # 历史准确率 / 现价。准确率由 export 按 streak 换算，
            # 这里只把它上清单 A 时的最长连续天数带出去。
            bl.append({"code": c, "name": info.get("name", ""),
                       "best": info["best"],
                       "days": int(info.get("days", 0)),
                       "streak": int(info.get("streak", 1) or 1),
                       "close": float(px[-1]),
                       "rise": round(100 * rise, 1),
                       "drop": round(100 * drop, 1),
                       "first": first})
    return pd.DataFrame(bl)


_CAL: dict = {}


def prev_trade_days(date: str, k: int) -> list[str]:
    """date 之前的 k 个交易日，由近到远。

    以前按自然日回退、只跳周末：每个法定假日（中秋、国庆）都让连续计数
    整榜归零，清单顺序、准确率那一列、A 池的 streak 全错一天以上。
    日历来自 datasource.trade_dates（接口 -> state/trade_dates.json 缓存）；
    两边都拿不到才退化成跳周末。
    """
    try:
        if "tds" not in _CAL:                 # 一次扫描里每只票都要查，缓存
            import datasource as ds
            _CAL["tds"] = sorted(ds.trade_dates())
        tds = [d for d in _CAL["tds"] if d < date]
        return list(reversed(tds[-k:]))
    except Exception as e:  # noqa: BLE001
        log.warning("交易日历拿不到（%s），连续天数按跳周末算", e)
        out, cur = [], dt.date.fromisoformat(date)
        while len(out) < k:
            cur -= dt.timedelta(days=1)
            if cur.weekday() < 5:
                out.append(cur.isoformat())
        return out


def count_streak(code: str, date: str) -> int:
    """这只票在 date **之前**已经连续够格几天。

    读的是每天落盘的 breakout_<date>.parquet。断一天就归零 ——
    「持续符合条件」指的是没断过，不是「最近几天里有几天符合」。
    """
    n = 0
    for prev in prev_trade_days(date, 12):     # 最多往回数 12 个交易日
        f = DATA / prev[:7] / f"breakout_{prev}.parquet"
        if not f.exists():
            break
        try:
            prev = pd.read_parquet(f, columns=["code"])
        except Exception:  # noqa: BLE001
            break
        if code not in set(prev["code"].astype(str)):
            break
        n += 1
    return n


def stage_scan(force_fit: bool = False) -> int:
    """打分日永远是特征表的最后一天。以前有个 --asof 能指定历史日，但风险
    剔除用的是今天的 ST/减持/解禁、A 池会被改坏进池日期，补发不安全，
    2026-09-15 删掉；补发历史清单走 tools/resend_breakout.py。"""
    t0 = time.time()
    tp = DATA / "train.parquet"
    if not tp.exists():
        log.error("缺 %s。先在控制台跑「起涨预测·补数据」和「建特征表」", tp)
        return 1
    df = pd.read_parquet(tp)
    fc = [c for c in df.columns if "__" in c]
    df[fc] = df[fc].astype("float32")

    date = str(df["date"].max())
    today = df[df["date"] == date].copy()
    log.info("打分日 %s，全市场 %d 只", date, len(today))
    if not len(today):
        log.error("%s 没有数据", date)
        return 1

    obj = load_or_fit(df, force_fit)
    X = np.nan_to_num(today[obj["feats"]].to_numpy(np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    proba = obj["booster"].predict(X)
    adj = today["board"].map(BOARD_ADJ).fillna(1.0).to_numpy(float)
    # 排序必须用**连续**的预测值，分数只是给人看的整数刻度。
    # 2026-09-13 实测：前几名的分数取整后都是 97~99，大量并列，
    # 按整数排序时排第 1 的往往不是真正得分最高的那只 ——
    # 「第 1 名」的命中率因此从 25.12% 掉到 21.74%，白丢 3.4 个百分点。
    today["_p"] = proba * adj
    today["score"] = to_score(today["_p"].to_numpy(), obj["quantiles"])
    today = today.sort_values("_p", ascending=False)

    # 次新股：上市不足 60 个交易日的特征算不出来，直接剔除
    cnt = df.groupby("code")["date"].size()
    today = today[today["code"].map(cnt).fillna(0) >= 120]

    log.info("风险剔除中（ST / 减持 / 解禁 / 增发）")
    cand = today.head(60)["code"].tolist()      # 只查前 60 只，省接口调用
    bad = risk_filter(cand)
    today["reject"] = today["code"].map(bad).fillna("")
    ok = today[(today["reject"] == "") & (today["score"] >= SCORE_MIN)]
    picks = ok.head(CAP_A).copy()
    log.info("清单 A：%d 只够格（≥%d 分），剔除 %d 只",
             len(picks), SCORE_MIN, len(bad))
    if not len(picks):
        log.info("今天没有够格的股票，清单 A 为空。这是正常的，不是故障。")

    # 连续够格天数：清单里最强的信号。从已落盘的历史里数，
    # 相邻交易日才算连续，断一天就重新计数。
    picks["streak"] = [count_streak(c, date) + 1 for c in picks["code"]]
    picks = picks.sort_values(["streak", "_p"], ascending=[False, False])

    # 名字从腾讯快照拿
    try:
        import datasource as ds
        q = ds.fetch_quotes([ds.to_symbol(c) for c in picks["code"]])
        picks["name"] = picks["code"].map(
            lambda c: getattr(q.get(ds.to_symbol(c)), "name", ""))
    except Exception:  # noqa: BLE001
        picks["name"] = ""

    pool = update_pool(picks, date)

    blist = build_list_b(df, pool)
    log.info("清单 B：%d 只（A 池 %d 只）", len(blist), len(pool))

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["code", "name", "score", "streak", "close", "board"]
    picks[cols].to_json(OUT / "list_a.json", orient="records",
                        force_ascii=False, indent=2)
    blist.to_json(OUT / "list_b.json", orient="records",
                  force_ascii=False, indent=2)
    import os
    (OUT / "run_meta.json").write_text(json.dumps(
        {"date": date, "n_a": len(picks), "n_b": len(blist),
         "score_min": SCORE_MIN,
         "n_streak3": int((picks["streak"] >= 3).sum()) if len(picks) else 0,
         "pool": len(pool), "model_date": obj["fit_date"],
         "rejected": len(bad),
         # 试跑也走到这里；不标 dry 的话计划任务会把试跑当「今天跑完了」
         "dry": bool(os.environ.get("DRY_RUN"))},
        ensure_ascii=False), encoding="utf-8")
    (DATA / date[:7]).mkdir(parents=True, exist_ok=True)
    # 模型日期和剔除数也落盘：面板按日期回看时抬头要印这两个数
    picks = picks.assign(model_date=obj["fit_date"], rejected=len(bad))
    picks.to_parquet(DATA / date[:7] / f"breakout_{date}.parquet", index=False)
    log.info("完成，用时 %.1f 分钟", (time.time() - t0) / 60)
    return 0


def load_env() -> None:
    """tools/local.env -> os.environ（实现在 src/localenv.py，三个入口共用）。"""
    import localenv
    localenv.load()


def stage_send() -> int:
    load_env()
    import export as E
    meta = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    date = meta["date"]
    # dtype 必须给：read_json 把 "002652" 推断成 int64 = 2652，邮件里就丢了
    # 前导零。至今没暴露只是因为两次真实运行恰好全是 688/920/600 段的票。
    a = pd.read_json(OUT / "list_a.json", dtype={"code": str})
    b = pd.read_json(OUT / "list_b.json", dtype={"code": str})
    for x in (a, b):
        if "code" in x.columns:
            x["code"] = x["code"].astype(str).str.zfill(6)
    try:
        hist = recent_history(30)
    except Exception as e:  # noqa: BLE001
        log.warning("历史清单回放失败（面板只带当天）: %s", e)
        hist = []
    E.write_panel(a, b, meta, OUT, date, history=hist)
    import os
    if os.environ.get("SKIP_MAIL"):
        log.info("SKIP_MAIL=1，只生成面板不发邮件")
        return 0
    E.send_mail(date, a, b, meta)
    return 0


def stage_refit() -> int:
    """只重训模型，不打分不发信。晚间系统的「模型自学」。

    平时不用手点：scan 阶段发现模型超过 30 天会自己重训。这个入口是给
    「我现在就想让它重新学一遍」用的，比如刚补了一批新数据。
    """
    tp = DATA / "train.parquet"
    if not tp.exists():
        log.error("缺 %s，先跑「补数据」和「建特征表」", tp)
        return 1
    df = pd.read_parquet(tp)
    fc = [c for c in df.columns if "__" in c]
    df[fc] = df[fc].astype("float32")
    obj = load_or_fit(df, force=True)
    log.info("重训完成：%d 个特征，数据截止 %s",
             len(obj["feats"]), obj["train_cut"])
    return 0


def model_status() -> dict:
    """模型的年龄和下次重训时间。控制台总览读它。"""
    mp = model_path().with_suffix(".json")
    if not mp.exists():
        return {"exists": False}
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        age = (now_bj().date() - dt.date.fromisoformat(meta["fit_date"])).days
        return {"exists": True, "fit_date": meta["fit_date"],
                "age_days": age, "n_feats": len(meta["feats"]),
                "train_cut": meta["train_cut"],
                "days_to_refit": max(MODEL_MAX_AGE - age, 0)}
    except Exception:  # noqa: BLE001
        return {"exists": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["scan", "send", "all", "refit"])
    ap.add_argument("--refit", action="store_true", help="强制重训模型")
    a = ap.parse_args()
    if a.stage == "refit":
        return stage_refit()
    rc = 0
    if a.stage in ("scan", "all"):
        rc = stage_scan(a.refit)
        if rc:
            return rc
    if a.stage in ("send", "all"):
        rc = stage_send()
    return rc


if __name__ == "__main__":
    sys.exit(main())
