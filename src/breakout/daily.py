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

TOP_A = 10          # 清单 A 每天几只（用户 2026-09-12 定的）
SCORE_B = 90        # 进 B 池的分数门槛
POOL_DAYS = 60      # A 池里的股票保留多少个交易日
MODEL_MAX_AGE = 30  # 模型多少天重训一次
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


def load_or_fit(df: pd.DataFrame, force: bool = False):
    """加载模型；没有或太旧就重训。

    重训只用到 TRAIN_END_GAP 个交易日之前的数据：更近的数据标签还没定
    （y_up 要看未来 20 个交易日），拿进去训练等于喂了一堆假的负样本。
    """
    p = model_path()
    meta_p = p.with_suffix(".json")
    if p.exists() and meta_p.exists() and not force:
        try:
            import lightgbm as lgb
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            age = (now_bj().date()
                   - dt.date.fromisoformat(meta["fit_date"])).days
            if age <= MODEL_MAX_AGE:
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
    feats_all = [c for c in df.columns if "__" in c]
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
         "fit_date": now_bj().strftime("%Y-%m-%d"), "train_cut": cut},
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

    # --- 未来 30 天解禁 ---
    try:
        import akshare as ak
        d = ak.stock_restricted_release_detail_em(
            start_date=today.strftime("%Y%m%d"),
            end_date=(today + dt.timedelta(days=30)).strftime("%Y%m%d"))
        cc = next((c for c in d.columns if "代码" in c), None)
        if cc:
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
    """把当天 90 分以上的票记进 A 池，顺便清掉过期的。"""
    p = pool_path()
    pool = {}
    if p.exists():
        try:
            pool = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pool = {}
    for _, r in picks[picks["score"] >= SCORE_B].iterrows():
        c = r["code"]
        e = pool.get(c, {"first": date, "best": 0.0})
        e["last"] = date
        e["best"] = max(float(e.get("best", 0)), float(r["score"]))
        e["name"] = r.get("name", "")
        pool[c] = e
    # 过期清理：POOL_DAYS 个交易日按 1.47 折算成自然日
    cut = (dt.date.fromisoformat(date)
           - dt.timedelta(days=int(POOL_DAYS * 1.47))).isoformat()
    pool = {k: v for k, v in pool.items() if v.get("last", "") >= cut}
    STATE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pool, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return pool


# ---------------------------------------------------------------
def stage_scan(asof: str = "", force_fit: bool = False) -> int:
    t0 = time.time()
    tp = DATA / "train.parquet"
    if not tp.exists():
        log.error("缺 %s。先在控制台跑「起涨预测·补数据」和「建特征表」", tp)
        return 1
    df = pd.read_parquet(tp)
    fc = [c for c in df.columns if "__" in c]
    df[fc] = df[fc].astype("float32")

    date = asof or str(df["date"].max())
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
    today["score"] = to_score(proba * adj, obj["quantiles"])
    today = today.sort_values("score", ascending=False)

    # 次新股：上市不足 60 个交易日的特征算不出来，直接剔除
    cnt = df.groupby("code")["date"].size()
    today = today[today["code"].map(cnt).fillna(0) >= 120]

    log.info("风险剔除中（ST / 减持 / 解禁 / 增发）")
    cand = today.head(60)["code"].tolist()      # 只查前 60 只，省接口调用
    bad = risk_filter(cand)
    today["reject"] = today["code"].map(bad).fillna("")
    picks = today[today["reject"] == ""].head(TOP_A).copy()
    log.info("清单 A：%d 只（剔除 %d 只）", len(picks), len(bad))

    # 名字从腾讯快照拿
    try:
        import datasource as ds
        q = ds.fetch_quotes([ds.to_symbol(c) for c in picks["code"]])
        picks["name"] = picks["code"].map(
            lambda c: getattr(q.get(ds.to_symbol(c)), "name", ""))
    except Exception:  # noqa: BLE001
        picks["name"] = ""

    pool = update_pool(picks, date)

    # ---- 清单 B：A 池里的票**涨上去之后**见顶 ----
    # 用户的原意是「A 清单的票涨了一波，现在到顶了」。所以三个条件缺一不可：
    #   1. 进池后至少过了 MIN_HOLD_DAYS 个交易日（当天进池当天见顶是荒谬的）
    #   2. 进池之后确实涨过 RISE_MIN（没涨过就谈不上「波段结束」）
    #   3. 现在从那个高点回落 8%~20%，且高点就在最近 10 天内
    #
    # 第一版漏了前两条，结果 A 池刚建立当天，5 只票同时出现在 A 和 B 上，
    # 一边说「接近起涨」一边说「见顶」。
    bl = []
    for c, info in pool.items():
        g = df[df["code"] == c].sort_values("date")
        first = info.get("first", "")
        after = g[g["date"] >= first]
        if len(after) < MIN_HOLD_DAYS:
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
            bl.append({"code": c, "name": info.get("name", ""),
                       "best": info["best"],
                       "rise": round(100 * rise, 1),
                       "drop": round(100 * drop, 1),
                       "first": first})
    blist = pd.DataFrame(bl)
    log.info("清单 B：%d 只（A 池 %d 只）", len(blist), len(pool))

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["code", "name", "score", "close", "board"]
    picks[cols].to_json(OUT / "list_a.json", orient="records",
                        force_ascii=False, indent=2)
    blist.to_json(OUT / "list_b.json", orient="records",
                  force_ascii=False, indent=2)
    (OUT / "run_meta.json").write_text(json.dumps(
        {"date": date, "n_a": len(picks), "n_b": len(blist),
         "pool": len(pool), "model_date": obj["fit_date"],
         "rejected": len(bad)}, ensure_ascii=False), encoding="utf-8")
    (DATA / date[:7]).mkdir(parents=True, exist_ok=True)
    picks.to_parquet(DATA / date[:7] / f"breakout_{date}.parquet", index=False)
    log.info("完成，用时 %.1f 分钟", (time.time() - t0) / 60)
    return 0


def load_env() -> None:
    """把 tools/local.env 读进环境变量。

    SMTP 配置在那个文件里，不在环境里。local_run.py 走的时候会先加载，
    但 daily.py 单独跑（控制台点按钮、手动命令行）时没人加载，
    发信就会报 KeyError: 'SMTP_HOST'。
    """
    f = ROOT / "tools" / "local.env"
    if not f.exists():
        return
    import os
    for ln in f.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def stage_send(asof: str = "") -> int:
    load_env()
    import export as E
    meta = json.loads((OUT / "run_meta.json").read_text(encoding="utf-8"))
    date = asof or meta["date"]
    a = pd.read_json(OUT / "list_a.json")
    b = pd.read_json(OUT / "list_b.json")
    E.write_panel(a, b, meta, OUT, date)
    import os
    if os.environ.get("SKIP_MAIL"):
        log.info("SKIP_MAIL=1，只生成面板不发邮件")
        return 0
    E.send_mail(date, a, b, meta)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["scan", "send", "all"])
    ap.add_argument("--asof", default="")
    ap.add_argument("--refit", action="store_true", help="强制重训模型")
    a = ap.parse_args()
    rc = 0
    if a.stage in ("scan", "all"):
        rc = stage_scan(a.asof, a.refit)
        if rc:
            return rc
    if a.stage in ("send", "all"):
        rc = stage_send(a.asof)
    return rc


if __name__ == "__main__":
    sys.exit(main())
