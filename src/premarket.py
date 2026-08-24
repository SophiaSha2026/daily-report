"""
盘前候选池构建（北京时间 08:35 运行，时间宽裕）。

两阶段收缩，避免对全市场 5000+ 只票逐个拉日线：
  阶段1  仅用一次全市场快照做廉价过滤  -> 约 400-800 只
  阶段2  只对存活者拉 90 日日线算形态  -> 每只 ~0.35s

产出 cache/universe.parquet，供 09:25 的竞价任务直接读取。
"""
from __future__ import annotations

import sys
import time
import logging
import datetime as dt
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datasource as ds
from datasource import limit_pct

ROOT = Path(__file__).resolve().parent.parent
TZ = dt.timezone(dt.timedelta(hours=8))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("premarket")


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
def stage1(spot: pd.DataFrame, c: dict) -> pd.DataFrame:
    """只用全市场快照做过滤。此时快照反映的是上一交易日收盘状态。"""
    u = c["universe"]
    df = spot.rename(columns={"代码": "code", "名称": "name"}).copy()
    for col in ("最新价", "涨跌幅", "成交额", "换手率", "总市值"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = df[~df["name"].astype(str).str.upper().str.contains("ST|退", na=False)]
    df = df[~df["name"].astype(str).str.match(r"^[NC] ", na=False)]   # 次新
    df = df[df["成交额"].fillna(0) > 0]                                # 停牌
    if u["exclude_bj"]:
        df = df[~df["code"].astype(str).str[0].isin(["8", "4"])]

    if u["max_mktcap_yi"]:
        df = df[df["总市值"] <= u["max_mktcap_yi"] * 1e8]
    if u["min_mktcap_yi"]:
        df = df[df["总市值"] >= u["min_mktcap_yi"] * 1e8]

    df["prev_amount"] = df["成交额"]
    df["prev_gain"] = df["涨跌幅"]
    df["lim"] = [limit_pct(r.code, r.name) for r in df.itertuples()]
    df["prev_limit_up"] = df["prev_gain"] >= df["lim"] - 0.4

    inc = c["universe"]["include_if"]
    keep = pd.Series(False, index=df.index)
    if inc.get("yesterday_limit_up"):
        keep |= df["prev_limit_up"]
    if inc.get("yesterday_gain_pct_gte") is not None:
        keep |= df["prev_gain"] >= inc["yesterday_gain_pct_gte"]
    keep |= df["换手率"].fillna(0) >= 5.0
    keep |= df["成交额"].rank(ascending=False) <= 600

    df = df[keep].copy()
    df = df[~df["prev_gain"].le(-9.0)]        # 昨日跌停
    df = df.nlargest(min(len(df), u["max_candidates"]), "成交额")
    log.info("阶段1: %d -> %d", n0, len(df))
    return df.reset_index(drop=True)


def stage2(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """
    对存活者拉 90 日日线，算位置/形态/连板高度。

    日线是并发拉的（datasource.daily_hist_many）：1600 只串行要 9 分钟以上，
    盘前到竞价只有 50 分钟，串行没有余量。源的优先级和熔断在 datasource 里。
    """
    end = dt.datetime.now(TZ).strftime("%Y%m%d")
    start = (dt.datetime.now(TZ) - dt.timedelta(days=140)).strftime("%Y%m%d")
    look = c["screen"]["breakout_lookback"]

    t0 = time.time()
    hists = ds.daily_hist_many([r.code for r in df.itertuples()], start, end)
    log.info("日线拉取完成 %d 只，耗时 %.0fs，来源 %s",
             len(hists), time.time() - t0, ds.hist_source_stats())

    rec = []
    empty = 0
    for r in df.itertuples():
        h = hists.get(r.code)
        d = dict(pos_pct_60d=0.5, ma_bull=False, platform_high=1e9,
                 board_height=0, prev_broken_board=False,
                 amount_ratio_5d=1.0)
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
            base = float(am.tail(6).iloc[:-1].mean()) or 1.0
            d["amount_ratio_5d"] = float(am.iloc[-1] / base)
        lim = r.lim - 0.4
        n = 0
        for v in reversed(pc.tolist()):
            if v >= lim:
                n += 1
            else:
                break
        d["board_height"] = n
        # 昨日炸板：盘中触及涨停但收盘未封
        if len(hi) and r.prev_gain < lim:
            prev_close = float(cl.iloc[-2]) if len(cl) >= 2 else 0
            if prev_close and float(hi.iloc[-1]) >= prev_close * (1 + r.lim / 100) - 0.01:
                d["prev_broken_board"] = True
        rec.append(d)

    if empty:
        log.warning("阶段2: %d/%d 只没拿到日线，形态字段用默认值",
                    empty, len(df))
    log.info("阶段2 完成 %d 只，耗时 %.0fs", len(df), time.time() - t0)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rec)], axis=1)


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
        d = dt.datetime.now(TZ).strftime("%Y-%m-%d")
        nt = ak.stock_notice_report(symbol="全部", date=d)
        if nt is not None and len(nt):
            col_c = next(x for x in nt.columns if "代码" in x)
            col_t = next(x for x in nt.columns if "标题" in x or "名称" in x)
            hit = {str(r[col_c]).zfill(6) for _, r in nt.iterrows()
                   if any(k in str(r[col_t]) for k in kws)}
            df["blacklisted"] = df["code"].isin(hit)
            log.info("公告黑名单命中 %d 只", int(df["blacklisted"].sum()))
    except Exception as e:  # noqa: BLE001
        log.warning("公告接口不可用(%s)，本次不做公告过滤", e)
    return df


def main() -> int:
    c = cfg()
    today = dt.datetime.now(TZ).strftime("%Y-%m-%d")
    try:
        if today not in ds.trade_dates():
            log.info("%s 非交易日", today)
            return 0
    except Exception:  # noqa: BLE001
        if dt.datetime.now(TZ).weekday() >= 5:
            return 0

    df = stage1(ds.spot_all(), c)
    df = stage2(df, c)
    df = attach_sector(df)
    df = attach_blacklist(df, c)

    cols = ["code", "name", "lim", "prev_amount", "prev_gain", "prev_limit_up",
            "prev_broken_board", "board_height", "pos_pct_60d", "ma_bull",
            "platform_high", "amount_ratio_5d", "sector",
            "sector_prev_limitups", "blacklisted"]
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    df[cols].to_parquet(out / "universe.parquet", index=False)
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
