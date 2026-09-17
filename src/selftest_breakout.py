"""
爆发线离线自测。和另外四条一样：不联网、不碰 state/、几秒内跑完。

钉住四类东西
------------
  1. 筹码算法的六项一致性检验（没法和东财对比，接口被掐，只能靠数学性质）
  2. 标签函数在人工构造的序列上给出正确答案
  3. **前视偏差**：构造一个「只有未来信息才能预测」的假数据集，
     整条管线必须学不会它（AUC ≈ 0.5）。学会了就说明某处漏了未来信息。
  4. 横截面百分位化真的抹掉了「今天大盘好」这个信息 ——
     这是整套设计对抗时间聚集的主力手段，它失效是静默的

第 3 条比任何 code review 都可靠：前视偏差的典型症状是回测惊艳、实盘崩溃，
而人工构造的数据集能在**开发期**就把它抓出来。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "breakout"))

fails: list[str] = []


def ck(cond: bool, msg: str) -> None:
    print(f"  {'✓' if cond else '✗'} {msg}")
    if not cond:
        fails.append(msg)


def synth(n: int = 600, seed: int = 3) -> pd.DataFrame:
    """合成一只票的日线。几何布朗 + 真实量级的振幅和换手。"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.022, n)
    close = 10 * np.exp(np.cumsum(ret))
    amp = np.abs(rng.normal(0.018, 0.010, n)) + 0.004
    high = close * (1 + amp * rng.uniform(.3, 1, n))
    low = close * (1 - amp * rng.uniform(.3, 1, n))
    turn = np.clip(np.abs(rng.normal(0.012, 0.010, n)), 0.001, 0.25)
    return pd.DataFrame({
        "date": pd.date_range("2023-05-01", periods=n, freq="B")
                  .strftime("%Y-%m-%d"),
        "open": close, "close": close, "high": high, "low": low,
        "volume": turn * 1e8, "turnover": turn, "code": "000001"})


# ---------------------------------------------------------------
def check_chips() -> None:
    print("\n[筹码算法·六项一致性]")
    from chips import chip_features
    d = synth(800)
    f = chip_features(d.high.to_numpy(), d.low.to_numpy(),
                      d.close.to_numpy(), d.turnover.to_numpy())
    d = pd.concat([d, f], axis=1)
    ok = d.dropna(subset=["chip_avg"])

    vwap60 = ((d.close * d.volume).rolling(60).sum()
              / d.volume.rolling(60).sum())
    corr = ok.chip_avg.corr(vwap60.loc[ok.index])
    ck(corr > 0.90, f"平均成本 vs 60日VWAP 相关 {corr:.3f} > 0.90")

    nh = d[d.close >= d.close.rolling(120).max() * 0.999].dropna(
        subset=["chip_win"])
    ck(nh.chip_win.median() > 0.90,
       f"创120日新高时获利盘 {nh.chip_win.median():.1%} > 90%")
    nl = d[d.close <= d.close.rolling(120).min() * 1.001].dropna(
        subset=["chip_win"])
    ck(nl.chip_win.median() < 0.10,
       f"创120日新低时获利盘 {nl.chip_win.median():.1%} < 10%")

    # 「获利盘与价格水平正相关」在随机游走上很弱（价格没有持续趋势，
    # 筹码一路跟着价格走，获利盘就一直在 50% 附近晃）。
    # 更本质的性质是**变化方向**：价格涨则获利盘增加，这在任何路径上都成立。
    dw = ok.chip_win.diff()
    dc = ok.close.pct_change()
    sp = dw.corr(dc, method="spearman")
    ck(sp > 0.5, f"价格涨跌 vs 获利盘增减 秩相关 {sp:.3f} > 0.5")
    ck(bool(((ok.chip_conc90 > 0) & (ok.chip_conc90 < 3)).all()),
       "集中度落在 (0,3) 区间")
    ck(bool((ok.chip_win.between(0, 1)).all()), "获利盘落在 [0,1]")

    # 换手率传成百分数是最容易犯的错，必须炸而不是静默算错
    bad = False
    try:
        chip_features(d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(),
                      d.turnover.to_numpy() * 100)
    except AssertionError:
        bad = True
    ck(bad, "换手率传成百分数时会断言失败（不静默算错）")


def check_labels() -> None:
    print("\n[标签]")
    from label import label_up, first_days, label_top
    # 人工序列：第 10 天起翻倍，之后腰斩
    c = np.array([10.0] * 10 + [20.0] * 10 + [9.0] * 30)
    h = c * 1.01
    y = label_up(c, h, window=20, threshold=0.50)
    ck(y[5] == 1.0, "第 5 天能看到 20 日内涨超 50% -> 正样本")
    ck(y[25] == 0.0, "第 25 天之后是下跌 -> 负样本")
    ck(bool(np.isnan(y[-3])), "末尾看不到完整未来窗口 -> NaN 而不是 0")

    t0 = first_days(y)
    ck(np.nansum(t0) < np.nansum(y),
       f"起涨点 {int(np.nansum(t0))} 个 < 连续正样本 {int(np.nansum(y))} 个")
    ck(t0[0] == 1.0 and t0[1] == 0.0, "连续区间只有第一天算起涨点")

    top = label_top(c, h, t0, window=20)
    idx = int(np.nanargmax(np.nan_to_num(top)))
    ck(bool(np.nanmax(top) == 1.0), "见顶标签有命中")
    ck(9 <= idx <= 20, f"见顶标在最高点附近（idx={idx}，最高点区间 10~19）")


def check_lookahead() -> None:
    """前视偏差：构造一个只有未来信息才能预测的标签，管线必须学不会。"""
    print("\n[前视偏差·最重要的一条]")
    from model import L1Lgbm, stratified_sample
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(11)
    n = 40000
    df = pd.DataFrame({
        "date": np.repeat(pd.date_range("2024-01-01", periods=200, freq="B")
                          .strftime("%Y-%m-%d"), n // 200),
        "code": np.tile(np.arange(n // 200).astype(str), 200),
    })
    # 特征纯噪音，和标签无关
    for i in range(8):
        df[f"f{i}__last"] = rng.normal(size=n)
    # 标签也是纯噪音（真实世界里它取决于未来，而未来不在特征里）
    df["y"] = (rng.random(n) < 0.03).astype(float)

    tr = df[df.date < "2024-07-01"]
    te = df[df.date >= "2024-07-01"]
    feats = [c for c in df.columns if "__" in c]
    m = L1Lgbm(n_estimators=120).fit(stratified_sample(tr, "y"), feats, "y")
    auc = roc_auc_score(te["y"], m.predict_proba(te))
    ck(0.44 < auc < 0.56,
       f"特征与标签无关时 AUC={auc:.3f} 应接近 0.5（偏离说明管线漏了信息）")


def check_holder_alignment() -> None:
    """股东人数必须按公告日期可用，不能按报告期。

    2026-09-15 发现：第一版按「报告期 + 15 天」对齐，而 99.7% 的公告晚于
    这个日子（中位 50 天）。模型提前一个多月看到户数变化，是前视偏差。
    """
    print("\n[股东人数·按公告日对齐]")
    import features as F
    dates = pd.date_range("2025-07-01", "2025-09-30", freq="B").strftime("%Y-%m-%d")
    panel = pd.DataFrame({"date": list(dates) * 2,
                          "code": ["000001"] * len(dates) + ["000002"] * len(dates),
                          "outstanding_share": 1e9})
    holders = pd.DataFrame({
        "代码": ["000001", "000002"], "报告期": ["20250630", "20250630"],
        "公告日期": ["2025-08-25", None],       # 第二只没有公告日：兜底 +120 天
        "股东户数-增减比例": [-5.0, 3.0], "股东户数-本次": [50000, 80000]})
    out = F.holder_features(panel, holders)
    a = out[out.code == "000001"].set_index("date")
    ck(np.isnan(a.loc["2025-08-22", "gdhs_chg1"]),
       "公告前一交易日（08-22）户数变化还看不到")
    ck(np.isnan(a.loc["2025-07-15", "gdhs_chg1"]),
       "报告期 + 15 天（07-15）也看不到 —— 那是旧口径的偷看点")
    ck(abs(a.loc["2025-08-25", "gdhs_chg1"] - (-0.05)) < 1e-9,
       "公告当天（08-25）起可用，值是 -5%")
    ck(abs(a.loc["2025-08-25", "gdhs_level"] - 5e-5) < 1e-12,
       "散户密度 = 户数 / 流通股本")
    ck(a.loc["2025-08-29", "gdhs_stale_days"] == 4, "数据年龄从公告日起算")
    b = out[out.code == "000002"].set_index("date")
    ck(np.isnan(b.loc["2025-09-30", "gdhs_chg1"]),
       "没有公告日期的按报告期 + 120 天（10-28）兜底，9 月底仍不可用")



def check_chip_causal() -> None:
    """筹码特征只能用过去：截到 t 算和用全序列算，前 t 行必须逐位相同。

    第一版用整段序列的 min/max 定价格网格，未来的最高价决定格宽，
    之前每一天的特征都被它扰动过（2026-09-15 审计）。

    ×12 是故意的：后段涨过 10 倍会触发网格外接（2026-09-16 起），
    外接如果没写成「只由 [..t] 决定」，前 500 行立刻和截断版对不上。
    """
    print("\n[筹码·只用过去]")
    from chips import chip_features
    d = synth(700, seed=9)
    # 后 200 天人为拉一波 12 倍的行情，让「全序列的最高价」冲出起始网格
    d.loc[d.index[-200:], ["high", "low", "close"]] *= 12.0
    h, l, c, t = (d[k].to_numpy() for k in ("high", "low", "close", "turnover"))
    k = 500
    full = chip_features(h, l, c, t).iloc[:k].to_numpy()
    part = chip_features(h[:k], l[:k], c[:k], t[:k]).to_numpy()
    diff = float(np.nanmax(np.abs(full - part)))
    ck(diff < 1e-12, f"截到 t 与全序列算出的前 t 行逐位相同（最大差 {diff:.2e}）")


def check_send_roundtrip() -> None:
    """清单经 to_json -> read_json 往返，代码不能丢前导零。"""
    print("\n[清单往返·代码前导零]")
    import tempfile
    a = pd.DataFrame({"code": ["002652", "000523", "688655"],
                      "name": ["扬子新材", "红棉股份", "迅捷兴"],
                      "score": [99.0, 98.0, 97.0], "streak": [3, 1, 5],
                      "close": [4.94, 4.36, 60.12], "board": ["main", "main", "star"]})
    tmp = Path(tempfile.mkdtemp(prefix="bk_")) / "list_a.json"
    a.to_json(tmp, orient="records", force_ascii=False, indent=2)
    naive = pd.read_json(tmp)
    ck(str(naive["code"].iloc[0]) != "002652",
       "不给 dtype 的 read_json 确实会把 002652 读成 2652（这条钉住的是坑本身）")
    b = pd.read_json(tmp, dtype={"code": str})
    b["code"] = b["code"].astype(str).str.zfill(6)
    ck(b["code"].tolist() == ["002652", "000523", "688655"],
       "daily.stage_send 的读法保住前导零")


def check_streak_calendar() -> None:
    """连续天数按交易日回数：跨过法定假日不归零。"""
    print("\n[连续天数·跨假日]")
    import tempfile
    import daily as D
    import datasource as ds
    cal = {"2026-09-22", "2026-09-23", "2026-09-24", "2026-09-28", "2026-09-29"}  # 25 日休市
    orig_td, orig_data = ds.trade_dates, D.DATA
    D.DATA = Path(tempfile.mkdtemp(prefix="streak_"))
    ds.trade_dates = lambda: cal
    D._CAL.clear()                       # 进程内缓存，别让别的用例的日历漏进来
    try:
        for day in ("2026-09-23", "2026-09-24"):
            (D.DATA / day[:7]).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"code": ["600000"]}).to_parquet(
                D.DATA / day[:7] / f"breakout_{day}.parquet", index=False)
        ck(D.prev_trade_days("2026-09-28", 3) == ["2026-09-24", "2026-09-23", "2026-09-22"],
           "09-28 往回数三个交易日跳过休市的 09-25")
        ck(D.count_streak("600000", "2026-09-28") == 2,
           "09-23、09-24 连续上榜，隔着 09-25 假日，09-28 的连续天数是 2（不归零）")
        ck(D.count_streak("600000", "2026-09-24") == 1, "09-24 往前只有 09-23 一天")
    finally:
        ds.trade_dates, D.DATA = orig_td, orig_data
        D._CAL.clear()


def check_pool_expiry_calendar() -> None:
    """A 池的有效期是 60 个**交易日**，不是 60×1.47 个自然日（S22）。

    旧写法 `date - int(60*1.47) 天` 在 2010~2026 的真日历上给的是 51~64 个
    交易日（均值 58.2，恰好 60 的只有 9.0%）：春节前上榜的票少留 8~9 天，
    6 月上榜的多留 4 天。清单 B 只从 A 池里出，池子提前空了就少发、晚清了
    就多发，两种都没有日志、没有报错。教训 29 的同一个毛病。
    """
    print("\n[A 池·有效期按交易日]")
    import datetime as _dt
    import daily as D
    import datasource as ds
    # 合成日历：2026-01-05~07-31 的工作日，挖掉 02-16~02-24（7 个工作日）当春节
    cal = [d for d in pd.date_range("2026-01-05", "2026-07-31", freq="B")
           .strftime("%Y-%m-%d") if not ("2026-02-16" <= d <= "2026-02-24")]
    orig_td = ds.trade_dates
    ds.trade_dates = lambda: set(cal)
    D._CAL.clear()                       # 别让别的用例的日历漏进来
    try:
        last = "2026-02-13"              # 春节前最后一个交易日上的榜
        after = [d for d in cal if d > last]
        d60, d61 = after[D.POOL_DAYS - 1], after[D.POOL_DAYS]
        ck(d60 == "2026-05-19" and d61 == "2026-05-20"
           and len([d for d in cal if last < d <= d60]) == D.POOL_DAYS,
           f"合成日历自校验：{last} 之后第 {D.POOL_DAYS} 个交易日是 {d60}")
        old_cut = (_dt.date.fromisoformat(d60)
                   - _dt.timedelta(days=int(D.POOL_DAYS * 1.47))).isoformat()
        ck(old_cut > last,
           f"旧的 88 自然日近似在 {d60} 这天的界线是 {old_cut}，"
           f"比 {last} 还晚 —— 它会把这只票提前清掉")
        pool = {"600000": {"first": last, "last": last, "best": 98.0,
                           "days": 1, "streak": 1, "name": "x"}}
        empty = pd.DataFrame({"code": pd.Series(dtype=str),
                              "score": pd.Series(dtype=float),
                              "streak": pd.Series(dtype=int),
                              "name": pd.Series(dtype=str)})
        ck("600000" in D.pool_step(pool, empty, d60),
           f"last 之后第 {D.POOL_DAYS} 个交易日（{d60}）仍在池，跨春节不提前过期")
        ck("600000" not in D.pool_step(pool, empty, d61),
           f"第 {D.POOL_DAYS + 1} 个交易日（{d61}）清出")
        # 当天又上榜就续命：清理在记账之后，别把刚上榜的票顺手清掉
        again = D.pool_step(pool, pd.DataFrame(
            {"code": ["600000"], "score": [float(D.SCORE_B)],
             "streak": [1], "name": ["x"]}), d61)
        ck("600000" in again and again["600000"]["days"] == 2,
           "第 61 个交易日当天又上榜就续命，上榜天数累加")
    finally:
        ds.trade_dates = orig_td
        D._CAL.clear()


def check_merge_authority() -> None:
    """重拉分片对它的代码整段权威：旧分片里该票的更早行也要丢掉。"""
    print("\n[分片合并·重拉整段替换]")
    import tempfile
    import backfill as B
    raw, out = B.RAW, B.OUT
    tmp = Path(tempfile.mkdtemp(prefix="merge_"))
    B.RAW, B.OUT = tmp / "raw", tmp
    B.RAW.mkdir()
    try:
        base = pd.DataFrame({"code": ["600000"] * 3 + ["600001"] * 3,
                             "date": ["2026-01-01", "2026-01-02", "2026-01-03"] * 2,
                             "close": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]})
        base.to_parquet(B.RAW / "sina_0000.parquet", index=False)
        upd = pd.DataFrame({"code": ["600000", "600001"], "date": ["2026-01-04"] * 2,
                            "close": [11.0, 6.0]})
        upd.to_parquet(B.RAW / "sina_9_20260104000000_upd.parquet", index=False)
        # 600000 除权重拉：整段只有后两天，价格减半
        ref = pd.DataFrame({"code": ["600000"] * 2, "date": ["2026-01-03", "2026-01-04"],
                            "close": [5.0, 5.5]})
        ref.to_parquet(B.RAW / "sina_9_20260105000000_ref.parquet", index=False)
        ck(B.merge_daily(pattern="sina_*.parquet") == 0, "合并成功")
        m = pd.read_parquet(B.OUT / "daily.parquet")
        a = m[m.code == "600000"].sort_values("date")
        ck(a["date"].tolist() == ["2026-01-03", "2026-01-04"]
           and a["close"].tolist() == [5.0, 5.5],
           "重拉的票只剩重拉分片的行（旧的 01-01/01-02 没有残留）")
        b = m[m.code == "600001"].sort_values("date")
        ck(len(b) == 4 and b["close"].iloc[-1] == 6.0, "没重拉的票旧行 + 增量行都在")

        # HIST_START 截断 + 零量占位行：老分片里的脏行在合并时就该清掉
        extra = pd.DataFrame({"code": ["600001"] * 2,
                              "date": ["2022-12-30", "2026-01-02"],
                              "close": [4.0, 5.0], "volume": [1e6, 0.0]})
        extra.to_parquet(B.RAW / "sina_0001.parquet", index=False)
        B.merge_daily(pattern="sina_*.parquet")
        m = pd.read_parquet(B.OUT / "daily.parquet")
        ck("2022-12-30" not in set(m["date"]),
           f"{B.HIST_START} 之前的行不进表")
        ck(not len(m[(m.code == "600001") & (m.date == "2026-01-02")
                     & (m.volume == 0)]),
           "零成交量的停牌占位行在合并时被清掉")

        # 起点不一致要出声：除权重拉的票一路补到 HIST_START，没重拉的停在原处
        warned = []
        ow = B.log.warning
        B.log.warning = lambda *a, **k: warned.append(a[0] if a else "")
        try:
            base2 = pd.DataFrame({
                "code": ["600000"] * 4 + [f"6000{i:02d}" for i in range(10, 14)],
                "date": ["2023-01-03", "2023-06-01", "2023-06-02", "2023-06-05"]
                        + ["2023-06-05"] * 4,
                "close": [10.0] * 8})
            for p in B.RAW.glob("sina_*.parquet"):
                p.unlink()
            base2.to_parquet(B.RAW / "sina_0000.parquet", index=False)
            B.merge_daily(pattern="sina_*.parquet")
        finally:
            B.log.warning = ow
        ck(any("起点不一致" in str(w) for w in warned),
           "最早日只有一只、最末日五只时会 warning（提示跑 refresh 对齐）")
    finally:
        B.RAW, B.OUT = raw, out

# ---------------------------------------------------------------
#  增量补数据（backfill.stage_update）的几条闸
#  全部离线：代码表取自 cache/codes.csv（教训 2：测试对象不能凭记忆编），
#  行情、日历、重拉、股东人数全换成假实现，产物只写 tempfile
# ---------------------------------------------------------------
def _real_codes(n: int) -> list[str]:
    p = ROOT / "cache" / "codes.csv"
    if p.exists():
        cs = sorted(set(pd.read_csv(p, dtype=str)["code"].astype(str).tolist()))
        if len(cs) >= n:
            return cs[:n]
    return [f"{600000 + i:06d}" for i in range(n)]


def _star_code() -> str:
    """一只真实的科创板代码。教训 2：测试对象从代码表里取，不许凭记忆写。"""
    p = ROOT / "cache" / "codes.csv"
    if p.exists():
        cs = sorted(c for c in pd.read_csv(p, dtype=str)["code"].astype(str)
                    if c.startswith("688"))
        if cs:
            return cs[0]
    return "688001"


def _quote(code: str, price: float, prev_close: float, vol_hand: float,
           turn: str, ts: str = "20260916150000"):
    """腾讯快照的一行。raw[33]/[34]/[38] = 最高/最低/换手率（百分数）。"""
    import datasource as ds
    raw = [""] * 40
    raw[33] = str(price * 1.02)
    raw[34] = str(price * 0.98)
    raw[38] = turn
    return ds.Quote(code=code, market=ds.to_symbol(code)[:2], name="测试",
                    price=price, prev_close=prev_close, open_=price,
                    volume_hand=vol_hand, amount_wan=price * vol_hand,
                    ts=ts, raw=raw)


def _fake_daily(cs, dates, close: float = 10.0, os_: float = 1e8):
    rows = []
    for c in cs:
        for d in dates:
            rows.append({"code": c, "date": d, "open": close,
                         "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": 1e6,
                         "amount": close * 1e6, "outstanding_share": os_,
                         "turnover": 0.01})
    return pd.DataFrame(rows)


def _offline_update(tmp, daily_df, quote_fn, cs, target, tds,
                    ref_result=None):
    """把 backfill 的目录和所有联网入口换成离线假实现。返回 (cap, restore)。"""
    import backfill as B
    import datasource as ds
    orig = (B.RAW, B.OUT, B.STATE, B.codes, B.last_closed_trade_day,
            B.refetch_codes, B.stage_holders, ds.fetch_quotes, ds.trade_dates)
    B.RAW, B.OUT, B.STATE = tmp / "raw", tmp, tmp / "state"
    B.RAW.mkdir(parents=True, exist_ok=True)
    daily_df.to_parquet(B.OUT / "daily.parquet", index=False)
    daily_df.to_parquet(B.RAW / "sina_0000.parquet", index=False)
    cap = {"ref": [], "fetch": 0}

    def _ref(codes_, target=None):
        cap["ref"] += list(codes_)
        if ref_result is not None:
            return ref_result
        return list(codes_), []

    def _fq(syms):
        cap["fetch"] += 1
        return quote_fn(list(syms))

    B.codes = lambda include_bj=True: list(cs)
    B.last_closed_trade_day = lambda *a, **k: target
    B.refetch_codes = _ref
    B.stage_holders = lambda: 0
    ds.fetch_quotes = _fq
    ds.trade_dates = lambda: set(tds)

    def restore():
        (B.RAW, B.OUT, B.STATE, B.codes, B.last_closed_trade_day,
         B.refetch_codes, B.stage_holders, ds.fetch_quotes,
         ds.trade_dates) = orig
    return cap, restore


def check_update_share_drift() -> None:
    """解禁/增发不触发除权判据，股本会一直沿用上一根新浪 K 线的值。

    daily.parquet 实测：股本跳升 ≥1.25 倍的 3649 次里 3624 次价格不变，
    昨收判据一个都接不住；2026-09-16 实测 13 只票的 turnover 偏离 >5%
    （最大 2.28 倍），而 turnover 喂给筹码衰减和 turn_pct/turn_std20。
    """
    print("\n[增量·流通股本漂移]")
    import ast
    import inspect
    import tempfile
    import backfill as B
    cs = _real_codes(1100)                      # 过 len(rows) >= 80% 的地板
    a, b, c_ok, d_low, e_small = cs[0], cs[1], cs[2], cs[3], cs[4]
    star = _star_code()                         # 科创板：快照的成交量是股不是手
    cs = cs + [star]
    tmp = Path(tempfile.mkdtemp(prefix="drift_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])
    target, tds = "2026-09-16", ["2026-09-14", "2026-09-15", "2026-09-16"]

    def qf(syms):
        out = {}
        for s in syms:
            code = s[2:]
            if code == a:            # 股本真涨 2 倍
                qt = _quote(code, 10.0, 10.0, 1e5, "5.00")
            elif code == b:          # 股本真涨 14.59 倍（IPO 解禁那种）
                qt = _quote(code, 10.0, 10.0, 1e5, "0.685")
            elif code == d_low:      # 低换手：两位小数的量化误差本身就 >5%
                qt = _quote(code, 10.0, 10.0, 149, "0.01")
            elif code == e_small:    # 股本只涨 6%（股权激励行权那种小变动）
                qt = _quote(code, 10.0, 10.0, 21200, "2.00")
            elif code == star:       # 688 的 volume_hand 已经是股，股本同时涨 1.5 倍
                qt = _quote(code, 10.0, 10.0, 1.5e6, "1.00")
            else:                    # 股本没变：turn = 1e7/1e8 = 10%
                qt = _quote(code, 10.0, 10.0, 1e5, "10.00")
            out[s] = qt
        return out

    cap, restore = _offline_update(tmp, daily, qf, cs, target, tds)
    try:
        rc = B.stage_update()
        m = pd.read_parquet(B.OUT / "daily.parquet")
        t = m[m["date"] == target].set_index("code")
        ck(rc == 0, "补数据成功")
        ck(abs(t.loc[a, "turnover"] - 0.05) < 1e-9
           and abs(t.loc[a, "outstanding_share"] / 2e8 - 1) < 1e-6
           and a in cap["ref"],
           f"股本涨 2 倍：换手 {t.loc[a, 'turnover']:.4f} 按交易所口径、"
           "股本按换手反推、进重拉名单")
        ck(abs(t.loc[b, "volume"] - 1e7) < 1
           and abs(t.loc[b, "turnover"] - 0.00685) < 1e-9,
           f"股本涨 14.59 倍：成交量单位没被判翻（{t.loc[b, 'volume']:.0f} 股）")
        ck(abs(t.loc[c_ok, "turnover"] - 0.10) < 1e-9
           and abs(t.loc[c_ok, "outstanding_share"] / 1e8 - 1) < 1e-9
           and c_ok not in cap["ref"],
           "股本没变的票：股本不动、不进重拉名单（否则天天全市场重拉）")
        ck(d_low not in cap["ref"],
           "换手 0.01% 的票不进重拉名单（死写 5% 阈值会天天误伤这类票）")
        ck(e_small in cap["ref"]
           and abs(t.loc[e_small, "outstanding_share"] / 1.06e8 - 1) < 1e-6,
           f"股本只涨 6% 也要抓住（容差 {B.OS_DRIFT_TOL:.0%} + 舍入项；"
           "10% 的固定阈值会放过这一档，而它天天在发生）")
        ck(abs(t.loc[star, "volume"] - 1.5e6) < 1
           and abs(t.loc[star, "turnover"] - 0.01) < 1e-9
           and star in cap["ref"],
           f"科创板股本漂 1.5 倍时单位仍判对（{t.loc[star, 'volume']:.0f} 股，"
           "不是 1.5 亿）且进重拉名单")
    finally:
        restore()

    # 接线：股本漂移这道检查在 _snap_row 里，stage_update 经 _scan_snapshot 调它。
    # 以后有人把它删成「只核对成交量单位」，这里立刻红（教训 11 的钉法）
    ss = inspect.getsource(B._snap_row)
    ck("os_q" in ss and "drift" in ss and "OS_DRIFT_TOL" in ss,
       "_snap_row 里有「按换手率反推股本 + OS_DRIFT_TOL 容差」这段")
    calls = [n.func.id for n in ast.walk(ast.parse(inspect.getsource(B._scan_snapshot)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    ck("_snap_row" in calls, "_scan_snapshot 真的调 _snap_row")
    ck("drift" in inspect.getsource(B._scan_snapshot)
       and "refetch" in inspect.getsource(B._scan_snapshot),
       "漂了的票进 refetch 名单（否则旧股本一直用下去）")


def check_update_coverage() -> None:
    """「覆盖到哪天」必须按覆盖完整的那天算，不是全表最大日期。"""
    print("\n[增量·覆盖率对账]")
    import json
    import tempfile
    import backfill as B
    cs = _real_codes(1100)
    target, tds = "2026-09-16", ["2026-09-14", "2026-09-15", "2026-09-16"]

    def mk_qf(frac: float):
        def qf(syms):
            keep = syms[:int(len(syms) * frac)]
            return {s: _quote(s[2:], 10.0, 10.0, 1e5, "10.00") for s in keep}
        return qf

    # A. 快照缺了几批：宁可不写也不写残表
    tmp = Path(tempfile.mkdtemp(prefix="cova_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])
    cap, restore = _offline_update(tmp, daily, mk_qf(0.70), cs, target, tds)
    try:
        rc = B.stage_update()
        mx = str(pd.read_parquet(B.OUT / "daily.parquet")["date"].max())
        ck(rc == 1, "快照缺 30% 时返回 1（不是照样写分片）")
        ck(list(B.RAW.glob("*_upd*.parquet")) == [], "一个增量分片都没写")
        ck(mx == "2026-09-15", "daily.parquet 没被动过")
    finally:
        restore()

    # B. 目标日只有 20 只有行：不能判成「已覆盖」
    tmp = Path(tempfile.mkdtemp(prefix="covb_"))
    daily = pd.concat([_fake_daily(cs, ["2026-09-14", "2026-09-15"]),
                       _fake_daily(cs[:20], [target])], ignore_index=True)
    cap, restore = _offline_update(tmp, daily, mk_qf(1.0), cs, target, tds)
    try:
        rc = B.stage_update()
        m = pd.read_parquet(B.OUT / "daily.parquet")
        ck(rc == 0 and cap["fetch"] > 0,
           "目标日只有 20/1100 只时仍然去取数（不是「不用追加」）")
        ck(m[m["date"] == target]["code"].nunique() == len(cs),
           f"跑完目标日有 {m[m['date'] == target]['code'].nunique()} 只（应 {len(cs)}）")
        ck(cap["ref"] == [],
           "已经有目标日行的那 20 只没被误判成除权（否则每轮重拉上千只）")
    finally:
        restore()

    # C. 正常：四个计数封闭、残差为 0
    tmp = Path(tempfile.mkdtemp(prefix="covc_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])
    cap, restore = _offline_update(tmp, daily, mk_qf(1.0), cs, target, tds)
    try:
        rc = B.stage_update()
        st = json.loads((B.STATE / "update_status.json").read_text("utf-8"))
        ck(rc == 0 and st["missing"] == 0, "正常时残差 0")
        ck(st["closed"] and st["appended"] + st["stale"] + st["newcodes"]
           + st["refetch_only"] + st["missing"] == st["requested"],
           "追加 + 无当日 + 无历史 + 重拉 + 没回 = 请求只数（再加一条静默 "
           "continue 这里立刻红）")
        ck(st["prev_n"] == len(cs) and st["covered"] == len(cs),
           "状态对象记下了「前一日几只、今天能覆盖几只」")
    finally:
        restore()

    # D. 快照全回来了，但两成的票时间戳还是上一日（盘中跑、行情源半挂）。
    #    这一档 missing=0、rows/len(cs)=0.82 过得了前两道闸，只有「和前一个
    #    交易日比」这道拦得住。放过去的话当天就在残缺池子里排百分位出清单，
    #    而且 hist_max 已到目标日，同一天重试也补不回来。
    tmp = Path(tempfile.mkdtemp(prefix="covd_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])

    def qf_stale(syms):
        out = {}
        for i, s in enumerate(syms):
            ts = "20260915150000" if i < int(len(syms) * 0.18) else "20260916150000"
            out[s] = _quote(s[2:], 10.0, 10.0, 1e5, "10.00", ts=ts)
        return out

    cap, restore = _offline_update(tmp, daily, qf_stale, cs, target, tds)
    try:
        rc = B.stage_update()
        st = json.loads((B.STATE / "update_status.json").read_text("utf-8"))
        mx = str(pd.read_parquet(B.OUT / "daily.parquet")["date"].max())
        ck(rc == 1 and st["ok"] is False,
           f"18% 的票没有当日数据时返回 1（覆盖 {st['covered']}/{st['prev_n']}）")
        ck(list(B.RAW.glob("*_upd*.parquet")) == [] and mx == "2026-09-15",
           "一行都没写进 daily.parquet（残缺的横截面不许出清单）")
        ck(st["missing"] == 0 and st["appended"] > 0.80 * len(cs),
           "拦住它的是覆盖率那道闸，不是「快照没回」也不是 80% 地板")
    finally:
        restore()


def check_update_gap() -> None:
    """漏采一天的票，平盘日昨收对得上，缺口会永久留下。"""
    print("\n[增量·逐票缺日]")
    import tempfile
    import backfill as B
    cs = _real_codes(1100)
    x, y = cs[0], cs[1]
    target, tds = "2026-09-14", ["2026-09-10", "2026-09-11", "2026-09-14"]
    tmp = Path(tempfile.mkdtemp(prefix="gap_"))
    full = _fake_daily([c for c in cs if c not in (x, y)],
                       ["2026-09-10", "2026-09-11"])
    part = _fake_daily([x, y], ["2026-09-10"])      # 两只都缺 09-11
    daily = pd.concat([full, part], ignore_index=True)

    def qf(syms):
        out = {}
        for s in syms:
            code = s[2:]
            # y 有涨跌（昨收对不上，原来的除权自愈能接住），x 平盘（接不住）
            prev = 10.6 if code == y else 10.0
            out[s] = _quote(code, 10.0, prev, 1e5, "10.00",
                            ts="20260914150000")
        return out

    cap, restore = _offline_update(tmp, daily, qf, cs, target, tds)
    try:
        rc = B.stage_update()
        m = pd.read_parquet(B.OUT / "daily.parquet")
        ck(x in cap["ref"], "平盘缺一天的票也要整段重拉（昨收比对看不见它）")
        ck(y in cap["ref"], "昨收对不上的票照旧重拉（除权自愈没被改坏）")
        ck(cs[5] not in cap["ref"] and len(cap["ref"]) <= 5,
           f"正常票不进重拉名单（本次 {len(cap['ref'])} 只）")
        ck(rc == 0 and target in set(m[m["code"] == cs[5]]["date"]),
           "正常票有目标日那根")
        ck(target in set(m[m["code"] == x]["date"]),
           "缺日的票当天这根照样写（重拉失败也不会又挖一个新洞）")
    finally:
        restore()


def check_refetch_short() -> None:
    """重拉「成功」只意味着拿到 ≥60 行，不意味着含目标日。"""
    print("\n[重拉·没拉到目标日]")
    import tempfile
    import backfill as B
    target = "2026-09-16"

    def fake(code):
        if code == "600002":
            return None
        end = "2026-09-15" if code == "600000" else target
        ds_ = pd.date_range("2026-01-01", end, freq="B").strftime("%Y-%m-%d")
        return pd.DataFrame({"code": code, "date": ds_, "close": 10.0})

    class _Pool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, it, chunksize=None):
            for c in it:
                yield c, fake(c)

    tmp = Path(tempfile.mkdtemp(prefix="short_"))
    raw, pool = B.RAW, B.ProcessPoolExecutor
    B.RAW = tmp
    B.ProcessPoolExecutor = lambda max_workers=None: _Pool()
    try:
        ok, short = B.refetch_codes(["600000", "600001", "600002"],
                                    target=target)
        ck(ok == ["600001"] and short == ["600000", "600002"],
           f"拉回来但不含目标日、以及整只失败，都要进 short（{short}）")
        ck(B.refetch_codes([], target=target) == ([], []), "空入参不炸")
    finally:
        B.RAW, B.ProcessPoolExecutor = raw, pool


def check_update_status() -> None:
    """除权票新浪没出当日 K 线时，要用快照补一根，并留下可查询的对象。"""
    print("\n[增量·状态对象与兜底行]")
    import json
    import tempfile
    import backfill as B
    cs = _real_codes(1200)
    xr = cs[0]
    target, tds = "2026-09-16", ["2026-09-14", "2026-09-15", "2026-09-16"]
    tmp = Path(tempfile.mkdtemp(prefix="stat_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])

    def qf(syms):
        return {s: _quote(s[2:], 10.0, 10.5 if s[2:] == xr else 10.0,
                          1e5, "10.00") for s in syms}

    cap, restore = _offline_update(tmp, daily, qf, cs, target, tds,
                                   ref_result=([], [cs[0]]))
    try:
        rc = B.stage_update()
        st = json.loads((B.STATE / "update_status.json").read_text("utf-8"))
        m = pd.read_parquet(B.OUT / "daily.parquet")
        ck(rc == 0 and st["short"] == [xr] and st["filled"] == 1,
           "update_status.json 记下了「没拉到目标日」和「补了几只」")
        ck(target in set(m[m["code"] == xr]["date"]),
           "除权票当天那一行被腾讯快照补上了（不会从候选池静默消失）")
        ck(set(m[m["date"] == target]["code"]) == set(m["code"]),
           "全局 max 到了目标日，不代表每只票都有那一行")
    finally:
        restore()


def check_refresh_rc() -> None:
    """股东人数失败不能把「日线已经补齐」拖成失败。"""
    print("\n[全量刷新·退出码]")
    import backfill as B
    orig = (B.stage_daily_sina, B.merge_daily, B.consolidate, B.stage_holders)
    B.stage_daily_sina = lambda: 0
    B.merge_daily = lambda pattern=None: 0
    B.consolidate = lambda: 0
    try:
        B.stage_holders = lambda: 1
        ck(B.stage_refresh() == 0, "日线补齐、只有股东人数失败时返回 0")

        def boom():
            raise RuntimeError("东财挂了")
        B.stage_holders = boom
        ck(B.stage_refresh() == 0, "股东人数抛异常也不阻断")
        B.stage_holders = lambda: 0
        B.stage_daily_sina = lambda: 1
        ck(B.stage_refresh() != 0, "日线本身失败仍然必须非 0")
    finally:
        (B.stage_daily_sina, B.merge_daily, B.consolidate,
         B.stage_holders) = orig


def check_holders_refresh() -> None:
    """股东人数表的新鲜度门槛：7 天 -> 20 小时（S5）。

    公告是成堆来的（2025-04-30 前 7 天 7123 条 = 全市场 131%），表龄 0~7 天
    意味着生产在那些天用的户数比训练行旧一整个披露期。
    """
    print("\n[股东人数·新鲜度门槛]")
    import inspect
    import os as _os
    import tempfile
    import backfill as B
    tmp = Path(tempfile.mkdtemp(prefix="hfresh_"))
    hp = tmp / "holders.parquet"
    ck(B.holders_stale(hp), "表根本不存在时要刷")
    pd.DataFrame({"报告期": ["20250331"]}).to_parquet(hp, index=False)
    old = time.time() - 25 * 3600
    _os.utime(hp, (old, old))
    ck(B.holders_stale(hp), "25 小时前的表要刷（旧口径 7 天不刷）")
    fresh = time.time() - 3600
    _os.utime(hp, (fresh, fresh))
    ck(not B.holders_stale(hp),
       "1 小时前刚刷过的不重复刷（同一晚每 30 分钟重试一次不会反复拉）")
    src = inspect.getsource(B.stage_update)
    ck("holders_stale(" in src and "7 * 86400" not in src,
       "stage_update 走 holders_stale，源码里没有残留的 7 天门槛")


def check_holders_partial() -> None:
    """掉一个报告期就用残缺表覆盖完整表，而且不会自愈。"""
    print("\n[股东人数·残缺不覆盖]")
    import datetime as dt
    import sys as _sys
    import tempfile
    import types
    import backfill as B
    tmp = Path(tempfile.mkdtemp(prefix="hold_"))
    old = pd.DataFrame({"代码": ["000001"] * 3,
                        "报告期": ["20250331", "20250630", "20250930"],
                        "股东户数-本次": [1, 2, 3]})
    old.to_parquet(tmp / "holders.parquet", index=False)

    class _NoSleep:
        def sleep(self, *a):
            pass

        def time(self):
            return time.time()

    fail = {"on": True}

    def gdhs(symbol):
        if fail["on"] and symbol == "20250630":
            raise RuntimeError("东财抖了")
        return pd.DataFrame({"代码": ["000001"], "股东户数-本次": [9]})

    fake_ak = types.ModuleType("akshare")
    fake_ak.stock_zh_a_gdhs = gdhs
    keep_ak = _sys.modules.get("akshare")
    _sys.modules["akshare"] = fake_ak
    out, nb, tm = B.OUT, B.now_bj, B.time
    B.OUT, B.now_bj, B.time = tmp, lambda: dt.datetime(2025, 10, 1), _NoSleep()
    try:
        ck(B.stage_holders() == 1, "丢了一个报告期要返回 1")
        back = pd.read_parquet(tmp / "holders.parquet")
        lost = back[back["报告期"].astype(str) == "20250630"]
        ck(len(lost) == 1 and int(lost["股东户数-本次"].iloc[0]) == 2,
           "没拉到的那一期沿用旧表的行（不是被残缺表整表覆盖）")
        got_ = back[back["报告期"].astype(str) == "20250930"]
        ck(len(got_) == 1 and int(got_["股东户数-本次"].iloc[0]) == 9,
           "拉到的期用这次的新数据（不是整表回退到旧表）")
        ck(back["报告期"].nunique() == 10,
           f"合并之后 10 个报告期一个不少（实际 {back['报告期'].nunique()}）")
        fail["on"] = False
        ck(B.stage_holders() == 0, "全部拿到时正常返回 0")
        back = pd.read_parquet(tmp / "holders.parquet")
        ck(back["报告期"].nunique() == 10,
           f"正常刷新写进去 10 个报告期（实际 {back['报告期'].nunique()}）")
    finally:
        B.OUT, B.now_bj, B.time = out, nb, tm
        if keep_ak is None:
            _sys.modules.pop("akshare", None)
        else:
            _sys.modules["akshare"] = keep_ak


def check_clean_bars() -> None:
    """停牌占位行（价照抄前一日、量为 0）和零价行必须进不了日线表。"""
    print("\n[日线·停牌占位行]")
    import backfill as B
    import features as F
    d = pd.DataFrame({
        "code": ["688089"] * 6,
        "date": [f"2024-11-0{i}" for i in range(1, 7)],
        "open": [20.0, 20.5, 0.0, 20.5, 21.0, 21.5],
        "high": [20.8, 21.0, 0.0, 21.0, 21.4, 22.0],
        "low": [19.8, 20.2, 0.0, 20.2, 20.6, 21.1],
        "close": [20.3, 20.75, 20.75, 20.75, 21.2, 21.8],
        "volume": [1e6, 1.1e6, 0.0, 0.0, 1.2e6, 1.3e6],
        "amount": [2e7, 2.2e7, 0.0, 0.0, 2.5e7, 2.8e7]})
    got = B.clean_bars(d)["date"].tolist()
    ck(got == ["2024-11-01", "2024-11-02", "2024-11-05", "2024-11-06"],
       f"零价行和零量占位行都被剔掉（剩 {got}）")
    ck(B.clean_bars(d)["close"].tolist() == [20.3, 20.75, 21.2, 21.8],
       "正常行原样保留")
    thin = pd.DataFrame({"code": ["600000"] * 2, "date": ["2026-01-01",
                                                         "2026-01-02"],
                         "close": [10.0, 11.0]})
    ck(len(B.clean_bars(thin)) == 2, "只有 code/date/close 的分片不被误删")

    # 特征侧：零价行当天 TR = |0 - 昨收|，atr_ratio 之后 14 天被抬高
    s = synth(200, seed=7)
    dirty = s.copy()
    i = dirty.index[100]
    dirty.loc[i, ["open", "high", "low"]] = 0.0
    dirty.loc[i, "volume"] = 0.0
    dirty.loc[i, "turnover"] = 0.0
    fd = F.per_stock(dirty)
    fc = F.per_stock(B.clean_bars(dirty).reset_index(drop=True))
    ck(bool((fd["atr_ratio"].iloc[101:114].to_numpy()
             > fc["atr_ratio"].iloc[100:113].to_numpy()).all()),
       "留着零价行会把之后 14 天的 atr_ratio 抬高")


def check_history_alignment() -> None:
    """训练表的行过滤要和生产口径一致：前 120 行只预热，薄横截面整天丢掉。"""
    print("\n[训练表·历史深度对齐]")
    import build as B
    long_c, short_c = _real_codes(3), _real_codes(40)[3:33]
    parts = []
    for k, c in enumerate(long_c):
        g = synth(400, seed=100 + k).assign(code=c)
        parts.append(g)
    for k, c in enumerate(short_c):
        g = synth(400, seed=200 + k).assign(code=c).iloc[100:]
        parts.append(g)
    daily = pd.concat(parts, ignore_index=True)
    daily["amount"] = daily["close"] * daily["volume"]
    daily["outstanding_share"] = 1e8
    daily = B.attach_turnover(daily)
    panel = B.assemble(daily)

    n = panel.groupby("date")["code"].nunique()
    ck(int(n.min()) >= 0.5 * n.median(),
       f"没有横截面不足半数的日期（最少 {int(n.min())} 只，中位 {n.median():.0f}）")
    bad = []
    for c, g in panel.groupby("code"):
        src = daily[daily["code"] == c].sort_values("date")
        if str(g["date"].min()) < str(src["date"].iloc[B.MIN_HIST]):
            bad.append(c)
    ck(not bad, f"每只票前 {B.MIN_HIST} 行只预热、不进表（越界 {bad[:3]}）")
    ck(bool(panel["chip_conc90"].notna().all()
            and panel["dist_52w_high"].notna().all()),
       "预热行没漏进来（筹码和 52 周位置全有值）")


def check_split_volume() -> None:
    """送转日成交股数按股本跳升而价格被复权除回去，量能会凭空放大。"""
    print("\n[量能·送转口径]")
    import features as F
    n = 120
    turn = np.full(n, 0.01)
    os_ = np.array([1e8] * 60 + [2e8] * 60)
    vol = turn * os_
    px = np.full(n, 20.0)
    px[:60] = 10.0                       # 前复权：送转前的价被除以 2
    d = pd.DataFrame({
        "code": ["000001"] * n,
        "date": pd.date_range("2025-01-01", periods=n, freq="B")
                  .strftime("%Y-%m-%d"),
        "open": px, "close": px, "high": px * 1.01, "low": px * 0.99,
        "volume": vol, "turnover": turn, "outstanding_share": os_,
        "amount": px * vol})
    f = F.per_stock(d)
    ck(np.allclose(f["vol_ratio20"].iloc[60:80], 1.0, atol=1e-6),
       "换手率不变的送转不产生假放量：vol_ratio20")
    ck(np.allclose(f["vol_compress"].iloc[60:80], 1.0, atol=1e-6),
       "…vol_compress")
    ck(np.allclose(f["vol_ratio5"].iloc[60:65], 1.0, atol=1e-6),
       "…vol_ratio5")
    ck(np.allclose(f["turn_pct"].iloc[40:80], 0.01),
       "turn_pct 本来就不受影响（对照）")
    # 整段成交量乘常数（单位换算）不改变任何量能特征
    d2 = d.copy()
    d2["volume"] = d2["volume"] * 100
    f2 = F.per_stock(d2)
    ck(np.allclose(f["vol_ratio20"].fillna(0), f2["vol_ratio20"].fillna(0)),
       "成交量整段乘常数时量能特征不变")


def check_float_mcap() -> None:
    """前复权价 × 真实股本 = 被复权因子压小的市值，跨除权日必须连续。"""
    print("\n[流通市值·复权口径]")
    import build as B
    n = 10
    close = [20.0] * n                       # qfq 价连续（10 送 10 之后都是 20）
    os_ = [1e8] * 5 + [2e8] * 5              # 除权日股本翻倍
    vol = [1e6] * n
    amt = [20 * 2 * 1e6] * 5 + [20 * 1e6] * 5   # 除权前真实均价是 qfq 价的 2 倍
    d = pd.DataFrame({"code": ["300260"] * n,
                      "date": [f"2023-06-{i + 1:02d}" for i in range(n)],
                      "open": close, "high": close, "low": close,
                      "close": close, "volume": vol, "amount": amt,
                      "outstanding_share": os_, "turnover": [0.01] * n})
    m = B.attach_turnover(d.copy())["float_mcap"].to_numpy(float)
    ck(0.95 < m[4] / m[5] < 1.05, f"跨除权日市值连续（比值 {m[4] / m[5]:.3f}）")
    ck(abs(m[0] / 4e9 - 1) < 0.02, "除权前按真实均价 40 元算，不是 qfq 的 20 元")
    d2 = d.copy()
    d2["amount"] = [20 * 1e6] * n
    m2 = B.attach_turnover(d2)["float_mcap"].to_numpy(float)
    ck(np.allclose(m2, np.array(os_) * 20.0, rtol=0.02),
       "没有除权的日子仍等于股本 × 收盘价")
    d3 = d.drop(columns=["amount"]).copy()
    d3.loc[3, "volume"] = 0
    m3 = B.attach_turnover(d3)["float_mcap"].to_numpy(float)
    ck(bool(np.isfinite(m3).all() and (m3 > 0).all()),
       "amount 缺失 / 零成交量时退回收盘价，不产生 NaN")
    # 停牌行（有 amount 但 volume=0）：VWAP 要除零，必须退回收盘价，
    # 不能落到 attach_turnover 末尾那句 fillna(amount * 50)
    d4 = d.copy()
    d4.loc[3, "volume"] = 0.0
    m4 = B.attach_turnover(d4)["float_mcap"].to_numpy(float)
    ck(abs(m4[3] - os_[3] * 20.0) < 1e-6,
       f"零成交量那天退回股本 × 收盘价（{m4[3]:.3e}，不是 NaN 也不是成交额 × 50）")


def check_pick_rule() -> None:
    """验收和生产必须是同一条规则，而且只有一份实现。"""
    print("\n[清单 A 规则·唯一实现]")
    import ast
    import daily as D
    import validate as V
    q = np.linspace(0, 1, 101)
    day = pd.DataFrame({"date": ["2025-06-02"] * 4,
                        "code": ["300001", "600000", "688001", "920001"],
                        "board": ["chinext", "main", "star", "bj"],
                        "y_up": [1.0, 0.0, 0.0, 0.0]})
    p = np.array([0.990, 0.985, 0.800, 0.400])
    sel = V.pick(day, p, q, score_min=0, cap=10)
    ck(sel["code"].iloc[0] == "600000",
       "板块系数真的参与排序：创业板原始分最高，校正后主板第一")
    # 门槛这条要和板块系数的**具体取值**解耦：以前这里写死 p=0.800 的科创票
    # 应该上榜，那是 star=1.27 时才成立的（0.8×1.27 过线）；2026-09-16 会诊把
    # star 改成 1.0 之后，这条断言红了，而规则本身一个字都没变。
    # 现在按当前系数反推「刚好过线 / 刚好不过线」，测的才是规则。
    adj = {b: float(D.BOARD_ADJ.get(b, 1.0))
           for b in ("chinext", "main", "star", "bj")}
    need = q[D.SCORE_MIN]            # 过线要的最小校正后预测值
    p2 = np.array([need * 0.5 / adj["chinext"],    # 差得远
                   need * 1.02 / adj["main"],      # 刚好过
                   need * 1.01 / adj["star"],      # 刚好过
                   need * 0.9 / adj["bj"]])        # 差一点
    sel = V.pick(day, p2, q)         # 生产门槛
    ck(len(sel) == 2 and set(sel["code"]) == {"600000", "688001"},
       f"≥{D.SCORE_MIN} 分才上（选中 {sel['code'].tolist()}）")
    ck(len(V.pick(day, p * 0.01, q)) == 0, "全场不够格时清单为空，不凑数")
    ck(len(V.pick(day, p, q, score_min=0, cap=2)) == 2, "截到 cap")
    elig = pd.Series([True, False, True, True], index=day.index)
    sel = V.pick(day, p, q, score_min=0, cap=10, eligible=elig)
    ck("600000" not in set(sel["code"]), "风险剔除在取前 N 之前生效")

    # 旧口径必须能逐行复算（旧实验的成绩还要对得上）
    rng = np.random.default_rng(4)
    many = pd.DataFrame({
        "date": np.repeat(["2025-06-02", "2025-06-03"], 50),
        "code": [f"{i:06d}" for i in range(100)],
        "board": ["main"] * 100,
        "y_up": rng.random(100).round()})
    pr = rng.random(100)
    old = V.daily_topn(many, pr, 10, "y_up")
    new = V.pick_days(many, pr, q, score_min=0, cap=10, board_adj={})
    ck(old["code"].tolist() == new["code"].tolist(),
       "score_min=0 且不校正时逐行退化成 daily_topn")

    # 空仓月 != 零命中月
    bm = pd.DataFrame([{"month": "2025-03", "n_pick": 0, "hit": np.nan,
                        "base": 0.03, "lift": np.nan},
                       {"month": "2025-04", "n_pick": 10, "hit": 0.0,
                        "base": 0.03, "lift": 0.0},
                       {"month": "2025-05", "n_pick": 10, "hit": 0.2,
                        "base": 0.03, "lift": 6.7}])
    acc = V.acceptance(bm, many.assign(month="2025-04"), many)
    ck(acc["zero_months"] == 1 and acc["empty_months"] == 1,
       f"空仓月记 empty 不记 zero（zero={acc['zero_months']}，"
       f"empty={acc['empty_months']}）")

    # 走向前真的走 pick：用桩模型跑一遍，不碰 LightGBM
    rng2 = np.random.default_rng(12)
    dates = pd.date_range("2024-11-01", "2025-03-31", freq="B").strftime("%Y-%m-%d")
    nc = 100
    wf = pd.DataFrame({"date": np.repeat(dates, nc),
                       "code": np.tile([f"{i:06d}" for i in range(nc)], len(dates)),
                       "board": "main"})
    wf["f0__last"] = rng2.normal(size=len(wf))
    wf["_p"] = rng2.random(len(wf))
    wf["y_up"] = (rng2.random(len(wf)) < 0.05).astype(float)

    class _Stub:
        def fit(self, d, feats, y):
            return self

        def predict_proba(self, d):
            return d["_p"].to_numpy(float)

    bm, ex = V.walk_forward(wf, ["f0__last"], _Stub, start="2025-03-01",
                            end="2025-04-01", top_n=10, score_min=D.SCORE_MIN,
                            board_adj={})
    pk = ex["picks"]
    ck(len(bm) == 1 and bool((pk["score"] >= D.SCORE_MIN).all()),
       f"走向前的选票全部够 {D.SCORE_MIN} 分（n={len(pk)}）")
    ck(int(pk.groupby("date").size().max()) <= 10, "每天不超过 CAP")
    _, ex0 = V.walk_forward(wf, ["f0__last"], _Stub, start="2025-03-01",
                            end="2025-04-01", top_n=10, score_min=0,
                            board_adj={})
    ck(bool((ex0["picks"].groupby("date").size() == 10).all()),
       "score_min=0 时退化成每天固定 10 只（旧口径可复算）")

    # 接线：arena 必须把生产常量传进 walk_forward（教训 11 的钉法）
    src = (ROOT / "src" / "breakout" / "arena.py").read_text(encoding="utf-8")
    call = [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "walk_forward"]
    kw = {k.arg: ast.dump(k.value) for c in call for k in c.keywords}
    ck("score_min" in kw and "SCORE_MIN" in kw["score_min"]
       and "'D'" in kw["score_min"], "arena 传 score_min=D.SCORE_MIN")
    ck("board_adj" in kw and "BOARD_ADJ" in kw["board_adj"]
       and "'D'" in kw["board_adj"], "arena 传 board_adj=D.BOARD_ADJ")
    vsrc = (ROOT / "src" / "breakout" / "validate.py").read_text(encoding="utf-8")
    ck("97" not in vsrc.split("def pick")[1].split("def summarize")[0],
       "validate 里没有手抄的 97 分门槛")
    ck("score_min" in V.walk_forward.__code__.co_varnames,
       "walk_forward 收 score_min")


def _day_fixture(n_hi: int, n_lo: int = 0):
    """n_hi 只够格 + n_lo 只差两分不够格的一天。返回 (day, proba, q)。

    分位点直接用 0,0.01,..,1：预测值 i/100 的分数就是 i，构造得出的分数
    一目了然。_p 严格降序、代码唯一，名次和构造顺序一致（第 k 名的代码是
    600000+k-1），这样断言里可以直接点名「第 61 名」。
    """
    import daily as D
    q = np.linspace(0, 1, 101)
    hi = [(D.SCORE_MIN + 1) / 100 - i * 1e-5 for i in range(n_hi)]
    lo = [(D.SCORE_MIN - 2) / 100 - i * 1e-5 for i in range(n_lo)]
    p = np.array(hi + lo)
    n = n_hi + n_lo
    day = pd.DataFrame({"date": ["2026-09-16"] * n,
                        "code": [f"{600000 + i:06d}" for i in range(n)],
                        "y_up": [0.0] * n})
    return day, p, q


def check_risk_scope() -> None:
    """风险剔除查的是**全部够格**的票，不是名次靠前的 60 只（S23），
    而且够格总数要落进产物（S7）。

    以前是 `today.head(60)` 送进 risk_filter，第 61 名起 reject 一律空串 ——
    「没查」和「查过没问题」不可区分，前 60 里剔掉 ≥51 只时（CAP_A=10）
    就会有一只从没查过 ST/减持/解禁的票上榜。同时 len(bad) 把 97 分以下、
    本来就上不了榜的票也数进「风险剔除 N 只」，面板和邮件的数字虚高。
    """
    print("\n[清单 A·风险检查范围 + 够格计数]")
    import ast
    import daily as D
    import export as E

    # 先验一遍夹具：够格的真够格、不够格的真不够格（教训 26：默认值绕开规则）
    day, p, q = _day_fixture(8, 72)
    sc = D.to_score(p, q)
    ck(bool((sc[:8] >= D.SCORE_MIN).all()) and bool((sc[8:] < D.SCORE_MIN).all()),
       f"夹具：前 8 只 {sc[0]:.0f} 分够格，后 72 只 {sc[8]:.0f} 分不够格")

    # a) 80 只全够格：查过的必须覆盖上榜的，第 61 名不能靠「没查」混上来
    day, p, q = _day_fixture(80)
    reject = {f"{600000 + i:06d}" for i in range(51)} | {"600060"}   # 前 51 名 + 第 61 名
    seen: list[list[str]] = []

    def risk_fn(codes):
        seen.append(list(codes))
        return {c: "假 ST" for c in codes if c in reject}

    picks, bad, n_q, n_ok = D.select_a(day, p, q, risk_fn=risk_fn)
    got = set(seen[0]) if seen else set()
    ck(len(seen) == 1 and len(got) == 80,
       f"80 只够格的票全部送进风险检查（实际送了 {len(got)} 只）")
    ck(set(picks["code"]).issubset(got),
       "上榜的每一只都查过：不存在「没查」冒充「查过没问题」")
    ck("600060" not in set(picks["code"]) and len(picks) == D.CAP_A,
       f"前 51 名被剔后第 61 名补位到清单边缘，它是 ST 所以不在榜"
       f"（清单 {picks['code'].tolist()}）")
    ck(n_q == 80 and len(bad) == 52 and n_ok == 28,
       f"够格 {n_q}、剔除 {len(bad)}、本可上榜 {n_ok}")

    # b) 计数口径：不够格的票既不送检，也不算进「风险剔除 N 只」
    day, p, q = _day_fixture(8, 72)
    seen2: list[str] = []

    def risk_fn2(codes):
        seen2.extend(codes)
        return {c: "假 ST" for c in codes if c == "600029"}   # 第 30 名，不够格

    picks, bad, n_q, n_ok = D.select_a(day, p, q, risk_fn=risk_fn2)
    ck(len(seen2) == 8 and "600029" not in seen2,
       f"只送够格的 8 只去查，第 30 名（不够格）不送（实际送 {len(seen2)} 只）")
    ck(len(bad) == 0 and n_q == 8 and n_ok == 8 and len(picks) == 8,
       f"「风险剔除」只数够格的：bad={len(bad)}，清单 {len(picks)} 只")

    # c) 接线：两处入口都走 select_a，没有第二份「前 60 名」的副本
    for rel in ("src/breakout/daily.py", "tools/rerun_breakout.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        used = ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
        ck("RISK_SCAN_N" not in used, f"{rel} 不再按名次截风险检查范围")
        ck(not [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "head"
                and n.args and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == 60], f"{rel} 里没有 head(60)")
        calls = {n.func.attr if isinstance(n.func, ast.Attribute)
                 else getattr(n.func, "id", "")
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        ck("select_a" in calls, f"{rel} 走 select_a 这一份实现")
    dsrc = (ROOT / "src" / "breakout" / "daily.py").read_text(encoding="utf-8")
    scan = next(f for f in ast.walk(ast.parse(dsrc))
                if isinstance(f, ast.FunctionDef) and f.name == "stage_scan")
    scalls = {n.func.attr if isinstance(n.func, ast.Attribute)
              else getattr(n.func, "id", "")
              for n in ast.walk(scan) if isinstance(n, ast.Call)}
    ck("select_a" in scalls and "risk_filter" not in scalls
       and "pick" not in scalls,
       "stage_scan 只调 select_a，不自己调 V.pick / risk_filter")
    rsrc = (ROOT / "tools" / "rerun_breakout.py").read_text(encoding="utf-8")
    rcalls = {n.func.attr if isinstance(n.func, ast.Attribute)
              else getattr(n.func, "id", "")
              for n in ast.walk(ast.parse(rsrc)) if isinstance(n, ast.Call)}
    ck("pick" not in rcalls, "rerun 里那份 V.pick 副本已经消掉")

    # d) 够格总数进产物（S7）：日志、run_meta、清单 parquet 三处
    ck('"n_qualified": n_q' in dsrc, "run_meta 落盘 n_qualified")
    ck(any("n_qualified" in [k.arg for k in c.keywords]
           for c in ast.walk(ast.parse(dsrc))
           if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
           and c.func.attr == "assign"),
       "清单 parquet 每行带 n_qualified（历史回放也看得到）")
    ck("只够格" not in dsrc,
       "日志不再拿截断后的只数冒充「够格 N 只」")

    # e) 抬头要印出来：满员日和刚好够 10 只的日子不能长得一模一样
    a = pd.DataFrame({"code": ["600000"], "name": ["测试"], "score": [98.0],
                      "streak": [1], "close": [10.0]})
    b = pd.DataFrame(columns=["code", "name", "best", "days", "streak",
                              "close", "rise", "drop"])
    h = E._day_block("2026-09-16", a, b,
                     {"score_min": 97, "cap_a": 10, "n_qualified": 37})
    ck("共 37 只" in h and "截断" in h,
       "抬头印「≥97 分共 37 只（已按前 10 名截断）」")
    h2 = E._day_block("2026-09-16", a, b,
                      {"score_min": 97, "cap_a": 10, "n_qualified": 6})
    ck("共 6 只" in h2 and "截断" not in h2, "没到上限就不说截断")
    h3 = E._day_block("2026-09-16", a, b, {"score_min": 97, "cap_a": 10})
    ck("共" not in h3.split("</div>")[0],
       "2026-09-16 之前的清单没有这个数，抬头就不印（不编个 0 出来）")

    # f) 上限主导那些天的成绩单独报，别拿总平均背书
    html = E._body("2026-09-16", a, b, {}, False)
    ck(f"{E.CAP_PERF[0]:.1f}%" in html and f"{E.CAP_PERF[1]} 个名额" in html,
       f"邮件里印了满员日的准确率 {E.CAP_PERF[0]:.1f}%（{E.CAP_PERF[1]} 个名额）")
    ck(E.CAP_PERF[0] < E.perf_row(1)[1],
       f"满员日 {E.CAP_PERF[0]}% 低于全部上榜的 {E.perf_row(1)[1]}%")
    import json
    wg = ROOT / "out_breakout" / "window_grid.json"
    w5c = [r for r in (json.loads(wg.read_text(encoding="utf-8"))["grid"]
                       if wg.exists() else [])
           if r.get("kind") == "W5c" and "满10只" in str(r.get("label", ""))]
    if w5c:
        r = w5c[0]
        ck(abs(100 * r["hit"] - E.CAP_PERF[0]) < 0.06
           and int(r["n"]) == E.CAP_PERF[1],
           f"CAP_PERF 和 window_grid 的 W5c 行一致（{100 * r['hit']:.2f}%/{r['n']}）")
    else:
        ck(True, "window_grid 还没有 W5c 行（exp_window 待补），"
                 "CAP_PERF 暂按逐月滚动缓存实算的 11.36%/220 填")


def check_board_factors() -> None:
    """板块系数的拟合窗口不能和评估窗口重叠（否则成绩是样本内的）。"""
    print("\n[板块校正·不许偷看]")
    import exp_window as EW
    picks = pd.DataFrame({
        "date": ["2025-03-05"] * 3 + ["2025-04-07"] * 3,
        "board": ["main", "bj", "main", "bj", "bj", "main"],
        "y_up": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]})
    f1 = EW.board_factors(picks, "2025-04", min_n=1, min_board=1)
    picks2 = picks.copy()
    picks2.loc[picks2["date"] == "2025-04-07", "y_up"] = 1.0
    f2 = EW.board_factors(picks2, "2025-04", min_n=1, min_board=1)
    ck(f1 == f2 and f1.get("bj", 1.0) == 0.0,
       f"因子只由 2025-04 之前的行决定（{f1} vs {f2}）")
    ck(EW.board_factors(picks, "2025-03", min_n=1, min_board=1) == {},
       "第一个月没有历史，不校正")
    ck(EW.board_factors(picks, "2025-05") == {},
       "样本不足时返回空（宁可不校正也不用噪音因子）")

    dates = ["2025-03-03", "2025-12-31"]
    pay = EW.grid_payload(207, 0.0293, 5, [], dates, "fixed",
                          dict(EW.D.BOARD_ADJ))
    for k in ("board_adj_applied", "adj_fit_window", "eval_window"):
        ck(k in pay, f"产物声明了 {k}")
    ck(pay["adj_inconsistent"] is True,
       "固定因子拟合于 2025-03..12、又在同段评估 -> 标成样本内")
    ck(EW.grid_payload(207, 0.0293, 5, [], dates, "none", {})
       ["adj_inconsistent"] is False, "不校正时没有这个问题")


def check_cross_section() -> None:
    """横截面百分位化必须抹掉「今天大盘好」这个信息。"""
    print("\n[横截面百分位·对抗时间聚集的主力]")
    import features as F
    rng = np.random.default_rng(5)
    rows = []
    for di, d in enumerate(["2024-09-24", "2024-09-25", "2025-06-02"]):
        # 前两天是普涨日（整体水平高 10 倍），第三天是平常日
        base = 10.0 if di < 2 else 1.0
        for k in range(400):
            rows.append({"date": d, "code": f"{k:06d}",
                         "vol_ratio5": base * abs(rng.normal(1, .3)),
                         "board": "main", "float_mcap": rng.random() * 1e10})
    p = pd.DataFrame(rows)
    raw_gap = (p[p.date == "2024-09-24"].vol_ratio5.mean()
               / p[p.date == "2025-06-02"].vol_ratio5.mean())
    ck(raw_gap > 5, f"变换前 普涨日/平常日 的均值差 {raw_gap:.1f} 倍")

    p = F.cross_section(p, ["vol_ratio5"])
    g = p.groupby("date")["vol_ratio5"].mean()
    spread = g.max() / g.min()
    ck(spread < 1.10,
       f"变换后各日均值几乎相同（最大/最小 {spread:.3f} < 1.10）")
    ck(bool(p.vol_ratio5.between(0, 1).all()), "百分位落在 [0,1]")


def check_wiring() -> None:
    """特征分组表和实际产出的列必须对得上。"""
    print("\n[接线]")
    import features as F
    d = synth(400)
    d = F.per_stock(d)
    d = F.add_chip_block(d)
    missing = []
    for grp, cols in F.GROUPS.items():
        for c in cols:
            # holders / regime 两组要全市场或低频数据，单只算不出来，跳过
            if grp in ("holders", "regime"):
                continue
            if c not in d.columns:
                missing.append(f"{grp}.{c}")
    ck(not missing, f"逐只特征都产出了（缺 {missing}）")
    ck(all(n not in d.columns for *_, n in F.INTERACTIONS),
       "交互项在这一步还不该存在（它在三层变换之后）")
    names = [n for *_, n in F.INTERACTIONS]
    ck(len(names) == len(set(names)) == 3, "三个交互项，名字不重复")
    ck(F.board_of("688001") == "star" and F.board_of("300001") == "chinext"
       and F.board_of("832735") == "bj" and F.board_of("600000") == "main",
       "板块判定正确")


def check_quote_wiring() -> None:
    """daily.py 调 fetch_quotes 时必须经过 to_symbol 转前缀。

    2026-09-12 踩过：直接把裸代码（600000）传进去，腾讯一个都不认，
    **静默返回 0 只**，于是 ST 剔除整个失效而日志只有一行「返回 0 只」。
    这是历史教训 11/13 那一类：代码跑得通、输出看着合理、线没接上。
    联网的东西自测测不了，就用 AST 钉调用点。
    """
    print("\n[行情接线]")
    import ast
    src = (ROOT / "src" / "breakout" / "daily.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "fetch_quotes"]
    ck(len(calls) >= 2, f"daily.py 里有 {len(calls)} 处 fetch_quotes 调用")
    ok = all("to_symbol" in ast.dump(c.args[0]) for c in calls if c.args)
    ck(ok, "每处 fetch_quotes 的参数都经过 to_symbol 转了市场前缀")


def check_panel_html() -> None:
    """面板/邮件的 HTML 里不许出现 markdown 语法。

    2026-09-13 犯了两次：DISCLAIMER 里写了 **排名**、分数对照表里写了
    **前 10 名**，在 HTML 里都原样显示成星号。写 HTML 就用 <b>，
    这条断言防的是「下次又顺手打了两个星号」。
    """
    print("\n[面板 HTML]")
    import export as E
    fake_a = pd.DataFrame({"code": ["600000"], "name": ["测试"],
                           "score": [95.0], "close": [10.0]})
    fake_b = pd.DataFrame(columns=["code", "name", "best", "days",
                                   "streak", "close", "rise", "drop"])
    for for_panel in (True, False):
        html = E._body("2026-09-11", fake_a, fake_b,
                       {"rejected": 0, "model_date": "2026-09-13"}, for_panel)
        tag = "面板" if for_panel else "邮件"
        ck("**" not in html, f"{tag}里没有 markdown 粗体（**）")
        ck("__STAMP" not in html and "__DATE__" not in html,
           f"{tag}里没有没替换的占位符")
    html = E._body("2026-09-11", fake_a, fake_b, {}, True)
    ck(html.count("<script>") == 1, "面板有且只有一个 script 标签")
    ck(E._body("2026-09-11", fake_a, fake_b, {}, False).count("<script") == 0,
       "邮件里没有 script（邮件客户端会剥掉）")
    check_rule_text()
    check_date_picker()
    check_table_shape()


def check_rule_text() -> None:
    """上榜规则从 run_meta 走，不写死 97/10（F5-4）。

    会诊批准 SCORE_MIN 97->96 之后，daily.py 立刻按 96 出清单，而邮件抬头
    还说「≥ 97 够不到就不上」、成绩表还说「口径完全一致」，那张表是 97/10
    规则下测的，拿它给 96 分的清单背书正是 export.py 开头明令禁止的事。
    """
    import ast
    import daily as D
    import export as E
    a = pd.DataFrame({"code": ["600000"], "name": ["测试"], "score": [96.0],
                      "streak": [1], "close": [10.0]})
    b = pd.DataFrame(columns=["code", "name", "best", "days", "streak",
                              "close", "rise", "drop"])
    h = E._body("2026-09-11", a, b, {"score_min": 95, "cap_a": 8}, False)
    ck("≥ 95" in h and "前 8 名" in h, "邮件文案跟 meta 里的规则走")
    ck("≥ 97" not in h and "到 97 分" not in h, "不再出现写死的 97")
    ck("旧规则" in h and "尚未单独实测" in h,
       "规则 ≠ PERF_RULE 时成绩表标注为旧规则的成绩")
    ck("%*" in h, "清单行的历史准确率打了星号（这张表不适用于当前规则）")
    ck("到 95 分" in E._rows_a(pd.DataFrame(columns=["code"]), 95),
       "空表占位也跟着规则走")
    h0 = E._body("2026-09-11", a, b, {}, False)
    ck(f"≥ {D.SCORE_MIN}" in h0 and "旧规则" not in h0,
       f"meta 缺规则时回落到当前生效值（≥ {D.SCORE_MIN}）")
    ck(E.PERF_RULE == (97, 10),
       f"PERF_RULE 就是 STREAK_PERF 那次实验的规则（实得 {E.PERF_RULE}）")
    ck(f"{D.MIN_HISTORY_DAYS} 个交易日" in h0 and "风险剔除" in h0,
       "口径那句写明「剔掉上市不足 N 个交易日」以及风险剔除回放不了")
    src = (ROOT / "src" / "breakout" / "daily.py").read_text(encoding="utf-8")
    ck('"cap_a": CAP_A' in src, "run_meta 带 cap_a（不然邮件永远说「前 10」）")
    ck('picks["streak"] >= MIN_STREAK' in src,
       "MIN_STREAK 真的过滤了清单（以前读进来没有一处用）")
    scan = next(f for f in ast.walk(ast.parse(src))
                if isinstance(f, ast.FunctionDef) and f.name == "stage_scan")
    ck(not [n for n in ast.walk(scan)
            if isinstance(n, ast.Constant) and n.value == 120],
       "stage_scan 里不再有写死的 120（走 MIN_HISTORY_DAYS）")


def check_date_picker() -> None:
    """面板的日期下拉（2026-09-15 加）。

    历史每天的清单 HTML 都嵌在页里，切换只换 innerHTML。要钉的三件事：
    仍然只有一个 script（第二个会被 REFRESH_JS 那条断言当成占位符没换）；
    邮件完全不受 history 影响；当天那块用调用方的 meta（有风险剔除数），
    不用回放出来的（rejected=None，印不出那段）。
    """
    import re
    import export as E
    a = pd.DataFrame({"code": ["600000"], "name": ["测试"], "score": [98.0],
                      "streak": [3], "close": [10.0]})
    b = pd.DataFrame(columns=["code", "name", "best", "days", "streak",
                              "close", "rise", "drop"])
    hist = [{"date": d, "a": a, "b": b, "meta": {"date": d, "rejected": None}}
            for d in ("2026-09-11", "2026-09-12", "2026-09-15")]
    meta = {"rejected": 2, "model_date": "2026-09-15"}
    html = E._body("2026-09-15", a, b, meta, True, hist)
    ck(html.count("<script>") == 1, "带下拉的面板仍然只有一个 script 标签")
    opts = re.findall(r'<option value="([\d-]+)"', html)
    ck(opts == ["2026-09-15", "2026-09-12", "2026-09-11"],
       f"下拉按日期倒序列出全部历史日（{opts}）")
    ck('<option value="2026-09-15" selected>' in html, "默认选中当天")
    ck('<div id="day">' in html and "风险剔除 2 只" in html,
       "当天那块用的是调用方的 meta（有风险剔除数）")
    ck(E._body("2026-09-15", a, b, meta, False, hist)
       == E._body("2026-09-15", a, b, meta, False),
       "邮件不受 history 影响")
    ck("daysel" not in E._body("2026-09-15", a, b, meta, True),
       "没给 history 就没有下拉")


def check_table_shape() -> None:
    """每张表的表头数、数据行列数、空表 colspan 三者必须相等。

    A/B 两张表的列是用户点名要的（代码 / 名称 / 分数 / 上榜天数 /
    历史准确率 / 现价）。加一列忘了改表头，浏览器不会报错，只会把整行
    错位一格；空表的 colspan 漂了也只是那句「今天没有」缩成一小格。
    两种都是静默的，所以用断言钉住。
    """
    import re
    import export as E
    full_a = pd.DataFrame({"code": ["600000"], "name": ["测试"],
                           "score": [98.0], "streak": [3], "close": [10.0]})
    full_b = pd.DataFrame({"code": ["600001"], "name": ["测试二"],
                           "best": [99.0], "days": [4], "streak": [2],
                           "close": [12.0], "rise": [25.0], "drop": [-12.0]})
    empty = pd.DataFrame(columns=list(full_b.columns))
    html = E._body("2026-09-11", full_a, full_b, {}, False)
    tables = re.findall(r"<table>(.*?)</table>", html, re.S)
    ck(len(tables) == 3, f"面板有 3 张表（实际 {len(tables)}）")
    for name, t in zip(("清单 A", "清单 B", "连续天数对照"), tables):
        nth = t.count("<th>")
        # `<tr ...>` 也要收进来：成绩表里样本少的那几行带 style 灰掉了，
        # 只匹配裸 <tr> 会把它们整行漏过去
        rows = [r for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S)
                if "<th" not in r]
        bad = [r for r in rows if r.count("<td") != nth]
        ck(rows and not bad, f"{name}：{nth} 个表头，{len(rows)} 个数据行都是 {nth} 列")
    for lbl, fn, df in (("清单 A", E._rows_a, pd.DataFrame(columns=["code"])),
                        ("清单 B", E._rows_b, empty)):
        blank = fn(df)
        m = re.search(r'colspan="(\d+)"', blank)
        head = tables[0 if lbl == "清单 A" else 1].count("<th>")
        ck(m is not None and int(m.group(1)) == head,
           f"{lbl} 空表的 colspan 等于表头数（{head}）")
    # 邮件里的成绩必须和实验产物一致：STREAK_PERF（hit / lift / n）对
    # window_grid.json 的 W5（生产口径）行，BASE 对 score_calibration.json。
    # 改了规则或模型只重跑实验不改常量，这里会红。
    # 两份 JSON 都已入库（git ls-files out_breakout），缺了就是被删了，
    # 不该走一条静默通过的分支。
    import json
    import re
    wg = ROOT / "out_breakout" / "window_grid.json"
    sc = ROOT / "out_breakout" / "score_calibration.json"
    ck(wg.exists() and sc.exists(),
       "实验产物 window_grid.json / score_calibration.json 在（两份都已入库）")
    grid = json.loads(wg.read_text(encoding="utf-8"))["grid"]
    w5 = {int(re.search(r"连续≥(\d)天", r["label"]).group(1)): r
          for r in grid if r["kind"] == "W5"}
    base = 100 * json.loads(sc.read_text(encoding="utf-8"))["base"]
    bad = []
    for need, hit, lift, n in E.STREAK_PERF:
        r = w5.get(need)
        if not r:
            bad.append(f"连续≥{need}天 实验里没有")
            continue
        # lift 是真正印进邮件和面板的那一列（「随便买的 4.3 倍」），
        # 以前只钉 hit/n：改了成绩忘改倍数，自测照绿而邮件自相矛盾
        if (abs(100 * r["hit"] - hit) > 0.06 or abs(r["lift"] - lift) > 0.06
                or abs(hit / E.BASE - lift) > 0.06 or int(r["n"]) != n):
            bad.append(f"连续≥{need}天 常量 {hit}%/{lift}倍/{n} vs 实验 "
                       f"{100 * r['hit']:.1f}%/{r['lift']:.2f}倍/{r['n']}")
    ck(not bad, "STREAK_PERF 的 hit / lift / n 和 window_grid.json 生产口径行一致"
       + ("：" + "；".join(bad) if bad else ""))
    ck(abs(E.BASE - base) < 0.06,
       f"BASE {E.BASE} 和 score_calibration 的验证集基准 {base:.2f} 一致")
    ck(all(f"{lift:.1f} 倍" in html for _k, _h, lift, _n in E.STREAK_PERF),
       "五档的倍数都印在对照表里")
    ck(not hasattr(E, "SCORE_TABLE"),
       "SCORE_TABLE 已删（从没渲染过，注释里的样本数早就和产物对不上了）")

    # 同一个准确率在清单行和对照表必须印成同一个字符串（F5-9）：
    # 以前清单行是 .0f、对照表是 .1f，15.7% 在两处印成 16% 和 15.7%，
    # 连续 1 天（12.6）和连续 4 天（13.0）在清单行都成了 13%，分不出来
    for need, hit, lift, n_ in E.STREAK_PERF:
        row = E._rows_a(pd.DataFrame({"code": ["600000"], "name": ["x"],
                                      "score": [98.0], "streak": [need],
                                      "close": [1.0]}))
        cell = re.findall(r"<td>([\d.]+%)</td>", row)[0]
        # 样本够的档：行内和对照表必须是同一个字符串（F5-9 那条）。
        # 样本不够的档（n < SMALL_N）：行内要退到下一个够样本的档，不能把
        # 24 个名额算出来的 0.0% 印成某只票的「历史准确率」（2026-09-16）。
        want = f"{hit:.1f}%" if n_ >= E.SMALL_N else f"{E.streak_perf(need)[0]:.1f}%"
        ck(cell == want and (n_ < E.SMALL_N or f'<td class="sc">{cell}</td>' in html),
           f"连续 {need} 天（n={n_}）：清单 A 行印 {cell}，应当是 {want}")
        rowb = E._rows_b(pd.DataFrame({"code": ["600001"], "name": ["x"],
                                       "best": [99.0], "days": [need],
                                       "streak": [need], "close": [1.0],
                                       "rise": [1.0], "drop": [-1.0]}))
        ck(re.findall(r"<td>([\d.]+%)</td>", rowb)[0] == cell,
           f"连续 {need} 天：清单 B 行和清单 A 行印同一个字符串")
    ck(f"{E.perf_row(1)[2]:.1f} 倍" in E.DISCLAIMER
       and f"{E.perf_row(1)[1]:.1f}%" in E.DISCLAIMER,
       "DISCLAIMER 的准确率和倍数与对照表同格式（.1f，不是「4 倍」）")
    ck(f"{E.perf_row(1)[1]}%" in html and f"{E.perf_row(3)[1]}%" in html,
       "邮件里印的是 STREAK_PERF 里的数（首日 / 连续 3 天）")

    # 样本少的档次要有 95% 区间、而且灰掉
    top = E.STREAK_PERF[0]
    lo, hi = E._wilson(top[1], top[3])
    # 命中率恰好是 0 时区间下界也是 0（Wilson 在 k=0 上就是这样），
    # 所以判 <=；区间必须真的把点估计盖住，而且印在表里
    ck(lo <= top[1] <= hi and f"{lo:.1f}~{hi:.1f}%" in html,
       f"连续 {top[0]} 天那档印了 Wilson 区间 {lo:.1f}~{hi:.1f}%")
    ck(E.STREAK_PERF[0][3] < E.SMALL_N and "color:#7c8794" in html,
       f"样本少于 {E.SMALL_N} 的行灰掉")


def check_chip_grid_overflow() -> None:
    """涨过 10 倍 / 跌破 1/10 的票，筹码不许整天丢掉（S1）。

    第一版的网格死死锚在 1/10~10 倍，越界那天 `s <= 0` 直接 continue：
    既不衰减也不写结果，特征 NaN。实测训练表 2739 行是这么来的，
    而这些行的 y_up 率是全表的 2.6 倍。更隐蔽的是没 NaN 但值错的行：
    涨过 10 倍再跌回来的票，10 倍以上的成交从没记进分布。
    """
    print("\n[筹码·网格越界]")
    from chips import chip_features
    d = synth(600, seed=5)
    path = np.r_[np.ones(200),
                 np.exp(np.linspace(0, np.log(15.0), 200)),
                 np.exp(np.linspace(np.log(15.0), np.log(6.0), 200))]
    up = d.copy()
    for k in ("high", "low", "close"):
        up[k] = up[k].to_numpy() * path
    f = chip_features(up.high.to_numpy(), up.low.to_numpy(),
                      up.close.to_numpy(), up.turnover.to_numpy())
    ck(bool(f.chip_avg.notna().all()),
       f"涨到 15 倍全程无 NaN（NaN {int(f.chip_avg.isna().sum())} 行）")
    ck(f.chip_win.iloc[399] > 0.9,
       f"15 倍新高处获利盘 {f.chip_win.iloc[399]:.1%} > 90%")
    ck(f.chip_win.iloc[-1] < f.chip_win.iloc[399],
       f"跌回 6 倍时获利盘 {f.chip_win.iloc[-1]:.1%} 低于 15 倍新高时 "
       f"{f.chip_win.iloc[399]:.1%}（10 倍以上的成交真的记进了分布）")

    dn = d.copy()
    for k in ("high", "low", "close"):
        dn[k] = dn[k].to_numpy() / path
    g = chip_features(dn.high.to_numpy(), dn.low.to_numpy(),
                      dn.close.to_numpy(), dn.turnover.to_numpy())
    ck(bool(g.chip_avg.notna().all()),
       f"跌到 1/15 全程无 NaN（NaN {int(g.chip_avg.isna().sum())} 行）")
    ck(g.chip_win.iloc[399] < 0.1,
       f"1/15 新低处获利盘 {g.chip_win.iloc[399]:.1%} < 10%")
    # 未越界的票必须逐位不变：外接只是把「丢弃」换成「补零的格」
    base = synth(400, seed=5)
    a = chip_features(base.high.to_numpy(), base.low.to_numpy(),
                      base.close.to_numpy(), base.turnover.to_numpy())
    ck(bool(a.notna().all().all()), "不越界的票照常全有值（对照）")


def check_warmup_nan() -> None:
    """ma60 没定义的前 59 行必须是 NaN 而不是 0（S4）。"""
    print("\n[暖机期·ma60 未定义的行]")
    import features as F
    n = 120
    c = np.linspace(10, 20, n)          # 单调上涨：成熟后 ma_align 必为 3
    d = pd.DataFrame({"open": c, "close": c, "high": c * 1.01, "low": c * 0.99,
                      "volume": np.full(n, 1e6), "turnover": np.full(n, .02),
                      "amount": c * 1e6})
    d = F.per_stock(d)
    ck(bool(d.ma_align.iloc[:59].isna().all()), "ma_align 前 59 行 NaN")
    ck(bool(d.above_ma60.iloc[:59].isna().all()), "above_ma60 前 59 行 NaN")
    ck(bool(d.ma_align.iloc[59:].notna().all()) and d.ma_align.iloc[59] == 3,
       "第 60 行起有值（单调上涨 = 3），没被一起抹掉")
    ck(d.above_ma60.isna().equals(d.dist_52w_high.isna()),
       "暖机窗口和 dist_52w_high（min_periods=60）完全一致")
    # 横截面：暖机票不能被排到底部
    rng = np.random.default_rng(0)
    m = 400
    p = pd.DataFrame({"date": "2025-06-02", "board": "main",
                      "float_mcap": rng.random(m + 1) * 1e10,
                      "above_ma60": np.r_[rng.integers(0, 61, m).astype(float),
                                          np.nan]})
    p = F.neutralize(F.cross_section(p, ["above_ma60"]), ["above_ma60"])
    ck(bool(np.isnan(p.above_ma60.iloc[-1])),
       "NaN 穿过 ①② 仍是 NaN（模型侧落 0 = 组均值，不是垫底）")


def check_market_breadth() -> None:
    """没有昨收的行不能算进市场宽度的分母（S19）。"""
    print("\n[市场宽度·NaN 不进分母]")
    import features as F
    rows = []
    # 第 2 天有效的三只里 2 涨 1 跌，第 3 天四只里 3 涨 1 跌
    px = {"A": [10.0, 11.0, 12.0], "B": [10.0, 11.0, 12.0],
          "C": [10.0, 9.0, 8.0], "D": [None, 10.0, 11.0]}
    for code, ps in px.items():
        for dt_, v in zip(("2024-01-02", "2024-01-03", "2024-01-04"), ps):
            if v is None:
                continue
            rows.append({"code": code, "date": dt_, "close": v,
                         "ret20": 0.1, "ret60": 0.1})
    q = F.add_market(pd.DataFrame(rows).sort_values(["code", "date"]))
    b = q.groupby("date")["mkt_breadth"].first()
    ck(bool(np.isnan(b["2024-01-02"])),
       f"整日没有昨收 -> NaN 而不是 0（实得 {b['2024-01-02']}）")
    ck(abs(b["2024-01-03"] - 2 / 3) < 1e-9,
       f"首行 NaN 的票不进分母：{b['2024-01-03']:.4f} 应是 2/3 不是 2/4")
    ck(abs(b["2024-01-04"] - 3 / 4) < 1e-9, "全员有昨收时照常")
    ck(int(q.groupby("date")["mkt_breadth"].nunique().max()) == 1,
       "市场宽度当日全市场同值")


def check_interactions() -> None:
    """交互项必须方向保持：假设象限和它的镜像象限不能拿同一个值（S18）。"""
    print("\n[交互项·方向保持]")
    import features as F
    ck(all(len(t) == 5 for t in F.INTERACTIONS)
       and all(s in (-1, 1) for _, s, _, s2, _ in F.INTERACTIONS
               for s in (s, s2)),
       "每条交互项都带 (a, sa, b, sb, name) 且指向是 ±1")
    allf = {c for g in F.GROUPS.values() for c in g}
    ck(all(a in allf and b in allf for a, _, b, _, _ in F.INTERACTIONS),
       "交互项的两条腿都在五组特征里（接线）")
    for a, sa, b, sb, name in F.INTERACTIONS:
        # 四个象限按**指向**造，不按原始正负：0=假设象限（两条腿都在假设那侧）、
        # 1=镜像象限（两条腿都在反侧）、2/3=各中一条。中性化之后两列都以 0
        # 为中心，乘法下 0 和 1 是同一个值 —— 这正是要钉住的那件事。
        p = pd.DataFrame({
            a: [0.3 * sa, -0.3 * sa, 0.3 * sa, -0.3 * sa, 0.6 * sa, 0.0],
            b: [0.3 * sb, -0.3 * sb, -0.3 * sb, 0.3 * sb, 0.6 * sb, 0.0]})
        v = F.add_interactions(p)[name].to_numpy(float)
        ck(v[0] > 0,
           f"{name}：假设象限 {v[0]:+.2f} > 0（乘法下是 {sa * sb * 0.09:+.2f}）")
        ck(v[0] > v[1],
           f"{name}：假设象限 {v[0]:+.2f} > 镜像象限 {v[1]:+.2f}"
           "（乘法下两者恰好相等，标签率却差 2~3 倍）")
        ck(v[1] < 0, f"{name}：镜像象限 {v[1]:+.2f} < 0")
        ck(v[2] < v[0] and v[3] < v[0],
           f"{name}：只中一条腿的两个象限都低于假设象限")
        ck(v[4] > v[0], f"{name}：指向侧更强时值更大（{v[4]:+.2f} > {v[0]:+.2f}）")
        ck(abs(v[5]) < 1e-12, f"{name}：两条腿都在中位时为 0")
    ck(isinstance(F.INTERACTION_VERSION, int) and F.INTERACTION_VERSION >= 2,
       f"交互项版本号 {F.INTERACTION_VERSION}（进特征指纹，改公式就重训）")


def check_holder_preipo() -> None:
    """招股书 / 新三板阶段的股东名册不是二级市场散户（S26）。"""
    print("\n[股东人数·上市前名册]")
    import features as F
    ck(1000 <= F.HOLDER_MIN <= 1600,
       f"户数下限 {F.HOLDER_MIN}：低于 1000 漏北交所新三板名册，"
       "高于 1600 会误杀 920033 那种真实小户数")
    dates = pd.date_range("2024-01-01", "2024-12-31",
                          freq="B").strftime("%Y-%m-%d")
    panel = pd.DataFrame({"date": list(dates) * 2,
                          "code": ["301999"] * len(dates)
                                  + ["600999"] * len(dates),
                          "outstanding_share": 1e9})
    holders = pd.DataFrame({
        "代码": ["301999", "301999", "301999", "600999", "600999"],
        "报告期": ["20230630", "20231231", "20240331", "20231231", "20240331"],
        "公告日期": ["2024-03-08", "2024-04-25", "2024-08-20",
                     "2024-04-25", "2024-08-20"],
        "股东户数-本次": [25, 30000, 28000, 5000, 4800],
        "股东户数-上次": [25, 25, 30000, 5200, 5000],
        "股东户数-增减比例": [0.0, 119900.0, -6.666667, -3.846154, -4.0]})
    out = F.holder_features(panel, holders)
    a = out[out.code == "301999"].set_index("date")
    ck(bool(np.isnan(a.loc["2024-04-01", "gdhs_level"]))
       and bool(np.isnan(a.loc["2024-04-01", "gdhs_chg1"])),
       "招股书那期（25 户）整行不进特征（现状 level=2.5e-8、chg1=0）")
    ck(abs(a.loc["2024-04-25", "gdhs_level"] - 3e-5) < 1e-12,
       "上市后第一期的户数本身照常可用")
    ck(bool(np.isnan(a.loc["2024-04-25", "gdhs_chg1"]))
       and bool(np.isnan(a.loc["2024-04-25", "gdhs_chg3"])),
       "但它的「增减比例」是拿招股书户数比出来的 1199 倍，置 NaN")
    ck(abs(a.loc["2024-08-20", "gdhs_chg1"] + 0.0666667) < 1e-5,
       f"下一期恢复正常：{a.loc['2024-08-20', 'gdhs_chg1']:.4f}")
    ck(abs(a.loc["2024-08-20", "gdhs_chg3"] + 0.0666667) < 1e-5,
       "三期滚动和不再被 1199 倍那期污染（现状 1198.9，拖三个季度）")
    ck(a.loc["2024-08-20", "gdhs_down_streak"] == 1, "连降期数照常")
    b = out[out.code == "600999"].set_index("date")
    ck(abs(b.loc["2024-08-20", "gdhs_chg1"] + 0.04) < 1e-9,
       f"正常票（5000 -> 4800）不受影响：{b.loc['2024-08-20', 'gdhs_chg1']:.4f}")


def check_common_start() -> None:
    """训练表的横截面必须在同一个池子里排（S2）。"""
    print("\n[训练表·共同起点]")
    import build as BD
    import fselect as FS
    dts = pd.date_range("2023-01-03", periods=300, freq="B").strftime("%Y-%m-%d")
    rows = []
    for k, c in enumerate(_real_codes(3)):
        start = 0 if k == 0 else 100        # 只有第一只有更早的历史
        for d in dts[start:]:
            rows.append({"code": c, "date": d, "close": 10.0})
    daily = pd.DataFrame(rows)
    cut, start = BD.trim_common_start(daily)
    ck(start == dts[100],
       f"共同起点取「全市场到齐」的那天（{start} 应是 {dts[100]}）")
    n = cut.groupby("date")["code"].nunique()
    ck(int(n.min()) == int(n.max()) == 3, "裁剪之后每天都是 3 只")
    ck(len(cut) < len(daily) and str(cut["date"].min()) == dts[100],
       "更早的那 100 天真的被砍掉了（不是只打了行日志）")

    # 没算出 IC 的月份不算符号翻转
    rng = np.random.default_rng(6)
    parts = []
    for i, m in enumerate(["2025-01", "2025-02", "2025-03", "2025-04",
                           "2025-05", "2025-06"]):
        k = 400 if i >= 4 else 600          # 后两个月有效行 <500 -> IC 哨兵 0.0
        x = rng.normal(size=k)
        y = (x + rng.normal(0, 0.5, k) > 0.8).astype(float)
        parts.append(pd.DataFrame({"date": f"{m}-15", "f__last": x, "y": y}))
    st = FS.ic_stability(pd.concat(parts, ignore_index=True), ["f__last"], "y")
    ck(int(st.loc["f__last", "n_month"]) == 4,
       f"只数真算出 IC 的月份（n_month={int(st.loc['f__last', 'n_month'])}，"
       "应是 4 不是 6）")
    ck(float(st.loc["f__last", "flip"]) == 0.0,
       f"哨兵 0.0 不再白送两次符号翻转（flip={st.loc['f__last', 'flip']:.2f}）")


def check_eligible_rule() -> None:
    """「够不够老」只能有一条规则，三处共用（F8-9）。"""
    print("\n[次新·一条规则三处共用]")
    import ast
    import build as BD
    import exp_window as EW
    a, b, c = _real_codes(3)
    g = synth(130, seed=31)
    daily = pd.concat([g.assign(code=a),                 # 130 根
                       g.iloc[10:].assign(code=b),       # 恰好 120 根
                       g.iloc[11:].assign(code=c)],      # 119 根
                      ignore_index=True)
    daily["amount"] = daily["close"] * daily["volume"]
    daily["outstanding_share"] = 1e8
    daily = BD.attach_turnover(daily)
    last = str(daily["date"].max())
    elig = set(BD.eligible(daily, last))
    ck(elig == {a}, f"只有 130 根那只够格（实得 {sorted(elig)}）")
    ck(b not in elig and c not in elig,
       f"恰好 {BD.MIN_HIST} 根还不够（前 {BD.MIN_HIST} 根只预热），119 根更不够")
    panel = BD.assemble(daily)
    bad = [d for d in sorted(set(daily["date"]))
           if set(panel[panel["date"] == d]["code"]) != set(BD.eligible(daily, d))]
    ck(not bad,
       f"eligible 和 assemble 的裁剪逐日一致（对不上的日子 {bad[:3]}）")
    ck(len(panel) == 10, f"130 根的票贡献最后 10 行（实得 {len(panel)}）")

    # 回测排名：剔掉之后要**补位**，不是只删
    d = pd.DataFrame({"date": ["2025-06-02"] * 12,
                      "code": [f"{i:06d}" for i in range(12)],
                      "_p": np.linspace(0.99, 0.50, 12),
                      "score": 97.0, "y_up": 0.0})
    r = EW.production_rank(d, {"000002", "000006"})
    top = set(r[r["rank"] <= 10]["code"])
    ck(len(top) == 10 and not (top & {"000002", "000006"}),
       f"剔掉两只之后前 10 还是 10 只（{sorted(top)}）")
    ck({"000010", "000011"} <= top, "原来第 11、12 名补位进来了")
    src = ast.parse((ROOT / "src" / "breakout" / "exp_window.py")
                    .read_text(encoding="utf-8"))
    called = {f.name: {n.func.id for n in ast.walk(f)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
              for f in ast.walk(src) if isinstance(f, ast.FunctionDef)}
    ck("production_rank" in called.get("build_cache", set())
       and "production_rank" in called.get("main", set()),
       "build_cache 和 main 都走 production_rank（名次只有一份实现）")


def check_backtest_universe() -> None:
    """回测的候选池要和生产一样剔 ST（S8）。"""
    print("\n[回测口径·ST]")
    import ast
    import json
    import tempfile
    import validate as V
    tmp = Path(tempfile.mkdtemp(prefix="st_")) / "st.json"
    tmp.write_text(json.dumps({"date": "2026-09-16",
                               "codes": ["600001", "600005"]}),
                   encoding="utf-8")
    got = V.st_codes(tmp)
    ck(got == {"600001", "600005"}, f"ST 名单读得出来（{sorted(got)}）")
    ck(V.st_codes(tmp.parent / "没有这个文件.json") == set(),
       "没有缓存就返回空集合（宁可不剔，也不凭空造名单）")

    day = pd.DataFrame({"date": ["2025-06-02"] * 12,
                        "code": [f"{600000 + i:06d}" for i in range(12)],
                        "board": ["main"] * 12,
                        "y_up": [0.0] * 12})
    p = np.linspace(0.99, 0.50, 12)
    q = np.linspace(0, 1, 101)
    st = {"600001", "600005"}
    sel = V.pick(day, p, q, score_min=0, cap=10, board_adj={}, st=st)
    ck(len(sel) == 10 and not (set(sel["code"]) & st),
       f"pick 剔 ST 之后仍给满 10 只（{sel['code'].tolist()}）")
    ck({"600010", "600011"} <= set(sel["code"]),
       "11、12 名补位（先取 10 再删的话这两只永远进不来）")
    tn = V.daily_topn(day.assign(p=p), p, 10, "y_up", st=st)
    ck(set(tn["code"]) == set(sel["code"]),
       "daily_topn 和 pick 选出同一批票（两条路同一个口径）")
    ck(len(V.pick(day, p, q, score_min=0, cap=10, board_adj={}, st=set())) == 10
       and "600001" in set(V.pick(day, p, q, score_min=0, cap=10,
                                  board_adj={}, st=set())["code"]),
       "st=空集合时行为和以前一模一样（没有 ST 缓存的机器不受影响）")

    vsrc = (ROOT / "src" / "breakout" / "validate.py").read_text(encoding="utf-8")
    fn = {f.name: f for f in ast.walk(ast.parse(vsrc))
          if isinstance(f, ast.FunctionDef)}
    for name in ("pick", "daily_topn"):
        calls = {n.func.id for n in ast.walk(fn[name])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        ck("st_codes" in calls, f"{name} 默认走 st_codes()（唯一来源）")
    ck(V.rule_stamp().get("st_excluded") is not None,
       "规则戳记里写明剔了几只 ST")


def check_acceptance_base() -> None:
    """验收的基准只能算被评估的月份，不能算整张表（F8-13，教训 30）。"""
    print("\n[验收基准·同一段时间]")
    import validate as V
    seg = []
    for d, y in (("2023-01-05", 0.20), ("2025-03-05", 0.02), ("2025-04-05", 0.02)):
        k = int(round(y * 1000))
        seg.append(pd.DataFrame({"date": d, "code": "600000",
                                 "y_up": [1.0] * k + [0.0] * (1000 - k)}))
    df = pd.concat(seg, ignore_index=True)          # 全表均值 0.08，评估段 0.02
    picks = pd.DataFrame({"date": ["2025-03-05", "2025-04-05"],
                          "code": ["600000", "600001"],
                          "y_up": [1.0, 0.0],
                          "month": ["2025-03", "2025-04"]})
    by_month = pd.DataFrame([{"month": "2025-03", "n_pick": 1, "hit": 1.0,
                              "base": 0.02, "lift": 50.0},
                             {"month": "2025-04", "n_pick": 1, "hit": 0.0,
                              "base": 0.02, "lift": 0.0}])
    acc = V.acceptance(by_month, picks, df)
    ck(abs(acc["base_rate"] - 0.02) < 1e-9,
       f"基准 = 评估月份的均值 0.02（实得 {acc['base_rate']:.4f}，全表是 0.08）")
    ck(abs(acc["lift"] - acc["overall_hit"] / 0.02) < 1e-6,
       "倍数和命中率用同一个分母")
    ck(acc["base_months"] == ["2025-03", "2025-04"] and acc["base_n"] == 2000,
       f"产物记下分母是哪几个月、多少行（{acc['base_months']}，{acc['base_n']} 行）")


def _fs_run_inputs(path: Path) -> list[str]:
    """把文件里每处 `FS.run(x, ...)` 的第一个实参还原成一段 AST 文本。

    实参是个名字就去同一个函数里找它的赋值右侧 —— 要钉的是「喂进特征
    选择的那份切片是怎么切出来的」，而切片和调用往往隔着一行。
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        asgn: dict[str, str] = {}
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        asgn[t.id] = ast.dump(n.value)
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "run" and n.args
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "FS"):
                a0 = n.args[0]
                out.append(asgn.get(a0.id, "") if isinstance(a0, ast.Name)
                           else ast.dump(a0))
    return out


def check_label_isolation() -> None:
    """标签只准看价格，而且这条隔离要靠代码强制（S11）。

    以前 build.assemble 把算完全部特征和筹码的 g（36 列）整个递给
    label_frame。标签只读 close/high，结果确实逐位相同，但 label.py 开头
    「不共享 DataFrame」那句当时只是注释：哪天 per_stock / add_chips 里
    有人给 close 做了复权或 fillna，标签会跟着静默漂移。
    """
    print("\n[标签隔离·只收 close/high]")
    import ast
    import features as F
    from label import label_frame, label_up
    raw = synth(400)
    d = F.add_chip_block(F.per_stock(raw.copy()))
    ck(len(d.columns) > 20 and np.array_equal(d["close"].to_numpy(),
                                              raw["close"].to_numpy())
       and np.array_equal(d["high"].to_numpy(), raw["high"].to_numpy())
       and d.index.equals(raw.index),
       f"特征步骤（{len(d.columns)} 列）不改写 close/high、不动 index")
    lab = label_frame(d[["close", "high"]])
    ck(list(lab.columns) == ["y_up", "y_t0", "y_top"]
       and lab.index.equals(d.index), "label_frame 只出三列标签、index 对齐")
    bad = False
    try:
        label_frame(d)
    except AssertionError:
        bad = True
    ck(bad, "label_frame 收到特征列时断言失败（隔离靠代码不靠自觉）")
    ck(np.array_equal(lab["y_up"].to_numpy(),
                      label_up(raw["close"].to_numpy(float),
                               raw["high"].to_numpy(float)), equal_nan=True),
       "label_frame 与直接 label_up(close, high) 逐位一致")
    src = ast.parse((ROOT / "src" / "breakout" / "build.py")
                    .read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(src) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "label_frame"]
    ck(len(calls) == 1 and "'close'" in ast.dump(calls[0].args[0])
       and "'high'" in ast.dump(calls[0].args[0])
       and "chip" not in ast.dump(calls[0].args[0]),
       f"build.py 唯一那处 label_frame 只把 close/high 递过去（{len(calls)} 处）")


def check_purge_suspension() -> None:
    """走向前的净化要按**这一行自己的**标签窗口切，不是按全市场日期（S10）。

    label.label_up 的窗口数的是该票自己的 K 线，停牌日在新浪的日线表里
    没有行，于是停牌票的第 20 根会从「往前数 20 个市场日」这条线底下钻
    过去。实测走向前十个月 1,884 行是这么漏进训练集的，其中 415 个正样本
    （正样本率 22.0%，全表 3.42% 的 6.4 倍）。
    """
    print("\n[走向前净化·停牌票的标签窗口]")
    import ast
    import re
    import label as L
    import validate as V
    ck(V.PURGE_DAYS == L.UP_WINDOW,
       f"净化长度 {V.PURGE_DAYS} 就是标签窗口 {L.UP_WINDOW}（同一个常量）")

    dates = (pd.date_range("2025-04-01", periods=60, freq="B")
             .strftime("%Y-%m-%d").tolist())
    first = next(i for i, d in enumerate(dates) if d >= "2025-06-01")
    a, b = _real_codes(2)
    gone = set(dates[first - 12:first])      # B 在测试月之前停牌 12 个市场日
    rows = []
    for d in dates:
        rows.append({"code": a, "date": d, "y_up": 0.0})
        if d not in gone:
            rows.append({"code": b, "date": d, "y_up": 0.0})
    df = pd.DataFrame(rows)
    tr = V.train_slice(df, "2025-06")
    ta = set(tr[tr["code"] == a]["date"])
    tb = set(tr[tr["code"] == b]["date"])
    ck(dates[first - 21] in ta and dates[first - 20] not in ta,
       f"不停牌的票：20 日净化线还在原处（留 {dates[first - 21]}、"
       f"剔 {dates[first - 20]}）")
    ck(dates[first - 32] in ta and dates[first - 32] not in tb,
       f"同一天 {dates[first - 32]}：停牌 12 天那只的第 20 根 K 线落进了"
       "测试月，被剔；不停牌那只留下")
    ck(dates[first - 33] in tb,
       f"再往前一天 {dates[first - 33]}，停牌票的第 20 根还在测试月之前，照留")
    end = (df.sort_values(["code", "date"]).groupby("code")["date"]
             .shift(-L.UP_WINDOW).reindex(df.index))
    e = end.reindex(tr.index)
    ck(bool(e.notna().all()) and str(e.max()) < "2025-06-01",
       f"独立重算：训练集里没有一行的第 {L.UP_WINDOW} 根 K 线碰到测试月"
       f"（最远 {e.max()}）")
    ck(len(tr) < len(df[df["date"] < V.purge_cut(df, "2025-06")]),
       "train_slice 真的比只按日期切少了几行（不是个恒等变换）")

    # 接线：改回 `df[df["date"] < purge_cut(...)]` 就立刻红
    for fn in ("validate.py", "exp_calib.py", "exp_window.py"):
        src = (ROOT / "src" / "breakout" / fn).read_text(encoding="utf-8")
        ck("train_slice(" in src, f"{fn} 走 train_slice")
        ck(not re.search(r'df\["date"\]\s*<\s*(V\.)?purge_cut', src),
           f"{fn} 里没有残留的「只按日期切」")
    vsrc = ast.parse((ROOT / "src" / "breakout" / "validate.py")
                     .read_text(encoding="utf-8"))
    wf = next(f for f in ast.walk(vsrc)
              if isinstance(f, ast.FunctionDef) and f.name == "walk_forward")
    names = {n.func.id for n in ast.walk(wf)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    ck("train_slice" in names and "purge_cut" not in names,
       "walk_forward 的训练集来自 train_slice")


def check_fselect_purge() -> None:
    """特征选择吃的那份切片也要净化（S21）。

    擂台和七个实验都在 `date < TRAIN_END` 上做特征选择，而走向前第一个
    测试月就是 2025-03：紧挨着的 20 个交易日（106,779 行、4,473 个正样本，
    占筛选集正样本 5.19%）的标签由 2025-03 的最高价算出来，等于让未来给
    特征的去留投票。净化后 range_compress__mean 的总 IC 0.0071 -> 0.0053，
    离 IC_MIN=0.005 只剩 0.0003。
    """
    print("\n[特征选择·净化线]")
    import re
    import label as L
    import validate as V
    dates = (pd.date_range("2025-01-01", periods=80, freq="B")
             .strftime("%Y-%m-%d").tolist())
    df = pd.DataFrame({"code": "600000", "date": dates, "y_up": 0.0})
    cut = V.purge_cut(df, "2025-03")
    first = next(i for i, d in enumerate(dates) if d >= "2025-03-01")
    ck(dates.index(cut) == first - V.PURGE_DAYS,
       f"净化线 {cut} 是测试月首日 {dates[first]} 往前 {V.PURGE_DAYS} 个交易日")
    ck(all(dates[i + L.UP_WINDOW] < dates[first]
           for i in range(dates.index(cut))),
       "净化集里没有一行的第 20 根 K 线碰到测试月（边界正好卡死）")

    for fn in ("arena.py", "exp_calib.py", "exp_window.py"):
        p = ROOT / "src" / "breakout" / fn
        src = p.read_text(encoding="utf-8")
        ck(not re.search(r"<(?!=)\s*V\.TRAIN_END", src),
           f"{fn} 里没有 `< V.TRAIN_END` 这种未净化的切法")
        got = _fs_run_inputs(p)
        ck(bool(got) and all("train_slice" in s for s in got),
           f"{fn} 的 FS.run 吃的是 train_slice 切出来的（{len(got)} 处）")


def check_fselect() -> None:
    """第四道筛（L1）必须真的接上，而且只记录不剔除（S12）。

    model.L0Logistic.surviving() 从上线起零调用，feature_select.json 里
    一直没有第 4 道的记录，读产物的人会以为 45 个生产特征过了 L1 压缩。
    真拿它剪枝又不行：线性 L1 看不见 GBDT 要的组合特征，
    IC_MIN 从 0.010 放到 0.005 换来的 +2.85 个百分点就白给了。
    """
    print("\n[特征选择·第四道 L1 只记录]")
    import ast
    import json
    import tempfile
    import fselect as FS
    import model as M
    rng = np.random.default_rng(17)
    parts = []
    for m in ("2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"):
        k = 600                       # 每月 ≥500 行，否则 single_ic 出哨兵 0.0
        sig = rng.normal(size=(k, 4))
        z = sig @ np.array([2.0, 1.8, 1.6, 1.4]) + rng.normal(0, 1.0, k)
        g = pd.DataFrame({"date": f"{m}-15",
                          "y_up": (z > np.quantile(z, 0.95)).astype(float)})
        for i in range(4):
            g[f"sig{i}__last"] = sig[:, i]
        for i in range(6):
            g[f"noise{i}__last"] = rng.normal(size=k)
        g["dup0__last"] = sig[:, 0]   # 和 sig0 逐位相同 -> 必落相关剪枝
        g["dup1__last"] = sig[:, 1]
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    cols = [c for c in df.columns if "__" in c]
    tmp = Path(tempfile.mkdtemp(prefix="fsel_"))
    rep = FS.run(df, cols, y="y_up", out_dir=tmp)

    ck({"ic_too_weak", "ic_unstable", "collinear"} <= set(rep["dropped"]),
       f"前三道各自留了记录（{sorted(rep['dropped'])}）")
    coll = set(rep["dropped"]["collinear"])
    for pair in (("sig0__last", "dup0__last"), ("sig1__last", "dup1__last")):
        ck(len(set(pair) & set(rep["keep"])) == 1
           and len(set(pair) & coll) == 1,
           f"逐位相同的 {pair} 只留一个，另一个进 collinear")
    ck(set(rep["keep"]) <= set(cols) and rep["end"] == len(rep["keep"]),
       "keep 是入参的子集，end 和它对得上")
    ck("l1_error" not in rep, f"第四道没走异常分支（{rep.get('l1_error')}）")
    ck("l1_zero" in rep and set(rep["l1_zero"]) <= set(rep["keep"]),
       f"第四道的记录在，且只提 keep 里的列（零系数 {len(rep['l1_zero'])} 个）")
    ck(set(rep["keep"]) & set(rep["l1_zero"]) == set(rep["l1_zero"])
       and len(rep["keep"]) > 0,
       "只记录不剔除：l1_zero 里的列仍然留在 keep 里")
    saved = json.loads((tmp / "feature_select.json").read_text(encoding="utf-8"))
    ck("l1_zero" in saved and saved["keep"] == rep["keep"],
       "l1_zero 真的落进了 feature_select.json（产物和返回值一致）")

    m0 = M.L0Logistic().fit(M.stratified_sample(df, "y_up"), cols, "y_up")
    alive = set(m0.surviving())
    ck(alive <= set(cols), "surviving 只报入参里的列")
    ck(len(alive) < len(cols),
       f"C=0.05 至少压掉一个（{len(alive)}/{len(cols)}，防止退化成「全留」）")
    ck(len({"sig2__last", "sig3__last"} & alive) == 2
       and bool({"sig0__last", "dup0__last"} & alive)
       and bool({"sig1__last", "dup1__last"} & alive),
       f"四个真信号都还在（孪生列各留一支）：{sorted(alive & set(cols))[:6]}")

    fsrc = (ROOT / "src" / "breakout" / "fselect.py").read_text(encoding="utf-8")
    ck(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "surviving" for n in ast.walk(ast.parse(fsrc))),
       "fselect 真的调了 surviving()（以前它定义了零调用）")
    ck("l1_zero" in fsrc and "只记录不剔除" in fsrc,
       "源码里写明第四道只记录不剔除")


def check_target_day() -> None:
    """目标日只能有一份算法，而且日历挂了要退化不要 raise（S20）。

    仓库里有四份「最近一个已收盘交易日」（local_run / evening_check /
    gui.status / backfill），只有 backfill 这份在 ds.trade_dates() 抛异常时
    直接向上抛，stage_update 和 main 都不接，整条晚间线以 traceback 退出 1。
    而 local_run.trade_dates 的文档明写「宁可节假日多跑一次空流程，也不能
    因为日历接口挂了整条线不跑」。
    """
    print("\n[目标日·一份算法]")
    import datetime as dt
    import inspect
    import tempfile
    import backfill as B
    import datasource as ds
    import local_run as LR
    BJ = dt.timezone(dt.timedelta(hours=8))
    cal = {"2026-09-11", "2026-09-14", "2026-09-15", "2026-09-30", "2026-10-08"}
    orig_td, orig_root = ds.trade_dates, LR.ROOT
    # ROOT 指到空临时目录：否则 LR.trade_dates 会读本机真实的
    # state/trade_dates.json，测试就不是在测我给的那份日历
    LR.ROOT = Path(tempfile.mkdtemp(prefix="cal_"))
    ds.trade_dates = lambda: set(cal)
    LR._TD.clear()
    try:
        cases = [(dt.datetime(2026, 9, 14, 16, 30, tzinfo=BJ), "2026-09-14"),
                 (dt.datetime(2026, 9, 14, 14, 0, tzinfo=BJ), "2026-09-11"),
                 (dt.datetime(2026, 9, 15, 7, 30, tzinfo=BJ), "2026-09-14"),
                 (dt.datetime(2026, 9, 13, 12, 0, tzinfo=BJ), "2026-09-11"),
                 (dt.datetime(2026, 10, 1, 17, 0, tzinfo=BJ), "2026-09-30"),
                 (dt.datetime(2026, 10, 7, 9, 0, tzinfo=BJ), "2026-09-30")]
        bad = [(str(n), B.last_closed_trade_day(n), LR.last_closed_trade_day(n), w)
               for n, w in cases
               if not (B.last_closed_trade_day(n)
                       == LR.last_closed_trade_day(n) == w)]
        ck(not bad, f"六个时点两份实现一致且正确（对不上 {bad[:2]}）")

        # 日历接口挂 + 缓存文件也没有：两份都要退化成工作日，不许 raise
        LR._TD.clear()

        def boom():
            raise RuntimeError("新浪日历挂了")

        ds.trade_dates = boom
        n = dt.datetime(2026, 9, 13, 12, 0, tzinfo=BJ)      # 周日
        try:
            got = B.last_closed_trade_day(n)
        except Exception as e:  # noqa: BLE001
            got = f"抛了 {type(e).__name__}"
        ck(got == LR.last_closed_trade_day(n) == "2026-09-11",
           f"日历和缓存都没有时退化成工作日（backfill 给 {got}，不是 traceback）")
    finally:
        ds.trade_dates, LR.ROOT = orig_td, orig_root
        LR._TD.clear()

    # 编排层传进来的目标日要压过自己算的那个
    cs = _real_codes(1100)
    tmp = Path(tempfile.mkdtemp(prefix="tgt_"))
    daily = _fake_daily(cs, ["2026-09-14", "2026-09-15"])
    tds = ["2026-09-14", "2026-09-15", "2026-09-16"]

    def qf(syms):
        return {s: _quote(s[2:], 10.0, 10.0, 1e5, "10.00") for s in syms}

    cap, restore = _offline_update(tmp, daily, qf, cs, "2026-09-15", tds)
    try:
        ck(B.stage_update() == 0
           and "2026-09-16" not in set(
               pd.read_parquet(B.OUT / "daily.parquet")["date"]),
           "不传 target 时按自己算的 09-15 走（已覆盖，不追加）")
        rc = B.stage_update(target="2026-09-16")
        m = pd.read_parquet(B.OUT / "daily.parquet")
        ck(rc == 0 and "2026-09-16" in set(m["date"]),
           "显式传进来的目标日压过自己算的那个")
    finally:
        restore()

    ck("target" in inspect.signature(B.stage_update).parameters,
       "stage_update 收 target（目标日只在编排层算一次）")
    msrc = inspect.getsource(B.main)
    ck("--target" in msrc and "target=a.target" in msrc,
       "CLI 把 --target 透传给 stage_update")
    ck("local_run" in inspect.getsource(B.last_closed_trade_day),
       "backfill 的目标日借 local_run 那一份，不再自己算一遍")


def check_backfill_cli() -> None:
    """--stage 必须显式给，没有 all（S27）。

    默认 all 跑的是「腾讯日线 + 股本 + 股东人数」：腾讯那条是兜底源、
    25~40 分钟，股本那份 build 压根不读（新浪日线自带逐日流通股本）。
    裸跑 backfill.py 会安静地跑错东西，跑完 build 还是报缺列。
    """
    print("\n[回填 CLI·没有 all]")
    import ast
    import contextlib
    import io
    import sys as _sys
    import backfill as B
    tree = ast.parse((ROOT / "src" / "breakout" / "backfill.py")
                     .read_text(encoding="utf-8"))
    arg = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "--stage"):
            arg = {k.arg: k.value for k in n.keywords}
    ck(arg is not None, "找得到 --stage 的定义")
    ck("default" not in arg, "--stage 没有默认值")
    ck(isinstance(arg.get("required"), ast.Constant)
       and arg["required"].value is True, "--stage 是必填")
    choices = [e.value for e in arg["choices"].elts]
    ck("all" not in choices, f"choices 里没有 all（{choices}）")
    ck({"sina", "update", "refresh", "merge"} <= set(choices),
       "在用的几个 stage 一个没少")
    argv, rc = _sys.argv, None
    _sys.argv = ["backfill.py"]
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            B.main()
    except SystemExit as e:
        rc = e.code
    finally:
        _sys.argv = argv
    ck(rc == 2, f"不给 --stage 直接报错退出（实得 {rc}）")
    bsrc = (ROOT / "src" / "breakout" / "build.py").read_text(encoding="utf-8")
    ck("--stage sina" in bsrc and "--stage daily" not in bsrc,
       "build 缺日线时叫人跑 --stage sina（不是腾讯兜底那条）")


def check_holdout_mark() -> None:
    """封存数据的标记要说清用掉它的是哪个模型（S27）。

    旧标记只有时间和 {top_n, quick} 两个开关。2026-09-12 那次 14.43% 后来
    发现协议本身含前视偏差，想追认「当时验收的是哪份特征、哪个训练表」
    都追认不了。
    """
    print("\n[封存数据·标记内容]")
    import json
    import tempfile
    import arena as A
    tmp = Path(tempfile.mkdtemp(prefix="hmark_"))
    out, mark, gh = A.OUT, A.HOLDOUT_MARK, A._git_head
    A.OUT, A.HOLDOUT_MARK = tmp, tmp / "holdout_used.json"
    A._git_head = lambda: "deadbeef1234"
    try:
        A.guard_holdout({"top_n": 10, "quick": False})
        pay = json.loads(A.HOLDOUT_MARK.read_text(encoding="utf-8"))
        ck({"at", "config", "commit"} <= set(pay)
           and pay["commit"] == "deadbeef1234",
           f"守卫一进场就写下时间 / 配置 / 提交（{sorted(pay)}）")
        df = pd.DataFrame({"date": ["2026-01-05", "2026-09-16"],
                           "code": ["600000", "600001"],
                           "a__last": [0.1, 0.2], "b__mean": [0.3, 0.4]})
        A.stamp_holdout(["a__last", "b__mean"], df)
        pay = json.loads(A.HOLDOUT_MARK.read_text(encoding="utf-8"))
        ck({"features", "fingerprint", "rows", "date_range"} <= set(pay),
           f"跑起来之后补记了特征 / 指纹 / 行数 / 区间（{sorted(pay)}）")
        ck(pay["features"] == ["a__last", "b__mean"] and pay["n_features"] == 2
           and pay["commit"] == "deadbeef1234"
           and pay["date_range"] == ["2026-01-05", "2026-09-16"],
           "补记不覆盖先写下的时间和提交")
        raised = None
        try:
            A.guard_holdout({"top_n": 10})
        except SystemExit as e:
            raised = e.code
        ck(raised == 2, f"第二次动 holdout 直接 SystemExit(2)（实得 {raised}）")
        ck(json.loads(A.HOLDOUT_MARK.read_text(encoding="utf-8"))["features"]
           == ["a__last", "b__mean"],
           "被拦下的那次没有把第一次的记录冲掉")
    finally:
        A.OUT, A.HOLDOUT_MARK, A._git_head = out, mark, gh


def check_release_share() -> None:
    """解禁占比是小数，>1 合法，不许按 max() 猜单位（S9）。"""
    print("\n[解禁占比·单位]")
    import ast
    import daily as D
    d = pd.DataFrame({
        "股票代码": ["000695", "301584", "920112", "920112", "688790"],
        "占解禁前流通市值比例": [0.00082, 3.4419, 0.03, 0.03, 0.363],
        "实际解禁市值": [1.54e7, 3.6e10, 8e6, 8e6, 6.76e8]})
    tot = D.release_share_by_code(d)
    ck(abs(tot["301584"] - 3.4419) < 1e-9,
       "比例 >1（首发限售大于现流通盘）原样保留，不除以 100")
    ck(abs(tot["688790"] - 0.363) < 1e-9,
       "同窗口有 >1 的行时，其它票不被整体缩 100 倍")
    ck(tot["000695"] < D.RELEASE_MIN_SHARE <= tot["688790"],
       f"0.08% 放行、36.3% 剔除（门槛 {D.RELEASE_MIN_SHARE:.0%}）")
    ck(abs(tot["920112"] - 0.06) < 1e-9, "同一只票多笔按票累计")
    ck(D.release_share_by_code(pd.DataFrame({"股票代码": ["000695"]})) is None,
       "没有占比列时返回 None（调用方退回「任何解禁都剔」）")

    # 复现旧口径的后果：窗口里有一笔 3.44 的 IPO 解禁，36.3% 那只就漏掉了
    old = d["占解禁前流通市值比例"] / 100.0
    ck(float(old.max()) < D.RELEASE_MIN_SHARE,
       "旧口径（整列 /100）之后全列都够不到 5%，一只都剔不掉")

    src = (ROOT / "src" / "breakout" / "daily.py").read_text(encoding="utf-8")
    ck("share.max()" not in src, "daily.py 里没有按 max() 猜单位的换算")
    rf = next(f for f in ast.walk(ast.parse(src))
              if isinstance(f, ast.FunctionDef) and f.name == "risk_filter")
    calls = {n.func.id for n in ast.walk(rf)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    ck("release_share_by_code" in calls, "risk_filter 走 release_share_by_code")


def check_board_adj_wiring() -> None:
    """板块系数的乘法只有 validate.pick 一处，生产和验收共用（S6）。"""
    print("\n[板块系数·只有一处乘法]")
    import ast
    import daily as D
    import validate as V
    q = np.linspace(0, 1, 101)
    n = 20
    day = pd.DataFrame({
        "date": ["2025-06-02"] * n,
        "code": [f"{600000 + i:06d}" for i in range(n)],
        "board": (["main", "star", "bj", "chinext"] * 5),
        "y_up": [0.0] * n})
    proba = np.linspace(0.999, 0.960, n)
    elig = pd.Series([True] * n, index=day.index)
    elig.iloc[0] = False                 # 次新：预测值最高也不许上

    got = V.pick(day, proba, q, eligible=elig, st=set())["code"].tolist()
    # 独立复算一遍 daily.py 以前那套算法
    adj = day["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
    d2 = day.assign(_p=proba * adj)
    d2["score"] = D.to_score(d2["_p"].to_numpy(), q)
    d2 = d2[elig.to_numpy(bool)].sort_values("_p", ascending=False)
    want = d2[d2["score"] >= D.SCORE_MIN].head(D.CAP_A)["code"].tolist()
    ck(got == want, f"pick 和 daily 那套算法逐位同序（{got[:4]} vs {want[:4]}）")
    ck(day["code"].iloc[0] not in got, "eligible=False 的那只被剔掉了")
    ck(got != V.pick(day, proba, q, board_adj={}, eligible=elig,
                     st=set())["code"].tolist(),
       "不带板块校正时选出来的不一样（证明系数真的参与了排序）")

    # 接线：仓库里不许再有第二处 `.map(BOARD_ADJ)`
    for rel in ("src/breakout/daily.py", "tools/rerun_breakout.py"):
        s = (ROOT / rel).read_text(encoding="utf-8")
        ck(".map(BOARD_ADJ)" not in s and ".map(D.BOARD_ADJ)" not in s,
           f"{rel} 里没有自己乘板块系数那一行")
    dsrc = (ROOT / "src" / "breakout" / "daily.py").read_text(encoding="utf-8")
    fns = {f.name: f for f in ast.walk(ast.parse(dsrc))
           if isinstance(f, ast.FunctionDef)}
    # 2026-09-16 这两处 V.pick 从 stage_scan 挪进了 select_a（S23 把风险检查
    # 范围从「前 60 名」改成「全部够格」，顺手消掉 rerun 里那份副本）
    picks = [c for c in ast.walk(fns["select_a"])
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
             and c.func.attr == "pick"]
    ck(len(picks) == 2,
       f"select_a 里两处 V.pick（够格名单 + 最后的清单），实得 {len(picks)}")
    ck(not [c for c in ast.walk(fns["stage_scan"])
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == "pick"],
       "stage_scan 自己不调 V.pick，全走 select_a")
    rsrc = (ROOT / "tools" / "rerun_breakout.py").read_text(encoding="utf-8")
    ck("D.select_a(" in rsrc and "V.pick(" not in rsrc,
       "rerun 经 D.select_a 走到 V.pick，不自己抄一份")


def check_model_fingerprint() -> None:
    """改了决定特征/标签数值的常量，指纹必须变（S14）。"""
    print("\n[模型指纹·常量与源码]")
    import datetime as dt
    import json
    import logging
    import io
    import tempfile
    import build as BD
    import chips as CH
    import daily as D
    import label as L
    import model as M
    df = pd.DataFrame({"date": ["2026-09-16"] * 3, "code": ["600000"] * 3,
                       "chip_win__last": [0.1, 0.2, 0.3],
                       "vol_ratio20__mean": [1.0, 1.1, 1.2],
                       "turn_pct__slope": [0.0, 0.1, 0.2]})
    base = D.feature_fingerprint(df)
    cases = [(CH, "DECAY", 0.8), (BD, "WIN", 10), (L, "UP_WINDOW", 40),
             (L, "UP_THRESHOLD", 0.3), (L, "DRAWDOWN_END", 0.1),
             (M, "NEG_PER_POS", 5), (M, "SEED", 99), (D, "TRAIN_END_GAP", 5)]
    for mod, name, val in cases:
        old = getattr(mod, name)
        try:
            setattr(mod, name, val)
            ck(D.feature_fingerprint(df) != base,
               f"改 {mod.__name__}.{name} 指纹必须变")
        finally:
            setattr(mod, name, old)
    ck(D.feature_fingerprint(df) == base, "常量还原之后指纹回到原值")

    old_adj = dict(D.BOARD_ADJ)
    try:
        D.BOARD_ADJ["star"] = 0.5
        ck(D.feature_fingerprint(df) == base,
           "改 BOARD_ADJ 指纹不变（它是打分时的乘数，不需要重训）")
    finally:
        D.BOARD_ADJ.clear()
        D.BOARD_ADJ.update(old_adj)

    # AST 兜底：features.py 的滚动窗口是字面量，常量列表钉不住
    here = ROOT / "src" / "breakout"
    srcs = [(here / f).read_text(encoding="utf-8")
            for f in ("features.py", "chips.py", "label.py", "build.py")]
    ck(D.feature_fingerprint(df, sources=srcs) == base,
       "显式传同一份源码时指纹不变（对照）")
    ck(D.feature_fingerprint(df, sources=[srcs[0] + "\n# 只加一行注释\n"] + srcs[1:])
       == base, "只加注释不触发重训（ast.dump 不含注释）")
    import re as _re
    m = _re.search(r"rolling\((\d+)", srcs[0])
    ck(m is not None, "features.py 里有 rolling(<数字>) 这种字面量窗口")
    if m:
        win = srcs[0][:m.start(1)] + str(int(m.group(1)) + 7) + srcs[0][m.end(1):]
        ck(D.feature_fingerprint(df, sources=[win] + srcs[1:]) != base,
           f"改 features.py 的滚动窗口（{m.group(1)} -> {int(m.group(1)) + 7}）指纹要变")

    # 日志：指纹分支不许再落到「超过 30 天上限」那行
    tmp = Path(tempfile.mkdtemp(prefix="fp_"))
    (tmp / "model.txt").write_text("not a real model", encoding="utf-8")
    (tmp / "model.json").write_text(json.dumps(
        {"feats": ["a__last"], "quantiles": [0.0] * 101,
         "fit_date": (D.now_bj().date() - dt.timedelta(days=1)).isoformat(),
         "train_cut": "2026-01-01", "fingerprint": "deadbeef"}),
        encoding="utf-8")

    class _Boom(Exception):
        pass

    def boom(*a, **k):
        raise _Boom()

    state, fs_run = D.STATE, D.FS.run
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    D.STATE, D.FS.run = tmp, boom
    D.log.addHandler(h)
    hit = False
    try:
        D.load_or_fit(df.assign(y_up=0.0))
    except _Boom:
        hit = True
    except Exception as e:  # noqa: BLE001
        print(f"    （load_or_fit 抛的是 {type(e).__name__}: {e}）")
    finally:
        D.log.removeHandler(h)
        D.STATE, D.FS.run = state, fs_run
    out = buf.getvalue()
    ck(hit, "指纹不同时真的走到了重训（FS.run 被调用）")
    ck("指纹变了" in out, f"日志说清是指纹变了（{out.strip()[:60]}）")
    ck("上限" not in out, "不再顺带打印「超过 30 天上限」那句假话")


def check_feature_select_record() -> None:
    """特征筛选的记录要和模型一起落盘，两边对得上（会诊-2）。"""
    print("\n[特征筛选记录·和模型同源]")
    import json
    import tempfile
    import daily as D
    rng = np.random.default_rng(23)
    # 六个月：IC 稳定性那道筛要每月 ≥500 行、至少 3 个算得出 IC 的月份，
    # 月份不够的话 keep 会是空的，LightGBM 直接报「0 feature」
    dates = (pd.date_range("2025-01-01", "2025-06-30", freq="B")
             .strftime("%Y-%m-%d").tolist())
    per = 40
    sig = rng.normal(size=(len(dates) * per, 4))
    z = sig @ np.array([2.0, 1.8, 1.6, 1.4]) + rng.normal(0, 1.0, len(sig))
    df = pd.DataFrame({"date": np.repeat(dates, per),
                       "code": np.tile([f"{i:06d}" for i in range(per)],
                                       len(dates)),
                       "y_up": (z > np.quantile(z, 0.95)).astype(float)})
    for i in range(4):
        df[f"sig{i}__last"] = sig[:, i]
    for i in range(4):
        df[f"noise{i}__last"] = rng.normal(size=len(df))
    tmp = Path(tempfile.mkdtemp(prefix="fsrec_"))
    state = D.STATE
    D.STATE = tmp
    try:
        obj = D.load_or_fit(df, force=True, gap=5)
        meta = json.loads((tmp / "model.json").read_text(encoding="utf-8"))
        rep = json.loads((tmp / "feature_select.json").read_text(encoding="utf-8"))
    finally:
        D.STATE = state
    ck(set(meta["feats"]) <= set(rep["keep"]) and len(meta["feats"]) > 0,
       f"model.json 的 {len(meta['feats'])} 个特征都在 feature_select.json 的 keep 里")
    ck(meta["fingerprint"] == rep["fingerprint"],
       "两份产物的指纹一致（能看出是不是同一次训练留下的）")
    ck(meta.get("select_file") == "feature_select.json",
       "model.json 指得出筛选记录在哪")
    ck(set(meta.get("importance_gain", {})) == set(meta["feats"]),
       "importance_gain 的键就是入模的那些特征（不装 lightgbm 也能读）")
    ck(rep["start"] >= rep["end"] == len(rep["keep"]),
       f"start {rep['start']} -> end {rep['end']} 和 keep 对得上")
    ck(obj["feats"] == meta["feats"], "返回值和落盘的特征表一致")


def check_send_marker() -> None:
    """真发出去才落 out_breakout/mail_sent.json（F5-3，教训 27）。"""
    print("\n[发信标记·只认真发了]")
    import json
    import os
    import tempfile
    import daily as D
    import export as E
    tmp = Path(tempfile.mkdtemp(prefix="sent_"))
    meta = {"date": "2026-09-16", "n_a": 1, "n_b": 0, "score_min": 97,
            "cap_a": 10, "rejected": 0, "model_date": "2026-09-15"}
    (tmp / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    pd.DataFrame({"code": ["600000"], "name": ["测试"], "score": [98.0],
                  "streak": [1], "close": [10.0], "board": ["main"]}).to_json(
        tmp / "list_a.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(columns=["code", "name", "best", "days", "streak", "close",
                          "rise", "drop"]).to_json(
        tmp / "list_b.json", orient="records", force_ascii=False, indent=2)
    cnt = {"n": 0}

    def fake_send(m, c):
        cnt["n"] += 1

    out, hist, lenv = D.OUT, D.recent_history, D.load_env
    conf, send, bht = E._conf, E._send, E.board_hit_table
    skip = os.environ.pop("SKIP_MAIL", None)
    D.OUT, D.recent_history, D.load_env = tmp, (lambda n=30: []), (lambda: None)
    E._conf = lambda: {"user": "a@b.c", "to": ["a@b.c"]}
    E._send = fake_send
    E.board_hit_table = lambda: {}        # 不碰 out_breakout/board_hit.json
    try:
        rc = D.stage_send()
        ms = json.loads((tmp / "mail_sent.json").read_text(encoding="utf-8"))
        ck(rc == 0 and cnt["n"] == 1 and ms["date"] == meta["date"],
           "真发信后落 mail_sent.json，日期和清单一致")
        (tmp / "mail_sent.json").unlink()
        os.environ["SKIP_MAIL"] = "1"
        rc = D.stage_send()
        ck(rc == 0 and cnt["n"] == 1 and not (tmp / "mail_sent.json").exists(),
           "SKIP_MAIL 下退出码一样是 0，但不落标记（宣布发了才是最坏的）")
        os.environ.pop("SKIP_MAIL")

        def boom(m, c):
            raise ConnectionRefusedError("SMTP 挂了")

        E._send = boom
        raised = False
        try:
            D.stage_send()
        except ConnectionRefusedError:
            raised = True
        ck(raised and not (tmp / "mail_sent.json").exists(),
           "SMTP 异常时不落标记")
    finally:
        D.OUT, D.recent_history, D.load_env = out, hist, lenv
        E._conf, E._send, E.board_hit_table = conf, send, bht
        os.environ.pop("SKIP_MAIL", None)
        if skip is not None:
            os.environ["SKIP_MAIL"] = skip


def check_evening_decide() -> None:
    """云端托底只认 state/sent 标记，不认 run_meta 的日期（F5-2）。"""
    print("\n[云端托底·判据]")
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location(
        "evening_check_t", ROOT / "tools" / "evening_check.py")
    ec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ec)
    m = {"date": "2026-09-16", "dry": False}
    ck(ec.decide(m, True, "2026-09-16")[0] is False, "sent 标记在 -> 不提醒")
    need, why = ec.decide(m, False, "2026-09-16")
    ck(need is True and why == ec.SENT_FAILED,
       "run_meta 是目标日但没有 sent 标记 -> 提醒「算了没发出去」")
    ck(ec.decide({**m, "dry": True}, False, "2026-09-16")
       == (True, ec.NOT_RUN), "dry 的 run_meta 不算跑完")
    ck(ec.decide({"date": "2026-09-15"}, False, "2026-09-16")
       == (True, ec.NOT_RUN), "run_meta 是昨天 -> 本机没跑")
    s1, _ = ec.build_mail("2026-09-16", m, ec.SENT_FAILED)
    s2, b2 = ec.build_mail("2026-09-16", m, ec.NOT_RUN)
    ck("没发出去" in s1 and "没跑" in s2 and s1 != s2, "两种原因的主题不一样")
    ck("多半是没开机" in b2, "本机没跑那封还是老说法")
    msrc = inspect.getsource(ec.main)
    ck("decide(" in msrc and "state/sent/breakout_" in msrc,
       "main 真的走 decide + sent 标记（防止有人把判据写回 run_meta）")
    ck(msrc.index("_origin_json(") < msrc.index("_origin_has(f\"state/sent"),
       "_origin_has 不自己 fetch，必须排在 _origin_json 之后")


def check_rerun_pool_seed() -> None:
    """重算/补发的 A 池要拿窗口之前的清单做种，不是从空池起算（S17）。"""
    print("\n[A 池做种·窗口之前不丢]")
    import json
    import tempfile
    import daily as D
    tmp = Path(tempfile.mkdtemp(prefix="pool_"))
    data, state = D.DATA, D.STATE
    D.DATA, D.STATE = tmp / "data", tmp / "state"
    (D.DATA / "2026-09").mkdir(parents=True)
    D.STATE.mkdir(parents=True)
    try:
        def put(date, code):
            pd.DataFrame({"code": [code], "name": ["测试"], "date": [date],
                          "score": [99.0], "streak": [1],
                          "close": [10.0]}).to_parquet(
                D.DATA / date[:7] / f"breakout_{date}.parquet", index=False)

        put("2026-09-01", "600000")
        put("2026-09-02", "600001")
        seed = D.replay_pool(before="2026-09-03")
        ck(set(seed) == {"600000", "600001"}, f"窗口之前两只都在（{sorted(seed)}）")
        ck(seed["600000"]["first"] == "2026-09-01", "first 是它真正第一次上榜那天")
        (D.STATE / "a_pool.json").write_text(json.dumps(seed), encoding="utf-8")
        newp = pd.DataFrame({"code": ["600002"], "name": ["新"],
                             "date": ["2026-09-03"], "score": [98.0],
                             "streak": [1], "close": [9.0]})
        pool = D.update_pool(newp, "2026-09-03")
        ck(set(pool) == {"600000", "600001", "600002"},
           f"续算之后三只都在（{sorted(pool)}）")
        ck(pool["600000"]["first"] == "2026-09-01"
           and pool["600000"]["days"] == 1,
           "窗口之前那只的 first / days 没被改（从空池起算会全错）")
        # 回放 == 逐日推进，这条把「按文件折叠出来的池」钉成不变量
        put("2026-09-03", "600002")
        step: dict = {}
        for d_, c_ in (("2026-09-01", "600000"), ("2026-09-02", "600001"),
                       ("2026-09-03", "600002")):
            step = D.pool_step(step, pd.read_parquet(
                D.DATA / d_[:7] / f"breakout_{d_}.parquet"), d_)
        ck(D.replay_pool() == step, "不带 before 的回放和逐日推进逐键相等")
    finally:
        D.DATA, D.STATE = data, state
    for rel in ("tools/rerun_breakout.py", "tools/resend_breakout.py"):
        s = (ROOT / rel).read_text(encoding="utf-8")
        ck("replay_pool(" in s, f"{rel} 用 replay_pool 做种（不是空池）")


def check_rerun_train_gap() -> None:
    """重算 N 天时训练截止要为**最早**那天留够余量（S17）。"""
    print("\n[重算·标签不许伸进重算窗口]")
    import inspect
    import daily as D
    import label as LB
    dates = (pd.date_range("2025-01-01", periods=60, freq="B")
             .strftime("%Y-%m-%d").tolist())
    days = 7

    def lab_end(gap: int) -> str:
        cut = dates[-gap]
        return dates[min(dates.index(cut) - 1 + LB.UP_WINDOW, len(dates) - 1)]

    ck(lab_end(D.TRAIN_END_GAP) > dates[-days],
       f"默认 gap={D.TRAIN_END_GAP} 时模型标签看到 {lab_end(D.TRAIN_END_GAP)}，"
       f"晚于重算起点 {dates[-days]}（重叠就在这）")
    g2 = D.TRAIN_END_GAP + days - 1
    ck(lab_end(g2) <= dates[-days],
       f"gap={g2} 时标签只到 {lab_end(g2)}，不碰重算窗口")
    ck("gap" in inspect.signature(D.load_or_fit).parameters,
       "load_or_fit 收 gap")
    s = (ROOT / "tools" / "rerun_breakout.py").read_text(encoding="utf-8")
    ck("gap=gap" in s and "a.days - 1" in s,
       "rerun 按 --days 把 gap 推出去（写死 25 这条立刻红）")


def check_expected_board() -> None:
    """期望命中率要按本份清单的板块构成加权（会诊-1）。"""
    print("\n[期望命中率·按板块构成加权]")
    import json
    import re
    import export as E
    tbl = {"overall": {"n": 783, "hit": 0.1264},
           "boards": {"main": {"n": 520, "hits": 86, "hit": 0.16538},
                      "star": {"n": 150, "hits": 4, "hit": 0.02667},
                      "bj": {"n": 113, "hits": 9, "hit": 0.07965}}}

    def mk(boards):
        return pd.DataFrame({"code": [f"{600000 + i:06d}" for i in
                                      range(len(boards))],
                             "name": ["x"] * len(boards),
                             "score": [98.0] * len(boards),
                             "streak": [1] * len(boards),
                             "close": [10.0] * len(boards),
                             "board": boards})

    bht = E.board_hit_table
    E.board_hit_table = lambda: tbl
    try:
        e_star, comp, fb = E.expected_for(mk(["star"] * 10))
        ck(abs(e_star - 2.667) < 0.01 and not fb,
           f"全科创的清单期望 {e_star:.1f}%（不是整体 12.6%）")
        ck(comp == "科创 10", f"构成文字用中文板块名（{comp}）")
        e_main, _, _ = E.expected_for(mk(["main"] * 10))
        ck(abs(e_main - 16.538) < 0.01, f"全主板的清单期望 {e_main:.1f}%")
        e_mix, comp2, _ = E.expected_for(mk(["star"] * 6 + ["main"] * 4))
        ck(abs(e_mix - (0.6 * 16.538 * 0 + 6 * 2.667 / 10 + 4 * 16.538 / 10)) < 0.01,
           f"混合构成按名额加权（{e_mix:.1f}%）")
        ck("科创 6" in comp2 and "主板 4" in comp2, f"构成把两个板块都列出来（{comp2}）")
        _, _, fb2 = E.expected_for(mk(["chinext"] * 5))
        ck(fb2, "验证集里没样本的板块（创业板）标成兜底")
        ck(E.expected_for(mk(["main"] * 3).drop(columns=["board"]))
           == (E.perf_row(1)[1], "", True),
           "清单没有 board 列（老产物）时退回整体平均，不许崩")
        E.board_hit_table = lambda: {}
        ck(E.expected_for(mk(["star"] * 10))[0] == E.perf_row(1)[1],
           "整张表读不到时退回整体平均")
        E.board_hit_table = lambda: tbl
        html = E._body("2026-09-16", mk(["star"] * 6 + ["main"] * 4),
                       pd.DataFrame(columns=["code"]),
                       {"score_min": 97, "cap_a": 10, "model_date": "2026-09-15"},
                       False)
        # 2026-09-16 用户定：抬头先说「是不是满员日 + 按近期基准折算」，
        # 按板块构成加权那个数降级成参考，所以断言跟着改
        ck("本份构成" in html
           and ("满员日" in html or "门槛卡得住" in html)
           and ("折算" in html or "基准还在累积" in html),
           "抬头印了满员与否、按近期基准折算、以及本份构成")
        ck(f"{E.BASE}%" in html, "同一行还印了同期全市场基准")
    finally:
        E.board_hit_table = bht

    # 板块表和 W5 那几行同源：n 必须对得上
    bh = ROOT / "out_breakout" / "board_hit.json"
    wg = ROOT / "out_breakout" / "window_grid.json"
    if bh.exists() and wg.exists():
        real = json.loads(bh.read_text(encoding="utf-8"))
        grid = json.loads(wg.read_text(encoding="utf-8"))["grid"]
        w5 = {int(re.search(r"连续≥(\d)天", r["label"]).group(1)): r
              for r in grid if r["kind"] == "W5"}
        ck(int(real["overall"]["n"]) == int(w5[1]["n"]),
           f"board_hit.json 的名额数 {real['overall']['n']} 和 W5 连续≥1 天"
           f" {w5[1]['n']} 同源")
        ck(abs(sum(b["n"] for b in real["boards"].values())
               - real["overall"]["n"]) == 0, "各板块名额加起来等于总数")
    else:
        ck(True, "没有 board_hit.json / window_grid.json，跳过同源核对")



def check_board_shrink() -> None:
    """板块系数的经验贝叶斯收缩（board_adj.py）的三条性质。"""
    print("\n[板块系数·收缩估计]")
    import board_adj as BA
    # 1. 板块之间看不出真实差异 -> tau2=0 -> 因子全是 1（自动退化成不校正）
    same = {"main": (50, 500), "star": (20, 200), "bj": (5, 50)}
    f = BA.shrink_factors(same)
    ck(all(abs(v - 1.0) < 1e-9 for v in f.values()) and len(f) == 3,
       f"命中率相同时因子全为 1（拿到 {f}）")
    d = BA.diagnostics(same)
    ck(abs(d["tau2"]) < 1e-12, f"命中率相同时 tau2=0（拿到 {d['tau2']:.2e}）")
    # 2. 样本越小被信得越少。真实数字：main 75/466、star 14/211、bj 5/19
    real = {"main": (75, 466), "star": (14, 211), "bj": (5, 19)}
    d = BA.diagnostics(real)
    w = {k: v["shrink_weight"] for k, v in d["boards"].items()}
    ck(w["main"] > w["star"] > w["bj"],
       f"样本大的被信得多：main {w['main']:.2f} > star {w['star']:.2f} > bj {w['bj']:.2f}")
    # 北交所裸因子 1.95，收缩后必须明显往 1 靠
    bj = d["boards"]["bj"]
    ck(abs(bj["factor"] - 1.0) < abs(bj["raw_factor"] - 1.0) * 0.8,
       f"小样本板块被拉回全市场：裸 {bj['raw_factor']:.2f} -> {bj['factor']:.2f}")
    # 3. 因子恒在夹子里，且样本不够时什么都不给
    for clamp in ((0.5, 1.5), (0.6, 1.4), (0.8, 1.25)):
        f = BA.shrink_factors(real, clamp=clamp)
        ck(all(clamp[0] - 1e-9 <= v <= clamp[1] + 1e-9 for v in f.values()),
           f"因子都在夹子 {clamp} 内（{ {k: round(v, 2) for k, v in f.items()} }）")
    ck(BA.shrink_factors({"main": (5, 30), "star": (2, 20)}) == {},
       "总名额不够 MIN_TOTAL 时返回空（= 不校正）")
    ck(BA.shrink_factors({"main": (75, 466)}) == {},
       "只有一个板块时返回空（估不出板块间方差）")


def check_rule_fingerprint() -> None:
    """成绩表的失效保护必须看得见板块系数（2026-09-16 对齐检查 3/4/5）。"""
    print("\n[成绩表·规则指纹含板块系数]")
    import export as E
    grid = dict(E.PERF_GRID)
    try:
        E.PERF_GRID.clear()
        E.PERF_GRID.update({"board_adj": {"main": 1.19, "star": 1.27,
                                          "bj": 0.9, "chinext": 0.59}})
        same = {"score_min": E.PERF_RULE[0], "cap_a": E.PERF_RULE[1],
                "board_adj": {"main": 1.19, "star": 1.27, "bj": 0.9, "chinext": 0.59}}
        ck(E._same_rule(same), "同一条规则判 True")
        diff = {**same, "board_adj": {**same["board_adj"], "star": 1.0}}
        ck(not E._same_rule(diff),
           "板块系数改了就判 False（star 1.27->1.0 换掉了近一半的榜）")
        ck(not E._same_rule({**same, "cap_a": 5}), "上限改了仍然判 False")
        # 老产物没有 board_adj 时退回只比 (分数线, 上限)
        E.PERF_GRID.clear()
        ck(E._same_rule(diff), "grid 里没有 board_adj 时退回老行为")
    finally:
        E.PERF_GRID.clear()
        E.PERF_GRID.update(grid)


def main() -> int:
    t0 = time.time()
    check_chips()
    check_chip_causal()
    check_chip_grid_overflow()
    check_send_roundtrip()
    check_streak_calendar()
    check_pool_expiry_calendar()
    check_merge_authority()
    check_clean_bars()
    check_refresh_rc()
    check_holders_refresh()
    check_holders_partial()
    check_refetch_short()
    check_update_share_drift()
    check_update_coverage()
    check_update_gap()
    check_update_status()
    check_target_day()
    check_backfill_cli()
    check_float_mcap()
    check_split_volume()
    check_history_alignment()
    check_common_start()
    check_eligible_rule()
    check_pick_rule()
    check_risk_scope()
    check_backtest_universe()
    check_acceptance_base()
    check_board_factors()
    check_purge_suspension()
    check_fselect_purge()
    check_fselect()
    check_holdout_mark()
    check_labels()
    check_label_isolation()
    check_lookahead()
    check_holder_alignment()
    check_holder_preipo()
    check_market_breadth()
    check_warmup_nan()
    check_cross_section()
    check_wiring()
    check_interactions()
    check_quote_wiring()
    check_release_share()
    check_board_adj_wiring()
    check_model_fingerprint()
    check_feature_select_record()
    check_rerun_pool_seed()
    check_rerun_train_gap()
    check_send_marker()
    check_evening_decide()
    check_expected_board()
    check_panel_html()
    check_board_shrink()
    check_rule_fingerprint()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {len(fails)} 个")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
