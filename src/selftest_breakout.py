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
    ck(all(n not in d.columns for _, _, n in F.INTERACTIONS),
       "交互项在这一步还不该存在（它在三层变换之后）")
    names = [n for _, _, n in F.INTERACTIONS]
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


def main() -> int:
    t0 = time.time()
    check_chips()
    check_labels()
    check_lookahead()
    check_cross_section()
    check_wiring()
    check_quote_wiring()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {len(fails)} 个")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
