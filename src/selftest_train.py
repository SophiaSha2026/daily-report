"""
回填训练表的口径自测（离线、不联网、不碰 state/ 和生产目录）。

    python src/selftest_train.py

钉的全是同一件事：**回测口径必须等于生产口径**（CLAUDE.md 历史教训 30）。
2026-09-16 那轮审计在这条线上一次查出八处，全都不报错、只是悄悄学歪：

  · stk_auction_o 的 bar 是「09:25 撮合价 -> 09:30 之后」，不是竞价段。
    以前把它的 close 当撮合价、open/vwap/close 当 T1/T2/T3，于是 slope
    算的是开盘后的涨跌，和标签 r 相关 +0.15（前视泄漏），而和线上真 slope
    相关 -0.03。
  · 回填池多一条生产没有的入池规则、少一条生产有的（换手率），
    两个池子日均只重合 70%。
  · prev_amount 是 收盘×量 估算，昨日涨停那一子群偏高 2.79%，
    它是量能准入的分母。
  · 除权日留下 236 行 gap_pct 低到 -66.76% 的不可能样本，次日整行丢掉。

所有用例都是合成数据，产物只写 tempfile。跑完 git status 必须没有变化
（历史教训 17）。
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

import cfg as C
import datasource as ds
import premarket
from learn import backfill as BF
from learn import objective as O
from learn import sources as SRC

ROOT = Path(__file__).resolve().parent.parent
BF_SRC = (ROOT / "src" / "learn" / "backfill.py").read_text(encoding="utf-8")


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


class Counter:
    bad = 0


def ck(ok: bool, msg: str) -> None:
    if not ok:
        Counter.bad += 1
    print(f"  {'✓' if ok else '✗'} {msg}")


# ---------------------------------------------------------------------
#  合成数据
# ---------------------------------------------------------------------
def row(code: str, date: str, close: float, chg: float, **kw) -> dict:
    """一根日线。列名用 hist_daily 的中文原名，daily_features 自己 rename。"""
    return {"code": code, "日期": date, "开盘": kw.get("open", close),
            "收盘": close, "最高": kw.get("high", close + 0.1),
            "最低": kw.get("low", close - 0.1),
            "成交量": kw.get("vol", 1.0e5),
            "成交额": kw.get("amount", 1.0e8),
            "换手率": kw.get("turn", 1.0), "涨跌幅": chg}


def days(n: int, start: int = 1) -> list[str]:
    """n 个连续日期字符串。只要能排序就行，不必是真交易日。"""
    out = []
    for i in range(start, start + n):
        out.append(f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}")
    return sorted(out)


def flat(code: str, dates: list[str], close: float = 10.0, **kw) -> list[dict]:
    return [row(code, d, close, 0.0, **kw) for d in dates]


def feats(rows: list[dict], auc: pd.DataFrame | None = None,
          names: dict | None = None, look: int = 20) -> pd.DataFrame:
    """走一遍完整链路：日线特征 -> 接竞价 -> 组特征。"""
    names = names or {}
    d = BF.daily_features(pd.DataFrame(rows), look, names)
    d = BF.merge_auction(d, auc)
    d = BF.merge_cauc(d, None)
    return BF.to_features(d, {}, names)


def auc_row(code: str, date: str, **kw) -> pd.DataFrame:
    """一行 stk_auction_o。open 才是撮合价，其余字段是撮合之后的。"""
    return pd.DataFrame([{
        "ts_code": f"{code}.SH", "trade_date": date.replace("-", ""),
        "open": kw.get("open", 10.2), "close": kw.get("close", 10.5),
        "high": kw.get("high", 10.6), "low": kw.get("low", 9.9),
        "vwap": kw.get("vwap", 10.4), "vol": kw.get("vol", 1.0e5),
        "amount": kw.get("amount", 1.05e6)}])


# ---------------------------------------------------------------------
#  1. 竞价口径：撮合价取 open，撮合之后的字段一个都不许进特征
# ---------------------------------------------------------------------
def check_auction_caliber() -> None:
    print("\n竞价口径（F2-1 / F8-1 / F8-2 / F8-3）")
    ds_ = days(8)
    rows = flat("600000", ds_[:-1], 10.0, high=10.3)
    # 最后一天：撮合价 10.2，收盘 11.0。ts 的 close=10.5 是开盘后的价
    rows.append(row("600000", ds_[-1], 11.0, 10.0, open=10.2, high=11.2,
                    low=10.1))
    a = auc_row("600000", ds_[-1], open=10.2, close=10.5, high=10.6, low=9.9,
                vwap=10.4, amount=1.05e6)
    f = feats(rows, a).iloc[-1]

    ck(abs(f.auc_price - 10.2) < 1e-9, "撮合价取 ts.open（不是 close 10.5）")
    ck(abs(f.gap_pct - 2.0) < 1e-9, "gap_pct = 2.0（按 close 算会是 5.0）")
    ck(abs(f.auc_amount - 1.05e6) < 1e-6, "竞价额接的是 ts.amount")
    ck(f.t1_chg == f.t2_chg == f.t3_chg == f.gap_pct,
       "t1/t2/t3 一律等于撮合价涨幅（历史拿不到三个采样点）")
    ck(f.slope == 0.0 and f.dive == 0.0, "slope / dive 给中性 0，不给假信号")
    ck(bool(f.monotonic) is False,
       "monotonic=False，和线上 traj_ok=False 同一口径（写 True 是白送趋势分）")

    merged = BF.merge_auction(pd.DataFrame(rows).assign(date="x"), a)
    leak = {"auc_close", "auc_high", "auc_low", "auc_vwap", "auc_open"}
    ck(not (set(merged.columns) & leak),
       "撮合之后的字段根本不 merge 进来（留着就会有人再用）")

    body = BF_SRC.split("def to_features(")[1].split("\ndef ")[0]
    ck(not any(k in body for k in ("auc_high", "auc_vwap", "auc_open",
                                   "auc_low")),
       "to_features 源码里不出现任何撮合后字段")
    ren = BF_SRC.split("def merge_auction(")[1].split("\ndef ")[0]
    ck('"open": "auc_price"' in ren and '"close"' not in ren,
       "merge_auction 的 rename：open -> auc_price，没有 close")

    # 恒等式：线上 slope 恒等于 t3-t1，回填以前 48.8% 的行不成立
    g = feats(rows, a)
    ck(float((g["t3_chg"] - g["t1_chg"] - g["slope"]).abs().max()) < 1e-9,
       "slope == t3_chg - t1_chg 逐行成立（F2-3：算完 slope 又改 t1）")
    tree = ast.parse(BF_SRC)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "to_features")
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and n.slice.value == "t1_chg"
            and isinstance(n.ctx, ast.Store)]
    ck(len(hits) == 1, f"t1_chg 只被赋值一次（发现 {len(hits)} 次，M12）")

    # M3：「稳步抬升」只能有一个口径。生产以前写的是 t1 <= t2+0.05 <= t3+0.10
    # （容差 0.05 个百分点，还按价格漂），回填是严格的 T1<=T2<=T3，
    # 同一列两种定义拼进一张训练表。现在生产统一走 score.is_monotonic，
    # 回填因为根本没有轨迹证据一律 False（和线上 traj_ok=False 同一口径）
    from score import is_monotonic
    ck(bool(is_monotonic(1.0, 1.0, 1.0))
       and not bool(is_monotonic(0.09, 0.21, 0.19)),
       "score.is_monotonic 是唯一定义：平盘算抬升，真实回落不算")
    ra = (ROOT / "src" / "run_auction.py").read_text(encoding="utf-8")
    ck("is_monotonic(" in ra and "t2 + 0.05" not in ra,
       "run_auction 不再有 0.05 的容差字面量")
    ck('d["monotonic"] = False' in BF_SRC and "0.05" not in
       BF_SRC.split("def to_features(")[1].split("\ndef ")[0],
       "回填的 monotonic 恒 False，没有自己那份容差")


def check_one_word() -> None:
    print("\n一字板判据（M12 / F8-1）")
    ds_ = days(8)
    rows = flat("600000", ds_[:-1], 10.0)
    rows.append(row("600000", ds_[-1], 11.0, 10.0, open=11.0))
    a = auc_row("600000", ds_[-1], open=11.0, low=9.0)   # low 是开盘后的，不该看
    f = feats(rows, a).iloc[-1]
    ck(bool(f.one_word), "撮合价即涨停价 -> 一字板（不再要求段内最低价）")
    body = BF_SRC.split('d["one_word"]')[1].split("\n\n")[0]
    ck("auc_low" not in body, "one_word 不读 auc_low")

    rows2 = flat("600001", ds_[:-1], 10.0)
    rows2.append(row("600001", ds_[-1], 10.3, 3.0, open=10.3))
    f2 = feats(rows2, auc_row("600001", ds_[-1], open=10.3)).iloc[-1]
    ck(not bool(f2.one_word), "高开 3% 不是一字板")


# ---------------------------------------------------------------------
#  2. 涨停幅度 / ST：只留一份实现
# ---------------------------------------------------------------------
def check_limit_pct() -> None:
    print("\n涨停幅度与 ST（M11）")
    codes = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
             .str.zfill(6).tolist())
    ck(all(BF._limit_pct_by_code(x) == ds.limit_pct(x, "") for x in codes),
       f"回填涨停幅度与生产 limit_pct 对代码表逐只一致（{len(codes)} 只）")
    # ST 按板块（2026-09-16 实测），5% 只剩未股改 S 股
    for code, name, want in (("000078", "ST海王", 10.0),
                             ("600000", "*ST 浦发", 10.0),
                             ("300068", "ST南都", 20.0),
                             ("688053", "ST思科瑞", 20.0),
                             ("920023", "*ST田野", 30.0),
                             ("600182", "S佳通", 5.0),
                             ("920159", "x", 30.0),
                             ("832735", "x", 30.0), ("688655", "x", 20.0),
                             ("300017", "x", 20.0), ("689009", "x", 20.0)):
        ck(BF._limit_pct_by_code(code, name) == ds.limit_pct(code, name) == want,
           f"{code}/{name or '无名'} -> {want}")

    ds_ = days(8)
    rows = flat("600000", ds_[:-1], 10.0) + flat("600001", ds_[:-1], 10.0)
    rows.append(row("600000", ds_[-1], 10.48, 4.8, open=10.48))
    rows.append(row("600001", ds_[-1], 10.48, 4.8, open=10.48))
    names = {"600000": "S甲", "600001": "乙"}     # S 股 5% vs 普通 10%
    f = feats(rows, None, names)
    last = f[f["date"] == ds_[-1]].set_index("code")
    ck(last.loc["600000", "limit_pct"] == 5.0
       and last.loc["600001", "limit_pct"] == 10.0,
       "未股改 S 股 5%，普通票 10%（names 为空时一律按板块）")
    ck(bool(last.loc["600000", "one_word"]),
       "S 股高开 4.8% 已经贴上 5% 涨停 -> 一字板")
    ck(not bool(last.loc["600001", "one_word"]),
       "同一个高开幅度在 10cm 票上不是一字板（limit_pct 真的参与了判定）")

    rows2 = flat("600002", ds_[:-1], 10.0)
    rows2.append(row("600002", ds_[-1], 10.48, 4.8, open=10.48))
    f2 = feats(rows2, None, {"600002": "*ST 丙"})
    ck(float(f2[f2["date"] == ds_[-1]]["limit_pct"].iloc[0]) == 10.0,
       "ST 不再降到 5%：主板 ST 也是 10%（201 只 ST 里 131 只走过 >5.5%）")


def check_names_guard() -> None:
    print("\n名字拿不到就不出表（M11）")
    c = cfg()
    try:
        BF.require_names({}, c)
        ck(False, "names 为空 + exclude_st 时应 raise")
    except RuntimeError:
        ck(True, "names 为空 + exclude_st -> RuntimeError，拒绝落表")
    c2 = {"universe": {"exclude_st": False}}
    BF.require_names({}, c2)
    ck(True, "exclude_st=False 时不拦（这条规则本来就不执行）")
    ck("holders.parquet" in BF_SRC, "load_names 有离线兜底（names.csv 不存在）")


# ---------------------------------------------------------------------
#  3. 候选池：和生产同一份规则
# ---------------------------------------------------------------------
def _pool_cases() -> tuple[list[str], list[dict]]:
    codes = ["600000", "600001", "600002", "600003", "600004", "600005",
             "600006", "600007", "920159"]
    cases = [
        {"name": "换手5",  "turn": 5.0, "amount": 1.0e6, "chg": 0.0},
        {"name": "换手49", "turn": 4.9, "amount": 1.1e6, "chg": 0.0},
        {"name": "昨涨5",  "turn": 0.5, "amount": 1.2e6, "chg": 5.0},
        {"name": "昨涨停", "turn": 0.5, "amount": 1.3e6, "chg": 10.0},
        {"name": "昨跌停", "turn": 9.0, "amount": 1.4e6, "chg": -9.5},
        {"name": "ST丙",   "turn": 9.0, "amount": 1.5e6, "chg": 0.0},
        {"name": "C丁",    "turn": 9.0, "amount": 1.6e6, "chg": 0.0},
        {"name": "巨量戊", "turn": 0.5, "amount": 9.9e9, "chg": 0.0},
        {"name": "北交所己", "turn": 9.0, "amount": 1.7e6, "chg": 0.0},
    ]
    return codes, cases


def check_pool_parity() -> None:
    print("\n候选池与生产同口径（F8-6 / M1）")
    c = cfg()
    c["universe"]["include_if"]["amount_rank_top"] = 1     # 只让巨量那只靠名次
    codes, cases = _pool_cases()

    spot = pd.DataFrame([{"代码": codes[i], "名称": x["name"], "最新价": 10.0,
                          "涨跌幅": x["chg"], "成交额": x["amount"],
                          "换手率": x["turn"], "总市值": float("nan")}
                         for i, x in enumerate(cases)])
    prod = set(premarket.stage1(spot, c)["code"])

    d1, d2 = days(2)
    rows = []
    for i, x in enumerate(cases):
        # 前一天承载「昨日」那一组值，第二天才是要入池的那一行
        rows.append(row(codes[i], d1, 10.0, x["chg"], amount=x["amount"],
                        turn=x["turn"]))
        rows.append(row(codes[i], d2, 10.3, 3.0, open=10.3, amount=1.0e8,
                        turn=1.0))
    names = {codes[i]: x["name"] for i, x in enumerate(cases)}
    f = feats(rows, None, names)
    pool = set(BF.build_pool(f, c)["code"])

    ck(prod == pool, f"两边入池集合完全相同（生产 {sorted(prod)} / "
                     f"回填 {sorted(pool)}）")
    for i, why in ((0, "只靠换手进池"), (2, "只靠昨涨 5% 进池"),
                   (3, "只靠昨日涨停进池"), (7, "只靠成交额名次进池")):
        ck(codes[i] in pool, f"每条入池规则都有用例真的碰到它：{why}")
    for i, why in ((1, "四条都不沾"), (4, "昨日跌停"), (5, "ST"), (6, "次新")):
        ck(codes[i] not in pool, f"剔除规则生效：{why}")

    ck(codes[8] in pool, "北交所默认保留（exclude_bj=false）")

    # 关掉/打开一条规则，两边必须同步变化（防止将来只改一边）
    c2 = cfg()
    c2["universe"]["include_if"]["turnover_pct_gte"] = None
    c2["universe"]["include_if"]["amount_rank_top"] = 1
    prod2 = set(premarket.stage1(spot, c2)["code"])
    pool2 = set(BF.build_pool(f, c2)["code"])
    ck(prod2 == pool2 and codes[0] not in pool2,
       "关掉换手率那条：两边同步少掉同一批票")
    c3 = cfg()
    c3["universe"]["exclude_bj"] = True
    c3["universe"]["include_if"]["amount_rank_top"] = 1
    prod3 = set(premarket.stage1(spot, c3)["code"])
    pool3 = set(BF.build_pool(f, c3)["code"])
    ck(prod3 == pool3 and codes[8] not in pool3,
       "exclude_bj=True：两边用同一条代码段判据（920 段也认）")

    body = BF_SRC.split("def build_pool(")[1].split("\ndef ")[0]
    ck("include_mask" in body and "amount_ratio_5d" not in body,
       "build_pool 调 premarket.include_mask，不再有自己的 amount_ratio_5d 分支")
    pm_tree = ast.parse((ROOT / "src" / "premarket.py")
                        .read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(pm_tree)
              if isinstance(n, ast.FunctionDef) and n.name == "include_mask")
    nums = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)]
    ck(all(v == 0.0 for v in nums),
       f"include_mask 里没有写死的阈值（发现常量 {nums}）")


def check_amount_ratio_denominator() -> None:
    print("\n昨日放量比值的分母（F8-6）")
    amt = [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]
    dts = days(len(amt))
    rows = [row("600000", dts[i], 10.0, 0.0, amount=amt[i])
            for i in range(len(amt))]
    d = BF.daily_features(pd.DataFrame(rows), 20, {})
    got = float(d["amount_ratio_5d"].iloc[-1])
    ck(abs(got - 3.0) < 1e-9, f"分母不含昨日：3.0（拿到 {got:.4f}）")
    ck(abs(got - 3.0 / 1.4) > 1e-6, "不是旧口径的 3/1.4 = 2.143")
    # 和生产 stage2 那一行逐位对齐
    am = pd.Series(amt[:-1])
    base = float(am.tail(6).iloc[:-1].mean())
    ck(abs(float(am.iloc[-1] / base) - got) < 1e-9,
       "和 premarket.stage2 的 am.tail(6).iloc[:-1] 逐位相等")
    amt2 = [1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0]
    rows2 = [row("600000", dts[i], 10.0, 0.0, amount=amt2[i])
             for i in range(len(amt2))]
    d2 = BF.daily_features(pd.DataFrame(rows2), 20, {})
    ck(abs(float(d2["amount_ratio_5d"].iloc[-1]) - 100.0) < 1e-9,
       "高倍区不失真（旧口径被数学压死在 5 以内，实测上限 4.968）")


def check_min_listed_days() -> None:
    print("\n次新剔除（F8-7）")
    c = cfg()
    c["universe"]["include_if"]["amount_rank_top"] = 600
    dts = days(180)
    rows = flat("600000", dts, 10.0, turn=6.0)                 # 老票，全程在表
    rows += flat("600001", dts[100:], 10.0, turn=6.0)          # 窗口内才上市
    names = {"600000": "老票", "600001": "新股"}
    f = feats(rows, None, names)
    pool = BF.build_pool(f, c)
    old = pool[pool["code"] == "600000"]
    new = pool[pool["code"] == "600001"]
    ck(set(old["hist_i"]) >= set(range(1, 60)),
       "老票前 60 行一行不少（裸 cumcount<60 会砍掉 15% 的表）")
    ck(new["hist_i"].min() == 60 and len(new) == len(dts) - 100 - 60,
       f"新股上市第 60 根起才进池（实得 {sorted(new['hist_i'])[:3]}…）")
    ck(not set(new["hist_i"]) & {1, 2, 3, 4},
       "上市第 2~5 天（生产里名字带 C 的那几天）一行不剩")
    c2 = cfg()
    c2["universe"]["min_listed_days"] = 10
    c2["universe"]["include_if"]["amount_rank_top"] = 600
    p2 = BF.build_pool(f, c2)
    ck(p2[p2["code"] == "600001"]["hist_i"].min() == 10,
       "阈值读 config（=10 时边界跟着变，不许再退化成死配置）")
    ck('u["min_listed_days"]' in BF_SRC
       and 'c["universe"]["min_listed_days"]' in
       (ROOT / "src" / "premarket.py").read_text(encoding="utf-8"),
       "两边读的是同一个配置键 universe.min_listed_days")


# ---------------------------------------------------------------------
#  4. 除权日
# ---------------------------------------------------------------------
def check_exdiv() -> None:
    print("\n除权日（F8-5）")
    c = cfg()
    c["universe"]["include_if"]["amount_rank_top"] = 600
    dts = days(4)
    rows = [row("600000", dts[0], 20.0, 0.0, open=20.0, turn=6.0),
            row("600000", dts[1], 20.0, 0.0, open=20.0, turn=6.0),
            # 10 送 10：原始口径 -50%，数据源给的 chg 就是这个数
            row("600000", dts[2], 10.0, -50.0, open=10.05, turn=6.0),
            row("600000", dts[3], 11.0, 10.0, open=10.10, turn=6.0)]
    names = {"600000": "甲"}
    f = feats(rows, None, names).set_index("date")
    ck(bool(np.isnan(f.loc[dts[2], "gap_pct"])),
       "除权日整行判废（以前留下 gap_pct -49.75 这种不可能样本）")
    ck(bool(np.isnan(f.loc[dts[3], "prev_gain"])),
       "除权次日的 prev_gain 置 NaN（那个 -50 不是真跌幅）")
    pool = BF.build_pool(feats(rows, None, names), c)
    ck(dts[3] in set(pool["date"]),
       "除权次日仍然入池（以前被「昨日跌停」整行丢掉，698 天里 0 行留下）")
    ok = pool["gap_pct"].abs() <= pool["limit_pct"] + 1e-6
    ck(bool(ok[pool["gap_pct"].notna()].all()),
       "池子里不存在 |高开| > 涨跌停幅度 的行")
    pm = (ROOT / "src" / "premarket.py").read_text(encoding="utf-8")
    ck('~d["prev_gain"].le(' in BF_SRC and '~df["prev_gain"].le(' in pm,
       "昨日跌停两边都是 `~le`（NaN 留下），不是 `gt`（NaN 丢掉）")
    # 反解昨收只许在 chg_adj=True 的行上做。以前无条件反解，而 100% 的数据
    # 是不复权口径，那段 np.where 是恒等变换（等于没修）
    body = BF_SRC.split("def daily_features(")[1].split("\ndef ")[0]
    ck("chg_adj" in body and "adj & d[\"chg\"].notna()" in body,
       "prev_close 的反解由 chg_adj 把门（不复权源一律退回 shift(1)）")
    ck("~adj" in body, "除权日判据只对不复权源生效（复权源没有假跌幅）")


def check_chg_adj() -> None:
    """三路日线的「涨跌幅」是两种语义，靠 chg_adj 区分（F6-5）。"""
    print("\n涨跌幅的复权语义（F6-5）")
    c = cfg()
    c["universe"]["include_if"]["amount_rank_top"] = 600
    dts = days(4)
    # 不复权源（腾讯/新浪）：10 送 10 那天 chg 是原始比值 −50%
    raw = [row("600000", dts[0], 20.0, 0.0, open=20.0, turn=6.0),
           row("600000", dts[1], 20.0, 0.0, open=20.0, turn=6.0),
           row("600000", dts[2], 10.0, -50.0, open=10.05, turn=6.0),
           row("600000", dts[3], 11.0, 10.0, open=10.10, turn=6.0)]
    for r in raw:
        r["chg_adj"] = False
    f = feats(raw, None, {"600000": "甲"}).set_index("date")
    ck(bool(np.isnan(f.loc[dts[2], "gap_pct"])),
       "不复权源：除权日整行判废（chg_adj=False 走退化判据）")
    ck(bool(np.isnan(f.loc[dts[3], "prev_gain"])), "除权次日 prev_gain 置 NaN")

    # 复权源（东财）：同一个除权日 chg 是真实涨跌幅 0%，昨收要反解成 5.0
    adj = [row("600001", dts[0], 20.0, 0.0, open=20.0, turn=6.0),
           row("600001", dts[1], 20.0, 0.0, open=20.0, turn=6.0),
           row("600001", dts[2], 5.0, 0.0, open=5.0, turn=6.0),
           row("600001", dts[3], 5.5, 10.0, open=5.05, turn=6.0)]
    for r in adj:
        r["chg_adj"] = True
    g = feats(adj, None, {"600001": "乙"}).set_index("date")
    ck(abs(float(g.loc[dts[2], "prev_close"]) - 5.0) < 1e-9,
       f"复权源：昨收由 chg 反解 = 5.0（拿到 {g.loc[dts[2], 'prev_close']}）")
    ck(abs(float(g.loc[dts[2], "gap_pct"])) < 1e-9,
       "复权源：除权日 gap_pct = 0（不是 −75%），整行照常留下")
    ck(abs(float(g.loc[dts[3], "prev_gain"])) < 1e-9,
       "复权源：除权次日 prev_gain = 0（不是 −50，不会被当昨日跌停丢掉）")
    ck(bool(g.loc[dts[3], "prev_limit_up"]) is False
       and bool(g.loc[dts[3], "is_limit_up"]),
       "复权源：+10% 那天判涨停，前一天不判")

    # 缺 chg_adj 列（老的 hist_daily.parquet）要退回不复权口径，不能崩
    plain = [dict(r) for r in raw]
    for r in plain:
        r.pop("chg_adj")
    f2 = feats(plain, None, {"600000": "甲"}).set_index("date")
    ck(bool(np.isnan(f2.loc[dts[2], "gap_pct"])),
       "老表没有 chg_adj 列时按不复权处理（退化，不报错）")

    ck("chg_adj" in (ROOT / "src" / "learn" / "sources.py")
       .read_text(encoding="utf-8"),
       "sources._normalize 把 chg_adj 留在长表里（丢了就整表退化）")


def check_broken_board() -> None:
    """昨日炸板的涨停价判据：−0.01 会整体放宽一分（M5）。"""
    print("\n昨日炸板的涨停价（M5）")
    g = np.random.default_rng(11)
    pcs = np.round(g.uniform(1.0, 300.0, 2000), 2)
    for pct in (5.0, 10.0, 20.0, 30.0):
        arr = ds.limit_price_arr(pd.Series(pcs), pct).to_numpy()
        one = np.array([ds.limit_price(float(x), pct) for x in pcs])
        ck(bool(np.all(np.abs(arr - one) < 1e-12)),
           f"向量化涨停价 == limit_price（{pct}cm，2000 个随机昨收）")

    # 北交所封板价向下取整到分（F6-3）：向量化版必须和标量版逐位一致，
    # 否则生产 premarket 判「昨日炸板」和回填判同一件事差 0.01
    bj = pd.Series(["920002", "920006", "920083", "920090"])
    pcs_bj = pd.Series([84.63, 17.09, 31.46, 2.85])
    arr_bj = ds.limit_price_arr(pcs_bj, 30.0, bj).to_numpy()
    want_bj = np.array([110.01, 22.21, 40.89, 3.70])
    ck(bool(np.all(np.abs(arr_bj - want_bj) < 1e-9)),
       f"向量化北交所涨停价向下取整（拿到 {list(np.round(arr_bj, 2))}）")
    mixed = pd.Series(["600000", "920002"])
    arr_mx = ds.limit_price_arr(pd.Series([10.05, 84.63]),
                                pd.Series([10.0, 30.0]), mixed).to_numpy()
    ck(abs(arr_mx[0] - 11.06) < 1e-9 and abs(arr_mx[1] - 110.01) < 1e-9,
       "同一张表里沪深四舍五入、北交所向下取整，互不串味")
    ck('d["code"]' in BF_SRC.split('d["prev_broken_board"]')[1].split("\n\n")[0],
       "回填把 code 传进 limit_price_arr（不传就全按四舍五入算）")

    dts = days(4)
    # 600654 2026-09-14：昨收 3.09，涨停价 3.40，最高 3.39 -> 不算触及
    for hi, want, why in ((3.39, False, "最高 3.39 < 涨停价 3.40"),
                          (3.40, True, "最高 3.40 == 涨停价")):
        rows = [row("600000", dts[0], 3.00, 0.0, turn=6.0),
                row("600000", dts[1], 3.09, 3.0, turn=6.0),
                row("600000", dts[2], 3.20, 3.5, high=hi, turn=6.0),
                row("600000", dts[3], 3.25, 1.5, open=3.25, turn=6.0)]
        d = BF.daily_features(pd.DataFrame(rows), 20, {})
        got = bool(d[d["date"] == dts[3]]["prev_broken_board"].iloc[0])
        ck(got is want, f"回填 prev_broken_board：{why} -> {want}")
    body = BF_SRC.split('d["prev_broken_board"]')[1].split("\n\n")[0]
    ck("limit_price_arr" in body and "- 0.01" not in body,
       "回填用 limit_price_arr，源码里不再出现 −0.01 的放宽")
    pm = (ROOT / "src" / "premarket.py").read_text(encoding="utf-8")
    # 只看代码行，注释里会引用那个旧写法
    seg = [x for x in pm.split("昨日炸板")[1].split("rec.append")[0].splitlines()
           if x.strip() and not x.strip().startswith("#")]
    ck(any("limit_price(" in x for x in seg)
       and not any("- 0.01" in x for x in seg),
       f"生产 premarket 那一处也是 limit_price（两边同一份公式）：{seg}")


def check_one_word_masked() -> None:
    """回填和线上的 one_word 定义不同，靠涨幅上限遮蔽（M10）。"""
    print("\n一字板两个定义的遮蔽边界（M10）")
    c = cfg()
    # 10 是非 ST 最小的涨停幅度，0.3 是回填 one_word 的容差。两边只在
    # gap_pct >= 9.7 时才可能为真；准入上限一旦抬到这里，407621 行里
    # 455 行（0.11%）的分歧就不再被 hard_reject 的涨幅区间盖住
    ck(c["screen"]["gap_pct_max"] < 10.0 - 0.3,
       f"gap_pct_max={c['screen']['gap_pct_max']} < 9.7：两个定义仍被完全遮蔽")

    dts = days(8)
    # 行 A：撮合价 10.98（回填判 True：gap 9.8 >= 10−0.3；线上判 False）
    # 行 B：撮合价 11.00（线上判 True：== 涨停价；回填也 True）
    rows = flat("600000", dts[:-1], 10.0) + flat("600001", dts[:-1], 10.0)
    rows.append(row("600000", dts[-1], 10.98, 9.8, open=10.98))
    rows.append(row("600001", dts[-1], 11.00, 10.0, open=11.00))
    f = feats(rows, None, {}).set_index("code")
    f = f[f["date"] == dts[-1]] if "date" in f.columns else f
    ck(bool(f.loc["600000", "one_word"]) and bool(f.loc["600001", "one_word"]),
       "回填按 gap >= limit−0.3 判：10.98 和 11.00 都算一字板")
    online = [abs(p - ds.limit_price(10.0, 10.0)) < 0.005
              for p in (10.98, 11.00)]
    ck(online == [False, True],
       "线上按 |撮合价 − 涨停价| < 0.005 判：只有 11.00 算（定义确实不同）")

    # 翻转 one_word 不改变 vscore 的剔除结果 —— 分歧全被涨幅上限盖住
    from learn import vscore as V
    fv = f.reset_index()
    fv["sector_members"] = 0
    fv["sector_prev_limitups"] = 0
    arr = V.prepare(fv)
    base = V.hard_reject(dict(arr), c["screen"])
    for v in (True, False):
        d2 = dict(arr)
        d2["one_word"] = np.full(len(base), v)
        ck(bool(np.array_equal(V.hard_reject(d2, c["screen"]), base))
           and bool(np.all(base)),
           f"one_word 全置 {v} 时剔除结果不变（两行都因涨幅超区间被剔）")


def check_labels() -> None:
    """标签只在收盘后打（F2-6），口径改了要能强制重打（F2-11）。"""
    import datetime as dtm
    import tempfile
    print("\n标签：收盘闸与强制覆盖（F2-6 / F2-11）")
    from learn import labels as L

    bj = dtm.timezone(dtm.timedelta(hours=8))
    calls: list[int] = []

    class Q:
        def __init__(self, code, ts):
            self.code, self.ts = code, ts
            self.open_, self.price, self.prev_close = 10.0, 10.5, 9.8

    orig = ds.fetch_quotes

    def fake(ts):
        def _f(syms, **k):
            calls.append(1)
            return {s: Q(s[2:], ts) for s in syms}
        return _f

    try:
        ds.fetch_quotes = fake("20260916100000")
        r = L.from_quotes(["600000"], "2026-09-16",
                          now=dtm.datetime(2026, 9, 16, 10, 0, tzinfo=bj))
        ck(r.empty and not calls, "盘中 10:00 给当天打标 -> 空表，且根本不发请求")
        r = L.from_quotes(["600000"], "2026-09-16",
                          now=dtm.datetime(2026, 9, 16, 16, 0, tzinfo=bj))
        ck(len(r) == 1 and bool(calls) and abs(float(r["close"].iloc[0]) - 10.5) < 1e-9,
           "收盘后 16:00 -> 照常返回")
        ds.fetch_quotes = fake("20260915150316")
        r = L.from_quotes(["600000"], "2026-09-15",
                          now=dtm.datetime(2026, 9, 16, 6, 30, tzinfo=bj))
        ck(len(r) == 1, "次日开盘前补昨天：ts 仍是昨天，允许")
        ds.fetch_quotes = fake("20260916100000")
        r = L.from_quotes(["600000"], "2026-09-15",
                          now=dtm.datetime(2026, 9, 16, 10, 0, tzinfo=bj))
        ck(r.empty, "隔天盘中补昨天：ts 是今天，全部丢弃（原有守卫仍在）")
    finally:
        ds.fetch_quotes = orig

    # 强制覆盖。标签目录换成临时目录，跑完还原（教训 17）
    old_dir = L.LABEL_DIR
    L.LABEL_DIR = Path(tempfile.mkdtemp(prefix="lab_"))
    try:
        d = "2099-01-01"

        def lab(n_ok: int, n_dirty: int, r0: float) -> pd.DataFrame:
            return pd.DataFrame({
                "date": [d] * (n_ok + n_dirty),
                "code": [f"{600000 + i}" for i in range(n_ok + n_dirty)],
                "open": [10.0] * (n_ok + n_dirty),
                "close": [10.0 * (1 + r0)] * (n_ok + n_dirty),
                "r": [r0] * (n_ok + n_dirty),
                "open_mismatch_pct": [0.0] * (n_ok + n_dirty),
                "dirty": [False] * n_ok + [True] * n_dirty})

        p, w = L.save(d, lab(100, 0, 0.0))
        ck(w and int((~pd.read_parquet(p)["dirty"]).sum()) == 100, "首次落盘")
        _, w = L.save(d, lab(90, 10, 0.0))
        ck(not w and int((~pd.read_parquet(p)["dirty"]).sum()) == 100,
           "可用行更少 -> 不覆盖（守卫仍在）")
        _, w = L.save(d, lab(90, 10, 0.0), force=True)
        ck(w and int((~pd.read_parquet(p)["dirty"]).sum()) == 90,
           "force=True -> 强制覆盖（口径改了要重打）")
        _, w = L.save(d, lab(90, 0, 0.01))
        ck(w and abs(float(pd.read_parquet(p)["r"].iloc[0]) - 0.01) < 1e-12,
           "可用行相等就覆盖（守卫只认严格少于）")
    finally:
        import shutil
        shutil.rmtree(L.LABEL_DIR, ignore_errors=True)
        L.LABEL_DIR = old_dir

    src = (ROOT / "src" / "learn" / "labels.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "save")
    ck(any(a.arg == "force" for a in fn.args.kwonlyargs),
       "save 的 force 是关键字参数（位置传参会被静默吃掉）")
    fq = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "from_quotes")
    body = ast.get_source_segment(src, fq) or ""
    ck("CLOSE_HM" in body and body.index("CLOSE_HM") < body.index("fetch_quotes"),
       "收盘闸写在 fetch_quotes 之前（先判再联网）")


def check_dataset_coverage() -> None:
    """标签覆盖不到快照时要出声（F2-11 附带）。"""
    import logging
    import tempfile
    print("\n标签覆盖率告警（F2-11）")
    from learn import dataset as DS
    from learn import labels as L

    tmp = Path(tempfile.mkdtemp(prefix="dsv_"))
    old_data, old_lab = DS.DATA, L.LABEL_DIR
    logs: list[str] = []

    class Grab(logging.Handler):
        def emit(self, rec):
            logs.append(rec.getMessage())

    h = Grab()
    DS.log.addHandler(h)
    try:
        DS.DATA = tmp
        L.LABEL_DIR = tmp / "labels"
        d = "2099-01-02"
        codes = [f"{600000 + i}" for i in range(300)]
        snap = pd.DataFrame({"code": codes, "auc_price": [10.0] * 300,
                             "one_word": [False] * 300,
                             "t1_chg": np.linspace(0, 3, 300),
                             "t2_chg": np.linspace(0, 3, 300),
                             "t3_chg": np.linspace(0.5, 3.5, 300)})
        (tmp / d[:7]).mkdir(parents=True, exist_ok=True)
        snap.to_parquet(tmp / d[:7] / f"auction_{d}.parquet", index=False)
        lb = pd.DataFrame({"date": [d] * 200, "code": codes[:200],
                           "r": np.linspace(-0.02, 0.02, 200),
                           "dirty": [False] * 200})
        (L.LABEL_DIR / d[:7]).mkdir(parents=True, exist_ok=True)
        lb.to_parquet(L.LABEL_DIR / d[:7] / f"label_{d}.parquet", index=False)
        out = DS.build([d], {"min_pool": 100})
        ck(len(out) == 200, f"inner merge 只剩 200 行（拿到 {len(out)}）")
        ck(any("只覆盖快照" in m for m in logs),
           "缩池要出声：标签可能是旧快照打的")
    finally:
        DS.log.removeHandler(h)
        DS.DATA, L.LABEL_DIR = old_data, old_lab
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def check_table_guard() -> None:
    print("\n落盘前的硬不变量")
    base = pd.DataFrame({"gap_pct": [3.0, -2.0], "limit_pct": [10.0, 10.0],
                         "slope": [0.0, 0.0], "monotonic": [False, False]})
    BF.check_table(base)
    ck(True, "干净的表通过")
    for name, patch in (("高开超过涨跌停幅度", {"gap_pct": [66.0, -2.0]}),
                        ("非零 slope", {"slope": [0.3, 0.0]}),
                        ("monotonic=True", {"monotonic": [True, False]})):
        bad = base.copy()
        for k, v in patch.items():
            bad[k] = v
        try:
            BF.check_table(bad)
            ck(False, f"{name} 应该拒绝落盘")
        except ValueError:
            ck(True, f"{name} -> 拒绝落盘")


def check_open_mismatch() -> None:
    """买入价与撮合价的偏离必须入表（F8-14）。

    回填表以前只有一字板一条脏样本规则，在线 labels.build 还判「日线开盘价
    与 auc_price 偏离 > 0.5%」：买入价和特征描述的价不是同一个价，那一行的
    标签就是错的。40.76 万行里这样的有 46190 行（11.53%），准入区间内占
    21.4%。缺了这一列，eval_daily._bf_dirty 只能退回旧口径（打一行 warning），
    两条线的可用样本不是同一批（教训 30）。
    """
    print("\n开盘价失配入表（F8-14）")
    ck("open_mismatch_pct" in BF.COLS,
       "open_mismatch_pct 进 COLS（不入表 = 下游一行都判不了）")
    seg = BF_SRC.split('d["open_mismatch_pct"] =')[1].split("\n\n")[0]
    ck(".abs()" in seg,
       "偏离取绝对值：不带 .abs() 的话 open 低于撮合价的那一半行是负数，"
       "`> 阈值` 永远不成立，实测漏掉六成脏行")

    ds_ = days(8)
    c = cfg()
    thr = float(c["learning"]["label"]["max_open_mismatch_pct"])

    def one(open_px: float, auc: float | None, whole: bool = False):
        rows = flat("600000", ds_[:-1], 10.0, high=10.3)
        rows.append(row("600000", ds_[-1], 11.0, 10.0, open=open_px,
                        high=11.2, low=10.1))
        a = auc_row("600000", ds_[-1], open=auc) if auc is not None else None
        g = feats(rows, a)
        return g if whole else g.iloc[-1]

    # COLS 里的每一列都得真的被链路产出，否则 build() 在 d[COLS] 那一步
    # KeyError —— 往 COLS 里加名字和往 to_features 里加计算是两件事
    full = BF.add_sector_stats(one(10.302, 10.2, whole=True), c)
    ck(not [k for k in BF.COLS if k not in full.columns],
       "COLS 里每一列链路都产出（含新加的 open_mismatch_pct）")

    hi = one(10.302, 10.2)      # 开盘比撮合价高 1.0%
    lo = one(10.098, 10.2)      # 开盘比撮合价低 1.0%（不带 .abs() 就漏掉）
    none_ = one(10.2, None)     # 没竞价数据：auc_price 由 open 兜底，必然相等
    ck(abs(hi.open_mismatch_pct - 1.0) < 1e-6,
       "偏离 = |open − auc_price| / auc_price × 100 = 1.0")
    ck(abs(lo.open_mismatch_pct - 1.0) < 1e-6, "开盘价低于撮合价时同样是 +1.0")
    ck(abs(none_.open_mismatch_pct) < 1e-12, "免费源（无竞价）那路偏离 0，不判脏")

    # 口径必须和在线 labels.build 逐位一致，两边是同一个公式
    from learn import labels as L
    lab = L.build(ds_[-1],
                  pd.DataFrame({"code": ["600000"], "auc_price": [10.2],
                                "one_word": [False]}),
                  pd.DataFrame({"code": ["600000"], "open": [10.302],
                                "close": [11.0]}), thr)
    ck(abs(float(lab["open_mismatch_pct"].iloc[0])
           - float(hi.open_mismatch_pct)) < 1e-9,
       "和在线 labels.build 逐位同口径")

    # 真的被判脏：_bf_dirty 是训练表可用样本的唯一口径
    import eval_daily as ED
    frame = pd.DataFrame({
        "one_word": [False, False, False, False],
        "open_mismatch_pct": [hi.open_mismatch_pct, lo.open_mismatch_pct,
                              none_.open_mismatch_pct, thr / 2]})
    got = ED._bf_dirty(frame, c).tolist()
    ck(got == [True, True, False, False],
       f"偏离 1% 的两行（高开/低开）被判脏，0 和 {thr / 2}% 的不判（实得 {got}）")


def check_auction_price_guard() -> None:
    print("\n撮合价哨兵（F2-1）")
    n = 1000
    d = pd.DataFrame({"auc_price": [10.0] * n, "open": [10.0] * n})
    ck(BF.check_auction_price(d) == 0.0, "逐行相等时不一致率 0")
    d2 = d.copy()
    d2.loc[:4, "auc_price"] = 10.5          # 0.5% 的行差 5%
    ck(BF.check_auction_price(d2) <= 0.01, "0.5% 的行不一致：放行（容忍噪声）")
    d3 = d.copy()
    d3.loc[:400, "auc_price"] = 10.5        # 旧口径（取 ts.close）就是这个量级
    try:
        BF.check_auction_price(d3)
        ck(False, "40% 的行不一致时应报错")
    except ValueError:
        ck(True, "40% 的行不一致 -> 报错（数据源口径变了）")


# ---------------------------------------------------------------------
#  5. 成交额真值
# ---------------------------------------------------------------------
def check_real_amount() -> None:
    print("\n成交额与换手率取真值（F8-4）")
    dts = days(2)
    h = pd.DataFrame([row("600000", dts[0], 11.0, 0.0, vol=1000.0,
                          amount=11.0 * 1000 * 100),
                      row("600000", dts[1], 11.0, 0.0, vol=1000.0,
                          amount=11.0 * 1000 * 100)])
    real = pd.DataFrame({"date": dts, "code": ["600000", "600000"],
                         "amount": [1_050_000.0, 1_050_000.0],
                         "turnover": [0.06, 0.06]})
    h2 = BF.attach_daily_truth(h, real)
    ck(abs(float(h2["成交额"].iloc[0]) - 1_050_000.0) < 1e-6,
       "成交额换成新浪真值（不再是 收盘×量×100 = 1.1e6）")
    ck(abs(float(h2["换手率"].iloc[0]) - 6.0) < 1e-9,
       "换手率按百分点入表（源是小数 0.06）")
    d = BF.daily_features(h2, 20, {})
    ck(abs(float(d["prev_amount"].iloc[-1]) - 1_050_000.0) < 1e-6,
       "prev_amount 用真值（auc_ratio 的分母就是它）")
    ck(abs(float(d["prev_turnover_pct"].iloc[-1]) - 6.0) < 1e-9,
       "prev_turnover_pct 供入池规则用")

    h3 = BF.attach_daily_truth(h, real.iloc[:0])
    ck(abs(float(h3["成交额"].iloc[0]) - 11.0 * 1000 * 100) < 1e-6,
       "真值缺失时保留估算，不报错")

    # 哪几行没接上真值要留痕（F6-7）。实测覆盖 99.97%，但训练表里必须能把
    # 估算行单独挑出来，否则「回测口径 == 生产口径」这句话没法核对
    ck(bool(h3["amount_est"].all()), "一行真值都没接上 -> amount_est 全 True")
    ck("amount_est" in h2.columns and not bool(h2["amount_est"].iloc[0]),
       "接上真值的行 amount_est=False")
    two = pd.DataFrame([row("600000", dts[0], 11.0, 0.0, vol=1000.0,
                            amount=11.0 * 1000 * 100),
                        row("600000", dts[1], 11.0, 0.0, vol=1000.0,
                            amount=11.0 * 1000 * 100),
                        row("000001", dts[0], 11.0, 0.0, vol=1000.0,
                            amount=11.0 * 1000 * 100),
                        row("000001", dts[1], 11.0, 0.0, vol=1000.0,
                            amount=11.0 * 1000 * 100)])
    only_one = real[real["code"] == "600000"]
    m = BF.daily_features(BF.attach_daily_truth(two, only_one), 20, {})
    hit = m[(m["code"] == "600000") & (m["date"] == dts[1])].iloc[0]
    miss = m[(m["code"] == "000001") & (m["date"] == dts[1])].iloc[0]
    ck(abs(float(hit["prev_amount"]) - 1_050_000.0) < 1e-6
       and not bool(hit["prev_amount_est"]),
       "有真值那只：prev_amount 是真值、prev_amount_est=False")
    ck(abs(float(miss["prev_amount"]) - 1.1e6) < 1e-6
       and bool(miss["prev_amount_est"]),
       "没真值那只：prev_amount 是估算、prev_amount_est=True")
    ck("prev_amount_est" in BF.COLS,
       "标记入表（prev_amount 是昨天的，标记也 shift(1) 过）")
    # 接线：覆盖那一步被顺手删掉的话，上面几条断言一条都发现不了
    bsrc = BF_SRC.split("def build(")[1]
    ck("attach_daily_truth(" in bsrc,
       "build() 里真的调了 attach_daily_truth（成交额/换手率的真值覆盖）")
    dsrc = (ROOT / "src" / "datasource.py").read_text(encoding="utf-8")
    ck("cl * vol * 100.0" not in dsrc
       and dsrc.count("(hi + lo + cl) / 3.0 * vol * 100.0") == 2,
       "datasource 的估算式改成 (高+低+收)/3×量（收盘×量 在涨停票偏高 2.79%）")
    ck('columns=["date", "code", "amount", "turnover"]' in BF_SRC,
       "只从新浪日线店取 amount/turnover（它的 close 是复权、volume 未复权）")


# ---------------------------------------------------------------------
#  6. 可学维度与箱约束
# ---------------------------------------------------------------------
def check_learnable_box() -> None:
    print("\n可学维度与箱约束裁剪")
    c = cfg()
    box = c["learning"]["box"]
    t0 = C.theta0(box)
    ck(set().union(*(set(v) for v in SRC.DIM_PARAMS.values())) == set(box),
       "DIM_PARAMS 恰好覆盖箱里的全部参数")

    r = SRC.restrict_box(box, t0, dims=SRC.DIMS_H)
    for k in ("scoring.weights.trend", "scoring.weights.volume",
              "screen.auc_ratio_score_hi", "screen.auc_ratio_decay"):
        ck(r[k] == [t0[k], t0[k]], f"{k} 被钉死在人工基线（不可学）")
    for k in ("scoring.weights.gap", "screen.gap_pct_peak",
              "scoring.weights.sector"):
        ck(r[k] == list(box[k]), f"{k} 仍然可学")
    ck(set(r) == set(box),
       "钉死而不是删键：删了会让剩下四个权重被归一到 1，总权重变成 1.4")

    g = np.random.default_rng(3)
    for _ in range(50):
        t = {k: float(g.uniform(lo - 0.3, hi + 0.3)) for k, (lo, hi) in box.items()}
        p = O.project(t, r)
        wk = [k for k in p if k.startswith("scoring.weights.")]
        if abs(sum(p[k] for k in wk) - 1.0) > 1e-9:
            ck(False, "裁过的箱仍然满足 Σw=1")
            break
        if abs(p["scoring.weights.trend"] - t0["scoring.weights.trend"]) > 1e-9:
            ck(False, "trend 权重被钉住不动")
            break
    else:
        ck(True, "裁过的箱：Σw=1 仍成立，且 trend/volume 一步都迈不出去")

    rv = SRC.restrict_box(box, t0, dims=SRC.DIMS_H + ["volume"])
    ck(rv["scoring.weights.volume"] == list(box["scoring.weights.volume"])
       and rv["scoring.weights.trend"] == [t0["scoring.weights.trend"]] * 2,
       "换到纯竞价额的源（H+V）时 volume 解锁、trend 仍锁")

    ck(SRC.dims_from_cfg(c) == ["gap", "position", "sector", "continuity"],
       "可学维度以 config.learning.backfill.learnable_dims 为准（以前是死键）")
    c2 = {"learning": {"backfill": {"learnable_dims": ["gap", "volume"]}}}
    ck(SRC.dims_from_cfg(c2) == ["gap", "volume"], "改配置真的能改可学维度")

    src = (ROOT / "src" / "learn" / "sources.py").read_text(encoding="utf-8")
    ck('"o": "H"' in src and '"o": "FULL"' not in src,
       "stk_auction_o 的完整度是 H 不是 FULL（它的竞价额含开盘后成交）")
    ck("trend" not in str(SRC.DIMS_H), "trend 不在任何历史源的可学集合里")


# ---------------------------------------------------------------------
#  7. 真实缓存上的数据级核对（文件缺了就跳过，自测不许依赖 data/）
# ---------------------------------------------------------------------
def check_cache_invariants() -> None:
    print("\n真实缓存核对（缺文件则跳过）")
    ap = ROOT / "cache" / "hist_auction.parquet"
    hp = ROOT / "cache" / "hist_daily.parquet"
    if not (ap.exists() and hp.exists()):
        print("  - 跳过：cache 里没有历史竞价/日线")
        return
    a = pd.read_parquet(ap, columns=["ts_code", "trade_date", "open", "close"])
    a["code"] = a["ts_code"].str.slice(0, 6)
    a["date"] = a["trade_date"].astype(str).str.replace(
        r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True)
    h = pd.read_parquet(hp, columns=["日期", "开盘", "code"])
    h = h.rename(columns={"日期": "date", "开盘": "open_d"})
    h["date"] = h["date"].astype(str).str[:10]
    m = a.merge(h, on=["code", "date"], how="inner")
    m = m[m["open_d"] > 0]
    same_open = float((np.abs(m["open"] / m["open_d"] - 1.0) < 0.005).mean())
    same_close = float((np.abs(m["close"] / m["open_d"] - 1.0) < 0.005).mean())
    ck(same_open >= 0.99,
       f"ts.open == 日线开盘 {same_open:.4f}（>=0.99）")
    ck(same_close < 0.95,
       f"ts.close 只有 {same_close:.4f} 等于开盘 —— 它不是撮合价，别再当撮合价用")


def main() -> int:
    t0 = time.time()
    check_auction_caliber()
    check_one_word()
    check_limit_pct()
    check_names_guard()
    check_pool_parity()
    check_amount_ratio_denominator()
    check_min_listed_days()
    check_exdiv()
    check_chg_adj()
    check_broken_board()
    check_one_word_masked()
    check_labels()
    check_dataset_coverage()
    check_table_guard()
    check_open_mismatch()
    check_auction_price_guard()
    check_real_amount()
    check_learnable_box()
    check_cache_invariants()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {Counter.bad} 个")
    return 1 if Counter.bad else 0


if __name__ == "__main__":
    sys.exit(main())
