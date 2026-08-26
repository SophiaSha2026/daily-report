"""
形态扫描的离线自测。不联网，构造合成日线覆盖每一条判定分支。

存在的理由和 selftest.py 一样：本地和沙箱都访问不了国内行情源，形态判定
如果只能在 GitHub Actions 上、只能在交易日、只能等到 17:00 才验证一次，
那基本等于没有测试。三段条件里任何一条写反了，线上表现都是「今天没票」，
而「今天没票」本来就是常态，错了根本看不出来。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import yaml                                            # noqa: E402
from pullback import find_pattern, score_one, norm_hist, hist_turnover  # noqa: E402


def cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def mk(days: list[dict]) -> pd.DataFrame:
    """days 按时间升序，每项给 gain/vol/low/high/close，缺的自动补。"""
    rec = []
    for i, d in enumerate(days):
        close = d.get("close", 10.0)
        rec.append({
            "日期": f"2026-08-{i + 1:02d}",
            "开盘": d.get("open", close),
            "收盘": close,
            "最高": d.get("high", close * 1.02),
            "最低": d.get("low", close * 0.98),
            "成交量": d["vol"],
            "涨跌幅": d.get("gain", 0.0),
        })
    return pd.DataFrame(rec)


# 一段标准形态：索引 -5 是启动日，之后 4 天缩量调整（-4..-1）
def base() -> list[dict]:
    return [
        {"vol": 10000, "gain": 0.5},        # 0  垫底，够 find_pattern 取 prev
        {"vol": 10000, "gain": -0.3},       # 1
        {"vol": 10000, "gain": 0.2},        # 2
        {"vol": 10000, "gain": 0.1},        # 3  启动日的前一日
        {"vol": 25000, "gain": 7.0,         # 4  启动日：量 2.5x，涨 7%
         "close": 10.7, "low": 10.0, "high": 10.8},
        {"vol": 12000, "gain": -1.0, "close": 10.6, "low": 10.4},   # 5
        {"vol": 9000,  "gain": -1.5, "close": 10.4, "low": 10.2},   # 6
        {"vol": 7000,  "gain": -0.5, "close": 10.3, "low": 10.1},   # 7
        {"vol": 6000,  "gain": 0.3,  "close": 10.4, "low": 10.2},   # 8  T-1
    ]


def run(label: str, days: list[dict], expect_hit: bool,
        today_to: float = 7.0, today_vol: float = 30000.0,
        tweak=None) -> bool:
    c = cfg()["pullback"]
    if tweak:
        tweak(c)
    got = find_pattern(mk(days), today_to, today_vol, c)
    ok = (got is not None) == expect_hit
    detail = ""
    if got:
        detail = (f"启动 {got['launch_date'][-2:]}日 涨{got['launch_gain']}% "
                  f"调整{got['adjust_days']}日 均量比{got['adjust_vol_mean_ratio']}")
    print(f"  {'✓' if ok else '✗'} {label:<28}"
          f"{'命中' if got else '未命中':<6}{detail}")
    return ok


def main() -> int:
    bad = 0
    print("形态判定")

    bad += not run("标准形态", base(), True)

    d = base(); d[6]["vol"] = 26000
    bad += not run("调整期某日量超启动日", d, False)

    d = base(); d[7]["low"] = 9.5
    bad += not run("调整期跌破启动日最低价", d, False)

    # 启动日推到更早，使调整天数 = 7 > max_days 6
    d = base()
    d = d[:4] + [d[4]] + [{"vol": 6000, "gain": -0.2, "close": 10.3, "low": 10.1}
                          for _ in range(7)]
    bad += not run("调整超过 6 个交易日", d, False)

    d = base()[:5]                       # 启动日就是 T-1，调整 0 天
    bad += not run("调整 0 天（启动日即昨日）", d, False)

    d = base(); d[4]["vol"] = 30000       # 换手反推 = 7.0*30000/30000 = 7.0 -> 合规
    bad += not run("启动日换手恰好在区间内", d, True)

    # 今日量放大到 6 万，则启动日反推换手 = 7.0*25000/60000 = 2.9% < 5%
    bad += not run("启动日换手低于 5%", base(), False, today_vol=60000.0)
    # 今日量只有 2 万，则启动日反推换手 = 7.0*25000/20000 = 8.75% 仍在区间
    bad += not run("启动日换手 8.75%（仍合规）", base(), True, today_vol=20000.0)
    # 今日量 1.5 万 -> 启动日 11.7% > 10%
    bad += not run("启动日换手高于 10%", base(), False, today_vol=15000.0)

    d = base(); d[4]["vol"] = 12000       # 12000/10000 = 1.2x < 1.5
    bad += not run("启动日量比不足 1.5", d, False)

    d = base(); d[4]["gain"] = 4.0
    bad += not run("启动日涨幅不足 5%", d, False)

    # 每日都 < 启动日量，但均量比 = 0.86 > 0.8
    d = base()
    for i in (5, 6, 7, 8):
        d[i]["vol"] = 21500
    bad += not run("每日缩量但均量比超上限", d, False)

    # 两个候选启动日，必须取更近的那个（8月9日那根）
    d = base() + [
        {"vol": 30000, "gain": 6.0, "close": 11.0, "low": 10.5, "high": 11.1},
        {"vol": 8000,  "gain": -1.0, "close": 10.9, "low": 10.7},
        {"vol": 7000,  "gain": -0.5, "close": 10.8, "low": 10.6},
    ]
    c = cfg()["pullback"]
    got = find_pattern(mk(d), 7.0, 30000.0, c)
    ok = got is not None and got["launch_date"].endswith("-10")
    bad += not ok
    print(f"  {'✓' if ok else '✗'} {'多个候选取最近那个':<28}"
          f"{got['launch_date'] if got else '未命中'}")

    # ---- 打分 ----
    print("\n打分")
    c = cfg()["pullback"]
    row = {"code": "600000", "name": "测试", "gain_pct": 7.0, "vol_ratio": 2.0,
           "launch_gain": 7.0, "launch_vol_ratio": 2.5, "launch_low": 10.0,
           "launch_high": 10.8, "adjust_vol_mean_ratio": 0.4,
           "adjust_low": 10.4, "adjust_days": 3}
    s = score_one(row, c)
    ok = 0 <= s["score"] <= 100
    bad += not ok
    print(f"  {'✓' if ok else '✗'} 分数在 [0,100]：{s['score']}  {s['parts']}")

    def sc(**kw):
        return score_one({**row, **kw}, c)["score"]

    checks = [
        ("今日涨幅越大分越高", sc(gain_pct=6.0) < sc(gain_pct=9.0)),
        ("今日量比越大分越高", sc(vol_ratio=1.6) < sc(vol_ratio=3.5)),
        ("缩得越狠分越高",     sc(adjust_vol_mean_ratio=0.7) < sc(adjust_vol_mean_ratio=0.2)),
        ("回踩越浅分越高",     sc(adjust_low=10.05) < sc(adjust_low=10.7)),
        ("调整越短分越高",     sc(adjust_days=6) < sc(adjust_days=2)),
        ("量比打分与准入下限解耦",
         abs(score_one(row, c)["parts"]["trigger"]
             - score_one(row, {**c, "trigger": {**c["trigger"],
                                                "vol_ratio_min": 1.5}})["parts"]["trigger"]) < 1e-9),
    ]
    for name, cond in checks:
        bad += not cond
        print(f"  {'✓' if cond else '✗'} {name}")

    # ---- 工具函数 ----
    print("\n工具函数")
    h = norm_hist(pd.DataFrame({
        "日期": ["2026-08-02", "2026-08-01"], "开盘": [1, 1], "收盘": [2, 1],
        "最高": [2, 1], "最低": [1, 1], "成交量": [10, 20], "涨跌幅": [1, 2],
        "成交额": [1, 1], "换手率": [3.0, 4.0]}))
    ok = list(h["日期"]) == ["2026-08-01", "2026-08-02"] and "换手率" in h.columns
    bad += not ok
    print(f"  {'✓' if ok else '✗'} norm_hist 升序且保留换手率")

    ok = norm_hist(pd.DataFrame({"日期": [1], "收盘": [1]})).empty
    bad += not ok
    print(f"  {'✓' if ok else '✗'} 缺列时返回空表而不是抛异常")

    r = pd.Series({"成交量": 5000.0, "换手率": 6.5})
    ok = abs(hist_turnover(r, 7.0, 10000.0) - 6.5) < 1e-9
    bad += not ok
    print(f"  {'✓' if ok else '✗'} 有真换手率时用真值")

    r = pd.Series({"成交量": 5000.0})
    ok = abs(hist_turnover(r, 7.0, 10000.0) - 3.5) < 1e-9
    bad += not ok
    print(f"  {'✓' if ok else '✗'} 无换手率时按今日等比反推（3.5%）")

    print(f"\n断言失败 {bad} 个")
    return 1 if bad else 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"耗时 {time.time() - t0:.2f}s")
    sys.exit(rc)
