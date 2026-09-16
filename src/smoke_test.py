"""冒烟测试：验证 GitHub runner 能否访问所需数据源，以及关键字段索引。"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

FAIL = []


def check(name, fn):
    t0 = time.time()
    try:
        msg = fn()
        print(f"  [OK]   {name:<26} {time.time()-t0:5.1f}s  {msg}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name:<26} {time.time()-t0:5.1f}s  {type(e).__name__}: {e}")
        FAIL.append(name)


def t_tencent():
    import datasource as ds
    q = ds.fetch_quotes(["sh600000", "sz000001", "sz300750", "sh688981"])
    assert len(q) >= 3, f"仅返回 {len(q)} 只"
    a = q["sh600000"]
    assert a.prev_close > 0 and a.name, "字段解析异常"
    return f"{len(q)}/4 只 | 浦发 昨收{a.prev_close} 额{a.amount_wan:.0f}万"


def t_codelist():
    import datasource as ds
    codes = ds.load_code_list()
    assert len(codes) > 3000, f"仅 {len(codes)} 只"
    return f"{len(codes)} 只（缓存或现拉）"


def t_batch():
    """
    竞价窗口真实负载。用**真实代码**测，不要用编造的连号——
    上一版用 sh600000..600799 这种连号，其中一半根本不存在，
    「1600 只 -> 1070 有效」是代码不存在，不是限流，属于测试设计错误。
    """
    import datasource as ds
    codes = ds.load_code_list()[:1600]
    syms = [ds.to_symbol(c) for c in codes]
    t0 = time.time()
    q = ds.fetch_quotes(syms)
    d = time.time() - t0
    rate = len(q) / len(syms)
    assert rate > 0.9, f"命中率仅 {rate:.0%}，疑似限流"
    assert d < 12, f"耗时 {d:.1f}s，竞价窗口余量不足"
    return f"{len(syms)} 只 -> {len(q)} 有效 ({rate:.0%}), {d:.1f}s"


def t_spot():
    import datasource as ds
    df = ds.spot_all()
    assert len(df) > 3000, f"仅 {len(df)} 行"
    nz = int((df["成交额"] > 0).sum())
    return f"{len(df)} 行, 有成交 {nz} 只"


def t_em_bulk():
    """
    东财 clist 批量接口。**失败不算致命**——流水线已改用腾讯，
    这里只做可用性记录，用于判断要不要把它重新加回兜底链。
    """
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    assert df is not None and len(df) > 3000
    return f"{len(df)} 行（可用，可作兜底）"


def t_calendar():
    import datasource as ds
    d = ds.trade_dates()
    assert len(d) > 1000
    return f"{len(d)} 个交易日"


# 成交量口径的哨兵。腾讯**两个接口**对科创板给的是「股」、其余板块给「手」，
# 由 datasource.tx_vol_hand 统一成手；新浪那路一律除以 100。两边都归一之后
# 同一天同一只票的成交量必须相等，2026-09-16 本机实测 688008/688981/688256
# 腾讯÷新浪恰好 100.000（未归一时），归一后 1.000。
# 少除一次 100 的后果不会报错：688008 那天的成交额会估成 7009 亿（真值 68.3 亿），
# 量比、换手、筹码全跟着错一个量级。所以拿一只 688 的票当口径哨兵。
VOL_CHECK_CODE = "688008"


def vol_ratio(a, b) -> float:
    """两路日线在相同交易日上的成交量之比（中位数）。两路都该是「手」，
    所以正常是 1.0；口径漂了会是 100 或 0.01，一眼看得出是乘错了 100。"""
    m = a.merge(b, on="日期", suffixes=("_a", "_b"))
    m = m[(m["成交量_a"] > 0) & (m["成交量_b"] > 0)]
    if not len(m):
        raise AssertionError("两路日线没有共同交易日，比不了口径")
    return float((m["成交量_a"] / m["成交量_b"]).median())


def t_hist():
    """
    日线取数。东财为主，腾讯为辅——只要有一路给出足够长度就算通过，
    因为盘前 stage2 用的就是这条带降级的链路。
    两路都单测一遍，把各自可用性打印出来，便于判断要不要调熔断阈值。

    外加一只科创板（688008）的成交量口径核对：腾讯对 688 段给「股」，
    这件事没有任何报错会提醒你，只能拿另一路比出来。
    """
    import datasource as ds
    detail = []
    for tag, fn in (("东财", ds.daily_hist_em), ("腾讯", ds.daily_hist_tx)):
        try:
            n = len(fn("600000", "20260101", "20260820"))
            detail.append(f"{tag} {n}根" if n > 50 else f"{tag} 仅{n}根")
        except Exception as e:  # noqa: BLE001
            detail.append(f"{tag} 不可用({type(e).__name__})")
    h = ds.daily_hist("600000", "20260101", "20260820")
    assert h is not None and len(h) > 50, f"两路都拿不到日线: {' / '.join(detail)}"

    tx = ds.daily_hist_tx(VOL_CHECK_CODE, "20260101", "20260820")
    sn = ds.daily_hist_sina(VOL_CHECK_CODE, "20260101", "20260820")
    r = vol_ratio(tx, sn)
    assert abs(r - 1.0) < 0.02, (
        f"{VOL_CHECK_CODE} 腾讯/新浪成交量之比 {r:.3f}，不是 1：成交量口径漂了"
        f"（腾讯对 688 段给「股」，见 datasource.tx_vol_hand）")
    detail.append(f"{VOL_CHECK_CODE} 腾讯÷新浪 {r:.3f}")
    return f"{len(h)} 根K线 | {' / '.join(detail)}"


def t_scoring():
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "selftest.py")],
                       capture_output=True, text=True, timeout=90)
    assert r.returncode == 0, r.stdout[-400:]
    return "12 边界用例 + 1000 压力样本 全通过"


if __name__ == "__main__":
    print("=" * 72)
    print("冒烟测试 —— 全绿才能开定时任务")
    print("=" * 72)
    check("腾讯批量行情", t_tencent)
    check("代码表", t_codelist)
    check("批量吞吐(1600只)", t_batch)
    check("全市场快照(腾讯)", t_spot)
    check("交易日历", t_calendar)
    check("日线历史(东财主/腾讯辅)", t_hist)
    check("打分逻辑自测", t_scoring)
    print("-" * 72)
    print("以下为非关键项，失败不影响运行：")
    soft = len(FAIL)
    check("东财批量接口(兜底)", t_em_bulk)
    optional = FAIL[soft:]
    hard = FAIL[:soft]
    print("=" * 72)
    if optional:
        print("提示：东财批量接口在本 runner 上不可用，流水线已改走腾讯，无影响。")
    if hard:
        print(f"致命失败 {len(hard)} 项: {', '.join(hard)}")
        sys.exit(1)
    print("关键项全部通过")
