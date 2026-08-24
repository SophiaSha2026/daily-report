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


def t_hist():
    import datasource as ds
    h = ds.daily_hist("600000", "20260101", "20260820")
    assert h is not None and len(h) > 50, "日线过短"
    return f"{len(h)} 根K线"


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
    check("日线历史(东财单只)", t_hist)
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
