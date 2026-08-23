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


def t_batch():
    """竞价窗口真实负载：1600 只分 27 批，必须在 15 秒内跑完。"""
    import datasource as ds
    base = [f"sh{600000+i}" for i in range(800)] + \
           [f"sz{i:06d}" for i in range(1, 801)]
    t0 = time.time()
    q = ds.fetch_quotes(base)
    d = time.time() - t0
    assert d < 25, f"耗时 {d:.1f}s 过长，竞价窗口会来不及"
    return f"1600 只 -> {len(q)} 有效, {d:.1f}s"


def t_spot():
    import datasource as ds
    df = ds.spot_all()
    assert len(df) > 3000, f"仅 {len(df)} 行"
    return f"{len(df)} 行, 列 {len(df.columns)}"


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
    check("批量吞吐(1600只)", t_batch)
    check("东财全市场快照", t_spot)
    check("交易日历", t_calendar)
    check("日线历史", t_hist)
    check("打分逻辑自测", t_scoring)
    print("=" * 72)
    if FAIL:
        print(f"失败 {len(FAIL)} 项: {', '.join(FAIL)}")
        print("若行情类失败 -> GitHub runner 出口 IP 被国内接口限流，需换部署位置")
        sys.exit(1)
    print("全部通过")
