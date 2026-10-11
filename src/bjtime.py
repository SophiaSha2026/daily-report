"""
北京时间和「最近一个已收盘交易日」的唯一实现。

2026-10-10 审计：`now_bj` 在仓库里有 8 份（两种实现，一半带时区一半 naive），
`last_closed_trade_day` 有 4 份（逻辑相同、取日历的方式不同）。教训 34：同一个名词
两处各算一遍就是待修的 bug。现在：

    now_bj()                      带时区（UTC+8）的当前时间
    today_bj()                    'YYYY-MM-DD'
    last_closed(tds, now)         最近一个已收盘的交易日。15:05 之后算当天，之前算上一个。
                                  tds 空集就退化成周一到周五（宁可节假日多跑一次空流程，
                                  也不能因为日历接口挂了整条线不跑）

取日历那一步各入口自己负责（local_run 带缓存和退化、pullback 直接问 datasource、
evening_check 在云端只问接口、控制台只读缓存文件），**判定逻辑只有这一份**。
不 import 任何重模块：控制台和云端都要用。
"""
from __future__ import annotations

import datetime as dt

BJ = dt.timezone(dt.timedelta(hours=8))


def now_bj() -> dt.datetime:
    return dt.datetime.now(BJ)


def today_bj() -> str:
    return now_bj().strftime("%Y-%m-%d")


def last_closed(tds: set[str] | None, now: dt.datetime | None = None) -> str:
    now = now or now_bj()
    d = now.date()
    closed = (now.hour, now.minute) >= (15, 5)
    tds = tds or set()
    for back in range(0, 15):
        day = d - dt.timedelta(days=back)
        ok = (day.isoformat() in tds) if tds else day.weekday() < 5
        if ok and (back > 0 or closed):
            return day.isoformat()
    return d.isoformat()
