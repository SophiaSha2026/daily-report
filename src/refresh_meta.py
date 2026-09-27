"""
刷新全市场代码表 cache/codes.csv。两条晚间线都用它：起涨预测的日线回填按它
逐只拉，长期调整突破的日线来自同一张表。

    python src/refresh_meta.py

多源兜底和「新表不许比旧表少超过 2%」那道闸都在 datasource.refresh_code_list 里。

2026-09-27 起只刷代码表。以前后半段还刷行业板块表 cache/sector_map.parquet
（东财 -> 同花顺 -> 新浪三个源），那张表只有早盘选股的板块打分用，随早盘系统
归档到 archive/morning/（原脚本原样在 archive/morning/src/refresh_meta.py）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("meta")


def main() -> int:
    import datasource as ds
    try:
        codes = ds.refresh_code_list()
    except Exception as e:  # noqa: BLE001
        # 沿用旧缓存是对的（残表比旧表坏），但退出码必须带出去：控制台的
        # 「更新股票名单」只看退出码，代码表一只都没刷到还显示绿，
        # 唯一的信号就只剩一行日志了（教训 16）
        log.error("代码表刷新失败，沿用旧缓存: %s", e)
        return 1
    log.info("代码表已刷新：%d 只", len(codes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
