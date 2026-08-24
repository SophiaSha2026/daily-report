"""每周刷新行业成分缓存。约 90 个行业板块，带限速，耗时 2-4 分钟。"""
from __future__ import annotations
import sys, time, logging
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("meta")


def main() -> int:
    import akshare as ak
    sys.path.insert(0, str(Path(__file__).parent))
    import datasource as ds
    try:
        ds.refresh_code_list()
    except Exception as e:  # noqa: BLE001
        log.warning("代码表刷新失败，沿用旧缓存: %s", e)

    names = ak.stock_board_industry_name_em()
    col = "板块名称" if "板块名称" in names.columns else names.columns[1]
    rec = []
    for i, nm in enumerate(names[col].tolist()):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=nm)
            ccol = "代码" if "代码" in cons.columns else cons.columns[1]
            for code in cons[ccol].astype(str):
                rec.append({"code": code.zfill(6), "sector": nm})
        except Exception as e:  # noqa: BLE001
            log.warning("行业 %s 失败: %s", nm, e)
        if i % 20 == 0:
            log.info("  %d/%d", i, len(names))
        time.sleep(0.25)

    if len(rec) < 2000:
        log.error("成分股仅 %d 条，疑似接口异常，保留旧缓存", len(rec))
        return 1
    out = ROOT / "cache"; out.mkdir(exist_ok=True)
    df = pd.DataFrame(rec).drop_duplicates("code")
    df.to_parquet(out / "sector_map.parquet", index=False)
    log.info("已写入 %d 只股票的行业归属，%d 个行业", len(df), df.sector.nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
