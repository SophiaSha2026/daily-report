"""十大流通股东缺失的月块（年报 / 季报月，整月深翻页报「服务器繁忙」）按自然日逐天拉，再拼成月块 parts/<月>.parquet。

    PYTHONIOENCODING=utf-8 python research/deepdata/fetch_fh_days.py

然后 fetch_dd.py merge 会把它们并进 raw/freeholders.parquet。纪律同 fetch_dd（禁区 / 超时 / 重试 / 熔断），单线程。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_dd as F  # noqa: E402


def main():
    cfg = dict(F.TABLES["freeholders"])
    d = F.RAW / "freeholders" / "parts"
    have = {p.name[:7] for p in d.glob("*")}
    months = [p for p in pd.period_range(F.BEGIN[:7], F.END[:7], freq="M") if p.strftime("%Y-%m") not in have]
    F.log(f"== freeholders 逐日补 {len(months)} 个月：{[m.strftime('%Y-%m') for m in months]}")
    for per in months:
        frames = []
        ok = True
        for day in pd.date_range(per.start_time, per.end_time, freq="D"):
            ds = day.strftime("%Y-%m-%d")
            try:
                df = F.pull_dc(cfg, ds, ds)
            except Exception as e:  # noqa: BLE001
                F.log(f"  {ds} 失败：{e}")
                ok = False
                if F._stop.is_set():
                    F.merge("freeholders")
                    return
                continue
            if len(df):
                frames.append(df)
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(d / f"{per.strftime('%Y-%m')}{'' if ok else '_partial'}.parquet", index=False)
            F.log(f"  {per.strftime('%Y-%m')}: {len(out)} 行（逐日 {len(frames)} 天{'，有失败日' if not ok else ''}）")
        else:
            F.log(f"  {per.strftime('%Y-%m')}: 0 行")
    F.merge("freeholders")


if __name__ == "__main__":
    main()
