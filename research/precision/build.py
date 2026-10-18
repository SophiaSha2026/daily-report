"""建 top50.parquet 并对账基线：B(10) 两段精确率要和 polish 的 11.34 / 16.54 逐位一致。

    PYTHONIOENCODING=utf-8 python research/precision/build.py
"""
from __future__ import annotations

import numpy as np

import common_pr as C


def main():
    C.P.wait_prod_window()
    t = C.build_top50(force=True)
    for seg in ("screen", "dev"):
        s = t[t["seg"] == seg]
        B = C.Baseline(s)
        say = C.say
        say("%s：%d 天，B(1..10) 精确率 %s" % (seg, B.D, " ".join("%.2f" % (100 * B.precision_k(k)) for k in range(1, 11))))
        say("  streak10 分布（前 10 行）：%s" % s[s["rank_ret20"] <= 10]["streak10"].value_counts().sort_index().head(8).to_dict())
        say("  model_rank 覆盖 %.1f%%，altdata lhb_today 覆盖 %.1f%%，rz_in 覆盖 %.1f%%" % (
            100 * np.isfinite(s["model_rank"]).mean(), 100 * np.isfinite(s["lhb_today"]).mean(), 100 * s["rz_in"].notna().mean()))
    C.progress("top50.parquet 建好：%d 行 %d 列；B(10) 筛选 %.2f%% / 开发 %.2f%%（对账 polish 11.34 / 16.54）" % (
        len(t), t.shape[1], 100 * C.Baseline(t[t["seg"] == "screen"]).precision_k(10), 100 * C.Baseline(t[t["seg"] == "dev"]).precision_k(10)))


if __name__ == "__main__":
    main()
