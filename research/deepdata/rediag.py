"""只重跑诊断（不重建特征表）：python research/deepdata/rediag.py sw_industry [events ...] -> results_<名>.txt 覆盖。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_dd as C  # noqa: E402

RESULT_NAME = {"sw_industry": "sw", "ths_theme": "ths"}

if __name__ == "__main__":
    pool = C.load_pool()
    for name in sys.argv[1:]:
        f = pd.read_parquet(HERE / f"features_{name}.parquet")
        cols = [c for c in f.columns if c not in ("code", "date") and not pd.api.types.is_string_dtype(f[c])]
        txt = C.diag_table(f, pool, cols, label=name)
        (HERE / f"results_{RESULT_NAME.get(name, name)}.txt").write_text(txt + "\n", encoding="utf-8")
        C.progress(f"重跑诊断 {name}（加前50控名次一栏）")
