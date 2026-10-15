"""已上线 / 待定结论的 Newey-West 复核（不重训）：实验 15 eqm 对 base（旧验证集 2025-03..12），
实验 18 g2 对 w1 base（开发集）在 exp_altpath.analyze 里。按天配对差值序列，滞后 20。"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "breakout"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd  # noqa: E401,E402
import validate as V  # noqa: E402
from exp_altpath import nw_t  # noqa: E402

p = pd.read_parquet(ROOT / "data/breakout/raw/exp_time/preds.parquet", columns=["date", "code", "y_up", "base", "eqm"])
lab = pd.read_parquet(ROOT / "data/breakout/raw/labels_open.parquet", columns=["code", "date", "y_open"])
p = p.merge(lab, on=["code", "date"], how="left")
p = p[~p["code"].isin(V.st_codes())]
out = []
for y in ("y_up", "y_open"):
    h = {}
    for arm in ("base", "eqm"):
        t = p[np.isfinite(p[y])].sort_values(["date", arm], ascending=[True, False])
        t["r"] = t.groupby("date").cumcount() + 1
        h[arm] = t[t["r"] <= 10].groupby("date")[y].mean()
    d = (h["eqm"] - h["base"]).dropna().sort_index()
    t_iid = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    t_nw, _ = nw_t(d)
    out.append("eqm 对 base（%s，%d 天）：%+.2fpp 按天 t=%.2f NW(20) t=%.2f" % (y, len(d), 100 * d.mean(), t_iid, t_nw))
print("\n".join(out))
(Path(__file__).parent / "nw_recheck.txt").write_text("\n".join(out), encoding="utf-8")
