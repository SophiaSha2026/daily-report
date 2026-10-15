"""
防偷看钉子：新特征族（build_new.FAMS，含 past_jackpot）在「日线截在 T 日」和「全段」上算出的 T 日那行必须逐位相同。
随机 300 只票，四个 T。任何一列不同就以非零退出。

    python research/features/test_trunc_new.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

import build_new as BN  # noqa: E402


def feats(px):
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px["_pos"] = px.groupby("code").cumcount().to_numpy()
    pos = px["_pos"].to_numpy()
    c = px["close"].to_numpy(float)
    r = c / BN.lag(c, 1, pos) - 1
    r[~np.isfinite(r)] = np.nan
    px["r"] = np.clip(r, -0.35, 0.35)
    m = px.groupby("date")["r"].mean()
    px["m0"] = px["date"].map(m).to_numpy(float)
    px["m1"] = px["date"].map(m.shift(1)).to_numpy(float)
    px["m2"] = px["date"].map(m.shift(2)).to_numpy(float)
    f = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for fn in BN.FAMS.values():
            g = {}
            fn(px, g)
            f.update(g)
    return px[["code", "date"]], f


def main():
    full = BN.load()
    rng = np.random.default_rng(3)
    codes = rng.choice(full["code"].unique(), 300, replace=False)
    sub = full[full["code"].isin(codes)][["code", "date", "open", "high", "low", "close", "volume", "amount",
                                          "turnover", "outstanding_share"]]
    k0, f0 = feats(sub.copy())
    bad = 0
    lines = []
    for T in ("2024-09-24", "2025-01-15", "2025-06-30", "2025-11-20"):
        k1, f1 = feats(sub[sub["date"] <= T].copy())
        i0 = np.flatnonzero(k0["date"].to_numpy() == T)
        i1 = np.flatnonzero(k1["date"].to_numpy() == T)
        assert (k0["code"].to_numpy()[i0] == k1["code"].to_numpy()[i1]).all()
        worst = {}
        for c in f0:
            a, b = np.asarray(f0[c], float)[i0], np.asarray(f1[c], float)[i1]
            same = (np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b)))
            if not same.all():
                worst[c] = int((~same).sum())
        bad += len(worst)
        lines.append("截断在 %s：%d 只 %d 列，不一致的列 %s" % (T, len(i0), len(f0), worst or "无"))
    txt = "\n".join(lines)
    print(txt)
    (C.HERE / "results_trunc_new.txt").write_text(txt, encoding="utf-8")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
