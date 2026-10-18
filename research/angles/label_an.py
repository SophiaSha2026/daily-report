"""
角度四：拆解标签（criteria 第 4 节）。只描述，不做选择，不改口径。

    PYTHONIOENCODING=utf-8 python research/angles/label_an.py   -> results_label.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common_an as A

OUT = A.HERE / "results_label.txt"
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def q(s, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    s = pd.Series(s).dropna()
    if not len(s):
        return "—"
    return "/".join("%.0f" % (100 * v) for v in s.quantile(list(qs)))


def main():
    top = pd.read_parquet(A.ROOT / "research" / "precision" / "top50.parquet",
                          columns=["date", "code", "rank_ret20", "y_open", "r20_open", "yizi1", "seg", "hit", "c20", "dd20", "board", "ret20"])
    top = top[top["rank_ret20"] <= 10]
    p = pd.read_parquet(A.PATHS).drop(columns=["c20", "c10"])
    t = top.merge(p, on=["code", "date"], how="left")
    t["grp"] = np.where(t["rank_ret20"] <= 3, "1-3名", "4-10名")
    say("角度四：拆解标签（候选池 lret20 名次前 10，两段；r20_open = 次日开盘起 20 根内最高价涨幅；一字板 y_open 记不中）")
    ok = np.isfinite(t["max20"]) & np.isfinite(t["r20_open"])
    agree = float(((t.loc[ok, "max20"] >= 0.5) == (t.loc[ok, "r20_open"] >= 0.5)).mean())
    say("路径表 max20 与标签表 r20_open 的 50%% 判定一致率 %.4f（%d 行）；|max20 − r20_open| 中位 %.4f" % (
        agree, int(ok.sum()), float((t.loc[ok, "max20"] - t.loc[ok, "r20_open"]).abs().median())))
    say()
    for seg in ("screen", "dev"):
        for grp in ("1-3名", "4-10名"):
            s = t[(t["seg"] == seg) & (t["grp"] == grp) & np.isfinite(t["r20_open"])]
            r = s["r20_open"]
            hits = s[s["hit"] > 0]
            miss = s[s["hit"] == 0]
            say("== %s %s：%d 行，命中 %.1f%%，一字板 %.1f%%（一字板里 r20>=50%% 的 %.1f%%）" % (
                seg, grp, len(s), 100 * s["hit"].mean(), 100 * s["yizi1"].mean(),
                100 * (s.loc[s["yizi1"] == True, "r20_open"] >= 0.5).mean() if (s["yizi1"] == True).any() else float("nan")))
            say("   r20_open 分位 10/25/50/75/90：%s%%" % q(r))
            bands = [("<20", r < 0.2), ("[20,30)", (r >= 0.2) & (r < 0.3)), ("[30,40)", (r >= 0.3) & (r < 0.4)),
                     ("[40,50)", (r >= 0.4) & (r < 0.5)), ("[50,60)", (r >= 0.5) & (r < 0.6)), (">=60", r >= 0.6)]
            say("   r20_open 分档占比：" + "  ".join("%s %.1f%%" % (k, 100 * m.mean()) for k, m in bands))
            say("   命中票到 +50%% 的第几根（d50）分位 10/25/50/75/90：%s；中位 %.0f 根；5 根内到的 %.0f%%，10 根内 %.0f%%" % (
                "/".join("%.0f" % v for v in hits["d50"].dropna().quantile([0.1, 0.25, 0.5, 0.75, 0.9])) if len(hits) else "—",
                hits["d50"].median() if len(hits) else float("nan"),
                100 * (hits["d50"] <= 5).mean() if len(hits) else float("nan"), 100 * (hits["d50"] <= 10).mean() if len(hits) else float("nan")))
            say("   命中票到 +30%% / +40%% 的中位根数 %.0f / %.0f；未命中票最高点在第几根分位 10/50/90：%s，未命中 max20 中位 %.0f%%" % (
                hits["d30"].median() if len(hits) else float("nan"), hits["d40"].median() if len(hits) else float("nan"),
                "/".join("%.0f" % v for v in miss["argmax20"].dropna().quantile([0.1, 0.5, 0.9])) if len(miss) else "—",
                100 * miss["max20"].median() if len(miss) else float("nan")))
            g1 = s["g1"]
            gb = [("<=0", g1 <= 0), ("(0,3]", (g1 > 0) & (g1 <= 0.03)), ("(3,7]", (g1 > 0.03) & (g1 <= 0.07)),
                  ("(7,10]", (g1 > 0.07) & (g1 <= 0.10)), (">10", g1 > 0.10)]
            say("   次日缺口分档 只数 / 精确率 / r20 中位：" + "  ".join(
                "%s %d/%.1f%%/%.0f%%" % (k, m.sum(), 100 * s.loc[m, "hit"].mean() if m.any() else float("nan"),
                                         100 * s.loc[m, "r20_open"].median() if m.any() else float("nan")) for k, m in gb))
            say("   缺口中位 %.1f%%；命中票缺口中位 %.1f%%，未命中 %.1f%%；缺口 > 10%% 里一字板占 %.0f%%" % (
                100 * g1.median(), 100 * hits["g1"].median() if len(hits) else float("nan"),
                100 * miss["g1"].median() if len(miss) else float("nan"),
                100 * s.loc[g1 > 0.10, "yizi1"].mean() if (g1 > 0.10).any() else float("nan")))
            # 敏感表：阈值 x 期限（一字板一律不中，和 y_open 同口径）
            buy = s["yizi1"] != True
            rows = []
            for hname, col in (("10 根", "max10"), ("20 根", "max20"), ("30 根", "max30")):
                cells = []
                for thr in (0.3, 0.4, 0.45, 0.5, 0.6):
                    cells.append("%.1f" % (100 * ((s[col] >= thr) & buy).mean()))
                rows.append("%s: %s" % (hname, " / ".join(cells)))
            say("   命中率敏感表（阈值 30/40/45/50/60%%，一字板不中）：%s" % "；".join(rows))
            say("   命中票 c20（第 20 根收盘）分位 10/50/90：%s%%；dd20（最大回撤）%s%%；未命中票 c20 %s%%，dd20 %s%%" % (
                q(hits["c20"], (0.1, 0.5, 0.9)), q(hits["dd20"], (0.1, 0.5, 0.9)), q(miss["c20"], (0.1, 0.5, 0.9)), q(miss["dd20"], (0.1, 0.5, 0.9))))
            say("   [40,50) 占比 / 命中占比 = %.2f；45%% 阈值下精确率 %.1f%% 对 50%% 的 %.1f%%（差 %+.1fpp）" % (
                bands[3][1].mean() / max(s["hit"].mean(), 1e-9), 100 * ((s["max20"] >= 0.45) & buy).mean(), 100 * ((s["max20"] >= 0.5) & buy).mean(),
                100 * (((s["max20"] >= 0.45) & buy).mean() - ((s["max20"] >= 0.5) & buy).mean())))
            say()
    # 板块 x 名次 1-3 的 r20 分布（描述）
    s = t[(t["grp"] == "1-3名") & np.isfinite(t["r20_open"])]
    say("== 1-3 名按板块（两段合并）：只数 / 精确率 / r20 中位 / [40,50) 占比")
    for b, g in s.groupby("board"):
        say("   %-8s %5d / %5.1f%% / %4.0f%% / %4.1f%%" % (b, len(g), 100 * g["hit"].mean(), 100 * g["r20_open"].median(),
                                                        100 * ((g["r20_open"] >= 0.4) & (g["r20_open"] < 0.5)).mean()))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    A.ledger_add({"kind": "diag", "name": "label_decomp", "note": "角度四描述（两段），不做选择"})
    A.progress("label_an：角度四拆解标签 -> results_label.txt")


if __name__ == "__main__":
    main()
