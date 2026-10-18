"""正式臂：预登记（看开发集前）+ 开发集一次看 + 两段判定 + 族内最大 t（照 research/polish/look_dev.py，族换成 altdata）。

    PYTHONIOENCODING=utf-8 python research/altdata/judge.py --prereg selection.json
    PYTHONIOENCODING=utf-8 python research/altdata/judge.py --judge altdata_v1

selection.json：{"exp": "altdata_v1", "hypothesis": "...", "arms": {臂名: {"col": hits10.parquet 的列名, "desc": "...", "screen_diff": x}}}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "research" / "polish", ROOT / "src", ROOT / "src" / "breakout"):
    sys.path.insert(0, str(p))
import common_pl as P  # noqa: E402

FAMILY = "breakout_y50_top10_altdata"
OLD = 71 + 3 + 2   # 旧族 breakout_y50_top10 71 + rulecore 3 + polish 2


def ledger_add(row: dict) -> None:
    from bjtime import now_bj
    with (HERE / "ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now_bj().strftime("%Y-%m-%d %H:%M"), **row}, ensure_ascii=False) + "\n")


def progress(text: str) -> None:
    from bjtime import now_bj
    with (HERE / "progress.txt").open("a", encoding="utf-8") as f:
        f.write("[%s BJ] %s\n" % (now_bj().strftime("%Y-%m-%d %H:%M"), text))


def do_prereg(sel: dict) -> None:
    import evalkit as K
    screen = {a: v["screen_diff"] for a, v in sel["arms"].items()}
    arms = {a: v["desc"] for a, v in sel["arms"].items()}
    gate = {**K.GATE_V3, "two_segments_same_sign": True, "top30_same_sign": True, "auc_eliminate": False,
            "family_maxt": "本族全部正式臂差值序列联合块自举最大 t 95% 分位", "k": 10}
    f = K.prereg(sel["exp"], sel["hypothesis"], arms, metric="top10 y_open 按天配对 对 ret20 基线（两段 + 合并 969 天）",
                 gate=gate, family=FAMILY, screen=screen)
    for a, v in sel["arms"].items():
        ledger_add({"kind": "arm", "exp": sel["exp"], "arm": a, "col": v["col"], "screen_diff": v["screen_diff"]})
    (P.EXP / sel["exp"] / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    P.say("预登记 -> %s" % f)
    progress("预登记 %s（族 %s）：%s（看开发集之前）" % (sel["exp"], FAMILY, list(sel["arms"])))


def do_judge(exp: str) -> str:
    sel = json.loads((P.EXP / exp / "selection.json").read_text(encoding="utf-8"))
    h10 = pd.read_parquet(HERE / "hits10.parquet")
    h30 = pd.read_parquet(HERE / "hits30.parquet")
    base10, base30 = h10["base_ret20"], h30["base_ret20"]
    D, res = {}, {}
    L = ["== %s：开发集一次看 + 两段判定（对 ret20 基线，每天前 10 买得到命中）==" % exp]
    for a, v in sel["arms"].items():
        d = (h10[v["col"]] - base10).dropna()
        d30 = (h30[v["col"]] - base30).dropna()
        D[a] = d
        j = P.judge(d, d30, a)
        h = h10[v["col"]]
        j["hit_screen"] = float(h[h.index < "2024-06-01"].mean())
        j["hit_dev"] = float(h[h.index >= "2024-06-01"].mean())
        j["hit_all"] = float(h.mean())
        res[a] = j
    DD = pd.DataFrame(D)
    null = P.maxt_null(DD)
    c_fam = float(np.quantile(null, 0.95))
    c = max(2.5, c_fam)
    c_strict = NormalDist().inv_cdf(1 - 0.05 / (OLD + len(sel["arms"])))
    L.append("族门槛（%d 个臂最大 t 零分布 95%% 分位）%.2f -> c = %.2f；严格门槛（旧族 %d + 本族）%.3f" % (len(D), c_fam, c, OLD, c_strict))
    for a, j in res.items():
        L.append("  命中：筛选 %.2f%% / 开发 %.2f%% / 合并 %.2f%%" % (100 * j["hit_screen"], 100 * j["hit_dev"], 100 * j["hit_all"]))
        L.append("  " + P.fmt_judge(j, c))
        j["fam_p"] = float((null >= j["t"]).mean())
        j["c"] = c
        L.append("  族 p（最大 t）%.3f" % j["fam_p"])
    import evalkit as K
    for a, j in res.items():
        K.record(exp, {"arm": a, **{k: v for k, v in j.items() if k != "label"}})
        ledger_add({"kind": "result", "exp": exp, "arm": a, "t": j["t"], "diff": j["diff"], "fam_p": j["fam_p"]})
    txt = "\n".join(L)
    (HERE / ("results_%s.txt" % exp)).write_text(txt + "\n", encoding="utf-8")
    json.dump(res, (P.EXP / exp / "results.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg")
    ap.add_argument("--judge")
    a = ap.parse_args()
    if a.prereg:
        do_prereg(json.loads((HERE / a.prereg).read_text(encoding="utf-8")))
    if a.judge:
        P.say(do_judge(a.judge))


if __name__ == "__main__":
    main()
