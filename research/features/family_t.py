"""
本轮（research/features）所有训练臂的逐日差值序列（各对同循环对照，前 10 买得到口径）-> 族内最大 t 的零分布门槛
（research/stats/multi.maxt_null：去均值后按天整行块重抽，保留臂间相关和自相关），加 Bonferroni 对照。

    python research/features/family_t.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import pandas as pd  # noqa: E402

import multi  # noqa: E402  (research/stats)

PAIRS = [("feat_seeds", f"a158_s{s}", f"base_s{s}") for s in (9, 10, 11)] + [("feat_seeds", "bag3_a158", "bag3_base")]
PAIRS += [("feat_decomp", a, "base") for a in ("prod_rank", "base_pv", "pv_only", "rank_pv", "coarse_a158")]
PAIRS += [("feat_fam_v1", f"a158_{f}", "base_a158") for f in ("on", "vc", "hi", "ext", "turn", "rm", "amp")]
PAIRS += [("feat_comb_v1", a, "base_a158") for a in ("comb_all", "comb_sel", "comb_sel_cs4")]
PAIRS += [("feat_final_v1", "cand_s7", "base_s7")]


def main():
    D = {}
    for exp, a, b in PAIRS:
        try:
            D[f"{exp}:{a}"] = C.daily_diff(C.top(exp, a), C.top(exp, b))
        except FileNotFoundError:
            print("缺", exp, a)
    D = pd.DataFrame(D).sort_index()
    r = multi.maxt_null(D, b=3000)
    m = C.n_trials()
    out = {"m_series": r["m"], "days": r["T"], "q95_one": r["q95_one"], "q90_one": r["q90_one"], "n_eff": r["n_eff"],
           "mean_abs_corr": r["mean_abs_corr"], "t_obs": r["t_obs"], "p_fwer": r["p_fwer"],
           "ledger_trials_this_round": m, "bonf_this_round": C.t_bonf(m)}
    txt = ["族内最大 t（本轮 %d 条序列，%d 天）：q95 单侧 %.2f，q90 %.2f，有效试验数 %.1f，平均 |相关| %.2f" % (
        r["m"], r["T"], r["q95_one"], r["q90_one"], r["n_eff"], r["mean_abs_corr"]),
        "本轮登记簿试验数 %d，单侧 Bonferroni 门槛 %.2f" % (m, C.t_bonf(m))]
    for k in D.columns:
        txt.append("  %-34s NW t %5.2f  族内 FWER p %.3f" % (k, r["t_obs"][k], r["p_fwer"][k]))
    print("\n".join(txt))
    (C.HERE / "results_family_t.txt").write_text("\n".join(txt), encoding="utf-8")
    (C.HERE / "family_t.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
