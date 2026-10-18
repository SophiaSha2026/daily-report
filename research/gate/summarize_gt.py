"""把 results/*.json 汇成一张表（筛选集；--dev 时两段都印）。

    PYTHONIOENCODING=utf-8 python research/gate/summarize_gt.py [--dev] [--out results_table.txt]
"""
import argparse
import json

import numpy as np

import common_gt as G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    segs = ("screen", "dev") if a.dev else ("screen",)
    L = []
    hdr = "%-40s %-5s %-6s %5s %4s %5s %6s %6s %6s %6s %6s %6s %6s %6s" % (
        "配置", "τ", "段", "V", "天", "月只", "空榜%", "P%", "下界", "上界", "B(k̄)%", "ΔP", "t", "p̂%")
    L.append(hdr)
    for f in sorted((G.HERE / "results").glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        if a.only and a.only not in j["name"]:
            continue
        for tau, r in j["rows"].items():
            for seg in segs:
                e = r[seg]
                if e["V"] <= 0:
                    L.append("%-40s %-5s %-6s %5d" % (j["name"], tau, seg, 0))
                    continue
                L.append("%-40s %-5s %-6s %5d %4d %5.1f %6.0f %6.1f %6.1f %6.1f %6.1f %+6.1f %+6.2f %6.1f" % (
                    j["name"], tau, seg, e["V"], e["days_out"], e.get("per_month", np.nan), 100 * e["empty_share"], 100 * e["P"],
                    100 * e.get("P_lo", np.nan), 100 * e.get("P_hi", np.nan), 100 * e["P_base"], 100 * e["dP"], e["t"],
                    100 * (e.get("p_hat_mean") or np.nan)))
            if a.dev and r.get("combined"):
                c = r["combined"]
                L.append("%-40s %-5s 合并  ΔP %+.1fpp NW t %.2f 块自举 p %.3f 去最好月 t %.2f 滚3 %+.1f@%s" % (
                    j["name"], tau, 100 * c["dP"], c["t"], c["boot_p"], c["t_drop_best"], c["roll3"][0], c["roll3"][1]))
    txt = "\n".join(L)
    print(txt)
    if a.out:
        (G.HERE / a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
