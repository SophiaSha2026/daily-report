"""汇总各 results_*.txt：挑出「前50控名次」两段同号、两段 |diff| 都 >= 0.8pp 的特征，和全池两段同号且 |t| 都 >= 2 的特征 -> results_summary.txt。"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(r"^(\S+)\s+(\S+)\s+\|\s+([+-]?[\d.]+|\+nan) \(t\s+([+-]?[\d.na]+),\s+(\d+)\)\s+\|\s+([+-]?[\d.]+|\+nan) \(t\s+([+-]?[\d.na]+),\s+(\d+)\)\s+\|\s+([+-]?[\d.]+|\+nan) \(t\s+([+-]?[\d.na]+)\) (同号|异号)\s+([\d.]+)%")


def rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        m = PAT.match(line)
        if not m:
            continue
        feat, uni, ds, ts, ns, dd, td, nd, da, ta, same, cov = m.groups()
        try:
            yield dict(src=path.stem.replace("results_", ""), feat=feat, uni=uni, ds=float(ds), ts=float(ts), ns=int(ns),
                       dd=float(dd), td=float(td), nd=int(nd), da=float(da), ta=float(ta), same=same == "同号", cov=float(cov))
        except ValueError:
            continue


def main():
    allr = []
    for p in sorted(HERE.glob("results_*.txt")):
        if p.name == "results_summary.txt":
            continue
        allr.extend(rows(p))
    out = ["# 汇总（summary_dd.py）：按天配对命中差 pp，NW t 滞后 20；筛选 2022-01..2024-05 / 开发 2024-06..2025-12",
           "# ths_theme 的筛选段只有 2023-10-25..2024-05；events 里股权激励只有 2023-08 起", "",
           "## A. 前50控名次：两段同号、两段 |diff| >= 0.8pp、两段天数 >= 100（去掉和 ret20 名次相关的那部分后仍有的）",
           "%-14s %-26s %-10s | %-20s | %-20s | %s" % ("源", "特征", "分组", "筛选 diff (t)", "开发 diff (t)", "合并 diff (t) 覆盖%")]
    A = [r for r in allr if r["uni"].startswith("前50控名次") and r["same"] and abs(r["ds"]) >= 0.8 and abs(r["dd"]) >= 0.8 and r["ns"] >= 100 and r["nd"] >= 100]
    A.sort(key=lambda r: -min(abs(r["ds"]), abs(r["dd"])))
    for r in A:
        out.append("%-14s %-26s %-10s | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f) %5.1f%%" % (
            r["src"], r["feat"], r["uni"].split("/")[1], r["ds"], r["ts"], r["dd"], r["td"], r["da"], r["ta"], r["cov"]))
    out += ["", "## B. 前50（不控名次）：两段同号、两段 |diff| >= 1pp、两段天数 >= 100",
            "%-14s %-26s %-10s | %-20s | %-20s | %s" % ("源", "特征", "分组", "筛选 diff (t)", "开发 diff (t)", "合并 diff (t) 覆盖%")]
    B = [r for r in allr if r["uni"].startswith("前50/") and r["same"] and abs(r["ds"]) >= 1 and abs(r["dd"]) >= 1 and r["ns"] >= 100 and r["nd"] >= 100]
    B.sort(key=lambda r: -min(abs(r["ds"]), abs(r["dd"])))
    for r in B:
        out.append("%-14s %-26s %-10s | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f) %5.1f%%" % (
            r["src"], r["feat"], r["uni"].split("/")[1], r["ds"], r["ts"], r["dd"], r["td"], r["da"], r["ta"], r["cov"]))
    out += ["", "## C. 全池：两段同号、两段 |t| >= 2（候选池里有分辨力，不一定进得了前 10）",
            "%-14s %-26s %-10s | %-20s | %-20s | %s" % ("源", "特征", "分组", "筛选 diff (t)", "开发 diff (t)", "合并 diff (t) 覆盖%")]
    Cc = [r for r in allr if r["uni"].startswith("全池") and r["same"] and abs(r["ts"]) >= 2 and abs(r["td"]) >= 2]
    Cc.sort(key=lambda r: -min(abs(r["ts"]), abs(r["td"])))
    for r in Cc:
        out.append("%-14s %-26s %-10s | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f)       | %+6.2f (t %5.2f) %5.1f%%" % (
            r["src"], r["feat"], r["uni"].split("/")[1], r["ds"], r["ts"], r["dd"], r["td"], r["da"], r["ta"], r["cov"]))
    out.append("")
    out.append("共 %d 条诊断行（%d 个特征 x 3 个宇宙）；A %d 条、B %d 条、C %d 条" % (len(allr), len({(r['src'], r['feat']) for r in allr}), len(A), len(B), len(Cc)))
    txt = "\n".join(out)
    (HERE / "results_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
