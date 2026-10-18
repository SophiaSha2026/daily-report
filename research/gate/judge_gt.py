"""正式臂：预登记（evalkit 新族 breakout_gate）+ 开发集一次读 + 合并判定（criteria.txt 第 4、5 节）。

    PYTHONIOENCODING=utf-8 python research/gate/judge_gt.py --arms arms.json

arms.json = [{"family": "F0", "score": "rank", "cal": "cum", "win": 36, "nmin": 100, "dmin": 20, "uni": "all"}, ...]
每个臂先读它筛选集的结果（results/<名>.json），τ=0.20 那一行的 ΔP 按天差作为 screen 登记；V=0 的臂登 None（筛选集：无）。
然后对每个臂跑 eval_gt --dev（开发集一次读），结果进 results_dev.txt、ledger kind=arm。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


import common_gt as G


def name_of(a: dict) -> str:
    if a["cal"] == "none":
        nm = a["score"]
    else:
        nm = "%s_%s%d" % (a["score"], a["cal"], a["win"])
        if a["cal"] == "cum":
            nm += "_n%d" % a["nmin"]
    if a.get("dmin"):
        nm += "_d%d" % a["dmin"]
    if a.get("uni", "all") != "all":
        nm += "_" + a["uni"]
    return nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--exp", default="gate_v1")
    a = ap.parse_args()
    import evalkit as K
    arms = json.loads((G.HERE / a.arms).read_text(encoding="utf-8"))
    screen = {}
    desc = {}
    for arm in arms:
        nm = name_of(arm)
        f = G.HERE / "results" / (nm + ".json")
        if not f.exists():
            raise SystemExit("筛选集结果不存在：%s（先跑 eval_gt 不带 --dev）" % f)
        j = json.loads(f.read_text(encoding="utf-8"))
        e = j["rows"]["0.20"]["screen"]
        key = "%s:%s" % (arm["family"], nm)
        desc[key] = "%s 族 %s，门槛 p̂ >= τ（τ 0.20..0.50）" % (arm["family"], nm)
        if e["V"] > 0:
            # 按天差 = ΔP x V / D（u 的均值）
            screen[key] = {"diff_day": float(e["dP"] * e["V"] / e["D"]), "days": int(e["D"]), "span": list(G.SEGS["screen"])}
        else:
            screen[key] = None
        G.say("%-40s 筛选集 τ=0.20：V %d ΔP %s" % (key, e["V"], ("%+.2fpp" % (100 * e["dP"])) if e["V"] else "无（整段空榜）"))
    hyp = ("绝对门槛：每只票每天一个校准概率 p̂（只用之前数据的样本外分数 + 过去 12/36 个月等渗或累计精确率校准），"
           "p̂ >= τ 就出；出的票的买得到精确率对同量名次基线 B(k̄) 有增量，且两段块自举下界 >= B(10)")
    # 筛选集 ΔP < 0 的臂 prereg 会整个拒绝；这里把它们单独列出，不登记、也不读开发集（criteria 第 5 节）
    ok_arms = {k: v for k, v in desc.items() if screen[k] is None or screen[k]["diff_day"] >= 0}
    bad = [k for k in desc if k not in ok_arms]
    for k in bad:
        G.say("!! %s 筛选集 ΔP < 0，不登记、不读开发集" % k)
        G.ledger_add({"kind": "screen_reject", "name": k, "screen_diff_day": screen[k]["diff_day"]})
    if ok_arms:
        p = K.prereg(a.exp, hyp, ok_arms, metric="精确率 P 对同量名次基线 B(k̄)，按天序列 u_d = h_d − h_B,d",
                     family=G.FAMILY, screen={k: screen[k] for k in ok_arms})
        G.say("预登记 -> %s（%d 个臂）" % (p, len(ok_arms)))
    for arm in arms:
        nm = name_of(arm)
        key = "%s:%s" % (arm["family"], nm)
        if key not in ok_arms:
            continue
        cmd = [sys.executable, str(G.HERE / "eval_gt.py"), "--score", arm["score"], "--cal", arm["cal"], "--win", str(arm["win"]),
               "--nmin", str(arm.get("nmin", 40)), "--dmin", str(arm.get("dmin", 0)), "--uni", arm.get("uni", "all"), "--dev"]
        G.say("== 开发集一次读：%s" % key)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-6000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
        K.record(a.exp, {"arm": key, "stdout_tail": r.stdout[-3000:]})
    G.progress("judge_gt：预登记 %s %d 个臂（筛选集 ΔP<0 剔 %d），开发集各读一次" % (a.exp, len(ok_arms), len(bad)))


if __name__ == "__main__":
    main()
