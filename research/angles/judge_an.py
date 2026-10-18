"""
正式臂：预登记（evalkit 新族 breakout_angles，实验 angles_v1）+ 开发集一次读（criteria 第 0、5、8e 节）。

    PYTHONIOENCODING=utf-8 python research/angles/judge_an.py

臂：regime R1 / R3（R2 = B(3) 不立，R4 筛选集 ΔP < 0 不登记）、crash C1、conformal 1c 挑出的每 τ 一个（筛选集 ΔP < 0 的不登记）。
筛选集按天差 = mean(u_screen)（ΔP x V / D）。prereg 后分别调各脚本 --dev 读开发集。
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

import common_an as A


def main():
    import evalkit as K
    arms, screen = {}, {}
    rj = json.loads((A.HERE / "results_regime_dev.json").read_text(encoding="utf-8"))
    for key in ("R1", "R3", "R4"):
        u = np.array(rj[key]["T"]["u_screen"])
        screen["regime:" + key] = {"diff_day": float(u.mean()), "days": int(len(u)), "span": list(A.SEGS["screen"])}
        arms["regime:" + key] = {"R1": "冰点 + 启动 期才出候选池 lret20 前 3（ClawHub 三条件阈值）",
                                 "R3": "冰点' + 启动' 期才出前 3（去掉跌停条件，8f）",
                                 "R4": "亢奋' + 退潮' 期不出、其余出前 3（8f）"}[key]
    cj = json.loads((A.HERE / "results_crash_dev.json").read_text(encoding="utf-8"))
    u = np.array(cj["C1"]["u_screen"])
    screen["crash:C1"] = {"diff_day": float(u.mean()), "days": int(len(u)), "span": list(A.SEGS["screen"])}
    arms["crash:C1"] = "候选池等权指数 252 日收益 < 0 且 21 日收益 > 0 的日子不出，其余出前 10"
    conf_arms = []
    fj = json.loads((A.HERE / "results_conformal_dev.json").read_text(encoding="utf-8"))
    for tau, pick in fj["picks"].items():
        key = pick[1]
        u = np.array(fj["res"][key]["u_screen"])
        screen["conformal:" + key] = {"diff_day": float(u.mean()), "days": int(len(u)), "span": list(A.SEGS["screen"])}
        arms["conformal:" + key] = "conformal p 值 + BH(α = 1 − τ)，校准前 12 个月，配置按筛选集 |P − τ| 最小挑：" + key
        conf_arms.append(key)
    ok, bad = {}, []
    for k, v in arms.items():
        if screen[k]["diff_day"] >= 0:
            ok[k] = v
        else:
            bad.append(k)
            A.say("!! %s 筛选集按天差 %+.4f < 0，不登记、不读开发集" % (k, screen[k]["diff_day"]))
            A.ledger_add({"kind": "screen_reject", "name": k, "screen_diff_day": screen[k]["diff_day"]})
    hyp = ("四个角度提精度：情绪阶段门控（只在冰点 / 启动期出前 3）、动量崩溃反弹日不出、conformal 静态门槛；"
           "出的票的买得到精确率对同量名次基线 B(k̄) 有增量，且两段块自举下界 >= B(10)")
    p = K.prereg(A.EXP_NAME, hyp, ok, metric="精确率 P 对同量名次基线 B(k̄)，按天序列 u_d = h_d − h_B,d", family=A.FAMILY,
                 screen={k: screen[k] for k in ok})
    A.say("预登记 -> %s（%d 个臂，拒绝 %d）" % (p, len(ok), len(bad)))
    for k in ok:
        A.ledger_add({"kind": "prereg", "name": k, "screen_diff_day": screen[k]["diff_day"]})
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env}
    runs = []
    reg = [k.split(":")[1] for k in ok if k.startswith("regime:")]
    if reg:
        runs.append(("regime", [sys.executable, str(A.HERE / "regime_an.py"), "--dev", "--arms", ",".join(reg)]))
    if "crash:C1" in ok:
        runs.append(("crash", [sys.executable, str(A.HERE / "crash_an.py"), "--dev"]))
    ca = [k.split(":", 1)[1] for k in ok if k.startswith("conformal:")]
    if ca:
        runs.append(("conformal", [sys.executable, str(A.HERE / "conformal_an.py"), "--dev", "--arms", "|".join(ca)]))
    for name, cmd in runs:
        A.say("== 开发集一次读：%s" % name)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        print(r.stdout[-8000:])
        if r.returncode != 0:
            print(r.stderr[-3000:])
        K.record(A.EXP_NAME, {"arm_group": name, "stdout_tail": r.stdout[-4000:]})
    A.progress("judge_an：预登记 %s %d 个臂（拒 %d：%s），开发集各读一次" % (A.EXP_NAME, len(ok), len(bad), bad))


if __name__ == "__main__":
    main()
