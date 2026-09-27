"""
提案的自动实验。LLM 只说「试什么」，数字全由这里的确定性代码给。

三类能自动做：
    param         早盘白名单参数。走 eval_daily.evaluate_candidate（和优化器候选同一套七道闸）
    threshold     起涨预测常量 SCORE_MIN / CAP_A / MIN_STREAK。用缓存的逐月滚动分数重算
                  生产口径命中率表（秒级）。BOARD_ADJ 等要重打分的标 needs_human
    feature_drop  起涨预测消融。exp_window.py 的 WF_DROP，先保证有一份和当前特征表
                  同步的全特征基线（各约 10 分钟）

过闸标准（写死在代码里，不由 LLM 定）：
    起涨：全部上榜命中率不低于基线 −1 个标准误 且 样本数 ≥ 基线 80%，
          或 连续≥2天档提升 ≥ 2 个标准误
    早盘：gate.evaluate 七道闸的 accepted

结果写回台账 decisions.json：status passed / failed / needs_human，result 里带数字。
过闸的等控制台批准（approve），批准后 apply() 落地：
    param         learn.apply.write -> state/learned.yaml（和自动接受同一条路）
    threshold /   state/breakout/overrides.json，daily.py 读；feature_drop 进指纹，下次打分自动重训
    feature_drop
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))   # daily / features / exp_window 按脚本方式 import

from learn.council import run as R     # noqa: E402

log = logging.getLogger("council.exp")
WF_CACHE = ROOT / "data" / "breakout" / "raw" / "wf_scores.parquet"
BASE_DIR = ROOT / "data" / "breakout" / "raw" / "council_base"
OVERRIDES = ROOT / "state" / "breakout" / "overrides.json"
TRAIN = ROOT / "data" / "breakout" / "train.parquet"
MAX_DROP_PER_RUN = 2

THRESHOLD_KEYS = {"SCORE_MIN", "CAP_A", "MIN_STREAK"}
BOARDS = ("main", "star", "bj", "chinext")


def _rescore_board_adj(d: pd.DataFrame, new_adj: dict) -> pd.DataFrame:
    """用缓存分数检验一组新的板块系数。

    缓存里的 _p 已经乘过旧系数，除回去得到原始预测值；每月「97 分」对应的
    _p 门槛取该月 score>=97 的最小 _p（分位映射单调，缓存每天前 100 名足够密）。
    重排后每天前 100 之外的票进不来，是近似；系数只降不升时没有影响。
    """
    import daily as D
    import features as F
    d = d.copy()
    d["board"] = d["code"].map(F.board_of)
    d["m"] = d["date"].str[:7]
    # 门槛要在改系数**之前**算：改完再算，被压低的那个板块会把最小值拖下去
    thr = d[d["score"] >= 97].groupby("m")["_p"].min()
    old = d["board"].map(D.BOARD_ADJ).fillna(1.0)
    new = d["board"].map({**D.BOARD_ADJ, **new_adj}).fillna(1.0)
    d["_p"] = d["_p"] / old * new
    d["score"] = np.where(d["_p"] >= d["m"].map(thr).fillna(np.inf), 97, 0)
    d = d.sort_values(["date", "_p"], ascending=[True, False])
    d["rank"] = d.groupby("date").cumcount() + 1
    return d.drop(columns=["m"])


# ---------------------------------------------------------------------
#  起涨：阈值类（缓存分数上重算）
# ---------------------------------------------------------------------
def _w5(d: pd.DataFrame, score_min: int, cap: int, min_streak: int = 1) -> dict:
    """生产口径：score ≥ score_min 且当天前 cap 名；连续按上榜天数。"""
    days = sorted(d["date"].unique())
    di = {x: i for i, x in enumerate(days)}
    ok = d[(d["score"] >= score_min) & (d["rank"] <= cap)].copy()
    ok["_i"] = ok["date"].map(di)
    by = {c: set(g["_i"].astype(int)) for c, g in ok.groupby("code")}
    st = []
    for c, i in zip(ok["code"], ok["_i"]):
        s, i = 1, int(i)
        while (i - s) in by[c]:
            s += 1
        st.append(s)
    ok["streak"] = st
    out = {}
    for k in (1, 2, 3):
        g = ok[ok["streak"] >= max(k, min_streak)]
        n = int(len(g))
        hit = float(g["y_up"].mean()) if n else float("nan")
        se = math.sqrt(hit * (1 - hit) / n) if n and hit == hit else float("nan")
        out[f"ge{k}"] = {"n": n, "hit": hit, "se": se,
                         "per_day": n / max(1, len(days))}
    out["empty_days"] = int(len(days) - ok["date"].nunique())
    return out


def _pass_breakout(base: dict, cand: dict) -> tuple[bool, str]:
    b1, c1 = base["ge1"], cand["ge1"]
    b2, c2 = base["ge2"], cand["ge2"]
    if not c1["n"]:
        return False, "候选清单为空"
    se = max(b1["se"] if b1["se"] == b1["se"] else 0.0, 1e-9)
    keep_hit = c1["hit"] >= b1["hit"] - se
    keep_n = c1["n"] >= 0.8 * b1["n"]
    up2 = (c2["n"] >= 30 and b2["n"] >= 30 and b2["se"] == b2["se"]
           and c2["hit"] - b2["hit"] >= 2 * max(b2["se"], c2["se"]))
    if keep_hit and keep_n:
        return True, (f"全部上榜 {100 * c1['hit']:.1f}% vs 基线 {100 * b1['hit']:.1f}%（−1SE 线 "
                      f"{100 * (b1['hit'] - se):.1f}%），样本 {c1['n']}/{b1['n']}")
    if up2:
        return True, (f"连续≥2天 {100 * c2['hit']:.1f}% vs {100 * b2['hit']:.1f}%，"
                      f"提升 ≥ 2SE（样本 {c2['n']}）")
    # 以覆盖换精度：全部上榜命中率提升 ≥ 2SE 且样本还剩四分之一以上。
    # 这是取舍不是改进，过闸但写明，批不批人定。
    se1 = max(b1["se"] if b1["se"] == b1["se"] else 0.0,
              c1["se"] if c1["se"] == c1["se"] else 0.0, 1e-9)
    if c1["hit"] - b1["hit"] >= 2 * se1 and c1["n"] >= 0.25 * b1["n"]:
        return True, (f"以覆盖换精度：全部上榜 {100 * c1['hit']:.1f}% vs {100 * b1['hit']:.1f}%"
                      f"（≥2SE），但样本从 {b1['n']} 降到 {c1['n']}，每天出票变少")
    return False, (f"全部上榜 {100 * c1['hit']:.1f}% vs 基线 {100 * b1['hit']:.1f}%，"
                   f"样本 {c1['n']}/{b1['n']}；连续≥2天 {100 * c2['hit']:.1f}% vs "
                   f"{100 * b2['hit']:.1f}%")


def _current_thresholds() -> dict:
    import daily as D  # noqa: WPS433
    return {"SCORE_MIN": int(D.SCORE_MIN), "CAP_A": int(D.CAP_A), "MIN_STREAK": 1}


def exp_threshold(p: dict) -> dict:
    """返回 {status, detail, base, cand}。"""
    params = p.get("params") or {}
    vals = {}
    for k, v in params.items():
        kk = str(k).upper()
        if kk in THRESHOLD_KEYS:
            try:
                vals[kk] = int(float(v))
            except Exception:  # noqa: BLE001
                pass
    # 板块系数：params 里 BOARD_ADJ.star / board_adj.star / star 这类键
    adj = {}
    for k, v in params.items():
        kk = str(k).lower().replace("board_adj.", "").replace("board_adj_", "")
        if kk in BOARDS:
            try:
                adj[kk] = float(v)
            except Exception:  # noqa: BLE001
                pass
    if not vals and not adj:
        m = re.search(r"(SCORE_MIN|CAP_A|MIN_STREAK)\D+(\d+)", p.get("change", ""), re.I)
        if m:
            vals[m.group(1).upper()] = int(m.group(2))
    if not vals and not adj:
        return {"status": "needs_human",
                "detail": "只能自动测 SCORE_MIN / CAP_A / MIN_STREAK / BOARD_ADJ.<板块>；其它常量要重打分"}
    if not WF_CACHE.exists():
        return {"status": "needs_human", "detail": "没有逐月滚动分数缓存（先跑 exp_window.py）"}
    d = pd.read_parquet(WF_CACHE)
    cur = _current_thresholds()
    new = {**cur, **vals}
    base = _w5(d, cur["SCORE_MIN"], cur["CAP_A"], cur["MIN_STREAK"])
    dc = _rescore_board_adj(d, adj) if adj else d
    cand = _w5(dc, new["SCORE_MIN"], new["CAP_A"], new["MIN_STREAK"])
    ok, why = _pass_breakout(base, cand)
    if adj:
        new["BOARD_ADJ"] = adj
    return {"status": "passed" if ok else "failed", "detail": why,
            "base": {**cur, **base}, "cand": {**new, **cand},
            "note": "缓存分数来自上次逐月滚动测试，和当前特征表可能差一天数据"
                    + ("；板块系数是在缓存的前 100 名上重排的近似" if adj else "")}


# ---------------------------------------------------------------------
#  起涨：消融
# ---------------------------------------------------------------------
def _run_wf(drop: list[str], cache: Path, out: Path, timeout: int = 1800) -> dict | None:
    """跑一次逐月滚动测试。返回 grid，并把 exp_window 打的「剩几列」记进 grid["_kept"]，
    这样「消融到底有没有真的去掉列」在台账里看得见（2026-09-16 两次消融都是空转，
    就是因为没人核对这个数）。"""
    env = dict(os.environ, WF_DROP=",".join(drop), WF_CACHE=str(cache), WF_OUT=str(out),
               PYTHONIOENCODING="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "src" / "breakout" / "exp_window.py"),
                        "--refit"], cwd=str(ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0 or not out.exists():
        log.warning("exp_window 失败 rc=%s: %s", r.returncode, (r.stderr or "")[-300:])
        return None
    g = json.loads(out.read_text(encoding="utf-8"))
    m = re.search(r"消融：排除 (.+)，剩 (\d+) 列", r.stdout or "")
    if m:
        g["_kept"] = int(m.group(2))
    m2 = re.search(r"筛3 相关剪枝: 丢 \d+，剩 (\d+)", r.stdout or "")
    if m2:
        g["_selected"] = int(m2.group(1))
    return g


def _w5_from_grid(grid: dict) -> dict:
    out = {}
    for r in grid.get("grid", []):
        if r.get("kind") != "W5":
            continue
        m = re.search(r"连续≥(\d)天", r["label"])
        if m and int(m.group(1)) in (1, 2, 3):
            out[f"ge{int(m.group(1))}"] = {"n": int(r["n"]), "hit": float(r["hit"]),
                                          "se": float(r["se"]), "per_day": float(r["per_day"])}
    return out


def _fresh(out: Path) -> dict | None:
    """比特征表新的成绩表直接复用（一次逐月滚动要 10 分钟，重复评估没必要再跑）。"""
    if out.exists() and TRAIN.exists() and out.stat().st_mtime >= TRAIN.stat().st_mtime:
        try:
            g = json.loads(out.read_text(encoding="utf-8"))
            g["_cached"] = True
            return g
        except Exception:  # noqa: BLE001
            return None
    return None


def _baseline_grid() -> dict | None:
    """和当前特征表同步的全特征基线；旧了就重算（约 10 分钟）。"""
    out = BASE_DIR / "window_grid.json"
    g = _fresh(out)
    if g:
        return g
    log.info("消融基线过期或缺失，重算全特征基线（约 10 分钟）")
    return _run_wf([], BASE_DIR / "wf_scores.parquet", out)


def exp_feature_drop(p: dict) -> dict:
    feats = (p.get("params") or {}).get("features") or []
    feats = [str(f).strip() for f in feats if str(f).strip()]
    if not feats:
        return {"status": "needs_human", "detail": "提案没给要去掉的特征名（params.features）"}
    known = _known_features()
    if not known:
        # 早先这里 import 失败会让 known 变空、校验被跳过，于是拿一个根本不存在的
        # 特征名跑了十分钟空转（2026-09-16）。拿不到特征表就别跑。
        return {"status": "failed", "detail": "读不到起涨预测的特征名单（features.GROUPS），不跑消融"}
    bad = [f for f in feats if f not in known
           and not (f.rsplit("__", 1)[0] in known and f.rsplit("__", 1)[-1] in ("last", "mean", "slope"))]
    if bad:
        return {"status": "failed",
                "detail": f"特征名不存在：{bad}（起涨预测的基础特征名见 features.GROUPS）"}
    base = _baseline_grid()
    if not base:
        return {"status": "failed", "detail": "基线重算失败"}
    tag = "_".join(feats)[:40]
    cout = BASE_DIR / f"drop_{tag}" / "window_grid.json"
    cand = _fresh(cout) or _run_wf(feats, cout.parent / "wf_scores.parquet", cout)
    if not cand:
        return {"status": "failed", "detail": "消融跑失败"}
    b, c = _w5_from_grid(base), _w5_from_grid(cand)
    if "ge1" not in b or "ge1" not in c:
        return {"status": "failed", "detail": "成绩表里没有 W5 行"}
    kept_b, kept_c = base.get("_kept"), cand.get("_kept")
    sel_b, sel_c = base.get("_selected"), cand.get("_selected")
    same = all(abs(b[k]["hit"] - c[k]["hit"]) < 1e-12 and b[k]["n"] == c[k]["n"]
               for k in ("ge1", "ge2", "ge3") if k in b and k in c)
    if same or (sel_b is not None and sel_b == sel_c and same):
        # 逐位相同 = 这些列本来就没进模型（筛选那一步已经丢了），实验什么也没测
        return {"status": "failed",
                "detail": f"空转：去掉 {feats} 前后成绩逐位相同，说明这些列本来就没被选进模型"
                          f"（入模 {sel_b} -> {sel_c} 列）。要测它们得先确认在 feature_select 的保留集里",
                "base": b, "cand": c, "dropped": feats, "kept": [kept_b, kept_c],
                "selected": [sel_b, sel_c]}
    ok, why = _pass_breakout(b, c)
    return {"status": "passed" if ok else "failed",
            "detail": why + f"；入模列数 {sel_b} -> {sel_c}",
            "base": b, "cand": c, "dropped": feats,
            "kept": [kept_b, kept_c], "selected": [sel_b, sel_c]}


def _known_features() -> set[str]:
    """起涨预测的基础特征名（六组 + 交互项）。

    交互项 2026-09-16 起是五元组 (a, sa, b, sb, name)，早先这里写的是
    `set(F.INTERACTIONS)`，并进来的是一堆元组，永远匹配不上特征名 ——
    消融会把所有提案判成「特征名不存在」。取最后一项才是名字。
    """
    try:
        import features as F
        base = {f for fs in F.GROUPS.values() for f in fs}
        inter = {t[-1] for t in getattr(F, "INTERACTIONS", []) if t}
        return base | inter
    except Exception as e:  # noqa: BLE001
        log.warning("读不到特征名单: %s", e)
        return set()


# ---------------------------------------------------------------------
#  早盘：参数类
# ---------------------------------------------------------------------
def exp_param(p: dict, c: dict, date: str = "") -> dict:
    """一组参数值走和优化器候选同一套七道闸。

    两个坑（2026-09-16 学习线那批修完后暴露的）：
    1. 以前返回的是这里自己夹箱得到的 theta，而 evaluate_candidate 内部还会
       再走一次 sparsify（权重要再归一到和为 1），过闸的和落地的不是同一份。
       现在一律返回 res["theta"]，也就是真正被判过的那份。
    2. 冷却期按「今天」算会差一个交易日：提案是那天的会诊出的，闸门要按
       提案日判，否则跨午夜重跑结果会变。
    """
    params = p.get("params") or {}
    import cfg as C
    import eval_daily as ED
    # 用**裁剪过的**箱：数据源给不出的维度（回填表没有竞价轨迹 -> trend、
    # 竞价额含盘后成交 -> volume）被钉死成基线值。拿未裁剪的箱夹提案值，
    # 一条动 trend 的提案会在 evaluate_candidate 里被投影回基线，闸门给出的
    # 理由是「参数没有实际改动」，看不出真正的原因是「这一维当前不可学」。
    box = ED._learn_box(c)
    pinned = [k for k, (lo, hi) in box.items() if abs(hi - lo) < 1e-12]
    prev = C.theta_now(box)          # 循环前的基准，下面 theta 会被原地改
    theta = dict(prev)
    asked = {}
    for k, v in params.items():
        if k in box:
            try:
                lo, hi = box[k]
                nv = float(np.clip(float(v), lo, hi))
                asked[k] = nv
                theta[k] = nv
            except Exception:  # noqa: BLE001
                pass
    if not asked:
        return {"status": "needs_human",
                "detail": f"参数不在可学白名单里（{sorted(box)}），或没给 params"}
    hit_pinned = [k for k in params if k in pinned]
    if hit_pinned and all(k in pinned for k in asked):
        return {"status": "needs_human",
                "detail": f"这几维当前数据源不可学，已钉死在人工基线：{hit_pinned}。"
                          f"回填表给不出竞价轨迹（trend）、竞价额含盘后成交（volume），"
                          f"要动它们得先有足够的在线真值天，或换数据源"}
    date = date or p.get("date") or dt.date.today().isoformat()
    res = ED.evaluate_candidate(c, theta, date)
    if not res:
        return {"status": "failed", "detail": "评估失败（数据不够或异常）"}
    final = res.get("theta") or theta
    v = res["verdict"]
    moved = {k: [prev[k], final[k]] for k in box
             if abs(float(final[k]) - float(prev[k])) > 1e-9}
    return {"status": "passed" if v.get("accepted") else "failed",
            "detail": "；".join(f"{ck['name']}{'过' if ck['passed'] else '不过'}" for ck in v.get("checks", [])),
            "moved": moved, "asked": asked,
            "verdict": v, "metrics": res.get("metrics"), "theta": final}


# ---------------------------------------------------------------------
def run_pending(c: dict, max_drop: int = MAX_DROP_PER_RUN,
                date: str = "") -> list[dict]:
    """把台账里 pending 的自动实验跑掉，结果写 decisions.json。"""
    done = []
    n_drop = 0
    box = set((c.get("learning") or {}).get("box") or {})
    for p in R.ledger_view():
        if p["status"] != "pending" or not p.get("auto_testable"):
            continue
        # LLM 常把早盘的可学参数（screen.auc_ratio_score_hi 之类）标成 threshold；
        # 只要 params 的键都在学习白名单里，就按 param 走七道闸
        params = p.get("params") or {}
        if p["kind"] == "threshold" and params and all(k in box for k in params):
            p = {**p, "kind": "param"}
        R.write_decision(p["id"], "testing")
        try:
            if p["kind"] == "threshold":
                res = exp_threshold(p)
            elif p["kind"] == "feature_drop" and p.get("line") == "morning":
                res = {"status": "needs_human",
                       "detail": "早盘的特征在 score.py 里，去掉要改代码（和 vscore 同步），不能自动消融"}
            elif p["kind"] == "feature_drop":
                if n_drop >= max_drop:
                    R.write_decision(p["id"], "pending", note="本次消融额度用完，下次再跑")
                    continue
                n_drop += 1
                res = exp_feature_drop(p)
            elif p["kind"] == "param":
                res = exp_param(p, c, date or p.get("date", ""))
            else:
                res = {"status": "needs_human", "detail": "不能自动做"}
        except Exception as e:  # noqa: BLE001
            res = {"status": "failed", "detail": f"实验异常 {type(e).__name__}: {e}"}
        R.write_decision(p["id"], res["status"], result=res)
        log.info("提案 %s %s：%s -> %s", p["id"], p["kind"], p.get("change", "")[:40],
                 res["status"])
        done.append({"id": p["id"], **res})
    return done


# ---------------------------------------------------------------------
#  批准后落地
# ---------------------------------------------------------------------
def read_overrides() -> dict:
    try:
        return json.loads(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_overrides(obj: dict) -> None:
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDES.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(OVERRIDES)


def apply(pid: str, c: dict, by: str = "gui") -> dict:
    """控制台批准 -> 落地。只接受 passed 的提案。"""
    rows = {p["id"]: p for p in R.ledger_view()}
    p = rows.get(pid)
    if not p:
        return {"ok": False, "error": "没有这条提案"}
    if p["status"] not in ("passed", "approved"):
        return {"ok": False, "error": f"状态是 {p['status']}，只有过闸的能落地"}
    res = p.get("result") or {}
    try:
        if p["kind"] == "param":
            from learn import apply as A, gate
            theta = res.get("theta")
            if not theta:
                return {"ok": False, "error": "实验结果里没有参数"}
            evidence = {"source": "council", "proposal": pid, "date": p["date"],
                        **(res.get("verdict") or {}).get("evidence", {})}
            A.write(theta, evidence, dt.date.today().isoformat())
            try:
                gate.HISTORY.parent.mkdir(parents=True, exist_ok=True)
                with gate.HISTORY.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"date": dt.date.today().isoformat(),
                                        "ts": dt.datetime.now().isoformat(timespec="seconds"),
                                        "theta": theta, "source": "council", "proposal": pid},
                                       ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                log.warning("theta_history 追加失败: %s", e)
            what = f"learned.yaml <- {res.get('moved')}"
        elif p["kind"] in ("threshold", "feature_drop"):
            ov = read_overrides()
            if p["kind"] == "threshold":
                cand = res.get("cand") or {}
                for k in THRESHOLD_KEYS:
                    if k in cand:
                        ov[k] = int(cand[k])
                if isinstance(cand.get("BOARD_ADJ"), dict):
                    ov["BOARD_ADJ"] = {**(ov.get("BOARD_ADJ") or {}),
                                       **{k: float(v) for k, v in cand["BOARD_ADJ"].items()}}
                what = f"overrides {({k: ov.get(k) for k in list(THRESHOLD_KEYS) + ['BOARD_ADJ']})}"
            else:
                cur = set(ov.get("drop_features") or [])
                cur |= set(res.get("dropped") or [])
                ov["drop_features"] = sorted(cur)
                what = f"drop_features {ov['drop_features']}（下次打分自动重训）"
            ov.setdefault("history", []).append(
                {"proposal": pid, "at": dt.datetime.now().isoformat(timespec="seconds"), "by": by})
            _write_overrides(ov)
        else:
            return {"ok": False, "error": "这类提案要人手工做"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    R.write_decision(pid, "applied", by=by, note=what)
    return {"ok": True, "what": what}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import cfg as C
    for r in run_pending(C.load()):
        print(json.dumps(r, ensure_ascii=False)[:300])
