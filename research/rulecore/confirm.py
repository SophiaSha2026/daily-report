"""
确认集 2026-01..08 那一次：规则核 对 现状模型（拍板 E）。**本脚本只写好不执行；主会话问过用户再跑。**
每季度只能看一次：进场先过 evalkit.confirm_guard（看了就记 out_breakout/confirm_used.json，本季度再跑直接拒绝）。

数据（和两段判定同一条路，只把截止日从 2025-12-31 推到 2026-09-30）
    1 pit 价格   research/screen/prices_screen.main（CS.KEEP_TO 改成 2026-09-30）-> px_screen_c.parquet
                 unadj_2020.parquet 是 2026-10-16 拉的，到 2026-10-15，8 月的 20 根标签窗口已关
    2 标签       research/screen/labels_screen.main -> labels_screen_c.parquet（y_up / y_open / r20_open / yizi1）
    3 生产特征   research/screen/build_screen.main -> prod_screen_c.parquet（约 1 小时，4 进程；股东户数公告日 <= 2026-09-30）
    4 事件表     events_rc 拉到 2026-09-30（独立目录 events_c/，全部重拉约 40 分钟）
    5 规则核     common_rc 同一套特征 / z 化 / 候选池，rulecore.fit_monotone 逐月 2026-01..08
    6 现状       pc_arm 同一条 LightGBM（生产特征 + 每月 fselect，缓存键 "<月>|screen_v1"，种子 7）逐月 2026-01..08
    7 判定       同一份候选池，每天前 10 买得到命中按天配对；前 30、AUC、三句事实并印

确认集的过线规则（预先写死，主会话问用户确认后再跑）
    8 个月约 160 天，按天差的 NW se 约 2pp，功效不够做 t >= 2.5 的检验，所以确认集只做「方向 + 不显著变坏」：
      a 规则核 − 现状 前 10 按天差 >= 0
      b NW t > -1.0（不显著为负）
      c 规则核前 10 命中 >= 2 x 候选池当日基准（按天平均）
      d 前 30 差和前 10 同号
    四条全满足 = 确认通过 -> 按教训 36 的流水换生产；a 或 c 不满足 = 不通过，规则核留作影子；其余 = 未定，再等一季实盘影子

    PYTHONIOENCODING=utf-8 python research/rulecore/confirm.py --selfcheck      只查依赖和参数，不读 2026 年任何一行
    PYTHONIOENCODING=utf-8 python research/rulecore/confirm.py --run            真跑（过 confirm_guard，分钟到小时级）
    PYTHONIOENCODING=utf-8 python research/rulecore/confirm.py --run --skip-build   价格 / 标签 / 特征表已建好时只跑 5~7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_rc as C  # noqa: E402

KEEP_TO_C = "2026-09-30"
CONFIRM = ("2026-01", "2026-08")
PX_C, LAB_C, FEAT_C = "px_screen_c.parquet", "labels_screen_c.parquet", "prod_screen_c.parquet"
NAME_C = "rulecore_confirm"
OUT_C = C.EXP / NAME_C
EVENTS_C = C.RC / "events_c"
# 主候选 ret20（用户 2026-10-17 拍板：两段判定后 ret20 主尺子过、卡在 AUC 淘汰条文和 2024-09，拿确认集验一次）；
# rc / pc 照原预登记并列报，不参与这次的上线决定
ARMS = ("ret20", "rc", "pc")
GATE_C = {"diff_day_min": 0.0, "t_nw_min": -1.0, "lift_min": 2.0, "top30_same_sign": True}


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------
#  1~4 建表（都是 research/screen 的脚本，只改截止日和文件名，不改它们）
# ---------------------------------------------------------------
def build_tables() -> None:
    import common_screen as CS
    import build_screen as BS
    import labels_screen as LS
    import prices_screen as PS
    CS.KEEP_TO = KEEP_TO_C
    assert CS.KEEP_TO == KEEP_TO_C
    if not (CS.SCR / PX_C).exists():
        sys.argv = ["prices_screen.py"]
        # prices_screen 的产物名写死为 px_screen.parquet：让它写进临时子目录，再搬到正式名字，原件不碰
        real = CS.SCR
        tmp = real / "_confirm_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        CS.SCR = tmp
        try:
            PS.main()
            (tmp / "px_screen.parquet").replace(real / PX_C)
        finally:
            CS.SCR = real
        say("价格表 ->", CS.SCR / PX_C)
    if not (CS.SCR / LAB_C).exists():
        sys.argv = ["labels_screen.py", "--px", PX_C, "--out", LAB_C]
        LS.main()
    if not (CS.SCR / FEAT_C).exists():
        sys.argv = ["build_screen.py", "--px", PX_C, "--out", FEAT_C, "--workers", "4"]
        BS.main()
    import events_rc as E
    E.OUT, E.PARTS, E.END = EVENTS_C, EVENTS_C / "parts", KEEP_TO_C
    if not (EVENTS_C / "status.json").exists() or not (EVENTS_C / "reduce_ann.parquet").exists():
        E.main()


def features_c() -> pd.DataFrame:
    """common_rc.build_features 的确认集版：换输入文件、不补 leak 标签（这张表自己的窗口到 2026-09-30 已关）。"""
    f = C.RC / "feat_c.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import common_screen as CS
    import daily as D
    import validate as V
    px = pd.read_parquet(CS.SCR / PX_C, columns=["code", "date", "open", "high", "low", "close", "turnover", "outstanding_share"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    px["ret20"] = g["close"].pct_change(20)
    px["lret20"] = np.log1p(px["ret20"].clip(lower=-0.99))
    lr = np.log(px["close"]).groupby(px["code"]).diff()
    px["vol60"] = lr.groupby(px["code"]).transform(lambda s: s.rolling(60).std())
    px["amp"] = px["high"] / px["low"] - 1
    px["amp20"] = px.groupby("code", sort=False)["amp"].transform(lambda s: s.rolling(20).mean())
    px["turn20"] = px.groupby("code", sort=False)["turnover"].transform(lambda s: s.rolling(20).mean())
    px["nbar"] = g.cumcount() + 1
    dates = np.array(sorted(px["date"].unique()))
    didx = pd.Series(np.arange(len(dates)), index=dates)
    px["_di"] = px["date"].map(didx).to_numpy()
    gap = (px["_di"] - px.groupby("code", sort=False)["_di"].shift(1) - 1).fillna(0)
    px["susp20"] = gap.groupby(px["code"]).transform(lambda s: s.rolling(20, min_periods=1).max()) >= C.SUSP_GAP
    px["end20"] = g["date"].shift(-20).fillna("9999-12-31")
    from numpy.lib.stride_tricks import sliding_window_view as swv

    def fut(gr):
        n = len(gr)
        lo_, cl, op = gr["low"].to_numpy(float), gr["close"].to_numpy(float), gr["open"].to_numpy(float)
        mn, c20 = np.full(n, np.nan), np.full(n, np.nan)
        if n > 20:
            mn[:n - 20] = swv(lo_[1:], 20).min(axis=1)
            c20[:n - 20] = cl[20:]
        o1 = np.append(op[1:], np.nan)
        return pd.DataFrame({"dd20": mn / o1 - 1, "c20": c20 / o1 - 1}, index=gr.index)

    px = px.join(pd.concat([fut(gr) for _, gr in px.groupby("code", sort=False)]))
    px["board"] = C.board_of(px["code"])
    st = V.st_codes()
    px["elig0"] = (~px["code"].isin(st) & (px["close"] >= C.PRICE_MIN) & (px["nbar"] >= D.MIN_HISTORY_DAYS)
                   & ~px["susp20"] & np.isfinite(px[list(C.INPUTS)]).all(axis=1))
    for c in C.INPUTS:
        s = px[c].where(px["elig0"])
        px["z_" + c] = (px[c] - s.groupby(px["date"]).transform("mean")) / (s.groupby(px["date"]).transform("std") + 1e-12)
    lab = pd.read_parquet(CS.SCR / LAB_C, columns=["code", "date", "y_up", "y_open", "r20_open", "yizi1"])
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    px = px.merge(lab, on=["code", "date"], how="left")
    px["patched"] = False
    keep = ["code", "date", "board", "close", "ret20", "nbar", "susp20", "elig0", "end20", "dd20", "c20",
            "y_up", "y_open", "r20_open", "yizi1", "patched"] + list(C.INPUTS) + ["z_" + c for c in C.INPUTS]
    px = px[keep].reset_index(drop=True)
    px.to_parquet(f, index=False)
    return px


# ---------------------------------------------------------------
#  5 规则核、6 现状（逐月，确认集 8 个月）
# ---------------------------------------------------------------
def run_rc(df: pd.DataFrame) -> pd.DataFrame:
    import model as M
    import rulecore as R
    import evalkit as K
    parts, coefs = [], {}
    for m in K.months_in(CONFIRM):
        tm = C.train_mask(df, m)
        tr = df[tm & df["elig0"].to_numpy() & np.isfinite(df["y_open"]).to_numpy()]
        te = df[(df["date"].str[:7] == m).to_numpy() & df["elig0"].to_numpy()]
        beta = R.fit_monotone(R.design(tr), tr["y_open"].to_numpy(), M.month_weights(tr))
        p = 1.0 / (1.0 + np.exp(-(R.design(te) @ beta)))
        parts.append(pd.DataFrame({"date": te["date"].to_numpy(), "code": te["code"].to_numpy(), "p": p.astype(np.float32)}))
        coefs[m] = {k: float(v) for k, v in zip(list(C.INPUTS) + ["b_" + b for b in C.BOARDS[1:]] + ["intercept"], beta)}
        say("  rc %s 训练 %d 行 w=%s" % (m, len(tr), np.round(beta[:4], 3)))
    (OUT_C / "coef_rc.json").write_text(json.dumps(coefs, ensure_ascii=False, indent=1), encoding="utf-8")
    o = pd.concat(parts, ignore_index=True)
    o.to_parquet(OUT_C / "all_rc.parquet", index=False)
    # ret20 一行字：同一候选池按 lret20 排（和 rulecore.py 两段判定的参照臂同一口径）
    te = df[df["date"].str[:7].isin(K.months_in(CONFIRM)).to_numpy() & df["elig0"].to_numpy()]
    pd.DataFrame({"date": te["date"].to_numpy(), "code": te["code"].to_numpy(),
                  "p": te["lret20"].to_numpy(np.float32)}).to_parquet(OUT_C / "all_ret20.parquet", index=False)
    return o


def run_base(feat: pd.DataFrame) -> pd.DataFrame:
    import common_screen as CS
    import evalkit as K
    import pc_arm as PA
    CS.KEEP_TO = KEEP_TO_C
    months = K.months_in(CONFIRM)
    lo = C.month_shift(months[0], -12 * C.TRAIN_YEARS) + "-01"
    hi = C.month_shift(months[-1], 1) + "-01"
    df = pd.read_parquet(CS.SCR / FEAT_C, filters=[("date", ">=", lo), ("date", "<", hi)])
    lab = pd.read_parquet(CS.SCR / LAB_C, columns=["code", "date", "y_up", "y_open"])
    df = df.merge(lab, on=["code", "date"], how="left")
    df["y_up"], df["y_open"] = df["y_up"].astype("float32"), df["y_open"].astype("float32")
    df = df.merge(feat[["code", "date", "lret20"]].rename(columns={"lret20": PA.RAW}), on=["code", "date"], how="left")
    df[PA.RAW] = df[PA.RAW].astype("float32")

    def fit_predict(tr, te, feats, month, state):
        import model as M
        trs = M.stratified_sample(tr, "y_up", seed=PA.SEED)
        ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
        out = {}
        for nm, cols in (("base", list(feats)), ("pc", PA.pc_cols(feats))):
            mdl = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
            out[f"{nm}_s{PA.SEED}"] = (mdl.predict_proba(te), mdl.predict_proba(ref))
        return out

    # evalkit.walk_forward 读的是 df 里的日期，确认集月份的 te 来自 prod_screen_c，没有 load_frame 那道 2026 封印
    # （封印由本脚本进场的 confirm_guard 承担）
    for m in months:
        sub = df[(df["date"] >= C.month_shift(m, -12 * C.TRAIN_YEARS) + "-01") & (df["date"] < C.month_shift(m, 1) + "-01")]
        K.walk_forward(sub, [m], fit_predict, f"{NAME_C}/parts/{m}", drop=[PA.DROP_KEY])
    outs = {}
    for nm in ("base", "pc"):
        o = pd.concat([pd.read_parquet(OUT_C / "parts" / m / f"all_{nm}_s{PA.SEED}.parquet") for m in months], ignore_index=True)
        o.to_parquet(OUT_C / f"all_{nm}.parquet", index=False)
        outs[nm] = o
    return outs["base"]


# ---------------------------------------------------------------
#  7 判定
# ---------------------------------------------------------------
def judge_c(df: pd.DataFrame) -> dict:
    import evalkit as K
    import judge as J
    C.EVENTS = EVENTS_C                       # event_flags 读确认集的事件表
    lo, hi = CONFIRM[0] + "-01", C.month_shift(CONFIRM[1], 1) + "-01"
    d = df[(df["date"] >= lo) & (df["date"] < hi) & np.isfinite(df["y_open"]) & np.isfinite(df["y_up"])].reset_index(drop=True)
    ev = C.event_flags(d)
    elig = d["elig0"] & ~ev["any"]
    tops, P = {}, {}
    for arm in ARMS + ("base",):
        ap = pd.read_parquet(OUT_C / f"all_{arm}.parquet")
        ap["code"] = ap["code"].astype(str).str.zfill(6)
        P[arm] = d[["code", "date"]].merge(ap, on=["code", "date"], how="left")["p"].to_numpy(float)
        tops[arm] = C.rank_table(d, P[arm], elig)
        tops[arm].to_parquet(OUT_C / f"top_{arm}.parquet", index=False)
    base_day = d[elig].groupby("date")["y_open"].mean()
    res = {"days": int(base_day.size), "pool_base": float(base_day.mean()), "gate": GATE_C}
    hb10, hb30 = C.daily_hit(tops["base"], 10), C.daily_hit(tops["base"], 30)
    for arm in ARMS:
        h10, h30 = C.daily_hit(tops[arm], 10), C.daily_hit(tops[arm], 30)
        d10 = (h10 - hb10.reindex(h10.index)).dropna()
        d30 = (h30 - hb30.reindex(h30.index)).dropna()
        m10, _, t10 = J.nw(d10)
        m30, _, t30 = J.nw(d30)
        cmp_ = K.compare(tops[arm], tops["base"], y="y_open", k=10, family=C.FAMILY, exp=NAME_C, arm=arm)
        lift = float(h10.mean() / base_day.mean())
        ok = {"a_diff": m10 >= GATE_C["diff_day_min"], "b_t": t10 > GATE_C["t_nw_min"], "c_lift": lift >= GATE_C["lift_min"],
              "d_top30": bool(np.sign(m30) == np.sign(m10))}
        verdict = "确认通过" if all(ok.values()) else ("不通过" if not (ok["a_diff"] and ok["c_lift"]) else "未定")
        sub = tops[arm][tops[arm]["rank"] <= 10].merge(d[["code", "date", "ret20", "r20_open", "c20", "dd20"]], on=["code", "date"])
        res[arm] = {"hit10": float(h10.mean()), "hit30": float(h30.mean()), "diff10": m10, "t10": t10, "diff30": m30, "t30": t30,
                    "lift": lift, "checks": ok, "verdict": verdict, "boot_p": cmp_["boot"]["p"], "facts": J.facts(sub),
                    "zero10": float((h10 == 0).mean())}
        say("[%s] 前10 %.2f%%（基准 %.2f%%，%.2f 倍）对现状 %+.2fpp NW t %.2f；前30 %+.2fpp；%s -> %s" % (
            arm, 100 * h10.mean(), 100 * base_day.mean(), lift, 100 * m10, t10, 100 * m30, ok, verdict))
    res["market_facts"] = J.facts(d[elig])
    (OUT_C / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    K.record(NAME_C, {"verdicts": {a: res[a]["verdict"] for a in ARMS}, "days": res["days"]})
    return res


def selfcheck() -> None:
    """不读 2026 年任何一行：只核依赖、函数签名、确认集月份、事件脚本常量可改。"""
    import common_screen as CS
    import evalkit as K
    import events_rc as E
    import pc_arm as PA
    import rulecore as R
    assert hasattr(K, "confirm_guard") and hasattr(K, "walk_forward") and K.CONFIRM == CONFIRM, K.CONFIRM
    assert callable(R.fit_monotone) and callable(R.design) and callable(PA.pc_cols)
    assert K.months_in(CONFIRM) == ["2026-0%d" % i for i in range(1, 9)]
    assert (CS.LEAK / "unadj_2020.parquet").exists(), "缺 unadj_2020.parquet（research/leak/fetch_unadj.py）"
    assert E.END == "2026-01-31" and KEEP_TO_C > E.END
    say("selfcheck 通过：确认集 %s..%s，截止 %s，产物目录 %s；没有读 2026 年数据" % (*CONFIRM, KEEP_TO_C, OUT_C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not a.run:
        ap.print_help()
        return
    if C.in_prod_window():
        raise SystemExit("北京 16:00~18:30 不跑")
    import evalkit as K
    K.confirm_guard(C.NAME, "ret20")             # 看了就记；本季度第二次直接 SystemExit
    OUT_C.mkdir(parents=True, exist_ok=True)
    (OUT_C / "prereg.json").write_text(json.dumps({
        "name": NAME_C, "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "family": C.FAMILY, "gate_version": 3,
        "hypothesis": "确认集 2026-01..08：主候选 ret20（每天候选池内 20 日涨幅前 10）对 现状 base（同一候选池、同一条 LightGBM "
                      "走向前），方向 + 不显著变坏。ret20 是两段判定之后经用户拍板加为主候选的（事后决定，如实记录）；"
                      "rc / pc 照原预登记并列报。",
        "arms": {"ret20": "20 日涨幅前 10（主候选）", "rc": "规则核", "pc": "百分位压缩对照"}, "gate": GATE_C, "span": list(CONFIRM), "results": []},
        ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    if not a.skip_build:
        build_tables()
    df = features_c()
    df["date"] = df["date"].astype(str)
    run_rc(df)
    run_base(df[["code", "date", "lret20"]])
    res = judge_c(df)
    C.progress("确认集看过一次：%s（%.0f 分钟）" % ({k: res[k]["verdict"] for k in ARMS}, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
