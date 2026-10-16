"""
给试验登记簿补登历史试验的逐日差值序列（evalkit v3，2026-10-16，docs/design_2026-10-16.md 7.2 第 1 条）。

登记簿 data/breakout/raw/exp/ledger.jsonl 里实验 15~21 的 47 条（ledger_seed.jsonl）只有计数、没有序列，
族门槛只能退到 Bonferroni。这里把能从已有产物重算的序列按 evalkit.backfill_series 的格式补上
（kind=backfill，count=0，不算新试验），让 family_threshold 走「族内最大 t 零分布」那条路。

来源三处，全部只读 <= 2025-12-31 的产物，确认集一行不读：
    1 research/stats/out/daily_hits.parquet（recompute.py 产物）：实验 15/16（旧验证集 207 天）和
      17/18/21（开发集 387 天）每天前 10 买得到口径命中，41 个模型臂对各自的 base。
      核对：重算的 NW20 t 必须和 out/arms.csv 里 (arm, 10, y_open, span) 那一行逐位相同。
    2 arms.py 的读法：实验 21 生产规则口径 4 个事后臂（族 breakout_y50_prodrule，k=prod）和实验 19
      强势延续（族 cont_y15，前 5，y_cont）。同样对 arms.csv 核对。
    3 research/features 三个实验（feat_decomp / feat_fam_v1 / feat_comb_v1，共 15 个臂）的 top_*.parquet：
      对照按各自 prereg.json（feat_decomp 对 base，另两个对 base_a158）。核对：按天差均值和 NW t 必须和
      research/features/ledger.jsonl 的 result 行一致（那是当时用同一口径算的）。

不能补的（写明，不补）：实验 15 滚动分数线 2 个、环境门控 7 个、实验 17 滚动刻度 1 个（只有日志，无逐日产物）；
实验 18 g2「2025 起」子区间（同一条序列的事后加看，序列已由 dev 全段那条覆盖，不再登一条重复的）；
实验 20 诊断切片 30 个（breakout_y50_diag 族，无逐日产物）。

    python research/stats/backfill_ledger.py          # 核对 + 写登记簿
    python research/stats/backfill_ledger.py --dry    # 只核对、只打印，不写
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import evalkit as K   # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
SOURCE = "backfill 2026-10-16"
TOP10 = "breakout_y50_top10"
W_EXP = {"avg3": 17, "g2": 18, "hp_l15": 18, "hp_l63": 18, "hp_mc400": 18, "hp_lr02": 18, "hp_cs05": 18,
         "nostar": 18, "base_re": 21, "y20": 21, "y20_blend": 21, "lr_blend": 21, "pool_small": 21}
FEAT = {"feat_decomp": ("base", ["prod_rank", "base_pv", "pv_only", "rank_pv", "coarse_a158"]),
        "feat_fam_v1": ("base_a158", ["a158_on", "a158_vc", "a158_hi", "a158_ext", "a158_turn", "a158_rm", "a158_amp"]),
        "feat_comb_v1": ("base_a158", ["comb_all", "comb_sel", "comb_sel_cs4"])}


def arms_csv() -> pd.DataFrame:
    a = pd.read_csv(OUT / "arms.csv", encoding="utf-8-sig")
    a["k"] = a["k"].astype(str)
    return a


def ref_nw(a: pd.DataFrame, arm: str, k: str, y: str, span: str) -> float:
    r = a[(a["arm"] == arm) & (a["k"] == k) & (a["y"] == y) & (a["span"] == span)]
    if len(r) != 1:
        raise SystemExit(f"arms.csv 里 ({arm},{k},{y},{span}) 有 {len(r)} 行，核不了")
    return float(r["nw20_t"].iloc[0])


def plan() -> list[dict]:
    """每条要补的序列：family / exp / arm / ctrl / span / label / k / d（按天差，日期索引）/ ref（核对用的 NW t）。"""
    a = arms_csv()
    rows: list[dict] = []
    # 1 daily_hits
    hits = pd.read_parquet(OUT / "daily_hits.parquet")
    hits.index = hits.index.astype(str)
    assert hits.index.max() <= "2025-12-31", "daily_hits 含开发集之后的日期"
    for col in hits.columns:
        pre, arm = col.split(":", 1)
        if arm == "base":
            continue
        d = (hits[col] - hits[f"{pre}:base"]).dropna()
        exp = int(pre) if pre != "W" else W_EXP[arm]
        span = "val" if pre != "W" else "dev"
        rows.append(dict(family=TOP10, exp=exp, arm=arm, ctrl="base", span=span, label="y_open", k=10, d=d,
                         ref=ref_nw(a, arm, "10", "y_open", span), src=f"{SOURCE} daily_hits.parquet"))
    # 2 arms.py：生产规则 4 个 + 强势延续 1 个
    import arms as AR
    for r in AR.REGISTRY:
        if r["fam"] not in ("P", "C"):
            continue
        ta = AR.span_filter(AR.top_of(r, r["arm"]), r["span"])
        tb = AR.span_filter(AR.top_of(r, r["ctrl"]), r["span"])
        d = (AR.daily_hit(ta, r["k"], r["y"]) - AR.daily_hit(tb, r["k"], r["y"])).dropna().sort_index()
        d.index = d.index.astype(str)
        fam = "breakout_y50_prodrule" if r["fam"] == "P" else "cont_y15"
        rows.append(dict(family=fam, exp=r["exp"], arm=r["arm"], ctrl=r["ctrl"], span=r["span"], label=r["y"],
                         k=r["k"], d=d, ref=ref_nw(a, r["arm"], str(r["k"]), r["y"], r["span"]),
                         src=f"{SOURCE} arms.py({r['loader']})"))
    # 3 research/features 三个实验
    led = [json.loads(x) for x in (ROOT / "research" / "features" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for exp, (ctrl, arms) in FEAT.items():
        tb = pd.read_parquet(K.EXP / exp / f"top_{ctrl}.parquet")
        assert tb["date"].max() <= "2025-12-31", f"{exp}/{ctrl} 含开发集之后的日期"
        for arm in arms:
            ta = pd.read_parquet(K.EXP / exp / f"top_{arm}.parquet")
            d = (ta[ta["rank"] <= 10].groupby("date")["y_open"].mean()
                 - tb[tb["rank"] <= 10].groupby("date")["y_open"].mean()).dropna().sort_index()
            d.index = d.index.astype(str)
            res = [x for x in led if x.get("exp") == exp and x.get("arm") == arm and x.get("kind") == "result"]
            if len(res) != 1:
                raise SystemExit(f"research/features/ledger.jsonl 里 {exp}/{arm} 有 {len(res)} 条 result，核不了")
            rows.append(dict(family=TOP10, exp=exp, arm=arm, ctrl=ctrl, span="dev", label="y_open", k=10, d=d,
                             ref=float(res[0]["t_nw"]), ref_diff=float(res[0]["diff"]),
                             src=f"{SOURCE} {exp}/top_*.parquet 对 top_{ctrl}"))
    return rows


def main() -> int:
    dry = "--dry" in sys.argv
    rows = plan()
    bad = 0
    print("%-24s %-4s %-16s %-10s %-7s %-4s %5s %8s %8s %8s" % ("族", "实验", "臂", "对照", "区间", "k", "天", "均值pp", "NW t", "核对值"))
    for r in rows:
        t = K.nw_t(r["d"].to_numpy())
        ok = abs(t - r["ref"]) < 0.005 and ("ref_diff" not in r or abs(float(r["d"].mean()) - r["ref_diff"]) < 5e-5)
        bad += not ok
        print("%-24s %-4s %-16s %-10s %-7s %-4s %5d %+8.2f %8.2f %8.2f %s" % (
            r["family"], r["exp"], r["arm"], r["ctrl"], r["span"], r["k"], len(r["d"]), 100 * r["d"].mean(), t,
            r["ref"], "" if ok else "  <-- 对不上"))
    print(f"\n共 {len(rows)} 条，核对不上 {bad} 条")
    if bad:
        print("有核不上的，不写登记簿")
        return 1
    if dry:
        print("--dry：不写")
        return 0
    before = {f: K.family_count(f) for f in (TOP10, "breakout_y50_prodrule", "cont_y15")}
    n_new = 0
    for r in rows:
        p = K.backfill_series(r["family"], r["exp"], r["arm"], r["ctrl"], r["span"], r["d"], r["src"],
                              label=r["label"], k=r["k"])
        n_new += p is not None
    after = {f: K.family_count(f) for f in before}
    assert before == after, f"补登改变了试验次数：{before} -> {after}"
    print(f"写入 {n_new} 条（已有的跳过 {len(rows) - n_new} 条）；试验次数不变 {after}")
    for f in before:
        th = K.family_threshold(f)
        print("%-24s 门槛 c=%.3f  %s  （纯 Bonferroni %.3f，只看序列的 95%% 分位 %.3f）" % (
            f, th["c"], th["method"], th["c_bonf"], th["c_maxt"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
