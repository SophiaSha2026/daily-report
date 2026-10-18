"""审计 A：基本事实（表的日期范围、2026 年标签有没有进研究表、族门槛现值、corp 缓存规模）。只读。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "breakout", ROOT / "src"):
    sys.path.insert(0, str(p))
EXP = ROOT / "data/breakout/raw/exp"
OUT = ROOT / "research/audit/pool"
L = []


def say(s=""):
    L.append(str(s))
    print(s, flush=True)


def rng(p, date="date", extra=None):
    cols = [date] + (extra or [])
    t = pd.read_parquet(p, columns=cols)
    d = t[date].astype(str)
    s = "%s: %d 行 %s..%s" % (Path(p).relative_to(ROOT), len(t), d.min(), d.max())
    if extra:
        for c in extra:
            m = np.isfinite(t[c].to_numpy(float))
            s += "; %s 有值的最大日期 %s（有值 %d 行，2026 年有值 %d 行）" % (c, d[m].max() if m.any() else "-", int(m.sum()), int((m & (d >= "2026-01-01")).sum()))
    say(s)
    return t


say("== 1 表的日期范围 / 2026 年标签 ==")
rng(EXP / "screen/px_screen.parquet")
rng(EXP / "screen/labels_screen.parquet", extra=["y_open", "y_up"])
rng(EXP / "rulecore/feat.parquet", extra=["y_open"])
rng(EXP / "polish/pool.parquet", extra=["y_open"])
rng(ROOT / "research/gate/table.parquet", extra=["y_open"])
rng(ROOT / "research/altdata/top60.parquet", extra=["y_open"])
rng(ROOT / "research/precision/top50.parquet", extra=["y_open"])
rng(EXP / "leak/train_snap.parquet", extra=["y_up"])
rng(EXP / "leak/labels_open_snap.parquet", extra=["y_open"])
rng(ROOT / "data/breakout/daily.parquet")
rng(ROOT / "data/breakout/train.parquet", extra=["y_up"])
for p in sorted((ROOT / "research/deepdata").glob("features_*.parquet")):
    rng(p)
rng(ROOT / "research/intraday/features_intraday.parquet")
# 确认集表
for n in ("px_screen_c.parquet", "labels_screen_c.parquet", "prod_screen_c.parquet"):
    p = EXP / "screen" / n
    if p.exists():
        rng(p, extra=(["y_open"] if "labels" in n else None))
p = EXP / "rulecore/feat_c.parquet"
if p.exists():
    rng(p, extra=["y_open"])

say("\n== 2 feat.parquet 的 2025-12 补标签行 ==")
f = pd.read_parquet(EXP / "rulecore/feat.parquet", columns=["date", "patched", "y_open", "end20"])
say("patched 行 %d，日期 %s..%s；其中 y_open 有值 %d" % (int(f["patched"].sum()), f.loc[f["patched"], "date"].min(), f.loc[f["patched"], "date"].max(),
                                                   int(np.isfinite(f.loc[f["patched"], "y_open"]).sum())))
dec = f[f["date"] >= "2025-12-01"]
say("2025-12 行 %d，y_open 有值 %d（%.1f%%），end20 > 2025-12-31 的行 %d（这些行的标签窗口按 pit 表没关，来自 leak 快照）" % (
    len(dec), int(np.isfinite(dec["y_open"]).sum()), 100 * np.isfinite(dec["y_open"]).mean(), int((dec["end20"] > "2025-12-31").sum())))
del f, dec

say("\n== 3 族门槛现值（evalkit.family_threshold）==")
import evalkit as K
for fam in ("breakout_y50_top10", "breakout_y50_top10_rulecore", "breakout_y50_top10_polish", "breakout_gate",
            "breakout_angles", "breakout_precision", "breakout_blind", "breakout_y50_top10_altdata"):
    try:
        th = K.family_threshold(fam)
        say("%-30s m=%d s=%d u=%d c=%.3f c_bonf=%.3f c_maxt=%s  %s" % (fam, th["m"], th["s"], th["u"], th["c"], th["c_bonf"],
                                                                        ("%.3f" % th["c_maxt"]) if np.isfinite(th["c_maxt"]) else "-", th["method"]))
    except Exception as e:  # noqa: BLE001
        say("%-30s ERR %s" % (fam, e))
rows = K._ledger_rows()
say("主登记簿 %d 行；按族 family_count：%s" % (len(rows), {f: K.family_count(f) for f in sorted({r.get("family") for r in rows if r.get("family")})}))
# 登记簿里 prereg 行的 gate_version / screen 字段
pr = [r for r in rows if r.get("kind") == "prereg"]
say("prereg 行 %d：gate_version 分布 %s；screen 字段分布 %s" % (len(pr), pd.Series([r.get("gate_version") for r in pr]).value_counts(dropna=False).to_dict(),
                                                        pd.Series([r.get("screen") for r in pr]).value_counts(dropna=False).to_dict()))
for r in rows:
    if r.get("family") in ("breakout_gate", "breakout_angles", "breakout_y50_top10_polish", "breakout_y50_top10_rulecore"):
        say("  " + json.dumps(r, ensure_ascii=False)[:230])

say("\n== 4 corp 缓存（长期调整突破逐只拉的公告缓存）==")
cc = list((ROOT / "data/pullback/corp").glob("*.json"))
say("data/pullback/corp/*.json %d 个" % len(cc))

say("\n== 5 ST 名单缓存 ==")
st = json.loads((ROOT / "cache/st_codes.json").read_text(encoding="utf-8"))
say("keys=%s n=%d" % (list(st.keys())[:6] if isinstance(st, dict) else "list", len(st.get("codes", st)) if isinstance(st, dict) else len(st)))
if isinstance(st, dict):
    for k in ("date", "at", "first_seen", "updated"):
        if k in st:
            say("  %s = %s" % (k, str(st[k])[:200]))
    seen = st.get("seen") or st.get("first_seen")
    if isinstance(seen, dict):
        say("  seen 样例 %s" % list(seen.items())[:5])

(OUT / "results_a_facts.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
