"""
预登记的候选（prereg_candidate.json，announce_c1）在开发集上算一次。
臂：base 模型前 10 / ann10 价格触发类公告里按模型分取前 10 / px10 日线 3 日涨幅过阈值里按模型分取前 10。
对照用 evalkit.compare（按天配对、按月配对、最差三个月、GATE），另加 Newey-West t 和 Bonferroni。
结果 append 进 prereg_candidate.json 的 results。只读开发集，不调 confirm_guard。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
import analyze as A  # noqa: E402
import evalkit as EK  # noqa: E402
from price_proxy import TH3  # noqa: E402

M_TESTS = 346
CATS = ("异常波动", "严重异常波动", "风险提示")


def pick(P: pd.DataFrame, flag: np.ndarray, k: int = 10) -> pd.DataFrame:
    x = P[flag].sort_values(["date", "p"], ascending=[True, False]).copy()
    x["rank"] = x.groupby("date").cumcount() + 1
    return x[x["rank"] <= k]


def main() -> None:
    P, E = A.load()
    # score 只在 top_base 前 200 里有，留着会让 evalkit.compare 按「生产规则」数空榜，这里只比前 10
    P = P.drop(columns=["score"]).reset_index(drop=True)
    stj = json.loads((ROOT / "cache" / "st_codes.json").read_text(encoding="utf-8"))
    st = {str(c).zfill(6) for c in stj["codes"]}
    notst = ~P["code"].isin(st).to_numpy()
    px = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet", columns=["code", "date", "close"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = px["date"].astype(str)
    px = px[(px["date"] >= "2024-04-01") & (px["date"] <= "2025-12-31")].sort_values(["code", "date"])
    px["r3"] = px["close"] / px.groupby("code")["close"].shift(3) - 1
    P = P.merge(px[["code", "date", "r3"]], on=["code", "date"], how="left")
    ann = A.dist_col(P, E[E["cat"].isin(CATS)], "avail_ei") <= 1
    prx = (P["r3"] >= P["board"].map(TH3).fillna(0.2)).to_numpy()

    base = P[P["rank"] <= 10].copy()
    arms = {"ann10": pick(P, ann & notst), "px10": pick(P, prx & notst)}
    out = {"at": dt.datetime.now().isoformat(timespec="seconds"),
           "note": "第 2 次运行：第 1 次 ST 名单读错（st_codes.json 是 dict，取成了键名，ST 没剔）且 score 列让空榜按生产规则数，两处机械修正，臂和过线标准不变"}
    lines = []
    for since, tag in (("", "全部"), ("2025-01-01", "2025 起")):
        b = base[base["date"] >= since] if since else base
        out[f"base|{tag}"] = EK.clustered(b)
        lines.append(f"[{tag}] base 前 10：{100 * out[f'base|{tag}']['hit']:.2f}% ±{100 * out[f'base|{tag}']['se']:.2f}"
                     f"（n={out[f'base|{tag}']['n']}）")
        for name, a in arms.items():
            aa = a[a["date"] >= since] if since else a
            c = EK.compare(aa, b, since="")
            da = aa.groupby("date")["y_open"].mean()
            db = b.groupby("date")["y_open"].mean()
            d = (da - db).dropna().sort_index().to_numpy()
            c["t_nw"] = A.nw_t(d)
            c["p_nw"] = A.p2(c["t_nw"])
            c["p_bonf"] = min(1.0, c["p_nw"] * M_TESTS)
            c["hit"] = EK.clustered(aa)
            c["per_day"] = float(len(aa) / max(b["date"].nunique(), 1))
            c["overlap_with_base"] = float(aa.merge(b[["date", "code"]], on=["date", "code"]).shape[0] / max(len(aa), 1))
            c["yizi1"] = float(aa["yizi1"].mean())
            c["hit_raw"] = float(aa["hit_raw"].mean())
            c["pass_all"] = bool(c["pass"] and c["t_nw"] >= 2.5 and c["p_bonf"] < 0.05)
            out[f"{name}|{tag}"] = c
            lines.append(f"[{tag}] {name}：命中 {100 * c['hit']['hit']:.2f}% ±{100 * c['hit']['se']:.2f}（n={c['hit']['n']}，"
                         f"每天 {c['per_day']:.1f} 只，和 base 重合 {100 * c['overlap_with_base']:.0f}%，次日一字 "
                         f"{100 * c['yizi1']:.1f}%，不管一字 {100 * c['hit_raw']:.2f}%）")
            lines.append("      " + EK.fmt_compare(name, c) + f" | NW t={c['t_nw']:.2f} Bonferroni p={c['p_bonf']:.3f}"
                         f" -> {'全部过线' if c['pass_all'] else '不过线'}")
            lines.append("      月度差：" + ", ".join(
                f"{k[2:]} {100 * v:+.1f}" for k, v in sorted((EK.monthly(aa) - EK.monthly(b)).dropna().items())))
    print("\n".join(lines))
    f = HERE / "prereg_candidate.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    j["results"].append(out)
    f.write_text(json.dumps(j, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (HERE / "cache" / "candidate.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
