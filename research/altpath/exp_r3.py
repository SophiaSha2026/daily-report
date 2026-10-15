"""
异动规则的日线版（research/announce 的建议，预登记 altpath_v3_r3）：生产特征 + 两列
    x_r3th_pct   r3 / 板块阈值（3 日涨幅；主板 20%、创业 / 科创 30%、北交 40%）的当日全市场排名
    x_r10_pct    10 日涨幅的当日全市场排名
对照同循环的 base（种子 7，等于 altpath_v1 的 base）。这块数据公告诊断已经看过，开发集结果带偏差。
和 altpath_v1 的 a158（含 ROC5/ROC10）、lu（涨停次数）信息有重叠，报告里写明。

    python research/altpath/exp_r3.py --fit / --analyze
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src" / "breakout", ROOT / "src", Path(__file__).parent, ROOT / "research" / "stats"):
    sys.path.insert(0, str(p))
import numpy as np, pandas as pd  # noqa: E401,E402
import evalkit as K  # noqa: E402
import exp_altpath as E  # noqa: E402
import hac  # noqa: E402

NAME = "altpath_v3_r3"
HYP = ("公告诊断发现 3 日 / 10 日涨幅相对板块异动阈值的票在模型 200 名外命中 12.4%。只加两列（r3/板块阈值、r10，"
       "当日全市场排名）对现状前 10 买得到口径按天配对，NW t >= 2.5 且最差三月不差且空榜不翻倍为过线；"
       "并入 altpath 族，共 11 个臂，Bonferroni 门槛 3.26。")
ARMS = {"r3r10": "生产特征 + x_r3th_pct + x_r10_pct"}
TH3 = {"main": 0.20, "chinext": 0.30, "star": 0.30, "bj": 0.40}
COLS = ["x_r3th_pct", "x_r10_pct"]


def build() -> pd.DataFrame:
    px = pd.read_parquet(K.DATA / "daily.parquet", columns=["code", "date", "close"])
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["date"] = px["date"].astype(str).str[:10]
    px = px[px["date"] < K.CONFIRM[0] + "-01"].sort_values(["code", "date"])
    g = px.groupby("code")["close"]
    px["r3"] = px["close"] / g.shift(3) - 1
    px["r10"] = px["close"] / g.shift(10) - 1
    return px


def fit_predict(tr, te, feats, month, state):
    import model as M
    trs = M.stratified_sample(tr, "y_up")
    ref = trs if len(trs) <= 600_000 else trs.sample(600_000, random_state=1)
    out = {}
    for name, cols in (("base", feats), ("r3r10", feats + COLS)):
        m = M.L1Lgbm(n_estimators=400, n_jobs=12).fit(trs, cols, "y_up")
        out[name] = (m.predict_proba(te), m.predict_proba(ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    K.prereg(NAME, HYP, ARMS)
    if a.fit:
        df = E.load()
        px = build()
        df = df.merge(px[["code", "date", "r3", "r10"]], on=["code", "date"], how="left")
        df["r3th"] = df["r3"] / df["board"].map(TH3).fillna(0.2)
        df["x_r3th_pct"] = df.groupby("date")["r3th"].rank(pct=True).astype("float32")
        df["x_r10_pct"] = df.groupby("date")["r10"].rank(pct=True).astype("float32")
        df = df.drop(columns=["r3", "r10", "r3th"])
        print("r3/r10 缺失 %.2f%%" % (100 * df["x_r3th_pct"].isna().mean()), flush=True)
        K.walk_forward(df, K.months_in(K.DEV), fit_predict, NAME)
    if a.analyze:
        d = K.EXP / NAME
        b = pd.read_parquet(d / "top_base.parquet")
        t = pd.read_parquet(d / "top_r3r10.parquet")
        b1 = pd.read_parquet(K.EXP / "altpath_v1" / "top_base.parquet")
        same = np.allclose(b1.sort_values(["date", "code"])["p"].to_numpy(), b.sort_values(["date", "code"])["p"].to_numpy())
        lines = ["自检：base 和 altpath_v1 base 逐位相同 = %s" % same]
        for since, lbl in (("", "全部"), ("2025-01-01", "2025 起")):
            c = E.cmp(t, b, since=since)
            lines.append("[%s] %s" % (lbl, E.fmt("r3r10", E.summ(t, since), c)))
            dd = E.daily_diff(t, b, since=since)
            lines.append("    块自举 95%% 区间 [%+.2f, %+.2f]pp" % tuple(100 * x for x in (hac.mbb(dd)["lo"], hac.mbb(dd)["hi"])))
        mo = sorted({x[:7] for x in b["date"]})
        for nm, x in (("base", b), ("r3r10", t)):
            r = K.monthly(x[x["rank"] <= 10])
            lines.append("%-6s" % nm + "".join("%6.1f" % (100 * r.get(m, np.nan)) for m in mo))
        txt = "\n".join(lines)
        print(txt)
        (Path(__file__).parent / "r3.txt").write_text(txt, encoding="utf-8")
        K.record(NAME, {"text": txt})
    return 0


if __name__ == "__main__":
    sys.exit(main())
