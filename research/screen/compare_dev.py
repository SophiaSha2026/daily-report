"""
第 5 步（附加）：同一张长表上 2024-06..2025-12 的 base（screen_base_v1_dev，种子 7，3 年窗口）对 leak 的 base_pit_s7
（生产口径的表：2023-05-30 起、扩张窗口、同一 pit 价格、开发集月份用生产缓存的特征列表）。回答：
    表变长（筹码记忆、250 日窗口、训练窗口）对开发集成绩有没有影响；两条线上的 base 是不是同一个量级。
不写主登记簿（compare 不带 exp/arm）。

    python research/screen/compare_dev.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import exp_screen as X  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evalkit as K  # noqa: E402

L: list[str] = []


def main():
    months = CS.months_between(*K.DEV)
    res = X.analyze("screen_base_v1_dev", seeds=(7,), book=True, months=months)
    a = X.with_r20(X.top("screen_base_v1_dev", "base_s7"))
    b = pd.read_parquet(K.EXP / "leak_v1" / "top_base_pit_s7.parquet")
    lab = pd.read_parquet(CS.LEAK / "labels_open_snap.parquet", columns=["code", "date", "r20_open", "yizi1"])
    b = b.merge(lab, on=["code", "date"], how="left")
    b["lr20"] = np.log1p(b["r20_open"].where(~b["yizi1"].fillna(False).astype(bool), 0.0).clip(lower=-0.99))
    CS.say("长表 base_s7（screen_base_v1_dev）%d 天，leak base_pit_s7 %d 天" % (a["date"].nunique(), b["date"].nunique()), L)
    r = X.screen_cmp(a, b)
    CS.say("长表 - 生产口径表：前 10 买得到 %.2f%% 对 %.2f%%，按天 %+.2fpp NW se %.2fpp NW t %.2f 块自举 p %.3f；"
           "lr20 %+.4f NW t %.2f（%d 天）" % (100 * r["hit_a"], 100 * r["hit_b"], 100 * r["diff_day"], 100 * r["se_nw"],
                                           r["t_nw"], r["boot"]["p"], r["aux_diff"], r["aux_t_nw"], r["days"]), L)
    ja = a[a["rank"] <= 10].groupby("date")["code"].apply(set)
    jb = b[b["rank"] <= 10].groupby("date")["code"].apply(set)
    jac = float(np.mean([len(ja[k]) and len(ja[k] & jb[k]) / len(ja[k] | jb[k]) for k in ja.index if k in jb.index]))
    CS.say("前 10 名单日均 Jaccard %.2f（换种子级 0.58、换价格版本 0.53）" % jac, L)
    c = K.compare(a, b)
    CS.say("evalkit.compare：%s" % K.fmt_compare("long_vs_prod", c), L)
    ma = a[a["rank"] <= 10].groupby(a["date"].str[:7])["y_open"].mean()
    mb = b[b["rank"] <= 10].groupby(b["date"].str[:7])["y_open"].mean()
    CS.say("月度（%）：\n" + (100 * pd.DataFrame({"长表": ma, "生产口径表": mb})).round(1).to_string(), L)
    CS.say("长表 base 开发集：按天 %.2f%% 基准 %.2f%% 倍数 %.2f" % (
        100 * res["base_s7"]["hit_day"], 100 * res["base_day"], res["base_s7"]["lift_day"]), L)
    (CS.HERE / "results_dev.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
