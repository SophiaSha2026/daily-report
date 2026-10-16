"""
第 3.5 步：一致性核对（重叠段）。
    特征：prod_screen（2020-07 起建）对 leak 的 prod_pit（2023-05-30 起建，同一 pit 口径）：行集合、逐列相关 /
          |差|<1e-4 的一致率，按半年分段看差异怎么随「离本表起点多远」收敛；再对 train_snap（生产 fwd 表）
    标签：y_up 对 train_snap 的 y_up、y_open 对 labels_open_snap 的 y_open（2024-06..2025-12 和全部重叠段）
产物 research/screen/results_check.txt、check_feats.csv

    python research/screen/check_screen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_screen as CS  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import features as F  # noqa: E402

L: list[str] = []
FAM = {c: g for g, cs in F.GROUPS.items() for c in cs}
FAM.update({n: "interaction" for *_, n in F.INTERACTIONS})


def cmp(name: str, m: pd.DataFrame, cols: list[str], sb: str = "_re", tol: float = 1e-4) -> pd.DataFrame:
    rows = []
    for c in cols:
        a, b = m[c].to_numpy(float), m[c + sb].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        agree = float((np.abs(a[ok] - b[ok]) < tol).mean()) if ok.any() else float("nan")
        nanmis = float((np.isfinite(a) != np.isfinite(b)).mean())
        cc = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan")
        rows.append((c, FAM.get(c.rsplit("__", 1)[0], "?"), agree, nanmis, cc))
    df = pd.DataFrame(rows, columns=["col", "family", "agree", "nanmis", "cc"]).sort_values("cc")
    CS.say("[%s] %d 列：相关中位 %.4f 最低 %.4f；一致率(|差|<%g) 中位 %.3f；NaN 位置不一致最大 %.4f" % (
        name, len(df), df["cc"].median(), df["cc"].min(), tol, df["agree"].median(), df["nanmis"].max()), L)
    fam = df.groupby("family")["cc"].agg(["median", "min"])
    CS.say("  按族相关（中位/最低）：" + "  ".join("%s %.3f/%.3f" % (k, r["median"], r["min"]) for k, r in fam.iterrows()), L)
    CS.say("  相关最低 8 列：" + "; ".join("%s %.3f" % (r.col, r.cc) for _, r in df.head(8).iterrows()), L)
    return df


def main():
    import pyarrow.parquet as pq
    pit_min = str(pq.read_table(CS.LEAK / "prod_pit.parquet", columns=["date"])["date"].to_pandas().min())
    keys = pd.read_parquet(CS.SCR / "prod_screen.parquet", columns=["code", "date"])
    ps = pd.read_parquet(CS.SCR / "prod_screen.parquet", filters=[("date", ">=", pit_min)])
    feats = [c for c in ps.columns if "__" in c]
    CS.say("prod_screen %d 行 %d 只 %s..%s，%d 列" % (len(keys), keys["code"].nunique(), keys["date"].min(), keys["date"].max(), len(feats)), L)
    cnt = keys.groupby("date")["code"].nunique()
    del keys
    CS.say("  每日只数：首日 %d，2022-01 %d，2023-06 %d，2024-05 %d，末日 %d" % (
        int(cnt.iloc[0]), int(cnt[cnt.index >= "2022-01-01"].iloc[0]), int(cnt[cnt.index >= "2023-06-01"].iloc[0]),
        int(cnt[cnt.index >= "2024-05-01"].iloc[0]), int(cnt.iloc[-1])), L)
    pit = pd.read_parquet(CS.LEAK / "prod_pit.parquet")
    CS.say("prod_pit（leak，起点 2023-05-30）%d 行 %s..%s" % (len(pit), pit["date"].min(), pit["date"].max()), L)
    sub = ps[ps["date"] >= pit["date"].min()]
    m = sub.merge(pit, on=["code", "date"], suffixes=("", "_re"), how="outer", indicator=True)
    CS.say("重叠段 %s 起：本表 %d 行，prod_pit %d 行，对上 %d；只在本表 %d（%d 只）只在 prod_pit %d（%d 只）" % (
        pit["date"].min(), len(sub), len(pit), int((m["_merge"] == "both").sum()),
        int((m["_merge"] == "left_only").sum()), m.loc[m["_merge"] == "left_only", "code"].nunique(),
        int((m["_merge"] == "right_only").sum()), m.loc[m["_merge"] == "right_only", "code"].nunique()), L)
    lo = m[m["_merge"] == "left_only"]
    if len(lo):
        CS.say("  只在本表的行按月（前 6）：%s" % lo.groupby(lo["date"].str[:7]).size().head(6).to_dict(), L)
    ro = m[m["_merge"] == "right_only"]
    if len(ro):
        CS.say("  只在 prod_pit 的行按月（前 6）：%s" % ro.groupby(ro["date"].str[:7]).size().head(6).to_dict(), L)
    m = m[m["_merge"] == "both"].drop(columns="_merge")
    del pit, sub
    CS.say("board 一致率 %.5f" % float((m["board"] == m["board_re"]).mean()), L)
    d_all = cmp("prod_screen 对 prod_pit，全部重叠行", m, feats)
    d_dev = cmp("prod_screen 对 prod_pit，开发集 2024-06..2025-12", m[m["date"] >= "2024-06-01"], feats)
    half = m["date"].str[:4] + "H" + ((m["date"].str[5:7].astype(int) > 6).astype(int) + 1).astype(str)
    for h in sorted(half.unique()):
        mm = m[half == h]
        d = cmp("prod_screen 对 prod_pit，%s" % h, mm, feats)
        d["half"] = h
    d_all["half"] = "all"
    d_dev["half"] = "dev"
    pd.concat([d_all, d_dev]).to_csv(CS.HERE / "check_feats.csv", index=False)
    # 和生产 fwd 表（train_snap）比
    snap = pd.read_parquet(CS.LEAK / "train_snap.parquet", columns=["code", "date", "y_up"] + feats,
                           filters=[("date", "<=", CS.KEEP_TO)])
    snap["code"] = snap["code"].astype(str).str.zfill(6)
    snap["date"] = snap["date"].astype(str).str[:10]
    ms = ps[ps["date"] >= "2024-06-01"].merge(snap, on=["code", "date"], suffixes=("", "_re"), how="inner")
    cmp("prod_screen 对 train_snap（生产 fwd 表），2024-06..2025-12", ms, feats)
    del m, ms
    # ---- 标签 ----
    lab = pd.read_parquet(CS.SCR / "labels_screen.parquet")
    ml = lab.merge(snap[["code", "date", "y_up"]], on=["code", "date"], suffixes=("", "_snap"), how="inner")
    del snap
    for lo_, hi_, tag in (("2024-06-01", "2025-12-31", "开发集"), (ml["date"].min(), "2025-12-31", "全部重叠")):
        x = ml[(ml["date"] >= lo_) & (ml["date"] <= hi_)]
        both = x[np.isfinite(x["y_up"]) & np.isfinite(x["y_up_snap"])]
        CS.say("y_up 对 train_snap（%s %s..%s）：对上 %d 行，两边都有标签 %d，一致 %.4f%%；本表 NaN 快照有值 %d，反之 %d；"
               "正样本率 本表 %.3f%% 快照 %.3f%%" % (
                   tag, lo_, hi_, len(x), len(both), 100 * float((both["y_up"] == both["y_up_snap"]).mean()),
                   int((~np.isfinite(x["y_up"]) & np.isfinite(x["y_up_snap"])).sum()),
                   int((np.isfinite(x["y_up"]) & ~np.isfinite(x["y_up_snap"])).sum()),
                   100 * both["y_up"].mean(), 100 * both["y_up_snap"].mean()), L)
    lo2 = pd.read_parquet(CS.LEAK / "labels_open_snap.parquet", columns=["code", "date", "y_open", "r20_open", "yizi1"])
    mo = lab.merge(lo2, on=["code", "date"], suffixes=("", "_snap"), how="inner")
    for lo_, hi_, tag in (("2024-06-01", "2025-12-31", "开发集"), (mo["date"].min(), "2025-12-31", "全部重叠")):
        x = mo[(mo["date"] >= lo_) & (mo["date"] <= hi_)]
        both = x[np.isfinite(x["y_open"]) & np.isfinite(x["y_open_snap"])]
        rr = both["r20_open"] - both["r20_open_snap"]
        CS.say("y_open 对 labels_open_snap（%s）：两边都有 %d，一致 %.4f%%（不一致 %d 行：一字板判定不同 %d，r20 跨 50%% %d）；"
               "r20_open |差| 中位 %.2e 最大 %.2e；一字板一致 %.4f%%" % (
                   tag, len(both), 100 * float((both["y_open"] == both["y_open_snap"]).mean()),
                   int((both["y_open"] != both["y_open_snap"]).sum()),
                   int(((both["y_open"] != both["y_open_snap"]) & (both["yizi1"] != both["yizi1_snap"])).sum()),
                   int(((both["y_open"] != both["y_open_snap"]) & (both["yizi1"] == both["yizi1_snap"])).sum()),
                   float(rr.abs().median()), float(rr.abs().max()), 100 * float((both["yizi1"] == both["yizi1_snap"]).mean())), L)
    (CS.HERE / "results_check.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
