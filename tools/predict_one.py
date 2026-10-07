"""
单只股票预测：输入代码，用生产模型给它打分（2026-10-07 用户要求）。

    python tools/predict_one.py 600000
    python tools/predict_one.py 600000 --json      给控制台用（/api/predict）

用的是**和每天清单完全同一套东西**：state/breakout/model.txt（模型）、model.json
（入模列、分数刻度）、state/breakout/board_adj.json（板块系数）、data/breakout/train.parquet
（特征表）。打分日 = 特征表最后一天，也就是最近一次「起涨预测」跑完补到的那天；要看今天的，
先在控制台跑一次起涨预测（或试跑）。

读表只读两小块：这只票的全部行（看近 20 天的分数走势）和打分日的全市场行（算名次），
几秒钟，不用把 2.7 GB 的表整个读进来。

输出和清单 A 同口径：预测值 × 板块系数 -> 0~100 分（参照样本的分位数，不是概率）->
全市场名次（剔 ST、剔上市不足 120 个交易日的）-> 够不够格（≥ SCORE_MIN 分且前 CAP_A）。
风险剔除里的减持 / 解禁 / 增发要联网拉全市场表，这里不查，只查 ST 名单缓存，输出里写明。
历史命中率印的是邮件里同一组常量（export.STREAK_PERF 按连续天数、score_calibration 按分数档），
都是验证集 2025-03..12 的数字；邮件口径是收盘买入，按次日开盘能买到的口径要打八折
（实验 16：17.6% -> 14.1%）。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

DATA = ROOT / "data" / "breakout"
STATE = ROOT / "state" / "breakout"
DAYS = 20


def _name_of(code: str) -> str:
    """名称要联网问腾讯，拿不到就空着（这不是主线，不能因为它失败）。"""
    try:
        import datasource as ds
        q = ds.fetch_quotes([ds.to_symbol(code)])
        for _, v in q.items():
            return str(getattr(v, "name", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return ""


def _why_missing(code: str) -> str:
    try:
        codes = set(pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"].str.zfill(6))
    except Exception:  # noqa: BLE001
        codes = set()
    if codes and code not in codes:
        return "不在代码表 cache/codes.csv 里（退市、代码错，或代码表没刷新）"
    dp = DATA / "daily.parquet"
    if dp.exists():
        try:
            n = len(pd.read_parquet(dp, filters=[("code", "==", code)], columns=["date"]))
        except Exception:  # noqa: BLE001
            n = -1
        if n == 0:
            return "日线表里没有这只票（北交所新股或日线没补到）"
        if 0 < n <= 120:
            return f"日线只有 {n} 根，上市不足 120 个交易日的票不打分（daily.MIN_HISTORY_DAYS）"
    return "特征表里没有这只票的行（特征表是最近一次起涨预测建的，先跑一次起涨预测）"


def predict(code: str, days: int = DAYS, with_name: bool = True) -> dict:
    import lightgbm as lgb
    import daily as D
    import export as E
    import validate as V

    code = str(code).strip().zfill(6)
    if not code.isdigit() or len(code) != 6:
        return {"ok": False, "code": code, "error": "代码要是 6 位数字"}
    mp = STATE / "model.txt"
    meta_p = STATE / "model.json"
    tp = DATA / "train.parquet"
    for f in (mp, meta_p, tp):
        if not f.exists():
            return {"ok": False, "code": code, "error": f"缺 {f.relative_to(ROOT)}，先跑一次起涨预测"}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    feats = list(meta["feats"])
    q = np.asarray(meta["quantiles"], dtype=float)
    booster = lgb.Booster(model_file=str(mp))

    mine = pd.read_parquet(tp, filters=[("code", "==", code)],
                           columns=["code", "date", "board", "close"] + feats)
    if mine.empty:
        return {"ok": False, "code": code, "error": _why_missing(code)}
    mine = mine.sort_values("date").tail(days).reset_index(drop=True)
    last = str(pd.read_parquet(tp, columns=["date"])["date"].max())
    today = pd.read_parquet(tp, filters=[("date", "==", last)],
                            columns=["code", "date", "board"] + feats)
    cnt = pd.read_parquet(tp, columns=["code"])["code"].value_counts()

    def _p(df: pd.DataFrame) -> np.ndarray:
        X = np.nan_to_num(df[feats].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        adj = df["board"].map(D.BOARD_ADJ).fillna(1.0).to_numpy(float)
        return booster.predict(X) * adj

    # 打分日全市场：剔 ST（缓存名单）、剔次新，按预测值排名
    st = V.st_codes()
    today["_p"] = _p(today)
    today["score"] = D.to_score(today["_p"].to_numpy(), q)
    elig = (today["code"].map(cnt).fillna(0) >= D.MIN_HISTORY_DAYS) & ~today["code"].isin(st)
    pool = today[elig].sort_values("_p", ascending=False).reset_index(drop=True)
    n_q = int((pool["score"] >= D.SCORE_MIN).sum())
    rank = int(pool.index[pool["code"] == code][0]) + 1 if (pool["code"] == code).any() else None

    mine["_p"] = _p(mine)
    mine["score"] = D.to_score(mine["_p"].to_numpy(), q)
    hist = [{"date": str(r.date), "score": int(r.score), "close": round(float(r.close), 2)}
            for r in mine.itertuples()]
    stale = str(mine["date"].iloc[-1]) != last
    sc = int(mine["score"].iloc[-1])
    streak = 0
    for s in mine["score"].to_numpy()[::-1]:
        if s >= D.SCORE_MIN:
            streak += 1
        else:
            break
    qualified = (not stale) and sc >= D.SCORE_MIN and rank is not None and rank <= D.CAP_A
    is_st = code in st
    hit_streak, lift_streak = E.streak_perf(max(streak, 1))
    bin_hit = None
    try:
        cal = json.loads((ROOT / "out_breakout" / "score_calibration.json").read_text(encoding="utf-8"))
        for b in cal.get("bins", []):
            if b["lo"] <= sc < b["hi"]:
                bin_hit = {"lo": b["lo"], "hi": b["hi"], "hit": b["hit"], "lift": b["lift"], "n": b["n"]}
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True, "code": code,
        "name": _name_of(code) if with_name else "",
        "score_date": last, "stale": stale, "last_row": str(mine["date"].iloc[-1]),
        "p": float(mine["_p"].iloc[-1]), "score": sc,
        "board": str(mine["board"].iloc[-1]),
        "board_adj": float(D.BOARD_ADJ.get(str(mine["board"].iloc[-1]), 1.0)),
        "rank": rank, "n_pool": int(len(pool)), "n_qualified": n_q,
        "score_min": int(D.SCORE_MIN), "cap_a": int(D.CAP_A),
        "qualified": bool(qualified), "streak": streak, "is_st": is_st,
        "eligible": bool(cnt.get(code, 0) >= D.MIN_HISTORY_DAYS),
        "perf_streak": {"hit": hit_streak, "lift": lift_streak, "k": streak},
        "perf_bin": bin_hit, "base": float(E.BASE),
        "history": hist,
        "note": "风险剔除只查了 ST 名单缓存，减持 / 解禁 / 增发没查；历史命中率是验证集"
                "收盘买入口径，按次日开盘能买到的口径约打八折",
    }


def fmt(r: dict) -> str:
    if not r.get("ok"):
        return f"{r.get('code', '')}：{r.get('error', '')}"
    nm = f" {r['name']}" if r.get("name") else ""
    lines = [f"{r['code']}{nm}  打分日 {r['score_date']}"
             + ("（这只票最后一行是 %s，打分日没有它的数据）" % r["last_row"] if r["stale"] else "")]
    lines.append("分数 %d / 100（预测值 %.4f，含%s系数 %.3f）  全市场第 %s / %d，当天够格 %d 只"
                 % (r["score"], r["p"], {"main": "主板", "star": "科创", "chinext": "创业",
                                          "bj": "北交"}.get(r["board"], r["board"]),
                    r["board_adj"], r["rank"] if r["rank"] else "-", r["n_pool"], r["n_qualified"]))
    if r["is_st"]:
        lines.append("ST：清单 A 不收")
    elif not r["eligible"]:
        lines.append("上市不足 120 个交易日：清单 A 不收")
    lines.append(("够格：会上清单 A" if r["qualified"]
                  else "不够格（上清单要 ≥%d 分且当天前 %d 名）" % (r["score_min"], r["cap_a"]))
                 + "  连续够格 %d 天" % r["streak"])
    hs = " ".join("%s:%d" % (h["date"][5:], h["score"]) for h in r["history"])
    lines.append("近 %d 天分数  %s" % (len(r["history"]), hs))
    pb = r.get("perf_bin")
    if pb:
        lines.append("历史：%d~%d 分这档验证集命中 %.1f%%（随便买 %.2f%%，%.1f 倍，n=%d）"
                     % (pb["lo"], pb["hi"] - 1, 100 * pb["hit"], r["base"], pb["lift"], pb["n"]))
    if r["qualified"]:
        ps = r["perf_streak"]
        lines.append("      连续够格 %d 天这档：%.1f%%（%.1f 倍）" % (ps["k"], ps["hit"], ps["lift"]))
    lines.append("注：" + r["note"])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--no-name", action="store_true", help="不联网查名称")
    a = ap.parse_args()
    import importlib
    importlib.import_module("localenv")   # 本机 env；和其它入口一致（pyflakes 不认 noqa）
    r = predict(a.code, a.days, with_name=not a.no_name)
    if a.json:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print(fmt(r))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
