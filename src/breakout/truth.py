"""
起涨预测的真值：历史清单 A 之后到底涨没涨。

清单 A 每天落盘在 data/breakout/YYYY-MM/breakout_<date>.parquet，标签
y_up 当时是 NaN（未来还没发生）。这里用 daily.parquet 把每只票之后的
走势补上，和 label.label_up 同一口径：**之后 20 根自身 K 线的最高价相对
上榜日收盘涨幅是否超 50%**，不含上榜当天。

窗口没满 20 根的算「进行中」，只给「到目前为止」的最高涨幅，不算命中率
里的分母（final=False）。同期全市场基准也按同一起点、同一窗口算：
上榜那天全市场随便买一只，之后 20 根内涨超 50% 的比例。**只有和同期基准比
才能分清「模型没用」和「那段时间谁都涨不了」。**

产物 state/breakout/truth.json，学习会诊的证据包和面板都读它。
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label import UP_THRESHOLD, UP_WINDOW   # noqa: E402

log = logging.getLogger("breakout.truth")
DATA = ROOT / "data" / "breakout"
STATE = ROOT / "state" / "breakout"
OUT = STATE / "truth.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 区间。n=0 返回 (0, 1)。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def list_files() -> list[Path]:
    return sorted(DATA.glob("*/breakout_*.parquet"))


def _future_max(px: pd.DataFrame, window: int) -> pd.DataFrame:
    """每行加两列：fut_max（之后 min(window, 可得) 根的最高价）、avail（可得根数）。

    和 label_up 一样从 t+1 起算，不含当天。反向 rolling 一次算完全表，
    4.3M 行几秒钟。
    """
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    g = px.groupby("code", sort=False)
    nxt = g["high"].shift(-1)
    # 反向 rolling：对每个 code，fut_max[i] = max(high[i+1 .. i+window])
    rev = nxt[::-1].groupby(px["code"][::-1], sort=False)
    px["fut_max"] = rev.rolling(window, min_periods=1).max().reset_index(
        level=0, drop=True)[::-1].to_numpy()
    n = g["date"].transform("size").to_numpy()
    idx = g.cumcount().to_numpy()
    px["avail"] = np.minimum(window, n - idx - 1)
    return px


def compute(window: int = UP_WINDOW, threshold: float = UP_THRESHOLD,
            asof: str = "") -> dict:
    """全部历史清单的真值 + 同期基准。asof 给了就只用那天及之前的行情（回放）。"""
    files = list_files()
    dp = DATA / "daily.parquet"
    if not files or not dp.exists():
        return {"lists": [], "picks": [], "by": {}, "window": window,
                "threshold": threshold}
    px = pd.read_parquet(dp, columns=["code", "date", "high", "close"])
    px["date"] = px["date"].astype(str)
    if asof:
        px = px[px["date"] <= asof]
    px = _future_max(px, window)
    px["rise"] = px["fut_max"] / px["close"] - 1.0
    px["rise"] = px["rise"].where(px["avail"] > 0)
    key = px.set_index(["code", "date"])

    lists, picks = [], []
    for f in files:
        d = pd.read_parquet(f)
        if not len(d):
            continue
        date = str(d["date"].iloc[0])
        if asof and date > asof:
            continue
        # 同期基准：上榜那天全市场、同一窗口
        day = px[px["date"] == date]
        fin = day[day["avail"] >= window]
        base_final = float((fin["rise"] > threshold).mean()) if len(fin) else None
        part = day[day["avail"] > 0]
        base_sofar = float((part["rise"] > threshold).mean()) if len(part) else None
        n_bars = int(day["avail"].max()) if len(day) else 0
        rows = []
        for r in d.itertuples():
            code = str(r.code).zfill(6)
            try:
                k = key.loc[(code, date)]
            except KeyError:
                rows.append({"code": code, "avail": 0, "rise": None, "hit": None})
                continue
            avail = int(k["avail"])
            rise = float(k["rise"]) if avail > 0 and np.isfinite(k["rise"]) else None
            rows.append({
                "date": date, "code": code, "name": getattr(r, "name", "") or "",
                "score": float(getattr(r, "score", 0) or 0),
                "streak": int(getattr(r, "streak", 1) or 1),
                "board": getattr(r, "board", "") or "",
                "close": float(getattr(r, "close", 0) or 0),
                "avail": avail, "final": avail >= window,
                "rise": rise,
                "hit": (rise > threshold) if rise is not None else None,
            })
        n_fin = sum(1 for x in rows if x.get("final"))
        k_fin = sum(1 for x in rows if x.get("final") and x.get("hit"))
        k_sofar = sum(1 for x in rows if x.get("hit"))
        lo, hi = wilson(k_fin, n_fin)
        lists.append({
            "date": date, "n": len(rows), "bars": n_bars,
            "final": n_fin >= len(rows) and len(rows) > 0,
            "n_final": n_fin, "hits_final": k_fin,
            "hit_rate": (k_fin / n_fin) if n_fin else None,
            "ci_lo": lo if n_fin else None, "ci_hi": hi if n_fin else None,
            "hits_sofar": k_sofar,
            "base_final": base_final, "base_sofar": base_sofar,
            "max_rise": max((x["rise"] for x in rows if x["rise"] is not None),
                            default=None),
        })
        picks.extend(rows)

    out = {"window": window, "threshold": threshold, "asof": asof,
           "lists": lists, "picks": picks, "by": _breakdowns(picks)}
    return out


def _breakdowns(picks: list[dict]) -> dict:
    """按连续档 / 分数段 / 板块汇总（只算窗口已满的）。"""
    fin = [p for p in picks if p.get("final")]
    out = {}

    def agg(keyf, name):
        buckets: dict[str, list] = {}
        for p in fin:
            buckets.setdefault(keyf(p), []).append(p)
        rows = []
        for k, v in sorted(buckets.items()):
            h = sum(1 for x in v if x["hit"])
            lo, hi = wilson(h, len(v))
            rows.append({"bucket": k, "n": len(v), "hits": h,
                         "hit_rate": h / len(v), "ci_lo": lo, "ci_hi": hi})
        out[name] = rows

    agg(lambda p: f"连续≥{min(p['streak'], 5)}天" if p["streak"] >= 2 else "首日",
        "streak")
    agg(lambda p: "99分" if p["score"] >= 99 else ("98分" if p["score"] >= 98
                                                   else "97分"), "score")
    agg(lambda p: p["board"] or "?", "board")
    if fin:
        h = sum(1 for x in fin if x["hit"])
        lo, hi = wilson(h, len(fin))
        out["all"] = {"n": len(fin), "hits": h, "hit_rate": h / len(fin),
                      "ci_lo": lo, "ci_hi": hi}
    return out


BOARD_FILE = ROOT / "out_breakout" / "board_hit.json"


def _rule_stamp() -> dict:
    """当前生产规则的指纹：分数线 / 上限 / 板块系数（含会诊批准的覆盖）。"""
    try:
        import daily as D
        return {"score_min": int(D.SCORE_MIN), "cap_a": int(D.CAP_A),
                "board_adj": {str(k): round(float(v), 4)
                              for k, v in D.BOARD_ADJ.items()}}
    except Exception:  # noqa: BLE001
        return {}


def _grid_arm() -> dict:
    """建缓存那一跑用的板块系数（out_breakout/window_grid.json 落的）。"""
    try:
        import json as _j
        g = _j.loads((ROOT / "out_breakout" / "window_grid.json")
                     .read_text(encoding="utf-8"))
        ba = g.get("board_adj")
        return {str(k): round(float(v), 4) for k, v in ba.items()} if ba else {}
    except Exception:  # noqa: BLE001
        return {}


def validation_by_board(refresh: bool = False) -> dict:
    """验证集里生产口径名额按板块的命中率。邮件的期望要按本份清单的板块构成加权。

    2026-09-16 会诊查出来的：验证集（逐月滚动 207 个交易日）里科创板名额命中
    只有 2.7%（n=150，和随便买的 2.93% 一个水平），主板 16.5%（n=520），
    北交 8.0%（n=113），创业板一个名额都没有。而生产清单里科创占六成，
    邮件印的 12.6% 是整体平均，对当前这种构成高估了近一倍。

    源数据是逐月滚动测试缓存 data/breakout/raw/wf_scores.parquet，和
    window_grid.json / export.STREAK_PERF 同一份，所以两边永远对得上。
    结果缓存到 out_breakout/board_hit.json，没有缓存也没有源数据就返回空表。
    """
    import json as _j
    cache = BOARD_FILE
    src = ROOT / "data" / "breakout" / "raw" / "wf_scores.parquet"
    rule = _rule_stamp()
    if not refresh and cache.exists():
        fresh = not src.exists() or cache.stat().st_mtime >= src.stat().st_mtime
        try:
            got = _j.loads(cache.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            got = None
        # 规则也要比。只看 mtime 的话，改 BOARD_ADJ（会诊批准写 overrides.json）
        # 既不动 parquet 也不动缓存，邮件里按板块加权的期望会一直停在旧系数上，
        # 而那正是「谁上榜」变了的那一维（2026-09-16 star 1.27->1.0 换掉了
        # 162 个出榜日里的 77 天）。
        if got is not None and fresh and got.get("rule") == rule:
            return got
    if not src.exists():
        return {}
    try:
        import daily as D
        import features as F
        d = pd.read_parquet(src)
        ok = d[(d["score"] >= D.SCORE_MIN) & (d["rank"] <= D.CAP_A)].copy()
        ok = ok[np.isfinite(ok["y_up"])]
        if not len(ok):
            return {}
        ok["board"] = ok["code"].map(F.board_of)
        out = {"overall": {"n": int(len(ok)), "hit": float(ok["y_up"].mean())},
               "boards": {}}
        for b, g in ok.groupby("board"):
            k, n = int(g["y_up"].sum()), int(len(g))
            lo, hi = wilson(k, n)
            out["boards"][str(b)] = {"n": n, "hits": k, "hit": k / n,
                                     "ci_lo": lo, "ci_hi": hi}
        out["rule"] = rule
        # 缓存里的 score / rank 两列是**建缓存那一跑**的板块系数烘进去的，
        # 只有 SCORE_MIN / CAP_A 是这里活取的。两者对不上时算出来的数不是
        # 当前规则的成绩，标出来让调用方（export）能说清楚。
        out["source_arm"] = _grid_arm()
        out["arm_matches"] = (not out["source_arm"]
                              or out["source_arm"] == rule.get("board_adj"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(_j.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("按板块命中率算不出来: %s", e)
        return {}


def save(res: dict) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return OUT


def load() -> dict:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = compute()
    p = save(res)
    for L in res["lists"]:
        hr = f"{100 * L['hit_rate']:.0f}%" if L["hit_rate"] is not None else "-"
        bf = f"{100 * L['base_final']:.1f}%" if L["base_final"] is not None else "-"
        print(f"{L['date']}  {L['n']} 只  已走 {L['bars']} 根  "
              f"{'已定' if L['final'] else '进行中'}  命中 {L['hits_sofar']}"
              f"  最终命中率 {hr}  同期基准 {bf}")
    print("->", p)
