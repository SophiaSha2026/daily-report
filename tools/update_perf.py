"""
把 export.py 里印给用户的成绩常量，按实验产物重写一遍。

改了口径就必须重跑实验、重写常量，否则邮件里是旧口径的成绩给新模型背书
（历史教训 30）。selftest_breakout 会逐位对账两边，所以这两件事只能一起做。

    python tools/update_perf.py --show          只打印产物里的数，不改文件
    python tools/update_perf.py --from fixed    按生产在用的板块系数那一臂写（默认）
    python tools/update_perf.py --from rolling  按滚动校正臂写（要显式指定）

**默认必须是 fixed，因为生产用的就是它。** 2026-09-16 之前默认是 rolling，
理由是「滚动臂每月只用过去的月份估系数，是最诚实的数」—— 那个理由对，
但结论错：滚动臂和固定臂选出来的不是同一批票（整体命中一样，构成从
主板 67%/科创 30% 变成主板 54%/北交 31%），它是**另一套策略**，不是同一套
策略的更诚实估法。把它的数字写进邮件，就是拿 A 策略的成绩给 B 策略背书。
当天例行跑一次 update_perf 就会静默发生这件事（2026-09-16 实测发生了）。

写之前还要核对产物的板块系数和生产此刻生效的是不是同一套（含 overrides.json
里会诊批准的覆盖），对不上直接拒绝写，让人先去重跑 exp_window。
两臂的数都写进 docs/breakout_log.md，邮件里印哪一个在这里选。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_breakout"
EXPORT = ROOT / "src" / "breakout" / "export.py"


def w5(grid: dict) -> dict[int, dict]:
    out = {}
    for r in grid.get("grid", []):
        if r.get("kind") != "W5":
            continue
        m = re.search(r"连续≥(\d)天", r.get("label", ""))
        if m:
            out[int(m.group(1))] = r
    return out


def w5c(grid: dict) -> dict | None:
    """满员日那一行（kind=W5c）。exp_window 还没写就返回 None。"""
    for r in grid.get("grid", []):
        if r.get("kind") == "W5c" and "满" in r.get("label", ""):
            return r
    return None


def cap_split(arm: str) -> tuple[tuple, tuple] | None:
    """满员日 vs 非满员日的命中率，从建 grid 的**同一份**缓存算。

    「满员」= 当天够格（score >= SCORE_MIN）的票超过 CAP_A 只，清单是被上限
    截断的；「非满员」= 门槛在决定清单。两者差一倍多（2026-09-16 实测
    7.7% vs 17.0%），而生产 2026-09 那 8 天全部满员，所以这两个数必须
    印在邮件里，且必须和 STREAK_PERF 同一次实验、同一套规则。

    返回 ((满员 hit%, 名额, 天数), (非满员 hit%, 名额, 天数))。
    """
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    _s.path.insert(0, str(ROOT / "src" / "breakout"))
    try:
        import pandas as pd
        import daily as D
    except Exception as e:  # noqa: BLE001
        print(f"[!] 满员日那组算不了（{e}）", flush=True)
        return None
    f = (ROOT / "data" / "breakout" / "raw"
         / ("wf_scores.parquet" if arm == "fixed" else "wf_rolling.parquet"))
    if not f.exists():
        print(f"[!] 没有 {f.name}，满员日那组跳过", flush=True)
        return None
    d = pd.read_parquet(f)
    d = d[d["y_up"].notna()]
    ok = d[d["score"] >= D.SCORE_MIN]
    if ok.empty:
        return None
    nq = ok.groupby("date").size()
    picks = ok[ok["rank"] <= D.CAP_A].copy()
    picks["capped"] = picks["date"].map(nq > D.CAP_A)
    out = []
    for flag in (True, False):
        g = picks[picks["capped"] == flag]
        n = int(len(g))
        out.append((round(100 * float(g["y_up"].mean()), 1) if n else float("nan"),
                    n, int(g["date"].nunique())))
    return out[0], out[1]


def load(arm: str) -> tuple[dict, dict, dict]:
    f = OUT / ("window_grid.json" if arm == "fixed" else "window_grid_rolling.json")
    if not f.exists():
        raise SystemExit(f"没有 {f.name}，先跑 exp_window.py"
                         + ("" if arm == "fixed" else " --adj rolling"))
    grid = json.loads(f.read_text(encoding="utf-8"))
    cal = json.loads((OUT / "score_calibration.json").read_text(encoding="utf-8"))
    board = {}
    bf = OUT / "board_hit.json"
    if bf.exists():
        board = json.loads(bf.read_text(encoding="utf-8"))
    return grid, cal, board


def show(arm: str) -> None:
    grid, cal, board = load(arm)
    rows = w5(grid)
    base = 100 * float(cal["base"])
    print(f"[{arm}] {grid.get('days')} 个交易日，基准 {base:.2f}%，"
          f"板块系数 {grid.get('board_adj_mode', '?')}")
    for k in sorted(rows, reverse=True):
        r = rows[k]
        print(f"  连续≥{k}天  n={r['n']:<5} 命中 {100 * r['hit']:.2f}%  "
              f"±{100 * r['se']:.2f}  倍数 {100 * r['hit'] / base:.2f}")
    c = w5c(grid)
    if c:
        print(f"  满员日   n={c['n']:<5} 命中 {100 * c['hit']:.2f}%  "
              f"天数 {c.get('empty', '?')}")
    if board.get("boards"):
        print("  按板块：", "，".join(
            f"{k} {100 * v['hit']:.1f}%（{v['n']}）"
            for k, v in sorted(board["boards"].items(), key=lambda x: -x[1]["n"])))
    for b in cal.get("bins", [])[::-1][:5]:
        print(f"  分数 {b['lo']}~{b['hi']}  n={b['n']:<7} 命中 {100 * b['hit']:.2f}%")


def _check_arm(arm: str, grid: dict) -> None:
    """产物里的板块系数必须等于生产此刻生效的那套，否则拒绝写。

    只有 fixed 臂才谈得上「和生产一致」：rolling 是逐月重估的另一套策略，
    显式指定它就是明知故犯，只警告。
    """
    if arm != "fixed":
        print("[!] 按 rolling 臂写：那是另一套策略的成绩，"
              "邮件里的免责声明要自己说清楚", flush=True)
        return
    want = grid.get("board_adj") or {}
    try:
        import sys as _s
        _s.path.insert(0, str(ROOT / "src"))
        _s.path.insert(0, str(ROOT / "src" / "breakout"))
        import daily as D
        have = {str(k): round(float(v), 4) for k, v in D.BOARD_ADJ.items()}
    except Exception as e:  # noqa: BLE001
        print(f"[!] 读不到生产的板块系数（{e}），跳过核对", flush=True)
        return
    want = {str(k): round(float(v), 4) for k, v in want.items()}
    if want and want != have:
        raise SystemExit(
            "产物的板块系数和生产不一致，拒绝写：\n"
            f"  window_grid.json: {want}\n"
            f"  生产（含 overrides）: {have}\n"
            "先跑 python src/breakout/exp_window.py --refit 重出成绩表。")


def rewrite(arm: str) -> None:
    grid, cal, board = load(arm)
    _check_arm(arm, grid)
    rows = w5(grid)
    if len(rows) < 2:
        raise SystemExit(f"W5 行只有 {len(rows)} 条，连「全部上榜」和「连续 2 天」都凑不齐")
    base = round(100 * float(cal["base"]), 2)
    lines = []
    # 有几档写几档。板块系数一改，高档次可能一个样本都没有（star=1.0 之后
    # 连续 5 天就没了），硬要 5 条会把上一轮的旧数字留在表里。
    for k in sorted(rows, reverse=True):
        r = rows[k]
        hit = round(100 * r["hit"], 1)
        lines.append(f"    ({k}, {hit}, {round(hit / base, 1)}, {int(r['n'])}),")
    s = EXPORT.read_text(encoding="utf-8")
    s2 = re.sub(r"BASE = [\d.]+", f"BASE = {base}", s, count=1)
    s2 = re.sub(r"STREAK_PERF = \[\n(?:.*\n)*?\]",
                "STREAK_PERF = [\n    # 连续天数下限, 准确率%, 相对随便买的倍数, 样本数\n"
                + "\n".join(lines) + "\n]", s2, count=1)
    s2 = re.sub(r'PERF = \{"base": BASE, "window": "[^"]*"\}',
                f'PERF = {{"base": BASE, "window": "验证集 {grid.get("days")} 个交易日"}}', s2, count=1)
    c = w5c(grid)
    if c:
        s2 = re.sub(r"CAP_PERF = \([\d., ]+\)",
                    f"CAP_PERF = ({round(100 * c['hit'], 1)}, {int(c['n'])}, "
                    f"{int(c.get('days') or c.get('empty') or 0)})", s2, count=1)
    else:
        sp = cap_split(arm)
        if sp:
            (h1, n1, d1), (h2, n2, d2) = sp
            s2 = re.sub(r"CAP_PERF = \([\d., ]+\)",
                        f"CAP_PERF = ({h1}, {n1}, {d1})", s2, count=1)
            s2 = re.sub(r"NONCAP_PERF = \([\d., ]+\)",
                        f"NONCAP_PERF = ({h2}, {n2}, {d2})", s2, count=1)
            print(f"满员日 {h1}%（{n1} 席 / {d1} 天） vs "
                  f"非满员 {h2}%（{n2} 席 / {d2} 天）")
    if s2 == s:
        raise SystemExit("一个常量都没改到，正则和文件对不上了")
    EXPORT.write_text(s2, encoding="utf-8")
    print(f"已按 [{arm}] 更新 export.py：BASE={base}，STREAK_PERF={lines}")
    print("接着跑 python src/selftest_breakout.py，它会逐位对账两边")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="arm", choices=["fixed", "rolling"], default="fixed")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show:
        for arm in ("fixed", "rolling"):
            try:
                show(arm)
            except SystemExit as e:
                print(e)
            print()
        return 0
    rewrite(a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
