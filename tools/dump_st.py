"""
把当前全市场的 ST / 退市整理名单落到 cache/st_codes.json。

为什么要有这份文件：生产（breakout/daily.risk_filter）按当天腾讯快照的**名称**
判 ST 并剔除，回测拿不到历史名称，validate.st_codes() 读不到缓存就干脆不剔，
于是「邮件里印的成绩」和「实际会发出去的清单」不是同一批票 —— 2026-09-16 实测
验证集 783 个名额里 98 个（12.5%、29 只代码）名称含 ST，剔掉之后各档命中率
从 12.64/15.74/14.29/12.96/6.67% 掉到 11.76/13.41/11.90/10.64/3.70%。

这份名单的局限，用之前必须知道：它是**今天**的名称，不是当时的。上面那 98 行里
有一半是现在戴帽、当时没戴的票，也有当时戴帽现在摘了的票它抓不到。所以它只能
把「数量级」做对，成员做不对。真正的解法是训练表带一列 is_st（要历史名称，
现在拿不到）。在那之前，宁可用这份保守一点的名单：生产确实不会买今天的 ST。

    python tools/dump_st.py            拉一次，写 cache/st_codes.json
    python tools/dump_st.py --dry      只看数量，不写

累积写：老名单里的代码不会被删掉（摘帽的票当时也是 ST，回测该剔的还是该剔），
每个代码记第一次和最后一次见到的日期，将来有了历史名称可以按日期回放。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "cache" / "st_codes.json"


def fetch() -> dict[str, str]:
    """{代码: 名称}，全市场。"""
    import pandas as pd
    import datasource as ds
    codes = [str(c).zfill(6) for c in
             pd.read_csv(ROOT / "cache" / "codes.csv", dtype={"code": str})["code"]]
    q = ds.fetch_quotes([ds.to_symbol(c) for c in codes])
    out = {}
    for c in codes:
        v = q.get(ds.to_symbol(c))
        if v is not None and getattr(v, "name", ""):
            out[c] = v.name
    return out


def is_st(name: str) -> bool:
    """和 daily.risk_filter 同一判据：名称里有 ST 或「退」。

    腾讯的名称里 ST 前面可能带 * 或空格（`*ST 华仪`），统一大写后按子串判。
    """
    n = str(name).upper().replace(" ", "")
    return "ST" in n or "退" in n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    today = dt.date.today().isoformat()
    names = fetch()
    if len(names) < 3000:
        print(f"只拿到 {len(names)} 只的名称，太少，不写（怕把名单写残）")
        return 1
    hit = {c: n for c, n in names.items() if is_st(n)}
    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8")).get("seen", {})
        except Exception:  # noqa: BLE001
            old = {}
    seen = dict(old)
    for c, n in hit.items():
        e = seen.get(c) or {"first": today}
        e.update({"last": today, "name": n})
        seen[c] = e
    print(f"全市场 {len(names)} 只，今天 ST/退市整理 {len(hit)} 只，"
          f"累积名单 {len(seen)} 只")
    if a.dry:
        print("  样本:", list(hit.items())[:8])
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"date": today, "n_market": len(names),
                               "codes": sorted(seen), "seen": seen},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(OUT)
    print("->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
