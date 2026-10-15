"""第 2 件：盲测 / 记忆泄漏检验（研究）。

    python research/ai_debate/blind_run.py                       sonnet 跑 20 只 x 两种条件
    python research/ai_debate/blind_run.py --model claude-opus-5-5 --n 20
    python research/ai_debate/blind_run.py --eval                只算结果，不调 AI
    python research/ai_debate/blind_run.py --dry                 打印两种条件的提示词各一份

条件 named：代码、名称、真实日期、真实价格；blind：无代码名称、T-19..T0、价格归一（blind_prep.py）。
两种条件都 --tools ""（读不到仓库、上不了网），系统提示整段替换成同一句，cwd 是空目录。
所以 named 能比 blind 多知道的，只能来自模型自己的记忆。
原始输出 raw/blind_<模型>.jsonl；结果 blind_result.json。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RAW, auth_ok, call, spent  # noqa: E402

SYSTEM = ("你是 A 股短线分析员。只根据用户给的材料判断，中文，简洁。"
          "给出概率时要校准：随便一只票 20 个交易日内最高价涨超 50% 的比例大约 3%，"
          "被模型选进前 10 的票历史上大约 13%。")
SCHEMA = {"type": "object", "properties": {
    "prob": {"type": "integer", "minimum": 0, "maximum": 100,
             "description": "次日开盘买入后 20 个交易日内最高价涨超 50%（且次日不是一字板）的概率，百分数"},
    "reasons": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
    "events_cited": {"type": "array", "maxItems": 5, "items": {"type": "string"},
                     "description": "理由里提到的具体事件（公告、新闻、题材），带日期；材料以外没有就留空"},
    "recognized": {"type": "boolean", "description": "你是否认出了这只股票或这段行情"}},
    "required": ["prob", "reasons", "events_cited", "recognized"]}

ASK = ("问题：次日开盘买入后，未来 20 个交易日内最高价相对买入价涨超 50% 的概率是多少（0~100）？"
       "次日开盘一字涨停买不进的算没涨。给概率和最多 5 条理由。")


def prompt(s: dict, cond: str) -> str:
    if cond == "named":
        m = s["named"]
        head = (f"今天是 {m['date']}，收盘后。股票：{m['code']} {m['name']}（{m['model']['board']}）。\n"
                "下面是截至今天收盘的材料。")
    else:
        m = s["blind"]
        head = (f"今天是 T0，收盘后。一只 A 股（板块类型 {m['model']['board']}，代码、名称、日期都隐去，"
                "价格已按 T0 收盘 = 100 归一）。\n下面是截至 T0 收盘的材料。")
    body = {"model": m["model"], "kline": m["kline"], "market": m["market"]}
    return (head + "model 是我们的起涨预测模型：score 是当天全市场排名（0~100），rank 是当天名次，p 是预测值。"
            "kline 是 K 线事实（recent_20_bars 的 v_rel 是当日量 / 近 60 日均量），market 是近 10 天全市场环境。\n"
            + json.dumps(body, ensure_ascii=False) + "\n\n" + ASK)


def run(model: str, n: int, effort: str) -> None:
    samples = json.loads((HERE / "blind_samples.json").read_text(encoding="utf-8"))[:n]
    tag = f"blind_{model.replace('claude-', '')}"
    done = set()
    p = RAW / f"{tag}.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("result"):
                done.add((r["meta"]["sid"], r["meta"]["cond"]))
    jobs = [(s, c) for s in samples for c in ("named", "blind") if (s["sid"], c) not in done]
    print(f"{model}：要跑 {len(jobs)} 次，已花 {spent()} 美元", flush=True)

    def one(sc):
        s, c = sc
        return call(prompt(s, c), SCHEMA, model=model, effort=effort, tools="", system=SYSTEM,
                    system_replace=True, kind="research-ai_debate-blind", subject=f"{s['sid']}-{c}",
                    timeout=300, tag=tag, meta={"sid": s["sid"], "cond": c, "y_open": s["y_open"]})
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, jobs):
            print(f"  {r['meta']['sid']} {r['meta']['cond']:5s} prob={(r['result'] or {}).get('prob')} "
                  f"{r['usd']} 美元 {r['error'][:60]}", flush=True)


# ---------------------------------------------------------------
#  评估
# ---------------------------------------------------------------
def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))


def boot(pos, neg, f, B=4000, seed=7):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    v = [f(list(rng.choice(pos, len(pos))), list(rng.choice(neg, len(neg)))) for _ in range(B)]
    return [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]


DATE_RE = re.compile(r"(20\d{2})[-年/.](\d{1,2})(?:[-月/.](\d{1,2}))?")


def later_dates(texts: list[str], d0: str) -> list[str]:
    """理由里出现的、晚于样本日的日期（只到月份的按月末算，宁可漏报不误报：同月不算）。"""
    hits = []
    for t in texts:
        for m in DATE_RE.finditer(t):
            y, mo, dd = int(m.group(1)), int(m.group(2)), m.group(3)
            key = f"{y:04d}-{mo:02d}" + (f"-{int(dd):02d}" if dd else "")
            if (dd and key > d0) or (not dd and key > d0[:7]):
                hits.append(m.group(0))
    return hits


def evaluate() -> dict:
    samples = {s["sid"]: s for s in json.loads((HERE / "blind_samples.json").read_text(encoding="utf-8"))}
    out = {}
    for p in sorted(RAW.glob("blind_*.jsonl")):
        rows = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("result") and isinstance(r["result"].get("prob"), (int, float)):
                rows[(r["meta"]["sid"], r["meta"]["cond"])] = r
        res = {"n_calls": len(rows), "usd": round(sum(r["usd"] for r in rows.values()), 3),
               "sec_mean": round(float(np.mean([r["sec"] for r in rows.values()])), 1) if rows else None}
        for c in ("named", "blind"):
            pos = [rows[(s, c)]["result"]["prob"] for s in samples if (s, c) in rows and samples[s]["y_open"] == 1]
            neg = [rows[(s, c)]["result"]["prob"] for s in samples if (s, c) in rows and samples[s]["y_open"] == 0]
            res[c] = {"n_pos": len(pos), "n_neg": len(neg),
                      "mean_pos": round(float(np.mean(pos)), 1) if pos else None,
                      "mean_neg": round(float(np.mean(neg)), 1) if neg else None,
                      "diff": round(float(np.mean(pos) - np.mean(neg)), 1) if pos and neg else None,
                      "diff_ci95": boot(pos, neg, lambda a, b: float(np.mean(a) - np.mean(b))) if pos and neg else None,
                      "auc": round(auc(pos, neg), 3) if pos and neg else None,
                      "auc_ci95": boot(pos, neg, auc) if pos and neg else None,
                      "recognized": sum(bool(rows[(s, c)]["result"].get("recognized")) for s in samples if (s, c) in rows)}
        # 配对：同一只票 named - blind
        pair = [(rows[(s, "named")]["result"]["prob"] - rows[(s, "blind")]["result"]["prob"], samples[s]["y_open"])
                for s in samples if (s, "named") in rows and (s, "blind") in rows]
        if pair:
            d1 = [d for d, y in pair if y == 1]
            d0 = [d for d, y in pair if y == 0]
            res["named_minus_blind"] = {"hit_mean": round(float(np.mean(d1)), 1) if d1 else None,
                                        "miss_mean": round(float(np.mean(d0)), 1) if d0 else None,
                                        "diff_in_diff": round(float(np.mean(d1) - np.mean(d0)), 1) if d1 and d0 else None,
                                        "did_ci95": boot(d1, d0, lambda a, b: float(np.mean(a) - np.mean(b))) if d1 and d0 else None}
        leaks = []
        for s, smp in samples.items():
            r = rows.get((s, "named"))
            if not r:
                continue
            txt = list(r["result"].get("reasons", [])) + list(r["result"].get("events_cited", []))
            ld = later_dates(txt, smp["date"])
            if ld or r["result"].get("events_cited") or r["result"].get("recognized"):
                leaks.append({"sid": s, "code": smp["code"], "date": smp["date"], "y_open": smp["y_open"],
                              "later_dates": ld, "events_cited": r["result"].get("events_cited"),
                              "recognized": r["result"].get("recognized"),
                              "reasons": r["result"].get("reasons")})
        res["named_mentions"] = leaks
        out[p.stem] = res
    (HERE / "blind_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.dry:
        s = json.loads((HERE / "blind_samples.json").read_text(encoding="utf-8"))[0]
        print(prompt(s, "named")[:1500], "\n----\n", prompt(s, "blind")[:1500])
        return 0
    if not a.eval:
        ok, how = auth_ok()
        if not ok:
            print(f"claude 命令行没登录（{how}），先在终端跑 claude 然后 /login")
            return 2
        run(a.model, a.n, a.effort)
    print(json.dumps(evaluate(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
