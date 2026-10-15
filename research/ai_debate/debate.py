"""第 1 件：辩论模式原型 vs 单分析师（研究，不进生产）。

    python research/ai_debate/debate.py                三只全跑（两种模式）
    python research/ai_debate/debate.py --only 002696   只跑一只
    python research/ai_debate/debate.py --mode debate   只跑辩论
    python research/ai_debate/debate.py --dry           只打印提示词长度，不调 AI

股票：out_breakout/list_a.json 前 2 只 + out_pullback/list_b.json 第 1 只（跑之前读当天文件）。
材料：materials/<代码>.json（tools/ai_analyze.py <代码> --materials --no-corp 的输出，跑之前备好）。

单分析师 = ai_chat 第一轮原样：ai_chat.SYSTEM + ai_chat.SCHEMA + first_prompt(FIRST_ASK["stock"])，
          模型 / 推理等级用生产默认（AI_MODEL = claude-opus-5-5，不设 effort），工具同 ai_chat。
辩论   = 正方、反方各一次（并行，claude-sonnet-5-5 low，WebSearch/WebFetch，提示词写死 8 次工具上限），
          裁判一次（claude-opus-5-5 high，只给 WebFetch 抽查，上限 3 次），只此一轮。
原始输出 raw/debate.jsonl；汇总 debate_summary.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from common import auth_ok, call, spent  # noqa: E402
import ai_chat  # noqa: E402

SIDE_MODEL, SIDE_EFFORT = "claude-sonnet-5-5", "low"
JUDGE_MODEL, JUDGE_EFFORT = "claude-opus-5-5", "high"
SINGLE_MODEL, SINGLE_EFFORT = "claude-opus-5-5", ""
SIDE_TOOL_CAP, JUDGE_TOOL_CAP = 8, 3

# ---------------------------------------------------------------
#  schema
# ---------------------------------------------------------------
EVID = {"type": "object", "properties": {
    "claim": {"type": "string"},
    "basis": {"type": "string", "enum": ["数据", "新闻", "公告", "推断"]},
    "source": {"type": "string", "description": "链接，或材料里的字段名（如 kline.ret_20d_pct）"},
    "published": {"type": "string", "description": "发布日期 YYYY-MM-DD；材料数据写材料日；查不到写 未知"},
    "hardness": {"type": "string", "enum": ["硬", "中", "软"],
                 "description": "硬 = 公告 / 交易所数据 / 材料里的数字；中 = 主流媒体报道；软 = 自媒体、传闻、推断"}},
    "required": ["claim", "basis", "source", "published", "hardness"]}
SIDE_SCHEMA = {"type": "object", "properties": {
    "side": {"type": "string", "enum": ["正方", "反方"]},
    "thesis": {"type": "string"},
    "evidence": {"type": "array", "minItems": 1, "maxItems": 8, "items": EVID},
    "weakest_point": {"type": "string", "description": "自己这一方最站不住的地方"},
    "tool_calls_used": {"type": "integer"}},
    "required": ["side", "thesis", "evidence", "weakest_point", "tool_calls_used"]}
JUDGE_SCHEMA = {"type": "object", "properties": {
    "stance": {"type": "string", "enum": ["看多", "看空", "观望"]},
    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    "one_line": {"type": "string"},
    "bull_hard": {"type": "integer", "description": "正方站得住的硬证据条数"},
    "bear_hard": {"type": "integer", "description": "反方站得住的硬证据条数"},
    "decisive": {"type": "array", "maxItems": 4, "items": EVID},
    "rejected": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
        "side": {"type": "string"}, "claim": {"type": "string"},
        "why": {"type": "string", "description": "没来源 / 链接打不开 / 内容对不上 / 晚于材料日 / 推断冒充事实"}},
        "required": ["side", "claim", "why"]}},
    "checked_links": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
        "url": {"type": "string"}, "ok": {"type": "boolean"}, "note": {"type": "string"}},
        "required": ["url", "ok"]}},
    "thin_evidence": {"type": "boolean", "description": "两边硬证据都不够，所以观望"},
    "change_mind": {"type": "string"},
    "model_vs_ai": {"type": "string"}},
    "required": ["stance", "confidence", "one_line", "bull_hard", "bear_hard", "decisive", "rejected",
                 "thin_evidence", "change_mind"]}

# ---------------------------------------------------------------
#  提示词
# ---------------------------------------------------------------
BASE_SYS = ("你在一个 A 股选股流水线里做研究分析。中文，简洁，不用破折号，要具体事实和数字，"
            "不说「值得关注」「有望受益」这类套话。清单排序由模型和规则决定，你只分析，不参与排序。"
            "只按证据说话，查不到的不要编，链接必须是你这次真的打开或搜到的。")


def side_prompt(side: str, code: str, mat: dict) -> str:
    want = ("找看多证据：未来 20 个交易日它为什么可能大涨（20 根内最高价涨超 50% 是我们关心的事件）"
            if side == "正方" else
            "找看空证据：未来 20 个交易日它为什么涨不起来或会跌（减持、解禁、业绩、监管、题材退潮、技术面破位）")
    return (
        f"你是辩论的{side}。对象：A 股 {code} {mat.get('name', '')}，材料日 {mat.get('asof')}（收盘后）。\n"
        f"任务：{want}。只替{side}找证据，另一方由别人负责，不要两边都说。\n"
        f"工具预算：WebSearch + WebFetch 合计最多 {SIDE_TOOL_CAP} 次；用到第 {SIDE_TOOL_CAP - 2} 次就停止查找、"
        "开始按 schema 写结论。用不完更好。\n"
        "搜索建议：「代码 名称 公告」「名称 最新」，优先交易所 / 巨潮 / 公司公告、证券时报 / 上证报 / 财联社等主流媒体。\n"
        "每条证据写清：说了什么、来源（链接或材料字段名）、发布日期（查不到写 未知）、硬度（硬 / 中 / 软）。"
        f"只用发布日期不晚于 {mat.get('asof')} 的信息。宁可少写，不许编链接。\n\n"
        "材料（本机算好的，model 是我们的起涨预测模型：分数 0~100 是全市场排名，≥97 且当天前 10 上清单；"
        "perf_bin 是这一档历史命中率，base 是随便买一只的比例；kline 是近 60 根 K 线事实，前复权）：\n"
        + json.dumps(mat, ensure_ascii=False, default=str))


def judge_prompt(code: str, mat: dict, bull: dict | None, bear: dict | None) -> str:
    return (
        f"你是裁判。对象：A 股 {code} {mat.get('name', '')}，材料日 {mat.get('asof')}。问题：未来 20 个交易日，"
        "次日开盘买入后最高价涨超 50% 的可能大不大，给立场（看多 / 看空 / 观望）和置信度。\n"
        "裁决规则：\n"
        "1. 按证据硬度裁决，不按条数、不按谁说得多。硬 = 公告、交易所数据、材料里的数字；中 = 主流媒体；"
        "软 = 自媒体、传闻、推断。没有来源的事实性说法当推断。\n"
        "2. 有分歧本身不是观望的理由。只有两边站得住的硬证据都太薄（各自不超过 1 条），才观望，并把 thin_evidence 设为 true。\n"
        f"3. 发布日期晚于 {mat.get('asof')} 或查不到日期的新闻，不能当硬证据。\n"
        f"4. 你可以用 WebFetch 抽查最关键的链接，最多 {JUDGE_TOOL_CAP} 次，打不开或内容对不上就驳回那条（写进 rejected）。"
        "抽查完立即按 schema 输出，不许再查。\n"
        "5. 把你和模型分数的关系写进 model_vs_ai：一致还是分歧、为什么。\n\n"
        "材料：\n" + json.dumps(mat, ensure_ascii=False, default=str)
        + "\n\n正方：\n" + json.dumps(bull or {"error": "正方没有给出结果"}, ensure_ascii=False)
        + "\n\n反方：\n" + json.dumps(bear or {"error": "反方没有给出结果"}, ensure_ascii=False))


def to_panel(j: dict | None, bull: dict | None, bear: dict | None) -> dict:
    """裁判结果 -> ai_chat 的 panel 形状。解析不出来记「待复核」，不默认观望（TradingAgents rating.py 的 REVIEW）。"""
    if not isinstance(j, dict) or j.get("stance") not in ("看多", "看空", "观望"):
        return {"stance": "待复核", "confidence": None, "conclusion": "裁判没有给出可解析的结论，需要人看正反方原文",
                "key_points": [], "bullets": [], "reasons": []}
    reasons = [{"point": e["claim"], "basis": {"公告": "新闻"}.get(e.get("basis"), e.get("basis", "推断")),
                "source": e.get("source", "")} for e in j.get("decisive", [])]
    return {"title": "正方 / 反方 / 裁判", "conclusion": j.get("one_line", ""), "stance": j["stance"],
            "confidence": j.get("confidence"),
            "key_points": [f"硬证据 正方 {j.get('bull_hard')} 条 / 反方 {j.get('bear_hard')} 条"]
                          + ([f"驳回 {len(j.get('rejected', []))} 条"] if j.get("rejected") else []),
            "bullets": [f"正方：{(bull or {}).get('thesis', '无')}", f"反方：{(bear or {}).get('thesis', '无')}"],
            "reasons": reasons, "unverified": [f"{r['side']}：{r['claim']}（{r['why']}）" for r in j.get("rejected", [])]}


# ---------------------------------------------------------------
def run_single(code: str, mat: dict) -> dict:
    prompt = ai_chat.first_prompt("stock", code, mat, ai_chat.FIRST_ASK["stock"])
    return call(prompt, ai_chat.SCHEMA, model=SINGLE_MODEL, effort=SINGLE_EFFORT, tools=ai_chat.TOOLS,
                system=ai_chat.SYSTEM, kind="research-ai_debate-single", subject=code, timeout=1500,
                tag="debate", meta={"mode": "single", "role": "analyst", "code": code},
                add_dirs=(str(ROOT),))


def run_debate(code: str, mat: dict) -> dict:
    def side(s: str) -> dict:
        return call(side_prompt(s, code, mat), SIDE_SCHEMA, model=SIDE_MODEL, effort=SIDE_EFFORT,
                    tools="WebSearch,WebFetch", system=BASE_SYS, kind="research-ai_debate-side",
                    subject=code, timeout=900, tag="debate",
                    meta={"mode": "debate", "role": s, "code": code})
    with ThreadPoolExecutor(2) as ex:
        fb, fr = ex.submit(side, "正方"), ex.submit(side, "反方")
        bull, bear = fb.result(), fr.result()
    j = call(judge_prompt(code, mat, bull["result"], bear["result"]), JUDGE_SCHEMA, model=JUDGE_MODEL,
             effort=JUDGE_EFFORT, tools="WebFetch", system=BASE_SYS, kind="research-ai_debate-judge",
             subject=code, timeout=900, tag="debate", meta={"mode": "debate", "role": "裁判", "code": code})
    return {"bull": bull, "bear": bear, "judge": j, "panel": to_panel(j["result"], bull["result"], bear["result"]),
            "sec_wall": round(max(bull["sec"], bear["sec"]) + j["sec"], 1),
            "usd": round(bull["usd"] + bear["usd"] + j["usd"], 4)}


def targets() -> list[str]:
    a = json.loads((ROOT / "out_breakout" / "list_a.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "out_pullback" / "list_b.json").read_text(encoding="utf-8"))
    return [str(x["code"]).zfill(6) for x in a[:2]] + [str(b[0]["code"]).zfill(6)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--mode", choices=["both", "single", "debate"], default="both")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    codes = [a.only] if a.only else targets()
    out = {}
    if not a.dry:
        ok, how = auth_ok()
        if not ok:
            print(f"claude 命令行没登录（{how}），先在终端跑 claude 然后 /login")
            return 2
    for code in codes:
        mat = json.loads((HERE / "materials" / f"{code}.json").read_text(encoding="utf-8"))
        if a.dry:
            print(code, mat.get("name"), "单分析师提示", len(ai_chat.first_prompt("stock", code, mat, "")),
                  "正方", len(side_prompt("正方", code, mat)), "裁判（不含正反方）", len(judge_prompt(code, mat, {}, {})))
            continue
        print(f"{code} 已花 {spent()} 美元", flush=True)
        rec = {}
        if a.mode in ("both", "single"):
            rec["single"] = run_single(code, mat)
            print(f"  单分析师 {rec['single']['usd']} 美元 {rec['single']['sec']} 秒 {rec['single']['error'][:80]}", flush=True)
        if a.mode in ("both", "debate"):
            rec["debate"] = run_debate(code, mat)
            print(f"  辩论 {rec['debate']['usd']} 美元 {rec['debate']['sec_wall']} 秒 立场 {rec['debate']['panel']['stance']}", flush=True)
        out[code] = rec
    if not a.dry:
        p = HERE / "debate_summary.json"
        old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        old.update(out)
        p.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
