"""
会诊输出的枚举和 JSON schema。

CLI 用 `--json-schema` 强制结构，所以正常情况下拿到的就是合法对象；
`sanitize_*` 是第二道保险：枚举越界改成保守值、数值裁到范围、长度截断。
下游（面板、台账、实验）只吃 sanitize 过的对象。
"""
from __future__ import annotations

from typing import Any

LENSES = ["gap_where", "noise_or_real", "features", "regime",
          "data_quality", "improve"]
LENS_NAME = {
    "gap_where": "差距定位", "noise_or_real": "波动还是真差距",
    "features": "特征体检", "regime": "市场环境",
    "data_quality": "数据与流程", "improve": "改进提案",
    "chair": "主审",
}

LINES = ["morning", "breakout", "both"]
CAUSES = ["模型偏差", "特征失效", "市场环境", "数据质量", "样本不足",
          "口径不一致", "未知"]
KINDS = ["param", "threshold", "feature_drop", "feature_add",
         "feature_modify", "data_fix", "scope", "process"]
NOISE = ["噪声", "真差距", "混合", "样本不足无法判断"]
CONF = ["低", "中", "高"]
PRIORITY = ["P0", "P1", "P2"]

_FINDING = {
    "type": "object",
    "properties": {
        "line": {"type": "string", "enum": LINES},
        "claim": {"type": "string", "description": "一句话结论，≤60 字"},
        "evidence": {"type": "string",
                     "description": "证据包里的具体数字或你查到的事实，≤120 字"},
        "magnitude": {"type": "string",
                      "description": "差距多大，带单位，如「−0.6 个百分点/天，n=17」"},
        "confidence": {"type": "string", "enum": CONF},
    },
    "required": ["line", "claim", "evidence", "magnitude", "confidence"],
}

_PROPOSAL = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": KINDS},
        "line": {"type": "string", "enum": LINES},
        "target": {"type": "string",
                   "description": "改哪个对象：参数名/特征名/常量名/流程步骤"},
        "change": {"type": "string",
                   "description": "改成什么，可执行的描述，如 SCORE_MIN 97->96、"
                                  "去掉 vol_ratio5、scoring.weights.volume 0.20->0.15"},
        "rationale": {"type": "string", "description": "依据哪些证据，≤150 字"},
        "expected_effect": {"type": "string",
                            "description": "预期改善多少、看哪个指标"},
        "test_plan": {"type": "string",
                      "description": "怎么检验：走向前/消融/七道闸/人工实现"},
        "priority": {"type": "string", "enum": PRIORITY},
        "params": {"type": "object",
                   "description": "机器可读参数：param 类给 {参数名: 新值}；"
                                  "threshold 类给 {常量名: 新值}；feature_drop 给 "
                                  "{features: [名]}",
                   "additionalProperties": True},
    },
    "required": ["kind", "line", "target", "change", "rationale",
                 "expected_effect", "test_plan", "priority"],
}

LENS_SCHEMA = {
    "type": "object",
    "properties": {
        "lens": {"type": "string", "enum": LENSES},
        "summary": {"type": "string", "description": "≤200 字，给人读"},
        "findings": {"type": "array", "items": _FINDING, "maxItems": 12},
        "proposals": {"type": "array", "items": _PROPOSAL, "maxItems": 6},
        "questions": {"type": "array", "items": {"type": "string"},
                      "description": "没法从证据包回答、需要更多数据的问题",
                      "maxItems": 5},
    },
    "required": ["lens", "summary", "findings", "proposals"],
}

_GAP = {
    "type": "object",
    "properties": {
        "line": {"type": "string", "enum": LINES},
        "where": {"type": "string", "description": "差距在哪个切片"},
        "expected": {"type": "number"}, "actual": {"type": "number"},
        "unit": {"type": "string", "description": "% / 百分点 / 倍"},
        "ci_lo": {"type": "number"}, "ci_hi": {"type": "number"},
        "n": {"type": "integer"},
        "baseline": {"type": "number",
                     "description": "同期全市场基准（有就填）"},
    },
    "required": ["line", "where", "expected", "actual", "unit", "n"],
}

_WHY = {
    "type": "object",
    "properties": {
        "cause": {"type": "string", "enum": CAUSES},
        "weight": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": ["cause", "weight", "evidence"],
}

CHAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "gap": {"type": "array", "items": _GAP, "maxItems": 12},
        "why": {"type": "array", "items": _WHY, "maxItems": 7},
        "noise_or_real": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": NOISE},
                "p_real": {"type": "number", "minimum": 0, "maximum": 1,
                           "description": "差距是真实的概率（你的主观后验）"},
                "reasoning": {"type": "string", "description": "≤200 字"},
            },
            "required": ["verdict", "p_real", "reasoning"],
        },
        "proposals": {"type": "array", "items": _PROPOSAL, "maxItems": 10},
        "dissent": {"type": "string",
                    "description": "六个视角意见不一致的地方和你的取舍，≤200 字"},
        "narrative": {"type": "string", "description": "≤600 字，给人读的结论"},
    },
    "required": ["gap", "why", "noise_or_real", "proposals", "narrative"],
}


def _s(x: Any, n: int) -> str:
    return str(x if x is not None else "")[:n]


def _enum(x: Any, allowed: list[str], default: str) -> str:
    return x if x in allowed else default


def _num(x: Any, lo: float | None = None, hi: float | None = None,
         default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:  # noqa: BLE001
        return default
    if v != v:
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def sanitize_proposal(p: dict) -> dict:
    params = p.get("params") if isinstance(p.get("params"), dict) else {}
    return {
        "kind": _enum(p.get("kind"), KINDS, "process"),
        "line": _enum(p.get("line"), LINES, "both"),
        "target": _s(p.get("target"), 80),
        "change": _s(p.get("change"), 200),
        "rationale": _s(p.get("rationale"), 300),
        "expected_effect": _s(p.get("expected_effect"), 200),
        "test_plan": _s(p.get("test_plan"), 200),
        "priority": _enum(p.get("priority"), PRIORITY, "P2"),
        "params": params,
    }


def sanitize_lens(obj: dict, lens: str) -> dict:
    fs = []
    for f in (obj.get("findings") or [])[:12]:
        if not isinstance(f, dict):
            continue
        fs.append({"line": _enum(f.get("line"), LINES, "both"),
                   "claim": _s(f.get("claim"), 120),
                   "evidence": _s(f.get("evidence"), 300),
                   "magnitude": _s(f.get("magnitude"), 120),
                   "confidence": _enum(f.get("confidence"), CONF, "低")})
    return {
        "lens": lens,
        "summary": _s(obj.get("summary"), 600),
        "findings": fs,
        "proposals": [sanitize_proposal(p) for p in (obj.get("proposals") or [])[:6]
                      if isinstance(p, dict)],
        "questions": [_s(q, 200) for q in (obj.get("questions") or [])[:5]],
    }


def sanitize_chair(obj: dict) -> dict:
    gaps = []
    for g in (obj.get("gap") or [])[:12]:
        if not isinstance(g, dict):
            continue
        gaps.append({"line": _enum(g.get("line"), LINES, "both"),
                     "where": _s(g.get("where"), 120),
                     "expected": _num(g.get("expected")),
                     "actual": _num(g.get("actual")),
                     "unit": _s(g.get("unit"), 20),
                     "ci_lo": _num(g.get("ci_lo"), default=float("nan")),
                     "ci_hi": _num(g.get("ci_hi"), default=float("nan")),
                     "n": int(_num(g.get("n"), 0)),
                     "baseline": _num(g.get("baseline"), default=float("nan"))})
    why = []
    for w in (obj.get("why") or [])[:7]:
        if not isinstance(w, dict):
            continue
        why.append({"cause": _enum(w.get("cause"), CAUSES, "未知"),
                    "weight": _num(w.get("weight"), 0, 1),
                    "evidence": _s(w.get("evidence"), 300)})
    tot = sum(w["weight"] for w in why)
    if tot > 0:
        for w in why:
            w["weight"] = round(w["weight"] / tot, 3)
    nr = obj.get("noise_or_real") if isinstance(obj.get("noise_or_real"), dict) else {}
    return {
        "gap": gaps,
        "why": why,
        "noise_or_real": {
            "verdict": _enum(nr.get("verdict"), NOISE, "样本不足无法判断"),
            "p_real": _num(nr.get("p_real"), 0, 1, 0.5),
            "reasoning": _s(nr.get("reasoning"), 600),
        },
        "proposals": [sanitize_proposal(p) for p in (obj.get("proposals") or [])[:10]
                      if isinstance(p, dict)],
        "dissent": _s(obj.get("dissent"), 600),
        "narrative": _s(obj.get("narrative"), 1500),
    }
