"""
LLM 审稿人：参数变更过完统计闸门之后、落地之前，Opus 审一遍证据包。

它在调优流程里的位置（和另外三条通道的关系）：

    通道 1  归因        每天，day_regime -> 优化器日权重（定量、已生效）
    通道 2  参数提案     Opus 提的和优化器提的走同一套闸门
    通道 3  特征提案     进积压清单，人工实现
    通道 4  审稿（本模块）统计闸门全过后，Opus 交叉检查数字和叙事

审稿人能看到统计检验看不到的东西：比如「这次要调低量能饱和点，但最近
两周的归因显示失败主要是板块轮动，跟量能无关」——数字对但故事不对，
这类矛盾正是 LLM 擅长抓的。

两种模式（config: learning.llm.review_mode）：

    advisory  意见附在变更邮件里，不拦。默认。
    veto      「反对」会把变更搁置：learned.yaml 不写，证据和意见落盘，
              邮件改发「变更被审稿搁置」，人工确认后手动落地：
                  python src/eval_daily.py --stage apply-held
              人永远有终审权，LLM 只能按暂停，不能按删除。

红线不变：无论哪种模式，LLM 都不写参数值、不排序。它的输出是枚举 + 文字，
落盘可审计。失败一律 fail-open（当 advisory 无意见处理），绝不因为审稿
崩了拦住一次统计上站得住的变更。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
OUTDIR = ROOT / "state" / "llm_review"

STANCES = {"支持", "保留意见", "反对"}

_PROMPT = """# 角色

你是量化团队的参数变更审稿人。一次筛选参数变更刚通过了全部统计闸门
（走向前样本外、按天自助、行为回放、在线稳健性等七道），现在需要你
做最后一道交叉检查：**数字和叙事对不对得上**。

你不是重算统计——那是闸门的事，已经过了。你要找的是统计检验找不到的矛盾：

- 变更方向和最近的归因记录矛盾（比如调量能参数，但近期失败全是板块轮动）
- 变更幅度虽小但方向与领域常识冲突（比如放松「越极端越警惕」）
- 证据窗口里有已知的特殊时期（长假前后、极端行情）可能污染结论
- 影子模型的对比数据和这次变更指向不同的方向

# 输入

下面的 JSON 包含：动了哪些参数（旧值/新值）、七道闸的证据、
最近的归因分布、影子模型对比。

# 输出

只输出一个 JSON 对象，不要围栏不要解释：

{"stance": "支持|保留意见|反对",
 "points": ["每条不超过 40 字的具体理由，1~4 条"],
 "suggest": "一句话建议，可为空"}

「反对」是很重的动作（veto 模式下会搁置变更），只在发现**实质矛盾**时用；
没有实质问题但有值得记录的顾虑，用「保留意见」。
"""


def run(date: str, package: dict, model: str = "claude-opus-5",
        timeout: int = 180) -> dict:
    """审一次。任何失败返回中性意见（fail-open）。"""
    neutral = {"stance": "支持", "points": ["审稿不可用，未做交叉检查"],
               "suggest": "", "source": "fallback"}
    try:
        from learn import llm_local
        if not llm_local.available():
            return neutral
        prompt = (_PROMPT + "\n\n# 证据包\n```json\n"
                  + json.dumps(package, ensure_ascii=False, indent=2,
                               default=str)
                  + "\n```")
        envj, err = llm_local._run_cli(prompt, model, timeout, tools="")
        if err:
            log.warning("审稿失败（按无意见处理）: %s", err[:100])
            return neutral
        obj = llm_local._extract_json(
            envj.get("result", "") if isinstance(envj, dict) else "")
        if not isinstance(obj, dict):
            return neutral
        res = {
            "stance": obj.get("stance") if obj.get("stance") in STANCES
                      else "支持",
            "points": [str(x)[:60] for x in (obj.get("points") or [])][:4],
            "suggest": str(obj.get("suggest", ""))[:80],
            "source": "opus",
        }
        OUTDIR.mkdir(parents=True, exist_ok=True)
        (OUTDIR / f"{date}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("审稿意见：%s（%d 条理由）", res["stance"], len(res["points"]))
        return res
    except Exception as e:  # noqa: BLE001
        log.warning("审稿异常（按无意见处理）: %s", e)
        return neutral


def as_note(res: dict) -> str:
    """渲染进变更邮件的那一段。"""
    mark = {"支持": "✓", "保留意见": "△", "反对": "✗"}.get(res["stance"], "")
    pts = "；".join(res.get("points", []))
    sug = res.get("suggest", "")
    return (f"<b>Opus 审稿：{mark} {res['stance']}</b><br>{pts}"
            + (f"<br>建议：{sug}" if sug else ""))
