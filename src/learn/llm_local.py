"""
本地 LLM 归因：直接调本机的 `claude` CLI。

为什么要有本地这条路
--------------------
远端那条走 claude-code-action，依赖 GitHub Actions 跑起来、依赖
CLAUDE_CODE_OAUTH_TOKEN 这个 secret 没过期。用户的方针是**本地为主、
远端为辅**，所以归因也要能在本机跑完，不依赖任何远端设施。

OAuth 自动识别（按顺序）
------------------------
1. 环境变量 CLAUDE_CODE_OAUTH_TOKEN
2. tools/local.env 里的 CLAUDE_CODE_OAUTH_TOKEN=...
3. CLI 自己的登录态（本机交互式登录过就有）

前两个是 `claude setup-token` 生成的长期 token，为非交互环境设计
（CLAUDE.md 历史教训第 5 条）。第三条在会话过期时会失败——CLI 的坑是
**退出码 1 但 stdout 仍是合法 JSON**，错误在 is_error/result 字段里，
stderr 是空的。所以这里解析 JSON 而不是看退出码，认证失败时把修法
直接打进日志。

三条约束
--------
1. **绝不参与排序。** 输出只影响报告文本和 regime_weight 查表出来的
   日权重。给定落盘的 JSON，优化器完全确定。
2. **失败不阻断。** 所有异常吞掉返回 None，那一天日权重按 1.0 算。
3. **枚举白名单。** 越界的 cause/day_regime 改写成 形态失效/正常。
   通道 2 按枚举查表，脏值会静默改变学习行为。
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT = ROOT / "prompts" / "eval_analyst.md"
OUTDIR = ROOT / "state" / "llm_eval"

CAUSES = {"个股利空", "个股利好", "板块轮动", "大盘系统性",
          "流动性陷阱", "情绪退潮", "形态失效", "数据异常"}
REGIMES = CAUSES | {"正常"}

_SCHEMA = """
只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码围栏。结构：

{
  "date": "YYYY-MM-DD",
  "day_regime": "<枚举>",
  "day_note": "<一句话，不超过 40 字>",
  "items": [
    {"code": "600000", "cause": "<枚举>", "evidence": "<不超过 30 字的具体事实>"}
  ],
  "param_proposal": {},
  "feature_proposal": []
}

枚举只能取：个股利空 个股利好 板块轮动 大盘系统性 流动性陷阱 情绪退潮 形态失效 数据异常
day_regime 还可以取「正常」。
items 要覆盖 brief 里 worst 和 best 的每一只，code 逐一对应，不遗漏不新增。
"""


def available() -> bool:
    return shutil.which("claude") is not None


def find_token() -> tuple[str, str]:
    """(token, 来源)。找不到返回 ("", "")。token 本身绝不进日志。"""
    t = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if t:
        return t, "环境变量"
    p = ROOT / "tools" / "local.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CLAUDE_CODE_OAUTH_TOKEN="):
                v = line.split("=", 1)[1].strip()
                if v and "FILLME" not in v:
                    return v, "tools/local.env"
    return "", ""


def _run_cli(prompt: str, model: str, timeout: int,
             tools: str = "WebSearch,WebFetch") -> tuple[dict | None, str]:
    """跑一次 CLI，返回 (外层JSON, 错误说明)。自动带上找到的 token。

    stdin 必须显式接 DEVNULL：CLI 会等 3 秒管道输入并打一条 warning，
    子进程环境里那 3 秒纯属浪费。
    """
    env = dict(os.environ)
    tok, src = find_token()
    if tok:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
    cmd = ["claude", "-p", prompt, "--model", model,
           "--output-format", "json", "--allowed-tools", tools]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL,
                       encoding="utf-8", errors="replace", timeout=timeout)
    j = _extract_json(r.stdout)
    if isinstance(j, dict) and j.get("is_error"):
        return None, str(j.get("result", ""))[:200]
    if r.returncode != 0:
        return None, (r.stderr or "")[:200] or f"退出码 {r.returncode}，无输出"
    return j, ""


def auth_status() -> tuple[bool, str]:
    """体检用：本地归因这条路通不通。会真调一次最小请求。"""
    if not available():
        return False, "本机没有 claude CLI"
    tok, src = find_token()
    try:
        j, err = _run_cli("回复一个字：好", "claude-haiku-4-5-20251001", 60, "")
        if err:
            hint = f"（已带 {src} 的长期 token）" if tok else \
                "（无长期 token，靠 CLI 登录态）"
            return False, err[:80] + hint
        return True, f"OK（{src}）" if tok else "OK（CLI 登录态）"
    except Exception as e:  # noqa: BLE001
        return False, type(e).__name__


def _extract_json(text: str) -> dict | None:
    """从模型输出里挖 JSON。实测的脏法：```json 围栏、前置寒暄、尾置总结。"""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:  # noqa: BLE001
        return None


def _sanitize(obj: dict, date: str, codes: list[str]) -> dict:
    """枚举校验 + 补全。脏值改写而不是丢弃，下游拿到的一定合法。"""
    out = {
        "date": date,
        "day_regime": obj.get("day_regime") if obj.get("day_regime") in REGIMES
                      else "正常",
        "day_note": str(obj.get("day_note", ""))[:80],
        "items": [],
        "param_proposal": obj.get("param_proposal") or {},
        "feature_proposal": obj.get("feature_proposal") or [],
        "source": "local-cli",
    }
    seen = {}
    for it in obj.get("items") or []:
        c = str(it.get("code", "")).zfill(6)
        if c in codes:
            seen[c] = {"code": c,
                       "cause": it.get("cause")
                       if it.get("cause") in CAUSES else "形态失效",
                       "evidence": str(it.get("evidence", ""))[:60]}
    for c in codes:
        out["items"].append(seen.get(c, {
            "code": c, "cause": "形态失效", "evidence": "模型未覆盖"}))
    return out


def run(date: str, brief_path: Path, model: str = "claude-opus-5",
        timeout: int = 300) -> dict | None:
    """跑一次归因。失败返回 None，绝不抛。"""
    try:
        if not available():
            log.warning("本机没有 claude CLI，跳过归因")
            return None
        if not brief_path.exists():
            log.warning("没有 %s，跳过归因", brief_path.name)
            return None
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        codes = [x["code"] for x in brief.get("worst", [])
                 + brief.get("best", [])]
        if not codes:
            log.warning("brief 里没有样本，跳过归因")
            return None

        prompt = (PROMPT.read_text(encoding="utf-8")
                  + "\n\n# 今日输入\n\n```json\n"
                  + json.dumps(brief, ensure_ascii=False, indent=2)
                  + "\n```\n\n# 输出格式\n" + _SCHEMA)

        tok, src = find_token()
        log.info("本地归因：%d 只，超时 %ds，认证=%s", len(codes), timeout,
                 src or "CLI 登录态")
        envj, err = _run_cli(prompt, model, timeout)
        if err:
            low = err.lower()
            if "authenticate" in low or "oauth" in low:
                log.warning("归因跳过：claude 认证失效（%s）。修法二选一：", err[:80])
                log.warning("  1) 交互式终端跑一次 `claude` 重新登录")
                log.warning("  2) `claude setup-token` 生成长期 token，写进 "
                            "tools/local.env 的 CLAUDE_CODE_OAUTH_TOKEN=")
            else:
                log.warning("claude CLI 失败: %s", err)
            return None

        text = envj.get("result") if isinstance(envj, dict) \
            and "result" in envj else ""
        obj = _extract_json(text if isinstance(text, str)
                            else json.dumps(text, ensure_ascii=False))
        if not obj:
            log.warning("归因输出解析不出 JSON，前 200 字: %s", str(text)[:200])
            return None

        res = _sanitize(obj, date, codes)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        p = OUTDIR / f"{date}.json"
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        log.info("归因落盘 %s：day_regime=%s，%d 条", p.name,
                 res["day_regime"], len(res["items"]))
        return res
    except subprocess.TimeoutExpired:
        log.warning("归因超时（%ds），跳过", timeout)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("归因失败（不影响学习流程）: %s", e)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        ok, msg = auth_status()
        print(("OK   " if ok else "FAIL ") + "claude auth  " + msg)
        sys.exit(0 if ok else 1)
    d = sys.argv[1] if len(sys.argv) > 1 else ""
    run(d, ROOT / "out_learn" / "eval_brief.json")
