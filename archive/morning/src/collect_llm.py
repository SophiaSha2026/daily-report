"""
把 claude-code-action 的 structured_output 转成 <目录>/commentary.json。
任何异常都不抛出——LLM 是可选增强，绝不能阻断发信。

目录由环境变量 LLM_OUT_DIR 指定（默认 out/）。竞价用 out/，
形态扫描用 out_pullback/，两条线共用这一份解析逻辑。
"""
from __future__ import annotations
import os, json, pathlib

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / (os.environ.get("LLM_OUT_DIR") or "out"))
OUT.mkdir(exist_ok=True)
dst = OUT / "commentary.json"

raw = (os.environ.get("RAW") or "").strip()
result: dict[str, dict] = {}

if raw:
    try:
        obj = json.loads(raw)
        for it in obj.get("items", []):
            code = str(it.get("code", "")).zfill(6)
            if len(code) == 6 and code.isdigit():
                result[code] = {
                    "reason": str(it.get("reason", ""))[:60],
                    "risk":   str(it.get("risk", ""))[:50],
                }
    except Exception as e:  # noqa: BLE001
        print(f"structured_output 解析失败: {type(e).__name__}: {e}")
        print(f"原始内容前 300 字: {raw[:300]}")
else:
    print("structured_output 为空（步骤失败/超时/未配置认证）")

dst.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{dst}: {len(result)} 条")
