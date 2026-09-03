"""
远端让位检查。只在 GitHub Actions 里跑，本地永远不需要它。

    python tools/yield_check.py auction   竞价线：紧贴发信双阈值
    python tools/yield_check.py pullback  形态线：时间宽裕，等 20 分钟

协议（另一半在 src/local_run.py 的模块注释里）：

    state/claim/<flow>_<date>.json   本地开跑时推送
    state/sent/<flow>_<date>.json    本地发信成功后推送

本步骤的产出是 GITHUB_OUTPUT 里的 skip_mail=0/1，
下游 enrich/send 步骤把它接到 SKIP_MAIL 环境变量上。

三种结局：
    无 claim            skip_mail=0，立即返回，发信时刻不受任何影响
    claim + sent        skip_mail=1，远端只发布面板
    claim + 无 sent     skip_mail=0，远端兜底发信（竞价线此时约 09:28:20，
                        仍在 send_deadline 09:28:30 之内）

fail-open：这个脚本自己崩了就输出 skip_mail=0，远端照发。
宁可用户收到两封，不能一封都没有。
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)


def out(skip: bool, why: str) -> None:
    print(f"skip_mail={'1' if skip else '0'}  ({why})")
    go = os.environ.get("GITHUB_OUTPUT")
    if go:
        with open(go, "a", encoding="utf-8") as f:
            f.write(f"skip_mail={'1' if skip else '0'}\n")


def marker_on_origin(kind: str, flow: str, date: str) -> bool:
    """标记是否已在 origin/main 上。查远端不查本地：
    本 runner checkout 的是触发时刻的树，本地标记是之后才推的。
    """
    path = f"state/{kind}/{flow}_{date}.json"
    subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT,
                   capture_output=True, timeout=60)
    r = subprocess.run(["git", "cat-file", "-e", f"origin/main:{path}"],
                       cwd=ROOT, capture_output=True, timeout=30)
    return r.returncode == 0


def sleep_to(hms: str) -> None:
    h, m, s = (int(x) for x in hms.split(":"))
    tgt = now_bj().replace(hour=h, minute=m, second=s, microsecond=0)
    delta = (tgt - now_bj()).total_seconds()
    if delta > 0:
        print(f"等到 {hms}（{delta:.0f} 秒）", flush=True)
        time.sleep(delta)


def main() -> int:
    flow = sys.argv[1] if len(sys.argv) > 1 else "auction"
    date = now_bj().strftime("%Y-%m-%d")
    try:
        import yaml
        rt = yaml.safe_load((ROOT / "config.yaml").read_text(
            encoding="utf-8"))["runtime"]
        soft, hard = rt["send_at"], rt.get("send_deadline", rt["send_at"])
    except Exception:  # noqa: BLE001
        soft, hard = "09:27:30", "09:28:30"

    try:
        if flow == "auction":
            # 发信前 30 秒查 claim（send_at 09:27:30 -> 09:27:00）。
            # 没有就立刻放行，发信时刻分毫不动。时点从 config 推，
            # 改了 send_at 这里自动跟着走，不再各写各的。
            h, m, s = (int(x) for x in soft.split(":"))
            chk = (dt.datetime(2000, 1, 1, h, m, s)
                   - dt.timedelta(seconds=30)).strftime("%H:%M:%S")
            sleep_to(chk)
            if not marker_on_origin("claim", "auction", date):
                out(False, "无本地接管声明，照常发信")
                return 0
            # 有 claim：等到硬上限前 10 秒，确认本地是否真的发出去了
            h, m, s = (int(x) for x in hard.split(":"))
            wait = dt.time(h, m, max(s - 10, 0)).strftime("%H:%M:%S")
            print(f"检测到本地接管声明，等到 {wait} 确认")
            sleep_to(wait)
            if marker_on_origin("sent", "auction", date):
                out(True, "本地已发信，远端只发布面板")
            else:
                out(False, "本地声明了接管但没发出，远端兜底")
        else:
            # 形态线数据定型，不抢秒。有 claim 就多等 20 分钟。
            if not marker_on_origin("claim", "pullback", date):
                out(False, "无本地接管声明，照常发信")
                return 0
            print("检测到本地接管声明，等 20 分钟确认")
            time.sleep(20 * 60)
            if marker_on_origin("sent", "pullback", date):
                out(True, "本地已发信，远端只发布面板")
            else:
                out(False, "本地声明了接管但没发出，远端兜底")
    except Exception as e:  # noqa: BLE001
        # fail-open：检查本身出错绝不能吞掉邮件
        out(False, f"检查异常({type(e).__name__})，照常发信")
    return 0


if __name__ == "__main__":
    sys.exit(main())
