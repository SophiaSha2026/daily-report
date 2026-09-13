"""
状态汇总：一次 HTTP 调用回答「今天这套东西到底跑没跑、有没有出事」。

设计动机
--------
2026-09-07 到 09-11，本地 push 连续失败五个交易日，没有任何人发现，
因为唯一的信号是日志里一行 warning。同期本地和云端各发各的邮件，
用户每天收到两封也没意识到是故障。

结论：**失败必须有一个能被界面查询的对象**，不能只写日志。
所以这里把四类信号收在一起，任何一条不正常，总览页就亮红：

    流程   今天各条线跑完没有（读各自的 run_meta.json）
    同步   上次 push 成没成（读 state/push_status.json）+ 本地和远端差几个 commit
    排期   计划任务是启用还是停用、上次跑的结果
    面板   三个面板各自是哪一天的

所有读取都必须容错：学习系统没跑过、state/ 整个不存在，也要能出页面。
CLAUDE.md 的硬约束第 8 条同理——三条自测线不许依赖 state/。
"""
from __future__ import annotations

import json
import subprocess
import time
import datetime as dt
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# 各条线的完成标记文件，和 local_run.py 的 FLOWS 保持一致。
# 这里重复一份而不是 import，是因为 GUI 不该因为 local_run 的依赖
# （akshare 之类）而起不来。字段少，漂了也一眼看得出来。
LINES = [
    {"key": "morning", "name": "早盘选股", "meta": "out/run_meta.json",
     "panel": "out/panel.html", "due": "09:27:30",
     "task": "DailyReport-Local-Morning"},
    {"key": "evening", "name": "回调形态", "meta": "out_pullback/run_meta.json",
     "panel": "out_pullback/panel.html", "due": "已停用自动",
     "task": None},
    {"key": "learn", "name": "参数自学", "meta": "state/learning_status.json",
     "panel": "out_learn/learn.html", "due": "收盘后",
     "task": "DailyReport-Local-Learn"},
    {"key": "breakout", "name": "起涨预测", "meta": "out_breakout/run_meta.json",
     "panel": "out_breakout/panel.html", "due": "17:00",
     "task": "DailyReport-Local-Evening"},
]


def now_bj() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=8)


def today_bj() -> str:
    return now_bj().strftime("%Y-%m-%d")


def _json(rel: str) -> dict:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=15)
        return (r.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# 查计划任务要起一个 powershell.exe，约 0.5 秒。总览页每 5 秒刷一次，
# 每次都付这个代价不值，缓存 10 秒。
_task_cache: dict = {"at": 0.0, "data": {}}
_task_lock = threading.Lock()


def scheduled_tasks(force: bool = False) -> dict:
    """查 DailyReport-* 计划任务的启用状态和上次运行结果。

    用 PowerShell 出 JSON 而不是 schtasks：schtasks 的 LIST 输出字段名
    跟着系统显示语言走，中文系统上解析出来的键和英文系统对不上。
    """
    with _task_lock:
        if (not force and _task_cache["data"]
                and time.time() - _task_cache["at"] < 10):
            return _task_cache["data"]
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-ScheduledTask -TaskName 'DailyReport-*' | ForEach-Object {"
        "  $i = $_ | Get-ScheduledTaskInfo;"
        "  [pscustomobject]@{"
        "    name=$_.TaskName; state=[string]$_.State;"
        "    last=if($i.LastRunTime){$i.LastRunTime.ToString('MM-dd HH:mm')}else{''};"
        "    rc=$i.LastTaskResult;"
        "    next=if($i.NextRunTime){$i.NextRunTime.ToString('MM-dd HH:mm')}else{''}"
        "  } } | ConvertTo-Json -Compress"
    )
    out = {}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=25,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
        data = json.loads((r.stdout or "").strip() or "[]")
        if isinstance(data, dict):
            data = [data]
        for d in data:
            # 从没跑过的任务，Windows 返回两个哨兵值而不是空：
            #   LastRunTime   = 1899-11-30（OLE 自动化的零日期）
            #   LastTaskResult= 267011 = 0x41303 SCHED_S_TASK_HAS_NOT_RUN
            # 照原样显示就成了「11-30 00:00 / 错误码 267011」，看着像出过错。
            if d.get("last", "").startswith("11-30"):
                d["last"] = ""
            if d.get("rc") == 267011:
                d["rc"] = None
                d["never"] = True
        out = {d["name"]: d for d in data}
    except Exception:  # noqa: BLE001
        out = {}
    with _task_lock:
        _task_cache.update(at=time.time(), data=out)
    return out


# 总览页每 5 秒刷一次，但 git fetch 是要打网络的，不该跟着刷。
# ahead/behind 本身是本地计算，快，每次都算。
_fetch_at = {"t": 0.0}


def sync_status() -> dict:
    """仓库同步状况。这是 09-07 那次失联唯一能提前发现的地方。"""
    push = _json("state/push_status.json")
    if time.time() - _fetch_at["t"] > 60:
        _git("fetch", "-q", "origin", "main")
        _fetch_at["t"] = time.time()
    ahead = _git("rev-list", "--count", "origin/main..main")
    behind = _git("rev-list", "--count", "main..origin/main")
    dirty = _git("status", "--porcelain")
    stuck = [d for d in ("rebase-merge", "rebase-apply", "MERGE_HEAD",
                         "CHERRY_PICK_HEAD") if (ROOT / ".git" / d).exists()]

    problems = []
    if push and not push.get("ok"):
        problems.append(f"上次推送失败：{(push.get('detail') or '')[:120]}")
    if ahead.isdigit() and int(ahead) > 0:
        problems.append(f"本地有 {ahead} 个 commit 没推上去")
    if behind.isdigit() and int(behind) > 0:
        problems.append(f"远端有 {behind} 个 commit 没拉下来")
    if stuck:
        problems.append(f"仓库卡在 {'/'.join(stuck)} 中间状态，"
                        f"下次推送会自动清理")
    return {
        "ok": not problems,
        "problems": problems,
        "ahead": ahead or "?", "behind": behind or "?",
        "dirty": len([x for x in dirty.splitlines() if x.strip()]),
        "last_push": push.get("at", ""),
        "last_push_ok": push.get("ok"),
        "last_push_msg": push.get("msg", ""),
    }


def line_status(tasks: dict) -> list[dict]:
    d = today_bj()
    out = []
    for ln in LINES:
        meta = _json(ln["meta"])
        done = meta.get("date") == d
        panel = ROOT / ln["panel"]
        try:
            pdate = _json(ln["meta"]).get("date") or ""
            pmtime = (dt.datetime.fromtimestamp(panel.stat().st_mtime)
                      .strftime("%m-%d %H:%M")) if panel.exists() else ""
        except Exception:  # noqa: BLE001
            pdate, pmtime = "", ""
        t = tasks.get(ln["task"] or "", {})
        out.append({
            "key": ln["key"], "name": ln["name"], "due": ln["due"],
            "done": done, "date": meta.get("date", ""),
            "n": meta.get("n"), "panel_date": pdate, "panel_mtime": pmtime,
            "task": ln["task"], "task_state": t.get("state", ""),
            "task_last": t.get("last", ""), "task_rc": t.get("rc"),
            "task_next": t.get("next", ""),
        })
    return out


def overview() -> dict:
    tasks = scheduled_tasks()
    lines = line_status(tasks)
    sync = sync_status()
    bj = now_bj()
    weekend = bj.weekday() >= 5

    # 云端是不是真的停干净了。改完 workflow 忘了推是很容易犯的错，
    # 这里直接读远端 main 上的文件，而不是读工作区。
    cloud = _git("grep", "-c", "^  *- cron:", "origin/main",
                 "--", ".github/workflows")
    cloud_live = [x for x in cloud.splitlines() if x.strip()]

    return {
        "bj": bj.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": "一二三四五六日"[bj.weekday()],
        "weekend": weekend,
        "lines": lines,
        "sync": sync,
        "tasks": tasks,
        "cloud_cron_live": cloud_live,
        "generated": dt.datetime.now().strftime("%H:%M:%S"),
    }
