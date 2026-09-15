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
    # system 字段决定总览页归到哪一组。仓库里就两个系统，别再加第三个。
    {"key": "morning", "name": "早盘选股", "system": "早盘系统",
     "meta": "out/run_meta.json", "panel": "out/panel.html",
     "due": "09:27:30", "task": "DailyReport-Local-Morning"},
    {"key": "learn", "name": "参数自学", "system": "早盘系统",
     "meta": "state/learning_status.json", "panel": "out_learn/learn.html",
     "due": "收盘后", "task": "DailyReport-Local-Learn"},
    {"key": "breakout", "name": "起涨预测", "system": "晚间系统",
     "meta": "out_breakout/run_meta.json", "panel": "out_breakout/panel.html",
     "due": "17:00 后，最晚次日 08:30", "task": "DailyReport-Local-Evening"},
    {"key": "evening", "name": "回调形态", "system": "辅助工具",
     "meta": "out_pullback/run_meta.json", "panel": "out_pullback/panel.html",
     "due": "已停用自动", "task": None},
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


# 交易日历：只读 local_run 落盘的缓存 state/trade_dates.json，**绝不在
# 控制台进程里 import akshare**。2026-09-15 实测：/api/status 被两个线程
# 同时打，两边同时 import akshare -> py_mini_racer 的 V8 进程级 FATAL
# （Check failed: !IsConfigurablePoolInitialized()），整个控制台直接没了，
# 历史教训 19 的翻版。缓存没有就按周一到周五（fail-open）。
_td_cache: dict = {"at": 0.0, "s": set()}
_td_lock = threading.Lock()


def _trade_dates() -> set:
    with _td_lock:
        if time.time() - _td_cache["at"] > 600:
            s: set = set()
            try:
                f = ROOT / "state" / "trade_dates.json"
                if f.exists() and time.time() - f.stat().st_mtime < 14 * 86400:
                    s = set(json.loads(f.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                s = set()
            _td_cache.update(at=time.time(), s=s)
        return _td_cache["s"]


def target_date(key: str) -> str:
    """这条线该产出哪一天。和 local_run.target_date 同一口径：
    竞价线是今天，其余是最近一个已收盘（15:05 后）的交易日。"""
    if key == "morning":
        return today_bj()
    now = now_bj()
    tds = _trade_dates()
    closed = (now.hour, now.minute) >= (15, 5)
    for back in range(15):
        day = now.date() - dt.timedelta(days=back)
        ok = (day.isoformat() in tds) if tds else day.weekday() < 5
        if ok and (back > 0 or closed):
            return day.isoformat()
    return today_bj()


def line_status(tasks: dict) -> list[dict]:
    out = []
    for ln in LINES:
        meta = _json(ln["meta"])
        d = target_date(ln["key"])
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
            "system": ln["system"],
            "done": done, "date": meta.get("date", ""), "target": d,
            "n": meta.get("n"), "panel_date": pdate, "panel_mtime": pmtime,
            "task": ln["task"], "task_state": t.get("state", ""),
            "task_last": t.get("last", ""), "task_rc": t.get("rc"),
            "task_next": t.get("next", ""),
        })
    return out


def breakout_model() -> dict:
    """起涨预测的模型状态。晚间系统的自学习是否在正常转，看这个。

    不 import breakout.daily（那会连带拉起 lightgbm/pandas，控制台就起不来了），
    直接读它写的 json。字段漂了这里会返回 exists=False，界面上看得见。
    """
    f = ROOT / "state" / "breakout" / "model.json"
    if not f.exists():
        return {"exists": False}
    try:
        m = json.loads(f.read_text(encoding="utf-8"))
        age = (now_bj().date()
               - dt.date.fromisoformat(m["fit_date"])).days
        return {"exists": True, "fit_date": m["fit_date"], "age_days": age,
                "n_feats": len(m.get("feats", [])),
                "train_cut": m.get("train_cut", ""),
                "days_to_refit": max(30 - age, 0)}
    except Exception:  # noqa: BLE001
        return {"exists": False}


def morning_perf() -> dict:
    """早盘选股的实测成绩，从参数自学的统计里读。

    口径要说准：learn/dataset.py 里的 y 是**按天中心化**的收益，
    所以 hit_rate 是「跑赢当天大盘的比例」（随机基准 50%），
    top_excess 是「平均超额收益」（随机基准 0%）。
    说成「上涨概率」就错了。
    """
    m = _json("state/learning_status.json").get("metrics") or {}
    if not m:
        return {"exists": False}
    return {"exists": True, "days": m.get("days"),
            "hit": m.get("hit_rate"), "excess": m.get("top_excess")}


# 触发规则，给排期页显式列出来。时刻两套：本机任务是美东时间（跟夏令时走），
# 开跑窗口是北京时间（local_run.FLOWS 判的就是北京时间）。
# 规则本身在 src/local_run.py（本机）、tools/yield_check.py 和
# tools/evening_check.py（云端）里，这里只是把它们用人话写出来。
RULES = {
    "morning": {
        "local_when": "手动：美东 18:00~20:30（冬令时 17:00~19:30）随时点。"
                      "自动：计划任务美东周日~周四 18:00 起每 15 分钟敲，北京 08:30 起没手点就跑",
        "target_rule": "今天（09:16 之后新起进程来不及赶上 09:25 采样）",
        "steps": "同步仓库 -> 推 claim -> 候选池 -> 09:14 预热、09:19/09:23/09:25 采样 -> "
                 "推数据快照 -> Claude 文案 -> 09:27:30 发信 -> 推 sent + 面板",
        "cloud": "auction.yml：cron 07:40/08:20/08:59/09:11 北京 + Cloudflare Worker 07:30 派发。"
                 "云端照常采样；09:25:50 看到本地已推数据快照就不跑 Claude；"
                 "09:27:00 看到本地 claim 就等到 09:28:20 确认 sent：有 sent -> 只发布 Pages 面板，"
                 "不发信、不提交数据；没 claim 或没 sent -> 云端 09:27:30 发信、提交数据。",
    },
    "breakout": {
        "local_when": "手动：北京 16:00 起到次日 08:30 随时点。"
                      "自动：计划任务美东周一~周五 04:30 起每 30 分钟敲，北京 16:30 起没手点就跑；"
                      "机器睡着就等醒了补，登录时也敲一次",
        "target_rule": "最近一个已收盘（15:05 后）的交易日。北京 09-15 早上补跑出的是 09-14 的清单",
        "steps": "同步仓库 -> 推 claim -> 腾讯快照追加目标日日线（追加不到就不出清单）-> "
                 "重算特征（约 13 分钟）-> 打分、风险剔除 -> 面板 + 发信 -> 推 sent + 产物",
        "cloud": "evening_check.yml：cron 20:30 北京 + Worker 20:45 派发。云端算不了这条线"
                 "（特征表 2.5GB 在本机，新浪源云端不通），只看 origin/main 有没有目标日的清单，"
                 "没有就发一封「本机没跑」提醒，同一天只发一次。",
    },
    "learn": {
        "local_when": "自动：计划任务美东周一~周五 04:40 起每 30 分钟敲，北京 16:40 起跑；手动随时",
        "target_rule": "同起涨预测：最近一个已收盘的交易日",
        "steps": "同步仓库 -> 标签 -> 归因 -> 拟合与闸门 -> 推学习产物",
        "cloud": "无。learn.yml 只留手动入口。",
    },
    "evening": {
        "local_when": "已停用自动（2026-09-12 用户取消每日形态报告），只剩控制台手动入口",
        "target_rule": "最近一个已收盘的交易日",
        "steps": "形态扫描 -> 发信 -> 学习线",
        "cloud": "无。pullback.yml 只留手动入口；Worker 09-15 起不再派发它。",
    },
}

# --if-needed 的五道检查，顺序就是 local_run.main 里的顺序
IF_NEEDED = [
    "北京时间周末：跳过",
    "这条线正在跑（state/lock 或进程表）：跳过",
    "目标日已经跑完（run_meta 日期 == 目标日）：跳过",
    "不在开跑窗口：只拉一次远端，不跑",
    "还没到自动开跑时刻：只拉一次远端，等手动",
    "都通过：先拉远端再核对一次（云端可能已经代跑），然后开跑",
]

PRIORITY = [
    "1. 手动：控制台点按钮，开跑窗口内随时。",
    "2. 本机自动：到了自动开跑时刻还没手点，计划任务自己跑（每 15~30 分钟敲一次，机器睡着就等醒了补）。",
    "3. 云端：本机根本没跑（没开机），GitHub cron + Cloudflare Worker 叫起云端兜底："
    "早盘由云端代发；起涨预测云端算不了，只发提醒。",
]

MANUAL_RULE = ("控制台按钮和计划任务走同一个入口、同一把锁，谁先起谁跑。手点不看跑没跑过、"
               "不看自动时刻，只看锁：已经在跑就直接退出。带「会发邮件」的按钮点了就真的发，点前会弹确认。")


def _local_run():
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import local_run
        return local_run
    except Exception:  # noqa: BLE001
        return None


def verdict(key: str, done: bool, target: str) -> str:
    """现在这一刻触发这条线，会发生什么。逐条复述 local_run.main 的判断。"""
    L = _local_run()
    now = now_bj()
    if key == "evening":
        return "自动已停用，只有手动"
    if now.weekday() >= 5:
        return "北京周末，触发了也跳过"
    if L:
        try:
            other = L.running_instance(key)
            if other:
                since = str(other.get("at", ""))[11:16]
                return (f"正在跑（pid {other.get('pid')}"
                        + (f"，{since} 起" if since else "") + "），再触发直接退出")
        except Exception:  # noqa: BLE001
            pass
    if done:
        return f"目标日 {target} 已跑完，再触发直接退出"
    if L:
        try:
            if not L.in_window(key):
                lo, hi = L.FLOWS[key][2:4]
                return (f"不在开跑窗口 {lo[0]:02d}:{lo[1]:02d}-{hi[0]:02d}:{hi[1]:02d}（北京），"
                        f"计划任务只拉远端不跑；手动也不建议")
            if not L.auto_due(key):
                auto = L.FLOWS[key][4]
                return (f"手动窗口内：等你在控制台点；{auto[0]:02d}:{auto[1]:02d}（北京）"
                        f"还没点，计划任务就自动跑")
        except Exception:  # noqa: BLE001
            pass
    return "过了自动开跑时刻、目标日还没跑：计划任务下一次敲就开跑（手动也可以）"


def window_text(key: str) -> str:
    L = _local_run()
    if not L or key not in L.FLOWS:
        return "?"
    lo, hi = L.FLOWS[key][2:4]
    s = f"北京 {lo[0]:02d}:{lo[1]:02d}-{hi[0]:02d}:{hi[1]:02d}"
    return s + ("（跨午夜到次日）" if hi < lo else "")


def auto_text(key: str) -> str:
    L = _local_run()
    if not L or key not in L.FLOWS or len(L.FLOWS[key]) < 5:
        return "?"
    a = L.FLOWS[key][4]
    return f"北京 {a[0]:02d}:{a[1]:02d}"


def rules_status(lines: list[dict]) -> dict:
    now = now_bj()
    edt = dt.datetime.now()
    out = []
    for ln in lines:
        r = RULES.get(ln["key"], {})
        out.append({
            "key": ln["key"], "name": ln["name"], "task": ln["task"],
            "local_when": r.get("local_when", ""),
            "window": window_text(ln["key"]),
            "auto_from": auto_text(ln["key"]),
            "target_rule": r.get("target_rule", ""),
            "target": ln.get("target", ""),
            "done": ln.get("done"),
            "verdict": verdict(ln["key"], bool(ln.get("done")), ln.get("target", "")),
            "steps": r.get("steps", ""),
            "cloud": r.get("cloud", ""),
        })
    return {
        "now": f"北京 {now.strftime('%m-%d %H:%M')}（本机 {edt.strftime('%m-%d %H:%M')}）",
        "lines": out,
        "if_needed": IF_NEEDED,
        "priority": PRIORITY,
        "manual": MANUAL_RULE,
        "worker": "Cloudflare Worker daily-report-trigger：每天 07:30 北京派发 auction.yml，"
                  "20:45 北京派发 evening_check.yml。第三层触发，改它要重新 deploy。",
    }


# 云端该开着的托底 workflow。多了是没停干净，少了是托底没开。
CLOUD_FALLBACK = {
    "auction.yml": "竞价线代跑（本地发了信就只发布面板）",
    "evening_check.yml": "晚间提醒（本地没跑起涨预测就发邮件）",
}


def overview() -> dict:
    tasks = scheduled_tasks()
    lines = line_status(tasks)
    sync = sync_status()
    bj = now_bj()
    weekend = bj.weekday() >= 5

    # 云端哪些 workflow 在自动跑。读远端 main 上的文件而不是工作区：
    # 改完 workflow 忘了推是很容易犯的错。
    # 该开着的只有两条托底（竞价线代跑、晚间提醒），别的还带 cron 就是
    # 没停干净，总览页亮黄。
    cloud = _git("grep", "-c", "^  *- cron:", "origin/main",
                 "--", ".github/workflows")
    live = {}
    for x in cloud.splitlines():
        # 形如 origin/main:.github/workflows/auction.yml:4
        parts = x.strip().split(":")
        if len(parts) >= 3:
            live[parts[-2].rsplit("/", 1)[-1]] = int(parts[-1] or 0)
    unexpected = sorted(k for k in live if k not in CLOUD_FALLBACK)
    missing = sorted(k for k in CLOUD_FALLBACK if k not in live)
    cloud_live = [f"{k}: {v} 个 cron 还开着" for k, v in live.items()
                  if k in unexpected]

    return {
        "bj": bj.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": "一二三四五六日"[bj.weekday()],
        "weekend": weekend,
        "lines": lines,
        "sync": sync,
        "tasks": tasks,
        "cloud_cron_live": cloud_live,
        "rules": rules_status(lines),
        "cloud_fallback": {
            "on": [f"{k}：{CLOUD_FALLBACK[k]}" for k in CLOUD_FALLBACK
                   if k in live],
            "missing": [f"{k}：{CLOUD_FALLBACK[k]}" for k in missing],
            "unexpected": unexpected,
        },
        "model": breakout_model(),
        "morning_perf": morning_perf(),
        "generated": dt.datetime.now().strftime("%H:%M:%S"),
    }
