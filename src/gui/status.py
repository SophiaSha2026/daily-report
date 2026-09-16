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
import os
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
    #
    # sent 字段回答的是另一个问题：「跑完了」不等于「发出去了」。
    # 竞价线 09:25:45 采样完就写 run_meta，enrich 在那之后才发信，而它有四条
    # 「没发信也 return 0」的分支（教训 27）；起涨预测的 run_meta 在 scan 阶段
    # 就落盘，send 是下一个子进程。两条线都可能「run_meta 是今天、邮件没出去」，
    # 以前界面对这种日子一律绿灯，用户看不出和正常日子的区别。
    #   ("json", 路径)   文件里的 date 等于目标日才算发过
    #   ("exists", 模板) 文件存在就算发过，{d} 填目标日
    # 不发信的线（参数自学）留 None，界面上不显示这一格。
    {"key": "morning", "name": "早盘选股", "system": "早盘系统",
     "meta": "out/run_meta.json", "panel": "out/panel.html",
     "due": "09:27:30", "task": "DailyReport-Local-Morning",
     "sent": ("json", "out/mail_sent.json")},
    {"key": "learn", "name": "参数自学", "system": "早盘系统",
     "meta": "state/learning_status.json", "panel": "out_learn/learn.html",
     "due": "收盘后", "task": "DailyReport-Local-Learn", "sent": None},
    {"key": "breakout", "name": "起涨预测", "system": "晚间系统",
     "meta": "out_breakout/run_meta.json", "panel": "out_breakout/panel.html",
     "due": "17:00 后，最晚次日 08:30", "task": "DailyReport-Local-Evening",
     "sent": ("exists", "state/sent/breakout_{d}.json")},
    {"key": "evening", "name": "回调形态", "system": "辅助工具",
     "meta": "out_pullback/run_meta.json", "panel": "out_pullback/panel.html",
     "due": "已停用自动", "task": None,
     "sent": ("exists", "state/sent/pullback_{d}.json")},
]

# 和 local_run.SENT_REQUIRED 是同一张表的两份副本（理由同 LINES：控制台
# 不该因为 local_run 的依赖起不来）。漂了的话总览页和计划任务对「跑完了」
# 的判断会分家：一边绿灯一边还在重跑。selftest_gui 钉住两边相等。
SENT_REQUIRED = {"breakout"}

# 这几个字段是 fail-open 的分支留下的错误对象：学习线的面板/影子段挂掉时
# eval_daily 把异常写进 learning_status.json 就继续往下走（研究性步骤不该
# 阻断业务），于是退出码 0、run_meta 日期也对，总览页一路绿灯 —— 而
# learn.html 其实是上一天的。失败必须有一个能被界面查询的对象（教训 16）。
ERR_KEYS = ("panel_error", "shadow_error", "error")


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
        # GIT_OPTIONAL_LOCKS=0 是 git 给后台轮询准备的开关。总览页每 5 秒跑一次
        # `git status --porcelain`，而 status 每次都会创建 .git/index.lock
        # （3000 文件的仓库持锁约 20ms，实测 20/20 次命中；带上这个环境变量后
        # 0/20）。流程那边的 `git add` 撞上就 rc=128，产物推不上去。
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=15,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
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
        # 年份的过滤只能放在这一侧：格式串里没有年，出了 PowerShell 就再也
        # 分不清「1999-11-30 的哨兵」和「真的 11 月 30 日跑过」。
        "    last=if($i.LastRunTime -and $i.LastRunTime.Year -ge 2000)"
        "{$i.LastRunTime.ToString('MM-dd HH:mm')}else{''};"
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
            #   LastRunTime   = 1999-11-30 00:00（本机 208 个任务实测 80 个，
            #                   和 rc 哨兵 80/80 完全重合）
            #   LastTaskResult= 267011 = 0x41303 SCHED_S_TASK_HAS_NOT_RUN
            # 照原样显示就成了「11-30 00:00 / 错误码 267011」，看着像出过错。
            #
            # 只能按 rc 判，不能按 last 的 "11-30" 前缀判：上面的格式串里没有
            # 年份，真的 11 月 30 日跑过（"11-30 18:07"）也是这个前缀，
            # 会被一起清空，排期页从当天一直到 12-01 都显示「没跑过却成功」。
            if d.get("rc") == 267011:
                d["rc"] = None
                d["never"] = True
                d["last"] = ""
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
    # index.lock 也算卡住：被 taskkill /F 打断的 git 会把它留下，之后每一次
    # add/commit 都 rc=128，而流程日志只会说「没有需要提交的产物」。
    stuck = [d for d in ("rebase-merge", "rebase-apply", "MERGE_HEAD",
                         "CHERRY_PICK_HEAD", "index.lock")
             if (ROOT / ".git" / d).exists()]

    problems = []
    if push and not push.get("ok"):
        problems.append(f"上次推送失败：{(push.get('detail') or '')[:120]}")
    if ahead.isdigit() and int(ahead) > 0:
        problems.append(f"本地有 {ahead} 个 commit 没推上去")
    if behind.isdigit() and int(behind) > 0:
        problems.append(f"远端有 {behind} 个 commit 没拉下来")
    if stuck:
        problems.append(f"仓库卡在 {'/'.join(stuck)} 中间状态，"
                        f"下次推送/同步会自动清理（也可以点「重试推送」）")
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
        if time.time() - _td_cache["at"] <= 600:
            return _td_cache["s"]
        prev = _td_cache["s"]
        f = ROOT / "state" / "trade_dates.json"
        s: set = set()
        keep = False
        try:
            # 不用 f.exists() 开路：Path.exists() 会把 stat 的 OSError 吞成
            # False，而写方是 tmp + os.replace（datasource.trade_dates），
            # Windows 上换名那一瞬间读方拿到的正是 OSError。于是「文件其实在」
            # 被当成「文件没有」，空集压 10 分钟，这 10 分钟里控制台按
            # 「周一到周五」算目标日，节假日前后就和调度器对不上。
            raw = f.read_text(encoding="utf-8")
            try:
                fresh = time.time() - f.stat().st_mtime < 14 * 86400
            except OSError:
                fresh = True        # 内容都读出来了，问不出修改时刻就当它新
            if fresh:
                s = set(json.loads(raw))
        except FileNotFoundError:
            pass                    # 真没有这个文件：合法降级，不是故障
        except Exception:  # noqa: BLE001
            keep = True             # 读到半截 JSON / 读不出来：别盖掉好结果
        if s:
            _td_cache.update(at=time.time(), s=s)
        else:
            # 空集只压 60 秒（读一个 123KB 的文件几乎不要钱），不是 10 分钟。
            _td_cache.update(at=time.time() - 540, s=prev if keep else set())
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


def _sent_ok(ln: dict, d: str) -> bool | None:
    """目标日这条线的邮件真发出去了没有。不发信的线返回 None。

    证据只认「真交给 SMTP 之后才落的那个文件」：
      早盘     out/mail_sent.json（run_auction 在 send_report 之后写）
      起涨预测 state/sent/breakout_<目标日>.json（local_run.push_marker 写，
               试跑的 dry 分支不写，所以它同时也是「不是试跑」的证据）
    run_meta 一律更早落盘，拿它当发信证据就是教训 27 那条「退出码 0 不等于
    做了事」的界面版。
    """
    spec = ln.get("sent")
    if not spec:
        return None
    kind, rel = spec
    if kind == "json":
        return _json(rel).get("date") == d
    return (ROOT / rel.format(d=d)).exists()


def line_status(tasks: dict) -> list[dict]:
    out = []
    L = _local_run()
    for ln in LINES:
        meta = _json(ln["meta"])
        d = target_date(ln["key"])
        sent = _sent_ok(ln, d)
        # 「跑完了」必须和计划任务用同一个函数判：以前这里只比日期，
        # 真跑完之后再点一次试跑，run_meta 被标成 dry，控制台说「已跑完」
        # 而计划任务下一次敲门就整条重跑重发。目标日由这里算好传进去
        # （local_run.target_date 会 import akshare，教训 19）。
        dry = bool(meta.get("dry")) and meta.get("date") == d
        done = meta.get("date") == d and not meta.get("dry")
        # local_run 起不来时的兜底也要带上 sent 那道门，否则「送信挂了」
        # 的那天两条路一条红一条绿，而界面显示的是哪一条全看 import 成没成。
        if done and ln["key"] in SENT_REQUIRED:
            done = bool(sent)
        if L:
            try:
                done = bool(L.done_for(ln["key"], d))
            except Exception:  # noqa: BLE001
                pass
        err = next((f"{k}: {meta[k]}"[:160] for k in ERR_KEYS if meta.get(k)), "")
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
            "done": done, "dry": dry, "sent": sent, "error": err,
            "date": meta.get("date", ""), "target": d,
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


def breakout_perf() -> dict:
    """起涨预测邮件里印的成绩，读它的来源 out_breakout/window_grid.json（W5 行），
    和 export.STREAK_PERF 同源，界面上不再手抄数字（2026-09-16 审计 S13）。"""
    f = ROOT / "out_breakout" / "window_grid.json"
    try:
        g = json.loads(f.read_text(encoding="utf-8"))
        rows = {}
        for r in g.get("grid", []):
            if r.get("kind") == "W5" and "连续≥" in r.get("label", ""):
                k = int(r["label"].split("连续≥")[1][0])
                rows[k] = r
        return {"exists": bool(rows), "hit_all": round(100 * rows[1]["hit"], 1),
                "hit2": round(100 * rows[2]["hit"], 1), "n_all": rows[1]["n"],
                "n2": rows[2]["n"], "base": round(100 * float(g.get("base", 0)), 2),
                "days": g.get("days")}
    except Exception:  # noqa: BLE001
        return {"exists": False}


def council_status() -> dict:
    """最近一次学习会诊 + 台账里等批准的提案数。读 state/council/，不 import 它。"""
    d = ROOT / "state" / "council"
    out = {"exists": False, "pending": 0}
    try:
        lp = d / "latest.json"
        if lp.exists():
            j = json.loads(lp.read_text(encoding="utf-8"))
            nr = j.get("verdict") or {}
            out.update({"exists": True, "date": j.get("date"), "ok": bool(j.get("ok")),
                        "error": j.get("error", ""), "verdict": nr.get("verdict", ""),
                        "p_real": nr.get("p_real"), "n_proposals": j.get("n_proposals", 0),
                        "seconds": j.get("seconds"), "cost_usd": j.get("cost_usd")})
        dp = d / "decisions.json"
        if dp.exists():
            dec = json.loads(dp.read_text(encoding="utf-8"))
            out["pending"] = sum(1 for v in dec.values() if v.get("status") == "passed")
            out["needs_human"] = sum(1 for v in dec.values() if v.get("status") == "needs_human")
    except Exception:  # noqa: BLE001
        pass
    return out


def morning_perf() -> dict:
    """早盘选股的实测成绩，从参数自学的统计里读。

    口径要说准：learn/dataset.py 里的 y 是**按天中心化**的收益，
    所以 hit_rate 是「跑赢当天大盘的比例」（随机基准 50%），
    top_excess 是「平均超额收益」（随机基准 0%）。
    说成「上涨概率」就错了。
    """
    st = _json("state/learning_status.json")
    m = st.get("metrics") or {}
    if not m:
        return {"exists": False}
    # metrics 是在**回填表**（代理特征）上回放的成绩，不是实盘；
    # 真值在 daily 里（每天真实榜单的前 10 超额），两者分开报。
    daily = [d for d in (st.get("daily") or []) if isinstance(d, dict)]
    ex = [float(d["top_excess"]) for d in daily
          if d.get("top_excess") is not None]
    online = {"days": len(ex),
              "excess": sum(ex) / len(ex) if ex else None,
              "hit": (sum(1 for x in ex if x > 0) / len(ex)) if ex else None}
    return {"exists": True, "days": m.get("days"),
            "hit": m.get("hit_rate"), "excess": m.get("top_excess"),
            "online": online}


# 触发规则，给排期页显式列出来。时刻一律写北京时间：本机计划任务的触发器是
# 按 UTC 锚定的（tools/setup_tasks.ps1 写死 22:00Z / 08:30Z / 08:40Z，
# StartBoundary 带偏移就不跟夏令时走），所以北京时刻是固定的，漂的是美东墙钟。
# 开跑窗口也是北京时间（local_run.FLOWS 判的就是北京时间）。
# 规则本身在 src/local_run.py（本机）、tools/yield_check.py 和
# tools/evening_check.py（云端）里，这里只是把它们用人话写出来。
RULES = {
    "morning": {
        "local_when": "手动：北京 06:00~08:30 随时点（美东夏令时 18:00~20:30、冬令时 17:00~19:30）。"
                      "自动：计划任务北京 06:00 起每 15 分钟敲（22:00Z 锚定，不随夏令时漂），"
                      "08:30 起没手点就跑",
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
                      "自动：计划任务北京 16:30 起每 30 分钟敲（08:30Z 锚定），16:30 起没手点就跑；"
                      "机器睡着就等醒了补，登录时也敲一次",
        "target_rule": "最近一个已收盘（15:05 后）的交易日。北京 09-15 早上补跑出的是 09-14 的清单",
        "steps": "同步仓库 -> 推 claim -> 腾讯快照追加目标日日线（追加不到就不出清单）-> "
                 "重算特征（约 13 分钟）-> 打分、风险剔除 -> 面板 + 发信 -> 推 sent + 产物",
        "cloud": "evening_check.yml：cron 20:30 北京 + Worker 20:45 派发。云端算不了这条线"
                 "（特征表 2.5GB 在本机，新浪源云端不通），只看 origin/main 有没有目标日的清单，"
                 "没有就发一封「本机没跑」提醒，同一天只发一次。",
    },
    "learn": {
        "local_when": "自动：计划任务北京 16:40 起每 30 分钟敲（08:40Z 锚定），16:40 起跑；手动随时",
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

# --if-needed 的几道检查，顺序就是 local_run.if_needed_skip 里的顺序
IF_NEEDED = [
    "北京时间周末：跳过（只挡目标日是今天的线；起涨预测/参数自学窗口跨午夜，"
    "北京周六凌晨正是补周五清单的时段，照跑）",
    "早盘选股遇非交易日：跳过（日历拿不到时不挡）",
    "这条线正在跑（state/lock 或进程表）：跳过",
    "目标日已经跑完（run_meta 日期 == 目标日；被试跑覆盖时认 sent 标记）：跳过",
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

MANUAL_RULE = ("控制台按钮和计划任务走同一个入口、同一把锁，谁先起谁跑。手点不看开跑窗口、"
               "不看自动时刻，只看两样：已经在跑（锁）就直接退出；目标日已经发过信也直接退出"
               "（早盘看 out/mail_sent.json，起涨预测看 state/sent/breakout_<目标日>.json，"
               "确要重发就删掉它）。带「会发邮件」的按钮点了就真的发，点前会弹确认。")


def _local_run():
    try:
        import sys
        # 总览页每 5 秒调好几次，无条件 insert 会让 sys.path 一直长下去
        p = str(ROOT / "src")
        if p not in sys.path:
            sys.path.insert(0, p)
        import local_run
        return local_run
    except Exception:  # noqa: BLE001
        return None


def verdict(key: str, done: bool, target: str,
            dry: bool = False, sent: bool | None = None) -> str:
    """现在这一刻触发这条线，会发生什么。逐条复述 local_run.main 的判断。

    dry/sent 不参与 local_run 的跑/跳判断，但排期页上必须说出来：
    只试跑过的那一天，计划任务到点仍会真跑并发信；run_meta 是今天而没有
    发信记录的那一天，计划任务反而会跳过（已跑完），邮件其实没出去。
    """
    L = _local_run()
    now = now_bj()
    if key == "evening":
        return "自动已停用，只有手动"
    # 周末拦截只管目标日是「今天」的线。起涨预测和学习线的窗口跨午夜，
    # 北京周六 00:00~08:30 正是补周五清单的时段，不能一刀切说「周末不跑」。
    weekend = (L.weekend_skip(key) if L and key in getattr(L, "FLOWS", {})
               else key == "morning" and now.weekday() >= 5)
    if weekend:
        return "北京周末，这条线的目标日是今天，触发了也跳过"
    tds = _trade_dates()
    if key == "morning" and tds and now.strftime("%Y-%m-%d") not in tds:
        return "非交易日，触发了也跳过"
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
        s = f"目标日 {target} 已跑完，再触发直接退出"
        if sent is False:
            s += ("；但没有发信记录，这一天的邮件不是本机发的"
                  "（早盘看 out/mail_sent.json，起涨预测看 state/sent/）")
        return s
    if dry:
        return (f"目标日 {target} 只试跑过，不算跑完："
                f"计划任务到点会真跑并发信")
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
            "done": ln.get("done"), "dry": ln.get("dry"), "sent": ln.get("sent"),
            "verdict": verdict(ln["key"], bool(ln.get("done")),
                               ln.get("target", ""),
                               dry=bool(ln.get("dry")), sent=ln.get("sent")),
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
        "breakout_perf": breakout_perf(),
        "council": council_status(),
        "generated": dt.datetime.now().strftime("%H:%M:%S"),
    }
