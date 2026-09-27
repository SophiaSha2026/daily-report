"""
控制台离线自测。和另外两条一样：不联网、不碰 state/、十几秒跑完。

钉的是**接线**，不是界面好不好看
--------------------------------
这个项目栽过两次接线的跟头（历史教训 11 的闸门传参、13 的邮件传参），
共同点是「代码跑得通、输出看着合理、其实那条线根本没接上」。
控制台有三处同样的风险：

  1. ACTIONS 里每个按钮指向一个脚本路径。脚本改名之后按钮不会报错，
     点下去才发现是死的。
  2. status.LINES 和 local_run.FLOWS 各自写了一份「哪条线读哪个
     run_meta」。两边漂了，总览页会长期显示「未完成」而流程其实跑完了。
  3. 前端 tab 的 data-p 和后端 PANELS 的键必须对上，否则点了出 404。

另外把两道防护（Host 白名单、写操作 token）也钉住：它们失效是静默的，
看界面完全正常。
"""
from __future__ import annotations

import ast
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

fails: list[str] = []


def ck(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ✓ {msg}")
    else:
        fails.append(msg)
        print(f"  ✗ {msg}")


# ---------------------------------------------------------------------
#  AST 小工具：这里钉的是「调用方有没有接对线」，不是输出好不好看。
#  手法和 selftest_learn.check_wiring 一样（教训 11：闸门那次就是调用方
#  传错了参数，代码跑得通、数字看着合理，四次点火都没人发现）。
# ---------------------------------------------------------------------
def _tree(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _parents(tree: ast.AST) -> dict:
    m = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            m[ch] = node
    return m


def _func(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _callname(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _consts(node: ast.Call) -> list:
    return [a.value for a in node.args if isinstance(a, ast.Constant)]


def _calls(node: ast.AST, name: str) -> list:
    return [c for c in ast.walk(node)
            if isinstance(c, ast.Call) and _callname(c) == name]


def _under_git_lock(node: ast.AST, parents: dict) -> bool:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.With):
            for it in cur.items:
                if (isinstance(it.context_expr, ast.Call)
                        and _callname(it.context_expr) == "git_lock"):
                    return True
        cur = parents.get(cur)
    return False


class _FakeTime:
    """把 time.sleep 变成空操作，重试逻辑就不用真的等（自测要一秒内跑完）。"""

    def __init__(self) -> None:
        import time as _t
        self.time = _t.time
        self.slept: list[float] = []

    def sleep(self, s: float) -> None:
        self.slept.append(s)


def check_actions() -> None:
    print("\n[按钮接线]")
    from gui.jobs import ACTIONS, CONFLICTS, _flow_of, Registry
    from gui import status as st
    for key, a in ACTIONS.items():
        target = ROOT / a["cmd"][0]
        ck(target.exists(), f"{key} -> {a['cmd'][0]} 存在")
    ck(all(a.get("desc") for a in ACTIONS.values()),
       "每个动作都有说明文字（界面靠它告诉人这一下会发生什么）")
    # 会发信的动作清单。加新的必须同时改这里 —— 这条断言的意义就是
    # 强迫「又多了一个会往外发邮件的按钮」这件事被人看见一次。
    mail_keys = {k for k, a in ACTIONS.items() if a.get("mail")}
    ck(mail_keys == {"breakout", "pullback"},
       f"会发信的正好是起涨预测 / 长期调整突破（实际 {sorted(mail_keys)}）")
    ck(all(ACTIONS[k]["danger"] for k in mail_keys),
       "会发信的动作都标了 danger（按钮是红边的）")
    # 早盘系统 2026-09-27 归档：它的按钮不许留下（点了就是死路径或者复活一条已停的线）
    gone = [k for k, a in ACTIONS.items()
            if any(x in " ".join(a["cmd"]) for x in
                   ("morning", "--flow learn", "--flow evening", "eval_daily",
                    "premarket", "run_auction", "refresh_sector", "selftest.py",
                    "selftest_learn", "selftest_train"))]
    ck(not gone, f"没有指向已归档早盘系统的按钮（{gone}）")
    # 说明文字里承诺「直接退出」的，代码里必须真有那道门（手动入口不带
    # --if-needed，跑过照样重跑重发）
    lr = _tree("src/local_run.py")
    for key, a in ACTIONS.items():
        if "直接退出" not in a.get("desc", ""):
            continue
        flow = _flow_of(key)
        if not flow or "--if-needed" in a["cmd"]:
            continue
        fn = _func(lr, f"flow_{flow}")
        guards = (_calls(fn, "done_for") + _calls(fn, "already_done")) if fn else []
        ck(bool(guards), f"{key} 的说明承诺「直接退出」，flow_{flow} 里真有那道门")
    ck("发过信" in st.MANUAL_RULE,
       "排期页的手动规则写明「目标日已发信就直接退出」")

    # 「补数据」必须是每日增量。--stage sina 是首次回填的断点续传：done 里的票
    # 一根都不拉（2026-09-16 实测 5515/5548 已 done），点了退出码 0、日志
    # 「日线合并完成」，一根新 K 线都没有 —— 按钮对它自己承诺的用途是空操作。
    bk = ACTIONS["bk_backfill"]["cmd"]
    ck(bk[1:] == ["--stage", "update"],
       f"「补数据」走每日增量 --stage update（实际 {bk[1:]}）")
    ck(ACTIONS["bk_refresh"]["cmd"][1:] == ["--stage", "refresh"],
       "另有「全量回填」按钮走 --stage refresh（首次回填从这儿走）")
    sina = [k for k, a in ACTIONS.items()
            if a["cmd"][0].endswith("backfill.py") and "sina" in a["cmd"]]
    ck(not sina, f"界面上没有任何按钮直接调 --stage sina（实际 {sina}）")
    # 和计划任务/两条线走的那条对齐：两处漂了，手点和自动补出来的数据就不一样
    import re as _re
    lr_src = (ROOT / "src" / "local_run.py").read_text(encoding="utf-8")
    ms = list(_re.finditer(r'py\("src/breakout/backfill\.py",\s*"--stage",\s*"(\w+)"'
                           r'(?P<rest>[^)]*)\)', lr_src))
    ck(len(ms) == 2 and all(["--stage", m.group(1)] == bk[1:] for m in ms),
       f"local_run 里两处补数据（起涨预测 / 长期调整突破）和按钮同一个 stage"
       f"（{[m.group(1) for m in ms]}）")
    # 目标日只该算一次：backfill 自己算的话，跨午夜补跑那一段（北京
    # 00:00~08:30）它按「今天」、编排层按「最近已收盘」，两边分叉（S20）
    ck(bool(ms) and all('"--target"' in m.group("rest") for m in ms),
       "local_run 把算好的目标日 --target 传给补数据那一步")
    ck(all(set(CONFLICTS.get(k, ())) == {"breakout", "pullback"}
           for k in ("bk_backfill", "bk_refresh")),
       "两个回填按钮都映射到两条晚间线的锁（它们和两条线抢 daily.parquet）")
    # 同一条线的两个按钮（正式 / 试跑）命令行一样，落到同一把进程锁上。
    # 这里只验「被拒」那一条路：start 一旦放行就会真的起子进程。
    ck(_flow_of("breakout") == _flow_of("breakout_dry") == "breakout"
       and _flow_of("pullback") == _flow_of("pullback_dry") == "pullback"
       and _flow_of("selftest_pullback") == "", "_flow_of 认得出按钮起的是哪条线")
    reg = Registry()

    class _Stub:
        key, running, id = "pullback_dry", True, "j0"

    reg._jobs["j0"] = _Stub()
    j, err = reg.start("pullback")
    ck(j is None and "同一条线" in err,
       "「长期调整突破」和它的试跑不能同时起（两个实例会互相看见、双双退出 0）")


def check_flow_tables() -> None:
    """status.LINES 和 local_run.FLOWS 是同一件事的两份副本，必须一致。"""
    print("\n[流程表一致性]")
    from gui import status
    import local_run
    gui_meta = {ln["key"]: ln["meta"] for ln in status.LINES}
    run_meta = {k: v[0] for k, v in local_run.FLOWS.items()}
    ck(set(gui_meta) == set(run_meta),
       f"两边的流程键一致（gui={sorted(gui_meta)} run={sorted(run_meta)}）")
    for k in sorted(set(gui_meta) & set(run_meta)):
        ck(gui_meta[k] == run_meta[k],
           f"{k} 的完成标记文件一致：{gui_meta[k]}")
    # 「跑完了要不要连 sent 标记一起看」也是同一张表的两份副本：漂了的话
    # 控制台绿灯而计划任务还在重跑（或者反过来），两边都不报错
    ck(status.SENT_REQUIRED == local_run.SENT_REQUIRED,
       f"SENT_REQUIRED 两边一致（gui={sorted(status.SENT_REQUIRED)} "
       f"run={sorted(local_run.SENT_REQUIRED)}）")
    ck(status.SENT_REQUIRED <= set(local_run.SENT_MARK),
       "SENT_REQUIRED 里的线都有 sent 标记可查（否则它永远「没跑完」）")


def check_windows() -> None:
    """--if-needed 的开跑窗口。上界写错会在盘中跑出一份脏清单。"""
    print("\n[开跑窗口]")
    import datetime as dt
    import local_run
    orig = local_run.now_bj
    # 两条线的窗口都跨午夜：16:00 到次日 08:30。目标日是最近一个
    # 已收盘交易日，所以北京 07:00 补跑出的仍是前一天的清单，不会错日。
    cases = [
        ("breakout", 15, 59, False), ("breakout", 16, 0, True),
        ("breakout", 23, 59, True), ("breakout", 7, 0, True),
        ("breakout", 8, 30, True), ("breakout", 8, 31, False),
        ("breakout", 12, 0, False),
        ("pullback", 15, 59, False), ("pullback", 16, 0, True),
        ("pullback", 17, 58, True), ("pullback", 0, 0, True),
        ("pullback", 8, 30, True), ("pullback", 8, 31, False),
        ("pullback", 12, 0, False),
    ]
    # 自动开跑时刻：在这之前计划任务只等手动。跨午夜的线午夜后也算到点（补跑）。
    # 自动时刻前 AUTO_GRACE_MIN（2）分钟内也算到点：2026-09-18 计划任务
    # 08:29:56 敲门、早 4 秒，按整分钟判就跳过了，晚起 15 分钟。
    auto_cases = [
        ("breakout", 16, 27, False), ("breakout", 16, 29, True), ("breakout", 16, 30, True),
        ("breakout", 2, 0, True), ("breakout", 8, 30, True), ("breakout", 8, 31, False),
        ("pullback", 16, 30, False), ("pullback", 17, 37, False),
        ("pullback", 17, 38, True), ("pullback", 17, 40, True),
        ("pullback", 2, 0, True), ("pullback", 8, 31, False),
    ]
    try:
        for flow, h, m, want in cases:
            local_run.now_bj = lambda h=h, m=m: dt.datetime(2026, 9, 14, h, m)
            got = local_run.in_window(flow)
            ck(got == want, f"{flow} {h:02d}:{m:02d} -> {'可跑' if want else '跳过'}")
        for flow, h, m, want in auto_cases:
            local_run.now_bj = lambda h=h, m=m: dt.datetime(2026, 9, 14, h, m)
            got = local_run.auto_due(flow)
            ck(got == want, f"{flow} {h:02d}:{m:02d} 自动 -> {'到点' if want else '等手动'}")
        for flow, v in local_run.FLOWS.items():
            ck(len(v) == 5, f"{flow} 的 FLOWS 有五项（含自动开跑时刻）")
        # 自动开跑要早于 17:58 发信：晚于它的话每天都是「补发」
        a = local_run.FLOWS["pullback"][4]
        ck(a < (17, 58), f"长期调整突破的自动开跑时刻 {a[0]:02d}:{a[1]:02d} 早于 17:58")
        # 周末拦截只管目标日是「今天」的线。两条线的窗口都跨午夜：北京周六
        # 07:00 = 美东周五中午前，计划任务正在那一段里补周五的清单，一刀切
        # 挡掉的话，周五收盘后机器没醒过的那一周，周五的清单永远补不出来。
        wk = [("breakout", 19, 7, 0, False), ("pullback", 19, 7, 0, False),
              ("breakout", 20, 7, 0, False), ("pullback", 14, 7, 0, False)]
        for flow, day, h, m, want in wk:
            local_run.now_bj = lambda d=day, h=h, m=m: dt.datetime(2026, 9, d, h, m)
            ck(local_run.weekend_skip(flow) == want,
               f"{flow} 09-{day} {h:02d}:{m:02d} 周末拦截 -> {want}")
        local_run.now_bj = lambda: dt.datetime(2026, 9, 19, 10, 0)
        ck(not local_run.weekend_skip("pullback")
           and not local_run.in_window("pullback"),
           "pullback 周六 10:00：拦截不管，但窗口已关，还是不跑")
    finally:
        local_run.now_bj = orig


def check_task_schedule() -> None:
    """计划任务什么时候敲 vs local_run 什么时候肯跑。两份代码，漂了不报错。

    2026-09-16 只读导出已注册任务，StartBoundary 全是
    `2026-09-12T18:00:00-04:00` 这种带偏移的形式：New-ScheduledTaskTrigger -At
    把**注册那一刻**的 UTC 偏移烘了进去。微软 ITrigger::put_StartBoundary
    写明「When an offset is specified ... the time and offset are always used
    regardless of the ... daylight saving settings」，所以触发器是绝对时刻，
    不跟夏令时走 —— 漂的是美东墙钟，北京时刻由「哪个季节重跑了脚本」决定。
    setup_tasks.ps1 因此改成直接写 UTC，这里按 UTC 复算北京敲击序列。

    钉的是这条线：ps1 里的时刻改了，而 FLOWS 的窗口/自动开跑时刻没跟着改，
    后果是整段敲击全落在窗口外（只拉远端），没有任何报错，
    check_windows 只看 FLOWS 自己，看不见这种漂。
    """
    print("\n[计划任务排期 vs 开跑窗口]")
    import datetime as dt
    import re
    import local_run
    from gui.status import RULES

    src = (ROOT / "tools" / "setup_tasks.ps1").read_text(encoding="utf-8-sig")
    blocks = list(re.finditer(
        r"(?P<name>Evening|Pullback|Sync)\s*=\s*@\{.*?Trigger\s*=\s*"
        r"(?:Weekly\s*@\((?P<days>[^)]*)\)|Daily)\s*"
        r'"(?P<at>\d{2}:\d{2})"\s*"(?P<every>[^"]+)"\s*"(?P<dur>[^"]+)"', src, re.S))
    got = {m.group("name"): m for m in blocks}
    ck(len(got) == 3, f"setup_tasks.ps1 里解析到三个任务（实际 {sorted(got)}）")
    ck(not re.search(r"^\s*(Morning|Learn)\s*=\s*@\{", src, re.M),
       "早盘系统的 Morning / Learn 两个任务定义已经删掉（2026-09-27 归档）")
    if len(got) != 3:
        return

    def iso_min(s: str) -> int:
        """PT15M / PT3H15M / PT16H / P1D -> 分钟"""
        m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", s)
        d, h, mi = (int(x or 0) for x in m.groups())
        return (d * 24 + h) * 60 + mi

    def ticks_bj(at_utc: str, every: int, dur: int) -> list[int]:
        """UTC 时刻 + 重复 -> 北京时间的敲击序列（分钟数）。
        末点不敲：Windows 的重复是 start + k*interval，k*interval < duration。"""
        h, m = (int(x) for x in at_utc.split(":"))
        start = (h * 60 + m + 8 * 60) % 1440          # UTC + 8 = 北京
        n = -(-dur // every)
        return [(start + k * every) % 1440 for k in range(n)]

    def hm(x: int) -> str:
        return "%02d:%02d" % divmod(x, 60)

    orig = local_run.now_bj

    def probe(flow: str, seq: list[int]) -> tuple[int, list[int]]:
        """返回（落在窗口内的次数, 到点可自动开跑的敲击）"""
        inw, auto = 0, []
        for t in seq:
            h, m = divmod(t, 60)
            local_run.now_bj = lambda h=h, m=m: dt.datetime(2026, 9, 14, h, m)
            if local_run.in_window(flow):
                inw += 1
            if local_run.auto_due(flow):
                auto.append(t)
        return inw, auto

    try:
        for name, flow in (("Evening", "breakout"), ("Pullback", "pullback")):
            m = got[name]
            every, dur = iso_min(m.group("every")), iso_min(m.group("dur"))
            seq = ticks_bj(m.group("at"), every, dur)
            inw, auto = probe(flow, seq)
            ck(inw >= 0.9 * len(seq),
               f"{name} 北京 {hm(seq[0])} 起敲 {len(seq)} 次，{inw} 次落在"
               f"{flow} 的开跑窗口内（要 ≥90%）")
            ck(len(auto) >= 1,
               f"{name} 至少敲中一次自动开跑时刻（实际 {len(auto)} 次）")
            if auto:
                want = local_run.FLOWS[flow][4]
                delay = (auto[0] - (want[0] * 60 + want[1])) % 1440
                # 首个到点敲击要是晚过一个敲击间隔，说明排期整体漂了
                # （冬天按本机时刻重跑脚本就是晚 1 小时）
                ck(delay <= every,
                   f"{name} 首个到点敲击 {hm(auto[0])} 离约定的 "
                   f"{'%02d:%02d' % want} 不超过一个间隔（实际 {delay} 分）")
            days = [d.strip().strip('"') for d in (m.group("days") or "").split(",")]
            # 08:30Z / 09:40Z 在美东是同一天凌晨，北京日期和美东日期同一天
            want_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            ck(days == want_days, f"{name} 的 DaysOfWeek 是 {want_days[0]}~{want_days[-1]}")
        # 长期调整突破 17:58 要发信：17:58 之前至少敲中一次
        pm = got["Pullback"]
        seq = ticks_bj(pm.group("at"), iso_min(pm.group("every")), iso_min(pm.group("dur")))
        ck(any(17 * 60 <= t < 17 * 60 + 58 for t in seq),
           "长期调整突破在北京 17:00~17:58 之间至少敲一次（开跑后等到 17:58 发）")

        # 反例：有人把 Pullback 写成「北京 17:40」当本机时刻（01:40Z = 北京 09:40）：
        # 整段敲击落在窗口外，自动开跑彻底失效，只剩登录触发。这个检查要抓得住
        bad = ticks_bj("01:40", 15, 180)
        inw_bad, auto_bad = probe("pullback", bad)
        ck(not auto_bad and inw_bad == 0,
           f"反例：挪到北京 {hm(bad[0])} 起，到点敲击 {len(auto_bad)} 次、"
           f"窗口内 {inw_bad}/{len(bad)} 次 -> 这个检查抓得住")
    finally:
        local_run.now_bj = orig

    # 时刻是 UTC 写死的才成立。谁改回裸 -At，季节依赖就回来了
    ck(re.search(r'\$t\.StartBoundary\s*=', src) is not None
       and ':00Z"' in src,
       "setup_tasks.ps1 用带 Z 的 StartBoundary 锚定，不靠 -At 的本机时刻")
    ck(re.search(r"-At\s+\$at\b", src) is None,
       "Weekly/Daily 不再把时刻透传给 -At（那样会把注册季节烘进 StartBoundary）")

    # 面板上的说明要跟着 ps1 走：改了时刻不改文案，用户看到的就是错的
    ui_src = (ROOT / "src" / "gui" / "ui.py").read_text(encoding="utf-8")
    st_src = (ROOT / "src" / "gui" / "status.py").read_text(encoding="utf-8")
    for name, flow in (("Evening", "breakout"), ("Pullback", "pullback")):
        z = got[name].group("at") + "Z"
        ck(z in ui_src and z in st_src, f"{name} 的 UTC 锚点 {z} 写进了排期页说明")
        start = hm(ticks_bj(got[name].group("at"), 1, 1)[0])
        ck(start in RULES[flow]["local_when"],
           f"{name} 的北京起敲时刻 {start} 和排期页说明一致")
    # 「北京时刻跟着夏令时漂」这句话是反的（实测触发器带偏移、绝对锚定），
    # 照它去改时刻就会掉进上面那个反例。负向断言只拦肯定句，「不跟夏令时走」放行
    ck(re.search(r"(?<!不)跟夏令时走", st_src + ui_src) is None,
       "控制台不再写「跟夏令时走」（实测反了：北京时刻才是固定的）")


def check_no_akshare_in_gui() -> None:
    """控制台进程里不许出现 akshare / datasource / py_mini_racer。

    2026-09-15 实测：/api/status 两个线程同时 import akshare，V8 进程级
    FATAL，整个控制台没了（历史教训 19）。这里用 AST 钉住 import 语句，
    连函数体内的延迟 import 也算。
    """
    print("\n[控制台不碰 akshare]")
    import ast
    banned = {"akshare", "datasource", "py_mini_racer"}
    for rel in ("src/gui/status.py", "src/gui/server.py", "src/gui/jobs.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hits += [a.name for a in node.names if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in banned:
                    hits.append(node.module)
        ck(not hits, f"{rel} 不 import {sorted(banned)}" + (f"，发现 {hits}" if hits else ""))


def check_target_and_lock() -> None:
    """目标日：两条线都是最近一个已收盘交易日。进程锁：活的挡、死的不挡。"""
    print("\n[目标日 / 进程锁]")
    import datetime as dt
    import json
    import os
    import tempfile
    import local_run
    tz = dt.timezone(dt.timedelta(hours=8))
    orig_td, orig_now = local_run.trade_dates, local_run.now_bj
    # 假日历：09-14（周一）是交易日，09-12/13 周末不是，09-11 是
    local_run.trade_dates = lambda: {"2026-09-11", "2026-09-14", "2026-09-15"}
    try:
        for (mo, d, h, m), want in [
            ((9, 14, 16, 30), "2026-09-14"),   # 收盘后：当天
            ((9, 14, 14, 0), "2026-09-11"),    # 盘中：上一交易日
            ((9, 15, 7, 30), "2026-09-14"),    # 次日早上补跑：仍是前一天
            ((9, 13, 12, 0), "2026-09-11"),    # 周末：上周五
        ]:
            local_run.now_bj = lambda: dt.datetime(2026, mo, d, h, m, tzinfo=tz)
            for flow in ("breakout", "pullback"):
                got = local_run.target_date(flow)
                ck(got == want, f"{flow} 目标日 {mo:02d}-{d:02d} {h:02d}:{m:02d} -> {want}")
        # 没有日历时按周一到周五
        local_run.trade_dates = lambda: set()
        local_run.now_bj = lambda: dt.datetime(2026, 9, 13, 12, 0, tzinfo=tz)
        ck(local_run.target_date("pullback") == "2026-09-11", "日历拿不到时按工作日退化")
    finally:
        local_run.trade_dates, local_run.now_bj = orig_td, orig_now

    # 锁：写到临时目录，不碰 state/。进程表那一层换成死实现：这里测的是锁的
    # 逻辑，真机上可能正好有一条线在跑，扫进程表会把它算进来。
    orig_root, orig_scan = local_run.ROOT, local_run._scan_processes
    local_run.ROOT = Path(tempfile.mkdtemp(prefix="lock_"))
    local_run._scan_processes = lambda flow: None
    try:
        ck(local_run.running_instance("breakout") is None, "没有锁文件 -> 没在跑")
        ck(local_run.acquire_lock("breakout"), "拿锁成功")
        ck(local_run.running_instance("breakout") is None, "自己的锁不算别人在跑")
        lp = local_run.lock_path("breakout")
        lp.write_text(json.dumps({"pid": os.getpid(), "flow": "breakout",
                                  "at": local_run.now_bj().isoformat(timespec="seconds")}),
                      encoding="utf-8")
        lp2 = local_run.lock_path("pullback")
        lp2.write_text(json.dumps({"pid": 999999, "flow": "pullback",
                                   "at": local_run.now_bj().isoformat(timespec="seconds")}),
                       encoding="utf-8")
        ck(local_run.running_instance("pullback") is None, "锁里的进程死了 -> 当没锁")
        stale = (local_run.now_bj() - dt.timedelta(hours=9)).isoformat(timespec="seconds")
        lp2.write_text(json.dumps({"pid": os.getpid(), "flow": "pullback", "at": stale}),
                       encoding="utf-8")
        ck(local_run.running_instance("pullback") is None, "锁超过最长运行时间 -> 当没锁")
        local_run.release_lock("breakout")
        ck(not lp.exists(), "释放后锁文件删掉")
        local_run._scan_processes = lambda flow: {"pid": 1, "flow": flow,
                                                  "at": "", "source": "进程表"}
        ck(local_run.running_instance("breakout") is not None,
           "没有锁但进程表里有 -> 算在跑")
        # 长期调整突破要等到 17:58：手动 16:00 起跑也不能被当成死锁
        ck(local_run.MAX_RUN["pullback"] * 60 >= (17 * 60 + 58) - 16 * 60,
           "pullback 的最长运行时间盖得住「16:00 手动起跑、等到 17:58 发信」")
    finally:
        local_run.ROOT, local_run._scan_processes = orig_root, orig_scan


def check_done_definition() -> None:
    """「跑完了」只能有一个定义：控制台和计划任务必须用同一个函数。"""
    print("\n[跑完了 = 什么]")
    import tempfile
    import local_run
    from gui import status as st
    tmp = Path(tempfile.mkdtemp(prefix="done_"))
    o = (local_run.ROOT, st.ROOT, st.target_date, local_run.target_date)
    local_run.ROOT = st.ROOT = tmp
    d = "2026-09-29"
    st.target_date = lambda key: d
    local_run.target_date = lambda flow: d
    try:
        for flow, meta_rel in (("breakout", "out_breakout/run_meta.json"),
                               ("pullback", "out_pullback/run_meta.json")):
            p = tmp / meta_rel
            p.parent.mkdir(parents=True, exist_ok=True)
            sp = tmp / "state" / "sent" / f"{flow}_{d}.json"
            sp.parent.mkdir(parents=True, exist_ok=True)
            # 试跑也写 run_meta。真跑完之后再点一次试跑，以前 already_done 变回
            # False，下一个整点计划任务就把整条线重跑并再发一封清单。
            #
            # 两条线都在 SENT_REQUIRED 里：run_meta 在扫描阶段就落盘，发信是
            # 下一个子进程。只认 run_meta 的话，send 挂掉的那天总览是绿的、
            # 计划任务也判「已跑完」整夜不补（F5-2）。
            cases = [
                ({"date": d, "dry": False}, True, True, "真跑完并发了信"),
                ({"date": d, "dry": False}, False, False,
                 "run_meta 是目标日但 send 挂了（以前这里绿灯，整夜没人补）"),
                ({"date": d, "dry": True}, False, False, "只试跑过（计划任务该接管）"),
                ({"date": d, "dry": True}, True, True, "真跑完之后又试跑了一次"),
                ({"date": "2026-09-28", "dry": False}, True, False, "run_meta 是别的日子"),
                ({"date": d}, True, True, "老格式（没有 dry 字段）"),
                ({"date": d}, False, False, "老格式但没发出去"),
            ]
            for meta, sent, want, msg in cases:
                p.write_text(json.dumps(meta), encoding="utf-8")
                if sent:
                    sp.write_text("{}", encoding="utf-8")
                elif sp.exists():
                    sp.unlink()
                ck(local_run.done_for(flow, d) is want, f"{flow}：{msg} -> 跑完={want}")
                ck(local_run.already_done(flow) is want, f"{flow}：{msg}（already_done）")
                row = {r["key"]: r for r in st.line_status({})}[flow]
                ck(row["done"] is want, f"{flow}：{msg}（控制台总览和计划任务口径一致）")

            # 控制台在 local_run import 不起来时走自己那份兜底判断（L is None），
            # 它也必须带上 sent 那道门：否则同一天一条路说没跑完、一条路说跑完了，
            # 界面显示哪一条全看 import 成没成。
            p.write_text(json.dumps({"date": d, "dry": False}), encoding="utf-8")
            if sp.exists():
                sp.unlink()
            orig_L = st._local_run
            st._local_run = lambda: None
            try:
                row = {r["key"]: r for r in st.line_status({})}[flow]
                ck(row["done"] is False and row["sent"] is False,
                   f"{flow}：local_run 起不来时的兜底也要求 sent 标记")
                sp.write_text("{}", encoding="utf-8")
                row = {r["key"]: r for r in st.line_status({})}[flow]
                ck(row["done"] is True, f"{flow}：兜底路径有 sent 标记才算跑完")
            finally:
                st._local_run = orig_L
    finally:
        local_run.ROOT, st.ROOT, st.target_date, local_run.target_date = o


def check_resend_guard() -> None:
    """发过信的那一天再点一次按钮，不许再发一封。"""
    print("\n[发过信就不重发]")
    import datetime as dt
    import tempfile
    import local_run
    tmp = Path(tempfile.mkdtemp(prefix="resend_"))
    o = (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.truth_and_regime, local_run.ensure_daily)
    tz = dt.timezone(dt.timedelta(hours=8))
    d = "2026-09-29"
    ran: list = []
    local_run.ROOT = tmp
    local_run.now_bj = lambda: dt.datetime(2026, 9, 29, 18, 0, tzinfo=tz)
    local_run.target_date = lambda flow: d
    local_run.py = lambda *a: (ran.append(a) or 1)       # 真往下跑就在第一步失败早退
    local_run.sync_repo = lambda: True
    local_run.push_marker = lambda *a, **k: ran.append(("push_marker",) + a[:2])
    local_run.push_all = lambda *a, **k: ran.append(("push_all", a[0]))
    local_run.truth_and_regime = lambda dry: ran.append(("truth",))
    local_run.ensure_daily = lambda target: (False, "自测：不往下跑")
    try:
        for flow, meta_rel in (("breakout", "out_breakout/run_meta.json"),
                               ("pullback", "out_pullback/run_meta.json")):
            fn = getattr(local_run, f"flow_{flow}")
            mp = tmp / meta_rel
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps({"date": d, "dry": False}), encoding="utf-8")
            sp = tmp / "state" / "sent" / f"{flow}_{d}.json"
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text("{}", encoding="utf-8")
            ran.clear()
            # 手动入口不带 --if-needed：2026-09-14 点一下就把上一交易日的清单又发了
            # 一遍。这道门只看「目标日跑完了并且发过信」。
            ck(fn(False) == 0 and not ran,
               f"{flow} 跑完并发过信之后再点：立刻退出，不推 claim、不发第二封")
            sp.unlink()
            ran.clear()
            ck(fn(False) == 1 and ran, f"{flow} 没有 sent 标记就照常往下跑（只挡「发过信」这一种）")
            sp.write_text("{}", encoding="utf-8")
            ran.clear()
            fn(True)
            ck(bool(ran), f"{flow} 试跑不受这道门挡（它不发信，也不写 sent 标记）")
    finally:
        (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.truth_and_regime, local_run.ensure_daily) = o


def check_scan_and_lock() -> None:
    """进程表过滤（探针不算在跑）+ 原子抢锁 + 进程身份（PID 会被复用）。"""
    print("\n[进程表 / 抢锁 / 进程身份]")
    import datetime as dt
    import os
    import subprocess
    import tempfile
    import local_run
    now = dt.datetime(2026, 9, 16, 12, 0, 0, tzinfo=dt.timezone.utc)
    mine = os.getpid()
    lines = [
        f"{mine}|2026-09-16T10:00:00|python.exe src/local_run.py --flow breakout",
        "4242|2026-09-16T11:59:57|python.exe src/local_run.py --flow breakout --if-needed",
        "4243|2026-09-16T11:58:00|python.exe src/local_run.py --flow breakout --if-needed",
        "4244|2026-09-16T11:00:00|python.exe src/local_run.py --flow pullback",
    ]
    got = local_run._parse_scan(lines, "breakout", now, mine)
    ck(got is not None and got["pid"] == 4243,
       "3 秒前起的 --if-needed 探针不算在跑，2 分钟的真流程才算")
    ck(got is not None and got["at"].startswith("2026-09-16T19:58"),
       "「几点起的」按北京时间填上（以前是空串，日志打成「pid X， 起」）")
    ck(local_run._parse_scan(lines[:2], "breakout", now, mine) is None,
       "只有自己和探针 -> 判没在跑（以前双方互相看见，双双退出 0，谁都不跑）")
    other = local_run._parse_scan(lines, "pullback", now, mine)
    ck(other is not None and other["pid"] == 4244, "按 --flow 分线匹配")
    ck(local_run._parse_scan(
        ["4245|问不出来|python.exe src/local_run.py --flow breakout"],
        "breakout", now, mine) is not None,
        "创建时刻问不出来 -> 按在跑算（宁可多退一次，也别双发）")

    tmp = Path(tempfile.mkdtemp(prefix="lock2_"))
    o = (local_run.ROOT, local_run._scan_processes, local_run.running_instance)
    local_run.ROOT = tmp
    local_run._scan_processes = lambda flow: None
    child = None
    try:
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        lp = local_run.lock_path("breakout")
        lp.parent.mkdir(parents=True, exist_ok=True)
        at = local_run.now_bj().isoformat(timespec="seconds")
        lp.write_text(json.dumps({"pid": child.pid, "flow": "breakout", "at": at}),
                      encoding="utf-8")
        ck(local_run.running_instance("breakout") is not None,
           "旧格式的锁（没有 ctime）照老规矩算：pid 活着就是在跑")
        if sys.platform == "win32":
            ct = local_run._pid_ctime(child.pid)
            ck(isinstance(ct, int) and ct > 0, "问得出进程的创建时刻（进程身份）")
            lp.write_text(json.dumps({"pid": child.pid, "flow": "breakout",
                                      "at": at, "ctime": ct + 1}), encoding="utf-8")
            ck(local_run.running_instance("breakout") is None,
               "pid 活着但不是写锁时那一代（号被复用了）-> 当没锁")
            lp.write_text(json.dumps({"pid": child.pid, "flow": "breakout",
                                      "at": at, "ctime": ct}), encoding="utf-8")
            ck(local_run.running_instance("breakout") is not None,
               "同一代进程 -> 确实在跑")
        # 查和抢必须同一个口径，否则会出现「查着说没人跑、抢的时候说有人跑」
        old = (local_run.now_bj() - dt.timedelta(hours=9)).isoformat(timespec="seconds")
        lp.write_text(json.dumps({"pid": child.pid, "flow": "breakout",
                                  "at": old}), encoding="utf-8")
        ck(local_run.running_instance("breakout") is None
           and local_run.acquire_lock("breakout") is True,
           "锁老过最长运行时间：查和抢都判「可以跑」")
        lp.write_text(json.dumps({"pid": child.pid, "flow": "breakout",
                                  "at": at}), encoding="utf-8")

        # 抢锁必须原子：装成「查的时候没看见」，锁文件仍不许被覆盖
        local_run.running_instance = lambda f: None
        ck(local_run.acquire_lock("breakout") is False,
           "别人的活锁在，先查后写的漏洞不再能把它覆盖掉")
        ck(json.loads(lp.read_text(encoding="utf-8"))["pid"] == child.pid,
           "别人的锁原样保留")
        child.kill(); child.wait(); child = None
        ck(local_run.acquire_lock("breakout") is True, "持锁进程死了 -> 抢得到")
        info = json.loads(lp.read_text(encoding="utf-8"))
        ck(info["pid"] == os.getpid(), "锁换成自己的")
        ck(info.get("ctime") == local_run._pid_ctime(os.getpid()),
           "新锁带上进程身份 ctime")
        local_run.release_lock("breakout")

        # index.lock：被 taskkill /F 打断的 git 留下的，没人清就是永久静默失败
        import time as _t
        gitd = tmp / ".git"
        gitd.mkdir(parents=True, exist_ok=True)
        lk = gitd / "index.lock"
        lk.write_text("", encoding="utf-8")
        os.utime(lk, (_t.time() - 600, _t.time() - 600))
        local_run._git_unstick()
        ck(not lk.exists(), "10 分钟前留下的 index.lock 被清掉")
        lk.write_text("", encoding="utf-8")
        local_run._git_unstick()
        ck(lk.exists(), "刚生成的 index.lock 留着（可能真有 git 在写）")
        lk.unlink()

        # 控制台「终止」走 taskkill /F，被杀的 local_run 不会执行 release_lock
        from gui import jobs as J
        oj = J.ROOT
        J.ROOT = tmp

        class _P:
            pid = 424242

        job = J.Job.__new__(J.Job)
        job.key, job.proc = "pullback", _P()
        try:
            mp = tmp / "state" / "lock" / "pullback.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps({"pid": 424242, "flow": "pullback"}),
                          encoding="utf-8")
            job._drop_lock()
            ck(not mp.exists(), "控制台终止一条流程后，把它留下的进程锁清掉")
            mp.write_text(json.dumps({"pid": 1, "flow": "pullback"}), encoding="utf-8")
            job._drop_lock()
            ck(mp.exists(), "不是自己起的那把锁不动")
        finally:
            J.ROOT = oj
    finally:
        if child is not None:
            child.kill(); child.wait()
        (local_run.ROOT, local_run._scan_processes,
         local_run.running_instance) = o


def check_push_wiring() -> None:
    """add 的退出码不能吞：暂存区空的时候 diff --cached 照样 rc=0。"""
    print("\n[推送：add 失败不能报成功]")
    import tempfile
    import local_run
    tmp = Path(tempfile.mkdtemp(prefix="push_"))
    o = (local_run.ROOT, local_run._git, local_run.time)
    ft = _FakeTime()
    local_run.ROOT = tmp
    local_run.time = ft
    seq: list = []

    def mk(add_results):
        it = iter(add_results)
        def fake(*args):
            seq.append(args[0])
            if args[0] == "add":
                return next(it)
            if args[0] == "diff":
                return (1, "")          # 有东西要提交
            return (0, "")
        return fake

    LOCKED = (128, "fatal: Unable to create '.git/index.lock': File exists.")
    try:
        local_run._git = mk([LOCKED] * 4)
        ck(local_run.git_commit_push("t", ["state/sent/x.json"], False) is False,
           "add 一直撞 index.lock -> 推送算失败（以前报「没有需要提交的产物」然后 return True）")
        ck("commit" not in seq and "push" not in seq, "失败之后不会接着 commit/push")
        ps = json.loads((tmp / "state" / "push_status.json").read_text(encoding="utf-8"))
        ck(ps["ok"] is False and "index.lock" in ps["detail"] and "add" in ps["detail"],
           "失败写进 push_status（界面查得到，不是只写日志）")
        ck(len(ft.slept) == 3, "撞锁时短重试 3 次（控制台的 git status 只持锁几十毫秒）")

        seq.clear(); ft.slept.clear()
        local_run._git = mk([LOCKED, (0, "")])
        ck(local_run.git_commit_push("t", ["out"], False) is True,
           "重试之后 add 成功 -> 照常提交推送")
        ck(seq.count("push") == 1, "推了一次")
        ck(json.loads((tmp / "state" / "push_status.json")
                      .read_text(encoding="utf-8"))["ok"] is True, "记成功")

        seq.clear()
        local_run._git = mk([(1, "The following paths are ignored by one of "
                                 "your .gitignore files")])
        ck(local_run.git_commit_push("t", ["out_learn/council.html"], False) is False,
           "被 .gitignore 挡住的产物不再静默丢（council.html 从上线起一次都没上过 Pages）")
        ck(json.loads((tmp / "state" / "push_status.json")
                      .read_text(encoding="utf-8"))["ok"] is False, "记失败")

        # 可选产物今天没生成，不算失败（早盘归档前学习线的推送列表里有
        # out_learn/council.html，只在会诊真跑过的那天才有）。按失败处理的话，
        # 同一次推送里的其它产物也推不上去。
        MISS = (128, "fatal: pathspec 'out_learn/council.html' did not "
                     "match any files")
        seq.clear()
        local_run._git = mk([MISS, (0, "")])
        ck(local_run.git_commit_push(
            "t", ["out_learn/council.html", "out_learn/learn.html"], False) is True,
           "推送列表里某个可选产物今天没生成 -> 跳过它，别的照推")
        ck(seq.count("push") == 1, "确实推了一次")
        # 同样的报错但文件在：那说明 git 出别的问题了，仍然要报
        seq.clear()
        (tmp / "out_learn").mkdir(parents=True, exist_ok=True)
        (tmp / "out_learn" / "council.html").write_text("x", encoding="utf-8")
        local_run._git = mk([MISS])
        ck(local_run.git_commit_push("t", ["out_learn/council.html"], False) is False,
           "文件明明在却 add 不进去 -> 还是算失败（只放过「真的不存在」这一种）")

        # 推送循环只留一份实现：控制台「重试上传」和流程共用 push_main。
        # 以前 GUI 那条自己拼 fetch+push，本地和远端分叉时按多少次都失败。
        seq.clear()
        calls: list = []

        def fake_push(*args):
            calls.append(args)
            if args[0] == "push":
                return (1, "! [rejected] non-fast-forward")
            if args[0] == "merge":
                return (0, "Auto-merging data/2026-09")
            return (0, "")

        local_run._git = fake_push
        ok, txt = local_run.push_main("manual push [gui]")
        ck(ok is False and calls.count(("push", "-q", "origin", "main")) == 3,
           "push 被拒 -> 重试三次（每次之间先 fetch + merge）")
        ck(("merge", "--no-edit", "-q", "-X", "ours", "origin/main") in calls,
           "分叉时真的去合并远端（本地优先），不是干推三次")
        ps = json.loads((tmp / "state" / "push_status.json")
                        .read_text(encoding="utf-8"))
        ck(ps["ok"] is False and ps["detail"] and "push" in txt,
           "推不上去要写 push_status（同步卡片读它）并把命令日志给界面")
        calls.clear()
        local_run._git = lambda *a: (calls.append(a) or (0, ""))
        ok, _ = local_run.push_main("manual push [gui]")
        ck(ok is True and json.loads((tmp / "state" / "push_status.json")
                                     .read_text(encoding="utf-8"))["ok"] is True,
           "推成功也要写 push_status（否则手动推完卡片还是红的）")
    finally:
        local_run.ROOT, local_run._git, local_run.time = o

    # 可选产物那条路的注释例子（out_learn/council.html）来自早盘归档前的学习线；
    # 逻辑本身和线无关，留着钉「不存在的路径跳过、存在但加不进去的报错」。


def check_git_lock() -> None:
    """两条线并行跑，共用一个 git 工作区，写操作必须串行。"""
    print("\n[git 互斥]")
    import os
    import subprocess
    import tempfile
    import time as _t
    import local_run
    tmp = Path(tempfile.mkdtemp(prefix="glock_"))
    orig_root = local_run.ROOT
    local_run.ROOT = tmp
    f = tmp / "state" / "lock" / "git.json"
    child = None
    try:
        with local_run.git_lock(timeout=1):
            ck(f.exists() and json.loads(f.read_text(encoding="utf-8"))["pid"]
               == os.getpid(), "拿到锁并写进 state/lock/git.json")
        ck(not f.exists(), "出了 with 就把锁还回去")

        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"pid": child.pid,
                                 "at": local_run.now_bj().isoformat(timespec="seconds")}),
                     encoding="utf-8")
        t0 = _t.time()
        busy = False
        try:
            with local_run.git_lock(timeout=0.3):
                pass
        except local_run.GitBusy:
            busy = True
        ck(busy, "别人占着就等，等不到抛 GitBusy（调用方据此落 push_status）")
        ck(_t.time() - t0 >= 0.4, "等的时候是真在等，不是立刻放弃")
        ck(json.loads(f.read_text(encoding="utf-8"))["pid"] == child.pid,
           "等待期间不动别人的锁")
        child.kill(); child.wait(); child = None
        with local_run.git_lock(timeout=1):
            ck(json.loads(f.read_text(encoding="utf-8"))["pid"] == os.getpid(),
               "持锁进程死了就清掉重抢（教训 15：自愈路径本身要能自愈）")
        # 日线表的写锁和 git 锁是同一份实现（file_lock），换个文件名
        dl = tmp / "state" / "lock" / "daily_update.json"
        with local_run.data_lock(timeout=1):
            ck(dl.exists(), "日线写锁落在 state/lock/daily_update.json")
        ck(not dl.exists(), "出了 with 就还回去")
        ck(local_run.DATA_LOCK_MAX >= 2 * 3600,
           "日线写锁的「太老就当残留」长过全量刷新（约 70 分钟）")
    finally:
        if child is not None:
            child.kill(); child.wait()
        local_run.ROOT = orig_root


def check_holiday() -> None:
    """节假日（工作日但不是交易日）：两条线的目标日落回节前最后一个交易日。"""
    print("\n[节假日]")
    import datetime as dt
    import tempfile
    import local_run
    from gui import status as st
    tmp = Path(tempfile.mkdtemp(prefix="holi_"))
    o = (local_run.ROOT, local_run.now_bj, local_run.trade_dates,
         local_run.running_instance, local_run._scan_processes, dict(local_run._TD))
    tz = dt.timezone(dt.timedelta(hours=8))
    local_run.ROOT = tmp
    local_run._scan_processes = lambda flow: None
    local_run.running_instance = lambda flow: None
    try:
        # 2026-10-01（周四）国庆，不开市；节前最后一个交易日是 09-30
        local_run.now_bj = lambda: dt.datetime(2026, 10, 1, 17, 45, tzinfo=tz)
        local_run._TD.clear()
        local_run.trade_dates = lambda: {"2026-09-29", "2026-09-30", "2026-10-08"}
        ck(local_run.target_date("pullback") == "2026-09-30",
           "国庆当天 17:45：目标日是节前最后一个交易日 09-30")
        for flow, meta_rel in (("breakout", "out_breakout/run_meta.json"),
                               ("pullback", "out_pullback/run_meta.json")):
            (tmp / meta_rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp / meta_rel).write_text(json.dumps({"date": "2026-09-30"}), encoding="utf-8")
            sp = tmp / "state" / "sent" / f"{flow}_2026-09-30.json"
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text("{}", encoding="utf-8")
            ck("已经跑完" in local_run.if_needed_skip(flow),
               f"{flow}：节前那份发过了 -> 节假日敲门直接跳过，不补发不空跑")
            sp.unlink()
            ck(local_run.if_needed_skip(flow) == "",
               f"{flow}：节前那份没发出去 -> 节假日照样补（窗口内、过了自动时刻）")
        local_run.now_bj = lambda: dt.datetime(2026, 10, 1, 12, 0, tzinfo=tz)
        ck(local_run.if_needed_skip("pullback").startswith(local_run.SYNC_ON[0]),
           "12:00 不在开跑窗口（这一类要顺手拉一次远端）")
        local_run.now_bj = lambda: dt.datetime(2026, 10, 1, 16, 45, tzinfo=tz)
        ck(local_run.if_needed_skip("pullback").startswith(local_run.SYNC_ON[1]),
           "16:45 还没到长期调整突破的自动开跑时刻 17:40（等手动，顺手拉远端）")

        # 上线日之前：首页和排期页都说「还没开始」，不报「没跑」
        orig = (st.now_bj, st._trade_dates)
        st.now_bj = lambda: dt.datetime(2026, 9, 27, 18, 0)
        st._trade_dates = lambda: {"2026-09-24", "2026-09-28"}
        try:
            v = st.verdict("pullback", False, "2026-09-24")
            ck("起才发清单" in v, f"排期页：目标日早于上线日 -> 照实说不补（{v[:30]}）")
        finally:
            st.now_bj, st._trade_dates = orig
    finally:
        (local_run.ROOT, local_run.now_bj, local_run.trade_dates,
         local_run.running_instance, local_run._scan_processes) = o[:5]
        local_run._TD.clear(); local_run._TD.update(o[5])


def check_wiring_ast() -> None:
    """接线：git 写操作在锁里、拉远端只有一条路、闸门抽成了纯函数。"""
    print("\n[接线：git 与闸门]")
    lr = _tree("src/local_run.py")
    par = _parents(lr)
    main = _func(lr, "main")
    ck(main is not None, "找得到 local_run.main")
    ck(not [c for c in ast.walk(main) if isinstance(c, ast.Call)
            and _callname(c) == "weekday"],
       "main 里不再直接判 now_bj().weekday()（周末拦截要按线区分，见 weekend_skip）")
    ck(bool(_calls(main, "if_needed_skip")),
       "--if-needed 的判定走 if_needed_skip（抽成纯函数才测得了）")
    ins = _func(lr, "if_needed_skip")
    ck(bool(_calls(ins, "weekend_skip")), "if_needed_skip 调 weekend_skip")
    ck(bool(_calls(ins, "target_date")) and bool(_calls(ins, "already_done")),
       "if_needed_skip 按目标日判（节假日的目标日是节前最后一个交易日，走「已跑完」）")
    ck(not [n for n in ast.walk(lr) if isinstance(n, ast.FunctionDef)
            and n.name in ("flow_morning", "flow_evening", "flow_learn",
                           "local_commentary", "_mail_sent_today")],
       "早盘系统的三个流程和 LLM 文案 2026-09-27 起不在 local_run 里")

    write = {"pull", "push", "commit", "add", "merge", "rebase", "fetch",
             "stash", "reset", "checkout"}
    # push_main 自己不开锁：它被 git_commit_push 和控制台的「重试上传」共用，
    # 两边都已经在 with git_lock() 里。所以改成钉它的**调用方**。
    pm = _func(lr, "push_main")
    pm_body = set(ast.walk(pm)) if pm else set()
    loose = [_consts(c)[0] for c in ast.walk(lr)
             if isinstance(c, ast.Call) and _callname(c) == "_git"
             and _consts(c)[:1] and _consts(c)[0] in write
             and c not in pm_body
             and not _under_git_lock(c, par)]
    ck(not loose, f"写 git 的调用全在 with git_lock() 里（漏网 {sorted(set(loose))}）")
    pm_calls = [c for c in ast.walk(lr) if isinstance(c, ast.Call)
                and _callname(c) == "push_main"]
    ck(bool(pm_calls) and all(_under_git_lock(c, par) for c in pm_calls),
       "push_main 的每个调用方都在 git_lock 里（它自己不开锁）")
    ck(not [c for c in ast.walk(lr) if isinstance(c, ast.Call)
            and _callname(c) == "sh" and _consts(c)[:1] == ["git"]],
       "不再有绕过 _git 的 sh('git', ...)（那条路不带 --autostash，工作区一脏就静默失败）")

    pulls = [c for c in ast.walk(lr) if isinstance(c, ast.Call)
             and _callname(c) == "_git" and _consts(c)[:1] == ["pull"]]
    ck(len(pulls) == 1, f"全模块只有一处 pull（实际 {len(pulls)} 处）")
    ck(all("--autostash" in _consts(c) for c in pulls), "那一处 pull 带 --autostash")
    for fn in ("flow_breakout", "flow_pullback"):
        f = _func(lr, fn)
        ck(not [c for c in _calls(f, "_git") if _consts(c)[:1] == ["pull"]],
           f"{fn} 不自己 pull")
        ck(bool(_calls(f, "sync_repo")),
           f"{fn} 拉远端走 sync_repo（它带 unstick + autostash + 「别的流程在跑就不动工作区」）")

    sv = _tree("src/gui/server.py")
    spar = _parents(sv)
    loose = []
    for c in ast.walk(sv):
        if isinstance(c, ast.Call) and _callname(c) == "run" and c.args:
            a0 = c.args[0]
            if (isinstance(a0, ast.List) and a0.elts
                    and isinstance(a0.elts[0], ast.Constant)
                    and a0.elts[0].value == "git"
                    and not _under_git_lock(c, spar)):
                loose.append(c.lineno)
    ck(not loose, f"控制台的 git 也在锁里（漏网行号 {loose}）")
    rp = _func(sv, "_retry_push")
    ck(rp is not None and bool(_calls(rp, "_git_unstick")),
       "「重试推送」先清残留（含 index.lock），否则按几次也好不了")
    ck(rp is not None and bool(_calls(rp, "push_main"))
       and bool(_calls(rp, "any_flow_running")),
       "「重试推送」走 local_run.push_main，并先查有没有流程在跑")
    ck(rp is not None and all(_under_git_lock(c, spar)
                              for c in _calls(rp, "push_main")),
       "控制台调 push_main 也在 git_lock 里")
    ck(not [c for c in ast.walk(sv) if isinstance(c, ast.Constant)
            and c.value == "git"],
       "控制台里再没有手拼的 git 命令（推送只留一份实现，见 push_main）")
    pm = _func(lr, "push_main")
    ck(pm is not None and bool(_calls(_func(lr, "git_commit_push"), "push_main")),
       "流程的推送也走同一个 push_main（两份实现必然漂）")
    ck(pm is not None and len(_calls(pm, "_write_push_status")) >= 2,
       "push_main 成功失败两条路都写 push_status（教训 16）")
    # 日线表两条线都会去补：backfill 里的写入口全在 data_lock 里
    bf = _tree("src/breakout/backfill.py")
    su = _func(bf, "stage_update")
    ck(su is not None and bool(_calls(su, "data_lock")) and bool(_calls(su, "_stage_update")),
       "backfill.stage_update 先拿日线写锁再干活")
    mn = _func(bf, "main")
    ck(mn is not None and bool(_calls(mn, "data_lock")),
       "merge / refresh / sina 三个入口也拿同一把锁（控制台「全量回填」按钮走 refresh）")
    md = _func(bf, "merge_daily")
    ck(md is not None and bool(_calls(md, "write_atomic"))
       and not [c for c in _calls(md, "to_parquet")
                if any(isinstance(a, ast.Name) and a.id == "target" for a in c.args)],
       "合并后的日线表先写临时文件再换名（读的一方看不到半截 parquet）")


def check_sync_clears_pull() -> None:
    """拉远端失败留下的红灯，下一次拉成功就清掉；推送失败的红灯不许被它清掉。"""
    print("\n[同步：拉取失败的红灯会自己好]")
    import tempfile
    import local_run
    tmp = Path(tempfile.mkdtemp(prefix="syncok_"))
    o = (local_run.ROOT, local_run._git, local_run.any_flow_running)
    local_run.ROOT = tmp
    local_run._git = lambda *a: (0, "")
    local_run.any_flow_running = lambda: ""
    ps = tmp / "state" / "push_status.json"
    ps.parent.mkdir(parents=True, exist_ok=True)
    try:
        ps.write_text(json.dumps({"ok": False, "msg": "pull", "detail": "Connection was reset"}),
                      encoding="utf-8")
        ck(local_run.sync_repo() is True
           and json.loads(ps.read_text(encoding="utf-8"))["ok"] is True,
           "上次拉取失败、这次拉成功 -> 红灯清掉（以前一直挂着「上次推送失败」）")
        ps.write_text(json.dumps({"ok": False, "msg": "长期调整突破 2026-09-29 [local]",
                                  "detail": "push 连续三次失败"}), encoding="utf-8")
        local_run.sync_repo()
        ck(json.loads(ps.read_text(encoding="utf-8"))["ok"] is False,
           "推送失败的红灯拉取成功也不清（产物还没推上去）")
    finally:
        local_run.ROOT, local_run._git, local_run.any_flow_running = o


def check_status_signals() -> None:
    """同步卡片：失败必须是一个能被界面查询的对象（教训 16）。"""
    print("\n[同步信号]")
    import subprocess
    import tempfile
    import time as _t
    from gui import status as st
    tmp = Path(tempfile.mkdtemp(prefix="sync_"))
    (tmp / ".git").mkdir(parents=True)
    (tmp / "state").mkdir()
    calls: list = []

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kw):
        calls.append((args, kw))
        return _R()

    o = (st.ROOT, subprocess.run, st._fetch_at["t"])
    st.ROOT = tmp
    subprocess.run = fake_run
    st._fetch_at["t"] = _t.time()        # 别去打网络
    try:
        st._git("status", "--porcelain")
        ck(bool(calls) and calls[-1][1].get("env", {}).get("GIT_OPTIONAL_LOCKS") == "0",
           "控制台的每条 git 都带 GIT_OPTIONAL_LOCKS=0（否则每 5 秒抢一次 index.lock）")
        (tmp / ".git" / "index.lock").write_text("", encoding="utf-8")
        ck(any("index.lock" in p for p in st.sync_status()["problems"]),
           "残留的 index.lock 在总览页亮红（它会让之后每一次 add 都失败）")
        (tmp / ".git" / "index.lock").unlink()
        (tmp / "state" / "push_status.json").write_text(json.dumps(
            {"ok": False, "msg": "sent: breakout", "detail": "add 失败: ..."}),
            encoding="utf-8")
        s = st.sync_status()
        ck(not s["ok"] and any("推送失败" in p for p in s["problems"]),
           "推送失败在总览页亮红")
    finally:
        st.ROOT, subprocess.run, st._fetch_at["t"] = o


def check_trade_dates_cache() -> None:
    """交易日历缓存：读到半截文件不能退化成「周一到周五」。"""
    print("\n[交易日历缓存]")
    import inspect
    import tempfile
    import time as _t
    import local_run
    from gui import status as st
    tmp = Path(tempfile.mkdtemp(prefix="td_"))
    (tmp / "state").mkdir()
    f = tmp / "state" / "trade_dates.json"
    o = (st.ROOT, dict(st._td_cache))
    st.ROOT = tmp
    try:
        f.write_text(json.dumps(["2026-09-14", "2026-09-15"]), encoding="utf-8")
        st._td_cache.update(at=0.0, s=set())
        ck(st._trade_dates() == {"2026-09-14", "2026-09-15"}, "完整日历读得出来")
        f.write_text('["2026-09-14","2026-09-1', encoding="utf-8")
        st._td_cache["at"] = 0.0
        ck(st._trade_dates() == {"2026-09-14", "2026-09-15"},
           "读到半截 JSON：留着上一次的日历，不退化成「周一到周五」")
        ck(_t.time() - st._td_cache["at"] >= 540, "半截文件只压 60 秒，不是 10 分钟")
        f.write_text("", encoding="utf-8")
        st._td_cache["at"] = 0.0
        ck(st._trade_dates() == {"2026-09-14", "2026-09-15"},
           "空文件（刚截断还没写）同样不清空")

        # Windows 上写方是 tmp + os.replace（datasource.trade_dates），换名
        # 那一瞬间读方拿到 OSError，而 Path.exists() 会把它吞成 False。
        # 以前的写法用 exists() 开路，于是「文件其实在」被当成「文件没有」，
        # 空集压 10 分钟，这 10 分钟里控制台按「周一到周五」算目标日。
        f.write_text(json.dumps(["2026-09-14", "2026-09-15"]), encoding="utf-8")
        st._td_cache.update(at=0.0, s={"2026-09-14", "2026-09-15"})

        class _Racy:
            def __init__(self, real: Path) -> None:
                self._r = real

            def exists(self) -> bool:
                return False              # OSError 被 pathlib 吞成 False

            def stat(self):
                raise PermissionError(13, "正在被 os.replace 换名")

            def read_text(self, **kw):
                return self._r.read_text(**kw)

        class _Dir:
            def __init__(self, p: Path) -> None:
                self.p = p

            def __truediv__(self, x):
                q = self.p / x
                return _Dir(q) if x == "state" else _Racy(q)

        st.ROOT = _Dir(tmp)
        st._td_cache["at"] = 0.0
        ck(st._trade_dates() == {"2026-09-14", "2026-09-15"},
           "撞上 os.replace 换名那一瞬间：日历照读得出来，不退化成「周一到周五」")
        st.ROOT = tmp

        f.unlink()
        st._td_cache["at"] = 0.0
        ck(st._trade_dates() == set(), "文件真没了才退化（合法降级，不是故障）")
        ck(_t.time() - st._td_cache["at"] >= 540, "退化成空集也只压 60 秒")
    finally:
        st.ROOT = o[0]
        st._td_cache.clear(); st._td_cache.update(o[1])
    ck("write_text" not in inspect.getsource(local_run.trade_dates),
       "local_run 不再重复写 state/trade_dates.json（datasource 已经写过同样的内容，写者减半）")

    # 探针每 15~30 分钟起一次，缓存还新就别再 import akshare 拉一遍日历
    import types
    tmp2 = Path(tempfile.mkdtemp(prefix="td2_"))
    (tmp2 / "state").mkdir()
    o2 = (local_run.ROOT, dict(local_run._TD), local_run.now_bj,
          sys.modules.get("datasource"))
    local_run.ROOT = tmp2
    local_run._TD.clear()
    local_run.now_bj = lambda: __import__("datetime").datetime(2026, 9, 16, 8, 0)

    def _boom():
        raise AssertionError("缓存还新的时候不该去拉日历")

    sys.modules["datasource"] = types.SimpleNamespace(trade_dates=_boom)
    try:
        (tmp2 / "state" / "trade_dates.json").write_text(
            json.dumps(["2026-09-15", "2026-09-16", "2026-12-31"]), encoding="utf-8")
        ck(local_run.trade_dates() == {"2026-09-15", "2026-09-16", "2026-12-31"},
           "12 小时内写的、覆盖到今天之后的缓存直接用，不联网")
    finally:
        local_run.ROOT, local_run.now_bj = o2[0], o2[2]
        local_run._TD.clear(); local_run._TD.update(o2[1])
        if o2[3] is None:
            sys.modules.pop("datasource", None)
        else:
            sys.modules["datasource"] = o2[3]


def check_panels() -> None:
    """前端 tab 的 data-p 必须和后端 PANELS 的键对得上。"""
    print("\n[面板路由]")
    import re
    from gui.server import PANELS
    from gui.ui import PAGE
    tabs = set(re.findall(r'data-p="([a-z]+)"', PAGE))
    ck(tabs == set(PANELS),
       f"前端 tab 和后端 PANELS 一致（前端 {sorted(tabs)}，后端 {sorted(PANELS)}）")
    ck("__TOKEN__" in PAGE, "页面里有 token 占位符（服务端要替换它）")


def check_http() -> None:
    """两道防护 + 读接口。失效是静默的，界面上看不出来。

    这里把 git 和 PowerShell 都换成假实现：自测必须离线（和另外三条
    一条规矩），而 /api/status 真跑起来会 git fetch 打网络、起
    powershell.exe 查计划任务，慢且结果不可复现。
    顺带这也是一次容错测试——外部命令全挂时页面照样要出得来。
    """
    print("\n[HTTP 防护]")
    from gui import server
    from gui import status as st

    orig_git, orig_tasks = st._git, st.scheduled_tasks
    st._git = lambda *_a: ""                      # 装成 git 不可用
    st.scheduled_tasks = lambda force=False: {}   # 装成查不到计划任务

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, method="GET", host=None, token=None, body=None):
        req = urllib.request.Request(
            base + path, method=method,
            data=(json.dumps(body).encode("utf-8") if body is not None
                  else (b"{}" if method == "POST" else None)))
        if host:
            req.add_header("Host", host)
        if token:
            req.add_header("X-Token", token)
        if method == "POST":
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        code, body = call("/")
        ck(code == 200, "GET / 返回 200")
        ck(server.TOKEN.encode() in body and b"__TOKEN__" not in body,
           "页面里的 token 占位符被真 token 替换了")

        code, body = call("/full")
        ck(code == 200 and server.TOKEN.encode() in body, "GET /full（详细控制台）返回 200")
        code, body = call("/api/simple")
        try:
            so = json.loads(body)
        except Exception:  # noqa: BLE001
            so = {}
        ck(code == 200 and len(so.get("lines", [])) == 2,
           "GET /api/simple 返回两条线（git 是死的也出得来）")

        code, body = call("/api/status")
        ck(code == 200, "GET /api/status 返回 200")
        try:
            o = json.loads(body)
        except Exception:  # noqa: BLE001
            o = {}
        ck({"lines", "sync", "tasks"} <= set(o), "状态里有 lines/sync/tasks")
        # 上面的 git 是死的，所以这同时证明了「外部命令全挂也出得来页面」
        from gui.status import LINES
        ck(len(o.get("lines", [])) == len(LINES),
           f"git 不可用时 {len(LINES)} 条流程照样列得出来")

        ck(call("/api/status", host="evil.example.com")[0] == 403,
           "伪造 Host 被挡（防 DNS rebinding）")
        ck(call("/api/run", "POST")[0] == 403, "POST 无 token 被挡")
        ck(call("/api/run", "POST", token="wrong")[0] == 403,
           "POST 错 token 被挡")
        ck(call("/api/task", "POST", token=server.TOKEN)[0] == 403,
           "带对 token 也改不了非 DailyReport-* 的计划任务")

        # 任务名会拼进 PowerShell 的**单引号字符串**，一个 ' 就出得来。
        # 实测 DailyReport-x'; Write-Output INJECTED-$PID; ' 通过了原来的
        # startswith 检查、第二条语句真的执行、退出码还是 0，接口回 ok:true。
        import subprocess as _sp
        import types
        ps_calls: list = []
        orig_run = _sp.run

        def fake_ps(*a, **k):
            ps_calls.append(a)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        _sp.run = fake_ps
        st.scheduled_tasks = lambda force=False: {
            "DailyReport-Local-Sync": {"name": "DailyReport-Local-Sync",
                                       "state": "Ready"}}
        try:
            evil = "DailyReport-Local-Sync'; Write-Output pwned; '"
            code, _ = call("/api/task", "POST", token=server.TOKEN,
                           body={"name": evil, "enable": True})
            ck(code == 403 and not ps_calls,
               "带单引号的任务名被挡，powershell 一次都没起（否则是任意命令执行）")
            code, _ = call("/api/task", "POST", token=server.TOKEN,
                           body={"name": "DailyReport-Nope", "enable": True})
            ck(code == 403 and not ps_calls,
               "前缀对、名字合法，但 Windows 上没有这个任务，也不放行")
            code, _ = call("/api/task", "POST", token=server.TOKEN,
                           body={"name": "DailyReport-Local-Sync", "enable": True})
            ck(code == 200 and len(ps_calls) == 1
               and "-TaskName 'DailyReport-Local-Sync'" in ps_calls[0][0][-1],
               "白名单里的任务照常放行，命令串里只有这个名字")
        finally:
            _sp.run = orig_run
            st.scheduled_tasks = lambda force=False: {}

        _check_push_api(call, server)

        ck(call("/panel/../../config.yaml")[0] in (403, 404),
           "面板路由不能拿它去读任意文件")
        for gone in ("auction", "learn", "council"):
            ck(call(f"/panel/{gone}")[0] == 404, f"已归档的 /panel/{gone} 返回 404")
        ck(call("/api/council/decide", "POST", token=server.TOKEN, body={})[0] == 404,
           "会诊批准接口随早盘系统删掉")
        ck(call("/nope")[0] == 404, "未知路径 404")
    finally:
        httpd.shutdown()
        httpd.server_close()
        st._git, st.scheduled_tasks = orig_git, orig_tasks


def _check_push_api(call, server) -> None:
    """「重试上传」按钮：有流程在跑不动工作区；分叉时真的合并；成败都写状态。"""
    import tempfile
    import local_run
    tmp = Path(tempfile.mkdtemp(prefix="gpush_"))
    o = (local_run.ROOT, local_run._git, local_run.any_flow_running)
    calls: list = []
    local_run.ROOT = tmp
    try:
        local_run.any_flow_running = lambda: "breakout"
        local_run._git = lambda *a: (calls.append(a) or (0, ""))
        code, _body = call("/api/push", "POST", token=server.TOKEN, body={})
        ck(code == 409 and not calls,
           "有流程正在跑 -> 「重试上传」回 409，一条 git 都不执行（它正在写产物）")

        local_run.any_flow_running = lambda: ""
        code, body = call("/api/push", "POST", token=server.TOKEN, body={})
        ck(code == 200 and json.loads(body).get("ok") is True,
           "没人在跑 -> 推一次并如实回 ok")
        ck(json.loads((tmp / "state" / "push_status.json")
                      .read_text(encoding="utf-8"))["ok"] is True,
           "手动推成功要改写 push_status（以前不写，卡片推完还是红的）")

        calls.clear()

        def diverged(*a):
            calls.append(a)
            if a[0] == "push":
                return (1, "! [rejected] non-fast-forward")
            if a[0] == "merge":
                return (0, "Auto-merging")
            return (0, "")

        local_run._git = diverged
        code, body = call("/api/push", "POST", token=server.TOKEN, body={})
        ck(code == 200 and json.loads(body).get("ok") is False,
           "推不上去如实回 ok:false（以前无论成败都 toast「已重试」）")
        ck(("merge", "--no-edit", "-q", "-X", "ours", "origin/main") in calls,
           "本地和远端分叉时真的去合并 —— 这按钮本来就是为 09-07 那次分叉加的，"
           "而它以前只有 fetch + push，按多少下都是 non-fast-forward")
        ps = json.loads((tmp / "state" / "push_status.json")
                        .read_text(encoding="utf-8"))
        ck(ps["ok"] is False and ps["detail"], "失败也写 push_status，红灯有理由")
    finally:
        (local_run.ROOT, local_run._git, local_run.any_flow_running) = o


def check_log_time_bj() -> None:
    """日志时间戳一律北京时间：控制台「运行记录」「运行」两页直接印日志。"""
    print("\n[日志时间是北京时间]")
    import logging
    import time as _time
    bad, n = [], 0
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").glob("*.py")):
        s = p.read_text(encoding="utf-8")
        if "logging.basicConfig(" in s and "%(asctime)s" in s:
            n += 1
            if "Formatter.converter = staticmethod(" not in s:
                bad.append(p.relative_to(ROOT).as_posix())
    ck(n >= 8 and not bad, f"带时间戳的 {n} 个入口都把日志换成北京时间（漏了：{bad}）")
    # 直接赋 lambda 会被绑成方法：每条日志 TypeError，logging 吞掉异常、消息丢失
    keep = logging.Formatter.converter
    try:
        logging.Formatter.converter = staticmethod(lambda t: _time.gmtime(t + 8 * 3600))
        f = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M")
        rec = logging.makeLogRecord({"msg": "x", "created": 1790000000.0})
        ck(f.format(rec).startswith(_time.strftime("%Y-%m-%d %H:%M",
                                                    _time.gmtime(1790000000 + 8 * 3600))),
           "staticmethod 包着的换算器按 UTC+8 出时间")
    finally:
        logging.Formatter.converter = keep
    cmd = (ROOT / "tools" / "run_local.cmd").read_text(encoding="ascii")
    ck(cmd.count("ToUniversalTime().AddHours(8)") == 2 and "Get-Date -Format s" not in cmd,
       "run_local.cmd 的开始 / 结束两行时间头也是北京时间")
    ck(r"[\d\-T:]+\] === run_local" in (ROOT / "src" / "gui" / "simple.py")
       .read_text(encoding="utf-8"),
       "首页按 [yyyy-MM-ddTHH:mm:ss] 切分运行块，时间头格式没变")


def check_task_sentinel() -> None:
    """「从没跑过」的哨兵只能按 rc 认，不能按 last 的 11-30 前缀认。"""
    print("\n[计划任务哨兵]")
    import inspect
    import subprocess
    import types
    from gui import status as st
    rows = [
        {"name": "DailyReport-Local-Evening", "state": "Ready",
         "last": "2026-11-30 10:07", "rc": 0, "next": "2026-12-01 08:30"},
        {"name": "DailyReport-Never", "state": "Ready",
         "last": "1999-11-30 05:00", "rc": 267011, "next": ""},
    ]
    orig = subprocess.run
    subprocess.run = lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps(rows), stderr="")
    try:
        out = st.scheduled_tasks(force=True)
        m = out.get("DailyReport-Local-Evening", {})
        n = out.get("DailyReport-Never", {})
        ck(m.get("last") == "11-30 18:07",
           "真的 11 月 30 日跑过，「上次」不被当哨兵清空（每年白一整天）")
        ck(m.get("rc") == 0 and m.get("never") is None, "跑过的任务 rc 原样保留")
        ck(m.get("next") == "12-01 16:30", "PowerShell 给 UTC，排期页印北京时间（08:30Z = 16:30）")
        ck(n.get("last") == "" and n.get("rc") is None and n.get("never") is True,
           "rc=267011 才是从没跑过：清空「上次」、标 never")
    finally:
        subprocess.run = orig
        st._task_cache.update(at=0.0, data={})
    ck("Year -ge 2000" in inspect.getsource(st.scheduled_tasks),
       "年份在 PowerShell 那一侧过滤（1999-11-30 哨兵是本机时间，转 UTC 之前判）")


def check_done_semantics() -> None:
    """「跑完了」和「发出去了」是两件事，控制台必须分得清。"""
    print("\n[试跑 / 发信记录]")
    import datetime as dt
    import tempfile
    import local_run
    from gui import status as st
    tmp = Path(tempfile.mkdtemp(prefix="sent_"))
    d, tz = "2026-09-29", dt.timezone(dt.timedelta(hours=8))
    o = (local_run.ROOT, st.ROOT, st.target_date, local_run.target_date,
         local_run.now_bj, st.now_bj, st._trade_dates,
         local_run.running_instance, local_run._scan_processes)
    local_run.ROOT = st.ROOT = tmp
    st.target_date = lambda key: d
    local_run.target_date = lambda flow: d
    local_run.now_bj = lambda: dt.datetime(2026, 9, 29, 18, 30, tzinfo=tz)
    st.now_bj = lambda: dt.datetime(2026, 9, 29, 18, 30)
    st._trade_dates = lambda: set()
    local_run._scan_processes = lambda flow: None
    local_run.running_instance = lambda flow: None
    try:
        for ln in st.LINES:
            (tmp / ln["meta"]).parent.mkdir(parents=True, exist_ok=True)
        rows = lambda: {r["key"]: r for r in st.line_status({})}  # noqa: E731

        # 主断言：两份「跑完了」的定义必须逐条相等。控制台不能调
        # already_done（那条路 import akshare，教训 19），所以它用的是
        # done_for(key, 自己算的目标日) —— 等价性只能靠这条断言钉住。
        marks: list[Path] = []
        for ln in st.LINES:
            k, mp = ln["key"], tmp / ln["meta"]
            mp.write_text(json.dumps({"date": d, "n": 3, "dry": True}),
                          encoding="utf-8")
            r = rows()[k]
            ck(r["done"] is False and r["done"] == local_run.done_for(k, d),
               f"{k}：只试跑过 -> 控制台和计划任务都说「没跑完」")
            ck(r["dry"] is True, f"{k}：试跑这件事在返回里说得出来")
            mp.write_text(json.dumps({"date": d, "n": 3, "dry": False}),
                          encoding="utf-8")
            if k in local_run.SENT_REQUIRED:
                mk = tmp / "state" / "sent" / f"{local_run.SENT_MARK[k]}_{d}.json"
                mk.parent.mkdir(parents=True, exist_ok=True)
                mk.write_text("{}", encoding="utf-8")
                marks.append(mk)
            r = rows()[k]
            ck(r["done"] is True and r["done"] == local_run.done_for(k, d),
               f"{k}：真跑完 -> 两边都说「跑完了」")
        for mk in marks:
            mk.unlink()

        # 发信记录：两条线都看 state/sent/<线>_<目标日>.json
        for k in ("breakout", "pullback"):
            sp = tmp / "state" / "sent" / f"{k}_{d}.json"
            ck(rows()[k]["sent"] is False, f"{k} 没有 sent 标记 -> 没发出去")
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text("{}", encoding="utf-8")
            ck(rows()[k]["sent"] is True, f"{k} 推过 sent 标记 -> 发出去了")
            sp.unlink()
        v = st.verdict("pullback", True, d, dry=False, sent=False)
        ck("没有发信记录" in v, f"排期页照实说「跑完了但没发信」（实际：{v[:40]}…）")
        # 试跑之后排期页不许说「已跑完，再触发直接退出」：计划任务到点真会跑
        v = st.verdict("pullback", False, d, dry=True, sent=False)
        ck("只试跑过" in v and "已跑完" not in v,
           f"试跑之后排期页不说「已跑完」（实际：{v[:40]}…）")
    finally:
        (local_run.ROOT, st.ROOT, st.target_date, local_run.target_date,
         local_run.now_bj, st.now_bj, st._trade_dates,
         local_run.running_instance, local_run._scan_processes) = o


def check_daily_coverage() -> None:
    """日线「补到目标日了没有」按覆盖率判，不按全表 max。两条线共用一份。"""
    print("\n[日线覆盖率]")
    import local_run
    lr = _tree("src/local_run.py")
    dr = _func(lr, "daily_ready")
    ck(dr is not None and bool(_calls(dr, "daily_coverage"))
       and bool(_calls(dr, "_daily_covers")),
       "daily_ready 同时过两道闸：覆盖率 daily_coverage + 目标日行数 _daily_covers")
    for fn in ("flow_breakout", "ensure_daily"):
        f = _func(lr, fn)
        ck(f is not None and bool(_calls(f, "daily_ready")),
           f"{fn} 判「补到了没有」走 daily_ready（两条线一份闸，教训 34）")
    for fn in ("flow_breakout", "daily_ready", "ensure_daily"):
        f = _func(lr, fn)
        ck(not [c for c in ast.walk(f) if isinstance(c, ast.Call)
                and _callname(c) == "max"],
           f"{fn} 里没有裸 max()（全表 max 一只新票就能顶起来）")
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        ck(True, "没有 pandas，覆盖率的数值断言跳过")
        return
    # 2026-09-16 实测：done_sina.json 有 33 只从没成功过的新票，任何一只
    # 补齐就让全表 max 等于目标日，另外 5515 只还停在旧日期。
    df = pd.DataFrame({
        "code": ["000001", "000002", "000003", "688797"],
        "date": ["2026-09-15", "2026-09-15", "2026-09-15", "2026-09-16"],
    })
    cov, mx = local_run.daily_coverage(df, "2026-09-16")
    ck(abs(cov - 0.25) < 1e-9 and mx == "2026-09-16",
       "3 只停在昨天、1 只到今天：覆盖率 25%，而全表 max 已经是今天")
    ck(cov < local_run.DAILY_COVER_MIN, "25% 不够 -> 不出清单（旧写法会放行）")
    df2 = pd.DataFrame({"code": [f"{i:06d}" for i in range(100)],
                        "date": ["2026-09-16"] * 95 + ["2026-09-10"] * 5})
    cov2, _ = local_run.daily_coverage(df2, "2026-09-16")
    ck(cov2 >= local_run.DAILY_COVER_MIN,
       "95% 的票补到了 -> 放行（长期停牌的那十几只不该卡住整条线）")

    # 目标日那一天的**行数**还要和前一个交易日比（F7-9）。合并之后才看得见
    # 的掉行：重拉分片对它覆盖的代码是「整段权威」，会把当日增量那一根清掉，
    # 而 backfill 自己的核对发生在合并之前。
    codes = [f"{i:06d}" for i in range(100)]
    full = pd.DataFrame({"code": codes * 2,
                         "date": ["2026-09-15"] * 100 + ["2026-09-16"] * 100})
    ok, why = local_run._daily_covers(full, "2026-09-16")
    ck(ok and "100.0%" in why, f"两天行数一样 -> 放行（{why}）")
    thin = pd.DataFrame({"code": codes + codes[:92],
                         "date": ["2026-09-15"] * 100 + ["2026-09-16"] * 92})
    cov3, _ = local_run.daily_coverage(thin, "2026-09-16")
    ok2, why2 = local_run._daily_covers(thin, "2026-09-16")
    ck(cov3 >= local_run.DAILY_COVER_MIN and not ok2,
       f"目标日比前一日少 8%：覆盖率那道闸放行，行数这道拦住（{why2}）")
    ok3, why3 = local_run._daily_covers(full, "2026-09-17")
    ck(not ok3 and "没有" in why3, f"表里根本没有目标日 -> 不出清单（{why3}）")
    one = pd.DataFrame({"code": codes, "date": ["2026-09-16"] * 100})
    ck(local_run._daily_covers(one, "2026-09-16")[0],
       "表里只有目标日一天（没有更早的可比）-> 放行，不拿空基准卡死")


def check_breakout_flow() -> None:
    """起涨预测的三道门：补到目标日、补不上的票不许成片、真发了信才推标记。

    这一条线上「退出码 0」特别不值钱：backfill 自己记的账在 json 里、
    send 阶段的 SKIP_MAIL 也返回 0。三道门都只认产物，不认退出码（教训 27）。
    """
    print("\n[起涨预测：补到没有 / 发出去没有]")
    import datetime as dt
    import tempfile
    import local_run
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        ck(True, "没有 pandas，起涨预测流程断言跳过")
        return

    d, prev = "2026-09-16", "2026-09-15"
    tz = dt.timezone(dt.timedelta(hours=8))
    tmp = Path(tempfile.mkdtemp(prefix="bkflow_"))
    o = (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.truth_and_regime)
    # 代码必须来自真实代码表（历史教训 2）
    real = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
            .str.zfill(6).tolist())[:100]
    ran: list = []
    local_run.ROOT = tmp
    local_run.now_bj = lambda: dt.datetime(2026, 9, 16, 18, 0, tzinfo=tz)
    local_run.target_date = lambda flow: d
    local_run.sync_repo = lambda: True
    local_run.push_marker = lambda *a, **k: ran.append(("mark",) + a[:2])
    local_run.push_all = lambda *a, **k: ran.append(("push_all", a[0]))
    # 市场环境 + 清单真值：真实现会起一个子进程读日线，这里只记它被叫到了没有
    local_run.truth_and_regime = lambda dry: ran.append(("truth", dry))

    def write_daily(n_target: int) -> None:
        (tmp / "data" / "breakout").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"code": real + real[:n_target],
                      "date": [prev] * len(real) + [d] * n_target}
                     ).to_parquet(tmp / "data" / "breakout" / "daily.parquet",
                                  index=False)

    def write_status(short: list) -> None:
        (tmp / "state" / "breakout").mkdir(parents=True, exist_ok=True)
        (tmp / "state" / "breakout" / "update_status.json").write_text(
            json.dumps({"date": d, "appended": len(real), "missing": 0,
                        "short": short, "ok": True}), encoding="utf-8")

    def mk_py(send_writes: str):
        """跑子阶段的假实现。send_writes 决定 send 那一步写哪个日期的
        mail_sent.json（空串 = 一个字都不写，模拟 SKIP_MAIL / 半路 return 0）。"""
        def _py(*a):
            ran.append(a)
            if a[:3] == ("src/breakout/daily.py", "--stage", "send") \
                    and send_writes:
                (tmp / "out_breakout").mkdir(parents=True, exist_ok=True)
                (tmp / "out_breakout" / "mail_sent.json").write_text(
                    json.dumps({"date": send_writes, "n_a": 3, "at": "18:02"}),
                    encoding="utf-8")
            return 0
        return _py

    try:
        write_daily(len(real))
        write_status([])
        local_run.py = mk_py(d)
        rc = local_run.flow_breakout(False)
        bf = [a for a in ran if a and a[0] == "src/breakout/backfill.py"]
        ck(rc == 0 and bf and list(bf[0][1:]) == ["--stage", "update",
                                                  "--target", d],
           f"正常日：补数据带上目标日 --target {d}（实际 {bf and bf[0][1:]}）")
        ck(("mark", "sent", "breakout") in [x[:3] for x in ran if x[0] == "mark"],
           "真发了信（mail_sent.json 是目标日的）-> 推 sent 标记")
        it = [i for i, x in enumerate(ran) if x[0] == "truth"]
        isend = [i for i, x in enumerate(ran)
                 if x[:3] == ("src/breakout/daily.py", "--stage", "send")]
        ck(len(it) == 1 and isend and it[0] < isend[0],
           "发信之前算一次市场环境 + 清单真值（邮件里的「近期基准」读它；"
           "以前只有每周一次的学习会诊顺手算）")

        # send 退出码 0 但没写 mail_sent：SKIP_MAIL、或者 send_mail 之前
        # 就 return 0 的那几条路。以前照推 sent 标记，云端 20:30 据此不提醒。
        (tmp / "out_breakout" / "mail_sent.json").unlink()
        ran.clear()
        local_run.py = mk_py("")
        rc = local_run.flow_breakout(False)
        marks = [x for x in ran if x[0] == "mark" and x[1] == "sent"]
        ck(rc == 1 and not marks,
           "send 退出码 0 却没有发信记录 -> rc=1，不推 sent 标记")
        ck(any(x[0] == "push_all" for x in ran),
           "产物照样提交（清单和面板已经生成了，别连它一起丢）")

        ran.clear()
        local_run.py = mk_py(prev)
        ck(local_run.flow_breakout(False) == 1,
           "mail_sent.json 是上一个交易日的 -> 同样算没发（教训 22 那种「拿旧的当今天」）")

        # 补不上目标日的票成片：那天的横截面百分位和市场宽度都在残缺池子里算
        ran.clear()
        write_status(real[:local_run.UPDATE_SHORT_MAX + 1])
        local_run.py = mk_py(d)
        rc = local_run.flow_breakout(False)
        ck(rc == 1 and not [a for a in ran if a and a[0] == "src/breakout/build.py"],
           f"没拉到目标日的票 {local_run.UPDATE_SHORT_MAX + 1} 只 -> 不建特征、不出清单")
        ran.clear()
        write_status(real[:3])
        ck(local_run.flow_breakout(False) == 0,
           "只差 3 只（新上市/长期停牌）-> 照常出清单，只留一条 warning")

        # 目标日行数比前一日少 8%：覆盖率那道闸放得过，行数这道放不过
        ran.clear()
        write_daily(92)
        ck(local_run.flow_breakout(False) == 1
           and not [a for a in ran if a and a[0] == "src/breakout/daily.py"],
           "目标日行数只有前一日的 92% -> 不打分不发信")
    finally:
        (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.truth_and_regime) = o


def check_pullback_flow() -> None:
    """长期调整突破：日线补到没有、谁来补、扫描产出对不对得上目标日、真发了信才推标记。"""
    print("\n[长期调整突破：补到没有 / 发出去没有]")
    import datetime as dt
    import tempfile
    import local_run
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        ck(True, "没有 pandas，长期调整突破流程断言跳过")
        return
    d, prev = "2026-09-29", "2026-09-28"
    tz = dt.timezone(dt.timedelta(hours=8))
    tmp = Path(tempfile.mkdtemp(prefix="pbflow_"))
    o = (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.running_instance, local_run.time)
    real = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
            .str.zfill(6).tolist())[:100]
    ran: list = []
    ft = _FakeTime()
    local_run.ROOT = tmp
    local_run.now_bj = lambda: dt.datetime(2026, 9, 29, 17, 45, tzinfo=tz)
    local_run.target_date = lambda flow: d
    local_run.sync_repo = lambda: True
    local_run.push_marker = lambda *a, **k: ran.append(("mark",) + a[:2])
    local_run.push_all = lambda *a, **k: ran.append(("push_all", a[0], tuple(a[1])))
    local_run.running_instance = lambda flow: None
    local_run.time = ft

    def write_daily(upto_target: bool) -> None:
        (tmp / "data" / "breakout").mkdir(parents=True, exist_ok=True)
        dates = [prev] * len(real) + ([d] * len(real) if upto_target else [])
        pd.DataFrame({"code": real + (real if upto_target else []), "date": dates}
                     ).to_parquet(tmp / "data" / "breakout" / "daily.parquet", index=False)

    def mk_py(scan_date: str, send_writes: str, update_fills: bool = True):
        def _py(*a):
            ran.append(a)
            if a[:3] == ("src/breakout/backfill.py", "--stage", "update") and update_fills:
                write_daily(True)
            if a[:3] == ("src/pullback.py", "--stage", "scan") and scan_date:
                (tmp / "out_pullback").mkdir(parents=True, exist_ok=True)
                (tmp / "out_pullback" / "run_meta.json").write_text(
                    json.dumps({"date": scan_date, "n": 1}), encoding="utf-8")
            if a[:3] == ("src/pullback.py", "--stage", "send") and send_writes:
                (tmp / "out_pullback" / "mail_sent.json").write_text(
                    json.dumps({"date": send_writes, "n": 1, "at": "17:58:00"}),
                    encoding="utf-8")
            return 0
        return _py

    def calls(prefix):
        return [a for a in ran if a[:len(prefix)] == prefix]

    try:
        # 1. 日线已经到目标日（起涨预测 16:30 补过）：不再补，直接扫、发
        write_daily(True)
        local_run.py = mk_py(d, d)
        rc = local_run.flow_pullback(False)
        ck(rc == 0 and not calls(("src/breakout/backfill.py",)),
           "日线已经补到目标日 -> 不再补一遍")
        sc = calls(("src/pullback.py", "--stage", "scan"))
        ck(len(sc) == 1 and list(sc[0][3:]) == ["--target", d],
           f"扫描带上编排层算好的目标日 --target {d}")
        se = calls(("src/pullback.py", "--stage", "send"))
        ck(len(se) == 1 and "--no-wait" not in se[0],
           "真跑时发信那一步不带 --no-wait（等到 17:58）")
        ck(("mark", "sent", "pullback") in [x[:3] for x in ran if x[0] == "mark"],
           "真发了信（mail_sent.json 是目标日的）-> 推 sent 标记")
        pa = [x for x in ran if x[0] == "push_all"]
        ck(pa and set(pa[-1][2]) == {f"data/pullback/{d[:7]}", "out_pullback"},
           f"推的是 data/pullback/{d[:7]} 和 out_pullback（{pa and pa[-1][2]}）")

        # 2. 日线还没到、起涨预测也没在跑：自己补一次（带 --target），补到了往下走
        ran.clear()
        write_daily(False)
        local_run.py = mk_py(d, d)
        rc = local_run.flow_pullback(False)
        bf = calls(("src/breakout/backfill.py",))
        ck(rc == 0 and len(bf) == 1 and list(bf[0][1:]) == ["--stage", "update", "--target", d],
           "日线没到、起涨预测没在跑 -> 自己补一次，带目标日")

        # 3. 自己补了还是没到：不扫描、不发信、rc=1
        ran.clear()
        write_daily(False)
        local_run.py = mk_py(d, d, update_fills=False)
        rc = local_run.flow_pullback(False)
        ck(rc == 1 and not calls(("src/pullback.py",)),
           "补完还没到目标日 -> 今天不出清单（不拿旧数据扫一份日期错位的）")

        # 4. 起涨预测正在跑：先等它补，不抢着补（补日线是它的第一步）
        ran.clear()
        write_daily(False)
        waits = {"n": 0}

        def busy(flow):
            if flow == "breakout" and waits["n"] < 2:
                waits["n"] += 1
                if waits["n"] == 2:
                    write_daily(True)       # 第二次问的时候它已经补好了
                return {"pid": 1}
            return None

        local_run.running_instance = busy
        ft.slept.clear()
        local_run.py = mk_py(d, d)
        rc = local_run.flow_pullback(False)
        ck(rc == 0 and not calls(("src/breakout/backfill.py",)) and ft.slept,
           "起涨预测正在跑 -> 等它把日线补上，自己不补（两边抢写 daily.parquet）")
        local_run.running_instance = lambda flow: None

        # 5. 扫描退出码 0 但没产出目标日的 run_meta（非交易日那条路）：不发信
        ran.clear()
        write_daily(True)
        local_run.py = mk_py(prev, d)
        ck(local_run.flow_pullback(False) == 1
           and not calls(("src/pullback.py", "--stage", "send")),
           "扫描没产出目标日的 run_meta -> 不发信（教训 22：退出码 0 不等于做了事）")

        # 6. send 退出码 0 但没写 mail_sent（SKIP_MAIL 那条路）：不推 sent，rc=1
        ran.clear()
        (tmp / "out_pullback" / "mail_sent.json").unlink()
        local_run.py = mk_py(d, "")
        rc = local_run.flow_pullback(False)
        ck(rc == 1 and not [x for x in ran if x[0] == "mark" and x[1] == "sent"],
           "send 退出码 0 却没有发信记录 -> rc=1，不推 sent 标记（云端 20:30 照样提醒）")
        ck(any(x[0] == "push_all" for x in ran), "产物照样提交")

        # 7. 试跑：发信那一步带 --no-wait，不写 sent 标记（push_marker 的 dry 分支）
        ran.clear()
        import os as _os
        keep = _os.environ.get("SKIP_MAIL")
        try:
            local_run.py = mk_py(d, "")
            local_run.flow_pullback(True)
            se = calls(("src/pullback.py", "--stage", "send"))
            ck(se and "--no-wait" in se[0], "试跑不等 17:58（--no-wait）")
            ck(_os.environ.get("SKIP_MAIL") == "1", "试跑设 SKIP_MAIL=1，不发信")
        finally:
            if keep is None:
                _os.environ.pop("SKIP_MAIL", None)
            else:
                _os.environ["SKIP_MAIL"] = keep
    finally:
        (local_run.ROOT, local_run.now_bj, local_run.target_date, local_run.py,
         local_run.sync_repo, local_run.push_marker, local_run.push_all,
         local_run.running_instance, local_run.time) = o

    # 上线日：09-28 之前的目标日计划任务不补（2026-09-27 注册任务时目标日还是 09-24）
    o2 = (local_run.target_date, local_run.weekend_skip, local_run.running_instance)
    try:
        local_run.weekend_skip = lambda flow: False
        local_run.running_instance = lambda flow: None
        local_run.target_date = lambda flow: "2026-09-24"
        why = local_run.if_needed_skip("pullback")
        ck("起才发清单" in why, f"目标日早于上线日 -> 计划任务不补（{why}）")
    finally:
        local_run.target_date, local_run.weekend_skip, local_run.running_instance = o2


def check_line_error() -> None:
    """fail-open 的分支把异常写进了 run_meta，界面必须把它显示出来（F3-7）。"""
    print("\n[出错要看得见]")
    import datetime as dt
    import tempfile
    from gui import status as st
    from gui.ui import PAGE
    tmp = Path(tempfile.mkdtemp(prefix="lerr_"))
    d = "2026-09-29"
    o = (st.ROOT, st.target_date, st.now_bj, st._trade_dates, st._local_run)
    st.ROOT = tmp
    st.target_date = lambda key: d
    st.now_bj = lambda: dt.datetime(2026, 9, 29, 18, 30)
    st._trade_dates = lambda: set()
    st._local_run = lambda: None
    try:
        f = tmp / "out_pullback" / "run_meta.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        sp = tmp / "state" / "sent" / f"pullback_{d}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("{}", encoding="utf-8")
        rows = lambda: {r["key"]: r for r in st.line_status({})}  # noqa: E731
        f.write_text(json.dumps({"date": d, "n": 0}), encoding="utf-8")
        r = rows()["pullback"]
        ck(r["done"] is True and r["error"] == "", "正常跑完：done=True，没有报错对象")
        f.write_text(json.dumps({"date": d, "n": 0, "error": "RuntimeError: 某一步炸了"}),
                     encoding="utf-8")
        r = rows()["pullback"]
        ck(r["error"].startswith("error: RuntimeError"),
           f"run_meta 里的报错被带到界面上（实际 {r['error'][:40]}）")
        ck(r["done"] is True, "它仍然算「跑完了」：两件事分开报")
    finally:
        (st.ROOT, st.target_date, st.now_bj, st._trade_dates, st._local_run) = o
    # 前端得真的用它，否则后端算了也白算（教训 11 那种「接了线没接上」）
    ck("l.error" in PAGE and 'return ["bad", "有报错"' in PAGE,
       "总览页把带报错的那一行标红")
    ck("l.sent === false || l.error" in PAGE,
       "顶栏「今天还没跑」的计数也把带报错的算进去")


def _wf(rel: str) -> dict:
    import yaml
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def _steps(rel: str, job: str) -> list:
    return _wf(rel)["jobs"][job]["steps"]


def _joined(run: str) -> list[str]:
    """把 shell 的反斜杠续行接回一行，方便按行找条件。"""
    out, buf = [], ""
    for ln in run.splitlines():
        if ln.rstrip().endswith("\\"):
            buf += ln.rstrip()[:-1]
            continue
        out.append(buf + ln)
        buf = ""
    if buf:
        out.append(buf)
    return out


def check_pages_wiring() -> None:
    """Pages 由 pages.yml 在推送后发布（2026-09-27 起，以前押在早盘的 auction.yml 上）。"""
    print("\n[Pages 发布]")
    wf = _wf(".github/workflows/pages.yml")
    on = wf.get("on") or wf.get(True) or {}
    paths = set((on.get("push") or {}).get("paths") or [])
    for p in ("out_breakout/panel.html", "out_pullback/panel.html",
              "out_breakout/stamp.txt", "out_pullback/stamp.txt"):
        ck(p in paths, f"推送了 {p} 就触发发布")
    ck("workflow_dispatch" in on, "也能手动点")
    ck("schedule" not in on, "不带 cron（只在面板真的变了的时候跑）")
    perm = wf.get("permissions") or {}
    ck(perm.get("pages") == "write" and perm.get("id-token") == "write",
       "有 pages: write 和 id-token: write（deploy-pages 要这两个）")
    steps = _steps(".github/workflows/pages.yml", "deploy")
    uses = [str(s.get("uses", "")) for s in steps]
    ck(any(u.startswith("actions/checkout") for u in uses)
       and any(u.startswith("actions/upload-pages-artifact") for u in uses)
       and any(u.startswith("actions/deploy-pages") for u in uses),
       "checkout -> build_site -> upload -> deploy 四步都在")
    co = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")]
    ck(co and "ref" in (co[0].get("with") or {}),
       "checkout 显式给 ref（workflow_dispatch 的 github.sha 钉死在派发那一刻）")
    ck(any("src/build_site.py" in str(s.get("run", "")) for s in steps),
       "发布走 build_site.py（一个仓库只有一份 Pages，两个面板要凑齐再发）")
    import build_site
    names = [x[1] for x in build_site.PAGES]
    ck(names == ["breakout.html", "pullback.html"],
       f"build_site 只发布两个晚间面板（{names}），早盘/学习/会诊页不再上线")
    ck("out_learn" not in (ROOT / "src" / "build_site.py").read_text(encoding="utf-8"),
       "build_site 不再去 out_learn 拷学习面板（已归档）")
    pg = build_site.with_nav("<html><body><h1>x</h1></body></html>", "pullback.html")
    ck('href="./"' in pg and 'href="breakout.html"' in pg
       and 'href="pullback.html"' not in pg and "<b>长期调整突破</b>" in pg
       and pg.index('class="nav"') < pg.index("<h1>"),
       "发布的面板顶上有导航：入口页 + 另一个面板，当前页不是链接")
    ck(build_site.with_nav("<html>x</html>", "pullback.html") == "<html>x</html>",
       "没有 <body> 的页面原样发布，不硬插")
    # 其余 workflow 都不许再发布 Pages：两处发布会互相冲掉
    other = [p.name for p in (ROOT / ".github" / "workflows").glob("*.yml")
             if p.name != "pages.yml" and "deploy-pages" in p.read_text(encoding="utf-8")]
    ck(not other, f"只有 pages.yml 发布 Pages（其它：{other}）")


def check_archived() -> None:
    """早盘系统 2026-09-27 整体归档：三层触发都停干净（教训 24），活代码不引用它。"""
    print("\n[早盘系统归档]")
    import re
    wfs = sorted(p.name for p in (ROOT / ".github" / "workflows").glob("*.yml"))
    for gone in ("auction.yml", "premarket.yml", "learn.yml", "intraday.yml",
                 "refresh_sector.yml", "pullback.yml"):
        ck(gone not in wfs, f".github/workflows/ 里没有 {gone}（移到 archive/ 才不会被 GitHub 执行）")
    toml = (ROOT / "tools" / "external-trigger" / "wrangler.toml").read_text(encoding="utf-8")
    js = (ROOT / "tools" / "external-trigger" / "worker.js").read_text(encoding="utf-8")
    crons = re.findall(r'"([\d*/ ,-]+)"', toml.split("crons", 1)[1]) if "crons" in toml else []
    ck(crons == ["45 12 * * *"], f"Cloudflare Worker 只剩 20:45 那条（{crons}）")
    ck('"auction.yml"' not in js, "Worker 的 ROUTES 里没有 auction.yml")
    routes = re.findall(r'"([\d*/ ,-]+)":\s*"([\w.]+)"', js)
    ck({c for c, _ in routes} == set(crons),
       f"worker.js 的 ROUTES 和 wrangler.toml 的 crons 逐字对应（{routes}）")
    ck(all((ROOT / ".github" / "workflows" / w).exists() for _, w in routes),
       "Worker 派发的 workflow 都还在")
    ps1 = (ROOT / "tools" / "setup_tasks.ps1").read_text(encoding="utf-8-sig")
    ck('"Morning"' not in ps1 and '"Learn"' not in ps1, "本机计划任务定义里没有 Morning / Learn")
    # 活代码不 import 已归档的模块
    banned = {"premarket", "run_auction", "score", "ths_export", "tdx_export",
              "collect_llm", "eval_daily", "cfg", "refresh_sector", "learn"}
    hits = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            hits += [f"{p.relative_to(ROOT).as_posix()}:{m}" for m in mods
                     if m.split(".")[0] in banned]
    ck(not hits, f"src/ 和 tools/ 不 import 已归档的早盘模块（{hits[:5]}）")
    for d in ("src/learn", "prompts", "out", "out_learn", "data/labels", "data/train"):
        ck(not (ROOT / d).exists(), f"{d} 已经搬进 archive/morning/")
    ck((ROOT / "archive" / "morning" / "src" / "run_auction.py").exists()
       and (ROOT / "archive" / "morning" / "config.yaml").exists(),
       "归档里有原样的代码和配置（要恢复的话从这里搬回去）")


def check_workflow_push() -> None:
    """云端提交：rebase 冲突之后 push 会「空成功」，不能认退出码。"""
    print("\n[云端提交的推送]")
    # 还在的两条会提交东西的 workflow 一个模板。漏掉一条不会报错，只会在某个
    # 提交丢掉的那天表现成「跑了但仓库里没有」，而那正是最难查的一种。
    for rel, job, nm in ((".github/workflows/evening_check.yml", "check", "记下"),
                         (".github/workflows/refresh_meta.yml", "refresh", "提交缓存")):
        step = [s for s in _steps(rel, job) if nm in str(s.get("name", ""))]
        ck(len(step) == 1, f"{rel}: 找得到「{nm}」那一步")
        if not step:
            continue
        # 只看真正执行的那些行：注释里为了说明「以前是怎么错的」会原样引用
        # 旧命令，按整段文本找会把注释当成代码。
        run = "\n".join(ln for ln in step[0]["run"].splitlines()
                        if not ln.lstrip().startswith("#"))
        ck("git pull --rebase --autostash || true" not in run,
           f"{rel}: 不再用 `pull --rebase --autostash || true`"
           f"（冲突被吞掉，HEAD 停在 origin/main 上，下一次 push 空成功返回 0）")
        ck("-X theirs" in run, f"{rel}: 冲突取 theirs（本 job 的产出为准）")
        ia = run.find("rebase --abort")
        ip = run.find("git push")
        ck(0 <= ia < ip, f"{rel}: 先清上一轮的 rebase 残留再 push（教训 15）")
        tail = run[run.rfind("git push"):]
        ck(("rev-parse" in tail or "cat-file" in tail) and "origin/" in tail,
           f"{rel}: 推完按 origin 上的东西核对，不认 push 的退出码")
        ck("::error::" in run and "::warning::push 失败" not in run,
           f"{rel}: 推丢了要 ::error::（这是唯一能被看见的信号）")
        # evening_check 推的是「今天已提醒」标记，提醒邮件那一步已经发出去了，
        # 最坏后果是下一个入口重复提醒一次；其余提交的是别人要读的产物，
        # 推丢了必须让 job 真的红。
        if "evening_check" not in rel:
            tail_err = run[run.find("::error::"):]
            ck("exit 1" in tail_err or "send_alert" in tail_err,
               f"{rel}: ::error:: 之后要 exit 1 或发告警邮件（光标红没人看得见）")
    ev = [s for s in _steps(".github/workflows/evening_check.yml", "check")
          if "记下" in str(s.get("name", ""))][0]["run"]
    ck("state/alert/breakout_$D.json" not in ev and "$NEW" in ev,
       "evening_check 推完逐个核对这次新写的提醒标记（两条线可能只缺一条）")


def check_push_paths_not_ignored() -> None:
    """推送列表里的路径一条都不许被 .gitignore 挡住（add 的 rc 会被吞）。"""
    print("\n[推送路径没被 gitignore 挡]")
    import subprocess
    paths = set()
    import build_site
    for src, *_ in build_site.PAGES:
        rel = src.relative_to(build_site.ROOT).as_posix()
        paths.add(f"{rel}/panel.html")
        paths.add(f"{rel}/stamp.txt")
    lr = _tree("src/local_run.py")
    for c in _calls(lr, "push_all") + _calls(lr, "push_marker"):
        for a in c.args:
            if isinstance(a, ast.List):
                paths |= {e.value for e in a.elts
                          if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    ck(len(paths) >= 7, f"收齐了要推的路径（{len(paths)} 条）")
    paths |= {"data/pullback/2026-09/pullback_2026-09-28.parquet",
              "out_pullback/selected.json", "out_pullback/history.json",
              "out_pullback/长期调整突破.txt", "state/sent/pullback_2026-09-28.json"}
    try:
        bad = []
        for p in sorted(paths):
            r = subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT,
                               capture_output=True, timeout=15)
            if r.returncode == 0:
                bad.append(p)
            elif r.returncode not in (0, 1):
                raise OSError(f"check-ignore rc={r.returncode}")
        ck(not bad, f"没有一条被 .gitignore 挡住（被挡的：{bad}）")
    except (OSError, subprocess.SubprocessError):
        ck(True, "git 不可用，跳过 check-ignore（自测必须离线可跑）")


def check_docs() -> None:
    """界面上的数字和排期必须从代码里来，不能手抄。"""
    print("\n[界面说明不漂]")
    import re
    from gui.jobs import ACTIONS
    from gui.status import RULES
    import local_run
    ck(not re.search(r"\d+ 条", ACTIONS["selftest_pullback"]["desc"]),
       "「检查长期调整突破」的说明不写死用例数（条数在自测输出里）")
    for k in ("breakout", "pullback"):
        a = "%02d:%02d" % local_run.FLOWS[k][4]
        ck(a in RULES[k]["local_when"], f"{k} 的自动开跑时刻和 FLOWS 一致（{a}）")
    raw = (ROOT / ".github/workflows/evening_check.yml").read_text(encoding="utf-8")
    crons = re.findall(r'^\s*-\s*cron:\s*"(\d+)\s+(\d+)\s', raw, re.M)
    bj = [f"{(int(h) + 8) % 24:02d}:{int(m):02d}" for m, h in crons]
    ck(bj == ["20:30"], f"evening_check 只有北京 20:30 一条 cron（{bj}）")
    ck(all("20:30" in RULES[k]["cloud"] for k in ("breakout", "pullback")),
       "排期页两条线的云端规则都写 20:30")
    import yaml
    send_at = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))[
        "pullback"]["send_at"][:5]
    ck(send_at in RULES["pullback"]["local_when"] and send_at in ACTIONS["pullback"]["desc"],
       f"排期页和按钮说明里的发信时刻和 config 一致（{send_at}）")


def check_simple_page() -> None:
    """首页傻瓜页（2026-09-18）：北京时间、多久没跑、失败原因、下次开机、进度。"""
    print("\n[首页·傻瓜页]")
    import datetime as dt
    import tempfile
    from gui import simple as S

    now = dt.datetime(2026, 9, 29, 8, 17)
    ck(S.ago(None, now) == "还没有", "没跑通过显示「上次跑通 还没有」")
    ck(S.ago(now - dt.timedelta(minutes=8), now) == "8 分钟前", "8 分钟前")
    ck(S.ago(now - dt.timedelta(hours=22, minutes=50), now) == "22 小时 50 分前",
       "22 小时 50 分前")
    ck(S.ago(now - dt.timedelta(days=3), now) == "3 天前", "超过两天按天数")

    # 下次开机：交易日表用假的（周三 09-30 是交易日、国庆不是、10-08 是）
    import local_run as L
    real = L.trade_dates
    L.trade_dates = lambda: {"2026-09-29", "2026-09-30", "2026-10-08"}
    try:
        w = S.next_window(dt.datetime(2026, 9, 29, 8, 17))
        ck(not w["now"] and w["start"] == "今天 16:30" and w["in"] == "8 小时 13 分",
           f"08:17 -> 今天 16:30、还有 8 小时 13 分（{w}）")
        w = S.next_window(dt.datetime(2026, 9, 29, 17, 50))
        ck(w["now"] and w["end"] == "18:05",
           f"17:50 在晚间窗口里，要开到 18:05（17:58 发完信）（{w}）")
        w = S.next_window(dt.datetime(2026, 9, 30, 18, 10))
        ck(w["start"].startswith("10-8") and "周四" in w["start"],
           f"09-30 18:10 -> 跳过国庆到 10-08（{w}）")
    finally:
        L.trade_dates = real

    # 失败原因从日志最后一次**真正开跑**的那一段取，秒退的敲门不算
    log = """[2026-09-17T04:30:02] === run_local breakout ===
04:30:09 本地全流程 breakout 启动 @ 16:30:09
04:39:50 补不上目标日的票 74 只，超过 50 只的上限，不出清单
[2026-09-17T04:39:50] === run_local breakout exit=1 ===
[2026-09-17T04:45:00] === run_local breakout ===
04:45:01 起涨预测 正在跑，这次不拉远端
[2026-09-17T04:45:01] === run_local breakout exit=0 ===
"""
    with tempfile.TemporaryDirectory() as td:
        real_root = S.ROOT
        try:
            S.ROOT = Path(td)
            (Path(td) / "tools").mkdir()
            (Path(td) / "tools" / "local_flow_breakout.log").write_text(
                log, encoding="utf-8")
            rc, why = S.last_run_block("breakout")
            ck(rc == 1 and "不出清单" in why,
               f"失败原因取到了，秒退的敲门被跳过（rc={rc}，{why}）")
        finally:
            S.ROOT = real_root

    # 进度：起涨预测第 2 步占九成，第 2 步做到一半时总进度应在 45% 上下；
    # 长期调整突破大部分时间在第 4 步等 17:58，等到一半应在六成上下
    with tempfile.TemporaryDirectory() as td:
        real_root = S.ROOT
        try:
            S.ROOT = Path(td)
            (Path(td) / "state" / "lock").mkdir(parents=True)
            (Path(td) / "state" / "lock" / "progress_breakout.json").write_text(
                json.dumps({"running": True, "step": 2, "total": 4, "sub": 0.5,
                            "text": "补 2026-09-17 日线 + 重算特征表"}),
                encoding="utf-8")
            pr = S.progress("breakout", {"pid": 1}, now)
            ck(45 <= pr["pct"] <= 47, f"第 2 步做一半 -> {pr['pct']:.1f}%")
            ck(S.progress("breakout", None, now) == {}, "没在跑就没有进度")
            (Path(td) / "state" / "lock" / "progress_pullback.json").write_text(
                json.dumps({"running": True, "step": 4, "total": 4, "sub": 0.5,
                            "text": "面板 + 邮件（北京 17:58 发）",
                            "updated_at": "2026-09-29T17:49:00+08:00"}),
                encoding="utf-8")
            pr = S.progress("pullback", {"pid": 1}, now)
            ck(55 <= pr["pct"] <= 65, f"长期调整突破等 17:58 等到一半 -> {pr['pct']:.1f}%")
            # 手点的那一次不写 tools/local_flow_<线>.log：只看日志会一开跑就判「没动静」
            idle = S._idle_min("pullback", dt.datetime(2026, 9, 29, 17, 50))
            ck(idle is not None and idle < 2,
               f"「多久没动静」也看进度文件（手点那次不写日志）-> {idle}")
        finally:
            S.ROOT = real_root

    snap = S.snapshot()
    ck({"now", "date", "wd", "power", "window", "lines", "charts"} <= set(snap),
       "snapshot 字段齐")
    ck([x["name"] for x in snap["lines"]] == ["起涨预测", "长期调整突破"],
       "两条线，顺序固定（早盘选股 / 参数自学 2026-09-27 归档）")
    ck(all({"state", "ago", "can_run", "key"} <= set(x) for x in snap["lines"]),
       "每条线都有 状态 / 多久没跑 / 能不能点 / 按钮键")
    ck({"lists", "pattern"} <= set(snap["charts"]), "两块小图的数据都在")
    ck("__TOKEN__" in S.PAGE and "/full" in S.PAGE, "页面有 token 占位符和详细控制台入口")
    ck("早盘" not in S.PAGE and "会诊" not in S.PAGE, "首页不再出现早盘 / 会诊")


def check_progress_file() -> None:
    """local_run 的进度文件和细进度（首页进度条的数据源）。不碰生产目录。"""
    print("\n[local_run·进度]")
    import tempfile
    import local_run as L
    with tempfile.TemporaryDirectory() as td:
        real = (L.PROGRESS_DIR, L._FLOW)
        try:
            L.PROGRESS_DIR = Path(td)
            L._FLOW = "selftest"
            L._PROG.clear()
            L._PROG.update({"flow": "selftest", "running": True})
            L.step(2, 4, "补日线 + 重算特征表")
            f = Path(td) / "progress_selftest.json"
            j = json.loads(f.read_text(encoding="utf-8"))
            ck(j["step"] == 2 and j["total"] == 4 and j["sub"] is None,
               "step() 写进度文件，换步时细进度清零")
            L._PROG_LAST[0] = 0
            L._scan_sub("12:00:01   逐只处理 2500/5000  用时 3 分钟\n".encode("utf-8"))
            j = json.loads(f.read_text(encoding="utf-8"))
            ck(abs(j["sub"] - 0.475) < 1e-6, f"认出「逐只处理 2500/5000」-> {j['sub']}")
            L._PROG_LAST[0] = 0
            L._scan_sub("逐只处理 1000/5000\n".encode("utf-8"))
            j = json.loads(f.read_text(encoding="utf-8"))
            ck(abs(j["sub"] - 0.475) < 1e-6, "细进度只增不减（晚到的旧行不会把进度条往回拉）")
            # 长期调整突破等 17:58 的那一段：pullback.wait_until 每分钟打一行
            L.step(4, 4, "面板 + 邮件（北京 17:58 发）")
            L._PROG_LAST[0] = 0
            L._scan_sub("等待发信 6/18 分钟\n".encode("utf-8"))
            j = json.loads(f.read_text(encoding="utf-8"))
            ck(abs(j["sub"] - 6 / 18) < 1e-3, f"认出「等待发信 6/18 分钟」-> {j['sub']}")
            src_pb = (ROOT / "src" / "pullback.py").read_text(encoding="utf-8")
            ck('等待发信 {k}/{total} 分钟' in src_pb,
               "pullback.wait_until 打的就是这一行（两边同一个格式）")
            # py() 透传 + 认进度：起一个真的子进程
            L.step(2, 4, "补日线 + 重算特征表")
            L._PROG["sub"] = None
            L._PROG_LAST[0] = 0
            code = ("import sys;print('逐只处理 4000/5000', flush=True);"
                    "sys.stderr.write('stderr 也要透传\\n');sys.exit(3)")
            src = Path(td) / "child.py"
            src.write_text(code, encoding="utf-8")
            rc = L.py(str(src))
            j = json.loads(f.read_text(encoding="utf-8"))
            ck(rc == 3, f"py() 原样返回子进程退出码（{rc}）")
            ck(abs(j["sub"] - 0.73) < 1e-6, f"py() 从子进程输出里认出进度 -> {j['sub']}")
        finally:
            L.PROGRESS_DIR, L._FLOW = real
            L._PROG.clear()


def main() -> int:
    import time
    t0 = time.time()
    check_actions()
    check_flow_tables()
    check_windows()
    check_task_schedule()
    check_no_akshare_in_gui()
    check_target_and_lock()
    check_done_definition()
    check_done_semantics()
    check_daily_coverage()
    check_breakout_flow()
    check_pullback_flow()
    check_line_error()
    check_resend_guard()
    check_scan_and_lock()
    check_push_wiring()
    check_push_paths_not_ignored()
    check_git_lock()
    check_holiday()
    check_wiring_ast()
    check_status_signals()
    check_sync_clears_pull()
    check_trade_dates_cache()
    check_pages_wiring()
    check_archived()
    check_workflow_push()
    check_docs()
    check_simple_page()
    check_progress_file()
    check_panels()
    check_log_time_bj()
    check_task_sentinel()
    check_http()
    print(f"\n耗时 {time.time() - t0:.2f}s | 断言失败 {len(fails)} 个")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
