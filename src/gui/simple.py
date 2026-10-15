"""
傻瓜页（控制台首页，2026-09-18 用户要求）。

一屏，字越少越好（2026-10-11 再瘦一轮，页面中文 315 -> 150 以内）：
    两条线各一张卡片：名称 / 状态词 / 一句话 / 跑按钮，在跑时才有进度条
    查一只股票：输入框 + 预测按钮 + AI 讨论（开 /chat，2026-10-13 从「AI 分析」升级）
    学习：一句话状态 + 等你决定的事（同意 / 不同意 / 现在重学）+ 问 AI（gui/learn.py，2026-10-13）
    一行系统状态：下次开机、插没插电、计划任务（/api/tasks，红了链到 /full 的计划任务区块）、推送成没成
    一张小图：起涨预测每份清单的命中数
全部北京时间。详细控制台在 /full。

2026-09-27 早盘系统（早盘选股 + 参数自学 + 学习会诊）整体归档，这一页只剩
晚间两条线：起涨预测、长期调整突破。

数据全部是现成产物，这里只读不写：
    跑没跑、跑到哪   local_run 的锁 + state/lock/progress_<线>.json（local_run 写）
    上次跑通         state/sent/<线>_<日>.json 的修改时间
    失败原因         tools/local_flow_<线>.log 里最后一次运行那一段
    小图             state/breakout/truth.json（长期调整突破那句「以前成立过 N 次」读
                     out_pullback/run_meta.json）
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from bjtime import BJ, now_bj  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 两条线。open = 自动开跑时刻（北京），close = 这条线通常做完的时刻。
# 起涨预测 16:30 起约 20 分钟；长期调整突破 17:40 起、17:58 发信。
LINES = [
    {"flow": "breakout", "name": "起涨预测", "key": "breakout",
     "open": (16, 30), "close": (17, 30), "wait": "16:30 自动跑"},
    {"flow": "pullback", "name": "长期调整突破", "key": "pullback",
     "open": (17, 40), "close": (18, 5), "wait": "17:58 发信"},
]
# 该开着电脑的时段（北京），页面顶上「下次要开机」的依据
WINDOWS = [((16, 30), (18, 5))]

# 各线每一步占多少进度。起涨预测第 2 步（补日线 + 重算特征）占了九成时间，
# 平均分的话进度条会在 25% 停十几分钟，看着像卡死。长期调整突破大部分时间
# 在第 4 步等 17:58，那一步按「等待发信 k/N 分钟」走（local_run._SUB）。
STEP_WEIGHT = {
    "breakout": {1: 2, 2: 88, 3: 7, 4: 3},
    "pullback": {1: 3, 2: 12, 3: 5, 4: 80},
}
STALL_MIN = 15      # 在跑、但日志和进度文件这么多分钟都没动 -> 标「没动静」


def _bj_of(ts: float) -> dt.datetime:
    """文件时间戳 -> 北京时间（带时区，和 now_bj 同类，能直接相减）。"""
    return dt.datetime.fromtimestamp(ts, tz=BJ)


def _aware(now: dt.datetime) -> dt.datetime:
    """自测会传 naive 的北京时间；生产传的是 now_bj()（带时区）。统一成带时区再比较。"""
    return now if now.tzinfo is not None else now.replace(tzinfo=BJ)


def ago(t: dt.datetime | None, now: dt.datetime) -> str:
    now = _aware(now)
    t = _aware(t) if t is not None else None
    if not t:
        return "还没有"
    m = int((now - t).total_seconds() // 60)
    if m < 1:
        return "刚刚"
    if m < 60:
        return f"{m} 分钟前"
    h = m // 60
    if h < 48:
        return f"{h} 小时 {m % 60} 分前"
    return f"{h // 24} 天前"


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------
#  每条线
# ---------------------------------------------------------------------
def last_ok(flow: str) -> tuple[dt.datetime | None, str]:
    """上次真的跑通（发出信）的北京时间和目标日。试跑不写 sent 标记，不算。"""
    fs = sorted((ROOT / "state" / "sent").glob(f"{flow}_*.json"))
    if fs:
        return _bj_of(fs[-1].stat().st_mtime), fs[-1].stem.split("_", 1)[1]
    return None, ""


def _log(flow: str) -> Path:
    return ROOT / "tools" / f"local_flow_{flow}.log"


def last_run_block(flow: str) -> tuple[int | None, str]:
    """日志里最后一次真正开跑（不是秒退的敲门）的退出码和一句原因。"""
    p = _log(flow)
    if not p.exists():
        return None, ""
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")[-400_000:]
    except Exception:  # noqa: BLE001
        return None, ""
    blocks = re.split(r"\n(?=\[[\d\-T:]+\] === run_local \w+ ===)", txt)
    for b in reversed(blocks):
        if "本地全流程" not in b:        # 秒退的敲门（没到点 / 已跑完 / 在跑）
            continue
        m = re.search(r"=== run_local \w+ exit=(\d+) ===", b)
        rc = int(m.group(1)) if m else None
        why = ""
        for ln in reversed(b.splitlines()):
            if re.search(r"不出清单|失败|放弃|错误|ERROR|Traceback|超过", ln):
                why = re.sub(r"^\S+\s+(\[\w+\]\s+)?", "", ln).strip()
                break
        return rc, why
    return None, ""


def progress(flow: str, running: dict | None, now: dt.datetime) -> dict:
    """进度 0~100 和一句在干什么。没有进度文件（老代码在跑）就只给不定进度。"""
    now = _aware(now)
    pj = _json(ROOT / "state" / "lock" / f"progress_{flow}.json")
    if not running:
        return {}
    if not pj or not pj.get("running"):
        return {"pct": None, "text": "在跑"}
    step, total = int(pj.get("step") or 0), int(pj.get("total") or 0)
    text = str(pj.get("text") or "")
    sub = pj.get("sub")                         # 0~1，长步骤里的细进度
    if total:
        w = STEP_WEIGHT.get(flow) or {i: 1 for i in range(1, total + 1)}
        tot = float(sum(w.get(i, 1) for i in range(1, total + 1)))
        done = sum(w.get(i, 1) for i in range(1, step))
        cur = w.get(step, 1) * (float(sub) if isinstance(sub, (int, float)) else 0.3)
        pct = 100 * (done + cur) / tot
    else:
        pct = None
    if pct is not None:
        pct = max(1.0, min(99.0, pct))
    return {"pct": pct, "text": re.sub(r"（.*?）", "", text)[:18]}


def _idle_min(flow: str, now: dt.datetime) -> float | None:
    """日志和进度文件里较新的那个，离现在多少分钟。都没有就 None。

    只看日志不够：控制台手点的那一次输出进的是控制台自己的任务窗口，不写
    tools/local_flow_<线>.log，而那个日志停在上一次计划任务的时刻，手点一开跑
    就会被判成「很久没动静」。进度文件两条路都写。
    """
    now = _aware(now)
    ts = []
    lp = _log(flow)
    if lp.exists():
        ts.append(_bj_of(lp.stat().st_mtime))
    pj = _json(ROOT / "state" / "lock" / f"progress_{flow}.json")
    try:
        t = dt.datetime.fromisoformat(str(pj["updated_at"]))
        ts.append(t if t.tzinfo else t.replace(tzinfo=BJ))   # 进度文件写的是北京时间
    except Exception:  # noqa: BLE001
        pass
    if not ts:
        return None
    return (now - max(ts)).total_seconds() / 60


def line(ln: dict, now: dt.datetime) -> dict:
    now = _aware(now)
    import local_run as L
    flow = ln["flow"]
    t_ok, d_ok = last_ok(flow)
    try:
        target = L.target_date(flow)
    except Exception:  # noqa: BLE001
        target = ""
    try:
        done = bool(target) and L.already_done(flow)
    except Exception:  # noqa: BLE001
        done = bool(target) and d_ok == target
    try:
        running = L.running_instance(flow)
    except Exception:  # noqa: BLE001
        running = None

    out = {"name": ln["name"], "key": ln["key"], "ago": ago(t_ok, now),
           "last_ok": t_ok.strftime("%m-%d %H:%M") if t_ok else "",
           "target": target, "state": "", "note": "", "pct": None, "can_run": True}
    if running:
        pr = progress(flow, running, now)
        out.update(state="running", pct=pr.get("pct"), note=pr.get("text", "在跑"),
                   can_run=False)
        idle = _idle_min(flow, now)
        if idle is not None and idle >= STALL_MIN:
            out["note"] = f"{int(idle)} 分钟没动静"
            out["state"] = "stalled"
        return out
    since = getattr(L, "START", {}).get(flow, "")
    if since and target and target < since:
        out.update(state="wait", note=f"{since[5:]} 起每天 {ln['wait'][:5]}")
        return out
    if done:
        # 两条线做的都是「最近一个已收盘交易日」那份，北京早上看到的往往是
        # 昨天的，写「今天跑过」会让人以为今天的也有了
        md = target[5:] if len(target) >= 10 else ""
        out.update(state="done", note=f"{md} 那份已发", can_run=False)
        return out
    rc, why = last_run_block(flow)
    h = (now.hour, now.minute)
    if rc not in (None, 0) and why:
        out.update(state="failed", note=why[:26])
    elif h < ln["open"] and h >= (8, 30):
        out.update(state="wait", note=ln["wait"])
    else:
        out.update(state="wait", note="到点了，等计划任务")
    return out


# ---------------------------------------------------------------------
#  顶上：电源 + 下次要开机
# ---------------------------------------------------------------------
def power() -> dict:
    """插没插电、电量。Windows 的 GetSystemPowerStatus，别的系统返回空。

    2026-09-17 17:02 起涨预测跑到一半停了 15 小时，Windows 事件日志写的是
    Sleep Reason: Battery —— 没插电、电量低到阈值自动休眠。程序拦不住
    这种保护性休眠，只能让人在该开机的时段插上电。
    """
    try:
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = SPS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return {}
        pct = int(s.BatteryLifePercent)
        return {"plugged": s.ACLineStatus == 1,
                "pct": None if pct == 255 else pct,
                "has_battery": s.BatteryFlag != 128}
    except Exception:  # noqa: BLE001
        return {}


def next_window(now: dt.datetime) -> dict:
    """现在在不在该开机的时段；不在的话下一段是几点、还有多久。"""
    now = _aware(now)
    try:
        import local_run as L
        tds = L.trade_dates()
    except Exception:  # noqa: BLE001
        tds = set()

    def trading(d: dt.date) -> bool:
        return d.weekday() < 5 and (not tds or d.isoformat() in tds)

    for add in range(0, 10):
        d = (now + dt.timedelta(days=add)).date()
        if not trading(d):
            continue
        for (oh, om), (ch, cm) in WINDOWS:
            a = dt.datetime(d.year, d.month, d.day, oh, om, tzinfo=BJ)
            b = dt.datetime(d.year, d.month, d.day, ch, cm, tzinfo=BJ)
            if a <= now < b:
                return {"now": True, "start": a.strftime("%H:%M"),
                        "end": b.strftime("%H:%M")}
            if now < a:
                mins = int((a - now).total_seconds() // 60)
                when = ("今天" if add == 0 else "明天" if add == 1
                        else f"{d.month}-{d.day} 周{'一二三四五六日'[d.weekday()]}")
                return {"now": False, "start": f"{when} {a:%H:%M}",
                        "end": b.strftime("%H:%M"),
                        "in": f"{mins // 60} 小时 {mins % 60} 分" if mins >= 60
                        else f"{mins} 分钟"}
    return {}


# ---------------------------------------------------------------------
#  小图（只要数，画在前端）
# ---------------------------------------------------------------------
def charts() -> dict:
    tr = _json(ROOT / "state" / "breakout" / "truth.json")
    lists = [{"d": str(r.get("date", ""))[5:], "bars": int(r.get("bars") or 0),
              "n": int(r.get("n") or 0),
              "hit": int(r.get("hits_final") if r.get("final") else r.get("hits_sofar") or 0),
              "final": bool(r.get("final"))}
             for r in (tr.get("lists") or [])[-12:]]
    meta = _json(ROOT / "out_pullback" / "run_meta.json")
    try:
        hist = json.loads((ROOT / "out_pullback" / "history.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        hist = []
    try:
        today = json.loads((ROOT / "out_pullback" / "selected.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        today = []
    recent = ([{"d": str(r.get("date", ""))[5:], "code": str(r.get("code", "")).zfill(6),
                "name": r.get("name", ""), "up": None, "new": True} for r in today]
              + [{"d": str(r.get("date", ""))[5:], "code": str(r.get("code", "")).zfill(6),
                  "name": r.get("name", ""), "up": r.get("max_up_pct"),
                  "n_after": r.get("n_after"), "new": False}
                 for r in reversed(hist)])[:8]
    h = meta.get("hist") or {}
    return {"lists": lists,
            "pattern": {"recent": recent, "per_month": h.get("per_month"),
                        "n": h.get("n"), "n_today": meta.get("n"), "n_b": meta.get("n_b"),
                        "date": meta.get("date", "")}}


def snapshot() -> dict:
    now = now_bj()
    return {"now": now.strftime("%H:%M"), "date": now.strftime("%m-%d"),
            "wd": "一二三四五六日"[now.weekday()],
            "power": power(), "window": next_window(now),
            "lines": [line(ln, now) for ln in LINES],
            "charts": charts(), "ai": _ai_usage()}


def _ai_usage() -> dict:
    """claude.exe 的用量（今天 / 本月美元，超警戒线标红）。读不到就空。"""
    try:
        import ai_usage
        t = ai_usage.totals()
        return {"today": t["today_usd"], "month": t["month_usd"], "n": t["today_n"], "warn": t["warn"]}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------
#  页面
#
# 2026-10-11 瘦身（用户：「文字越少越好，界面越干净越好，只留必要交互」）。
# 页面里只放给人看的字，说明写在这里，不写进 JS（JS 注释也算进页面字数）：
#   - 状态词只有四个：已完成 / 在跑 / 还没跑 / 出错。stalled（在跑但
#     STALL_MIN 分钟没动静）也叫「在跑」，状态块标红，一句话里写多久没动。
#   - 「清单 N 只」：起涨预测从小图数据（truth.json 的 lists）里找目标日那份，
#     长期调整突破读 charts.pattern（它的 run_meta）。都在 /api/simple 里。
#   - 「以前成立过 N 次」只给长期调整突破：空榜是常态，不写这一句会以为坏了。
#   - 「推送失败」只在 /api/status 里（sync.last_push_ok），那个接口最多每分钟
#     git fetch 一次，首页一分钟问一次，红了链到 /full 的「修 -> 重试推送」。
#   - 底部两个面板链接按 /api/simple 的 lines 拼（名称和键都从 LINES 来，
#     页面里不再写死线名）。
#   - 长期调整突破「今天的和以前成立过的」那张表 2026-10-11 删掉，charts() 的
#     pattern.recent 留着没删（后端数据结构不为界面改）。
PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股流水线</title>
<style>
:root{--bg:#121417;--card:#1b1e23;--line:#2a2e35;--fg:#e8eaed;--dim:#7d8590;--run:#4b8ef0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.45 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:16px 16px 28px}
.top{display:flex;align-items:baseline;gap:10px;margin-bottom:14px}
.clock{font-size:32px;font-weight:600;letter-spacing:1px}
.dim{color:var(--dim);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
.card{background:var(--card);border-radius:12px;padding:14px 16px}
.hd{display:flex;align-items:center;gap:10px}
.name{font-size:17px;font-weight:600}
.st{font-size:12px;padding:2px 9px;border-radius:10px;background:var(--line);color:var(--dim)}
.st.done{background:#1d3324;color:#7ee29a}.st.running{background:#1b2a44;color:#8fb8ff}
.st.failed,.st.stalled{background:#3a1f1f;color:#ff8a87}
.hd button{margin-left:auto}
.say{font-size:13px;color:var(--dim);margin-top:8px}
.say .bad{color:#ff8a87}
.prog{height:6px;background:var(--line);border-radius:3px;margin-top:10px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--run);transition:width .6s}
.prog.ind i{width:30%;animation:ind 1.4s infinite}
@keyframes ind{0%{margin-left:-30%}100%{margin-left:100%}}
button{border:0;border-radius:8px;padding:7px 16px;font-family:inherit;font-size:14px;
font-weight:600;background:#2f6fe0;color:#fff;cursor:pointer}
button:disabled{background:var(--line);color:#6b7280;cursor:default}
.ask{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px}
input{font:inherit;padding:6px 10px;width:150px;background:transparent;color:inherit;
border:1px solid var(--line);border-radius:8px}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--dim)}
#pout{white-space:pre-wrap;margin:8px 0 0;font:13px/1.5 ui-monospace,Consolas,monospace;color:var(--dim)}
#pout:empty{display:none}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.pill{padding:4px 10px;border-radius:14px;background:var(--card);font-size:13px;color:var(--fg);text-decoration:none}
.pill.ok{background:#1d3324;color:#7ee29a}.pill.bad{background:#3a1f1f;color:#ff8a87}
.pill.warn{background:#3a2e17;color:#f3c26b}
h2{font-size:13px;color:var(--dim);font-weight:500;margin:0 0 8px}
.lists{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:6px}
.lst{background:#15171b;border-radius:8px;padding:6px 8px;font-size:12px;color:var(--dim)}
.lst b{display:block;font-size:17px;color:var(--fg)}
.mini{height:3px;background:var(--line);border-radius:2px;margin-top:5px;overflow:hidden}
.mini i{display:block;height:100%;background:#6b7280}
.learn{margin-top:10px}.learn .say div{margin:2px 0}
.dec{border-top:1px solid var(--line);margin-top:10px;padding-top:10px;font-size:14px}
.dec .why{font-size:12px;color:var(--dim);margin:4px 0 6px}.dec summary{cursor:pointer}
.dec button{padding:5px 12px;font-size:13px;margin-right:6px}
.hd a.ai{margin-left:6px;font-size:13px;color:var(--dim);text-decoration:none;border:1px solid var(--line);
border-radius:8px;padding:5px 10px}
.links{display:flex;gap:16px;margin-top:16px;flex-wrap:wrap}
.links a{color:var(--dim);font-size:13px}
</style></head><body><div class="wrap">
<div class="top"><div class="clock" id="clock">--:--</div><div class="dim" id="date"></div></div>
<div class="cards" id="lines"></div>
<div class="ask"><input id="pcode" inputmode="numeric" maxlength="6" placeholder="代码"
 onkeydown="if(event.key==='Enter')predict()"><button onclick="predict()" id="pbtn">预测</button>
<button onclick="aiChat()" class="ghost">AI 讨论</button>
<span class="dim" id="phint"></span></div>
<pre id="pout"></pre>
<div class="card learn" id="learn" hidden><div class="hd"><span class="name">学习</span>
<span class="st" id="lhead"></span><a class="ai" href="/chat?kind=learn&token=__TOKEN__" target="_blank">问 AI</a></div>
<div class="say" id="llines"></div><div id="ldec"></div></div>
<div class="pills" id="pills"></div>
<div class="card" id="chart" hidden><h2>起涨预测清单 · 20 天内涨超 50% 的只数</h2>
<div class="lists" id="lists"></div></div>
<div class="links" id="links"><a href="/full?token=__TOKEN__">详细控制台</a></div>
</div>
<script>
const TOKEN="__TOKEN__";
const $=s=>document.querySelector(s);
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
const WORD={done:"已完成",running:"在跑",failed:"出错",wait:"还没跑"};WORD.stalled=WORD.running;
let snap=null,push=null;
async function run(key,btn){
  btn.disabled=true;btn.textContent="…";
  try{
    const r=await fetch("/api/run?token="+TOKEN,{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({key})});
    const j=await r.json();
    if(!r.ok) alert(j.error||r.status);
  }catch(e){alert(e)}
  setTimeout(load,800);
}
function count(l,c){
  const md=(l.target||"").slice(5);
  if(l.key==="pullback") return (c.pattern.date||"").slice(5)===md?c.pattern.n_today:null;
  const x=(c.lists||[]).find(r=>r.d===md);
  return x?x.n:null;
}
function say(l,c){
  const b=[],red=l.state==="failed"||l.state==="stalled";
  if(l.state!=="done"&&l.note) b.push(red?`<span class="bad">${esc(l.note)}</span>`:esc(l.note));
  if(l.pct!=null) b.push(l.pct.toFixed(0)+"%");
  b.push("上次跑通 "+esc(l.ago));
  if(l.target) b.push("目标日 "+esc(l.target.slice(5)));
  const n=l.state==="done"?count(l,c):null;
  if(n!=null) b.push(`清单 ${n} 只`);
  if(l.key==="pullback"&&c.pattern.n!=null) b.push(`以前成立过 ${c.pattern.n} 次`);
  return b.join(" · ");
}
function lines(s){
  $("#lines").innerHTML=s.lines.map(l=>{
    const on=l.state==="running"||l.state==="stalled";
    const bar=!on?"":l.pct==null?`<div class="prog ind"><i></i></div>`
      :`<div class="prog"><i style="width:${l.pct.toFixed(0)}%"></i></div>`;
    return `<div class="card"><div class="hd"><span class="name">${esc(l.name)}</span>
      <span class="st ${l.state}">${WORD[l.state]||esc(l.state)}</span>
      <button ${l.can_run?"":"disabled"} onclick="run('${l.key}',this)">跑</button>
      <a class="ai" href="/chat?kind=${esc(l.key)}&token=${TOKEN}" target="_blank">AI</a></div>
      <div class="say">${say(l,s.charts)}</div>${bar}</div>`;
  }).join("");
  const lk=$("#links");
  if(!lk.dataset.done){lk.dataset.done=1;lk.insertAdjacentHTML("beforeend",s.lines.map(l=>
    `<a href="/panel/${esc(l.key)}" target="_blank">${esc(l.name)}面板</a>`).join(""))}
}
function pills(s){
  const p=[],w=s.window||{},pw=s.power||{};
  if(w.now) p.push(`<span class="pill warn">保持开机到 ${esc(w.end)}</span>`);
  else if(w.start) p.push(`<span class="pill">下次开机 ${esc(w.start)}–${esc(w.end)}</span>`);
  if(pw.has_battery) p.push(pw.plugged?`<span class="pill ok">插着电</span>`
    :`<span class="pill bad">没插电 ${pw.pct??""}%</span>`);
  if(tasks) p.push(tasks.ok?`<a class="pill ok" href="/full?token=${TOKEN}#d-tasks">计划任务正常</a>`
    :`<a class="pill bad" href="/full?token=${TOKEN}#d-tasks">${esc(tasks.head)}</a>`);
  if(push===false) p.push(`<a class="pill bad" href="/full?token=${TOKEN}">推送失败</a>`);
  else if(push===true) p.push(`<span class="pill ok">推送正常</span>`);
  const ai=s.ai||{};
  if(ai.n) p.push(`<span class="pill ${ai.warn?"bad":""}">AI 今天 ${ai.today} 美元</span>`);
  $("#pills").innerHTML=p.join("");
}
function chart(rows){
  $("#chart").hidden=!rows.length;
  $("#lists").innerHTML=rows.map(r=>`<div class="lst">${esc(r.d)}<b>${r.hit}/${r.n}</b>
    第 ${r.final?20:r.bars} 天<div class="mini"><i style="width:${Math.min(100,r.bars/20*100)}%"></i></div></div>`).join("");
}
function aiChat(){
  const code=$("#pcode").value.trim();
  if(!/^\d{6}$/.test(code)){$("#phint").textContent="6 位数字";return}
  window.open(`/chat?kind=stock&code=${code}&token=${TOKEN}`,"_blank");
}
async function learn(){
  try{
    const s=await (await fetch("/api/learn")).json();
    $("#learn").hidden=false;
    const n=(s.decide||[]).length;
    $("#lhead").textContent=s.head;$("#lhead").className="st "+(n?"stalled":s.drift?"running":"done");
    $("#llines").innerHTML=s.lines.map(x=>`<div>${esc(x)}</div>`).join("")+`<div>${esc(s.pullback.line)}</div>`;
    const all=s.decide||[],rest=all.length>3?`<div class="dec dim"><a href="/full?token=${TOKEN}" style="color:inherit">还有 ${all.length-3} 件，去详细控制台</a></div>`:"";
    $("#ldec").innerHTML=all.slice(0,3).map(d=>{
      if(d.kind==="relearn") return `<div class="dec">${esc(d.text)}<div class="why"></div>
        <button onclick="relearn(this)">现在重学</button></div>`;
      const more=[d.detail,d.effect&&"预期："+d.effect,d.risk&&"风险："+d.risk,
        (d.files||[]).length&&"改："+d.files.join("、"),d.note].filter(Boolean).map(esc).join("<br>");
      const chat=d.session?` <a class="ai" href="/chat?sid=${esc(d.session)}&token=${TOKEN}" target="_blank">看讨论</a>`:"";
      return `<div class="dec">${esc(d.text)}<details class="why"><summary>细节</summary>${more}</details>
        <button onclick="decide('${esc(d.id)}','approve',this)">同意</button>
        <button class="ghost" onclick="decide('${esc(d.id)}','reject',this)">不同意</button>${chat}</div>`;
    }).join("")+rest;
  }catch(e){}
}
async function decide(id,action,b){
  if(action==="approve"&&!confirm("同意后会执行这条建议：参数会改并重算邮件里的准确率；代码会改文件、跑自测，不过就撤回。"))return;
  b.disabled=true;
  try{
    const r=await fetch("/api/proposal?token="+TOKEN,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({id,action})});
    const j=await r.json();if(!r.ok)alert(j.error||r.status);
  }catch(e){alert(e)}
  setTimeout(learn,800);
}
async function relearn(b){
  if(!confirm("现在重学？约 20 分钟，起涨预测在跑时不能点。"))return;
  b.disabled=true;
  try{
    const r=await fetch("/api/run?token="+TOKEN,{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({key:"learn_now"})});
    const j=await r.json();b.textContent=r.ok?"在重学":"没开始";if(!r.ok)alert(j.error||r.status);
  }catch(e){alert(e)}
}
async function predict(){
  const code=$("#pcode").value.trim();
  if(!/^\d{6}$/.test(code)){$("#phint").textContent="6 位数字";return}
  const b=$("#pbtn");b.disabled=true;$("#phint").textContent="…";$("#pout").textContent="";
  try{
    const r=await (await fetch("/api/predict?code="+code)).json();
    $("#phint").textContent="";
    if(!r.ok){$("#pout").textContent=String(r.error);return}
    const L=[];
    L.push(`${r.code} ${r.name||""}  ${r.score_date}`+(r.stale?`（数据到 ${r.last_row}）`:""));
    L.push(`分数 ${r.score}/100  第 ${r.rank??"-"}/${r.n_pool}  当天够格 ${r.n_qualified} 只`);
    L.push(`预测值 ${r.p.toFixed(4)} × 板块系数 ${r.board_adj.toFixed(3)}`);
    if(r.is_st||!r.eligible)L.push((r.is_st?"ST":"上市不足 120 个交易日")+"，不进清单 A");
    const k=r.perf_streak;
    L.push((r.qualified?"够格，会上清单 A":`不够格：要 ≥${r.score_min} 分且前 ${r.cap_a} 名`)+`  连续够格 ${r.streak} 天`
      +(r.qualified?`（这档 ${k.hit.toFixed(1)}%，${k.lift.toFixed(1)} 倍）`:""));
    L.push(r.history.map(h=>h.date.slice(5)+":"+h.score).join(" "));
    if(r.perf_bin){const p=r.perf_bin;L.push(`${p.lo}~${p.hi-1} 分历史命中 ${(100*p.hit).toFixed(1)}%（随便买 ${r.base}%，${p.lift.toFixed(1)} 倍，n=${p.n}）`)}
    L.push(r.note);
    $("#pout").textContent=L.join("\n");
  }catch(e){$("#phint").textContent="";$("#pout").textContent=String(e)}
  finally{b.disabled=false}
}
async function load(){
  try{
    const s=await (await fetch("/api/simple")).json();
    $("#clock").textContent=s.now;$("#date").textContent=`北京 ${s.date} 周${s.wd}`;
    snap=s;lines(s);pills(s);chart(s.charts.lists||[]);
  }catch(e){}
}
let tasks=null;
async function loadTasks(){
  try{tasks=await (await fetch("/api/tasks")).json();if(snap)pills(snap)}catch(e){}
}
async function loadPush(){
  try{const s=await (await fetch("/api/status")).json();push=s.sync.last_push_ok;if(snap)pills(snap)}
  catch(e){}
}
load();loadPush();learn();loadTasks();setInterval(load,5000);setInterval(loadPush,60000);setInterval(learn,60000);
setInterval(loadTasks,60000);
</script></body></html>"""
