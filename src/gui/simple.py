"""
傻瓜页（控制台首页，2026-09-18 用户要求）。

一屏，字越少越好：
    每条线一行：按钮 / 跑了没 / 多久没跑 / 进度条
    下面：学习的图
全部北京时间。旧的详细控制台挪到 /full。

数据全部是现成产物，这里只读不写：
    跑没跑、跑到哪   local_run 的锁 + state/lock/progress_<线>.json（local_run 写）
    上次跑通         发信戳 / learning_status.json 的修改时间
    失败原因         tools/local_flow_<线>.log 里最后一次运行那一段
    学习的图         state/learning_status.json 的 daily、state/breakout/truth.json、
                     state/council/latest.json + 台账
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 三条线。窗口 = 该开着电脑的时段（北京），也是页面顶上「下次要开机」的依据。
# 早盘到 09:30 发完信；起涨预测 16:30 起约 20 分钟，参数自学 16:40 起、
# 会诊那天要再加半小时，所以晚上那段给到 17:30。
LINES = [
    {"flow": "morning", "name": "早盘选股", "key": "morning",
     "open": (8, 30), "close": (9, 30)},
    {"flow": "breakout", "name": "起涨预测", "key": "breakout",
     "open": (16, 30), "close": (17, 30)},
    {"flow": "learn", "name": "参数自学", "key": "learn",
     "open": (16, 40), "close": (17, 30)},
]
WINDOWS = [((8, 30), (9, 30)), ((16, 30), (17, 30))]

# 各线每一步占多少进度。起涨预测第 2 步（补日线 + 重算特征）占了九成时间，
# 平均分的话进度条会在 25% 停十几分钟，看着像卡死。
STEP_WEIGHT = {
    "breakout": {1: 2, 2: 88, 3: 7, 4: 3},
    "learn": {1: 2, 2: 95, 3: 3},
}
STALL_MIN = 15      # 在跑、但日志这么多分钟没动 -> 标「没动静」


def now_bj() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=8)


def _bj_of(ts: float) -> dt.datetime:
    return dt.datetime.utcfromtimestamp(ts) + dt.timedelta(hours=8)


def ago(t: dt.datetime | None, now: dt.datetime) -> str:
    if not t:
        return "从没跑通"
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
    """上次真的跑通（发出信 / 学完）的北京时间和目标日。试跑不算。"""
    if flow == "morning":
        p = ROOT / "out" / "mail_sent.json"
        if p.exists():
            return _bj_of(p.stat().st_mtime), str(_json(p).get("date", ""))
        return None, ""
    if flow == "breakout":
        fs = sorted((ROOT / "state" / "sent").glob("breakout_*.json"))
        if fs:
            return _bj_of(fs[-1].stat().st_mtime), fs[-1].stem.split("_", 1)[1]
        return None, ""
    p = ROOT / "state" / "learning_status.json"
    j = _json(p)
    if p.exists() and not j.get("dry"):
        return _bj_of(p.stat().st_mtime), str(j.get("date", ""))
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
    pj = _json(ROOT / "state" / "lock" / f"progress_{flow}.json")
    if not running:
        return {}
    if not pj or not pj.get("running"):
        return {"pct": None, "text": "在跑"}
    step, total = int(pj.get("step") or 0), int(pj.get("total") or 0)
    text = str(pj.get("text") or "")
    sub = pj.get("sub")                         # 0~1，长步骤里的细进度
    if flow == "morning":
        # 早盘大部分时间在等 09:25 采样、09:27:30 发信，按钟点算最诚实
        try:
            t0 = dt.datetime.fromisoformat(str(pj["started_at"])).replace(tzinfo=None)
        except Exception:  # noqa: BLE001
            t0 = now
        end = now.replace(hour=9, minute=27, second=30, microsecond=0)
        span = max(60.0, (end - t0).total_seconds())
        pct = 100 * (now - t0).total_seconds() / span
    elif total:
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


def line(ln: dict, now: dt.datetime) -> dict:
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
    try:
        tds = L.trade_dates()
    except Exception:  # noqa: BLE001
        tds = set()
    trading_today = (not tds or now.strftime("%Y-%m-%d") in tds) and now.weekday() < 5

    out = {"name": ln["name"], "key": ln["key"], "ago": ago(t_ok, now),
           "last_ok": t_ok.strftime("%m-%d %H:%M") if t_ok else "",
           "target": target, "state": "", "note": "", "pct": None, "can_run": True}
    if running:
        pr = progress(flow, running, now)
        out.update(state="running", pct=pr.get("pct"), note=pr.get("text", "在跑"),
                   can_run=False)
        lp = _log(flow)
        if lp.exists():
            idle = (now - _bj_of(lp.stat().st_mtime)).total_seconds() / 60
            if idle >= STALL_MIN:
                out["note"] = f"{int(idle)} 分钟没动静"
                out["state"] = "stalled"
        return out
    if done:
        # 起涨预测和参数自学做的是「最近一个已收盘交易日」那份，北京早上看到的
        # 往往是昨天的，写「今天跑过」会让人以为今天的也有了
        md = target[5:] if len(target) >= 10 else ""
        note = {"morning": "今天已发", "breakout": f"{md} 那份已发",
                "learn": f"{md} 已学完"}.get(flow, "已完成")
        out.update(state="done", note=note, can_run=False)
        return out
    rc, why = last_run_block(flow)
    h = (now.hour, now.minute)
    if rc not in (None, 0) and why:
        out.update(state="failed", note=why[:26])
    elif flow == "morning" and not trading_today:
        out.update(state="off", note="今天不开盘")
    elif h < ln["open"]:
        out.update(state="wait", note=f"{ln['open'][0]:02d}:{ln['open'][1]:02d} 自动跑")
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
            a = dt.datetime(d.year, d.month, d.day, oh, om)
            b = dt.datetime(d.year, d.month, d.day, ch, cm)
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
#  学习的图（只要数，画在前端）
# ---------------------------------------------------------------------
def learn_charts() -> dict:
    st = _json(ROOT / "state" / "learning_status.json")
    morning = [{"d": r.get("date", "")[5:], "v": round(100 * float(r["top_excess"]), 2),
                "n": r.get("sent_n")}
               for r in (st.get("daily") or [])[-20:]
               if isinstance(r.get("top_excess"), (int, float))]
    tr = _json(ROOT / "state" / "breakout" / "truth.json")
    lists = [{"d": str(r.get("date", ""))[5:], "bars": int(r.get("bars") or 0),
              "n": int(r.get("n") or 0),
              "hit": int(r.get("hits_final") if r.get("final") else r.get("hits_sofar") or 0),
              "final": bool(r.get("final"))}
             for r in (tr.get("lists") or [])[-12:]]
    lat = _json(ROOT / "state" / "council" / "latest.json")
    dec = _json(ROOT / "state" / "council" / "decisions.json")
    pend = sum(1 for v in dec.values() if v.get("status") in ("passed", "pending"))
    todo = sum(1 for v in dec.values() if v.get("status") == "needs_human")
    return {"morning": morning, "lists": lists,
            "council": {"date": lat.get("date", ""), "ok": bool(lat.get("ok")),
                        "approve": pend, "todo": todo}}


def snapshot() -> dict:
    now = now_bj()
    return {"now": now.strftime("%H:%M"), "date": now.strftime("%m-%d"),
            "wd": "一二三四五六日"[now.weekday()],
            "power": power(), "window": next_window(now),
            "lines": [line(ln, now) for ln in LINES],
            "learn": learn_charts()}


# ---------------------------------------------------------------------
#  页面
# ---------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股流水线</title>
<style>
:root{--bg:#121417;--card:#1b1e23;--line:#2a2e35;--fg:#e8eaed;--dim:#7d8590;
--ok:#3fb950;--run:#4b8ef0;--bad:#f0524f;--warn:#e3a23b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.4 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:720px;margin:0 auto;padding:18px 16px 40px}
.top{display:flex;align-items:baseline;gap:12px;margin-bottom:6px}
.clock{font-size:34px;font-weight:600;letter-spacing:1px}
.dim{color:var(--dim);font-size:13px}
.bar-top{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 18px}
.pill{padding:5px 10px;border-radius:14px;background:var(--card);font-size:13px}
.pill.bad{background:#3a1f1f;color:#ff8a87}.pill.ok{background:#1d3324;color:#7ee29a}
.pill.warn{background:#3a2e17;color:#f3c26b}
.row{display:grid;grid-template-columns:14px 1fr auto;gap:12px;align-items:center;
background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:10px}
.dot{width:12px;height:12px;border-radius:50%;background:var(--dim)}
.dot.done{background:var(--ok)}.dot.running{background:var(--run);animation:p 1.2s infinite}
.dot.failed,.dot.stalled{background:var(--bad)}.dot.wait,.dot.off{background:#4a4f57}
@keyframes p{50%{opacity:.35}}
.name{font-size:17px;font-weight:600}
.meta{font-size:13px;color:var(--dim);margin-top:2px}
.meta b{color:var(--fg);font-weight:500}
.note.failed,.note.stalled{color:#ff8a87}.note.done{color:#7ee29a}.note.running{color:#8fb8ff}
.prog{height:6px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden}
.prog i{display:block;height:100%;background:var(--run);transition:width .6s}
.prog.ind i{width:30%;animation:ind 1.4s infinite}
@keyframes ind{0%{margin-left:-30%}100%{margin-left:100%}}
button{border:0;border-radius:9px;padding:9px 18px;font-size:15px;font-weight:600;
background:#2f6fe0;color:#fff;cursor:pointer;min-width:72px}
button:disabled{background:#2a2e35;color:#6b7280;cursor:default}
h2{font-size:14px;color:var(--dim);font-weight:500;margin:26px 0 10px}
.card{background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:10px}
svg{display:block;width:100%}
.lists{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px}
.lst{background:#15171b;border-radius:8px;padding:8px;font-size:12px;color:var(--dim)}
.lst b{display:block;font-size:18px;color:var(--fg)}
.mini{height:4px;background:var(--line);border-radius:2px;margin-top:6px;overflow:hidden}
.mini i{display:block;height:100%;background:#6b7280}
a.more{color:var(--dim);font-size:13px}
.links{display:flex;gap:16px;margin-top:22px}
</style></head><body><div class="wrap">
<div class="top"><div class="clock" id="clock">--:--</div><div class="dim" id="date"></div></div>
<div class="dim">北京时间</div>
<div class="bar-top" id="pills"></div>
<div id="lines"></div>
<h2>早盘 · 每天前 10 比全池多赚（%）</h2>
<div class="card"><svg id="mchart" height="120"></svg></div>
<h2>起涨预测 · 每份清单 20 天内涨超 50% 的只数</h2>
<div class="card"><div class="lists" id="lists"></div></div>
<h2>学习会诊</h2>
<div class="card" id="council"></div>
<div class="links"><a class="more" href="/full?token=__TOKEN__">详细控制台</a></div>
</div>
<script>
const TOKEN="__TOKEN__";
const $=s=>document.querySelector(s);
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function run(key,btn){
  btn.disabled=true;btn.textContent="启动中";
  try{
    const r=await fetch("/api/run?token="+TOKEN,{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({key})});
    const j=await r.json();
    if(!r.ok) alert(j.error||"没启动成功");
  }catch(e){alert("没启动成功")}
  setTimeout(load,800);
}
function pills(s){
  const p=[],w=s.window||{},pw=s.power||{};
  if(w.now) p.push(`<span class="pill warn">现在要开着电脑 · 到 ${w.end}</span>`);
  else if(w.start) p.push(`<span class="pill">下次开机 ${esc(w.start)} · 还有 ${esc(w.in)}</span>`);
  if(pw.has_battery){
    if(pw.plugged) p.push(`<span class="pill ok">插着电</span>`);
    else p.push(`<span class="pill bad">没插电 ${pw.pct??""}%</span>`);
  }
  $("#pills").innerHTML=p.join("");
}
function lines(s){
  $("#lines").innerHTML=s.lines.map(l=>{
    const bar=l.state==="running"||l.state==="stalled"
      ?(l.pct==null?`<div class="prog ind"><i></i></div>`
        :`<div class="prog"><i style="width:${l.pct.toFixed(0)}%"></i></div>`):"";
    const btn=`<button ${l.can_run?"":"disabled"} onclick="run('${l.key}',this)">`+
      (l.state==="running"||l.state==="stalled"?"在跑":l.state==="done"?"已完成":"跑")+`</button>`;
    return `<div class="row"><div class="dot ${l.state}"></div><div>
      <div class="name">${esc(l.name)}</div>
      <div class="meta">上次跑通 <b>${esc(l.ago)}</b>
        ${l.note?` · <span class="note ${l.state}">${esc(l.note)}</span>`:""}
        ${l.pct!=null?` · ${l.pct.toFixed(0)}%`:""}</div>${bar}</div>${btn}</div>`;
  }).join("");
}
function mchart(rows){
  const svg=$("#mchart");
  if(!rows.length){svg.innerHTML='<text x="10" y="60" fill="#7d8590" font-size="13">还没有数据</text>';return}
  const W=svg.clientWidth||680,H=120,pad=18,mx=Math.max(1,...rows.map(r=>Math.abs(r.v)));
  const bw=(W-pad*2)/rows.length,mid=H/2;
  let g=`<line x1="${pad}" x2="${W-pad}" y1="${mid}" y2="${mid}" stroke="#2a2e35"/>`;
  rows.forEach((r,i)=>{
    const h=Math.abs(r.v)/mx*(H/2-14),x=pad+i*bw+bw*.15,w=bw*.7;
    const y=r.v>=0?mid-h:mid;
    g+=`<rect x="${x}" y="${y}" width="${w}" height="${Math.max(1,h)}" rx="2"
      fill="${r.v>=0?"#3fb950":"#f0524f"}"><title>${r.d}  ${r.v>0?"+":""}${r.v}%</title></rect>`;
    if(i===0||i===rows.length-1) g+=`<text x="${x+w/2}" y="${H-2}" fill="#7d8590"
      font-size="10" text-anchor="middle">${r.d}</text>`;
  });
  const avg=rows.reduce((a,r)=>a+r.v,0)/rows.length;
  g+=`<text x="${W-pad}" y="12" fill="#7d8590" font-size="11" text-anchor="end">
    ${rows.length} 天平均 ${avg>0?"+":""}${avg.toFixed(2)}%</text>`;
  svg.innerHTML=g;
}
function lists(rows){
  if(!rows.length){$("#lists").innerHTML='<div class="dim">还没有清单</div>';return}
  $("#lists").innerHTML=rows.map(r=>`<div class="lst">${esc(r.d)}
    <b>${r.hit}/${r.n}</b>${r.final?"已到期":`第 ${r.bars}/20 天`}
    <div class="mini"><i style="width:${Math.min(100,r.bars/20*100)}%"></i></div></div>`).join("");
}
function council(c){
  const bits=[];
  if(c.approve) bits.push(`<b>${c.approve}</b> 条等你批`);
  $("#council").innerHTML=`<div class="meta">上次 ${esc(c.date||"没跑过")}
    ${c.date?(c.ok?"· 成功":"· <span class='note failed'>失败</span>"):""}
    ${bits.length?" · "+bits.join(" · "):""}</div>
    <div style="margin-top:10px"><a class="more" href="/panel/council" target="_blank">打开会诊结果</a>
    &nbsp;&nbsp;<a class="more" href="/panel/learn" target="_blank">打开学习面板</a></div>`;
}
async function load(){
  try{
    const s=await (await fetch("/api/simple")).json();
    $("#clock").textContent=s.now;$("#date").textContent=`${s.date} 周${s.wd}`;
    pills(s);lines(s);
    mchart(s.learn.morning);lists(s.learn.lists);council(s.learn.council);
  }catch(e){}
}
load();setInterval(load,5000);
</script></body></html>"""
