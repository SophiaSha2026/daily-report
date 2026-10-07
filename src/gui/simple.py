"""
傻瓜页（控制台首页，2026-09-18 用户要求）。

一屏，字越少越好：
    每条线一行：按钮 / 跑了没 / 多久没跑 / 进度条
    下面：两张小图
全部北京时间。旧的详细控制台挪到 /full。

2026-09-27 早盘系统（早盘选股 + 参数自学 + 学习会诊）整体归档，这一页只剩
晚间两条线：起涨预测、长期调整突破。

数据全部是现成产物，这里只读不写：
    跑没跑、跑到哪   local_run 的锁 + state/lock/progress_<线>.json（local_run 写）
    上次跑通         state/sent/<线>_<日>.json 的修改时间
    失败原因         tools/local_flow_<线>.log 里最后一次运行那一段
    小图             state/breakout/truth.json、out_pullback/history.json
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

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


def now_bj() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=8)


def _bj_of(ts: float) -> dt.datetime:
    return dt.datetime.utcfromtimestamp(ts) + dt.timedelta(hours=8)


def ago(t: dt.datetime | None, now: dt.datetime) -> str:
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
    ts = []
    lp = _log(flow)
    if lp.exists():
        ts.append(_bj_of(lp.stat().st_mtime))
    pj = _json(ROOT / "state" / "lock" / f"progress_{flow}.json")
    try:
        ts.append(dt.datetime.fromisoformat(str(pj["updated_at"])).replace(tzinfo=None))
    except Exception:  # noqa: BLE001
        pass
    if not ts:
        return None
    return (now - max(ts)).total_seconds() / 60


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
            "charts": charts()}


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
.lists{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:8px}
.lst{background:#15171b;border-radius:8px;padding:8px;font-size:12px;color:var(--dim)}
.lst b{display:block;font-size:18px;color:var(--fg)}
.lst.new{outline:1px solid #2f6fe0}
.mini{height:4px;background:var(--line);border-radius:2px;margin-top:6px;overflow:hidden}
.mini i{display:block;height:100%;background:#6b7280}
a.more{color:var(--dim);font-size:13px}
.links{display:flex;gap:16px;margin-top:22px;flex-wrap:wrap}
</style></head><body><div class="wrap">
<div class="top"><div class="clock" id="clock">--:--</div><div class="dim" id="date"></div></div>
<div class="dim">北京时间</div>
<div class="bar-top" id="pills"></div>
<div id="lines"></div>
<h2>查一只股票 · 用起涨预测的模型给它打分</h2>
<div class="card"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
<input id="pcode" inputmode="numeric" maxlength="6" placeholder="6 位代码，如 300829"
 style="font:inherit;padding:8px 10px;width:150px;background:transparent;color:inherit;border:1px solid var(--line);border-radius:6px"
 onkeydown="if(event.key==='Enter')predict()">
<button onclick="predict()" id="pbtn">预测</button><span class="dim" id="phint"></span></div>
<pre id="pout" class="dim" style="white-space:pre-wrap;margin:10px 0 0;font:13px/1.5 ui-monospace,Consolas,monospace"></pre></div>
<h2>起涨预测 · 每份清单 20 天内涨超 50% 的只数</h2>
<div class="card"><div class="lists" id="lists"></div></div>
<h2 id="pt">长期调整突破 · 今天的和以前成立过的</h2>
<div class="card"><div class="lists" id="pattern"></div><div class="meta" id="pm"></div></div>
<div class="links"><a class="more" href="/full?token=__TOKEN__">详细控制台</a>
<a class="more" href="/panel/breakout" target="_blank">起涨预测面板</a>
<a class="more" href="/panel/pullback" target="_blank">长期调整突破面板</a></div>
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
function lists(rows){
  if(!rows.length){$("#lists").innerHTML='<div class="dim">还没有清单</div>';return}
  $("#lists").innerHTML=rows.map(r=>`<div class="lst">${esc(r.d)}
    <b>${r.hit}/${r.n}</b>${r.final?"已到期":`第 ${r.bars}/20 天`}
    <div class="mini"><i style="width:${Math.min(100,r.bars/20*100)}%"></i></div></div>`).join("");
}
// 走满 20 个交易日的写「20 天最高」，还在走的写「至今最高」：两者不是一回事
function upTxt(r){
  if(r.up==null) return "";
  const v=(r.up>0?"+":"")+Number(r.up).toFixed(1)+"%";
  return (r.n_after!=null&&r.n_after<20?"至今最高 ":"20 天最高 ")+v;
}
function pattern(p){
  const rows=p.recent||[];
  $("#pattern").innerHTML=rows.length?rows.map(r=>`<div class="lst${r.new?" new":""}">${esc(r.d)}
    <b>${esc(r.code)}</b>${esc(r.name||"")}<div>${r.new?"今天":upTxt(r)}</div></div>`).join("")
    :'<div class="dim">还没有</div>';
  const bits=[];
  if(p.per_month!=null) bits.push(`三年同口径 ${p.n} 次，约每月 ${p.per_month} 次，空榜是常态`);
  if(p.n_today!=null) bits.push(`${esc((p.date||"").slice(5))} 清单 A ${p.n_today} 只`
    +(p.n_b!=null?` · B（二次进攻前）${p.n_b} 只`:""));
  $("#pm").textContent=bits.join(" · ");
}
async function predict(){
  const code=$("#pcode").value.trim();
  if(!/^\d{6}$/.test(code)){$("#phint").textContent="要 6 位数字";return}
  const b=$("#pbtn");b.disabled=true;$("#phint").textContent="算中（几秒）";$("#pout").textContent="";
  try{
    const r=await (await fetch("/api/predict?code="+code)).json();
    $("#phint").textContent="";
    if(!r.ok){$("#pout").textContent=r.error||"失败";return}
    const bd={main:"主板",star:"科创",chinext:"创业",bj:"北交"}[r.board]||r.board;
    const L=[];
    L.push(`${r.code} ${r.name||""}  打分日 ${r.score_date}`+(r.stale?`（这只票最后一行是 ${r.last_row}）`:""));
    L.push(`分数 ${r.score} / 100（预测值 ${r.p.toFixed(4)}，含${bd}系数 ${r.board_adj.toFixed(3)}）  全市场第 ${r.rank??"-"} / ${r.n_pool}，当天够格 ${r.n_qualified} 只`);
    if(r.is_st)L.push("ST：清单 A 不收");else if(!r.eligible)L.push("上市不足 120 个交易日：清单 A 不收");
    L.push((r.qualified?"够格：会上清单 A":`不够格（上清单要 ≥${r.score_min} 分且当天前 ${r.cap_a} 名）`)+`  连续够格 ${r.streak} 天`);
    L.push(`近 ${r.history.length} 天分数  `+r.history.map(h=>h.date.slice(5)+":"+h.score).join(" "));
    if(r.perf_bin){const p=r.perf_bin;L.push(`历史：${p.lo}~${p.hi-1} 分这档验证集命中 ${(100*p.hit).toFixed(1)}%（随便买 ${r.base}%，${p.lift.toFixed(1)} 倍，n=${p.n}）`)}
    if(r.qualified){const p=r.perf_streak;L.push(`      连续够格 ${p.k} 天这档：${p.hit.toFixed(1)}%（${p.lift.toFixed(1)} 倍）`)}
    L.push("注："+r.note);
    $("#pout").textContent=L.join("\n");
  }catch(e){$("#phint").textContent="";$("#pout").textContent="请求失败："+e}
  finally{b.disabled=false}
}
async function load(){
  try{
    const s=await (await fetch("/api/simple")).json();
    $("#clock").textContent=s.now;$("#date").textContent=`${s.date} 周${s.wd}`;
    pills(s);lines(s);
    lists(s.charts.lists);pattern(s.charts.pattern);
  }catch(e){}
}
load();setInterval(load,5000);
</script></body></html>"""
