"""
控制台的单页界面。

为什么把 HTML 塞在 Python 字符串里而不是放 static/ 目录
------------------------------------------------------
只有一个页面，而且它要跟着后端的字段一起改。拆成文件之后，改一个
字段名要同时动两处、还要处理静态文件路由和缓存。等到第二个页面出现
再拆不迟。

样式沿用 ths_export.PANEL_CSS 的配色（#14161a 底、#c1440e 强调），
这样右侧 iframe 里嵌的三个面板和外壳是同一套视觉，不会像两个软件。
故意不 import 那份 CSS：面板的 CSS 要跟着邮件正文走，控制台的不用，
共用会让两边互相绊住。
"""

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>A股流水线 控制台</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,'Microsoft YaHei',sans-serif;margin:0;
     background:#14161a;color:#e6e6e6;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:10px 16px;
       background:#1a1d23;border-bottom:1px solid #2a2f38;flex:none}
header h1{font-size:15px;margin:0;font-weight:600}
header .clock{color:#8f9aa8;font-size:12px;font-family:Consolas,monospace}
header .spacer{flex:1}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.ok{background:#3fb950}.dot.bad{background:#f85149}.dot.warn{background:#d29922}
.dot.idle{background:#484f58}
main{flex:1;display:flex;min-height:0}
nav{width:132px;flex:none;background:#1a1d23;border-right:1px solid #2a2f38;
    padding:10px 0;display:flex;flex-direction:column}
nav button{background:none;border:none;color:#8f9aa8;text-align:left;
           padding:9px 16px;font-size:13px;cursor:pointer;border-left:2px solid transparent}
nav button:hover{color:#e6e6e6;background:#20242b}
nav button.on{color:#e6e6e6;background:#20242b;border-left-color:#c1440e}
nav .gap{flex:1}
section{flex:1;overflow:auto;padding:16px;min-width:0}
section.flush{padding:0}
h2{font-size:14px;margin:0 0 10px;font-weight:600}
h2 .hint{font-weight:400;color:#7d8590;font-size:12px;margin-left:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.card{background:#1a1d23;border:1px solid #2a2f38;border-radius:6px;padding:12px}
.card.bad{border-color:#5c2626;background:#1f1717}
.card.warn{border-color:#5c4a1a}
.card h3{margin:0 0 8px;font-size:13px;font-weight:600;display:flex;align-items:center}
.card h3 .t{flex:1}
.kv{display:flex;justify-content:space-between;font-size:12px;color:#8f9aa8;
    padding:3px 0;gap:10px}
.kv b{color:#e6e6e6;font-weight:600;font-family:Consolas,monospace;
      text-align:right;word-break:break-all}
.prob{color:#ff8f8f;font-size:12px;margin-top:8px;line-height:1.6}
.prob div{margin-top:3px}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
button.act{background:#2a2f38;color:#e6e6e6;border:1px solid #3a4149;
     border-radius:5px;padding:6px 12px;font-size:13px;cursor:pointer}
button.act:hover{background:#39404b}
button.act:disabled{opacity:.4;cursor:not-allowed}
button.act.danger{border-color:#7a3520}
button.act.danger:hover{background:#8a3a18}
button.act.on{background:#c1440e;border-color:#c1440e}
.grp{margin-bottom:16px}
.grp>.lbl{color:#7d8590;font-size:12px;margin-bottom:6px}
.desc{color:#7d8590;font-size:12px;margin:-6px 0 10px;line-height:1.6;
      min-height:2.6em;border-left:2px solid #2a2f38;padding-left:10px}
pre.log{background:#0e1014;border:1px solid #2a2f38;border-radius:6px;
    padding:10px 12px;font:12px/1.6 Consolas,monospace;color:#c9d1d9;
    white-space:pre-wrap;word-break:break-all;margin:0;
    max-height:calc(100vh - 300px);overflow:auto;min-height:180px}
pre.log .e{color:#ff8f8f}
pre.log .w{color:#e3b341}
pre.log .s{color:#7fb3ff}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#1e2229;text-align:left;padding:7px 9px;border-bottom:1px solid #333;
   font-weight:600;white-space:nowrap}
td{padding:7px 9px;border-bottom:1px solid #232830;vertical-align:top}
tr:hover td{background:#1b1f26}
.mono{font-family:Consolas,monospace;font-size:12px}
.muted{color:#7d8590}
iframe{width:100%;height:100%;border:none;background:#14161a;display:block}
.tabs{display:flex;gap:2px;background:#1a1d23;border-bottom:1px solid #2a2f38;
   flex:none;overflow-x:auto;white-space:nowrap}
.tabs button{flex:none}
.tabs button{background:none;border:none;color:#8f9aa8;padding:9px 16px;
   font-size:13px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:#e6e6e6;border-bottom-color:#c1440e}
.tabs .tabgrp{align-self:center;font-family:var(--mono);font-size:10px;
   letter-spacing:.1em;color:#6B7683;padding:0 8px 0 14px}
.tabs .tabsep{align-self:center;width:1px;height:16px;background:#2a2f38;
   margin:0 4px}
.frame{display:flex;flex-direction:column;height:100%}
.frame>.body{flex:1;min-height:0}

/* ==== 系统卡片（总览）==== */
.sys{background:var(--surface,#1a1d23);border:1px solid #2a2f38;border-radius:8px;
     margin-bottom:14px;overflow:hidden}
.sys>.hd{display:flex;align-items:baseline;gap:10px;padding:12px 16px;
     background:#20242b;border-bottom:1px solid #2a2f38}
.sys>.hd .n{font:700 15px/1 var(--mono,Consolas);color:#c1440e}
.sys>.hd .t{font-size:15px;font-weight:650}
.sys>.hd .w{margin-left:auto;font-size:12px;color:#7d8590}
.item{display:grid;grid-template-columns:44px 1fr auto;gap:12px;
      align-items:center;padding:12px 16px;border-bottom:1px solid #232830}
.item:last-child{border-bottom:none}
.item .no{font:700 13px/1 var(--mono,Consolas);color:#7d8590}
.item .nm{font-size:14px;font-weight:600}
.item .sub{font-size:12px;color:#7d8590;margin-top:3px}
.item .st{text-align:right;font-size:13px;white-space:nowrap}
.item .st b{display:block;font:700 17px/1.2 var(--mono,Consolas)}
.st.ok b{color:#3fb950}.st.no b{color:#7d8590}.st.warn b{color:#d29922}
.st.bad b{color:#f85149}
/* ==== 运行页的大按钮 ==== */
.act-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
     gap:10px;margin-bottom:20px}
.act-card{background:#1a1d23;border:1px solid #2a2f38;border-radius:8px;
     padding:14px 16px;display:flex;flex-direction:column;gap:8px}
.act-card.danger{border-color:#5c2626}
.act-card .top{display:flex;align-items:baseline;gap:9px}
.act-card .no{font:700 12px/1 var(--mono,Consolas);color:#c1440e}
.act-card .nm{font-size:15px;font-weight:650}
.act-card .ml{margin-left:auto;font-size:11px;color:#ff8f8f;
     border:1px solid #5c2626;border-radius:3px;padding:1px 6px}
.act-card .what{font-size:13px;color:#b4becA;line-height:1.55;flex:1}
.act-card .go{align-self:flex-start;background:#2a2f38;color:#e6e6e6;
     border:1px solid #3a4149;border-radius:5px;padding:6px 16px;
     font-size:13px;cursor:pointer}
.act-card .go:hover{background:#39404b}
.act-card.danger .go{border-color:#7a3520}
.act-card.danger .go:hover{background:#8a3a18}
.act-card .go:disabled{opacity:.45;cursor:not-allowed}
.act-card .more{font-size:11.5px;color:#6B7683;line-height:1.5;display:none}
.act-card.open .more{display:block}
.act-card .tg{font-size:11px;color:#6B7683;background:none;border:none;
     cursor:pointer;padding:0;align-self:flex-start}
/* ==== 成绩块（嵌在系统卡片里）==== */
.perfbox{padding:12px 16px 14px;background:#171a1f;border-top:1px solid #232830}
.perfbox .pt{font-size:12px;color:#7d8590;margin-bottom:9px}
.perfbox .pn{font-size:12.5px;color:#b4beca;line-height:1.65;margin-top:9px}
/* ==== 成绩条 ==== */
.perf{display:flex;align-items:center;gap:10px;margin-top:6px}
.perf .lbl{font-size:11.5px;color:#7d8590;width:64px;flex:none}
.perf .bar2{flex:1;height:12px;background:#232830;border-radius:3px;overflow:hidden}
.perf .bar2 i{display:block;height:100%;background:#c1440e}
.perf .bar2.grey i{background:#3a4149}
.perf .v{font:600 12px/1 var(--mono,Consolas);width:52px;text-align:right}
#toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);
   background:#c1440e;padding:8px 18px;border-radius:5px;opacity:0;
   transition:.25s;font-size:13px;pointer-events:none;z-index:9}
#toast.show{opacity:1}
.spin{display:inline-block;width:9px;height:9px;border:2px solid #c1440e;
   border-top-color:transparent;border-radius:50%;animation:sp .7s linear infinite;
   margin-right:6px;vertical-align:-1px}
@keyframes sp{to{transform:rotate(360deg)}}
</style></head><body>

<header>
  <h1>A股流水线</h1>
  <span class="clock" id="clock">...</span>
  <span class="spacer"></span>
  <span class="clock" id="health"></span>
</header>

<main>
  <nav>
    <button data-v="home" class="on">总览</button>
    <button data-v="run">运行</button>
    <button data-v="panel">面板</button>
    <button data-v="sched">排期</button>
    <button data-v="log">运行记录</button>
    <button data-v="conf">配置</button>
    <span class="gap"></span>
  </nav>

  <section id="v-home"></section>

  <section id="v-run" hidden>
    <div id="runbar"></div>
      <h2>输出 <span class="hint" id="joblbl"></span>
      <button class="act" id="stopbtn" hidden style="float:right;padding:3px 10px">停止</button>
    </h2>
    <pre class="log" id="joblog">还没有跑过任何东西。</pre>
  </section>

  <section id="v-panel" hidden class="flush">
    <div class="frame">
      <div class="tabs">
        <span class="tabgrp">早盘系统</span>
        <button data-p="auction" class="on">早盘选股</button>
        <button data-p="learn">参数自学</button>
        <span class="tabsep"></span>
        <span class="tabgrp">晚间系统</span>
        <button data-p="breakout">起涨预测</button>
        <button data-p="pullback">回调形态</button>
      </div>
      <div class="body"><iframe id="pframe" src="/panel/auction"></iframe></div>
    </div>
  </section>

  <section id="v-sched" hidden></section>

  <section id="v-log" hidden>
    <h2>自动运行记录<span class="hint">计划任务每天自己跑的输出。手动跑的记录在「运行」页</span></h2>
    <pre class="log" id="flowlog">读取中...</pre>
  </section>

  <section id="v-conf" hidden>
    <h2>参数配置<span class="hint">只读。所有阈值都在这里，要改请直接编辑 config.yaml 并跑一次自检</span></h2>
    <pre class="log" id="confbody" style="max-height:calc(100vh - 150px)">读取中...</pre>
  </section>
</main>

<div id="toast"></div>

<script>
const TOKEN = "__TOKEN__";
const $ = s => document.querySelector(s);
const el = (t, c, x) => { const e = document.createElement(t);
  if (c) e.className = c; if (x !== undefined) e.textContent = x; return e; };
const esc = s => String(s ?? "").replace(/[&<>]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function toast(m) { const t = $("#toast"); t.textContent = m;
  t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200); }

async function api(path, body) {
  const o = body ? { method: "POST", headers: { "Content-Type": "application/json",
    "X-Token": TOKEN }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, o);
  const j = await r.json().catch(() => ({ error: "响应不是 JSON" }));
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

/* ---------------- 导航 ---------------- */
let view = "home";
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x =>
    x.classList.toggle("on", x === b));
  view = b.dataset.v;
  ["home","run","panel","sched","log","conf"].forEach(v =>
    $("#v-" + v).hidden = v !== view);
  if (view === "log") loadFlowLog();
  if (view === "conf") loadConf();
  if (view === "home" || view === "sched") refresh();
});

document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x =>
    x.classList.toggle("on", x === b));
  $("#pframe").src = "/panel/" + b.dataset.p;
});

/* ---------------- 总览 ---------------- */
function card(title, dotClass, rows, probs, cls) {
  const c = el("div", "card" + (cls ? " " + cls : ""));
  const h = el("h3");
  h.appendChild(Object.assign(el("span", "dot " + dotClass), {}));
  h.appendChild(el("span", "t", title));
  c.appendChild(h);
  rows.forEach(([k, v]) => {
    const r = el("div", "kv"); r.appendChild(el("span", null, k));
    r.appendChild(el("b", null, v)); c.appendChild(r);
  });
  if (probs && probs.length) {
    const p = el("div", "prob");
    probs.forEach(x => p.appendChild(el("div", null, "• " + x)));
    c.appendChild(p);
  }
  return c;
}

function sysItem(no, name, sub, stateCls, big, small) {
  const d = el("div", "item");
  d.appendChild(el("div", "no", no));
  const mid = el("div");
  mid.appendChild(el("div", "nm", name));
  if (sub) mid.appendChild(el("div", "sub", sub));
  d.appendChild(mid);
  const st = el("div", "st " + stateCls);
  st.appendChild(el("b", null, big));
  if (small) st.appendChild(el("div", null, small));
  d.appendChild(st);
  return d;
}

function sysCard(no, title, when, items) {
  const c = el("div", "sys");
  const hd = el("div", "hd");
  hd.appendChild(el("span", "n", no));
  hd.appendChild(el("span", "t", title));
  hd.appendChild(el("span", "w", when));
  c.appendChild(hd);
  items.forEach(x => c.appendChild(x));
  return c;
}

function lineOf(s, key) { return s.lines.find(l => l.key === key) || {}; }

function pickState(l, s) {
  // 三种状态：跑完了(ok) / 周末不用跑(no) / 该跑没跑(warn)
  if (l.done) return ["ok", (l.n === undefined || l.n === null)
    ? "已完成" : l.n + " 只", "数据 " + (l.date || "-")];
  if (s.weekend) return ["no", "周末", "不用跑"];
  return ["warn", "没跑", "数据 " + (l.date || "无")];
}

function perfRow(lbl, val, unit, barPct, grey) {
  const r = el("div", "perf");
  r.appendChild(el("div", "lbl", lbl));
  const bar = el("div", "bar2" + (grey ? " grey" : ""));
  const i2 = el("i"); i2.style.width = Math.max(barPct, 2) + "%";
  bar.appendChild(i2); r.appendChild(bar);
  r.appendChild(el("div", "v", val + unit));
  return r;
}

function perfBlock(title, rows, note) {
  // 成绩块，塞在各自系统卡片的**里面**。它属于哪个系统就放哪个系统，
  // 不单独成块 —— 单独摆出来的话，看的人不知道它说的是早还是晚。
  const d = el("div", "perfbox");
  d.appendChild(el("div", "pt", title));
  rows.forEach(r => d.appendChild(r));
  if (note) d.appendChild(el("div", "pn", note));
  return d;
}

function renderHome(s) {
  const box = $("#v-home"); box.innerHTML = "";
  const h = el("h2", null, "今天");
  h.appendChild(Object.assign(el("span", "hint"),
    { textContent: `北京 ${s.bj} 周${s.weekday}` +
      (s.weekend ? "（周末，两个系统都不该跑）" : "") }));
  box.appendChild(h);

  // ===== 1 早盘系统 =====
  const m = lineOf(s, "morning"), lr = lineOf(s, "learn");
  const [ms, mb, mm] = pickState(m, s);
  const [ls, lb, lm] = pickState(lr, s);
  const c1 = sysCard("1", "早盘系统", "每天早上 9:27 发邮件", [
    sysItem("1-1", "早盘选股", "挑出当天最强的股票发给你", ms, mb, mm),
    sysItem("1-2", "参数自学", "回头检查打分准不准，自动改进", ls, lb, lm),
  ]);
  const mp = s.morning_perf || {};
  if (mp.exists) {
    const hitPct = 100 * mp.hit, exc = 100 * mp.excess;
    c1.appendChild(perfBlock("这个系统准不准（" + mp.days + " 个交易日实测）", [
      perfRow("榜上的票", hitPct.toFixed(1), "%", (hitPct - 45) / 10 * 100, false),
      perfRow("随便买", "50.0", "%", (50 - 45) / 10 * 100, true),
    ], "榜上的票有 " + hitPct.toFixed(1) + "% 跑赢当天大盘（随便买是 50%），"
     + "平均每只比大盘多赚 " + exc.toFixed(2) + "%。优势不大但是稳定的。"));
  }
  box.appendChild(c1);

  // ===== 2 晚间系统 =====（结构和上面一模一样）
  const bk = lineOf(s, "breakout");
  const [bs, bb, bm] = pickState(bk, s);
  const mo = s.model || {};
  const c2 = sysCard("2", "晚间系统", "每天下午 5:00 发邮件", [
    sysItem("2-1", "起涨预测", "挑出可能要涨的股票发给你", bs, bb, bm),
    sysItem("2-2", "模型自学", "用最新数据重新学习，每 30 天一次",
            mo.exists ? "ok" : "warn",
            mo.exists ? mo.age_days + " 天前" : "没训练",
            mo.exists ? mo.days_to_refit + " 天后重学" : "点 2-2 训练"),
  ]);
  const hit = 14.43, base = 2.91;
  c2.appendChild(perfBlock("这个系统准不准（模型从没见过的 8 个月实测）", [
    perfRow("清单里的票", hit.toFixed(2), "%", 100, false),
    perfRow("随便买", base.toFixed(2), "%", base / hit * 100, true),
  ], "清单里每 10 只，大约 " + (hit / 10).toFixed(1)
   + " 只会在接下来一个月内涨超 50%，是随便买的 " + (hit / base).toFixed(1)
   + " 倍。不是「选出来的都会涨」。"));
  box.appendChild(c2);

  return renderSync(box, s);
}

function renderSync(box, s) {
  const h2 = el("h2", null, "系统健康");
  h2.appendChild(Object.assign(el("span", "hint"),
    { textContent: "这里红了才需要管，平时不用看" }));
  box.appendChild(h2);

  const g2 = el("div", "cards");
  const sy = s.sync;
  const syncCard = card("和 GitHub 的同步", sy.ok ? "ok" : "bad", [
    ["本地没上传的", sy.ahead + " 次改动"],
    ["远端没下载的", sy.behind + " 次改动"],
    ["上次上传", (sy.last_push || "无记录")
      + (sy.last_push_ok === false ? " 失败" : "")],
  ], sy.problems, sy.ok ? "" : "bad");
  if (!sy.ok) {
    const b = el("button", "act", "重试上传");
    b.style.marginTop = "10px";
    b.onclick = async () => {
      b.disabled = true; b.textContent = "上传中…";
      try { const r = await api("/api/push", {});
        toast("已重试"); alert(r.log || "(无输出)"); refresh(); }
      catch (e) { toast("失败: " + e.message); }
      finally { b.disabled = false; b.textContent = "重试上传"; }
    };
    syncCard.appendChild(b);
  }
  g2.appendChild(syncCard);

  const live = s.cloud_cron_live || [];
  g2.appendChild(card("云端自动运行", live.length ? "warn" : "ok",
    [["状态", live.length ? live.length + " 个还在自动跑" : "已全部关掉"],
     ["现在的角色", live.length ? "仍在云端跑" : "只存代码和数据"]],
    live.length ? live : null, live.length ? "warn" : ""));
  box.appendChild(g2);
}

/* ---------------- 运行 ---------------- */
let curJob = null, logOffset = 0, logTimer = null;

function renderRun(a) {
  const bar = $("#runbar");
  if (bar.dataset.built) { syncRunButtons(a); return; }
  bar.innerHTML = "";
  const groups = {};
  a.actions.forEach(x => (groups[x.group] = groups[x.group] || []).push(x));
  Object.keys(groups).sort().forEach(name => {
    const h = el("h2", null, name);
    h.style.cssText = "margin:4px 0 12px";
    bar.appendChild(h);
    const grid = el("div", "act-grid");
    groups[name].forEach(x => grid.appendChild(actCard(x)));
    bar.appendChild(grid);
  });
  bar.dataset.built = "1";
  syncRunButtons(a);
}

function actCard(x) {
  const c = el("div", "act-card" + (x.danger ? " danger" : ""));
  const top = el("div", "top");
  top.appendChild(el("span", "no", x.no || ""));
  top.appendChild(el("span", "nm", x.name));
  if (x.mail) top.appendChild(el("span", "ml", "会发邮件"));
  c.appendChild(top);
  c.appendChild(el("div", "what", x.what || x.desc || ""));

  const tg = el("button", "tg", "详细说明 ▾");
  const more = el("div", "more", x.desc || "");
  tg.onclick = () => {
    c.classList.toggle("open");
    tg.textContent = c.classList.contains("open") ? "收起 ▴" : "详细说明 ▾";
  };
  if (x.desc && x.desc !== x.what) { c.appendChild(tg); c.appendChild(more); }

  const go = el("button", "go", "运行");
  go.dataset.key = x.key;
  go.onclick = () => run(x.key);
  c.appendChild(go);
  return c;
}

function syncRunButtons(a) {
  const running = new Set(a.running);
  document.querySelectorAll("#runbar button.go").forEach(b => {
    const r = running.has(b.dataset.key);
    b.disabled = r;
    b.textContent = r ? "正在跑…" : "运行";
  });
}

async function run(key) {
  try {
    const j = await api("/api/run", { key });
    curJob = j.id; logOffset = 0; $("#joblog").textContent = "";
    $("#stopbtn").hidden = false;
    toast("已启动 " + j.name);
    pollLog();
  } catch (e) { toast(e.message); }
}

$("#stopbtn").onclick = async () => {
  if (!curJob) return;
  if (!confirm("终止这次运行？整棵进程树都会被杀掉。")) return;
  try { await api("/api/stop", { id: curJob }); toast("已终止"); }
  catch (e) { toast(e.message); }
};

function colorize(line) {
  const s = esc(line);
  if (/ERROR|错误|失败|Traceback|退出码 [1-9]/.test(line))
    return '<span class="e">' + s + "</span>";
  if (/WARN|警告|跳过/.test(line)) return '<span class="w">' + s + "</span>";
  if (/^##STEP|步骤|启动 @/.test(line)) return '<span class="s">' + s + "</span>";
  return s;
}

async function pollLog() {
  if (!curJob) return;
  try {
    const j = await api(`/api/job/${curJob}?from=${logOffset}`);
    if (j.new && j.new.length) {
      const pre = $("#joblog");
      const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
      pre.insertAdjacentHTML("beforeend",
        j.new.map(colorize).join("\n") + "\n");
      if (stick) pre.scrollTop = pre.scrollHeight;
      logOffset = j.next;
    }
    $("#joblbl").innerHTML = j.running
      ? `<span class="spin"></span>${esc(j.name)} 已跑 ${j.secs}s`
      : `${esc(j.name)} 结束，退出码 ${j.rc}，耗时 ${j.secs}s`;
    $("#stopbtn").hidden = !j.running;
    clearTimeout(logTimer);
    if (j.running) logTimer = setTimeout(pollLog, 1000);
    else { refreshActions(); refresh(); }
  } catch (e) {
    $("#joblbl").textContent = "日志中断: " + e.message;
  }
}

/* ---------------- 排期 ---------------- */
function renderSched(s) {
  const box = $("#v-sched"); box.innerHTML = "";
  const h = el("h2", null, "Windows 计划任务");
  h.appendChild(Object.assign(el("span", "hint"),
    { textContent: "每天自动运行靠这个。时刻是本机（美东）时间，括号里是对应的北京时间" }));
  box.appendChild(h);

  const t = el("table");
  t.innerHTML = "<tr><th>任务</th><th>状态</th><th>上次</th><th>结果</th>" +
    "<th>下次</th><th></th></tr>";
  const names = Object.keys(s.tasks).sort();
  if (!names.length) t.innerHTML += '<tr><td colspan="6" class="muted">' +
    "查不到 DailyReport-* 任务</td></tr>";
  names.forEach(n => {
    const d = s.tasks[n], tr = el("tr");
    const on = d.state === "Ready" || d.state === "Running";
    const legacy = n.indexOf("Trigger") >= 0;
    tr.innerHTML =
      `<td class="mono">${esc(n)}${legacy ?
        '<div class="muted" style="font-size:11px">旧的云端派发，已弃用</div>' : ""}</td>` +
      `<td><span class="dot ${on ? (legacy ? "warn" : "ok") : "idle"}"></span>` +
        `${esc(d.state)}</td>` +
      `<td class="mono muted">${esc(d.last)}</td>` +
      `<td class="mono ${d.rc ? "" : "muted"}">${d.never ? "还没跑过" :
        (d.rc === 0 ? "成功" :
         (d.rc === null || d.rc === undefined ? "" : "码 " + d.rc))}</td>` +
      `<td class="mono muted">${esc(d.next)}</td>`;
    const td = el("td");
    const b = el("button", "act", on ? "停用" : "启用");
    b.style.padding = "3px 10px";
    b.onclick = async () => {
      b.disabled = true;
      try { await api("/api/task", { name: n, enable: !on }); toast("已改"); }
      catch (e) { toast(e.message); }
      finally { b.disabled = false; refresh(true); }
    };
    td.appendChild(b); tr.appendChild(td); t.appendChild(tr);
  });
  box.appendChild(t);

  const tip = el("div", "muted");
  tip.style.cssText = "font-size:12px;margin-top:14px;line-height:1.8";
  tip.innerHTML =
    "早盘选股 <b>DailyReport-Local-Morning</b>：美东周日到周四 18:00 起每 15 分钟，" +
    "持续 3 小时 15 分。夏令时对应北京 06:00-09:15，冬令时 07:00-10:15，" +
    "两种时令都盖得住 09:16 这条开跑上界。<br>" +
    "参数自学 <b>DailyReport-Local-Learn</b>：美东周一到周五 04:40 起每 30 分钟。<br>" +
    "反复重试是因为笔记本可能整段时间不在线（2026-08-27 就漏发过一次）。" +
    "反复敲是安全的：流程会先查今天跑过没有，跑完了再敲直接退出，不会重发邮件。";
  box.appendChild(tip);
}

/* ---------------- 其它页 ---------------- */
async function loadFlowLog() {
  try { const j = await api("/api/logfile");
    $("#flowlog").innerHTML = j.text.split("\n").map(colorize).join("\n");
    const p = $("#flowlog"); p.scrollTop = p.scrollHeight;
  } catch (e) { $("#flowlog").textContent = "读取失败: " + e.message; }
}
async function loadConf() {
  try { const j = await api("/api/config"); $("#confbody").textContent = j.text; }
  catch (e) { $("#confbody").textContent = "读取失败: " + e.message; }
}

/* ---------------- 刷新循环 ---------------- */
let lastStatus = null;
async function refresh(force) {
  try {
    const s = await api("/api/status");
    lastStatus = s;
    $("#clock").textContent = "北京 " + s.bj;
    const bad = !s.sync.ok || (s.cloud_cron_live || []).length;
    const undone = s.lines.filter(l => !l.done && l.key !== "evening").length;
    $("#health").innerHTML = bad
      ? '<span class="dot bad"></span>有问题，看总览'
      : (s.weekend ? '<span class="dot idle"></span>周末'
        : (undone ? '<span class="dot warn"></span>' + undone + " 条线今天还没跑"
          : '<span class="dot ok"></span>一切正常'));
    if (view === "home") renderHome(s);
    if (view === "sched") renderSched(s);
  } catch (e) { $("#health").textContent = "状态获取失败"; }
}
async function refreshActions() {
  try { renderRun(await api("/api/actions")); } catch (e) {}
}

refresh(); refreshActions();
setInterval(refresh, 5000);
setInterval(refreshActions, 3000);
setInterval(() => { const d = new Date(Date.now() + 8 * 3600e3);
  if (lastStatus) $("#clock").textContent = "北京 " +
    d.toISOString().slice(0, 19).replace("T", " "); }, 1000);
</script></body></html>
"""
