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
.tabs{display:flex;gap:2px;background:#1a1d23;border-bottom:1px solid #2a2f38;flex:none}
.tabs button{background:none;border:none;color:#8f9aa8;padding:9px 16px;
   font-size:13px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:#e6e6e6;border-bottom-color:#c1440e}
.frame{display:flex;flex-direction:column;height:100%}
.frame>.body{flex:1;min-height:0}
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
    <div class="desc" id="rundesc">把鼠标放到按钮上看它具体做什么</div>
    <h2>输出 <span class="hint" id="joblbl"></span>
      <button class="act" id="stopbtn" hidden style="float:right;padding:3px 10px">停止</button>
    </h2>
    <pre class="log" id="joblog">还没有跑过任何东西。</pre>
  </section>

  <section id="v-panel" hidden class="flush">
    <div class="frame">
      <div class="tabs">
        <button data-p="auction" class="on">早盘选股</button>
        <button data-p="pullback">回调形态</button>
        <button data-p="learn">参数自学</button>
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

function renderHome(s) {
  const box = $("#v-home"); box.innerHTML = "";
  const h = el("h2", null, "今天");
  h.appendChild(Object.assign(el("span", "hint"),
    { textContent: `北京 ${s.bj} 周${s.weekday}` +
      (s.weekend ? "（周末，两条线都不该运行）" : "") }));
  box.appendChild(h);

  const g = el("div", "cards");
  s.lines.forEach(l => {
    const dot = l.done ? "ok" : (s.weekend ? "idle" : "warn");
    const rows = [["今日产出", l.done ? (l.n === undefined || l.n === null
        ? "已完成" : l.n + " 只")
        : (s.weekend ? "周末停跑" : "未完成")],
      ["数据日期", l.date || "无"],
      ["应跑时刻", l.due]];
    if (l.task) rows.push(["排期", (l.task_state || "?") +
      (l.task_next ? " 下次 " + l.task_next : "")]);
    g.appendChild(card(l.name, dot, rows, null,
      (!l.done && !s.weekend && l.key !== "evening") ? "warn" : ""));
  });
  box.appendChild(g);

  const h2 = el("h2", null, "同步");
  h2.appendChild(Object.assign(el("span", "hint"),
    { textContent: "上次连错五天没人发现，就是因为这里以前只写日志不上界面" }));
  box.appendChild(h2);

  const g2 = el("div", "cards");
  const sy = s.sync;
  const syncCard = card("仓库", sy.ok ? "ok" : "bad", [
    ["本地未推送", sy.ahead + " 个 commit"],
    ["远端未拉取", sy.behind + " 个 commit"],
    ["工作区改动", sy.dirty + " 个文件"],
    ["上次推送", (sy.last_push || "无记录") +
      (sy.last_push_ok === false ? " 失败" : "")],
  ], sy.problems, sy.ok ? "" : "bad");
  if (!sy.ok) {
    const b = el("button", "act", "重试推送");
    b.style.marginTop = "10px";
    b.onclick = async () => { b.disabled = true; b.textContent = "推送中...";
      try { const r = await api("/api/push", {});
        toast("已重试"); alert(r.log || "(无输出)"); refresh(); }
      catch (e) { toast("失败: " + e.message); }
      finally { b.disabled = false; b.textContent = "重试推送"; } };
    syncCard.appendChild(b);
  }
  g2.appendChild(syncCard);

  const live = s.cloud_cron_live || [];
  g2.appendChild(card("云端", live.length ? "warn" : "ok",
    [["自动触发", live.length ? live.length + " 个 workflow 仍有 cron" : "已全停"],
     ["仓库角色", live.length ? "仍在自动跑" : "只做版本控制"]],
    live.length ? live.map(x => x) : null, live.length ? "warn" : ""));
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
  Object.entries(groups).forEach(([name, items]) => {
    const g = el("div", "grp");
    g.appendChild(el("div", "lbl", name));
    const row = el("div", "bar");
    items.forEach(x => {
      const b = el("button", "act" + (x.danger ? " danger" : ""),
        x.name + (x.mail ? " ✉" : ""));
      b.dataset.key = x.key;
      b.onmouseenter = () => $("#rundesc").textContent =
        x.desc + (x.mail ? "  【会真的发邮件】" : "");
      b.onclick = () => run(x.key);
      row.appendChild(b);
    });
    g.appendChild(row); bar.appendChild(g);
  });
  bar.dataset.built = "1";
  syncRunButtons(a);
}

function syncRunButtons(a) {
  const running = new Set(a.running);
  document.querySelectorAll("#runbar button").forEach(b => {
    const r = running.has(b.dataset.key);
    b.classList.toggle("on", r);
    b.disabled = r;
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
