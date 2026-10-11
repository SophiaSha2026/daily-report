"""
详细控制台（/full）。首页傻瓜页在 simple.py。

为什么把 HTML 塞在 Python 字符串里而不是放 static/ 目录
------------------------------------------------------
页面要跟着后端的字段一起改。拆成文件之后，改一个字段名要同时动两处、
还要处理静态文件路由和缓存。

布局（2026-10-11 瘦身，用户：「文字越少越好，界面越干净越好，只留必要交互」）
------------------------------------------------------------------------------
    顶栏   北京时间、健康灯、首页 / 两个面板的链接（新标签页打开）
    左栏   三组按钮：跑 / 查 / 修。按钮来自 jobs.ACTIONS（按 group 分组、组内按 no
           排序），每个只印 name 和 what 一句，desc 收进鼠标悬停提示。
           「修」组另外挂两样不在 ACTIONS 里的：重试推送（/api/push）、
           计划任务启停（/api/task，任务表来自 /api/status 的 tasks）
    右边   「状态」一页：两条线 + 同步的状态表 -> 当前任务的输出（打开页面时
           自动接上最近一个任务）-> 折叠区：排期、自动运行日志、配置
以前的总览 / 运行 / 面板 / 排期 / 运行记录 / 配置六页合成了这一页；面板 iframe
改成顶栏链接；成绩说明、触发规则长表、优先级和 --if-needed 清单不再平铺，
排期表的细则收进悬停提示（数据照旧来自 status.RULES）。

接线要点（selftest_gui 钉住的，删文字时别删到它们）
--------------------------------------------------
- 顶栏面板链接的 data-p 必须和 server.PANELS 的键一致，href 由 data-p 拼出来。
- pickState：报错对象最优先（run_meta 带 error 时退出码 0、日期也对，只看 done
  会一路绿灯，教训 16）；done 和 sent 是两件事（run_meta 扫描阶段就落盘，发信是
  后面的子进程），「跑完了但邮件没发出去」标红。
- 顶栏「N 条线还没跑」把没发出去和带报错的也算进去，否则写「正常」就没人去补。
- 会发邮件的按钮点前确认一次（2026-09-14 误点重发过一次上一交易日的清单）。
- 排期说明里的 UTC 锚点要和 tools/setup_tasks.ps1 一致（selftest 逐字比对）。
- 计划任务「上次结果」是 Windows HRESULT：0x41301 正在运行、0x41303 没跑过、
  0x41306 被终止、0x800710E0 上次触发时已有实例在跑（IgnoreNew）。
"""

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>A股流水线 控制台</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,'Microsoft YaHei',sans-serif;margin:0;
     background:#14161a;color:#e6e6e6;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:14px;padding:10px 16px;flex-wrap:wrap;
       background:#1a1d23;border-bottom:1px solid #2a2f38;flex:none}
header h1{font-size:15px;margin:0;font-weight:600}
header .spacer{flex:1}
header a{color:#8f9aa8;font-size:13px;text-decoration:none}
header a:hover{color:#e6e6e6}
.clock{color:#8f9aa8;font-size:12px;font-family:Consolas,monospace}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot.ok{background:#3fb950}.dot.bad{background:#f85149}.dot.warn{background:#d29922}
.dot.idle{background:#484f58}
main{flex:1;display:flex;min-height:0}
aside{width:300px;flex:none;overflow:auto;background:#1a1d23;
      border-right:1px solid #2a2f38;padding:4px 0 16px}
aside h2{font-size:13px;color:#c1440e;font-weight:600;margin:14px 16px 4px}
.act{display:flex;align-items:center;gap:10px;padding:6px 16px}
.act:hover{background:#20242b}
.act .tx{flex:1;min-width:0}
.act .nm{font-size:13.5px;font-weight:600}
.act .ml{font-size:11px;font-weight:400;color:#ff8f8f;border:1px solid #5c2626;
         border-radius:3px;padding:0 5px;margin-left:6px}
.act .what{font-size:12px;color:#7d8590;line-height:1.45}
button.go{flex:none;background:#2a2f38;color:#e6e6e6;border:1px solid #3a4149;
          border-radius:5px;padding:4px 12px;font-size:13px;font-family:inherit;cursor:pointer}
button.go:hover{background:#39404b}
button.go.danger{border-color:#7a3520}
button.go.danger:hover{background:#8a3a18}
button.go:disabled{opacity:.45;cursor:not-allowed}
section{flex:1;overflow:auto;padding:16px;min-width:0}
h3{font-size:14px;margin:18px 0 8px;font-weight:600;display:flex;align-items:center;gap:10px}
.hint{font-weight:400;color:#7d8590;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#1e2229;text-align:left;padding:6px 9px;border-bottom:1px solid #333;
   font-weight:600;white-space:nowrap}
td{padding:6px 9px;border-bottom:1px solid #232830;vertical-align:top}
table.sum td:first-child{font-weight:600;white-space:nowrap;width:1%}
table.sum td:nth-child(2){white-space:nowrap;width:1%}
.muted{color:#7d8590}
.mono{font-family:Consolas,monospace;font-size:12px}
.prob{color:#ff8f8f;font-size:12px;margin-top:8px;line-height:1.6}
pre.log{background:#0e1014;border:1px solid #2a2f38;border-radius:6px;
    padding:10px 12px;font:12px/1.6 Consolas,monospace;color:#c9d1d9;
    white-space:pre-wrap;word-break:break-all;margin:0;
    max-height:60vh;overflow:auto;min-height:120px}
pre.log .e{color:#ff8f8f}
pre.log .w{color:#e3b341}
pre.log .s{color:#7fb3ff}
details{margin-top:14px;border-top:1px solid #2a2f38;padding-top:10px}
summary{cursor:pointer;color:#8f9aa8;font-size:13px;margin-bottom:8px}
#toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);
   background:#c1440e;padding:8px 18px;border-radius:5px;opacity:0;
   transition:.25s;font-size:13px;pointer-events:none;z-index:9}
#toast.show{opacity:1}
.spin{display:inline-block;width:9px;height:9px;border:2px solid #c1440e;
   border-top-color:transparent;border-radius:50%;animation:sp .7s linear infinite;
   margin-right:6px;vertical-align:-1px}
@keyframes sp{to{transform:rotate(360deg)}}
@media (max-width:760px){body{height:auto}main{flex-direction:column}
  aside{width:auto;border-right:none;border-bottom:1px solid #2a2f38}section{overflow:visible}}
</style></head><body>

<header>
  <h1>A股流水线</h1>
  <span class="clock" id="clock">...</span>
  <span class="clock" id="health"></span>
  <span class="spacer"></span>
  <a href="/">首页</a>
  <a data-p="breakout" target="_blank">起涨预测面板</a>
  <a data-p="pullback" target="_blank">长期调整突破面板</a>
</header>

<main>
  <aside id="acts"></aside>
  <section>
    <div id="sum"></div>
    <h3>输出 <span class="hint" id="joblbl"></span>
      <button class="go" id="stopbtn" hidden>停止</button></h3>
    <pre class="log" id="joblog">还没有</pre>
    <details id="d-ai" open><summary>AI 复盘与提案</summary><div id="ai"></div></details>
    <details id="d-sched"><summary>排期</summary><div id="sched"></div></details>
    <details id="d-log"><summary>自动运行日志</summary><pre class="log" id="flowlog"></pre></details>
    <details id="d-conf"><summary>配置（只读）</summary><pre class="log" id="confbody"></pre></details>
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

document.querySelectorAll("a[data-p]").forEach(a => a.href = "/panel/" + a.dataset.p);

/* ---------------- 状态 ---------------- */
function pickState(l) {
  const day = "目标日 " + (l.target || "-") +
    ((l.n === undefined || l.n === null) ? "" : " · 清单 " + l.n + " 只");
  if (l.before_start) return ["idle", "未上线", (l.since || "").slice(5) + " 起"];
  if (l.error) return ["bad", "有报错", String(l.error).slice(0, 60)];
  if (l.done && l.sent === false) return ["bad", "邮件没发出去", day];
  if (l.done) return ["ok", "已完成", day];
  if (l.dry) return ["warn", "只试跑过", "没发信"];
  return ["warn", "没跑", "目标日 " + (l.target || "-")];
}

function sumRow(t, name, cls, big, small) {
  const tr = el("tr"), td = el("td");
  tr.appendChild(el("td", null, name));
  td.appendChild(el("span", "dot " + cls)); td.appendChild(document.createTextNode(big));
  tr.appendChild(td); tr.appendChild(el("td", "muted", small || ""));
  t.appendChild(tr);
}

function renderSum(s) {
  const t = el("table", "sum");
  s.lines.forEach(l => { const [c, big, small] = pickState(l); sumRow(t, l.name, c, big, small); });
  const sy = s.sync;
  sumRow(t, "同步", sy.ok ? "ok" : "bad", sy.ok ? "正常" : "有问题",
    "上次推送 " + (sy.last_push || "无记录") + (sy.last_push_ok === false ? " 失败" : ""));
  const fb = s.cloud_fallback || {};
  const cloud = (s.cloud_cron_live || []).concat((fb.missing || []).map(x => "托底没开：" + x));
  if (cloud.length) sumRow(t, "云端", "warn", "有问题", "");
  const box = $("#sum"); box.innerHTML = ""; box.appendChild(t);
  const probs = (sy.problems || []).concat(cloud);
  if (probs.length) {
    const p = el("div", "prob");
    probs.forEach(x => p.appendChild(el("div", null, x)));
    box.appendChild(p);
  }
}

/* ---------------- 左栏：跑 / 查 / 修 ---------------- */
let curJob = null, logOffset = 0, logTimer = null, acts = [];
const byNo = (a, b) => String(a.no).localeCompare(String(b.no), undefined, { numeric: true });

function actRow(nm, what, tip, btn) {
  const r = el("div", "act"), tx = el("div", "tx");
  if (tip) r.title = tip;
  tx.appendChild(nm);
  if (what) tx.appendChild(el("div", "what", what));
  r.appendChild(tx); r.appendChild(btn);
  return r;
}

function buildActs(list) {
  const box = $("#acts"), groups = {};
  list.slice().sort(byNo).forEach(x => {
    if (x.hidden) return;
    if (!groups[x.group]) {
      box.appendChild(el("h2", null, x.group));
      groups[x.group] = box.appendChild(el("div"));
    }
    const nm = el("div", "nm", x.name);
    if (x.mail) nm.appendChild(el("span", "ml", "会发邮件"));
    const b = el("button", "go" + (x.danger ? " danger" : ""), "跑");
    b.dataset.key = x.key; b.onclick = () => run(x.key);
    groups[x.group].appendChild(actRow(nm, x.what, x.desc, b));
  });
  const fix = groups["修"] || box.appendChild(el("div"));
  const pb = el("button", "go", "推送");
  pb.onclick = () => retryPush(pb);
  fix.appendChild(actRow(el("div", "nm", "重试推送"), "推送失败时用", "", pb));
  const tk = el("div"); tk.id = "tasks"; fix.appendChild(tk);
}

function renderActs(a) {
  acts = a.actions || [];
  const box = $("#acts");
  if (!box.dataset.built) {
    buildActs(acts); box.dataset.built = "1";
    if (lastStatus) renderTasks(lastStatus);
  }
  const running = new Set(a.running);
  box.querySelectorAll("button[data-key]").forEach(b => {
    const r = running.has(b.dataset.key);
    b.disabled = r; b.textContent = r ? "在跑" : "跑";
  });
  if (!curJob && (a.jobs || []).length) {
    curJob = a.jobs[0].id; logOffset = 0; $("#joblog").textContent = ""; pollLog();
  }
}

function taskName(s, n) {
  const l = s.lines.find(x => x.task === n);
  return l ? l.name : (/Sync$/.test(n) ? "同步" : n);
}

function renderTasks(s) {
  const box = $("#tasks");
  if (!box) return;
  const names = Object.keys(s.tasks || {}).sort();
  const sig = JSON.stringify(names.map(n => [n, s.tasks[n].state]));
  if (box.dataset.sig === sig) return;
  box.dataset.sig = sig; box.innerHTML = "";
  names.forEach(n => {
    const on = s.tasks[n].state === "Ready" || s.tasks[n].state === "Running";
    const b = el("button", "go", on ? "停用" : "启用");
    b.onclick = async () => {
      b.disabled = true;
      try { await api("/api/task", { name: n, enable: !on }); toast("已改"); }
      catch (e) { toast(e.message); }
      finally { b.disabled = false; refresh(); }
    };
    box.appendChild(actRow(el("div", "nm", "计划任务 · " + taskName(s, n)),
      on ? "启用中" : "已停用", n, b));
  });
}

async function retryPush(b) {
  b.disabled = true;
  try {
    const r = await api("/api/push", {});
    toast(r.ok ? "已推送" : "推送失败");
    alert(r.log || ""); refresh();
  } catch (e) { toast(e.message); }
  finally { b.disabled = false; }
}

async function run(key) {
  const a = acts.find(x => x.key === key);
  if (a && a.mail && !confirm(`「${a.name}」会真的发邮件，确定？`)) return;
  try {
    const j = await api("/api/run", { key });
    curJob = j.id; logOffset = 0; $("#joblog").textContent = "";
    $("#stopbtn").hidden = false;
    toast("已启动 " + j.name);
    pollLog();
  } catch (e) { toast(e.message); }
}

$("#stopbtn").onclick = async () => {
  if (!curJob || !confirm("停掉这次运行？")) return;
  try { await api("/api/stop", { id: curJob }); toast("已停"); }
  catch (e) { toast(e.message); }
};

/* ---------------- 输出 ---------------- */
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
      pre.insertAdjacentHTML("beforeend", j.new.map(colorize).join("\n") + "\n");
      if (stick) pre.scrollTop = pre.scrollHeight;
      logOffset = j.next;
    }
    $("#joblbl").innerHTML = j.running
      ? `<span class="spin"></span>${esc(j.name)} ${j.secs}s`
      : `${esc(j.name)} 结束，退出码 ${j.rc}，${j.secs}s`;
    $("#stopbtn").hidden = !j.running;
    clearTimeout(logTimer);
    if (j.running) logTimer = setTimeout(pollLog, 1000);
    else { refreshActions(); refresh(); }
  } catch (e) {
    $("#joblbl").textContent = e.message;
  }
}

/* ---------------- 折叠区 ---------------- */
function rcText(rc) {
  if (rc === 0) return "成功";
  if (rc === null || rc === undefined) return "";
  const m = { 267009: "在跑", 267011: "没跑过", 267014: "被终止",
              2147946720: "已在跑，忽略", 1: "退出码 1" };
  return m[rc] || ("码 " + rc);
}
const TASK_ST = { Ready: "启用", Running: "在跑", Disabled: "停用" };

function renderSched(s) {
  const box = $("#sched"); box.innerHTML = "";
  const t = el("table");
  t.innerHTML = "<tr><th>计划任务</th><th>状态</th><th>上次</th><th>结果</th><th>下次</th></tr>";
  const names = Object.keys(s.tasks || {}).sort();
  if (!names.length) t.innerHTML += '<tr><td colspan="5" class="muted">查不到</td></tr>';
  names.forEach(n => {
    const d = s.tasks[n], tr = el("tr");
    tr.title = n;
    tr.innerHTML = `<td>${esc(taskName(s, n))}</td><td>${esc(TASK_ST[d.state] || d.state)}</td>` +
      `<td class="mono muted">${esc(d.last)}</td>` +
      `<td class="mono">${d.never ? "没跑过" : rcText(d.rc)}</td>` +
      `<td class="mono muted">${esc(d.next)}</td>`;
    t.appendChild(tr);
  });
  box.appendChild(t);
  const r = s.rules || {}, t2 = el("table");
  t2.style.marginTop = "10px";
  t2.innerHTML = "<tr><th></th><th>开跑窗口</th><th>自动开跑</th><th>现在触发会怎样</th></tr>";
  (r.lines || []).forEach(x => {
    const tr = el("tr");
    tr.title = [x.local_when, x.steps, x.cloud].join("\n\n");
    tr.innerHTML = `<td><b>${esc(x.name)}</b></td><td class="mono">${esc(x.window)}</td>` +
      `<td class="mono">${esc(x.auto_from)}</td><td>${esc(x.verdict)}</td>`;
    t2.appendChild(tr);
  });
  box.appendChild(t2);
  const tip = el("div", "muted",
    "触发器按 UTC 锚定（起涨预测 08:30Z、长期调整突破 09:40Z），北京时刻固定；" +
    "定义在 tools/setup_tasks.ps1。细则悬停看。");
  tip.style.cssText = "font-size:12px;margin-top:8px";
  tip.title = [r.manual].concat(r.priority || [], r.if_needed || [], [r.worker]).join("\n");
  box.appendChild(tip);
}

async function loadFlowLog() {
  try { const j = await api("/api/logfile");
    $("#flowlog").innerHTML = j.text.split("\n").map(colorize).join("\n");
    const p = $("#flowlog"); p.scrollTop = p.scrollHeight;
  } catch (e) { $("#flowlog").textContent = e.message; }
}
async function loadConf() {
  try { const j = await api("/api/config"); $("#confbody").textContent = j.text; }
  catch (e) { $("#confbody").textContent = e.message; }
}
$("#d-sched").addEventListener("toggle", () => {
  if ($("#d-sched").open && lastStatus) renderSched(lastStatus); });
$("#d-log").addEventListener("toggle", () => { if ($("#d-log").open) loadFlowLog(); });
$("#d-conf").addEventListener("toggle", () => { if ($("#d-conf").open) loadConf(); });

/* ---------------- 刷新循环 ---------------- */
let lastStatus = null;
async function loadAi() {
  try {
    const a = await api("/api/ai");
    const al = a.alert || {}, u = a.usage || {}, ps = a.proposals || [];
    let h = "";
    if (al.date) h += `<div class="sub">${esc(al.date)} ${esc(al.kind||"")}：<b class="${al.severity==="严重"?"bad":al.severity==="注意"?"warn":""}">${esc(al.severity||"")}</b> ${esc(al.summary||"")}</div>`;
    else h += `<div class="sub">还没有复盘记录</div>`;
    if (u.today_usd != null) h += `<div class="sub${u.warn?" bad":""}">用量 今天 ${u.today_usd} 美元（${u.today_n} 次） · 本月 ${u.month_usd} 美元</div>`;
    for (const p of ps) {
      h += `<div class="prop"><b>[${esc(p.type)}] ${esc(p.title)}</b><div class="sub">${esc(p.detail)}</div>`
        + `<div class="sub">预期：${esc(p.expected_effect||"")} · 风险：${esc(p.risk||"")}</div>`
        + `<button class="go" onclick="proposal('${esc(p.id)}','approve')">批准</button> `
        + `<button class="go" onclick="proposal('${esc(p.id)}','reject')">拒绝</button></div>`;
    }
    if (!ps.length) h += `<div class="sub">没有待批准的提案</div>`;
    $("#ai").innerHTML = h;
  } catch (e) { $("#ai").innerHTML = `<div class="sub">读不到：${esc(e)}</div>`; }
}
async function proposal(id, action) {
  if (action === "approve" && !confirm("批准这个提案？参数改动会写进 overrides.json 并重跑月度重估。")) return;
  try { const r = await api("/api/proposal", { id, action }); toast(r.error || (action === "approve" ? "已批准，正在落地" : "已拒绝")); loadAi(); }
  catch (e) { toast("失败：" + e); }
}
async function refresh() {
  try {
    const s = await api("/api/status");
    loadAi();
    lastStatus = s;
    $("#clock").textContent = "北京 " + s.bj;
    const bad = !s.sync.ok || (s.cloud_cron_live || []).length;
    const undone = s.lines.filter(l => !l.before_start
      && (!l.done || l.sent === false || l.error)).length;
    $("#health").innerHTML = bad
      ? '<span class="dot bad"></span>有问题'
      : (undone ? '<span class="dot warn"></span>' + undone + " 条线还没跑"
        : (s.weekend ? '<span class="dot idle"></span>周末'
          : '<span class="dot ok"></span>正常'));
    renderSum(s); renderTasks(s);
    if ($("#d-sched").open) renderSched(s);
  } catch (e) { $("#health").textContent = "状态获取失败"; }
}
async function refreshActions() {
  try { renderActs(await api("/api/actions")); } catch (e) {}
}

refresh(); refreshActions();
setInterval(refresh, 5000);
setInterval(refreshActions, 3000);
setInterval(() => { const d = new Date(Date.now() + 8 * 3600e3);
  if (lastStatus) $("#clock").textContent = "北京 " +
    d.toISOString().slice(0, 19).replace("T", " "); }, 1000);
</script></body></html>
"""
