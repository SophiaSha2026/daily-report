"""
AI 讨论窗口（/chat，2026-10-13）。后端在 src/ai_chat.py，这里只有页面。

    /chat?kind=stock&code=300829   一只股票（首页「查一只股票」旁边的「AI 讨论」）
    /chat?kind=breakout            起涨预测今天的清单（首页卡片上的「AI」）
    /chat?kind=pullback            长期调整突破今天的清单
    /chat?kind=learn               学习状态（「学习」卡片上的「问 AI」）
    /chat?sid=<会话>               打开一场以前的讨论

左边聊天，右边是这一轮的结构化输出（结论 / 立场和置信度 / 重点 / 要点 / 原因 / 没核实的 / 改动提案）。
点左边任何一条 AI 回答，右边换成那一轮的。模型和推理等级的下拉框读 /api/chat/models
（本机 claude 本体里的模型表，CLI 升级后自动更新），选择记在浏览器里。
打开页面不自动花钱：第一句默认问题填在输入框里，点「发送」才开始。
改动提案的「批准」走 /api/proposal（和学习卡片同一个接口），落地结果在提案卡片上刷新。
"""
from __future__ import annotations

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 讨论</title>
<style>
:root{--bg:#121417;--card:#1b1e23;--line:#2a2e35;--fg:#e8eaed;--dim:#8a929c;--blue:#2f6fe0;
--ok:#7ee29a;--bad:#ff8a87;--warn:#f3c26b}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line)}
.top a{color:var(--dim);text-decoration:none;font-size:14px}
.top b{font-size:16px;margin-right:auto}
select,textarea,button{font:inherit;color:inherit}
select{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:5px 8px;max-width:190px}
button{border:0;border-radius:8px;padding:7px 14px;font-weight:600;background:var(--blue);color:#fff;cursor:pointer}
button:disabled{background:var(--line);color:#6b7280;cursor:default}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--dim);font-weight:500}
.cost{font-size:13px;color:var(--dim)}
.main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);height:calc(100% - 58px)}
.chat{display:flex;flex-direction:column;border-right:1px solid var(--line);min-height:0}
#msgs{flex:1;overflow:auto;padding:14px 16px}
.m{max-width:92%;margin:0 0 12px;padding:9px 12px;border-radius:12px;white-space:pre-wrap;word-break:break-word}
.m.u{margin-left:auto;background:#1f3357}
.m.a{background:var(--card);cursor:pointer;border:1px solid transparent}
.m.a.sel{border-color:#3d5f9e}
.m.err{background:#3a1f1f;color:var(--bad)}
.meta{font-size:12px;color:var(--dim);margin-top:4px}
#prog{padding:0 16px 8px;font-size:13px;color:var(--dim)}
#prog:empty{display:none}
.ask{display:flex;gap:8px;padding:10px 16px;border-top:1px solid var(--line)}
textarea{flex:1;min-height:58px;max-height:200px;resize:vertical;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:8px 10px}
.side{overflow:auto;padding:14px 18px}
.side:empty::before{content:"这里显示 AI 每一轮的结构化结论";color:var(--dim)}
h3{font-size:13px;color:var(--dim);font-weight:500;margin:16px 0 6px}
.concl{font-size:17px;font-weight:600;line-height:1.5}
.stance{display:inline-block;padding:2px 10px;border-radius:10px;font-size:13px;margin-right:8px;background:var(--line)}
.stance.看多{background:#1d3324;color:var(--ok)}.stance.看空{background:#3a1f1f;color:var(--bad)}
.stance.观望{background:#3a2e17;color:var(--warn)}
.bar{display:inline-block;width:120px;height:6px;background:var(--line);border-radius:3px;vertical-align:middle;overflow:hidden}
.bar i{display:block;height:100%;background:var(--blue)}
.key{background:#1f2a3d;border-radius:8px;padding:6px 10px;margin:4px 0}
ul{margin:0;padding-left:20px}li{margin:3px 0}
.tag{font-size:11px;padding:1px 6px;border-radius:6px;margin-right:6px;background:var(--line);color:var(--dim)}
.tag.新闻{color:#8fb8ff}.tag.代码{color:#d6a8ff}.tag.推断{color:var(--warn)}.tag.数据{color:var(--ok)}
.src{font-size:12px;color:var(--dim);word-break:break-all}
.src a{color:#8fb8ff}
.chg{border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:8px 0}
.chg .st{font-size:12px;color:var(--dim)}
.chg .st.applied{color:var(--ok)}.chg .st.failed,.chg .st.rejected{color:var(--bad)}
pre.diff{font:12px/1.45 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-all;margin:6px 0;
background:#15171b;border-radius:8px;padding:8px}
.del{color:var(--bad)}.add{color:var(--ok)}
details summary{cursor:pointer;color:var(--dim);font-size:13px}
@media (max-width:820px){.main{grid-template-columns:1fr;height:auto}.chat{border-right:0;min-height:70vh}
#msgs{max-height:60vh}.side{border-top:1px solid var(--line)}}
</style></head><body>
<div class="top"><a href="/?token=__TOKEN__">← 首页</a><b id="title">AI 讨论</b>
<select id="model" title="模型"></select><select id="effort" title="推理等级"></select>
<select id="hist" title="以前的讨论"><option value="">以前的讨论</option></select>
<button class="ghost" onclick="fresh()">新讨论</button><span class="cost" id="cost"></span></div>
<div class="main"><div class="chat"><div id="msgs"></div><div id="prog"></div>
<div class="ask"><textarea id="q" placeholder="问点什么。要改代码或参数就直说，AI 会给提案，你批准才改"
 onkeydown="if(event.key==='Enter'&&(event.ctrlKey||event.metaKey))send()"></textarea>
<button id="go" onclick="send()">发送</button></div></div>
<div class="side" id="side"></div></div>
<script>
const TOKEN="__TOKEN__";
const $=s=>document.querySelector(s);
const P=new URLSearchParams(location.search);
let kind=P.get("kind")||"",code=P.get("code")||"",sid=P.get("sid")||"",S=null,sel=-1,timer=null,firstAsk="";
const props={};
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function link(s){return esc(s).replace(/(https?:\/\/[^\s<]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>')}
function ls(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
async function post(u,b){const r=await fetch(u+"?token="+TOKEN,{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(b)});const j=await r.json();if(!r.ok)throw new Error(j.error||r.status);return j}
async function models(){
  const m=await (await fetch("/api/chat/models")).json();
  const want=ls("chat_model")||m.default_model;
  const ids=m.models.map(x=>x.id);if(!ids.includes(m.default_model))m.models.unshift({id:m.default_model,verified:true});
  $("#model").innerHTML=m.models.map(x=>`<option value="${esc(x.id)}"${x.id===want?" selected":""}>${esc(x.id)}${x.verified?"":" · 没用过"}</option>`).join("");
  const we=ls("chat_effort")||"";
  $("#effort").innerHTML=`<option value="">推理：默认</option>`+(m.efforts||[]).map(e=>`<option value="${esc(e)}"${e===we?" selected":""}>推理：${esc(e)}</option>`).join("");
  $("#model").onchange=()=>ls("chat_model",$("#model").value);
  $("#effort").onchange=()=>ls("chat_effort",$("#effort").value);
}
async function hist(){
  const q=new URLSearchParams({kind,code});
  const r=await (await fetch("/api/chat/list?"+q)).json();
  $("#hist").innerHTML=`<option value="">以前的讨论（${r.sessions.length}）</option>`+r.sessions.map(s=>
    `<option value="${esc(s.sid)}"${s.sid===sid?" selected":""}>${esc(s.created.slice(5,16).replace("T"," "))} · ${s.turns} 句 · ${s.usd} 美元</option>`).join("");
  $("#hist").onchange=()=>{const v=$("#hist").value;if(v)location.href=`/chat?sid=${v}&token=${TOKEN}`};
  firstAsk=r.first_ask||"";
}
function render(){
  if(!S)return;
  $("#title").textContent="AI 讨论 · "+(S.title||"");
  document.title="AI 讨论 · "+(S.title||"");
  const T=S.turns||[];let usd=0,lastA=-1;
  $("#msgs").innerHTML=T.map((t,i)=>{
    if(t.role==="user")return `<div class="m u">${esc(t.text)}</div>`;
    usd+=t.usd||0;
    if(t.error)return `<div class="m err">${esc(t.error)}<div class="meta">${esc(t.model||"")} ${t.usd?"· "+t.usd+" 美元":""}</div></div>`;
    lastA=i;
    return `<div class="m a${i===sel?" sel":""}" onclick="pick(${i})">${link(t.reply)}<div class="meta">${esc(t.model)}${t.effort?" · "+esc(t.effort):""} · ${t.usd} 美元${(t.changes||[]).length?" · 有改动提案":""}</div></div>`;
  }).join("");
  $("#cost").textContent=usd?`这场 ${usd.toFixed(2)} 美元`:"";
  if(sel<0||!T[sel]||T[sel].role!=="ai"||T[sel].error)sel=lastA;
  side();
  const busy=!!S.busy;$("#go").disabled=busy;
  $("#prog").innerHTML=busy?"在想… "+(S.progress||[]).slice(-4).map(esc).join("<br>"):"";
  if(!T.length&&!busy&&!$("#q").value)$("#q").value=firstAsk;
  $("#msgs").scrollTop=1e9;
}
function pick(i){sel=i;render()}
function side(){
  const t=(S.turns||[])[sel];
  if(!t||!t.panel){$("#side").innerHTML="";return}
  const p=t.panel,h=[];
  if(p.title)h.push(`<h3>${esc(p.title)}</h3>`);
  h.push(`<div>${p.stance&&p.stance!=="不适用"?`<span class="stance ${esc(p.stance)}">${esc(p.stance)}</span>`:""}`
    +(p.confidence!=null?`<span class="dim">置信 ${p.confidence}</span> <span class="bar"><i style="width:${p.confidence}%"></i></span>`:"")+`</div>`);
  h.push(`<h3>结论</h3><div class="concl">${esc(p.conclusion)}</div>`);
  if((p.key_points||[]).length)h.push(`<h3>重点</h3>`+p.key_points.map(k=>`<div class="key">${esc(k)}</div>`).join(""));
  if((p.bullets||[]).length)h.push(`<h3>要点</h3><ul>${p.bullets.map(b=>`<li>${esc(b)}</li>`).join("")}</ul>`);
  if((p.reasons||[]).length)h.push(`<h3>原因</h3><ul>${p.reasons.map(r=>`<li><span class="tag ${esc(r.basis)}">${esc(r.basis)}</span>${esc(r.point)}${r.source?`<div class="src">${link(r.source)}</div>`:""}</li>`).join("")}</ul>`);
  if((p.unverified||[]).length)h.push(`<h3>没核实到</h3><ul>${p.unverified.map(b=>`<li>${esc(b)}</li>`).join("")}</ul>`);
  if((t.changes||[]).length)h.push(`<h3>改动提案（你批准才会执行）</h3>`+t.changes.map(chg).join(""));
  if(t.changes_error)h.push(`<div class="src">提案没存上：${esc(t.changes_error)}</div>`);
  $("#side").innerHTML=h.join("");
  (t.changes||[]).forEach(c=>refreshProp(c.id));
}
function chg(c){
  const st=props[c.id]||{status:"pending"};
  const ed=(c.edits||[]).map(e=>`<div class="src">${esc(e.file)}</div><pre class="diff">`
    +String(e.old||"").split("\n").map(l=>`<span class="del">- ${esc(l)}</span>`).join("\n")+"\n"
    +String(e.new||"").split("\n").map(l=>`<span class="add">+ ${esc(l)}</span>`).join("\n")+`</pre>`).join("");
  const what=c.type==="param"?`参数 ${esc(c.param)} 改成 ${esc(c.value)}`:`改 ${(c.edits||[]).length} 处代码`;
  const word={pending:"等你批准",approved:"已批准，正在落地",applied:"已落地",failed:"没落地",rejected:"已拒绝"}[st.status]||st.status;
  return `<div class="chg" id="c-${esc(c.id)}"><b>${esc(c.title)}</b> <span class="st ${esc(st.status)}">${esc(word)}</span>
    <div class="src">${what}</div><div>${esc(c.detail)}</div>
    ${c.expected_effect?`<div class="src">预期：${esc(c.expected_effect)}</div>`:""}
    <div class="src">风险：${esc(c.risk)}</div>${c.how_to_verify?`<div class="src">怎么验证：${esc(c.how_to_verify)}</div>`:""}
    ${ed?`<details><summary>看改动</summary>${ed}</details>`:""}${st.note?`<div class="src">${esc(st.note)}</div>`:""}
    ${st.status==="pending"?`<div style="margin-top:8px"><button onclick="decide('${esc(c.id)}','approve')">批准</button>
    <button class="ghost" onclick="decide('${esc(c.id)}','reject')">拒绝</button></div>`:""}</div>`;
}
async function refreshProp(id){
  try{const j=await (await fetch("/api/proposal?id="+encodeURIComponent(id))).json();
    const old=props[id]&&props[id].status+props[id].note;props[id]=j;
    if(old!==j.status+j.note){const t=(S.turns||[])[sel];const c=(t.changes||[]).find(x=>x.id===id);
      const el=document.getElementById("c-"+id);if(c&&el)el.outerHTML=chg(c)}
    if(j.status==="approved")setTimeout(()=>refreshProp(id),4000);
  }catch(e){}
}
async function decide(id,action){
  if(action==="approve"&&!confirm("批准这个改动？参数会写进 overrides 并跑月度重估；代码会改文件、跑三条自测和 pyflakes，不过就撤回，过了本地提交。"))return;
  try{await post("/api/proposal",{id,action});props[id]={status:action==="approve"?"approved":"rejected"};side()}
  catch(e){alert(e.message)}
}
async function load(){
  if(!sid)return;
  try{S=await (await fetch("/api/chat/s/"+sid)).json();kind=S.kind;code=S.code||"";render();
    clearTimeout(timer);if(S.busy)timer=setTimeout(load,2000)}catch(e){timer=setTimeout(load,4000)}
}
async function send(){
  const q=$("#q").value.trim();
  $("#go").disabled=true;
  try{
    if(!sid){const s=await post("/api/chat/new",{kind,code});sid=s.sid;history.replaceState(null,"",`/chat?sid=${sid}&token=${TOKEN}`)}
    await post("/api/chat/send",{sid,message:q,model:$("#model").value,effort:$("#effort").value});
    $("#q").value="";sel=-1;load();
  }catch(e){alert(e.message);$("#go").disabled=false}
}
function fresh(){location.href=`/chat?kind=${encodeURIComponent(kind)}&code=${encodeURIComponent(code)}&token=${TOKEN}`}
(async()=>{
  await models();
  if(sid){await load()}
  else{$("#title").textContent="AI 讨论 · "+({stock:code,breakout:"起涨预测清单",pullback:"长期调整突破清单",learn:"学习状态"}[kind]||"")}
  await hist();
  if(!sid&&!$("#q").value)$("#q").value=firstAsk;
})();
</script></body></html>"""
