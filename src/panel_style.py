"""
面板共用的样式和自动刷新脚本。起涨预测、长期调整突破两个面板都用它。

2026-09-27 从 ths_export.py 抽出来：那个文件是早盘选股（竞价面板 + 同花顺
分层板块）专用的，早盘系统归档到 archive/morning/ 之后，两个晚间面板不能
还从一个已归档的模块里拿样式。内容逐字搬过来，没改一个字符，面板长相不变。
"""
from __future__ import annotations

PANEL_CSS = """*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,'Microsoft YaHei',sans-serif;margin:0;
     padding:14px;background:#14161a;color:#e6e6e6}
h1{font-size:15px;margin:0 0 4px}
.sub{color:#888;font-size:12px;margin-bottom:12px}
.bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
button{background:#2a2f38;color:#e6e6e6;border:1px solid #3a4149;border-radius:5px;
       padding:6px 12px;font-size:13px;cursor:pointer}
button:hover{background:#39404b}
button.on{background:#c1440e;border-color:#c1440e}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:#1e2229;text-align:left;padding:7px 8px;position:sticky;top:0;
   border-bottom:1px solid #333;white-space:nowrap;font-weight:600}
td{padding:7px 8px;border-bottom:1px solid #232830;vertical-align:top}
tr:hover{background:#1b1f26}
.code{font-family:Consolas,monospace;font-weight:600;color:#7fb3ff;cursor:pointer}
.up{color:#ff6b6b}.sc{font-weight:700;color:#ffb347}
.t强{color:#ff6b6b}.t中{color:#ffb347}.t观察{color:#8f9aa8}
.rn{color:#c9d1d9;font-size:12px}.rz{color:#d0a34a;font-size:12px;margin-top:2px}
.tip{color:#7d8590;font-size:12px;margin-top:14px;line-height:1.7}
#toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
   background:#c1440e;padding:8px 18px;border-radius:5px;opacity:0;
   transition:.25s;font-size:13px;pointer-events:none}
#toast.show{opacity:1}
#stale{display:none;background:#3a2d16;border:1px solid #6b5320;color:#e8c877;
   padding:8px 12px;border-radius:5px;font-size:12px;margin-bottom:12px}
"""

# 自动刷新脚本，三个面板共用。__STAMP__ / __DATE__ / __LAGOK__ 由调用方替换。
# __LAGOK__：这条线的面板日期本来就落后今天吗。竞价面板 false（它的日期
# 就是今天），起涨预测和形态面板 true（它们的日期是最近一个已收盘交易日）。
# 不替换时表达式退化成 false，也就是保持老行为，不会因为漏了一处就炸掉脚本。
REFRESH_JS = """/* ---------------------------------------------------------------------
   自动刷新。为什么需要：
   GitHub Pages 给 index.html 挂的是 Cache-Control: max-age=600，
   邮件 09:27:31 到、Pages 09:27:42 才部署完，中间还隔着 CDN 那 10 分钟。
   用户点邮件里的链接进来，拿到的常常是上一个交易日的面板。
   页面本身没法改响应头，但可以自己发现「我过期了」然后跳到一个新 URL：
   带上 ?v=<新stamp> 就是不同的缓存键，必然回源。
   stamp.txt 的请求也带 cb=<随机> 绕开缓存，否则查的还是旧的。
   --------------------------------------------------------------------- */
const STAMP='__STAMP__', PDATE='__DATE__', LAGOK=('__LAGOK__'==='true');
function bjToday(){
  // Date.now() 已经是 UTC 毫秒，加 8 小时再按 UTC 取日期就是北京日期。
  // 以前还加了一次 getTimezoneOffset，美东浏览器算成北京 +4 小时，
  // 每天 20:00 之后横幅误报「数据过期」。
  return new Date(Date.now()+8*36e5).toISOString().slice(0,10);
}
function banner(msg){
  const e=document.getElementById('stale');
  e.textContent=msg; e.style.display=msg?'block':'none';
}
let tries=0;
function poll(){
  fetch('__STAMPFILE__?cb='+Date.now()+Math.random(),{cache:'no-store'})
    .then(r=>r.ok?r.text():null)
    .then(s=>{
      if(!s) return;
      s=s.trim();
      if(s && s!==STAMP){
        /* 防死循环：同一个 stamp 只跳一次 */
        if(sessionStorage.getItem('jumped')===s) return;
        sessionStorage.setItem('jumped',s);
        location.replace(location.pathname+'?v='+encodeURIComponent(s));
      }
    }).catch(()=>{});
}
(function(){
  /* 横幅只在「这条线的面板日期本来该等于今天」且「是工作日」时才有意义。
     起涨预测/形态的面板日期就是最近一个已收盘交易日，按旧口径 720/720 小时
     全在挂「数据过期」，等于没有过期检测；竞价面板周末两天也是无条件误报。 */
  const t=bjToday(), wd=new Date(Date.now()+8*36e5).getUTCDay();
  if(!LAGOK && PDATE!==t && wd>=1 && wd<=5){
    banner('面板数据日期 '+PDATE+'，当前北京 '+t+
           '。若今日榜单已发布，本页会自动刷新（每 15 秒检查一次）。');
  }
  poll();
  /* 前 20 分钟每 15 秒查一次，够覆盖发信到 Pages 部署完成的窗口 */
  const id=setInterval(()=>{ if(++tries>80){clearInterval(id);return;} poll(); },15000);
})();
"""
