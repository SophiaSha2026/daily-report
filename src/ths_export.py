"""
同花顺输出。

⚠️ 重要结论（查证后更正上一轮）：
    同花顺**没有**通达信那种「自定义数据管理器」——不能把外部算好的
    数值/字符串导入成行情列表里的一列。
    社区工具链的方向全部是「同花顺 -> 通达信」：把同花顺的表头数据导出，
    再做成通达信的自定义外部数据。反方向不存在。

同花顺实际支持的两条：
  1) 自选股板块设置 -> 导入 -> 文件类型选 TXT -> 纯代码列表
  2) 剪贴板识别：复制一列 6 位代码，同花顺会自动弹出识别框，
     点「加入自选股/板块股」即可

因此路线 C 在同花顺上降级为：
  · 用**分层板块**表达排名（强 / 中 / 观察 三个板块）
  · 理由与风险放在本地 HTML 面板，配一键复制按钮，
    利用剪贴板识别把任意子集推进同花顺

如果你愿意额外装一个通达信（免费、体积小、可与同花顺共存）当评分看板，
tdx_export.py 里的完整版随时可用——那边能做到真正的可排序评分列。
"""
from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path


def _tier(score: float, tiers: list[float]) -> str:
    if score >= tiers[0]:
        return "强"
    if score >= tiers[1]:
        return "中"
    return "观察"


def write_ths_blocks(rows: list[dict], out_dir: Path,
                     tiers: list[float], date: str) -> list[Path]:
    """
    生成同花顺可导入的分层板块 TXT。
    格式：每行一个 6 位代码，无前缀、无表头。GBK + CRLF。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[str]] = {"强": [], "中": [], "观察": []}
    for r in rows:
        buckets[_tier(r["score"], tiers)].append(r["code"])

    paths = []
    for name, codes in buckets.items():
        if not codes:
            continue
        p = out_dir / f"竞价_{name}.txt"
        p.write_bytes(("\r\n".join(codes) + "\r\n").encode("gbk"))
        paths.append(p)

    p = out_dir / "竞价_全部.txt"
    p.write_bytes(("\r\n".join(r["code"] for r in rows) + "\r\n").encode("gbk"))
    paths.insert(0, p)
    return paths


# ---------------------------------------------------------------------
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

# 自动刷新脚本，竞价面板和形态面板共用。__STAMP__ / __DATE__ 由调用方替换。
REFRESH_JS = """/* ---------------------------------------------------------------------
   自动刷新。为什么需要：
   GitHub Pages 给 index.html 挂的是 Cache-Control: max-age=600，
   邮件 09:27:31 到、Pages 09:27:42 才部署完，中间还隔着 CDN 那 10 分钟。
   用户点邮件里的链接进来，拿到的常常是上一个交易日的面板。
   页面本身没法改响应头，但可以自己发现「我过期了」然后跳到一个新 URL：
   带上 ?v=<新stamp> 就是不同的缓存键，必然回源。
   stamp.txt 的请求也带 cb=<随机> 绕开缓存，否则查的还是旧的。
   --------------------------------------------------------------------- */
const STAMP='__STAMP__', PDATE='__DATE__';
function bjToday(){
  const d=new Date(Date.now()+(new Date().getTimezoneOffset()*6e4)+8*36e5);
  return d.toISOString().slice(0,10);
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
  const t=bjToday();
  if(PDATE!==t){
    banner('面板数据日期 '+PDATE+'，当前北京 '+t+
           '。若今日榜单已发布，本页会自动刷新（每 15 秒检查一次）。');
  }
  poll();
  /* 前 20 分钟每 15 秒查一次，够覆盖发信到 Pages 部署完成的窗口 */
  const id=setInterval(()=>{ if(++tries>80){clearInterval(id);return;} poll(); },15000);
})();
"""

_PANEL = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>竞价榜 __DATE__</title><style>
""" + PANEL_CSS + """</style></head><body>
<div id="stale"></div>
<h1>集合竞价榜 · __DATE__</h1>
<div class="sub">__SUB__</div>
<div class="bar">
  <button onclick="cp('all',this)">复制全部代码</button>
  <button onclick="cp('强',this)">仅「强」</button>
  <button onclick="cp('中',this)">仅「中」</button>
</div>
<table><thead><tr>
<th>#</th><th>代码</th><th>名称</th><th>层</th><th>竞价价</th><th>高开</th>
<th>量能(量比)</th><th>形态</th><th>板块</th><th>分</th><th>理由 / 风险</th>
</tr></thead><tbody>__ROWS__</tbody></table>
__SHADOW__
<div class="tip">
<a href="learn.html" style="color:#8ab4f8">→ 学习面板（自学习系统状态、影子榜战绩、闸门裁决）</a> ·
<a href="pullback.html" style="color:#8ab4f8">→ 形态面板</a><br>
点任意代码即复制该代码；上方按钮批量复制。<br>
复制后切到同花顺，剪贴板识别框会自动弹出 → 点「加入自选股/板块股」。<br>
或：自选股板块设置 → 导入 → 文件类型选 TXT → 选 out/ 目录下的 竞价_*.txt。
</div>
<div id="toast"></div>
<script>
const D=__DATA__;
function toast(m){const t=document.getElementById('toast');t.textContent=m;
  t.className='show';setTimeout(()=>t.className='',1300);}
function put(txt,msg){
  navigator.clipboard.writeText(txt).then(()=>toast(msg))
  .catch(()=>{const a=document.createElement('textarea');a.value=txt;
    document.body.appendChild(a);a.select();document.execCommand('copy');
    a.remove();toast(msg);});
}
function cp(k,btn){
  const s=(k==='all'?D:D.filter(x=>x.tier===k)).map(x=>x.code);
  if(!s.length){toast('该层为空');return;}
  put(s.join('\\n'), '已复制 '+s.length+' 个代码');
  document.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
}
function one(c){put(c,'已复制 '+c);}

""" + REFRESH_JS + """</script></body></html>"""


def _criteria_line(sc: dict) -> str:
    """把当前生效的准入区间渲染成一行。

    写死一段文案的话，改 config 之后面板会继续显示旧口径，看不出改动生效没有。
    量比是 AUC_RATIO 的换算显示值，筛选本身仍然用 AUC_RATIO。
    """
    k = sc.get("liangbi_per_auc_ratio", 240)
    return (f'准入：竞价涨幅 {sc["gap_pct_min"]:.0f}%~{sc["gap_pct_max"]:.0f}% · '
            f'量比 {sc["auc_ratio_min"]*k:.1f}~{sc["auc_ratio_max"]*k:.0f} '
            f'（竞价量能 {sc["auc_ratio_min"]*100:.2f}%~{sc["auc_ratio_max"]*100:.2f}%）'
            f' · 竞价额 ≥ {sc["min_auc_amount_wan"]:.0f} 万')


def write_ths_panel(rows: list[dict], texts: dict, out_dir: Path,
                    tiers: list[float], date: str, notice: str = "",
                    screen: dict | None = None,
                    shadow_rows: list | None = None) -> Path:
    tr = []
    data = []
    for i, r in enumerate(rows, 1):
        tier = _tier(r["score"], tiers)
        data.append({"code": r["code"], "tier": tier})
        t = texts.get(r["code"], {})
        shape = "抬升" if (r["monotonic"] and r["slope"] > 0) else (
                "走弱" if r["slope"] < 0 else "震荡")
        rs = r["risk_tags"]
        cell = (f'<div class="rn">{t.get("reason","")}</div>'
                if t.get("reason") else "")
        rk = t.get("risk") or (" / ".join(rs) if rs else "")
        if rk:
            cell += f'<div class="rz">⚠ {rk}</div>'
        tr.append(
            f'<tr><td>{i}</td>'
            f'<td class="code" onclick="one(\'{r["code"]}\')">{r["code"]}</td>'
            f'<td>{r["name"]}</td><td class="t{tier}">{tier}</td>'
            f'<td>{r["auc_price"]:.2f}</td>'
            f'<td class="up">+{r["gap_pct"]:.2f}%</td>'
            f'<td>{r["auc_ratio"]*100:.2f}% ({r.get("liangbi", 0):.1f})</td>'
            f'<td>{shape} {r["slope"]:+.1f}</td>'
            f'<td>{r["sector"]}'
            + (f'·{r["sector_members"]}' if r["sector_members"] >= 3 else '')
            + f'</td><td class="sc">{r["score"]:.0f}</td>'
            f'<td>{cell}</td></tr>'
        )
    sub = f'共 {len(rows)} 只 · 采集于 09:25:10'
    if screen:
        sub += ' · ' + _criteria_line(screen)
    if notice:
        sub += f' · <span style="color:#d0a34a">{notice}</span>'
    # 每次生成都换一个 stamp。页面拿它跟 stamp.txt 比对，不一致就跳新 URL，
    # 借此绕开 GitHub Pages 那 600 秒的 CDN 缓存（详见 _PANEL 里的注释）。
    stamp = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime(
        "%Y%m%d-%H%M%S")
    (out_dir / "stamp.txt").write_text(stamp, encoding="utf-8")

    shadow_html = ""
    if shadow_rows:
        body = "".join(
            f"""<tr><td>{i}</td><td class="code" onclick="one('{r["code"]}')">{r["code"]}</td><td>{r["name"]}</td><td class="up">+{r["gap_pct"]:.2f}%</td><td>{r.get("liangbi", 0):.1f}</td><td>{r["sscore"]:+.2f}</td></tr>"""
            for i, r in enumerate(shadow_rows, 1))
        shadow_html = (
            '<h1 style="font-size:15px;margin-top:18px">影子参考榜（试运行）</h1>'
            '<div class="sub">候任 27 特征线性模型的前 10 · 并行考核中 · '
            '正式榜在上方，此榜仅参考</div>'
            '<table><thead><tr><th>#</th><th>代码</th><th>名称</th>'
            '<th>高开</th><th>量比</th><th>影子分</th></tr></thead>'
            '<tbody>' + body + '</tbody></table>')

    html = (_PANEL.replace("__DATE__", date).replace("__SUB__", sub)
            .replace("__STAMP__", stamp)
            .replace("__STAMPFILE__", "stamp.txt")
            .replace("__ROWS__", "".join(tr))
            .replace("__SHADOW__", shadow_html)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    p = out_dir / "panel.html"
    p.write_text(html, encoding="utf-8")
    return p
