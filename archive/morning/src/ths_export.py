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
import html as _h
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
        # 空层也写（空文件）：以前空层不写，上一个有「强」的交易日留下的
        # 竞价_强.txt 一直躺在 out/ 里，被 build_site 发布到 Pages、
        # 被人当成今天的导入同花顺。
        p = out_dir / f"竞价_{name}.txt"
        p.write_bytes(("\r\n".join(codes) + ("\r\n" if codes else ""))
                      .encode("gbk"))
        paths.append(p)

    # 和上面分层那三个同口径：空榜写 0 字节，不是一个孤零零的换行
    p = out_dir / "竞价_全部.txt"
    p.write_bytes(("\r\n".join(r["code"] for r in rows)
                   + ("\r\n" if rows else "")).encode("gbk"))
    paths.insert(0, p)
    return paths


# ---------------------------------------------------------------------
from panel_style import PANEL_CSS, REFRESH_JS  # noqa: E402,F401  2026-09-27 抽成共用模块

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


def _criteria_line(sc: dict, late: bool = False) -> str:
    """把当前生效的准入区间渲染成一行。

    写死一段文案的话，改 config 之后面板会继续显示旧口径，看不出改动生效没有。
    量比是 AUC_RATIO 的换算显示值，筛选本身仍然用 AUC_RATIO。

    late=True 是抢救日：那一跑的量能区间被放成 0~1e9（run_auction.salvage_screen），
    照着渲染会印成「量比 0.0~240000000000」，不如直接说已放开。
    """
    k = sc.get("liangbi_per_auc_ratio", 240)
    vol = ('量比 已放开（抢救：累计额混入连续竞价，量能维度停用）' if late else
           f'量比 {sc["auc_ratio_min"]*k:.1f}~{sc["auc_ratio_max"]*k:.0f} '
           f'（竞价量能 {sc["auc_ratio_min"]*100:.2f}%~'
           f'{sc["auc_ratio_max"]*100:.2f}%）')
    return (f'准入：竞价涨幅 {sc["gap_pct_min"]:.0f}%~{sc["gap_pct_max"]:.0f}% · '
            f'{vol} · 竞价额 ≥ {sc["min_auc_amount_wan"]:.0f} 万')


def write_ths_panel(rows: list[dict], texts: dict, out_dir: Path,
                    tiers: list[float], date: str, notice: str = "",
                    screen: dict | None = None,
                    shadow_rows: list | None = None,
                    collected: str = "", late: bool = False) -> Path:
    """collected = 实际采集时刻（run_meta.captured_at），以前写死 09:25:10。"""
    mm = int((screen or {}).get("sector_min_members", 3))
    tr = []
    data = []
    for i, r in enumerate(rows, 1):
        tier = _tier(r["score"], tiers)
        data.append({"code": r["code"], "tier": tier})
        t = texts.get(r["code"], {})
        shape = "抬升" if (r["monotonic"] and r["slope"] > 0) else (
                "走弱" if r["slope"] < 0 else "震荡")
        rs = r["risk_tags"]
        # LLM 文案和名称是外部输入，转义后再拼（同 mailer._rows_html）。
        # 面板会发布到公开的 GitHub Pages，未转义的 `<script>` 就在那上面执行。
        cell = (f'<div class="rn">{_h.escape(t.get("reason", ""))}</div>'
                if t.get("reason") else "")
        rk = _h.escape(t.get("risk") or "") or (" / ".join(rs) if rs else "")
        if rk:
            cell += f'<div class="rz">⚠ {rk}</div>'
        tr.append(
            f'<tr><td>{i}</td>'
            f'<td class="code" onclick="one(\'{r["code"]}\')">{r["code"]}</td>'
            f'<td>{_h.escape(str(r["name"]))}</td>'
            f'<td class="t{tier}">{tier}</td>'
            f'<td>{r["auc_price"]:.2f}</td>'
            f'<td class="up">+{r["gap_pct"]:.2f}%</td>'
            f'<td>{r["auc_ratio"]*100:.2f}% ({r.get("liangbi", 0):.1f})</td>'
            f'<td>{shape} {r["slope"]:+.1f}</td>'
            f'<td>{_h.escape(str(r["sector"]))}'
            + (f'·{r["sector_members"]}'
               if r["sector_members"] >= mm else '')
            + f'</td><td class="sc">{r["score"]:.0f}</td>'
            f'<td>{cell}</td></tr>'
        )
    sub = f'共 {len(rows)} 只 · 采集于 {collected or "未知"}'
    if screen:
        sub += ' · ' + _criteria_line(screen, late=late)
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
            f"""<tr><td>{i}</td><td class="code" onclick="one('{r["code"]}')">{r["code"]}</td><td>{_h.escape(str(r["name"]))}</td><td class="up">+{r["gap_pct"]:.2f}%</td><td>{r.get("liangbi", 0):.1f}</td><td>{r["sscore"]:+.2f}</td></tr>"""
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
            # 竞价面板的日期就是今天，日期横幅对它有意义
            .replace("__LAGOK__", "false")
            .replace("__ROWS__", "".join(tr))
            .replace("__SHADOW__", shadow_html)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    p = out_dir / "panel.html"
    p.write_text(html, encoding="utf-8")
    return p
