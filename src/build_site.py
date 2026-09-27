"""
组装 GitHub Pages 的 _site 目录。

为什么要单独一个脚本
--------------------
一个仓库只有**一份** Pages 部署。两条线如果各自 `mkdir _site && cp 自己的东西
&& deploy`，后跑的那个会把先跑的那个页面整个冲掉，用户点进去就只剩一半。
所以发布一律走这个脚本：它从仓库里已提交的 `out_breakout/` 和 `out_pullback/`
各取一份凑齐再发布。谁后跑，发布的都是完整站点。

谁来调它：.github/workflows/pages.yml（本机推送了面板就触发）。
2026-09-27 以前 Pages 是早盘那条 auction.yml 顺手发布的，早盘系统归档之后
改成推送触发，面板推上去一两分钟就能在手机上看到。

只发布**当天新鲜**的那一份吗？不。旧的也照发：留着上一个交易日的面板不会
误导（抬头自带日期，页面还会自己检查更新），换成一句「今日暂无数据」才是
净损失。

根目录 index.html 以前是早盘竞价面板，现在是一个入口页，指向两个面板。
"""
from __future__ import annotations

import html
import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("site")

# (源目录, 发布成什么名字, stamp 发布成什么名字, 入口页上的名字)
PAGES = [
    (ROOT / "out_breakout", "breakout.html", "stamp-breakout.txt", "起涨预测"),
    (ROOT / "out_pullback", "pullback.html", "stamp-pullback.txt", "长期调整突破"),
]


def _date(src: Path) -> str:
    try:
        return str(json.loads((src / "run_meta.json").read_text(encoding="utf-8")).get("date", ""))
    except Exception:  # noqa: BLE001
        return ""


def index_html(items: list[tuple[str, str, str]]) -> str:
    """入口页：两个面板各一行（名字 + 数据日期）。"""
    rows = "".join(
        f'<a class="it" href="{html.escape(href)}"><b>{html.escape(name)}</b>'
        f'<span>{html.escape(d) or "还没有"}</span></a>'
        for name, href, d in items)
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>A股流水线</title><style>'
            'body{margin:0;background:#14161a;color:#e6e6e6;'
            'font:15px/1.5 -apple-system,"Microsoft YaHei",sans-serif}'
            '.w{max-width:560px;margin:0 auto;padding:28px 16px}'
            'h1{font-size:18px;margin:0 0 18px}'
            '.it{display:flex;justify-content:space-between;align-items:center;'
            'background:#1b1e23;border-radius:10px;padding:16px;margin-bottom:10px;'
            'color:#e6e6e6;text-decoration:none}'
            '.it span{color:#8f9aa8;font-size:13px}'
            '.tip{color:#7d8590;font-size:12px;margin-top:16px}'
            '</style></head><body><div class="w"><h1>A股流水线</h1>' + rows +
            '<div class="tip">日期是面板里清单对应的交易日（北京时间）。</div>'
            '</div></body></html>')


def main() -> int:
    # 先清空：本机的 _site 里可能还留着早盘归档前的 index/learn/council 页，
    # 发布出去就成了指向已归档内容的死页面（云端每次都是新 checkout，不受影响）
    shutil.rmtree(SITE, ignore_errors=True)
    SITE.mkdir(exist_ok=True)
    got = 0
    items = []
    for src, name, stamp_name, label in PAGES:
        panel = src / "panel.html"
        if not panel.exists():
            log.warning("跳过 %s：没有 panel.html", src.name)
            continue
        shutil.copy2(panel, SITE / name)
        s = src / "stamp.txt"
        if s.exists():
            shutil.copy2(s, SITE / stamp_name)
        else:
            # 页面靠轮询 stamp 发现自己被 CDN 缓存住了，缺了就失去自愈能力
            log.warning("%s 缺 stamp.txt，该页失去自动刷新", src.name)
        # 同花顺自选股 txt 一并发布，手机上也能直接下
        for t in src.glob("*.txt"):
            if t.name != "stamp.txt":
                shutil.copy2(t, SITE / t.name)
        items.append((label, name, _date(src)))
        got += 1
        log.info("已发布 %s -> %s", panel, name)

    if not got:
        log.error("两个面板都不存在，_site 是空的")
        return 1
    (SITE / "index.html").write_text(index_html(items), encoding="utf-8")
    log.info("_site 内容：%s", sorted(p.name for p in SITE.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
