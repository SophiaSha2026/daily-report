"""
组装 GitHub Pages 的 _site 目录。

为什么要单独一个脚本
--------------------
一个仓库只有**一份** Pages 部署。竞价流水线和形态扫描是两个 workflow，
如果各自 `mkdir _site && cp 自己的东西 && deploy`，后跑的那个会把先跑的
那个页面整个冲掉，用户点进去就只剩一半。

所以两条线都调这个脚本：它从仓库里已提交的 `out/` 和 `out_pullback/`
各取一份，凑齐再发布。谁后跑，发布的都是完整站点。

只发布**当天新鲜**的那一份吗？不。旧的也照发。理由和竞价那边一样：
留着上一个交易日的面板不会误导（抬头自带日期，页面还会自己检查更新），
把它换成一句「今日暂无数据」才是净损失。
"""
from __future__ import annotations

import sys
import shutil
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("site")

# (源目录, 发布成什么名字, stamp 发布成什么名字)
PAGES = [
    (ROOT / "out",          "index.html",    "stamp.txt"),
    (ROOT / "out_pullback", "pullback.html", "stamp-pullback.txt"),
]


def main() -> int:
    SITE.mkdir(exist_ok=True)
    got = 0
    for src, name, stamp_name in PAGES:
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
        got += 1
        log.info("已发布 %s -> %s", panel, name)

    if not got:
        log.error("两个面板都不存在，_site 是空的")
        return 1
    log.info("_site 内容：%s", sorted(p.name for p in SITE.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
