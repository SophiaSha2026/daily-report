"""
刷新同花顺行业成分表（无人值守）。

为什么必须开真浏览器
--------------------
10jqka 的成分 ajax 接口挡的是**客户端指纹**，不是 IP：用 requests 带上
akshare 那套 ths.js 算出来的 v cookie，照样 403 Nginx forbidden；同一台
机器同一个出口 IP，在真实浏览器会话里 fetch 同一个 URL 直接 200。
所以这里用 Playwright 起 chromium，在页面上下文里发请求。

分页截断
--------
接口在第 5 页（100 只）硬截断，但 page_info 会报真实页数（比如 1/14）。
用同一排序字段 desc + asc 各取 5 页合并，覆盖 200 只以内的行业；
超过 200 只的行业再换几个排序字段补中段（field id 从表头 <a field=...> 读到）。

失败不覆盖
----------
跑不通就保留仓库里已提交的表。宁可用一张上个月的表，也不要用一张残表——
板块共振是打分的一个维度，成分缺一半比整个维度关掉更糟。
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "cache" / "sector_map.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sector")

LIST_URL = "https://q.10jqka.com.cn/thshy/"

# 表头里读到的排序字段。前两个够覆盖 200 只以内的行业，
# 后面几个只在超大行业上追加，用来补中段。
FIELD_MAIN = "199112"                                  # 涨跌幅
FIELD_EXTRA = ["10", "19", "3475914", "2034120"]       # 现价/成交额/流通市值/市盈率

# 在页面上下文里跑：同源 fetch，带完整 cookie，绕开指纹拦截。
JS_GRAB = """
async ([code, field, order]) => {
  const out = []; let pages = 1;
  for (let p = 1; p <= 5; p++) {
    const url = `https://q.10jqka.com.cn/thshy/detail/field/${field}/order/${order}`
              + `/page/${p}/ajax/1/code/${code}`;
    const r = await fetch(url, {credentials: 'include',
                               headers: {'X-Requested-With': 'XMLHttpRequest'}});
    if (r.status !== 200) break;
    const t = new TextDecoder('gbk').decode(await r.arrayBuffer());
    if (p === 1) { const m = t.match(/page_info[^>]*>1\\/(\\d+)</); pages = m ? +m[1] : 1; }
    const doc = new DOMParser().parseFromString(t, 'text/html');
    doc.querySelectorAll('tr').forEach(tr => {
      const td = tr.querySelectorAll('td');
      if (td.length > 1) { const c = td[1].textContent.trim();
                           if (/^\\d{6}$/.test(c)) out.push(c); }
    });
    if (p >= pages) break;
    await new Promise(s => setTimeout(s, 200));
  }
  return {codes: out, pages};
}
"""

JS_LIST = """
() => { const m = {};
  document.querySelectorAll('a[href*="/thshy/detail/code/"]').forEach(a => {
    const c = a.href.match(/code\\/(\\d+)/); if (c) m[c[1]] = a.textContent.trim(); });
  return m; }
"""


def scrape() -> dict[str, str]:
    from playwright.sync_api import sync_playwright

    merged: dict[str, str] = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        ctx = br.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            locale="zh-CN")
        pg = ctx.new_page()
        pg.goto(LIST_URL, wait_until="domcontentloaded", timeout=60_000)
        pg.wait_for_timeout(1500)

        inds: dict[str, str] = pg.evaluate(JS_LIST)
        log.info("行业 %d 个", len(inds))
        if len(inds) < 50:
            raise RuntimeError(f"行业列表只拿到 {len(inds)} 个，页面可能被拦")

        big: list[tuple[str, str]] = []
        for i, (code, name) in enumerate(inds.items(), 1):
            got = 0
            for order in ("desc", "asc"):
                res = pg.evaluate(JS_GRAB, [code, FIELD_MAIN, order])
                for c in res["codes"]:
                    merged.setdefault(c, name)
                got = max(got, res["pages"])
                pg.wait_for_timeout(150)
            if got > 10:                       # 超过 200 只，两头取不全
                big.append((code, name))
            if i % 20 == 0:
                log.info("  %d/%d，累计 %d 只", i, len(inds), len(merged))

        for code, name in big:
            log.info("  大行业补抓 %s", name)
            for field in FIELD_EXTRA:
                for order in ("desc", "asc"):
                    res = pg.evaluate(JS_GRAB, [code, field, order])
                    for c in res["codes"]:
                        merged.setdefault(c, name)
                    pg.wait_for_timeout(150)

        br.close()
    return merged


def main() -> int:
    # 抓不到不算失败：仓库里那张表还在，板块共振照常工作，只是不更新。
    # 让 workflow 保持绿色是刻意的——用户不需要每周去看一眼红叉。
    # 真出问题的信号是每天的竞价邮件，不是这个。
    def _warn(msg: str) -> int:
        print(f"::warning::板块表未更新：{msg}")
        log.warning(msg)
        return 0 if DST.exists() else 1

    try:
        merged = scrape()
    except Exception as e:  # noqa: BLE001
        return _warn(f"抓取失败 {type(e).__name__}: {e}")

    df = pd.DataFrame([{"code": c, "sector": s} for c, s in sorted(merged.items())])
    log.info("抓到 %d 只，%d 个行业", len(df), df.sector.nunique())

    if len(df) < 3000:
        return _warn(f"只抓到 {len(df)} 只，明显不完整，不覆盖")
    if DST.exists():
        old_n = len(pd.read_parquet(DST))
        if len(df) < old_n * 0.9:
            return _warn(f"新表 {len(df)} 只 < 现有 {old_n} 只的 90%，不覆盖")

    codes_csv = ROOT / "cache" / "codes.csv"
    if codes_csv.exists():
        codes = [l.strip() for l in
                 codes_csv.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
        hit = sum(1 for c in codes if c in merged)
        log.info("全市场 %d 只，覆盖 %d (%.1f%%)", len(codes), hit, hit / len(codes) * 100)

    DST.parent.mkdir(exist_ok=True)
    df.to_parquet(DST, index=False)
    log.info("已写入 %s", DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
