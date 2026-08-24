"""
每周刷新行业成分缓存 + 代码表。

源的优先级：东财 -> 同花顺 -> 新浪。
东财分类更细（~90 个行业），是首选；但它在 GitHub runner 上会整段时间
RemoteDisconnected（2026-08-24 实测：板块列表第一个请求就被掐断）。
上一版没有兜底、也没有 try，直接抛异常退出，连旧缓存都保不住。

同花顺（10jqka）分类细且覆盖全，带反爬 cookie，走 akshare 的封装。
新浪那张 newSinaHy 表已经过期：只认 3000 只，688/300/301/920 整段缺，
所以排在最后，只当最后一道兜底。

⚠️ 三个源在 GitHub runner 上都可能不通（2026-08-24 实测东财全站 5xx/断连、
新浪 vip 主机返回空体）。板块表实际是由 `4-刷新板块成分表` 这个 workflow
用 Playwright 真浏览器抓的（见 `src/refresh_sector.py`），本脚本只负责代码表，
板块那部分是历史遗留的兜底，失败时保留已提交的表。
"""
from __future__ import annotations
import sys, time, json, logging
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("meta")

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 "
                     "Safari/537.36"),
      "Referer": "https://finance.sina.com.cn"}

SINA_HY = "https://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
SINA_NODE = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/"
             "json_v2.php/Market_Center.getHQNodeData"
             "?page={p}&num=100&sort=symbol&asc=1&node={node}")


# ---------------------------------------------------------------------
def sectors_em() -> list[dict]:
    """东财行业成分。分类细，首选。每个板块失败只丢该板块。"""
    import akshare as ak
    names = None
    for attempt in range(3):
        try:
            names = ak.stock_board_industry_name_em()
            break
        except Exception as e:  # noqa: BLE001
            log.warning("东财板块列表第%d次失败: %s", attempt + 1, e)
            time.sleep(1.0 * (attempt + 1))
    if names is None or not len(names):
        return []

    col = "板块名称" if "板块名称" in names.columns else names.columns[1]
    rec = []
    for i, nm in enumerate(names[col].tolist()):
        for attempt in range(2):
            try:
                cons = ak.stock_board_industry_cons_em(symbol=nm)
                ccol = "代码" if "代码" in cons.columns else cons.columns[1]
                rec += [{"code": str(c).zfill(6), "sector": nm}
                        for c in cons[ccol].astype(str)]
                break
            except Exception as e:  # noqa: BLE001
                if attempt:
                    log.warning("行业 %s 失败: %s", nm, e)
                time.sleep(0.3)
        if i % 20 == 0:
            log.info("  东财 %d/%d，已收 %d 条", i, len(names), len(rec))
        time.sleep(0.25)
    return rec


# ---------------------------------------------------------------------
def sectors_ths() -> list[dict]:
    """
    同花顺行业成分。分类细、覆盖全（含科创板/北交所/次新），是东财之外
    唯一粒度够用的全覆盖源。

    10jqka 的成分接口带反爬：直接请求返回 401，要带一个由页面 JS 算出来的
    v 值 Cookie。akshare 内置了那段 ths.js 并用 py_mini_racer 执行，
    所以这里直接用 akshare 的封装，不自己复刻算法。
    py_mini_racer 已写进 requirements.txt，缺它 akshare 会在导入时报错。
    """
    import akshare as ak
    names = None
    for attempt in range(3):
        try:
            names = ak.stock_board_industry_name_ths()
            break
        except Exception as e:  # noqa: BLE001
            log.warning("同花顺行业列表第%d次失败: %s", attempt + 1, e)
            time.sleep(1.0 * (attempt + 1))
    if names is None or not len(names):
        return []

    col = "name" if "name" in names.columns else names.columns[0]
    rec = []
    for i, nm in enumerate(names[col].tolist()):
        for attempt in range(3):
            try:
                cons = ak.stock_board_industry_cons_ths(symbol=str(nm))
                ccol = next(c for c in cons.columns if "代码" in str(c))
                rec += [{"code": str(c).zfill(6), "sector": str(nm)}
                        for c in cons[ccol].astype(str)]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    log.warning("行业 %s 失败: %s", nm, e)
                time.sleep(0.6 * (attempt + 1))
        if i % 20 == 0:
            log.info("  同花顺 %d/%d，已收 %d 条", i, len(names), len(rec))
        time.sleep(0.2)
    return rec


# ---------------------------------------------------------------------
def _sina_industries() -> list[tuple[str, str]]:
    """
    [(node, 行业名)]。行业清单变动极慢，优先读仓库里的静态表。

    newSinaHy.php 在 GitHub runner 上返回的不是那段 JS 赋值（2026-08-24 实测
    解析直接 substring not found），但成分股接口 Market_Center 是好的。
    所以清单走静态文件，只有本地手动刷新时才去请求它。
    """
    p = ROOT / "cache" / "sina_industries.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return [(d["node"], d["name"]) for d in data]

    r = requests.get(SINA_HY, headers=UA, timeout=15)
    r.encoding = "gbk"
    if "{" not in r.text:
        raise RuntimeError(
            f"newSinaHy 返回异常 status={r.status_code} len={len(r.text)} "
            f"前120字={r.text[:120]!r}")
    body = r.text[r.text.index("{"): r.text.rindex("}") + 1]
    out = []
    for node, val in json.loads(body).items():
        parts = val.split(",")
        if len(parts) >= 2:
            out.append((node, parts[1]))
    return out


def _sina_cons(node: str) -> list[str]:
    codes = []
    for p in range(1, 12):                 # 单个行业超过 1100 只不可能
        r = requests.get(SINA_NODE.format(p=p, node=node), headers=UA,
                         timeout=15)
        r.encoding = "gbk"
        t = r.text.strip()
        if not t or t in ("null", "[]"):
            break
        part = json.loads(t)
        codes += [str(d["code"]).zfill(6) for d in part]
        if len(part) < 100:
            break
    return codes


def sectors_sina() -> list[dict]:
    """新浪行业成分。粒度粗，但主机稳定。"""
    inds = _sina_industries()
    log.info("新浪行业 %d 个", len(inds))
    rec = []
    for i, (node, name) in enumerate(inds):
        try:
            rec += [{"code": c, "sector": name} for c in _sina_cons(node)]
        except Exception as e:  # noqa: BLE001
            log.warning("行业 %s(%s) 失败: %s", name, node, e)
        if i % 10 == 0:
            log.info("  新浪 %d/%d，已收 %d 条", i, len(inds), len(rec))
        time.sleep(0.1)
    return rec


# ---------------------------------------------------------------------
def main() -> int:
    import datasource as ds
    try:
        ds.refresh_code_list()
    except Exception as e:  # noqa: BLE001
        log.warning("代码表刷新失败，沿用旧缓存: %s", e)

    rec: list[dict] = []
    for tag, fn in (("东财", sectors_em), ("同花顺", sectors_ths),
                    ("新浪", sectors_sina)):
        try:
            rec = fn()
        except Exception as e:  # noqa: BLE001
            log.warning("%s 板块源异常: %s", tag, e)
            rec = []
        if len(rec) >= 2000:
            log.info("板块源采用 %s：%d 条", tag, len(rec))
            break
        log.warning("%s 板块源只拿到 %d 条，换下一个源", tag, len(rec))

    if len(rec) < 2000:
        log.error("所有板块源都不可用，保留旧缓存，本次不覆盖")
        return 1

    out = ROOT / "cache"; out.mkdir(exist_ok=True)
    dst = out / "sector_map.parquet"
    df = pd.DataFrame(rec).drop_duplicates("code")

    # 仓库里那张表是浏览器里抓的同花顺全量（5400+ 只，候选池覆盖 99%）。
    # 这里能跑通的源多半是新浪那张过期表（3000 只，688/300/301/920 整段缺），
    # 拿它盖掉好表是净损失。所以只有新表不比旧表差才允许覆盖。
    if dst.exists():
        old_n = len(pd.read_parquet(dst))
        if len(df) < old_n * 0.9:
            log.warning("新表 %d 只，明显少于现有的 %d 只，放弃覆盖",
                        len(df), old_n)
            return 1

    df.to_parquet(dst, index=False)
    log.info("已写入 %d 只股票的行业归属，%d 个行业", len(df), df.sector.nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
