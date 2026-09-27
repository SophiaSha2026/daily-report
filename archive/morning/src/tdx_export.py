"""
路线 C —— 通达信「自定义数据」导出。

⚠️ 上一轮我把两套机制说混了，这里更正：
  · 扩展数据 (EXTDATA_USER)  —— 数据来源只能是「本地技术指标公式」，
    由通达信自己算，**不能从外部导入**。这条路对我们没用。
  · 自定义数据 (.901 自定义数据管理器) —— 才是能外部导入的那个。
    分两类：
      外部数据(字符串,数值) -> EXTERNSTR / EXTERNVALUE
      序列数据(日期,数值)   -> SIGNALS_USER

本模块生成「外部数据(字符串,数值)」格式，导入后可在
《行情报价 -> 列头右键 -> 自定义数据》里作为一列显示并点击排序。

格式（每行，竖线分隔，GBK 编码，CRLF 换行）：
    市场|代码|字符串值|数值
    市场：1=沪(6开头)  0=深(0/3开头)
例：
    1|600519|A#1 竞价+3.2 量能2.4% 共振4|86.3
    0|300750|B#1 低位 竞价+2.1 量能1.8%|71.5

来源：通达信社区文档与实测样例。字段含义一致，但不同版本/券商版
      通达信对条数与编码可能有差异，首次导入请用 3 行小样本验证。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 通达信自定义数据的字符串列宽度有限，控制在 ~24 个全角字符内更稳妥
MAX_STR = 48


def _mkt(code: str) -> str:
    return "1" if str(code).zfill(6)[0] == "6" else "0"


def _label(r: dict[str, Any], rank: int) -> str:
    """一行摘要，尽量短。通达信列宽有限，长了会被截断。"""
    bits = [f"{r['group']}#{rank}", f"竞价{r['gap_pct']:+.1f}",
            f"量{r['auc_ratio']*100:.1f}%"]
    if r["monotonic"] and r["slope"] > 0:
        bits.append("抬升")
    if r["sector_members"] >= 3:
        bits.append(f"共振{r['sector_members']}")
    if r["risk_tags"]:
        bits.append("!" + r["risk_tags"][0][:6])
    s = " ".join(bits)
    return s[:MAX_STR]


def write_tdx_custom(rows: list[dict], out_dir: Path,
                     data_no: int = 1) -> list[Path]:
    """
    生成两个文件：
      外部数据(字符串,数值)_{N}.txt   -> 摘要字符串 + 评分
      watchlist.ebk                    -> 自选股板块（一键导入）
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    lines = []
    for grp in ("A", "B"):
        sel = [r for r in rows if r["group"] == grp]
        for i, r in enumerate(sel, 1):
            lines.append(
                f"{_mkt(r['code'])}|{r['code']}|{_label(r, i)}|{r['score']:.1f}"
            )

    p = out_dir / f"外部数据(字符串,数值)_{data_no}.txt"
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("gbk", errors="replace"))
    paths.append(p)

    # .ebk 自选股：每行 市场前缀(1沪/0深) + 6位代码，无分隔符
    ebk = out_dir / "watchlist.ebk"
    ebk.write_bytes(
        ("\r\n".join(f"{_mkt(r['code'])}{r['code']}" for r in rows) + "\r\n")
        .encode("gbk", errors="replace")
    )
    paths.append(ebk)
    return paths


# ---------------------------------------------------------------------
#  本地一键同步脚本（Windows）—— 解决「电脑不常开」的问题
#  用户开机后双击一次，5 秒内把当日数据拉到通达信目录
# ---------------------------------------------------------------------
SYNC_PS1 = r'''# sync_tdx.ps1  —— 放桌面，开机后双击运行
# 首次使用请把 $TDX 改成你的通达信安装目录
$REPO = "https://raw.githubusercontent.com/{OWNER}/{REPO}/main/out"
$TDX  = "C:\new_tdx"

$sig = Join-Path $TDX "T0002\signals"
$blk = Join-Path $TDX "T0002\blocknew"
New-Item -ItemType Directory -Force -Path $sig, $blk | Out-Null

$stamp = Get-Date -Format "yyyyMMddHHmmss"
try {
    Invoke-WebRequest "$REPO/%E5%A4%96%E9%83%A8%E6%95%B0%E6%8D%AE(%E5%AD%97%E7%AC%A6%E4%B8%B2,%E6%95%B0%E5%80%BC)_1.txt?t=$stamp" `
        -OutFile (Join-Path $sig "extern_user_1.txt") -TimeoutSec 15
    Invoke-WebRequest "$REPO/watchlist.ebk?t=$stamp" `
        -OutFile (Join-Path $blk "AUCTION.blk") -TimeoutSec 15
    Write-Host "同步完成。通达信内按 .901 -> 修改数据 -> 导入，选 extern_user_1.txt" -ForegroundColor Green
} catch {
    Write-Host "同步失败: $_" -ForegroundColor Red
}
Start-Sleep -Seconds 3
'''


def write_sync_script(out_dir: Path, owner: str, repo: str) -> Path:
    p = out_dir / "sync_tdx.ps1"
    p.write_text(SYNC_PS1.replace("{OWNER}", owner).replace("{REPO}", repo),
                 encoding="utf-8-sig")
    return p
