"""
控制台体检用的探针：行情源可达性 + 本地依赖。

单独成文件而不是内嵌在 panel.ps1 里，有两个原因：
  1. PowerShell 的 here-string（@'...'@）在某些 shell 里会破坏命令解析
  2. 内嵌 python 的中文输出经 PowerShell 管道会变成 ??????，
     所以这里标签一律 ASCII

输出格式固定为 "OK   xxx" / "FAIL xxx" / "MISS xxx"，
panel.ps1 靠开头那个词决定显示成绿色还是红色。
"""
import importlib
import pathlib
import socket
import sys
import time
import urllib.request

socket.setdefaulttimeout(10)

# 晚间两条线实际要连的主机（2026-09-27 早盘归档后重排，逐个本机实测过）：
#   每天：腾讯快照追加当天 K 线、拿名称；新浪交易日历；东财 datacenter 刷股东户数
#         （起涨预测 backfill --stage update 顺手刷，失败沿用上一份）
#   重下三年日线时：新浪日线（前复权 + 股本）为主，腾讯日 K 兜底
# 东财 push2his 日线（归档前早盘盘前用）本机也不通，已不再探测。
SOURCES = [
    ("Tencent quote ", "https://qt.gtimg.cn/q=sh600000"),
    ("Sina calendar ", "https://finance.sina.com.cn/realstock/company/klc_td_sh.txt"),
    ("Sina daily    ", "https://finance.sina.com.cn/realstock/company/sh600000/qfq.js"),
    ("Tencent kline ", "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                       "?param=sh600000,day,,,10,"),
    ("EastMoney(opt)", "https://datacenter-web.eastmoney.com/api/data/v1/get"
                       "?reportName=RPT_HOLDERNUMLATEST&columns=SECURITY_CODE,END_DATE"
                       "&pageSize=1&pageNumber=1&source=WEB&client=WEB"),
]

DEPS = ("pandas", "pyarrow", "yaml", "requests", "akshare")

ENV = pathlib.Path(__file__).resolve().parent / "local.env"


def _port(cfg: dict) -> int:
    """SMTP 端口。local.env 里 `SMTP_PORT=`（键在、值是空串）是常见的手写残留，
    而 dict.get 的默认值只在**键不存在**时生效，int("") 直接 ValueError：
    体检在这里整个炸掉，用户看到的是一段 traceback 而不是「SMTP 没配」。
    """
    return int((cfg.get("SMTP_PORT") or "").strip() or 587)


def check_smtp() -> None:
    """真连一次 SMTP 并登录，不发信。

    只检查「几个键存不存在」是不够的：密码填错、应用专用密码被吊销、
    Gmail 改了策略，这些都要等到真跑一次才暴露，而那时候已经错过时点了。
    这里连上去 login 一下就断，代价几百毫秒。

    密码永远不打印，失败也只报异常类型和 SMTP 的返回码。
    """
    if not ENV.exists():
        print("      MISS local.env not configured (local run will not mail)")
        return
    cfg = {}
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    need = ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "MAIL_TO")
    miss = [k for k in need if not cfg.get(k)]
    if miss:
        print("      MISS local.env missing: " + " ".join(miss))
        return
    if "FILLME" in cfg["SMTP_PASS"]:
        print("      MISS local.env SMTP_PASS still a placeholder")
        return
    import smtplib
    import ssl
    t0 = time.time()
    try:
        s = smtplib.SMTP(cfg["SMTP_HOST"], _port(cfg), timeout=15)
        s.starttls(context=ssl.create_default_context())
        s.login(cfg["SMTP_USER"], cfg["SMTP_PASS"].replace(" ", ""))
        s.quit()
        print("      OK   SMTP login    %.2fs  -> %s" % (time.time() - t0, cfg["MAIL_TO"]))
    except smtplib.SMTPAuthenticationError as e:
        print("      FAIL SMTP auth rejected (code %s) - app password wrong or revoked"
              % getattr(e, "smtp_code", "?"))
    except Exception as e:  # noqa: BLE001
        print("      FAIL SMTP %s" % type(e).__name__)


def main() -> int:
    for name, url in SOURCES:
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            urllib.request.urlopen(req).read()
            print("      OK   %-15s %.2fs" % (name, time.time() - t0))
        except Exception as e:  # noqa: BLE001
            print("      FAIL %-15s %s" % (name, type(e).__name__))

    miss = []
    for m in DEPS:
        try:
            importlib.import_module(m)
        except Exception:  # noqa: BLE001
            miss.append(m)
    if miss:
        print("      MISS Python deps: " + " ".join(miss))
        print("      MISS fix: pip install " + " ".join(miss))
    else:
        print("      OK   Python deps complete")

    check_smtp()
    # claude CLI 那一项 2026-09-27 随早盘系统归档删掉：两条晚间线都不用 LLM
    return 0


if __name__ == "__main__":
    sys.exit(main())
