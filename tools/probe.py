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

SOURCES = [
    ("Tencent quote ", "https://qt.gtimg.cn/q=sh600000"),
    ("Tencent kline ", "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                       "?param=sh600000,day,,,10,"),
    ("Sina calendar ", "https://finance.sina.com.cn/realstock/company/"
                       "sh000001/hisdata/klc_kl.js"),
    # 东财是可选源：日线优先走它（成交额是真值），不通就熔断走腾讯。
    # 所以这一项 FAIL 不影响出榜，只是慢一点、成交额变成估算。
    ("EastMoney(opt)", "https://push2his.eastmoney.com/api/qt/stock/kline/get"
                       "?secid=1.600000&fields1=f1&fields2=f51&klt=101&fqt=0"
                       "&end=20500101&lmt=5"),
]

DEPS = ("pandas", "pyarrow", "yaml", "requests", "akshare")

ENV = pathlib.Path(__file__).resolve().parent / "local.env"


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
        s = smtplib.SMTP(cfg["SMTP_HOST"], int(cfg.get("SMTP_PORT", "587")), timeout=15)
        s.starttls(context=ssl.create_default_context())
        s.login(cfg["SMTP_USER"], cfg["SMTP_PASS"].replace(" ", ""))
        s.quit()
        print("      OK   SMTP login    %.2fs  -> %s" % (time.time() - t0, cfg["MAIL_TO"]))
    except smtplib.SMTPAuthenticationError as e:
        print("      FAIL SMTP auth rejected (code %s) - app password wrong or revoked"
              % getattr(e, "smtp_code", "?"))
    except Exception as e:  # noqa: BLE001
        print("      FAIL SMTP %s" % type(e).__name__)


def check_claude() -> None:
    """本地 LLM 归因走的 claude CLI 通不通。真调一次最小请求（haiku）。

    这条 FAIL 不影响发信：竞价线拿不到文案会按「本次无 LLM 分析」照发，
    学习线归因失败那天日权重按 1.0 算。它影响的是本地跑有没有 LLM 文案。
    """
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                               / "src"))
        from learn import llm_local
        ok, msg = llm_local.auth_status()
        print(("      OK   claude auth   " if ok else "      FAIL claude auth  ")
              + msg)
    except Exception as e:  # noqa: BLE001
        print("      FAIL claude auth  " + type(e).__name__)


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
    check_claude()
    return 0


if __name__ == "__main__":
    sys.exit(main())
