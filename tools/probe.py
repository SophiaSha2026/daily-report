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
    return 0


if __name__ == "__main__":
    sys.exit(main())
