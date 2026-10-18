"""a12：BaoStock 逐日 isST（沪深全部代码，2020-07-01..2026-10-16），写 raw/bs_isst.parquet。研究 venv 跑（baostock 装在那）。
    单连接顺序查，每只一次请求；每 200 只落一次盘可续跑。北交所 BaoStock 不支持。
    用途：量静态 ST 名单（cache/st_codes.json，2026-09-16）对历史回测的误差。"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402
import baostock as bs  # noqa: E402

LOG = open(OUT / "a12.log", "a", encoding="utf-8")


def say(*a):
    s = "[%s] %s" % (bjtime.now_bj().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def guard():
    t = bjtime.now_bj()
    if 16 * 60 <= t.hour * 60 + t.minute < 18 * 60 + 30:
        say("北京 16:00~18:30，停")
        sys.exit(2)


def main():
    codes = pd.read_csv(ROOT / "cache/codes.csv", dtype=str)["code"].str.zfill(6)
    codes = [c for c in codes if c[0] in "0369" and not c.startswith("9")]
    extra = sys.argv[1:]          # 另给的代码（退市票）
    codes = extra or codes
    part = RAW / ("bs_isst_delist.parquet" if extra else "bs_isst.parquet")
    done = set()
    frames = []
    if part.exists():
        old = pd.read_parquet(part)
        frames.append(old)
        done = set(old["code"].unique())
        say("续跑：已有", len(done), "只")
    lg = bs.login()
    say("login", lg.error_code, lg.error_msg)
    buf, fails = [], []
    todo = [c for c in codes if c not in done]
    say("待查", len(todo), "只")
    for i, c in enumerate(todo):
        if i % 100 == 0:
            guard()
        sym = ("sh." if c[0] == "6" else "sz.") + c
        rows = None
        for attempt in range(3):
            try:
                rs = bs.query_history_k_data_plus(sym, "date,code,close,isST,tradestatus", start_date="2020-07-01",
                                                  end_date="2026-10-16", frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    raise RuntimeError(rs.error_msg)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                break
            except Exception as e:
                say("查", c, "失败", repr(e)[:80])
                time.sleep(3 * (attempt + 1))
                try:
                    bs.logout()
                except Exception:
                    pass
                bs.login()
        if rows is None:
            fails.append(c)
            continue
        if rows:
            x = pd.DataFrame(rows, columns=["date", "sym", "close", "isST", "tradestatus"])
            x["code"] = c
            buf.append(x[["code", "date", "close", "isST", "tradestatus"]])
        else:
            buf.append(pd.DataFrame({"code": [c], "date": [None], "close": [None], "isST": [None], "tradestatus": [None]}))
        if (i + 1) % 200 == 0 or i + 1 == len(todo):
            frames.append(pd.concat(buf, ignore_index=True))
            buf = []
            pd.concat(frames, ignore_index=True).to_parquet(part, index=False)
            say("进度", i + 1, "/", len(todo), "失败", len(fails))
    bs.logout()
    say("完成，失败", fails)


if __name__ == "__main__":
    main()
