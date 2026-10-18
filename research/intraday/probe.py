"""步骤 1：实测三条分时取数路径（baostock 5 分钟 / pytdx 历史分时 + 分笔 / 通达信 g4tic 包 HEAD）。

    research/datasrc/.venv/Scripts/python.exe research/intraday/probe.py

测试代码从 cache/codes.csv 取（教训 2）。每个请求带超时；北京 15:55~18:30 不发请求。
结果写 research/intraday/probe_result.json，样本 CSV 写 raw/probe_*.csv。
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bjtime  # noqa: E402

RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)
OUT = {}


def say(*a):
    print(*a, flush=True)


def blackout():
    t = bjtime.now_bj().time()
    if dt.time(15, 55) <= t <= dt.time(18, 30):
        raise SystemExit("北京 %s 在禁区内，停" % t)


blackout()
CODES = pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"].str.zfill(6)
rng = np.random.default_rng(1)
TEST = {pre: list(rng.choice(CODES[CODES.str.startswith(pre)], 2)) for pre in ("00", "60", "30", "68", "92")}
say("测试代码", TEST)
DATES = ["2022-03-08", "2024-03-05", "2025-03-05"]
daily = pd.read_parquet(ROOT / "data" / "breakout" / "daily.parquet",
                        filters=[("code", "in", sum(TEST.values(), [])), ("date", "in", DATES)])
daily["code"] = daily["code"].astype(str).str.zfill(6)
daily = daily.set_index(["code", "date"])
say("日线对照行", len(daily))


def dref(code, date):
    try:
        return daily.loc[(code, date)]
    except KeyError:
        return None


# ---------------- baostock ----------------
def probe_bs():
    import baostock as bs
    r = {}
    t0 = time.time()
    lg = bs.login()
    r["login"] = lg.error_msg
    r["login_sec"] = round(time.time() - t0, 2)
    rows_all = []
    for pre, codes in TEST.items():
        code = codes[0]
        bsc = ("sh." if pre in ("60", "68") else "sz." if pre in ("00", "30") else "bj.") + code
        for date in DATES:
            t0 = time.time()
            rs = bs.query_history_k_data_plus(bsc, "date,time,code,open,high,low,close,volume,amount,adjustflag",
                                              start_date=date, end_date=date, frequency="5", adjustflag="3")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            sec = time.time() - t0
            df = pd.DataFrame(rows, columns=rs.fields)
            ref = dref(code, date)
            info = {"code": bsc, "date": date, "err": rs.error_msg, "rows": len(df), "sec": round(sec, 2)}
            if len(df):
                v = df["volume"].astype(float).sum()
                a = df["amount"].astype(float).sum()
                info.update(first_time=df["time"].iloc[0], last_time=df["time"].iloc[-1],
                            vol_sum=v, amt_sum=a, close_last=float(df["close"].iloc[-1]),
                            high_max=float(df["high"].astype(float).max()))
                if ref is not None:
                    info.update(ref_vol=float(ref["volume"]), ref_amt=float(ref["amount"]), ref_close=float(ref["close"]),
                                vol_ratio=v / float(ref["volume"]) if ref["volume"] else None,
                                amt_ratio=a / float(ref["amount"]) if ref["amount"] else None)
                df.insert(0, "_date", date)
                rows_all.append(df)
            say("bs", info)
            r.setdefault("tests", []).append(info)
    # 一个月一只票的吞吐
    code = TEST["30"][1]
    t0 = time.time()
    rs = bs.query_history_k_data_plus("sz." + code, "date,time,open,high,low,close,volume,amount",
                                      start_date="2024-06-03", end_date="2024-06-28", frequency="5", adjustflag="3")
    n = 0
    while rs.error_code == "0" and rs.next():
        n += 1
    r["month_rows"] = n
    r["month_sec"] = round(time.time() - t0, 2)
    say("bs 一只票一个月", n, "行", r["month_sec"], "秒")
    bs.logout()
    if rows_all:
        pd.concat(rows_all).to_csv(RAW / "probe_baostock_5min.csv", index=False)
    return r


# ---------------- pytdx ----------------
def tdx_market(code):
    if code.startswith(("6", "9")) and not code.startswith("92"):
        return 1
    if code.startswith(("92", "8", "4")):
        return 2
    return 0


def probe_tdx():
    from pytdx.hq import TdxHq_API
    from pytdx.config.hosts import hq_hosts
    r = {"hosts_tried": 0, "hosts_ok": []}
    seen = set()
    for name, ip, port in hq_hosts:
        if (ip, port) in seen:
            continue
        seen.add((ip, port))
        r["hosts_tried"] += 1
        t0 = time.time()
        try:
            s = socket.create_connection((ip, port), timeout=2.5)
            s.close()
            r["hosts_ok"].append({"name": name, "ip": ip, "port": port, "sec": round(time.time() - t0, 2)})
        except OSError:
            pass
        if len(r["hosts_ok"]) >= 6:
            break
    say("tdx 能连的主机", r["hosts_ok"])
    if not r["hosts_ok"]:
        return r
    api = TdxHq_API(heartbeat=False, auto_retry=False, raise_exception=False)
    ok_host = None
    for h in r["hosts_ok"]:
        if api.connect(h["ip"], h["port"], time_out=5):
            ok_host = h
            break
    r["host_used"] = ok_host
    if ok_host is None:
        return r
    rows_min, rows_tx = [], []
    for pre, codes in TEST.items():
        code = codes[0]
        mk = tdx_market(code)
        for date in DATES:
            di = int(date.replace("-", ""))
            ref = dref(code, date)
            t0 = time.time()
            try:
                m = api.get_history_minute_time_data(mk, code, di)
            except Exception as e:  # noqa: BLE001
                m = None
                say("tdx 分时异常", code, date, repr(e))
            sec = time.time() - t0
            info = {"code": code, "market": mk, "date": date, "kind": "minute", "rows": None if m is None else len(m), "sec": round(sec, 2)}
            if m:
                df = pd.DataFrame(m)
                v = float(df["vol"].sum())
                info.update(vol_sum=v, last_price=float(df["price"].iloc[-1]), max_price=float(df["price"].max()),
                            first_price=float(df["price"].iloc[0]))
                if ref is not None:
                    info.update(ref_vol=float(ref["volume"]), ref_close=float(ref["close"]), ref_high=float(ref["high"]),
                                vol_ratio=v / float(ref["volume"]) if ref["volume"] else None)
                df.insert(0, "code", code)
                df.insert(1, "date", date)
                df["i"] = np.arange(len(df))
                rows_min.append(df)
            say("tdx", info)
            r.setdefault("tests", []).append(info)
            # 分笔：翻页到空
            t0 = time.time()
            ticks, start, pages = [], 0, 0
            while pages < 40:
                try:
                    tk = api.get_history_transaction_data(mk, code, start, 2000, di)
                except Exception as e:  # noqa: BLE001
                    say("tdx 分笔异常", code, date, repr(e))
                    tk = None
                pages += 1
                if not tk:
                    break
                ticks.extend(tk)
                if len(tk) < 2000:
                    break
                start += len(tk)
            sec = time.time() - t0
            info = {"code": code, "market": mk, "date": date, "kind": "tx", "rows": len(ticks), "pages": pages, "sec": round(sec, 2)}
            if ticks:
                df = pd.DataFrame(ticks)
                v = float(df["vol"].sum())
                info.update(vol_sum=v, first_time=df["time"].iloc[0], last_time=df["time"].iloc[-1],
                            buyorsell_values=sorted(df["buyorsell"].unique().tolist()),
                            buy_share=float(df.loc[df["buyorsell"] == 0, "vol"].sum() / max(v, 1)))
                if ref is not None:
                    info.update(ref_vol=float(ref["volume"]), vol_ratio=v / float(ref["volume"]) if ref["volume"] else None)
                df.insert(0, "code", code)
                df.insert(1, "date", date)
                rows_tx.append(df)
            say("tdx", info)
            r["tests"].append(info)
    api.disconnect()
    if rows_min:
        pd.concat(rows_min).to_csv(RAW / "probe_tdx_minute.csv", index=False)
    if rows_tx:
        pd.concat(rows_tx).to_csv(RAW / "probe_tdx_tx.csv", index=False)
    return r


# ---------------- g4tic HEAD ----------------
def probe_g4tic():
    r = {}
    for d in ("20240102", "20250305", "20231229"):
        url = f"https://www.tdx.com.cn/products/data/data/g4tic/{d}.zip"
        try:
            h = requests.head(url, timeout=20, allow_redirects=True)
            r[d] = {"status": h.status_code, "len_mb": round(int(h.headers.get("Content-Length", 0)) / 1e6, 1)}
        except Exception as e:  # noqa: BLE001
            r[d] = {"err": repr(e)}
        say("g4tic", d, r[d])
    return r


if __name__ == "__main__":
    which = sys.argv[1:] or ["bs", "tdx", "g4tic"]
    f = HERE / "probe_result.json"
    if f.exists():
        OUT = json.loads(f.read_text(encoding="utf-8"))
    for w in which:
        blackout()
        try:
            OUT[w] = {"bs": probe_bs, "tdx": probe_tdx, "g4tic": probe_g4tic}[w]()
        except Exception as e:  # noqa: BLE001
            OUT[w] = {"fatal": repr(e)}
            say("!!", w, repr(e))
        OUT[w]["at_bj"] = bjtime.now_bj().strftime("%Y-%m-%d %H:%M")
        f.write_text(json.dumps(OUT, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say("done ->", f)
