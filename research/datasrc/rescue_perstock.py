"""抢救：逐只拉「只给近期」的东财数据，全市场一遍，可续跑。

  python research/datasrc/rescue_perstock.py fflow     个股资金流（push2his fflow/daykline，近 120 个交易日）
  python research/datasrc/rescue_perstock.py hotrank   人气榜个股历史（emappdata getHisList + getHisProfileList，近一年）
  python research/datasrc/rescue_perstock.py merge     把分片合并成 archive/<种类>.parquet

东财全机 2 路并发（common.EMGate，两个进程一起跑也只占 2 路），请求间 sleep 0.4~0.9 秒，
连续失败 5 次全局停 10 分钟（教训 20），北京 16:00~18:30 自动停（common.check_blackout）。
每 100 只落一个分片 archive/<种类>/part_*.parquet；进度在 rescue_<种类>.progress.json，
重跑跳过已完成的代码（空结果也算完成，记 0）。

字段原样保留：
  fflow  f51 日期 f52 主力净流入 f53 小单净流入 f54 中单净流入 f55 大单净流入 f56 超大单净流入（元）
         f57~f61 对应净占比（%） f62 收盘价 f63 涨跌幅（%） f64 f65 实测恒为 0（意义未核实）
  hotrank    getHisList：date（calcTime）, rank
  hotprofile getHisProfileList：profile_time（calcTime 带时刻）, newUidRate, oldUidRate, *ChangeRank, uidCount
  python research/datasrc/rescue_perstock.py hotprofile  第二遍（时间够再跑）
"""
import sys
import time

import pandas as pd
import requests

import common

KIND = sys.argv[1] if len(sys.argv) > 1 else "fflow"
OUT = common.ARCHIVE / KIND
log = common.logger(common.ROOT / f"rescue_{KIND}.log")
FF = ",".join(f"f{i}" for i in range(51, 66))


def get_fflow(s, code):
    m = 1 if common.market(code) == "SH" else 0
    r = s.get("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get", params={
        "lmt": "0", "klt": "101", "secid": f"{m}.{code}", "fields1": "f1,f2,f3,f7",
        "fields2": FF, "ut": "b2884a393a59ad64002292a3e90d46a5"}, timeout=20)
    r.raise_for_status()
    d = r.json().get("data")
    if not d or not d.get("klines"):
        return pd.DataFrame()
    df = pd.DataFrame([k.split(",") for k in d["klines"]], columns=FF.split(","))
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "code", code)
    return df.rename(columns={"f51": "date"})


def hot_symbol(code):
    """人气榜的市场前缀：北交所 920 段要写 SZ（2026-10-15 实测 BJ920001 0 行、SZ920001 366 行）。"""
    m = common.market(code)
    return ("SZ" if m == "BJ" else m) + code


def _hot(s, code, ep):
    body = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38", "marketType": "",
            "srcSecurityCode": hot_symbol(code), "yearType": "5"}
    r = s.post("https://emappdata.eastmoney.com/stockrank/" + ep, json=body, timeout=20)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("data") or [])


def get_hotrank(s, code):
    """人气排名（calcTime 日期, rank）。emappdata 一次约 2 秒，粉丝结构另一遍（hotprofile）。"""
    df = _hot(s, code, "getHisList")
    if df.empty:
        return df
    df = df.rename(columns={"calcTime": "date"})
    df.insert(0, "code", code)
    return df


def get_hotprofile(s, code):
    """人气榜粉丝结构（calcTime 带时刻, newUidRate / oldUidRate / *ChangeRank / uidCount）。"""
    df = _hot(s, code, "getHisProfileList")
    if df.empty:
        return df
    df = df.rename(columns={"calcTime": "profile_time"}).drop(columns=["srcSecurityCode"], errors="ignore")
    df["date"] = df["profile_time"].str[:10]
    df.insert(0, "code", code)
    return df


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    fn = {"fflow": get_fflow, "hotrank": get_hotrank, "hotprofile": get_hotprofile}[KIND]
    prog = common.Progress(common.ROOT / f"rescue_{KIND}.progress.json")
    todo = [c for c in common.codes() if c not in prog.d["done"]]
    log(f"{KIND}: 待拉 {len(todo)} 只，已完成 {len(prog.d['done'])}")
    s = requests.Session()
    s.headers["User-Agent"] = common.UA
    # push2his 的 fflow 路径 2026-10-15 04:28 BJ 实测连 8 次就被掐（同主机 kline 路径照常），
    # 所以资金流走慢档：请求间 2~4 秒，连续 3 次失败停 15 分钟，最多停 20 次（约 5 小时）再放弃。
    br = common.Breaker(n=3, pause=900, max_trips=20, log=log) if KIND == "fflow" else common.Breaker(log=log)
    gap = (2.0, 4.0) if KIND == "fflow" else (0.4, 0.9)
    buf, t0 = [], time.time()
    part = len(list(OUT.glob("part_*.parquet")))

    def flush():
        nonlocal buf, part
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(OUT / f"part_{part:04d}.parquet", index=False)
            part += 1
            buf = []
        prog.save()

    try:
        for i, code in enumerate(todo, 1):
            try:
                with common.EMGate(gap):
                    df = fn(s, code)
                br.ok()
            except common.Blackout:
                raise
            except Exception as e:  # noqa: BLE001
                prog.d["fail"][code] = repr(e)[:150]
                if not br.bad(f"{code} {e!r}"[:200]):
                    break
                continue
            prog.d["fail"].pop(code, None)
            prog.d["done"][code] = len(df)
            if len(df):
                buf.append(df)
            if i % 100 == 0:
                flush()
                log(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/只  失败 {len(prog.d['fail'])}")
    except common.Blackout as e:
        log(str(e))
    finally:
        flush()
    log(f"{KIND} 结束：完成 {len(prog.d['done'])}，非空 {sum(1 for v in prog.d['done'].values() if v)}，"
        f"失败 {len(prog.d['fail'])}")


def merge():
    for kind in ("fflow", "hotrank", "hotprofile"):
        parts = sorted((common.ARCHIVE / kind).glob("part_*.parquet"))
        if not parts:
            continue
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if kind == "hotrank":  # part_0000 是第一版（排名 + 粉丝合在一起）留下的，粉丝列以 hotprofile 为准
            df = df[["code", "date", "rank"]]
        df = df.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])
        df.to_parquet(common.ARCHIVE / f"{kind}.parquet", index=False)
        print(kind, len(df), "行", df["code"].nunique(), "只", df["date"].min(), "~", df["date"].max())


if __name__ == "__main__":
    merge() if KIND == "merge" else run()
