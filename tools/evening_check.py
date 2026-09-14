"""
晚间托底检查。只在 GitHub Actions 里跑（.github/workflows/evening_check.yml）。

    python tools/evening_check.py            检查 + 该发提醒就发
    python tools/evening_check.py --dry      只检查不发信

起涨预测（晚间系统）云端**算不了**：特征表 2.5GB 在本机，新浪日线源在
GitHub runner 上也不通。所以云端在这条线上的托底不是「替本地跑」，
而是「本地没跑就告诉用户」：

    目标日 = 最近一个已收盘的交易日
    origin/main 上 out_breakout/run_meta.json 的 date == 目标日  -> 本地跑过，退出
    否则                                                      -> 发一封提醒

提醒里写清楚本机开机后会自动补跑（窗口到次日 08:30 北京），以及手动入口。
同一天只提醒一次：发过就把 state/alert/breakout_<date>.json 提交到仓库，
下一个入口（GitHub cron 和 Cloudflare Worker 各触发一次）看到标记就跳过。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def now_bj() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


def last_closed_trade_day(tds: set[str], now: dt.datetime) -> str:
    d = now.date()
    closed = (now.hour, now.minute) >= (15, 5)
    for back in range(0, 15):
        day = d - dt.timedelta(days=back)
        ok = (day.isoformat() in tds) if tds else day.weekday() < 5
        if ok and (back > 0 or closed):
            return day.isoformat()
    return d.isoformat()


def _origin_json(path: str) -> dict:
    subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT,
                   capture_output=True, timeout=60)
    r = subprocess.run(["git", "show", f"origin/main:{path}"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=30)
    if r.returncode != 0:
        return {}
    try:
        return json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {}


def _origin_has(path: str) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"origin/main:{path}"],
                       cwd=ROOT, capture_output=True, timeout=30)
    return r.returncode == 0


def build_mail(date: str, meta: dict) -> tuple[str, str]:
    last = meta.get("date") or "无"
    subject = f"[起涨预测] {date} 本机没跑，清单未出"
    body = (
        f"目标交易日 {date} 的起涨预测清单还没出。\n"
        f"仓库里最新的一份是 {last}。\n\n"
        "原因：这条线只能在本机跑（特征表 2.5GB 在本机，云端拉不到日线），"
        "今天本机在北京 16:00 到现在没有跑完它，多半是没开机。\n\n"
        "接下来会怎样：\n"
        "  1. 本机开机后计划任务 DailyReport-Local-Evening 会自动补跑，"
        "最晚到北京次日 08:30（美东 20:30）。补出来的清单和 17:00 跑的一样，"
        "赶在下一个交易日开盘前。\n"
        "  2. 想立刻要：打开桌面「A股流水线」控制台，点「起涨预测」。\n\n"
        f"发自云端托底检查 {now_bj().strftime('%Y-%m-%d %H:%M')} 北京时间。"
    )
    return subject, body


def send(subject: str, body: str) -> None:
    from mailer import _conf, _send
    c = _conf()
    m = EmailMessage()
    m["Subject"] = subject
    m["From"] = c["user"]
    m["To"] = ", ".join(c["to"])
    m.set_content(body)
    _send(m, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    try:
        import datasource as ds
        tds = set(ds.trade_dates())
    except Exception as e:  # noqa: BLE001
        print(f"交易日历拿不到（{e}），按周一到周五算")
        tds = set()
    now = now_bj()
    today = now.strftime("%Y-%m-%d")
    if (today not in tds) if tds else now.weekday() >= 5:
        print(f"{today} 不是交易日，不检查")
        return 0
    date = last_closed_trade_day(tds, now)
    if date != today:
        print(f"现在 {now.strftime('%H:%M')} 还没收盘，目标日 {date} 不是今天，不检查")
        return 0

    meta = _origin_json("out_breakout/run_meta.json")
    if meta.get("date") == date:
        print(f"本地已出 {date} 的清单（A {meta.get('n_a')} 只，B {meta.get('n_b')} 只），不用提醒")
        return 0
    marker = f"state/alert/breakout_{date}.json"
    if _origin_has(marker):
        print(f"{date} 已经提醒过（{marker}），不重复")
        return 0

    subject, body = build_mail(date, meta)
    print(subject)
    print(body)
    if a.dry:
        print("[dry] 不发信")
        return 0
    send(subject, body)
    p = ROOT / marker
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": date, "at": now.isoformat(timespec="seconds"),
                             "last": meta.get("date", "")}, ensure_ascii=False),
                 encoding="utf-8")
    print(f"提醒已发，标记写到 {marker}（由 workflow 提交）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
