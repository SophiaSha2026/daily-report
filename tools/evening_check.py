"""
晚间托底检查。只在 GitHub Actions 里跑（.github/workflows/evening_check.yml）。

    python tools/evening_check.py            检查 + 该发提醒就发
    python tools/evening_check.py --dry      只检查不发信

晚间两条线（起涨预测、长期调整突破）云端都**算不了**：日线表和特征表在本机，
新浪日线源在 GitHub runner 上也不通。所以云端的托底不是「替本地跑」，而是
「本地没跑就告诉用户」：

    目标日 = 最近一个已收盘的交易日
    每条线：origin/main 上有 state/sent/<线>_<目标日>.json -> 邮件真发出去了
    缺了哪条线，就在**一封**提醒里写哪条（2026-09-27 起两条线合并成一封，
    以前只查起涨预测）

判据只认 sent 标记，不认 run_meta 的日期：run_meta 是扫描阶段写的，
local_run 的 push_all 不看退出码就把它推上 origin，于是「清单算出来了、
send 阶段 SMTP 挂了」那一天，run_meta.date 照样等于目标日，云端会打印
「本地已出」退出，用户一封信都收不到而三层保护全报正常（2026-09-16 审计）。
run_meta 仍然读，但只用来在正文里说「仓库里最新一份是哪天」和区分两种原因。

提醒里写清楚本机开机后会自动补跑（窗口到次日 08:30 北京），以及手动入口。
同一天同一条线只提醒一次：发过就把 state/alert/<线>_<date>.json 提交到仓库，
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

# 要检查的线：名字、run_meta、计划任务、日志、控制台按钮
LINES = {
    "breakout": {"name": "起涨预测", "meta": "out_breakout/run_meta.json",
                 "task": "DailyReport-Local-Evening",
                 "log": "tools/local_flow_breakout.log",
                 "why_local": "特征表 2.5GB 在本机，云端拉不到日线"},
    "pullback": {"name": "长期调整突破", "meta": "out_pullback/run_meta.json",
                 "task": "DailyReport-Local-Pullback",
                 "log": "tools/local_flow_pullback.log",
                 "why_local": "三年全市场日线在本机，云端拉不到"},
}


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


SENT_FAILED = "sent_failed"     # 清单算出来了、邮件没发出去
NOT_RUN = "not_run"             # 本机根本没跑


def decide(meta: dict, has_sent: bool, date: str) -> tuple[bool, str]:
    """返回 (要不要提醒, 原因)。sent 标记是唯一的「发了」证据。

    run_meta.date == 目标日只说明扫描跑过（它 dry 跑也写，所以还要看 dry），
    不说明邮件发出去了。教训 27：退出码 0 / 产物存在不等于做了事。
    """
    if has_sent:
        return False, ""
    if meta.get("date") == date and not meta.get("dry"):
        return True, SENT_FAILED
    return True, NOT_RUN


def _cause(key: str, date: str, meta: dict, why: str) -> str:
    ln = LINES[key]
    last = meta.get("date") or "无"
    head = f"【{ln['name']}】仓库里最新的一份是 {last}。\n"
    if why == SENT_FAILED:
        return head + (f"原因：本机把 {date} 的清单算出来了（{ln['meta']} 已经推到仓库），"
                       "但发信那一步没成功，多半是 SMTP 超时/认证失败或者 tools/local.env 缺项。\n"
                       f"怎么办：本机看 {ln['log']} 末尾那段报错；想立刻要就打开桌面"
                       f"「A股流水线」控制台，点「{ln['name']}」重发（只走发信那一步）。\n")
    return head + (f"原因：这条线只能在本机跑（{ln['why_local']}），今天本机在北京 16:00 "
                   "到现在没有跑完它，多半是没开机。\n"
                   f"接下来：本机开机后计划任务 {ln['task']} 会自动补跑，最晚到北京次日 08:30，"
                   "补出来的清单和按时跑的一样，赶在下一个交易日开盘前。"
                   f"想立刻要：打开桌面「A股流水线」控制台，点「{ln['name']}」。\n")


def build_mail(date: str, meta: dict, why: str = NOT_RUN,
               key: str = "breakout") -> tuple[str, str]:
    """一条线的提醒。两条线都缺时 main 用 build_mail_many 合成一封。"""
    return build_mail_many(date, [(key, meta, why)])


def build_mail_many(date: str, items: list[tuple[str, dict, str]]) -> tuple[str, str]:
    names = "、".join(LINES[k]["name"] for k, _, _ in items)
    whys = {w for _, _, w in items}
    if whys == {SENT_FAILED}:
        subject = f"[晚间提醒] {date} {names} 清单算了，但邮件没发出去"
    elif whys == {NOT_RUN}:
        subject = f"[晚间提醒] {date} {names} 本机没跑，清单未出"
    else:
        subject = f"[晚间提醒] {date} {names} 没发出去 / 本机没跑"
    body = (f"目标交易日 {date} 的 {names} 还没发到你邮箱。\n\n"
            + "\n".join(_cause(k, date, m, w) for k, m, w in items)
            + f"\n发自云端托底检查 {now_bj().strftime('%Y-%m-%d %H:%M')} 北京时间。")
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

    items = []
    for key, ln in LINES.items():
        # _origin_json 负责 fetch，_origin_has 自己不 fetch，顺序不能反
        meta = _origin_json(ln["meta"])
        need, why = decide(meta, _origin_has(f"state/sent/{key}_{date}.json"), date)
        if not need:
            print(f"{ln['name']}：本地已发 {date} 的清单邮件（state/sent 标记在），不用提醒")
            continue
        marker = f"state/alert/{key}_{date}.json"
        if _origin_has(marker):
            print(f"{ln['name']}：{date} 已经提醒过（{marker}），不重复")
            continue
        items.append((key, meta, why))
    if not items:
        return 0

    subject, body = build_mail_many(date, items)
    print(subject)
    print(body)
    if a.dry:
        print("[dry] 不发信")
        return 0
    send(subject, body)
    for key, meta, why in items:
        p = ROOT / f"state/alert/{key}_{date}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"date": date, "at": now.isoformat(timespec="seconds"),
                                 "last": meta.get("date", ""), "why": why},
                                ensure_ascii=False), encoding="utf-8")
        print(f"提醒已发，标记写到 {p.relative_to(ROOT)}（由 workflow 提交）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
