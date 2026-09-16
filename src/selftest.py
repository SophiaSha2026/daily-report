"""
离线端到端测试：用合成行情跑通 特征->打分->排序->通达信导出->HTML 渲染。
不联网、不发邮件。目的：证明流程不会卡住，且边界条件都被正确拒绝。

    python src/selftest.py
"""
from __future__ import annotations

import sys
import time
import random
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from score import AuctionFeature, score_one, rank, f_gap, f_volume
from tdx_export import write_tdx_custom
from mailer import build_html

ROOT = Path(__file__).resolve().parent.parent


def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def mk(**kw) -> AuctionFeature:
    """构造一只票，未指定字段用中性默认值。"""
    d = dict(
        code="600000", name="测试股", limit_pct=10.0, prev_close=10.0,
        auc_price=10.3, gap_pct=3.0, gap_norm=0.30,
        auc_amount=2.0e7, prev_amount=1.0e9, auc_ratio=0.02,
        t1_chg=2.0, t2_chg=2.6, t3_chg=3.0, slope=1.0, monotonic=True, dive=-0.4,
        pos_pct_60d=0.6, ma_bull=True, breakout=True,
        prev_limit_up=True, prev_broken_board=False, board_height=1,
        sector="半导体", sector_members=4, sector_prev_limitups=3,
        blacklisted=False, one_word=False,
    )
    d.update(kw)
    return AuctionFeature(**d)


# --- 覆盖每一条硬性排除规则 + 正常样本 -------------------------------
CASES = [
    ("正常A组强势",   mk(code="600111", name="强势A"),                       None),
    # 绝对流动性下限：比例指标在小盘票上失真，300 万以下买不进去。
    # 2026-09-15 前这条只用来数板块成员，从不剔除，竞价额 127 万的票进过前 10
    ("竞价额不足",    mk(code="600112", name="小票", auc_amount=1.27e6,
                        prev_amount=8.0e7, auc_ratio=0.0159),     "竞价额"),
    ("正常B组低位",   mk(code="000222", name="低位B", prev_limit_up=False,
                        board_height=0, breakout=False, pos_pct_60d=0.25),  None),
    ("创业板20cm",    mk(code="300333", name="创业板", limit_pct=20.0,
                        gap_pct=4.2, gap_norm=0.21, t1_chg=2.6, t2_chg=3.5,
                        t3_chg=4.2, slope=1.6),                             None),
    # 涨幅上限是绝对 5%，不随 20cm 放宽：同样一只创业板高开 6.4% 要出局
    ("20cm高开超5%",  mk(code="300334", name="创业板高", limit_pct=20.0,
                        gap_pct=6.4, gap_norm=0.32, t1_chg=4.0, t2_chg=5.5,
                        t3_chg=6.4, slope=2.4),                   "竞价涨幅"),
    ("低开出局",      mk(code="600444", gap_pct=-1.2, gap_norm=-0.12,
                        t1_chg=-1.5, t2_chg=-1.3, t3_chg=-1.2),   "竞价涨幅"),
    # 2026-09-02 退回原版，+2% 动能下限重新变成硬性剔除（此前只扣分）
    ("动能不足1.5%",  mk(code="600445", gap_pct=1.5, gap_norm=0.15,
                        t1_chg=1.0, t2_chg=1.3, t3_chg=1.5),      "竞价涨幅"),
    ("刚好2%放行",    mk(code="600446", gap_pct=2.1, gap_norm=0.21,
                        t1_chg=1.6, t2_chg=1.9, t3_chg=2.1),               None),
    ("高开过头",      mk(code="600555", gap_pct=7.0, gap_norm=0.70,
                        t1_chg=6.0, t2_chg=6.5, t3_chg=7.0),      "竞价涨幅"),
    ("量能不足",      mk(code="600666", auc_ratio=0.004),        "竞价量能"),
    ("量能过载",      mk(code="600777", auc_ratio=0.30),         "竞价量能"),
    # 量比上限退回 10：量比 12 出局，量比 9 放行
    ("量比12超上限",  mk(code="600778", auc_ratio=0.05),         "竞价量能"),
    ("量比9放行",     mk(code="600779", auc_ratio=0.0375),                None),
    ("尾盘跳水",      mk(code="600888", t2_chg=5.4, t3_chg=3.0,
                        dive=2.4, monotonic=False),              "尾盘跳水"),
    ("假涨停撤单",    mk(code="600999", t1_chg=9.6, t2_chg=5.0, t3_chg=3.0,
                        dive=2.0, monotonic=False),              "假涨停撤单"),
    ("一字板",        mk(code="601000", one_word=True),          "一字板"),
    ("公告黑名单",    mk(code="601111", blacklisted=True),       "公告黑名单"),
    ("停牌零价",      mk(code="601222", auc_price=0.0, prev_close=0.0),
                                                                 "停牌或数据缺失"),
]


def check_curves(c: dict) -> int:
    """打分曲线的形状不变量。

    存在的理由：2026-08-24 改阈值时踩过一次——只是把 auc_ratio_max 从 8%
    放宽到 20.8%，f_volume 的衰减分母是 (hi - sat)，衰减速率跟着被摊平，
    量比 19 的得分从 0.61 悄悄变成 0.89。阈值和曲线形状必须解耦，
    这里用断言钉住，改配置时不会再顺带改掉打分口径。
    """
    s = c["screen"]
    lo, hi, pk = s["gap_pct_min"], s["gap_pct_max"], s["gap_pct_peak"]
    vlo, vhi = s["auc_ratio_min"], s["auc_ratio_max"]
    sat, dec = s["auc_ratio_score_hi"], s.get("auc_ratio_decay", 0.40)
    bad = 0

    def ck(cond: bool, msg: str) -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  {'✓' if cond else '✗'} {msg}")

    print("\n打分曲线不变量")
    ck(abs(f_gap(pk, lo, hi, pk) - 1.0) < 1e-9, "gap 峰值处得满分")
    ck(f_gap(lo, lo, hi, pk) == 0.0 and f_gap(hi, lo, hi, pk) == 0.0,
       "gap 两个边界都归零")
    # 边界连续：紧贴边界的取值必须接近 0，不能出现「区间内 0.57、边界 0」的断崖
    eps = (hi - lo) / 1000
    ck(f_gap(hi - eps, lo, hi, pk) < 0.02 and f_gap(lo + eps, lo, hi, pk) < 0.02,
       "gap 在两个边界处连续（无断崖）")
    ck(all(f_gap(g, lo, hi, pk) < f_gap(g + 0.05, lo, hi, pk)
           for g in [lo + 0.1 + i * 0.1 for i in range(int((pk - lo) * 10) - 2)]),
       "gap 在峰值左侧单调递增（涨幅越大动能越强）")

    ck(abs(f_volume(sat, vlo, vhi, sat, dec) - 1.0) < 1e-9, "volume 饱和点得满分")
    # 饱和点必须落在准入区间内部。这条同时挡住一类致命配置错误：
    # 把「竞价量能 >= 昨日全天 10%」当成 auc_ratio_min 打开（=0.10），
    # 它比 auc_ratio_max(0.0417) 还大，准入区间成空集，每天发空榜而且不报错。
    ck(vlo < sat < vhi, "量能区间自洽：min < 饱和点 < max（空集会静默发空榜）")
    # 在实际区间内几何取点，取点数不随上下限变化
    probes = [sat * (vhi / sat) ** (i / 8.0) for i in range(9)]
    ck(all(f_volume(probes[i], vlo, vhi, sat, dec)
           > f_volume(probes[i + 1], vlo, vhi, sat, dec) for i in range(8)),
       "volume 超过饱和点后单调递减（越极端越警惕）")
    # 关键：衰减速率不能随上限漂移。这两条都显式传 hi，是**纯形状**断言，
    # 与 config 当前的 auc_ratio_max 无关——否则收窄上限时断言会跟着一起失效。
    probe = sat * 2
    ck(abs(f_volume(probe, vlo, 0.08, sat, dec)
           - f_volume(probe, vlo, 0.30, sat, dec)) < 1e-9,
       "volume 衰减速率与 auc_ratio_max 无关")
    ck(abs(f_volume(0.0792, vlo, 0.30, sat, dec) - 0.61) < 0.02,
       "volume 每 e 倍于饱和点扣 0.40（量比 19 处 0.61，与旧配置口径一致）")
    return bad


def check_rules(c: dict) -> int:
    """非「硬性排除」类规则的行为断言。

    这些规则不会出现在 CASES 的 rejected 字段里，但同样会因为改阈值而
    静默失效——比如「高位极端放量」的触发线一度写成 auc_ratio_score_hi*2，
    量比上限收回 10 之后那个值落到准入区间之外，扣分永远不会发生。
    """
    bad = 0

    def ck(cond: bool, msg: str) -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(f"  {'✓' if cond else '✗'} {msg}")

    print("")
    print("规则不变量")
    hi_vol = score_one(mk(code="600801", pos_pct_60d=0.95, auc_ratio=0.0375), c)
    lo_vol = score_one(mk(code="600802", pos_pct_60d=0.95, auc_ratio=0.02), c)
    mid_pos = score_one(mk(code="600803", pos_pct_60d=0.60, auc_ratio=0.0375), c)
    ck("高位极端放量" in hi_vol["risk_tags"],
       "高位(0.95)+量比9 触发「高位极端放量」扣分")
    ck("高位极端放量" not in lo_vol["risk_tags"],
       "高位(0.95)+量比4.8 不触发")
    ck("高位极端放量" not in mid_pos["risk_tags"],
       "中位(0.60)+量比9 不触发")
    ck(hi_vol["penalty"] >= c["scoring"]["penalties"]["high_pos_extreme_volume"],
       "扣分额度按 config 生效")

    # M6：斜率缺失。以前 min(1.0, nan) 因参数顺序返回 1.0，没轨迹的票白拿
    # 趋势满分（权重 0.20 = 20 分）且不被剔除
    from score import f_trend, hard_reject
    ck(abs(f_trend(float("nan"), False, 10.0) - 0.5) < 1e-12,
       "斜率缺失得 0.5（中性），不再白给趋势满分")
    ck(abs(f_trend(float("nan"), True, 10.0) - 0.65) < 1e-12,
       "斜率缺失 + 单调 = 0.5 + 0.15，和 slope=0 同值")
    ck(f_trend(float("nan"), False, 10.0) == f_trend(0.0, False, 10.0),
       "缺失按 0 处理（和 build_features「T1 漏采、斜率 0」同口径）")

    # M8：min_auc_amount_wan 缺键必须响。.get(…, 0) 那个默认值恰好等于
    # 2026-09-15 之前「规则写在配置里没人执行」的状态（教训 26）
    small = mk(code="600113", auc_amount=1.27e6, prev_amount=8.0e7,
               auc_ratio=0.0159)
    got = hard_reject(small, c["screen"])
    ck(got is not None and "竞价额" in got,
       f"竞价额 127 万被这条下限真的剔除（拿到 {got}）")
    sc_nokey = dict(c["screen"])
    sc_nokey.pop("min_auc_amount_wan")
    try:
        r = hard_reject(small, sc_nokey)
        ck(False, f"缺 min_auc_amount_wan 时静默放行了（rejected={r}）")
    except KeyError:
        ck(True, "缺 min_auc_amount_wan -> KeyError，不静默放行（教训 26）")
    return bad


def check_misc(c: dict) -> int:
    """零散但都真出过事的规则。"""
    from datasource import limit_pct
    from score import f_volume, rank
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n[零散规则]")
    from datasource import limit_price
    # 北交所三个代码段都是 30%。920 段以前落到 10%，假涨停判据、斜率归一全错
    ck(limit_pct("920159", "农大科技") == 30.0, "北交所 920 段涨停幅度 30%")
    ck(limit_pct("832735", "x") == 30.0 and limit_pct("430139", "x") == 30.0,
       "北交所 8/4 老段 30%")
    ck(limit_pct("688655", "迅捷兴") == 20.0 and limit_pct("300017", "x") == 20.0,
       "科创/创业 20%")
    # ST 不再一律 5%（2026-09-16 实测 201 只 ST 的真实涨跌停）
    ck(limit_pct("000078", "ST海王") == 10.0
       and limit_pct("600000", "*ST 浦发") == 10.0,
       "主板 ST 10%（不再 5%；名字带空格也认）")
    ck(limit_pct("300068", "ST南都") == 20.0
       and limit_pct("688053", "ST思科瑞") == 20.0, "创业/科创 ST 20%")
    ck(limit_pct("920023", "*ST田野") == 30.0, "北交所 ST 30%")
    ck(limit_pct("600182", "S佳通") == 5.0, "未股改 S 股 5%（全市场只此一只）")
    ck(limit_price(12.78, limit_pct("600182", "S佳通")) == 13.42,
       "S佳通涨停价 = 腾讯 f[47]（12.78 -> 13.42）")
    ck(limit_price(1.50, limit_pct("000078", "ST海王")) == 1.65,
       "ST海王涨停价 = 腾讯 f[47]（1.50 -> 1.65）")
    # 昨日炸板的涨停价判据：−0.01 会把「最高价 = 涨停价 − 1 分」判成触及涨停
    ck(3.39 < limit_price(3.09, 10.0),
       "600654 昨收 3.09 最高 3.39 不算触及涨停（涨停价 3.40）")
    ck(3.40 >= limit_price(3.09, 10.0), "最高 3.40 才算触及")
    ck(22.63 < limit_price(20.58, 10.0),
       "600184 昨收 20.58 最高 22.63 不算（涨停价 22.64）")
    ck(11.0 >= limit_price(10.0, 10.0),
       "浮点 10.0*1.1=11.000000000000002 不漏判")
    # 北交所封板价**向下取整**到分：920 段 978 个封板日实测，
    # 「最高价×100 − 昨收×(100+pct)」的平台在 (-1, 0]，而沪主板在 [-0.4, +0.5]
    ck(limit_price(84.63, 30, "920002") == 110.01
       and limit_price(17.09, 30, "920006") == 22.21
       and limit_price(31.46, 30, "920083") == 40.89
       and limit_price(2.85, 30, "920090") == 3.70,
       "北交所涨停价向下取整（四只票的实测封板价）")
    ck(limit_price(0.70, 30, "920000") == 0.91
       and limit_price(43.00, 30, "920002") == 55.90,
       "北交所取整不被浮点误差掉档（0.70*1.3=0.9099999）")
    ck(limit_price(10.05, 10, "600000") == 11.06
       and limit_price(4.35, 10) == 4.79
       and limit_price(10.05, 20, "300001") == 12.06,
       "沪深仍四舍五入（.xx5 平局进位），不传 code 时行为不变")
    ck(abs(3.70 - limit_price(2.85, 30, "920090")) < 0.005,
       "北交所一字板判据（run_auction 的 |价−涨停价|<0.005）在封板价上成立")
    ck(abs(3.71 - limit_price(2.85, 30, "920090")) >= 0.005,
       "四舍五入那个价（3.71）不再被当成封板价")
    ra_src = (ROOT / "src" / "run_auction.py").read_text(encoding="utf-8")
    ck("limit_price(a3.prev_close, lp, row.code)" in ra_src,
       "run_auction 的一字板判据按代码取整（否则上面几条再对也没用）")
    # 抢救模式把量能下限放到 0：以前 log(ratio/0) 直接 ZeroDivisionError
    try:
        v = f_volume(0.02, 0.0, 1e9, 0.03)
        ck(v == 0.0, "量能下限为 0（维度停用）时 f_volume 返回 0 不抛异常")
    except ZeroDivisionError:
        ck(False, "量能下限为 0 时 f_volume 抛了 ZeroDivisionError")
    # 排序用未取整的分数，取整并列的票不再按候选池顺序定先后
    a = score_one(mk(code="600901", name="a"), c)
    b = score_one(mk(code="600902", name="b"), c)
    a["score"], b["score"] = 70.0, 70.0
    a["score_raw"], b["score_raw"] = 69.96, 70.04
    top = rank([a, b], c)["all"]
    ck(top[0]["code"] == "600902", "同分（取整后）按未取整分数排，b 在前")

    # M9：select() 不许再按取整后的 score 排一次。它现在是对的只因为
    # round 单调 + sorted 稳定，谁改成 numpy argsort 或 key=(-score, code)
    # 都会悄悄退回「按候选池顺序（昨日成交额）定先后」
    from run_auction import select
    a2 = score_one(mk(code="600903", name="c"), c)
    for r, raw in ((a, 69.96), (b, 70.04), (a2, 70.00)):
        r["score"], r["score_raw"] = 70.0, raw
        r["rejected"], r["group"] = None, "A"
    sel = select([a, a2, b], c)          # 输入顺序故意和 raw 顺序不同
    ck([r["code"] for r in sel] == ["600902", "600903", "600901"],
       f"select() 取整同分按 score_raw 排（b>c>a），拿到 "
       f"{[r['code'] for r in sel]}")
    ck([r["code"] for r in sel]
       == [r["code"] for r in rank([a, a2, b], c)["all"]][: c["output"]["top_n"]],
       "select() 就是 rank()['all'] 的前 top_n，没有第二次排序")

    # M14：过 hard_deadline 之后「放弃还是抢救」只有一条分界线。
    # 以前 local_run 自己按 09:27 整分判，run_auction 按 09:26:30 秒级判，
    # 中间 32 秒的缝里手点会发一封措辞错误的告警邮件，本该出抢救榜
    from run_auction import quick_mode
    tz = dt.timezone(dt.timedelta(hours=8))

    def t(h, m, s):
        return dt.datetime(2026, 9, 16, h, m, s, tzinfo=tz)

    hd = c["runtime"]["hard_deadline"]
    ck(quick_mode(t(9, 26, 30), hd, False, True) == "normal",
       f"恰好 {hd} 仍走正常采样")
    ck(quick_mode(t(9, 26, 31), hd, False, True) == "salvage",
       "本地（--salvage-if-late）：过线 1 秒即抢救，不告警")
    ck(quick_mode(t(9, 26, 45), hd, False, True) == "salvage",
       "09:26:30~09:27:00 不再是告警缝")
    ck(quick_mode(t(9, 26, 31), hd, False, False) == "abort",
       "云端（不带 auto）：过线照旧告警放弃（硬约束 3）")
    ck(quick_mode(t(9, 40, 0), hd, True, False) == "salvage",
       "显式 --late 永远抢救")
    ra_src2 = (ROOT / "src" / "run_auction.py").read_text(encoding="utf-8")
    ck("--salvage-if-late" in ra_src2 and "quick_mode(" in ra_src2,
       "stage_quick 真的走 quick_mode，main 真的有这个开关")

    # F5-14：SKIP_MAIL 只有一个判定。以前三条业务线用真值判断（"0" 也跳过），
    # 学习线用 == "1"，同一个变量两套语义，两边自测各钉各的
    import os as _os
    import mailer as M
    keep = _os.environ.get("SKIP_MAIL")
    try:
        for v, want in ((None, False), ("", False), ("0", False),
                        ("false", False), ("no", False), ("1", True),
                        ("true", True), ("TRUE", True), (" 1 ", True)):
            if v is None:
                _os.environ.pop("SKIP_MAIL", None)
            else:
                _os.environ["SKIP_MAIL"] = v
            ck(M.skip_mail() is want, f"SKIP_MAIL={v!r} -> 跳过={want}")
    finally:
        _os.environ.pop("SKIP_MAIL", None)
        if keep is not None:
            _os.environ["SKIP_MAIL"] = keep
    for path in ("src/run_auction.py", "src/pullback.py"):
        s = (ROOT / path).read_text(encoding="utf-8")
        ck('environ.get("SKIP_MAIL")' not in s and "skip_mail()" in s,
           f"{path} 走 mailer.skip_mail()，不再自己真值判断")

    # F5-15：SMTP_PORT 空串。workflow 里 secret 没配时 GitHub 把 env 设成空串，
    # `.get("SMTP_PORT", "587")` 的默认值不会命中，int("") 在 _conf() 里抛
    # ValueError —— 而 _conf 是所有发信路径的唯一入口，正文信和兜底告警信
    # 一起炸，一封都发不出去
    import localenv as LE
    import tempfile as _tf
    envkeys = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "MAIL_TO")
    saved = {k: _os.environ.get(k) for k in envkeys}
    orig_env = LE.ENV
    try:
        # 指到一个不存在的文件：_conf 里的 localenv.load() 不许把本机真凭证
        # 读进来，否则「缺省」这条用例测的就不是代码里的默认值
        LE.ENV = Path(_tf.gettempdir()) / "selftest_no_such_local.env"
        _os.environ.update({"SMTP_HOST": "localhost", "SMTP_USER": "u",
                            "SMTP_PASS": "p", "MAIL_TO": "a@b.c"})
        for val, want in ((None, 587), ("", 587), ("   ", 587),
                          ("465", 465), (" 587 ", 587)):
            if val is None:
                _os.environ.pop("SMTP_PORT", None)
            else:
                _os.environ["SMTP_PORT"] = val
            try:
                port = M._conf()["port"]
            except Exception as e:  # noqa: BLE001
                port = f"抛了 {type(e).__name__}"
            ck(port == want, f"SMTP_PORT={val!r} -> port {want}（拿到 {port}）")
    finally:
        LE.ENV = orig_env
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    return bad


def check_monotonic(c: dict) -> int:
    """「稳步抬升」只能有一个口径（M3）。

    生产以前写的是 `t1 <= t2 + 0.05 <= t3 + 0.10`，有效容差 0.05 个百分点，
    而回填表和文档都是严格的 T1<=T2<=T3。同一列两种定义拼进一张训练表，
    学到的 trend 权重是在严格口径上拟合、在容差口径上执行（教训 30）。
    那个容差还按价格漂：0.05% 在 10 元票上是半分钱，在 100 元票上是 5 分钱。
    """
    import pandas as pd
    import datasource as ds
    import run_auction as RA
    from score import is_monotonic
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n稳步抬升的口径（M3）")
    ck(is_monotonic(1.0, 1.0, 1.0) is True or bool(is_monotonic(1.0, 1.0, 1.0)),
       "平盘 t1=t2=t3 算抬升（两边一致）")
    ck(bool(is_monotonic(1.0, 1.0 + 1e-12, 1.0)),
       "1e-12 的浮点噪声仍算抬升（eps 只吸收噪声）")
    # 三个真实案例：现口径下靠 0.05 容差全判 True，严格口径全 False
    for t1, t2, t3, who in ((0.09, 0.21, 0.19, "08-25 001337 t2>t3"),
                            (0.14, 0.12, 2.70, "08-24 002428 t1>t2"),
                            (2.07, 2.04, 2.61, "08-27 600378 t1>t2")):
        ck(not bool(is_monotonic(t1, t2, t3)),
           f"真实回落不算抬升：{who}（{t1}/{t2}/{t3}）")

    # 走一遍生产的 build_features：接线错了标量函数再对也没用
    def q(code: str, chg: float) -> "object":
        prev = 10.0
        return ds.Quote(code=code, market="sh", name="测试", price=prev * (1 + chg / 100),
                        prev_close=prev, open_=prev, volume_hand=1.0e4,
                        amount_wan=2000.0, ts="20260916092510", raw=[])

    uni = pd.DataFrame([{
        "code": "600000", "prev_amount": 1.0e9, "pos_pct_60d": 0.5,
        "ma_bull": True, "platform_high": 1e9, "prev_limit_up": False,
        "prev_broken_board": False, "board_height": 0, "sector": "半导体",
        "sector_prev_limitups": 0, "blacklisted": False}])

    def feat(t1, t2, t3, drop_t1=False):
        snaps = {"T1": {} if drop_t1 else {"sh600000": q("600000", t1)},
                 "T2": {"sh600000": q("600000", t2)},
                 "T3": {"sh600000": q("600000", t3)}, "T4": {}}
        return RA.build_features(uni, snaps, c)[0]

    ck(not feat(0.09, 0.21, 0.19).monotonic,
       "build_features：t2 0.21 > t3 0.19 -> monotonic False（旧口径给 True）")
    ck(feat(2.0, 2.5, 3.0).monotonic, "真抬升 2.0/2.5/3.0 -> True")
    ck(feat(3.0, 3.0, 3.0).monotonic, "三点相等 -> True")
    ck(not feat(2.0, 2.5, 3.0, drop_t1=True).monotonic,
       "T1 漏采（没有轨迹证据）-> False，不白送 +0.15")

    # 源码里不许再出现那两个容差字面量
    src = (ROOT / "src" / "run_auction.py").read_text(encoding="utf-8")
    line = [x for x in src.splitlines() if "monotonic=" in x]
    ck(len(line) == 1 and "is_monotonic(" in line[0]
       and "0.05" not in line[0] and "0.10" not in line[0],
       f"run_auction 的 monotonic 只调 is_monotonic（现为 {line}）")
    return bad


def _mk_hist(dates, close, high, low, amount, chg):
    """一张合成日线（列名和 datasource._HIST_COLS 一致）。"""
    import pandas as pd
    return pd.DataFrame({"日期": dates, "开盘": close, "收盘": close,
                         "最高": high, "最低": low,
                         "成交量": [1.0e4] * len(dates), "成交额": amount,
                         "涨跌幅": chg, "chg_adj": [False] * len(dates)})


def check_premarket_stage2(c: dict) -> int:
    """盘中/盘后建池时，日线尾根是今天那根未定型 K 线（M4 / M5 / M20 / F6-5）。"""
    import copy
    import math
    import pandas as pd
    import premarket
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n候选池形态字段：以 T-1 为最后一根（M4/M5/M20）")
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
    n = 70
    dates = [(today - dt.timedelta(days=n - 1 - i)).isoformat() for i in range(n)]
    close = [10.0 + i * 0.01 for i in range(n - 2)] + [10.40, 12.00]
    high = [x + 0.05 for x in close[:-1]] + [12.50]
    low = [x - 0.05 for x in close]
    amount = [1.0e8] * (n - 2) + [3.0e8, 5.0e9]
    chg = [0.1] * (n - 2) + [10.0, -3.0]
    h_all = _mk_hist(dates, close, high, low, amount, chg)     # 含今天那根
    h_prev = h_all.iloc[:-1].reset_index(drop=True)            # 盘前看到的

    df = pd.DataFrame([{"code": "600000", "name": "甲", "lim": 10.0,
                        "prev_gain": 10.0, "prev_amount": 3.0e8,
                        "成交额": 3.0e8}])
    look = c["screen"]["breakout_lookback"]
    c1 = copy.deepcopy(c)
    c1["universe"]["min_listed_days"] = 0
    orig = premarket.ds.daily_hist_many
    try:
        premarket.ds.daily_hist_many = lambda *a, **k: {"600000": h_all}
        a = premarket.stage2(df.copy(), c1).iloc[0]
        premarket.ds.daily_hist_many = lambda *a_, **k: {"600000": h_prev}
        b = premarket.stage2(df.copy(), c1).iloc[0]
        cols = ["pos_pct_60d", "ma_bull", "platform_high", "board_height",
                "prev_broken_board", "amount_ratio_5d"]
        same = [k for k in cols if a[k] == b[k]]
        ck(len(same) == len(cols),
           f"含今天那根 == 不含今天那根（差的列：{sorted(set(cols) - set(same))}）")
        # 值本身要是 T-1 口径的。数据故意造成「今天那根会改掉每一个字段」
        ck(a["board_height"] == 1,
           f"连板高度按昨日算 = 1（拿到 {a['board_height']}）")
        want_ph = float(h_prev["最高"].tail(look).max())
        ck(abs(a["platform_high"] - want_ph) < 1e-9
           and abs(float(h_all["最高"].tail(look).max()) - want_ph) > 1.0,
           f"平台高点不含今天最高价（{want_ph:.2f}，含今天会变成 12.50）")
        ck(abs(a["amount_ratio_5d"] - 3.0) < 1e-9,
           f"放量比值 = 昨日额/前5日均额 = 3.0（拿到 {a['amount_ratio_5d']:.3f}）")
        w = pd.Series(close[:-1]).tail(60)
        want_pos = (close[-2] - w.min()) / (w.max() - w.min())
        ck(abs(a["pos_pct_60d"] - want_pos) < 1e-9 and a["pos_pct_60d"] < 0.99,
           f"60 日位置按截至昨日的窗口算（{want_pos:.4f}，含今天会是 1.0）")

        # M20：成交额整列 NaN / 只有最后一根 NaN，都要落回默认 1.0
        for label, amt in (("整列 NaN", [float("nan")] * n),
                           ("只有昨日那根 NaN",
                            [1.0e8] * (n - 2) + [float("nan"), 5.0e9])):
            hn = _mk_hist(dates, close, high, low, amt, chg)
            premarket.ds.daily_hist_many = lambda *a_, **k: {"600000": hn}
            r = premarket.stage2(df.copy(), c1).iloc[0]
            ck(math.isfinite(r["amount_ratio_5d"])
               and r["amount_ratio_5d"] == 1.0,
               f"成交额{label} -> amount_ratio_5d 落回 1.0（`or 1.0` 兜不住 NaN）")

        # M5：昨日炸板的涨停价判据。600654 2026-09-14 昨收 3.09 最高 3.39，
        # 涨停价 3.40；−0.01 的老写法算出 3.389，把它判成触及涨停
        for hi_last, want, why in ((3.39, False, "最高 3.39 < 涨停价 3.40"),
                                   (3.40, True, "最高 3.40 == 涨停价")):
            # 倒数第 3 根（剔掉今天之后就是 T-2）收 3.09，T-1 最高 hi_last
            cl2 = [3.00] * (n - 3) + [3.09, 3.15, 3.20]
            hb = _mk_hist(dates, cl2,
                          [x + 0.01 for x in cl2[:-2]] + [hi_last, 3.30],
                          [x - 0.05 for x in cl2],
                          [1.0e8] * n, [0.1] * (n - 2) + [3.0, 1.0])
            premarket.ds.daily_hist_many = lambda *a_, **k: {"600000": hb}
            df2 = df.copy()
            df2["prev_gain"] = 3.0          # 昨日没封板，炸板判据才会走到
            r = premarket.stage2(df2, c1).iloc[0]
            ck(bool(r["prev_broken_board"]) is want, f"昨日炸板：{why} -> {want}")
    finally:
        premarket.ds.daily_hist_many = orig

    # 空池也要有完整的六列，否则 main() 的 df[cols] 抛一个指不到根因的 KeyError
    empty = df.iloc[0:0].copy()
    premarket.ds.daily_hist_many = lambda *a_, **k: {}
    try:
        out = premarket.stage2(empty, c1)
        ck(all(k in out.columns for k in premarket._SHAPE_COLS),
           "0 只的候选池仍然带齐六个形态列（不是 KeyError）")
    finally:
        premarket.ds.daily_hist_many = orig
    return bad


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def check_hist_units(c: dict) -> int:
    """三路日线的单位和「涨跌幅」语义（F6-1 / F6-2 / F6-5）。"""
    import math
    import datasource as D
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n日线/快照的成交量单位与涨跌幅语义（F6-1/F6-2/F6-5）")
    ck(D.tx_vol_hand("688008", 100.0) == 1.0, "科创板 688：腾讯给股，折成手")
    ck(D.tx_vol_hand("300750", 100.0) == 100.0
       and D.tx_vol_hand("600000", 100.0) == 100.0
       and D.tx_vol_hand("920159", 100.0) == 100.0,
       "创业板/主板/北交所：本来就是手，不动")

    def body(code, price, vol, amt_wan):
        f = [""] * 45
        f[1], f[2] = "测试", code
        f[3], f[4], f[5] = str(price), str(price * 0.99), str(price * 0.995)
        f[6] = str(vol)
        f[30] = "20260916161457"
        f[37] = str(amt_wan)
        f[38] = "3.06"
        return "~".join(f)

    # 2026-09-16 16:14 实测值
    q688 = D._parse_tx_body("sh688008", body("688008", 199.89, 35064294, 682899))
    q600 = D._parse_tx_body("sh600000", body("600000", 9.10, 723404, 65635))
    ck(abs(q688.volume_hand - 350642.94) < 1e-6,
       f"sh688008 快照成交量折成手 350642.94（拿到 {q688.volume_hand:.2f}）")
    ck(q600.volume_hand == 723404.0, "sh600000 快照成交量原样 723404 手")
    for q, nm in ((q688, "688008"), (q600, "600000")):
        ratio = q.amount_yuan / (q.price * q.volume_hand * 100.0)
        ck(abs(ratio - 1.0) < 0.05,
           f"{nm} 成交额/(价×量×100) = {ratio:.3f}（错单位时是 0.0097）")

    tx = {"data": {"sh688008": {"day": [
              ["2026-09-15", "197.0", "198.0", "200", "196", "30000000.000"],
              ["2026-09-16", "198.5", "199.89", "203", "197", "35064294.000"]]},
                   "sh600000": {"day": [
              ["2026-09-15", "9.00", "9.05", "9.10", "8.99", "700000.000"],
              ["2026-09-16", "9.05", "9.10", "9.15", "9.00", "723404.000"]]}}}
    sina688 = [{"day": "2026-09-15", "open": "197.0", "high": "200",
                "low": "196", "close": "198.0", "volume": "30000000"},
               {"day": "2026-09-16", "open": "198.5", "high": "203",
                "low": "197", "close": "199.89", "volume": "35064294"}]

    class FakeSession:
        def get(self, url, **kw):
            if "fqkline" in url:
                sym = "sh688008" if "sh688008" in url else "sh600000"
                return _FakeResp({"data": {sym: tx["data"][sym]}})
            if "getKLineData" in url:
                return _FakeResp(sina688)
            raise AssertionError(f"自测不该访问 {url}")

    orig = D._SESSION
    D._SESSION = FakeSession()
    try:
        h = D.daily_hist_tx("688008", "20260901", "20260916")
        ck(abs(float(h["成交量"].iloc[-1]) - 350642.94) < 1e-6,
           f"腾讯日 K 688 成交量折成手（拿到 {float(h['成交量'].iloc[-1]):.2f}）")
        want_amt = (203 + 197 + 199.89) / 3.0 * 350642.94 * 100.0
        ck(abs(float(h["成交额"].iloc[-1]) / want_amt - 1.0) < 1e-9,
           "成交额跟着回到元（不折算会大 100 倍：7009 亿 vs 真值 68 亿）")
        ck(bool(math.isnan(float(h["涨跌幅"].iloc[0]))),
           "首根涨跌幅是 NaN，不是 0.0（0.0 是「昨天平盘」这个假事实）")
        ck(abs(float(h["涨跌幅"].iloc[1])
               - round((199.89 - 198.0) / 198.0 * 100, 2)) < 1e-9,
           "第二根 = 相邻不复权收盘价之比")
        ck(not bool(h["chg_adj"].any()), "腾讯路 chg_adj 全 False（不复权口径）")

        h6 = D.daily_hist_tx("600000", "20260901", "20260916")
        ck(float(h6["成交量"].iloc[-1]) == 723404.0, "主板日 K 成交量原样")

        s = D.daily_hist_sina("688008", "20260901", "20260916")
        ck(abs(float(s["成交量"].iloc[-1]) - float(h["成交量"].iloc[-1])) < 1e-6,
           "新浪与腾讯同日成交量相等（两路单位统一了）")
        ck(bool(math.isnan(float(s["涨跌幅"].iloc[0]))),
           "新浪首根没有前置行时涨跌幅也是 NaN")
        s2 = D.daily_hist_sina("688008", "20260916", "20260916")
        ck(not math.isnan(float(s2["涨跌幅"].iloc[0])),
           "窗口起点之前有行时，首个在窗口内的行有真实涨跌幅")
        ck(not bool(s["chg_adj"].any()), "新浪路 chg_adj 全 False")
    finally:
        D._SESSION = orig

    # 形态线：快照量和日线量必须同单位，否则 688 的量比差 100 倍
    from pullback import hist_turnover
    import pandas as pd
    row = pd.Series({"成交量": 175321.47})        # 手
    ck(abs(hist_turnover(row, 3.06, 350642.94) - 1.53) < 1e-6,
       "688 历史换手反推 1.53%（快照按股时会算成 0.0153%，直接出界）")
    ck("chg_adj" in D._HIST_COLS, "_HIST_COLS 带 chg_adj，三路语义能被区分")
    return bad


def _snapshot(codes, rows) -> "object":
    """拼一张 spot_all 形状的快照。代码必须来自真实代码表（历史教训 2）。"""
    import pandas as pd
    rec = []
    for i, r in enumerate(rows):
        rec.append({"代码": codes[i], "名称": r.get("name", f"票{i}"),
                    "最新价": 10.0, "涨跌幅": r.get("chg", 0.0),
                    "成交额": r.get("amount", 1.0e7),
                    "换手率": r.get("turn", 0.5),
                    "总市值": float("nan")})
    return pd.DataFrame(rec)


def check_pool(c: dict) -> int:
    """候选池的每一条入池/剔除规则都要有一个用例真的碰到它（教训 26）。

    2026-09-16 前 stage1 里真正生效的三条规则（换手>=5、成交额前 600、
    昨跌<=-9）是写死的字面量，而 config.yaml 的 universe 段有 6/11 个键
    没有任何代码读：用户改 max_3d_gain_pct / min_listed_days / in_hot_board
    一点效果都没有，面板上却照写「已剔次新」。
    """
    import copy
    import pandas as pd
    import premarket
    from datasource import is_new_listing, is_st_name
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n候选池规则")
    real = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
            .str.zfill(6).tolist())
    codes = [x for x in real if x.startswith(("600", "601", "603"))][:8]

    # 八只票，每一条规则都有一只**只**靠它进池（或只被它剔）
    rows = [
        {"name": "换手5",   "turn": 5.0, "amount": 1.0e6},   # 0 只靠换手
        {"name": "换手49",  "turn": 4.9, "amount": 1.1e6},   # 1 哪条都不沾
        {"name": "昨涨5",   "turn": 0.5, "amount": 1.2e6, "chg": 5.0},
        {"name": "昨涨停",  "turn": 0.5, "amount": 1.3e6, "chg": 10.0},
        {"name": "昨跌停",  "turn": 9.0, "amount": 1.4e6, "chg": -9.0},
        {"name": "ST测试",  "turn": 9.0, "amount": 1.5e6},   # 5 名字剔除
        {"name": "C测试",   "turn": 9.0, "amount": 1.6e6},   # 6 次新剔除
        {"name": "巨量甲",  "turn": 0.5, "amount": 9.9e9},   # 7 只靠成交额名次
    ]
    spot = _snapshot(codes, rows)
    c1 = copy.deepcopy(c)
    c1["universe"]["include_if"]["amount_rank_top"] = 1   # 只让第 7 只靠名次进
    got = set(premarket.stage1(spot, c1)["code"])
    ck(codes[0] in got and codes[1] not in got,
       "换手 5.0 进池、4.9 不进（阈值 5.0 真的被判到）")
    ck(codes[2] in got and codes[3] in got, "昨涨 5% / 昨日涨停 各自进池")
    ck(codes[4] not in got, "昨日跌停 -9.0 剔除（换手 9 也救不回来）")
    ck(codes[5] not in got and codes[6] not in got, "ST / 次新按名字剔除")
    ck(codes[7] in got, "成交额名次进池")

    c2 = copy.deepcopy(c1)
    c2["universe"]["include_if"]["turnover_pct_gte"] = 6.0
    got2 = set(premarket.stage1(spot, c2)["code"])
    ck(codes[0] not in got2, "换手阈值读 config（改成 6.0 后 5.0 那只出局）")

    c3 = copy.deepcopy(c1)
    c3["universe"]["exclude_yesterday_limit_down"] = False
    ck(codes[4] in set(premarket.stage1(spot, c3)["code"]),
       "exclude_yesterday_limit_down=False 时昨日跌停留下")
    c4 = copy.deepcopy(c1)
    c4["universe"]["exclude_st"] = False
    ck(codes[5] in set(premarket.stage1(spot, c4)["code"]),
       "exclude_st=False 时 ST 留下（这条键真的有人读）")
    c5 = copy.deepcopy(c1)
    c5["universe"]["min_listed_days"] = 0
    ck(codes[6] in set(premarket.stage1(spot, c5)["code"]),
       "min_listed_days=0 时次新留下")
    c6 = copy.deepcopy(c1)
    c6["universe"]["include_if"]["amount_rank_top"] = 3
    got6 = set(premarket.stage1(spot, c6)["code"])
    ck(codes[1] not in got6 and codes[0] in got6,
       "成交额名次阈值读 config（前 3 仍不含那只 1.1e6 的票）")

    # 换手率字段缺失时必须报错：以前 _turnover 取不到返回 0，整条规则被
    # 静默关掉（08-28 池子只有 635 只，正常 1100~1400）
    dead = _snapshot(codes, [dict(r, turn=0.0) for r in rows])
    try:
        premarket.stage1(dead, c1)
        ck(False, "换手率整列为 0 时 stage1 应报错")
    except RuntimeError:
        ck(True, "换手率整列为 0 -> 报错发告警邮件，不静默关掉入池规则")

    # 快照给不出总市值：配置一旦要求按市值筛，必须报错而不是筛空池子
    for key, val in (("max_mktcap_yi", 500), ("min_mktcap_yi", 20)):
        c7 = copy.deepcopy(c1)
        c7["universe"][key] = val
        try:
            premarket.stage1(spot, c7)
            ck(False, f"总市值全 NaN + {key} 非 0 应报错")
        except RuntimeError:
            ck(True, f"{key} 非 0 而快照没有市值 -> 报错（否则当天发空榜）")
    no_col = spot.drop(columns=["总市值"])
    c7b = copy.deepcopy(c1)
    c7b["universe"]["max_mktcap_yi"] = 500
    try:
        premarket.stage1(no_col, c7b)
        ck(False, "快照连总市值这一列都没有时也要报错")
    except RuntimeError:
        ck(True, "快照没有总市值列 -> 报错，不是 KeyError")

    # 将来真补上市值源时，这条过滤本身必须是对的
    cap_rows = [dict(r, turn=6.0) for r in rows[:3]]
    cap_spot = _snapshot(codes[:3], cap_rows)
    cap_spot["总市值"] = [50e8, 150e8, 80e8]
    c8 = copy.deepcopy(c1)
    c8["universe"]["max_mktcap_yi"] = 100
    got8 = set(premarket.stage1(cap_spot, c8)["code"])
    ck(got8 == {codes[0], codes[2]},
       f"市值有真值时按亿元阈值过滤（150 亿那只出局），拿到 {sorted(got8)}")

    # 过滤到一只不剩不是合法结果：真实市场每天 1000~1400 只
    zero = _snapshot(codes, [dict(r, amount=0.0) for r in rows])
    try:
        premarket.stage1(zero, c1)
        ck(False, "候选池被过滤成 0 只时 stage1 应报错")
    except RuntimeError:
        ck(True, "空候选池 -> 报错（否则 09:25 照跑，发一封看起来合法的空榜邮件）")

    # 次新判据：旧正则 ^[NC] 要求跟一个空格，而腾讯真实的次新名字里没有空格，
    # 那条规则从上线起一次都没生效（688836/688826/301655 都进过池并被打了分）
    for nm in ("C频准", "C绿控传动", "C宇树-W", "N拓竹"):
        ck(is_new_listing(nm), f"次新识别：{nm}")
    for nm in ("万  科Ａ", "农 产 品", "中信证券", "*ST 浦发"):
        ck(not is_new_listing(nm), f"不是次新：{nm}")
    ck(is_st_name("*ST 浦发") and is_st_name("退市美置")
       and not is_st_name("中信证券"), "ST/退市名字判据")

    # config 里不许有没人读的键，stage1 里也不许再出现裸阈值
    src_pm = (ROOT / "src" / "premarket.py").read_text(encoding="utf-8")
    src_bf = (ROOT / "src" / "learn" / "backfill.py").read_text(encoding="utf-8")
    u = c["universe"]
    leaves = [k for k in u if k != "include_if"] + list(u["include_if"])
    dead_keys = [k for k in leaves
                 if f'"{k}"' not in src_pm and f'"{k}"' not in src_bf]
    ck(not dead_keys, f"universe 段没有死配置键（漏读的：{dead_keys}）")
    import ast
    fn = next(n for n in ast.walk(ast.parse(src_pm))
              if isinstance(n, ast.FunctionDef) and n.name == "stage1")
    nums = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)}
    hard = sorted(nums & {5.0, 600, 600.0, 9.0})
    ck(not hard, f"stage1 里没有写死的阈值（发现：{hard}）")
    return bad


def check_stage2_listing(c: dict) -> int:
    """stage2 的次新剔除：min_listed_days 以前是死配置。

    「拉不到日线」和「拿到了但根数不足」必须分开，否则日线源一降级
    整池都会被当次新剔光。
    """
    import copy
    import pandas as pd
    import premarket
    bad = 0

    def ck(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad += 1

    print("\n次新剔除（stage2）")

    # 日期一直排到今天：stage2 会把「今天那根」剔掉（M4），所以这里造 n 根
    # 拿到手的是 n-1 根，次新阈值判的正是剔完之后的根数
    tz = dt.timezone(dt.timedelta(hours=8))
    tdy = dt.datetime.now(tz).date()

    def fake_hist(n: int):
        return pd.DataFrame({
            "日期": [(tdy - dt.timedelta(days=n - 1 - i)).isoformat()
                   for i in range(n)],
            "收盘": [10.0 + i * 0.01 for i in range(n)],
            "最高": [10.1 + i * 0.01 for i in range(n)],
            "最低": [9.9 + i * 0.01 for i in range(n)],
            "成交额": [1.0e8] * n,
            "涨跌幅": [0.5] * n,
        })

    real = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
            .str.zfill(6).tolist())
    # 池子要够大：次新占比超 5% 会走「疑似源降级」分支，那是另一条用例
    codes = [x for x in real if x.startswith("600")][:42]
    n = len(codes)
    df = pd.DataFrame({"code": codes, "name": [f"票{i}" for i in range(n)],
                       "lim": [10.0] * n, "prev_gain": [1.0] * n,
                       "prev_amount": [1e8] * n, "成交额": [1e8] * n})
    hists = {x: fake_hist(95) for x in codes}
    hists[codes[1]] = fake_hist(31)      # 次新（剔掉今天那根后 30 根）
    hists[codes[2]] = None               # 日线拉不到
    orig = premarket.ds.daily_hist_many
    premarket.ds.daily_hist_many = lambda *a, **k: hists      # 离线
    try:
        c1 = copy.deepcopy(c)
        c1["universe"]["min_listed_days"] = 60
        got = set(premarket.stage2(df.copy(), c1)["code"])
        ck(codes[0] in got, "95 根日线的老票保留")
        ck(codes[1] not in got, "30 根日线的次新剔除（min_listed_days=60）")
        ck(codes[2] in got, "拉不到日线的票保留（源降级不等于次新）")
        c2 = copy.deepcopy(c)
        c2["universe"]["min_listed_days"] = 10
        ck(codes[1] in set(premarket.stage2(df.copy(), c2)["code"]),
           "阈值读 config（=10 时那只 30 根的票留下）")
        # 超过 5% 判成次新 = 日线源普遍返回得少，只告警不剔
        many = {k: fake_hist(31) for k in codes}
        premarket.ds.daily_hist_many = lambda *a, **k: many
        got3 = set(premarket.stage2(df.copy(), c1)["code"])
        ck(len(got3) == n, "次新占比超 5% 时不剔除（防日线源降级误杀整池）")
    finally:
        premarket.ds.daily_hist_many = orig
    return bad


def _literals(mod_path: str, needle: str) -> list[str]:
    """模块里**会进输出**的字符串字面量里有没有 needle。

    文档字符串不算：说明「以前写死 09:25:10」是解释，不是在渲染它。
    """
    import ast
    tree = ast.parse((ROOT / mod_path).read_text(encoding="utf-8"))
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            b = n.body[0] if n.body else None
            if (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                    and isinstance(b.value.value, str)):
                docs.add(id(b.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs and needle in n.value]


def _counter():
    """返回 (ck, get_bad)：每个 check_* 都重复写一遍太吵。"""
    box = {"bad": 0}

    def ck(ok: bool, msg: str) -> None:
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            box["bad"] += 1
    return ck, lambda: box["bad"]


def check_calendar(c: dict) -> int:
    """交易日历：格式归一 + 原子写（M19 / F6-10）。

    akshare 1.18 的 tool_trade_date_hist_sina 返回 datetime.date，
    `str(d)` 恰好是 YYYY-MM-DD 所以一直没出事；requirements 写的是
    `akshare>=1.16.0`，上游一改格式，`today not in s` 就恒真，
    早盘/盘前/形态三条线每天「非交易日」退出 0，云端托底也一起哑。
    """
    import ast
    import pandas as pd
    import datasource as D
    ck, get = _counter()
    print("\n交易日历（M19 / F6-10）")

    t = dt.date.today()
    iso, ymd = t.isoformat(), t.strftime("%Y%m%d")
    for label, col in (("datetime.date", pd.Series([t])),
                       ("datetime64", pd.Series(pd.to_datetime([iso]))),
                       ("object/Timestamp",
                        pd.Series([pd.Timestamp(iso)], dtype=object)),
                       ("'YYYYMMDD' 字符串", pd.Series([ymd])),
                       ("YYYYMMDD 整数", pd.Series([int(ymd)]))):
        try:
            got = D._norm_trade_dates(col, today=t)
        except Exception as e:  # noqa: BLE001
            got = f"抛了 {type(e).__name__}"
        ck(got == {iso}, f"日历列（{label}）归一成 {iso}，拿到 {got}")
    for col, why in ((pd.Series(["abc"]), "垃圾"),
                     (pd.Series([], dtype=object), "空"),
                     (pd.Series(["2020-01-01"]), "不覆盖最近 30 天")):
        try:
            D._norm_trade_dates(col, today=t)
            ck(False, f"日历{why}时必须抛（否则坏日历会写进缓存，兜底一起废）")
        except Exception:  # noqa: BLE001
            ck(True, f"日历{why}时抛异常 -> 调用方走缓存兜底")

    # 原子写：Path.write_text 先把文件截断成 0 字节再写，读方（控制台、
    # 学习闸门）在那 1ms 里读到空文件（实测 3 秒 337 次读里 138 次）
    src = (ROOT / "src" / "datasource.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "trade_dates")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    ck(not any(isinstance(x.func, ast.Attribute) and x.func.attr == "write_text"
               and getattr(x.func.value, "id", "") == "cache" for x in calls),
       "trade_dates 不再 cache.write_text（那是先截断再写）")
    ck(any(isinstance(x.func, ast.Attribute) and x.func.attr == "replace"
           and getattr(x.func.value, "id", "") == "os" for x in calls),
       "trade_dates 走 tmp + os.replace 原子换名")
    ck("_norm_trade_dates(" in src.split("def trade_dates")[1],
       "trade_dates 用 _norm_trade_dates 归一后才写缓存")
    return get()


def _fake_quotes(D, ts: str, amount_wan: float = 100.0):
    """造一批 Quote。ts 是腾讯 f[30]，spot_all 拿它判有没有翻篇。"""
    def _f(syms, **kw):
        return {s: D.Quote(code=s[2:], market=s[:2], name="测试",
                           price=10.0, prev_close=10.0, open_=0.0,
                           volume_hand=0.0, amount_wan=amount_wan,
                           ts=ts, raw=[]) for s in syms}
    return _f


def check_spot_all(c: dict) -> int:
    """盘前快照的翻篇闸 + 建池时刻闸（F6-9 / F6-8）。"""
    import datasource as D
    import pandas as pd
    import premarket
    ck, get = _counter()
    print("\n盘前快照的时间假设（F6-9）")

    # 代码必须来自真实代码表（历史教训 2）
    real = (pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
            .str.zfill(6).tolist())[:100]
    orig_codes, orig_fetch = D.load_code_list, D.fetch_quotes
    try:
        D.load_code_list = lambda: real
        D.fetch_quotes = _fake_quotes(D, "20260917090500")
        try:
            D.spot_all(expect_date="2026-09-16")
            ck(False, "快照时间戳全是今天时必须报错（此刻建池会把竞价态当昨收）")
        except RuntimeError as e:
            ck("翻篇" in str(e), f"翻篇的快照被拒：{e}")

        D.fetch_quotes = _fake_quotes(D, "20260916161427")
        df = D.spot_all(expect_date="2026-09-16")
        ck(len(df) == 100 and bool((df["成交额"] > 0).all()),
           "时间戳是最近已收盘交易日 -> 正常返回 100 只")
        ck("总市值" not in df.columns,
           "快照不再输出恒为 NaN 的「总市值」列（不输出一列永远是假的数据）")

        def mixed(n_today: int):
            a = _fake_quotes(D, "20260917090500")
            b = _fake_quotes(D, "20260916161427")
            def _f(syms, **kw):
                out = b(syms, **kw)
                for s in list(syms)[:n_today]:
                    out.update(a([s]))
                return out
            return _f

        D.fetch_quotes = mixed(5)
        ck(len(D.spot_all(expect_date="2026-09-16")) == 100,
           "5/100 只时间戳不一致（<=5%）放行，留给零星停牌/未更新")
        D.fetch_quotes = mixed(6)
        try:
            D.spot_all(expect_date="2026-09-16")
            ck(False, "6/100（>5%）应报错")
        except RuntimeError:
            ck(True, "6/100（>5%）报错")
    finally:
        D.load_code_list, D.fetch_quotes = orig_codes, orig_fetch

    tz = dt.timezone(dt.timedelta(hours=8))
    premarket.assert_pre_open(now=dt.datetime(2026, 9, 17, 8, 59, tzinfo=tz))
    ck(True, "08:59 建池放行")
    for hh, mm in ((9, 12), (10, 30)):
        try:
            premarket.assert_pre_open(
                now=dt.datetime(2026, 9, 17, hh, mm, tzinfo=tz))
            ck(False, f"{hh:02d}:{mm:02d} 建池必须报错（快照可能已翻成竞价态）")
        except RuntimeError:
            ck(True, f"{hh:02d}:{mm:02d} 建池报错，走告警邮件而不是发脏清单")

    tds = {"2026-09-15", "2026-09-16"}
    ck(premarket.last_closed_day(
        tds, dt.datetime(2026, 9, 17, 8, 30, tzinfo=tz)) == "2026-09-16",
       "盘前 08:30 的最近已收盘交易日是昨天")
    ck(premarket.last_closed_day(
        tds, dt.datetime(2026, 9, 16, 15, 30, tzinfo=tz)) == "2026-09-16",
       "15:05 之后算当天")
    ck(premarket.last_closed_day(
        tds, dt.datetime(2026, 9, 16, 8, 30, tzinfo=tz)) == "2026-09-15",
       "15:05 之前算上一个交易日（周末/节假日跳过）")
    ck(premarket.last_closed_day(set()) == "",
       "日历拿不到时返回空串，调用方跳过校验而不是拿错日期去比")

    pm = (ROOT / "src" / "premarket.py").read_text(encoding="utf-8")
    ck("assert_pre_open()" in pm and "spot_all(expect_date=" in pm,
       "premarket.main 真的接了这两道闸（接线错了函数再对也没用）")
    return get()


def check_codes(c: dict) -> int:
    """代码表：缺页不许返回残表，短表不许覆盖缓存（F6-11）。

    页是按 symbol 排序切的，丢一页就是丢连续一段代码（离线复现：第 3 页
    超时 -> 5448 只，丢的是 920433~920837 一整段北交所），而下游
    早盘候选池 / 形态扫描 / 起涨预测的当日 K 线追加 / 学习回填全读它，
    唯一的信号是一行 log.warning（教训 16）。
    """
    import json as _json
    import re
    import tempfile
    import types
    import pandas as pd
    import requests
    import datasource as D
    ck, get = _counter()
    print("\n代码表刷新（F6-11）")

    allc = sorted((pd.read_csv(ROOT / "cache" / "codes.csv", dtype=str)["code"]
                   .str.zfill(6).tolist()), key=D.to_symbol)
    pages = {i + 1: allc[i * 100:(i + 1) * 100]
             for i in range((len(allc) + 99) // 100)}

    class _Resp:
        def __init__(self, text: str) -> None:
            self.text, self.encoding = text, "gbk"

    def make_get(bad_pages: set[int]):
        def _get(url, **kw):
            p = int(re.search(r"page=(\d+)", url).group(1))
            if p in bad_pages:
                raise requests.exceptions.ReadTimeout("自测：假超时")
            return _Resp(_json.dumps([{"code": x} for x in pages.get(p, [])]))
        return _get

    class _NoSleep:
        """重试退避 3 次要 3 秒，自测里不等（datasource 只用 time.sleep）。"""
        @staticmethod
        def sleep(_):
            return None

    fake_ak = types.ModuleType("akshare")

    def _boom(*a, **k):
        raise RuntimeError("离线自测：不许调 akshare")
    fake_ak.stock_info_a_code_name = _boom
    fake_ak.stock_zh_a_spot_em = _boom

    orig = (D._SESSION.get, D.time, D._ROOT, D._sina_code_list,
            sys.modules.get("akshare"))
    tmp = Path(tempfile.mkdtemp(prefix="selftest_codes_"))
    try:
        D.time = _NoSleep
        sys.modules["akshare"] = fake_ak
        D._SESSION.get = make_get({3})
        try:
            got = D._sina_code_list()
            ck(False, f"单页失败必须抛，不能返回 {len(got)} 只的残表")
        except RuntimeError as e:
            ck("不返回残表" in str(e), f"第 3 页超时 -> 抛异常：{e}")

        D._SESSION.get = make_get(set(range(33, 41)))
        try:
            D._sina_code_list()
            ck(False, "整组失败必须抛（got==0 那条 break 不许放残表出去）")
        except RuntimeError:
            ck(True, "第 33~40 页整组失败 -> 抛异常，不是 3200 只的残表")

        D._SESSION.get = make_get(set())
        ck(len(D._sina_code_list()) == len(set(allc)),
           "全部页正常时返回完整代码表")

        # 写盘闸：新表比现有表少超过 2% 就拒绝覆盖
        (tmp / "cache").mkdir()
        dst = tmp / "cache" / "codes.csv"
        pd.DataFrame({"code": allc}).to_csv(dst, index=False)
        D._ROOT = tmp
        # 少 300 只 = 少 5.4%，超过 2% 的闸（少 100 只在容差内，照写）
        D._sina_code_list = lambda: allc[:len(allc) - 300]
        try:
            D.refresh_code_list()
            ck(False, "比现有表少 300 只时应拒绝覆盖并最终报错")
        except RuntimeError:
            ck(True, "新表比现有表少超过 2% -> 拒绝覆盖，换下一个源")
        ck(len(pd.read_csv(dst, dtype=str)) == len(allc),
           f"旧缓存原样保留 {len(allc)} 只")

        grown = allc + [f"9209{i:02d}" for i in range(12)]
        D._sina_code_list = lambda: sorted(set(grown))
        ck(len(D.refresh_code_list()) == len(set(grown)), "正常变长时照常写入")
        ck(len(pd.read_csv(dst, dtype=str)) == len(set(grown)),
           "缓存被更新成新表")
    finally:
        (D._SESSION.get, D.time, D._ROOT, D._sina_code_list,
         old_ak) = orig
        if old_ak is None:
            sys.modules.pop("akshare", None)
        else:
            sys.modules["akshare"] = old_ak
    return get()


def check_em_circuit(c: dict) -> int:
    """东财熔断只认「不通」，不认「没数据」（F6-6）。"""
    import pandas as pd
    import datasource as D
    ck, get = _counter()
    print("\n东财熔断的语义（F6-6）")

    saved = dict(D._em_state)
    D._em_state.update(fail_streak=0, tripped=False, since_probe=0,
                       em=0, tx=0, sina=0)
    short = pd.DataFrame([{"日期": f"2026-08-{d:02d}", "开盘": 1.0, "收盘": 1.0,
                           "最高": 1.0, "最低": 1.0, "成交量": 1.0,
                           "成交额": 100.0, "涨跌幅": 0.0, "chg_adj": True}
                          for d in range(1, 11)])          # 10 根，新股
    empty = pd.DataFrame(columns=D._HIST_COLS)
    orig = (D.daily_hist_em, D.daily_hist_tx, D.daily_hist_sina)
    try:
        D.daily_hist_em = lambda *a, **k: short.copy()
        D.daily_hist_tx = lambda *a, **k: empty.copy()
        D.daily_hist_sina = lambda *a, **k: empty.copy()
        r = None
        for i in range(12):
            r = D.daily_hist(f"6889{i:02d}", "20260401", "20260902")
        ck(r is not None and len(r) == 0,
           "东财只有 10 根时 daily_hist 仍返回空表（>=25 根的返回契约不变）")
        ck(D.hist_source_stats()["熔断中"] is False,
           "连续 12 只东财短历史（正常返回 <25 根）不触发熔断")
        ck(D.hist_source_stats()["东财"] == 0,
           "没用上的短历史不计进「东财命中数」")
        ck("新浪" in D.hist_source_stats(),
           "统计里有新浪那一路（以前 1125 只里 780 只去向不明）")

        def boom(*a, **k):
            raise ConnectionError("push2his 被掐")
        D.daily_hist_em = boom
        for i in range(12):
            D.daily_hist(f"6000{i:02d}", "20260401", "20260902")
        ck(D.hist_source_stats()["熔断中"] is True, "连续 12 只东财抛异常才熔断")
    finally:
        D.daily_hist_em, D.daily_hist_tx, D.daily_hist_sina = orig
        D._em_state.clear()
        D._em_state.update(saved)
    return get()


def check_exports(c: dict) -> int:
    """空榜日的导入文件 / 采集时刻 / 板块阈值 / 面板横幅
    （F5-1 / F5-11 / M22 / F5-10）。

    全部写 tempfile：out/ 和 out_pullback/ 是当天的生产产物（教训 17）。
    """
    import ast
    import re
    import inspect
    import tempfile
    import mailer as M
    import pullback_export as PE
    from ths_export import (write_ths_blocks, write_ths_panel,
                            _criteria_line, REFRESH_JS)
    ck, get = _counter()
    print("\n空榜产物与面板文案（F5-1 / F5-11 / M22 / F5-10）")

    tmp = Path(tempfile.mkdtemp(prefix="selftest_exp_"))
    tiers = c["output"]["ths_tiers"]
    sel = [score_one(mk(code="600111", name="甲"), c),
           score_one(mk(code="300750", name="乙", sector_members=2), c)]

    # 1) 空榜日：先写一份有票的，再写空榜，旧代码必须被冲掉
    write_ths_blocks(sel, tmp, tiers, "2026-09-04")
    write_tdx_custom(sel, tmp)
    ths_paths = write_ths_blocks([], tmp, tiers, "2026-08-28")
    tdx_paths = write_tdx_custom([], tmp)
    for p in ths_paths:
        ck(p.stat().st_size == 0, f"空榜后 {p.name} 是 0 字节")
    for p in tdx_paths:
        # tdx_export.py 不在本组白名单，它那两行还无条件补一个 "\r\n"
        ck(re.search(rb"\d{6}", p.read_bytes()) is None,
           f"空榜后 {p.name} 不含任何 6 位代码")
    pb = PE.write_blocks([], tmp, "2026-09-11")
    ck(pb[0].stat().st_size == 0, "形态线空榜也是 0 字节（30 天里 26 天是空榜）")

    # 接线：这两个写文件的调用不许再躲在 `if sel` 里面
    def _guarded(path: str, fname: str, names: set[str]) -> list[str]:
        fn = next(n for n in ast.walk(ast.parse(
            (ROOT / path).read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == fname)
        hit = []
        for node in ast.walk(fn):
            if isinstance(node, (ast.If, ast.IfExp)):
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call)
                            and getattr(sub.func, "id", "") in names):
                        hit.append(sub.func.id)
        return hit
    ck(not _guarded("src/run_auction.py", "stage_enrich",
                    {"write_ths_blocks", "write_tdx_custom"}),
       "run_auction.stage_enrich 无条件写导入文件（空榜也写）")
    ck(not _guarded("src/pullback.py", "stage_send", {"write_blocks"}),
       "pullback.stage_send 无条件写导入文件")

    # 2) 采集时刻不再写死 09:25:10
    res = {"A": sel, "B": []}
    h_late = M.build_html("2026-08-28", res, {}, "自测", "抢救",
                          collected="今开（抢救，采样于 09:51:12）")
    ck("09:25:10" not in h_late and h_late.count("今开") >= 2,
       "抢救日邮件抬头和页脚都不印 09:25:10")
    h_norm = M.build_html("2026-09-04", res, {}, "自测",
                          collected=c["runtime"]["snapshot_t3"])
    ck(c["runtime"]["snapshot_t3"] in h_norm, "正常日采集时刻从 config/run_meta 取")
    hard = (_literals("src/mailer.py", "09:25:10")
            + _literals("src/ths_export.py", "09:25:10"))
    ck(not hard, f"mailer / ths_export 里不再有写死的采集时刻：{hard}")

    # 3) 抢救日的准入行：quick 按 0~1e9 筛的，面板不能印「量比 2.5~10」
    sc_late = dict(c["screen"])
    sc_late["auc_ratio_min"], sc_late["auc_ratio_max"] = 0.0, 1e9
    cl = _criteria_line(sc_late, late=True)
    ck("已放开" in cl and "1e+09" not in cl and "2.5~10" not in cl,
       f"抢救日准入行：{cl}")
    ck("已放开" not in _criteria_line(c["screen"]),
       "正常日准入行照常渲染区间")
    p_late = write_ths_panel(sel, {}, tmp, tiers, "2026-08-28", "抢救",
                             sc_late, [], collected="今开（抢救，采样于 09:51:12）",
                             late=True)
    t_late = p_late.read_text(encoding="utf-8")
    ck("09:25:10" not in t_late and "今开" in t_late and "已放开" in t_late,
       "抢救日面板不印 09:25:10、不印正常量比区间")

    # 4) 板块成员数的显示阈值绑 config（以前写死 3，members=2 拿了 3.75 分却不显示）
    one2 = {"A": [score_one(mk(code="300750", name="乙", sector_members=2), c)],
            "B": []}
    h2 = M.build_html("2026-09-04", one2, {}, "自测", collected="09:25:10",
                      screen={"sector_min_members": 2})
    h3 = M.build_html("2026-09-04", one2, {}, "自测", collected="09:25:10",
                      screen={"sector_min_members": 3})
    ck("半导体·2" in h2 and "半导体·2" not in h3,
       "邮件板块列的显示阈值读 config（2 时显示 ·2，3 时不显示）")
    ck(">= 3" not in inspect.getsource(M._rows_html),
       "_rows_html 里不再有写死的 3")
    sc2 = dict(c["screen"])
    sc2["sector_min_members"] = 2
    p2 = write_ths_panel(one2["A"], {}, tmp, tiers, "2026-09-04", "", sc2, [],
                         collected="09:25:10")
    ck("半导体·2" in p2.read_text(encoding="utf-8"),
       "面板板块列同一份阈值")

    # 5) 面板过期横幅：收盘后的两条线日期本来就落后今天
    ck("if(!LAGOK && PDATE!==t" in REFRESH_JS, "横幅条件受 LAGOK 门控")
    t_norm = write_ths_panel(sel, {}, tmp, tiers, "2026-09-04", "",
                             c["screen"], [],
                             collected="09:25:10").read_text(encoding="utf-8")
    ck("LAGOK=('false'==='true')" in t_norm and "__LAGOK__" not in t_norm,
       "竞价面板 LAGOK=false（它的日期就是今天），占位符已替换")
    t_pb = PE.write_panel([], {}, tmp, "2026-09-04").read_text(encoding="utf-8")
    ck("LAGOK=('true'==='true')" in t_pb and "__LAGOK__" not in t_pb,
       "形态面板 LAGOK=true（日期本来就是最近已收盘交易日，不挂过期横幅）")

    # 6) enrich 的透传接线
    import run_auction as RA
    esrc = inspect.getsource(RA.stage_enrich)
    ck("collected=collected" in esrc and "late=late" in esrc,
       "stage_enrich 把采集时刻和抢救标记传给面板")
    ck('"captured_at"' in inspect.getsource(RA.write_prompt),
       "run_meta.json 里有 captured_at，enrich 才读得到")
    ck("salvage_screen" in esrc,
       "enrich 用和 quick 同一份抢救口径（两个进程各自 cfg()）")

    # 7) F5-12：notice 是直接插值进 HTML 的，中间没有 markdown -> HTML 转换。
    #    写 `**已停用**` 在邮件里就是两个星号
    from run_auction import LATE_NOTICE
    h_ln = M.build_html("2026-08-28", res, {}, "自测", LATE_NOTICE,
                        collected="今开（抢救，采样于 09:51:12）")
    ck("**" not in h_ln and "<b>已停用</b>" in h_ln,
       "抢救 notice 在邮件里是真加粗，不是两个星号")
    p_ln = write_ths_panel(sel, {}, tmp, tiers, "2026-08-28", LATE_NOTICE,
                           sc_late, [], collected="今开（抢救）", late=True)
    ck("**" not in p_ln.read_text(encoding="utf-8"),
       "面板副标题同样没有 markdown 星号")

    # 形态邮件的「本次无 LLM 分析」以前用 class="notice"，而 mailer._CSS 里
    # 根本没这个类：和正文同色、没边框，硬约束 2 要的顶部声明看不见
    hp = PE.build_html("2026-09-04", [], {}, "本次无 LLM 分析", "", {})
    cls = set(re.findall(r'class=["\']([a-z]+)["\']', hp))
    missing = sorted(x for x in cls if f".{x}{{" not in M._CSS)
    ck(not missing, f"形态邮件用到的 class 都在 mailer._CSS 里（缺：{missing}）")
    ck("本次无 LLM 分析" in hp and "**" not in hp, "声明还在，且没有星号")

    # 8) F5-13：LLM 文案和名称都是外部输入（analyst.md 让模型用 WebSearch），
    #    不转义的话一句 `</td><td>` 就把整张表挪位，`<img onerror=>` 会原样
    #    发布到公开的 Pages
    code0 = sel[0]["code"]
    bad_txt = {code0: {"reason": "<b>x</b>&y", "risk": "</td><td>z"}}
    hx = M.build_html("2026-09-16", res, bad_txt, "自测", collected="09:25:10")
    ck("&lt;b&gt;x&lt;/b&gt;&amp;y" in hx and "<b>x</b>" not in hx
       and "</td><td>z" not in hx, "邮件把 LLM 文案转义后再拼")
    trs = [x for x in hx.split("<tr>") if "<td" in x]
    n_td = sorted({x.count("<td") for x in trs})
    ck(n_td == [len(M._HDR)],
       f"每行 td 数都等于表头 {len(M._HDR)} 列（拿到 {n_td}）")
    px = write_ths_panel(sel, bad_txt, tmp, tiers, "2026-09-16", "",
                         c["screen"], [], collected="09:25:10"
                         ).read_text(encoding="utf-8")
    ck("&lt;b&gt;x&lt;/b&gt;" in px and "<b>x</b>" not in px
       and "</td><td>z" not in px, "面板同样转义（它会发布到公开的 Pages）")
    evil = score_one(mk(code="600114", name="<img src=x onerror=1>"), c)
    h_nm = M.build_html("2026-09-16", {"A": [evil], "B": []}, {}, "自测",
                        collected="09:25:10")
    ck("<img" not in h_nm and "&lt;img" in h_nm,
       "名称（来自行情源）也转义")
    sh_evil = [{"code": "600000", "name": "<i>影子</i>", "gap_pct": 3.1,
                "liangbi": 4.2, "sscore": 0.5}]
    h_sh = M.build_html("2026-09-16", res, {}, "自测", shadow_rows=sh_evil,
                        collected="09:25:10")
    ck("<i>影子</i>" not in h_sh and "&lt;i&gt;影子&lt;/i&gt;" in h_sh,
       "影子榜的名称也转义")

    # 形态线两个渲染点同样
    tmp2 = Path(tempfile.mkdtemp(prefix="selftest_exp2_"))
    prow = {"code": "600111", "name": "<b>甲</b>", "close": 10.0,
            "gain_pct": 6.0, "vol_ratio": 1.8, "turnover": 7.0,
            "launch_date": "2026-09-01", "launch_gain": 8.0,
            "adjust_days": 3, "adjust_vol_mean_ratio": 0.5,
            "adjust_drawdown_pct": -3.0, "score": 70.0,
            "break_launch_high": False}
    ptx = {"600111": {"reason": "<i>r</i>", "risk": "</td><td>q"}}
    hb = PE.build_html("2026-09-04", [prow], ptx, "", "", {})
    pb2 = PE.write_panel([prow], ptx, tmp2, "2026-09-04").read_text(
        encoding="utf-8")
    for label, txt in (("邮件", hb), ("面板", pb2)):
        ck("&lt;i&gt;r&lt;/i&gt;" in txt and "<i>r</i>" not in txt
           and "</td><td>q" not in txt and "&lt;b&gt;甲&lt;/b&gt;" in txt,
           f"形态{label}把 LLM 文案和名称都转义了")
    return get()


def check_vscore_twin(c: dict) -> int:
    """孪生体在**缺失值**和 dtype 上也要和 score.py 一致（M6 / M7 / M8）。

    硬约束 9 那条 2000 样本的等价性断言抽的是有限值、干净 dtype，
    下面三件事全在它的盲区里：
      · slope=NaN：标量侧 min(1.0, nan) 因参数顺序返回 1.0（趋势满分 = 20 分），
        向量侧 np.clip(nan) 是 NaN（整天目标函数作废），两边都不剔除
      · 数值列以 object 到达：prepare 按 dtype 分流，会 astype(bool)，
        gap_pct 变 True 之后 `True >= 2.0` 恒 False，整天 100% 被剔除
      · min_auc_amount_wan 两边同时 .get(…, 0)：等价性照样全绿，
        而准入线被静默关掉，回测池比生产池多 8.2% 的行（实测 44/534）
    """
    import numpy as np
    import pandas as pd
    from dataclasses import asdict
    from learn import vscore as V
    ck, get = _counter()
    print("\n向量化孪生体的缺失值与 dtype（M6/M7/M8）")

    feats = [
        mk(code="600111", name="正常"),
        mk(code="600112", name="斜率缺失", slope=float("nan"),
           monotonic=False),
        mk(code="600113", name="高开缺失", gap_pct=float("nan"),
           gap_norm=float("nan"), t3_chg=float("nan")),
        mk(code="600114", name="量能缺失", auc_ratio=float("nan")),
        mk(code="600115", name="竞价额不足", auc_amount=1.27e6,
           prev_amount=8.0e7, auc_ratio=0.0159),
    ]
    rows = [score_one(f, c) for f in feats]
    df = pd.DataFrame([asdict(f) for f in feats])
    s_vec, rej_vec = V.score_df(df, c)

    ck(rows[1]["rejected"] is None and abs(rows[1]["parts"]["trend"] - 0.5) < 1e-9,
       f"slope=NaN 的票趋势分 0.5 不是 1.0（拿到 {rows[1]['parts']['trend']}）")
    ck(rows[2]["rejected"] is not None and rows[4]["rejected"] is not None,
       "gap=NaN 和竞价额不足两行在标量侧被剔除")
    ck(bool(np.isfinite(s_vec).all()),
       f"向量侧总分无 NaN（含 slope/gap/量能缺失行）：{s_vec}")
    same_rej = [bool(rej_vec[i]) == (rows[i]["rejected"] is not None)
                for i in range(len(rows))]
    ck(all(same_rej), f"剔除判定逐行一致（不一致行：{[i for i, x in enumerate(same_rej) if not x]}）")
    diffs = [abs(float(s_vec[i]) - rows[i]["score_raw"]) for i in range(len(rows))]
    ck(max(diffs) < 1e-9, f"总分逐位一致，最大差 {max(diffs):.3g}")

    # M7：prepare 认列名不认 dtype
    num = [k for k in V.NEEDED if k not in V._BOOL_COLS]
    dfo = df.copy()
    for k in num:
        dfo[k] = dfo[k].astype(object)
    dfo.loc[dfo.index[0], "pos_pct_60d"] = None      # 真实混入路径：一个 None
    d = V.prepare(dfo)
    ck(all(d[k].dtype == np.float64 for k in num),
       "数值列以 object 到达仍是 float64（以前整列变 bool）")
    ck(bool(np.isnan(d["pos_pct_60d"][0])),
       "object 里的 None 变 NaN 而不是 False")
    s1, r1 = V.score_df(dfo, c)
    ck(np.array_equal(r1[1:], rej_vec[1:]) and np.allclose(s1[1:], s_vec[1:]),
       "object 数值列的分数与剔除和 float 列逐位一致")

    # M8：缺键必须响，两边一样
    d0 = V.prepare(df)
    sc_nokey = dict(c["screen"])
    sc_nokey.pop("min_auc_amount_wan")
    try:
        V.hard_reject(d0, sc_nokey)
        ck(False, "vscore.hard_reject 缺 min_auc_amount_wan 时静默放行了")
    except KeyError:
        ck(True, "向量化剔除对缺键和 score.py 一样是响的")
    ck(bool(V.hard_reject(d0, c["screen"])[4]),
       "竞价额 127 万那行在向量化侧也真的被这条下限剔除")
    return get()


def check_salvage_whitelist(c: dict) -> int:
    """抢救模式的权重分摊 + learned.yaml 白名单（M17）。"""
    import copy
    import tempfile
    import cfg as C
    import run_auction as RA
    from score import PART_KEYS
    ck, get = _counter()
    print("\n抢救分摊与 learned.yaml 白名单（M17）")

    ck(tuple(score_one(mk(), c)["parts"]) == PART_KEYS,
       "score_one 的 parts 就是 PART_KEYS（分摊口径和打分口径是同一组键）")

    c1 = copy.deepcopy(c)
    c1["scoring"]["weights"]["foo"] = 0.30      # 手改 learned.yaml 能塞进来的死键
    RA.salvage_screen(c1)
    w = c1["scoring"]["weights"]
    real = sum(w[k] for k in PART_KEYS)
    ck(abs(real - 1.0) < 1e-9,
       f"抢救分摊后六个真实维度之和仍是 1.0（拿到 {real:.4f}，旧写法 0.9286）")
    ck(w["volume"] == 0.0 and w["foo"] == 0.30,
       "量能权重清零，死键既不被分摊也不被改")

    c2 = copy.deepcopy(c)
    RA.salvage_screen(c2)
    w2 = c2["scoring"]["weights"]
    ck(abs(sum(w2[k] for k in PART_KEYS) - 1.0) < 1e-9 and w2["volume"] == 0.0,
       "没有死键时分摊结果不变")
    ck(c2["screen"]["auc_ratio_min"] == 0.0
       and c2["screen"]["auc_ratio_max"] == 1e9,
       "抢救口径把量能准入区间放开（面板据此印「已放开」）")

    known = frozenset(C._flat(C.base()))
    for k in ("screen.gap_pct_peakXYZ", "screen.auc_ratio_decay_x",
              "scoring.weights.foo", "screen.gap_pct_min",
              "runtime.send_at"):
        ck(not C._allowed(k, known), f"白名单不认 {k}")
    for k in ("scoring.weights.gap", "screen.gap_pct_peak",
              "screen.auc_ratio_score_hi", "screen.auc_ratio_decay"):
        ck(C._allowed(k, known), f"白名单认 {k}")

    # 越界键整份忽略。learned.yaml 写临时目录，不碰 state/（硬约束 8、教训 17）
    tmp = Path(tempfile.mkdtemp(prefix="selftest_cfg_"))
    orig = C.LEARNED
    try:
        p = tmp / "learned.yaml"
        C.LEARNED = p
        p.write_text("params:\n  scoring.weights.foo: 0.3\n"
                     "  screen.gap_pct_peakXYZ: 9.9\n", encoding="utf-8")
        ck(C.learned() == {}, "含越界键的 learned.yaml 整份忽略")
        ck(set(C.load()["scoring"]["weights"]) == set(PART_KEYS),
           "合并后权重仍只有六个维度（死键进不来）")
        ck(C.diff() == [], "cfg.diff() 不再列出假变更")
        p.write_text("params:\n  scoring.weights.gap: 0.31\n", encoding="utf-8")
        ck(C.learned() == {"scoring.weights.gap": 0.31}, "合法键照常生效")
        ck(abs(C.load()["scoring"]["weights"]["gap"] - 0.31) < 1e-12,
           "合法键真的合并进配置")
    finally:
        C.LEARNED = orig
    return get()


def main() -> int:
    c = cfg()
    t0 = time.time()
    print(f"{'用例':<14}{'代码':<9}{'分':>6}  {'组':<3}判定")
    print("-" * 62)

    rows, bad = [], 0
    for label, feat, expect in CASES:
        r = score_one(feat, c)
        rows.append(r)
        got = r["rejected"] or "通过"
        ok = (expect is None and r["rejected"] is None) or \
             (expect is not None and r["rejected"] and expect in r["rejected"])
        if not ok:
            bad += 1
        print(f"{label:<14}{r['code']:<9}{r['score']:>6.1f}  "
              f"{r['group']:<3}{'✓' if ok else '✗'} {got}")

    bad += check_curves(c)
    bad += check_rules(c)
    bad += check_misc(c)
    bad += check_pool(c)
    bad += check_stage2_listing(c)
    bad += check_monotonic(c)
    bad += check_premarket_stage2(c)
    bad += check_hist_units(c)
    bad += check_calendar(c)
    bad += check_spot_all(c)
    bad += check_codes(c)
    bad += check_em_circuit(c)
    bad += check_exports(c)
    bad += check_vscore_twin(c)
    bad += check_salvage_whitelist(c)

    # 压力：1000 只随机票，确认打分不发散、不抛异常、不卡住
    random.seed(7)
    stress = []
    for i in range(1000):
        lp = random.choice([10.0, 20.0])
        gn = random.uniform(-0.2, 0.9)
        stress.append(mk(
            code=f"{600000+i}", name=f"S{i}", limit_pct=lp,
            gap_norm=gn, gap_pct=gn * lp,
            t1_chg=gn * lp - random.uniform(-2, 2),
            t2_chg=gn * lp - random.uniform(-1, 1), t3_chg=gn * lp,
            slope=random.uniform(-3, 3), dive=random.uniform(-2, 4),
            auc_ratio=random.uniform(0.001, 0.2),
            pos_pct_60d=random.random(),
            sector=random.choice(["半导体", "光模块", "军工", "券商", "锂电"]),
            sector_members=random.randint(0, 8),
            monotonic=random.random() > 0.5,
            prev_limit_up=random.random() > 0.6,
            board_height=random.choice([0, 0, 1, 2, 3]),
        ))
    st_rows = [score_one(f, c) for f in stress]
    scores = [r["score"] for r in st_rows]
    assert all(0 <= s <= 100 for s in scores), "分数越界"

    res = rank(rows + st_rows, c)
    print("-" * 62)
    print(f"压力样本 1000 只 | 分数 min={min(scores):.1f} "
          f"max={max(scores):.1f} | 通过 {len(res['all'])} 只")
    print(f"输出 A组 {len(res['A'])} / B组 {len(res['B'])}"
          f"（上限 {c['output']['top_n_a']}/{c['output']['top_n_b']}）")

    assert len(res["A"]) <= c["output"]["top_n_a"]
    assert len(res["B"]) <= c["output"]["top_n_b"]
    assert all(res["A"][i]["score"] >= res["A"][i + 1]["score"]
               for i in range(len(res["A"]) - 1)), "A组未按分降序"

    # 下游产物。**一律写临时目录，不许碰 out/。**
    # 自测用的是编造的代码（历史教训 2），而 out/ 是当天的生产产物：
    # watchlist.ebk 是给同花顺导入的自选股，out/*.txt 会被 build_site.py
    # 发布到 Pages。跑一次自测就把当天真实的榜换成假数据，人在手机上
    # 点开、或者导进同花顺，拿到的是 17 个测试用例里编出来的票。
    # panel.html 一直有这个保护（见下面影子榜那段），preview.html、
    # watchlist.ebk、外部数据.txt 这三个漏了。
    import tempfile
    OUT = Path(tempfile.mkdtemp(prefix="selftest_out_"))
    sel = res["A"] + res["B"]
    paths = write_tdx_custom(sel, OUT)
    html = build_html(dt.date.today().isoformat(), res, {}, "自测")
    (OUT / "preview.html").write_text(html, encoding="utf-8")

    # 影子参考榜：邮件和面板都要能带上第二个榜，也要能不带（影子失败时）。
    # 面板写到临时目录，不许碰 out/panel.html（那是当天的生产产物）。
    import tempfile
    from ths_export import write_ths_panel
    shadow_rows = [{"code": "600000", "name": "影子甲", "gap_pct": 3.1,
                    "liangbi": 4.2, "sscore": 0.512},
                   {"code": "300750", "name": "影子乙", "gap_pct": 2.4,
                    "liangbi": 6.0, "sscore": -0.08}]
    h2 = build_html("2026-09-04", res, {}, "自测", shadow_rows=shadow_rows)
    assert "影子参考榜" in h2 and "600000" in h2 and "影子乙" in h2, "邮件缺影子榜"
    assert "影子参考榜" not in build_html("2026-09-04", res, {}, "自测"), \
        "无影子行时邮件不该出现影子榜"
    tmp = Path(tempfile.mkdtemp(prefix="selftest_panel_"))
    pp = write_ths_panel(sel, {}, tmp, c["output"]["ths_tiers"], "2026-09-04",
                         "", c["screen"], shadow_rows)
    ptxt = pp.read_text(encoding="utf-8")
    assert "影子参考榜" in ptxt and "300750" in ptxt, "面板缺影子榜"
    assert "learn.html" in ptxt, "面板缺学习面板入口"
    pp2 = write_ths_panel(sel, {}, tmp, c["output"]["ths_tiers"], "2026-09-04",
                          "", c["screen"], [])
    assert "影子参考榜" not in pp2.read_text(encoding="utf-8"), \
        "无影子行时面板不该出现影子榜"

    print("\n通达信自定义数据前 3 行:")
    for line in paths[0].read_bytes().decode("gbk").splitlines()[:3]:
        print("   ", line)

    dur = time.time() - t0
    print(f"\n耗时 {dur:.2f}s | 断言失败 {bad} 个")
    assert dur < 20, "耗时异常，可能存在阻塞"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
