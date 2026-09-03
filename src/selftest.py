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
OUT = ROOT / "out"


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
    return bad


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

    # 下游产物
    OUT.mkdir(exist_ok=True)
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
