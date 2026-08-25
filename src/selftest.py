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
    ("高开过头",      mk(code="600555", gap_pct=7.0, gap_norm=0.70,
                        t1_chg=6.0, t2_chg=6.5, t3_chg=7.0),      "竞价涨幅"),
    ("量能不足",      mk(code="600666", auc_ratio=0.004),        "竞价量能"),
    ("量能过载",      mk(code="600777", auc_ratio=0.30),         "竞价量能"),
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
    ck(all(f_volume(r, vlo, vhi, sat, dec) > f_volume(r * 1.2, vlo, vhi, sat, dec)
           for r in [sat * 1.1 * 1.3 ** i for i in range(6)]
           if r * 1.2 <= vhi),
       "volume 超过饱和点后单调递减（越极端越警惕）")
    # 关键：衰减速率不能随上限漂移
    probe = sat * 2
    ck(abs(f_volume(probe, vlo, 0.08, sat, dec)
           - f_volume(probe, vlo, 0.30, sat, dec)) < 1e-9,
       "volume 衰减速率与 auc_ratio_max 无关")
    ck(abs(f_volume(0.0792, vlo, vhi, sat, dec) - 0.61) < 0.02,
       "volume 在量比 19 处仍是 0.61（与旧配置口径一致）")
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

    print("\n通达信自定义数据前 3 行:")
    for line in paths[0].read_bytes().decode("gbk").splitlines()[:3]:
        print("   ", line)

    dur = time.time() - t0
    print(f"\n耗时 {dur:.2f}s | 断言失败 {bad} 个")
    assert dur < 20, "耗时异常，可能存在阻塞"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
