import json
import sys

cat, head, out = sys.argv[1:4]
d = json.load(open(cat, encoding="utf-8"))
F = d["factors"]
L = [open(head, encoding="utf-8").read().rstrip("\n"), ""]
fams = sorted({f["family"] for f in F})
for fam in fams:
    fs = sorted([f for f in F if f["family"] == fam], key=lambda f: (f["rank"], f["name"]))
    L.append(f"【{fam}】{len(fs)} 条")
    for f in fs:
        rk = f"rank {f['rank']}" if f["rank"] <= 41 else "未排序"
        L.append(f"  {f['id']} {f['name']}（priority {f['priority']}，{rk}，窗口 {f['windows']}）")
        L.append(f"    公式：{f['formula']}")
        L.append(f"    含义：{f['rationale']}")
        L.append(f"    重叠：{f['overlap']}")
        L.append(f"    成本：{f['cost']}；偷看：{f['lookahead_risk']}")
        L.append(f"    出处：{f['source_url']}（{f['origin']}；{f['verified']}）")
        if f.get("notes"):
            L.append(f"    备注：{f['notes']}")
    L.append("")
L.append("五、总排序前 41（rank）")
for f in sorted([f for f in F if f["rank"] <= 41], key=lambda f: f["rank"]):
    L.append(f"  {f['rank']:>2}. {f['name']}  [{f['family']}]  priority {f['priority']}")
L.append("")
L.append("六、需要新数据才能算、值得存档的")
for i, n in enumerate(d["needs_new_data"], 1):
    L.append(f"  {i}. {n['name']}")
    L.append(f"     数据：{n['data']}")
    L.append(f"     理由：{n['why']}")
    if n.get("source_url"):
        L.append(f"     出处：{n['source_url']}")
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(len(L), "lines")
