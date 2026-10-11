"""
仓库审计：死代码、重复实现、文档漂移、接线漂移。零输出才算干净。

    python tools/audit_repo.py            全部检查，逐条打印，有问题退出码 1
    python tools/audit_repo.py --json     机器可读

查什么（计划 2026-10-10 第 1、2 项；归档目录 archive/ 一律跳过）：
  模块     .py 没被任何 import / 命令行 / 计划任务 / workflow / 控制台按钮引用
  函数     顶层函数 / 类在全仓库（代码 + 工作流 + 文档）只出现一次
  别名属性 `X.attr` 在被导入模块里不存在（pyflakes 抓不到这一类）
  重复实现 同名顶层函数在两个非归档文件里各有一份（教训 34）
  产物     out_* / state/ 下跟踪的文件没有任何代码读写
  文档     CLAUDE.md 文件地图提到但不存在的路径；跟踪的脚本没在文件地图里
  配置     config.yaml 的键没有代码读
  依赖     requirements*.txt 列了但没有一处 import 的包
  gitignore 规则指向的路径既不存在也没人写
  按钮     gui.jobs.ACTIONS 的 cmd 指向不存在的脚本
  工作流   .github/workflows 和 tools/run_local.cmd 调的脚本不存在
  流程表   gui.status.LINES 和 local_run.FLOWS 的键不一致

已知的例外写在 ALLOW 里，每条带理由；新例外要带理由加进去，不许静默放过。
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "breakout"))
sys.path.insert(0, str(ROOT / "tools"))

STDLIB_OR_THIRD = set(sys.stdlib_module_names) | {
    "np", "pd", "requests", "akshare", "lightgbm", "sklearn", "torch", "scipy", "yaml",
    "pyarrow", "matplotlib", "tqdm", "numpy", "pandas", "lgb", "ak"}
# 例外：(类别, 名字) -> 理由
ALLOW: dict[tuple[str, str], str] = {
    ("module", "src/breakout/exp_time.py"): "实验 15 复现脚本，手动跑",
    ("module", "src/breakout/exp_acc.py"): "实验 16 复现脚本，手动跑",
    ("module", "src/breakout/exp_w1.py"): "实验 17 复现脚本，手动跑",
    ("module", "src/breakout/exp_w2.py"): "实验 18 复现脚本，手动跑",
    ("module", "src/breakout/exp_cont.py"): "实验 19 复现脚本，手动跑",
    ("module", "tools/dump_st.py"): "手动刷新 cache/st_codes.json",
    ("module", "tools/audit_repo.py"): "本脚本",
    ("func", "selftest_gui.py"): "selftest 的 check_* 是入口调的",
    ("dup", "last_closed_trade_day"): "判定只有 bjtime.last_closed 一份；local_run / pullback / backfill 是各自取日历的薄包装",
    ("dup", "target_date"): "local_run 和控制台各一份薄包装，逻辑在 bjtime.last_closed；控制台不能 import 重模块",
    ("dup", "trade_dates"): "datasource 拉日历并写缓存，local_run 是带 12 小时缓存和退化的包装",
    ("dup", "walk_forward"): "validate 的是 2026-09-12 旧协议（比模型按钮用），evalkit 的是新协议",
    ("dup", "write_panel"): "两条线各自的面板，互不相干是设计（CLAUDE.md 文件地图）",
    ("dup", "send_mail"): "两条线各自的邮件，同上",
    ("dup", "board_of"): "features.board_of 在模型指纹里不能动；datasource.board_of 是唯一可改的那份，自测钉住两份逐只相同",
    ("requirement", "requirements.txt: py-mini-racer"): "akshare 新浪日线解密的运行时依赖，不直接 import",
}


def git_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                         text=True, encoding="utf-8").stdout.split("\n")
    return [ROOT / f for f in out if f and not f.startswith("archive/")]


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


class Audit:
    def __init__(self) -> None:
        self.files = git_files()
        self.py = [f for f in self.files if f.suffix == ".py"]
        self.text = {f: read(f) for f in self.files
                     if f.suffix in (".py", ".cmd", ".ps1", ".yml", ".yaml", ".vbs", ".js",
                                     ".md", ".toml", ".txt", ".json", ".html")
                     and f.stat().st_size < 3_000_000}
        self.code_text = {f: t for f, t in self.text.items()
                          if f.suffix not in (".md", ".json", ".html", ".txt")}
        self.issues: list[dict] = []

    def add(self, kind: str, where: str, what: str) -> None:
        if (kind, where) in ALLOW or (kind, Path(where).name) in ALLOW:
            return
        self.issues.append({"kind": kind, "where": where, "what": what})

    def rel(self, f: Path) -> str:
        return f.relative_to(ROOT).as_posix()

    # ---- 模块 ----
    def modules(self) -> None:
        for f in self.py:
            stem = f.stem
            if stem in ("__init__", "__main__"):
                continue
            pat = re.compile(r"(import\s+%s\b|from\s+[\w.]*\b%s\s+import|\b%s\.py\b|"
                             r"import\s+[\w, ]*\b%s\b|from\s+\.\s+import\s+[\w, ]*\b%s\b)"
                             % ((re.escape(stem),) * 5))
            if not any(g != f and pat.search(t) for g, t in self.code_text.items()):
                self.add("module", self.rel(f), "没有任何引用")

    # ---- 函数 ----
    def functions(self) -> None:
        for f in self.py:
            try:
                tree = ast.parse(self.text[f])
            except SyntaxError as e:
                self.add("syntax", self.rel(f), str(e))
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    n = node.name
                    if n == "main" or (n.startswith("check_") and "selftest" in f.name):
                        continue
                    pat = re.compile(r"\b%s\b" % re.escape(n))
                    cnt = sum(len(pat.findall(t)) for t in self.text.values())
                    if cnt <= 1:
                        self.add("func", f"{self.rel(f)}:{node.lineno}", f"{n} 全仓库只出现一次")

    # ---- 别名属性 ----
    def attrs(self) -> None:
        for f in self.py:
            try:
                tree = ast.parse(self.text[f])
            except SyntaxError:
                continue
            # 同一个别名在不同函数里可能指向不同模块（自测里 B 既是 backfill 也是 build），
            # 所以按函数作用域各算各的
            scopes = [tree] + [n for n in ast.walk(tree)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for scope in scopes:
                alias = {}
                body = scope.body if hasattr(scope, "body") else []
                for n in body:
                    if isinstance(n, ast.Import):
                        for a in n.names:
                            alias[a.asname or a.name.split(".")[0]] = a.name
                if not alias:
                    continue
                for n in ast.walk(scope):
                    if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id in alias):
                        mod = alias[n.value.id]
                        if mod.split(".")[0] in STDLIB_OR_THIRD:
                            continue
                        try:
                            m = importlib.import_module(mod)
                        except Exception:  # noqa: BLE001
                            continue
                        if not hasattr(m, n.attr):
                            self.add("attr", f"{self.rel(f)}:{n.lineno}",
                                     f"{n.value.id}.{n.attr} 在 {mod} 里不存在")

    # ---- 重复实现 ----
    def duplicates(self) -> None:
        seen: dict[str, list[str]] = {}
        for f in self.py:
            if "selftest" in f.name or f.name.startswith("exp_"):
                continue
            try:
                tree = ast.parse(self.text[f])
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") \
                        and node.name not in ("main", "load", "save", "compute", "run", "fmt",
                                              "send", "render", "build", "serve", "fit",
                                              "predict", "record", "parse_args", "stage_scan",
                                              "stage_send", "status", "overview"):
                    seen.setdefault(node.name, []).append(self.rel(f))
        for n, fs in seen.items():
            if len(set(fs)) > 1:
                self.add("dup", n, "同名顶层函数在 " + " / ".join(sorted(set(fs))))

    # ---- 产物 ----
    def products(self) -> None:
        for f in self.files:
            r = self.rel(f)
            if not (r.startswith("out_") or r.startswith("state/")):
                continue
            if "/sent/" in r or "/claim/" in r or "/alert/" in r:
                continue
            name = f.name
            if not any(name in t for g, t in self.code_text.items() if g != f):
                self.add("product", r, "没有任何代码读写它")

    # ---- 文档 ----
    def docs(self) -> None:
        cl_all = self.text.get(ROOT / "CLAUDE.md", "")
        # 只扫「文件地图」那段围栏：教训里提到的历史路径（select.py）和归档路径不算
        m_ = re.search(r"## 文件地图.*?```(.*?)```", cl_all, re.S)
        cl = m_.group(1) if m_ else cl_all
        for m in sorted(set(re.findall(r"(?:src|tools|docs|\.github/workflows)/[A-Za-z0-9_/.\-]+\.(?:py|ps1|cmd|vbs|yml|md)", cl))):
            if not (ROOT / m).exists():
                self.add("doc", f"CLAUDE.md -> {m}", "文件地图提到的路径不存在")
        for m in sorted(set(re.findall(r"breakout/[a-z_]+\.py", cl))):
            if not (ROOT / "src" / m).exists():
                self.add("doc", f"CLAUDE.md -> src/{m}", "文件地图提到的路径不存在")
        for f in self.files:
            r = self.rel(f)
            if not (r.startswith("src/") or r.startswith("tools/") or r.startswith(".github/")):
                continue
            if f.name in ("__init__.py", "__main__.py"):
                continue
            if f.name not in cl_all and f.stem not in cl_all:
                self.add("doc", r, "CLAUDE.md 文件地图没提到")

    # ---- 配置 ----
    def config(self) -> None:
        try:
            import yaml
            c = yaml.safe_load(read(ROOT / "config.yaml"))
        except Exception as e:  # noqa: BLE001
            self.add("config", "config.yaml", f"读不出来 {e}")
            return
        src = "\n".join(self.code_text.values())

        def walk(d, path):
            for k, v in d.items():
                if not re.search(r"[\"']%s[\"']" % re.escape(str(k)), src):
                    self.add("config", ".".join(path + [str(k)]), "没有代码读这个键")
                if isinstance(v, dict):
                    walk(v, path + [str(k)])
        walk(c, [])

    # ---- 依赖 ----
    def requirements(self) -> None:
        alias = {"scikit-learn": "sklearn", "pyyaml": "yaml", "beautifulsoup4": "bs4",
                 "pillow": "PIL", "python-dateutil": "dateutil", "py-mini-racer": "py_mini_racer"}
        src = "\n".join(t for f, t in self.text.items() if f.suffix == ".py")
        for rf in ("requirements.txt", "requirements-breakout.txt"):
            for line in read(ROOT / rf).splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith("-"):
                    continue
                pkg = re.split(r"[<>=!~\[ ]", line)[0].strip().lower()
                mod = alias.get(pkg, pkg.replace("-", "_"))
                if not re.search(r"\b(import|from)\s+%s\b" % re.escape(mod), src):
                    self.add("requirement", f"{rf}: {pkg}", "没有一处 import")

    # ---- gitignore ----
    def gitignore(self) -> None:
        src = "\n".join(self.code_text.values())
        for line in read(ROOT / ".gitignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!") or "*" in line:
                continue
            p = line.rstrip("/")
            if p.startswith("archive/"):
                continue
            if "/" not in p:
                continue                      # __pycache__/ node_modules/ 这类裸目录名，随处可能出现
            if not (ROOT / p).exists() and Path(p).name not in src:
                self.add("gitignore", line, "路径不存在，代码里也没人写它")

    # ---- 按钮 / 工作流 / 流程表 ----
    def wiring(self) -> None:
        try:
            from gui.jobs import ACTIONS
            for k, a in ACTIONS.items():
                if not (ROOT / a["cmd"][0]).exists():
                    self.add("button", k, f"cmd 指向不存在的 {a['cmd'][0]}")
        except Exception as e:  # noqa: BLE001
            self.add("button", "gui.jobs", f"读不出 ACTIONS: {e}")
        for f in self.files:
            if f.suffix in (".yml", ".cmd", ".vbs", ".ps1") and (
                    "workflows" in f.parts or f.parent.name == "tools"):
                for m in re.findall(r"(?:src|tools)[/\\][A-Za-z0-9_/\\.\-]+\.(?:py|cmd|vbs)", self.text[f]):
                    if not (ROOT / m.replace("\\", "/")).exists():
                        self.add("workflow", self.rel(f), f"调的 {m} 不存在")
        try:
            import local_run as L
            from gui.status import LINES
            a, b = set(L.FLOWS), {x["key"] for x in LINES} if isinstance(LINES, list) else set(LINES)
            if a != b:
                self.add("flows", "local_run.FLOWS vs gui.status.LINES", f"{sorted(a)} vs {sorted(b)}")
        except Exception as e:  # noqa: BLE001
            self.add("flows", "FLOWS/LINES", f"读不出来: {e}")

    def run(self) -> list[dict]:
        for step in (self.modules, self.functions, self.attrs, self.duplicates, self.products,
                     self.docs, self.config, self.requirements, self.gitignore, self.wiring):
            step()
        return self.issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    issues = Audit().run()
    if a.json:
        print(json.dumps(issues, ensure_ascii=False, indent=1))
    else:
        for i in issues:
            print(f"[{i['kind']:<11}] {i['where']}: {i['what']}")
        print(f"\n{len(issues)} 条" if issues else "干净")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
