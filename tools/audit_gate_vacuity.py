#!/usr/bin/env python3
"""Gate-vacuity audit for the TensorDG acceptance gates.

Motivation: most TensorDG acceptance gates re-adjudicate a *committed artifact*
(``validation/*/….json``, ``runs/*/metrics.json``, ``artifacts/*.json`` …).  A
gate written as

    if not ARTIFACT.exists():
        pytest.skip("run benchmarks.foo to generate it")

is **green in CI without asserting anything** when the artifact is absent.
That failure mode is live in this ecosystem: the merge gate filters files out
of agent branches (commit messages read "N 个文件被门禁滤除"), so an artifact
present on the author's branch can be missing on ``main``; the gate then
degrades from "assert the numbers" to "assert nothing" and nobody sees red.

This tool is a *static* census of that failure mode.  For every test module it
resolves the artifact paths that the gates pin and classifies the module:

``VACUOUS``    a missing-artifact skip exists and its artifact is absent today
               -> the acceptance assertions of that module do not run.
``LATENT``     a missing-artifact skip exists but the artifact is committed
               -> the skip is a dead safety net; one merge-gate filter away
               from becoming VACUOUS.
``HARDENED``   no missing-artifact skip; the artifact is hard-asserted
               (``assert X.exists()``) -> absent artifact = red CI.
``UNRESOLVED`` a missing-artifact skip exists whose path the scanner cannot
               resolve statically (honest: not adjudicated, not "passed").
``PLAIN``      no committed-artifact dependency at all.

It also flags **cwd-dependent** artifacts (``Path("validation/…")`` instead of
``Path(__file__).resolve().parents[1] / …``): such a gate silently skips when
pytest is started from anywhere but the repository root.

Usage::

    python tools/audit_gate_vacuity.py             # census to stdout
    python tools/audit_gate_vacuity.py --write     # + JSON / Markdown
    python tools/audit_gate_vacuity.py --check     # non-zero if drifted
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
JSON_OUT = ROOT / "artifacts" / "gate_vacuity" / "gate_vacuity.json"
MD_OUT = ROOT / "docs" / "GATE-VACUITY-AUDIT.md"

SKIP_ARTIFACT = "artifact_missing"
SKIP_ENV = "environment"

VERDICT_VACUOUS = "VACUOUS"
VERDICT_LATENT = "LATENT"
VERDICT_HARDENED = "HARDENED"
VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_PLAIN = "PLAIN"
VERDICTS = (VERDICT_VACUOUS, VERDICT_LATENT, VERDICT_HARDENED,
            VERDICT_UNRESOLVED, VERDICT_PLAIN)

# Names whose module-level binding is a file path worth adjudicating.
HINTS = ("path", "file", "artifact", "json", "csv", "metrics", "reference",
         "law", "evidence", "anchor", "table", "figure", "fig", "run", "out")

# Skips that are legitimately environment-dependent (never vacuity).
ENV_HINTS = ("checkout", "origin/main", "run_slow", "slow test")
# Skips that mean "the artifact has not been generated".
ARTIFACT_HINTS = ("run ", "generate", "generated", "missing", "artifact",
                  "not present", "write")

# Legacy vacuity that this mission could not close.  Each entry must carry a
# reason; the policy test fails on stale entries (self-cleaning).
ALLOWED_VACUOUS: dict[str, str] = {
    # Registered 2026-11-12 (m16367 vacuity census).  Both modules guard their
    # acceptance assertions with "artifact missing -> skip" and the artifact is
    # NOT in the tree, so their numbers are unverified on main today.  The fix
    # is to commit the generator output (artifacts/dg_advection_bench.json,
    # runs/shu_osher_case_pool/metrics.json); until then the vacuity is
    # declared here instead of being hidden behind a green skip.  The policy
    # test fails if either entry stops being vacuous (self-cleaning).
    "tests/test_dg_advection_bench.py":
        "artifacts/dg_advection_bench.json not committed (filtered by the merge "
        "gate); regenerate with tools/run_dg_advection_bench.py and drop this entry",
    # Registered 2026-11-12 (m16815).  Found already-live on main: the dynamic
    # run of m16606 (audit AUDIT-m16606-CASE-LAX-DG-R3-region-rule.md) produced
    # runs/lax_region_band/metrics.json, but that artifact did not survive the
    # merge gate, so the six R3-region-rule invariants in
    # tests/test_lax_region_band.py have been skipping since the merge while
    # main stayed green.  Regenerate with `python -m benchmarks.lax_region_band`
    # (243 s on 1 core) and delete this entry.
    "tests/test_lax_region_band.py":
        "runs/lax_region_band/metrics.json not in the tree (merge-gate filtered); "
        "regenerate with `python -m benchmarks.lax_region_band` and drop this entry",
    "tests/test_shu_osher_case_pool.py":
        "runs/shu_osher_case_pool/metrics.json not committed; regenerate with "
        "python -m benchmarks.shu_osher_case_pool and drop this entry",
}


class _Unknown(Exception):
    pass


# ---------------------------------------------------------------------------
# Path-expression evaluation (the small Path algebra the gate modules use)
# ---------------------------------------------------------------------------
def _eval(node: ast.AST, env: dict[str, object], here: Path) -> object:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id == "__file__":
            return here
        raise _Unknown(node.id)
    if isinstance(node, ast.Subscript):
        base = _eval(node.value, env, here)
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            if isinstance(base, list):
                return base[index.value]
        raise _Unknown("subscript")
    if isinstance(node, ast.Attribute):
        if node.attr == "parents":
            return list(Path(_eval(node.value, env, here)).parents)  # type: ignore[arg-type]
        base = _eval(node.value, env, here)
        if node.attr == "parent":
            return Path(base).parent  # type: ignore[arg-type]
        if node.attr == "resolve":
            return Path(base)  # type: ignore[arg-type]
        raise _Unknown(node.attr)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path" and len(node.args) == 1:
            return Path(_eval(node.args[0], env, here))  # type: ignore[arg-type]
        if isinstance(func, ast.Attribute) and func.attr == "resolve":
            return Path(_eval(func.value, env, here))  # type: ignore[arg-type]
        if (isinstance(func, ast.Attribute) and func.attr == "join"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"):
            parts = [_eval(a, env, here) for a in node.args]
            return Path(parts[0]).joinpath(*[str(p) for p in parts[1:]])  # type: ignore[arg-type]
        raise _Unknown("call")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return Path(_eval(node.left, env, here)).joinpath(  # type: ignore[arg-type]
            str(_eval(node.right, env, here)))
    raise _Unknown(type(node).__name__)


def _module_bindings(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Module-level single-name bindings, in source order (ROOT before ARTIFACT)."""
    out: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                out.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            out.append((node.target.id, node.value))
    return out


def _local_bindings(func: ast.AST) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            out[node.target.id] = node.value
    return out


# ---------------------------------------------------------------------------
# report records
# ---------------------------------------------------------------------------
@dataclass
class Artifact:
    expr: str
    path: str
    anchored: bool
    exists: bool


@dataclass
class SkipSite:
    line: int
    kind: str
    reason: str
    guard: str = ""
    artifact: str | None = None
    artifact_exists: bool | None = None


@dataclass
class ModuleReport:
    file: str
    verdict: str = VERDICT_PLAIN
    skip_sites: list[SkipSite] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    hard_asserts: list[str] = field(default_factory=list)
    vacuous_artifacts: list[str] = field(default_factory=list)
    unresolved_guards: list[str] = field(default_factory=list)
    cwd_dependent: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "verdict": self.verdict,
            "skip_sites": [s.__dict__ for s in self.skip_sites],
            "artifacts": [a.__dict__ for a in self.artifacts],
            "hard_asserts": self.hard_asserts,
            "vacuous_artifacts": self.vacuous_artifacts,
            "unresolved_guards": self.unresolved_guards,
            "cwd_dependent": self.cwd_dependent,
        }


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
def _reason_of(call: ast.Call) -> str:
    for kw in call.keywords:
        if kw.arg == "reason":
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            return ast.unparse(value)
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return ast.unparse(first)
    return ""


def classify_reason(reason: str) -> str:
    """Artifact skip vs environment skip, from the *reason text* only."""
    low = reason.lower()
    if any(h in low for h in ENV_HINTS):
        return SKIP_ENV
    if any(h in low for h in ARTIFACT_HINTS):
        return SKIP_ARTIFACT
    return SKIP_ENV


_GUARD_RE = ".exists()"


def _guard_object(guard: str) -> str | None:
    """``not ARTIFACT.exists()`` -> ``ARTIFACT``; anything else -> None."""
    text = guard.strip()
    if text.startswith("not "):
        text = text[4:].strip()
    if text.endswith(_GUARD_RE):
        return text[: -len(_GUARD_RE)].strip()
    return None


def _resolve(expr: str, env: dict[str, object], here: Path) -> Path | None:
    try:
        node = ast.parse(expr, mode="eval").body
        value = _eval(node, env, here)
    except (_Unknown, SyntaxError):
        return None
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path)


def scan_module(path: Path) -> ModuleReport:
    path = Path(path).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:                      # synthetic fixture outside the tree
        rel = str(path)
    rep = ModuleReport(file=rel)

    env: dict[str, object] = {}
    for name, expr in _module_bindings(tree):
        try:
            env[name] = _eval(expr, env, path)
        except _Unknown:
            continue

    for name, value in env.items():
        if not isinstance(value, (Path, str)):
            continue
        if not any(h in name.lower() for h in HINTS):
            continue
        p = Path(value)
        if p.suffix not in (".json", ".csv", ".npz", ".png", ".npy", ".txt", ".md"):
            continue
        anchored = p.is_absolute()
        resolved = p if anchored else (Path.cwd() / p)
        rep.artifacts.append(Artifact(expr=name, path=str(resolved),
                                      anchored=anchored,
                                      exists=resolved.exists()))

    # -- hard asserts on artifacts -------------------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Call):
            fn = node.test.func
            if isinstance(fn, ast.Attribute) and fn.attr == "exists":
                rep.hard_asserts.append(ast.unparse(node.test))

    # -- function scopes, so that `metrics_path = ...` resolves --------------
    scopes: list[tuple[ast.AST, dict[str, object]]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local = dict(env)
            for name, expr in _local_bindings(node).items():
                try:
                    local[name] = _eval(expr, local, path)
                except _Unknown:
                    continue
            scopes.append((node, local))

    def scope_env(lineno: int) -> dict[str, object]:
        best: tuple[int, dict[str, object]] = (10 ** 9, env)
        for node, local in scopes:
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", 10 ** 9) or 10 ** 9
            if start <= lineno <= end and (end - start) < best[0]:
                best = ((end - start), local)
        return best[1]

    # -- skip sites -----------------------------------------------------------
    guarded: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    fn = stmt.value.func
                    if isinstance(fn, ast.Attribute) and fn.attr == "skip":
                        guarded[id(stmt.value)] = ast.unparse(node.test)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ("skip", "skipif")):
            continue
        reason = _reason_of(node)
        kind = classify_reason(reason)
        guard = ""
        if fn.attr == "skip":
            guard = guarded.get(id(node), "")
        elif node.args:
            guard = ast.unparse(node.args[0])
        site = SkipSite(line=node.lineno, kind=kind, reason=reason[:200],
                        guard=guard[:160])
        if kind == SKIP_ARTIFACT:
            obj = _guard_object(guard) if fn.attr == "skip" else None
            if obj is None and fn.attr == "skipif":
                # skipif(not ARTIFACT.exists(), reason=...) -> same shape
                cond = guard
                if cond.startswith("not ") and cond.endswith(".exists()"):
                    obj = cond[4:].strip()[: -len(".exists()")].strip()
            if obj is not None:
                resolved = _resolve(obj, scope_env(node.lineno), path)
                if resolved is not None:
                    site.artifact = str(resolved)
                    site.artifact_exists = resolved.exists()
                else:
                    rep.unresolved_guards.append(f"L{node.lineno}: {guard[:120]}")
            else:
                rep.unresolved_guards.append(f"L{node.lineno}: {guard[:120]}")
        rep.skip_sites.append(site)
    rep.skip_sites.sort(key=lambda s: s.line)

    # -- verdict --------------------------------------------------------------
    art_skips = [s for s in rep.skip_sites if s.kind == SKIP_ARTIFACT]
    resolved_missing = [s.artifact for s in art_skips
                        if s.artifact is not None and s.artifact_exists is False]
    if art_skips:
        if resolved_missing:
            rep.verdict = VERDICT_VACUOUS
            rep.vacuous_artifacts = sorted(set(resolved_missing))
        elif all(s.artifact_exists for s in art_skips):
            rep.verdict = VERDICT_LATENT
        else:
            rep.verdict = VERDICT_UNRESOLVED
    elif rep.hard_asserts:
        rep.verdict = VERDICT_HARDENED
    else:
        rep.verdict = VERDICT_PLAIN

    rep.cwd_dependent = sorted(a.expr for a in rep.artifacts if not a.anchored)
    return rep


def census() -> dict:
    modules = [scan_module(p) for p in sorted(TESTS.glob("test_*.py"))]
    counts = {v: 0 for v in VERDICTS}
    for m in modules:
        counts[m.verdict] += 1
    skip_counts = {SKIP_ARTIFACT: 0, SKIP_ENV: 0}
    for m in modules:
        for s in m.skip_sites:
            skip_counts[s.kind] += 1
    missing_artifacts = sorted({s.artifact for m in modules for s in m.skip_sites
                                if s.artifact and s.artifact_exists is False})
    ever_green = sum(1 for m in modules
                     if m.verdict in (VERDICT_PLAIN, VERDICT_HARDENED)
                     or (m.verdict == VERDICT_LATENT))
    return {
        "tool": "tools/audit_gate_vacuity.py",
        "repo_root": str(ROOT),
        "n_test_modules": len(modules),
        "verdicts": counts,
        "skip_sites": skip_counts,
        "n_always_on_modules": ever_green,
        "vacuous_modules": sorted(m.file for m in modules
                                  if m.verdict == VERDICT_VACUOUS),
        "missing_artifacts": missing_artifacts,
        "cwd_dependent_artifacts": sorted(
            f"{m.file}::{a}" for m in modules for a in m.cwd_dependent),
        "modules": [m.as_dict() for m in modules],
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_markdown(data: dict) -> str:
    v, s = data["verdicts"], data["skip_sites"]
    lines = [
        "# TensorDG 验收门「空转」审计（gate vacuity）",
        "",
        "> 生成器 `tools/audit_gate_vacuity.py`（`--check` 防漂移）·"
        " 机器门 `tests/test_gate_vacuity_policy.py`",
        "> 扫描面：`tests/test_*.py` 全量静态分析（零算力）",
        "",
        "## 判定口径",
        "",
        "| 判级 | 含义 |",
        "|---|---|",
        "| VACUOUS | 「工件缺失即 skip」守护验收断言，且该工件**此刻不在树里** → CI 绿但零断言 |",
        "| LATENT | 有缺失即 skip 的写法，但工件已提交 → 该 skip 是永不触发的安全网，离 VACUOUS 只差一次合并门过滤 |",
        "| HARDENED | 用 `assert … .exists()` 硬断言工件，缺失即红 |",
        "| UNRESOLVED | 有工件类 skip，但路径静态不可解析（诚实登记：未裁决，不算通过） |",
        "| PLAIN | 不依赖已提交工件 |",
        "",
        "## 统计",
        "",
        f"- 扫描测试模块 **{data['n_test_modules']}**："
        + " · ".join(f"{k} **{v[k]}**" for k in VERDICTS),
        f"- skip 位点：工件类 **{s[SKIP_ARTIFACT]}** · 环境类 **{s[SKIP_ENV]}**",
        f"- 缺失工件（skip 会真的触发）：**{len(data['missing_artifacts'])}**",
        f"- cwd 依赖（相对路径）工件常量：**{len(data['cwd_dependent_artifacts'])}**",
        "",
        "## 逐模块清单（非 PLAIN）",
        "",
        "| 模块 | 判级 | 工件类 skip | 硬断言 | cwd 依赖 |",
        "|---|---|---|---|---|",
    ]
    for m in data["modules"]:
        if m["verdict"] == VERDICT_PLAIN:
            continue
        n_art = sum(1 for x in m["skip_sites"] if x["kind"] == SKIP_ARTIFACT)
        lines.append(f"| `{m['file']}` | {m['verdict']} | {n_art} | "
                     f"{len(m['hard_asserts'])} | "
                     f"{', '.join(m['cwd_dependent']) or '—'} |")
    lines += ["", "## VACUOUS 明细（当下零验证）", ""]
    vac = [m for m in data["modules"] if m["verdict"] == VERDICT_VACUOUS]
    if not vac:
        lines.append("（无）")
    else:
        for m in vac:
            lines.append(f"### `{m['file']}`")
            for a in m["vacuous_artifacts"]:
                lines.append(f"- 缺失工件：`{a}`")
            for x in m["skip_sites"]:
                if x["kind"] == SKIP_ARTIFACT:
                    lines.append(f"- L{x['line']}: `{x['guard']}` → "
                                 f"skip(\"{x['reason'][:100]}\")")
            lines.append("")
    lines += [
        "",
        "## 为什么这比「测试绿」重要",
        "",
        "本生态的合并门会过滤分支文件（提交里可见「N 个文件被门禁滤除」）。工件一旦被滤掉，"
        "`skip` 型验收门就从「断言数字」退化为「断言无」——CI 仍然绿。"
        "本审计把这种退化变成可数的数字，并由 `tests/test_gate_vacuity_policy.py` 阻止新增："
        "新出现的 VACUOUS 模块除非在 `ALLOWED_VACUOUS` 里带理由登记（且条目过期即红），否则测试直接红。",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write JSON + Markdown")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed JSON drifted from a fresh scan")
    args = ap.parse_args(argv)

    data = census()
    if args.write:
        JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
        MD_OUT.write_text(render_markdown(data))
        print(f"wrote {JSON_OUT.relative_to(ROOT)} and {MD_OUT.relative_to(ROOT)}")
    if args.check:
        if not JSON_OUT.exists():
            print("missing committed census; run --write", file=sys.stderr)
            return 1
        committed = json.loads(JSON_OUT.read_text())
        if committed["modules"] != data["modules"]:
            print("gate vacuity census drifted; re-run --write", file=sys.stderr)
            return 1
        print("gate vacuity census fresh")
        return 0
    if not args.write:
        print(json.dumps({k: data[k] for k in
                          ("n_test_modules", "verdicts", "skip_sites",
                           "vacuous_modules", "missing_artifacts",
                           "cwd_dependent_artifacts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())