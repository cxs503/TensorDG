#!/usr/bin/env python
"""TensorDG public API-surface audit (machine-readable, no hand copies).

Why this exists
---------------
The library grew module by module, and the package root ``tensordg/__init__.py``
grew only when somebody remembered to touch it.  The failure mode is the one
already found in this repo's audit trail: a merged capability (``elastic_1d``,
``swe_exact``, ``poisson_2d_ldg``, ``dispersion_analysis``, ...) exists, is
tested, is documented -- and cannot be reached from the documented public entry
point ``from tensordg import ...``.  Two independent instances were on record
before this audit: ``tensordg/audit_report_m16246_dalembert.md`` finding #1
(``elastic_1d`` imported but absent from ``__all__``) and
``docs/ldg_poisson_status.md`` §1 (a tutorial claiming an ``LDG`` solver that did
not exist).

This script answers three falsifiable questions and writes the answers to
``validation/api_contract/contract.json``:

1. **Coverage.** Is every public symbol (module-level ``def``/``class``, no
   leading underscore, not a ``main`` CLI hook) of every ``tensordg`` submodule
   reachable from the package root, and is the root name the *same object*
   (``__module__``/``__qualname__`` identity, so a mis-wired alias is caught)?
2. **Consistency.** Is ``__all__`` duplicate-free, fully resolvable, and exactly
   the set of public names the package namespace exposes?
3. **Claims.** Do the import snippets printed in ``docs/``/``README`` still
   resolve, and does every ``tensordg/<path>.py`` referenced by a document
   actually exist?

Everything is derived from source (AST + ``__all__`` + a real import), so the
numbers cannot drift from the code; nothing here is transcribed.

Usage
-----
    python tools/api_contract.py                 # human summary
    python tools/api_contract.py --json OUT      # write the artifact
    python tools/api_contract.py --check         # exit 1 if any gate is red
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "tensordg"
PACKAGE = "tensordg"
#: CLI hooks are not API: every ``main``/``__main__`` entry is excluded.
EXCLUDED_NAMES = {"main"}

# Modules that are deliberately outside the public contract because they are
# unvalidated prototypes.  Two merged missions had pulled in opposite
# directions -- the API-coverage sweep re-exported ``poisson_2d_ldg`` while
# ``tests/test_ldg_closure_rank.py`` forbade it -- which left the suite red on
# ``main``.  The prototype is not a capability: it is excluded from the coverage
# denominator *and* required to carry its status marker (see
# ``prototype_report``), so neither silent export nor silent deletion is possible.
#: Modules carrying a "STATUS: PROTOTYPE" marker that must stay out of the
#: public namespace.  Empty since the 2-D LDG closure was validated
#: (docs/CASE-DG-LDG-2D.md).
PROTOTYPE_MODULES: dict[str, str] = {}
DOC_GLOBS = ("docs/**/*.md", "README.md", "AGENTS.md", "TASK_COMPLETION_SUMMARY.md")


# --------------------------------------------------------------------------- #
# inventory: what does the source tree actually define?
# --------------------------------------------------------------------------- #
def _module_path(file: Path) -> str:
    """``tensordg/x/y.py`` -> ``tensordg.x.y``."""
    return ".".join(file.with_suffix("").relative_to(ROOT).parts)


def module_inventory() -> list[dict]:
    """Public symbols defined by every submodule, read from the AST."""
    records = []
    for file in sorted(PKG.rglob("*.py")):
        if file.name == "__init__.py":
            continue
        tree = ast.parse(file.read_text())
        symbols = [
            (node.name, type(node).__name__)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not node.name.startswith("_")
            and node.name not in EXCLUDED_NAMES
        ]
        module = _module_path(file)
        records.append({
            "module": module,
            "file": str(file.relative_to(ROOT)),
            "public": [name for name, _kind in symbols],
            "kinds": {name: kind for name, kind in symbols},
            "prototype": module in PROTOTYPE_MODULES,
        })
    return records


def export_registry() -> dict[str, dict]:
    """``root name -> {module, source name}``, read from ``__init__.py`` AST."""
    tree = ast.parse((PKG / "__init__.py").read_text())
    registry: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        base = PACKAGE if node.level else ""
        module = f"{base}.{node.module}" if node.module else base
        for alias in node.names:
            registry[alias.asname or alias.name] = {
                "module": module,
                "name": alias.name,
                "renamed": bool(alias.asname) and alias.asname != alias.name,
            }
    return registry


# --------------------------------------------------------------------------- #
# the three questions
# --------------------------------------------------------------------------- #
def prototype_report(inventory: list[dict], registry: dict[str, dict]) -> dict:
    """Prototype modules: status marker present, and no symbol exported.

    Encodes the policy that a coverage sweep may not promote an unvalidated
    prototype into the public contract.
    """
    exported = {(entry["module"], entry["name"]) for entry in registry.values()}
    records = []
    for record in inventory:
        if not record.get("prototype"):
            continue
        marker = PROTOTYPE_MODULES[record["module"]]
        text = (ROOT / record["file"]).read_text()
        leaked = [n for n in record["public"]
                  if (record["module"], n) in exported]
        records.append({
            "module": record["module"],
            "marker": marker,
            "marker_present": marker in text,
            "leaked_exports": leaked,
            "ok": (marker in text) and not leaked,
        })
    # The gate reads "every registered prototype is still marked and still
    # unexported".  It is vacuously satisfied when no prototype is registered
    # any more (the 2-D LDG module was promoted after validation -- see
    # docs/CASE-DG-LDG-2D.md); requiring a non-empty registry would make the
    # gate impossible to keep green once the last prototype is validated.
    return {"prototypes": records,
            "all_ok": all(r["ok"] for r in records)}


def coverage_report(inventory: list[dict], registry: dict[str, dict]) -> dict:
    exported_pairs = {(entry["module"], entry["name"]) for entry in registry.values()}
    per_module = []
    total_public = 0
    total_exported = 0
    for record in inventory:
        if record.get("prototype"):
            per_module.append({
                "module": record["module"], "public": len(record["public"]),
                "exported": 0, "missing": [], "aliases": {}, "prototype": True,
            })
            continue
        covered = [n for n in record["public"] if (record["module"], n) in exported_pairs]
        missing = [n for n in record["public"] if (record["module"], n) not in exported_pairs]
        aliases = {
            root: entry["name"]
            for root, entry in registry.items()
            if entry["module"] == record["module"] and entry["name"] in record["public"]
            and root != entry["name"]
        }
        total_public += len(record["public"])
        total_exported += len(covered)
        per_module.append({
            "module": record["module"],
            "public": len(record["public"]),
            "exported": len(covered),
            "missing": missing,
            "aliases": aliases,
        })
    return {
        "public_symbols": total_public,
        "exported_symbols": total_exported,
        "ratio": (total_exported / total_public) if total_public else 1.0,
        "modules": per_module,
        # prototypes are excluded by design (see PROTOTYPE_MODULES); they are
        # reported separately by prototype_report(), not as a coverage failure
        "modules_with_zero_exports": [m["module"] for m in per_module
                                      if m["exported"] == 0 and not m.get("prototype")],
    }


def identity_report(package, registry: dict[str, dict]) -> dict:
    """Every exported root name must be the *same object* as its source symbol."""
    violations = []
    for root_name, entry in sorted(registry.items()):
        obj = getattr(package, root_name, None) if hasattr(package, root_name) else None
        if obj is None:
            violations.append({"name": root_name, "problem": "missing from namespace"})
            continue
        module = importlib.import_module(entry["module"])
        source = getattr(module, entry["name"], None)
        if source is None:
            violations.append({"name": root_name,
                               "problem": f"{entry['module']}.{entry['name']} does not exist"})
            continue
        if isinstance(source, type) or callable(source):
            if obj is not source:
                violations.append({"name": root_name,
                                   "problem": "not the same object as its source symbol"})
    return {"checked": len(registry), "violations": violations}


def dunder_all_report(package, registry: dict[str, dict]) -> dict:
    names = list(getattr(package, "__all__", []))
    duplicates = sorted({n for n in names if names.count(n) > 1})
    unresolved = [n for n in names if not hasattr(package, n)]
    module_objects = [n for n in names if hasattr(package, n)
                      and type(getattr(package, n)).__name__ == "module"]
    namespace_public = sorted(
        n for n in dir(package)
        if not n.startswith("_") and type(getattr(package, n)).__name__ != "module"
    )
    return {
        "count": len(names),
        "duplicates": duplicates,
        "unresolved": unresolved,
        "modules_in_all": module_objects,
        "in_all_not_in_namespace": sorted(set(names) - set(namespace_public)),
        "in_namespace_not_in_all": sorted(set(namespace_public) - set(names)),
        "registry_names_not_in_all": sorted(set(registry) - set(names)),
        "in_all_not_in_registry": sorted(set(names) - set(registry)),
    }


SNIPPET_RE = re.compile(
    r"from\s+(tensordg(?:[.\w]+)?)\s+import\s+\(?\s*"
    r"([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)")
MODULE_CLAIM_RE = re.compile(r"tensordg/[A-Za-z0-9_/]+\.py")
#: A line that says a path does *not* exist is not a claim that it does.
NEGATION_MARKERS = ("不存在", "未提升", "未实现", "not implemented", "missing", "已删除", "0 命中")
#: Registered, explained stale references (ratchet): every entry carries a reason
#: and a status.  New stale claims are not allowed; known ones stay visible here.
BASELINE_FILE = ROOT / "validation" / "api_contract" / "doc_claims_baseline.json"


def load_claim_baseline() -> dict[str, dict]:
    if not BASELINE_FILE.exists():
        return {}
    payload = json.loads(BASELINE_FILE.read_text())
    return {entry["claim"]: entry for entry in payload.get("known_stale", [])}


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for pattern in DOC_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return [f for f in files if f.is_file()]


def docs_report() -> dict:
    """Do the import snippets and module paths printed in the docs resolve?"""
    snippets = []
    bad_snippets = []
    claimed_modules = set()
    for file in _doc_files():
        text = file.read_text(errors="replace")
        for match in SNIPPET_RE.finditer(text):
            module, raw = match.group(1), match.group(2)
            line = match.group(0).replace("\n", " ")
            names = [n.strip().split(" as ")[0].strip()
                     for n in raw.replace("(", " ").replace(")", " ").split(",")]
            names = [n for n in names if n and n.isidentifier()]
            target = importlib.import_module(module)
            missing = [n for n in names if not hasattr(target, n)]
            entry = {"file": str(file.relative_to(ROOT)), "line": line,
                     "module": module, "names": names, "missing": missing}
            snippets.append(entry)
            if missing:
                bad_snippets.append(entry)
        for line in text.splitlines():
            if any(marker in line for marker in NEGATION_MARKERS):
                continue  # explicitly negative statement, not an existence claim
            for claim in MODULE_CLAIM_RE.findall(line):
                if claim.startswith("tensordg/") and not (ROOT / claim).exists():
                    claimed_modules.add(claim)
    baseline = load_claim_baseline()
    missing = sorted(claimed_modules)
    registered = [c for c in missing if c in baseline]
    return {
        "doc_files_scanned": len(_doc_files()),
        "snippets_checked": len(snippets),
        "snippets_unresolved": bad_snippets,
        "missing_module_claims": missing,
        "registered_claims": registered,
        "unregistered_claims": [c for c in missing if c not in baseline],
        "baseline_entries": len(baseline),
    }


def audit() -> dict:
    package = importlib.import_module(PACKAGE)
    inventory = module_inventory()
    registry = export_registry()
    coverage = coverage_report(inventory, registry)
    identity = identity_report(package, registry)
    dunder = dunder_all_report(package, registry)
    docs = docs_report()
    prototypes = prototype_report(inventory, registry)
    gates = {
        "coverage_100pct": coverage["ratio"] == 1.0,
        "identity_clean": not identity["violations"],
        "dunder_all_consistent": not (dunder["duplicates"] or dunder["unresolved"]
                                      or dunder["in_all_not_in_namespace"]
                                      or dunder["in_namespace_not_in_all"]
                                      or dunder["registry_names_not_in_all"]
                                      or dunder["in_all_not_in_registry"]),
        "docs_snippets_resolve": not docs["snippets_unresolved"],
        "no_unregistered_module_claims": not docs["unregistered_claims"],
        "prototypes_excluded_and_marked": prototypes["all_ok"],
    }
    return {
        "package": PACKAGE,
        "inventory": inventory,
        "coverage": coverage,
        "identity": identity,
        "dunder_all": dunder,
        "docs": docs,
        "prototypes": prototypes,
        "gates": gates,
        "all_gates_green": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full audit result here")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when any gate is red")
    args = parser.parse_args(argv)

    result = audit()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    cov = result["coverage"]
    print(f"{PACKAGE}: {cov['exported_symbols']}/{cov['public_symbols']} public symbols "
          f"reachable from the package root ({cov['ratio']:.1%})")
    print(f"  modules with zero exports: {len(cov['modules_with_zero_exports'])}")
    for name in cov["modules_with_zero_exports"]:
        print(f"    - {name}")
    print(f"  exported under an alias: "
          f"{sum(len(m['aliases']) for m in cov['modules'])}")
    print(f"  __all__ entries: {result['dunder_all']['count']}; identity violations: "
          f"{len(result['identity']['violations'])}")
    print(f"  docs: {result['docs']['snippets_checked']} import snippets, "
          f"{len(result['docs']['snippets_unresolved'])} unresolved; "
          f"{len(result['docs']['missing_module_claims'])} missing module claims")
    print("  gates: " + " ".join(f"{k}={'GREEN' if v else 'RED'}"
                                for k, v in result["gates"].items()))
    if args.check and not result["all_gates_green"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())