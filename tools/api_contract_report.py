#!/usr/bin/env python
"""Before/after report for the TensorDG public API surface.

"Before" is read from the *committed* ``tensordg/__init__.py`` (``git show
HEAD:tensordg/__init__.py``), "after" is the working tree, so the delta cannot
be hand-copied.  Writes:

* ``validation/api_contract/before_after.json`` -- per-module numbers
* ``validation/api_contract/table_coverage.md`` -- the markdown table used in
  ``docs/API-CONTRACT-TENSORDG.md``
* ``docs/figures/api_contract_coverage.png``   -- per-module coverage bars
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "api_contract"


def _load_tool():
    spec = importlib.util.spec_from_file_location("api_contract", ROOT / "tools" / "api_contract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry_from_source(text: str) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ImportFrom) or node.module == "__future__":
            continue
        base = "tensordg" if node.level else ""
        module = f"{base}.{node.module}" if node.module else base
        for alias in node.names:
            registry[alias.asname or alias.name] = {"module": module, "name": alias.name}
    return registry


def coverage(inventory, registry) -> dict[str, dict]:
    pairs = {(e["module"], e["name"]) for e in registry.values()}
    rows = {}
    for record in inventory:
        public = record["public"]
        hit = [n for n in public if (record["module"], n) in pairs]
        rows[record["module"]] = {
            "public": len(public), "exported": len(hit),
            "ratio": (len(hit) / len(public)) if public else 1.0,
        }
    return rows


def main() -> int:
    tool = _load_tool()
    inventory = tool.module_inventory()
    before = registry_from_source(
        subprocess.check_output(["git", "show", "HEAD:tensordg/__init__.py"],
                                cwd=ROOT, text=True))
    after = tool.export_registry()
    rows_before = coverage(inventory, before)
    rows_after = coverage(inventory, after)
    totals = {
        "public_symbols": sum(r["public"] for r in rows_before.values()),
        "exported_before": sum(r["exported"] for r in rows_before.values()),
        "exported_after": sum(r["exported"] for r in rows_after.values()),
    }
    totals["ratio_before"] = totals["exported_before"] / totals["public_symbols"]
    totals["ratio_after"] = totals["exported_after"] / totals["public_symbols"]
    payload = {"totals": totals, "modules": {
        m: {"before": rows_before[m], "after": rows_after[m]} for m in rows_before}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "before_after.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["| module | public | exported before | exported after |",
             "|---|---|---|---|"]
    for name in sorted(rows_before):
        b, a = rows_before[name], rows_after[name]
        lines.append(f"| `{name}` | {b['public']} | {b['exported']} | {a['exported']} |")
    lines.append(f"| **total** | **{totals['public_symbols']}** | "
                 f"**{totals['exported_before']}** | **{totals['exported_after']}** |")
    (OUT / "table_coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n.replace("tensordg.", "") for n in sorted(rows_before)]
    before_pct = [100 * rows_before[n]["ratio"] for n in sorted(rows_before)]
    after_pct = [100 * rows_after[n]["ratio"] for n in sorted(rows_before)]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    idx = range(len(names))
    ax.bar([i - 0.2 for i in idx], before_pct, width=0.4, label="before (HEAD)", color="#c0392b")
    ax.bar([i + 0.2 for i in idx], after_pct, width=0.4, label="after (this mission)", color="#27ae60")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("public symbols reachable from `tensordg` [%]")
    ax.set_title(f"TensorDG public API coverage: {totals['exported_before']}/"
                 f"{totals['public_symbols']} ({100*totals['ratio_before']:.1f}%) -> "
                 f"{totals['exported_after']}/{totals['public_symbols']} "
                 f"({100*totals['ratio_after']:.1f}%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="grey", lw=0.6, ls=":")
    ax.legend(loc="upper left")
    fig.tight_layout()
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "docs" / "figures" / "api_contract_coverage.png", dpi=150)
    print("figure -> docs/figures/api_contract_coverage.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
