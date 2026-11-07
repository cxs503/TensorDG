#!/usr/bin/env python3
"""Render the markdown tables of the ice-edge DG benchmark artifact.

Numbers are injected programmatically from ``artifacts/dg_advection_bench.json``
so that the documentation never contains hand-copied values::

    python tools/render_bench_tables.py > /tmp/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "dg_advection_bench.json"

NAMES = {
    "shear_transport": "periodic shear (analytic)",
    "solid_body_rotation_front": "rotating front (disk norm)",
    "reentrant_deformational_vortex": "reentrant vortex",
}


def main() -> int:
    d = json.loads(ARTIFACT.read_text())
    out = []
    add = out.append

    add("### Observed order of accuracy (least-squares fit of log err vs log n)\n")
    add("| case | p | n cells | L2 errors | observed order | required (p+0.9) | verdict |")
    add("|---|---|---|---|---|---|---|")
    for case, per in d["cases"].items():
        for p, t in sorted(per.items()):
            errs = " / ".join("%.3e" % e for e in t["errors_l2"])
            verdict = ("PASS" if t["order_pass"] else
                       "PASS (reported)" if t["order_pass"] is None else "FAIL")
            add(f"| {NAMES[case]} | {p} | {t['cells']} | {errs} | "
                f"**{t['observed_order']:.2f}** | {t['required_order']:.1f} | {verdict} |")

    add("\n### Pairwise rates, mass drift and step counts\n")
    add("| case | p | pairwise orders | max relative mass drift | RK4 steps |")
    add("|---|---|---|---|---|")
    for case, per in d["cases"].items():
        for p, t in sorted(per.items()):
            pw = " / ".join("%.2f" % v for v in t["pairwise_orders"])
            add(f"| {NAMES[case]} | {p} | {pw} | {max(t['mass_drift_rel']):.2e} | "
                f"{', '.join(str(s) for s in t['steps'])} |")

    add("\n### Peak preservation after one revolution (exact solution = initial field)\n")
    add("| case | p | n | steps | peak initial | peak final | retention | rel. L2 | mass drift |")
    add("|---|---|---|---|---|---|---|---|---|")
    peak = d["peak_preservation"]
    for r in peak["solid_body_rotation_front"]:
        add(f"| rotating front | {r['degree']} | {r['cells'][0]} | {r['steps']} | "
            f"{r['peak_initial']:.5f} | {r['peak_final']:.5f} | "
            f"**{r['peak_retention']:.5f}** | {r['relative_l2_disk']:.3e} (disk) | "
            f"{r['mass_drift_rel']:.1e} |")
    for r in peak["solid_body_rotation_resolution_study"]:
        add(f"| rotating front, res. study | {r['degree']} | {r['cells'][0]} | {r['steps']} | "
            f"{r['peak_initial']:.5f} | {r['peak_final']:.5f} | "
            f"**{r['peak_retention']:.5f}** | {r['relative_l2_disk']:.3e} (disk) | "
            f"{r['mass_drift_rel']:.1e} |")
    for r in peak["reentrant_deformational_vortex"]:
        add(f"| reentrant vortex | {r['degree']} | {r['cells'][0]} | {r['steps']} | "
            f"{r['peak_initial']:.5f} | {r['peak_final']:.5f} | "
            f"**{r['peak_retention']:.5f}** | {r['relative_l2']:.3e} (global) | "
            f"{r['mass_drift_rel']:.1e} |")

    mass = d["mass_balance"]
    add("\n### Global mass balance (all runs)\n")
    add(f"Max relative drift over every run of the campaign: "
        f"**{mass['max_relative_drift_observed']:.3e}** (acceptance {mass['tolerance']:.0e}). "
        f"Traced case: p={mass['trace']['degree']}, n={mass['trace']['cells'][0]}, "
        f"{mass['trace']['steps']} RK4 steps, final drift "
        f"{mass['trace']['relative_drift_final']:.3e}.")

    add("\n### Environment and provenance\n")
    add("```")
    add(json.dumps(d["environment"], indent=1))
    add(f"commit {d['commit']} | config {d['config']} | generated {d['generated_utc']}")
    add("```")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())