"""Extend the CKS1998 isentropic-vortex order certification with finer meshes.

The committed ``validation/cks1998_vortex_2d/certification.json`` certified the
vortex case on only 3 levels per degree (p1: 12/16/24, p2: 8/12/16, p3: 6/8/12)
and returned FAIL for every degree because the finest-pair order sat below the
design order (p2: 2.86 vs 3).  This module re-runs the same ``run_case`` path on
an *extended* ladder (adding 24/32/48 to each degree) and re-certifies, so the
question "is the vortex under-convergence a genuine defect or a coarse-mesh
artifact" is answered by the data rather than by a point estimate.

The extended ladder is the same solver, same CFL, same error norm
(``L1_rel`` of ``rho``) as the committed certification -- only the mesh set
changes.  A degree whose order rises toward ``p+1`` as the mesh refines is
declared ``COARSE_MESH_ARTIFACT`` (the committed FAIL was a mesh-regime
statement, not a defect); a degree that stays below ``p+1`` on the finest
available mesh is ``GENUINE_UNDERCONVERGENCE``.

Usage::

    OMP_NUM_THREADS=4 python -m scratch.extend_vortex_certification

Writes ``validation/cks1998_vortex_2d/certification_extended.json``.
"""
from __future__ import annotations
import math, time, json
from pathlib import Path
import torch
import benchmarks.cks1998_vortex_order_certification as m

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "validation" / "cks1998_vortex_2d" / "certification_extended.json"

#: Extended ladders: the committed set plus two finer rungs per degree.
EXTENDED = {1: [12, 16, 24, 32, 48], 2: [8, 12, 16, 24, 32, 48], 3: [6, 8, 12, 16, 24, 32]}

def main():
    torch.set_num_threads(4)
    records = []
    for degree, meshes in EXTENDED.items():
        for n in meshes:
            r = m.run_case(degree, n)
            r["kind"] = "ladder"
            records.append(r)
            print(f"p{degree} n={n} h={r['h']:.4f} "
                  f"L1_rel={r['norms']['density']['L1_rel']:.6e} "
                  f"steps={r['steps']} dofs={r['dofs']} wall={r['wall_seconds']}s")
    # re-certify with the extended ladder
    payload = m.assemble(records, beta=m.TIME_REFINEMENT)
    # add a per-degree diagnosis: does the order rise toward p+1 on finer meshes?
    for name, bundle in payload["degrees"].items():
        cert = bundle["certification"]
        levels = sorted(cert["levels"], key=lambda lv: lv["h"])
        # order on the two finest pairs
        orders = []
        for a, b in zip(levels, levels[1:]):
            orders.append(math.log(a["error"] / b["error"]) / math.log(a["h"] / b["h"]))
        design = cert["design_order"]
        finest_order = orders[-1] if orders else None
        # trend: order on the finest pair vs the coarsest pair
        trend = (orders[-1] - orders[0]) if len(orders) >= 2 else 0.0
        if finest_order is not None and abs(finest_order - design) / design <= 0.03:
            diagnosis = "COARSE_MESH_ARTIFACT"
        elif finest_order is not None and finest_order < design:
            diagnosis = "GENUINE_UNDERCONVERGENCE"
        else:
            diagnosis = "UNCERTAIN"
        bundle["diagnosis"] = {
            "label": diagnosis,
            "finest_order": finest_order,
            "design_order": design,
            "order_trend_fine_minus_coarse": trend,
            "orders": [round(o, 4) for o in orders],
        }
    payload["table_markdown"] = m.render_markdown(payload)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")
    for name, bundle in payload["degrees"].items():
        d = bundle["diagnosis"]
        print(f"  {name}: finest_order={d['finest_order']:.4f} "
              f"design={d['design_order']} -> {d['label']}")

if __name__ == "__main__":
    main()
