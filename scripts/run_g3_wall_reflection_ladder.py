"""G3 ladder: shock reflection off an *embedded* solid wall vs the exact two-shock solution.

Anchor ``REPRO-DG-CKS1998`` — the forward-facing step (Cockburn & Shu, JCP 141 (1998)
199-224, §5.3; Woodward & Colella, JCP 54 (1984) 115-173) needs solid walls *inside*
the domain, and ``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md`` §4(a) left the
quantitative correctness of that internal-wall flux as an honest FAIL/WITHHELD.

This driver runs the *paired* experiment the gate supports but which had never been
executed and recorded:

* ``wall_mode="boundary"`` — the wall is the domain face ``x == x_wall`` (the
  pre-existing ``bcs right="wall"`` path, already covered by
  ``tests/test_euler_dg_2d_wall_bc.py``);
* ``wall_mode="solid"`` — the wall is a frozen solid block behind that same face
  (the code path under test).

Both arms see the same analytic reference (exact incident + reflected normal
shocks computed from the Rankine-Hugoniot jump conditions and a bisection on the
reflected shock Mach number — independent of the DG solver).  Comparing the two
arms at *identical* mesh separates "the embedded-wall flux is wrong" from
"DG smears a discontinuity on this mesh", which is what the earlier single-arm
FAIL could not distinguish.

Usage::

    PYTHONPATH=. python -m scripts.run_g3_wall_reflection_ladder --out validation/forward_step/g3

Every number in the emitted JSON/CSV is printed by the run itself; nothing is
hand-copied into the markdown report.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from benchmarks.forward_facing_step import gate_wall_reflection

#: (ncells, degree) — an h-ladder in P1 plus a P1/P2 pair on the finest mesh.
#: The march to the wall dominates the cost (``steps ~ t_end / (cfl*dx/(2k+1)/c)``),
#: so the shock starts close to the wall and the domain ends just behind it.  The
#: reflected plateau then occupies ``|reflected_speed| * extra_time`` -- at
#: ``dx = xmax/nx`` that is 6/12/24 cells on the three P1 meshes.
LADDER = (
    ((29, 1), 1),
    ((58, 1), 1),
    ((116, 1), 1),
    ((58, 1), 2),
)

#: ``ymax`` is deliberately large: with ``ny = 1`` the y-direction is uniform and
#: its term ``speed_y/dy`` only shrinks the time step (the flow is 1-D in x).
GEOMETRY = dict(x_wall=0.3, x_shock=0.25, xmax=0.35, ymax=0.5, extra_time=0.06)


def run(out: Path, ladder=LADDER, arms=("boundary", "solid"), limiter=None,
        tvb_constant=1.0, incident_mach=2.0):
    rows = []
    for ncells, degree in ladder:
        for arm in arms:
            started = time.time()
            res = gate_wall_reflection(ncells=ncells, degree=degree, wall_mode=arm,
                                       limiter=limiter, tvb_constant=tvb_constant,
                                       incident_mach=incident_mach, **GEOMETRY)
            rows.append({
                "ncells": list(ncells), "degree": degree, "wall_mode": arm,
                "limiter": limiter, "tvb_constant": tvb_constant,
                "incident_mach": incident_mach,
                "status": res["status"], "pass": res["pass"],
                "dx": res["dx"], "steps": res["steps"],
                "wall_time": res["wall_time"], "final_time": res["final_time"],
                "plateau_cells": res["plateau_cells"],
                "plateau_cells_available": res["plateau_cells_available"],
                "exact_rho5": res["reflected_rho_exact"],
                "exact_p5": res["reflected_p_exact"],
                "worst_pct": res["worst_relative_error_percent"],
                "mean_rho_pct": res["plateau_mean_rho_error_percent"],
                "mean_p_pct": res["plateau_mean_p_error_percent"],
                "l1_rho": res["l1_relative"]["density"],
                "l1_p": res["l1_relative"]["pressure"],
                "linf_rho": res["linf_relative"]["density"],
                "linf_p": res["linf_relative"]["pressure"],
                "mass_residual": res["mass_relative_residual"],
                "energy_residual": res["energy_relative_residual"],
                "seconds": round(time.time() - started, 2),
            })
            print(json.dumps(rows[-1]), flush=True)

    payload = {
        "case": "G3 shock reflection off an embedded solid wall",
        "anchor": "REPRO-DG-CKS1998",
        "reference": ("exact two-shock solution: Rankine-Hugoniot + bisection on the "
                      "reflected shock Mach number (benchmarks.euler_dg_2d_wall_bc."
                      "WallReflection), independent of the DG solver"),
        "geometry": GEOMETRY,
        "limiter": limiter, "tvb_constant": tvb_constant,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "g3_ladder.json").write_text(json.dumps(payload, indent=2))
    (out / "g3_ladder.csv").write_text(to_csv(rows))
    return payload


def to_csv(rows) -> str:
    if not rows:
        return ""
    keys = list(rows[0])
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join("" if row[k] is None else str(row[k]) for k in keys))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="validation/forward_step/g3")
    parser.add_argument("--limiter", default=None)
    parser.add_argument("--tvb-constant", type=float, default=1.0)
    args = parser.parse_args(argv)
    payload = run(Path(args.out), limiter=args.limiter,
                  tvb_constant=args.tvb_constant)
    print(f"wrote {args.out}/g3_ladder.json ({len(payload['rows'])} rows)")
    return payload


if __name__ == "__main__":
    main()