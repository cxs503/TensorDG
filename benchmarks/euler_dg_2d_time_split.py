"""Error budget of the 2-D high-order Euler DG solver: time vs space.

Motivation (open debt filed by the design-order study, m3024)

``benchmarks/design_order_euler_dg_2d`` sweeps P1/P2/P3 on the smooth isentropic
vortex with a *CFL-driven* step, so ``dt`` scales with ``h``.  For P3 the spatial
design order is 4 while SSP-RK3 is only third order in time, so the measured P3
slopes (3.85 / 3.52 / 3.48) were suspected of being capped by temporal error.
That study could not separate the two, and the documented ``< 3 %`` gate at a
given mesh mixes them.  This benchmark does the separation with the fixed-step
option (``dt=``) added to ``solve_euler_dg_2d`` for this purpose:

Part A -- temporal refinement at a fixed mesh.  The step is halved repeatedly.
The error against the exact solution is reported *and* the successive-solution
self-difference ``||u_{k+1} - u_k||`` is fitted, because the exact-solution error
is dominated by the spatial error and saturates; the self-difference isolates the
temporal contribution and must decay at the SSP-RK3 rate (3).

Part B -- the same mesh ladder that m3024 used, re-run here for like-for-like
comparison, plus a *quadrature* refinement.  If Part A shows the temporal error
is negligible, any P3 order deficit is spatial; the candidate is the aliasing of
the nonlinear (HLLC) face flux, which is integrated with the same
``max(degree + 2, 3)`` Gauss rule, so the ladder is repeated with a higher rule.

Run (single thread keeps this inside a shared box)::

    PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.euler_dg_2d_time_split
    PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.euler_dg_2d_time_split --quick

Writes ``validation/euler_dg_2d/time_split.json`` (after every block, so an
interrupted sweep still leaves usable evidence) and the two-panel figure
``docs/figures/euler_dg_2d_time_split.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "euler_dg_2d" / "time_split.json"
FIG = ROOT / "docs" / "figures" / "euler_dg_2d_time_split.png"
REFERENCE_JSON = ROOT / "validation" / "euler_dg_2d" / "design_order.json"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))  # length 10, vortex strength 5, background (1, 1)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
FINAL_TIME = 1.0
CFL = 0.2
# The CFL run reports only its *last* step, which is trimmed onto final_time;
# prescribing a step 10 % below it keeps every refinement level stable.
DTSAFE = 0.9


def slope(xs, ys):
    """Least-squares slope of ``log(y)`` against ``log(x)``."""
    return float(
        np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0]
    )


def pairwise_orders(values, refinements):
    return [
        float(math.log(values[i] / values[i + 1]) / math.log(refinements[i]))
        for i in range(len(values) - 1)
    ]


def run_case(degree, ncells, dt=None, limiter=None, quadrature=None,
             limiter_constant=1.0, log=None):
    """One vortex run; returns relative L2 errors, metadata and the coefficients."""
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, order)
    started = time.time()

    def initial(x, y):
        return dg2.vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))

    # The named vortex wrapper does not expose the quadrature order, so the
    # generic entry point is used for every case (identical set-up).
    _, _, coeff, metadata = dg2.solve_euler_dg_2d(
        initial, ncells=ncells, degree=degree, domain=DOMAIN,
        final_time=FINAL_TIME, cfl=CFL, dt=dt, limiter=limiter, quadrature=order,
        tvb_constant=limiter_constant,
    )
    errors = _vortex_error(coeff, basis, nodes, DOMAIN, FINAL_TIME)
    row = {
        "degree": degree,
        "cells": ncells,
        "h": 10.0 / ncells,
        "quadrature": order,
        "dt_mode": metadata["dt_mode"],
        "dt_last": float(metadata["dt"]),
        "dt_mean": FINAL_TIME / metadata["steps"],
        "prescribed_dt": metadata["prescribed_dt"],
        "steps": metadata["steps"],
        "wall_seconds": round(time.time() - started, 2),
        **{name: float(errors[name]) for name in FIELDS},
    }
    if log is not None:
        log(
            f"  P{degree} {ncells:3d}^2 quad={order} dt={row['dt_mean']:.3e} "
            f"({row['dt_mode']:5s}) steps={row['steps']:5d} "
            f"rho_err={row['density']:.4e} ({row['wall_seconds']}s)"
        )
    return row, coeff


def modal_norm(coeff):
    """Modal L2 weight: sum of squared coefficients over the (2k+1)(2l+1) weights."""
    degree = coeff.shape[2] - 1
    modes = torch.arange(degree + 1, dtype=coeff.dtype)
    weight = torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0)
    return float(torch.sqrt((coeff**2 / weight[None, None, :, :, None]).sum()).item())


def self_difference(coarse, fine):
    """Relative modal L2 difference between two solutions of the same problem."""
    return modal_norm(coarse - fine) / modal_norm(fine)


def temporal_study(degree, ncells, halvings, log=None):
    """Part A: halve the time step ``halvings`` times at a fixed mesh."""
    rows, coeffs = [], []
    base, coeff = run_case(degree, ncells, log=log)
    rows.append(dict(base, level=0, effective_dt=base["dt_mean"]))
    coeffs.append(coeff)
    dt0 = DTSAFE * float(base["dt_last"])
    for level in range(1, halvings + 1):
        row, coeff = run_case(degree, ncells, dt=dt0 / 2.0 ** (level - 1), log=log)
        rows.append(dict(row, level=level, effective_dt=row["dt_mean"]))
        coeffs.append(coeff)
    # Self-differences use the prescribed (exactly halved) levels only.
    differences = [
        self_difference(coeffs[i], coeffs[i + 1]) for i in range(1, len(coeffs) - 1)
    ]
    halving = [rows[i]["effective_dt"] / rows[i + 1]["effective_dt"]
               for i in range(1, len(rows) - 1)]
    exact = [row["density"] for row in rows]
    return {
        "kind": "temporal",
        "degree": degree,
        "cells": ncells,
        "qoi": "relative modal L2 self-difference of the conservative state",
        "theory_order": 3.0,
        "dt": [row["effective_dt"] for row in rows],
        "exact_error": exact,
        "self_difference": differences,
        "refinements": halving,
        "pairwise_orders": pairwise_orders(differences, halving),
        "fit_order": (
            slope([row["effective_dt"] for row in rows[len(rows) - len(differences):]],
                  differences)
            if differences else None
        ),
        "exact_error_relative_change": [
            abs(exact[i + 1] - exact[i]) / exact[i] for i in range(len(exact) - 1)
        ],
        "rows": rows,
    }


def quadrature_study(degree, meshes, rules, log=None):
    """Part B: re-run the m3024 ladder at two Gauss rules."""
    blocks = {}
    for rule in rules:
        rows, errors = [], []
        for ncells in meshes:
            row, _ = run_case(degree, ncells, quadrature=rule, log=log)
            rows.append(row)
            errors.append(row["density"])
        hs = [row["h"] for row in rows]
        ratios = [meshes[i + 1] / meshes[i] for i in range(len(meshes) - 1)]
        blocks[str(rule)] = {
            "quadrature": rule,
            "rows": rows,
            "errors": errors,
            "pairwise_orders": pairwise_orders(errors, ratios),
            "fit_order": slope(hs, errors),
        }
    return {
        "kind": "quadrature",
        "degree": degree,
        "theory_order": float(degree + 1),
        "meshes": list(meshes),
        "rules": blocks,
    }


def limiter_inertness(degree=3, ncells=8, log=None):
    """The TVB limiter must be inert on a smooth flow (no troubled cells)."""
    plain, _ = run_case(degree, ncells, log=log)
    limited, _ = run_case(degree, ncells, limiter="tvb", log=log)
    return {
        "kind": "limiter_inertness",
        "degree": degree,
        "cells": ncells,
        "unlimited_error": plain["density"],
        "tvb_error": limited["density"],
        "relative_change": abs(limited["density"] - plain["density"]) / plain["density"],
        "note": "smooth isentropic vortex: the troubled-cell detector should fire nowhere",
    }


def m3024_cross_check(degree, blocks):
    """Compare this study's default-rule ladder with the m3024 design-order JSON."""
    if not REFERENCE_JSON.exists():
        return {"available": False}
    data = json.loads(REFERENCE_JSON.read_text())
    reference = None
    for block in data.get("blocks", []):
        if block.get("degree") == degree:
            reference = block
    if reference is None:
        return {"available": False}
    by_cells = {row["cells"]: row["errors"]["density"] for row in reference["rows"]}
    default_rule = blocks[str(min(int(key) for key in blocks))]
    comparison = []
    for row in default_rule["rows"]:
        if row["cells"] in by_cells:
            old = by_cells[row["cells"]]
            comparison.append(
                {
                    "cells": row["cells"],
                    "m3024_error": old,
                    "this_study_error": row["density"],
                    "relative_difference": abs(row["density"] - old) / old,
                }
            )
    return {"available": True, "degree": degree, "comparison": comparison}


def dump(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")


def make_figure(payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    for block in payload["temporal"]:
        # self-differences pair consecutive prescribed levels: dt[2], dt[3], ...
        steps = block["dt"][2:]
        left.loglog(steps, block["self_difference"], "o-",
                    label=f"P{block['degree']} self-difference (fit {block['fit_order']:.2f})")
        left.loglog(block["dt"], block["exact_error"], "s:",
                    label=f"P{block['degree']} exact-solution error")
        if steps:
            scale = block["self_difference"][0] / steps[0] ** 3.0
            left.loglog(steps, [scale * t**3.0 for t in steps], "k--",
                        label="slope 3 (SSP-RK3)")
    left.set_xlabel("time step dt")
    left.set_ylabel("relative error / self-difference")
    left.set_title("Part A: temporal refinement at fixed mesh")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7)

    for block in payload["quadrature"]:
        for key, rule in block["rules"].items():
            right.loglog(block["meshes"], rule["errors"], "o-" if key == str(min(
                int(k) for k in block["rules"])) else "s--",
                label=f"P{block['degree']} quad={key} (fit {rule['fit_order']:.2f})")
        default_key = str(min(int(k) for k in block["rules"]))
        base = block["rules"][default_key]["errors"][0]
        order = block["degree"] + 1
        first = float(block["meshes"][0])
        right.loglog(block["meshes"], [base * (first / m) ** order for m in block["meshes"]],
                     "k:", label=f"slope {order} (P{block['degree']} design order)")
    right.set_xlabel("cells per direction")
    right.set_ylabel(r"relative $L_2$ error of density")
    right.set_title("Part B: mesh ladder at two Gauss rules")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=7)

    figure.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG, dpi=140)
    return FIG


def main(argv=None):
    parser = argparse.ArgumentParser(description="time/space error split for 2-D Euler DG")
    parser.add_argument("--quick", action="store_true", help="cheapest cut")
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--halvings", type=int, default=3)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    log = print
    payload = {
        "benchmark": "euler_dg_2d_time_split",
        "problem": "periodic smooth isentropic vortex (exact solution of 2-D Euler)",
        "domain": [list(DOMAIN[0]), list(DOMAIN[1])],
        "vortex_strength": 5.0,
        "background_velocity": [1.0, 1.0],
        "final_time": FINAL_TIME,
        "cfl": CFL,
        "temporal": [],
        "quadrature": [],
    }

    log("Part A: temporal refinement (fixed mesh, dt halved)")
    for degree in args.degrees:
        payload["temporal"].append(
            temporal_study(degree, 8, 2 if args.quick else args.halvings, log=log)
        )
        dump(payload)

    log("Part B: mesh ladder at two Gauss rules")
    rules = (5, 8) if not args.quick else (5, 7)
    for degree in args.degrees:
        payload["quadrature"].append(
            quadrature_study(degree, (6, 8, 12) if degree == 3 else (8, 12, 16),
                             rules, log=log)
        )
        dump(payload)
    payload["m3024_cross_check"] = m3024_cross_check(3, payload["quadrature"][-1]["rules"])

    log("Limiter inertness on the smooth vortex")
    payload["limiter_inertness"] = limiter_inertness(log=log)
    dump(payload)

    if not args.no_figure:
        payload["figure"] = str(make_figure(payload).relative_to(ROOT))
        dump(payload)
    log(f"wrote {OUT.relative_to(ROOT)}")
    return payload


if __name__ == "__main__":
    main()