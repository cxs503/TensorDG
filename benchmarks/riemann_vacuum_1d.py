"""Near-vacuum double-rarefaction verification for the 1-D Euler DG solver.

Verification-debt harness for the claim in ``README.md`` / ``euler_dg_1d.py``:
"general P0 Euler Riemann solvers and high-order HLLC DG with **local
positivity fallback**".  ``docs/shu_osher_verification.md`` §5.8 states that the
scope of the limiter study is *moderate shock strength + smooth perturbation*
and that near-vacuum / strong-rarefaction problems (Toro test 2, Einfeldt's
"123" problem) were **not** covered.  This module covers that missing case.

Physics
-------
Toro (2009, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, §4.5,
test 2; equivalently Einfeldt et al. 1991 "123 problem") is the standard
near-vacuum test: two symmetric rarefactions pull the gas apart, the density in
the middle of the domain drops by a factor ~50 while the pressure drops by a
factor ~200.  Its exact solution is available in closed form
(``benchmarks/riemann_exact.py``), so the gate is an absolute one (no
self-referential reference profile).

Pre-registered criteria (fixed before looking at the numbers)
-------------------------------------------------------------
C1 admissibility   : with the shipped fallback, every recorded state has
                     ``rho > 0``, ``p > 0`` and finite values (no abort).
C2 accuracy        : relative L2 error of density / velocity / pressure against
                     the exact Riemann solution < 3 % at the finest mesh.
C3 fallback value  : the identical run with ``_positivity_filter`` disabled
                     aborts (non-physical state) or produces non-physical
                     values -> the fallback is *load bearing* here, not
                     cosmetic.
C4 locality        : the fallback stays local - the number of reverted cells in
                     any single call is a small fraction of the mesh (documented
                     per case; a global revert would destroy the solution).
C5 convergence     : errors must not grow under refinement (monotone decay,
                     measured EOC reported honestly whatever it is).

Everything is CPU/float64/single-threaded, deterministic; the JSON artifact is
machine readable and the Markdown report is generated from it (no hand-copied
numbers).

Usage
-----
    python -m benchmarks.riemann_vacuum_1d                # full sweep (~6 min)
    python -m benchmarks.riemann_vacuum_1d --quick        # smoke sweep
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import tensordg.euler_dg_1d as euler_dg_module
from benchmarks.riemann_exact import cell_averages
from tensordg.euler_dg_1d import solve_euler_dg_1d

GAMMA = 1.4

#: (rho, u, p) left / right, final time, domain.  ``toro2`` is the near-vacuum
#: double rarefaction (Einfeldt "123"); ``toro1`` is the modified-Sod control
#: that the repository already covers, included so the harness itself is
#: calibrated against a known-good case.
CASES = {
    "toro2_near_vacuum": {
        "label": "Toro test 2 (Einfeldt 123): symmetric double rarefaction, near-vacuum",
        "left": (1.0, -2.0, 0.4),
        "right": (1.0, 2.0, 0.4),
        "final_time": 0.15,
        "domain": (0.0, 1.0),
        "diaphragm": 0.5,
    },
    "toro1_modified_sod": {
        "label": "Toro test 1: modified Sod (strong rarefaction + shock) control",
        "left": (1.0, 0.75, 1.0),
        "right": (0.125, 0.0, 0.1),
        "final_time": 0.25,
        "domain": (0.0, 1.0),
        "diaphragm": 0.5,
    },
}


def _conservative_to_primitive_averages(coeff):
    """Cell averages of (rho, u, p) from modal coefficients (mode 0 = average)."""
    average = coeff[:, 0].detach().cpu().numpy()
    rho = average[:, 0]
    momentum = average[:, 1]
    energy = average[:, 2]
    velocity = momentum / np.where(np.abs(rho) > 0.0, rho, np.nan)
    pressure = (GAMMA - 1.0) * (energy - 0.5 * momentum * velocity)
    return rho, velocity, pressure


def _relative_l2(numerical, exact):
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def run_configuration(case_key, ncells, degree, cfl=0.12, fallback=True, profile=False):
    """Run one (case, mesh, degree, fallback) configuration and return metrics.

    The shipped solver always applies ``_positivity_filter``.  To measure its
    contribution without touching production code, the module-level symbol is
    temporarily replaced by (i) an instrumented wrapper for the ``fallback=True``
    runs and (ii) the identity map for the ``fallback=False`` runs.
    """
    case = CASES[case_key]
    counters = {"rhs_calls": 0, "filter_calls": 0, "cells_reverted": 0,
                "max_cells_reverted_per_call": 0, "calls_with_revert": 0}
    original_filter = euler_dg_module._positivity_filter
    original_rhs = euler_dg_module._rhs

    def counted_rhs(*args, **kwargs):
        counters["rhs_calls"] += 1
        return original_rhs(*args, **kwargs)

    def counted_filter(coeff, check_points):
        out = original_filter(coeff, check_points)
        reverted = int((out != coeff).any(dim=2).any(dim=1).sum().item())
        counters["filter_calls"] += 1
        counters["cells_reverted"] += reverted
        counters["calls_with_revert"] += int(reverted > 0)
        counters["max_cells_reverted_per_call"] = max(
            counters["max_cells_reverted_per_call"], reverted,
        )
        return out

    euler_dg_module._rhs = counted_rhs
    euler_dg_module._positivity_filter = counted_filter if fallback else (lambda c, p: c)
    started = time.time()
    record = {
        "case": case_key,
        "label": case["label"],
        "ncells": ncells,
        "degree": degree,
        "cfl": cfl,
        "fallback": bool(fallback),
        "flux": "hllc",
        "integrator": "SSP-RK3",
    }
    try:
        x, coeff, meta = solve_euler_dg_1d(
            case["left"], case["right"], ncells=ncells, degree=degree,
            final_time=case["final_time"], diaphragm=case["diaphragm"],
            domain=case["domain"], cfl=cfl,
        )
        record["status"] = "ok"
        record["steps"] = int(meta["steps"])
        rho, velocity, pressure = _conservative_to_primitive_averages(coeff)
        record["min_density"] = float(np.min(rho))
        record["max_density"] = float(np.max(rho))
        record["min_pressure"] = float(np.min(pressure))
        record["max_pressure"] = float(np.max(pressure))
        record["finite"] = bool(np.isfinite(rho).all() and np.isfinite(pressure).all())
        centers, exact = cell_averages(
            ncells, case["final_time"], case["left"], case["right"],
            diaphragm=case["diaphragm"], domain=case["domain"],
        )
        for field, values in (("density", rho), ("velocity", velocity),
                              ("pressure", pressure)):
            record[f"rel_l2_{field}"] = _relative_l2(values, exact[field])
            record[f"rel_l1_{field}"] = float(
                np.linalg.norm(values - exact[field], ord=1) / np.linalg.norm(exact[field], ord=1)
            )
        record["max_field_error"] = max(
            record["rel_l2_density"], record["rel_l2_velocity"], record["rel_l2_pressure"],
        )
        record["exact_min_density"] = float(np.min(exact["density"]))
        record["exact_min_pressure"] = float(np.min(exact["pressure"]))
        if profile:
            record["profile"] = {
                "centers": centers.tolist(),
                "density": rho.tolist(),
                "velocity": velocity.tolist(),
                "pressure": pressure.tolist(),
                "exact_density": exact["density"].tolist(),
                "exact_velocity": exact["velocity"].tolist(),
                "exact_pressure": exact["pressure"].tolist(),
            }
    except FloatingPointError as error:
        record["status"] = "aborted"
        record["abort_reason"] = str(error)
    finally:
        euler_dg_module._positivity_filter = original_filter
        euler_dg_module._rhs = original_rhs
        record["wall_time_s"] = round(time.time() - started, 2)

    record["steps_reached"] = counters["rhs_calls"] // 3
    record["filter_calls"] = counters["filter_calls"]
    record["cells_reverted_total"] = counters["cells_reverted"]
    record["max_cells_reverted_per_call"] = counters["max_cells_reverted_per_call"]
    record["max_reverted_fraction_per_call"] = (
        counters["max_cells_reverted_per_call"] / ncells if ncells else 0.0
    )
    record["calls_with_revert"] = counters["calls_with_revert"]
    record["revert_call_fraction"] = (
        counters["calls_with_revert"] / counters["filter_calls"] if counters["filter_calls"] else 0.0
    )
    return record


def observed_orders(records, key="max_field_error"):
    """Pairwise observed order of accuracy over the mesh sequence in ``records``."""
    ordered = sorted(records, key=lambda r: r["ncells"])
    orders, previous = [], None
    for record in ordered:
        if previous is not None and record["status"] == "ok" and previous["status"] == "ok":
            ratio = record["ncells"] / previous["ncells"]
            error_ratio = previous[key] / record[key]
            orders.append(float(np.log(error_ratio) / np.log(ratio)))
        previous = record
    return orders


def build_sweep(quick=False):
    """Full sweep definition: two cases x degrees x meshes x fallback on/off."""
    meshes = (100, 200) if quick else (100, 200, 400)
    degrees = (1, 2) if quick else (1, 2, 3)
    sweep = []
    for case_key in CASES:
        for degree in degrees:
            records = []
            for ncells in meshes:
                records.append(run_configuration(
                    case_key, ncells, degree, fallback=True, profile=(ncells == meshes[-1]),
                ))
            for record in records:
                record["observed_orders"] = observed_orders(
                    [r for r in records if r["status"] == "ok"]
                )
            # Fallback disabled: only the two coarsest meshes are informative
            # (the run aborts almost immediately, that is the point).
            for ncells in meshes[:2]:
                records.append(run_configuration(case_key, ncells, degree, fallback=False))
            sweep.extend(records)
    return sweep


def verdicts(sweep):
    """Apply the pre-registered criteria C1-C5 to the recorded sweep.

    Criteria C2/C3 are scoped to the near-vacuum case (``toro2_near_vacuum``),
    which is the subject of this study; the modified-Sod control is reported
    alongside but kept out of the gate because unlimited high-order DG is
    expected to overshoot across its shock (documented in
    ``docs/shu_osher_verification.md``).
    """
    primary, control = "toro2_near_vacuum", "toro1_modified_sod"
    finest = {}
    for record in sweep:
        if not record["fallback"] or record["status"] != "ok":
            continue
        key = (record["case"], record["degree"])
        if key not in finest or record["ncells"] > finest[key]["ncells"]:
            finest[key] = record

    def finest_for(case):
        return {k: v for k, v in finest.items() if k[0] == case}

    c1 = all(
        r["status"] == "ok" and r["min_density"] > 0.0 and r["min_pressure"] > 0.0 and r["finite"]
        for r in sweep if r["fallback"]
    )
    primary_finest = finest_for(primary)
    control_finest = finest_for(control)
    c2 = all(r["max_field_error"] < 0.03 for r in primary_finest.values()) and bool(primary_finest)
    c2b = all(r["max_field_error"] < 0.03 for r in control_finest.values()) and bool(control_finest)

    def off_runs(case):
        return [r for r in sweep if not r["fallback"] and r["case"] == case]

    c3 = all(r["status"] == "aborted" for r in off_runs(primary)) and bool(off_runs(primary))
    c4 = all(r["max_reverted_fraction_per_call"] < 0.10 for r in sweep if r["fallback"])
    monotone = {}
    for key in finest:
        series = sorted(
            [(r["ncells"], r["max_field_error"]) for r in sweep
             if r["fallback"] and r["status"] == "ok"
             and r["case"] == key[0] and r["degree"] == key[1]],
        )
        monotone[f"{key[0]}|P{key[1]}"] = {
            "observed_orders": list(finest[key].get("observed_orders", [])),
            "errors": series,
        }
    c5 = all(
        all(later[1] <= earlier[1] for earlier, later in zip(items["errors"], items["errors"][1:]))
        for items in monotone.values()
    )
    return {
        "C1_admissibility_with_fallback": bool(c1),
        "C2_near_vacuum_finest_mesh_error_below_3pct": bool(c2),
        "C2b_control_finest_mesh_error_below_3pct": bool(c2b),
        "C3_near_vacuum_fallback_disabled_aborts": bool(c3),
        "C4_fallback_locality_below_10pct_cells": bool(c4),
        "C5_errors_monotone_under_refinement": bool(c5),
        "all_criteria_pass": bool(c1 and c2 and c3 and c4 and c5),
        "control_no_fallback_status": [
            {"degree": r["degree"], "ncells": r["ncells"], "status": r["status"],
             "min_density": r.get("min_density"), "min_pressure": r.get("min_pressure"),
             "steps_reached": r.get("steps_reached")}
            for r in sorted(off_runs(control), key=lambda r: (r["degree"], r["ncells"]))
        ],
        "finest_mesh": {
            f"{case}|P{degree}": {
                "ncells": record["ncells"],
                "max_field_error": record["max_field_error"],
                "rel_l2_density": record["rel_l2_density"],
                "rel_l2_velocity": record["rel_l2_velocity"],
                "rel_l2_pressure": record["rel_l2_pressure"],
                "min_density": record["min_density"],
                "min_pressure": record["min_pressure"],
            }
            for (case, degree), record in finest.items()
        },
        "orders": monotone,
    }


def recompute_verdicts(path):
    """Re-apply :func:`verdicts` to an existing artifact (no re-simulation)."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["verdicts"] = verdicts(payload["records"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload["verdicts"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two meshes x P1/P2 smoke sweep")
    parser.add_argument("--output", default="validation/riemann_vacuum/vacuum_sweep.json")
    parser.add_argument("--verdicts-only", action="store_true",
                        help="recompute the verdict block of an existing artifact")
    args = parser.parse_args(argv)

    if args.verdicts_only:
        print(json.dumps(recompute_verdicts(args.output), indent=2, sort_keys=True))
        return 0

    torch.set_num_threads(1)
    sweep = build_sweep(quick=args.quick)
    payload = {
        "harness": "benchmarks/riemann_vacuum_1d.py",
        "solver": "tensordg.euler_dg_1d.solve_euler_dg_1d (modal Legendre DG, HLLC, SSP-RK3)",
        "reference": "benchmarks/riemann_exact.py (exact Riemann solution, 8-point Gauss cell averages)",
        "gamma": GAMMA,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "quick": bool(args.quick),
        "records": sweep,
        "verdicts": verdicts(sweep),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload["verdicts"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())