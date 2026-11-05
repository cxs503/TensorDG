"""Gate-power (negative-control) audit of the documented 3% tolerances.

Motivation
----------
Thirteen verification reports in this repository assert a documented ``< 0.03``
relative-error gate against an analytic solution, but until now only two of them
(differentiability, wall-scope) shipped a *negative control* -- evidence that the
gate can actually fail.  A tolerance without a negative control is a claim we
cannot size: an error of 2.4e-6 against a 3e-2 gate passes a 12000x regression
just as happily as the verified code does.

This driver measures, per documented gate:

1. ``documented_error`` -- the error the committed test measures (reproduced);
2. ``margin`` -- ``GATE / documented_error``, i.e. how large a regression the
   gate would still accept;
3. ``crossing`` -- the coarsening factor ``cells_doc / cells`` at which the same
   case first violates the gate (resolution negative control).  A crossing at
   2x means the gate is resolution-sharp; a crossing at 16x means the gate is
   slack by four refinement levels;
4. ``sabotage`` -- a deliberate implementation defect at the documented
   resolution (SIPG penalty forced below the stability threshold, a time-offset
   in which the numerical solution is compared against the exact solution of a
   *different* time, and -- for the high-order Euler solver -- a degree drop to
   P0).  A gate that survives sabotage has no teeth.

Everything is deterministic (no RNG, float64) and every number in the report is
read back from the JSON this script writes; nothing is transcribed by hand.

Usage
-----
    python -m benchmarks.audit_gate_power --case advection_1d_p1
    python -m benchmarks.audit_gate_power --case all
    python -m benchmarks.audit_gate_power --aggregate

Each ``--case`` run writes ``validation/gate_power/<case>.json`` incrementally so
a slow case (the 192^2 vortex) never blocks the rest; ``--aggregate`` merges the
per-case files into ``validation/gate_power_audit.json`` and draws
``docs/figures/gate_power_audit.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tensordg import (  # noqa: E402
    heat_error,
    l2_error,
    poisson_error_2d,
    solve_advection,
    solve_advection_2d,
    solve_heat_1d,
    solve_poisson_2d,
    solve_isentropic_vortex,
    vortex_error,
    error_2d,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d  # noqa: E402
from benchmarks.euler_sod_1d import solve_sod  # noqa: E402
from benchmarks.sod_exact import cell_averages  # noqa: E402
from benchmarks.lax_1d import LEFT, RIGHT  # noqa: E402
from benchmarks.riemann_dg_error import riemann_dg_errors  # noqa: E402

GATE = 0.03
OUT_DIR = os.path.join(ROOT, "validation", "gate_power")
FIG_PATH = os.path.join(ROOT, "docs", "figures", "gate_power_audit.png")


# --------------------------------------------------------------------------- #
# case runners: (cells, overrides) -> {field: relative L2 error}
# --------------------------------------------------------------------------- #
def run_advection_1d(cells, **kw):
    final_time = kw.get("final_time", 0.25)
    compare_time = kw.get("compare_time", final_time)
    centers, coeff, _ = solve_advection(ncells=cells, degree=1, final_time=final_time)
    # l2_error returns an absolute L2 norm; the sine IC has L2 norm 1/sqrt(2).
    return {"u": float(l2_error(centers, coeff, compare_time) * math.sqrt(2.0))}


def run_advection_2d(cells, **kw):
    final_time = kw.get("final_time", 1.0)
    compare_time = kw.get("compare_time", final_time)
    x, y, coeff, _ = solve_advection_2d(ncells=cells, degree=1, final_time=final_time)
    return {"u": float(error_2d(x, y, coeff, compare_time)["relative_l2"])}


def run_poisson_2d(cells, **kw):
    penalty = kw.get("penalty", 20.0)
    x, y, coeff, _ = solve_poisson_2d(ncells=cells, degree=1, penalty=penalty)
    return {"u": float(poisson_error_2d(x, y, coeff)["relative_l2"])}


def run_heat_1d(cells, **kw):
    penalty = kw.get("penalty", 20.0)
    final_time = kw.get("final_time", 0.1)
    compare_time = kw.get("compare_time", final_time)
    x, coeff, _ = solve_heat_1d(
        ncells=cells, degree=2, final_time=final_time, penalty=penalty
    )
    return {"u": float(heat_error(x, coeff, compare_time)["relative_l2"])}


def run_sod(cells, **kw):
    cfl = kw.get("cfl", 0.4)
    final_time = kw.get("final_time", 0.2)
    compare_time = kw.get("compare_time", final_time)
    _, fields, _ = solve_sod(ncells=cells, final_time=final_time, cfl=cfl)
    _, reference = cell_averages(cells, compare_time)
    out = {}
    for key in ("density", "velocity", "pressure"):
        numerical = np.asarray(fields[key].detach().cpu().numpy(), dtype=float)
        ref = np.asarray(reference[key], dtype=float)
        if not np.all(np.isfinite(numerical)):
            raise FloatingPointError(f"non-finite {key}")
        out[key] = float(np.sqrt(np.mean((numerical - ref) ** 2))
                         / np.sqrt(np.mean(ref ** 2)))
    return out


def run_lax(cells, **kw):
    degree = kw.get("degree", 3)
    cfl = kw.get("cfl", 0.35)
    final_time = kw.get("final_time", 0.13)
    x, coeff, _ = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=cells, degree=degree, final_time=final_time, cfl=cfl
    )
    errors = riemann_dg_errors(x, coeff, final_time, LEFT, RIGHT)
    return {key: float(metrics["relative_l2"]) for key, metrics in errors.items()}


def run_vortex(cells, **kw):
    final_time = kw.get("final_time", 1.0)
    compare_time = kw.get("compare_time", final_time)
    x, y, fields, _ = solve_isentropic_vortex(ncells=cells, final_time=final_time)
    errors = vortex_error(x, y, fields["conservative"], compare_time)
    return {key: float(metrics["relative_l2"]) for key, metrics in errors.items()}


# name -> (runner, documented cells, smallest cells worth trying, documented kwargs)
CASES = {
    "advection_1d_p1": (run_advection_1d, 40, 5, {}),
    "advection_2d_p1": (run_advection_2d, 24, 6, {}),
    "poisson_2d_p1": (run_poisson_2d, 16, 4, {}),
    "heat_1d_p2": (run_heat_1d, 16, 2, {}),
    "sod_p0": (run_sod, 3200, 100, {}),
    "lax_p3": (run_lax, 400, 25, {}),
    "vortex_p0": (run_vortex, 192, 12, {}),
}

# Sabotage schedules applied at the documented resolution of each gate.
SABOTAGE = {
    "poisson_2d_p1": [("penalty", p) for p in (5.0, 1.0, 0.2, 0.05, 0.02, 0.005)],
    "heat_1d_p2": [("penalty", p) for p in (5.0, 1.0, 0.2, 0.05)],
    "advection_1d_p1": [("compare_time", 0.375)],
    "advection_2d_p1": [("compare_time", 1.5)],
    "sod_p0": [("compare_time", 0.15)],
    "lax_p3": [("degree", 1)],
}

FIELD_MAX = True  # gate is applied to the largest field error of a case


def _evaluate(runner, cells, kwargs):
    """Run one case; return {"errors": {...}, "max_error": float} or a failure."""
    started = time.time()
    try:
        errors = runner(cells, **kwargs)
    except Exception as exc:  # noqa: BLE001 - failure is a result, not a crash
        return {
            "cells": cells,
            "kwargs": {k: v for k, v in kwargs.items()},
            "status": "raised",
            "exception": f"{type(exc).__name__}: {exc}",
            "max_error": float("inf"),
            "seconds": round(time.time() - started, 3),
        }
    finite = all(math.isfinite(v) for v in errors.values())
    return {
        "cells": cells,
        "kwargs": {k: v for k, v in kwargs.items()},
        "status": "ok" if finite else "non_finite",
        "errors": {k: v for k, v in errors.items()},
        "max_error": max(errors.values()) if finite else float("inf"),
        "seconds": round(time.time() - started, 3),
    }


def audit_case(name, max_levels=5, reuse=True):
    runner, documented_cells, min_cells, kwargs = CASES[name]
    path = os.path.join(OUT_DIR, f"{name}.json")
    record = {
        "case": name,
        "gate": GATE,
        "documented_cells": documented_cells,
        "documented_kwargs": kwargs,
        "refinements": [],
        "sabotage": [],
    }

    cached = None
    if reuse and os.path.exists(path):
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("documented_cells") == documented_cells:
            record["refinements"] = cached.get("refinements", [])
            print(f"[{name}] reused {len(record['refinements'])} refinement "
                  f"levels from {os.path.relpath(path, ROOT)}", flush=True)
        else:
            cached = None

    cells = documented_cells
    level = 0
    need_levels = not record["refinements"]
    while need_levels and level <= max_levels:
        result = _evaluate(runner, cells, kwargs)
        result["coarsening_factor"] = documented_cells / cells
        result["level"] = level
        record["refinements"].append(result)
        print(f"[{name}] level {level} cells={cells} "
              f"max_err={result['max_error']:.6g} ({result['status']}, "
              f"{result['seconds']}s)", flush=True)
        if level > 0 and result["max_error"] >= GATE:
            break
        level += 1
        cells = documented_cells // (2 ** level)
        if cells < min_cells:
            break

    if not record["refinements"]:
        raise RuntimeError(f"{name}: no refinement level could be evaluated")

    documented = record["refinements"][0]
    record["documented_error"] = documented["max_error"]
    record["margin"] = (GATE / documented["max_error"]
                        if documented["max_error"] > 0 else float("inf"))
    crossing = None
    for entry in record["refinements"][1:]:
        if entry["max_error"] >= GATE:
            crossing = entry["coarsening_factor"]
            break
    record["crossing_factor"] = crossing
    record["crossing_levels"] = (math.log2(crossing) if crossing else None)

    for axis, value in SABOTAGE.get(name, []):
        sabotage_kwargs = dict(kwargs)
        sabotage_kwargs[axis] = value
        result = _evaluate(runner, documented_cells, sabotage_kwargs)
        result["axis"] = axis
        result["value"] = value
        # A rejected input (ValueError from argument validation) is a *different*
        # guard than the 3% gate: it must not be counted as gate power.
        result["rejected"] = (
            result["status"] == "raised"
            and isinstance(result.get("exception"), str)
            and result["exception"].startswith("ValueError")
        )
        result["broke_gate"] = (
            result["status"] != "raised" and result["max_error"] >= GATE
        )
        record["sabotage"].append(result)
        print(f"[{name}] sabotage {axis}={value} "
              f"max_err={result['max_error']:.6g} "
              f"broke_gate={result['broke_gate']}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    print(f"[{name}] wrote {os.path.relpath(path, ROOT)}", flush=True)
    return record


# --------------------------------------------------------------------------- #
# aggregation + figure
# --------------------------------------------------------------------------- #
def aggregate():
    records = []
    for name in CASES:
        path = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path) as handle:
                records.append(json.load(handle))
    missing = [name for name in CASES if name not in {r["case"] for r in records}]
    verdicts = []
    for record in records:
        sabotage_broken = [s for s in record["sabotage"] if s["broke_gate"]]
        sabotage_total = len(record["sabotage"])
        crossing = record["crossing_factor"]
        verdicts.append({
            "case": record["case"],
            "documented_error": record["documented_error"],
            "margin": record["margin"],
            "crossing_factor": crossing,
            "crossing_levels": record["crossing_levels"],
            "sabotage_broken": len(sabotage_broken),
            "sabotage_total": sabotage_total,
            "teeth": _teeth_verdict(record, sabotage_broken),
        })
    summary = {
        "gate": GATE,
        "cases_audited": [r["case"] for r in records],
        "cases_missing": missing,
        "verdicts": verdicts,
        "records": records,
    }
    path = os.path.join(ROOT, "validation", "gate_power_audit.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    if records:
        plot(records)
    return summary


def _teeth_verdict(record, sabotage_broken):
    """Classify how well a gate discriminates.

    ``sharp``    -- some sabotage breaks the gate at the documented resolution;
    ``resolution-sensitive`` -- no sabotage broke it, but coarsening does;
    ``slack``    -- neither sabotage nor the coarsening cap broke it.
    """
    if sabotage_broken:
        return "sharp"
    if record["crossing_factor"] is not None:
        return "resolution-sensitive"
    return "slack"


def plot(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for marker, record in zip(markers, records):
        levels = [r["coarsening_factor"] for r in record["refinements"]]
        errors = [max(r["max_error"], 1e-16) for r in record["refinements"]]
        ax.plot(levels, errors, marker=marker, label=record["case"], lw=1.4)
    ax.axhline(GATE, color="crimson", ls="--", lw=1.6, label="documented gate 3%")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("coarsening factor $N_{\\rm doc}/N$\n(resolution negative control)")
    ax.set_ylabel("max relative $L_2$ error")
    ax.set_title("(a) how far each 3% gate can be coarsened before it fails")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    plotted = False
    for record in records:
        sweep = [s for s in record["sabotage"] if s["axis"] == "penalty"]
        if not sweep:
            continue
        xs = [s["value"] for s in sweep]
        ys = [max(s["max_error"], 1e-16) for s in sweep]
        ax.plot(xs, ys, marker="o", lw=1.4, label=record["case"])
        plotted = True
    ax.axhline(GATE, color="crimson", ls="--", lw=1.6, label="documented gate 3%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SIPG penalty $\\sigma$ (default 20; stability needs $\\sigma\\geq\\sigma_{\\min}$)")
    ax.set_ylabel("max relative $L_2$ error")
    if plotted:
        ax.set_title("(b) SIPG penalty sabotage at the documented resolution")
        ax.legend(fontsize=8)
    else:
        ax.set_title("(b) no penalty sabotage recorded")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("TensorDG gate-power audit: do the documented 3% tolerances have teeth?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"wrote {os.path.relpath(FIG_PATH, ROOT)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=None,
                        help="case name, or 'all' for every case")
    parser.add_argument("--levels", type=int, default=5,
                        help="maximum number of halvings (default 5)")
    parser.add_argument("--aggregate", action="store_true",
                        help="merge per-case JSON and draw the figure")
    args = parser.parse_args(argv)

    torch.set_grad_enabled(False)
    if args.aggregate:
        aggregate()
        return 0
    if args.case in (None, ""):
        parser.error("pass --case <name>|all or --aggregate")
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        if name not in CASES:
            parser.error(f"unknown case {name}; known: {sorted(CASES)}")
        audit_case(name, max_levels=args.levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())