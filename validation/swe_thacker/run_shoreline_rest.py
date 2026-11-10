#!/usr/bin/env python3
"""Rest state of a lake with a *dry bank* — shoreline positivity filter validation.

Scenario
--------
A still lake (``eta = h + z_b = eta0``, ``u = 0``) whose surface cuts the bed,
so that part of the domain is dry.  Two geometries are used:

``parabolic_hump``
    ``z_b(x) = 1.5 - 0.5 x^2`` on ``[-2, 2]`` with ``eta0 = 1 m``.  The crest
    (1.5 m) rises above the still-water level, the shoreline sits at
    ``x = ±sqrt(2) = 1.41421…``, i.e. *inside* a cell for the meshes used here.

``tent_on_face``
    ``z_b(x) = |x| / 1.5`` with ``eta0 = 1 m``: a piecewise-linear tent whose
    shoreline ``x = ±1.5`` falls **exactly on a cell face** for ``dx = 1/16``
    (``N = 64`` on ``[-2, 2]``), the case the previous filter could not hold.

Why this is not covered by ``run_hydrostatic_wb.py``
----------------------------------------------------
The Thacker basin of ``run_hydrostatic_wb.py`` is *wet everywhere*: every cell
has a strictly positive depth, so the positivity filter never fires and the
well-balanced cancellation is never tested against a wet/dry trace.

The defect
----------
``_positivity_filter`` used to test ``h(x_q) > h_min`` (``1e-12``) at every
quadrature and interface point and, on failure, replaced the whole cell by its
cell average.  A shoreline cell carries ``h = 0`` **exactly** at its dry node,
so it failed that test even though it is perfectly admissible; zeroing its
first mode destroys ``eta = h + z_b = const`` *in that cell*, the reconstructed
surface jumps by half the cell's bed drop, and the resulting imbalance drives a
spurious current that lasts for the whole run (the initial condition is no
longer a fixed point).

Measured before the fix (``N = 64``, ``t = 1 s``, this host): ``max|eta-eta0|``
over wet cells ``3.0e-2 m`` (P1) and ``5.0e-1 m`` (P2), ``max|u|`` ``0.54 m/s``
(P1) and ``36 m/s`` (P2).

The fix
-------
``_positivity_filter`` now uses a Zhang & Shu (2010) style *scaling*: the higher
modes are multiplied by ``theta = min_q h_bar / (h_bar - h_q)`` clipped to
``[0, 1]`` (``1`` when no point is negative), the cell average is untouched
(conservation) and admissible cells pass through unchanged (so a rest state
stays a rest state).  This script measures the resulting drift and records the
before/after contrast in one artifact.

References
----------
* Audusse et al. (2004), "A fast and stable well-balanced scheme with
  hydrostatic reconstruction for shallow water flows", SIAM J. Sci. Comput.
* Zhang & Shu (2010), "On positivity-preserving high order discontinuous
  Galerkin schemes for compressible Euler equations on rectangular meshes",
  J. Comput. Phys. 229, 8918-8934.
* Xing & Shu (2011), "Positivity-preserving high order well-balanced
  discontinuous Galerkin methods for the shallow water equations",
  Adv. Water Resour. 34, 1500-1513.

Usage
-----
    PYTHONPATH=. python validation/swe_thacker/run_shoreline_rest.py
    # optional: --out <path>, --final-time <s>, --quick (skip the coarsest runs)

The JSON written next to this file is guarded by
``tests/test_swe_shoreline_rest_artifact.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tensordg import swe_1d
from tensordg.swe_1d import DRY_DEPTH, reconstruct_fields, solve_swe_1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hydrostatic_wb import collect_provenance  # noqa: E402  (same directory)

GENERATOR = "validation/swe_thacker/run_shoreline_rest.py"
OUT_DEFAULT = Path(__file__).resolve().parent / "shoreline_rest_state_results.json"

ETA0 = 1.0
DOMAIN = (-2.0, 2.0)
CFL = 0.3


# --------------------------------------------------------------------------- #
# geometries
# --------------------------------------------------------------------------- #
def parabolic_hump(x: torch.Tensor) -> torch.Tensor:
    """``z_b = 1.5 - 0.5 x^2``: crest above the still-water level."""
    return 1.5 - 0.5 * x ** 2


def tent(x: torch.Tensor) -> torch.Tensor:
    """``z_b = |x| / 1.5``: piecewise-linear, shoreline exactly at ``x = ±1.5``."""
    return x.abs() / 1.5


def _initial_depth(bed):
    def initial_h(x: torch.Tensor) -> torch.Tensor:
        return (ETA0 - bed(x)).clamp_min(0.0)
    return initial_h


def _zero_u(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def shoreline_on_face(ncells: int, x_shore: float) -> bool:
    """True when ``x_shore`` coincides with a cell face of the uniform mesh."""
    dx = (DOMAIN[1] - DOMAIN[0]) / ncells
    k = (x_shore - DOMAIN[0]) / dx
    return abs(k - round(k)) < 1e-12


# --------------------------------------------------------------------------- #
# pre-fix filter, kept only for the before/after contrast
# --------------------------------------------------------------------------- #
def _legacy_positivity_filter(coeff: torch.Tensor, check_points: torch.Tensor,
                              h_min: float = 1e-12,
                              well_balanced: bool = False) -> torch.Tensor:
    """The replace-with-cell-average filter that this change removed.

    Reproduced verbatim (``h > h_min`` pointwise test, average-only fallback)
    so the artifact can quote a *measured* before/after contrast instead of a
    remembered number.  Never used by :func:`solve_swe_1d` -- and note that this
    *is* the action the default (non-hydrostatic) path still takes: the scaling
    limiter is only active where the Audusse reconstruction runs, see
    ``swe_1d._positivity_filter``.  The signature mirrors the current one so the
    patch fits the solver's call site.
    """
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    h_vals = values[..., 0]
    valid = (h_vals > h_min) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=1)
    average_only = torch.zeros_like(coeff)
    average_only[:, 0] = coeff[:, 0]
    return torch.where(valid_cell[:, None, None], coeff, average_only)


def _solve(bed, degree: int, ncells: int, final_time: float, filter_fn=None,
           keep_profile: bool = False):
    """Solve one rest-state case, optionally with a patched positivity filter."""
    original = swe_1d._positivity_filter
    if filter_fn is not None:
        swe_1d._positivity_filter = filter_fn
    try:
        x, coeff, zb_faces, meta = solve_swe_1d(
            _initial_depth(bed), _zero_u, bed,
            ncells=ncells, degree=degree, final_time=final_time,
            domain=DOMAIN, cfl=CFL, rk_order=3, well_balanced=True)
    finally:
        swe_1d._positivity_filter = original
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    wet = h > DRY_DEPTH
    record = {
        "degree": degree,
        "ncells": ncells,
        "dt_last": float(meta["dt"]),
        "steps": int(meta["steps"]),
        "reached_final_time": bool(meta["reached_final_time"]),
        # the exact solution is the rest state, so the error is measured
        # against the analytic value eta0 / u = 0 (no reference run needed)
        "eta_error_max": float((eta[wet] - ETA0).abs().max()),
        "u_max": float(u.abs().max()),
        "h_min": float(h.min()),
        "dry_cell_count": int((~wet).sum()),
        "reconstruction": "zb_center=zb_modes[:,0] (cell-average bed)",
        "diverged": not bool(meta["reached_final_time"]),
    }
    if keep_profile:
        # cell-centre surface and depth, so the artifact carries the field
        # itself (a resting surface vs. the jagged one of the old filter)
        record["x"] = [round(float(v), 8) for v in x]
        record["eta_profile"] = [round(float(v), 12) for v in eta]
        record["h_profile"] = [round(float(v), 12) for v in h]
    return record


def _case(name: str, bed, description: str, x_shore: float,
          mesh: list, final_time: float) -> dict:
    runs = []
    for degree, ncells in mesh:
        t0 = time.time()
        run = _solve(bed, degree, ncells, final_time,
                     keep_profile=(ncells == 64))
        run["wall_time_s"] = round(time.time() - t0, 3)
        runs.append(run)
        print(f"  {name} P{degree} N={ncells}: "
              f"eta={run['eta_error_max']:.3e} |u|={run['u_max']:.3e} "
              f"({run['wall_time_s']:.1f} s)", flush=True)
    return {
        "name": name,
        "description": description,
        "shoreline_x": x_shore,
        "shoreline_on_face": {
            str(ncells): shoreline_on_face(ncells, x_shore) for _, ncells in mesh
        },
        "eta0": ETA0,
        "domain": list(DOMAIN),
        "cfl": CFL,
        "final_time": final_time,
        "runs": runs,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true",
                        help="skip the coarsest mesh (fast smoke run)")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    mesh = [(1, 32), (1, 64), (1, 128), (2, 32), (2, 64), (2, 128)]
    if args.quick:
        mesh = [(1, 64), (2, 64)]
    contrast_mesh = [(1, 64), (2, 64)]

    print("rest-state cases (fixed point of the well-balanced scheme):", flush=True)
    cases = [
        _case("parabolic_hump", parabolic_hump,
              "z_b = 1.5 - 0.5 x^2, eta0 = 1 m: shoreline at x = +-sqrt(2) "
              "inside a cell", 2.0 ** 0.5, mesh, args.final_time),
        _case("tent_on_face", tent,
              "z_b = |x|/1.5, eta0 = 1 m: shoreline at x = +-1.5 exactly on a "
              "cell face for N = 64", 1.5, mesh, args.final_time),
    ]

    print("before/after contrast with the pre-fix filter:", flush=True)
    legacy = []
    for name, bed in (("parabolic_hump", parabolic_hump), ("tent_on_face", tent)):
        for degree, ncells in contrast_mesh:
            t0 = time.time()
            run = _solve(bed, degree, ncells, args.final_time,
                         filter_fn=_legacy_positivity_filter,
                         keep_profile=True)
            run["filter"] = "legacy_flatten_to_cell_average"
            run["wall_time_s"] = round(time.time() - t0, 3)
            legacy.append({"case": name, **run})
            print(f"  {name} P{degree} N={ncells}: "
                  f"eta={run['eta_error_max']:.3e} |u|={run['u_max']:.3e}",
                  flush=True)

    eta_tol, u_tol = 1e-8, 1e-8
    fixed = [run for case in cases for run in case["runs"]]
    worst_eta = max(r["eta_error_max"] for r in fixed)
    worst_u = max(r["u_max"] for r in fixed)
    worst_legacy_eta = max(r["eta_error_max"] for r in legacy)
    worst_legacy_u = max(r["u_max"] for r in legacy)
    # The pre-fix filter does not merely lose accuracy on one of the cases, it
    # blows up (the surface error feeds the source term every step).  The
    # improvement factor is therefore quoted against the worst pre-fix run that
    # still reached the requested final time; the diverged run is recorded
    # separately so the factor is not inflated by an overflow.
    legacy_converged = [r for r in legacy if not r["diverged"]]

    provenance = collect_provenance()
    provenance["generator"] = GENERATOR
    provenance["command"] = f"python {GENERATOR}"
    provenance["torch_num_threads"] = 1

    payload = {
        "provenance": provenance,
        "test_description": (
            "Rest state of a still lake with a dry bank: hydrostatic "
            "well-balancing across a wet/dry shoreline after the positivity "
            "filter was changed from replace-by-cell-average to a "
            "Zhang-Shu scaling limiter"
        ),
        "exact_solution": {"eta": ETA0, "u": 0.0},
        "error_definition": (
            "max|eta - eta0| over cells with h > DRY_DEPTH (dry cells have no "
            "water surface, their eta = z_b is not part of the solution) and "
            "max|u| over the whole field"
        ),
        "dry_depth": DRY_DEPTH,
        "cases": cases,
        "legacy_filter_contrast": legacy,
        "summary": {
            "pass_criteria": {"eta_error_max": "< 1e-8 m", "u_max": "< 1e-8 m/s"},
            "actual_results": {"eta_error_max": worst_eta, "u_max": worst_u},
            "legacy_filter_results": {
                "eta_error_max": worst_legacy_eta, "u_max": worst_legacy_u,
                "diverged_runs": [f"{r['case']} P{r['degree']} N={r['ncells']}"
                                  for r in legacy if r["diverged"]],
            },
            "legacy_filter_converged": {
                "eta_error_max": max(r["eta_error_max"] for r in legacy_converged),
                "u_max": max(r["u_max"] for r in legacy_converged),
            },
            "improvement_factor": {
                "eta_error_max": (max(r["eta_error_max"] for r in legacy_converged)
                                  / worst_eta if worst_eta > 0 else None),
                "u_max": (max(r["u_max"] for r in legacy_converged)
                          / worst_u if worst_u > 0 else None),
                "basis": "worst pre-fix run that reached final_time vs worst fixed run",
            },
            "pass": bool(worst_eta < eta_tol and worst_u < u_tol),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}  pass={payload['summary']['pass']} "
          f"(worst eta {worst_eta:.3e} m, worst |u| {worst_u:.3e} m/s)", flush=True)
    return 0 if payload["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())