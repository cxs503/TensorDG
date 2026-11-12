"""Measurement floor of the p-refinement study (closes audit m16499 finding F5).

The published p-refinement benchmark measures the L2 error of a DG solution
against an analytic wave.  As ``p`` grows the error falls towards the roundoff
level of the discrete solution itself, where the number stops being a
discretisation measurement: ``u_h - u`` becomes a difference of two O(1)
float64 numbers.  Audit m16499 flagged p >= 7 as *unmeasured* for exactly this
reason but left the boundary unquantified.

This script quantifies it:

1. For each degree it re-solves the same case (temporally converged, CFL taken
   from the published artifact so the numbers are comparable).
2. It measures the L2 error along three independent paths: the production
   float64 quadrature, a float64 quadrature of a different order, and an
   extended-precision (``np.longdouble``) accumulation with its own Legendre
   recurrence.  The spread between paths bounds the *measurement* error.
3. It compares the measured error with the representable-solution floor
   ``eps64 * ||u_h||_L2`` and reports the signal-to-noise ratio per level.

A level is usable for a decay-rate fit only while ``error > 10 * floor``; the
resulting ``max_resolvable_reduction`` says how many decades of error the
float64 pipeline can certify at all.

Run::

    python -m benchmarks.p_refinement_floor --degrees 5 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import _basis, _sine, l2_error, solve_advection
from tensordg.convergence import (
    FLOAT64_EPS,
    modal_l2_norm_1d,
    resolved_levels,
    roundoff_floor,
)

SCHEMA = "tensordg-p-refinement-measurement-floor/1"
ARTIFACT_RELPATH = Path("docs") / "benchmarks" / "results" / "p-refinement-exponential"
PRIMARY = "p_refinement_results.json"
NCELLS = 16
FINAL_TIME = 1.0
SPEED = 1.0
FALLBACK_CFL = 0.08
SNR_MIN = 10.0


def _legendre_ld(x: np.ndarray, degree: int) -> np.ndarray:
    """Legendre values on ``x`` via a longdouble three-term recurrence."""
    out = [np.ones_like(x)]
    if degree >= 1:
        out.append(x)
    for k in range(2, degree + 1):
        out.append(((2 * k - 1) * x * out[-1] - (k - 1) * out[-2]) / np.longdouble(k))
    return np.stack(out, axis=1)


def l2_error_extended(centers, coeff, final_time: float = FINAL_TIME,
                      speed: float = SPEED, nq: int | None = None) -> float:
    """L2 error accumulated in ``np.longdouble`` (80-bit) instead of float64."""
    values = np.asarray(
        coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff
    ).astype(np.longdouble, copy=False)
    c = np.asarray(
        centers.detach().cpu().numpy() if hasattr(centers, "detach") else centers
    ).astype(np.longdouble, copy=False)
    degree = values.shape[1] - 1
    if nq is None:
        nq = max(2 * degree + 4, 6)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    nodes = nodes.astype(np.longdouble)
    weights = weights.astype(np.longdouble)
    basis = _legendre_ld(nodes, degree)
    uh = values @ basis.T
    h = np.longdouble(1.0) / np.longdouble(c.size)
    x = c[:, None] + np.longdouble(0.5) * h * nodes[None, :]
    two_pi = np.longdouble(2.0) * np.pi
    truth = np.sin(two_pi * (x - np.longdouble(speed * final_time)))
    err2 = np.longdouble(0.5) * h * np.sum(weights[None, :] * (uh - truth) ** 2)
    return float(np.sqrt(err2))


def l2_error_numpy(centers, coeff, final_time: float = FINAL_TIME,
                   speed: float = SPEED, nq: int | None = None) -> float:
    """Same quadrature as :func:`tensordg.advection_1d.l2_error` but in NumPy.

    Used as a third path so that the spread between the torch and NumPy float64
    paths is visible (they share the discretisation but not the arithmetic).
    """
    values = np.asarray(
        coeff.detach().cpu().numpy() if hasattr(coeff, "detach") else coeff
    )
    c = np.asarray(centers.detach().cpu().numpy() if hasattr(centers, "detach") else centers)
    degree = values.shape[1] - 1
    if nq is None:
        nq = max(2 * degree + 4, 6)
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    basis = _legendre_ld(nodes, degree).astype(float)
    uh = values @ basis.T
    h = 1.0 / c.size
    x = c[:, None] + 0.5 * h * nodes[None, :]
    truth = np.sin(2.0 * np.pi * (x - speed * final_time))
    err2 = 0.5 * h * np.sum(weights[None, :] * (uh - truth) ** 2)
    return float(math.sqrt(err2))


def _cfl_for(degree: int, primary: dict | None) -> float:
    """CFL used by the published study for this degree (keeps runs comparable)."""
    if primary:
        for level in primary.get("results", {}).get("levels", []):
            if int(level.get("degree", -1)) == degree:
                return float(level["cfl"])
    return FALLBACK_CFL


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def study(degrees: list[int], ncells: int = NCELLS) -> dict:
    root = Path(__file__).resolve().parent.parent
    outdir = root / ARTIFACT_RELPATH
    primary_path = outdir / PRIMARY
    primary = json.loads(primary_path.read_text()) if primary_path.exists() else None

    levels = []
    for degree in degrees:
        cfl = _cfl_for(degree, primary)
        centers, coeff, meta = solve_advection(
            ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
            cfl=cfl, time_integrator="rk4",
        )
        prod = l2_error(centers, coeff, FINAL_TIME, speed=SPEED)
        np_same = l2_error_numpy(centers, coeff, nq=max(2 * degree + 4, 6))
        np_hi = l2_error_numpy(centers, coeff, nq=4 * degree + 16)
        ext = l2_error_extended(centers, coeff, nq=max(2 * degree + 4, 6))
        ext_hi = l2_error_extended(centers, coeff, nq=4 * degree + 16)
        norm = modal_l2_norm_1d(centers, coeff)
        floor = roundoff_floor(norm)
        spread = max(
            abs(prod - ext), abs(np_hi - ext), abs(np_same - prod), abs(ext_hi - ext),
        )
        levels.append({
            "degree": degree,
            "ncells": ncells,
            "dofs": ncells * (degree + 1),
            "cfl": cfl,
            "dt": meta["dt"],
            "steps": meta["steps"],
            "l2_production_float64": prod,
            "l2_numpy_float64": np_same,
            "l2_numpy_float64_hiorder": np_hi,
            "l2_extended_longdouble": ext,
            "l2_extended_longdouble_hiorder": ext_hi,
            "path_spread_abs": spread,
            "path_spread_rel": spread / ext if ext > 0 else float("inf"),
            "solution_norm_l2": norm,
            "roundoff_floor": floor,
            "snr_measured_over_floor": (prod / floor) if floor > 0 else float("inf"),
            "resolved": bool(prod / floor >= SNR_MIN) if floor > 0 else True,
        })

    errors = [lv["l2_extended_longdouble"] for lv in levels]
    floor = max(lv["roundoff_floor"] for lv in levels)
    resolved = resolved_levels(degrees, errors, floor, snr_min=SNR_MIN)
    published_range = None
    if primary:
        published_range = resolved_levels(
            primary["results"]["degrees"], primary["results"]["errors"], floor,
            snr_min=SNR_MIN,
        )

    # What the published decay rate predicts for the next degrees, and whether
    # that prediction would still be measurable above the floor.
    projection = None
    if primary:
        sigma = float(primary["results"]["diagnostics"]["lsq_decay_rate"])
        p_last = int(levels[-1]["degree"])
        e_last = errors[-1]
        projection = {
            "sigma": sigma,
            "anchor_degree": p_last,
            "anchor_error": e_last,
            "predicted": {
                str(p_last + k): {
                    "predicted_error": e_last * math.exp(-sigma * k),
                    "predicted_snr": (e_last * math.exp(-sigma * k)) / floor,
                }
                for k in (1, 2, 3)
            },
            "first_floor_limited_degree_projection": next(
                (
                    p_last + k
                    for k in (1, 2, 3, 4, 5, 6)
                    if (e_last * math.exp(-sigma * k)) / floor < SNR_MIN
                ),
                None,
            ),
        }

    return {
        "schema": SCHEMA,
        "case": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "question": (
            "At which degree does the float64 L2-error measurement stop resolving "
            "the discretisation error (audit m16499 F5)?"
        ),
        "config": {
            "ncells": ncells,
            "final_time": FINAL_TIME,
            "speed": SPEED,
            "snr_min": SNR_MIN,
            "float64_eps": FLOAT64_EPS,
            "cfl_source": "published artifact levels" if primary else "fallback 0.08",
        },
        "levels": levels,
        "floor": {
            "value": floor,
            "definition": "max over levels of eps64 * ||u_h||_L2",
            "snr_min": SNR_MIN,
            "resolved_degrees": resolved["resolved_degrees"],
            "floor_limited_degrees": resolved["floor_limited_degrees"],
            "max_resolvable_reduction": resolved["max_resolvable_reduction"],
            "published_series": published_range,
            "note": (
                "max_resolvable_reduction is quoted for the degrees measured here; "
                "published_series repeats the test over the full published p ladder"
            ),
        },
        "projection": projection,
        "artifact_provenance": {
            "primary_artifact": PRIMARY,
            "primary_sha256": _sha256(primary_path),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_commit": _git_commit(),
        },
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # pragma: no cover
        return None


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--ncells", type=int, default=NCELLS)
    ap.add_argument("--out", type=str,
                    default=str(ARTIFACT_RELPATH / "measurement_floor.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    degrees = sorted(set(args.degrees))
    t0 = time.time()
    payload = study(degrees, ncells=args.ncells)
    payload["runtime_seconds"] = time.time() - t0

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent.parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"{'p':>3} {'L2 (f64 prod)':>16} {'L2 (longdouble)':>18} {'spread rel':>12} "
          f"{'floor':>10} {'SNR':>10} resolved")
    for lv in payload["levels"]:
        print(f"{lv['degree']:>3} {lv['l2_production_float64']:>16.6e} "
              f"{lv['l2_extended_longdouble']:>18.6e} {lv['path_spread_rel']:>12.2e} "
              f"{lv['roundoff_floor']:>10.3e} {lv['snr_measured_over_floor']:>10.1f} "
              f"{lv['resolved']}")
    print(f"floor = {payload['floor']['value']:.3e}  snr_min = {SNR_MIN}")
    print(f"max resolvable reduction = {payload['floor']['max_resolvable_reduction']:.3e}")
    if payload["projection"]:
        print("projection:", json.dumps(payload["projection"]["predicted"], indent=1))
    print(f"written: {out}")
    return payload


if __name__ == "__main__":
    main()