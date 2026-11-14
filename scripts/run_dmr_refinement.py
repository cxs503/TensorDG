"""Mesh-refinement verification of the double Mach reflection quantitative gates.

Anchor ``REPRO-DG-CKS1998`` (Cockburn & Shu, JCP 141 (1998) 199--224, Part V).

Why this exists
---------------
``docs/CASE-REPRO-DG-CKS1998-DOUBLE-MACH.md`` (m20397) measured, on a single
64 x 16 P1 mesh at t = 0.02, two quantities that came out *outside* the 3 %
standard and attributed them to coarse-grid shock smearing plus a sampling
point that had drifted into the reflected structure:

    post-shock platform  rho  -21.15 %,  p  -26.37 %,  |V|  -3.62 %
    front position       up to 3.64 cells (24.7 %) at y = 0.906

That attribution was an *assumption*, not a measurement.  This script measures
the same quantities on a resolution ladder so the attribution can be confirmed
or falsified, and adds two quantities that separate the candidate mechanisms:

``plateau_*``
    mean of rho / p over a window that is far behind the analytic front and far
    from the wall (``x in [0.25, 0.45]``, ``y in [0.55, 0.90]`` at t = 0.02).
    The exact solution there is the *constant* Rankine--Hugoniot post-shock
    state, which a DG method reproduces to round-off once the jump is resolved.
    A large error here cannot be blamed on smearing; it would point at the
    limiter or at a boundary / initial-projection defect.

``smear_width_cells``
    width of the captured shock at ``y = 0.70`` measured between the 10 % and
    90 % density crossings, in cells.  If the -21 % offset sample is a smearing
    artefact, this width must scale like ``h`` (constant in cells) while the
    fixed-*physical*-offset sample error falls as ``h``.

``undisturbed_l1_rel``
    relative L1 error of the density over every point that is at least six
    cells away from the analytic front and above ``y = 0.35`` (i.e. where the
    exact solution is still the two-state piecewise constant).  This is the one
    norm-based convergence statement the case admits at this time.

Every number lands in ``metrics.json``; nothing is hand-copied.

Example
-------
    python scripts/run_dmr_refinement.py --nx 384 --ny 96 --degree 1 \
        --time 0.02 --out validation/dmr/refinement/p1_384x96
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensordg.euler_dg_2d import cell_fields
from benchmarks.double_mach_reflection import (
    DENSITY_LEVEL, POST_SHOCK, PRE_SHOCK, analytic_front_x, rh_residuals,
    solve_double_mach, _interp_crossing,
)

GAMMA = 1.4
POST_RHO = float(POST_SHOCK[0])
POST_P = float(POST_SHOCK[3])
POST_SPEED = float(np.hypot(POST_SHOCK[1], POST_SHOCK[2]))
PRE_RHO = float(PRE_SHOCK[0])

#: heights at which the undisturbed part of the incident front is located
FRONT_Y_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90)
#: row used for the shock-width / profile measurements
PROFILE_Y = 0.70
#: windows where the exact solution is the constant post-shock state.  ``mid``
#: is far from every boundary; ``top`` deliberately reaches the row 2-4 cells
#: below ``y = 1`` so that top-boundary contamination is measured, not hidden.
PLATEAU_X = (0.25, 0.45)
PLATEAU_WINDOWS = {"mid": (0.55, 0.75), "top": (0.80, 0.95),
                   "full": (0.55, 0.90)}


def _index(xc, target):
    return int(np.argmin(np.abs(xc - target)))


def _crossing(xc, prof, level, side="right"):
    """Sub-cell crossing of ``level``.

    ``side="right"`` scans from ``x = +inf`` (used for the incident front and
    for the 10 % point, whose exact value lies further left); ``side="left"``
    scans from ``x = -inf`` (used for the 90 % point, whose exact value lies
    further right than the captured profile's start).
    """
    if side == "right":
        return _interp_crossing(xc, prof, level)
    diff = prof - level
    for i in range(0, len(diff) - 1):
        if diff[i] > 0.0 >= diff[i + 1]:
            w = diff[i] - diff[i + 1]
            frac = diff[i] / w if w > 0 else 0.0
            return 0.5 * (xc[i] + xc[i + 1]) + (frac - 0.5) * (xc[i + 1] - xc[i])
    return None


def measure(x, y, coeff, t, meta, degree, dx, cfl, limiter, wall_seconds):
    """All refinement metrics of one run (programmatic, no hand-copied numbers)."""
    fields = cell_fields(coeff)
    rho = np.asarray(fields["density"].cpu().numpy())
    ux = np.asarray(fields["velocity_x"].cpu().numpy())
    uy = np.asarray(fields["velocity_y"].cpu().numpy())
    p = np.asarray(fields["pressure"].cpu().numpy())
    xc = np.asarray(x).ravel()
    yc = np.asarray(y).ravel()

    # --- incident front: measured x vs exact plane-front law, in cells ------
    j_last = len(xc) - 1
    front, exact, err, err_cells = {}, {}, {}, {}
    for target in FRONT_Y_LEVELS:
        j = _index(yc, target)
        prof = rho[:, j]
        # a real front must exist on this row; otherwise record null honestly
        crossing = _crossing(xc, prof, DENSITY_LEVEL, side="right")
        exact_v = float(analytic_front_x(yc[j], t))
        front[float(yc[j])] = None if crossing is None else float(crossing)
        exact[float(yc[j])] = exact_v
        err[float(yc[j])] = None if crossing is None else float(crossing) - exact_v
        err_cells[float(yc[j])] = (None if crossing is None
                                   else (float(crossing) - exact_v) / dx)

    # --- captured shock width at PROFILE_Y (10 % / 90 % of the density jump)
    j_p = _index(yc, PROFILE_Y)
    prof = rho[:, j_p]
    lo = PRE_RHO + 0.1 * (POST_RHO - PRE_RHO)
    hi = PRE_RHO + 0.9 * (POST_RHO - PRE_RHO)
    x_lo = _crossing(xc, prof, lo, side="right")
    x_hi = _crossing(xc, prof, hi, side="left")
    width_phys = None if (x_lo is None or x_hi is None) else float(x_lo - x_hi)
    width_cells = None if width_phys is None else width_phys / dx

    # --- plateau far behind the front: exact solution is the constant state -
    mi = (xc >= PLATEAU_X[0]) & (xc <= PLATEAU_X[1])
    plateaus, plateau_rel = {}, {}
    for key, (ylo, yhi) in PLATEAU_WINDOWS.items():
        mj = (yc >= ylo) & (yc <= yhi)
        mean_rho = float(rho[np.ix_(mi, mj)].mean())
        mean_p = float(p[np.ix_(mi, mj)].mean())
        plateaus[key] = {"window_x": list(PLATEAU_X),
                         "window_y": [ylo, yhi],
                         "cells": int(mi.sum() * mj.sum()),
                         "rho": mean_rho, "p": mean_p}
        plateau_rel[key] = {"rho": (mean_rho - POST_RHO) / POST_RHO,
                            "p": (mean_p - POST_P) / POST_P}

    # --- per-row contamination profile --------------------------------------
    # For every row well above the wall, the density error of a sample placed a
    # fixed number of *cells* behind (a) the analytic front and (b) the
    # *measured* front.  (a) - (b) isolates the front-position error, which is
    # what the m20397 report attributed to shock smearing.
    rows, err_analytic, err_measured = [], [], []
    for j, yv in enumerate(yc):
        if yv < 0.40:
            continue
        xa = float(analytic_front_x(yv, t))
        ia = _index(xc, xa - 8.0 * dx)
        xm = _crossing(xc, rho[:, j], DENSITY_LEVEL, side="right")
        measured_row = None
        if xm is not None:
            im = _index(xc, xm - 8.0 * dx)
            measured_row = float(rho[im, j])
        rows.append(float(yv))
        err_analytic.append(float((rho[ia, j] - POST_RHO) / POST_RHO))
        err_measured.append(None if measured_row is None
                            else float((measured_row - POST_RHO) / POST_RHO))

    # --- samples behind the incident front at fixed offsets -----------------
    samples = {}
    for offset_cells, offset_phys in ((5.0, None), (8.0, None), (None, 0.10)):
        off = (offset_cells * dx) if offset_cells is not None else offset_phys
        x_target = float(analytic_front_x(PROFILE_Y, t)) - off
        i = _index(xc, x_target)
        st = {"x": float(xc[i]), "y": float(yc[j_p]),
              "rho": float(rho[i, j_p]), "u": float(ux[i, j_p]),
              "v": float(uy[i, j_p]), "p": float(p[i, j_p])}
        speed = float(np.hypot(st["u"], st["v"]))
        key = f"{offset_cells:g}cells" if offset_cells is not None else f"{offset_phys:g}phys"
        samples[key] = {
            "offset": off, "offset_cells": off / dx, "measured": st,
            "rel_error": {"rho": (st["rho"] - POST_RHO) / POST_RHO,
                          "p": (st["p"] - POST_P) / POST_P,
                          "speed": (speed - POST_SPEED) / POST_SPEED},
        }

    # --- L1 error of density where the exact solution is piecewise constant -
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    dist = X - analytic_front_x(Y, t)          # > 0 ahead (pre-shock side)
    mask = (np.abs(dist) > 6.0 * dx) & (Y > 0.35)
    exact_rho = np.where(dist > 0.0, PRE_RHO, POST_RHO)
    l1 = float(np.abs(rho - exact_rho)[mask].sum())
    l1_ref = float(exact_rho[mask].sum())
    undisturbed = {"mask_cells": int(mask.sum()),
                   "l1_abs": l1, "l1_rel": l1 / l1_ref,
                   "l1_rel_normalised": l1 / l1_ref / 1.0}

    return {
        "case": "double_mach_reflection_refinement",
        "anchor": "REPRO-DG-CKS1998",
        "grid": [int(len(xc)), int(len(yc))],
        "degree": degree,
        "final_time": t,
        "cfl": cfl,
        "limiter": limiter,
        "steps": meta.get("steps"),
        "wall_seconds": wall_seconds,
        "dx": dx,
        "front_x": front,
        "front_x_exact": exact,
        "front_error": err,
        "front_error_cells": err_cells,
        "max_front_error_cells": max(abs(v) for v in err_cells.values()
                                     if v is not None),
        "smear_width_phys": width_phys,
        "smear_width_cells": width_cells,
        "plateau": plateaus,
        "plateau_rel_error": plateau_rel,
        "row_profile": {"y": rows, "rel_error_from_analytic_front": err_analytic,
                        "rel_error_from_measured_front": err_measured,
                        "offset_cells_behind_front": 8.0},
        "samples_behind_front": samples,
        "undisturbed_l1": undisturbed,
        "density_min": float(rho.min()),
        "density_max": float(rho.max()),
        "rh_residuals": {k: float(v) for k, v in
                         rh_residuals(PRE_SHOCK, POST_SHOCK).items()},
        "profile_y": float(yc[j_p]),
        "profile_x": [float(v) for v in xc],
        "profile_rho": [float(v) for v in prof],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--time", type=float, default=0.02)
    ap.add_argument("--cfl", type=float, default=0.15)
    ap.add_argument("--limiter", default="tvb")
    ap.add_argument("--tvb-constant", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    limiter = None if args.limiter == "none" else args.limiter
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    x, y, coeff, meta = solve_double_mach(
        ncells=(args.nx, args.ny), degree=args.degree,
        final_time=args.time, cfl=args.cfl, limiter=limiter,
        tvb_constant=args.tvb_constant, limiter_stage="final",
    )
    wall = time.time() - t0
    metrics = measure(x, y, coeff, args.time, meta, args.degree,
                      4.0 / args.nx, args.cfl, args.limiter, wall)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()