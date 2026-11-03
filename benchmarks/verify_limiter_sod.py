"""Verification harness for the TVB/WENO shock limiters on the Sod problem.

The limiters were merged without a quantitative pass/fail criterion: the
original development report recorded "accuracy target (errors <3% with
limiter): NOT MET" and diagnosed it as a tuning problem of the TVB constant
``M``.  This harness turns that open verification debt into a reproducible
measurement:

* each configuration is integrated with the *same* solver, mesh, CFL and
  final time, so the only difference is the limiter;
* the reference is the **exact Riemann solution** (analytic, no numerical
  reference is involved), sampled pointwise through the modal polynomial;
* in addition to the discretisation error we record the algebraic *slope
  amplification factor* of the limiter, i.e. the ratio between the
  mode-1 coefficient written by the limiter and the one that the minmod
  slope of the cell averages corresponds to.  That quantity is independent
  of the time integration and exposes scaling bugs directly.

Usage
-----
    PYTHONPATH=. python -m benchmarks.verify_limiter_sod --tag pre_fix

Results are written to ``validation/limiter_sod/<tag>.json``.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.riemann_dg_error import riemann_dg_errors
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.limiters import _minmod, _troubled_cell_indicator

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)
FINAL_TIME = 0.2

# (label, limiter kind or None, TVB constant)
CONFIGS = [
    ("none", None, 1.0),
    ("tvb_M1", "tvb", 1.0),
    ("tvb_M50", "tvb", 50.0),
    ("tvb_M300", "tvb", 300.0),
    ("weno_M1", "weno", 1.0),
]


def slope_amplification(coeff: torch.Tensor, dx: float, tvb_constant: float = 1.0) -> float:
    """Ratio of the limiter-output mode-1 to the mode-1 of a minmod slope.

    The normalised Legendre mode-1 on the reference cell is ``xi``, so a cell
    average slope ``s`` (physical units) is stored as ``c1 = s * dx / 2``.
    A TVD limiter must therefore write ``c1 = minmod(d_left, d_right) / 2``.
    The returned scalar is ``max |c1_limited / (minmod / 2)|`` over the
    cells whose neighbouring averages are not monotone; a correct limiter
    returns a value <= 1, a doubled slope returns ~2.
    """
    avg = coeff[:, 0]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg
    target = _minmod(d_left, d_right) / 2.0
    # Only cells the detector actually flags are constrained; untouched cells
    # legitimately keep their unlimited slope.
    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant)
    floor = 1e-10 * max(1.0, float(avg.abs().max()))
    active = troubled[:, None] & (target.abs() > floor)
    if not bool(active.any()):
        return 0.0
    ratio = (coeff[:, 1].abs() / target.abs())[active]
    return float(ratio.max())


def run_config(kind, tvb_constant, ncells, degree, cfl):
    started = time.time()
    x, coeff, metadata = solve_euler_dg_1d(
        SOD_LEFT,
        SOD_RIGHT,
        ncells=ncells,
        degree=degree,
        final_time=FINAL_TIME,
        cfl=cfl,
        limiter=kind,
        tvb_constant=tvb_constant,
    )
    errors = riemann_dg_errors(x, coeff, FINAL_TIME, SOD_LEFT, SOD_RIGHT)
    dx = 1.0 / ncells
    return {
        "limiter": kind or "none",
        "tvb_constant": tvb_constant,
        "steps": metadata["steps"],
        "wall_s": round(time.time() - started, 1),
        "relative_l2_percent": {
            field: errors[field]["relative_l2_percent"] for field in ("density", "velocity", "pressure")
        },
        "max_relative_l2_percent": max(errors[f]["relative_l2_percent"] for f in ("density", "velocity", "pressure")),
        "slope_amplification": slope_amplification(coeff, dx, tvb_constant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run", help="artifact name under validation/limiter_sod/")
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    rows = []
    for label, kind, tvb_constant in CONFIGS:
        row = run_config(kind, tvb_constant, args.cells, args.degree, args.cfl)
        row["label"] = label
        rows.append(row)
        print(
            f"{label:10s} steps={row['steps']:5d} "
            f"rho={row['relative_l2_percent']['density']:.4f}% "
            f"u={row['relative_l2_percent']['velocity']:.4f}% "
            f"p={row['relative_l2_percent']['pressure']:.4f}% "
            f"max={row['max_relative_l2_percent']:.4f}% "
            f"slope_amp={row['slope_amplification']:.3f} "
            f"{row['wall_s']}s",
            flush=True,
        )

    out = {
        "tag": args.tag,
        "case": "Sod shock tube, exact Riemann reference",
        "left_state": SOD_LEFT,
        "right_state": SOD_RIGHT,
        "final_time": FINAL_TIME,
        "ncells": args.cells,
        "degree": args.degree,
        "cfl": args.cfl,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configs": rows,
        "summary": {
            "best_max_percent": min(r["max_relative_l2_percent"] for r in rows),
            "all_below_three_percent": all(r["max_relative_l2_percent"] < 3.0 for r in rows),
        },
    }
    path = Path("validation/limiter_sod") / f"{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()