"""One-shot limiter diagnostic: separate the limiter *formula* from its use
inside the SSP-RK loop.

The Sod sweep in ``benchmarks/verify_limiter_sod.py`` shows that enabling the
limiter increases the L2 error against the exact Riemann solution, even after
the mode-1 scaling bug is fixed.  Two mechanisms are possible:

1. the limiting *formula* itself is harmful (wrong projector), or
2. the formula is fine but applying it after every SSP-RK substage over-limits
   consistently-consistent stage states.

This script isolates the two by taking the *unlimited* final solution, applying
the limiter to it exactly once, and recomputing the error.  If the one-shot
application reduces the error, the formula is sound and the in-the-loop
application is the problem.

Usage
-----
    PYTHONPATH=. python -m benchmarks.limiter_oneshot --tag post_fix
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.verify_limiter_sod import FINAL_TIME, SOD_LEFT, SOD_RIGHT
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.limiters import _troubled_cell_indicator, apply_limiter

FIELDS = ("density", "velocity", "pressure")


def _errors(x, coeff):
    err = riemann_dg_errors(x, coeff, FINAL_TIME, SOD_LEFT, SOD_RIGHT)
    return {f: err[f]["relative_l2_percent"] for f in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="post_fix")
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--tvb-constant", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    x, coeff, metadata = solve_euler_dg_1d(
        SOD_LEFT,
        SOD_RIGHT,
        ncells=args.cells,
        degree=args.degree,
        final_time=FINAL_TIME,
        cfl=args.cfl,
        limiter=None,
    )
    dx = 1.0 / args.cells
    base = _errors(x, coeff)

    rows = []
    for kind in ("tvb", "weno"):
        limited = apply_limiter(coeff, dx, kind=kind, tvb_constant=args.tvb_constant)
        troubled = _troubled_cell_indicator(coeff, dx, args.tvb_constant)
        # Only troubled cells change, so the perturbation is local.
        if bool(troubled.any()):
            delta = float((limited[troubled, 1:] - coeff[troubled, 1:]).abs().max())
        else:
            delta = 0.0
        rows.append(
            {
                "limiter": kind,
                "errors_on_unlimited_state": _errors(x, limited),
                "troubled_cells": int(troubled.sum()),
                "troubled_cell_fraction": round(float(troubled.sum()) / args.cells, 4),
                "max_mode_perturbation": delta,
            }
        )
        print(f"{kind}: {rows[-1]['errors_on_unlimited_state']} troubled={rows[-1]['troubled_cells']}", flush=True)

    detector = []
    for m in (0.0, 1.0, 50.0, 300.0, 1.0e6):
        flag = _troubled_cell_indicator(coeff, dx, m)
        detector.append({"tvb_constant": m, "flagged_cells": int(flag.sum()),
                         "flagged_fraction": round(float(flag.sum()) / args.cells, 4)})
        print(f"detector M={m:g}: flagged={int(flag.sum())}/{args.cells}", flush=True)

    out = {
        "tag": args.tag,
        "detector_selectivity": detector,
        "note": "limiter applied once to the unlimited final state (mechanism separation)",
        "case": {"left": SOD_LEFT, "right": SOD_RIGHT, "final_time": FINAL_TIME},
        "ncells": args.cells,
        "degree": args.degree,
        "cfl": args.cfl,
        "tvb_constant": args.tvb_constant,
        "steps": metadata["steps"],
        "unlimited_errors": base,
        "one_shot": rows,
        "wall_s": round(time.time() - started, 1),
    }
    path = Path("validation/limiter_sod") / f"oneshot_{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"unlimited: {base}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()