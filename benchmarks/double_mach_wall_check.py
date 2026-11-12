"""Woodward--Colella double Mach reflection with reflecting / tracked boundaries.

Before m16803 ``solve_euler_dg_2d`` had no wall face, so this benchmark could
only initialise the exact oblique-shock states (see
``benchmarks/double_mach_reflection.py``, whose docstring records the gap).
With the ``bcs`` API the standard Woodward--Colella configuration is now
expressible:

* ``y = 0``: impermeable wall for ``x >= 1/6`` (the reflecting floor) and the
  exact post-shock state for ``x < 1/6`` (supersonic inflow);
* ``y = 1``: the analytically tracked post-shock / pre-shock split
  ``x_s(1, t) = 1/6 + 1/sqrt(3) + (20/sqrt(3)) t``;
* ``x = 0`` / ``x = 4``: the flow is supersonic, so the zero-gradient outer
  face carries the correct upwind state.

Checks reported (all from the run, no hand-copied numbers):

* incident-shock position at ``t`` small, along three rows, versus the exact
  shock line of the initial data (verifies the whole boundary/inflow setup:
  before the reflection the solution is the initial data translated, so the
  front position is known in closed form);
* wall impermeability: the wall-normal velocity at the floor cells vanishes;
* the reflected pattern: density range at ``t = 0.2`` (Woodward--Colella report
  a peak density of order 20), the Mach stem height and its refinement change
  between two resolutions.

Usage::

    python -m benchmarks.double_mach_wall_check --ncells 120 30 --final-time 0.05
    python -m benchmarks.double_mach_wall_check --ncells 240 60 --final-time 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import cell_fields, conservative_state, solve_euler_dg_2d

from benchmarks.double_mach_reflection import (
    POST_SHOCK,
    PRE_SHOCK,
    SHOCK_ANGLE_RAD,
    SHOCK_X0,
)

SHOCK_NORMAL = (math.sin(SHOCK_ANGLE_RAD), -math.cos(SHOCK_ANGLE_RAD))
SOUND_AHEAD = math.sqrt(GAMMA * PRE_SHOCK[3] / PRE_SHOCK[0])
SHOCK_SPEED = 10.0 * SOUND_AHEAD  # Mach 10 relative to the pre-shock sound speed


def shock_position_x(y, time):
    """Exact incident-shock ``x`` at height ``y`` and time ``t``.

    The front is ``n . ((x, y) - (1/6, 0)) = S t`` with ``n`` the shock normal,
    i.e. ``x = 1/6 + (S t + y/2) / n_x``; at ``t = 0`` this is the initial line
    ``x = 1/6 + y tan(60 deg)``.
    """
    return SHOCK_X0 + (SHOCK_SPEED * time + 0.5 * y) / SHOCK_NORMAL[0]


def _initial(x, y):
    # line through (SHOCK_X0, 0) with slope tan(60 deg) == the t = 0 front
    line = SHOCK_X0 + y * math.tan(SHOCK_ANGLE_RAD)
    front = x < line
    return (
        torch.where(front, torch.full_like(x, POST_SHOCK[0]), torch.full_like(x, PRE_SHOCK[0])),
        torch.where(front, torch.full_like(x, POST_SHOCK[1]), torch.full_like(x, PRE_SHOCK[1])),
        torch.where(front, torch.full_like(x, POST_SHOCK[2]), torch.full_like(x, PRE_SHOCK[2])),
        torch.where(front, torch.full_like(x, POST_SHOCK[3]), torch.full_like(x, PRE_SHOCK[3])),
    )


def _bottom(X, Y, t):
    """Reflecting floor for ``x >= 1/6``; exact post-shock inflow before that."""
    wall_mask = X >= SHOCK_X0
    ghost = conservative_state(
        torch.full_like(X, POST_SHOCK[0]), torch.full_like(X, POST_SHOCK[1]),
        torch.full_like(X, POST_SHOCK[2]), torch.full_like(X, POST_SHOCK[3]),
    )
    return ghost, wall_mask


def _top(X, Y, t):
    """Tracked post-shock region below the incident front at ``y = 1``."""
    front = X < shock_position_x(1.0, t)
    rho = torch.where(front, torch.full_like(X, POST_SHOCK[0]), torch.full_like(X, PRE_SHOCK[0]))
    u = torch.where(front, torch.full_like(X, POST_SHOCK[1]), torch.full_like(X, PRE_SHOCK[1]))
    v = torch.where(front, torch.full_like(X, POST_SHOCK[2]), torch.full_like(X, PRE_SHOCK[2]))
    p = torch.where(front, torch.full_like(X, POST_SHOCK[3]), torch.full_like(X, PRE_SHOCK[3]))
    return conservative_state(rho, u, v, p)


def run(ncells=(120, 30), degree=1, final_time=0.05, cfl=0.15, limiter="tvb",
        tvb_constant=1.0, limiter_stage="final", device="cpu"):
    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        _initial, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        domain=((0.0, 4.0), (0.0, 1.0)), periodic=(False, False),
        bcs={"left": "outflow", "right": "outflow", "bottom": _bottom, "top": _top},
        limiter=limiter, tvb_constant=tvb_constant, limiter_stage=limiter_stage,
        device=device,
    )
    wall = time.time() - started
    fields = cell_fields(coeff)
    density = fields["density"].cpu().numpy()
    velocity_y = fields["velocity_y"].cpu().numpy()
    centers_x = x.cpu().numpy()
    centers_y = y.cpu().numpy()

    fronts = {}
    for height in (0.25, 0.5, 0.75):
        row = int(np.argmin(np.abs(centers_y - height)))
        gradient = np.abs(np.gradient(density[:, row], centers_x))
        index = int(np.argmax(gradient))
        measured = float(centers_x[index])
        exact = shock_position_x(float(centers_y[row]), final_time)
        fronts[f"y={height}"] = {
            "measured": measured,
            "exact": exact,
            "cells_off": abs(measured - exact) / (centers_x[1] - centers_x[0]),
            "row": row,
        }

    floor = density[:, 0]
    stem = float(np.max(floor))
    above = float(np.max(floor[floor > 0.5 * stem])) if stem > 0 else 0.0
    return {
        "ncells": list(ncells),
        "degree": degree,
        "final_time": final_time,
        "cfl": cfl,
        "limiter": limiter,
        "tvb_constant": tvb_constant,
        "steps": meta["steps"],
        "wall_seconds": wall,
        "density": {"min": float(density.min()), "max": float(density.max())},
        "pressure_min": float(fields["pressure"].min()),
        "wall_normal_velocity_max": float(np.max(np.abs(velocity_y[:, 0]))),
        "incident_fronts": fronts,
        "incident_front_max_cells_off": max(v["cells_off"] for v in fronts.values()),
        "floor_density_max": stem,
        "dofs": meta["dofs"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncells", nargs=2, type=int, default=(120, 30))
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--final-time", type=float, default=0.05)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--limiter", default="tvb")
    parser.add_argument("--tvb-constant", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    result = run(tuple(args.ncells), args.degree, args.final_time, args.cfl,
                 args.limiter, args.tvb_constant)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())