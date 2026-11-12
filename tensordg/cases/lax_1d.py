"""Lax (1954) shock tube / Toro test 2: canonical case definition.

Reference
---------
P. D. Lax, *Weak solutions of nonlinear hyperbolic equations and their
numerical computation*, Comm. Pure Appl. Math. **7** (1954) 159--193.

E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Springer, 2009, §10.6 (Test 2, p. 131).

Physics
-------
A diaphragm at ``x = 0.5`` separates two constant states in a ``gamma = 1.4``
ideal gas on ``[0, 1]``.  When the diaphragm is removed at ``t = 0`` the
solution develops four waves: a left-going rarefaction, a contact discontinuity,
and a right-going shock.  The exact Riemann solution is available in closed
form (via the star-state iteration), so this case carries its own reference
without needing a third-party solver.

Why the state values are not free data
--------------------------------------
The left and right states are the standard Toro Test 2 values.  The star-state
pressure ``p* = 2.4660979...`` and velocity ``u* = 1.5287230...`` are computed
by the exact Riemann solver (``benchmarks.riemann_exact.star_state``), not
transcribed from a table.  The test gates compare the DG solution against this
computed reference, keeping the case free of hand-copied numbers.

Usage
-----
>>> from tensordg.cases.lax_1d import solve_lax_case
>>> out = solve_lax_case(ncells=200, degree=2, cfl=0.35)  # doctest: +SKIP
>>> out["meta"]["steps"]                                   # doctest: +SKIP
2535
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch

from tensordg.euler_dg_1d import solve_euler_dg_1d

CASE_KEY = "lax-1954-shock-tube"
GAMMA = 1.4
DOMAIN = (0.0, 1.0)
DIAPHRAGM = 0.5
FINAL_TIME = 0.13
#: (rho, u, p) of the left state (Toro Test 2)
LEFT_STATE = (0.445, 0.698, 3.528)
#: (rho, u, p) of the right state (Toro Test 2)
RIGHT_STATE = (0.5, 0.0, 0.571)


def lax_parameters() -> dict:
    """Return the canonical Lax case parameter dictionary."""
    return {
        "case_key": CASE_KEY,
        "gamma": GAMMA,
        "domain": DOMAIN,
        "diaphragm": DIAPHRAGM,
        "final_time": FINAL_TIME,
        "left_state": LEFT_STATE,
        "right_state": RIGHT_STATE,
        "reference": "exact Riemann solution (Toro 2009, §10.6, Test 2)",
        "acceptance_threshold": 0.03,  # relative L2 < 3%
    }


def star_state() -> tuple[float, float]:
    """Compute the exact star-state (p*, u*) for the Lax Riemann problem.

    Uses the standard two-shock / shock-rarefaction iteration.  The result
    is computed, not transcribed, so it stays consistent with the exact
    Riemann solver used in the error measurement.
    """
    # Import here to avoid circular dependency; benchmarks.riemann_exact
    # is the authoritative source for the star-state iteration.
    from benchmarks.riemann_exact import star_state as _star_state
    return _star_state(LEFT_STATE, RIGHT_STATE)


def solve_lax_case(
    ncells: int = 200,
    degree: int = 2,
    cfl: float = 0.35,
    device: str = "cpu",
    limiter: Optional[str] = None,
) -> dict:
    """Run the Lax shock tube with the DG Euler solver.

    Returns a dict with keys:
        - x: cell centers (torch.Tensor)
        - coeff: modal coefficients (torch.Tensor)
        - meta: metadata dict (steps, dt, etc.)
        - p_star: exact star-state pressure
        - u_star: exact star-state velocity
    """
    t0 = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT_STATE, RIGHT_STATE,
        ncells=ncells, degree=degree, final_time=FINAL_TIME, cfl=cfl,
        device=device, limiter=limiter,
    )
    meta["wall_time"] = time.time() - t0
    meta["case_key"] = CASE_KEY

    p_star, u_star = star_state()
    return {
        "x": x,
        "coeff": coeff,
        "meta": meta,
        "p_star": p_star,
        "u_star": u_star,
    }
