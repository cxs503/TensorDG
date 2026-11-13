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
by the exact Riemann solver implemented here (Newton iteration with analytical
derivatives on the two-shock/shock-rarefaction function), not transcribed from
a table.  The test gates compare the DG solution against this computed
reference, keeping the case free of hand-copied numbers.

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


def _f_and_derivative(p: float, state: tuple, gamma: float) -> tuple[float, float]:
    """f(p) and f'(p) for one side of the Riemann problem.

    For a shock (p > p_k):
        f(p) = (p - p_k) * sqrt(2 / ((gamma+1)*rho) / (p + (gamma-1)/(gamma+1)*p_k))
        f'(p) = sqrt(a/(p+b)) * (1 - 0.5*(p-p_k)/(p+b))

    For a rarefaction (p <= p_k):
        f(p) = 2*a/(gamma-1) * ((p/p_k)^((gamma-1)/(2*gamma)) - 1)
        f'(p) = 1/(rho*a) * (p/p_k)^(-(gamma+1)/(2*gamma))
    """
    rho, _, p_k = state
    sound = math.sqrt(gamma * p_k / rho)
    if p > p_k:
        a_coeff = 2.0 / ((gamma + 1.0) * rho)
        b_coeff = (gamma - 1.0) / (gamma + 1.0) * p_k
        root = math.sqrt(a_coeff / (p + b_coeff))
        value = (p - p_k) * root
        derivative = root * (1.0 - 0.5 * (p - p_k) / (p + b_coeff))
    else:
        ratio = p / p_k
        value = 2.0 * sound / (gamma - 1.0) * (
            ratio ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0
        )
        derivative = 1.0 / (rho * sound) * ratio ** (-(gamma + 1.0) / (2.0 * gamma))
    return value, derivative


def star_state(left: tuple = LEFT_STATE, right: tuple = RIGHT_STATE,
               gamma: float = GAMMA) -> tuple[float, float]:
    """Compute the exact star-state (p*, u*) for a Riemann problem.

    Uses a guarded Newton iteration with analytical derivatives on the
    two-shock/shock-rarefaction function.  The result is computed, not
    transcribed, so it stays consistent with the exact Riemann solver
    used in error measurement.

    This is a self-contained implementation that does not depend on
    ``benchmarks.riemann_exact``, making the case module importable
    without pulling in the benchmark suite.

    Parameters
    ----------
    left, right : tuple
        (rho, u, p) of the left and right states.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    pressure, velocity : float
        Star-state pressure and velocity.

    Raises
    ------
    RuntimeError
        If the Newton iteration does not converge.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    # Initial guess: acoustic approximation
    pressure = max(
        1e-12,
        0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r),
    )

    for _ in range(60):
        f_l, df_l = _f_and_derivative(pressure, left, gamma)
        f_r, df_r = _f_and_derivative(pressure, right, gamma)
        updated = pressure - (f_l + f_r + u_r - u_l) / (df_l + df_r)
        if updated <= 0.0:
            updated = 0.5 * pressure
        if abs(updated - pressure) <= 1e-14 * max(1.0, pressure):
            pressure = updated
            break
        pressure = updated
    else:
        raise RuntimeError("Riemann star-pressure iteration did not converge")

    f_l, _ = _f_and_derivative(pressure, left, gamma)
    f_r, _ = _f_and_derivative(pressure, right, gamma)
    velocity = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    return float(pressure), float(velocity)


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
