"""Verification of the impulsively started Couette wall transient (m3661).

The analytic reference is the Stokes series of the first problem in a channel
(:func:`tensordg.ns_dg_2d.impulsive_couette_exact`).  Three layers of evidence:

1. the series itself is checked against an **independent** finite-difference
   solution of ``u_t = nu u_yy`` (no DG code involved);
2. the series satisfies its own boundary/initial data and bounds;
3. the DG wall path reproduces it below the 3 % gate at ``P1/8`` cells -- the
   case that used to be "reaches t = 0.05, accuracy unknown".

Numbers behind the assertions: ``benchmarks/verify_ns_impulse_couette.py`` ->
``validation/ns_dg_2d_impulse/impulse.json`` and
``docs/reports/tensordg-ns-impulse-transient.md``.
"""
import numpy as np
import pytest
import scipy.linalg as sla
import torch

import tensordg.ns_dg_2d as ns
from benchmarks.verify_ns_impulse_couette import run_case

U = 0.3
NU = 0.05
T_END = 0.2


def _finite_difference_reference(time, nodes=401, dt=2e-4, speed=U, nu=NU):
    """Implicit-Euler solution of ``u_t = nu u_yy`` with the impulsive start."""
    dy = 1.0 / (nodes - 1)
    r = nu * dt / dy ** 2
    matrix = (np.diag(np.full(nodes, 1.0 + 2.0 * r))
              + np.diag(-r * np.ones(nodes - 1), 1)
              + np.diag(-r * np.ones(nodes - 1), -1))
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    matrix[-1, :] = 0.0
    matrix[-1, -1] = 1.0
    lu = sla.lu_factor(matrix)
    u = np.zeros(nodes)
    for _ in range(int(round(time / dt))):
        u = sla.lu_solve(lu, u)
        u[-1] = speed
    return np.linspace(0.0, 1.0, nodes), u


def test_series_matches_an_independent_finite_difference_solve():
    """The reference series is not self-certifying: it must agree with a
    black-box finite-difference solve of the same initial-boundary problem."""
    y, reference = _finite_difference_reference(T_END)
    series = ns.impulsive_couette_exact(
        torch.as_tensor(y), T_END, wall_speed=U, kinematic_viscosity=NU
    ).numpy()
    relative = np.linalg.norm(series - reference) / np.linalg.norm(reference)
    assert relative < 5e-3, relative  # floor is the FD scheme's own dt error


def test_series_satisfies_boundary_and_initial_data():
    y = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    inside = ns.impulsive_couette_exact(y, 0.05, wall_speed=U, kinematic_viscosity=NU)
    assert float(inside[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(inside[-1]) == pytest.approx(U, abs=1e-12)
    assert bool(((inside >= -1e-9) & (inside <= U + 1e-9)).all())
    assert bool((inside[1:] >= inside[:-1] - 1e-12).all())  # monotone in y
    early = ns.impulsive_couette_exact(y, 1e-8, wall_speed=U, kinematic_viscosity=NU)
    # still (almost) fluid just after t = 0; the 1e-3 floor is the round-off of
    # the truncated series, whose coefficients cancel to 13 digits at t -> 0
    assert float(early[:-1].abs().max()) < 5e-3
    late = ns.impulsive_couette_exact(y, 50.0, wall_speed=U, kinematic_viscosity=NU)
    assert float((late - U * y).abs().max()) < 1e-6     # relaxes to linear Couette


def test_impulse_start_transient_matches_the_series_below_three_percent():
    """P1 / 8 cells at t = 0.2: the wall transient is accurate to < 3 %."""
    row = run_case(1, 8, final_time=T_END)
    assert row["finite"]
    assert row["velocity_l2_relative"] < 0.03, row["velocity_l2_relative"]
    assert row["u_monotone_in_y"]
    assert row["u_min"] >= -0.02 * U
    assert row["u_max"] <= U * 1.001
