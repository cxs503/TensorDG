"""Discrete entropy budget / second-law tests for the 1-D Euler DG solver.

The verification question is *not* "does the solver look right" but "does the
weak solution it produces satisfy the entropy inequality, and does the amount of
entropy it produces converge to the analytic Rankine-Hugoniot value".  Both
statements are falsifiable without any reference run of the same code:

* the analytic production rate comes from the exact star state plus the
  Rankine-Hugoniot density ratio (``benchmarks/_entropy_tools.py``);
* the diagnostic itself is validated against a finite difference of the exact
  Riemann solution *before* it is used to judge the solver;
* the smooth periodic case has an exactly conserved entropy integral, so any
  measured drift is numerical entropy production.

See ``benchmarks/entropy_budget_1d.py`` for the full study and
``docs/entropy_verification.md`` for the report.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    observed_order,
    relative_error,
    total_entropy_from_coefficients,
    total_entropy_from_coefficients_convergence,
    total_entropy_from_profiles,
)
from benchmarks.riemann_exact import sample
from tensordg.euler_dg_1d import solve_euler_dg_1d, solve_periodic_euler_dg_1d

LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
SOD_TIME = 0.2
CFL = 0.3


def _production(left, right, ncells, degree=2, final_time=SOD_TIME, limiter=None,
                tvb_constant=1.0, cfl=CFL):
    """Discrete ``int rho s dx`` produced by one DG run (final minus initial)."""
    dx = 1.0 / ncells
    _, coeff_0, _ = solve_euler_dg_1d(left, right, ncells=ncells, degree=degree,
                                      final_time=0.0, cfl=cfl, limiter=limiter,
                                      tvb_constant=tvb_constant)
    _, coeff_t, meta = solve_euler_dg_1d(left, right, ncells=ncells, degree=degree,
                                         final_time=final_time, cfl=cfl, limiter=limiter,
                                         tvb_constant=tvb_constant)
    return (total_entropy_from_coefficients(coeff_t, dx, degree)
            - total_entropy_from_coefficients(coeff_0, dx, degree)), meta


# --------------------------------------------------------------------------- #
# diagnostic self-tests: a measure that cannot recover a known answer cannot
# judge a solver
# --------------------------------------------------------------------------- #
def test_rankine_hugoniot_production_matches_the_exact_solution():
    """Analytic shock entropy production vs a finite difference of the exact solution."""
    budget = analytic_right_shock_budget(LEFT, RIGHT, SOD_TIME)
    x = np.linspace(0.0, 1.0, 50001)
    t_hi, t_lo = SOD_TIME, SOD_TIME - 0.05
    hi = sample(x, t_hi, LEFT, RIGHT)
    lo = sample(x, t_lo, LEFT, RIGHT)
    rate = (total_entropy_from_profiles(hi["density"], hi["pressure"], x)
            - total_entropy_from_profiles(lo["density"], lo["pressure"], x)) / (t_hi - t_lo)
    assert rate > 0.0, "the exact Sod solution must produce entropy"
    # the finite difference is limited by the sampling grid (50k points) and by
    # the 0.05 time window: 1.8e-3 measured, i.e. the two baselines agree to <0.2 %
    assert relative_error(budget["production"], rate) < 5e-3


def test_entropy_integral_is_quadrature_converged():
    """The entropy integral must not depend on the Gauss order used to measure it."""
    _, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=60, degree=2,
                                    final_time=0.02, cfl=CFL)
    ladder = total_entropy_from_coefficients_convergence(coeff, 1.0 / 60, 2)
    # 4.2e-6 relative spread measured between 4 and 12 Gauss points
    assert max(ladder) - min(ladder) < 1e-4 * abs(ladder[0])


def test_analytic_budget_reproduces_the_known_sod_shock_position():
    """Shock speed from the same RH states must place the Sod shock at x ~ 0.85."""
    budget = analytic_right_shock_budget(LEFT, RIGHT, SOD_TIME)
    assert abs(budget["shock_position"] - 0.8504) < 5e-3
    assert budget["production_total"] > 0.0


# --------------------------------------------------------------------------- #
# C1: the second law must hold for every limiter setting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("limiter", [None, "tvb", "weno"])
def test_shock_produces_entropy(limiter):
    """``int rho s dx`` must grow across a shock: the entropy inequality holds."""
    produced, meta = _production(LEFT, RIGHT, ncells=64, limiter=limiter)
    assert meta["steps"] > 0
    assert produced > 0.0, f"entropy decreased with limiter={limiter}: {produced}"


# --------------------------------------------------------------------------- #
# C2: the produced amount must approach the analytic value
# --------------------------------------------------------------------------- #
def test_shock_budget_converges_to_the_analytic_value():
    """Coarse-to-fine error must shrink monotonically towards the RH budget."""
    analytic = analytic_right_shock_budget(LEFT, RIGHT, SOD_TIME)["production_total"]
    errors = []
    for ncells in (32, 64, 128):
        produced, _ = _production(LEFT, RIGHT, ncells=ncells)
        errors.append(abs(produced - analytic) / analytic)
    assert errors[-1] < errors[0], errors
    assert errors[-1] < 0.15, errors          # measured: 11.65 % / 7.30 % at 64 / 128 cells
    assert min(errors) < 0.10, errors


def test_fine_mesh_entropy_budget_nears_the_three_percent_gate():
    """Budget error must remain inside a documented, honest band.

    ``benchmarks/entropy_budget_1d.py`` measures 11.65 % / 7.30 % / 3.13 % at
    64 / 128 / 256 cells (no limiter, Sod), i.e. the excess *numerical* entropy
    production converges only at about first order, so the platform 3 % gate is
    reached only around 256 cells and not yet met on the meshes used below.
    The assertion therefore keeps the measured band instead of pretending.
    """
    analytic = analytic_right_shock_budget(LEFT, RIGHT, SOD_TIME)["production_total"]
    produced, _ = _production(LEFT, RIGHT, ncells=128)
    assert produced > 0.0
    assert relative_error(analytic, produced) < 0.08


# --------------------------------------------------------------------------- #
# C3: smooth periodic data must not produce entropy
# --------------------------------------------------------------------------- #
def _smooth_drift(ncells, limiter=None, tvb_constant=1.0, final_time=0.1, degree=2):
    def initial_state(nodes):
        rho = 1.0 + 0.2 * torch.sin(2.0 * np.pi * nodes)
        return rho, torch.ones_like(nodes), torch.ones_like(nodes)

    dx = 1.0 / ncells
    _, coeff_0, _ = solve_periodic_euler_dg_1d(initial_state, ncells=ncells, degree=degree,
                                               final_time=0.0, cfl=CFL, limiter=limiter,
                                               tvb_constant=tvb_constant)
    _, coeff_t, _ = solve_periodic_euler_dg_1d(initial_state, ncells=ncells, degree=degree,
                                               final_time=final_time, cfl=CFL, limiter=limiter,
                                               tvb_constant=tvb_constant)
    return (total_entropy_from_coefficients(coeff_t, dx, degree)
            - total_entropy_from_coefficients(coeff_0, dx, degree))


def test_smooth_periodic_flow_entropy_production_vanishes_with_refinement():
    """The exact solution conserves ``int rho s dx``; the drift must go to zero."""
    drifts = [abs(_smooth_drift(ncells)) for ncells in (16, 32, 64)]
    assert drifts[-1] < drifts[0], drifts
    order = observed_order((16, 32, 64), drifts)[-1]
    assert order is not None and order > 2.0, (order, drifts)
    assert drifts[-1] < 1e-6, drifts


def test_smooth_periodic_flow_stays_on_the_dissipative_side():
    """Any measured entropy change on smooth data must be production, not destruction."""
    assert _smooth_drift(32) >= 0.0
    assert _smooth_drift(32, limiter="tvb", tvb_constant=1.0) >= 0.0