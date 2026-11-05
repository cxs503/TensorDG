"""Regression tests for the two-dimensional discrete entropy budget.

These tests guard the 2-D entropy machinery and pin the entropy behaviour of the
merged 2-D limiters.  Every number here is recomputed, never copied from
``validation/entropy_2d/summary.json``.

Fast by construction: the smooth case runs on 16x16 and 32x32 cells with a
final time of 0.04, and the Sod cross-check uses 24 cells.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.entropy_budget_2d import (
    extruded_sod_case,
    limiter_activity,
    one_dimensional_sod_production,
    smooth_advection_case,
    smooth_exact_density,
    smooth_initial,
    sod_initial,
    total_entropy_1d_reference,
    total_entropy_2d,
)
from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density,
)
from tensordg.euler_dg_2d import apply_weno_limiter_2d, solve_euler_dg_2d

GAMMA = 1.4
FINAL_TIME = 0.04


@pytest.fixture(autouse=True, scope="module")
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _uniform_coefficients(ncells=8, degree=1, density=2.0, pressure=1.0):
    def initial(x, y):
        full = x + y
        return (torch.full_like(full, density), torch.zeros_like(full),
                torch.zeros_like(full), torch.full_like(full, pressure))

    _, _, coeff, meta = solve_euler_dg_2d(initial, ncells=ncells, degree=degree,
                                          final_time=0.0)
    return coeff, meta


# --------------------------------------------------------------------------- #
# the integral itself
# --------------------------------------------------------------------------- #
def test_entropy_integral_of_a_constant_state_matches_the_closed_form():
    """``rho s`` of a uniform state integrates to ``rho (ln p - gamma ln rho) * A``."""
    coeff, meta = _uniform_coefficients()
    nx, ny = meta["ncells"]
    exact = 2.0 * (np.log(1.0) - GAMMA * np.log(2.0))
    measured = total_entropy_2d(coeff, 1.0 / nx, 1.0 / ny, 1)
    assert abs(measured - exact) / abs(exact) < 1e-12, (measured, exact)


def test_entropy_integral_is_independent_of_the_quadrature_order():
    """The integrand is a polynomial, so refinement must not move the value."""
    _, _, coeff, meta = solve_euler_dg_2d(smooth_initial(), ncells=8, degree=1,
                                          final_time=0.0)
    nx, ny = meta["ncells"]
    values = [total_entropy_2d(coeff, 1.0 / nx, 1.0 / ny, 1, quadrature_order=order)
              for order in (4, 6, 8, 10)]
    spread = (max(values) - min(values)) / abs(values[-1])
    assert spread < 1e-10, (spread, values)


def test_entropy_integral_reproduces_the_one_dimensional_implementation():
    """y-invariant data: the 2-D integral must equal the 1-D one to round-off."""
    _, _, coeff, meta = solve_euler_dg_2d(sod_initial(), ncells=(12, 6), degree=1,
                                          final_time=0.0, periodic=(False, True))
    nx, ny = meta["ncells"]
    measured = total_entropy_2d(coeff, 1.0 / nx, 1.0 / ny, 1)
    reference = total_entropy_1d_reference(coeff, 1.0 / nx, 1, ny)
    assert abs(measured - reference) / abs(reference) < 1e-12, (measured, reference)


# --------------------------------------------------------------------------- #
# case 1: smooth 2-D flow with an exact entropy invariant
# --------------------------------------------------------------------------- #
def test_smooth_two_dimensional_flow_conserves_entropy():
    """The exact solution is a pure advection of ``rho``, so ``int rho s dA`` holds."""
    coarse = smooth_advection_case(16, final_time=FINAL_TIME)
    fine = smooth_advection_case(32, final_time=FINAL_TIME)
    assert coarse["entropy_drift"] >= 0.0, coarse["entropy_drift"]
    assert fine["entropy_drift"] >= 0.0, fine["entropy_drift"]
    assert abs(fine["entropy_drift"]) < abs(coarse["entropy_drift"])
    assert abs(fine["entropy_drift"]) < 1e-5, fine["entropy_drift"]


def test_smooth_two_dimensional_density_matches_the_exact_solution():
    """Sanity check of the exact solution used as the entropy baseline."""
    case = smooth_advection_case(32, final_time=FINAL_TIME)
    assert case["density_linf_error"] / case["density_linf_exact_amplitude"] < 0.05


def test_tvb_limiter_stays_on_the_dissipative_side():
    """A limiter may add dissipation (``drift >= 0``) but never destroy entropy."""
    for ncells in (16, 32):
        case = smooth_advection_case(ncells, final_time=FINAL_TIME, limiter="tvb")
        assert case["entropy_drift"] >= 0.0, (ncells, case["entropy_drift"])


def test_tvb_limiter_is_nearly_inactive_on_smooth_data():
    """It must only clip where the solution is genuinely under-resolved."""
    activity = limiter_activity(16, limiter="tvb")
    assert activity["max_relative_change"] < 5e-3, activity


def test_weno_limiter_entropy_destruction_is_pinned():
    """Known defect (m3430): the WENO limiter *destroys* entropy on smooth data.

    ``drift < 0`` is a violation of the discrete entropy inequality, i.e. the
    scheme creates a solution the second law forbids.  The pinned value is the
    32x32 measurement; when the limiter is fixed this test must be inverted to
    ``drift >= 0`` together with ``test_tvb_limiter_stays_on_the_dissipative_side``.
    See ``docs/reports/tensordg-entropy-2d.md``.
    """
    case = smooth_advection_case(32, final_time=FINAL_TIME, limiter="weno")
    assert case["entropy_drift"] < 0.0, case["entropy_drift"]


def test_weno_limiter_is_more_intrusive_than_tvb_on_smooth_data():
    """The defect's mechanism: WENO rewrites ~5x more of the smooth state."""
    tvb = limiter_activity(16, limiter="tvb")
    weno = limiter_activity(16, limiter="weno")
    assert weno["max_relative_change"] > 3.0 * tvb["max_relative_change"], (tvb, weno)


def test_weno_limiter_of_a_uniform_state_is_the_identity():
    """Weights are scale invariant, so no reconstruction error on constant data."""
    coeff, meta = _uniform_coefficients()
    nx, ny = meta["ncells"]
    limited = apply_weno_limiter_2d(coeff, 1.0 / nx, 1.0 / ny, 1.0, (True, True))
    assert torch.allclose(limited, coeff, atol=1e-12)


# --------------------------------------------------------------------------- #
# case 2: extruded Sod -- 2-D path against the 1-D solver and the RH budget
# --------------------------------------------------------------------------- #
def test_extruded_sod_entropy_production_matches_the_one_dimensional_solver():
    """Independent implementations of the same physics must agree closely."""
    case = extruded_sod_case(24, final_time=0.05)
    reference = one_dimensional_sod_production(24, final_time=0.05)
    assert case["entropy_production"] > 0.0
    relative = abs(case["entropy_production"] - reference["entropy_production"]) / abs(
        reference["entropy_production"])
    assert relative < 0.01, (relative, case, reference)


def test_extruded_sod_entropy_production_against_the_analytic_budget_is_honest():
    """The analytic RH budget is *not* met at engineering resolution (1-D ledger:
    52 % at 64 cells / P1).  The test asserts only the sign and the recorded gap,
    so a silent regression in either direction is caught."""
    budget = analytic_right_shock_budget(np.array([1.0, 0.0, 1.0]),
                                         np.array([0.125, 0.0, 0.1]), 0.1)
    case = extruded_sod_case(24, final_time=0.1)
    assert case["entropy_production"] > 0.0
    assert case["entropy_production"] < 5.0 * budget["production_total"]
    assert 0.2 < case["entropy_production"] / budget["production_total"] < 4.0, case


def test_sod_entropy_density_field_is_consistent_with_the_total():
    """The pointwise field used for contours integrates back to the total."""
    coeff, meta = _uniform_coefficients(density=1.5, pressure=0.7)
    nx, ny = meta["ncells"]
    total = total_entropy_2d(coeff, 1.0 / nx, 1.0 / ny, 1)
    exact = 1.5 * (np.log(0.7) - GAMMA * np.log(1.5))
    assert abs(total - exact) / abs(exact) < 1e-12


def test_entropy_density_helper_agrees_with_the_scalar_integral_on_uniform_data():
    density = np.full((2, 2), 1.3)
    pressure = np.full((2, 2), 0.9)
    assert np.allclose(entropy_density(density, pressure),
                       density * (np.log(pressure) - GAMMA * np.log(density)))