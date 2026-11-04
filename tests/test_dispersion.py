"""Verification-debt closure for the wave-propagation accuracy of the DG advection solvers.

``tensordg.advection_1d`` / ``advection_2d`` were previously verified only through
end-point L2 error against a translated sine wave (order of accuracy, conservation).
That is a *solution* metric; it says nothing about the two properties this code base
relies on when it is used as a wave solver, and nothing about the time-step rule the
solver hard-codes:

* **dispersion** — the numerical phase speed of a Fourier mode versus the exact ``a``;
* **dissipation** — the decay rate of that mode (upwind DG must never amplify);
* **the time-step rule** ``dt = cfl * h / ((2 p + 1) |a|)`` — its divisor ``2p+1`` is
  an unverified heuristic; the spectral radius of the semi-discrete operator fixes the
  exact RK4 stability boundary, so the heuristic can be checked for safety.

The operator analysed here is *the production one*: it is recovered by applying
``tensordg.advection_1d._rhs`` to the identity, so no copy of the assembly can drift
away from the solver.  Tests are anchored to closed-form results wherever possible:

* ``p = 0``: upwind DG is first-order upwind finite volume, whose Bloch symbol is
  ``lambda = (a/h)(exp(-i theta) - 1)`` in closed form — exact to round-off;
* ``theta -> 0``: ``lambda -> -i a k`` (exact continuum dispersion) and the measured
  phase-error order must be ``2p + 2`` (dispersion superconvergence);
* energy stability: ``Re lambda <= 0`` for every mode;
* a deliberately oversized step must actually destabilise, otherwise the CFL bound
  above proves nothing (negative control).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.advection_1d import solve_advection  # noqa: E402
from tools.dg_dispersion import (  # noqa: E402
    RK4_REAL_LIMIT,
    bloch_eigen,
    dispersion_curve,
    eigenvalue_set_distance,
    refine_order,
    semidiscrete_matrix,
    semidiscrete_matrix_2d,
    symbol_blocks,
    symbol_eigen,
    theta_grid,
    track_physical_branch,
)

DEGREES = (0, 1, 2, 3)


def _symbol(degree: int, ncells: int, speed: float = 1.0):
    A, h = semidiscrete_matrix(ncells, degree, speed)
    return A, h, symbol_blocks(A, ncells, degree)


def test_p0_symbol_is_exactly_first_order_upwind():
    """Upwind DG-P0 must reproduce the closed-form FTBS symbol to round-off."""
    speed, ncells = 1.0, 16
    _A, h, blocks = _symbol(0, ncells, speed)
    theta = theta_grid(ncells)
    lam, _vec = symbol_eigen(blocks, theta)
    exact = (speed / h) * (np.exp(-1j * theta) - 1.0)
    assert np.abs(lam[:, 0] - exact).max() < 1e-12


def test_block_symbol_matches_full_operator_projection():
    """The explicit block symbol and the projection of the full operator must agree."""
    for p in DEGREES:
        A, _h, blocks = _symbol(p, 16)
        theta = theta_grid(16)
        lam_blocks, _ = symbol_eigen(blocks, theta)
        lam_proj, _ = bloch_eigen(p, A, 16, theta)
        assert eigenvalue_set_distance(lam_blocks, lam_proj) < 1e-10, p


def test_spectral_operator_reproduces_production_rk4_step():
    """The analysed operator must be the solver's operator, not an approximation."""
    from tools.dg_dispersion import one_step_faithfulness

    for p in DEGREES:
        assert one_step_faithfulness(16, p) < 1e-12, p


@pytest.mark.parametrize("degree", DEGREES)
def test_every_semidiscrete_mode_is_dissipative(degree):
    """Upwind DG is energy stable: no semi-discrete eigenvalue may have Re > 0."""
    for ncells in (8, 16, 32):
        curve = dispersion_curve(degree, ncells)
        assert curve["max_real_part_all_modes"] <= 1e-10, (degree, ncells)


@pytest.mark.parametrize("degree", DEGREES)
def test_phase_error_order_is_two_p_plus_two(degree):
    """Dispersion superconvergence: phase error ~ (k h)^(2p+2), not the L2 order 2p+1."""
    res = refine_order(degree, (8, 16, 32))
    expected = 2 * degree + 2
    assert abs(res["observed_order"] - expected) < 0.35, (degree, res)


@pytest.mark.parametrize("degree", (1, 2, 3))
def test_phase_speed_converges_to_the_continuum_limit(degree):
    """As theta -> 0 the tracked branch must tend to the exact dispersion -i a k."""
    speed, ncells = 1.0, 16
    _A, h, blocks = _symbol(degree, ncells, speed)
    theta = np.array([1e-3])
    lam, vec = symbol_eigen(blocks, theta)
    _idx, tracked, _amb = track_physical_branch(lam, vec, theta, speed, h)
    c_num = -tracked.imag[0] / (theta[0] / h)
    assert abs(c_num - speed) / speed < 1e-9, degree


@pytest.mark.parametrize("degree", DEGREES)
def test_timestep_rule_is_inside_the_rk4_stability_region(degree):
    """The hard-coded 2p+1 divisor must give a step the RK4 stability region accepts.

    ``max|lambda| * dt_heuristic = cfl * R_p / (2p+1)`` with ``R_p = max|lambda| h/a``,
    so the admissible CFL is ``RK4_limit * (2p+1) / R_p``.  Default ``cfl = 0.08``
    must sit far inside it.
    """
    speed, ncells = 1.0, 32
    _A, h, blocks = _symbol(degree, ncells, speed)
    lam, _vec = symbol_eigen(blocks, theta_grid(ncells))
    radius = float(np.abs(lam).max() * h / speed)
    cfl_max = RK4_REAL_LIMIT * (2 * degree + 1) / radius
    assert cfl_max > 1.0, (degree, cfl_max)
    assert cfl_max / 0.08 > 5.0, (degree, cfl_max)
    # the heuristic is conservative, never optimistic, for every degree analysed
    assert radius / (2 * degree + 1) <= 3.0, (degree, radius)


def test_2d_timestep_rule_is_inside_the_rk4_stability_region():
    """The 2-D rule ``dt = cfl/((2p+1)(|ax|/hx+|ay|/hy))`` shares the same safety."""
    velocity = (1.0, 0.5)
    for p in (0, 1, 2):
        A, hx, hy = semidiscrete_matrix_2d(5, p, velocity)
        radius = float(np.abs(np.linalg.eigvals(A)).max())
        wave_rate = abs(velocity[0]) / hx + abs(velocity[1]) / hy
        cfl_max = RK4_REAL_LIMIT * (2 * p + 1) / (radius / wave_rate)
        assert cfl_max > 0.8, (p, cfl_max)


def test_negative_control_oversized_step_destabilises():
    """A step above the spectral-radius limit must blow up, so the bound is meaningful."""
    stable = solve_advection(ncells=24, degree=1, final_time=0.2, cfl=0.08)[1]
    assert np.isfinite(stable.numpy()).all()

    from tensordg.advection_1d import l2_error

    centers, coeff, meta = solve_advection(ncells=24, degree=1, final_time=0.2, cfl=6.0)
    err = l2_error(centers, coeff, final_time=0.2)
    # the solver may only shorten dt relative to the heuristic nominal value
    assert meta["dt"] <= 6.0 * (1.0 / 24) / (2 * 1 + 1) + 1e-15
    assert meta["steps"] == 3
    # cfl = 6 for p = 1 gives max|lambda| dt = 6 * R_1 / 3 ~= 12 > RK4 limit 2.79
    assert not math.isfinite(err) or err > 0.5, err


def test_dissipation_not_dispersion_limits_coarse_resolution():
    """Physically relevant ordering: at coarse resolution upwind DG damps first.

    Pinned as a property test so that a future flux/basis change which silently makes
    the scheme much more (or much less) dissipative is caught.
    """
    for degree in (1, 2):
        curve = dispersion_curve(degree, 16)
        # at the worst-resolved mode the per-wavelength damping exceeds the phase error
        assert curve["diss_per_wavelength"][-1] > curve["phase_err"][-1], degree