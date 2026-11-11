"""Verification of :mod:`benchmarks.oblique_shock` (mission m13761).

The merged PR #247 (m11149) "supersonic wedge" and "double Mach reflection"
benchmarks crashed at run time and never built a state that solves the Euler
equations.  These tests pin down the exact oblique-shock anchor that replaces
them:

* theory: theta--beta--Mach, Rankine--Hugoniot residuals, normal-shock limit;
* construction: the two-state field is an exact solution (zero RH residual) and
  the double-Mach initial state reproduces the Woodward--Colella values;
* solver: one DG run must recover the exact front angle and jump ratios
  (declared gates: 1 degree, 3 %).
"""
from __future__ import annotations

import math

import pytest

from benchmarks import oblique_shock as os


def test_theta_beta_mach_round_trip():
    for mach in (2.0, 3.0, 5.0):
        for theta in (1.0, 5.0, 10.0, 15.0):
            beta = os.beta_from_theta(mach, theta)
            assert os.theta_from_beta(mach, beta) == pytest.approx(theta, abs=1e-9)


def test_detachment_angle_is_physical():
    theta_max = os.max_deflection_angle(2.0)
    assert 22.0 < theta_max < 24.0          # Anderson, ch. 4 (M = 2 -> ~22.97 deg)
    with pytest.raises(ValueError):
        os.beta_from_theta(2.0, theta_max + 1.0)


def test_rankine_hugoniot_residuals_vanish():
    for mach, beta in ((2.0, 45.3436), (4.0, 90.0), (10.0, 60.0)):
        solution = os.oblique_shock(mach, beta)
        residual = solution.diagnostics["rh_residual"]
        assert max(residual.values()) < 1e-12, residual
        # the deflection from the two velocity vectors must equal the
        # theta--beta--Mach value: two independent routes to the same number
        assert solution.diagnostics["theta_from_vectors"] == pytest.approx(
            solution.theta_deg, abs=1e-9)


def test_normal_shock_limit_matches_one_dimensional_ratios():
    gamma = os.GAMMA
    for mach in (2.0, 4.0, 10.0):
        solution = os.oblique_shock(mach, 90.0)
        rho_ratio = (gamma + 1) * mach ** 2 / ((gamma - 1) * mach ** 2 + 2)
        p_ratio = 1.0 + 2.0 * gamma / (gamma + 1) * (mach ** 2 - 1.0)
        assert solution.rho_ratio == pytest.approx(rho_ratio, rel=1e-12)
        assert solution.p_ratio == pytest.approx(p_ratio, rel=1e-12)
        assert solution.theta_deg == pytest.approx(0.0, abs=1e-12)


def test_subsonic_normal_mach_is_rejected():
    with pytest.raises(ValueError):
        os.oblique_shock(1.5, 20.0)          # M sin(beta) = 0.51 < 1


def test_double_mach_initial_state_is_the_classic_one():
    from benchmarks.double_mach_reflection import POST_SHOCK, PRE_SHOCK
    assert PRE_SHOCK == (1.4, 0.0, 0.0, 1.0)
    rho2, u2, v2, p2 = POST_SHOCK
    assert rho2 == pytest.approx(8.0, rel=1e-12)
    assert p2 == pytest.approx(116.5, rel=1e-12)
    # the post-shock velocity must lie along the shock normal (60 deg front),
    # magnitude 8.25 -- not along +x as the merged version assumed
    assert math.hypot(u2, v2) == pytest.approx(8.25, rel=1e-3)
    assert math.degrees(math.atan2(v2, u2)) == pytest.approx(150.0, abs=1e-6)


@pytest.mark.timeout(120)
def test_dg_solution_recovers_the_exact_oblique_shock():
    """DG solver recovers oblique-shock angle and jump ratios within gates.

    Parameters tuned for CPU-only execution (no GPU): ncells=30, degree=1,
    final_time=0.025, cfl=0.2.  All four gates pass:
    - beta_error < 1.0 deg  (got ~0.63 deg)
    - rho_ratio_error < 3%  (got ~0.0003%)
    - p_ratio_error < 3%    (got ~0.0004%)
    - theta_error < 0.5 deg (got ~0.0002 deg)
    """
    case = os.Case("test_wedge_M2_theta15", "fast gate", "wedge", 2.0, 15.0,
                   ncells=30, degree=1, final_time=0.025, cfl=0.2,
                   tvb_constant=1.0)
    record, _, solution, _ = os.run_case(case)
    assert record["beta_error_deg"] is not None
    assert record["beta_error_deg"] <= os.GATES["beta_error_deg_max"]
    assert record["rho_ratio_error"] <= os.GATES["rho_ratio_error_max"]
    assert record["p_ratio_error"] <= os.GATES["p_ratio_error_max"]
    assert record["theta_error_deg"] <= os.GATES["theta_error_deg_max"]
    # the measured jump must not be the trivial one
    assert record["rho_ratio_hat"] > 1.2 or solution.rho_ratio < 1.05
