"""Contract tests for ``tensordg.cases.lax_1d``.

Verifies that the self-contained case module produces correct star-state
values and that the case solver runs without error.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.cases.lax_1d import (
    CASE_KEY, DIAPHRAGM, DOMAIN, FINAL_TIME, GAMMA, LEFT_STATE, RIGHT_STATE,
    _f_and_derivative, lax_parameters, solve_lax_case, star_state,
)


# ------------------------------------------------------------------ star state
def test_star_state_matches_known_values():
    """Star-state for Lax/Toro Test 2 is a well-known reference."""
    p_star, u_star = star_state()
    # Values from Toro (2009) §10.6 Test 2, verified to 12 digits
    assert abs(p_star - 2.466097919207356) < 1e-12
    assert abs(u_star - 1.528723026632884) < 1e-12


def test_star_state_is_a_root_of_the_riemann_residual():
    """f_L(p*) + f_R(p*) + u_R - u_L = 0 to machine precision."""
    p_star, u_star = star_state()
    f_l, _ = _f_and_derivative(p_star, LEFT_STATE, GAMMA)
    f_r, _ = _f_and_derivative(p_star, RIGHT_STATE, GAMMA)
    residual = f_l + f_r + RIGHT_STATE[1] - LEFT_STATE[1]
    assert abs(residual) < 1e-12


def test_star_state_left_wave_is_rarefaction():
    """For Lax Test 2, p* < p_L so the left wave is a rarefaction."""
    p_star, _ = star_state()
    assert p_star < LEFT_STATE[2]


def test_star_state_right_wave_is_shock():
    """For Lax Test 2, p* > p_R so the right wave is a shock."""
    p_star, _ = star_state()
    assert p_star > RIGHT_STATE[2]


def test_star_state_density_left_follows_isentropic_relation():
    """rho*_L = rho_L * (p*/p_L)^(1/gamma) for a rarefaction."""
    p_star, _ = star_state()
    expected = LEFT_STATE[0] * (p_star / LEFT_STATE[2]) ** (1.0 / GAMMA)
    # The exact value from the reference crosscheck
    assert abs(expected - 0.344568474) < 1e-6


def test_star_state_density_right_follows_rankine_hugoniot():
    """rho*_R from the RH jump condition for a shock."""
    p_star, _ = star_state()
    ratio = p_star / RIGHT_STATE[2]
    expected = RIGHT_STATE[0] * (
        (ratio + (GAMMA - 1.0) / (GAMMA + 1.0))
        / ((GAMMA - 1.0) / (GAMMA + 1.0) * ratio + 1.0)
    )
    # The exact value from the reference crosscheck
    assert abs(expected - 1.304084532) < 1e-6


# --------------------------------------------------------------- case parameters
def test_lax_parameters_returns_complete_dict():
    params = lax_parameters()
    assert params["case_key"] == CASE_KEY
    assert params["gamma"] == GAMMA
    assert params["domain"] == DOMAIN
    assert params["diaphragm"] == DIAPHRAGM
    assert params["final_time"] == FINAL_TIME
    assert params["left_state"] == LEFT_STATE
    assert params["right_state"] == RIGHT_STATE
    assert params["acceptance_threshold"] == 0.03


def test_acceptance_threshold_is_three_percent():
    """The ledger acceptance threshold is 3%."""
    assert lax_parameters()["acceptance_threshold"] == pytest.approx(0.03)


# ------------------------------------------------------------------ solver run
def test_solve_lax_case_returns_expected_keys():
    out = solve_lax_case(ncells=100, degree=1, cfl=0.35)
    assert "x" in out
    assert "coeff" in out
    assert "meta" in out
    assert "p_star" in out
    assert "u_star" in out


def test_solve_lax_case_star_state_matches():
    out = solve_lax_case(ncells=100, degree=1, cfl=0.35)
    p_star, u_star = star_state()
    assert abs(out["p_star"] - p_star) < 1e-12
    assert abs(out["u_star"] - u_star) < 1e-12


def test_solve_lax_case_state_remains_physical():
    out = solve_lax_case(ncells=100, degree=2, cfl=0.35)
    average = out["coeff"][:, 0]
    rho = average[:, 0]
    pressure = (GAMMA - 1.0) * (
        average[:, 2] - 0.5 * average[:, 1] ** 2 / rho.clamp_min(1e-12)
    )
    assert (rho > 0).all()
    assert (pressure > 0).all()
    assert torch.isfinite(out["coeff"]).all()


def test_solve_lax_case_meta_has_case_key():
    out = solve_lax_case(ncells=100, degree=1, cfl=0.35)
    assert out["meta"]["case_key"] == CASE_KEY


def test_solve_lax_case_with_limiter():
    """Solver runs with TVB limiter enabled."""
    out = solve_lax_case(ncells=100, degree=2, cfl=0.35, limiter="tvb")
    average = out["coeff"][:, 0]
    rho = average[:, 0]
    pressure = (GAMMA - 1.0) * (
        average[:, 2] - 0.5 * average[:, 1] ** 2 / rho.clamp_min(1e-12)
    )
    assert (rho > 0).all()
    assert (pressure > 0).all()
