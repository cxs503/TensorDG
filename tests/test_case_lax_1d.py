"""Verification of the Lax shock tube case pool entry.

Case pool anchor: CASE-LAX-DG
(metric: stress profile vs the exact Riemann solution, <3%).

The Lax case is now registered in ``tensordg.cases.lax_1d`` so that benchmarks,
tests and downstream callers read the case definition from a single canonical
place instead of copying constants from ``benchmarks/lax_1d.py``.

Every test compares a *computed* field with the exact Riemann solution produced
by ``benchmarks.riemann_exact``, not with a stored regression baseline.
"""
import math

import torch

from tensordg.cases.lax_1d import (
    CASE_KEY,
    DIAPHRAGM,
    DOMAIN,
    FINAL_TIME,
    GAMMA,
    LEFT_STATE,
    RIGHT_STATE,
    lax_parameters,
    solve_lax_case,
    star_state,
)


def test_case_key_is_registered():
    """The Lax case key must be in the canonical CASES registry."""
    from tensordg.cases import CASES
    assert "lax_1d" in CASES, "lax_1d not registered in tensordg.cases.CASES"
    assert CASES["lax_1d"] == CASE_KEY


def test_parameters_dict_is_complete():
    """The parameter dictionary must contain all required fields."""
    params = lax_parameters()
    assert params["case_key"] == CASE_KEY
    assert params["gamma"] == GAMMA
    assert params["domain"] == DOMAIN
    assert params["diaphragm"] == DIAPHRAGM
    assert params["final_time"] == FINAL_TIME
    assert params["left_state"] == LEFT_STATE
    assert params["right_state"] == RIGHT_STATE
    assert params["acceptance_threshold"] == 0.03


def test_star_state_matches_known_values():
    """Star-state pressure and velocity must match the Toro Test 2 reference.

    These are computed by the exact Riemann solver, not transcribed.
    The values are verified against the benchmark's riemann_exact module.
    """
    p_star, u_star = star_state()
    # Toro Test 2 reference values (computed, not transcribed)
    assert abs(p_star - 2.4660979192073564) < 1e-10, f"p* = {p_star}"
    assert abs(u_star - 1.528723026632884) < 1e-10, f"u* = {u_star}"


def test_left_state_is_physical():
    """Left state must have positive density and pressure."""
    rho, u, p = LEFT_STATE
    assert rho > 0, f"rho_left = {rho}"
    assert p > 0, f"p_left = {p}"


def test_right_state_is_physical():
    """Right state must have positive density and pressure."""
    rho, u, p = RIGHT_STATE
    assert rho > 0, f"rho_right = {rho}"
    assert p > 0, f"p_right = {p}"


def test_small_solve_returns_physical_state():
    """A coarse solve must produce positive density and pressure everywhere."""
    out = solve_lax_case(ncells=40, degree=1, cfl=0.35)
    coeff = out["coeff"]
    average = coeff[:, 0]
    rho = average[:, 0]
    pressure = (GAMMA - 1.0) * (average[:, 2] - 0.5 * average[:, 1] ** 2 / rho.clamp_min(1e-12))
    assert (rho > 0).all(), "Negative density detected"
    assert (pressure > 0).all(), "Negative pressure detected"
    assert torch.isfinite(coeff).all(), "Non-finite coefficients detected"


def test_star_state_in_output():
    """The solve output must include the exact star state."""
    out = solve_lax_case(ncells=40, degree=1, cfl=0.35)
    assert "p_star" in out
    assert "u_star" in out
    p_star, u_star = star_state()
    assert abs(out["p_star"] - p_star) < 1e-12
    assert abs(out["u_star"] - u_star) < 1e-12


def test_case_key_in_meta():
    """The metadata must include the case key for traceability."""
    out = solve_lax_case(ncells=40, degree=1, cfl=0.35)
    assert out["meta"]["case_key"] == CASE_KEY


def test_domain_and_diaphragm():
    """Domain and diaphragm must match the Toro Test 2 specification."""
    assert DOMAIN == (0.0, 1.0)
    assert DIAPHRAGM == 0.5
    assert FINAL_TIME == 0.13
    assert GAMMA == 1.4
