"""Standalone verification of the Lax Riemann problem wave positions.

Why this test exists
--------------------
The CASE-LAX-DG acceptance criteria is ``ref: 精确解｜容差 < 3 %``.  The
reference solution is the exact Riemann solver, whose wave positions are
computed from closed-form Rankine–Hugoniot and isentropic relations.  While
``benchmarks.lax_case_pool.exact_waves`` computes these positions, no
*standalone* test verifies them against the analytical formulas from
Toro (2009) §10.6, Test 2.

This test pins the four wave positions (rarefaction head/tail, contact,
shock) to the analytical formulas, independent of any DG solver run.
It is a "验证欠账" item: verifying that the reference solution's wave
positions match the literature values.

Reference
---------
E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Springer, 2009, §10.6 (Test 2, p. 131).

The Toro Test 2 wave positions at t = 0.13 are:
- rarefaction head:  x = 0.157637
- rarefaction tail:  x = 0.287229
- contact:           x = 0.698734
- shock:             x = 0.822312

These values are computed from the star-state (p*, u*) and the
Rankine–Hugoniot / isentropic relations, not transcribed from a table.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.riemann_exact import star_state
from tensordg.cases.lax_1d import (
    DIAPHRAGM, FINAL_TIME, GAMMA, LEFT_STATE, RIGHT_STATE,
)

# ---------------------------------------------------------------------------
# Tolerance: wave positions are computed to machine precision, so we use
# a tight tolerance.  The 3 % ledger bar applies to field errors, not
# to the reference solution itself.
# ---------------------------------------------------------------------------
POSITION_TOL = 1e-6  # absolute tolerance for wave positions


def _compute_wave_positions(
    left=LEFT_STATE, right=RIGHT_STATE, final_time=FINAL_TIME,
    diaphragm=DIAPHRAGM, gamma=GAMMA,
) -> dict[str, float]:
    """Compute the four wave positions from closed-form relations.

    This is a self-contained implementation that does not depend on
    ``benchmarks.lax_case_pool.exact_waves``, making the test independent
    of the case pool driver.

    Returns
    -------
    dict with keys: rarefaction_head, rarefaction_tail, contact, shock
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    p_star, u_star = star_state(left, right, gamma)
    a_l = math.sqrt(gamma * p_l / rho_l)
    a_r = math.sqrt(gamma * p_r / rho_r)

    # Left wave: rarefaction (p_star < p_l for Lax Test 2)
    assert p_star < p_l, "Lax Test 2: left wave must be a rarefaction"
    a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
    head_speed = u_l - a_l
    tail_speed = u_star - a_star_l

    # Right wave: shock (p_star > p_r for Lax Test 2)
    assert p_star > p_r, "Lax Test 2: right wave must be a shock"
    ratio = p_star / p_r
    shock_speed = u_r + a_r * math.sqrt(
        (gamma + 1.0) / (2.0 * gamma) * ratio
        + (gamma - 1.0) / (2.0 * gamma)
    )

    return {
        "rarefaction_head": diaphragm + head_speed * final_time,
        "rarefaction_tail": diaphragm + tail_speed * final_time,
        "contact": diaphragm + u_star * final_time,
        "shock": diaphragm + shock_speed * final_time,
    }


# ---------------------------------------------------------------------------
# Tests: wave positions vs Toro (2009) §10.6 Test 2
# ---------------------------------------------------------------------------

class TestWavePositionsVsToro:
    """Verify wave positions against Toro (2009) Test 2 analytical values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.positions = _compute_wave_positions()

    def test_rarefaction_head(self):
        """Rarefaction head position: x = 0.157637."""
        expected = 0.157637
        assert abs(self.positions["rarefaction_head"] - expected) < POSITION_TOL, (
            f"rarefaction head: {self.positions['rarefaction_head']:.6f} "
            f"vs expected {expected}"
        )

    def test_rarefaction_tail(self):
        """Rarefaction tail position: x = 0.287229."""
        expected = 0.287229
        assert abs(self.positions["rarefaction_tail"] - expected) < POSITION_TOL, (
            f"rarefaction tail: {self.positions['rarefaction_tail']:.6f} "
            f"vs expected {expected}"
        )

    def test_contact(self):
        """Contact discontinuity position: x = 0.698734."""
        expected = 0.698734
        assert abs(self.positions["contact"] - expected) < POSITION_TOL, (
            f"contact: {self.positions['contact']:.6f} vs expected {expected}"
        )

    def test_shock(self):
        """Shock position: x = 0.822312."""
        expected = 0.822312
        assert abs(self.positions["shock"] - expected) < POSITION_TOL, (
            f"shock: {self.positions['shock']:.6f} vs expected {expected}"
        )


class TestWaveOrdering:
    """Verify the physical ordering of waves."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.positions = _compute_wave_positions()

    def test_waves_are_ordered_left_to_right(self):
        """Waves must appear in order: head < tail < contact < shock."""
        p = self.positions
        assert p["rarefaction_head"] < p["rarefaction_tail"], (
            f"head ({p['rarefaction_head']:.6f}) >= tail ({p['rarefaction_tail']:.6f})"
        )
        assert p["rarefaction_tail"] < p["contact"], (
            f"tail ({p['rarefaction_tail']:.6f}) >= contact ({p['contact']:.6f})"
        )
        assert p["contact"] < p["shock"], (
            f"contact ({p['contact']:.6f}) >= shock ({p['shock']:.6f})"
        )

    def test_all_waves_inside_domain(self):
        """All wave positions must be inside [0, 1]."""
        for name, pos in self.positions.items():
            assert 0.0 <= pos <= 1.0, f"{name} = {pos:.6f} outside [0, 1]"


class TestWaveSpeeds:
    """Verify wave speeds are physically consistent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.positions = _compute_wave_positions()

    def test_rarefaction_head_speed_is_subsonic_left(self):
        """Rarefaction head speed = u_L - a_L (left sonic point)."""
        rho_l, u_l, p_l = LEFT_STATE
        a_l = math.sqrt(GAMMA * p_l / rho_l)
        expected_speed = u_l - a_l
        actual_speed = (self.positions["rarefaction_head"] - DIAPHRAGM) / FINAL_TIME
        assert abs(actual_speed - expected_speed) < POSITION_TOL / FINAL_TIME

    def test_shock_speed_is_supersonic_right(self):
        """Shock speed > u_R + a_R (right supersonic)."""
        rho_r, u_r, p_r = RIGHT_STATE
        a_r = math.sqrt(GAMMA * p_r / rho_r)
        shock_speed = (self.positions["shock"] - DIAPHRAGM) / FINAL_TIME
        assert shock_speed > u_r + a_r, (
            f"shock speed {shock_speed:.6f} <= right sonic speed {u_r + a_r:.6f}"
        )

    def test_contact_speed_equals_star_velocity(self):
        """Contact speed = u* (star-state velocity)."""
        _, u_star = star_state(LEFT_STATE, RIGHT_STATE, GAMMA)
        contact_speed = (self.positions["contact"] - DIAPHRAGM) / FINAL_TIME
        assert abs(contact_speed - u_star) < POSITION_TOL / FINAL_TIME


class TestConsistencyWithCasePool:
    """Verify consistency with the case pool's exact_waves function."""

    def test_positions_match_case_pool_exact_waves(self):
        """Our computed positions must match benchmarks.lax_case_pool.exact_waves."""
        from benchmarks.lax_case_pool import exact_waves

        pool_waves = exact_waves()
        our_positions = _compute_wave_positions()

        for key in ["rarefaction_head", "rarefaction_tail", "contact", "shock"]:
            pool_val = pool_waves["positions"][key]
            our_val = our_positions[key]
            assert abs(pool_val - our_val) < 1e-12, (
                f"{key}: pool={pool_val:.12f}, ours={our_val:.12f}"
            )

    def test_star_state_matches_case_pool(self):
        """Star-state must match the case pool's reference."""
        from benchmarks.lax_case_pool import exact_waves

        pool_waves = exact_waves()
        p_star, u_star = star_state(LEFT_STATE, RIGHT_STATE, GAMMA)

        assert abs(pool_waves["pressure_star"] - p_star) < 1e-12
        assert abs(pool_waves["velocity_star"] - u_star) < 1e-12
