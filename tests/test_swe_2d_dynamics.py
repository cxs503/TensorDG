"""Unsteady verification gates for the 2-D shallow-water solver.

These tests pin the two defects fixed in mission m6386 and the dynamical
behaviour that no previous 2-D test covered:

* the y-direction interface traces used an ``"xyklv,l,ak->xykav"`` contraction
  that kept the x-mode index and multiplied every mode by ``phi_k(x_a)``
  instead of summing the x-modes, so an otherwise y-invariant unsteady flow
  received a spurious y-momentum source in the cross mode and the solver blew
  up in one time step;
* ``solve_swe_2d(limiter=...)`` accepted the argument and silently ignored it.

References are closed form (``tensordg.swe_exact`` for the Stoker solution and
the linear shallow-water standing wave), never the solver's own output.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.swe_2d import solve_swe_2d
from tensordg.swe_exact import dam_break_cell_averages

from benchmarks.swe_2d_verification import standing_wave, stoker_2d

GRAVITY = 9.81


def test_stoker_depth_and_momentum_match_exact_solution():
    """y-invariant 2-D run must reproduce the exact 1-D Stoker solution."""
    r = stoker_2d(nx=100, ny=4, degree=1, final_time=0.3)
    assert r["steps"] > 50
    assert r["rel_l1_h"] < 0.03, f"rel L1(h) = {r['rel_l1_h']:.4%}"
    assert r["rel_l1_hu"] < 0.03, f"rel L1(hu) = {r['rel_l1_hu']:.4%}"
    assert r["star_depth_error"] < 0.03


def test_y_invariance_is_machine_zero():
    """y-invariant data must keep h independent of y and hv identically zero."""
    r = stoker_2d(nx=60, ny=4, degree=1, final_time=0.3)
    assert r["y_spread"] < 1e-12, r["y_spread"]
    assert r["hv_coefficient"] < 1e-12, r["hv_coefficient"]
    assert np.isfinite(r["h_numeric"]).all()


def test_standing_wave_matches_linear_theory_at_half_period():
    """Exact linear SWE reference, skewed wave vector, both axes active."""
    r = standing_wave(nx=32, ny=32, quarter_periods=2)
    assert r["rel_l2_vs_exact"] < 0.03, f"rel L2 = {r['rel_l2_vs_exact']:.4%}"
    assert np.isfinite(r["h"]).all()


def test_standing_wave_closes_after_one_period():
    """After one exact linear period the field must return to its initial state."""
    r = standing_wave(nx=32, ny=32, quarter_periods=4)
    assert r["phase_closure"] < 0.03, f"closure = {r['phase_closure']:.4%}"


@pytest.mark.parametrize("limiter", ["tvb", "weno"])
def test_limiter_is_applied_and_keeps_the_solution_admissible(limiter):
    """The limiter argument used to be dropped silently; it must actually run."""
    r = stoker_2d(nx=60, ny=4, degree=1, final_time=0.2, limiter=limiter)
    assert r["steps"] > 20
    assert r["rel_l1_h"] < 0.05, f"rel L1(h) = {r['rel_l1_h']:.4%}"
    assert np.isfinite(r["h_numeric"]).all()
    assert r["h_min"] > 0.0


def test_unknown_limiter_is_rejected():
    with pytest.raises(ValueError):
        solve_swe_2d(lambda X, Y: torch.ones_like(X + Y),
                     lambda X, Y: torch.zeros_like(X + Y),
                     lambda X, Y: torch.zeros_like(X + Y),
                     lambda X, Y: torch.zeros_like(X + Y),
                     nx=8, ny=8, degree=1, final_time=0.1,
                     limiter="minmod")


def test_limiter_preserves_a_lake_at_rest():
    """Limiting must not disturb the well-balanced steady state."""
    H = 2.0
    flat = lambda X, Y: torch.zeros_like(X + Y)
    x, y, coeff, zb_faces, meta = solve_swe_2d(
        lambda X, Y: H + flat(X, Y), flat, flat, flat,
        nx=16, ny=16, degree=1, final_time=0.4, limiter="tvb",
        domain=((-2.0, 2.0), (-2.0, 2.0)), cfl=0.2)
    assert meta["steps"] > 5
    assert float((coeff[:, :, 0, 0, 0] - H).abs().max()) < 1e-11
    assert float(coeff[:, :, :, :, 1].abs().max()) < 1e-11
    assert float(coeff[:, :, :, :, 2].abs().max()) < 1e-11
