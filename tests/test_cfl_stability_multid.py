"""Gates for the multi-dimensional CFL study (``benchmarks.cfl_stability_limit_multid``).

Each test asserts an *invariant* rather than replaying a number:

G1  the Kronecker-sum bound scales as ``1/d`` (structural, matched theta grids);
G2  the code-unit conversion ``cfl = d * dt_max * (2p+1)`` is direction-exact;
G3  the candidate pruning used for the 3-D sum set reproduces the exact 2-D bound;
G4  the shipped 2-D solver really is stable far inside the bound and unstable far
    outside it (the bound is not vacuous);
G5  the shipped default ``cfl`` of every multi-dimensional solver is inside the
    measured limit at the highest degree the solver accepts.
"""
from __future__ import annotations

import math

import pytest

from benchmarks.cfl_stability_limit_multid import (
    DEFAULTS,
    _growth_2d,
    directional_limit,
    euler_ceiling,
    same_mesh_bound,
)

NT = 20  # coarse theta grid: the tests check structure, not digits


@pytest.mark.parametrize("integrator", ["rk4", "ssp_rk3"])
@pytest.mark.parametrize("degree", [0, 1])
def test_double_directional_bound_equals_half_of_one_d(degree, integrator):
    one = directional_limit(degree, 1, integrator, NT)
    two = directional_limit(degree, 2, integrator, NT, prune=False)
    ratio = 2.0 * two["dt_max"] / one["dt_max"]
    assert abs(ratio - 1.0) < 1e-6, ratio


@pytest.mark.parametrize("degree", [0, 1])
def test_code_unit_cfl_is_direction_invariant(degree):
    """``cfl = d * dt_max * (2p+1)`` must be the same number in every dimension."""
    one = directional_limit(degree, 1, "rk4", NT)
    two = directional_limit(degree, 2, "rk4", NT, prune=False)
    three = directional_limit(degree, 3, "rk4", NT, prune=True)
    for multi in (two, three):
        rel = abs(multi["cfl_max_code_units"] / one["cfl_max_code_units"] - 1.0)
        assert rel < 1e-6, (degree, multi["dims"], rel)


@pytest.mark.parametrize("degree", [1, 2])
def test_pruning_reproduces_the_exact_two_dimensional_bound(degree):
    exact = directional_limit(degree, 2, "rk4", NT, prune=False)
    pruned = directional_limit(degree, 2, "rk4", NT, prune=True)
    assert abs(pruned["dt_max"] / exact["dt_max"] - 1.0) < 1e-9


def test_two_dimensional_solver_is_stable_inside_and_unstable_outside_the_bound():
    bound = same_mesh_bound(1, 8, (1.0, 1.0), "rk4")["cfl_max_code_units"]
    inside = _growth_2d(1, 0.25 * bound, 8, 0.2)
    outside = _growth_2d(1, 4.0 * bound, 8, 0.2)
    assert inside["exponent"] <= 1e-3, inside
    assert outside["exponent"] > 1e-3, outside


def test_shipped_two_dimensional_default_is_inside_the_measured_limit():
    for degree in (0, 1, 2, 3):
        bound = same_mesh_bound(degree, 8, (1.0, 1.0), "rk4")["cfl_max_code_units"]
        assert bound > DEFAULTS["advection_2d"], (degree, bound)
        run = _growth_2d(degree, DEFAULTS["advection_2d"], 8, 0.2)
        assert run["exponent"] <= 1e-3, (degree, run)


def test_euler_default_is_inside_the_blow_up_ceiling():
    row = euler_ceiling(degree=1, limiter="tvb", ncells=16, final_time=0.1,
                        cfl_lo=0.02, cfl_hi=0.4, iters=3)
    assert row["cfl_ceiling"] > DEFAULTS["euler_dg_2d"], row
    assert math.isfinite(row["cfl_ceiling"])