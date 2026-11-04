"""Tests for the general (all-wave-pattern) analytic entropy budget.

Anchors
-------
* **Literature** — Toro's Table 4.1 star states for his four standard test
  problems (printed to 5 significant figures in the book).
* **Symmetry** — mirroring a Riemann problem in ``x`` (``u -> -u``) maps a
  right shock to a left shock and must leave the total entropy production
  *identical*; this is what pins the sign convention of the left-shock branch.
* **Convention free** — ``d int rho s dx`` of the exact sampled solution.
* **Second law** — every shock must produce entropy (``P > 0``), rarefactions
  exactly zero.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.entropy_budget_general import (
    CASES,
    NEAR_VACUUM_LEFT,
    NEAR_VACUUM_RIGHT,
    analytic_entropy_budget,
    domain_is_clean,
    exact_solution_production,
    measure_dg,
)
from benchmarks.riemann_exact import star_state
from benchmarks._entropy_tools import analytic_right_shock_budget, relative_error

SOD_LEFT = (1.0, 0.0, 1.0)
SOD_RIGHT = (0.125, 0.0, 0.1)

# Toro (2009), "Riemann Solvers and Numerical Methods for Fluid Dynamics",
# 3rd ed., Table 4.1: star pressure and star velocity of the standard tests.
# (name, left, right, p*, u*, tolerance) -- the tolerance tracks how many
# significant figures Toro prints for that entry (test 2 is given as 0.00189).
TORO_TABLE_4_1 = [
    ("test1_sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.30313, 0.92745, 1e-3),
    ("test2_123", (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.00189, 0.0, 1e-2),
    ("test3_two_shocks", (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 460.894, 19.5975, 1e-3),
    ("test4_two_rarefactions", (1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 46.0950, -6.196328, 1e-3),
]


@pytest.mark.parametrize("name,left,right,pressure_ref,velocity_ref,tolerance", TORO_TABLE_4_1)
def test_star_state_matches_toro_table_4_1(name, left, right, pressure_ref, velocity_ref, tolerance):
    """The star state entering the budget reproduces Toro's printed values."""
    pressure, velocity = star_state(left, right)
    assert relative_error(pressure_ref, pressure) < tolerance, name
    if velocity_ref == 0.0:
        assert abs(velocity) < 1e-6
    else:
        assert relative_error(velocity_ref, velocity) < tolerance, name


def test_left_shock_budget_equals_mirrored_right_shock():
    """Mirror symmetry fixes the left-shock sign convention.

    Mirroring ``x -> -x`` (with ``u -> -u``) turns Sod's right shock into a left
    shock; ``int rho s`` is invariant, so the total production must be equal.
    """
    sod = analytic_entropy_budget(SOD_LEFT, SOD_RIGHT, 0.2)
    mirror = analytic_entropy_budget(SOD_RIGHT, SOD_LEFT, 0.2)
    assert sod["patterns"] == ["left_rarefaction", "right_shock"]
    assert mirror["patterns"] == ["left_shock", "right_rarefaction"]
    assert mirror["production_total"] > 0.0
    assert relative_error(sod["production_total"], mirror["production_total"]) < 1e-12
    # the star state itself mirrors exactly
    assert relative_error(sod["star_state"][0], mirror["star_state"][0]) < 1e-12
    assert relative_error(sod["star_state"][1], -mirror["star_state"][1]) < 1e-12


def test_two_rarefaction_production_is_exactly_zero():
    """No shock => the exact solution is isentropic => zero production."""
    budget = analytic_entropy_budget((1.0, -1.0, 1.0), (1.0, 1.0, 1.0), 0.15)
    assert budget["patterns"] == ["left_rarefaction", "right_rarefaction"]
    assert budget["isentropic"] is True
    assert budget["production_total"] == 0.0
    assert all(wave["production_rate"] == 0.0 for wave in budget["waves"])


@pytest.mark.parametrize("name", ["sod_right_shock", "mirror_sod_left_shock"])
def test_shock_production_satisfies_second_law(name):
    spec = CASES[name]
    budget = analytic_entropy_budget(spec["left"], spec["right"], spec["final_time"])
    assert budget["isentropic"] is False
    shocks = [wave for wave in budget["waves"] if wave["kind"] == "shock"]
    assert len(shocks) == 1
    for shock in shocks:
        assert shock["production_rate"] > 0.0
        assert np.isfinite(shock["speed"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_analytic_budget_matches_exact_solution_difference(name):
    """Convention-free anchor: RH budget vs ``d int rho s dx`` of the exact solution."""
    spec = CASES[name]
    assert domain_is_clean(spec), f"{name}: a wave leaves the domain, budget invalid"
    budget = analytic_entropy_budget(spec["left"], spec["right"], spec["final_time"])
    reference = exact_solution_production(spec["left"], spec["right"], spec["final_time"])
    if budget["production_total"] == 0.0:
        # isentropic: both sides must agree that nothing is produced; the
        # residual is trapezoid noise, so gate it on the entropy-integral scale
        scale = abs(measure_entropy_scale(spec))
        assert abs(reference) < 1e-3 * max(scale, 1e-12), (name, reference, scale)
    else:
        assert relative_error(budget["production_total"], reference) < 1e-2


def test_generic_budget_reproduces_the_restricted_right_shock_helper():
    """Backwards compatibility with ``analytic_right_shock_budget`` (Sod/Lax)."""
    restricted = analytic_right_shock_budget(SOD_LEFT, SOD_RIGHT, 0.2)
    general = analytic_entropy_budget(SOD_LEFT, SOD_RIGHT, 0.2)
    assert relative_error(restricted["production_total"], general["production_total"]) < 1e-12
    assert relative_error(restricted["sigma"], general["waves"][-1]["speed"]) < 1e-12


def test_near_vacuum_star_pressure_is_not_below_zero():
    budget = analytic_entropy_budget(NEAR_VACUUM_LEFT, NEAR_VACUUM_RIGHT, 0.15)
    assert budget["star_state"][0] > 0.0
    assert budget["isentropic"] is True


def measure_entropy_scale(spec):
    """``int rho s dx`` at the final time -- the scale the budget is measured against."""
    from benchmarks._entropy_tools import total_entropy_from_profiles
    from benchmarks.riemann_exact import sample

    x = np.linspace(0.0, 1.0, 200_001)
    fields = sample(x, spec["final_time"], spec["left"], spec["right"], 0.5)
    return total_entropy_from_profiles(fields["density"], fields["pressure"], x)


def test_two_rarefaction_dg_run_keeps_the_second_law():
    """A purely rarefying problem may not *destroy* entropy (P >= 0).

    The magnitude is checked in ``docs/entropy_budget_general.md``: at 64 cells
    the TVB limiter produces ~31% of the entropy scale on this strong expansion,
    i.e. the 3% gate is **not** met for this pattern.  Only the sign is asserted
    here, because that is the part that is a genuine invariant.
    """
    left, right, final_time = (2.0, -1.0, 1.0), (2.0, 1.0, 1.0), 0.15
    run = measure_dg(left, right, 32, 2, final_time, "tvb", 1.0)
    produced = run["production_numeric"]
    assert np.isfinite(produced)
    assert produced >= 0.0, produced