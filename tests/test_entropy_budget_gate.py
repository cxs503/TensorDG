"""Regression tests for the entropy-budget *harness* (not the solver).

Three kinds of defects are pinned here, all of them found while re-running
``benchmarks/entropy_budget_1d.py`` on finer meshes:

1. **Gate semantics.** The "relative error at the finest mesh" verdict used
   ``min(rows, key=cells)`` -- it graded the *coarsest* mesh, so the honest-FAIL
   verdict in the first version of ``docs/entropy_verification.md`` was partly a
   labelling bug.  :func:`benchmarks.entropy_budget_1d.finest_row` is the single
   place that decides which mesh is the gate; it is tested against a shuffled row
   list.
2. **Meaningless profile metric.** ``rho s`` is signed and crosses zero, so
   ``||num-exact|| / ||exact||`` reached 163068 % on Sod 128 cells and carried no
   information.  :func:`benchmarks._entropy_tools.entropy_profile_error` is
   pinned to give an interpretable RMS error in entropy-density units and a
   gap-normalised value.
3. **Rankine-Hugoniot production identity.** ``P = sigma[rho s] - [rho u s]``
   must equal ``m [s]`` with the mass flux ``m = rho_a (sigma - u_a)``.  The two
   formulas are algebraically independent given the implementation, so the
   identity is a real check of the analytical baseline every shock run is
   graded against.

Run: ``pytest -q tests/test_entropy_budget_gate.py``
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density as _entropy_density,
    entropy_jump_scale,
    entropy_profile_error,
    gas_entropy,
    observed_order,
    shock_speed_and_production,
)
from benchmarks.entropy_budget_1d import (
    LAX_LEFT,
    LAX_RIGHT,
    LEFT,
    RIGHT,
    SOD_TIME,
    finest_row,
    shock_study,
)
from benchmarks.riemann_exact import sample

SOD_BUDGET = analytic_right_shock_budget(LEFT, RIGHT, SOD_TIME)


# --------------------------------------------------------------------------- #
# 1. finest-mesh gate semantics
# --------------------------------------------------------------------------- #
def test_finest_row_selects_the_largest_cell_count():
    rows = [
        {"cells": 256, "relative_error": 0.031},
        {"cells": 64, "relative_error": 0.117},
        {"cells": 1024, "relative_error": 0.009},
        {"cells": 128, "relative_error": 0.073},
    ]
    assert finest_row(rows)["cells"] == 1024
    # the historical defect: min() would have returned the 64-cell row at 11.7 %
    assert min(rows, key=lambda row: row["cells"])["relative_error"] == pytest.approx(0.117)
    assert finest_row(rows)["relative_error"] < 0.03


def test_finest_row_is_order_independent():
    rows = [{"cells": n, "relative_error": 1.0 / n} for n in (64, 128, 256)]
    forward = finest_row(rows)
    backward = finest_row(list(reversed(rows)))
    assert forward == backward == {"cells": 256, "relative_error": 1.0 / 256}


# --------------------------------------------------------------------------- #
# 2. analytic Rankine-Hugoniot production identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("left,right", [(LEFT, RIGHT), (LAX_LEFT, LAX_RIGHT)])
def test_production_equals_mass_flux_times_entropy_jump(left, right):
    """P = m [s] with m = rho_a (sigma - u_a): independent of eq. (3) as coded."""
    budget = analytic_right_shock_budget(left, right, 1.0)
    # the shock's upstream side is the *right* Riemann state, by construction
    rho_a, u_a, p_a = (float(v) for v in budget["upstream"])
    down = budget["downstream"]
    rho_b = down[0]
    sigma = budget["sigma"]
    mass_flux = rho_a * (sigma - u_a)
    assert rho_b * (sigma - down[1]) == pytest.approx(mass_flux, rel=1e-12)
    jump = float(gas_entropy(down[0], down[2]) - gas_entropy(rho_a, p_a))
    assert budget["production"] == pytest.approx(mass_flux * jump, rel=1e-12)
    assert budget["production"] > 0.0  # second law for a compressive shock


def test_sod_budget_matches_the_documented_value():
    """Pin the baseline the report and the docs quote (mass-flux identity)."""
    assert SOD_BUDGET["production_total"] == pytest.approx(2.36462089397162e-03, rel=1e-12)
    assert SOD_BUDGET["sigma"] == pytest.approx(1.752155732030178, rel=1e-12)
    assert SOD_BUDGET["shock_position"] == pytest.approx(0.850431, abs=1e-6)


def test_entropy_jump_scale_is_the_shock_entropy_step():
    scale = entropy_jump_scale(SOD_BUDGET["upstream"], SOD_BUDGET["downstream"])
    rho_a, _, p_a = SOD_BUDGET["upstream"]
    rho_b, _, p_b = SOD_BUDGET["downstream"]
    assert scale == pytest.approx(float(gas_entropy(rho_b, p_b) - gas_entropy(rho_a, p_a)))
    assert scale > 0.0


# --------------------------------------------------------------------------- #
# 3. the profile metric must survive a sign-changing entropy density
# --------------------------------------------------------------------------- #
def _sod_exact_profile(samples=1001):
    x = np.linspace(0.0, 1.0, samples)
    fields = sample(x, SOD_TIME, LEFT, RIGHT)
    return x, _entropy_density(fields["density"], fields["pressure"])


def test_entropy_density_really_changes_sign_on_sod():
    """Documents *why* a relative L2 norm of rho s is meaningless here."""
    _, exact = _sod_exact_profile()
    assert exact.min() < 0.0 < exact.max()
    assert abs(exact.min()) < 1.0  # the two signs nearly cancel in ||exact||


def test_profile_metric_is_finite_and_scales_with_the_perturbation():
    x, exact = _sod_exact_profile()
    scale = entropy_jump_scale(SOD_BUDGET["upstream"], SOD_BUDGET["downstream"])
    zero = entropy_profile_error(exact, exact, x, scale)
    assert zero["rms"] == pytest.approx(0.0)
    assert zero["rms_over_entropy_jump"] == pytest.approx(0.0)

    step = 0.01
    perturbed = exact + step
    metrics = entropy_profile_error(perturbed, exact, x, scale)
    # a constant offset of 0.01 must be reported as an RMS error of 0.01 (in rho s units)
    assert metrics["rms"] == pytest.approx(step, rel=1e-12)
    assert metrics["rms_over_entropy_jump"] == pytest.approx(step / scale, rel=1e-12)
    # the deprecated normalisation is a different (useless) number, kept only for provenance
    assert metrics["legacy_relative_l2"] > 0.0


def test_legacy_metric_is_dominated_by_one_near_vacuum_sample():
    """Why ``||e|| / ||exact||`` is not a field error for ``rho s``.

    ``s = ln(p / rho**gamma)`` diverges as ``p -> 0``: a single sample inside a
    smeared shock where the pressure is nearly zero is enough to make the legacy
    ratio O(100 %) -- this is the mechanism behind the 163068 % entry that the
    first version of the report quoted as a profile error.
    """
    x, exact = _sod_exact_profile()
    scale = entropy_jump_scale(SOD_BUDGET["upstream"], SOD_BUDGET["downstream"])
    spike = exact.copy()
    spike[len(spike) // 2] -= 20.0  # one near-vacuum sample, rho*s ~ -20
    metrics = entropy_profile_error(spike, exact, x, scale)
    assert metrics["legacy_relative_l2"] > 5.0
    assert metrics["rms"] < 1.0
    assert metrics["rms_over_entropy_jump"] < 20.0


def test_profile_metric_reports_the_deprecated_value_for_provenance():
    """The legacy number is kept in the JSON, so it must still be computable."""
    x, exact = _sod_exact_profile()
    metrics = entropy_profile_error(exact, exact, x)
    assert "legacy_relative_l2" in metrics
    assert "rms_over_entropy_jump" not in metrics  # only defined with a scale


def test_profile_metric_is_sample_count_independent():
    """An RMS error must not change when the same field is sampled more densely."""
    coarse_x, coarse = _sod_exact_profile(1001)
    fine_x, fine = _sod_exact_profile(2001)
    scale = entropy_jump_scale(SOD_BUDGET["upstream"], SOD_BUDGET["downstream"])
    coarse_metrics = entropy_profile_error(coarse + 0.02, coarse, coarse_x, scale)
    fine_metrics = entropy_profile_error(fine + 0.02, fine, fine_x, scale)
    assert coarse_metrics["rms"] == pytest.approx(0.02, rel=1e-3)
    assert fine_metrics["rms"] == pytest.approx(coarse_metrics["rms"], rel=1e-3)


# --------------------------------------------------------------------------- #
# 4. limiter-stage coverage and second law on a tiny case
# --------------------------------------------------------------------------- #
def test_shock_study_records_the_limiter_stage_and_skips_it_without_limiter():
    """``limiter_stage`` is inert without a limiter: do not spend a run on it."""
    results = shock_study([32], [(None, 1.0), ("tvb", 1.0)], degree=2, cfl=0.3,
                          limiter_stages=("every", "final"))
    rows = results["sod"]["rows"]
    none_rows = [row for row in rows if row["limiter"] == "none"]
    tvb_rows = [row for row in rows if row["limiter"] == "tvb"]
    assert len(none_rows) == 1 and none_rows[0]["limiter_stage"] == "every"
    assert {row["limiter_stage"] for row in tvb_rows} == {"every", "final"}
    assert {row["label"] for row in tvb_rows} == {"tvb", "tvb@final"}
    # C1 on the smiling case, including the previously uncovered "final" branch
    assert all(row["second_law"] for row in rows)


def test_budget_error_order_helper_matches_an_analytic_series():
    """The EOC ladder used for the C2 convergence claim is itself verified."""
    cells = [64, 128, 256]
    order = observed_order(cells, [c ** -1.0 * 0.5 for c in cells])
    assert order == pytest.approx([1.0, 1.0], rel=1e-12)