"""Tests for the CASE-LAX-DG norm-convergence audit.

The properties pinned here are the ones an acceptance verdict rests on:

* the audit's single-panel path *is* the shipped measurement (bit-reproduces the
  committed metric), so the comparison is not comparing two different things;
* the reference evaluation is an honest integral -- it covers the domain exactly
  (measure invariant), it is converged in its own rule, and it agrees with an
  independent brute-force integration;
* the corrected A0 arm is still a **FAIL** -- the audit must not be usable to
  flip an honest FAIL into a PASS by re-post-processing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lax_case_pool import production_errors_from_profiles
from benchmarks.lax_norm_convergence import (
    FIELDS, PROFILES, REFERENCE_ORDER, _grid, _physical_to_xi,
    bias_table, breakpoint_resolved_norms, cell_polynomials, convergence_study,
    error_share, evaluate_cells, load_profiles, oscillation_metrics,
    re_adjudicate, single_panel_norms)

OUT = Path("docs/benchmarks/results/lax_norm_convergence")


@pytest.fixture(scope="module")
def prof():
    return load_profiles(PROFILES)


@pytest.fixture(scope="module")
def reference(prof):
    return breakpoint_resolved_norms(prof, order=REFERENCE_ORDER)


@pytest.fixture(scope="module")
def verdict(prof):
    return re_adjudicate(prof)


# --------------------------------------------------------------------------- #
# 1. the audit measures the shipped quantity
# --------------------------------------------------------------------------- #
def test_single_panel_path_bit_reproduces_the_committed_metric(prof):
    committed = production_errors_from_profiles(PROFILES, quadrature_order=10)
    mine = single_panel_norms(prof, order=10)
    for field in FIELDS:
        assert mine["l2"][field] == pytest.approx(committed["l2"][field],
                                                  rel=1e-12)
        assert mine["l1"][field] == pytest.approx(committed["l1"][field],
                                                  rel=1e-12)


def test_committed_arm_matches_the_published_number(verdict):
    # docs/CASE-LAX-DG.md publishes 3.11556e-02 as the A0 ledger arm.
    assert verdict["committed"]["worst"] == pytest.approx(3.11556e-02, rel=1e-6)
    assert verdict["committed"]["status"] == "FAIL"
    assert verdict["production_leg"] == "p3.N400"


# --------------------------------------------------------------------------- #
# 2. the reference evaluation is an honest integral
# --------------------------------------------------------------------------- #
def test_reference_grid_covers_the_domain_exactly(prof):
    """Catches truncated element extents (the class of bug this audit found)."""
    nodes, weights = np.polynomial.legendre.leggauss(REFERENCE_ORDER)
    breaks = sorted(np.asarray(reference_breakpoints()))
    _, w = _grid(prof, nodes, weights, breaks)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
    # the single-panel grid must cover it too
    _, w_single = _grid(prof, nodes, weights, [])
    assert float(w_single.sum()) == pytest.approx(1.0, abs=1e-12)


def reference_breakpoints():
    from benchmarks.lax_case_pool import exact_waves
    return list(exact_waves()["positions"].values())


def test_element_polynomial_is_recovered_to_roundoff(prof):
    for field in FIELDS:
        coeffs = cell_polynomials(prof, field)
        samples = prof["fields"][field]["numerical"]
        worst = max(np.max(np.abs(np.polyval(coef, prof["xi"][c]) - samples[c]))
                    for c, coef in enumerate(coeffs))
        assert worst < 1e-10, f"{field}: interpolation residual {worst:.2e}"


def test_reference_norms_are_converged_in_their_own_rule():
    study = convergence_study(load_profiles(PROFILES), orders=(12, 16, 24, 32))
    assert study["converged_at_1e_6"], study["worst_l2_spread"]
    assert study["worst_l2_spread"] < 1e-6


def test_reference_agrees_with_an_independent_brute_force_integral(prof):
    """Dense midpoint rule per element: shares no code with the Gauss path."""
    npts = 800
    dx, centres = prof["dx"], prof["centres"]
    xs = centres[:, None] + dx * ((np.arange(npts) + 0.5) / npts - 0.5)[None, :]
    ws = np.full_like(xs, dx / npts)
    from benchmarks.lax_case_pool import exact_reference
    exact = exact_reference(xs)
    reference = breakpoint_resolved_norms(prof, order=REFERENCE_ORDER)
    for field in FIELDS:
        numerical = evaluate_cells(cell_polynomials(prof, field),
                                   _physical_to_xi(prof, xs))
        l2 = np.sqrt(np.sum(ws * (numerical - exact[field]) ** 2)
                     / np.sum(ws * exact[field] ** 2))
        assert l2 == pytest.approx(reference["l2"][field], rel=1e-3)


# --------------------------------------------------------------------------- #
# 3. the defect is real and quantified
# --------------------------------------------------------------------------- #
def test_shipped_single_panel_rule_overstates_the_error(prof, reference):
    q10 = single_panel_norms(prof, order=10)
    biases = {f: q10["l2"][f] / reference["l2"][f] - 1.0 for f in FIELDS}
    for field in FIELDS:
        assert biases[field] > 0.0, f"{field}: expected an over-estimate"
    assert biases["velocity"] > 0.02
    assert biases["pressure"] > 0.02
    assert biases["density"] > 0.02


def test_no_single_panel_order_is_trustworthy(prof):
    """The bias oscillates instead of vanishing: this is why q=10 could not be
    tuned away, and why the audit must split elements rather than raise q."""
    rows = bias_table(prof, orders=(4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96))
    osc = oscillation_metrics(rows)
    for field in FIELDS:
        assert osc[field]["monotone_under_refinement"] is False
        assert osc[field]["sign_changes"] >= 1
    assert max(osc[f]["max_abs_bias"] for f in FIELDS) > 0.05


def test_the_verdict_is_decided_by_a_handful_of_wave_cells(prof):
    share = error_share(prof)
    assert share["cells"] <= 0.05 * share["cells_total"]
    assert share["per_field"]["velocity"]["numerator_share"] > 0.4
    # ... while those same cells hold almost none of the solution norm
    assert share["per_field"]["velocity"]["denominator_share"] < 0.1


# --------------------------------------------------------------------------- #
# 4. honest FAIL is preserved
# --------------------------------------------------------------------------- #
def test_corrected_ledger_arm_is_still_a_fail(verdict):
    corrected = verdict["corrected"]
    assert corrected["status"] == "FAIL"
    assert corrected["worst"] > verdict["tolerance"]
    assert verdict["verdict_flips"] is False


def test_the_correction_shrinks_but_does_not_erase_the_miss(verdict):
    """The *excess over the bar* is what the bias inflated, not the ratio."""
    committed = verdict["committed"]
    corrected = verdict["corrected"]
    assert corrected["worst"] < committed["worst"]
    assert committed["excess_over_bar"] > 0.0
    assert corrected["excess_over_bar"] > 0.0   # honest FAIL survives
    # the published margin overstated the miss by a factor > 3
    assert (committed["excess_over_bar"]
            / corrected["excess_over_bar"]) > 3.0


def test_ladder_l1_arm_is_reported_as_not_re_measured(verdict):
    arm = verdict["ladder_l1_arm"]
    assert arm["re_measured"] is False
    assert arm["reason"]


# --------------------------------------------------------------------------- #
# 5. committed artifacts are self-consistent
# --------------------------------------------------------------------------- #
def test_committed_metrics_json_is_present_and_self_consistent(verdict):
    stored = json.loads((OUT / "metrics.json").read_text())
    assert stored["reference_convergence"]["converged_at_1e_6"] is True
    assert stored["re_adjudication"]["verdict_flips"] is False
    assert stored["re_adjudication"]["corrected"]["worst"] == pytest.approx(
        verdict["corrected"]["worst"], rel=1e-9)
    assert stored["anchor"] == "CASE-LAX-DG"
    for field in FIELDS:
        assert stored["interpolation_residual"][field] < 1e-10


def test_committed_figure_exists():
    assert (OUT / "figures" / "norm_quadrature.png").stat().st_size > 10_000