"""Invariants of the shock-regime band-honesty study.

These tests exist to make the study's *claims* falsifiable rather than to show
that the script runs:

* the manufactured truth and its modelled ladders are **exact** (closed-form
  cell averages), so any error the study reports is solver error and any failure
  of nesting consistency is a bug in the harness;
* the shock ladders really are in the ``p ~ 1`` regime (the regime CASE-SHU-OSHER
  is certified in), not silently smooth;
* the measured under-coverage factor stays bounded by
  :data:`BAND_FACTOR_SHOCK`, i.e. the recommended safety factor is a *measured*
  quantity that a future change to the GCI machinery would break;
* the vacuity rule of :func:`tensordg.certified_reference.judge` never produces a
  PASS for a candidate whose true error exceeds the gate (no false PASS);
* a ladder whose differences cancel to the round-off floor is refused, and the
  error it hides is orders of magnitude larger than the difference that hid it.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.certified_reference_shock_coverage import (
    BAND_FACTOR_SHOCK,
    cell_averages,
    jump_integral,
    modeled_levels,
    ramp_integral,
    run_study,
    shifted_levels,
    smooth_error_levels,
    smooth_integral,
    study_case,
)
from tensordg.certified_reference import (
    CERTIFIED,
    WITHHELD,
    WITHHELD_ROUNDOFF_FLOOR,
    certify,
    coarsen,
    judge,
    rel_l1,
)


# ------------------------------------------------------------------ exactness
def test_smooth_integral_matches_quadrature():
    a, b = np.array([0.13, 0.51]), np.array([0.29, 0.77])
    x, w = np.polynomial.legendre.leggauss(40)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    numeric = np.array([half[i] * np.sum(w * (1.0 + 0.2 * np.sin(6.0 * (mid[i] + half[i] * x))))
                        for i in range(a.size)])
    exact = smooth_integral(a, b, 1.0, 0.2, 6.0)
    assert np.allclose(exact, numeric, rtol=0.0, atol=1e-13)


def test_ramp_and_jump_integrals_are_exact_and_consistent():
    # a ramp of width w integrates to w/2 plus the length beyond the ramp
    w, xs = 0.02, 0.41
    total = ramp_integral(np.array([0.0]), np.array([1.0]), xs, w)
    assert total[0] == pytest.approx(0.5 * w + (1.0 - xs - 0.5 * w), rel=0.0, abs=1e-15)
    # the ramp vanishes everywhere left of the ramp, exactly
    assert ramp_integral(np.array([0.0]), np.array([xs - 0.6 * w]), xs, w)[0] == 0.0
    # and a vanishing width reproduces the step exactly
    assert ramp_integral(np.array([0.0]), np.array([1.0]), xs, 1e-14)[0] == pytest.approx(
        jump_integral(np.array([0.0]), np.array([1.0]), xs)[0], rel=1e-9)
    assert jump_integral(np.array([0.2]), np.array([0.6]), 0.4)[0] == pytest.approx(0.2)


def test_nested_cell_averages_nest_exactly():
    """The study's whole "no ambiguity" claim: coarsening is exact, not interpolated."""
    for width in (0.0, 0.01, 0.04):
        fine = cell_averages(400, xs=0.4131, jump=1.0, width=width, s1=0.2, k=6.0)
        coarse = cell_averages(200, xs=0.4131, jump=1.0, width=width, s1=0.2, k=6.0)
        assert np.allclose(coarsen(fine, 2), coarse, rtol=0.0, atol=1e-13)


# ---------------------------------------------------------- regime sanity
def test_shock_shift_ladder_is_first_order():
    levels = shifted_levels((50, 100, 200), shift=0.5, xs=0.41, jump=1.0, s1=0.2, k=6.0)
    cert = certify(levels, tolerance_percent=3.0)
    assert cert["verdict"] == CERTIFIED
    assert cert["observed_order_used"] == pytest.approx(1.0, abs=0.05)


def test_smooth_control_ladder_recovers_its_modeled_order():
    levels = smooth_error_levels((50, 100, 200), p=2.0, jump=0.0)
    cert = certify(levels, tolerance_percent=3.0)
    assert cert["verdict"] == CERTIFIED
    assert cert["observed_order_used"] == pytest.approx(2.0, abs=0.05)


def test_exact_data_is_refused_not_certified():
    """Zero differences carry no order information: the engine must withhold."""
    levels = [{"ncells": n, "field": cell_averages(n, xs=0.41, jump=1.0)}
              for n in (50, 100, 200)]
    cert = certify(levels, tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_ROUNDOFF_FLOOR
    assert cert["gate_usable"] is False
    # and a candidate cannot be blessed against a refused reference
    dec = judge(cell_averages(200, xs=0.41, jump=1.0), cert, tolerance_percent=3.0)
    assert dec["decision"] == WITHHELD


# ------------------------------------------------------- band honesty bounds
def test_band_covers_in_the_smooth_regime_and_undercovers_in_the_shock_regime():
    smooth = study_case(regime="smooth", jump=0.0, error_order=2.0)
    assert smooth["coverage"] is True
    assert smooth["true_over_band"] < 0.05          # band is very conservative here

    shock = study_case(regime="shift", beta=2.0, xs=0.5, jump=1.0)
    assert shock["certified"] is True
    # the measured failure mode: the band can UNDER-cover in this regime
    assert shock["true_over_band"] > 1.0
    assert shock["true_over_band"] <= BAND_FACTOR_SHOCK + 1e-9


def test_masked_error_of_a_roundoff_floor_ladder():
    """Differences that vanish do not imply a converged profile."""
    row = study_case(regime="shock", beta=2.0, xs=0.41, jump=1.0, k=6.0)
    assert row["verdict"] == WITHHELD_ROUNDOFF_FLOOR
    assert row["d_fine_percent"] < 1e-10           # the ladder looks converged
    assert row["raw_finest_true_error_percent"] > 0.01   # truth disagrees
    assert row["masked_error_factor"] > 1e6


# ------------------------------------------------------------ whole study
@pytest.fixture(scope="module")
def study():
    return run_study(ladder=(50, 100, 200), tolerance_percent=3.0)


def test_study_summary_is_bounded_and_falsifiable(study):
    s = study["summary"]
    al = s["shock_all"]
    assert al["n_cases"] == 192
    assert al["n_certified"] > 0
    # coverage in the shock regime is genuinely worse than the smooth control
    assert al["coverage_fraction_certified"] < 1.0
    assert s["smooth_error_control"]["coverage_fraction_certified"] == 1.0
    # the recommended safety factor is measured, not chosen: nothing may exceed it
    for key in ("shock_smearing", "shock_shift", "shock_all"):
        assert s[key]["worst_true_over_band"] <= BAND_FACTOR_SHOCK + 1e-9
    # observed order in the shock regime is ~1, i.e. the Shu-Osher regime
    assert al["min_observed_order"] >= 0.99
    # exact data is always refused
    assert s["exact_control"]["all_refused"] is True


def test_study_produces_no_false_pass(study):
    """A candidate worse than the gate in truth must never come back as PASS."""
    rows = study["cases"] + study["controls"]
    assert sum(1 for r in rows if r["false_pass"]) == 0
    checked = 0
    for r in rows:
        for probe in r["probes"]:
            if probe["false_pass"]:
                checked += 1
    assert checked == 0
    # at least some probes really are worse than the gate, otherwise the check
    # above would be vacuous
    assert any(not p["passes_gate_in_truth"] for r in rows for p in r["probes"])


def test_study_is_deterministic():
    a = study_case(regime="shock", beta=2.0, xs=0.41, jump=0.5, k=6.0)
    b = study_case(regime="shock", beta=2.0, xs=0.41, jump=0.5, k=6.0)
    assert a["band_percent"] == b["band_percent"]
    assert a["true_error_percent"] == b["true_error_percent"]
    np.testing.assert_allclose(
        modeled_levels((30, 60), beta=1.0, xs=0.41, jump=1.0)[-1]["field"],
        modeled_levels((30, 60), beta=1.0, xs=0.41, jump=1.0)[-1]["field"])


def test_markdown_and_json_are_writable(tmp_path, study):
    from benchmarks.certified_reference_shock_coverage import _jsonable, write_markdown
    import json

    md = write_markdown(study, tmp_path / "study.md")
    text = md.read_text()
    assert "Band honesty" in text or "band honesty" in text.lower()
    js = json.dumps(_jsonable(study))
    assert "coverage_fraction_certified" in js
    assert "nan" not in js.lower()