"""Gates for the *discriminating* spectral-convergence criterion.

Case: CASE-DG-PREFINEMENT-EXPONENTIAL
Closes audit finding F4 of docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md:
the published pass criterion ``sigma > 0.5`` accepts algebraic error laws, so the
verdict is now gated on a composite, predeclared criterion

    sigma > 1  AND  pairwise decay rates non-decreasing
               AND  e(p_min)/e(p_max) > 1e3  AND  error monotone decreasing

These tests pin the criterion from three sides: synthetic laws whose answer is
known a priori, *real solver output* for non-smooth data (which must be
rejected), and the shipped artifact (which must be accepted).

Run::
    pytest -q tests/test_p_refinement_criterion.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.p_refinement_exponential import (
    DEGREES_FULL,
    ERROR_RATIO_MIN,
    SIGMA_MIN_DISCRIMINATING,
    check_criterion,
    check_discriminating_criterion,
    falsification_control,
)
from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import (
    FLOAT64_EPS,
    p_decay_diagnostics,
    resolved_levels,
    roundoff_floor,
    signal_to_noise,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "p-refinement-exponential" / "p_refinement_results.json"
FLOOR_ARTIFACT = ARTIFACT.parent / "measurement_floor.json"


# --- synthetic laws: the criterion must land on the predeclared side ---------

def test_falsification_suite_meets_every_predeclared_verdict():
    control = falsification_control(DEGREES_FULL)
    assert control["gate_pass"] is True, control["laws"]
    for name, row in control["laws"].items():
        assert row["as_expected"] is True, (name, row)


def test_algebraic_laws_are_rejected_by_the_discriminating_criterion():
    control = falsification_control(DEGREES_FULL)
    for name in ("constant", "algebraic_p^-1", "algebraic_p^-2", "algebraic_p^-3",
                 "algebraic_p^-4", "algebraic_p^-5"):
        row = control["laws"][name]
        assert row["passes_discriminating"] is False, (name, row)
    # the reduction floor disposes of the low-power laws on its own ...
    for name in ("constant", "algebraic_p^-1", "algebraic_p^-2", "algebraic_p^-3"):
        assert control["laws"][name]["error_ratio"] < ERROR_RATIO_MIN, name


def test_high_power_algebraic_laws_are_only_caught_by_the_shape_condition():
    """p^-4 (sigma=1.37) and p^-5 (sigma=1.71) clear both scalar gates.

    Over p=1..6 their total reduction is 6**4 = 1296x and 6**5 = 7776x, i.e. above
    the 1e3 floor, and sigma is above 1 -- only the non-decreasing pairwise decay
    rates reveal that the law is algebraic, which is why that condition is not
    optional.
    """
    control = falsification_control(DEGREES_FULL)
    for name, dofs_power in (("algebraic_p^-4", 4), ("algebraic_p^-5", 5)):
        row = control["laws"][name]
        assert row["error_ratio"] == pytest.approx(6.0 ** dofs_power), (name, row)
        assert row["error_ratio"] > ERROR_RATIO_MIN, (name, row)
        assert row["lsq_decay_rate"] > SIGMA_MIN_DISCRIMINATING, (name, row)
        rates = row["decay_rates"]
        assert all(b < a for a, b in zip(rates, rates[1:])), (name, rates)
        assert row["passes_discriminating"] is False, (name, row)


def test_legacy_threshold_has_documented_false_positives():
    """The point of the change: sigma > 0.5 alone accepts algebraic laws."""
    control = falsification_control(DEGREES_FULL)
    false_positives = set(control["legacy_threshold_false_positives"])
    assert {"algebraic_p^-2", "algebraic_p^-3"} <= false_positives, control["laws"]
    # ... and the composite criterion rejects every one of them
    for name in false_positives:
        assert control["laws"][name]["passes_discriminating"] is False, name


def test_sigma_alone_cannot_see_an_algebraic_law_above_one():
    """p^-3 has sigma = 1.03 > 1, so the *shape* condition is what saves us."""
    control = falsification_control(DEGREES_FULL)
    coarse = control["laws"]["algebraic_p^-3"]
    assert coarse["lsq_decay_rate"] > SIGMA_MIN_DISCRIMINATING, coarse
    assert coarse["passes_discriminating"] is False, coarse
    rates = coarse["decay_rates"]
    assert all(b < a for a, b in zip(rates, rates[1:])), rates


def test_genuine_exponential_laws_are_accepted():
    control = falsification_control(DEGREES_FULL)
    for name in ("exponential_sigma_1.5", "exponential_sigma_3.93"):
        assert control["laws"][name]["passes_discriminating"] is True, name


def test_threshold_band_exponential_is_not_certified():
    """A 12x reduction over p=1..6 cannot certify spectral decay, whatever sigma says."""
    control = falsification_control(DEGREES_FULL)
    row = control["laws"]["exponential_sigma_0.5"]
    assert row["error_ratio"] < ERROR_RATIO_MIN
    assert row["passes_discriminating"] is False, row


# --- real solver output: non-smooth data must never be certified -------------

def _p_ladder(profile, degrees, ncells=16, cfl=0.08):
    errors = []
    for degree in degrees:
        centers, coeff, _ = solve_advection(
            ncells=ncells, degree=degree, final_time=1.0, cfl=cfl,
            initial=profile, time_integrator="rk4",
        )
        errors.append(l2_error(centers, coeff, 1.0, exact=profile))
    return errors


def _triangle(x):
    """C^0 kink: DG errors decay algebraically in p (no analyticity strip)."""
    xr = torch.remainder(x, 1.0)
    return torch.where(xr < 0.5, 2.0 * xr, 2.0 * (1.0 - xr))


def _square(x):
    return torch.where(torch.remainder(x, 1.0) < 0.5,
                       torch.ones_like(x), -torch.ones_like(x))


def test_real_kinked_solution_passes_the_legacy_gate_but_is_rejected():
    """Live counterexample: the old sigma > 0.5 gate certifies an algebraic case."""
    degrees = (1, 2, 3, 4)
    errors = _p_ladder(_triangle, degrees)
    diag = p_decay_diagnostics(degrees, errors)
    assert diag["monotone_decreasing"] is True, errors
    # algebraic decay of a kinked profile: sigma hovers at the legacy threshold
    assert diag["lsq_decay_rate"] > 0.5, (diag["lsq_decay_rate"], errors)
    verdict = check_discriminating_criterion(diag)
    assert verdict["pass"] is False, verdict
    assert verdict["conditions"]["decay_rates_non_decreasing"]["pass"] is False
    assert verdict["conditions"]["error_ratio_above_threshold"]["pass"] is False


def test_real_shock_solution_is_rejected():
    degrees = (1, 2, 3)
    errors = _p_ladder(_square, degrees)
    diag = p_decay_diagnostics(degrees, errors)
    verdict = check_discriminating_criterion(diag)
    assert verdict["pass"] is False, verdict
    assert verdict["conditions"]["sigma_above_threshold"]["pass"] is False


# --- shipped artifact -------------------------------------------------------

def _artifact():
    if not ARTIFACT.exists():
        pytest.skip("run python -m benchmarks.p_refinement_exponential")
    return json.loads(ARTIFACT.read_text())


def test_artifact_verdict_is_gated_on_the_discriminating_criterion():
    art = _artifact()
    conditions = art["checks"]["discriminating"]["conditions"]
    for name, cond in conditions.items():
        assert cond["pass"] is True, (name, cond)
    assert art["checks"]["discriminating"]["pass"] is True
    assert art["checks"]["overall_pass"] is True
    assert art["verdict"] == "PASS"


def test_artifact_records_the_criterion_and_the_falsification_evidence():
    art = _artifact()
    assert art["criterion"]["legacy_threshold"] == 0.5
    assert "sigma > 1" in art["criterion"]["statement"]
    control = art["falsification_control"]
    assert control["gate_pass"] is True, control["laws"]
    assert "algebraic_p^-3" in control["legacy_threshold_false_positives"]


def test_artifact_criterion_reproduces_from_stored_errors():
    art = _artifact()
    diag = p_decay_diagnostics(tuple(art["results"]["degrees"]), art["results"]["errors"])
    recomputed = check_criterion(diag)
    assert recomputed["discriminating"] == art["checks"]["discriminating"], (
        recomputed["discriminating"], art["checks"]["discriminating"])
    assert recomputed["overall_pass"] == art["checks"]["overall_pass"]


def test_artifact_numbers_agree_with_the_documented_sigma():
    art = _artifact()
    sigma = art["results"]["diagnostics"]["lsq_decay_rate"]
    assert 3.5 < sigma < 4.5, sigma  # published value 3.9271
    ratio = art["results"]["errors"][0] / art["results"]["errors"][-1]
    assert ratio > 1.0e8, ratio


# --- measurement floor helpers (audit F5) -----------------------------------

def test_roundoff_floor_is_eps_times_norm():
    assert roundoff_floor(0.0) == 0.0
    assert roundoff_floor(1.0) == pytest.approx(FLOAT64_EPS)
    assert roundoff_floor(2.0) == pytest.approx(2.0 * FLOAT64_EPS)
    with pytest.raises(ValueError):
        roundoff_floor(-1.0)


def test_signal_to_noise_and_zero_floor():
    snr = signal_to_noise([1e-3, 1e-6], 2e-6)
    assert snr == pytest.approx([500.0, 0.5])
    assert signal_to_noise([1e-3], 0.0) == [float("inf")]


def test_resolved_levels_splits_at_the_floor():
    floor = 1e-15
    errors = [1e-3, 1e-8, 5e-15, 1e-16]   # levels 3-4 sit at/below 10 * floor
    out = resolved_levels([1, 2, 3, 4], errors, floor, snr_min=10.0)
    assert out["resolved_degrees"] == [1, 2]
    assert out["floor_limited_degrees"] == [3, 4]
    assert out["max_resolvable_reduction"] == pytest.approx(1e-3 / 1e-14)
    assert out["snr"][0] == pytest.approx(1e12)
    # a level exactly at 10x the floor is still usable (>= boundary)
    edge = resolved_levels([1, 2], [1e-3, 1e-14], floor, snr_min=10.0)
    assert edge["resolved_degrees"] == [1, 2]


def test_resolved_levels_validates_inputs():
    with pytest.raises(ValueError):
        resolved_levels([1, 2], [1e-3], 1e-15)
    with pytest.raises(ValueError):
        resolved_levels([1, 2], [1e-3, 1e-9], 1e-15, snr_min=0.0)
    with pytest.raises(ValueError):
        resolved_levels([1, 2], [1e-3, -1e-9], 1e-15)


# --- measurement floor artifact --------------------------------------------

def _floor_artifact():
    if not FLOOR_ARTIFACT.exists():
        pytest.skip("run python -m benchmarks.p_refinement_floor --degrees 5 6")
    return json.loads(FLOOR_ARTIFACT.read_text())


def test_floor_artifact_provenance_binds_the_primary_artifact():
    primary = _artifact()
    floor_art = _floor_artifact()
    import hashlib
    assert floor_art["artifact_provenance"]["primary_sha256"] == hashlib.sha256(
        ARTIFACT.read_bytes()).hexdigest()
    # the primary artifact is the same study the floor was measured on
    assert floor_art["case"] == primary["case"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"


def test_floor_artifact_floor_is_eps_times_the_solution_norm():
    floor_art = _floor_artifact()
    for level in floor_art["levels"]:
        assert level["roundoff_floor"] == pytest.approx(
            FLOAT64_EPS * level["solution_norm_l2"], rel=1e-9), level["degree"]


def test_floor_artifact_paths_agree_far_below_the_discretisation_error():
    """float64 vs longdouble measurement differ by far less than the error itself."""
    floor_art = _floor_artifact()
    for level in floor_art["levels"]:
        assert level["path_spread_rel"] < 1e-5, (level["degree"], level["path_spread_rel"])


def test_floor_artifact_marks_the_last_measured_degrees_resolved():
    floor_art = _floor_artifact()
    assert floor_art["floor"]["floor_limited_degrees"] == []
    for level in floor_art["levels"]:
        assert level["resolved"] is True
        assert level["snr_measured_over_floor"] > 1e4


def test_floor_artifact_projection_predicts_the_floor_limited_degree():
    """p=7/8 stay measurable, p=9 is inside the float64 measurement floor."""
    floor_art = _floor_artifact()
    projection = floor_art["projection"]
    predicted = projection["predicted"]
    assert predicted["7"]["predicted_snr"] > 1e3
    assert predicted["9"]["predicted_snr"] < floor_art["floor"]["snr_min"]
    assert projection["first_floor_limited_degree_projection"] == 9