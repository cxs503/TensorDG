"""Tests for the set-level feature gate and the cell-average extremum estimator.

Two claims are asserted, both against **analytic** truth rather than against a
number somebody once wrote down:

1. :func:`tensordg.feature_gate.cell_average_extremum` recovers the peak of a
   smooth field from its *cell averages* with fourth-order accuracy, while the
   raw cell-average maximum is second order and its location is quantised to cell
   centres.  The estimator never sees the exact solution, so this is a property
   of the estimator, not of a particular data set.
2. :func:`judge_feature_set` is a real gate: it accepts the reference itself,
   rejects a perturbed candidate, refuses to pass a set that misses a required
   functional kind, and refuses to pass when the evidence for that kind is
   withheld (missing evidence is not a pass).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.certified_reference import (
    CERTIFIED,
    WITHHELD_STAGNANT,
    certify_functional,
)
from tensordg.feature_gate import (
    DEFAULT_REQUIRED,
    GATE_FAIL,
    GATE_PASS,
    GATE_PASS_MARGINAL,
    GATE_WITHHELD,
    cell_average_extremum,
    gate_teeth_check,
    judge_feature_set,
    kind_of,
)


# ----------------------------------------------------------------- analytic data
def sin_cell_averages(n: int, *, sign: float = 1.0, a: float = 0.0,
                      b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact cell averages of ``sign * sin(2 pi t)`` -- peak known to be 1."""
    h = (b - a) / n
    edges = a + np.arange(n + 1) * h
    means = sign * (-np.cos(2 * math.pi * edges[1:]) + np.cos(2 * math.pi * edges[:-1])) \
        / (2 * math.pi * h)
    return a + (np.arange(n) + 0.5) * h, means


def quadratic_cell_averages(n: int, a0: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact cell averages of ``a0 - (t - 0.37)**2`` (peak ``a0`` at ``t = 0.37``)."""
    h = 1.0 / n
    edges = np.arange(n + 1) * h
    means = a0 - ((edges[1:] - 0.37) ** 3 - (edges[:-1] - 0.37) ** 3) / (3.0 * h)
    return (edges[:-1] + edges[1:]) / 2.0, means


def _orders(errors) -> list[float]:
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


# ------------------------------------------------------- extremum deconvolution
def test_quadratic_peak_is_exact():
    """A parabola is reproduced exactly: value and sub-cell position."""
    for n in (200, 400, 800):
        x, f = quadratic_cell_averages(n)
        est = cell_average_extremum(f, x=x, mode="max")
        assert est["status"] == "ok"
        assert est["value"] == pytest.approx(2.0, abs=1e-4)
        assert est["position"] == pytest.approx(0.37, abs=2e-3)   # sub-cell, not 0.375
        # the raw cell-average maximum is biased and quantised
        assert est["raw_value"] < est["value"] + 1e-12
        h = x[1] - x[0]
        assert abs(est["position"] - 0.37) <= 0.5 * h + 1e-12
        assert abs(est["position"] - 0.37) < abs(est["raw_position"] - 0.37) + 1e-12


def test_deconvolution_restores_fourth_order():
    """Raw max is O(h^2); the deconvolved estimate is O(h^4)."""
    ns = (100, 200, 400, 800, 1600)
    raw, deconv = [], []
    for n in ns:
        x, f = sin_cell_averages(n)
        est = cell_average_extremum(f, x=x, mode="max")
        assert est["status"] == "ok"
        raw.append(abs(est["raw_value"] - 1.0))
        deconv.append(abs(est["value"] - 1.0))
    p_raw = _orders(raw)
    p_dec = _orders(deconv)
    assert all(abs(p - 2.0) < 0.05 for p in p_raw[1:])
    assert all(p > 3.5 for p in p_dec[1:])
    assert deconv[-1] < 1e-8                      # n=1600: 1.6e-7 raw -> ~1e-10
    assert deconv[-1] * 1e3 < raw[-1]             # >= 3 orders better at the finest


def test_subcell_position_beats_cell_centre_staircase():
    """The deconvolved location is sub-cell; the raw argmax is a staircase.

    For a symmetric peak the three cell means bracket the peak symmetrically, so
    the parabola vertex is exact to round-off -- a much stronger statement than
    "second order", and one the raw cell-centre argmax can never make.
    """
    raw_err = []
    for n in (100, 200, 400, 800):
        x, f = sin_cell_averages(n)
        est = cell_average_extremum(f, x=x, mode="max")
        assert est["status"] == "ok"
        raw_err.append(abs(est["raw_position"] - 0.25))   # sin peak at t = 0.25
        assert abs(est["position"] - 0.25) < 1e-12
    assert min(raw_err) > 1e-4                 # quantised to ~h/2 at best


def test_min_mode_and_sign_symmetry():
    """``mode='min'`` deconvolves a trough with the same accuracy."""
    ns = (200, 400, 800)
    errs = []
    for n in ns:
        x, f = sin_cell_averages(n, sign=-1.0)
        est = cell_average_extremum(f, x=x, mode="min")
        assert est["status"] == "ok"
        assert est["value"] == pytest.approx(-1.0, abs=1e-4)
        errs.append(abs(est["value"] + 1.0))
    assert errs[-1] < errs[0]


def test_bias_correction_is_the_documented_term():
    """The reported correction is exactly ``h^2/24 f''`` with the fitted ``f''``."""
    x, f = sin_cell_averages(400)
    est = cell_average_extremum(f, x=x, mode="max")
    h = x[1] - x[0]
    assert est["bias_correction"] == pytest.approx(h * h / 24.0 * est["curvature"])
    assert est["curvature"] < 0.0                     # maximum -> concave
    no_fix = cell_average_extremum(f, x=x, mode="max", correct_bias=False)
    assert no_fix["estimator"] == "parabola_vertex"
    assert est["value"] > no_fix["value"]             # bias is a deficit at a peak


@pytest.mark.parametrize("kwargs,fragment", [
    ({"mode": "max"}, "boundary"),
    ({"mode": "max", "x": np.array([0.0, 0.1, 0.35, 0.4, 0.9])}, "nonuniform"),
])
def test_refusals_are_reported_not_repaired(kwargs, fragment):
    """Boundary extremum and non-uniform grid fall back to the raw estimate."""
    if "x" in kwargs:
        vals = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    else:
        vals = np.array([3.0, 2.0, 1.0])      # extremum on the first cell
    est = cell_average_extremum(vals, **kwargs)
    assert est["estimator"] == "raw_cell_average"
    assert est["status"] == fragment
    assert est["value"] == est["raw_value"]
    assert est["curvature"] is None


def test_degenerate_and_invalid_inputs():
    with pytest.raises(ValueError):
        cell_average_extremum([1.0, 2.0])                     # too few cells
    with pytest.raises(ValueError):
        cell_average_extremum([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        cell_average_extremum([1.0, 2.0, 1.0], mode="peak")
    flat = cell_average_extremum([2.0, 2.0, 2.0, 2.0], mode="max")
    assert flat["status"] == "boundary"        # plateau: argmax lands on a cell edge
    assert flat["value"] == flat["raw_value"]


# --------------------------------------------------------------- gate semantics
def _reference_set():
    """Two certified functionals (one integral, one extremum) with real bands."""
    integral = certify_functional([0.300444853847175, 0.30010642666753573,
                                   0.29992366843862206, 0.2999031],
                                  tolerance_percent=3.0)
    extremum = certify_functional([4.678201310902247, 4.679750288115062,
                                   4.680389190376859, 4.680413723239022],
                                  tolerance_percent=3.0)
    assert integral["verdict"] == CERTIFIED and extremum["verdict"] == CERTIFIED
    return {"rms": integral, "crest": extremum}


KINDS = {"rms": "integral", "crest": "extremum"}


def _reference_values(refs):
    return {n: c["certified_value"] for n, c in refs.items()}


def test_gate_passes_reference_and_fails_perturbation():
    refs = _reference_set()
    ok = judge_feature_set(_reference_values(refs), references=refs, kinds=KINDS)
    assert ok["verdict"] == GATE_PASS
    bad = dict(_reference_values(refs))
    bad["crest"] *= 1.2
    verdict = judge_feature_set(bad, references=refs, kinds=KINDS)
    assert verdict["verdict"] == GATE_FAIL
    assert verdict["kinds"]["extremum"]["failing"] == ["crest"]
    assert verdict["kinds"]["integral"]["status"] == GATE_PASS


def test_gate_withholds_when_a_required_kind_is_missing():
    """Integral-only evidence must never satisfy an integral+extremum gate."""
    refs = _reference_set()
    partial = {"rms": _reference_values(refs)["rms"]}
    out = judge_feature_set(partial, references=refs, kinds=KINDS)
    assert out["verdict"] == GATE_WITHHELD
    assert out["kinds"]["extremum"]["status"] == GATE_WITHHELD
    assert "not a pass" in out["kinds"]["extremum"]["reason"]


def test_gate_withholds_when_the_reference_is_withheld():
    """A stagnant ladder gives no usable reference -> the gate cannot pass."""
    refs = _reference_set()
    stagnant = certify_functional([4.6731615458847, 4.673356413442839,
                                   4.679291490460577, 4.6795854646991595],
                                  tolerance_percent=3.0)
    assert stagnant["verdict"] == WITHHELD_STAGNANT
    refs = dict(refs, crest=stagnant)
    vals = {"rms": refs["rms"]["certified_value"], "crest": 4.6795}
    out = judge_feature_set(vals, references=refs, kinds=KINDS)
    assert out["verdict"] == GATE_WITHHELD
    assert out["functionals"]["crest"]["decision"] == "WITHHELD"


def test_quota_of_two_and_empty_requirement_rejected():
    refs = _reference_set()
    vals = _reference_values(refs)
    two = judge_feature_set(vals, references=refs, kinds=KINDS,
                            required={"integral": 2, "extremum": 1})
    assert two["verdict"] == GATE_WITHHELD        # only one integral exists
    with pytest.raises(ValueError):
        judge_feature_set(vals, references=refs, kinds=KINDS, required={})
    with pytest.raises(ValueError):
        judge_feature_set(vals, references=refs, kinds=KINDS,
                          required={"integral": 0})


def test_unknown_kind_cannot_satisfy_a_requirement():
    refs = _reference_set()
    vals = _reference_values(refs)
    # drop the kind table: every functional becomes "other", so both required
    # kinds are unreachable and the gate must withhold rather than pass.
    out = judge_feature_set(vals, references=refs, kinds=None)
    assert out["verdict"] == GATE_WITHHELD
    assert kind_of("crest") == "other"
    assert DEFAULT_REQUIRED == {"integral": 1, "extremum": 1}


def test_marginal_credit_is_opt_in():
    """``PASS_MARGINAL`` (error inside the reference band) is not a pass by default."""
    refs = _reference_set()
    ladder = [1.0, 1.016, 1.024, 1.028]          # band ~0.16 %, well under the gate
    wide = certify_functional(ladder, tolerance_percent=3.0)
    assert wide["verdict"] == CERTIFIED
    assert wide["band_percent"] < 1.0
    refs = {"rms": refs["rms"], "crest": wide}
    val = wide["certified_value"]
    cand = dict(_reference_values(refs))
    cand["crest"] = val * (1.0 + 2.9 / 100.0)     # 2.9 % < 3 %, but 2.9 + band > 3
    strict = judge_feature_set(cand, references=refs, kinds=KINDS)
    loose = judge_feature_set(cand, references=refs, kinds=KINDS, allow_marginal=True)
    assert strict["functionals"]["crest"]["decision"] == "PASS_MARGINAL"
    assert strict["verdict"] == GATE_FAIL          # a marginal pass is not a pass
    assert loose["verdict"] in (GATE_PASS, GATE_PASS_MARGINAL)
    assert loose["kinds"]["extremum"]["status"] != GATE_FAIL


def test_teeth_check_detects_a_battery_that_cannot_fail():
    refs = _reference_set()
    teeth = gate_teeth_check(refs, kinds=KINDS)
    assert teeth["has_teeth"] is True
    assert teeth["accepts_identity"] == GATE_PASS
    # a gate with no requirement is rejected outright (vacuous by construction)
    with pytest.raises(ValueError):
        judge_feature_set(_reference_values(refs), references=refs, kinds=KINDS,
                          required={})