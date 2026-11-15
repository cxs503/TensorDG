"""CKS1998 §5.3 vortex 2-D order certification tests.

Tests that the certification module correctly classifies ladders and that the
committed artefact is reproducible from its own levels.

Run::

    PYTHONPATH=. python -m pytest tests/test_cks1998_vortex_order_certification.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.cks1998_vortex_order_certification import (
    SIGMA_BAND,
    TOLERANCE_REL,
    adjacent_order,
    assemble,
    asymptotic_ratio,
    certify,
    extrapolation_residual,
    least_squares_order,
    render_markdown,
    richardson_extrapolated,
    time_error_fraction,
)

ARTIFACT = (Path(__file__).resolve().parents[1] / "validation"
            / "cks1998_vortex_2d" / "certification.json")


def _ladder(order: float, h=(1 / 16, 1 / 32, 1 / 64), constant=0.5, scale=1.0):
    """Synthetic ``L1_rel = C h^order`` ladder (coarse -> fine)."""
    return [{"h": hi, "error": constant * hi ** order * scale} for hi in h]


# ---------------------------------------------------------------------------
# 1. Estimators (synthetic)
# ---------------------------------------------------------------------------
def test_adjacent_order_recovers_an_exact_power_law():
    assert adjacent_order(1e-2, 1e-2 / 4.0, 2.0) == pytest.approx(2.0, abs=1e-12)
    assert adjacent_order(8.0, 1.0, 2.0) == pytest.approx(3.0, abs=1e-12)


def test_adjacent_order_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        adjacent_order(1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        adjacent_order(0.0, 1.0, 2.0)


def test_least_squares_order_separates_clean_from_degraded_ladders():
    clean = least_squares_order(
        [1 / 8, 1 / 16, 1 / 32],
        [0.5 / 8 ** 2, 0.5 / 16 ** 2, 0.5 / 32 ** 2])
    assert clean == pytest.approx(2.0, abs=1e-12)
    degraded = least_squares_order(
        [1 / 8, 1 / 16, 1 / 32],
        [0.5 / 8 ** 1.5, 0.5 / 16 ** 1.5, 0.5 / 32 ** 1.5])
    assert degraded == pytest.approx(1.5, abs=1e-12)


def test_richardson_limit_and_residual_match_the_closed_form():
    h_c, h_f = 1 / 16, 1 / 32
    e_c, e_f = 0.5 * h_c ** 2, 0.5 * h_f ** 2
    assert richardson_extrapolated(e_c, e_f, 2.0, 2.0) == pytest.approx(0.0, abs=1e-18)
    assert extrapolation_residual(e_c, e_f, 2.0, 2.0) == pytest.approx(0.0, abs=1e-15)
    wrong = extrapolation_residual(e_c, e_f, 2.0, 1.6)
    assert wrong > 0.4


def test_asymptotic_ratio_is_order_over_order():
    assert asymptotic_ratio(2.02, 1.99) == pytest.approx(2.02 / 1.99, rel=1e-12)
    with pytest.raises(ValueError):
        asymptotic_ratio(2.0, 0.0)


# ---------------------------------------------------------------------------
# 2. The verdict must be able to fail (non-vacuity)
# ---------------------------------------------------------------------------
def test_clean_second_order_ladder_is_certified():
    cert = certify(_ladder(2.0), design_order=2)
    assert cert["verdict"] == "CERTIFIED"
    assert cert["order_fine"] == pytest.approx(2.0, abs=1e-10)
    assert cert["asymptotic_ok"] is True


def test_degraded_order_is_reported_as_fail():
    """1.6 instead of 2 is a 20 % deviation: no honest gate may call that a pass."""
    cert = certify(_ladder(1.6), design_order=2)
    assert cert["verdict"] == "FAIL"
    assert cert["delta_rel"] > TOLERANCE_REL


def test_an_out_of_range_coarse_level_cannot_be_certified():
    """The finest pair looks perfect; the ladder as a whole is *not* asymptotic."""
    ladder = _ladder(2.0)
    ladder[0]["error"] *= 1.6
    cert = certify(ladder, design_order=2)
    assert cert["order_fine"] == pytest.approx(2.0, abs=1e-10)
    assert cert["verdict"] != "CERTIFIED"
    assert cert["asymptotic_ok"] is False
    lo, hi = SIGMA_BAND
    assert not (lo <= cert["asymptotic_ratio"] <= hi)


def test_estimator_uncertainty_wider_than_the_tolerance_is_not_a_certificate():
    """A perfect finest pair cannot be certified if the estimators disagree."""
    ladder = _ladder(2.0)
    ladder[1]["error"] *= 1.02
    cert = certify(ladder, design_order=2)
    assert cert["delta_rel"] <= TOLERANCE_REL
    assert cert["verdict"] in ("UNVERIFIED_BAND", "FAIL")
    assert cert["delta_band_rel"] > TOLERANCE_REL
    assert cert["uncertainty"] > 0.0


def test_two_level_ladder_is_refused():
    with pytest.raises(ValueError):
        certify(_ladder(2.0)[:2], design_order=2)


# ---------------------------------------------------------------------------
# 3. Time-error separation
# ---------------------------------------------------------------------------
def test_time_error_fraction_is_the_relative_shift():
    assert time_error_fraction([1.0, 1.02]) == pytest.approx(0.02, rel=1e-12)
    assert time_error_fraction([1.0, 1.0]) == 0.0
    with pytest.raises(ValueError):
        time_error_fraction([1.0])


def test_markdown_renders_the_verdict_column():
    payload = assemble([{"kind": "ladder", "degree": 1, "ncells": n,
                         "h": 2.0 / n, "error": 0.5 * (2.0 / n) ** 2,
                         "norms": {"density": {"L1_rel": 0.5 * (2.0 / n) ** 2}}}
                        for n in (8, 16, 32)])
    table = render_markdown(payload)
    assert "CERTIFIED" in table
    rows = [line.split("|")[1:-1] for line in table.splitlines() if "|" in line]
    assert len({len(r) for r in rows}) == 1, "table rows must all have the same width"


# ---------------------------------------------------------------------------
# 4. The committed artefact (re-derived, not trusted)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="certification artefact not committed yet")
def test_committed_certification_is_reproducible_from_its_own_levels():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["field"] == "density" and payload["norm"] == "L1_rel"
    assert payload["anchor"] == "REPRO-DG-CKS1998"
    assert payload["case"] == "vortex_2d_certification"
    for name, bundle in payload["degrees"].items():
        degree = int(name[1:])
        recomputed = certify(bundle["certification"]["levels"], degree + 1,
                             payload["tolerance_rel"])
        stored = bundle["certification"]
        assert stored["verdict"] == recomputed["verdict"], name
        assert stored["order_fine"] == pytest.approx(recomputed["order_fine"], rel=1e-9)
        assert stored["uncertainty"] == pytest.approx(recomputed["uncertainty"], rel=1e-9)
        assert stored["delta_band_rel"] == pytest.approx(recomputed["delta_band_rel"], rel=1e-9)
        # the ladder must actually converge
        errors = [level["error"] for level in stored["levels"]]
        assert errors == sorted(errors, reverse=True)
