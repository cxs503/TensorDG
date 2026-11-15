"""Does the order gate *certify*, or does it merely happen to pass?

The CKS1998 §5.1 smooth 2-D Euler case is the anchor's headline accuracy claim.
``benchmarks/cks1998_smooth_2d.py`` measures an order per degree; the platform
gate ``sm2d_order_p{1,2,3}`` accepts it if it is within 3 % of the design order
``p + 1``.  ``benchmarks/cks1998_order_certification.py`` adds the estimator
uncertainty, the asymptotic-range test and the space/time separation, and turns
them into a three-valued verdict.

Two kinds of test live here:

1. **Estimator tests on synthetic data** (no solver, milliseconds).  These are
   the ones that decide whether the verdict can fail at all: a ladder whose
   true order is 1.6 must be reported ``FAIL`` against design order 2; a ladder
   that is off its asymptotic range must not be reported ``CERTIFIED``; a
   Richardson/GCI pair must match the closed form for ``e = C h^p``.
2. **One artefact test** that does not trust the committed bundle: the observed
   orders, the estimator spread and the verdict are all re-derived from the
   stored ``(h, error)`` levels and must reproduce.

Run::

    PYTHONPATH=. python -m pytest tests/test_cks1998_order_certification.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.cks1998_order_certification import (
    ORDER_DRIFT_MAX,
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
            / "cks1998_order_certification" / "certification.json")


def _ladder(order: float, h=(1 / 16, 1 / 32, 1 / 64), constant=0.5, scale=1.0):
    """Synthetic ``L1 = C h^order`` ladder (coarse -> fine)."""
    return [{"h": hi, "error": constant * hi ** order * scale} for hi in h]


# ---------------------------------------------------------------------------
# 1. Estimators (synthetic)
# ---------------------------------------------------------------------------
def test_adjacent_order_recovers_an_exact_power_law():
    assert adjacent_order(1e-2, 1e-2 / 4.0, 2.0) == pytest.approx(2.0, abs=1e-12)
    # cubic refinement, third order
    assert adjacent_order(8.0, 1.0, 2.0) == pytest.approx(3.0, abs=1e-12)


def test_adjacent_order_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        adjacent_order(1.0, 1.0, 0.5)          # mesh ratio <= 1
    with pytest.raises(ValueError):
        adjacent_order(0.0, 1.0, 2.0)          # non-positive error


def test_least_squares_order_separates_clean_from_degraded_ladders():
    clean = least_squares_order([1 / 8, 1 / 16, 1 / 32], [0.5 / 8 ** 2, 0.5 / 16 ** 2, 0.5 / 32 ** 2])
    assert clean == pytest.approx(2.0, abs=1e-12)
    degraded = least_squares_order([1 / 8, 1 / 16, 1 / 32], [0.5 / 8 ** 1.5, 0.5 / 16 ** 1.5, 0.5 / 32 ** 1.5])
    assert degraded == pytest.approx(1.5, abs=1e-12)


def test_richardson_limit_and_residual_match_the_closed_form():
    h_c, h_f = 1 / 16, 1 / 32
    e_c, e_f = 0.5 * h_c ** 2, 0.5 * h_f ** 2
    # e = C h^2 -> extrapolating with the true order leaves exactly zero
    assert richardson_extrapolated(e_c, e_f, 2.0, 2.0) == pytest.approx(0.0, abs=1e-18)
    assert extrapolation_residual(e_c, e_f, 2.0, 2.0) == pytest.approx(0.0, abs=1e-15)
    # a wrong order leaves a residual of the size of the miss, relative to e_fine
    wrong = extrapolation_residual(e_c, e_f, 2.0, 1.6)
    assert wrong > 0.4
    with pytest.raises(ValueError):
        extrapolation_residual(e_c, 0.0, 2.0, 2.0)


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
    ladder[0]["error"] *= 1.6                      # coarsest level off-band
    cert = certify(ladder, design_order=2)
    assert cert["order_fine"] == pytest.approx(2.0, abs=1e-10)
    assert cert["verdict"] != "CERTIFIED"
    assert cert["asymptotic_ok"] is False
    lo, hi = SIGMA_BAND
    assert not (lo <= cert["asymptotic_ratio"] <= hi)


def test_estimator_uncertainty_wider_than_the_tolerance_is_not_a_certificate():
    """A perfect finest pair cannot be certified if the estimators disagree."""
    ladder = _ladder(2.0)
    ladder[1]["error"] *= 1.02                     # 2 % wiggle on the middle level
    cert = certify(ladder, design_order=2)
    assert cert["delta_rel"] <= TOLERANCE_REL      # point estimate still inside 3 %
    assert cert["verdict"] in ("UNVERIFIED_BAND", "FAIL")
    assert cert["delta_band_rel"] > TOLERANCE_REL  # the band is not
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


def test_time_separation_uses_the_same_mesh_pair_at_both_cfl():
    """The assembled bundle must compare orders on a *matched* pair, not two meshes."""
    h = {8: 2.0 / 8, 16: 2.0 / 16}
    records = []
    for ncells in (8, 16):
        for cfl in (0.2, 0.05):
            # spatial error dominates: e ~ h^2 with a small dt^3 tail
            records.append({"kind": "time", "degree": 1, "ncells": ncells, "cfl": cfl,
                            "h": h[ncells], "error": 0.5 * h[ncells] ** 2 + 0.4 * 1e-4 * cfl ** 3})
    payload = assemble(records)
    sep = payload["degrees"]["p1"]["time_separation"]
    assert set(sep["pairs"]) == {"cfl0.2", "cfl0.05"}
    assert sep["pairs"]["cfl0.2"]["order"] == pytest.approx(2.0, abs=2e-3)
    assert sep["pairs"]["cfl0.05"]["order"] == pytest.approx(2.0, abs=2e-3)
    assert sep["eta"] <= 0.05 and sep["order_drift"] <= ORDER_DRIFT_MAX
    assert sep["spatial_isolated"] is True


def test_markdown_renders_the_verdict_column():
    payload = assemble([{"kind": "ladder", "degree": 1, "ncells": n,
                         "h": 2.0 / n, "error": 0.5 * (2.0 / n) ** 2}
                        for n in (8, 16, 32)])
    table = render_markdown(payload)
    assert "CERTIFIED" in table
    rows = [line.split("|")[1:-1] for line in table.splitlines()]
    assert len({len(r) for r in rows}) == 1, "table rows must all have the same width"
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# 4. The committed artefact (re-derived, not trusted)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="certification artefact not committed yet")
def test_committed_certification_is_reproducible_from_its_own_levels():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["field"] == "density" and payload["norm"] == "L1"
    for name, bundle in payload["degrees"].items():
        degree = int(name[1:])
        recomputed = certify(bundle["certification"]["levels"], degree + 1,
                             payload["tolerance_rel"])
        stored = bundle["certification"]
        assert stored["verdict"] == recomputed["verdict"], name
        assert stored["order_fine"] == pytest.approx(recomputed["order_fine"], rel=1e-9)
        assert stored["uncertainty"] == pytest.approx(recomputed["uncertainty"], rel=1e-9)
        assert stored["delta_band_rel"] == pytest.approx(recomputed["delta_band_rel"], rel=1e-9)
        # a certified verdict must have both halves of the claim in hand
        if stored["verdict"] == "CERTIFIED":
            assert stored["delta_band_rel"] <= payload["tolerance_rel"]
            assert stored["asymptotic_ok"] is True
        # the ladder must actually converge
        errors = [level["error"] for level in stored["levels"]]
        assert errors == sorted(errors, reverse=True)