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
    order_drift,
    render_time_markdown,
    richardson_extrapolated,
    temporal_share_of_deviation,
    time_error_fraction,
    time_separation,
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


def test_order_drift_is_relative_and_rejects_zero():
    assert order_drift(2.0, 2.1) == pytest.approx(0.05, rel=1e-12)
    assert order_drift(2.07, 2.07) == 0.0
    with pytest.raises(ValueError):
        order_drift(0.0, 1.0)


def test_temporal_share_of_deviation_saturates():
    # order 2.07 -> 2.00 against design 2: the whole deviation is temporal
    assert temporal_share_of_deviation(2.07, 2.0, 2) == pytest.approx(1.0)
    # order does not move at all under dt refinement
    assert temporal_share_of_deviation(2.07, 2.07, 2) == 0.0
    # half of the deviation disappears
    assert temporal_share_of_deviation(2.10, 2.05, 2) == pytest.approx(0.5)


def _time_runs(order_base, order_reduced, eta=0.01, design_constant=0.5):
    """Four synthetic runs: two meshes x two CFLs, with prescribed orders.

    ``order_base`` is the finest-pair order at ``CFL``, ``order_reduced`` the one
    at ``CFL/beta``; ``eta`` is the fraction of the fine-mesh error that the
    time-step refinement removes (the temporal share).
    """
    h = [2.0 / 16, 2.0 / 32]                     # h_coarse / h_fine = 2
    ratio = h[0] / h[1]
    e_fine_base = design_constant * h[1] ** 2
    e_coarse_base = e_fine_base * ratio ** order_base
    e_fine_red = e_fine_base * (1.0 - eta)
    e_coarse_red = e_fine_red * ratio ** order_reduced
    runs = []
    for cfl, e_c, e_f in ((0.2, e_coarse_base, e_fine_base),
                          (0.05, e_coarse_red, e_fine_red)):
        runs.append({"ncells": 16, "cfl": cfl, "h": h[0], "error": e_c})
        runs.append({"ncells": 32, "cfl": cfl, "h": h[1], "error": e_f})
    return runs


def test_time_separation_attributes_a_vanishing_deviation_to_the_time_stepper():
    sep = time_separation(_time_runs(2.07, 2.00), design_order=2)
    assert sep["status"] == "OK"
    assert sep["spatial_isolated"] is False, "a 3.5 % order shift is not a converged dt"
    assert sep["deviation_direction"] == "SHRINKS"
    assert sep["attribution"] == "TIME_DOMINATED"
    assert sep["order_base"] == pytest.approx(2.07, abs=1e-9)
    assert sep["order_reduced"] == pytest.approx(2.00, abs=1e-9)
    assert sep["delta_rel_base"] > 0.03 > sep["delta_rel_reduced"]
    assert sep["within_tolerance_at_reduced_cfl"] is True
    assert sep["order_bracket"] == pytest.approx([2.00, 2.07])


def test_time_separation_keeps_a_persistent_deviation_as_spatial():
    # order moves by 0.05 % and the 10 % deviation stays: time is not the cause
    sep = time_separation(_time_runs(2.20, 2.199), design_order=2)
    assert sep["spatial_isolated"] is True
    assert sep["attribution"] == "SPATIAL_ISOLATED"
    assert sep["within_tolerance_at_reduced_cfl"] is False


def test_time_separation_calls_a_partial_shift_time_sensitive():
    sep = time_separation(_time_runs(2.10, 2.06), design_order=2)
    assert sep["deviation_direction"] == "SHRINKS"
    assert sep["deviation_share"] == pytest.approx(0.4, abs=1e-9)
    assert sep["attribution"] == "TIME_SENSITIVE"


def test_time_separation_refuses_to_attribute_when_the_deviation_grows():
    sep = time_separation(_time_runs(2.07, 2.60), design_order=2)
    assert sep["deviation_direction"] == "GROWS"
    assert sep["spatial_isolated"] is False
    assert sep["attribution"] == "UNRESOLVED"
    assert sep["within_tolerance_at_reduced_cfl"] is False


def test_time_separation_marks_a_large_temporal_share_as_unresolved():
    # eta above the 5 % ceiling: the reduced-CFL run is not a spatial order yet
    sep = time_separation(_time_runs(2.07, 2.00, eta=0.2), design_order=2)
    assert sep["eta"] > sep["eta_max"]
    assert sep["temporal_share_ok"] is False
    assert sep["spatial_isolated"] is False
    assert sep["attribution"] != "SPATIAL_ISOLATED"


def test_time_separation_returns_status_instead_of_a_number_on_bad_input():
    runs = _time_runs(2.07, 2.0)
    only_one_cfl = [r for r in runs if r["cfl"] == 0.2]
    assert time_separation(only_one_cfl, 2)["status"] == "INSUFFICIENT_CFLS"
    mismatched = [dict(r, cfl=0.1 if r["cfl"] == 0.05 else r["cfl"]) for r in runs]
    assert time_separation(mismatched, 2, beta=4)["status"] == "CFL_RATIO_MISMATCH"
    one_mesh = [r for r in runs if r["ncells"] == 32]
    assert time_separation(one_mesh, 2)["status"] == "INSUFFICIENT_MESHES"


def test_assemble_keeps_the_ladder_and_the_time_run_separate():
    records = []
    for n in (8, 16, 32):
        h = 2.0 / n
        records.append({"kind": "ladder", "degree": 1, "ncells": n, "h": h,
                        "norms": {"density": {"L1_rel": 0.5 * h ** 2}}})
    for run in _time_runs(2.07, 2.0):
        records.append({"kind": "time", "degree": 1, "ncells": run["ncells"],
                        "cfl": run["cfl"], "h": run["h"],
                        "norms": {"density": {"L1_rel": run["error"]}}})
    payload = assemble(records)
    cert = payload["degrees"]["p1"]["certification"]
    assert cert["verdict"] == "CERTIFIED", "time runs must not enter the ladder"
    assert len(cert["levels"]) == 3
    sep = payload["degrees"]["p1"]["time_separation"]
    assert sep["attribution"] == "TIME_DOMINATED"
    assert "TIME_DOMINATED" in render_time_markdown(payload)


def test_a_two_rung_ladder_is_refused_not_point_estimated():
    payload = assemble([{"kind": "ladder", "degree": 1, "ncells": n,
                         "h": 2.0 / n,
                         "norms": {"density": {"L1_rel": 0.5 * (2.0 / n) ** 2}}}
                        for n in (8, 16)])
    assert payload["degrees"]["p1"]["certification"]["verdict"] == "INSUFFICIENT_LEVELS"
    assert "INSUFFICIENT_LEVELS" in render_markdown(payload)


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


@pytest.mark.skipif(not ARTIFACT.exists(), reason="certification artefact not committed yet")
def test_committed_artefact_carries_a_re_derived_time_attribution():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["time_refinement"] > 1.0
    for name, bundle in payload["degrees"].items():
        sep = bundle["time_separation"]
        assert sep["status"] == "OK", name
        # re-derive the whole attribution from the recorded raw runs
        raw = [{"ncells": r["ncells"], "cfl": r["cfl"], "h": r["h"],
                "error": r["norms"]["density"]["L1_rel"]}
               for r in bundle["runs"] if r.get("kind") == "time"]
        redo = time_separation(raw, int(name[1:]) + 1, beta=payload["time_refinement"])
        for key in ("eta", "order_drift", "order_base", "order_reduced",
                    "delta_rel_base", "delta_rel_reduced", "deviation_share"):
            assert sep[key] == pytest.approx(redo[key], rel=1e-9), (name, key)
        assert sep["attribution"] == redo["attribution"], name
        # the time runs must be the finest ladder pair, at CFL and CFL/beta
        ladder_meshes = sorted(lv["ncells"] for lv in
                               bundle["certification"]["levels"])[-2:]
        assert sep["meshes"] == ladder_meshes, name
        assert sep["cfl_reduced"] == pytest.approx(sep["cfl_base"] / sep["beta"])
