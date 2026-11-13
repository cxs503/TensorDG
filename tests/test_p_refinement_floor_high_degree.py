"""Gates for the high-degree extension of the measurement-floor study (m17687).

Case: CASE-DG-PREFINEMENT-EXPONENTIAL, audit m16499 finding F5.

The floor study used to take the CFL from the published ladder and trust it, so
the p=7/8 levels were only ever *projected* (``measurement_floor.json`` carried
SNR estimates, no measurement).  Upstream m17067 measured p=7/8 with an exact
(semi-discrete matrix-exponential) time integration and showed the binding cost
is the RK4 temporal error, not the roundoff floor.

This module gates the two protocol additions that close the remaining hole:

1. ``_temporal_converged_chain`` -- every degree is re-solved with
   ``cfl -> cfl/2`` until the error stops moving (threshold
   ``max(1%, 10 x roundoff relative uncertainty)``); a level that never settles
   inside the step budget must be reported ``temporal_converged = False`` with a
   ``stop_reason``, never silently promoted to a spatial measurement.
2. ``_extrapolation_check`` -- ``error(p) = e(6) exp(-sigma (p-6))`` from the
   *published* ladder is tested out of sample against every degree measured here
   above 6, i.e. Babuska & Suri (1994) analytic-solution theory is predicted,
   not re-fitted.

No error value is hard-coded below: the thresholds are the predeclared ones
(``SNR_MIN`` from the benchmark, factor-2 extrapolation band).

Run::

    pytest -q tests/test_p_refinement_floor_high_degree.py
"""
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "benchmarks" / "results" / "p-refinement-exponential"
FLOOR_ARTIFACT = RESULTS / "measurement_floor.json"
PRIMARY_ARTIFACT = RESULTS / "p_refinement_results.json"

SNR_MIN = 10.0            # predeclared in benchmarks/p_refinement_floor.py
EXTRAPOLATION_BAND = 2.0  # measured/predicted tolerance
CERTIFIED_MAX_DEGREE = 6  # degrees whose published CFL is already converged


def _artifact() -> dict:
    if not FLOOR_ARTIFACT.exists():
        pytest.skip("run python -m benchmarks.p_refinement_floor --degrees ... 7")
    art = json.loads(FLOOR_ARTIFACT.read_text())
    degrees = [int(lv["degree"]) for lv in art.get("levels", [])]
    if max(degrees, default=0) <= CERTIFIED_MAX_DEGREE:
        pytest.skip("measurement_floor.json has no degree above p=6; re-run with p=7")
    if "temporal" not in art["levels"][0]:
        pytest.skip("measurement_floor.json predates the temporal-certification protocol")
    return art


def _level(art: dict, degree: int) -> dict:
    return next(lv for lv in art["levels"] if int(lv["degree"]) == degree)


# --- artifact identity / provenance ----------------------------------------


def test_artifact_identity_and_schema():
    art = _artifact()
    assert art["schema"] == "tensordg-p-refinement-measurement-floor/1"
    assert art["case"] == "CASE-DG-PREFINEMENT-EXPONENTIAL"
    degrees = [int(lv["degree"]) for lv in art["levels"]]
    assert degrees == sorted(degrees)
    assert degrees[-1] > CERTIFIED_MAX_DEGREE, degrees


def test_artifact_provenance_binds_the_published_ladder():
    art = _artifact()
    assert art["artifact_provenance"]["primary_sha256"] == hashlib.sha256(
        PRIMARY_ARTIFACT.read_bytes()).hexdigest()


def test_levels_are_solved_on_the_published_mesh():
    art = _artifact()
    ncells = art["config"]["ncells"]
    primary = json.loads(PRIMARY_ARTIFACT.read_text())
    assert ncells == primary["results"]["levels"][0]["ncells"] == 16
    for lv in art["levels"]:
        assert lv["ncells"] == ncells
        assert lv["dofs"] == ncells * (lv["degree"] + 1)


# --- temporal certification is reported, not assumed -----------------------


def test_temporal_protocol_is_declared_and_self_consistent():
    art = _artifact()
    proto = art["temporal_protocol"]
    converged = sorted(lv["degree"] for lv in art["levels"]
                       if lv["temporal"]["temporal_converged"])
    uncertified = sorted(lv["degree"] for lv in art["levels"]
                         if not lv["temporal"]["temporal_converged"])
    assert sorted(proto["converged_degrees"]) == converged
    assert sorted(proto["uncertified_degrees"]) == uncertified
    assert proto["all_converged"] == (not uncertified)
    assert proto["step_budget_per_degree"] > 0


def test_every_level_reports_a_resolved_cfl_and_a_stop_reason():
    art = _artifact()
    for lv in art["levels"]:
        t = lv["temporal"]
        chain = t["refinements"]
        assert len(chain) >= 2, (lv["degree"], "no refinement attempted")
        assert chain[-1]["cfl"] < chain[0]["cfl"], lv["degree"]
        assert lv["cfl"] == pytest.approx(t["final_cfl"], rel=0)
        assert t["stop_reason"] in (
            "converged", "refinement_budget_exhausted", "step_budget",
            "no_refinement_allowed"), (lv["degree"], t["stop_reason"])
        if t["temporal_converged"]:
            assert t["stop_reason"] == "converged"
            assert t["temporal_change_rel"] <= t["threshold"]
        else:
            # an uncertified level must say *why* it stopped and keep the budget
            assert t["total_steps"] > 0
            assert t["temporal_change_rel"] > 0


def test_degrees_inside_the_published_ladder_are_temporally_certified():
    art = _artifact()
    for lv in art["levels"]:
        if lv["degree"] <= CERTIFIED_MAX_DEGREE:
            assert lv["temporal"]["temporal_converged"], (lv["degree"], lv["temporal"])


def test_errors_decrease_over_the_measured_ladder():
    art = _artifact()
    errors = [lv["l2_extended_longdouble"] for lv in art["levels"]]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] > 0.0


def test_levels_above_the_floor_are_not_noise():
    art = _artifact()
    floor = art["floor"]["value"]
    for lv in art["levels"]:
        snr = lv["l2_production_float64"] / floor
        assert snr == pytest.approx(lv["snr_measured_over_floor"], rel=1e-12)
        assert lv["resolved"] is (snr >= SNR_MIN)
        if lv["resolved"]:
            assert lv["path_spread_rel"] < 0.05, (lv["degree"], lv["path_spread_rel"])


# --- the analytic-solution law, tested out of sample ----------------------


def test_extrapolation_covers_every_degree_above_six():
    art = _artifact()
    check = art["extrapolation_check"]
    assert check is not None, "artifact has no out-of-sample extrapolation check"
    assert check["reference_degree"] == CERTIFIED_MAX_DEGREE
    high = {str(lv["degree"]) for lv in art["levels"]
            if lv["degree"] > CERTIFIED_MAX_DEGREE}
    assert set(check["rows"]) == high
    for degree, row in check["rows"].items():
        assert row["measured"] == pytest.approx(
            _level(art, int(degree))["l2_extended_longdouble"], rel=1e-12)


def test_measured_high_degrees_match_the_exponential_prediction():
    art = _artifact()
    check = art["extrapolation_check"]
    assert check["within_factor_2"], check["rows"]
    for degree, row in check["rows"].items():
        ratio = row["measured_over_predicted"]
        assert 1.0 / EXTRAPOLATION_BAND <= ratio <= EXTRAPOLATION_BAND, (degree, row)


def test_local_decay_rates_of_certified_degrees_stay_in_the_analytic_band():
    art = _artifact()
    sigma = art["extrapolation_check"]["sigma"]
    assert sigma > 1.0
    levels = art["levels"]
    for prev, cur in zip(levels, levels[1:]):
        if not (prev["temporal"]["temporal_converged"]
                and cur["temporal"]["temporal_converged"]):
            continue  # an uncertified level carries temporal error; skip the band
        local = (math.log(prev["l2_extended_longdouble"] / cur["l2_extended_longdouble"])
                 / (cur["degree"] - prev["degree"]))
        assert 0.5 * sigma <= local <= 2.0 * sigma, (cur["degree"], local, sigma)


# --- live guards on the new code paths (fast, no p>=7 solve) -------------


def test_temporal_chain_terminates_on_a_known_convergent_level():
    from benchmarks.p_refinement_floor import _temporal_converged_chain

    chain = _temporal_converged_chain(degree=2, cfl=0.4, max_refinements=3)
    assert chain["temporal_converged"], chain
    assert chain["stop_reason"] == "converged"
    assert len(chain["refinements"]) >= 2
    errors = [r["l2_extended_longdouble"] for r in chain["refinements"]]
    assert errors[-1] <= errors[0] * 1.05
    assert chain["temporal_change_rel"] <= chain["threshold"]


def test_temporal_chain_reports_budget_exhaustion_honestly():
    from benchmarks.p_refinement_floor import _temporal_converged_chain

    chain = _temporal_converged_chain(degree=4, cfl=2.0, max_refinements=0)
    assert len(chain["refinements"]) == 1
    assert chain["temporal_converged"] is False
    assert chain["stop_reason"] != "converged"
    assert chain["final_cfl"] == chain["refinements"][-1]["cfl"]
