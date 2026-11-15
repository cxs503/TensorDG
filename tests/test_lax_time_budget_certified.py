"""Contract tests for the CASE-LAX-DG temporal budget at the certified legs.

The artifact ``docs/benchmarks/results/lax_time_budget_certified/metrics.json``
is the only thing asserted here -- every claim is recomputed from its raw
series with the case pool's own diagnostics, never read off a stored verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.lax_case_pool import (
    TOL_TIME_ERROR_SHARE, ledger_acceptance_tolerance, temporal_bound,
    temporal_order, temporal_share)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_time_budget_certified" / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-time-budget-certified-legs.md"
SUMMARY = METRICS.parent / "summary.md"
RESOLUTION = ROOT / "docs" / "benchmarks" / "results" / "lax_resolution_requirement" / "metrics.json"

FIELDS = ("density", "velocity", "pressure")
LEG_NAMES = ("p3.N400", "p3.N800")


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip(f"artifact not committed: {METRICS.relative_to(ROOT)}")
    return json.loads(METRICS.read_text())


def _leg(metrics, name):
    for row in metrics["legs"]:
        if row["leg"] == name:
            return row
    raise AssertionError(f"leg {name} absent from the artifact")


def _temporal(series):
    order = temporal_order(series)
    return order, temporal_share(series, order), temporal_bound(series)


def test_artifact_is_anchored_and_carries_the_rule(metrics):
    assert metrics["case"] == "CASE-LAX-DG"
    assert metrics["anchor"] == "CASE-LAX-DG"
    for token in ("T1", "T2", "T3", "T4", "T5"):
        assert token in metrics["rule"]
    assert metrics["config"]["time_error_share"] == TOL_TIME_ERROR_SHARE
    # the pre-registered bar is the registered ledger bar, not a local copy
    assert metrics["config"]["tolerance"] == pytest.approx(ledger_acceptance_tolerance())


def test_both_certified_legs_and_three_exact_halvings(metrics):
    assert tuple(row["leg"] for row in metrics["legs"]) == LEG_NAMES
    assert tuple(metrics["config"]["divisors"]) == (1, 2, 4)
    for row in metrics["legs"]:
        assert len(row["levels"]) == 3
        cells = {level["cells"] for level in row["levels"]}
        degree = {level["degree"] for level in row["levels"]}
        # only dt moved inside a leg: the mesh is byte-identical across levels
        assert cells == {row["cells"]} and degree == {row["degree"]}
        cfls = [level["cfl"] for level in row["levels"]]
        assert cfls == pytest.approx([c / d for c, d in
                                      zip([metrics["config"]["cfl"]] * 3,
                                          metrics["config"]["divisors"])])


def test_every_field_is_measured_positive_and_finite(metrics):
    """No field exemption: all three primitive fields in every level."""
    for row in metrics["legs"]:
        for level in row["levels"]:
            for field in FIELDS:
                value = level["l2_errors"][field]
                assert math.isfinite(value) and value > 0.0, (row["leg"], field)
            assert level["density_min"] > 0.0 and level["pressure_min"] > 0.0


def test_temporal_budget_is_recomputed_below_the_registered_gate(metrics):
    """T1: the stored shares must follow the pool's semantics from the raw series."""
    for row in metrics["legs"]:
        for field in FIELDS:
            series = [level["l2_errors"][field] for level in row["levels"]]
            order, share, bound = _temporal(series)
            stored = row["fields"][field]
            assert stored["temporal_order"] == (None if order is None
                                                else pytest.approx(order))
            if share is None:
                assert stored["temporal_share"] is None
            else:
                assert stored["temporal_share"] == pytest.approx(share)
            assert stored["temporal_bound"] == pytest.approx(bound)
            reading = share if share is not None else bound
            assert reading is not None and reading < TOL_TIME_ERROR_SHARE, (
                f"{row['leg']} {field}: temporal reading {reading:.3e} exceeds "
                f"the registered {TOL_TIME_ERROR_SHARE:.0%} gate")
    assert metrics["checks"]["temporal_budget"]["status"] == "PASS"


def test_resolution_pass_survives_dt_refinement(metrics):
    """T2: p3.N800 stays below the ledger bar at *every* CFL level."""
    tolerance = ledger_acceptance_tolerance()
    row = _leg(metrics, "p3.N800")
    for level in row["levels"]:
        worst = max(level["l2_errors"].values())
        assert worst < tolerance, (level["cfl"], worst)
    entry = metrics["checks"]["ledger"]["per_leg"]["p3.N800"]
    assert entry["uniform_status"] == "PASS"
    assert metrics["checks"]["ledger"]["T2_resolution_pass_is_dt_robust"] == "PASS"


def test_pinned_fail_is_not_a_time_step_artifact(metrics):
    """T3: shrinking dt alone must not rescue the pinned A0 leg."""
    tolerance = ledger_acceptance_tolerance()
    row = _leg(metrics, "p3.N400")
    for level in row["levels"]:
        worst = max(level["l2_errors"].values())
        assert worst >= tolerance, (level["cfl"], worst)
    assert metrics["checks"]["ledger"]["per_leg"]["p3.N400"]["uniform_status"] == "FAIL"
    assert metrics["checks"]["ledger"]["T3_pinned_fail_is_not_dt_artifact"] == "PASS"


def test_baseline_reproduces_the_committed_resolution_artifact(metrics):
    """T4: the baseline CFL readings are the committed resolution numbers."""
    assert RESOLUTION.exists()
    committed = {f"p{leg['degree']}.N{leg['cells']}": leg
                 for leg in json.loads(RESOLUTION.read_text())["legs"]}
    for row in metrics["legs"]:
        baseline = row["levels"][0]
        for field in FIELDS:
            reference = committed[row["leg"]]["l2_errors"][field]
            assert baseline["l2_errors"][field] == pytest.approx(reference, rel=1e-9), (
                f"{row['leg']} {field}: baseline rerun {baseline['l2_errors'][field]!r} "
                f"vs committed {reference!r}")
    assert metrics["checks"]["reuse"]["status"] == "PASS"


def test_dt_refinement_really_refines_the_step(metrics):
    """T5: steps and dt_mean must halve with the CFL divisor."""
    for row in metrics["legs"]:
        steps = [level["steps"] for level in row["levels"]]
        dt_mean = [level["dt_mean"] for level in row["levels"]]
        assert steps == sorted(steps)
        for i in range(len(steps) - 1):
            assert 1.9 <= steps[i + 1] / steps[i] <= 2.1
            assert 1.9 <= dt_mean[i] / dt_mean[i + 1] <= 2.1
        # the mean step is the representative one (never the truncated last step)
        for level, count in zip(row["levels"], steps):
            assert level["dt_mean"] == pytest.approx(
                metrics["config"]["final_time"] / count, rel=1e-12)
    assert metrics["checks"]["dt_refinement"]["status"] == "PASS"


def test_doc_and_summary_are_rendered_from_the_artifact(metrics):
    from benchmarks.lax_time_budget_certified import render_doc, render_summary

    assert DOC.exists() and SUMMARY.exists()
    assert render_doc(metrics) == DOC.read_text(), "doc drift vs metrics.json"
    assert render_summary(metrics) == SUMMARY.read_text(), "summary drift vs metrics.json"


def test_overall_verdict_is_the_conjunction_of_the_gates(metrics):
    checks = metrics["checks"]
    gates = [checks["temporal_budget"]["status"],
             checks["ledger"]["T2_resolution_pass_is_dt_robust"],
             checks["ledger"]["T3_pinned_fail_is_not_dt_artifact"],
             checks["reuse"]["status"],
             checks["dt_refinement"]["status"]]
    expected = "PASS" if all(gate == "PASS" for gate in gates) else "FAIL"
    assert metrics["verdict"] == expected
    assert expected == "PASS", f"one of the T-gates failed: {gates}"