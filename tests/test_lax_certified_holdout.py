"""Machine checks for the CASE-LAX-DG certified-frontier hold-out study.

These assert *invariants of the study*, not that the study succeeded:

* the pre-registration exists and predates the measurement;
* the two hold-out legs are genuinely unmeasured by the enumerated grid, and the
  second one carries exactly the certified leg's DOF;
* every prediction in ``predictions.json`` can be regenerated from the committed
  enumerated artifact (so the file is not hand-written);
* every measured number in ``metrics.json`` is consistent with the same tolerance
  the ledger uses, and the in-window / verdict flags recompute;
* the re-derived frontier (over enumerated + hold-out legs) matches its own rows.

A failed *prediction* is data, not a red test: the honest outcome lives in
``metrics.json['evaluation']``.  What is enforced here is that the recorded
outcome is the one the arithmetic gives.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from benchmarks.lax_certified_holdout import (
    DOC,
    FIELDS,
    HOLDOUT,
    METRICS,
    PREDICTIONS,
    build_predictions,
    dof,
    enumerated_rows,
    frontier,
    leg_name,
    leave_one_out,
    tolerance,
    worst_field,
)

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "docs/benchmarks/results/lax_certified_holdout"


@pytest.fixture(scope="module")
def predictions():
    if not PREDICTIONS.exists():
        pytest.skip("pre-registration not present")
    return json.loads(PREDICTIONS.read_text())


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip("hold-out measurement not present")
    return json.loads(METRICS.read_text())


# --------------------------------------------------------- pre-registration
def test_predictions_predate_measurement(predictions, metrics):
    registered = predictions["registered_at_utc"]
    measured = metrics["measured_at_utc"]
    assert registered <= measured, (registered, measured)
    assert predictions["registered_at_utc"] == metrics["registered_at_utc"]


def test_holdout_legs_are_unmeasured(predictions):
    enumerated = {row["leg"] for row in predictions["enumerated_legs"]}
    holdout = {row["leg"] for row in
               ({"leg": leg_name(c, d)} for c, d in HOLDOUT)}
    assert holdout and not (holdout & enumerated)
    assert set(predictions["legs"]) == holdout


def test_second_holdout_shares_certified_dof(predictions):
    by_leg = {row["leg"]: row for row in predictions["enumerated_legs"]}
    certified = by_leg["p2.N800"]
    same_dof = [name for name, design in predictions["legs"].items()
                if design["dof"] == certified["dof"]]
    assert same_dof, "no hold-out shares the certified leg's DOF"
    for name in same_dof:
        design = predictions["legs"][name]
        assert design["dof"] == dof(design["cells"], design["degree"])


def test_window_is_loo_max(predictions):
    for model, table in predictions["calibration"].items():
        errors = [item["abs_error"] for item in table["loo"]]
        assert table["window"] == pytest.approx(max(errors), rel=1e-12)
        for item in table["loo"]:
            assert item["abs_error"] == pytest.approx(
                abs(item["predicted"] - item["observed"]), rel=1e-12)
        for name, design in predictions["legs"].items():
            assert design["per_model"][model]["window"] == pytest.approx(
                table["window"], rel=1e-12)


def test_predictions_regenerate_from_committed_enumerated_legs(predictions):
    """The JSON must be the output of the fit, not a transcription."""
    rebuilt = build_predictions(rows=enumerated_rows(),
                                tol=predictions["tolerance"])
    for name, design in predictions["legs"].items():
        for model, item in design["per_model"].items():
            assert item["predicted_worst_l2"] == pytest.approx(
                rebuilt["legs"][name]["per_model"][model]["predicted_worst_l2"],
                rel=1e-12)
            assert item["predicted_verdict"] == \
                rebuilt["legs"][name]["per_model"][model]["predicted_verdict"]
    for model in predictions["calibration"]:
        assert predictions["calibration"][model]["window"] == pytest.approx(
            rebuilt["calibration"][model]["window"], rel=1e-12)


def test_loo_calibration_is_leave_one_out(predictions):
    rows = predictions["enumerated_legs"]
    for model, table in predictions["calibration"].items():
        assert len(table["loo"]) == len(rows)
        assert {item["leg"] for item in table["loo"]} == {r["leg"] for r in rows}
        recomputed = leave_one_out(rows, model)
        assert [i["abs_error"] for i in recomputed] == pytest.approx(
            [i["abs_error"] for i in table["loo"]], rel=1e-12)


# ------------------------------------------------------------- measurement
def test_tolerance_comes_from_the_registered_case(predictions, metrics):
    tol = tolerance()
    assert predictions["tolerance"] == pytest.approx(tol, rel=1e-12)
    assert metrics["tolerance"] == pytest.approx(tol, rel=1e-12)


def test_measured_legs_are_internally_consistent(metrics):
    for row in metrics["legs"]:
        assert row["dof"] == dof(row["cells"], row["degree"])
        worst, field = worst_field(row)
        assert row["worst_l2"] == pytest.approx(worst, rel=1e-12)
        assert row["worst_field"] == field
        assert set(row["l2_errors"]) == set(FIELDS)
        assert row["verdict"] == ("PASS" if worst < metrics["tolerance"] else "FAIL")


def test_measured_legs_are_the_preregistered_legs(metrics):
    assert {row["leg"] for row in metrics["legs"]} == \
        {leg_name(c, d) for c, d in HOLDOUT}


def test_evaluation_matches_measurement(metrics):
    by_leg = {row["leg"]: row for row in metrics["legs"]}
    inside = correct = 0
    for row in metrics["evaluation"]["legs"]:
        measured = by_leg[row["leg"]]
        assert row["dof"] == measured["dof"]
        for model, item in row["per_model"].items():
            predicted = metrics["predictions"]["legs"][row["leg"]]["per_model"][model]
            assert item["predicted_worst_l2"] == pytest.approx(
                predicted["predicted_worst_l2"], rel=1e-12)
            assert item["observed_worst_l2"] == pytest.approx(
                measured["worst_l2"], rel=1e-12)
            delta = measured["worst_l2"] - predicted["predicted_worst_l2"]
            assert item["signed_error"] == pytest.approx(delta, rel=1e-9, abs=1e-15)
            assert item["abs_error"] == pytest.approx(abs(delta), rel=1e-9, abs=1e-15)
            assert item["inside_window"] == (abs(delta) <= item["window"] + 1e-15)
            assert item["observed_verdict"] == measured["verdict"]
            assert item["verdict_correct"] == \
                (item["predicted_verdict"] == measured["verdict"])
            inside += bool(item["inside_window"])
            correct += bool(item["verdict_correct"])
    summary = metrics["evaluation"]["summary"]
    assert summary["predictions"] == 2 * len(metrics["legs"])
    assert summary["inside_window"] == inside
    assert summary["verdict_correct"] == correct


def test_consensus_rule_is_honest(predictions):
    for name, design in predictions["legs"].items():
        verdicts = {item["predicted_verdict"]
                    for item in design["per_model"].values()}
        consensus = design["consensus"]
        assert consensus["agree"] == (len(verdicts) == 1)
        if consensus["agree"]:
            assert consensus["verdict"] == verdicts.pop()
        else:
            assert consensus["verdict"] == "DISPUTED"


# ---------------------------------------------------------------- frontier
def test_frontier_recomputes_from_rows(metrics):
    tol = metrics["tolerance"]
    rows = metrics["frontier"]["rows"]
    enumerated = {row["leg"] for row in metrics["predictions"]["enumerated_legs"]}
    holdout = {row["leg"] for row in metrics["legs"]}
    assert {row["leg"] for row in rows} == enumerated | holdout
    for row in rows:
        assert row["source"] == ("hold-out" if row["leg"] in holdout else "enumerated")
        assert row["worst_l2"] == pytest.approx(
            next(r["worst_l2"] for r in metrics["legs"] + metrics["predictions"]["enumerated_legs"]
                 if r["leg"] == row["leg"]), rel=1e-12)
    ordered = sorted(rows, key=lambda r: (r["dof"], r["degree"]))
    passing = [r for r in ordered if r["worst_l2"] < tol]
    expected = passing[0] if passing else None
    recorded = metrics["frontier"]["minimal_dof_passing"]
    if expected is None:
        assert recorded is None
    else:
        assert recorded["leg"] == expected["leg"]
        assert recorded["dof"] == expected["dof"]
    assert metrics["frontier"]["minimality_restated"] == (
        expected is not None and expected["leg"] != "p2.N800")


def test_frontier_helper_matches_artifact(metrics):
    rebuilt = frontier(metrics["predictions"]["enumerated_legs"], metrics["legs"],
                       tol=metrics["tolerance"])
    assert rebuilt["minimal_dof_passing"] == metrics["frontier"]["minimal_dof_passing"]
    assert rebuilt["minimality_restated"] == metrics["frontier"]["minimality_restated"]


# ------------------------------------------------------- metric provenance
def test_driver_does_not_reimplement_the_metric():
    source = (ROOT / "benchmarks/lax_certified_holdout.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert "benchmarks.lax_case_pool.run_ladder" in imported
    assert not any("solve_euler_dg_1d" in name for name in imported)


def test_artifacts_and_doc_present(metrics):
    for name in ("predictions.json", "metrics.json", "summary.md"):
        assert (RESULT_DIR / name).exists(), name
    assert DOC.exists()
    for figure in metrics["figures"]:
        assert (ROOT / figure).exists(), figure