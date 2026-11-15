"""Machine checks for the retrospective out-of-sample extrapolation study.

The invariant enforced here is *provenance*, not success: the tested legs must not
be part of the enumerated grid the models were fitted on, every prediction and
window must regenerate from the committed artifacts, and every recorded flag must
be the one the arithmetic gives (so a bad model would show up as recorded
``inside_window = False``, never as an edited table).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.lax_certified_extrapolation import EXTRAPOLATION, unfitted_legs
from benchmarks.lax_certified_holdout import (
    FIELDS,
    calibrate,
    dof,
    enumerated_rows,
    predict,
    tolerance,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def payload():
    if not EXTRAPOLATION.exists():
        pytest.skip("extrapolation artifact not present")
    return json.loads(EXTRAPOLATION.read_text())


def test_tested_legs_were_never_fitted(payload):
    enumerated = {row["leg"] for row in payload["enumerated_legs"]}
    tested = {row["leg"] for row in payload["legs"]}
    assert tested and not (tested & enumerated)
    assert [row["leg"] for row in unfitted_legs(enumerated_rows())] == \
        [row["leg"] for row in payload["legs"]]


def test_tolerance_is_imported(payload):
    assert payload["tolerance"] == pytest.approx(tolerance(), rel=1e-12)


def test_predictions_regenerate_from_enumerated_legs(payload):
    rows = payload["enumerated_legs"]
    calibration = calibrate(rows)
    for row in payload["legs"]:
        assert row["dof"] == dof(row["cells"], row["degree"])
        assert set(row["l2_errors"]) == set(FIELDS)
        assert row["worst_l2"] == pytest.approx(max(row["l2_errors"].values()), rel=1e-12)
        for model, table in calibration.items():
            assert table["window"] == pytest.approx(
                payload["calibration"][model]["window"], rel=1e-12)
            expected = predict(model, rows, row["cells"], row["degree"])
            recorded = next(e for e in payload["evaluation"]["legs"]
                            if e["leg"] == row["leg"])["per_model"][model]
            assert recorded["predicted_worst_l2"] == pytest.approx(expected, rel=1e-12)


def test_flags_recompute(payload):
    tol = payload["tolerance"]
    by_leg = {row["leg"]: row for row in payload["legs"]}
    inside = correct = 0
    for entry in payload["evaluation"]["legs"]:
        measured = by_leg[entry["leg"]]
        for model, item in entry["per_model"].items():
            delta = measured["worst_l2"] - item["predicted_worst_l2"]
            assert item["signed_error"] == pytest.approx(delta, rel=1e-9, abs=1e-15)
            assert item["inside_window"] == (abs(delta) <= item["window"] + 1e-15)
            expected = "PASS" if item["predicted_worst_l2"] < tol else "FAIL"
            observed = "PASS" if measured["worst_l2"] < tol else "FAIL"
            assert item["predicted_verdict"] == expected
            assert item["observed_verdict"] == observed
            assert item["verdict_correct"] == (expected == observed)
            inside += bool(item["inside_window"])
            correct += bool(item["verdict_correct"])
    summary = payload["evaluation"]["summary"]
    assert summary["predictions"] == 2 * len(payload["legs"])
    assert summary["inside_window"] == inside
    assert summary["verdict_correct"] == correct


def test_design_is_labelled_as_retrospective(payload):
    design = payload["design"]
    assert "retrospective" in design["kind"]
    assert "not blind" in design["caveat"]
    numeric = [r["dof"] for r in payload["enumerated_legs"]]
    spans = any(r["dof"] > max(numeric) or r["dof"] < min(numeric)
                for r in payload["legs"])
    assert design["spans_beyond_fitted_dof_range"] is spans


def test_frontier_recomputes(payload):
    tol = payload["tolerance"]
    rows = payload["frontier"]["rows"]
    passing = [r for r in sorted(rows, key=lambda r: (r["dof"], r["degree"]))
               if r["worst_l2"] < tol]
    recorded = payload["frontier"]["minimal_dof_passing"]
    assert recorded["leg"] == passing[0]["leg"]
    assert recorded["dof"] == passing[0]["dof"]


def test_doc_and_figures_are_on_disk(payload):
    for figure in payload["figures"]:
        assert (ROOT / figure).exists(), figure
    doc = ROOT / payload["doc"]
    assert doc.exists()
    assert "回顾性 out-of-sample" in doc.read_text()
