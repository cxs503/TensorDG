"""Tests for ``tools/vortex_longtime_horizon.py``.

The tool's whole value is the *order* of operations: the ``t = 40`` prediction is
frozen from the archive (``final_time <= 20``) before any ``t = 40`` record
exists, and the measurement is then scored against it.  These tests pin that
order down, not the numbers themselves (the numbers move when new checkpoints
land; the protocol must not).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import vortex_longtime_horizon as vh

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "case_euler_vortex_errconserve" / "results"


def _record(task, cells, final_time, max_percent, degree=None, cfl=0.2,
            reference="lattice", background_velocity=None):
    config = {"task": task, "cells": cells, "final_time": final_time,
              "cfl": cfl, "strength": 5.0, "reference": reference}
    if degree is not None:
        config["degree"] = degree
    if background_velocity is not None:
        config["background_velocity"] = list(background_velocity)
    record = {"task": task, "cells": cells, "final_time": final_time,
              "cfl": cfl, "max_percent": max_percent, "config": config,
              "steps": 1, "wall_seconds": 1.0}
    if degree is not None:
        record["degree"] = degree
    return record


@pytest.fixture(scope="module")
def predictions(tmp_path_factory):
    out = tmp_path_factory.mktemp("horizon")
    if not ARCHIVE.is_dir():
        pytest.skip("archive checkpoints are not present")
    return vh.predict(ARCHIVE, out), out


def test_prediction_is_frozen_before_the_runs(predictions):
    doc, _ = predictions
    assert doc["frozen_before_runs"] is True
    for entry in doc["families"].values():
        assert max(entry["archive_times"]) <= vh.ARCHIVE_TIME_CAP


def test_predictions_are_reproducible_from_the_archive(predictions, tmp_path):
    doc, _ = predictions
    again = vh.predict(ARCHIVE, tmp_path)
    assert again["archive_digest_sha256"] == doc["archive_digest_sha256"]
    for family, entry in doc["families"].items():
        for name, model in entry["models"].items():
            assert again["families"][family]["models"][name]["predicted_percent"] == (
                pytest.approx(model["predicted_percent"], rel=1e-12, abs=0.0))


def test_committed_prediction_file_matches_a_fresh_freeze(predictions):
    """The published ``predictions.json`` must be reproducible, not hand-edited."""
    doc, _ = predictions
    committed = ROOT / "case_euler_vortex_errconserve_longtime" / "predictions.json"
    if not committed.exists():
        pytest.skip("no published prediction file yet")
    stored = json.loads(committed.read_text())
    assert stored["archive_digest_sha256"] == doc["archive_digest_sha256"]
    for family, entry in doc["families"].items():
        assert stored["families"][family]["models"]["tail_local"]["alpha"] == (
            pytest.approx(entry["models"]["tail_local"]["alpha"], rel=1e-12))


def test_the_prediction_has_power_the_models_disagree(predictions):
    """A prediction whose models agree cannot arbitrate between hypotheses."""
    doc, _ = predictions
    design = doc["families"]["dg_d2_n32_cfl0.2"]
    low, high = design["prediction_spread_percent"]
    assert high / low > 1.2, "models too close to separate conservation from accumulation"
    assert design["models"]["linear_bound"]["predicted_percent"] >= high - 1e-12


def test_a_late_record_is_rejected_as_a_prediction_leak(tmp_path):
    late = tmp_path / "archive"
    late.mkdir()
    records = [_record("dg", 32, t, 0.01 * t, degree=2) for t in (1.0, 2.0, 5.0)]
    records.append(_record("dg", 32, 40.0, 0.03, degree=2))
    for i, record in enumerate(records):
        (late / f"r{i}.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="leak guard"):
        vh.predict(late, tmp_path / "out")


def test_analyze_scores_the_measurement_and_extends_the_window(tmp_path):
    out = tmp_path / "case"
    (out / "results").mkdir(parents=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    truth = {1.0: 0.0100, 2.0: 0.0102, 5.0: 0.0110, 10.0: 0.0120}
    for t, e in truth.items():
        (archive / f"a{t}.json").write_text(
            json.dumps(_record("dg", 32, t, e, degree=2, reference="min_image")))
    vh.predict(archive, out, targets={"dg_d2_n32_cfl0.2": "design family"})

    (out / "results" / "t40.json").write_text(
        json.dumps(_record("dg", 32, 40.0, 0.0135, degree=2)))
    doc = vh.analyze(out / "predictions.json", [out / "results"], out)

    design = doc["families"]["dg_d2_n32_cfl0.2"]
    assert design["status"] == "measured"
    assert design["measured_percent"] == pytest.approx(0.0135)
    assert design["extended"]["window"] == [1.0, 40.0]
    assert set(design["scores"]) == {"ols_window", "robust_window",
                                    "tail_local", "linear_bound"}
    assert design["within_tolerance"] is True
    assert doc["verdict"]["status"] in {"PASS", "FAIL"}
    assert (out / "horizon.json").exists() and (out / "horizon.md").exists()


def test_missing_design_checkpoint_is_incomplete_not_pass(tmp_path):
    out = tmp_path / "case"
    archive = tmp_path / "archive"
    archive.mkdir()
    (out / "results").mkdir(parents=True)
    for t, e in {1.0: 0.01, 2.0: 0.0102, 5.0: 0.011}.items():
        (archive / f"a{t}.json").write_text(
            json.dumps(_record("dg", 32, t, e, degree=2, reference="min_image")))
    vh.predict(archive, out, targets={"dg_d2_n32_cfl0.2": "design family"})
    doc = vh.analyze(out / "predictions.json", [out / "results"], out)
    assert doc["verdict"]["status"] == "INCOMPLETE"


def test_secular_accumulation_is_not_called_conservation(tmp_path):
    """A family whose error grows like t must fail the extended C2 shoulder."""
    out = tmp_path / "case"
    archive = tmp_path / "archive"
    archive.mkdir()
    (out / "results").mkdir(parents=True)
    for t, e in {1.0: 0.01, 2.0: 0.02, 5.0: 0.05}.items():
        (archive / f"a{t}.json").write_text(
            json.dumps(_record("dg", 32, t, e, degree=2, reference="min_image")))
    vh.predict(archive, out, targets={"dg_d2_n32_cfl0.2": "design family"})
    (out / "results" / "t40.json").write_text(
        json.dumps(_record("dg", 32, 40.0, 0.90, degree=2)))
    doc = vh.analyze(out / "predictions.json", [out / "results"], out)
    design = doc["families"]["dg_d2_n32_cfl0.2"]
    assert design["extended"]["gate_holds"] is False
    assert doc["verdict"]["status"] == "FAIL"
    assert "extended_alpha_within_bound" in doc["verdict"]["failed"]


def test_horizon_extrapolation_matches_its_closed_form():
    assert vh._horizon_time(10.0, 1.0, 1.0, 6.0) == pytest.approx(60.0)
    assert vh._horizon_time(10.0, 1.0, 0.5, 6.0) == pytest.approx(360.0)
    # a flat family never reaches the tolerance on its own
    assert vh._horizon_time(10.0, 1.0, 0.0, 6.0) == float("inf")


def test_stored_family_labels_are_not_trusted():
    """Four archived records carry the pre-CFL label; the tool must see one family."""
    records = [json.loads(p.read_text()) for p in ARCHIVE.glob("*.json")] if ARCHIVE.is_dir() else []
    if not records:
        pytest.skip("archive checkpoints are not present")
    stale = [r for r in records if r.get("family") != vh._record_family(r)]
    assert stale, "fixture drift: expected at least one record with a stale label"
    times, _ = vh._series(records, "dg_d2_n32_cfl0.2")
    assert len(times) >= 4, "the C2 gate family must not be split by a stale label"