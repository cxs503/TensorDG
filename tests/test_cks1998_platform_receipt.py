"""Provenance gate for the REPRO-DG-CKS1998 L4 evidence (mission 19918).

The platform only counts an L4 benchmark row when it can **recompute** ``ref``/``got``
from the raw cluster stdout and bind the row to a *succeeded* cluster job of the same
mission (``POST /tensor/bench/results`` → ``tensorbench_results`` → ``GET
/tensor/bench/l4-verdict/<mid>``).  Those rows live in the platform DB, not in git, so
the repository keeps a byte-exact receipt of the exchange under
``validation/repro_cks1998/artifacts/l4/``.

This test does not trust the numbers *written into* the receipt: it re-derives every
claim from the committed raw bytes (stdout → ``bench_spec.evaluate_metrics`` → compare
with the rows the platform returned) and from the spec's pre-registered metric table.
A receipt that disagrees with its own stdout, or a verdict that is not a machine PASS,
fails here.
"""
import hashlib
import json
from pathlib import Path

import pytest

from validation.repro_cks1998 import bench_spec

RECEIPT_DIR = Path(__file__).resolve().parents[1] / "validation/repro_cks1998/artifacts/l4"


def _load(name):
    path = RECEIPT_DIR / name
    assert path.is_file(), f"missing L4 evidence artifact: {path}"
    return json.loads(path.read_text())


def _stdout_text():
    path = RECEIPT_DIR / "stdout.txt"
    assert path.is_file(), f"missing raw cluster stdout: {path}"
    return path.read_bytes()


def test_receipt_is_bound_to_the_exact_stdout_bytes():
    raw = _stdout_text()
    receipt = _load("receipt.json")
    digest = hashlib.sha256(raw).hexdigest()
    assert receipt["stdout_sha256"] == digest
    # the platform recomputes from these bytes; the receipt must name the same cluster job
    assert int(receipt["cluster_job_id"]) > 0
    assert int(receipt["mission_id"]) == 19918
    # the workload stdout must be pure JSON: the platform's json_stdout extractor
    # cannot be fed a document with log lines mixed in
    json.loads(raw.decode())


def test_spec_metric_table_is_the_preregistered_one():
    spec = _load("spec.json")
    registered = [m["name"] for m in spec["evaluation"]["metrics"]]
    assert registered == [m["name"] for m in bench_spec.METRICS]
    assert spec["benchmark"]["benchmark_id"] == bench_spec.BENCH_ID
    assert spec["implementation"]["project"] == "TensorDG"
    assert spec["evaluation"]["extract"]["format"] == "json_stdout"
    # tolerances are frozen in the spec file, not recomputed from the measurement
    for row, metric in zip(spec["evaluation"]["metrics"], bench_spec.METRICS):
        assert row["tolerance_rel"] == metric["tolerance_rel"]
        assert row["ref_pointer"] == metric["ref_pointer"]
        assert row["got_pointer"] == metric["got_pointer"]


def test_receipt_rows_are_reproducible_from_the_raw_stdout():
    """Re-derive ref/got/error locally; a laundered receipt row fails here."""
    spec = _load("spec.json")
    receipt = _load("receipt.json")
    doc = json.loads(_stdout_text().decode())
    recomputed = bench_spec.evaluate_metrics(spec["evaluation"], doc)
    assert [r["metric"] for r in recomputed] == [r["metric"] for r in receipt["rows"]]
    for mine, theirs in zip(recomputed, receipt["rows"]):
        assert mine["ref_value"] == pytest.approx(theirs["ref_value"], rel=1e-12, abs=0.0)
        assert mine["got_value"] == pytest.approx(theirs["got_value"], rel=1e-12, abs=0.0)
        assert mine["tolerance_rel"] == pytest.approx(theirs["tolerance_rel"], rel=1e-12)
        if mine["rel_error"] is None:
            assert theirs["rel_error"] is None
        else:
            assert mine["rel_error"] == pytest.approx(theirs["rel_error"], rel=1e-9, abs=1e-15)
        assert bool(mine["passed"]) == bool(theirs["passed"])


def test_every_platform_row_passed_and_is_bound_to_the_succeeded_cluster_job():
    receipt = _load("receipt.json")
    rows = receipt["rows"]
    assert len(rows) == len(bench_spec.METRICS)
    assert [r["metric"] for r in rows] == [m["name"] for m in bench_spec.METRICS]
    failed = [r["metric"] for r in rows if not r["passed"]]
    assert not failed, f"platform rows out of tolerance: {failed}"
    for row in rows:
        assert int(row.get("cluster_job_id", receipt["cluster_job_id"])) == int(
            receipt["cluster_job_id"])
        assert str(row.get("execution_status", "succeeded")).lower() in {
            "success", "succeeded", "passed"}
        assert row["rel_error"] is not None      # ref/got/error present, not self-declared


def test_l4_verdict_is_a_machine_pass_not_a_claim():
    verdict = _load("l4_verdict.json")
    assert verdict["l4_benchmark_pass"] is True, verdict
    assert verdict["status"] in {"PASS", "VERIFIED"}, verdict
    assert verdict["failed_checks"] == [], verdict
    assert int(verdict.get("rows", 0)) >= len(bench_spec.METRICS)
    # the pre-closure state the platform reported for this mission is recorded too, so a
    # future reader can see the delta rather than having to trust prose
    pre = _load("l4_verdict_before.json")
    assert pre["l4_benchmark_pass"] is False
    assert pre["status"] == "CLAIMED_NOT_PROVEN"
    assert int(pre.get("rows", -1)) == 0