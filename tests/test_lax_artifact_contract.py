"""Gate for the CASE-LAX-DG cross-artifact contract (C1-C6).

These tests never re-run the solver: the *measurement* half of C6 lives in the
committed reconciliation artifact (produced on the compute node, where the legs
are cheap to run), and what is asserted here is that the record exists, passes,
and is provably about the artifacts currently in the tree (``publisher_digests``).

Why the gate matters: the case family has ten sibling artifacts and, before this
contract, the same key ``dt`` meant the mean step in four of them and the
time-truncated final step in a fifth.  That drift was consumed as evidence of a
non-reproducible artifact (``docs/CASE-LAX-DG-residual-gap.md``) and queued a
full-ladder regeneration.  A silent re-introduction of the drift must fail here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import lax_artifact_reconciliation as reconciliation
from benchmarks.lax_artifact_contract import (
    PUBLISHERS,
    TOL,
    audit,
    dt_identity,
    extract,
    reconcile,
)
from benchmarks.lax_case_pool import FINAL_TIME

ROOT = Path(".")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ C1 - C4
def test_contract_audit_passes():
    result = audit()
    assert result["verdict"] == "PASS", json.dumps(result["publishers"], indent=1)


@pytest.mark.parametrize("spec", PUBLISHERS, ids=lambda spec: spec.name)
def test_semantics_resolved_for_every_leg(spec):
    table = extract(spec)
    assert table, f"{spec.name}: no legs extracted -- artifact missing or path drift?"
    rows = dt_identity(table)
    for row in rows:
        # the bare (or explicit) mean-step key must satisfy its own definition
        assert row["mean_relative_error"] is not None, row
        assert row["mean_relative_error"] <= TOL, row
        if row["last_within_mean"] is not None:
            assert row["last_within_mean"], row


def test_no_artifact_publishes_the_final_step_under_the_bare_dt_key():
    """The drift that was repaired: ``dt`` must not secretly be ``dt_last``."""
    for spec in PUBLISHERS:
        for record in extract(spec).values():
            mean = record["dt_mean"]
            assert mean is not None, f"{spec.name}: leg without a mean step: {record}"
            expected = FINAL_TIME / record["steps"]
            assert abs(mean - expected) / expected <= TOL, (
                f"{spec.name} p{record['degree']}.N{record['cells']}: recorded step "
                f"{mean!r} is not final_time/steps ({expected!r}) -- semantic drift"
            )


def test_attach_script_check_exits_zero():
    completed = subprocess.run(
        [sys.executable, "scripts/attach_lax_dt_semantics.py", "--check"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


# --------------------------------------------------------------------- C5
def test_cross_artifact_agreement():
    result = reconcile()
    assert result["legs_compared"] >= 2, result
    assert result["verdict"] == "PASS", [
        row for row in result["rows"] if row["status"] != "PASS"
    ]
    assert result["worst_relative_difference"] <= TOL


def test_cross_definition_gap_is_not_confused_with_drift():
    """Mismatched definitions produce O(1e-2) gaps; matched ones produce 0.0."""
    gap = reconciliation.cross_definition_gap()
    result = reconcile()
    if gap["worst_relative_difference"] is None:
        pytest.skip("no artifact publishes both definitions")
    # the diagnostic must be strictly larger than the real disagreement, which is
    # what makes "the probe artifact drifted" a mis-reading rather than a finding
    assert gap["worst_relative_difference"] > 100 * max(result["worst_relative_difference"], 1e-15)


# ---------------------------------------------------------------- repair R5
def test_limiter_arms_dt_repair_is_provenanced():
    provenanced = reconciliation.repair_provenance()
    assert provenanced["status"] == "PASS", provenanced
    assert provenanced["solver_rerun"] is False
    assert provenanced["legacy_value_retained_on_every_leg"] is True
    assert provenanced["legs"] >= 1


# ------------------------------------------------------- measurement record
ARTIFACT = reconciliation.OUT / "metrics.json"


def _measurement() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} not committed yet")
    return json.loads(ARTIFACT.read_text())


def test_live_reproduction_record_is_reported_on_both_arms():
    """The floor passes; the pre-registered 1e-12 arm keeps its own (possibly FAIL) verdict.

    A bitwise match between the loong64 artifacts and this host is not achievable,
    so the record must show *both* arms and the reason the floor is post-hoc --
    never a single, convenient number.
    """
    payload = _measurement()
    live = payload["payload_live"]
    assert live["legs"], live
    assert live["worst_relative_difference"] <= live["floor"], [
        row for row in live["rows"] if row["status_at_floor"] != "PASS"
    ]
    assert live["verdict_at_floor"] == "PASS"
    assert live["pre_registered_tolerance"] == TOL
    assert live["pre_registered_verdict"] in ("PASS", "FAIL")
    assert live["floor_reason"], "a post-hoc floor must carry its justification"
    assert live["floor"] > live["pre_registered_tolerance"]


def test_measurement_is_about_the_current_artifacts():
    """Staleness gate: the record names the sha256 of every artifact it read."""
    payload = _measurement()
    digests = payload.get("publisher_digests")
    assert digests, "reconciliation artifact carries no publisher_digests"
    for spec in PUBLISHERS:
        path = ROOT / spec.artifact
        if not path.exists():
            continue
        assert spec.name in digests, spec.name
        assert digests[spec.name] == _sha256(path), (
            f"{spec.artifact} changed since the reconciliation ran; re-run the driver"
        )


def test_quoted_dt_numbers_belong_to_one_run():
    payload = _measurement()
    quoted = payload["quoted_dt"]
    assert quoted["status"] == "PASS", quoted
    assert quoted["both_quoted_values_belong_to_the_same_run"] is True
    # the two numbers differ substantially -- that is exactly why they looked like
    # a drift when read as two separate runs
    values = [row["artifact"] for row in quoted["rows"]]
    assert max(values) / min(values) > 1.05


def test_doc_is_rendered_from_the_committed_measurement():
    payload = _measurement()
    payload["metrics_sha256"] = _sha256(ARTIFACT)
    rendered = reconciliation.render_doc(payload)
    assert reconciliation.DOC.exists(), f"{reconciliation.DOC} missing"
    assert reconciliation.DOC.read_text() == rendered, (
        f"{reconciliation.DOC} is stale; run `--render`"
    )
    sidecar = (reconciliation.OUT / "metrics.sha256").read_text().split()[0]
    assert sidecar == _sha256(ARTIFACT)