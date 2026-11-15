"""C11 must validate a replay certificate itself, not trust the landing step.

``tools/land_provenance_certificates.py`` filters certificates before they are
written, but a certificate bundle is a plain JSON file: a hand-edited bundle, a
future merge bug or a replayed key can put a claim on disk that the landing step
never saw.  Gate C11 therefore re-derives three bindings from the record and the
certificate alone:

* the physics/discretisation fields (a certificate for another mesh is evidence
  for another problem);
* the checkpoint name the certificate says it replayed;
* the stored percentage it claims to reproduce.

These tests pin the gate (not the tool).  They also pin the *other* direction: a
certificate that does describe its checkpoint must still be admitted, otherwise
the hardening would silently turn into a stricter acceptance criterion, which
would be moving the goalposts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import case_euler_vortex_errconserve as ec

BUNDLE_DIR = Path("case_euler_vortex_errconserve")

RECORD = {
    "task_key": "dg_d2_n32_t2_cfl0.2",
    "config": {"task": "dg", "degree": 2, "cells": 32, "cfl": 0.2,
               "final_time": 2.0, "strength": 5.0},
    "reference_model": None,      # a legacy record: it needs a certificate
    "errors": {"density": 1e-4, "velocity_x": 2e-4,
               "velocity_y": 2e-4, "pressure": 3e-4},
    "max_percent": 0.0231,        # percent, as written by the benchmark
    "family": "dg_d2_n32_cfl0.2",
}


def certificate(**overrides) -> dict:
    base = {
        "task_key": RECORD["task_key"],
        "config": dict(RECORD["config"]),
        "reference_model": ec.REFERENCE_MODEL,
        "replay_delta": 1e-13,
        "stored_max_percent": RECORD["max_percent"],
        "replayed_max_percent": RECORD["max_percent"],
        "wall_seconds": 12.0,
    }
    base.update(overrides)
    return base


def audit(cert: dict) -> dict:
    return ec.provenance_audit([dict(RECORD)], {RECORD["task_key"]: cert})


# ---------------------------------------------------------------- admitted
def test_matching_certificate_is_replayed():
    rows = audit(certificate())
    row = rows["rows"][RECORD["task_key"]]
    assert row["status"] == "replayed", row.get("problems")
    assert rows["counts"].get("replayed", 0) == 1
    assert rows["counts"].get("unverified", 0) == 0


def test_no_certificate_stays_unverified():
    rows = ec.provenance_audit([dict(RECORD)], {})
    assert rows["rows"][RECORD["task_key"]]["status"] == "unverified"


# ---------------------------------------------------------------- rejected
@pytest.mark.parametrize("override, field", [
    ({"config": {**RECORD["config"], "cells": 64}}, "cells"),
    ({"config": {**RECORD["config"], "final_time": 10.0}}, "final_time"),
    ({"config": {**RECORD["config"], "cfl": 0.4}}, "cfl"),
    ({"config": {**RECORD["config"], "degree": 3}}, "degree"),
    ({"task_key": "dg_d2_n64_t2_cfl0.2"}, "task_key"),
    ({"stored_max_percent": 0.5 * RECORD["max_percent"]}, "stored_max_percent"),
])
def test_certificate_for_another_checkpoint_is_rejected(override, field):
    """A certificate whose geometry/time/numbers belong to another checkpoint is
    not evidence for this one, even when its replay delta is perfect."""
    rows = audit(certificate(**override))
    row = rows["rows"][RECORD["task_key"]]
    assert row["status"] == "certificate_rejected", row
    assert field in row["certificate_mismatch"]
    assert any("does not describe this checkpoint" in p for p in row["problems"])
    assert rows["counts"].get("certificate_rejected", 0) == 1
    assert rows["counts"].get("replayed", 0) == 0


def test_wrong_reference_model_is_rejected():
    rows = audit(certificate(reference_model="free_space"))
    assert rows["rows"][RECORD["task_key"]]["status"] == "certificate_rejected"


def test_over_tolerance_delta_is_rejected():
    rows = audit(certificate(replay_delta=1e-6))
    assert rows["rows"][RECORD["task_key"]]["status"] == "certificate_rejected"


def test_nan_delta_is_rejected():
    rows = audit(certificate(replay_delta=float("nan")))
    assert rows["rows"][RECORD["task_key"]]["status"] == "certificate_rejected"


# ------------------------------------------------- gate-level consequence
# The gate rule itself is one line of ``evaluate_gates``: a ``certificate_rejected``
# row forces C11 to ``failed`` (it can never read as passed), so the audit-level
# assertions above are what decides the verdict.  ``evaluate_gates`` additionally
# needs the exact-invariant bundle, which is produced by a run rather than by a
# synthetic record; ``tests/test_case_euler_vortex_errconserve.py`` covers it on
# the real bundle.


# ------------------------------------------------------ no goalpost moved
@pytest.mark.skipif(not (BUNDLE_DIR / "provenance_certificates.json").exists(),
                    reason="no certificate bundle on disk")
def test_landed_bundle_is_still_admitted():
    """Every certificate that already landed must survive the stricter gate: the
    hardening closes a hole, it does not tighten acceptance."""
    records = [json.loads(path.read_text())
               for path in sorted((BUNDLE_DIR / "results").glob("*.json"))]
    certificates = ec.load_provenance_certificates(BUNDLE_DIR)
    audit_rows = ec.provenance_audit(records, certificates)
    rejected = [key for key, row in audit_rows["rows"].items()
                if row["status"] == "certificate_rejected"]
    assert not rejected, {key: audit_rows["rows"][key]["problems"] for key in rejected}
    assert audit_rows["counts"].get("replayed", 0) == len(certificates)