"""The verdict must not lag the evidence: `metrics.json` is a projection of
`results/*.json` + `provenance_certificates.json`, and projections go stale.

Round 15 of CASE-DG-EULER-VORTEX-ERR found exactly that: the bundle on disk
carried 14 replay certificates while `metrics.json` still reported
`certificates_on_disk: 12` and `unverified: 7` -- two certificates had landed
(`dg_d2_n32_t2_cfl0.2`, `dg_d2_n64_t1_cfl0.2`) and nobody re-ran the aggregator,
so the package verdict understated its own evidence.

These tests pin the identity instead of the discipline:

    provenance_audit(records, certificates_on_disk)  ==  metrics["provenance"]

If a certificate lands and the aggregate is not refreshed, or a record is
rewritten without re-aggregating, this fails and names the stale keys.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import case_euler_vortex_errconserve as ec

CASE_DIR = Path("case_euler_vortex_errconserve")
RESULTS = CASE_DIR / "results"
CERTIFICATES = CASE_DIR / ec.PROVENANCE_CERTIFICATES
METRICS = CASE_DIR / "metrics.json"


def _records():
    return [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*.json"))]


def _certificates():
    if not CERTIFICATES.exists():
        return {}
    return json.loads(CERTIFICATES.read_text()).get("certificates", {})


@pytest.fixture(scope="module")
def fresh():
    if not RESULTS.exists() or not METRICS.exists():
        pytest.skip("bundle not present")
    records = _records()
    certificates = _certificates()
    return records, certificates, ec.provenance_audit(records, certificates)


@pytest.fixture(scope="module")
def stored():
    if not METRICS.exists():
        pytest.skip("metrics.json not present")
    return json.loads(METRICS.read_text())


def test_counts_match_a_fresh_audit(fresh, stored):
    """The status histogram in metrics.json equals a fresh re-audit."""
    _, _, audit = fresh
    assert stored["provenance"]["counts"] == audit["counts"], (
        "metrics.json provenance counts are stale with respect to "
        f"{CERTIFICATES.name}: stored {stored['provenance']['counts']} vs "
        f"fresh {audit['counts']} -- re-run "
        "`python3 benchmarks/case_euler_vortex_errconserve.py --aggregate`"
    )


def test_per_row_status_matches_a_fresh_audit(fresh, stored):
    """Per-checkpoint status must agree too -- a matching histogram can still
    hide two swapped rows (e.g. one certificate landed, another was rejected)."""
    _, _, audit = fresh
    stored_rows = stored["provenance"]["rows"]
    stale = {
        key: (stored_rows.get(key, {}).get("status"), row["status"])
        for key, row in audit["rows"].items()
        if stored_rows.get(key, {}).get("status") != row["status"]
    }
    assert stale == {}, f"stale provenance rows (stored, fresh): {stale}"


def test_certificate_file_count_matches_a_fresh_audit(fresh, stored):
    """`certificates_on_disk` lives on the C11 gate row and is what the stale
    bundle got wrong first (12 reported, 14 actually on disk)."""
    _, certificates, _ = fresh
    gate = stored["gates"]["C11_provenance_integrity"]
    assert gate["certificates_on_disk"] == len(certificates)


def test_every_replayed_row_has_a_certificate_on_disk(fresh):
    """A `replayed` verdict with no certificate file behind it would be an
    evidence claim with no evidence."""
    _, certificates, audit = fresh
    replayed = [key for key, row in audit["rows"].items() if row["status"] == "replayed"]
    assert replayed, "expected at least one replayed row"
    assert set(replayed) <= set(certificates)


def test_c11_is_not_passed_while_any_record_is_unverified(fresh, stored):
    """Honest-fail guard: C11 cannot be `passed` while an `unverified` row exists."""
    _, _, audit = fresh
    gate = stored["gates"]["C11_provenance_integrity"]
    unverified = [key for key, row in audit["rows"].items()
                  if row["status"] == "unverified"]
    if unverified:
        assert gate["status"] != "passed", (
            f"C11 reported passed while unverified: {unverified}")
        assert gate["passed"] is False
    else:
        assert gate["status"] in {"passed", "not_evaluated"}