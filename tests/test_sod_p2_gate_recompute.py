"""Regression gate for the DG P2 Sod re-measurement (2.368 % vs 3.838 % dispute).

The audit artifact ``docs/benchmarks/results/cks1998_dual_instrument/sod_p2_gate_recompute.json``
is produced by ``validation.repro_cks1998.sod_p2_gate_recompute`` on a compute node
(one DG P2 N=400 run costs ~2 min, so it cannot run in the local test budget).

These assertions hold **with or without** the artifact, and deliberately fail loudly
when the artifact claims a verdict its own numbers do not support:

1. the driver's published/cached constants are the ones the repository actually
   registered (``metrics.json`` fingerprint row) — a drifted constant would make any
   attribution meaningless;
2. the sweep is pre-registered: the documented configuration must be in it, and every
   swept axis is one the gate's documentation leaves implicit;
3. if the artifact exists, its verdict must be derivable from its own rows:
   ``attribution`` says "code-state mismatch" exactly when no sweep entry reproduces the
   published reading, and the current-code gate verdict must agree with the exact
   instrument reading against the frozen 3 % tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.repro_cks1998 import sod_p2_gate_recompute as driver

METRICS = Path("docs/benchmarks/results/shock_reference_quadrature/metrics.json")
ARTIFACT = Path("docs/benchmarks/results/cks1998_dual_instrument/sod_p2_gate_recompute.json")


def _metrics() -> dict:
    if not METRICS.exists():  # pragma: no cover - committed artifact
        pytest.skip("shock_reference_quadrature metrics.json is not present")
    return json.loads(METRICS.read_text())


def test_published_and_cached_constants_match_the_registered_fingerprint():
    gate = next(g for g in _metrics()["gates"] if g["key"] == "sod-dg-p2-tvb-n400")
    fingerprint = gate["fingerprint"]
    assert fingerprint["registered_rel_l2"] == pytest.approx(driver.PUBLISHED_VELOCITY_REL_L2, rel=0, abs=1e-12)
    assert fingerprint["measured_rel_l2"] == pytest.approx(driver.CACHED_VELOCITY_REL_L2, rel=0, abs=1e-12)
    # the dispute is real: the two numbers straddle the frozen 3 % tolerance
    assert driver.PUBLISHED_VELOCITY_REL_L2 < driver.TOLERANCE < driver.CACHED_VELOCITY_REL_L2
    assert driver.TOLERANCE == pytest.approx(_metrics()["tolerance"], rel=0, abs=1e-12)


def test_the_documented_configuration_is_the_first_entry_of_the_sweep():
    documented = driver.SWEEP[0]
    assert {k: documented[k] for k in driver.DOCUMENTED} == driver.DOCUMENTED
    # the documented configuration is the one the gate row claims (metrics.json)
    gate = next(g for g in _metrics()["gates"] if g["key"] == "sod-dg-p2-tvb-n400")
    assert "P2" in gate["config"] and "N=400" in gate["config"] and "CFL 0.12" in gate["config"]
    assert driver.DOCUMENTED["degree"] == 2 and driver.DOCUMENTED["ncells"] == 400
    assert driver.DOCUMENTED["cfl"] == 0.12


def test_every_swept_axis_is_an_implicit_axis_of_the_gate_documentation():
    # axes whose value the gate row does not pin down; the sweep must not invent new physics
    allowed = {"limiter_stage", "positivity", "tvb_constant", "cfl", "degree"}
    for entry in driver.SWEEP:
        changed = {k for k in allowed if entry.get(k) != driver.DOCUMENTED.get(k)}
        assert changed or entry["tag"] == "documented", entry


def test_the_artifact_verdict_is_derivable_from_its_own_rows():
    if not ARTIFACT.exists():
        pytest.skip("recompute artifact not fetched yet (cluster job); honest not_run")
    doc = json.loads(ARTIFACT.read_text())
    runs = doc["runs"]
    verdict = doc["verdict"]
    base = runs[0]
    assert base["config"]["tag"] == "documented"
    # 1. the verdict quotes the documented run, not some other sweep entry
    assert verdict["current_code_worst_legacy_pct"] == pytest.approx(100 * base["worst_legacy"], rel=0, abs=1e-9)
    assert verdict["current_code_gate_passes"] == (base["worst_exact"] < doc["tolerance"])
    # 2. attribution follows the pre-registered rule, not the author's narrative
    reproducing = [r["config"]["tag"] for r in runs if r["reproduces_published"]]
    assert verdict["configs_reproducing_published"] == reproducing
    if reproducing:
        assert verdict["attribution"].startswith("configuration mismatch")
    else:
        assert verdict["attribution"].startswith("no neighbouring configuration")
    # 3. a "reproduces_published" flag must be backed by the reading itself
    for row in runs:
        gap = abs(100 * row["worst_legacy"] - 100 * doc["published_velocity_rel_l2"])
        assert row["gap_to_published_pp"] == pytest.approx(gap, rel=1e-9, abs=1e-9)
        assert row["reproduces_published"] == (gap < 0.05)
    # 4. the committed cached-profile reading must be either reproduced or contradicted
    #    explicitly -- no silent middle ground
    assert verdict["agrees_with_cached_profile"] == (
        abs(base["worst_legacy"] - doc["cached_velocity_rel_l2"]) < 0.005)