"""Machine gate for the dual-instrument Sod gate readings.

The committed artifact ``docs/benchmarks/results/shock_reference_quadrature/metrics.json``
now carries a reading for **all three** registered Sod gates.  Two of those profiles were
computed on the cluster (job cj3966), so the numbers can only be trusted if

* the artifact really has no ``not_run`` gate left (otherwise a hand-edit could quietly
  restore the old "we did not measure it" state),
* every reading is reproducible from the committed profile bytes plus the two reference
  instruments (i.e. the score is not typed into the JSON),
* each profile carries provenance and its sha256 still matches,
* a profile that does **not** reproduce the published gate reading is flagged, so the
  artifact cannot be cited as closing that gate.

The last point is the reason this file exists: the DG P2 profile from cj3966 scores
3.8384 % on velocity while this repository published 2.368 % for the same configuration.
That is a material discrepancy, not a rounding question, and it must stay visible.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import shock_reference_quadrature as srq
from benchmarks import shock_reference_quadrature_audit as audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "shock_reference_quadrature"
CACHE = OUT / "cache"
HEAVY = ("sod-2d-p0-n3200.npz", "sod-dg-p2-n400.npz")


def metrics():
    path = OUT / "metrics.json"
    if not path.exists():
        pytest.fail(f"missing artifact: {path}")
    return json.loads(path.read_text())


def profile(name):
    path = CACHE / name
    if not path.exists():
        pytest.fail(f"missing profile: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: np.asarray(data[key], dtype=float) for key in srq.FIELDS}, json.loads(str(data["meta"]))


def test_artifact_has_no_unmeasured_gate():
    verdict = metrics()["verdict"]
    assert verdict["gates_audited"] == 3
    assert verdict["gates_measured"] == 3, verdict
    assert verdict["gates_not_run"] == [], verdict


@pytest.mark.parametrize("name", HEAVY)
def test_heavy_profiles_have_provenance_and_match_their_sha256(name):
    provenance = json.loads((CACHE / "PROVENANCE.json").read_text())
    entry = provenance["profiles"][name]
    assert entry["sha256"] == hashlib.sha256((CACHE / name).read_bytes()).hexdigest()
    assert "cj3966" in entry["origin"], entry["origin"]
    assert entry["solver_meta"]["final_time"] == pytest.approx(0.2)


@pytest.mark.parametrize("gate", audit.GATES, ids=[g["key"] for g in audit.GATES])
def test_every_committed_reading_rescores_from_the_profile(gate):
    """The JSON must be a score of the committed profile, not a hand-typed number."""
    committed = {g["key"]: g for g in metrics()["gates"]}[gate["key"]]
    assert committed["status"] == "measured"
    import numpy as np  # noqa: F401  (kept local to mirror the driver's numerics)

    stem = {"sod-1d-p0-n3200": "sod-1d-p0-n3200.npz",
            "sod-2d-p0-extruded-n3200": "sod-2d-p0-n3200.npz",
            "sod-dg-p2-tvb-n400": "sod-dg-p2-n400.npz"}[gate["key"]]
    numerical, meta = profile(stem)
    n = gate["ncells"]
    _, exact = srq.cell_averages(n, audit.SOD_TIME, *audit.SOD)
    _, legacy = audit.legacy_cell_averages(n, audit.SOD_TIME, quadrature_order=audit.LEGACY_ORDER)
    for key in srq.FIELDS:
        e_legacy = srq.relative_l2(numerical[key], legacy[key])
        e_exact = srq.relative_l2(numerical[key], exact[key])
        assert committed["fields"][key]["legacy_rel_l2"] == pytest.approx(e_legacy, rel=1e-12)
        assert committed["fields"][key]["exact_rel_l2"] == pytest.approx(e_exact, rel=1e-12)
    assert committed["steps"] == int(meta["steps"])


def test_the_two_instruments_never_disagree_on_a_verdict():
    """The instrument policy question: does swapping the reference flip any gate?"""
    verdict = metrics()["verdict"]
    assert verdict["verdict_flips"] == []
    assert verdict["largest_gate_shift_pp"] < 0.5  # well inside FINGERPRINT_TOL


def test_gates_that_fail_the_exact_reference_are_recorded():
    """Honest FAIL must stay visible: the DG P2 velocity reading is above 3 %."""
    verdict = metrics()["verdict"]
    assert verdict["gates_failing_exact"] == ["sod-dg-p2-tvb-n400:velocity"]
    assert verdict["tightest_gate_margin_pp"] == pytest.approx(-0.7702, abs=5e-4)


def test_the_stale_dg_profile_is_flagged_not_laundered():
    """A profile that misses the published reading must be marked, not accepted."""
    verdict = metrics()["verdict"]
    assert verdict["fingerprint_mismatches"] == ["sod-dg-p2-tvb-n400"]
    gate = next(g for g in metrics()["gates"] if g["key"] == "sod-dg-p2-tvb-n400")
    mark = gate["fingerprint"]
    assert mark["status"] == "mismatch"
    assert mark["delta_pp"] == pytest.approx(1.4704, abs=1e-3)
    others = [g for g in metrics()["gates"] if g["key"] != "sod-dg-p2-tvb-n400"]
    assert {g["fingerprint"]["status"] for g in others} == {"match"}


def test_the_extruded_2d_gate_is_the_1d_gate():
    """2-D extruded P0 (ny=2, periodic) is the 1-D run to 3e-6: not independent evidence."""
    a, _ = profile("sod-1d-p0-n3200.npz")
    b, _ = profile("sod-2d-p0-n3200.npz")
    for key in srq.FIELDS:
        scale = np.max(np.abs(a[key]))
        assert np.max(np.abs(a[key] - b[key])) / scale < 1e-5


def test_a_missing_profile_cannot_be_reported_as_measured(tmp_path, monkeypatch):
    """Negative control: no profile, no reading -- the gate must come back not_run."""
    monkeypatch.setattr(audit, "CACHE", tmp_path)
    monkeypatch.setenv("SRQ_ALLOW_MISSING", "1")
    gate = next(g for g in audit.GATES if g["key"] == "sod-dg-p2-tvb-n400")
    result = audit.measure_gate(gate)
    assert result["status"] == "not_run"
    assert "cj3966" in result["reason"]
    assert result["fingerprint"]["status"] == "not_compared"


def test_provenance_records_every_committed_profile():
    provenance = json.loads((CACHE / "PROVENANCE.json").read_text())
    on_disk = {p.name for p in CACHE.glob("*.npz")}
    assert on_disk <= set(provenance["profiles"])
    for name, entry in provenance["profiles"].items():
        assert entry["sha256"] == hashlib.sha256((CACHE / name).read_bytes()).hexdigest(), name
        assert entry["origin"]
        assert os.path.getsize(CACHE / name) == entry["bytes"]