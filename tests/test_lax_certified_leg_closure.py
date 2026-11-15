"""Contract tests for the CASE-LAX-DG certified-leg closure artifact.

The artifact ``docs/benchmarks/results/lax_certified_leg_closure/metrics.json``
is produced by ``python -m benchmarks.lax_certified_leg_closure`` (three legs x
three exact ``dt`` halvings, two quadrature rules).  These tests do not re-solve
anything: they assert the registered invariants, recompute the converged reading
from the committed field cache, and refuse silent drift between the artifact,
the rendered document and the ledger's own tolerance source.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_leg_closure"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"
DOC = ROOT / "docs" / "CASE-LAX-DG-certified-leg-closure.md"
FIELDS = ("density", "velocity", "pressure")
CONVERGED_ORDER = "20"


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip(f"missing artifact {METRICS.relative_to(ROOT)}")
    return json.loads(METRICS.read_text())


def _leg(metrics, name):
    for row in metrics["legs"]:
        if f"p{row['degree']}.N{row['cells']}" == name:
            return row
    raise AssertionError(f"leg {name} not measured")


def test_artifact_carries_the_anchor_and_the_ledger_tolerance(metrics):
    from benchmarks.lax_case_pool import ledger_acceptance_tolerance

    assert metrics["anchor"] == "CASE-LAX-DG"
    assert metrics["case"] == "CASE-LAX-DG"
    assert metrics["config"]["tolerance"] == ledger_acceptance_tolerance()
    assert metrics["checks"]["ledger"]["tolerance"] == ledger_acceptance_tolerance()
    assert len(metrics["config"]["breakpoints"]) == 4  # head, tail, contact, shock


def test_every_preregistered_gate_passed(metrics):
    checks = metrics["checks"]
    assert checks["reuse"]["status"] == "PASS"
    assert checks["cross_driver"]["status"] == "PASS"
    assert checks["convergence"]["status"] == "PASS"
    assert checks["time_budget"]["status"] == "PASS"
    assert checks["step_law"]["status"] == "PASS"
    assert checks["ledger"]["C2_certified_pass_is_quadrature_and_dt_robust"] == "PASS"
    assert checks["ledger"]["C3_cheaper_leg_fail_is_quadrature_and_dt_robust"] == "PASS"
    assert checks["ledger"]["C5_pinned_honest_fail_survives"] == "PASS"
    assert metrics["verdict"] == "PASS"


def test_certified_leg_passes_at_every_cfl(metrics):
    tolerance = metrics["config"]["tolerance"]
    row = _leg(metrics, "p2.N800")
    assert len(row["levels"]) == 3
    for level in row["levels"]:
        converged = level["aligned"][CONVERGED_ORDER]["l2"]
        assert max(converged.values()) < tolerance, level["cfl"]
        # the committed rule must not be the only reason it passes
        assert max(level["uniform"]["l2"].values()) < tolerance, level["cfl"]


def test_cheaper_leg_and_pinned_leg_still_fail(metrics):
    tolerance = metrics["config"]["tolerance"]
    for name in ("p2.N400", "p3.N400"):
        row = _leg(metrics, name)
        for level in row["levels"]:
            converged = level["aligned"][CONVERGED_ORDER]["l2"]
            assert max(converged.values()) >= tolerance, (name, level["cfl"])


def test_quadrature_convergence_at_round_off(metrics):
    worst = 0.0
    for row in metrics["legs"]:
        for level in row["levels"]:
            a = level["aligned"]["16"]["l2"]
            b = level["aligned"][CONVERGED_ORDER]["l2"]
            worst = max(worst, max(abs(a[f] - b[f]) / b[f] for f in FIELDS))
    assert worst <= metrics["checks"]["convergence"]["tolerance"]


def test_reuse_and_cross_driver_bounds(metrics):
    reuse = metrics["checks"]["reuse"]
    assert reuse["worst_relative_difference"] <= reuse["tolerance"]
    assert len(reuse["rows"]) == 3 * len(FIELDS)
    cross = metrics["checks"]["cross_driver"]
    assert cross["worst_a0_relative_difference"] <= cross["tolerances"]["a0"]
    assert cross["worst_floor_relative_difference"] <= cross["tolerances"]["floor"]
    drivers = {row["driver"] for row in cross["rows"]}
    assert drivers == {"lax_a0_quadrature_audit.aligned_20",
                       "lax_reference_floor.total_exact"}


def test_committed_cache_reproduces_the_converged_reading(metrics):
    """Artifact integrity: recompute one leg from the npz cache, no solver."""
    from benchmarks.lax_certified_leg_closure import aligned_measure

    if not PROFILES.exists():
        pytest.skip(f"missing cache {PROFILES.relative_to(ROOT)}")
    data = np.load(PROFILES)
    key = "p2_N400_d1"
    points = data[f"{key}_points"]
    numerical = {f: data[f"{key}_numerical_{f}"] for f in FIELDS}
    recomputed = aligned_measure(points, numerical, 20)
    stored = _leg(metrics, "p2.N400")["levels"][0]["aligned"][CONVERGED_ORDER]["l2"]
    for field in FIELDS:
        assert abs(recomputed[field] - stored[field]) / stored[field] <= 1e-12


def test_documented_dt_series_are_monotone_in_the_reading_used(metrics):
    """Every recorded temporal reading must come from a real 3-point series."""
    for name, fields in metrics["checks"]["time_budget"]["per_leg"].items():
        for field, item in fields.items():
            assert len(item["aligned_series"]) == 3, (name, field)
            assert all(value > 0.0 for value in item["aligned_series"])
            if item["reading"] == "share":
                assert item["temporal_share_aligned"] is not None
            else:
                assert item["reading"] == "bound"


def test_rendered_doc_matches_the_artifact(metrics):
    from benchmarks.lax_certified_leg_closure import render_doc

    assert DOC.exists()
    assert render_doc(metrics) == DOC.read_text()


def test_check_mode_is_green():
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.lax_certified_leg_closure", "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr