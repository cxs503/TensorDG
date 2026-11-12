"""Gates for the case-pool anchor ``CASE-DG-SOLID-BODY-ROTATION-2D``.

Audit context (m16228, verdict id=452): the previous delivery let three
acceptance assertions ``pytest.skip`` when the artifact was missing, so
``4 passed`` covered *none* of the declared criterion.  Here a missing or
unreadable artifact is a **failure**, never a skip, and a gate that was not
executed must be recorded as ``NOT_RUN`` -- it is never counted as a pass.

Artifact: ``validation/solid_body_rotation_2d/results.json`` (produced by
``benchmarks/solid_body_rotation_case_pool.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / "validation" / "solid_body_rotation_2d" / "results.json"
VALID_STATUS = {"PASS", "FAIL", "NOT_RUN"}


def _load() -> dict:
    assert ARTIFACT.exists(), (
        f"missing case-pool artifact {ARTIFACT}; run "
        "`PYTHONPATH=. python3 benchmarks/solid_body_rotation_case_pool.py` "
        "(a missing artifact is a hard failure, not a skip)")
    data = json.loads(ARTIFACT.read_text())
    assert data.get("runs"), "artifact carries no runs"
    return data


def test_artifact_is_provenanced():
    data = _load()
    assert data["anchor"] == "CASE-DG-SOLID-BODY-ROTATION-2D"
    prov = data["provenance"]
    assert prov["git_commit"] and prov["git_commit"] != "unknown"
    assert prov["dtype"] == "float64"
    assert len(prov["git_commit"]) >= 7
    case = data["case"]
    assert case["declared_config"] == [100, 2]
    assert case["final_time"] == 1.0
    assert case["disk_radius"] == 0.45


def test_every_gate_has_an_explicit_status():
    data = _load()
    gates = data["gate_results"]
    assert set(gates) == set(data["gates_preregistered"]), "gate set drift"
    for name, g in gates.items():
        assert g["status"] in VALID_STATUS, (name, g["status"])
        if g["status"] != "NOT_RUN":
            assert "got" in g, f"{name} ran but recorded no measured value"


def test_no_not_run_gate_is_reported_as_a_pass():
    """The verdict flags must not turn unexecuted gates into passes (m16228 F1)."""
    data = _load()
    verdict = data["verdict"]
    not_run = [k for k, g in data["gate_results"].items() if g["status"] == "NOT_RUN"]
    assert verdict["any_not_run"] == bool(not_run)
    if not_run:
        assert verdict["all_gates_pass"] is False, not_run
    if verdict["measured_gates_pass"]:
        assert all(g["status"] == "PASS" for k, g in data["gate_results"].items()
                   if g["status"] != "NOT_RUN")


def test_measured_gates_pass():
    data = _load()
    failures = {k: g for k, g in data["gate_results"].items() if g["status"] == "FAIL"}
    assert not failures, failures


def test_negative_control_coarse_p1_misses_the_five_percent_line():
    data = _load()
    rec = next(r for r in data["runs"] if (r["ncells"], r["degree"]) == (16, 1))
    assert rec["metrics"]["relative_l1_disk"] > 0.05, rec["metrics"]
    assert data["gate_results"]["G7_NEGCTRL"]["status"] == "PASS"


def test_mass_is_conserved_to_round_off():
    data = _load()
    worst = max(r["metrics"]["mass_drift_rel"] for r in data["runs"])
    assert worst < 1e-12, worst


def test_disk_norm_coverage_is_recorded_and_full():
    """The bell trajectory (annulus r in [0.10,0.40]) must be inside the disk."""
    data = _load()
    for rec in data["runs"]:
        cov = rec["metrics"]["coverage"]
        assert cov is not None and cov == pytest.approx(1.0, abs=1e-12), (rec["ncells"], cov)


def test_global_norm_exceeds_disk_norm_everywhere():
    """Documents the seam pollution instead of hiding it behind the mask."""
    data = _load()
    for rec in data["runs"]:
        m = rec["metrics"]
        assert m["relative_l1_global"] >= m["relative_l1_disk"] - 1e-15, m


def test_convergence_is_monotone_in_the_measured_ladder():
    data = _load()
    for deg in {r["degree"] for r in data["runs"]}:
        pts = sorted((r["ncells"], r["metrics"]["relative_l1_disk"])
                     for r in data["runs"] if r["degree"] == deg)
        for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
            assert e1 < e0, f"P{deg}: error grew from n={n0} to n={n1}"


def test_p1_observed_order_is_at_least_design_minus_slack():
    data = _load()
    orders = data["orders"]["1"]
    slope = orders.get("fit_slope")
    if slope is None:
        pytest.fail("P1 ladder has too few points to estimate an order")
    assert slope >= 1.2, orders


def test_declared_case_is_measured_or_explicitly_pending():
    """G1 either has a number, or the artifact must say NOT_RUN (never silent)."""
    data = _load()
    got = data["gate_results"]["G1_DECLARED"]
    if got["status"] == "NOT_RUN":
        assert any((r["ncells"], r["degree"]) != (100, 2) for r in data["runs"])
        assert data["verdict"]["all_gates_pass"] is False
    else:
        assert math.isfinite(got["got"])


def test_artifact_is_reproducible_from_the_committed_code():
    """Re-run the cheapest configuration in-process and match the artifact."""
    data = _load()
    ref = next(r for r in data["runs"] if (r["ncells"], r["degree"]) == (16, 1))
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.solid_body_rotation_case_pool import _norm_pair

    again = _norm_pair(16, 1)
    assert again["steps"] == ref["steps"]
    assert again["metrics"]["relative_l1_disk"] == pytest.approx(
        ref["metrics"]["relative_l1_disk"], rel=1e-12)
    assert again["metrics"]["relative_l1_global"] == pytest.approx(
        ref["metrics"]["relative_l1_global"], rel=1e-12)