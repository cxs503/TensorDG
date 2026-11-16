"""Tests for the CASE-LAX-DG admissibility audit (node-level positivity, TV budget,
and whether the certified margin leans on the Zhang-Shu positivity scaling).

These tests never run the DG solver: they re-derive every committed number from
the profile files the audit itself read, and they exercise the gate semantics on
synthetic legs so that a PASS cannot be produced by code that merely executes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import lax_admissibility_audit as audit

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "benchmarks" / "results" / "lax_admissibility" / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-admissibility-audit.md"


@pytest.fixture(scope="module")
def metrics():
    assert METRICS_PATH.exists(), (
        f"missing {METRICS_PATH.relative_to(ROOT)}; regenerate with "
        "`python -m benchmarks.lax_admissibility_audit`")
    return json.loads(METRICS_PATH.read_text())


def _leg(points, density, exact_density, pressure=None, exact_pressure=None):
    """Build one audit entry through the production path, no shortcuts."""
    pts = np.asarray(points, dtype=float).reshape(-1)
    pressure = 1.0 + 0.1 * pts if pressure is None else pressure
    exact_pressure = (1.0 + 0.1 * pts if exact_pressure is None else exact_pressure)
    # velocity is a strictly increasing field on both sides so that its TV ratio
    # is well defined (1.0) and never divides by zero
    velocity = np.asarray(points, dtype=float).reshape(-1)
    numerical = {"density": density, "pressure": pressure, "velocity": velocity}
    exact = {"density": exact_density, "pressure": exact_pressure,
             "velocity": velocity.copy()}
    return audit.admissibility(points, numerical, exact)


# --------------------------------------------------------------- artifact layer
def test_doc_is_rendered_from_the_artifact(metrics):
    """The doc must be a pure function of metrics.json (zero hand-copied numbers)."""
    assert DOC.exists()
    assert DOC.read_text().strip() == audit.render(metrics).strip()


def test_every_committed_leg_recomputes_from_its_profile_file(metrics):
    """Re-derive the stored numbers from the committed profiles, bit for bit."""
    assert metrics["committed_legs"], "audit audited nothing"
    for name, entry in metrics["committed_legs"].items():
        path = ROOT / entry["source"]
        assert path.exists(), f"{name}: missing source {entry['source']}"
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        # a leg name is either "pX.NY" (flat block) or "pX_NY_dK" (prefixed block)
        if "points" in keys:
            prefix = ""
        else:
            prefix = name.split("[")[-1].rstrip("]") + "_"
            assert f"{prefix}points" in keys, f"{name}: no block {prefix}* in {entry['source']}"
        pts_key = f"{prefix}points"
        points = np.asarray(data[pts_key])
        numerical = {f: data[f"{prefix}numerical_{f}"] for f in audit.FIELDS}
        exact = {f: data[f"{prefix}exact_{f}"] for f in audit.FIELDS}
        again = audit.admissibility(points, numerical, exact)
        for key in ("min_density", "min_pressure", "exact_min_density",
                    "tv_ratio_density", "tv_ratio_velocity", "tv_ratio_pressure",
                    "nonpositive_density_nodes", "nonpositive_pressure_nodes"):
            assert again[key] == pytest.approx(entry[key], rel=0, abs=1e-12), (
                f"{name}:{key} drifted from its profile file")


def test_gate_results_are_consistent_with_the_per_leg_numbers(metrics):
    gates = metrics["gates"]
    legs = dict(metrics["committed_legs"])
    for leg, runs in metrics["fresh_runs"].items():
        for mode, entry in runs.items():
            if entry.get("status") == "ok":
                legs[f"{leg}[{mode}]"] = entry
    expect_g1 = all(entry["admissible"] for entry in legs.values())
    assert (gates["G1"]["status"] == "PASS") is expect_g1
    expect_g2 = all(
        entry["undershoot_density"] <= audit.UNDERSHOOT_BOUND
        and entry["undershoot_pressure"] <= audit.UNDERSHOOT_BOUND
        and entry["tv_ratio_density"] <= audit.TV_RATIO_BOUND
        and entry["tv_ratio_velocity"] <= audit.TV_RATIO_BOUND
        and entry["tv_ratio_pressure"] <= audit.TV_RATIO_BOUND
        for entry in legs.values())
    assert (gates["G2"]["status"] == "PASS") is expect_g2


def test_thresholds_and_tolerance_come_from_their_single_source(metrics):
    """No threshold is a private literal: config, gates and the doc must agree."""
    cfg = metrics["config"]
    assert cfg["thresholds"]["tv_ratio"] == audit.TV_RATIO_BOUND
    assert cfg["thresholds"]["undershoot"] == audit.UNDERSHOOT_BOUND
    assert cfg["thresholds"]["scaling_delta"] == audit.SCALING_DELTA_BOUND
    assert cfg["tolerance"] == pytest.approx(audit.ledger_acceptance_tolerance())
    assert metrics["gates"]["G2"]["thresholds"]["tv_ratio"] == audit.TV_RATIO_BOUND
    assert audit.TV_RATIO_BOUND == pytest.approx(1.10)
    assert audit.UNDERSHOOT_BOUND == pytest.approx(0.05)


def test_ledger_numbers_are_not_rewritten_by_this_audit(metrics):
    """The pinned FAIL and the certified PASS keep their committed values."""
    committed = metrics["committed_legs"]
    assert committed["p3.N400"]["source"].endswith("lax_case_pool/profiles.npz")
    assert committed["p2.N800"]["source"].endswith("lax_certified_configuration/profiles.npz")
    fresh = metrics.get("fresh_runs", {})
    if "p3.N400" in fresh and fresh["p3.N400"].get("scale", {}).get("status") == "ok":
        scale = fresh["p3.N400"]["scale"]
        assert scale["worst_rel_l2"] > metrics["config"]["tolerance"], (
            "p3.N400 must stay an honest FAIL; an audit may not upgrade it")
    if "p2.N800" in fresh and fresh["p2.N800"].get("scale", {}).get("status") == "ok":
        scale = fresh["p2.N800"]["scale"]
        assert scale["worst_rel_l2"] < metrics["config"]["tolerance"]


def test_certified_leg_re_run_reproduces_the_committed_number(metrics):
    """Reuse check: the fresh p2.N800 run must land on the certified 2.57e-02."""
    runs = metrics.get("fresh_runs", {}).get("p2.N800", {})
    scale = runs.get("scale")
    if not scale or scale.get("status") != "ok":
        pytest.skip("fresh runs not present in this artifact")
    certified = json.loads(
        (ROOT / "docs/benchmarks/results/lax_certified_configuration/metrics.json")
        .read_text())
    leg = next(row for row in certified["legs"]
               if row["degree"] == 2 and row["cells"] == 800)
    worst = max(leg["l2_errors"].values())
    assert scale["worst_rel_l2"] == pytest.approx(worst, rel=0.02)


# ------------------------------------------------------------- gate semantics
def test_channel_negative_node_fails_admissibility():
    points = np.linspace(0.0, 1.0, 20)
    density = np.ones(20)
    density[7] = -1e-3
    entry = _leg(points, density, np.ones(20))
    assert entry["nonpositive_density_nodes"] == 1
    assert entry["admissible"] is False
    assert audit.gate_g1({"leg": entry})["status"] == "FAIL"


def test_admissible_leg_passes_g1():
    points = np.linspace(0.0, 1.0, 20)
    entry = _leg(points, np.ones(20), np.ones(20))
    assert entry["admissible"] is True
    assert audit.gate_g1({"leg": entry})["status"] == "PASS"


def test_tv_ratio_counts_spurious_oscillation():
    points = np.linspace(0.0, 1.0, 200)
    exact = np.where(points < 0.5, 0.4, 1.0)
    smooth = exact.copy()
    smooth[100:] = smooth[100:]  # exact jump is the only variation
    wiggly = exact + 0.02 * (-1.0) ** np.arange(points.size)
    assert np.isfinite(wiggly).all()
    calm, noisy = _leg(points, smooth, exact), _leg(points, wiggly, exact)
    assert calm["tv_ratio_density"] == pytest.approx(1.0, abs=1e-12)
    assert noisy["tv_ratio_density"] > calm["tv_ratio_density"]
    assert noisy["tv_ratio_density"] > audit.TV_RATIO_BOUND
    assert audit.gate_g2({"calm": calm})["status"] == "PASS"
    assert audit.gate_g2({"noisy": noisy})["status"] == "FAIL"


def test_undershoot_is_measured_against_the_exact_floor():
    points = np.linspace(0.0, 1.0, 100)
    exact = np.where(points < 0.5, 1.0, 0.5)
    dipped = exact.copy()
    dipped[20] = 0.45                      # 10 % below the exact floor of 0.5
    entry = _leg(points, dipped, exact)
    assert entry["undershoot_density"] == pytest.approx(0.10, rel=1e-12)
    assert entry["admissible"] is True       # still positive, but beyond budget
    assert audit.gate_g2({"leg": entry})["status"] == "FAIL"


def test_g3_reports_load_bearing_instead_of_faking_a_delta():
    fresh = {"p2.N800": {"scale": {"status": "ok", "worst_rel_l2": 0.0257},
                         "none": {"status": "aborted", "detail": "nonphysical average"}}}
    gate = audit.gate_g3(fresh)
    assert gate["per_leg"]["p2.N800"]["status"] == "load_bearing"
    assert "nonphysical average" in gate["per_leg"]["p2.N800"]["detail"]


def test_g3_passes_when_disabling_the_limiter_barely_changes_the_norm():
    fresh = {"p2.N800": {"scale": {"status": "ok", "worst_rel_l2": 0.0257},
                         "none": {"status": "ok", "worst_rel_l2": 0.0261,
                                  "admissibility": {"admissible": True}}}}
    gate = audit.gate_g3(fresh)
    assert gate["per_leg"]["p2.N800"]["status"] == "PASS"


def test_g3_fails_when_the_limiter_is_worth_more_than_the_budget():
    fresh = {"p2.N800": {"scale": {"status": "ok", "worst_rel_l2": 0.0257},
                         "none": {"status": "ok", "worst_rel_l2": 0.0310,
                                  "admissibility": {"admissible": True}}}}
    gate = audit.gate_g3(fresh)
    assert gate["per_leg"]["p2.N800"]["status"] == "FAIL"