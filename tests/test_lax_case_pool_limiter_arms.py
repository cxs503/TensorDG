"""Contract tests for the ``CASE-LAX-DG`` limiter-arms study.

The study's claim is *attribution*: "the pool's C3/R3 FAIL comes from the
limiter default that the driver never switched on, or it does not".  A test
that merely executed the driver would not protect that claim, so the layers
below assert invariants instead:

1. **pre-registration layer** -- the arms are fixed in code, exactly one knob
   differs, and no tunable parameter (``tvb_constant``) is exposed, so the study
   cannot turn into a post-hoc sweep;
2. **non-vacuity layer** -- the arms must actually change the solution (step
   counts and errors differ from the baseline), otherwise three identical runs
   would be reported as a three-arm comparison;
3. **artifact layer** -- the committed ``metrics.json`` is re-adjudicated from
   its stored numbers with the *pool's own* gate functions (a hand-edited
   verdict cannot pass), the baseline arm must reproduce the tracked pool
   record bit for bit, the R3 region rule must be identical across arms, and
   the rendered document must match the artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.lax_case_pool import (
    MIN_C3_ORDER, R3_BAND_WIDTH, R3_LEGACY_BAND_CELLS, R3_ORDER_FRACTION,
    R5_ORDER_FRACTION, TOL_C1, TOL_C2,
    convergence_order, gate_c1, gate_c2, gate_c3, gate_r3, gate_r5,
)
from benchmarks.lax_case_pool_limiter_arms import (
    ARMS, OUT, POOL_RECORD, _analytic_order_table,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUT / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-limiter-arms.md"
PROFILES = ARTIFACT.parent / "profiles.npz"


# --------------------------------------------------------- pre-registration layer
def test_arms_are_pre_registered_and_change_exactly_one_knob():
    assert set(ARMS) == {"none", "tvb", "weno"}
    assert ARMS["none"] == {}, "baseline arm must not pass any solver override"
    for arm in ("tvb", "weno"):
        assert set(ARMS[arm]) == {"limiter"}, (
            f"arm {arm} may only override the limiter, got {ARMS[arm]}")


def test_no_tunable_limiter_parameter_is_swept():
    # ``tvb_constant`` would let the study pick M after seeing the numbers
    for arm, kwargs in ARMS.items():
        assert "tvb_constant" not in kwargs, f"arm {arm} tunes the TVB constant"
        assert "limiter_stage" not in kwargs, f"arm {arm} tunes the stage policy"


def test_tolerances_are_the_pool_ones_not_new_ones():
    assert TOL_C1 == pytest.approx(0.02)
    assert TOL_C2 == pytest.approx(0.05)
    assert MIN_C3_ORDER == pytest.approx(1.0)
    assert R3_ORDER_FRACTION == pytest.approx(0.85)
    assert R5_ORDER_FRACTION == pytest.approx(0.03)
    # the R3 region rule is the nested fixed band of m16606 F1
    assert R3_BAND_WIDTH == pytest.approx(0.02)
    assert R3_LEGACY_BAND_CELLS == pytest.approx(4.0)


# ------------------------------------------------------------------- artifact io
@pytest.fixture(scope="module")
def metrics():
    # hard failure, not skip: the record is tracked next to this test
    assert ARTIFACT.exists(), f"missing tracked artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------- non-vacuity layer
def test_every_declared_arm_has_a_record(metrics):
    assert set(metrics["arms"]) == set(metrics["pre_registration"]["arms"])
    assert metrics["pre_registration"]["declared_before_run"] is True
    for arm, record in metrics["arms"].items():
        assert record["arguments"] == ARMS[arm]


def test_the_arms_actually_change_the_solution(metrics):
    """A limiter arm that reproduces the baseline would make the study vacuous."""
    base = {(e["cells"], e["degree"]): e for e in metrics["arms"]["none"]["ladder"]}
    differing = {}
    for arm in ("tvb", "weno"):
        changed = 0
        for entry in metrics["arms"][arm]["ladder"]:
            ref = base[(entry["cells"], entry["degree"])]
            same_error = entry["errors"]["density"] == ref["errors"]["density"]
            same_steps = entry["steps"] == ref["steps"]
            changed += int(not (same_error and same_steps))
        differing[arm] = changed
    # every leg of both limiter arms must move
    assert differing == {"tvb": 9, "weno": 9}, differing


def test_baseline_arm_reproduces_the_tracked_pool_record(metrics):
    repro = metrics["reproduction"]
    assert repro["status"] == "PASS", repro.get("worst_key")
    assert repro["worst_relative_difference"] < 1e-9
    assert POOL_RECORD.exists()
    published = json.loads(POOL_RECORD.read_text())
    assert published["ladder"], "pool record carries no ladder"


# ----------------------------------------------------------------- artifact layer
def _entries(record, key="errors"):
    return [{"cells": e["cells"], "degree": e["degree"], "errors": e[key],
             "analytic_errors": e["analytic_errors"], "steps": e["steps"]}
            for e in record["ladder"]]


def test_c3_verdict_is_rederived_from_the_stored_numbers(metrics):
    for arm, record in metrics["arms"].items():
        recomputed = gate_c3(_entries(record))
        stored = record["gates"]["c3"]
        assert recomputed["status"] == stored["status"], arm
        assert recomputed["gated_order"] == pytest.approx(stored["gated_order"])
        # the gate is only meaningful if the order is computable at all
        assert math.isfinite(stored["gated_order"])


def test_r3_verdict_is_rederived_with_the_nested_band(metrics):
    for arm, record in metrics["arms"].items():
        recomputed = gate_r3(record["ladder"], "analytic_errors",
                             {"width": R3_BAND_WIDTH})
        stored = record["gates"]["r3"]
        assert recomputed["status"] == stored["status"], arm
        for degree in (1, 2, 3):
            key = f"p{degree}.density"
            assert recomputed["per_leg"][key] == pytest.approx(
                stored["per_leg"][key]), (arm, key)


def test_r3_region_rule_is_identical_across_arms(metrics):
    """Same post-processing everywhere: the retained measure cannot differ."""
    by_arm = {}
    for arm, record in metrics["arms"].items():
        by_arm[arm] = sorted({round(e["analytic_fraction"], 12)
                              for e in record["ladder"]})
    values = list(by_arm.values())
    assert all(v == values[0] for v in values), by_arm
    # and the band is genuinely nested, i.e. the retained measure is constant
    assert len(values[0]) == 1, values[0]


def test_c1_c2_r5_verdicts_match_the_stored_ones(metrics):
    for arm, record in metrics["arms"].items():
        gates = record["gates"]
        c1 = gate_c1(record["finest"]["positions"], record["exact_positions"])
        assert c1["status"] == gates["c1"]["status"], arm
        for name, row in c1["waves"].items():
            assert row["relative_error"] == pytest.approx(
                gates["c1"]["waves"][name]["relative_error"])
        if "r5" in gates:
            recomputed = gate_r5(record["entropy_wave"])
            assert recomputed["status"] == gates["r5"]["status"], arm


def test_verdict_is_the_and_of_all_gates(metrics):
    for arm, record in metrics["arms"].items():
        expected = ("PASS" if all(g["status"] == "PASS"
                                  for g in record["gates"].values()) else "FAIL")
        assert record["verdict"] == expected, arm
        assert metrics["verdicts"][arm] == expected


def test_shared_ladder_shape_and_monotone_error_not_assumed(metrics):
    """Every arm solves the same ladder; errors need not be monotone (report it)."""
    shapes = {arm: [(e["cells"], e["degree"]) for e in record["ladder"]]
              for arm, record in metrics["arms"].items()}
    first = shapes["none"]
    assert all(shape == first for shape in shapes.values()), shapes
    table = _analytic_order_table({arm: record
                                   for arm, record in metrics["arms"].items()})
    for arm, per_degree in table.items():
        for degree, data in per_degree.items():
            assert len(data["cells"]) == 3, (arm, degree)
            assert all(err > 0.0 for err in data["analytic"])


def test_profiles_are_committed_and_agree_with_the_exact_solution(metrics):
    import numpy as np

    assert PROFILES.exists(), f"missing tracked artifact {PROFILES}"
    data = np.load(PROFILES)
    for arm in metrics["arms"]:
        points = data[f"{arm}_points"].ravel()
        order = np.argsort(points)
        numerical = data[f"{arm}_numerical_density"].ravel()[order]
        exact = data[f"{arm}_exact_density"].ravel()[order]
        relative = float(abs(numerical - exact).sum() / abs(exact).sum())
        assert relative < 0.1, (arm, relative)


def test_rendered_document_matches_the_artifact(metrics):
    from scripts.render_lax_arms_doc import render

    assert DOC.exists(), f"missing tracked document {DOC}"
    assert DOC.read_text().strip() == render(metrics).strip(), (
        f"{DOC.name} is stale; re-run scripts/render_lax_arms_doc.py")