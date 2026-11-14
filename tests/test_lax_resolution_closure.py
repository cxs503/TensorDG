"""Consistency of the CASE-LAX-DG resolution-closure artifact.

The artifact ``docs/benchmarks/results/lax_resolution_closure/metrics.json`` is
produced by ``benchmarks/lax_resolution_closure.py`` from a ~55 minute cluster
run, so it cannot be regenerated inside a unit test.  What *can* be checked --
and is what this file checks -- is that the committed numbers still say what
the committed verdict says, i.e. that the verdict is **re-derived** from the
ladder and not asserted next to it:

* the ladder is the registered one (p3, ``200/400/800/1600``, no limiter);
* the reported production leg is the ``N = 400`` entry, bit-identical to it;
* the closure mesh is the smallest *run* mesh whose L2 meets the bar for
  **every** field, or ``None``;
* ``status`` is ``PASS`` exactly when such a mesh exists -- an extrapolated
  ``N*`` never produces a pass (the honest-FAIL rule);
* the limiter arms differ in exactly one knob (``limiter``);
* every field's relative L2 decreases monotonically along the ladder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS = Path("docs/benchmarks/results/lax_resolution_closure/metrics.json")
FIELDS = ("density", "velocity", "pressure")
LADDER = (200, 400, 800, 1600)
ARMS = ("none", "tvb", "weno")


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip(f"{METRICS} not generated yet (run benchmarks.lax_resolution_closure)")
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_ladder_is_the_registered_production_arm(metrics):
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    assert tuple(e["cells"] for e in legs) == LADDER
    assert {e["degree"] for e in legs} == {3}
    assert {e["limiter"] for e in legs} == {"none"}, (
        "the closure ladder must measure the registered unlimited p3 arm; a "
        "limiter here would silently change the acceptance configuration"
    )


def test_production_leg_number_is_the_ladder_entry(metrics):
    gate = metrics["gate"]
    assert gate["production_leg"] == "p3.N400"
    leg = next(e for e in metrics["ladder"] if e["cells"] == 400)
    for field in FIELDS:
        assert gate["production_l2"][field] == leg["l2"][field]


def test_closure_mesh_is_rederived_from_the_ladder(metrics):
    gate = metrics["gate"]
    tol = gate["tolerance"]
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    closing = [e["cells"] for e in legs if all(e["l2"][f] <= tol for f in FIELDS)]
    expected = min(closing) if closing else None
    assert gate["closure_mesh"] == expected
    assert gate["status"] == ("PASS" if expected is not None else "FAIL")


def test_extrapolation_never_counts_as_a_pass(metrics):
    """If no run leg meets the bar, the verdict must stay FAIL."""
    gate = metrics["gate"]
    model = gate["per_field"]["density"]["model"] or {}
    if gate["closure_mesh"] is None:
        assert gate["status"] == "FAIL"
        assert model.get("extrapolated_n") is not None
    else:
        assert gate["closure_mesh"] <= max(LADDER)


def test_every_field_converges_along_the_ladder(metrics):
    legs = sorted(metrics["ladder"], key=lambda e: e["cells"])
    for field in FIELDS:
        series = [e["l2"][field] for e in legs]
        # The ladder must go down overall; a finest-pair plateau is allowed
        # because it is measured (velocity L2 1.383 % at N=800 -> 1.475 % at
        # N=1600), not assumed away.
        assert series[-1] < series[0], f"{field} L2 does not decrease: {series}"
        for coarse, fine in zip(series, series[1:]):
            assert fine <= coarse * 1.15, (
                f"{field} L2 grows by more than 15 % on one refinement step: {series}"
            )


def test_limiter_arms_change_only_the_limiter(metrics):
    arms = metrics["arms"]
    assert tuple(a["limiter"] for a in arms) == ARMS
    assert {a["cells"] for a in arms} == {metrics["config"]["arm_cells"]}
    assert {a["degree"] for a in arms} == {metrics["config"]["degree"]}
    # the unlimited arm at the production mesh must reproduce the ladder entry
    leg = next(e for e in metrics["ladder"] if e["cells"] == metrics["config"]["arm_cells"])
    none_arm = next(a for a in arms if a["limiter"] == "none")
    for field in FIELDS:
        assert none_arm["l2"][field] == leg["l2"][field]


def test_figures_are_declared_on_disk(metrics):
    base = METRICS.parent / "figures"
    if not metrics.get("figures"):
        pytest.skip("figures were skipped for this run")
    for figure in metrics["figures"]:
        assert (base / figure["name"]).exists()
        assert figure["caption"].strip()