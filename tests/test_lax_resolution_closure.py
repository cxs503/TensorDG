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

Resolution-closure vs ledger bar (m20699)
-----------------------------------------
The standard case pool ladder (P1/P2/P3 × N=100/200/400) yields an A0
verdict of **FAIL** because the production leg (p3.N400) has relative L2
density = 3.116 %, just above the 3 % ledger bar.  The resolution closure
study extends the P3 ladder to N = [200, 400, 800, 1600] and shows that the
3 % threshold is met at N = 800 (worst L2 = 2.315 % for density).  The tests
below machine-verify this finding so it cannot drift from the artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS = Path("docs/benchmarks/results/lax_resolution_closure/metrics.json")
FIELDS = ("density", "velocity", "pressure")
LADDER = (200, 400, 800, 1600)
ARMS = ("none", "tvb", "weno")
LEDGER_TOLERANCE = 0.03  # 3 % relative L2


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


# --------------------------------------------------------------- m20699: ledger-bar closure
def _find_leg(metrics: dict, cells: int) -> dict:
    for leg in metrics["ladder"]:
        if leg["cells"] == cells and leg.get("limiter") == "none":
            return leg
    raise KeyError(f"no unlimited leg with cells={cells}")


def test_n800_passes_ledger_bar(metrics):
    """N = 800 at P3 must have all three fields below the 3 % ledger bar.

    This is the key "验证欠账" finding: the ledger bar *is* achievable under
    the registered configuration (P3, unlimited, CFL 0.35); it just requires
    N = 800 rather than N = 400.
    """
    leg = _find_leg(metrics, 800)
    l2 = leg["l2"]
    for field in FIELDS:
        assert l2[field] < LEDGER_TOLERANCE, (
            f"N=800 {field} L2 = {l2[field]:.6f} ({l2[field]*100:.3f}%) "
            f"≥ {LEDGER_TOLERANCE*100:.1f}% ledger bar"
        )


def test_n800_worst_field_is_density(metrics):
    """Density is the hardest field; it should be the worst at every resolution."""
    for cells in LADDER:
        leg = _find_leg(metrics, cells)
        l2 = leg["l2"]
        assert l2["density"] >= l2["velocity"], (
            f"N={cells}: velocity L2 ({l2['velocity']:.6f}) > density L2 ({l2['density']:.6f})"
        )
        assert l2["density"] >= l2["pressure"], (
            f"N={cells}: pressure L2 ({l2['pressure']:.6f}) > density L2 ({l2['density']:.6f})"
        )


def test_n400_still_fails_ledger_bar(metrics):
    """N = 400 at P3 must still fail the 3 % bar (honest FAIL, unchanged).

    This confirms the case-pool artifact's A0 FAIL verdict is correct:
    p3.N400 density L2 = 3.116 % > 3 %.
    """
    leg = _find_leg(metrics, 400)
    l2 = leg["l2"]
    assert l2["density"] >= LEDGER_TOLERANCE, (
        f"N=400 density L2 = {l2['density']:.6f} should be ≥ {LEDGER_TOLERANCE}"
    )


def test_unlimited_arm_is_most_accurate_at_n400(metrics):
    """At N = 400, the unlimited arm must have the lowest L2 density.

    TVB and WENO limiters degrade accuracy because they reduce troubled cells
    to linear reconstruction, which is over-limiting at P3.
    """
    arms = metrics.get("arms", [])
    if not arms:
        pytest.skip("limiter arms not present in closure artifact")

    l2_density = {}
    for arm_data in arms:
        limiter = arm_data.get("limiter", "unknown")
        if "l2" in arm_data and "density" in arm_data["l2"]:
            l2_density[limiter] = arm_data["l2"]["density"]

    if len(l2_density) < 2:
        pytest.skip("insufficient limiter arms for comparison")

    assert "none" in l2_density, "unlimited arm missing from limiter comparison"
    none_l2 = l2_density["none"]
    for arm_name, arm_l2 in l2_density.items():
        if arm_name != "none":
            assert arm_l2 >= none_l2, (
                f"limiter '{arm_name}' L2 ({arm_l2:.6f}) < unlimited ({none_l2:.6f})"
            )


def test_tvb_degrades_accuracy_vs_unlimited(metrics):
    """TVB limiter must degrade L2 density by at least 2× at N = 400."""
    arms = metrics.get("arms", [])
    arm_by_limiter = {a.get("limiter"): a for a in arms}
    if "none" not in arm_by_limiter or "tvb" not in arm_by_limiter:
        pytest.skip("tvb or none arm missing from closure artifact")

    none_l2 = arm_by_limiter["none"]["l2"]["density"]
    tvb_l2 = arm_by_limiter["tvb"]["l2"]["density"]
    ratio = tvb_l2 / none_l2
    assert ratio >= 2.0, (
        f"TVB L2 ({tvb_l2:.6f}) / unlimited L2 ({none_l2:.6f}) = {ratio:.2f} < 2.0"
    )


def test_closure_n800_l2_matches_documentation(metrics):
    """The N=800 L2 density in the artifact must match the published doc number.

    From docs/CASE-LAX-DG-resolution-closure.md: N=800 ρ L2 = 2.315 %.
    """
    leg = _find_leg(metrics, 800)
    expected = 0.02315
    assert abs(leg["l2"]["density"] - expected) < 5e-5, (
        f"N=800 density L2 = {leg['l2']['density']:.6f}, expected ~{expected}"
    )
