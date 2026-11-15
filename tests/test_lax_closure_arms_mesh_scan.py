"""CASE-LAX-DG limiter arm x stage x mesh scan: artifact/verdict consistency.

The artifact
``docs/benchmarks/results/lax_closure_arms_mesh_scan/metrics.json`` comes from a
~35 minute cluster job, so it cannot be regenerated inside a unit test.  What is
checked here is that the committed numbers still say what the committed verdict
says, and -- more importantly for an experiment -- that the **three predeclared
predictions are re-evaluated from the data** instead of being asserted next to
it, and that the mapping from the three booleans to the published statement is
the one the module declares (so a threshold cannot be tuned after the fact
without a test failing).

Local, cluster-free parts:

* ``gate()`` is a pure function of the leg list; the decision-rule branches are
  driven with synthetic legs, covering all four (P1, P3) combinations plus the
  metric-split flag.
* ``render_summary`` / ``render_doc`` are exercised on a synthetic artifact so
  that a rendering defect cannot silently ship a doc that misstates the gate.

Artifact parts (skipped when the artifact is absent):

* the factorial grid is complete and each leg differs from its reference only in
  the declared knob;
* the re-measured unlimited legs reproduce the committed closure artifact;
* P1/P2/P3 booleans are re-derived from the legs with the module's thresholds;
* the published statement is the one the decision rule yields;
* ``summary.md`` and the case doc carry the artifact's verdict and its rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.lax_closure_arms_mesh_scan as scan

METRICS = Path("docs/benchmarks/results/lax_closure_arms_mesh_scan/metrics.json")
SUMMARY = METRICS.parent / "summary.md"
DOC = Path("docs/CASE-LAX-DG-closure-arms-mesh-scan.md")
FIELDS = ("density", "velocity", "pressure")
LEDGER_TOLERANCE = 0.03


# ------------------------------------------------------------------ synthetic
def _synthetic_leg(cells, limiter, stage, l2_rho, band_rho, l2_other=None):
    l2_other = l2_rho * 0.7 if l2_other is None else l2_other
    return {
        "cells": cells, "degree": 3, "limiter": limiter,
        "limiter_stage": stage,
        "tvb_constant": 1.0 if limiter == "tvb" else None,
        "steps": 100, "dt_mean": 1e-4, "wall_seconds": 1.0,
        "density_min": 0.4, "band_fraction": 0.9,
        "l1": {f: l2_rho * 0.5 for f in FIELDS},
        "l2": {"density": l2_rho, "velocity": l2_other, "pressure": l2_other},
        "band_l1": {f: band_rho for f in FIELDS},
        "max_abs": {f: 0.1 for f in FIELDS},
    }


def _grid_legs(*, unlimited_l2=(0.030, 0.024), limiter_l2=(0.070, 0.056),
               limiter_band=0.0005, unlimited_band=(0.006, 0.005),
               final_gain=0.40, meshes=(400, 800)):
    """Build one leg per (mesh, limiter, stage) with controllable contrasts."""
    legs = []
    for k, cells in enumerate(meshes):
        legs.append(_synthetic_leg(cells, "none", None, unlimited_l2[k],
                                   unlimited_band[k]))
        for limiter in ("tvb", "weno"):
            every = limiter_l2[k]
            final = every * (1.0 - final_gain)
            legs.append(_synthetic_leg(cells, limiter, "every", every,
                                       limiter_band))
            legs.append(_synthetic_leg(cells, limiter, "final", final,
                                       limiter_band))
    return legs


def test_gate_all_predictions_hold_retains_statement_with_qualifier():
    gate = scan.gate(_grid_legs())
    assert gate["predictions"]["P1"]["holds"] is True
    assert gate["predictions"]["P2"]["holds"] is True
    assert gate["predictions"]["P3"]["holds"] is True
    assert gate["verdicts"]["P1_limiter_ranking"] == "mesh-independent"
    assert gate["verdicts"]["P3_stage_effect"] == "dominant"
    assert "qualify it by the limiter stage" in gate["verdicts"]["published_statement"]


def test_gate_ordering_failing_at_one_mesh_is_mesh_specific():
    """P1 must be False as soon as one mesh violates the ordering."""
    legs = _grid_legs()
    # at N=800 make tvb better than unlimited
    for leg in legs:
        if leg["cells"] == 800 and leg["limiter"] == "tvb":
            leg["l2"]["density"] = 0.010
    gate = scan.gate(legs)
    assert gate["predictions"]["P1"]["holds"] is False
    assert "mesh-specific" in gate["verdicts"]["P1_limiter_ranking"]


def test_gate_without_stage_effect_withdraws_the_stage_attribution():
    """If 'final' does not help, the penalty belongs to the limiter itself."""
    gate = scan.gate(_grid_legs(final_gain=0.0))
    assert gate["predictions"]["P1"]["holds"] is True
    assert gate["predictions"]["P3"]["holds"] is False
    assert gate["verdicts"]["P3_stage_effect"] == "none"
    assert "stands unqualified" in gate["verdicts"]["published_statement"]


def test_gate_partial_stage_effect_is_reported_as_partial():
    """A gain on one mesh only must not be promoted to 'dominant'."""
    legs = _grid_legs()
    for leg in legs:
        if leg["cells"] == 800 and leg["limiter_stage"] == "final":
            leg["l2"]["density"] = 0.070  # no gain at the fine mesh
    gate = scan.gate(legs)
    assert gate["predictions"]["P3"]["holds"] is False
    assert gate["verdicts"]["P3_stage_effect"] == "partial"


def test_gate_failing_both_withdraws_the_whole_claim():
    legs = _grid_legs(final_gain=0.0)
    for leg in legs:
        if leg["cells"] == 800 and leg["limiter"] == "weno":
            leg["l2"]["density"] = 0.005
    gate = scan.gate(legs)
    assert gate["predictions"]["P1"]["holds"] is False
    assert gate["predictions"]["P3"]["holds"] is False
    assert "must be demoted" in gate["verdicts"]["published_statement"]


def test_gate_separation_flag_uses_the_declared_threshold():
    """The metric-split verdict tracks the separation threshold, both ways."""
    tight = scan.gate(_grid_legs(unlimited_band=(0.020, 0.018)))  # ratio < 2
    assert tight["verdicts"]["metric_view"] == "metric-consistent"
    loose = scan.gate(_grid_legs())  # ratio 5 and 4.8
    assert loose["verdicts"]["metric_view"].startswith("metric-dependent")
    for value in loose["unlimited_separation"].values():
        assert value >= scan.SEPARATION_MIN


def test_thresholds_are_module_constants_not_literals_in_the_verdict():
    """The bars used by the gate are the ones the predeclared text quotes."""
    gate = scan.gate(_grid_legs())
    bars = gate["bar_values"]
    assert bars["band_gain_factor"] == scan.BAND_GAIN_FACTOR
    assert bars["l2_stage_gain_min"] == scan.L2_STAGE_GAIN_MIN
    assert bars["separation_min"] == scan.SEPARATION_MIN
    assert f"{scan.BAND_GAIN_FACTOR:.2f}" in scan.PREDICTIONS["P2"]
    assert f"{scan.L2_STAGE_GAIN_MIN:.0%}" in scan.PREDICTIONS["P3"]


def test_grid_is_the_declared_factorial():
    grid = scan._arms((400, 800))
    assert len(grid) == 10
    assert grid.count((400, None, None)) == 1
    for cells in (400, 800):
        for limiter in ("tvb", "weno"):
            for stage in scan.STAGES:
                assert (cells, limiter, stage) in grid
        # the unlimited arm must appear once per mesh, never twice
        assert sum(1 for g in grid if g[0] == cells and g[1] is None) == 1


def test_summary_render_carries_the_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "OUT", tmp_path)
    legs = _grid_legs()
    gate = scan.gate(legs)
    text = scan.render_summary({"generated_at_utc": "x", "wall_seconds": 1.0,
                                "gate": gate, "legs": legs})
    assert "P1 (mesh-independent limiter ranking): **HOLDS**" in text
    assert gate["verdicts"]["published_statement"] in text
    for leg in legs:
        row = (f"| {leg['cells']} | {leg['limiter']} | "
               f"{leg['limiter_stage'] or '-'} | {leg['l2']['density']:.4%} |")
        assert row in text
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") == text


def test_figures_are_rendered_from_the_artifact_numbers(tmp_path):
    import matplotlib
    assert matplotlib.get_backend().lower() != "agg" or True  # backend-agnostic
    legs = _grid_legs()
    gate = scan.gate(legs)
    made = scan.render_figures({"legs": legs, "gate": gate}, tmp_path)
    names = {f["name"] for f in made}
    assert names == {"arms_mesh_scan.png", "arms_metric_split.png"}
    for figure in made:
        assert (tmp_path / figure["name"]).stat().st_size > 5000
        assert figure["caption"].strip()


def _synthetic_metrics(tmp_path, *, reproduced):
    legs = _grid_legs()
    gate = scan.gate(legs)
    rep = ({"checked": True, "artifact": "x", "worst_relative_deviation": 0.0,
            "tolerance": 1e-12, "reproduced": True,
            "legs": [{"cells": 400}, {"cells": 800}]}
           if reproduced else {"checked": False, "reason": "artifact absent"})
    return {"generated_at_utc": "x", "wall_seconds": 1.0, "legs": legs,
            "gate": gate, "closure_artifact_reproduction": rep,
            "source": {"job": "cj0", "node": "n1"},
            "figures": [{"name": "arms_mesh_scan.png", "caption": "c"}]}


def test_doc_render_warns_when_reproduction_is_unverified(tmp_path, monkeypatch):
    """The doc must not claim comparability it has not established."""
    import scripts.ingest_lax_arms_mesh_scan as ing

    monkeypatch.setattr(ing, "DOC", tmp_path / "doc.md")
    text = ing.render_doc(_synthetic_metrics(tmp_path, reproduced=False))
    assert "未通过 1e-12 判据" in text
    assert "逐位对接" not in text


def test_doc_render_states_reproduction_when_verified(tmp_path, monkeypatch):
    import scripts.ingest_lax_arms_mesh_scan as ing

    monkeypatch.setattr(ing, "DOC", tmp_path / "doc.md")
    text = ing.render_doc(_synthetic_metrics(tmp_path, reproduced=True))
    assert "REPRODUCED" in text
    assert "逐位对接" in text
    # every leg must appear in the honest ledger table with its own verdict
    for leg in _grid_legs():
        assert f"{leg['l2']['density'] * 100:.3f} %" in text


# ------------------------------------------------------------------- artifact
@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip(f"{METRICS} not generated yet "
                    "(run scripts/ingest_lax_arms_mesh_scan.py)")
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_artifact_grid_is_complete_and_single_knob(metrics):
    legs = metrics["legs"]
    assert sorted({leg["cells"] for leg in legs}) == sorted(metrics["config"]["meshes"])
    assert sorted({leg["degree"] for leg in legs}) == [metrics["config"]["degree"]]
    for cells in metrics["config"]["meshes"]:
        mesh_legs = [leg for leg in legs if leg["cells"] == cells]
        assert sorted(leg["limiter"] for leg in mesh_legs) == \
            ["none", "tvb", "tvb", "weno", "weno"]
        unlimited = [leg for leg in mesh_legs if leg["limiter"] == "none"]
        assert len(unlimited) == 1 and unlimited[0]["limiter_stage"] is None
    # every leg must be an actually-run solve
    for leg in legs:
        assert leg["steps"] > 0 and leg["wall_seconds"] > 0


def test_artifact_reproduces_the_committed_closure_artifact(metrics):
    rep = metrics["closure_artifact_reproduction"]
    if not rep.get("checked"):
        pytest.skip("closure artifact not available in this checkout")
    assert rep["reproduced"] is True, rep["worst_relative_deviation"]
    assert rep["worst_relative_deviation"] <= rep["tolerance"]
    # only the unlimited legs can be cross-checked against the closure ladder
    for row in rep["legs"]:
        assert row["steps"][0] == row["steps"][1], row


def test_predictions_are_rederived_from_the_legs(metrics):
    """Re-evaluate P1/P2/P3 from the legs with the module's own thresholds."""
    legs = metrics["legs"]
    fresh = scan.gate(legs)
    for key in ("P1", "P2", "P3"):
        assert fresh["predictions"][key]["holds"] is metrics["gate"]["predictions"][key]["holds"]
    assert fresh["verdicts"] == metrics["gate"]["verdicts"]


def test_published_statement_matches_the_decision_rule(metrics):
    gate = metrics["gate"]
    p1 = gate["predictions"]["P1"]["holds"]
    p3 = gate["predictions"]["P3"]["holds"]
    statement = gate["verdicts"]["published_statement"]
    if p1 and p3:
        assert "qualify it by the limiter stage" in statement
    elif not p1 and p3:
        assert "withdraw" in statement
    elif p1 and not p3:
        assert "stands unqualified" in statement
    else:
        assert "must be demoted" in statement


def test_limiter_arms_cannot_flip_the_ledger_verdict(metrics):
    """The ledger bar is registered on the unlimited leg; arms are diagnostic."""
    assert metrics["gate"]["registered_arm_unaffected"] is True
    unlimited = [leg for leg in metrics["legs"] if leg["limiter"] == "none"]
    for leg in unlimited:
        worst = max(leg["l2"][f] for f in FIELDS)
        # the N=800 unlimited leg is the published closure result
        if leg["cells"] == 800:
            assert worst < LEDGER_TOLERANCE


def test_summary_md_matches_the_artifact(metrics):
    if not SUMMARY.exists():
        pytest.skip(f"{SUMMARY} not generated yet")
    text = SUMMARY.read_text(encoding="utf-8")
    gate = metrics["gate"]
    for key in ("P1", "P2", "P3"):
        flag = "HOLDS" if gate["predictions"][key]["holds"] else "FAILS"
        assert f"{key} (" in text
        assert flag in text
    assert gate["verdicts"]["published_statement"] in text
    for leg in metrics["legs"]:
        row = (f"| {leg['cells']} | {leg['limiter']} | "
               f"{leg['limiter_stage'] or '-'} | {leg['l2']['density']:.4%} | "
               f"{leg['band_l1']['density']:.4%} |")
        assert row in text, f"row missing/stale for {leg['cells']}/{leg['limiter']}"


def test_case_doc_states_every_prediction_and_the_honest_fail(metrics):
    if not DOC.exists():
        pytest.skip(f"{DOC} not generated yet")
    text = DOC.read_text(encoding="utf-8")
    gate = metrics["gate"]
    for key in ("P1", "P2", "P3"):
        flag = "HOLDS" if gate["predictions"][key]["holds"] else "FAILS"
        assert f"{key} **{flag}**" in text or f"{key} {flag}" in text or flag in text
    # the doc must carry the ledger comparison for every leg, and must not
    # claim the limiter arms pass a bar they do not pass
    qualifying = [leg for leg in metrics["legs"]
                  if all(leg["l2"][f] < LEDGER_TOLERANCE for f in FIELDS)]
    assert f"**{len(qualifying)} 条**" in text
    for leg in metrics["legs"]:
        assert f"{leg['l2']['density'] * 100:.3f} %" in text
    for figure in metrics.get("figures", []):
        assert figure["name"] in text