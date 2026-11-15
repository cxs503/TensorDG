"""CASE-LAX-DG strict-invariance audit: machine checks on the committed artifact.

Every number is either read out of `docs/benchmarks/results/lax_invariance_audit/`
or re-measured live through the operator arms, so this file is a regression guard
for the *audit's own* claims (not a re-run of the audit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import lax_invariance_audit as aud

METRICS = aud.METRICS
FIG = aud.FIG
FIELDS = aud.FIELDS


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():
        pytest.skip(f"missing artifact {METRICS}; run python -m benchmarks.lax_invariance_audit")
    return json.loads(METRICS.read_text())


def _arm(metrics, leg, name):
    return next(a for a in metrics["arms"] if a["leg"] == leg and a["arm"] == name)


def test_operator_invariants_hold_live():
    """O1-O5 are closed-form identities: re-measure them in this process."""
    rows = aud.operator_rows()
    assert [r["id"] for r in rows] == ["O1", "O2", "O3", "O4", "O5"]
    for row in rows:
        assert row["defect"] <= aud.INVARIANT_TOL, (row["id"], row["defect"])
        assert row["pass"] is True


def test_mirror_parity_is_at_round_off(metrics):
    """T1: the mirrored problem reproduces the mirrored solution on every leg."""
    rows = [a for a in metrics["arms"] if a["arm"] == "mirror"]
    assert rows, "no mirror arm in the artifact"
    for row in rows:
        assert row["worst_deviation"] <= aud.INVARIANT_TOL, (row["leg"], row["worst_deviation"])
        assert set(row["deviation"]) == set(FIELDS)


def test_ledger_metric_is_frame_invariant(metrics):
    """T2: the registered metric is the same number in both frames."""
    for row in metrics["verdict"]["metric_parity"]:
        assert row["rel_diff"] <= aud.INVARIANT_TOL, row
        assert row["base"] == pytest.approx(row["mirror"], rel=aud.INVARIANT_TOL)


def test_step_and_admissibility_parity(metrics):
    """T3/T4: same step sequence and identical positivity statistics."""
    for row in metrics["verdict"]["step_parity"]:
        assert row["same_steps"], row
        assert row["dt_mean_rel_diff"] <= aud.INVARIANT_TOL, row
        assert row["positivity_equal"], row


def test_control_arms_have_discriminating_power(metrics):
    """The audit must be able to *fail*: both control arms deviate loudly."""
    controls = [a for a in metrics["arms"] if a["arm"].startswith("control_")]
    assert len(controls) >= 2
    for row in controls:
        assert row["worst_deviation"] >= aud.CONTROL_FLOOR, (row["arm"], row["worst_deviation"])
    assert metrics["verdict"]["discrimination"] >= aud.DISCRIMINATION_MIN


def test_verdict_is_pass(metrics):
    verdict = metrics["verdict"]
    assert verdict["invariants_ok"] is True
    assert verdict["controls_ok"] is True
    assert verdict["status"] == "PASS"
    assert metrics["reuse_check"]["status"] in ("PASS", "not_run")


def test_pinned_leg_reproduces_the_committed_pool_number(metrics):
    reuse = metrics["reuse_check"]
    if reuse["status"] == "not_run":
        pytest.skip(reuse.get("reason", "reuse check not run"))
    assert reuse["worst_rel_diff"] <= reuse["tolerance"]
    for field, row in reuse["per_field"].items():
        assert row["measured"] == pytest.approx(row["committed"], rel=reuse["tolerance"])


def test_pinned_leg_still_misses_the_ledger_bar(metrics):
    """honest FAIL: the audit must not have moved the ledger bar."""
    bias = metrics["bias_bound"]
    assert bias["pinned_verdict"] == "FAIL"
    assert bias["pinned_density_rel_l2"] > bias["tolerance"]

    # and the direction bound is tiny compared with the bar
    assert bias["bias_in_units_of_bar"] <= 1e-3


def test_bias_bound_is_documented_against_the_other_resolution_artifacts(metrics):
    """The audit's conclusion must point at the resolution explanation."""
    doc = aud.DOC.read_text()
    assert "resolution-requirement" in doc
    assert "FAIL" in doc
    assert "是否 <3%：否" in doc


def test_figures_are_committed(metrics):
    names = [f["name"] for f in metrics["figures"]]
    assert len(names) == 3
    for name in names:
        assert (FIG / name).exists() and (FIG / name).stat().st_size > 1024


def test_artifact_carries_the_design_and_norm_contract(metrics):
    design = metrics["design"]["strict_invariants"]
    assert set(design) == {"O1", "O2", "O3", "O4", "O5", "T1", "T2", "T3", "T4"}
    assert "not_checked" in metrics["design"]
    assert "hllc_galilean" in metrics["design"]["not_checked"]
    assert "never elementwise" in metrics["config"]["norm"]
    md5 = aud.hashlib.sha256(METRICS.read_bytes()).hexdigest()
    assert isinstance(md5, str) and len(md5) == 64
