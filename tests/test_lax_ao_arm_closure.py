"""CASE-LAX-DG -- machine checks on the AO limiter arm in the scheme_variants study.

These tests verify that the AO arm (once run) produces physically reasonable
results and is consistent with the scheme_variants framework.

The AO arm was previously marked as ``not_run`` because the leg file was missing.
This test suite ensures that once the leg file exists (from cluster job cj4072),
the results are validated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import lax_scheme_variants as study

ROOT = Path(__file__).resolve().parents[1]
LEGS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "legs"
METRICS = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants" / "metrics.json"


@pytest.fixture(scope="module")
def ao_leg():
    """Load the AO leg file, skip if not yet available."""
    leg_path = LEGS / "ao.json"
    if not leg_path.exists():
        pytest.skip("AO leg file not yet available (cluster job cj4072 pending)")
    return json.loads(leg_path.read_text())


@pytest.fixture(scope="module")
def metrics_with_ao():
    """Load the aggregated metrics, skip if AO not yet aggregated."""
    if not METRICS.exists():
        pytest.skip("metrics.json not yet aggregated with AO arm")
    m = json.loads(METRICS.read_text())
    ao_row = next((r for r in m["rows"] if r["variant"] == "ao"), None)
    if ao_row is None or ao_row.get("not_run"):
        pytest.skip("AO arm not yet aggregated into metrics.json")
    return m


# ----------------------------------------------------------------- basic sanity
class TestAOLegSanity:
    """Basic sanity checks on the AO leg file."""

    def test_leg_has_required_fields(self, ao_leg):
        required = ["variant", "cells", "degree", "dof", "steps", "l1", "l2",
                     "worst_field", "worst_l2", "density_min", "pressure_min"]
        for field in required:
            assert field in ao_leg, f"missing field: {field}"

    def test_variant_is_ao(self, ao_leg):
        assert ao_leg["variant"] == "ao"

    def test_discretisation_matches_pinned(self, ao_leg):
        assert ao_leg["cells"] == study.PINNED_CELLS
        assert ao_leg["degree"] == study.PINNED_DEGREE
        assert ao_leg["dof"] == (study.PINNED_DEGREE + 1) * study.PINNED_CELLS

    def test_steps_positive(self, ao_leg):
        assert ao_leg["steps"] > 0

    def test_wall_time_positive(self, ao_leg):
        assert ao_leg["wall_seconds"] > 0


# --------------------------------------------------------------- physical bounds
class TestAOPhysicalBounds:
    """Physical reasonableness checks on the AO solution."""

    def test_density_positive(self, ao_leg):
        assert ao_leg["density_min"] > 0, "density must be positive"

    def test_pressure_positive(self, ao_leg):
        assert ao_leg["pressure_min"] > 0, "pressure must be positive"

    def test_density_reasonable_range(self, ao_leg):
        """Density should be within a physically reasonable range for Lax problem."""
        # Lax problem: left ρ=0.445, right ρ=0.5, max shock compression ~3x
        assert 0.1 < ao_leg["density_min"] < 3.0

    def test_pressure_reasonable_range(self, ao_leg):
        """Pressure should be within a physically reasonable range for Lax problem."""
        # Lax problem: left p=3.528, right p=0.571
        assert 0.1 < ao_leg["pressure_min"] < 10.0


# --------------------------------------------------------------- comparison with other arms
class TestAOComparison:
    """Compare AO arm with other arms in the study."""

    def test_ao_l2_finite(self, ao_leg):
        for field in ("density", "velocity", "pressure"):
            assert math.isfinite(ao_leg["l2"][field]), f"L2 {field} is not finite"

    def test_ao_worst_l2_is_max(self, ao_leg):
        worst = max(ao_leg["l2"].values())
        assert math.isclose(ao_leg["worst_l2"], worst, rel_tol=1e-12)

    def test_ao_vs_base(self, ao_leg, metrics_with_ao):
        """AO should not be dramatically worse than base (unlimited)."""
        base_row = next(r for r in metrics_with_ao["rows"] if r["variant"] == "base")
        # AO should not be more than 2x worse than base on any field
        for field in ("density", "velocity", "pressure"):
            ratio = ao_leg["l2"][field] / base_row["l2"][field]
            assert ratio < 2.0, f"AO L2 {field} is {ratio:.2f}x base (expected < 2x)"

    def test_ao_vs_av(self, ao_leg, metrics_with_ao):
        """AO should generally be better than or comparable to AV (artificial viscosity)."""
        av_row = next(r for r in metrics_with_ao["rows"] if r["variant"] == "av")
        if study._failed(av_row):
            pytest.skip("AV arm failed, cannot compare")
        # AO worst L2 should not be dramatically worse than AV
        ratio = ao_leg["worst_l2"] / av_row["worst_l2"]
        assert ratio < 1.5, f"AO worst L2 is {ratio:.2f}x AV (expected < 1.5x)"


# ------------------------------------------------------------ framework consistency
class TestAOFrameworkConsistency:
    """Verify AO arm is consistent with the scheme_variants framework."""

    def test_ao_in_metrics_rows(self, metrics_with_ao):
        ao_row = next((r for r in metrics_with_ao["rows"] if r["variant"] == "ao"), None)
        assert ao_row is not None, "AO arm not found in metrics rows"

    def test_ao_not_marked_not_run(self, metrics_with_ao):
        ao_row = next(r for r in metrics_with_ao["rows"] if r["variant"] == "ao")
        assert not ao_row.get("not_run"), "AO arm should not be marked as not_run"

    def test_ao_not_errored(self, metrics_with_ao):
        ao_row = next(r for r in metrics_with_ao["rows"] if r["variant"] == "ao")
        assert "error" not in ao_row, f"AO arm has error: {ao_row.get('error')}"

    def test_ao_decision_computed(self, metrics_with_ao):
        ao_row = next(r for r in metrics_with_ao["rows"] if r["variant"] == "ao")
        assert "decision" in ao_row, "AO arm decision not computed"
        assert "admissible" in ao_row["decision"]

    def test_scope_complete_after_ao(self, metrics_with_ao):
        """With AO arm present, scope should be complete (no not_run except pos_none errored)."""
        verdict = metrics_with_ao["verdict"]
        not_run = verdict.get("arms_not_run", [])
        errored = verdict.get("arms_errored", [])
        # After AO is filled, only pos_none should be errored
        assert "ao" not in not_run, "AO should not be in arms_not_run"
        assert "ao" not in errored, "AO should not be in arms_errored"


# ------------------------------------------------------- closed-set completeness
class TestClosedArmSetComplete:
    """The pre-registered arm set must be fully *measured*, not merely declared closed.

    ``closed_arm_set`` in the artifact means "no arm was added or dropped after
    the numbers were seen".  It must not be readable as "every arm has a number":
    the AO arm spent one full round as ``not_run`` after a cluster job that
    crashed on a stale checkout (rc=1) while the platform still showed
    ``succeeded``.  These checks bind the closure claim to files on disk.
    """

    def test_every_pre_registered_arm_has_a_leg_file(self):
        missing = [name for name in study.VARIANT_ORDER
                   if not (LEGS / f"{name}.json").exists()]
        assert not missing, f"pre-registered arms without a leg record: {missing}"

    def test_metrics_has_one_row_per_arm(self):
        metrics = json.loads(METRICS.read_text())
        assert [r["variant"] for r in metrics["rows"]] == list(study.VARIANT_ORDER)

    def test_no_arm_is_silently_unknown(self):
        metrics = json.loads(METRICS.read_text())
        unknown = [r["variant"] for r in metrics["rows"]
                   if r.get("not_run") or "error" in r]
        # pos_none is a registered crash (positivity=none is nonphysical at this
        # mesh, recorded with its exception); it is an outcome, not a hole.
        assert set(unknown) <= {"pos_none"}, (
            f"arms left without a measurement: {unknown}")

    def test_ao_row_is_bound_to_its_leg_file(self):
        metrics = json.loads(METRICS.read_text())
        leg = json.loads((LEGS / "ao.json").read_text())
        ao_row = next(r for r in metrics["rows"] if r["variant"] == "ao")
        for field in study.FIELDS:
            assert math.isclose(ao_row["l2"][field], leg["l2"][field], rel_tol=1e-12)
        assert ao_row["steps"] == leg["steps"]

    def test_ao_leg_reports_its_provenance(self):
        leg = json.loads((LEGS / "ao.json").read_text())
        assert leg["solver_kwargs"]["limiter"] == "ao"
        assert leg["density_min"] > 0 and leg["pressure_min"] > 0
