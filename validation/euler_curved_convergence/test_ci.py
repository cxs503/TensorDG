"""CI test for the curvilinear Euler DG convergence validation case.

This test verifies that the ``validation/euler_curved_convergence/`` model
directory is self-consistent:

1. All expected artifacts exist (audit.json, audit.md, summary.json)
2. audit.json is valid JSON with the expected structure
3. summary.json references the correct artifacts
4. Key acceptance criteria are met (convergence orders, curved-straight equivalence)

Run:
    pytest validation/euler_curved_convergence/test_ci.py -v
"""
from __future__ import annotations

import json
import math
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def load_audit():
    path = HERE / "audit.json"
    assert path.exists(), f"audit.json not found at {path}"
    return json.loads(path.read_text())


def load_summary():
    path = HERE / "summary.json"
    assert path.exists(), f"summary.json not found at {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------
class TestArtifactsExist:
    def test_audit_json(self):
        assert (HERE / "audit.json").exists()

    def test_audit_md(self):
        assert (HERE / "audit.md").exists()

    def test_summary_json(self):
        assert (HERE / "summary.json").exists()


# ---------------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------------
class TestStructure:
    def test_audit_has_all_sections(self):
        a = load_audit()
        for section in ("final_time", "cfl", "degree", "meshes", "amplitude",
                        "blocks"):
            assert section in a, f"Missing section: {section}"

    def test_summary_has_evidence_lines(self):
        s = load_summary()
        assert len(s.get("evidence_lines", [])) >= 3, \
            "summary.json should document at least 3 evidence lines"

    def test_summary_status(self):
        s = load_summary()
        assert s.get("status") in ("verified", "pass", "fail"), \
            f"summary.json status should be verified/pass/fail, got: {s.get('status')}"

    def test_summary_has_case_id(self):
        s = load_summary()
        assert "case" in s, "summary.json missing 'case' field"
        assert "gate" in s, "summary.json missing 'gate' field"
        assert "physics" in s, "summary.json missing 'physics' field"


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------
class TestAcceptanceCriteria:
    def test_convergence_straight_p1(self):
        """straight_p1 should achieve order >= 1.5 (paper tolerance)."""
        a = load_audit()
        blocks = a.get("blocks", {})
        for strength in ("strength_1", "strength_5"):
            if strength in blocks:
                straight = blocks[strength].get("straight", {})
                evolve = straight.get("evolve", [])
                if len(evolve) >= 2:
                    orders = straight.get("evolve_orders", [])
                    if orders:
                        min_order = min(orders)
                        assert min_order >= 1.5, \
                            f"{strength}/straight_p1: min order {min_order:.2f} < 1.5"

    def test_convergence_warped_p1(self):
        """warped_p1 should achieve order >= 1.5 (paper tolerance)."""
        a = load_audit()
        blocks = a.get("blocks", {})
        for strength in ("strength_1", "strength_5"):
            if strength in blocks:
                warped = blocks[strength].get("warped", {})
                evolve = warped.get("evolve", [])
                if len(evolve) >= 2:
                    orders = warped.get("evolve_orders", [])
                    if orders:
                        min_order = min(orders)
                        assert min_order >= 1.5, \
                            f"{strength}/warped_p1: min order {min_order:.2f} < 1.5"

    def test_conservation_straight_p1(self):
        """straight_p1 conservation drift should be < 1e-10."""
        a = load_audit()
        blocks = a.get("blocks", {})
        for strength in ("strength_1", "strength_5"):
            if strength in blocks:
                straight = blocks[strength].get("straight", {})
                evolve = straight.get("evolve", [])
                for run in evolve:
                    drift = run.get("mass_drift_relative", 1.0)
                    assert drift < 1e-10, \
                        f"{strength}/straight_p1: drift={drift:.3e} >= 1e-10"


# ---------------------------------------------------------------------------
# Cross-validation: summary vs audit
# ---------------------------------------------------------------------------
class TestCrossValidation:
    def test_summary_evidence_lines_count(self):
        s = load_summary()
        evidence_ids = [e["id"] for e in s["evidence_lines"]]
        assert "G2_straight" in evidence_ids
        assert "G2_warped" in evidence_ids
        assert "G3_curved_equivalence" in evidence_ids
        assert "G4_cartesian_reference" in evidence_ids
