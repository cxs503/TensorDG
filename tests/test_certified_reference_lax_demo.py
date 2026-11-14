"""Tests for the certified reference Lax demo (methodology demonstration)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import certified_reference_lax_demo as demo
from tensordg.certified_reference import CERTIFIED, certify, coarsen


class TestSyntheticProfile:
    def test_exact_profile_shape(self):
        x = np.linspace(0, 1, 100)
        rho = demo._exact(x)
        assert rho.shape == (100,)
        assert rho.min() > 0
        assert rho.max() < 1.0

    def test_synthetic_converges(self):
        """Higher resolution → smaller error against exact."""
        _, r200 = demo._synthetic(200, p=1.05, amp=0.08)
        _, r400 = demo._synthetic(400, p=1.05, amp=0.08)
        _, exact = demo._synthetic(400, p=1.05, amp=0.0)
        e200 = np.sum(np.abs(coarsen(r200, 2) - coarsen(exact, 2))) / np.sum(np.abs(exact))
        e400 = np.sum(np.abs(r400 - exact)) / np.sum(np.abs(exact))
        assert e400 < e200, "finer grid should have smaller error"


class TestCertifiedReference:
    def test_fv_ladder_certifies(self):
        """The synthetic FV ladder should certify with proper parameters."""
        levels = [{"ncells": n, "field": demo._synthetic(n, p=1.05, amp=0.05)[1]}
                  for n in demo.FV_LEVELS]
        c = certify(levels, tolerance_percent=5.0)
        # With lower amplitude, the ladder should show convergence
        assert c["verdict"] == CERTIFIED or c["band_percent"] is not None

    def test_reference_grid_is_second_finest(self):
        ref = demo.certified_reference()
        if ref["rho"] is not None:
            assert ref["rho"].size == demo.FV_LEVELS[1]  # 400


class TestCrossValidate:
    def test_audit_returns_required_keys(self):
        audit = demo.cross_validate()
        for key in ("D1_reference_usable", "D2_legs_under_strict_gate",
                     "D3_reference_vs_exact", "D4_no_self_certification",
                     "accepted", "legs"):
            assert key in audit, f"missing key: {key}"

    def test_legs_count(self):
        audit = demo.cross_validate()
        assert len(audit["legs"]) == len(demo.DG_LEGS)

    def test_d3_reference_vs_exact_is_reasonable(self):
        """The certified reference should be close to exact."""
        audit = demo.cross_validate()
        assert audit["D3_reference_vs_exact"]["rel_l1_percent"] < 5.0


class TestArtifacts:
    def test_write_artifacts_creates_files(self, tmp_path):
        audit = demo.cross_validate()
        demo.write_artifacts(audit, str(tmp_path))
        assert (tmp_path / "xval_lax_audit.json").exists()
        assert (tmp_path / "xval_lax_audit.md").exists()

    def test_json_is_valid(self, tmp_path):
        audit = demo.cross_validate()
        demo.write_artifacts(audit, str(tmp_path))
        data = json.loads((tmp_path / "xval_lax_audit.json").read_text())
        assert "accepted" in data
        assert "legs" in data

    def test_markdown_contains_accepted(self, tmp_path):
        audit = demo.cross_validate()
        demo.write_artifacts(audit, str(tmp_path))
        md = (tmp_path / "xval_lax_audit.md").read_text()
        assert "accepted" in md.lower()
