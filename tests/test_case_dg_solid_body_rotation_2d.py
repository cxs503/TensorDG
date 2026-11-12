"""Acceptance test for CASE-DG-SOLID-BODY-ROTATION-2D.

LeVeque 2002, *Finite Volume Methods for Hyperbolic Problems*, Ch. 19.
DOI: 10.1017/cbo9780511791253

Acceptance criteria (from mission total ledger):
  - Grid: 100 x 100
  - CFL: 0.5
  - Metric: L1 error and extremum (peak) decay vs initial value after one rotation
  - Tolerance: < 5%

Note: The 100x100 case is computationally expensive and is run via the cluster.
This test reads the pre-computed results from the artifact JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
TOLERANCE = 0.05  # 5%


def _load_artifact(name: str) -> dict:
    """Load a JSON artifact, skipping if not found."""
    path = ARTIFACT_DIR / name
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    with open(path) as f:
        return json.load(f)


class TestSolidBodyRotationAcceptance:
    """Acceptance tests for the LeVeque Ch.19 solid-body rotation benchmark."""

    def test_p2_n100_cfl05_l1_error(self):
        """P2, 100x100, CFL=0.5: L1 relative error < 5%."""
        data = _load_artifact("case_dg_solid_body_rotation_2d_p2_n100.json")
        case = data.get("case", data)
        assert case["l1_relative"] < TOLERANCE, (
            f"L1 relative error {case['l1_relative']:.4f} >= {TOLERANCE}"
        )

    def test_p2_n100_cfl05_peak_retention(self):
        """P2, 100x100, CFL=0.5: peak retention > 95%."""
        data = _load_artifact("case_dg_solid_body_rotation_2d_p2_n100.json")
        case = data.get("case", data)
        assert case["peak_retention_nodal"] > (1.0 - TOLERANCE), (
            f"Peak retention {case['peak_retention_nodal']:.4f} <= {1-TOLERANCE}"
        )

    def test_p2_n100_mass_conservation(self):
        """P2, 100x100, CFL=0.5: mass conserved to machine precision."""
        data = _load_artifact("case_dg_solid_body_rotation_2d_p2_n100.json")
        case = data.get("case", data)
        assert case["mass_drift_rel"] < 1e-12, (
            f"Mass drift {case['mass_drift_rel']:.2e} exceeds 1e-12"
        )


class TestConvergence:
    """Convergence tests across resolutions."""

    def test_p1_converges(self):
        """P1 L1 error decreases with resolution."""
        r16 = _load_artifact("case_dg_solid_body_rotation_2d_p1_n16.json")
        r32 = _load_artifact("case_dg_solid_body_rotation_2d_p1_n32.json")
        e16 = r16.get("case", r16)["l1_relative"]
        e32 = r32.get("case", r32)["l1_relative"]
        assert e32 < e16, f"P1 not converging: {e16:.4e} -> {e32:.4e}"

    def test_p2_converges(self):
        """P2 L1 error decreases with resolution."""
        r16 = _load_artifact("case_dg_solid_body_rotation_2d_p2_n16.json")
        r25 = _load_artifact("case_dg_solid_body_rotation_2d_p2_n25.json")
        e16 = r16.get("case", r16)["l1_relative"]
        e25 = r25.get("case", r25)["l1_relative"]
        assert e25 < e16, f"P2 not converging: {e16:.4e} -> {e25:.4e}"

    def test_p1_n32_passes_5pct_gate(self):
        """P1 at 32x32 passes the 5% L1 gate."""
        data = _load_artifact("case_dg_solid_body_rotation_2d_p1_n32.json")
        case = data.get("case", data)
        assert case["l1_relative"] < TOLERANCE, (
            f"P1 n=32 L1 error {case['l1_relative']:.4f} >= {TOLERANCE}"
        )

    def test_p2_n25_passes_5pct_gate(self):
        """P2 at 25x25 passes the 5% L1 gate."""
        data = _load_artifact("case_dg_solid_body_rotation_2d_p2_n25.json")
        case = data.get("case", data)
        assert case["l1_relative"] < TOLERANCE, (
            f"P2 n=25 L1 error {case['l1_relative']:.4f} >= {TOLERANCE}"
        )
