"""CKS1998 §5.1 — smooth 2-D Euler order verification from pre-computed artifact.

Tests that the 2-D Euler DG solver reproduces the design order of accuracy
on the smooth entropy wave case, as required by the CKS1998 paper.

The benchmark is run by ``benchmarks/cks1998_smooth_2d.py`` and writes
``validation/cks1998_smooth_2d/order.json``.  These tests read that artifact
and assert the observed orders meet the paper's tolerance (>= p + 0.5).

This is the *artifact-based* variant; the *run-based* variant lives in
``tests/test_cks1998_smooth_2d_order.py`` (P1 only, due to compute budget).
"""
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "validation" / "cks1998_smooth_2d" / "order.json"


def _load():
    assert ARTIFACT.is_file(), f"missing artifact: {ARTIFACT} (run: python -m benchmarks.cks1998_smooth_2d --quick)"
    return json.loads(ARTIFACT.read_text())


@pytest.mark.parametrize("degree_key,design_order", [("p1", 2), ("p2", 3), ("p3", 4)])
def test_smooth_2d_order_meets_paper_tolerance_l1(degree_key, design_order):
    """Observed L1 order >= p + 0.5 (the Cockburn-Shu tolerance for smooth data)."""
    data = _load()
    runs = data["runs"][degree_key]
    assert len(runs) >= 2, f"need >= 2 meshes for EOC, got {len(runs)}"
    # Use the last two meshes for EOC
    coarse, fine = runs[-2], runs[-1]
    ratio = coarse["h"] / fine["h"]
    eoc_l1 = math.log(
        coarse["norms"]["density"]["L1"] / fine["norms"]["density"]["L1"]
    ) / math.log(ratio)
    tolerance = design_order - 0.5
    assert eoc_l1 >= tolerance, (
        f"{degree_key}: EOC L1 = {eoc_l1:.3f} < {tolerance} (design {design_order})"
    )


@pytest.mark.parametrize("degree_key,design_order", [("p1", 2), ("p2", 3), ("p3", 4)])
def test_smooth_2d_order_meets_paper_tolerance_linf(degree_key, design_order):
    """Observed Linf order >= p + 0.5 (the Cockburn-Shu tolerance for smooth data)."""
    data = _load()
    runs = data["runs"][degree_key]
    assert len(runs) >= 2, f"need >= 2 meshes for EOC, got {len(runs)}"
    coarse, fine = runs[-2], runs[-1]
    ratio = coarse["h"] / fine["h"]
    eoc_linf = math.log(
        coarse["norms"]["density"]["Linf"] / fine["norms"]["density"]["Linf"]
    ) / math.log(ratio)
    tolerance = design_order - 0.5
    assert eoc_linf >= tolerance, (
        f"{degree_key}: EOC Linf = {eoc_linf:.3f} < {tolerance} (design {design_order})"
    )


def test_smooth_2d_finest_error_below_one_percent():
    """The finest mesh L1 relative error must be < 1% (engineering gate)."""
    data = _load()
    for key, runs in data["runs"].items():
        finest = runs[-1]
        l1_rel = finest["norms"]["density"]["L1_rel"]
        assert l1_rel < 0.01, (
            f"{key}: L1_rel = {l1_rel:.4e} >= 1% on finest mesh"
        )


def test_smooth_2d_anchor_metadata():
    """Artifact is bound to the REPRO-DG-CKS1998 anchor."""
    data = _load()
    assert data["anchor"] == "REPRO-DG-CKS1998"
    assert data["case"] == "smooth_2d_entropy_wave"
    assert "analytic" in data["reference"].lower()


def test_smooth_2d_tvb_threshold_prediction():
    """The predicted TVB threshold M* = 0.2 * pi^2 is declared and matches theory."""
    data = _load()
    predicted = data["predicted_tvb_threshold"]
    expected = 0.2 * math.pi ** 2
    assert abs(predicted - expected) < 1e-10, f"M* = {predicted} != {expected}"
    assert 1.9 < predicted < 2.0
