"""CKS1998 §5.3 — 2D isentropic vortex order verification (REPRO-DG-CKS1998 anchor).

Tests that the 2-D Euler DG solver reproduces the design order of accuracy
on the smooth isentropic vortex case, as required by the CKS1998 paper.

The benchmark is run by ``benchmarks/cks1998_vortex_2d.py`` and writes
``validation/cks1998_vortex_2d/order.json``.  These tests read that artifact
and assert the observed orders meet the paper's tolerance (>= p + 0.5).
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "validation" / "cks1998_vortex_2d" / "order.json"


def _load():
    assert ARTIFACT.is_file(), f"missing artifact: {ARTIFACT} (run: python -m benchmarks.cks1998_vortex_2d --quick)"
    return json.loads(ARTIFACT.read_text())


@pytest.mark.parametrize("degree_key,design_order", [("p1", 2), ("p2", 3), ("p3", 4)])
def test_vortex_2d_order_meets_paper_tolerance(degree_key, design_order):
    """Observed L1 order >= p + 0.5 (the Cockburn-Shu tolerance for smooth data)."""
    data = _load()
    runs = data["by_degree"][degree_key]
    assert len(runs) >= 2, f"need >= 2 meshes for EOC, got {len(runs)}"
    finest = runs[-1]
    eoc_l1 = finest.get("eoc_l1")
    assert eoc_l1 is not None, "EOC L1 not computed"
    tolerance = design_order - 0.5
    assert eoc_l1 >= tolerance, (
        f"{degree_key}: EOC L1 = {eoc_l1:.3f} < {tolerance} (design {design_order})"
    )


def test_vortex_2d_finest_error_below_three_percent():
    """The finest mesh L1 relative error must be < 3% (engineering gate)."""
    data = _load()
    for key, runs in data["by_degree"].items():
        finest = runs[-1]
        l1_rel = finest["norms"]["density"]["L1_rel"]
        assert l1_rel < 0.03, (
            f"{key}: L1_rel = {l1_rel:.4e} >= 3% on finest mesh"
        )


def test_vortex_2d_anchor_metadata():
    """Artifact is bound to the REPRO-DG-CKS1998 anchor."""
    data = _load()
    assert data["anchor"] == "REPRO-DG-CKS1998"
    assert data["case"] == "vortex_2d"
    assert "analytic" in data["reference"].lower()
