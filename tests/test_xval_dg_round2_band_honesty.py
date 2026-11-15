"""Tests for the cross-order band-honesty audit of the round-2 certified reference.

What is asserted here is the *apparatus*, not the audit's conclusion: the derived
WENO tables are exact, the new solver's reconstruction is symmetric in the two
face directions and converges at least at the order of the shipped scheme, and
the coverage classifier is a correct three-way decision.  The audit's numbers
themselves live in ``validation/xval_dg_round2/artifacts/band_honesty/`` and are
asserted only for internal consistency (they are recomputed by the driver, so a
test that pinned them would just duplicate the driver).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.fv_weno7_1d import (
    WENO_TABLES,
    reconstruction_order_study,
    selftest_tables,
    weno_reconstruct,
)
from benchmarks.xval_dg_round2_band_honesty import coverage_verdict

OUTDIR = Path("validation/xval_dg_round2/artifacts/band_honesty")


# --------------------------------------------------------------- WENO tables
@pytest.mark.parametrize("order", [5, 7])
def test_tables_exact_on_polynomials(order):
    """Ideal-weight blend reproduces degree ``order-1`` polynomials exactly."""
    res = selftest_tables(order)
    assert res["r"] == (order + 1) // 2
    assert res["worst_polynomial_error"] < 1e-10


def test_order5_optimal_weights_are_jiang_shu():
    assert WENO_TABLES[5]["d"] == [0.1, 0.6, 0.3]


def test_order7_optimal_weights_match_literature():
    """Jiang--Shu / Balsara--Shu optimal weights for the seven-cell stencil."""
    d = WENO_TABLES[7]["d"]
    assert d == [1 / 35, 12 / 35, 18 / 35, 4 / 35]


def test_smoothness_indicators_vanish_for_constant_data():
    """Jin--Shu beta must vanish on a constant field (the beta-zero reference point).

    Note it does *not* vanish for linear data: the ``l=1`` term ``int (p')^2 dx``
    is non-zero there, which is exactly what makes the indicators scale as
    ``(Delta u)^2`` and the JS weights resolution dependent.
    """
    from benchmarks.fv_weno7_1d import _beta
    r = 4
    vals = [torch.full((1,), 2.5, dtype=torch.float64) for _ in range(r)]
    assert abs(float(_beta(vals, WENO_TABLES[7]["Q"]))) < 1e-12
    lin = [torch.full((1,), 2.5 + k, dtype=torch.float64) for k in range(r)]
    assert float(_beta(lin, WENO_TABLES[7]["Q"])) > 0.0


# --------------------------------------------------------- reconstruction
@pytest.mark.parametrize("order", [5, 7])
def test_reconstruction_is_direction_symmetric(order):
    """``u_left`` and ``u_right`` must carry the same error against a smooth field.

    A one-cell misalignment of the mirrored evaluation (the classic flip-index
    bug) shows up here as a factor-``O(1)`` asymmetry long before it shows up in
    a convergence study.
    """
    n = 64
    dx = 1.0 / n
    j = torch.arange(n, dtype=torch.float64)
    ubar = (torch.cos(2 * math.pi * j * dx) - torch.cos(2 * math.pi * (j + 1) * dx)) / (
        2 * math.pi * dx)
    u = torch.stack((ubar,) * 3, dim=-1)
    ul, ur = weno_reconstruct(u, order=order)
    xf = torch.arange(n + 1, dtype=torch.float64) * dx
    ex = torch.sin(2 * math.pi * xf)
    el = float((ul[3:-3, 0] - ex[3:-3]).abs().max())
    er = float((ur[3:-3, 0] - ex[3:-3]).abs().max())
    assert el < 1e-4 and er < 1e-4
    assert abs(el - er) <= 0.2 * max(el, er)


@pytest.mark.parametrize("order,floor", [(5, 4.3), (7, 4.8)])
def test_reconstruction_order(order, floor):
    """Observed order on a smooth periodic field (smooth-regime WENO behaviour).

    The JS weights degrade the formal order in smooth regions; the measured
    values on this ladder are ~4.5 (order 5) and ~5.0 (order 7), so the floors are
    set below both rather than at the formal order -- an honest lower bound.
    """
    st = reconstruction_order_study(order)
    assert st["finest_observed_order"] >= floor
    assert st["rows"][-1]["linf"] < st["rows"][0]["linf"]


def test_order7_is_more_accurate_than_order5():
    """The higher order must actually buy accuracy at the same resolution."""
    r5 = reconstruction_order_study(5)["rows"][-1]["linf"]
    r7 = reconstruction_order_study(7)["rows"][-1]["linf"]
    assert r7 < r5 / 2.0


# ------------------------------------------------------- coverage classifier
@pytest.mark.parametrize("delta,u5,u7,expected", [
    (0.05, 0.083, 0.050, "COVERED"),
    (0.083, 0.083, 0.050, "COVERED"),
    (0.10, 0.083, 0.050, "MARGINAL"),
    (0.133, 0.083, 0.050, "MARGINAL"),
    (0.134, 0.083, 0.050, "UNDER_COVERS"),
])
def test_coverage_three_way(delta, u5, u7, expected):
    out = coverage_verdict(delta, u5, u7)
    assert out["level"] == expected
    assert out["band_covers_cross_family"] is (expected == "COVERED")
    assert out["combined_bands_cover"] is (expected != "UNDER_COVERS")


def test_coverage_reports_ratio():
    out = coverage_verdict(0.166, 0.083, 0.05)
    assert pytest.approx(out["ratio_delta_over_band"], rel=1e-12) == 2.0


# ----------------------------------------------------------- audit artifacts
def test_audit_artifacts_consistent_if_present():
    path = OUTDIR / "band_honesty_audit.json"
    if not path.exists():
        pytest.skip("audit not run yet (python -m benchmarks.xval_dg_round2_band_honesty)")
    r = json.loads(path.read_text())
    for key in ("certified_reference_weno5", "certified_reference_weno7",
                "distances", "coverage", "gate_survival", "dg_legs", "verdict"):
        assert key in r
    d = r["distances"]["cross_order_4level_percent"]
    u5 = r["certified_reference_weno5"]["band_percent"]
    u7 = r["certified_reference_weno7"]["band_percent"]
    assert r["coverage"]["level"] == coverage_verdict(d, u5, u7)["level"]
    # the reference grid must be shared (no interpolation in the comparison)
    assert (r["certified_reference_weno5"]["reference_ncells"]
            == r["certified_reference_weno7"]["reference_ncells"])
    # every re-judged leg carries both errors
    for row in r["dg_legs"]["legs"]:
        assert row["error_percent_vs_weno5_ref"] > 0
        assert row["error_percent_vs_weno7_ref"] > 0