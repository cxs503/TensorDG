"""Gates for the shipped-vs-paper TVB troubled-cell criterion A/B.

Anchor ``REPRO-DG-CKS1998``, verification debt: the A1 accuracy gate in
``bench_spec`` runs the solver *unlimited*, and every limiter gate that existed
uses ``M = 0`` (strict TVD).  The regime the paper actually designs the
indicator for -- finite ``M``, smooth solution with extrema -- had no assertion
at all.  These tests pin it down.

The assertions are stated against the *exact* second derivative of the test
profile (``sin(2*pi*x)`` gives ``max|u''| = (2*pi)**2``); no threshold here is
fitted to the measurement.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensordg import limiters as limiters_mod
from tensordg.tvb import troubled_from_differences
from validation.repro_cks1998 import tvb_criterion_ab as ab

CURV = ab.SINE_CURVATURE_MAX
CELLS = (20, 40)          # short ladder: the assertion is order *identity*, not asymptotic
DEGREES = (1, 2)


def test_curvature_constant_is_the_exact_profile_value():
    """The gate's M is derived from the test function, not tuned."""
    assert CURV == pytest.approx((2.0 * math.pi) ** 2, rel=0, abs=0)
    # max|u''| of sin(2*pi*x): |4*pi^2 * sin| maxes at 1.
    assert CURV == pytest.approx(4.0 * math.pi**2, rel=1e-15)


def test_paper_criterion_releases_only_where_a_side_is_second_order():
    """Cockburn-Shu form: released iff a one-sided difference is O(h**2)."""
    h = 0.1
    M = 1.0
    # Constant data: every side is (numerically) zero -> released.
    zeros = torch.zeros(5, 1, dtype=torch.float64)
    troubled, _ = ab.paper_troubled_from_differences(zeros, zeros, h, M)
    assert not bool(troubled.any())
    # O(h) differences with M h^2 far below them -> troubled everywhere.
    lin = torch.full((5, 1), 0.5 * h, dtype=torch.float64)
    troubled, _ = ab.paper_troubled_from_differences(lin, lin.clone(), h, M)
    assert bool(troubled.all())
    # Shrink the differences to O(h**2) -> released again, even at the same M.
    small = torch.full((5, 1), 0.5 * M * h * h, dtype=torch.float64)
    troubled, _ = ab.paper_troubled_from_differences(small, small.clone(), h, M)
    assert not bool(troubled.any())


def test_shipped_criterion_differs_from_paper_in_a_monotone_stretch():
    """The two readings are not the same test -- otherwise the A/B is pointless."""
    h = 0.01
    M = 1.0
    # Monotone smooth data: d ~ u' h = O(h) >> M h^2, with |d_left - d_right| = 0
    # (exactly linear stretch) so the shipped deviation test releases the cell
    # while the paper's one-sided test flags it.
    d_left = torch.tensor([[2e-2]], dtype=torch.float64)
    d_right = torch.tensor([[2e-2]], dtype=torch.float64)
    shipped, _ = troubled_from_differences(d_left, d_right, h, M)
    paper, _ = ab.paper_troubled_from_differences(d_left, d_right, h, M)
    assert not bool(shipped.item()), "shipped test should release a linear stretch"
    assert bool(paper.item()), "paper test should flag it (|a| = 2h >> M h^2)"


def test_criterion_context_manager_restores_the_kernel():
    """The A/B must not leak a patched limiter kernel into the rest of the suite."""
    original = limiters_mod.troubled_from_differences
    with ab.criterion("paper"):
        assert limiters_mod.troubled_from_differences is ab.paper_troubled_from_differences
        assert ab._ACTIVE_CRITERION == "paper"
    assert limiters_mod.troubled_from_differences is original
    assert ab._ACTIVE_CRITERION == "shipped"
    with pytest.raises(ValueError):
        with ab.criterion("nonsense"):
            pass


def test_solve_cache_is_keyed_on_the_active_criterion():
    """Regression: a cache keyed only on (ncells, degree, M) would return the
    *shipped* solution for a *paper*-criterion request and silently make the A/B
    identical."""
    ab._SOLVE_CACHE.clear()
    shipped = ab.order_ladder(1, CELLS, 0.25 * CURV, "shipped")
    paper = ab.order_ladder(1, CELLS, 0.25 * CURV, "paper")
    assert shipped["l2"] != paper["l2"], "paper and shipped rows must be distinct solves"


@pytest.mark.parametrize("degree", DEGREES)
def test_g1_indicator_is_inert_at_the_curvature_threshold(degree):
    """M = max|u''| => no cell flagged, solution bit-identical to unlimited,
    order at least p + 1/2.  This is the paper's design claim, measured."""
    row = ab.order_ladder(degree, CELLS, CURV, "shipped")
    assert max(row["troubled_fraction_initial"]) == 0.0
    assert max(row["max_abs_diff_vs_unlimited"]) == 0.0
    assert row["observed_order"] >= row["order_floor"]


@pytest.mark.parametrize("degree", DEGREES)
def test_g1_counters_g2_the_indicator_does_fire_below_threshold(degree):
    """Non-vacuity: at M = max|u''|/4 the same indicator flags cells and the
    solution departs from the unlimited one, so G1 is not passing by inertia."""
    row = ab.order_ladder(degree, CELLS, 0.25 * CURV, "shipped")
    assert max(row["troubled_fraction_initial"]) > 0.5
    assert max(row["max_abs_diff_vs_unlimited"]) > 0.0


def test_enabling_the_limiter_at_the_public_default_costs_order():
    """The public default ``tvb_constant=1.0`` is an absolute M and is not tied to
    the solution's curvature; on this smooth problem it behaves like M = 0 and
    the observed order drops below the theoretical p + 1."""
    payload = ab.evaluate("smoke")
    d1 = payload["gates"]["D1_public_default_constant"]
    for degree in (1, 2):
        entry = d1[f"p{degree}"]
        assert entry["troubled_fraction"] > 0.5
        assert entry["order_loss"] > 0.25


def test_paper_form_is_never_more_permissive_than_the_shipped_form():
    payload = ab.evaluate("smoke")
    c1 = payload["gates"]["C1_paper_form_is_stricter"]
    assert c1, "C1 must produce comparisons"
    for key, entry in c1.items():
        assert entry["satisfied"], f"{key}: paper form flagged fewer cells than shipped"


def test_artifact_matches_a_fresh_evaluation():
    """The committed artifact must be reproducible from the module (no hand editing)."""
    import json
    from pathlib import Path

    path = Path(ab.ARTIFACT_DIR) / "tvb_criterion_ab_smoke.json"
    if not path.exists():
        pytest.skip("smoke artifact not generated in this checkout")
    committed = json.loads(path.read_text())
    fresh = ab.evaluate("smoke")
    assert committed["gates"] == fresh["gates"]
    assert committed["all_pass"] == fresh["all_pass"]
    assert [r["l2"] for r in committed["rows"]] == [r["l2"] for r in fresh["rows"]]