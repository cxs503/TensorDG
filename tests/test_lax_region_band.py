"""Gates for the CASE-LAX-DG R3 region-rule audit (``benchmarks/lax_region_band``).

The tests are deliberately *invariant* tests, not smoke tests:

1. the published ``4 h`` band is **not nested** -- its retained region grows by
   tens of percent along the resolution ladder, while a fixed *physical* band
   keeps the same region to within a fraction of a percent;
2. the two rules coincide exactly when ``4 h`` happens to equal the physical
   band (``N = 200`` for a unit domain) -- an internal consistency check that
   the two code paths really describe the same geometry;
3. on the same DG solutions the ``4 h`` rule produces a **non-decreasing**
   error (negative measured order) at P1 while every nested rule stays above
   first order.  That is the discriminating measurement: it shows the published
   ``R3`` failure is dominated by moving region geometry;
4. if the committed artifact exists, it is re-derived from its own ladder table
   (no trust in the stored verdicts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.lax_region_band import (BASELINE_RULE, NESTED_BANDS, QUICK_LADDER,
                                        region_masks, region_verdicts, rule_orders)
from benchmarks.lax_case_pool import (CFL, FINAL_TIME, LEFT, RIGHT,
                                      convergence_order, exact_positions,
                                      exact_reference, exact_waves, modal_fields,
                                      solve_euler_dg_1d, _weighted_l1)

ROOT = Path(__file__).resolve().parents[1]
# The record lives on a git-tracked path (``docs/``).  ``artifacts/`` and ``runs/``
# are filtered out of agent branches by the merge gate, so a gate pinned there
# either skips forever (vacuous) or fails on every clean checkout.
ARTIFACT = ROOT / "docs" / "benchmarks" / "results" / "lax_region_band" / "metrics.json"


def _uniform_points(cells):
    """``cells`` sub-cell samples per element on a unit domain."""
    edges = np.linspace(0.0, 1.0, cells + 1)
    offsets = (np.arange(10) + 0.5) / 10.0
    return edges[:-1, None] + offsets[None, :] * (1.0 / cells)


def test_published_band_is_not_nested_but_physical_bands_are():
    positions = exact_positions(exact_waves())
    fractions = {name: [] for name in (BASELINE_RULE, *NESTED_BANDS)}
    for cells in (100, 200, 400):
        points = _uniform_points(cells)
        masks, _ = region_masks(points, 1.0 / cells, positions)
        for name, mask in masks.items():
            fractions[name].append(float(mask.mean()))
    for name, values in fractions.items():
        change = abs(values[-1] - values[0]) / values[0]
        if name == BASELINE_RULE:
            assert change > 0.05, (name, values)  # region moves with dx
            assert values[-1] > values[0]         # and it moves the same way
        else:
            assert change < 0.02, (name, values)  # region is fixed in space


def test_band_rule_matches_the_physical_band_when_four_h_equals_it():
    positions = exact_positions(exact_waves())
    points = _uniform_points(200)               # h = 1/200 -> 4 h = 0.02
    masks, _ = region_masks(points, 1.0 / 200, positions)
    np.testing.assert_array_equal(masks[BASELINE_RULE], masks["nested_0p02"])


def test_every_region_excludes_the_analytic_wave_locations():
    positions = exact_positions(exact_waves())
    points = _uniform_points(400)
    masks, _ = region_masks(points, 1.0 / 400, positions)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        nearest = int(np.argmin(np.abs(points.ravel() - positions[name])))
        row, col = np.unravel_index(nearest, points.shape)
        for rule, mask in masks.items():
            assert not mask[row, col], (rule, name)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_band_rule_error_does_not_decay_while_nested_rules_do():
    """Same DG solutions, four regions: only the region rule differs."""
    positions = exact_positions(exact_waves())
    errors = {}
    for cells in (64, 128):
        x, coeff, _meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=cells, degree=1,
                                            final_time=FINAL_TIME, cfl=CFL)
        points, primitive, dx, weights = modal_fields(x, coeff)
        exact = exact_reference(points)
        masks, _ = region_masks(points, dx, positions)
        errors[cells] = {
            name: _weighted_l1(points, primitive["density"], exact["density"],
                               weights, dx, mask=mask)
            for name, mask in masks.items()}
    orders = {name: convergence_order(64, errors[64][name], 128, errors[128][name])
              for name in errors[64]}
    # the moving-band rule measures region geometry, not discretisation error
    assert orders[BASELINE_RULE] < 0.5, orders
    # every fixed-physical-band rule stays above first order on the same data
    for name in NESTED_BANDS:
        assert orders[name] > 1.0, (name, orders)


def test_quick_ladder_keeps_a_coarse_fine_pair_per_degree():
    degrees = [degree for _cells, degree in QUICK_LADDER]
    for degree in set(degrees):
        assert degrees.count(degree) >= 2


def test_committed_artifact_is_self_consistent():
    """Re-derive every stored number from the stored ladder (no trust in verdicts)."""
    assert ARTIFACT.exists(), (
        f"missing committed record {ARTIFACT.relative_to(ROOT)}; regenerate with "
        "`python -m benchmarks.lax_region_band`")
    metrics = json.loads(ARTIFACT.read_text())
    entries = metrics["ladder"]
    orders = rule_orders(entries)
    verdicts = region_verdicts(orders)
    assert set(verdicts) == set(metrics["verdicts"])
    for name, gate in verdicts.items():
        assert gate["status"] == metrics["verdicts"][name]["status"], name
        for key, row in gate["per_degree"].items():
            stored = metrics["verdicts"][name]["per_degree"][key]
            if row["order"] is None:
                assert stored["order"] is None
            else:
                assert row["order"] == pytest.approx(stored["order"], rel=1e-12)
    # every figure the artifact advertises must live inside the worktree
    for figure in metrics["figures"]:
        path = Path(figure)
        path = path if path.is_absolute() else ROOT / path
        assert path.exists(), figure
        assert path.resolve().is_relative_to(ROOT), figure
    # the ladder must be strictly finer in cells for every degree
    for degree in {e["degree"] for e in entries}:
        cells = sorted(e["cells"] for e in entries if e["degree"] == degree)
        assert cells == sorted(set(cells)) and len(cells) >= 2