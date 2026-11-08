"""Detector-level pins for the m6495 TVB-constant adjudication (no time marching).

Companion of ``benchmarks/tvb_constant_adjudication.py``: the long-time error
sweep lives there (and in ``tests/test_euler_dg_2d_time_split.py``); the pins
here only touch the *detector*, so they run in well under a second and can guard
the adjudication on every CI pass.

Claim under test (retired): "the default ``tvb_constant = 1.0`` fires on the
smooth isentropic vortex because the threshold scale or its units are wrong".
The measurements below are the refutation: the mask follows
``M`` against the data's own second-derivative bound, and the flagged fraction
saturates instead of growing with refinement.
"""
from __future__ import annotations

from benchmarks.tvb_constant_adjudication import (
    analytic_second_derivative_bound,
    mask_record,
)


def test_analytic_derivative_bound_exceeds_the_default_constant():
    """M = 1 is *below* the vortex's second-derivative bound, so firing is correct."""
    bound = analytic_second_derivative_bound(64)
    peak = max(value for row in bound.values() for value in row.values())
    assert 1.7 < peak < 1.9, peak


def test_mask_follows_the_derivative_bound_criterion():
    at_one, *_ = mask_record(16, 1.0)
    assert at_one["flagged_any"] > 0, "M=1 is below the bound: it must fire"
    # The shipped detector on the projected cell averages must reproduce the same
    # routine on the analytic point values, cell by cell (a units/scale slip
    # would not agree to a handful of cells).
    disagreements = at_one["mismatch_x"] + at_one["mismatch_y"]
    assert disagreements <= 0.01 * 16 * 16, (disagreements, at_one)

    at_four, *_ = mask_record(16, 4.0)
    assert at_four["flagged_any"] == 0, "M=4 is above the bound: it must stay quiet"


def test_flagged_fraction_saturates_instead_of_running_away():
    """A dimension slip would drive the fraction to 1 as h -> 0; it saturates."""
    fractions = [mask_record(n, 1.0)[0]["fraction_any"] for n in (12, 16, 32)]
    assert all(value < 0.10 for value in fractions), fractions
    assert fractions[-1] < 2.0 * max(fractions[0], 1e-12), fractions