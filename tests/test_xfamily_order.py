"""Contract tests for the cross-family convergence-order protocol.

These tests assert *invariants of the protocol* (what a verdict must refuse to
credit) and one measurement against families whose orders are known in advance.
They are deliberately independent of any external repository: the numbers come
from :mod:`benchmarks.xfamily_order_protocol` solvers or from synthetic series.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg.xfamily_order import (
    cross_family_difference,
    cross_family_verdict,
    family_error_ratios,
    half_gate_budget,
    max_point_error,
    normalisation_ratio,
    order_bracket,
    order_law,
    percent_of,
    required_resolution,
)

LADDER = (8, 16, 32, 64)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def test_order_bracket_and_min_order():
    assert order_bracket(2.0, 4.0) == (2.0, 4.0)
    assert order_law(5.0, 3.0, 4.0) == 3.0
    with pytest.raises(ValueError):
        order_bracket(0.0, 2.0)


def test_half_gate_budget_is_half_the_gate():
    # |A-B| <= e_A + e_B means a <3% agreement needs each family <1.5%.
    assert half_gate_budget(3.0) == 1.5


def test_normalisation_ratio_reconciles_the_two_percentages():
    # The anchor quotes the same absolute difference as 3.495% (v normalised) and
    # 0.890% (lid speed normalised); the ratio is the amplitude ratio.
    ratio = normalisation_ratio(1.0, 0.2546)
    assert ratio == pytest.approx(3.928, rel=1e-3)
    assert 3.495 / 0.890 == pytest.approx(ratio, rel=2e-2)


def test_percent_of_and_field_metrics():
    assert percent_of(0.03, 1.0) == pytest.approx(3.0)
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    b[1, 1] = -0.04
    assert cross_family_difference(a, b) == pytest.approx(0.04)
    exact = np.zeros((3, 3))
    exact[1, 1] = 1.0
    assert max_point_error(b, exact, relative=True) == pytest.approx(1.04)
    # interior=True drops the outer node layer (here the only node carrying
    # the 1.0 spike), so the remaining field is identically zero.
    b5 = np.zeros((5, 5))
    b5[1, 1] = -0.04
    exact5 = np.zeros((5, 5))
    assert max_point_error(b5, exact5, interior=True) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        cross_family_difference(a, np.zeros((4, 4)))
    with pytest.raises(ValueError):
        max_point_error(np.full((3, 3), np.nan), exact)


def test_family_error_ratios_can_exceed_one_but_not_two():
    # diff <= e_A + e_B (sum) but diff may exceed max(e_A, e_B) when the smaller
    # error adds on top of the larger one instead of cancelling it.
    ratio = family_error_ratios([1.5e-7], [5.6e-8], [9.7e-8])
    assert 1.0 < ratio[0] < 2.0


def test_required_resolution_inverts_the_measured_decay():
    diffs = [1e-2, 2.5e-3, 6.25e-4]
    assert required_resolution(LADDER[:-1], diffs, 1e-4) == pytest.approx(80.0, rel=1e-6)
    assert required_resolution([8, 16, 32], [1e-2, 1e-2, 1e-2], 1e-4) == math.inf


# ---------------------------------------------------------------------------
# verdict rules
# ---------------------------------------------------------------------------
def _series(order: float, amplitudes=None, ladder=LADDER):
    if amplitudes is None:
        amplitudes = [1.0] * len(ladder)
    assert len(amplitudes) == len(ladder)
    return [a * (1.0 / n) ** order for a, n in zip(amplitudes, ladder)]


def test_degenerate_pair_is_not_credited():
    diffs = [1e-16] * len(LADDER)
    verdict = cross_family_verdict(
        LADDER, diffs, gate_percent=3.0, gate_order=1.8, family_orders=[2.0, 2.0],
    )
    assert verdict.degenerate is True
    assert verdict.passed is False
    assert verdict.order_attribution == "not_run"


def test_alias_pair_is_refused_even_though_both_gates_are_green():
    # Both families carry the same 2nd-order error (e = 1e-2 h^2); their
    # difference is 1% of it and decays one order faster -> order gate would
    # read 3.0, which must not be credited.
    own = _series(2.0)
    diffs = [1e-2 * e for e in own]
    verdict = cross_family_verdict(
        LADDER, diffs, gate_percent=3.0, gate_order=1.8,
        family_orders=[2.0, 2.0], family_errors_a=own, family_errors_b=own,
    )
    assert verdict.gate_pct_passed is True
    assert verdict.gate_order_passed is True
    assert verdict.order_attribution == "alias_pair_shared_error"
    assert verdict.passed is False
    assert any("alias" in note for note in verdict.notes)


def test_dominant_family_regime_is_credited_and_reports_the_dominant_order():
    errors_b = _series(2.0)
    errors_a = _series(4.0, amplitudes=[1e-3] * len(LADDER))
    diffs = [max(a, b) for a, b in zip(errors_a, errors_b)]
    verdict = cross_family_verdict(
        LADDER, diffs, gate_percent=30.0, gate_order=1.8,
        family_orders=[4.0, 2.0], family_errors_a=errors_a, family_errors_b=errors_b,
    )
    assert verdict.order_attribution == "dominant_family_controls"
    assert verdict.p_finest == pytest.approx(2.0, abs=0.05)
    assert verdict.passed is True


def test_order_below_the_bracket_is_a_violation_not_a_pass():
    diffs = _series(1.0)
    verdict = cross_family_verdict(
        LADDER, diffs, gate_percent=30.0, gate_order=1.8, family_orders=[2.0, 2.0],
    )
    assert verdict.order_attribution == "below_range_violation"
    assert verdict.passed is False


def test_unconverged_family_blocks_the_verdict():
    diffs = _series(2.0)
    verdict = cross_family_verdict(
        LADDER, diffs, gate_percent=30.0, gate_order=1.8,
        family_orders=[2.0, 2.0], converged_b=False,
    )
    assert verdict.gate_order_passed is True
    assert verdict.passed is False
    assert any("R2" in note for note in verdict.notes)


def test_gate_percent_uses_the_declared_normaliser():
    diffs = _series(2.0)  # 1.56e-2 at n=8 -> 1.56% of unit amplitude
    coarse = cross_family_verdict(
        LADDER, diffs, gate_percent=3.0, gate_order=1.8, family_orders=[2.0, 2.0],
    )
    tight = cross_family_verdict(
        LADDER, diffs, gate_percent=3.0, gate_order=1.8, family_orders=[2.0, 2.0],
        scale=0.25,  # same absolute difference, four times the percentage
    )
    assert coarse.gate_pct_passed is True
    assert tight.gate_pct_passed is False


# ---------------------------------------------------------------------------
# measurement: families with known orders
# ---------------------------------------------------------------------------
def test_measured_family_orders_and_cross_family_order_law():
    """fd2 (O(h^2)) vs fd4 (O(h^4)): the difference must follow the dominant family."""
    from benchmarks.xfamily_order_protocol import exact_field, solve_fd2, solve_fd4
    from tensordg.convergence import observed_order

    ladder = (8, 16, 32)
    fields = {name: {n: fn(n) for n in ladder} for name, fn in
              (("fd2", solve_fd2), ("fd4", solve_fd4))}
    own = {
        name: [max_point_error(fields[name][n], exact_field(n)) for n in ladder]
        for name in fields
    }
    diffs = [cross_family_difference(fields["fd2"][n], fields["fd4"][n]) for n in ladder]

    assert observed_order(ladder, own["fd2"]) == pytest.approx(2.0, abs=0.05)
    assert observed_order(ladder, own["fd4"]) == pytest.approx(4.0, abs=0.15)
    # fd2's own error is ~40x fd4's on this ladder, so fd2 owns the difference
    # and the cross-family order falls to the *lower* family order (2).
    assert observed_order(ladder, diffs) == pytest.approx(2.0, abs=0.1)
    verdict = cross_family_verdict(
        ladder, diffs, gate_percent=3.0, gate_order=1.8,
        family_orders=[2.0, 4.0], family_errors_a=own["fd2"], family_errors_b=own["fd4"],
    )
    assert verdict.order_attribution == "dominant_family_controls"
    assert verdict.gate_pct_passed is True
    assert verdict.p_finest >= 1.8


def test_alias_detection_on_the_real_fd2_vs_sipg_p1_pair():
    """SIPG p=1 on a Cartesian grid and 5-point FD2 share their leading error term."""
    from benchmarks.xfamily_order_protocol import exact_field, solve_dg, solve_fd2

    ladder = (8, 16, 32)
    dg = {n: solve_dg(n, 1) for n in ladder}
    fd = {n: solve_fd2(n) for n in ladder}
    own_dg = [max_point_error(dg[n], exact_field(n)) for n in ladder]
    own_fd = [max_point_error(fd[n], exact_field(n)) for n in ladder]
    diffs = [cross_family_difference(dg[n], fd[n]) for n in ladder]
    ratios = family_error_ratios(diffs, own_dg, own_fd)
    # The two families agree to ~1% of their own error at n=8 because the
    # leading 2nd-order term is the same stencil.
    assert ratios[0] < 0.05
    verdict = cross_family_verdict(
        ladder, diffs, gate_percent=3.0, gate_order=1.8,
        family_orders=[2.0, 2.0], family_errors_a=own_dg, family_errors_b=own_fd,
    )
    assert verdict.order_attribution == "alias_pair_shared_error"
    assert verdict.passed is False