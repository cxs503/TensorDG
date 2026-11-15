"""Gates for the declared-design-order ceiling of the round-2 certification.

Background (the defect these tests pin)
--------------------------------------
``certify`` refused any ladder whose observed order exceeded ``P_MAX = 6.0``.
That bound is a *policy* number, not a physical one: a DG scheme of degree ``k``
on a smooth solution has cell averages that superconverge at ``O(h**(2k+1))``, so
a P3 ladder is *expected* to report ``p ~ 7`` and was therefore refused **by
construction**.  ``docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md`` measured exactly
that on this repository's own solvers (``advection_sine_P3`` p=7.1240,
``advection_gaussian_P3`` p=6.9364, both ``WITHHELD_ORDER_TOO_HIGH`` even though
their certified profiles were 2-3 orders of magnitude more accurate than the raw
finest level and their bands still covered the true error).

The fix is opt-in: a caller that knows its scheme declares ``design_order=k`` and
the absolute ceiling is replaced by the two-sided superconvergence window
``2k+1 +/- SUPERCONVERGENCE_SLACK`` (:func:`tensordg.certified_reference.order_ceiling`).
Nothing changes when the argument is absent, so every published number stays
comparable.

The tests below assert the *invariants* of that change, not merely that code ran:
the window is two-sided (a merely-above-6 order is still refused), it cannot be
bought by a non-contracting ladder, it never shrinks the region accepted at
``P_MAX``, and the gate-validity rule (``U / tau <= 1/3``) still vetoes a
certification whose band eats the gate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordg import certified_reference as cr


# ------------------------------------------------------------------ fixtures
def _base(ncells: int) -> np.ndarray:
    """Cell averages of ``1 + 0.2 x`` on ``[0, 1)`` -- a smooth, O(1) profile."""
    x = (np.arange(ncells) + 0.5) / ncells
    return 1.0 + 0.2 * x


def _sine(ncells: int) -> np.ndarray:
    """Cell averages of ``sin(2 pi x)`` on ``[0, 1)`` (mean |v| = 2/pi)."""
    x = (np.arange(ncells) + 0.5) / ncells
    return np.sin(2.0 * np.pi * x)


def _cell_averages(fn, ncells: int, nq: int = 64) -> np.ndarray:
    """Cell averages of ``fn`` on ``[0, 1)`` by ``nq``-point Gauss quadrature.

    Pointwise truth is *not* good enough here: for a cell-average field the
    midpoint value differs from the cell average by ``O(h**2)`` (0.04 % at
    N=64 for ``sin 2 pi x``), which swamps the ``O(h**(2k+1))`` error a P3
    ladder is being certified at.  The round-2 practice consumes cell averages,
    so the truth must be cell averages too.
    """
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    xc = (np.arange(ncells) + 0.5) / ncells
    h = 1.0 / ncells
    # (nq) x (ncells) quadrature on each cell, mapped to [0, 1)
    xs = xc[None, :] + 0.5 * h * nodes[:, None]
    # mean over the cell: (1/h) * integral = 0.5 * sum(w * f)
    return 0.5 * (weights[:, None] * fn(xs)).sum(axis=0)


def _ladder_from_relative_differences(d: list[float],
                                      ncells: tuple[int, ...] = (16, 32, 64)):
    """Build a nesting ladder whose consecutive ``rel_l1`` differences are ~``d``.

    ``d[i]`` is the relative L1 difference between level ``i`` and ``i+1``.  The
    profiles are scaled copies of a common shape, so the realised differences are
    ``d`` up to the O(1) normalisation of the shape (asserted by the caller via
    ``certify``'s own ``differences`` block, never assumed here).
    """
    assert len(ncells) == len(d) + 1
    shape = _base
    n0 = ncells[0]
    profiles = [shape(n0)]
    for i, ratio in enumerate(d):
        prev = profiles[-1]
        # profile i+1 = prev / (1 + ratio) reproduces rel_l1(prev, next) = ratio
        profiles.append(prev / (1.0 + ratio))
    levels = []
    for n, prof in zip(ncells, profiles):
        # the profile was built on the coarsest grid; lift it to grid n by
        # exact nesting (fine cell averages of a piecewise-constant field are
        # the block means, so coarsening returns ``prof`` exactly)
        factor = n // n0
        levels.append({"ncells": int(n), "field": np.repeat(prof, factor)})
    return levels


# ------------------------------------------------------- window definition
def test_expected_cell_average_rate_is_two_k_plus_one():
    assert cr.order_ceiling(1)[0] == pytest.approx(3.0)
    assert cr.order_ceiling(2)[0] == pytest.approx(5.0)
    assert cr.order_ceiling(3)[0] == pytest.approx(7.0)
    rate, ceiling = cr.order_ceiling(3, slack=0.5)
    assert ceiling == pytest.approx(rate + 0.5)


# ---------------------------------------------- default path is untouched
def test_default_policy_still_refuses_an_order_above_six():
    """No ``design_order`` -> the shipped behaviour, bit for bit."""
    levels = _ladder_from_relative_differences([1.0 / 128.0, 1.0 / 128.0 ** 2])
    cert = cr.certify(levels, tolerance_percent=3.0)
    assert cert["design_order"] is None
    assert cert["order_ceiling"] is None
    assert cert["superconvergent"] is False
    # p ~ 7 by construction, so the absolute bound must still veto it
    assert cert["classification"] == "order_too_high"
    assert cert["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH


def test_declaring_the_order_never_shrinks_the_accepted_region():
    """An order already inside ``P_MAX`` stays accepted whatever ``design_order`` is.

    Regression guard: for a P1 shock-capturing ladder (p ~ 1-2) the declared
    window is [2.5, 3.5]; the declaration must not turn a previously accepted
    ladder into a refused one.
    """
    # p = 2 with a d_fine of 1/512 so the band (0.08 % of a 3 % gate) is usable
    levels = _ladder_from_relative_differences([1.0 / 128.0, 1.0 / 512.0])
    plain = cr.certify(levels, tolerance_percent=3.0)
    declared = cr.certify(levels, tolerance_percent=3.0, design_order=1)
    assert plain["verdict"] == cr.CERTIFIED
    assert declared["verdict"] == cr.CERTIFIED
    assert declared["superconvergent"] is False
    assert declared["observed_order_used"] == pytest.approx(plain["observed_order_used"])
    assert declared["band_percent"] == pytest.approx(plain["band_percent"])


# ------------------------------------------------------- two-sided window
def test_window_accepts_an_order_matching_the_declared_rate():
    levels = _ladder_from_relative_differences([1.0 / 128.0, 1.0 / 128.0 ** 2])
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=3)
    assert cert["verdict"] == cr.CERTIFIED
    assert cert["superconvergent"] is True
    assert cert["order_rate_expected"] == pytest.approx(7.0)
    assert cert["order_ceiling"] == pytest.approx(7.5)


def test_window_refuses_an_order_that_is_merely_above_six():
    """p = 6.2 clears the old policy bound but matches neither 2k+1 nor [2k+1-s, 2k+1+s]."""
    levels = _ladder_from_relative_differences([1.0 / 73.5, 1.0 / 73.5 ** 2])  # p ~ 6.2
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=3)
    assert 6.0 < cert["observed_order_used"] < 6.5
    assert cert["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH
    assert cert["superconvergent"] is False


def test_window_refuses_an_order_above_the_ceiling():
    levels = _ladder_from_relative_differences([1.0 / 194.0, 1.0 / 194.0 ** 2])  # p ~ 7.6
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=3)
    assert cert["observed_order_used"] > 7.5
    assert cert["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH


def test_a_wrong_declaration_cannot_hide_a_superconvergent_ladder():
    """Declaring P1 (window [2.5, 3.5]) does not accept a p ~ 7 ladder."""
    levels = _ladder_from_relative_differences([1.0 / 128.0, 1.0 / 128.0 ** 2])
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=1)
    assert cert["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH


# ------------------------------------------- the window cannot be bought
def test_a_non_contracting_ladder_cannot_buy_the_window():
    """Four levels, local orders ~ (-1, 7): the finest triple alone must not certify.

    This is the noise signature the guard exists for -- one level pair happens to
    give a high order while an earlier pair does not contract at all.
    """
    # realised differences: d1 = 0.5 (weak contraction), d2 = 2.0 (the ladder
    # *grows* -> local order -2), d3 = 2/128 (the finest triple alone gives p = 7)
    levels = _ladder_from_relative_differences(
        [0.5, 2.0, 2.0 / 128.0], ncells=(16, 32, 64, 128))
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=3)
    orders = [o["observed_order"] for o in cert["orders"]]
    assert orders[0] is not None and orders[0] < 0.0      # ladder is not contracting
    assert cert["observed_order_used"] > 6.5              # the finest triple looks fine
    assert cert["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH
    assert cert["superconvergent"] is False


# ------------------------------------- gate validity still has the last word
def test_gate_validity_still_vetoes_a_superconvergent_ladder():
    """A window-accepted order whose band eats the gate stays ``WITHHELD_BAND_...``."""
    # p = log2(512/4) = 7.0 inside the P3 window, but d_fine = 4 gives
    # U = 100*1.25*4/(2**7 - 1) = 3.94 % > tau/3 = 1 % (share 1.31 of the 3 % gate)
    levels = _ladder_from_relative_differences([512.0, 4.0])
    cert = cr.certify(levels, tolerance_percent=3.0, design_order=3)
    assert 6.5 < cert["observed_order_used"] < 7.5     # the window did open
    assert cert["band_share_of_gate"] > 1.0
    assert cert["verdict"] == cr.WITHHELD_BAND_CONSUMES_GATE
    assert cert["gate_usable"] is False


# ------------------------------------------------- physical ladder (solver)
@pytest.fixture(scope="module")
def advection_sine_p3_levels():
    """P3 advection ladder [16, 32, 64] from the production kernel (no limiter)."""
    from tensordg import advection_1d

    levels = []
    for n in (16, 32, 64):
        _, coeff, _ = advection_1d.solve_advection(
            ncells=n, degree=3, final_time=1.0, cfl=0.08, limiter=None,
            time_integrator="rk4")
        levels.append({"ncells": int(n), "field": coeff[:, 0].detach().cpu().numpy().astype(float)})
    return levels


def test_physical_p3_ladder_is_certified_only_with_a_declared_design_order(advection_sine_p3_levels):
    """The measured defect on a real ladder, and its resolution.

    ``p_obs ~ 7.1`` -- above ``P_MAX`` but inside the P3 window -- and the
    certified profile's true error must stay inside its own band, i.e. accepting
    the ladder must not cost band coverage.
    """
    plain = cr.certify(advection_sine_p3_levels, tolerance_percent=3.0)
    declared = cr.certify(advection_sine_p3_levels, tolerance_percent=3.0, design_order=3)

    assert plain["verdict"] == cr.WITHHELD_ORDER_TOO_HIGH
    assert declared["verdict"] == cr.CERTIFIED
    assert declared["superconvergent"] is True
    assert 5.0 < declared["observed_order_used"] < 7.5
    assert declared["band_percent"] < 1e-6          # a machine-level band
    assert declared["band_share_of_gate"] < cr.U_SHARE_MAX

    # truth: exact cell averages of u0(x - t) with u0 = sin(2 pi x), u_x = c u_t -> u = sin(2 pi (x - t))
    n_ref = int(declared["reference_ncells"])
    exact = _cell_averages(lambda x: np.sin(2.0 * np.pi * (x - 1.0)), n_ref)
    true_percent = 100.0 * cr.rel_l1(declared["reference"], exact)
    assert true_percent < declared["band_percent"], (
        f"band {declared['band_percent']:.3e} % does not cover true error {true_percent:.3e} %")
    # and the certified profile beats the raw finest level it replaces
    n_fine = int(advection_sine_p3_levels[-1]["ncells"])
    raw_percent = 100.0 * cr.rel_l1(
        advection_sine_p3_levels[-1]["field"],
        _cell_averages(lambda x: np.sin(2.0 * np.pi * (x - 1.0)), n_fine))
    assert true_percent < raw_percent


def test_judge_uses_the_declared_window_end_to_end(advection_sine_p3_levels):
    """``judge`` must inherit the widened classification, not silently WITHHOLD."""
    cert = cr.certify(advection_sine_p3_levels, tolerance_percent=3.0, design_order=3)
    n_fine = int(advection_sine_p3_levels[-1]["ncells"])
    verdict = cr.judge(_cell_averages(lambda x: np.sin(2.0 * np.pi * (x - 1.0)), n_fine), cert)
    assert verdict["decision"] == cr.PASS
    assert verdict["reference_verdict"] == cr.CERTIFIED