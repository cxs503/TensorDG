"""Tests for functional-level (feature) certification, ``certify_functional``.

The engine is asserted against **exact** synthetic ladders -- a functional whose
continuum limit and convergence order are known by construction -- so a PASS
here means the method recovers a known answer, not that it reproduces a number
somebody once wrote down.  The refusal paths are asserted too: a ladder that
stagnates, oscillates, sits on the round-off floor, or converges at an order
outside the admissible window must be WITHHELD, because a band computed from a
non-asymptotic ladder is a fabricated uncertainty.
"""
from __future__ import annotations

import math

import pytest

from tensordg.certified_reference import (
    CERTIFIED,
    FAIL,
    PASS,
    PASS_MARGINAL,
    WITHHELD,
    WITHHELD_ORDER_TOO_HIGH,
    WITHHELD_ORDER_TOO_LOW,
    WITHHELD_ROUNDOFF_FLOOR,
    WITHHELD_STAGNANT,
    WITHHELD_TOO_FEW_LEVELS,
    WITHHELD_ZERO_FUNCTIONAL,
    certify_functional,
    judge_functional,
)

NC = (800, 1600, 3200, 6400)


def exact_ladder(f0: float, c: float, p: float, n=NC) -> list[float]:
    """``F(h) = f0 + c * h**p`` sampled on the nesting ladder."""
    return [f0 + c * (1.0 / nc) ** p for nc in n]


# ------------------------------------------------------------------ exactness
@pytest.mark.parametrize("p", [1.0, 1.5, 2.0])
def test_recovers_known_order_and_limit(p):
    cert = certify_functional(exact_ladder(2.0, 3.0, p), tolerance_percent=3.0)
    assert cert["verdict"] == CERTIFIED
    assert cert["observed_order"] == pytest.approx(p, rel=1e-6)
    assert cert["certified_value"] == pytest.approx(2.0, rel=1e-9)
    assert cert["band_percent"] < 0.05          # exact power law -> band ~ d_fine/10
    assert cert["gate_usable"] is True


def test_sign_of_the_correction_is_preserved():
    """A functional converging from below must not be extrapolated upwards."""
    below = certify_functional(exact_ladder(2.0, -3.0, 1.5), tolerance_percent=3.0)
    above = certify_functional(exact_ladder(2.0, +3.0, 1.5), tolerance_percent=3.0)
    assert below["certified_value"] == pytest.approx(2.0, rel=1e-9)
    assert above["certified_value"] == pytest.approx(2.0, rel=1e-9)
    # the two ladders are on opposite sides of the limit
    assert below["values"][-1] < 2.0 < above["values"][-1]


def test_band_is_the_roache_index_of_the_finest_difference():
    from tensordg.certified_reference import FS_GCI

    cert = certify_functional(exact_ladder(1.0, 0.5, 1.5), tolerance_percent=3.0)
    p = cert["observed_order"]
    f_fine = cert["values"][-1]
    d_fine = abs(cert["values"][-1] - cert["values"][-2])
    expected = FS_GCI * d_fine / (2.0 ** p - 1.0)
    assert cert["band_absolute"] == pytest.approx(expected, rel=1e-9)
    assert cert["band_percent"] == pytest.approx(100.0 * expected / abs(cert["certified_value"]),
                                                 rel=1e-9)


def test_a_converging_ladder_has_a_smaller_band_than_a_slow_one():
    fast = certify_functional(exact_ladder(2.0, 1.0, 3.0), tolerance_percent=3.0)
    slow = certify_functional(exact_ladder(2.0, 1.0, 1.0), tolerance_percent=3.0)
    assert fast["band_percent"] < slow["band_percent"]


# ------------------------------------------------------------------ refusals
def test_too_few_levels_is_refused():
    cert = certify_functional([1.0, 1.01], tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_TOO_FEW_LEVELS
    assert cert["certified_value"] is None


def test_stagnant_ladder_is_refused():
    # non-monotone differences: a ladder that oscillates carries no order
    cert = certify_functional([1.0, 1.01, 1.005, 1.012], tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_STAGNANT
    assert cert["band_percent"] is None


def test_a_frozen_ladder_is_refused_as_roundoff():
    cert = certify_functional([1.0, 1.0 + 1e-6, 1.0 + 1e-3, 1.0 + 1e-3], tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_ROUNDOFF_FLOOR


def test_roundoff_floor_is_refused():
    base = 1.0
    cert = certify_functional([base, base + 1e-15, base + 2e-15, base + 3e-15],
                              tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_ROUNDOFF_FLOOR


def test_order_too_low_is_refused():
    # differences barely shrink: p ~ 0.1
    vals = [1.0]
    for k in range(3):
        vals.append(vals[-1] + 0.1 * (0.5 ** k) ** 0.1)
    cert = certify_functional(vals, tolerance_percent=3.0)
    assert cert["verdict"] == WITHHELD_ORDER_TOO_LOW


def test_order_too_high_is_refused_without_a_declared_design_order():
    # differences shrink by 2**7 per doubling -> p = 7 > P_MAX = 6
    d = [1.0, 1.0 / 128.0, 1.0 / 128.0 ** 2]
    vals = [1.0 + sum(d[k:]) for k in range(3)] + [1.0]
    cert = certify_functional(vals, tolerance_percent=3.0)
    assert cert["observed_order"] == pytest.approx(7.0, rel=1e-9)
    assert cert["verdict"] == WITHHELD_ORDER_TOO_HIGH


def test_vanishing_functional_is_refused_instead_of_dividing_by_zero():
    # a functional that converges *to zero*: relative band is unbounded by
    # construction, so it must be refused rather than reported as infinite
    cert = certify_functional([1e-2, 2.5e-3, 6.25e-4, 1.5625e-4], tolerance_percent=3.0)
    assert cert["verdict"] in (WITHHELD_ZERO_FUNCTIONAL, WITHHELD_ORDER_TOO_LOW)


def test_band_consuming_the_gate_is_not_usable():
    """A band wider than a third of the gate cannot certify a verdict there."""
    # slow, low-order ladder at moderate resolution: big band
    cert = certify_functional(exact_ladder(2.0, 3.0, 0.6, n=(50, 100, 200, 400)),
                              tolerance_percent=3.0)
    assert cert["verdict"] == CERTIFIED
    assert cert["band_share_of_gate"] > 1.0 / 3.0
    assert cert["gate_usable"] is False
    verdict = judge_functional(2.0, cert, tolerance_percent=3.0)
    assert verdict["decision"] == WITHHELD


# ------------------------------------------------------------------ judging
def test_judge_decisions_are_band_aware():
    cert = certify_functional(exact_ladder(2.0, 3.0, 1.5), tolerance_percent=3.0)
    band = cert["band_percent"]
    assert judge_functional(2.0, cert)["decision"] == PASS
    assert judge_functional(2.0 * (1.0 + 2.0 * band / 100.0), cert)["decision"] == PASS
    # inside the band's shadow: passes the tolerance but not resolvably
    marginal = judge_functional(2.0 * (1.0 + (3.0 - band / 2.0) / 100.0), cert)
    assert marginal["decision"] == PASS_MARGINAL
    assert judge_functional(2.0 * 1.05, cert)["decision"] == FAIL


def test_judge_ignores_a_reference_that_is_not_certified():
    cert = certify_functional([1.0, 1.0 + 1e-6, 1.0 + 1e-3, 1.0 + 1e-3], tolerance_percent=3.0)
    out = judge_functional(1.0, cert, tolerance_percent=3.0)
    assert out["decision"] == WITHHELD
    assert out["error_percent"] is None


def test_judge_re_evaluates_the_share_rule_at_the_asked_tolerance():
    """A reference certified for a loose gate is not silently usable at a tight one."""
    cert = certify_functional(exact_ladder(2.0, 1.0, 0.6, n=(400, 800, 1600, 3200)),
                              tolerance_percent=3.0)
    assert cert["verdict"] == CERTIFIED
    assert cert["gate_usable"] is True
    tight = judge_functional(2.0, cert, tolerance_percent=0.5)
    assert tight["decision"] == WITHHELD