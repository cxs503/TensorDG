"""Regression gate: Thacker (1981) long-time behaviour and P2 order.

Locks in the findings of the 2026-11-11 closure of the verification debt left
by ``docs/thacker_validation_gap_corrected.md`` (m6377, long-time ~12 % error
marked "unresolved") and ``docs/thacker_validation_fix.md`` (m7536, fix only
shown at ``t = 0.3 s``, P1 only):

* with the consistent configuration (exact bed slope + exact boundary flux)
  the planar sloshing solution is reproduced at ``t = 0.5 s`` *and*
  ``t = 1.0 s`` with an error three to four orders of magnitude below the 3 %
  acceptance gate;
* ``P2`` is *not* worse than ``P1``: the exact solution of this benchmark is a
  quadratic in ``x`` for the depth, so a consistent ``P2`` discretisation
  represents it to machine precision;
* the free-surface error previously reported for these runs is the ``O(dx^2)``
  constant vertical offset of ``reconstruct_fields``' default trapezoidal bed
  average -- it is independent of the polynomial degree and therefore masks the
  order gain.

The numbers come from ``validation/swe_thacker/run_thacker_longtime.py`` so the
test and the published artifact share one code path.
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "validation", "swe_thacker"))

import run_thacker_longtime as thk  # noqa: E402

T_CLOSE = 0.5     # the horizon at which m6377 measured the ~12 % gap


@pytest.fixture(scope="module")
def runs():
    cache = {}

    def get(n, p, t):
        key = (n, p, t)
        if key not in cache:
            cache[key] = thk.run_case(n, p, t)
        return cache[key]

    return get


@pytest.mark.parametrize("t_end", [0.5, 1.0])
def test_long_time_error_far_below_three_percent(runs, t_end):
    """P1 N=40 at t=0.5 s and t=1.0 s: <0.01 % (the 3 % gate is not the bar)."""
    r = runs(40, 1, t_end)
    assert r["err_eta_correct_pct"] < 0.01, r
    assert r["err_u_pct"] < 0.01, r


def test_p1_depth_second_order(runs):
    """P1 depth error decreases by ~4x per mesh doubling (order 2)."""
    coarse, fine = runs(20, 1, T_CLOSE), runs(40, 1, T_CLOSE)
    assert fine["err_h_pct"] < 0.25 * coarse["err_h_pct"], (coarse, fine)


def test_p2_is_exact_not_worse_than_p1(runs):
    """P2 represents the planar solution: machine-precision depth, |u| tiny."""
    r = runs(20, 2, T_CLOSE)
    assert r["err_h_pct"] < 1e-4, r
    assert r["err_eta_correct_pct"] < 1e-2, r
    assert r["err_u_pct"] < 1e-3, r
    # ... and strictly more accurate than P1 on the same mesh.
    p1 = runs(20, 1, T_CLOSE)
    assert r["err_eta_correct_pct"] < 0.1 * p1["err_eta_correct_pct"], (r, p1)


def test_default_bed_postprocessing_dominates_and_is_degree_blind(runs):
    """The reported free-surface error is the O(dx^2) bed-average offset.

    At N=40 the default (trapezoidal) bed estimate is at least 20x larger than
    the solver error, and its value is the *same* for P1 and P2 -- the signature
    of a post-processing artifact, not a discretisation error.
    """
    p2 = runs(40, 2, T_CLOSE)
    p1 = runs(40, 1, T_CLOSE)
    assert p2["err_eta_default_pct"] == pytest.approx(p1["err_eta_default_pct"],
                                                      rel=1e-3)
    assert p2["err_eta_default_pct"] > 20 * p2["err_eta_correct_pct"]
    # Predicted constant offset (dx^2/6) z_b'' relative to the free surface:
    # (h0/L^2) dx^2/6 / RMS(eta).  RMS(eta) ~ 1 inside the sloshing range, and
    # it is not stored in the result dict, hence the 8 % band (the two effects
    # left out -- the O(dx^2) h-average term and RMS(eta) != 1 -- are < 5 %).
    predicted = (thk.H0 / thk.L ** 2) * (2 * thk.L / 40) ** 2 / 6 * 100
    assert p2["err_eta_default_pct"] == pytest.approx(predicted, rel=0.08)


def test_volume_conserved_to_roundoff(runs):
    """Exact boundary flux removes the +6.6 % mass drift of the legacy closure."""
    for n, p in ((20, 1), (40, 2)):
        assert runs(n, p, T_CLOSE)["vol_rel_err"] < 1e-10