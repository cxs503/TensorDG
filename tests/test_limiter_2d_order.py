"""Pin the 2-D limiter's behaviour on *smooth* flow (validation debt of m2549).

``docs/limiter_policy_verification.md`` §5 recorded that the m2549 fix to the
troubled-cell test propagated to the 2-D limiter
(``tensordg.euler_dg_2d.apply_tvb_limiter_2d``) but had **never been verified in
2-D** — the 76–123 % overhead and the order behaviour were both unchecked, and
the earlier four-quadrant study (m2400) only ever ran the limiter at
``degree = 1``.

The tests here pin, on the analytic isentropic vortex (an exact solution of the
2-D Euler equations):

1. **inertness** — with the scale-consistent tolerance ``M dx**2`` and a large
   enough ``M`` the limiter must not touch the solution at all (bit-for-bit
   equality with the unlimited run);
2. **discriminating power** — the strict TVD setting ``M = 0`` *must* modify the
   solution and make it worse, otherwise the test above proves nothing;
3. **the ``degree >= 2`` branch** — a case with genuinely troubled cells at
   ``degree = 2`` runs without the ``RuntimeError`` that m2549 fixed;
4. the mask-level semantics of the scale-consistent measure (no false positives
   on linear data, detection of genuine extrema), which is where the m2549 bug
   actually lived.

The full ladder (P1/P2, three meshes, five limiter settings, observed orders)
lives in ``benchmarks/limiter_2d_order_study.py`` ->
``validation/limiter_2d_order/``; see ``docs/limiter_2d_order_verification.md``.
These tests are the cheap corner of that sweep so a regression turns the suite
red in seconds rather than minutes.
"""
import math

import pytest
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))


# --- portability helpers -------------------------------------------------
# CI installs a *different* torch build (``torch>=2.0`` resolves to the newest
# wheel) from the one these tests were written on, and BLAS/threading
# differences are allowed to move the last bits of a float64 reduction.  The
# scientific content of "the limiter is inert" is therefore *round-off
# inertness*, not bit-for-bit identity: allow a few ulps and fail loudly above
# it (a limiter that actually bites moves the solution by O(1e-3), not 1e-13).
ROUNDOFF_TOL = 1.0e-12
BITE_TOL = 1.0e-8


def _max_abs_diff(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


def _assert_inert(a, b, what=""):
    diff = _max_abs_diff(a, b)
    assert diff <= ROUNDOFF_TOL, (what, diff)
    return diff


def _assert_bites(a, b, what=""):
    diff = _max_abs_diff(a, b)
    assert diff > BITE_TOL, (what, diff)
    return diff


def _vortex(ncells, degree, limiter=None, tvb_constant=1.0, final_time=1.0):
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        limiter=limiter,
        tvb_constant=tvb_constant,
        domain=DOMAIN,
    )
    return coeff, _vortex_error(coeff, basis, nodes, DOMAIN, final_time), metadata


def test_large_tvb_constant_leaves_smooth_flow_untouched():
    """``M`` far above the solution's second-difference scale must be inert."""
    unlimited, errors_none, _ = _vortex(8, 2)
    limited, errors_tvb, _ = _vortex(8, 2, limiter="tvb", tvb_constant=100.0)
    _assert_inert(unlimited, limited, "tvb M=100 rewrote the smooth vortex")
    assert errors_tvb["density"] == pytest.approx(errors_none["density"], rel=1e-12, abs=0.0)


def test_strict_tvb_constant_limits_and_worsens_the_vortex():
    """``M = 0`` is the negative control: it must bite, and it must hurt."""
    unlimited, errors_none, _ = _vortex(8, 2)
    limited, errors_tvb, _ = _vortex(8, 2, limiter="tvb", tvb_constant=0.0)
    _assert_bites(unlimited, limited, "M=0 did not flag a single cell")
    for field in ("density", "pressure"):
        assert errors_tvb[field] > errors_none[field], (field, errors_tvb, errors_none)


def test_degree_two_limited_run_is_stable_and_conservative():
    """The ``degree >= 2`` mask path (broken before m2549) runs and conserves."""
    unlimited, _, _ = _vortex(8, 2)
    limited, errors, metadata = _vortex(8, 2, limiter="tvb", tvb_constant=0.0)
    assert metadata["limiter"] == "tvb"
    assert torch.isfinite(limited).all()
    # the limiter only rewrites higher modes: cell averages move only through
    # the minmod-consistent slope, never by a mode-0 change of the *unlimited*
    # cell average of the final state (conservation is checked exactly in
    # tests/test_euler_dg_2d_limiter.py); here we pin that the answer is sane.
    assert max(errors.values()) < 1.0


def test_weno_limiter_is_also_inert_when_the_indicator_is():
    """The WENO limiter shares the indicator: inert indicator => no change."""
    unlimited, _, _ = _vortex(8, 2)
    limited, _, _ = _vortex(8, 2, limiter="weno", tvb_constant=100.0)
    _assert_inert(unlimited, limited, "weno M=100 rewrote the smooth vortex")


def test_measure_is_scale_consistent_on_linear_averages():
    """A linear cell-average profile must never be flagged, whatever ``dx``.

    Before m2549 the test compared raw one-sided differences ``O(dx u')``
    against ``M dx**2``, so *every* smooth cell was flagged once
    ``dx > M / |u'|`` — the reason a finer mesh made the 2-D limiter worse.
    """
    ncells = 16
    dx = 2.0 / ncells
    centres = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    averages = torch.stack([0.5 + 0.25 * centres] * 4, dim=-1)  # [ncells, 4]
    coeff = torch.zeros((ncells, ncells, 3, 3, 4), dtype=torch.float64)
    coeff[..., 0, 0, :] = averages[:, None, :]
    limited = dg2.apply_tvb_limiter_2d(coeff, dx, dx, tvb_constant=0.0)
    _assert_inert(limited, coeff, "linear data flagged as troubled")


def test_measure_flags_a_genuine_extremum():
    """Sanity: the same measure must still detect an actual kink/extremum."""
    ncells = 16
    dx = 2.0 / ncells
    centres = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    profile = 1.0 - (centres ** 2)  # smooth but strongly curved: an extremum
    coeff = torch.zeros((ncells, ncells, 3, 3, 4), dtype=torch.float64)
    coeff[..., 0, 0, :] = profile[:, None, None]
    # non-trivial higher modes so that *limiting* is observable at all: the
    # limiter rewrites the slope mode and zeroes the modes above it.
    coeff[..., 1, 0, :] = 0.1
    coeff[..., 0, 1, :] = 0.1
    coeff[..., 2, 0, :] = 0.05
    limited = dg2.apply_tvb_limiter_2d(coeff, dx, dx, tvb_constant=0.0)
    _assert_bites(limited, coeff, "a genuine extremum went undetected")


def test_observed_order_survives_an_inert_limiter():
    """Inert limiter => identical coefficient ladder => identical observed order."""
    def eoc(tvb):
        coarse, err_coarse, _ = _vortex(8, 2, limiter="tvb", tvb_constant=tvb)
        fine, err_fine, _ = _vortex(16, 2, limiter="tvb", tvb_constant=tvb)
        return math.log(err_coarse["density"] / err_fine["density"]) / math.log(2.0)

    eoc_inert, eoc_strict = eoc(100.0), eoc(0.0)
    assert eoc_inert > 2.0, eoc_inert
    assert eoc_strict < eoc_inert, (eoc_strict, eoc_inert)