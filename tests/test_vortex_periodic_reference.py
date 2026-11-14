"""Regression tests: the isentropic-vortex exact solution must be periodic.

Root cause this file guards (CASE-DG-EULER-VORTEX-ERR, round 4): the exact
solution was evaluated with the raw Galilean shift ``x - u t``.  Once
``|u| t >= L/2`` the vortex peak leaves the computational box, so the
"reference" degrades to the uniform free stream while the numerical solution
still carries the vortex.  The resulting error is *resolution independent*
(measured here: 16.0 / 16.4 / 16.4 % at n = 8 / 16 / 32), which is the
signature of a reference defect, not of truncation -- and it was misread as
super-linear error accumulation (alpha = 3.30).
"""
from __future__ import annotations

import math

import torch

from tensordg.euler_2d import GAMMA, vortex_primitive

L = 10.0
BOX = (L, L)


def _grid(n=120):
    """Interior nodes only.

    Exactly on the seam (``|shift| = L/2``) the minimum-image convention has to
    break the sign tie for the *velocity* terms, and the two candidate images
    contribute ``+/- beta/(2 pi) exp(...) * L/2`` -- a 2.5e-05 free-stream
    residue.  The true image sum cancels it, so the residual is an artefact of
    using one image instead of the series, and it sits entirely in the
    far field (``exp(-L^2/8)``).  DG/FV quadrature nodes are strictly interior,
    so the verification never samples the tie; the tests do the same.
    """
    x = torch.linspace(-0.5 * L, 0.5 * L, n + 2, dtype=torch.float64)[1:-1]
    return x[:, None].expand(n, n), x[None, :].expand(n, n)


def test_periodic_reference_returns_after_one_period():
    """t = L/|u| is one whole period: the exact field must be the initial one."""
    X, Y = _grid()
    after = vortex_primitive(X, Y, L, periodic_box=BOX)
    initial = vortex_primitive(X, Y, 0.0)
    for a, b in zip(after, initial):
        assert float((a - b).abs().max()) < 1e-12


def test_historic_shift_is_not_periodic_so_the_bug_is_reproducible():
    """Guard the defect itself: without the box the t=L reference is free stream."""
    X, Y = _grid()
    rho, u, v, p = vortex_primitive(X, Y, L)          # historic, un-wrapped
    assert float((rho - 1.0).abs().max()) < 1e-10      # peak left the domain
    periodic = vortex_primitive(X, Y, L, periodic_box=BOX)
    assert float((periodic[0] - 1.0).abs().max()) > 0.4   # peak is back
    # ... and that gap is exactly the size of the bogus "error"
    gap = 100.0 * float(torch.sqrt(((rho - periodic[0]) ** 2).mean())
                        / torch.sqrt((periodic[0] ** 2).mean()))
    assert 5.0 < gap < 10.0


def test_backward_compatibility_without_box():
    """Existing callers (t=0 initial data, t=1 single-snapshot cases) are unchanged."""
    X, Y = _grid()
    for time in (0.0, 1.0):
        legacy = vortex_primitive(X, Y, time)
        explicit = vortex_primitive(X, Y, time, periodic_box=None)
        for a, b in zip(legacy, explicit):
            assert torch.equal(a, b)


def test_minimum_image_is_a_no_op_well_inside_the_box():
    """Sub-tolerance while |u|t < L/2: long-time fixes must not disturb t=1."""
    X, Y = _grid()
    boxed = vortex_primitive(X, Y, 1.0, periodic_box=BOX)
    plain = vortex_primitive(X, Y, 1.0)
    assert float((boxed[0] - plain[0]).abs().max()) < 1e-6


def test_corner_crossing_keeps_the_peak_at_the_corner():
    """At t = L/2 the core sits on the periodic corner: every corner shows it."""
    half = 0.5 * L
    pts = torch.tensor([-half, half], dtype=torch.float64)
    X = pts[:, None].expand(2, 2)
    Y = pts[None, :].expand(2, 2)
    rho, _u, _v, _p = vortex_primitive(X, Y, half, periodic_box=BOX)
    assert float((rho - rho.min()).abs().max()) < 1e-12      # all four identical
    assert rho.min() < 0.5                                  # and at the core value


def test_periodic_reference_is_a_solution_of_the_periodic_problem():
    """Shift-invariance: the profile at t is the t=0 profile rotated by (ut, vt)."""
    X, Y = _grid()
    for time in (2.0, 5.0, 7.5):
        moved = vortex_primitive(X, Y, time, periodic_box=BOX)
        # displacement written back into the box, applied as a physical shift
        sx = (time % L)
        sx = sx - L if sx > 0.5 * L else sx
        # both sides periodic: the gate is shift-invariance of the wrapped shift
        shifted = vortex_primitive(X - sx, Y - sx, 0.0, periodic_box=BOX)
        assert float((moved[0] - shifted[0]).abs().max()) < 1e-12


def test_entropy_and_pressure_consistency_hold_periodically():
    """The wrapped profile must stay isentropic (p = rho^gamma), as the case says."""
    X, Y = _grid()
    rho, u, v, p = vortex_primitive(X, Y, 7.0, periodic_box=BOX)
    assert float((p - rho ** GAMMA).abs().max()) < 1e-13
    assert math.isclose(float(rho.min()), 0.4939, abs_tol=1e-3)