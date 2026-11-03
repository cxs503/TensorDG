"""Validation tests for the TVB / WENO limiters against analytic references.

These tests encode the *dimensional* consistency of the limiter, which is the
easiest thing to get wrong in modal DG: with the normalised Legendre basis on
``[-1, 1]`` the cell-average difference across one cell is ``O(dx * du/dx)``
while the mode-1 coefficient is ``(dx / 2) * du/dx`` (LeVeque 2002, ch. 6;
Cockburn & Shu 1989 for the TVB criterion).  A factor-of-two slip here makes
the limiter *increase* the error instead of reducing it, so every assertion
below is anchored on an analytic value rather than on a previous output.
"""
import numpy as np
import torch

from tensordg.limiters import (
    _troubled_cell_indicator,
    apply_tvb_limiter,
    apply_weno_limiter,
)

DTYPE = torch.float64


def _gauss_cell_averages(fn, ncells, domain=(0.0, 1.0), order=10):
    lower, upper = domain
    dx = (upper - lower) / ncells
    centres = lower + (np.arange(ncells) + 0.5) * dx
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = centres[:, None] + 0.5 * dx * nodes[None, :]
    averages = 0.5 * np.sum(weights[None, :] * fn(points), axis=1)
    return centres, averages, dx


def _modal(averages, slope, dx):
    """Modal tensor (ncells, 3, 1): mode-0 = average, mode-1 = (dx/2) * slope."""
    ncells = len(averages)
    coeff = torch.zeros(ncells, 3, 1, dtype=DTYPE)
    coeff[:, 0, 0] = torch.as_tensor(np.asarray(averages), dtype=DTYPE)
    if slope is not None:
        coeff[:, 1, 0] = torch.as_tensor(0.5 * dx * np.asarray(slope), dtype=DTYPE)
    return coeff


def test_tvb_minmod_slope_uses_half_dx_scaling():
    """The limited mode-1 coefficient must equal (dx/2) * minmod slope.

    The cell-average sequence below is built by hand so that the minmod
    selection is unambiguous: at cell 2 the left difference is ``2h`` and the
    right difference is ``h``, so ``minmod = h`` and the limited physical slope
    is ``h / dx``.  The modal coefficient is then ``(dx / 2) * h / dx = h / 2``.
    Using the raw difference (``h``) would double the reconstructed slope.
    """
    h, dx = 0.3, 0.1
    averages = [0.0, 0.0, 2.0 * h, 3.0 * h, 3.0 * h, 3.0 * h, 3.0 * h, 3.0 * h]
    coeff = _modal(averages, None, dx)

    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0)
    assert bool(troubled[2]), "cell 2 must be flagged (minmod != left difference)"

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    expected = 0.5 * h
    assert abs(float(limited[2, 1, 0]) - expected) < 1e-14, (
        f"mode-1 = {float(limited[2, 1, 0])!r}, expected {expected!r}"
    )
    # Cell average untouched.
    assert abs(float(limited[2, 0, 0]) - 2.0 * h) < 1e-15


def test_tvb_limited_slope_matches_analytic_derivative_on_quadratic():
    """Away from smooth extrema the limited slope must match du/dx to O(dx).

    Cells straddling the smooth extremum are excluded on purpose: there the
    minmod of a positive and a negative difference is identically zero, so the
    limited slope is zero *by construction* (that is the TVB clipping, not an
    error).  For the remaining troubled cells the limited physical slope must
    reproduce the analytic derivative.
    """
    fn = lambda x: 1.0 + 2.0 * x - 3.0 * x ** 2
    dfn = lambda x: 2.0 - 6.0 * x
    for ncells, tolerance_percent in ((40, 15.0), (160, 8.0)):
        centres, averages, dx = _gauss_cell_averages(fn, ncells)
        coeff = _modal(averages, dfn(centres), dx)
        troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
        limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0).numpy()
        interior = troubled.copy()
        interior[0] = interior[-1] = False
        selected = interior & (np.abs(dfn(centres)) >= 0.25 * np.abs(dfn(centres)).max())
        assert selected.sum() > 0, "window must contain troubled cells away from the extremum"
        expected = 0.5 * dx * dfn(centres)
        relative = np.abs(limited[selected, 1, 0] - expected[selected]) / np.abs(
            expected[selected]
        )
        worst = 100.0 * float(relative.max())
        assert worst < tolerance_percent, (
            f"n={ncells}: limited slope deviates {worst:.1f}% from du/dx"
        )


def test_tvb_slope_consistency_improves_with_refinement():
    """The consistency error of the limited slope must vanish as dx -> 0."""
    fn = lambda x: 1.0 + 2.0 * x - 3.0 * x ** 2
    dfn = lambda x: 2.0 - 6.0 * x
    errors = []
    for ncells in (40, 160):
        centres, averages, dx = _gauss_cell_averages(fn, ncells)
        coeff = _modal(averages, dfn(centres), dx)
        troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
        limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0).numpy()
        interior = troubled.copy()
        interior[0] = interior[-1] = False
        interior &= np.abs(dfn(centres)) >= 0.25 * np.abs(dfn(centres)).max()
        expected = 0.5 * dx * dfn(centres)
        errors.append(
            float(
                np.mean(
                    np.abs(limited[interior, 1, 0] - expected[interior])
                    / np.abs(expected[interior])
                )
            )
        )
    ratio = errors[0] / errors[-1]
    assert ratio > 2.0, f"refinement by 4x only reduced the error {ratio:.2f}x"


def test_tvb_reconstruction_stays_strictly_inside_neighbour_averages():
    """Monotonicity preservation: p(x_{i±1/2}) must stay strictly between neighbours."""
    fn = lambda x: 1.0 + 2.0 / (1.0 + np.exp(-20.0 * (x - 0.5)))
    centres, averages, dx = _gauss_cell_averages(fn, 80)
    coeff = _modal(averages, None, dx)
    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0).numpy()
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0).numpy()

    avg = limited[:, 0, 0]
    pad = np.concatenate([avg[:1], avg, avg[-1:]])
    u_im1, u_ip1 = pad[:-2], pad[2:]
    a1 = limited[:, 1, 0]
    interior = troubled.copy()
    interior[0] = interior[-1] = False
    assert interior.sum() > 0
    slack = np.minimum(u_ip1 - (avg + a1), (avg - a1) - u_im1)[interior]
    assert (slack > 0.0).all(), (
        "limited reconstruction touches a neighbour cell average "
        f"(min slack {slack.min():.3e})"
    )


def test_tvb_detector_releases_smooth_extremum_at_second_derivative_scale():
    """Cockburn-Shu tolerance: M > |u''| should leave a smooth extremum unlimited."""
    fn = lambda x: 1.0 + 0.5 * np.sin(2.0 * np.pi * x)
    centres, averages, dx = _gauss_cell_averages(fn, 160)
    coeff = _modal(averages, None, dx)
    strict = int(_troubled_cell_indicator(coeff, dx, tvb_constant=0.0).sum())
    relaxed = int(_troubled_cell_indicator(coeff, dx, tvb_constant=1000.0).sum())
    assert strict > 0, "strict TVD test must flag the smooth extrema"
    assert relaxed == 0, f"M=1000 still flags {relaxed} cells"


def test_affine_data_is_untouched_by_both_limiters():
    """Affine data: detector silent and coefficients bit-identical."""
    fn = lambda x: 1.0 + 2.0 * x
    centres, averages, dx = _gauss_cell_averages(fn, 60)
    coeff = _modal(averages, np.full(60, 2.0), dx)
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0)
        assert not bool(troubled.any())
        assert torch.equal(limiter(coeff, dx, tvb_constant=0.0), coeff)


def test_limiters_never_touch_cell_averages():
    """Mode-0 (mass) preservation on random data, both limiters."""
    torch.manual_seed(7)
    coeff = torch.randn(40, 3, 3, dtype=DTYPE)
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        limited = limiter(coeff, 0.025, tvb_constant=0.5)
        assert torch.allclose(limited[:, 0], coeff[:, 0], atol=0.0)
        assert torch.equal(limited[:, 0], coeff[:, 0])