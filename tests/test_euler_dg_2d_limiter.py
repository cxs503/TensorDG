"""Regression tests for the 2-D TVB limiter (mission m2549).

The two defects these tests lock down:

1. ``_troubled_1d`` in ``tensordg/euler_dg_2d.py`` used the *old*, dimensionally
   inconsistent troubled-cell measure (``max|d_left|, |d_right|`` -- O(dx u') --
   against ``M dx**2``) that mission m2500 had already replaced in the 1-D
   limiter.  Every smooth extremum was flagged as soon as ``dx > 1/M``, the
   flagged fraction grew under refinement and ``tvb_constant`` had no effect on
   whether a feature was released.
2. ``apply_tvb_limiter_2d`` raised ``RuntimeError`` for ``degree >= 2`` in the
   branch that removes the higher modes of a troubled direction, because the
   boolean mask was broadcast against the mode axes.  No test ever ran the 2-D
   limiter above P1, so the branch was dead code.

See ``benchmarks/verify_limiter_2d.py`` and ``docs/limiter_2d_verification.md``
for the before/after numbers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_dg_2d import (
    _initial_coefficients,
    _troubled_1d,
    apply_tvb_limiter_2d,
    legendre_operators,
)
from tensordg.tvb import troubled_from_differences
from benchmarks.verify_euler_dg_2d import _vortex_error
from tensordg.euler_1d import GAMMA
from tensordg.limiters import _troubled_cell_indicator, apply_tvb_limiter

DTYPE = torch.float64


def _wave(x, y):
    wave = torch.sin(math.pi * x) * torch.cos(0.5 * math.pi * y)
    zeros = torch.zeros_like(x)
    return 1.0 + 0.05 * wave, zeros, zeros, 1.0 + 0.05 * wave


def _project(ncells, degree, domain=(-1.0, 1.0)):
    nodes, weights, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))
    lo, hi = domain
    dx = (hi - lo) / ncells
    centres = lo + (torch.arange(ncells, dtype=DTYPE) + 0.5) * dx
    coeff = _initial_coefficients(
        _wave, centres, centres, dx, dx, nodes, weights, basis,
        torch.device("cpu"), DTYPE,
    )
    return coeff, dx, nodes, basis


def _curvature(coeff, dx, axis=0):
    averages = coeff[..., 0, 0, :]
    second = (
        torch.roll(averages, -1, axis) - 2.0 * averages + torch.roll(averages, 1, axis)
    ) / (dx * dx)
    return float(second.abs().max())


def _safe_constant(coeff, dx, factor=4.0):
    """TVB constant comfortably above the measured curvature on both axes."""
    return factor * max(_curvature(coeff, dx, 0), _curvature(coeff, dx, 1))


def _reference_curvature():
    """max |d2u/dx2| measured on a fine mesh (the coarsest meshes smear it)."""
    coeff, dx, _, _ = _project(256, 2)
    return _curvature(coeff, dx)


# ---------------------------------------------------------------------------
# 1-D shared test
# ---------------------------------------------------------------------------

def test_shared_troubled_test_releases_smooth_data_above_curvature():
    dx = 0.05
    x = (torch.arange(200, dtype=DTYPE) + 0.5) * dx
    values = torch.sin(2.0 * math.pi * x)
    d_left = values - torch.roll(values, 1, 0)
    d_right = torch.roll(values, -1, 0) - values
    curvature = float(
        (torch.roll(values, -1, 0) - 2.0 * values + torch.roll(values, 1, 0)).abs().max()
        / dx ** 2
    )
    strict, _ = troubled_from_differences(d_left, d_right, dx, 0.0)
    released, _ = troubled_from_differences(d_left, d_right, dx, 2.0 * curvature)
    assert int(strict.sum()) > 0
    assert int(released.sum()) == 0


# ---------------------------------------------------------------------------
# 2-D troubled-cell detector
# ---------------------------------------------------------------------------

def test_detector_releases_smooth_wave_on_every_mesh():
    for degree in (1, 2):
        for ncells in (8, 16, 32):
            coeff, dx, _, _ = _project(ncells, degree)
            tvb = _safe_constant(coeff, dx)
            for axis in (0, 1):
                flagged, _ = _troubled_1d(coeff[..., 0, 0, :], axis, dx, tvb)
                assert int(flagged.sum()) == 0, (degree, ncells, axis)


def test_detector_fraction_stays_bounded_under_refinement():
    """Pre-fix this fraction grew to 0.44; it must stay a bounded set."""
    fractions = []
    for ncells in (8, 16, 32, 64):
        coeff, dx, _, _ = _project(ncells, 2)
        flagged, _ = _troubled_1d(coeff[..., 0, 0, :], 0, dx, 1.0)
        fractions.append(int(flagged.sum()) / (ncells * ncells))
    assert max(fractions) <= 0.10, fractions
    assert fractions[-1] <= 2.0 * fractions[0], fractions


def test_detector_still_flags_a_jump():
    ncells, degree = 24, 2
    coeff, dx, _, _ = _project(ncells, degree)
    averages = torch.zeros(ncells, ncells, 4, dtype=DTYPE)
    averages[..., 0] = torch.where(
        torch.arange(ncells)[:, None] < ncells // 2, 1.0, 0.125
    )
    averages[..., 3] = torch.where(
        torch.arange(ncells)[:, None] < ncells // 2, 1.0, 0.1
    )
    flagged, _ = _troubled_1d(averages, 0, dx, 1.0)
    assert int(flagged.sum()) > 0
    # y-invariant data must not be flagged along y
    flagged_y, _ = _troubled_1d(averages, 1, dx, 1.0)
    assert int(flagged_y.sum()) == 0


def test_detector_never_flags_boundary_rows_on_outflow_axis():
    ncells = 20
    averages = torch.zeros(ncells, ncells, 4, dtype=DTYPE)
    averages[..., 0] = torch.arange(ncells, dtype=DTYPE)[:, None] ** 2
    flagged, _ = _troubled_1d(averages, 0, 1.0 / ncells, 0.0, periodic=False)
    assert not bool(flagged[0].any())
    assert not bool(flagged[-1].any())
    wrapped, _ = _troubled_1d(averages, 0, 1.0 / ncells, 0.0, periodic=True)
    assert bool(wrapped[0].any())  # wrap-around sees the far side as a jump


# ---------------------------------------------------------------------------
# 2-D limiter itself
# ---------------------------------------------------------------------------

def _jump_coefficients(ncells, degree):
    """y-invariant step in x at the mid plane, higher modes zero."""
    coeff = torch.zeros(ncells, ncells, degree + 1, degree + 1, 4, dtype=DTYPE)
    left = torch.arange(ncells)[:, None] < ncells // 2
    coeff[..., 0, 0, 0] = torch.where(left, 1.0, 0.125)
    coeff[..., 0, 0, 3] = torch.where(left, 1.0, 0.1)
    return coeff


def test_limiter_degree_two_does_not_raise_and_preserves_cell_averages():
    """Regression for the mask-broadcast RuntimeError at degree >= 2."""
    for degree in (1, 2, 3):
        ncells = 16
        coeff = _jump_coefficients(ncells, degree)
        averages = coeff[..., 0, 0, :].clone()
        try:
            limited = apply_tvb_limiter_2d(
                coeff, 1.0 / ncells, 1.0 / ncells, tvb_constant=1.0
            )
        except RuntimeError as error:  # pragma: no cover - the m2549 defect
            raise AssertionError(f"limiter raised at degree {degree}: {error}")
        assert limited.shape == coeff.shape
        assert torch.equal(limited[..., 0, 0, :], averages)
        flagged, _ = _troubled_1d(averages, 0, 1.0 / ncells, 1.0)
        if degree >= 2:
            higher_x = limited[..., 2:, :, :]
            inside = higher_x[flagged]
            assert float(inside.abs().max()) == 0.0


def test_limiter_is_inert_above_the_data_curvature():
    for degree in (1, 2):
        for ncells in (8, 16, 32):
            coeff, dx, _, _ = _project(ncells, degree)
            tvb = _safe_constant(coeff, dx)
            limited = apply_tvb_limiter_2d(coeff, dx, dx, tvb_constant=tvb)
            assert torch.equal(limited, coeff), (degree, ncells)


def test_limited_reconstruction_stays_inside_neighbour_bounds_on_a_jump():
    ncells, degree = 24, 2
    coeff = _jump_coefficients(ncells, degree)
    limited = apply_tvb_limiter_2d(coeff, 1.0 / ncells, 1.0 / ncells, tvb_constant=1.0)
    nodes, _, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))
    state = torch.einsum("xyklv,ak,bl->xyabv", limited, basis, basis)
    rho = state[..., 0]
    assert float(rho.min()) >= 0.125 - 1e-12
    assert float(rho.max()) <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# end-to-end: the limiter must not cost design order on smooth flow
# ---------------------------------------------------------------------------

def _vortex_density_error(coeff, final_time):
    nodes, _, basis, _, _ = legendre_operators(1, 3)
    ncells = coeff.shape[0]
    dx = 10.0 / ncells
    centres = -5.0 + (torch.arange(ncells, dtype=DTYPE) + 0.5) * dx
    xq = centres[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = centres[None, :, None, None] + 0.5 * dx * nodes[None, None, None, :]
    exact = dg2.vortex_primitive(xq, yq, final_time, 5.0, (1.0, 1.0))
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = state[..., 0]
    weights = torch.as_tensor(
        np.polynomial.legendre.leggauss(3)[1], dtype=DTYPE
    )
    weight = weights[:, None] * weights[None, :]
    return float(
        torch.sqrt(
            (((rho - exact[0]) ** 2) * weight).sum()
            / ((exact[0] ** 2) * weight).sum()
        )
    )


def test_vortex_evolution_keeps_second_order_with_limiter():
    """P1 vortex, final_time 0.3: limited run must track the unlimited one."""
    meshes = (12, 16, 24)
    final_time = 0.3
    rows = []
    for ncells in meshes:
        errors = {}
        for label, limiter in (("none", None), ("tvb", "tvb")):
            _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
                ncells=ncells,
                degree=1,
                final_time=final_time,
                limiter=limiter,
                tvb_constant=50.0,
            )
            errors[label] = _vortex_density_error(coeff, final_time)
        rows.append(errors)

    orders = [
        math.log(previous["tvb"] / current["tvb"]) / math.log(h_fine / h_coarse)
        for (h_coarse, previous), (h_fine, current) in zip(
            zip(meshes, rows), zip(meshes[1:], rows[1:])
        )
    ]
    assert min(orders) > 1.7, orders
    for row in rows:
        assert row["tvb"] <= 1.10 * row["none"], row


# ----------------------------------------------------------------------
DEVICE = torch.device("cpu")


def _legendre(degree, quadrature=None):
    return dg2.legendre_operators(degree, quadrature or 2 * degree + 1, DEVICE, DTYPE)


def _smooth_1d_coefficients(ncells, degree, seed=0, amplitude=0.3):
    """Modal coefficients of a smooth, strictly convex/concave 1-D profile."""
    generator = torch.Generator().manual_seed(seed)
    coeff = torch.zeros(ncells, degree + 1, 4, dtype=DTYPE)
    coeff[:, 0, 0] = 1.0 + amplitude * torch.sin(
        2.0 * np.pi * (torch.arange(ncells, dtype=DTYPE) + 0.5) / ncells
    )
    coeff[:, 0, 1] = 0.2 * torch.cos(
        2.0 * np.pi * (torch.arange(ncells, dtype=DTYPE) + 0.5) / ncells
    )
    coeff[:, 0, 3] = coeff[:, 0, 0] ** GAMMA
    if degree >= 1:
        coeff[:, 1:, :] = 0.02 * torch.randn(
            ncells, degree, 4, generator=generator, dtype=DTYPE
        )
    return coeff


def _extrude(coeff_1d):
    """Place 1-D modal coefficients into a one-cell-thick 2-D array."""
    ncells, nmodes, nvar = coeff_1d.shape
    coeff_2d = torch.zeros(ncells, 1, nmodes, nmodes, nvar, dtype=DTYPE)
    coeff_2d[:, 0, :, 0, :] = coeff_1d
    return coeff_2d


def test_higher_modes_are_zeroed_without_broadcast_error():
    """The m2546 defect: limiting a P2 cell raised a broadcast RuntimeError."""
    degree, ncells = 2, 9
    coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=DTYPE)
    # Saw-tooth cell averages: every interior cell is a local extremum, so the
    # M=0 detector flags the whole row and the higher-mode branch is executed.
    averages = torch.tensor(
        [1.0, 1.2, 2.2, 3.0, 2.4, 1.1, 0.9, 1.0, 1.0], dtype=DTYPE
    )
    coeff[..., 0, 0, 0] = averages[:, None]
    coeff[..., 0, 0, 3] = coeff[..., 0, 0, 0]
    coeff[..., 1:, :, :] = 0.1  # non-zero higher modes everywhere

    limited = dg2.apply_tvb_limiter_2d(coeff, 1.0, 1.0, tvb_constant=0.0)

    assert torch.isfinite(limited).all()
    # cell averages are conserved exactly
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])
    # flagged cells (2, 3, 5, 6, 7) are reduced to a linear reconstruction,
    # while the monotone cells (1, 4) keep their higher modes untouched
    for cell in (2, 3, 5, 6, 7):
        assert float(limited[cell, :, 2:, :, :].abs().max()) == 0.0, cell
    for cell in (1, 4):
        assert float(limited[cell, :, 2:, :, :].abs().max()) == 0.1, cell
    # and its slope obeys the minmod bound |c1| <= max(|d_left|, |d_right|)/2
    differences = (averages[1:] - averages[:-1]).abs()
    local_scale = torch.maximum(differences[:-1], differences[1:])
    assert float(limited[1:-1, :, 1, 0, 0].abs().max()) <= 0.5 * float(
        local_scale.max()
    )
    assert float(limited[1:-1, :, 1, 0, 0].abs().max()) > 0.0


def test_mode_one_slope_is_the_minmod_increment_at_an_extremum():
    """At a discrete minimum the minmod slope vanishes, so mode-1 is zeroed."""
    coeff = torch.zeros(7, 1, 2, 2, 4, dtype=DTYPE)
    coeff[..., 0, 0, 0] = torch.tensor(
        [1.0, 0.5, 0.2, 0.8, 1.4, 2.0, 2.4], dtype=DTYPE
    )[:, None]
    coeff[..., 0, 0, 3] = coeff[..., 0, 0, 0]
    coeff[..., 1, 0, :] = 0.3

    limited = dg2.apply_tvb_limiter_2d(coeff, 1.0, 1.0, tvb_constant=0.0)

    assert float(limited[2, 0, 1, 0, 0].abs()) == 0.0  # u_2 is the local min
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])


def test_detector_agrees_with_the_one_dimensional_limiter():
    """Single source of truth: extruded 2-D limiting equals the 1-D limiter."""
    ncells, degree = 12, 2
    coeff_1d = _smooth_1d_coefficients(ncells, degree)
    coeff_2d = _extrude(coeff_1d)
    spacing = 1.0 / ncells

    for tvb_constant in (0.0, 0.1, 1.0, 10.0):
        mask_1d = _troubled_cell_indicator(
            coeff_1d, spacing, tvb_constant=tvb_constant
        )
        mask_2d, _ = dg2._troubled_1d(
            coeff_2d[..., 0, 0, :], 0, spacing, tvb_constant, periodic=False
        )
        assert torch.equal(mask_1d, mask_2d[:, 0]), tvb_constant

        limited_1d = apply_tvb_limiter(
            coeff_1d, spacing, tvb_constant=tvb_constant
        )
        limited_2d = dg2.apply_tvb_limiter_2d(
            coeff_2d, spacing, spacing, tvb_constant=tvb_constant,
            periodic=(False, True),
        )
        assert torch.allclose(
            limited_2d[:, 0, 1, 0, :], limited_1d[:, 1, :], atol=1e-15
        ), tvb_constant
        assert torch.equal(limited_2d[..., 0, 0, :], coeff_2d[..., 0, 0, :])


def test_non_periodic_axis_leaves_boundary_cells_unlimited():
    """No wrapped neighbour on a non-periodic axis: boundary cells stay free."""
    ncells = 8
    averages = torch.zeros(ncells, 1, 4, dtype=DTYPE)
    averages[..., 0] = torch.tensor(
        [5.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 5.0], dtype=DTYPE
    )[:, None]
    averages[..., 3] = averages[..., 0]
    troubled, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, 0.0, periodic=False)
    assert not bool(troubled[0, 0]) and not bool(troubled[-1, 0])
    assert bool(troubled[1:-1, 0].any())


def test_large_tvb_constant_is_inert_on_a_smooth_vortex():
    """M is resolution/behaviour independent: a resolved smooth flow is untouched."""
    common = dict(ncells=8, degree=2, final_time=0.05, cfl=0.2)
    _, _, unlimited, _ = dg2.solve_euler_dg_2d_vortex(limiter=None, **common)
    _, _, limited, _ = dg2.solve_euler_dg_2d_vortex(
        limiter="tvb", tvb_constant=50.0, **common
    )
    assert float((limited - unlimited).abs().max()) < 1e-13


def test_shock_limiting_preserves_cell_averages_and_fires():
    """On a shock the limiter must fire, stay finite and conserve cell averages."""
    common = dict(ncells=64, degree=2, final_time=0.05, cfl=0.2)
    _, _, unlimited, _ = dg2.solve_euler_dg_2d_sod(limiter=None, **common)
    _, _, limited, metadata = dg2.solve_euler_dg_2d_sod(
        limiter="tvb", tvb_constant=0.0, **common
    )
    assert metadata["limiter"] == "tvb"
    assert torch.isfinite(limited).all()
    assert torch.isfinite(unlimited).all()

    troubled, _ = dg2._troubled_1d(
        limited[..., 0, 0, :], 0, 1.0 / 64, 0.0, periodic=False
    )
    assert int(troubled.sum()) > 0
    # the limiter really changed the high-order content (it is not a no-op)
    assert float((limited[:, :, 1:, :, :] - unlimited[:, :, 1:, :, :]).abs().max()) > 1e-6
    # the limited density stays non-negative at the cell-average level
    assert float(limited[..., 0, 0, 0].min()) > 0.0


def test_vortex_wrapper_accepts_tvb_constant():
    """The public vortex helper must expose the limiter tolerance."""
    x, y, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=6, degree=1, final_time=0.02, limiter="tvb", tvb_constant=3.0
    )
    assert metadata["limiter"] == "tvb"
    _, nodes, basis, _, _ = _legendre(1)
    domain = ((-5.0, 5.0), (-5.0, 5.0))
    errors = _vortex_error(coeff, basis, nodes, domain, 0.02)
    assert all(np.isfinite(value) for value in errors.values())
