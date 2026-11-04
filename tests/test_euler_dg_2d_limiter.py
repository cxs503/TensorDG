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
from tensordg.limiters import _troubled_cell_indicator

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
TUNED_M = 10.0


def _averages_from_profile(profile, ncells, ny=4, axis=0):
    values = torch.as_tensor(profile, dtype=DTYPE)
    if axis == 0:
        return values.view(ncells, 1, 1).expand(ncells, ny, 1).clone()
    return values.view(1, ncells, 1).expand(ny, ncells, 1).clone()


def test_two_dimensional_detector_matches_the_one_dimensional_indicator():
    ncells = 64
    x = (np.arange(ncells) + 0.5) / ncells
    dx = 1.0 / ncells
    for tvb_constant in (1.0, 10.0):
        for profile in (
            1.0 + 0.2 * np.sin(2.0 * np.pi * x),          # smooth periodic wave
            np.where(x < 0.5, 1.0, 0.125),                # Sod jump
        ):
            averages = _averages_from_profile(profile, ncells)
            mask_2d, slope_2d = dg2._troubled_1d(averages, 0, dx, tvb_constant)
            coeff_1d = torch.zeros(ncells, 2, 1, dtype=DTYPE)
            coeff_1d[:, 0, 0] = torch.as_tensor(profile, dtype=DTYPE)
            mask_1d = _troubled_cell_indicator(coeff_1d, dx, tvb_constant)
            # the 1-D indicator freezes the two outermost cells (no full
            # stencil); the 2-D axis detector wraps periodically.
            interior = slice(1, ncells - 1)
            assert torch.equal(mask_2d[:, 0][interior], mask_1d[interior]), (
                tvb_constant, profile[:4]
            )


def test_two_dimensional_detector_is_mesh_independent_on_a_smooth_wave():
    """With a scale-consistent detector the flagged count does not grow with h."""
    tvb_constant = TUNED_M
    counts = []
    for ncells in (20, 40, 80, 160):
        x = (np.arange(ncells) + 0.5) / ncells
        profile = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
        averages = _averages_from_profile(profile, ncells, ny=2)
        mask, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, tvb_constant)
        counts.append(int(mask.sum()))
    assert counts == [0, 0, 0, 0], counts


def test_shock_still_flags_cells_in_both_directions():
    """The repaired detector must remain aggressive at a genuine jump."""
    ncells = 32
    x = (np.arange(ncells) + 0.5) / ncells
    profile = np.where(x < 0.5, 1.0, 0.125)
    averages = _averages_from_profile(profile, ncells, ny=3, axis=1)
    mask_x, _ = dg2._troubled_1d(averages, 1, 1.0 / ncells, 0.0)
    mask_y, _ = dg2._troubled_1d(averages, 0, 1.0 / ncells, 0.0)
    assert int(mask_x.sum()) > 0
    assert int(mask_y.sum()) == 0


def _discontinuous_coefficients(degree, nx=10, ny=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    coeff = 0.05 * torch.rand(
        nx, ny, degree + 1, degree + 1, 4, generator=generator, dtype=DTYPE
    )
    coeff[:, :, 0, 0, 0] = torch.where(
        torch.arange(nx, dtype=DTYPE).view(nx, 1) < nx // 2,
        torch.tensor(1.0, dtype=DTYPE),
        torch.tensor(0.125, dtype=DTYPE),
    ).expand(nx, ny)
    coeff[:, :, 0, 0, 1] = 0.3 * torch.sign(coeff[:, :, 0, 0, 2])
    coeff[:, :, 0, 0, 3] = torch.where(
        torch.arange(nx, dtype=DTYPE).view(nx, 1) < nx // 2,
        torch.tensor(2.5, dtype=DTYPE),
        torch.tensor(0.25, dtype=DTYPE),
    ).expand(nx, ny)
    return coeff


def test_limiter_runs_for_every_degree_and_keeps_the_cell_average():
    """D2 regression: a flagged P>=2 state used to raise a broadcast error."""
    for degree in (1, 2, 3):
        coeff = _discontinuous_coefficients(degree)
        limited = dg2.apply_tvb_limiter_2d(coeff, 0.1, 0.1, 0.0)
        assert torch.isfinite(limited).all(), degree
        # conservation: mode 0 is untouched
        assert torch.equal(limited[:, :, 0, 0, :], coeff[:, :, 0, 0, :]), degree
        # some cell must have been limited
        changed = (limited - coeff).abs().reshape(
            coeff.shape[0], coeff.shape[1], -1
        ).max(-1).values > 0
        assert int(changed.sum()) > 0, degree


def test_limiter_flags_along_both_axes_without_broadcast_error():
    """Axis-1 limiting of a P3 state used to raise a RuntimeError."""
    degree = 3
    nx, ny = 8, 8
    coeff = torch.zeros(nx, ny, degree + 1, degree + 1, 4, dtype=DTYPE)
    coeff[..., 0, 0, 0] = 1.0
    coeff[::2, :, 0, 0, 0] = 1.5      # variation along x
    coeff[:, ::2, 0, 0, 0] = 0.5      # variation along y
    coeff[..., 1, 1, :] = 0.2
    limited = dg2.apply_tvb_limiter_2d(coeff, 0.1, 0.25, 0.0)
    assert torch.isfinite(limited).all()
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])
    assert float(limited[..., 2:, :, :].abs().max()) == 0.0


def _vortex_errors(ncells, degree, limiter=None, tvb_constant=None,
                   final_time=1.0):
    from benchmarks.verify_euler_dg_2d import _vortex_error

    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=final_time, limiter=limiter,
        **({} if tvb_constant is None else {"tvb_constant": tvb_constant}),
    )
    return _vortex_error(coeff, basis, nodes, ((-5.0, 5.0), (-5.0, 5.0)),
                         final_time)


def test_vortex_wrapper_forwards_the_tvb_constant():
    """M must reach the limiter: M=1 limits, a large M is inert (defect D3)."""
    reference = _vortex_errors(8, 2, limiter=None)["density"]
    aggressive = _vortex_errors(8, 2, limiter="tvb", tvb_constant=0.0)["density"]
    inert = _vortex_errors(8, 2, limiter="tvb", tvb_constant=1e6)["density"]
    assert aggressive != reference
    assert inert == reference, (inert, reference)


def test_limiter_is_exactly_inert_on_a_resolved_smooth_vortex():
    for degree in (1, 2):
        nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
        _, _, smooth, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=12, degree=degree, final_time=1.0)
        _, _, limited, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=12, degree=degree, final_time=1.0, limiter="tvb",
            tvb_constant=TUNED_M)
        assert torch.equal(smooth, limited), degree


def test_two_dimensional_limiter_preserves_design_order():
    """P1 on the smooth vortex must keep EOC ~ 2 with the limiter enabled."""
    errors = [
        _vortex_errors(ncells, 1, limiter="tvb", tvb_constant=TUNED_M)["density"]
        for ncells in (8, 12, 16)
    ]
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(16.0 / 12.0)
        if i else math.log(errors[0] / errors[1]) / math.log(12.0 / 8.0)
        for i in range(len(errors) - 1)
    ]
    assert min(orders) > 1.7, (orders, errors)
    assert errors[-1] < 5e-3, errors


def test_limiter_keeps_a_two_dimensional_shock_positive():
    _, _, coeff, _ = dg2.solve_euler_dg_2d_sod(
        ncells=60, degree=2, final_time=0.2, limiter="tvb", tvb_constant=0.5)
    rho = coeff[:, 0, 0, 0, 0]
    momentum = coeff[:, 0, 0, 0, 1]
    energy = coeff[:, 0, 0, 0, 3]
    pressure = (dg2.GAMMA - 1.0) * (energy - 0.5 * momentum ** 2 / rho)
    assert bool(torch.isfinite(coeff).all())
    assert float(rho.min()) > 0.0
    assert float(pressure.min()) > 0.0
