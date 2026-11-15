"""Regression tests for the *two-sided* TVB troubled-cell indicator (m21413).

The indicator in :func:`tensordg.tvb.troubled_from_differences` compares the
distance of a one-sided cell-average difference from the minmod slope against
``M dx**2``.  Between missions m2500 and m21413 that distance was measured on
``d_left`` only, so it vanished identically in every cell with
``|d_left| <= |d_right|`` -- i.e. in the right-facing half of every front.  The
unlimited polynomial then survived those cells and its point values left the
exact data range (1.40 % overshoot at P1 on the CKS1998 square wave, 2.58 % on
N = 50, both above the 1 % gate of ``REPRO-DG-CKS1998``).

These tests pin the two properties that make the fix safe:

* a right-facing front is detected (the defect),
* locally affine cell averages are still released (no over-limiting), so the
  ``M`` semantics and the limited scheme's accuracy are untouched.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.advection_1d import solve_advection
from tensordg.tvb import minmod, troubled_from_differences

LEGENDRE = np.polynomial.legendre


def _square_wave(x):
    return ((x >= 0.25) & (x <= 0.75)).to(x.dtype)


def _nodal_envelope(coeff: torch.Tensor, npts: int = 41) -> tuple[float, float]:
    """Max/min of the piecewise polynomial over ``npts`` equally spaced nodes."""
    nmodes = coeff.shape[1]
    xi = torch.linspace(-1.0, 1.0, npts, dtype=coeff.dtype)
    basis = torch.stack(
        [torch.as_tensor(LEGENDRE.legval(xi.numpy(), np.eye(nmodes)[k])) for k in range(nmodes)],
        dim=1,
    )
    values = coeff @ basis.T
    return float(values.max()), float(values.min())


def _square_wave_case(ncells: int, degree: int):
    _, coeff, _ = solve_advection(
        ncells=ncells, degree=degree, final_time=1.0, cfl=0.2,
        initial=_square_wave, limiter="tvb", tvb_constant=0.0,
        time_integrator="ssp_rk3",
    )
    return coeff


def test_right_facing_front_is_flagged():
    """A cell whose left increment is the smaller one must still be detected."""
    # Cell averages of a right-facing front: 0, 0, 0.01, 0.05, 0.4, 0.9, 1, 1.
    avg = torch.tensor([0.0, 0.0, 0.01, 0.05, 0.4, 0.9, 1.0, 1.0],
                       dtype=torch.float64).reshape(-1, 1)
    d_left = avg - torch.roll(avg, 1, dims=0)
    d_right = torch.roll(avg, -1, dims=0) - avg
    # Replicate padding at the two ends, exactly as the 1-D limiter does.
    d_left[0] = 0.0
    d_right[-1] = 0.0
    troubled, slope = troubled_from_differences(d_left, d_right, 0.1, 0.0)
    troubled = troubled.reshape(-1)
    # Cell 3 (avg 0.05) has |d_left| = 0.04 < |d_right| = 0.35: the one-sided
    # indicator had deviation == 0 there and left it unlimited.
    assert troubled[3], "right-facing front cell was not flagged"
    # Cells 4 and 5 are the steep part of the same front.
    assert troubled[4] and troubled[5]
    # The two constant tail cells (padded differences are zero) stay released;
    # at M = 0 a cell with *any* non-zero variation is legitimately flagged.
    assert not troubled[0] and not troubled[7]
    # The returned slope is still the (bounded) minmod slope.
    assert torch.equal(slope, minmod(d_left, d_right))


def test_locally_affine_data_is_released():
    """Affine cell averages must never be flagged, hence never over-limited."""
    n = 32
    avg = torch.linspace(-1.0, 1.0, n, dtype=torch.float64).reshape(-1, 1)
    avg = 0.25 + 0.5 * avg                     # affine, non-trivial slope
    d_left = avg - torch.roll(avg, 1, dims=0)
    d_right = torch.roll(avg, -1, dims=0) - avg
    d_left[0] = 0.0
    d_right[-1] = 0.0
    for tvb_constant in (0.0, 1.0, 1e6):
        troubled, _ = troubled_from_differences(d_left, d_right, 0.05, tvb_constant)
        # End cells carry the replicate padding, not real neighbours: callers
        # mask or override them, so only the interior is asserted here.
        assert not troubled.reshape(-1)[1:-1].any(), tvb_constant


def test_two_sided_deviation_matches_numpy_reference():
    """Tensor kernel == explicit loop, including mixed-sign and zero increments."""
    rng = np.random.default_rng(7)
    d_left = torch.as_tensor(rng.normal(size=(9, 17, 3)))
    d_right = torch.as_tensor(rng.normal(size=(9, 17, 3)))
    d_left[0, :, 0] = 0.0                       # zero left increments
    spacing, tvb_constant = 0.125, 0.0
    troubled, slope = troubled_from_differences(d_left, d_right, spacing, tvb_constant)
    dev_ref = np.maximum(np.abs(d_left.numpy() - slope.numpy()),
                         np.abs(d_right.numpy() - slope.numpy())).max(axis=-1)
    scale = np.maximum(np.abs(d_left.numpy()), np.abs(d_right.numpy())).max(axis=-1)
    want = dev_ref > tvb_constant * spacing ** 2 + 1e-12 * np.clip(scale, 1.0, None)
    np.testing.assert_array_equal(troubled.numpy(), want)


@pytest.mark.parametrize("degree,ncells", [(1, 50), (1, 100), (2, 50), (2, 100)])
def test_square_wave_pointwise_overshoot_meets_one_percent_gate(degree, ncells):
    """CKS1998 square wave: no new pointwise extrema above the 1 % gate."""
    coeff = _square_wave_case(ncells, degree)
    high, low = _nodal_envelope(coeff)
    overshoot = 100.0 * max(0.0, high - 1.0, -low)
    assert overshoot <= 1.0, (degree, ncells, high, low, overshoot)


@pytest.mark.parametrize("degree", [1, 2])
def test_square_wave_cell_average_overshoot_meets_gate(degree):
    coeff = _square_wave_case(100, degree)
    avg = coeff[:, 0].numpy()
    overshoot = 100.0 * max(0.0, avg.max() - 1.0, -avg.min())
    assert overshoot <= 1.0, (degree, overshoot)


def test_limited_scheme_keeps_half_order_law():
    """The extra limiting must not change the CKS1998 limited-scheme order."""
    errors = []
    for ncells in (50, 100, 200):
        coeff = _square_wave_case(ncells, 1)
        centres = (np.arange(ncells) + 0.5) / ncells
        exact = np.where(((centres - 1.0) % 1.0 >= 0.25) & ((centres - 1.0) % 1.0 <= 0.75), 1.0, 0.0)
        errors.append(float(np.abs(coeff[:, 0].numpy() - exact).mean()))
    orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    # CKS1998 report ~0.5 for the limited scheme on a discontinuity; the
    # implemented scheme has always measured ~0.65 on this mesh ladder.
    assert all(order >= 0.5 for order in orders), (errors, orders)
    assert all(order <= 0.85 for order in orders), (errors, orders)