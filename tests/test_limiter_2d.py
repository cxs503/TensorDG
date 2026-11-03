"""Regression tests for the two-dimensional limiters.

The 2-D TVB/WENO limiters are exercised on the properties that matter for
correctness: the cell average is never modified, the troubled-cell test uses
one-sided stencils on outflow boundaries instead of wrapping across the mesh,
and the dimension-by-dimension WENO reduction reproduces the verified 1-D
limiter on y-invariant data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from benchmarks.sod_exact import cell_averages as sod_cell_averages
from tensordg.euler_dg_2d import (
    _troubled_1d,
    apply_tvb_limiter_2d,
    apply_weno_limiter_2d,
    cell_fields,
    legendre_operators,
    solve_euler_dg_2d_sod,
)
from tensordg.limiters import apply_tvb_limiter, apply_weno_limiter

LEFT_STATE = (1.0, 0.0, 1.0)
RIGHT_STATE = (0.125, 0.0, 0.1)


def _jump_coefficients(ncells=16, degree=2):
    """Modal coefficients of a cell-wise Riemann jump in density and pressure."""
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) / ncells
    averages = torch.zeros(ncells, 1, 4, dtype=torch.float64)
    averages[..., 0] = torch.where(
        x < 0.5, torch.full_like(x, LEFT_STATE[0]), torch.full_like(x, RIGHT_STATE[0])
    )[:, None]
    averages[..., 3] = torch.where(
        x < 0.5, torch.full_like(x, LEFT_STATE[2]), torch.full_like(x, RIGHT_STATE[2])
    )[:, None]
    coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=torch.float64)
    coeff[..., 0, 0, :] = averages
    # a non-trivial sub-cell profile so that limiting has something to remove
    coeff[..., 1, 0, :] = 0.4 * averages
    coeff[..., 2, 0, :] = -0.2 * averages
    return coeff


def test_troubled_detector_does_not_wrap_across_an_outflow_boundary():
    """Cell 0 of the Sod data is uniform; the wrapped test invents a jump."""
    coeff = _jump_coefficients()
    averages = coeff[..., 0, 0, :]
    wrapped, _ = _troubled_1d(averages, 0, 1.0 / coeff.shape[0], 1.0, True)
    one_sided, _ = _troubled_1d(averages, 0, 1.0 / coeff.shape[0], 1.0, False)
    assert wrapped[0, 0].item() is True
    assert one_sided[0, 0].item() is False
    # the real discontinuity is detected by both variants
    assert wrapped[8, 0].item() and one_sided[8, 0].item()


@pytest.mark.parametrize("limiter", [apply_tvb_limiter_2d, apply_weno_limiter_2d])
def test_cell_averages_are_never_modified(limiter):
    coeff = _jump_coefficients()
    limited = limiter(coeff, 1.0 / coeff.shape[0], 0.25, 1.0, (False, True))
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])
    # the limiter must actually do something on this data
    assert float((limited - coeff).abs().max()) > 1e-3


@pytest.mark.parametrize("limiter", [apply_tvb_limiter_2d, apply_weno_limiter_2d])
def test_affine_data_is_untouched(limiter):
    ncells, degree = 12, 2
    x = (np.arange(ncells) + 0.5) / ncells
    coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=torch.float64)
    for comp, slope in enumerate((0.7, 0.2, -0.4, 0.9)):
        coeff[..., 0, 0, comp] = torch.as_tensor(1.0 + slope * x, dtype=torch.float64)[:, None]
    limited = limiter(coeff, 1.0 / ncells, 0.25, 1.0, (False, True))
    assert torch.equal(limited, coeff)


@pytest.mark.parametrize("limiter", [apply_tvb_limiter_2d, apply_weno_limiter_2d])
def test_multi_row_meshes_are_supported(limiter):
    """Regression: the troubled mask was broadcastable only for ny == 1."""
    coeff = torch.zeros(6, 5, 3, 3, 4, dtype=torch.float64)
    torch.manual_seed(0)
    coeff[..., 0, 0, :] = torch.rand(6, 5, 4, dtype=torch.float64)
    coeff[..., 1, 0, :] = torch.rand(6, 5, 4, dtype=torch.float64)
    limited = limiter(coeff, 0.1, 0.1, 1.0, (False, False))
    assert limited.shape == coeff.shape
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])


def test_weno_2d_reduces_to_the_verified_1d_limiter():
    """On y-invariant data the dimension-by-dimension limiter is the 1-D one."""
    coeff = _jump_coefficients(ncells=16, degree=2)
    limited = apply_weno_limiter_2d(coeff, 1.0 / 16, 0.25, 1.0, (False, True))
    reference = apply_weno_limiter(
        coeff[:, 0, :, 0, :].clone(), 1.0 / 16, 1.0, limit_boundaries=True
    )
    assert float((limited[:, 0, :, 0, :] - reference).abs().max()) == 0.0


def test_1d_limiters_can_include_the_boundary_cells():
    """``limit_boundaries`` exposes the end cells that the default masks out."""
    coeff = torch.zeros(8, 3, 3, dtype=torch.float64)
    averages = torch.tensor(
        [[1.0, 0.0, 1.0], [1.1, 0.0, 1.1], [1.2, 0.0, 1.1], [1.3, 0.0, 1.2],
         [1.4, 0.0, 1.2], [1.5, 0.0, 1.3], [1.6, 0.0, 1.3], [1.2, 0.0, 1.1]],
        dtype=torch.float64,
    )
    coeff[:, 0] = averages
    coeff[:, 1] = 0.3 * averages
    masked = apply_tvb_limiter(coeff, 0.125, 1.0)
    exposed = apply_tvb_limiter(coeff, 0.125, 1.0, limit_boundaries=True)
    # the last cell is a local extremum in the cell averages
    assert torch.equal(masked[-1, 1], coeff[-1, 1])       # left unlimited
    assert float(exposed[-1, 1].abs().max()) < 1e-12      # limited on request


def test_weno_2d_preserves_a_smooth_stationary_profile():
    """A smooth steady state must survive one limiter application to round-off."""
    ncells, degree = 16, 2
    dx, dy = 1.0 / ncells, 0.25
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    y = torch.tensor([0.125], dtype=torch.float64)
    nodes, weights, basis, _, _ = legendre_operators(degree, max(degree + 2, 3))

    def initial(xx, yy):
        shape = torch.broadcast_shapes(xx.shape, yy.shape)
        rho = torch.broadcast_to(1.0 + 0.2 * torch.cos(math.pi * xx), shape)
        return rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.ones_like(rho)

    from tensordg.euler_dg_2d import _initial_coefficients

    coeff = _initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, "cpu", torch.float64)
    limited = apply_weno_limiter_2d(coeff, dx, dy, 1.0, (False, True))
    physical = x[:, None] + 0.5 * dx * nodes[None, :]
    exact = 1.0 + 0.2 * torch.cos(math.pi * physical)
    values = torch.einsum("xyklv,ak,bl->xyabv", limited, basis, basis)[..., 0, :, :, 0]
    assert float(torch.sqrt(((values[:, 0] - exact) ** 2).mean())) < 0.05
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])


def test_weno_limited_squeezed_sod_meets_the_three_percent_gate():
    """The 2-D WENO path must reproduce the exact Riemann solution below 3 %."""
    _, reference = sod_cell_averages(100, 0.2)
    _, _, coeff, metadata = solve_euler_dg_2d_sod(
        ncells=100, degree=1, final_time=0.2, cfl=0.2, limiter="weno"
    )
    assert metadata["limiter"] == "weno"
    fields = cell_fields(coeff)
    for name in ("density", "pressure"):
        numerical = fields[name][:, 0].detach().cpu().numpy()
        exact = np.asarray(reference[name], dtype=float)
        error = float(np.sqrt(np.mean((numerical - exact) ** 2))) / float(
            np.sqrt(np.mean(exact**2))
        )
        assert error < 0.03, (name, error)
    # the WENO reconstruction must not overshoot the Sod plateau density
    assert float(fields["density"].max()) < 1.001


def test_unknown_limiter_name_is_rejected():
    with pytest.raises(ValueError):
        solve_euler_dg_2d_sod(ncells=20, degree=1, final_time=0.001, limiter="muscl")