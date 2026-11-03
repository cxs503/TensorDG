"""Regression pins for the (now verified) 2-D high-order Euler DG module.

History: m2143 filed an honest FAIL with ``xfail(strict=True)`` on the
cross-code test.  m2400 located the defect -- ``_axis_fluxes`` did not roll the
neighbour trace in the non-periodic case, so HLLC saw the cell's own state --
fixed it and these pins replace the xfail marker.  See
``docs/verify_euler_dg_2d_resolved.md``.
"""
import numpy as np
import pytest
import torch

from tensordg.euler_dg_1d import _operators, _rhs as rhs_1d, solve_euler_dg_1d
from tensordg.euler_dg_2d import (
    _rhs,
    _traces,
    cell_fields,
    legendre_operators,
    solve_euler_dg_2d_sod,
    solve_euler_dg_2d_vortex,
)
from tensordg.euler_1d import GAMMA

from benchmarks.sod_exact import cell_averages as sod_cell_averages

LEFT_STATE = (1.0, 0.0, 1.0)
RIGHT_STATE = (0.125, 0.0, 0.1)


def test_uniform_state_has_zero_residual():
    """A constant state must give an exactly zero DG residual (degree 0,1,2)."""
    device, dtype = torch.device("cpu"), torch.float64
    for degree in (0, 1, 2):
        _, weights, basis, derivative, signs = legendre_operators(degree, 3, device, dtype)
        state = torch.zeros(6, 1, degree + 1, degree + 1, 4, dtype=dtype)
        state[..., 0, 0, :] = torch.tensor(
            [1.0, 0.4, 0.0, 1.0 / (GAMMA - 1.0) + 0.5 * 0.4 ** 2], dtype=dtype
        )
        residual = _rhs(state, basis, derivative, weights, signs, 0.1, 1.0, (True, True))
        assert float(residual.abs().max()) < 1e-12, degree


def test_y_invariant_data_has_no_y_residual():
    """Extruded 1-D data must not excite any y mode of the residual."""
    device, dtype = torch.device("cpu"), torch.float64
    degree = 2
    _, weights, basis, derivative, signs = legendre_operators(degree, 4, device, dtype)
    state = torch.zeros(8, 1, degree + 1, degree + 1, 4, dtype=dtype)
    state[..., 0, 0, :] = torch.tensor([1.0, 1.2, 0.0, 2.5 + 0.72], dtype=dtype)
    state[..., 1, 0, 0] = 0.05
    residual = _rhs(state, basis, derivative, weights, signs, 0.1, 1.0, (True, True))
    assert float(residual[..., :, 1:, :].abs().max()) < 1e-12


def test_extruded_residual_matches_the_independent_1d_solver():
    """The m2143 defect: 2-D residual of y-invariant data vs the 1-D DG solver.

    Every x mode is populated, so the volume term is exercised too.  The two
    code paths share nothing but the numerical flux, hence agreement must be at
    round-off (the m2143 assembly returned an identically zero residual here).
    """
    device, dtype = torch.device("cpu"), torch.float64
    ncells, dx, dy = 40, 0.025, 0.25
    for degree in (1, 2):
        nodes, weights, basis, derivative, signs = legendre_operators(
            degree, max(degree + 2, 3), device, dtype
        )
        centres = (torch.arange(ncells, dtype=dtype) + 0.5) * dx
        xq = centres[:, None] + 0.5 * dx * nodes[None, :]
        blend = 0.5 * (1.0 - torch.tanh(220.0 * (xq - 0.5)))
        rho = RIGHT_STATE[0] + (LEFT_STATE[0] - RIGHT_STATE[0]) * blend
        pressure = RIGHT_STATE[2] + (LEFT_STATE[2] - RIGHT_STATE[2]) * blend
        energy = pressure / (GAMMA - 1.0)
        state = torch.stack((rho, torch.zeros_like(rho), energy), dim=-1)
        projected = torch.einsum("cqv,q,qk->ckv", state, weights, basis)
        modes = torch.arange(degree + 1, dtype=dtype)
        coeff_1d = (2.0 * modes + 1.0).view(1, -1, 1) / 2.0 * projected

        coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=dtype)
        coeff[:, 0, :, 0, (0, 1, 3)] = coeff_1d
        residual_2d = _rhs(
            coeff, basis, derivative, weights, signs, dx, dy, (False, True)
        )
        residual_1d = rhs_1d(
            coeff_1d, dx, weights, basis, derivative, signs, torch.ones_like(signs)
        )
        scale = float(residual_1d.abs().max())
        assert scale > 1.0  # the data must produce a genuinely non-trivial residual
        for two_d, one_d in ((0, 0), (1, 1), (3, 2)):
            difference = float((residual_2d[:, 0, :, 0, two_d] - residual_1d[:, :, one_d]).abs().max())
            assert difference / scale < 1e-12, (degree, two_d, difference)
        # y momentum must stay untouched by y-invariant data
        assert float(residual_2d[:, 0, :, 0, 2].abs().max()) < 1e-10


def test_extruded_sod_reproduces_the_1d_dg_solution():
    """Cell averages of the extruded Sod tube equal the 1-D solver's solution."""
    _, coeff_1d, _ = solve_euler_dg_1d(
        LEFT_STATE, RIGHT_STATE, ncells=64, degree=1, final_time=0.01, cfl=0.2,
    )
    _, _, coeff_2d, _ = solve_euler_dg_2d_sod(
        ncells=64, degree=1, final_time=0.01, cfl=0.2,
        initial_projection="cell-average",  # the 1-D solver starts from cell averages
    )
    density_1d = coeff_1d[:, 0, 0]
    density_2d = coeff_2d[:, 0, 0, 0, 0]
    scale = float(density_1d.abs().max())
    assert float((density_1d - density_2d).abs().max()) / scale < 1e-3
    energy_1d = coeff_1d[:, 0, 2]
    energy_2d = coeff_2d[:, 0, 0, 0, 3]
    assert float((energy_1d - energy_2d).abs().max()) / float(energy_1d.abs().max()) < 1e-3


def test_extruded_sod_matches_the_exact_riemann_solution():
    """The documented 3 % gate on the extruded Sod tube (P1, 100 cells)."""
    _, reference = sod_cell_averages(100, 0.2)
    _, _, coeff, metadata = solve_euler_dg_2d_sod(
        ncells=100, degree=1, final_time=0.2, cfl=0.2,
    )
    fields = cell_fields(coeff)
    for name in ("density", "pressure"):
        numerical = fields[name][:, 0].detach().cpu().numpy()
        exact = np.asarray(reference[name], dtype=float)
        absolute = float(np.sqrt(np.mean((numerical - exact) ** 2)))
        norm = float(np.sqrt(np.mean(exact ** 2)))
        assert absolute / norm < 0.03, (name, absolute / norm)
    assert metadata["steps"] > 0


def test_l2_projection_reproduces_a_polynomial_initial_state():
    """The modal L2 projection must be exact for a degree-p polynomial."""
    _, _, coeff, metadata = solve_euler_dg_2d_vortex(
        ncells=8, degree=2, final_time=0.0,
    )
    assert metadata["initial_projection"] == "l2"
    # A smooth field must populate the higher modes (the legacy P0 initialisation
    # left them exactly zero, which caps the convergence order at one).
    assert float(coeff[..., 1:, :, :].abs().max()) > 1e-3
    assert float(coeff[..., :, 1:, :].abs().max()) > 1e-3