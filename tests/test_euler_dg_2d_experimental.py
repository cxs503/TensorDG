"""Regression pins for the experimental 2-D high-order Euler DG module.

The 2-D assembly is not verified yet, so the cross-code test is marked
``xfail(strict=True)``: the suite stays green today, and the moment the defect
documented in ``docs/verify_2d_euler_dg_attempt.md`` is fixed the test turns
into an XPASS which *fails* the suite and forces the pin to be removed.
"""
import pytest
import torch

from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_dg_2d import (
    _rhs,
    _axis_fluxes,
    _traces,
    cell_fields,
    legendre_operators,
    solve_euler_dg_2d_sod,
)
from tensordg.euler_1d import GAMMA


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


@pytest.mark.xfail(
    strict=True,
    reason="2-D face-flux projection collapses to zero; see docs/verify_2d_euler_dg_attempt.md",
)
def test_extruded_sod_reproduces_the_1d_dg_solution():
    """The 2-D solver must match the 1-D solver cell by cell for y-invariant data."""
    x1, coeff_1d, _ = solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
        ncells=64, degree=1, final_time=0.01, cfl=0.2,
    )
    _, _, coeff_2d, _ = solve_euler_dg_2d_sod(
        ncells=64, degree=1, final_time=0.01, cfl=0.2,
    )
    fields = cell_fields(coeff_2d)
    for column, key in ((0, "density"), (1, "velocity_x"), (2, "pressure")):
        reference = coeff_1d[:, 0, column]
        assert float((fields[key][:, 0] - reference).abs().max()) < 1e-10, key
