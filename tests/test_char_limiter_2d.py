"""Verification of the characteristic-variable TVB limiter for the 2-D Euler system.

Every assertion here is a property of the *mathematics*, not of a stored number:
the eigenvectors are checked against an autograd Jacobian of the flux, the
transform is checked to be an involution-free exact change of basis, and the
limiter is checked to conserve cell averages bit-for-bit and to be inert on data
the Cockburn--Shu test does not flag.  The quantitative comparison against the
componentwise limiter lives in ``benchmarks/char_limiter_2d.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

import tensordg.euler_dg_2d as e2d
from tensordg.char_limiter import (
    apply_characteristic_tvb_limiter_2d,
    euler_eigenvectors_2d,
)

GAMMA = 1.4


def _random_states(n: int = 17, seed: int = 20260101) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rho = 0.2 + 2.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    u = -3.0 + 6.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    v = -3.0 + 6.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    p = 0.1 + 5.0 * torch.rand(n, generator=generator, dtype=torch.float64)
    return e2d.conservative_state(rho, u, v, p)


@pytest.mark.parametrize("axis", [0, 1])
def test_eigenvectors_diagonalise_the_flux_jacobian(axis):
    """``R^{-1} A_n R = diag(u_n - a, u_n, u_n, u_n + a)`` with autograd ``A_n``."""
    state = _random_states()
    R, R_inv = euler_eigenvectors_2d(state, axis, GAMMA)
    # Independent Jacobian: autograd of the flux of each single state, so the
    # only input to the check is the flux function itself.
    jacobian = torch.stack(
        [
            torch.autograd.functional.jacobian(
                lambda s: e2d._physical_flux(s, axis), single
            )
            for single in state
        ]
    )  # (n, 4, 4), rows = flux component, columns = state component
    # R has the right eigenvectors as columns, so A R = R Lambda, i.e. the
    # diagonalising similarity is R^{-1} A R.
    diagonal = torch.einsum("nij,njk,nkl->nil", R_inv, jacobian, R)
    target = torch.zeros_like(diagonal)
    rho, u, v, p = e2d._primitive(state)
    a = torch.sqrt(GAMMA * p / rho)
    un = u if axis == 0 else v
    values = torch.stack((un - a, un, un, un + a), dim=-1)
    target[:, range(4), range(4)] = values
    assert torch.allclose(diagonal, target, atol=1e-10, rtol=1e-10)

    # R R^{-1} = I to machine precision (the inverse is numerical, not asserted).
    identity = torch.einsum("nij,njk->nik", R, R_inv)
    assert torch.allclose(identity, torch.eye(4, dtype=R.dtype).expand_as(identity),
                          atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("axis", [0, 1])
def test_eigen_order_is_physical(axis):
    """An entropy perturbation must project onto the double root's second slot."""
    state = _random_states()
    R, R_inv = euler_eigenvectors_2d(state, axis, GAMMA)
    rho, u, v, _ = e2d._primitive(state)
    kinetic = 0.5 * (u ** 2 + v ** 2)
    entropy = torch.stack((rho, rho * u, rho * v, rho * kinetic), dim=-1)
    amplitudes = torch.einsum("nij,nj->ni", R_inv, entropy)
    # A pure entropy perturbation carries no acoustic content: slots 0 and 3 vanish.
    assert torch.allclose(
        amplitudes[:, [0, 3]], torch.zeros_like(amplitudes[:, [0, 3]]), atol=1e-10
    )


def test_transform_is_a_basis_change_not_a_projection():
    """The transform pair reproduces the state exactly (no information loss)."""
    state = _random_states(9)
    identity = torch.eye(4, dtype=state.dtype).expand(9, 4, 4)
    for axis in (0, 1):
        R, R_inv = euler_eigenvectors_2d(state, axis, GAMMA)
        round_trip = torch.einsum("nij,njk->nik", R, R_inv)
        assert torch.allclose(round_trip, identity, atol=1e-12, rtol=0.0)


def test_inadmissible_states_fall_back_to_the_identity():
    """A negative-density state must not raise and must degrade to componentwise."""
    bad = torch.tensor([[-1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    R, R_inv = euler_eigenvectors_2d(bad, 0, GAMMA)
    assert torch.allclose(R[0], torch.eye(4, dtype=torch.float64))
    assert torch.allclose(R_inv[0], torch.eye(4, dtype=torch.float64))


def _smooth_wave(nx: int, ny: int, degree: int) -> torch.Tensor:
    """The anchor paper's smooth 2-D Euler solution at t = 0, projected modally."""
    x = torch.linspace(0.0, 2.0, nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0.0, 2.0, ny + 1, dtype=torch.float64)[:-1]

    def initial(X, Y):
        rho = 1.0 + 0.2 * torch.sin(math.pi * (X + Y))
        ones = torch.ones_like(rho)
        return rho, ones.clone(), ones.clone(), ones

    _, _, coeff, _ = e2d.solve_euler_dg_2d(
        initial,
        ncells=(nx, ny),
        degree=degree,
        final_time=0.0,
        cfl=0.2,
        limiter=None,
    )
    return coeff


def test_limiter_is_inert_when_no_cell_is_troubled():
    """With ``M`` above the smoothness threshold the state must be returned as is."""
    coeff = _smooth_wave(12, 12, 2)
    limited, diagnostics = apply_characteristic_tvb_limiter_2d(
        coeff, 2.0 / 12, 2.0 / 12, tvb_constant=1e3,
        periodic=(True, True), return_diagnostics=True,
    )
    assert diagnostics["troubled_x"] == 0 and diagnostics["troubled_y"] == 0
    assert torch.equal(limited, coeff)


def test_limiter_conserves_cell_averages_bit_for_bit():
    """Averages are restored verbatim, so conservation is exact (not 1e-13 close)."""
    coeff = _smooth_wave(10, 10, 2)
    # Force the limiter to fire everywhere with the strict TVD test.
    limited, diagnostics = apply_characteristic_tvb_limiter_2d(
        coeff, 2.0 / 10, 2.0 / 10, tvb_constant=0.0,
        periodic=(True, True), return_diagnostics=True,
    )
    assert diagnostics["troubled_x"] > 0
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])


def test_p0_is_returned_untouched():
    coeff = _smooth_wave(6, 6, 0)
    out = apply_characteristic_tvb_limiter_2d(coeff, 2.0 / 6, 2.0 / 6, 1.0)
    assert out is coeff


def test_solver_accepts_the_char_tvb_limiter_and_stays_admissible():
    """End-to-end smoke: the limiter name is wired into ``solve_euler_dg_2d``."""
    def initial(X, Y):
        rho = 1.0 + 0.2 * torch.sin(math.pi * (X + Y))
        ones = torch.ones_like(rho)
        return rho, ones.clone(), ones.clone(), ones

    x, y, coeff, metadata = e2d.solve_euler_dg_2d(
        initial, ncells=(10, 10), degree=2, final_time=0.05, cfl=0.2,
        limiter="char_tvb", tvb_constant=1.0,
    )
    assert metadata["limiter"] == "char_tvb"
    assert torch.isfinite(coeff).all()
    fields = e2d.cell_fields(coeff)
    assert float(fields["density"].min()) > 0.0
    assert float(fields["pressure"].min()) > 0.0


def test_unknown_limiter_name_still_rejected():
    def initial(X, Y):
        ones = torch.ones_like(X + Y)
        return ones, torch.zeros_like(ones), torch.zeros_like(ones), ones

    with pytest.raises(ValueError):
        e2d.solve_euler_dg_2d(
            initial, ncells=(4, 4), degree=1, final_time=0.01, cfl=0.2,
            limiter="bogus",
        )