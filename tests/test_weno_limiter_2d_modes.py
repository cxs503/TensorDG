"""Mode-level contract of the 2-D WENO limiter (m14777).

The limiter is documented as "dimension by dimension": in a troubled cell only
the *pure* mode 1 of the offending axis is replaced by the WENO slope and that
axis' modes >= 2 are zeroed, while the modal content of the other axis is kept.
A slice-wide assignment (``limited[..., 1, :, :]``) silently violated this for
``degree >= 2`` by stamping the one-dimensional slope into every mixed mode
(1, j >= 1).  These tests pin the contract.
"""
import math

import torch

from tensordg.euler_dg_2d import (
    apply_tvb_limiter_2d,
    apply_weno_limiter_2d,
    legendre_operators,
)
from tensordg.limiters import apply_weno_limiter

torch.set_default_dtype(torch.float64)


def _wavy_state(nx=6, ny=6, degree=2, ncomp=4):
    """Cell averages that vary (so the limiter can flag and produce slopes)."""
    i = torch.arange(nx)[:, None].repeat(1, ny).double()
    j = torch.arange(ny)[None, :].repeat(nx, 1).double()
    averages = 1.0 + 0.3 * torch.sin(0.7 * i + 0.3 * j)
    coeff = torch.zeros(nx, ny, degree + 1, degree + 1, ncomp)
    for a in range(degree + 1):
        for b in range(degree + 1):
            coeff[..., a, b, :] = 1000.0 * a + 100.0 * b
    coeff[..., 0, 0, :] = averages[..., None]
    return coeff


def test_pure_modes_carry_the_slope_and_mixed_modes_are_preserved():
    coeff = _wavy_state()
    dx = dy = 1.0 / 6.0
    limited = apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
    changed_x = limited[..., 1, 0, :] != coeff[..., 1, 0, :]
    changed_y = limited[..., 0, 1, :] != coeff[..., 0, 1, :]
    assert bool(changed_x.any()) and bool(changed_y.any()), "limiter must flag cells at M=0"
    # the defect: (1,1) used to be overwritten with the 1-D slope
    assert torch.equal(limited[..., 1, 1, :], coeff[..., 1, 1, :])
    # and (1,2) / (2,1) used to be overwritten too; they are now zeroed only
    # in cells flagged along the corresponding axis.
    for (a, b), mask in (((1, 2), changed_y), ((2, 1), changed_x), ((2, 0), changed_x)):
        kept = limited[..., a, b, :] == coeff[..., a, b, :]
        zero = limited[..., a, b, :] == 0.0
        assert bool((kept | zero).all()), (a, b)
        assert bool(torch.equal(limited[..., a, b, :][mask], torch.zeros_like(limited[..., a, b, :][mask])))
    # cell averages are never modified (exact conservation of the mean)
    assert torch.equal(limited[..., 0, 0, :], coeff[..., 0, 0, :])
    # the TVB limiter keeps the same contract
    tvb = apply_tvb_limiter_2d(coeff, dx, dy, 0.0, (True, True))
    assert torch.equal(tvb[..., 1, 1, :], coeff[..., 1, 1, :])
    assert torch.equal(tvb[..., 0, 0, :], coeff[..., 0, 0, :])


def test_degree_three_mixed_marker_survives():
    coeff = _wavy_state(degree=3)
    limited = apply_weno_limiter_2d(coeff, 1.0 / 6.0, 1.0 / 6.0, 0.0, (True, True))
    assert torch.equal(limited[..., 1, 1, :], coeff[..., 1, 1, :])


def test_reduces_to_the_1d_limiter_on_y_invariant_data():
    nx, ny, degree = 8, 3, 2
    averages = 1.0 + 0.2 * torch.sin(0.9 * torch.arange(nx).double())
    base = torch.zeros(nx, degree + 1, 2)
    base[:, 0, :] = averages[:, None]
    base[:, 1, :] = 0.05
    coeff = torch.zeros(nx, ny, degree + 1, degree + 1, 2)
    coeff[..., :, 0, :] = base[:, None, :, :]
    dx = dy = 1.0 / nx
    got = apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
    interior = slice(1, nx - 1)  # the 1-D limiter leaves outflow end cells alone
    for k in range(2):
        one = apply_weno_limiter(base[..., k : k + 1], dx, 0.0)
        assert torch.allclose(got[interior, 0, :, 0, k], one[interior, :, 0], atol=1e-14), k
