"""Unit-consistency tests for the modal TVB / WENO limiters.

The modal representation used everywhere in TensorDG is the *unnormalised*
Legendre basis ``P_k(xi)`` on ``xi in [-1, 1]``, i.e.

    u(x) = sum_k c_k P_k(xi),     xi = 2 (x - x_center) / dx.

For this basis ``P_0 = 1`` and ``P_1(xi) = xi``, therefore

    c_0  = cell average,
    c_1  = (du/dx) * dx / 2.        (*)

Requirement (*) is the convention already used by the exact-representation
test in ``tests/test_limiters.py`` (for ``u = x`` it fixes ``c_1 = dx/2``) and
by :func:`tensordg.limiters.apply_weno_limiter`, which converts its physical
WENO slope with the explicit ``dx / 2`` factor.

A minmod limiter working with *cell averages* produces a physical slope

    m = minmod( (u_i - u_{i-1})/dx , (u_{i+1} - u_i)/dx ),

so its modal representation must be ``c_1 = m * dx / 2``.  Writing the
cell-average increments ``d_left = u_i - u_{i-1}`` and ``d_right = u_{i+1} - u_i``
this is simply

    c_1 = 0.5 * minmod(d_left, d_right).

The tests below pin that conversion down: a limiter must be exact for data it
has no reason to change, and its reconstruction must interpolate the classical
minmod edge values ``u_i +/- minmod(d_left, d_right)/2``.
"""
import torch

from tensordg.limiters import apply_tvb_limiter, apply_weno_limiter


def _minmod(a: float, b: float) -> float:
    """Scalar reference minmod."""
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


def test_tvb_limiter_is_exact_for_linear_data_in_troubled_cells():
    """A limiter must not corrupt a field that is already a linear polynomial.

    Variable 0 is exactly linear in ``x`` (``c_0 = 1 + x_i``, ``c_1 = dx/2``),
    which is the *same* representation as the unlimited data, because the
    minmod slope of a linear sequence equals its exact slope.  Variable 1
    contains a jump, so the neighbouring cell (index 4 here) is flagged as
    troubled and *every* variable of that cell is passed through the limiter.

    With the ``dx/2`` conversion (*) the limited mode-1 coefficient of
    variable 0 is unchanged; without it the slope is doubled, i.e. the
    limiter introduces a 100 % error on an exactly represented field.
    """
    ncells, dx = 6, 0.2
    coeff = torch.zeros(ncells, 3, 2, dtype=torch.float64)
    centers = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx

    # Variable 0: u(x) = 1 + x, exactly represented by (c_0, c_1) = (1+x_i, dx/2).
    coeff[:, 0, 0] = 1.0 + centers
    coeff[:, 1, 0] = dx / 2.0

    # Variable 1: decelerating rise that flags cell 3 (d_left = 0.8,
    # d_right = 0.2 -> minmod = 0.2 != d_left) as troubled while still having a
    # non-zero minmod slope, so variable 0 is genuinely passed through the
    # limiter and must come out unchanged.
    coeff[:, 0, 1] = torch.tensor([0.0, 0.0, 0.0, 0.8, 1.0, 1.05], dtype=torch.float64)

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)

    # Sanity: the intended cell really is flagged by the public limiter path.
    assert not torch.equal(limited[:, 1, 1], coeff[:, 1, 1]), "test cell was not limited"

    assert torch.allclose(limited[3, :, 0], coeff[3, :, 0], atol=1e-13), (
        "TVB limiter must reproduce the exact linear profile of variable 0; "
        f"got c_1 = {limited[3, 1, 0].item():.17g}, "
        f"expected {coeff[3, 1, 0].item():.17g}"
    )


def test_tvb_and_weno_limiters_agree_on_the_mode1_conversion():
    """Both limiters must express the same physical slope in the same units.

    For decelerating monotone data the minmod slope is ``d_right`` and the WENO
    weights collapse onto the same slope, so the two limiters must produce the
    identical mode-1 coefficient ``0.5 * d_right``.
    """
    ncells, dx = 5, 0.25
    averages = [0.00, 0.40, 0.46, 0.47, 0.4702]
    coeff = torch.zeros(ncells, 2, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor(averages, dtype=torch.float64)

    cell = 2
    d_left = averages[cell] - averages[cell - 1]
    d_right = averages[cell + 1] - averages[cell]
    expected = 0.5 * _minmod(d_left, d_right)

    tvb = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    weno = apply_weno_limiter(coeff, dx, tvb_constant=0.0)

    # WENO combines three candidate slopes with data-dependent weights, so it
    # agrees with the ideal minmod slope only to O(1 %), but it must be in the
    # same units (it is: 2 % tolerance, versus the factor-2 discrepancy the TVB
    # path used to have).
    assert abs(weno[cell, 1, 0].item() - expected) < 0.02 * abs(expected), (
        f"WENO conversion drifted: {weno[cell, 1, 0].item():.6e} vs {expected:.6e}"
    )
    assert abs(tvb[cell, 1, 0].item() - expected) < 1e-14, (
        "TVB limiter must use the h/2 modal conversion of equation (*); "
        f"got c_1 = {tvb[cell, 1, 0].item():.17g}, expected {expected:.17g}"
    )


def test_tvb_limiter_reconstruction_interpolates_minmod_edge_values():
    """The limited polynomial must take the minmod edge values ``u_i +/- m/2``.

    Evaluated at the cell interfaces (``xi = +/-1``) the limited reconstruction
    is ``c_0 +/- c_1``.  Classical minmod reconstruction gives
    ``u_left = u_i - m*dx/2`` and ``u_right = u_i + m*dx/2``, hence
    ``|c_1|`` must equal ``|m|*dx/2`` and never the full increment ``|d_right|``.
    """
    ncells, dx = 5, 0.25
    averages = [0.00, 0.40, 0.46, 0.47, 0.4702]
    coeff = torch.zeros(ncells, 2, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor(averages, dtype=torch.float64)

    cell = 2
    d_left = averages[cell] - averages[cell - 1]
    d_right = averages[cell + 1] - averages[cell]
    m = _minmod(d_left, d_right)

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    u_left = (limited[cell, 0, 0] - limited[cell, 1, 0]).item()
    u_right = (limited[cell, 0, 0] + limited[cell, 1, 0]).item()

    assert abs(u_right - (averages[cell] + 0.5 * m)) < 1e-13
    assert abs(u_left - (averages[cell] - 0.5 * m)) < 1e-13
    # TVD bookkeeping: the reconstruction slope must not exceed half the
    # smaller neighbour jump, otherwise the limiter is not the minmod one.
    assert abs(limited[cell, 1, 0].item()) <= 0.5 * abs(m) + 1e-15