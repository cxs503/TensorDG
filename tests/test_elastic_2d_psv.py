"""Gates for the experimental 2-D P-SV elastic DG solver (``tensordg.elastic_2d``).

Status (2026-11-12, m16925): **NOT VERIFIED** -- see
``docs/CASE-DG-ELASTIC-2D-PSV.md``.  The tests below are deliberately split in
two groups:

* gates that the implementation *passes* (operator algebra, free-stream
  preservation, physical flux/Riemann consistency, material speeds);
* a *defect gate* (``test_known_semidiscrete_instability``) that asserts the
  currently observed instability, so that it turns red the moment the defect is
  fixed and has to be updated together with the docs.  It is marked ``xfail``
  in the other direction: it is a positive assertion on the buggy behaviour, so
  a fix shows up as a failing test rather than silent rot.
"""
import math

import numpy as np
import pytest
import torch

from tensordg.elastic_2d import (ElasticMedium, flux_x, flux_y,
                                 legendre_operators_2d, modal_to_quadrature,
                                 plane_wave, quadrature_grid, riemann_flux_x,
                                 riemann_flux_y, solve_elastic_2d, _rhs)

MEDIUM = ElasticMedium(rho=1.0, lam=1.0, mu=1.0)


def test_material_speeds_and_impedances():
    assert MEDIUM.cp == pytest.approx(math.sqrt(3.0))
    assert MEDIUM.cs == pytest.approx(1.0)
    assert MEDIUM.zp == pytest.approx(MEDIUM.rho * MEDIUM.cp)
    assert MEDIUM.zs == pytest.approx(MEDIUM.rho * MEDIUM.cs)
    # physical ordering: the P wave is always faster than the S wave
    assert MEDIUM.cp > MEDIUM.cs
    with pytest.raises(ValueError):
        ElasticMedium(rho=0.0)
    with pytest.raises(ValueError):
        ElasticMedium(mu=-1.0)


def test_nodal_modal_transform_is_the_identity():
    for degree in (0, 1, 2, 3, 5):
        ops = legendre_operators_2d(degree)
        eye = ops["proj"] @ ops["phi"]
        assert torch.allclose(eye, torch.eye(degree + 1, dtype=torch.float64),
                              atol=1e-12)


def test_derivative_matrix_differentiates_legendre_modes_exactly():
    for degree in (1, 2, 3, 4):
        ops = legendre_operators_2d(degree)
        for k in range(degree + 1):
            c = torch.zeros(degree + 1, dtype=torch.float64)
            c[k] = 1.0
            got = ops["dmat"] @ c
            poly = np.zeros(k + 1)
            poly[-1] = 1.0
            deriv = np.polynomial.legendre.legder(poly)
            if deriv.size == 0:
                continue
            expected = ops["proj"] @ torch.as_tensor(
                np.polynomial.legendre.legval(ops["nodes"].numpy(), deriv),
                dtype=torch.float64)
            assert torch.allclose(got, expected, atol=1e-12)


def test_volume_operator_reproduces_the_analytic_flux_divergence():
    """``-dF/dx`` of an exactly representable state must come out exactly."""
    degree, nx, ny = 2, 4, 1
    ops = legendre_operators_2d(degree)
    hx, hy = 1.0 / nx, 1.0 / ny
    x, y = quadrature_grid(nx, ny, hx, hy, ops["nodes"])

    def initial(xx, yy):
        out = torch.zeros(*xx.shape, 5, dtype=torch.float64)
        out[..., 2] = xx ** 2            # sigma_xx = x^2 (degree 2 <= p)
        return out

    u0 = initial(x, y)
    c0 = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], u0)
    fx = flux_x(u0, MEDIUM)
    cfx = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"], fx)
    volume = -(2.0 / hx) * torch.einsum("kn,ijnlm->ijklm", ops["dmat"], cfx)
    got = modal_to_quadrature(volume, ops["phi"])[..., 0]
    expected = 2.0 * x / MEDIUM.rho      # -d/dx(-sigma_xx/rho)
    assert torch.allclose(got[:, 0, :, 0], expected[:, 0, :, 0], atol=1e-13)


def test_uniform_state_has_zero_rhs_at_machine_precision():
    """Free-stream preservation: a constant state must not move."""
    for degree in (1, 2, 3):
        ops = legendre_operators_2d(degree)
        k = degree + 1
        c = torch.zeros(4, 4, k, k, 5, dtype=torch.float64)
        c[..., 0, 0, 0] = 0.4           # vx
        c[..., 0, 0, 2] = 1.7           # sxx
        c[..., 0, 0, 3] = -0.3          # syy
        rhs = _rhs(c, ops, MEDIUM, 0.25, 0.25, "periodic", "periodic")
        assert rhs.abs().max().item() < 1e-12


def test_riemann_flux_reduces_to_the_physical_flux_for_equal_states():
    torch.manual_seed(0)
    state = torch.randn(3, 4, 7, 5, dtype=torch.float64)
    assert torch.allclose(riemann_flux_x(state, state, MEDIUM),
                          flux_x(state, MEDIUM), atol=1e-14)
    assert torch.allclose(riemann_flux_y(state, state, MEDIUM),
                          flux_y(state, MEDIUM), atol=1e-14)


def test_riemann_flux_passes_a_right_going_wave_unchanged():
    """A pure +x P wave on the left must leave the right state untouched."""
    zp = MEDIUM.zp
    amp = 0.3
    left = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    right = torch.zeros(1, 1, 1, 5, dtype=torch.float64)
    right[..., 0] = amp                 # vx
    right[..., 2] = -zp * amp           # sxx of a +x travelling P wave
    star = riemann_flux_x(left, right, MEDIUM)
    assert torch.allclose(star, flux_x(right, MEDIUM), atol=1e-14)


def test_plane_wave_reference_solves_the_pde_in_the_strong_sense():
    """Finite-difference the exact solution and check it against ``_rhs``.

    This is the gate that caught the missing isotropic part of the P-wave
    stress: without it the residual is O(1) relative.
    """
    k = 2.0 * math.pi
    ex = plane_wave("P", k, 0.0, 1.0, MEDIUM)
    degree, nx, ny = 2, 8, 1
    ops = legendre_operators_2d(degree)
    hx, hy = 1.0 / nx, 1.0 / ny
    x, y = quadrature_grid(nx, ny, hx, hy, ops["nodes"])
    c0 = torch.einsum("ka,lb,ijabm->ijklm", ops["proj"], ops["proj"],
                      ex(x, y, 0.0))
    rhs = modal_to_quadrature(_rhs(c0, ops, MEDIUM, hx, hy, "periodic", "periodic"),
                              ops["phi"])
    dt = 1e-6
    fd = (ex(x, y, dt) - ex(x, y, -dt)) / (2.0 * dt)
    scale = (fd ** 2).sum().sqrt()
    assert ((rhs - fd) ** 2).sum().sqrt() / scale < 5e-2


def test_known_semidiscrete_instability_is_registered():
    """DEFECT GATE: the face assembly is still anti-dissipative.

    Assembles the semi-discrete operator of a ``2 x 1`` cell, ``p = 2``,
    periodic box and asserts that a *positive* real eigenvalue exists.  The
    value is recorded in ``docs/CASE-DG-ELASTIC-2D-PSV.md``.  When the flux
    assembly is fixed this test must start failing -- that is the intended
    signal to flip it into ``max Re < 1e-8`` and update the docs.
    """
    degree, nx, ny = 2, 2, 1
    ops = legendre_operators_2d(degree)
    k = degree + 1
    hx, hy = 1.0 / nx, 1.0 / ny
    n = nx * ny * k * k * 5
    mat = np.zeros((n, n))
    for j in range(n):
        c = torch.zeros(nx, ny, k, k, 5, dtype=torch.float64).reshape(-1)
        c[j] = 1.0
        mat[:, j] = _rhs(c.reshape(nx, ny, k, k, 5), ops, MEDIUM, hx, hy,
                         "periodic", "periodic").reshape(-1).numpy()
    ev = np.linalg.eigvals(mat)
    assert ev.real.max() > 1.0, (
        f"no positive eigenvalue found (max Re = {ev.real.max():.3e}); if the "
        "assembly was fixed, update this gate and the docs")
    # the growth is mild but real: the plane wave energy creeps up (see docs)
    assert np.abs(ev.imag).max() < 1e3


def test_plane_wave_error_is_not_converging_and_is_h_independent():
    """DEFECT GATE: the measured plane-wave error does not fall with h.

    Records the observation that the error is identical (to 12 digits) for
    five different CFL numbers, i.e. it is a semi-discrete, not a time-stepping,
    defect.
    """
    k = 2.0 * math.pi
    ex = plane_wave("P", k, 0.0, 1.0, MEDIUM)

    def run(n, cfl):
        res = solve_elastic_2d(lambda x, y: ex(x, y, 0.0), nx=n, ny=1,
                               degree=2, final_time=0.02, cfl=cfl)
        x, y = res.quadrature_grid()
        num, exact = res.fields(), ex(x, y, 0.02)
        w = res.ops["weights"]
        w2 = torch.outer(w, w).reshape(1, 1, w.numel(), w.numel(), 1)
        return ((((num - exact) ** 2 * w2).sum()
                 / ((exact ** 2 * w2).sum())).sqrt().item())

    e16 = run(16, 0.1)
    e64 = run(64, 0.1)
    assert e64 > 1e-4                      # still far from converged
    assert abs(math.log(e16 / e64) / math.log(4.0)) < 1.0   # EOC well below p+1
    e64_small_cfl = run(64, 0.025)
    assert e64_small_cfl == pytest.approx(e64, rel=1e-6)    # CFL independent