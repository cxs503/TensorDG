"""Unit tests for the layered-Poiseuille EDM verification instrument.

These tests pin the *instrument*, in this order:

* the analytic reference is verified against a second, independent route
  (closed-form hand algebra for two layers, and a finite-volume BVP solve for
  three layers) -- no solver code is involved, so a drift here is a drift in the
  yardstick the anchor is scored against;
* the EDM forcing term is shown to inject exactly ``F`` into the first moment
  (the invariant that makes a steady profile comparable with Stokes);
* the half-way bounce-back wall holds a resting fluid exactly at rest;
* a short single-phase EDM channel run reproduces the analytic parabola.
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorlbm.lattice import D2Q9  # noqa: E402
from tensorlbm.layered_poiseuille import (  # noqa: E402
    CS2,
    ShanChenLayeredPoiseuille,
    edm_collide,
    layered_poiseuille_analytic,
    layered_poiseuille_two_layer_closed_form,
    local_viscosity,
    profile_error,
    viscosity_from_omega,
    wall_aware_cross_force,
)


# ---------------------------------------------------------------------------
# 1. Analytic reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nu1,nu2,h1,H", [
    (1 / 6, 0.5, 31.0, 62.0),
    (0.05, 0.5, 20.0, 62.0),
    (0.4, 0.4, 30.0, 62.0),
])
def test_two_numerical_routes_agree(nu1, nu2, h1, H):
    """Dense linear solve (generic N-layer) vs hand-eliminated closed form."""
    g = 2e-5
    y = np.linspace(0.0, H, 501)
    a = layered_poiseuille_analytic([nu1, nu2], [h1], H, g, y=y)
    b = layered_poiseuille_two_layer_closed_form(nu1, nu2, h1, H, g, y=y)
    scale = np.max(np.abs(a["u"]))
    assert scale > 0
    assert np.max(np.abs(a["u"] - b["u"])) < 1e-12 * scale


def test_two_layer_conditions_hold():
    """No-slip, velocity continuity and shear-stress continuity."""
    nu1, nu2, h1, H, g = 1 / 6, 0.35, 27.0, 62.0, 2e-5
    res = layered_poiseuille_analytic([nu1, nu2], [h1], H, g,
                                      y=[0.0, H, h1 - 1e-12, h1 + 1e-12])
    y, u = res["y"], res["u"]
    scale = np.max(np.abs(res["u"]))
    assert abs(u[0]) < 1e-12 * scale and abs(u[1]) < 1e-12 * scale
    assert abs(u[2] - u[3]) < 1e-9 * scale

    # stress: mu du/dy from the coefficients (du/dy = a - g y / nu)
    coeffs = res["coeffs"]
    dud1 = coeffs[0, 0] - g * h1 / nu1
    dud2 = coeffs[1, 0] - g * h1 / nu2
    assert abs(nu1 * dud1 - nu2 * dud2) < 1e-12 * max(abs(nu1 * dud1), 1.0)


def test_equal_viscosity_recovers_poiseuille_parabola():
    nu, H, g = 0.21, 62.0, 3e-5
    y = np.linspace(0.0, H, 401)
    res = layered_poiseuille_analytic([nu, nu], [H / 2], H, g, y=y)
    exact = g * y * (H - y) / (2.0 * nu)
    assert np.max(np.abs(res["u"] - exact)) < 1e-12 * np.max(np.abs(exact))


def test_three_layer_solution_matches_finite_volume_bvp():
    """Independent route: flux-form BVP on a fine grid (no LBM, no hand algebra)."""
    nu = np.array([0.10, 0.40, 0.20])
    ifs = [18.0, 41.0]
    H, g = 62.0, 2e-5
    y = np.linspace(0.0, H, 2001)
    res = layered_poiseuille_analytic(nu, ifs, H, g, y=y)

    # finite-volume discretisation of d/dy(mu du/dy) = -rho g, rho = 1
    n = y.size
    dy = y[1] - y[0]
    mu_node = np.where(y <= ifs[0], nu[0], np.where(y <= ifs[1], nu[1], nu[2]))
    mu_face = 2.0 * mu_node[:-1] * mu_node[1:] / (mu_node[:-1] + mu_node[1:])
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = mu_face[i - 1] / dy ** 2
        A[i, i] = -(mu_face[i - 1] + mu_face[i]) / dy ** 2
        A[i, i + 1] = mu_face[i] / dy ** 2
        rhs[i] = -g
    u_fv = np.linalg.solve(A, rhs)
    assert np.max(np.abs(u_fv - res["u"])) < 1e-3 * np.max(np.abs(res["u"]))


def test_analytic_rejects_bad_input():
    with pytest.raises(ValueError):
        layered_poiseuille_analytic([0.1, 0.2], [], 10.0, 1e-5)
    with pytest.raises(ValueError):
        layered_poiseuille_analytic([0.1, 0.2], [5.0, 4.0], 10.0, 1e-5)
    with pytest.raises(ValueError):
        layered_poiseuille_analytic([0.1, -0.2], [5.0], 10.0, 1e-5)


# ---------------------------------------------------------------------------
# 2. EDM forcing
# ---------------------------------------------------------------------------

def test_edm_injects_exactly_the_force():
    torch.manual_seed(0)
    ny, nx = 6, 5
    rho = 0.8 + 0.2 * torch.rand(ny, nx, dtype=torch.float64)
    ux = 0.01 * torch.randn(ny, nx, dtype=torch.float64)
    uy = 0.01 * torch.randn(ny, nx, dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy)
    fx = 0.03 * torch.randn(ny, nx, dtype=torch.float64)
    fy = 0.03 * torch.randn(ny, nx, dtype=torch.float64)

    f_post = edm_collide(f, rho, ux, uy, fx, fy, omega=1.3)
    e = D2Q9.e
    dpx = sum(((f_post[i] - f[i]) * e[i, 0]).sum().item() for i in range(9))
    dpy = sum(((f_post[i] - f[i]) * e[i, 1]).sum().item() for i in range(9))
    assert dpx == pytest.approx(float(fx.sum()), rel=1e-12)
    assert dpy == pytest.approx(float(fy.sum()), rel=1e-12)


def test_edm_without_force_is_plain_bgk():
    torch.manual_seed(1)
    rho = torch.ones(4, 3, dtype=torch.float64)
    ux = 0.02 * torch.randn(4, 3, dtype=torch.float64)
    uy = 0.02 * torch.randn(4, 3, dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy) + 0.01 * torch.randn(9, 4, 3,
                                                           dtype=torch.float64)
    zero = torch.zeros(4, 3, dtype=torch.float64)
    got = edm_collide(f, rho, ux, uy, zero, zero, omega=0.7)
    want = f - 0.7 * (f - D2Q9.equilibrium(rho, ux, uy))
    assert torch.allclose(got, want, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Walls
# ---------------------------------------------------------------------------

def test_halfway_bounce_back_keeps_resting_fluid_at_rest():
    s = ShanChenLayeredPoiseuille(ny=16, nx=3, omegas=[1.0], G12=0.0, g_body=0.0)
    m0 = s.total_mass()
    s.run(200)
    prof = s.profiles()
    assert np.max(np.abs(prof["ux0"])) < 1e-16
    assert np.max(np.abs(prof["uy0"])) < 1e-16
    m1 = s.total_mass()
    assert abs(m1[0] - m0[0]) < 1e-13 * abs(m0[0])


def test_wall_velocity_extrapolates_to_zero():
    """Half-way wall at y=0.5: linear extrapolation of u must vanish there."""
    s = ShanChenLayeredPoiseuille(ny=32, nx=3, omegas=[1.0], G12=0.0,
                                  g_body=2e-5)
    s.run(1200)
    prof = s.profiles()
    y, u = prof["y"], prof["ux0"]
    u_wall = u[0] - (y[0] - 0.0) / (y[1] - y[0]) * (u[1] - u[0])
    assert abs(u_wall) < 2e-2 * np.max(np.abs(u))


# ---------------------------------------------------------------------------
# 4. Single-phase channel vs analytic (short run from the analytic state)
# ---------------------------------------------------------------------------

def test_single_phase_edm_poiseuille_holds_analytic_profile():
    ny, nx = 64, 3
    g = 2e-5
    nu = viscosity_from_omega(1.0)
    H = float(ny - 2)
    y = np.arange(1, ny - 1) - 0.5
    u_ref = layered_poiseuille_analytic([nu], [], H, g, y=y)["u"]

    s = ShanChenLayeredPoiseuille(ny=ny, nx=nx, omegas=[1.0], G12=0.0, g_body=g)
    ux = torch.zeros(ny, nx, dtype=torch.float64)
    ux[1:-1, :] = torch.as_tensor(u_ref).view(-1, 1)
    s.set_fields([torch.ones(ny, nx, dtype=torch.float64)], ux=[ux])
    s.run(600)

    prof = s.profiles()
    err = profile_error(prof["y"], prof["u_mix"],
                        lambda yy: layered_poiseuille_analytic([nu], [], H, g, y=yy)["u"])
    assert err["err_max"] < 5e-2, err


def test_local_viscosity_inversion_recovers_bgk_value():
    nu = viscosity_from_omega(0.7)
    g, H = 2e-5, 62.0
    y = np.arange(1, 63) - 0.5
    u = layered_poiseuille_analytic([nu], [], H, g, y=y)["u"]
    nu_meas = local_viscosity(y, u, g, half_width=1)[5:-5]
    assert np.nanmax(np.abs(nu_meas - nu) / nu) < 1e-6


# ---------------------------------------------------------------------------
# 5. Wall-aware cross force
# ---------------------------------------------------------------------------

def test_uniform_density_has_no_cross_force():
    psi = [torch.full((12, 4), 0.6, dtype=torch.float64),
           torch.full((12, 4), 0.4, dtype=torch.float64)]
    F = wall_aware_cross_force(psi, G12=-2.0, solid_rows=(0, 11))
    assert torch.max(torch.abs(F)) < 1e-15


def test_cross_force_ignores_solid_row_values():
    """Wall-aware stencil must not see whatever is stored in the solid rows."""
    ny, nx = 20, 3
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    p1 = 0.5 + 0.25 * torch.tanh((10.0 - y) / 2.0)
    psi_a = [p1.expand(ny, nx).clone(), (1.0 - p1).expand(ny, nx).clone()]
    psi_b = [q.clone() for q in psi_a]
    psi_b[0][0, :] = 1e6
    psi_b[1][ny - 1, :] = -1e6
    F_a = wall_aware_cross_force(psi_a, -2.2, solid_rows=(0, ny - 1))
    F_b = wall_aware_cross_force(psi_b, -2.2, solid_rows=(0, ny - 1))
    assert torch.allclose(F_a[:, :, 1:-1, :], F_b[:, :, 1:-1, :], atol=1e-14)


def test_cross_force_action_reaction_on_a_periodic_domain():
    """On a doubly periodic domain sum_x (F_1 + F_2) telescopes to zero.

    Note this holds *only* for a periodic field: with a wall the ghost rows
    break the telescope and the wall exerts a net force on the fluid (measured
    and reported as a limitation of the wall-aware stencil in the report).
    """
    ny, nx = 16, 24
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    a = 0.5 + 0.25 * torch.sin(2.0 * torch.pi * x / nx)
    p1 = a.expand(ny, nx).clone()
    p2 = (1.0 - a).expand(ny, nx).clone()
    F = wall_aware_cross_force([p1, p2], G12=-2.2, solid_rows=())
    tot = F[0] + F[1]
    scale = torch.abs(F[0]).sum() + torch.abs(F[1]).sum()
    # *global* conservation: the pointwise residual is O(F) by construction (the
    # Shan-Chen stencil is not a locally conservative pair), only its domain sum
    # has to vanish.
    assert abs(tot.sum().item()) < 1e-12 * scale


def test_profile_error_reports_mask_fraction():
    y = np.linspace(0.0, 10.0, 101)
    u = np.zeros_like(y)
    res = profile_error(y, u, lambda yy: np.ones_like(yy), exclude=[(3.0, 5.0)])
    assert res["n_total"] == 101
    assert res["scored_fraction"] < 1.0
    assert res["err_max"] == pytest.approx(1.0)