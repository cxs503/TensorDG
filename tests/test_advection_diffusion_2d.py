"""Verification tests for :mod:`tensordg.advection_diffusion` (node P165W3-N05).

The criteria asserted here are the campaign gates E1, E2 and E5:

* E1 solid-body rotation (omega = 2 pi, one revolution): relative L2 <= 5 %
* E2 deformational (LeVeque cosine-hill) flow at T = 1: relative L2 <= 5 %
* E5 total mass change of the advection solver: <= 1 %

The remaining tests are the *consistency* layer that makes those numbers
trustworthy: point-wise operator exactness on exactly representable data, the
divergence-free property of the two velocity fields, an analytic
advection--diffusion decay anchor with ``nu > 0``, and the transparency of the
chosen boundary treatment.  Everything is deterministic (no random numbers).
"""
from __future__ import annotations

import math

import torch

from tensordg import advection_diffusion as ad

CFL = 0.7
UNIT = ad.UNIT_SQUARE


def _ops(degree):
    return ad.Operators(degree, torch.device("cpu"), torch.float64)


def _mesh(nx, ny=None):
    return ad._mesh_arrays(nx, ny or nx, UNIT, torch.device("cpu"), torch.float64)


def _residual_on_nodes(coeff, velocity, nu, nx, degree, boundary="extrapolate"):
    ops = _ops(degree)
    hx, hy, xc, yc, xf, yf = _mesh(nx)
    r = ad.rhs(coeff, 0.0, velocity, nu, hx, hy, ops, boundary, 2.0, (xc, yc, xf, yf))
    return torch.einsum("xykl,ak,bl->xyab", r, ops.phi, ops.phi)


def test_advection_operator_exact_on_linear_field():
    """A p=1 field that is linear in (x, y) has residual ``-a.grad u`` exactly."""
    for degree in (1, 2):
        coeff, _ = ad.project(lambda x, y: 1.0 + 2.0 * x + 3.0 * y, 6, 6, degree,
                              UNIT, "cpu", torch.float64)
        residual = _residual_on_nodes(coeff, ad.translation_velocity((1.0, 0.5)),
                                      0.0, 6, degree)
        # interior cells only: the boundary face is a transparent (Neumann-like) ghost
        inner = residual[1:-1, 1:-1]
        assert float(torch.max(torch.abs(inner + 3.5))) < 1e-11


def test_diffusion_operator_exact_on_quadratic_field():
    """SIPG Laplacian of an exactly representable quadratic field, nu = 0.1."""
    coeff, _ = ad.project(
        lambda x, y: 1.0 + 0.3 * x + 0.2 * y + 0.5 * x * x + 0.25 * y * y,
        6, 6, 2, UNIT, "cpu", torch.float64)
    residual = _residual_on_nodes(coeff, ad.vanishing_velocity(), 0.1, 6, 2)
    assert float(torch.max(torch.abs(residual[1:-1, 1:-1] - 0.1 * 1.5))) < 1e-10


def test_projection_is_exact_for_representable_polynomials():
    ops = _ops(3)
    field = lambda x, y: 0.5 + 0.3 * x - 0.7 * y + 1.2 * x * y - 0.9 * x * x * y  # noqa: E731
    coeff, _ = ad.project(field, 5, 4, 3, UNIT, "cpu", torch.float64)
    hx, hy, xc, yc, xf, yf = _mesh(5, 4)
    xq, yq = ad._quad_points(xc, yc, hx, hy, ops)
    values = ad.cell_quadrature(coeff, ops)
    assert float(torch.max(torch.abs(values - field(xq, yq)))) < 1e-13


def test_velocity_fields_are_divergence_free():
    """Both benchmark velocity fields must satisfy div(a) = 0 (mass conservation)."""
    eps = 1e-5
    x = torch.linspace(0.0, 1.0, 41, dtype=torch.float64).view(-1, 1)
    y = torch.linspace(0.0, 1.0, 41, dtype=torch.float64).view(1, -1)
    for velocity in (ad.solid_body_rotation_velocity(), ad.deformational_flow_velocity()):
        for time in (0.0, 0.37):
            ax_p, _ = velocity(time, x + eps, y)
            ax_m, _ = velocity(time, x - eps, y)
            _, ay_p = velocity(time, x, y + eps)
            _, ay_m = velocity(time, x, y - eps)
            divergence = (ax_p - ax_m + ay_p - ay_m) / (2.0 * eps)
            assert float(torch.max(torch.abs(divergence))) < 1e-9


def test_cosine_bell_is_compactly_supported_and_away_from_boundary():
    """Precondition for the transparent boundary: the data never reaches it."""
    bell = ad.cosine_bell((0.5, 0.75), 0.15)
    x = torch.linspace(-0.2, 1.2, 281, dtype=torch.float64).view(-1, 1)
    y = torch.linspace(-0.2, 1.2, 281, dtype=torch.float64).view(1, -1)
    values = bell(x, y)
    assert float(values.max()) <= 1.0 + 1e-14
    assert float(torch.abs(values[0]).max()) < 1e-15  # x = -0.2 plane is empty
    # every point of the support is at distance >= 0.1 from the unit-square border
    inside = values > 1e-12
    xs = x.expand(-1, 281)[inside]
    ys = y.expand(281, -1)[inside]
    margin = min(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    assert margin >= 0.1 - 1e-12


def test_constant_translation_is_second_order():
    k = 2.0 * math.pi

    def exact(x, y):
        return torch.sin(k * (x - 0.2)) * torch.sin(k * (y - 0.1))

    errors = []
    for ncells in (8, 16):
        out = ad.solve(ncells, ncells, 1, 0.2, ad.translation_velocity((1.0, 0.5)),
                       ad.sine_wave(), boundary="periodic", cfl=CFL)
        errors.append(ad._finish_metrics(out, exact, "translation")["relative_l2"])
    rate = math.log2(errors[0] / errors[1])
    assert rate > 1.8, (errors, rate)


def test_periodic_advection_diffusion_matches_exact_solution():
    """``u_t + a.grad u = nu lap u`` for one Fourier mode with nu = 0.02."""
    metrics = ad.run_advection_diffusion_decay(ncells=8, degree=2, nu=0.02)
    assert metrics["relative_l2"] < 5e-3, metrics


def test_advection_diffusion_decay_rate_is_correct():
    """The L2 norm must follow ``exp(-nu (kx^2+ky^2) t)`` to better than 1 %."""
    nu, final_time = 0.02, 0.25
    decay = nu * 2.0 * (2.0 * math.pi) ** 2
    out = ad.solve(8, 8, 2, final_time, ad.translation_velocity((1.0, 0.5)),
                   ad.sine_wave(), nu=nu, boundary="periodic", cfl=CFL)
    meta = out["metadata"]
    ops = meta["ops"]
    values = ad.cell_quadrature(out["coeff"], ops)
    w2 = ops.weights[None, None, :, None] * ops.weights[None, None, None, :]
    norm = float(torch.sqrt(0.25 * meta["hx"] * meta["hy"]
                            * torch.sum(w2 * values ** 2)).item())
    # exact L2 norm of sin(2 pi x) sin(2 pi y) exp(-decay t) on the unit square
    expected = 0.5 * math.exp(-decay * final_time)
    assert abs(norm - expected) / expected < 1e-2, (norm, expected)


def test_solid_body_rotation_criterion_e1():
    """E1: one full rigid rotation returns the initial bell (relative L2 <= 5 %)."""
    metrics = ad.run_solid_body_rotation(ncells=16, degree=3, cfl=CFL)
    assert metrics["relative_l2"] <= 0.05, metrics
    assert metrics["steps"] > 100
    assert metrics["omega"] == 2.0 * math.pi


def test_solid_body_rotation_quarter_turn_matches_analytic_rotation():
    """Exact solution at T/4 is the bell rotated by -omega t: same order of error."""
    metrics = ad.run_solid_body_rotation(ncells=16, degree=3, cfl=CFL,
                                         revolutions=0.25)
    assert metrics["relative_l2"] <= 0.05, metrics


def test_deformational_flow_criterion_e2():
    """E2: reversible LeVeque deformation returns the hill at T = 1."""
    metrics = ad.run_deformational_flow(ncells=16, degree=3, cfl=CFL)
    assert metrics["relative_l2"] <= 0.05, metrics


def test_mass_drift_criterion_e5():
    """E5: total mass change <= 1 % for both advection benchmarks."""
    rotation = ad.run_solid_body_rotation(ncells=16, degree=3, cfl=CFL)
    deformation = ad.run_deformational_flow(ncells=16, degree=3, cfl=CFL)
    assert rotation["mass_drift"] <= 0.01, rotation["mass_drift"]
    assert deformation["mass_drift"] <= 0.01, deformation["mass_drift"]


def test_mass_is_conserved_exactly_for_periodic_data():
    """Single-mode periodic data: the discrete mass is conserved to round-off."""
    out = ad.solve(12, 12, 2, 0.2, ad.translation_velocity((0.7, -0.4)),
                   lambda x, y: 1.0 + 0.5 * torch.sin(2 * math.pi * x) * torch.cos(2 * math.pi * y),
                   boundary="periodic", cfl=CFL)
    meta = out["metadata"]
    drift = abs(meta["mass_final"] - meta["mass_initial"]) / abs(meta["mass_initial"])
    assert drift < 1e-12, drift


def test_boundary_treatment_is_transparent_for_compact_data():
    """Periodic and transparent boundaries agree while the data is interior."""
    bell = ad.cosine_bell((0.5, 0.75), 0.15)
    velocity = ad.translation_velocity((0.5, -0.25))
    periodic = ad.solve(16, 16, 2, 0.1, velocity, bell, boundary="periodic", cfl=CFL)
    transparent = ad.solve(16, 16, 2, 0.1, velocity, bell, boundary="extrapolate", cfl=CFL)
    ops = periodic["metadata"]["ops"]
    hx, hy = periodic["metadata"]["hx"], periodic["metadata"]["hy"]
    diff = periodic["coeff"] - transparent["coeff"]
    scale = float(torch.max(torch.abs(transparent["coeff"])))
    assert float(torch.max(torch.abs(diff))) / scale < 1e-4


def test_solver_is_deterministic_and_reports_metadata():
    bell = ad.cosine_bell((0.5, 0.75), 0.15)
    first = ad.solve(10, 10, 2, 0.05, ad.solid_body_rotation_velocity(), bell, cfl=CFL)
    second = ad.solve(10, 10, 2, 0.05, ad.solid_body_rotation_velocity(), bell, cfl=CFL)
    assert torch.equal(first["coeff"], second["coeff"])
    meta = first["metadata"]
    for key in ("nx", "ny", "degree", "dt", "steps", "boundary", "mass_history",
                "time_history", "max_speed"):
        assert key in meta
    assert abs(meta["time_history"][-1] - 0.05) < 1e-12