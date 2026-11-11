"""Pins for the public wall heat-flux API of ``tensordg.ns_dg_2d`` (m14554).

The wall flux used to be readable only by hand-mirroring private internals
(``_axis_fluxes_ns`` plus the ``int phi_0 dxi = 2`` inverse projection) inside
``benchmarks/verify_ns_wall_heat_flux.py``.  ``tensordg.ns_dg_2d.wall_heat_flux``
promotes exactly that measurement to a supported entry point and the solver
reports it in ``metadata["wall_heat_flux"]``, so these pins freeze:

1. the API reproduces the operator projection bit-for-bit (no re-derivation);
2. the value is Fourier's ``q = -kappa dT/dy`` with the same sign on both walls;
3. it converges to the analytic ``kappa (T_hot - T_cold) / L`` at the design
   order of the chosen wall-gradient estimator;
4. the solver integration path (``solve_navier_stokes_dg_2d`` metadata) reports
   the same numbers as the standalone API call on the returned state.
"""
from __future__ import annotations

import math

import pytest
import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
T_HOT, T_COLD, PRESSURE = 1.0, 0.5, 1.0
EXACT = KAPPA * (T_HOT - T_COLD)
NX = 2


def _steady(x, y):
    reference = x * y + 0.0 * x + 0.0 * y
    temperature = T_HOT + (T_COLD - T_HOT) * (y + 0.0 * x)
    return (PRESSURE / temperature, torch.zeros_like(reference),
            torch.zeros_like(reference), PRESSURE * torch.ones_like(reference))


def _boundary():
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
            "ymax": {"velocity": (0.0, 0.0), "temperature": T_COLD}}


def _state(ny, degree):
    dx, dy = 1.0 / NX, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(NX, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    return coeff, (nodes, weights, basis, derivative, signs), (dx, dy)


def _api_flux(ny, degree, wall_gradient="face"):
    coeff, ops, (dx, dy) = _state(ny, degree)
    nodes, weights, basis, derivative, signs = ops  # noqa: F841
    penalty = 2.0 * (degree + 1.0) ** 2
    flux = ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             _boundary(), MU, KAPPA, penalty, wall_gradient)
    return flux, (coeff, ops, (dx, dy))


def _order(values, cells):
    xs = [math.log(n) for n in cells]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs)


def test_api_is_the_inverse_of_the_operator_projection():
    """The API must not re-derive the flux: it inverts the same projection."""
    ny, degree = 16, 1
    coeff, ops, (dx, dy) = _state(ny, degree)
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (degree + 1.0) ** 2
    boundary = _boundary()
    flux = ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             boundary, MU, KAPPA, penalty, "face")
    right, left, _, _ = ns._axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, False,
        boundary["ymin"], boundary["ymax"], MU, KAPPA, penalty, 1, "face")
    norm0 = 2.0  # int_{-1}^{1} P_0 dxi
    assert torch.allclose(flux["ymin"], left[:, 0, 0, 3] / norm0, rtol=0.0, atol=0.0)
    assert torch.allclose(flux["ymax"], right[:, -1, 0, 3] / norm0, rtol=0.0, atol=0.0)
    # one value per wall cell: a y-wall spans the NX cells in x
    assert flux["ymin"].shape == (NX,)
    assert flux["ymax"].shape == (NX,)


def test_api_reports_fourier_flux_with_the_same_sign_on_both_walls():
    flux, _ = _api_flux(32, 2)
    assert set(flux) == {"ymin", "ymax"}
    bottom = float(flux["ymin"].mean())
    top = float(flux["ymax"].mean())
    assert bottom > 0.0 and top > 0.0, (
        "heat crosses the hot wall into the domain and leaves through the cold "
        f"wall: both q_y = -kappa dT/dy values must be positive, got {bottom}, {top}")
    assert abs(bottom - EXACT) / EXACT < 3.0e-2
    assert abs(top - EXACT) / EXACT < 3.0e-2


@pytest.mark.parametrize("degree,wall_gradient,min_order", [
    (1, "face", 0.9), (1, "taylor", 0.9), (2, "face", 1.8), (2, "taylor", 1.8)])
def test_api_flux_converges_to_the_analytic_value(degree, wall_gradient, min_order):
    cells = (8, 16, 32, 64)
    errors = []
    for ny in cells:
        flux, _ = _api_flux(ny, degree, wall_gradient)
        errors.append(abs(float(flux["ymin"].mean()) - EXACT))
    order = _order(errors, cells)
    assert order > min_order, (degree, wall_gradient, order, errors)
    # the finest mesh must be inside the 3%-acceptance band of the platform
    assert errors[-1] / EXACT < 3.0e-2, (degree, wall_gradient, errors[-1])


def test_api_returns_empty_dict_without_walls():
    coeff, ops, (dx, dy) = _state(8, 1)
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (1 + 1.0) ** 2
    assert ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             {}, MU, KAPPA, penalty) == {}
    outflow = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
    assert ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             outflow, MU, KAPPA, penalty) == {}
    # a periodic axis is not a wall even if a boundary dict is handed in
    assert ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             _boundary(), MU, KAPPA, penalty, periodic=(False, True)) == {}


def test_summary_averages_the_wall_cells():
    coeff, ops, (dx, dy) = _state(16, 1)
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (1 + 1.0) ** 2
    flux = ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                             _boundary(), MU, KAPPA, penalty)
    summary = ns.wall_heat_flux_summary(coeff, basis, derivative, signs, weights,
                                        dx, dy, _boundary(), MU, KAPPA, penalty)
    for name, value in summary.items():
        assert value == pytest.approx(float(flux[name].mean()), abs=0.0)


def test_solver_metadata_exposes_the_wall_heat_flux():
    """Integration path: the solver itself reports the flux of its own state."""
    _, _, coeff, metadata = ns.solve_navier_stokes_dg_2d(
        _steady, ncells=(NX, 16), degree=1, final_time=0.002, cfl=0.3,
        mu=MU, prandtl=PRANDTL, periodic=(True, False), boundary=_boundary(),
        dtype=torch.float64)
    reported = metadata["wall_heat_flux"]
    assert set(reported) == {"ymin", "ymax"}
    for name, value in reported.items():
        assert math.isfinite(value)
        assert abs(value - EXACT) / EXACT < 3.0e-2, (name, value)
    # the reported number is the standalone API applied to the returned state
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        metadata["degree"], max(metadata["degree"] + 2, 3), torch.device("cpu"),
        torch.float64)
    dx, dy = metadata["domain"][0][1] / NX, metadata["domain"][1][1] / 16
    direct = ns.wall_heat_flux_summary(
        coeff, basis, derivative, signs, weights, dx, dy, _boundary(), MU,
        metadata["kappa"], metadata["penalty"], metadata["wall_gradient"],
        periodic=metadata["periodic"])
    assert set(direct) == set(reported)
    for name in reported:
        assert reported[name] == pytest.approx(direct[name], abs=1e-12)


def test_x_walls_reproduce_the_y_wall_numbers():
    """Axis symmetry: the x-wall path must give the y-wall numbers bit-for-bit.

    The operator is axis-agnostic (``_axis_fluxes_ns`` is called with ``axis``),
    so rotating the conduction case by 90 degrees must not change the flux.  This
    gives the ``xmin``/``xmax`` branch a real reference: the identical number is
    benchmarked against ``kappa/2`` on the y-walls.
    """
    def steady_x(x, y):
        reference = x * y + 0.0 * x + 0.0 * y
        temperature = T_HOT + (T_COLD - T_HOT) * (x + 0.0 * y)
        return (PRESSURE / temperature, torch.zeros_like(reference),
                torch.zeros_like(reference), PRESSURE * torch.ones_like(reference))

    for degree in (1, 2):
        nx, ny = 16, 2
        dx, dy = 1.0 / nx, 1.0 / ny
        nodes, weights, basis, derivative, signs = ns.legendre_operators(
            degree, degree + 2, torch.device("cpu"), torch.float64)
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
        coeff = ns._initial_coefficients(steady_x, x, y, dx, dy, nodes, weights,
                                         basis, torch.device("cpu"), torch.float64)
        penalty = 2.0 * (degree + 1.0) ** 2
        boundary = {"xmin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
                    "xmax": {"velocity": (0.0, 0.0), "temperature": T_COLD},
                    "ymin": None, "ymax": None}
        flux = ns.wall_heat_flux(coeff, basis, derivative, signs, weights, dx, dy,
                                 boundary, MU, KAPPA, penalty, "face")
        reference, _ = _api_flux(nx, degree, "face")  # same case, y as the normal
        assert flux["xmin"].shape == (ny,)
        # axis=0 vs axis=1 only reorder the einsums: the measured difference is
        # 1.3e-14 absolute (5e-14 relative) at ny=16, i.e. round-off, not physics
        for name_x, name_y in (("xmin", "ymin"), ("xmax", "ymax")):
            delta = float((flux[name_x] - reference[name_y]).abs().max())
            assert delta < 1e-12, (degree, name_x, delta)
