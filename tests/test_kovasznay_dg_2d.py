"""Verification tests for the steady Kovasznay case and the through-flow face.

The tests are judges, not smoke tests:

* :func:`test_closed_form_is_exact` -- the benchmark reference is a closed form,
  so it is only usable once the three steady Navier--Stokes residuals of that
  closed form are at round-off.  This is what makes ``ref`` a formula instead of
  a transcribed table.
* :func:`test_decay_rate_solves_the_characteristic_relation` -- ``lam`` is the
  negative root of ``lam**2 - Re lam - k**2 = 0`` for a range of ``Re``/``k``.
* :func:`test_boundary_data_is_periodic_in_y` -- the ``y`` period of the closed
  form is ``2 pi / k``, which is what makes the ``y`` boundaries periodic.
* :func:`test_prescribed_face_carries_mass_flux` -- the wall closure of
  ``ns_dg_2d`` transports no mass; the prescribed-state face must, otherwise an
  inflow is a solid wall and the whole case is meaningless.
* :func:`test_modal_restart_is_lossless` -- the chunked march re-projects the
  polynomial through :func:`~tensordg.kovasznay.modal_evaluator`; that step has
  to be exact to round-off or the chunking would pollute the steady state.
* :func:`test_steady_error_shrinks_with_degree` -- on a small mesh, the P2
  steady error must be far below the P1 steady error (design order, not a
  smoke test).

Every measured number comes from the module under test; the only hard-coded
numbers are the analytic formulas and their tolerances.
"""
import math

import numpy as np
import pytest
import torch

from tensordg.euler_dg_2d import legendre_operators
from tensordg.kovasznay import (
    DEFAULT_WAVENUMBER,
    decay_rate,
    default_domain,
    kovasznay_exact,
    kovasznay_residual,
    modal_evaluator,
    solve_kovasznay_dg_2d,
)
from tensordg.ns_dg_2d import _boundary_flux, _face_primitives, _rhs_ns


def test_closed_form_is_exact():
    """Kovasznay's closed form solves steady NS to round-off (it *is* the ref)."""
    residual = kovasznay_residual(reynolds=40.0, points=48)
    assert residual["continuity_max"] < 1e-14
    assert residual["momentum_x_max"] < 1e-13
    assert residual["momentum_y_max"] < 1e-13
    assert residual["continuity_relative"] < 1e-13
    assert residual["momentum_x_relative"] < 1e-13
    assert residual["momentum_y_relative"] < 1e-13


@pytest.mark.parametrize("reynolds,wavenumber", [(40.0, 2 * math.pi), (10.0, 3.0),
                                                 (100.0, 2 * math.pi)])
def test_decay_rate_solves_the_characteristic_relation(reynolds, wavenumber):
    lam = decay_rate(reynolds, wavenumber)
    assert lam < 0.0
    assert abs(lam ** 2 - reynolds * lam - wavenumber ** 2) < 1e-12


def test_boundary_data_is_periodic_in_y():
    """The closed form repeats every ``2 pi / k`` in y, so y is periodic."""
    (xmin, xmax), (ymin, ymax) = default_domain(DEFAULT_WAVENUMBER)
    period = 2.0 * math.pi / DEFAULT_WAVENUMBER
    assert abs((ymax - ymin) - 2.0 * period) < 1e-12
    y = torch.linspace(ymin, ymax - period, 17, dtype=torch.float64)
    x = torch.linspace(xmin, xmax, 9, dtype=torch.float64)
    xg, yg = x[:, None], y[None, :]
    base = kovasznay_exact(xg, yg, 40.0, DEFAULT_WAVENUMBER)
    shifted = kovasznay_exact(xg, yg + period, 40.0, DEFAULT_WAVENUMBER)
    for key in ("u", "v", "p"):
        assert torch.max(torch.abs(base[key] - shifted[key])) < 1e-12


def _face_arrays(degree=2, ncells=4, sound_speed=40.0):
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(degree, order)
    (xmin, xmax), (ymin, ymax) = default_domain()
    dx = (xmax - xmin) / ncells
    dy = (ymax - ymin) / ncells
    y_centres = ymin + (torch.arange(ncells, dtype=torch.float64) + 0.5) * dy
    y_face = (y_centres[:, None] + 0.5 * dy * nodes[None, :])[None, :, :]
    x_face = torch.full_like(y_face, xmin)
    state = kovasznay_exact(x_face, y_face, 40.0, DEFAULT_WAVENUMBER)
    t0 = sound_speed ** 2 / 1.4
    pressure = t0 + state["p"]
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, dx=dx, dy=dy, ncells=ncells, x_face=x_face,
                y_face=y_face, u=state["u"], v=state["v"], pressure=pressure, t0=t0)


def test_prescribed_face_carries_mass_flux():
    """A wall dict carries no mass; ``"state"`` makes the face a through-flow."""
    arrays = _face_arrays()
    modes = arrays["basis"].shape[1]
    side = _face_primitives(
        torch.zeros(arrays["ncells"], arrays["ncells"], modes, modes, 4,
                    dtype=torch.float64),
        arrays["basis"], arrays["derivative"], arrays["signs"], 0, "left",
    )
    side = {key: value + 0.0 for key, value in side.items()}
    # a non-zero mass flux needs a non-zero interior velocity: perturb ``u``
    side["u"] = side["u"] + 1.0
    side["state"] = side["state"] + 0.0
    side["state"][..., 1] = side["state"][..., 1] + 1.0
    common = dict(axis=0, h=arrays["dx"], mu=0.025, kappa=0.025 * 3.5 / 0.71,
                  penalty=2.0 * 9.0, normal_scale=2.0 / arrays["dx"],
                  tangential_scale=2.0 / arrays["dy"], side_name="left",
                  outward_sign=-1.0)
    wall_only = {"velocity": (arrays["u"], arrays["v"]),
                 "temperature": arrays["pressure"] / arrays["t0"]}
    flux_wall = _boundary_flux(side, wall_only, **common)
    wall_state = dict(wall_only)
    wall_state["state"] = {"rho": torch.ones_like(arrays["u"]),
                           "velocity_x": arrays["u"],
                           "velocity_y": arrays["v"],
                           "pressure": arrays["pressure"]}
    flux_inlet = _boundary_flux(side, wall_state, **common)
    assert torch.max(torch.abs(flux_wall[..., 0])) < 1e-14
    assert torch.min(flux_inlet[..., 0]) > 0.1
    # the momentum in the normal direction must still carry the pressure
    assert torch.max(torch.abs(flux_wall[..., 1])) > 1.0
    assert torch.max(torch.abs(flux_inlet - flux_wall)) > 0.1


def test_modal_restart_is_lossless():
    """Re-projecting the modal evaluator reproduces the coefficients."""
    x, y, coeff, metadata = solve_kovasznay_dg_2d(
        ncells=6, degree=2, sound_speed=40.0, final_time=6e-4, chunks=1
    )
    evaluate = modal_evaluator(coeff, 6, 2, default_domain(), DEFAULT_WAVENUMBER)
    grid = 4
    order = max(2 + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(2, order)
    (xmin, xmax), (ymin, ymax) = default_domain()
    dx = (xmax - xmin) / 6
    dy = (ymax - ymin) / 6
    xq = xmin + (torch.arange(6, dtype=torch.float64)[:, None, None, None]
                 + 0.5 * (nodes[None, None, :, None] + 1.0)) * dx
    yq = ymin + (torch.arange(6, dtype=torch.float64)[None, :, None, None]
                 + 0.5 * (nodes[None, None, None, :] + 1.0)) * dy
    xq = xq.expand(6, 6, nodes.numel(), nodes.numel()).clone()
    yq = yq.expand(6, 6, nodes.numel(), nodes.numel()).clone()
    rho, u, v, p = evaluate(xq, yq)
    from tensordg.euler_dg_2d import _conservative

    state = _conservative(rho, u, v, p)
    projected = torch.einsum("xyabv,a,b,ak,bl->xyklv", state, weights, weights,
                             basis, basis)
    modes = torch.arange(3, dtype=torch.float64)
    factor = 0.25 * torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0)
    projected = projected * factor.view(1, 1, 3, 3, 1)
    scale = coeff.abs().max()
    assert torch.max(torch.abs(projected - coeff)) / scale < 1e-12
    del grid


def test_steady_error_shrinks_with_degree():
    """Design order: the P2 steady error is far below the P1 steady error."""
    errors = {}
    for degree in (1, 2):
        x, y, coeff, metadata = solve_kovasznay_dg_2d(
            ncells=8, degree=degree, sound_speed=60.0, final_time=0.1, chunks=4
        )
        errors[degree] = metadata["history"][-1]["velocity"]
    print("steady velocity error on 8x8:", errors)
    assert errors[1] < 0.10
    # at 8 cells the P1/P2 errors are still in the pre-asymptotic regime
    # (k h is O(1)), so the ratio is 6.2 here and approaches 1/h = 8 on refine;
    # a ratio of 5 already rules out a non-design-order (e.g. clipping) error
    assert errors[2] < errors[1] / 5.0