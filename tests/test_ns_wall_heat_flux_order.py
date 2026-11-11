"""Order pins for the isothermal-wall heat flux of ``tensordg.ns_dg_2d`` (m13815).

Re-adjudication of the two ``xfail(strict=True)`` pins in
``tests/test_ns_heat_conduction.py``.  Those pins ask the *wall-row* energy
residual of the exact steady conduction state to decay under refinement and
read its stagnation as an "inconsistent isothermal wall energy closure".  The
measurements below show that reading is wrong:

* the wall heat flux the closure puts into the strong-form divergence converges
  to the exact ``-kappa dT/dn = kappa / 2`` at the **design order** (1.03 at P1,
  2.07 at P2), i.e. the closure *is* consistent;
* the neighbouring interior face is superconvergent (order 2.06 at both
  degrees), so the wall row sees a *flux mismatch* of size ``O(h**p)``;
* the strong-form residual divides that mismatch by ``h``, hence the wall-row
  residual can only decay at order ``p - 1`` (0 at P1, 1 at P2) -- which is
  exactly what is measured (0.08 / 0.19 at P1, 1.06 at P2);
* the volume-weighted residual of the whole operator does decay
  (order 0.48 / 0.73 at P1, 0.98 at P2), i.e. the O(1) part is confined to one
  measure-``O(h)`` boundary row.

So the stagnation of the wall-row residual is a property of the strong-form
operator, not a defect of the boundary condition.  These pins freeze the
mechanism with machine-checkable inequalities; the evidence file is
``validation/ns_heat_conduction/wall_flux_order.json``
(``python -m benchmarks.verify_ns_wall_heat_flux``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tensordg.ns_dg_2d as ns

MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
T_HOT, T_COLD, PRESSURE = 1.0, 0.5, 1.0
EVIDENCE = Path("validation/ns_heat_conduction/wall_flux_order.json")


def _steady(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    temperature = T_HOT + (T_COLD - T_HOT) * (y + 0.0 * x)
    return (PRESSURE / temperature, torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def _measure(ny, degree, wall_gradient):
    nx = 2
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(_steady, x, y, dx, dy, nodes, weights, basis,
                                     torch.device("cpu"), torch.float64)
    penalty = 2.0 * (degree + 1.0) ** 2
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
                "ymax": {"velocity": (0.0, 0.0), "temperature": T_COLD}}
    rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                     (True, False), boundary, MU, KAPPA, penalty, wall_gradient)
    right_y, left_y, _, _ = ns._axis_fluxes_ns(
        coeff, basis, derivative, signs, weights, dx, dy, False,
        boundary["ymin"], boundary["ymax"], MU, KAPPA, penalty, 1, wall_gradient)
    modes = torch.arange(degree + 1, dtype=torch.float64)
    mass = 1.0 / ((2.0 * modes + 1.0) * (2.0 * modes + 1.0))
    modal = rhs[..., 3]
    l2 = float((modal.pow(2) * mass.view(1, 1, *mass.shape, 1)).sum().sqrt()) * math.sqrt(dx * dy / 4.0)
    return {
        "wall_flux": float(left_y[:, 0, 0, 3].mean()) / 2.0,
        "interior_flux": float(right_y[:, 0, 0, 3].mean()) / 2.0,
        "row_residual": float(modal[:, 0].abs().max()),
        "l2_residual": l2,
    }


def _order(values):
    cells = (8, 16, 32, 64)
    xs = [math.log(n) for n in cells]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


@pytest.mark.parametrize("degree,min_order", [(1, 0.9), (2, 1.8)])
def test_wall_heat_flux_converges_at_the_design_order(degree, min_order):
    """The Dirichlet wall energy closure is *consistent*: its flux converges."""
    exact = KAPPA * 0.5
    for wall_gradient in ("taylor", "face"):
        errors = [abs(_measure(ny, degree, wall_gradient)["wall_flux"] - exact)
                  for ny in (8, 16, 32, 64)]
        order = _order(errors)
        assert order > min_order, (degree, wall_gradient, order, errors)
        assert errors[-1] < 5.0e-3 / degree, (degree, wall_gradient, errors[-1])


def test_interior_face_flux_is_superconvergent():
    """The neighbouring interior face is O(h^2) at both degrees -- the mismatch
    that the wall row divides by h is therefore O(h^p) vs O(h^2), not a closure error."""
    exact = KAPPA * 0.5
    for degree in (1, 2):
        errors = [abs(_measure(ny, degree, "taylor")["interior_flux"] - exact)
                  for ny in (8, 16, 32, 64)]
        assert _order(errors) > 1.9, (degree, errors)


@pytest.mark.parametrize("degree,row_order", [(1, (0.0, 0.45)), (2, (0.9, 1.35))])
def test_wall_row_residual_order_equals_flux_order_minus_one(degree, row_order):
    """Mechanism pin: the strong-form wall-row residual decays at ``p - 1``,
    because it is the O(h^p) boundary-flux error divided by the cell width h."""
    residuals = [_measure(ny, degree, "taylor")["row_residual"] for ny in (8, 16, 32, 64)]
    order = _order(residuals)
    assert row_order[0] <= order <= row_order[1], (degree, order, residuals)


@pytest.mark.parametrize("degree,min_order", [(1, 0.4), (2, 0.9)])
def test_volume_weighted_residual_decays(degree, min_order):
    """The O(1) wall residual is confined to one boundary row: the L2 norm of the
    residual of the whole operator decays under refinement."""
    l2 = [_measure(ny, degree, "taylor")["l2_residual"] for ny in (8, 16, 32, 64)]
    order = _order(l2)
    assert order > min_order, (degree, order, l2)


@pytest.mark.skipif(not EVIDENCE.exists(), reason="run benchmarks.verify_ns_wall_heat_flux")
def test_evidence_artifact_matches_the_pins():
    data = json.loads(EVIDENCE.read_text())
    assert abs(data["kappa"] - KAPPA) < 1e-12
    assert abs(data["exact_wall_flux"] - KAPPA * 0.5) < 1e-12
    orders = data["orders"]
    assert orders["p1_taylor"]["wall_flux_error"] > 0.9
    assert orders["p2_taylor"]["wall_flux_error"] > 1.8
    assert orders["p1_taylor"]["row_residual"] < 0.45
    assert orders["p1_taylor"]["l2_residual"] > 0.4