"""Pins for the cell-centre stencil wall heat-flux estimator (m16493).

What is frozen here
-------------------
* the stencil weights are the exact first-derivative weights at a Dirichlet wall
  (moment conditions, two independent formulas for ``points = 1`` / ``2``);
* the estimator is **exact** on a state whose temperature is a polynomial the
  stencil resolves (quadratic temperature, ``points = 2``);
* ``points = 2`` reproduces the operator's own opt-in ``fd2`` wall flux
  elementwise (independent code path -> the sign/normal conventions are
  validated, not assumed);
* measured convergence orders on the exact steady conduction state: order 3 at
  P2 (one above the solver flux) and the order-1 floor at P1, which is why the
  face extrapolation stays the P1 recommendation;
* the API contract: both walls share a sign, tangential cells keep their own
  value, a periodic axis / empty boundary yields nothing, and a stencil that
  would leave the domain or an adiabatic wall raises ``ValueError``;
* the evidence artifact ``validation/ns_heat_conduction/wall_flux_stencil.json``
  exists, is self-consistent (its stored order is re-derived from its stored
  errors) and its predeclared verdicts all pass.

Run: ``PYTHONPATH=. python -m pytest -q tests/test_ns_wall_flux_stencil.py``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import tensordg.ns_dg_2d as ns
from benchmarks.verify_ns_wall_flux_stencil import (
    EXACT_WALL_FLUX,
    KAPPA,
    MU,
    T_COLD,
    T_HOT,
    _boundary_x,
    _projected_state,
    _projected_state_x,
    observed_order,
)
from tensordg.wall_flux_stencil import (
    stencil_weights,
    wall_heat_flux_stencil,
    wall_heat_flux_stencil_summary,
)

EVIDENCE = Path("validation/ns_heat_conduction/wall_flux_stencil.json")
CELLS = (8, 16, 32, 64)
DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _boundary(gradient="face"):
    return {"xmin": None, "xmax": None,
            "ymin": {"velocity": (0.0, 0.0), "temperature": T_HOT},
            "ymax": {"velocity": (0.0, 0.0), "temperature": T_COLD}}


def _values(ncells, degree, points, family="y", boundary=None, periodic=None):
    project = _projected_state if family == "y" else _projected_state_x
    coeff, _, (dx, dy) = project(ncells, degree)
    if boundary is None:
        boundary = _boundary() if family == "y" else _boundary_x()
    if periodic is None:
        periodic = (True, False) if family == "y" else (False, True)
    return wall_heat_flux_stencil_summary(coeff, dx, dy, boundary, KAPPA,
                                          points, periodic)


# ----------------------------------------------------------------------
# stencil weights
# ----------------------------------------------------------------------

def test_two_point_weights_are_the_one_sided_difference():
    assert stencil_weights(1) == [-2.0, 2.0]


def test_parabola_weights_match_the_closed_form():
    weights = stencil_weights(2)
    assert weights == pytest.approx([-8.0 / 3.0, 3.0, -1.0 / 3.0], abs=0.0)


@pytest.mark.parametrize("points", [1, 2, 3, 4, 5])
def test_weights_are_the_exact_derivative_weights(points):
    """``sum_j w_j s_j**k == delta_{k,1}`` for the nodes ``s = 0, 1/2, 3/2, ...``."""
    weights = stencil_weights(points)
    nodes = [0.0] + [index - 0.5 for index in range(1, points + 1)]
    assert len(weights) == points + 1
    for power in range(points + 1):
        moment = sum(w * s ** power for w, s in zip(weights, nodes))
        target = 1.0 if power == 1 else 0.0
        assert moment == pytest.approx(target, abs=1e-12), (points, power, moment)


@pytest.mark.parametrize("points", [0, -1])
def test_weights_reject_non_positive_points(points):
    with pytest.raises(ValueError):
        stencil_weights(points)


# ----------------------------------------------------------------------
# exactness on a representable state
# ----------------------------------------------------------------------

def _quadratic_state(x, y):
    """Conserved state whose temperature is ``1 + y + y**2`` (no flow).

    ``_initial_coefficients`` calls ``initial(xq, yq)`` with quadrature-point
    tensors and maps the result through ``_conservative(rho, u, v, p)`` with
    ``R = 1``, so the pressure passed here is the temperature itself (``rho = 1``).
    """
    temperature = 1.0 + y + y * y + 0.0 * x
    zeros = torch.zeros_like(temperature)
    return (torch.ones_like(temperature), zeros, zeros, temperature)


def _quadratic_coeff(ny, degree):
    dx, dy = 0.25, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, DEVICE, DTYPE)
    x = (torch.arange(4, dtype=DTYPE) + 0.5) * dx
    y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    return (ns._initial_coefficients(_quadratic_state, x, y, dx, dy, nodes,
                                     weights, basis, DEVICE, DTYPE),
            (nodes, weights, basis, derivative, signs), (dx, dy))


def test_parabola_stencil_is_exact_on_a_representable_parabola():
    """``T = 1 + y + y**2``: ``dT/dy = 1`` at ``y = 0`` and ``3`` at ``y = 1``,
    so ``q = -kappa dT/dy`` must be exact to round-off at both walls."""
    coeff, _, (dx, dy) = _quadratic_coeff(8, 2)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0 + 1.0 + 1.0}}
    flux = wall_heat_flux_stencil_summary(coeff, dx, dy, boundary, KAPPA,
                                          points=2, periodic=(True, False))
    assert flux["ymin"] == pytest.approx(-KAPPA, abs=1e-12), flux
    assert flux["ymax"] == pytest.approx(-3.0 * KAPPA, abs=1e-12), flux


def test_two_point_stencil_is_not_exact_on_a_parabola():
    """The negative control: ``points = 1`` misses the curvature by ``O(h)``."""
    coeff, _, (dx, dy) = _quadratic_coeff(8, 2)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (0.0, 0.0), "temperature": 3.0}}
    flux = wall_heat_flux_stencil_summary(coeff, dx, dy, boundary, KAPPA,
                                          points=1, periodic=(True, False))
    assert abs(flux["ymin"] + KAPPA) > 1e-3


# ----------------------------------------------------------------------
# cross-validation against the operator's own fd2 stencil
# ----------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("family", ["y", "x"])
def test_parabola_stencil_reproduces_the_operator_fd2_flux(degree, family):
    """Same stencil, independent code paths: agreement must be round-off."""
    project = _projected_state if family == "y" else _projected_state_x
    coeff, ops, (dx, dy) = project(16, degree)
    nodes, weights, basis, derivative, signs = ops
    boundary = _boundary() if family == "y" else _boundary_x()
    periodic = (True, False) if family == "y" else (False, True)
    penalty = 2.0 * (degree + 1.0) ** 2
    operator = ns.wall_heat_flux(
        coeff, basis, derivative, signs, weights, dx, dy, boundary, MU, KAPPA,
        penalty, "fd2", periodic=periodic)
    stencil = wall_heat_flux_stencil(coeff, dx, dy, boundary, KAPPA, points=2,
                                    periodic=periodic)
    assert set(operator) == set(stencil)
    for name in operator:
        worst = float((operator[name] - stencil[name]).abs().max())
        assert worst <= 1e-12, (name, worst)


# ----------------------------------------------------------------------
# measured orders (recomputed here, not read from the artifact)
# ----------------------------------------------------------------------

def _order(points, degree, family="y", wall=None):
    values = [_values(ny, degree, points, family) for ny in CELLS]
    wall = wall or ("ymin" if family == "y" else "xmin")
    errors = [abs(entry[wall] - EXACT_WALL_FLUX) for entry in values]
    return observed_order(errors, CELLS), errors


def test_stencil_is_third_order_at_p2():
    """One order above the solver flux: the reason this estimator exists."""
    order, errors = _order(1, 2)
    assert order > 2.75, (order, errors)
    assert errors[-1] < 1e-8, errors


def test_stencil_floors_at_first_order_at_p1():
    """Honest floor: the ``O(h**2)`` centre bias over a one-sided ``O(1/h)`` weight."""
    for points in (1, 2, 3):
        order, errors = _order(points, 1)
        assert 0.7 < order <= 1.3, (points, order, errors)


def test_p2_stencil_error_is_far_below_the_solver_flux():
    coeff, ops, (dx, dy) = _projected_state(32, 2)
    nodes, weights, basis, derivative, signs = ops
    penalty = 2.0 * (2 + 1.0) ** 2
    solver = ns.wall_heat_flux_summary(coeff, basis, derivative, signs, weights,
                                       dx, dy, _boundary(), MU, KAPPA, penalty,
                                       "face", periodic=(True, False))
    stencil = wall_heat_flux_stencil_summary(coeff, dx, dy, _boundary(), KAPPA, 1,
                                            (True, False))
    solver_err = abs(solver["ymin"] - EXACT_WALL_FLUX)
    stencil_err = abs(stencil["ymin"] - EXACT_WALL_FLUX)
    assert solver_err > 100.0 * stencil_err, (solver_err, stencil_err)


def test_x_walls_reproduce_the_y_walls_result():
    """Independent analytic family (walls on the other axis), not rotation-only."""
    y_order, y_errors = _order(1, 2, "y")
    x_order, x_errors = _order(1, 2, "x")
    assert x_order == pytest.approx(y_order, abs=0.1)
    assert x_errors[-1] == pytest.approx(y_errors[-1], rel=1e-6)


# ----------------------------------------------------------------------
# sign convention, tangential values and API contract
# ----------------------------------------------------------------------

def test_both_walls_report_the_same_positive_flux():
    coeff, _, (dx, dy) = _projected_state(32, 2)
    flux = wall_heat_flux_stencil(coeff, dx, dy, _boundary(), KAPPA, 1,
                                  (True, False))
    bottom = float(flux["ymin"].mean())
    top = float(flux["ymax"].mean())
    assert bottom > 0.0 and top > 0.0
    assert bottom == pytest.approx(EXACT_WALL_FLUX, rel=1e-4)
    assert top == pytest.approx(bottom, rel=1e-3)


def test_tangential_cells_keep_their_own_stencil():
    """A wall cell's value depends only on its own normal column of centres."""
    coeff, _, (dx, dy) = _projected_state(16, 1)
    flux = wall_heat_flux_stencil(coeff, dx, dy, _boundary(), KAPPA, 1,
                                  (True, False))
    assert flux["ymin"].shape == (coeff.shape[0],)
    assert float(flux["ymin"].std()) < 1e-12  # the analytic state is x-uniform


def test_anisotropic_mesh_uses_only_the_normal_spacing():
    """``dx`` must not enter a ``y``-wall stencil: same ``ny`` -> same flux."""
    flux = {}
    for nx in (2, 8):
        dx, dy = 1.0 / nx, 1.0 / 16
        nodes, weights, basis, derivative, signs = ns.legendre_operators(
            2, 4, DEVICE, DTYPE)
        x = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
        y = (torch.arange(16, dtype=DTYPE) + 0.5) * dy

        def state(xx, yy):
            temperature = T_HOT + (T_COLD - T_HOT) * yy + 0.0 * xx
            zeros = torch.zeros_like(temperature)
            return (1.0 / temperature, zeros, zeros, torch.ones_like(temperature))

        coeff = ns._initial_coefficients(state, x, y, dx, dy, nodes, weights,
                                         basis, DEVICE, DTYPE)
        flux[nx] = wall_heat_flux_stencil_summary(coeff, dx, dy, _boundary(),
                                                  KAPPA, 1, (True, False))
    assert flux[2]["ymin"] == pytest.approx(flux[8]["ymin"], rel=1e-13)


def test_empty_and_periodic_axes_report_nothing():
    coeff, _, (dx, dy) = _projected_state(8, 1)
    assert wall_heat_flux_stencil(coeff, dx, dy, {}, KAPPA) == {}
    assert wall_heat_flux_stencil(coeff, dx, dy, {"xmin": None, "xmax": None,
                                                  "ymin": None, "ymax": None},
                                  KAPPA) == {}
    # a periodic axis is skipped even when it carries a wall description
    periodic = wall_heat_flux_stencil(coeff, dx, dy, _boundary(), KAPPA, 1,
                                      (True, False))
    assert set(periodic) == {"ymin", "ymax"}
    assert wall_heat_flux_stencil(coeff, dx, dy, _boundary(), KAPPA, 1,
                                  (True, True)) == {}


def test_stencil_wider_than_the_axis_raises():
    coeff, _, (dx, dy) = _projected_state(2, 1)
    with pytest.raises(ValueError):
        wall_heat_flux_stencil(coeff, dx, dy, _boundary(), KAPPA, 3, (True, False))


def test_adiabatic_or_free_slip_wall_raises():
    """The estimator needs the imposed temperature; a wall without one is refused."""
    coeff, _, (dx, dy) = _projected_state(8, 1)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0)}, "ymax": None}
    with pytest.raises(ValueError):
        wall_heat_flux_stencil(coeff, dx, dy, boundary, KAPPA)


def test_estimator_is_blind_to_the_wall_closure():
    """Structural pin: no ``wall_gradient``/``wall_penalty`` argument exists."""
    import inspect

    params = set(inspect.signature(wall_heat_flux_stencil).parameters)
    assert not params & {"wall_gradient", "wall_penalty", "penalty", "basis"}


# ----------------------------------------------------------------------
# evidence artifact
# ----------------------------------------------------------------------

def test_evidence_artifact_is_self_consistent():
    data = json.loads(EVIDENCE.read_text())
    assert data["meta"]["kappa"] == pytest.approx(KAPPA)
    assert data["meta"]["ref_flux"] == pytest.approx(EXACT_WALL_FLUX)
    assert data["verdict"]["all_pass"] is True
    assert {case["family"] for case in data["cases"]} == {"y", "x"}
    for case in data["cases"]:
        for entry in case["estimators"].values():
            for wall in case["walls"]:
                stored = entry[wall]
                rederived = observed_order(stored["errors"], case["cells"])
                assert rederived == pytest.approx(stored["order"], abs=1e-9), (
                    case["family"], case["degree"], wall)
                assert stored["errors"][-1] == pytest.approx(
                    abs(stored["values"][-1] - EXACT_WALL_FLUX), rel=1e-12)


def test_evidence_artifact_records_the_ladder():
    data = json.loads(EVIDENCE.read_text())
    orders = {}
    for case in data["cases"]:
        if case["family"] != "y":
            continue
        for label, entry in case["estimators"].items():
            orders[(case["degree"], label)] = entry["ymin"]["order"]
    # P2: the stencil is one order above the solver flux and above the
    # extrapolation, while at P1 it floors at order 1 (the honest negative)
    assert orders[(2, "stencil1")] > orders[(2, "solver")] + 0.5
    assert orders[(2, "stencil1")] > orders[(2, "extrapolated")] + 0.5
    assert orders[(1, "stencil1")] < 1.3 < orders[(1, "extrapolated")]
    assert math.isfinite(orders[(3, "solver")])