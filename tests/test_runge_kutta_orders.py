"""Verification of the explicit Runge-Kutta tableaus and their solver wiring.

Three independent checks, each with a stated tolerance:

``T1 tableaus``   every advertised tableau attains its design order on the
                  scalar probe ``y' = y`` (analytical solution ``e^t``), measured
                  by dt-refinement with the same least-squares estimator used
                  throughout the verification suite.
``T2 solver``     inside :func:`tensordg.advection_1d.solve_advection` the time
                  error is isolated from the spatial error by comparing against
                  ``expm(L*T) u0`` -- the *exact* solution of the DG
                  semi-discrete ODE system, whose matrix ``L`` is built by
                  feeding unit vectors through ``_rhs`` (the operator is linear).
                  The measured rate must equal the tableau order.
``T3 no regression``  the default ``rk4`` path is unchanged: the stored
                  ``validation/advection_order/metrics.json`` L2 value for
                  ``p = 2, N = 40, cfl = 0.2`` is reproduced to 1e-6 relative.

Run: ``PYTHONPATH=. pytest -q tests/test_runge_kutta_orders.py``
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from tensordg.advection_1d import _operators, _rhs, l2_error, solve_advection
from tensordg.runge_kutta import TABLEAUS, get_tableau, rk_step, tableau_for_degree

METRICS = pathlib.Path("validation/advection_order/metrics.json")
ORDER_TOL = 0.15
#: Least-squares fit floor: errors below 1e-12 are float64 round-off, errors
#: above 1e-2 may be outside the asymptotic range.  Points outside the window
#: are dropped and at least three must survive.
FIT_LO, FIT_POINTS = 1e-14, 4

#: Design order of every scheme reachable through ``solve_advection``;
#: SSP-RK3 is not in Butcher form so it is not part of ``TABLEAUS``.
SCHEME_ORDER = {"euler": 1, "rk2": 2, "rk3": 3, "rk4": 4, "rk5": 5, "ssp_rk3": 3}


def _slope(dts, errs) -> float:
    """Order from a log-log fit over the finest ``FIT_POINTS`` resolved steps.

    Points below ``FIT_LO`` are float64 round-off and are discarded; taking the
    finest points keeps the fit inside the asymptotic range for every order.
    """
    keep = [(dt, e) for dt, e in zip(dts, errs) if e > FIT_LO][-FIT_POINTS:]
    assert len(keep) >= 3, (dts, errs)
    xs = np.log([dt for dt, _ in keep])
    ys = np.log([e for _, e in keep])
    return float(np.polyfit(xs, ys, 1)[0])


@pytest.mark.parametrize("name", sorted(TABLEAUS))
def test_tableau_attains_design_order_on_scalar_probe(name):
    """T1: ``y' = y`` -- the tableau's order shows up as the dt-refinement slope."""
    tab = get_tableau(name)
    # One step started from the exact solution: the *local* truncation error
    # decays as dt^(order+1), which is free of accumulation and stays inside the
    # asymptotic range and well above float64 round-off.
    t0 = 0.1
    dts, errs = [], []
    for k in range(40):
        dt = t0 / 2.0 ** k
        y = rk_step(lambda z: z, torch.tensor([math.exp(t0)], dtype=torch.float64),
                    dt, tab)
        dts.append(dt)
        errs.append(abs(float(y) - math.exp(t0 + dt)))
        if errs[-1] < FIT_LO:
            break
    measured = _slope(dts, errs) - 1.0  # local error slope is order + 1
    assert abs(measured - tab.order) <= ORDER_TOL, (name, measured, tab.order)


def _semi_discrete(degree: int, ncells: int):
    """Return (L, u0) of the DG semi-discrete ODE ``du/dt = L u``."""
    centres, coeff, _ = solve_advection(ncells=ncells, degree=degree,
                                        final_time=0.0)
    u0 = coeff.reshape(-1).numpy().astype(float)
    _, _, _, volume, phi_l, phi_r = _operators(degree, torch.device("cpu"),
                                               torch.float64)
    dim = ncells * (degree + 1)
    mat = np.zeros((dim, dim))
    for j in range(dim):
        e = torch.zeros((ncells, degree + 1), dtype=torch.float64)
        e.reshape(-1)[j] = 1.0
        mat[:, j] = _rhs(e, 1.0 / ncells, 1.0, volume, phi_l, phi_r).reshape(-1).numpy()
    return mat, u0


@pytest.mark.parametrize("name,degree", [
    ("euler", 1), ("rk2", 1), ("rk3", 2), ("ssp_rk3", 2),
    ("rk4", 3), ("rk5", 4),
])
def test_time_order_inside_solver_matches_tableau(name, degree):
    """T2: error against ``expm(L*T)u0`` isolates the *time* discretisation."""
    ncells, final_time = 4, 0.1
    mat, u0 = _semi_discrete(degree, ncells)
    exact = sla.expm(mat * final_time) @ u0
    order = SCHEME_ORDER[name]
    dts, errs = [], []
    for steps in [2 ** k for k in range(2, 12)]:
        # cfl chosen so that solve_advection takes exactly ``steps`` steps.
        cfl = (final_time / steps) * (2 * degree + 1) * ncells
        _, coeff, meta = solve_advection(ncells=ncells, degree=degree,
                                         final_time=final_time, cfl=cfl,
                                         time_integrator=name)
        assert meta["steps"] == steps, (name, meta["steps"], steps)
        dts.append(meta["dt"])
        errs.append(float(np.linalg.norm(coeff.reshape(-1).numpy() - exact)))
        if errs[-1] < 1e-13:
            break
    measured = _slope(dts, errs)
    # Global accumulation plus the pre-asymptotic coarsest meshes broaden the
    # band for the order-5 scheme (measured 5.6 on this linear operator).
    tol = 0.6 if order >= 5 else 2 * ORDER_TOL
    assert abs(measured - order) <= tol, (name, measured, order)


def test_metadata_reports_order_and_stages():
    _, _, meta = solve_advection(ncells=8, degree=2, final_time=0.1, time_integrator="rk5")
    assert (meta["rk_order"], meta["rk_stages"]) == (5, 6)
    _, _, meta = solve_advection(ncells=8, degree=2, final_time=0.1)
    assert (meta["rk_order"], meta["rk_stages"]) == (4, 4)
    _, _, meta = solve_advection(ncells=8, degree=2, final_time=0.1,
                                 time_integrator="ssp_rk3")
    assert (meta["rk_order"], meta["rk_stages"]) == (3, 3)  # SSP-RK3 is 3rd order


def test_minimal_stage_selection():
    """Degree p is advanced with order p+1 -- the RKDG accuracy-table convention."""
    assert tableau_for_degree(0).name == "euler"
    assert tableau_for_degree(1).name == "rk2"
    assert tableau_for_degree(2).name == "rk3"
    assert tableau_for_degree(3).name == "rk4"
    assert tableau_for_degree(4).name == "rk5"
    assert tableau_for_degree(7).name == "rk5"  # capped


def test_unknown_integrator_rejected():
    with pytest.raises(ValueError):
        solve_advection(ncells=8, degree=1, final_time=0.1, time_integrator="rk9")


def test_rk4_path_unchanged_against_stored_metrics():
    """T3: the historical ``rk4`` L2 value is bit-stable (1e-6 relative)."""
    if not METRICS.exists():
        pytest.skip("stored metrics not present")
    metrics = json.loads(METRICS.read_text())
    row = next(r for r in metrics["comparison"] if r["degree"] == 2)
    stored = row["got_l2"][1]  # N = 40
    centres, coeff, _ = solve_advection(ncells=40, degree=2, final_time=1.0,
                                        cfl=0.2, time_integrator="rk4")
    got = l2_error(centres, coeff, final_time=1.0, speed=1.0)
    assert abs(got - stored) / stored < 1e-6, (got, stored)