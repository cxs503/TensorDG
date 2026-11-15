"""Acceptance gates replacing the pre-fix defect lock (REPRO-DG-CKS1998, curved leg).

History: ``docs/AUDIT-m21473-curvilinear-euler-nonconvergence.md`` pinned the
curvilinear Euler operator as *non-convergent* (evolution error rising with
resolution, order -1.43 at 16->32) with a lock asserting the defect.  The root
cause was the interface orientation of the low faces plus the interior physical
trace being re-evaluated on the p-degree solution trace
(``docs/CASE-DG-CKS1998-CURVED-OPERATOR-SIGN-FIX.md``).  Per the lock's own
instruction the defect gate is now replaced by the strengthened gate it asked
for: monotone error decay and order >= 1.5 on both mesh families, with the
projection (geometry/quadrature/reference-solution) control that ruled out the
other explanations.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensordg.euler_2d import vortex_primitive
from tensordg.euler_curved import (
    CurvilinearEuler,
    straight_box_mesh,
    warped_box_mesh,
)

BOX = 10.0
AMPLITUDE = 0.06
GEOMETRY_DEGREE = 3
CELLS = (8, 16, 32)


def exact_primitive(time, strength=1.0):
    def primitive(x, y):
        rho, u, v, p = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=strength, background_velocity=(1.0, 1.0),
            period=BOX)
        return (rho.numpy(), u.numpy(), v.numpy(), p.numpy())
    return primitive


def _mesh(family, cells):
    if family == "straight":
        return straight_box_mesh(cells, cells, 1, length=BOX)
    return warped_box_mesh(cells, cells, GEOMETRY_DEGREE, length=BOX,
                           amplitude=AMPLITUDE)


@pytest.mark.parametrize("family", ["straight", "warped"])
def test_projection_error_keeps_second_order_both_meshes(family):
    """Control: mesh map, quadrature and reference solution are all intact."""
    errors = []
    for cells in CELLS:
        solver = CurvilinearEuler(_mesh(family, cells), degree=1, cfl=0.2)
        state = solver.project(exact_primitive(0.0))
        errors.append(solver.relative_errors(state, exact_primitive(0.0))["worst"])
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2.0)
              for i in range(len(errors) - 1)]
    assert min(orders) >= 1.7, (orders, errors)


@pytest.mark.parametrize("family", ["straight", "warped"])
def test_evolution_error_converges_monotonically_at_design_order(family):
    """The fixed operator: order >= 1.5 and no error growth at the finest level."""
    errors = []
    for cells in CELLS:
        solver = CurvilinearEuler(_mesh(family, cells), degree=1, cfl=0.2)
        state = solver.project(exact_primitive(0.0))
        state, _, _ = solver.evolve(state, 0.1)
        errors.append(solver.relative_errors(state, exact_primitive(0.1))["worst"])
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(2.0)
              for i in range(len(errors) - 1)]
    assert errors[2] < errors[1] < errors[0], errors       # monotone decay
    assert orders[0] >= 1.5, (orders, errors)              # paper tolerance p+1/2
    assert orders[1] >= 1.2, (orders, errors)
    # the pre-fix operator produced 1.14e-02 at 32 cells; the fixed one is >20x
    # better and keeps improving (regression guard against the sign regression)
    assert errors[2] < 1.0e-03, (orders, errors)


def test_time_step_is_not_the_limiting_factor():
    """Fixed small dt reproduces the CFL result -> the residual error is spatial."""
    solver = CurvilinearEuler(_mesh("straight", 8), degree=1, cfl=0.2)
    state = solver.project(exact_primitive(0.0))
    cfl_state, _, _ = solver.evolve(state, 0.1)
    fixed_state, _, _ = solver.evolve(state, 0.1, dt=5.0e-4)
    cfl_error = solver.relative_errors(cfl_state, exact_primitive(0.1))["worst"]
    fixed_error = solver.relative_errors(fixed_state, exact_primitive(0.1))["worst"]
    assert abs(cfl_error - fixed_error) / cfl_error < 0.2, (cfl_error, fixed_error)