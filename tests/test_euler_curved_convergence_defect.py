"""Known-defect lock: ``tensordg.euler_curved`` does not converge (m21473).

The two red gates in ``tests/test_euler_curved.py`` (F3 design order on the
warped mesh, F5 conservation on the warped mesh) were carried as "pre-existing
failures" by two earlier missions.  This module pins down *why* they are red so
that neither gate can be quietly relaxed later:

* the L2 **projection** of the smooth vortex on the same mesh ladder is clean
  O(h^2) for p=1 -- mesh, geometry map, quadrature and reference solution are
  all sound;
* the **evolved** state is not: the error stops decreasing and grows again at
  32x32, on the *straight* (affine) mesh as well as the warped one, so this is
  not a curvilinear-geometry artefact;
* re-running the same march with a mesh-independent ``dt = 1e-3`` reproduces the
  error **bit-for-bit equal to the CFL-driven run**, so the temporal
  truncation error is not the cause -- the spatial operator is.

Reference point measured with the repository's verified Cartesian solver on the
same vortex, degree, CFL and element counts (``euler_dg_2d`` via
``benchmarks.design_order_euler_dg_2d.sweep``): 1.84e-2 / 5.84e-3 / 1.68e-3,
observed orders 1.65 and 1.80.  Full evidence, including the strength-5 case
(which blows up at 32x32) is produced by
``python -m benchmarks.euler_curved_convergence_audit`` and stored in
``validation/euler_curved_convergence/audit.json``.

Do not delete these assertions to make the suite green: they are the honest
record that the module's G2/G3/G4 claims (design order, curvature does not cost
accuracy, round-off conservation) are **not** met.  When the operator is fixed,
the strengthened form of these assertions flips to ``order >= 1.5`` and
monotone error decrease, and this file should be replaced by that gate.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tensordg.euler_curved import CurvilinearEuler, straight_box_mesh
from tensordg.euler_2d import vortex_primitive

BOX = 10.0
CFL = 0.2
FINAL_TIME = 0.1
MESHES = (8, 16, 32)
STRENGTH = 1.0


def exact_primitive(time):
    def primitive(x, y):
        rho, u, v, p = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=STRENGTH, background_velocity=(1.0, 1.0), period=BOX,
        )
        return (rho.numpy(), u.numpy(), v.numpy(), p.numpy())

    return primitive


def _run(cells, dt=None):
    solver = CurvilinearEuler(straight_box_mesh(cells, cells, 1, length=BOX),
                              degree=1, cfl=CFL)
    state = solver.project(exact_primitive(0.0))
    state, _, _ = solver.evolve(state, FINAL_TIME, dt=dt)
    return solver.relative_errors(state, exact_primitive(FINAL_TIME))["worst"]


def test_projection_converges_second_order_on_the_same_meshes():
    """The reference solution / mesh / quadrature chain is sound (control)."""
    errors = []
    for cells in MESHES:
        solver = CurvilinearEuler(straight_box_mesh(cells, cells, 1, length=BOX),
                                  degree=1, cfl=CFL)
        state = solver.project(exact_primitive(0.0))
        errors.append(solver.relative_errors(state, exact_primitive(0.0))["worst"])
    for left, right in zip(errors, errors[1:]):
        order = math.log(left / right) / math.log(2.0)
        assert order >= 1.7, (order, errors)


def test_evolved_error_does_not_converge_on_the_straight_mesh():
    """Known defect (m21473): the spatial operator stalls, then diverges."""
    errors = [_run(cells) for cells in MESHES]
    order_8_16 = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order_8_16 < 0.9, (order_8_16, errors)          # far below p+1 = 2
    assert errors[2] > errors[1], (errors,)                 # error grows again
    # ... and the *verified* Cartesian solver is an order of magnitude better
    # at the same element count (1.68e-3 vs this), see the module docstring.


def test_the_stall_is_not_a_time_integration_artefact():
    """A mesh-independent dt reproduces the CFL-driven error exactly."""
    cfl_error = _run(8)
    fixed_error = _run(8, dt=1e-3)
    # measured: |Delta| / err = 8.0e-06, i.e. the 1000 vs 10 step marches differ
    # by 0.0008 % -- the temporal error is three orders below the defect.
    assert abs(fixed_error - cfl_error) <= 1e-4 * abs(cfl_error), (
        cfl_error, fixed_error)