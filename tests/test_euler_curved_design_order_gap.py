"""Honest-negative pins for ``tensordg.euler_curved`` (curved-mesh Euler DG).

Both tests document defects that exist at HEAD; they are *pins*, not
acceptances.  When either defect is fixed the corresponding test fails, which is
the intended trigger to delete that test together with its entry in
``docs/CASE-DG-CURVED-DESIGN-ORDER-RED.md``.

P1  the warped-mesh P1 accuracy gate (``order >= p + 0.5 = 1.5``) is **not** met:
    the observed order is ~1.15 with 5.4% worst relative L2 error at 8x8.  The
    acceptance test that owns this gate lives in ``tests/test_euler_curved.py``
    and is red; this test only pins the *number* for the report.
P2  the blow-up of an SSP-RK3 march is **not** a boundary effect: it reproduces
    on a fully periodic mesh with no wall and no outflow edge, where a 1e-6
    stationary entropy wave (an exact steady state) eventually explodes.
    ``tensordg/euler_curved.py`` currently attributes this to the reflective
    wall; the attribution is falsified here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import tensordg.euler_curved as ec
from tensordg.curved_geometry import CurvedMesh, IsoparametricMap

from tensordg.euler_2d import vortex_primitive
import torch

BOX = 10.0
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
AMPLITUDE = 0.06
GEOMETRY_DEGREE = 3


def _exact_primitive(time):
    def primitive(x, y):
        rho, velocity_x, velocity_y, pressure = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=STRENGTH, background_velocity=BACKGROUND, period=BOX,
        )
        return (rho.numpy(), velocity_x.numpy(), velocity_y.numpy(), pressure.numpy())

    return primitive


def _periodic_mesh(nx, ny, lx=4.0, ly=1.0, periodic=(True, True), walls=(False,) * 4):
    maps = [[IsoparametricMap.affine(lx * i / nx, lx * (i + 1) / nx,
                                     ly * j / ny, ly * (j + 1) / ny, 1)
             for j in range(ny)] for i in range(nx)]
    return CurvedMesh(maps, periodic=periodic, walls=walls)


def _rest(x, y):
    return (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def _entropy_wave(x, y):
    """rho perturbed, p and velocity untouched -- an exact *steady* solution."""
    return (1.0 + 1e-6 * np.sin(np.pi * x) * np.sin(2 * np.pi * y),
            np.zeros_like(x), np.zeros_like(x), np.ones_like(x))


def test_warped_p1_design_order_is_currently_missed():
    """P1: warped-mesh P1 order ~1.15 instead of >= 1.5 (defect pin)."""
    errors = []
    for cells in (8, 16):
        solver = ec.CurvilinearEuler(
            ec.warped_box_mesh(cells, cells, GEOMETRY_DEGREE, length=BOX,
                               amplitude=AMPLITUDE),
            degree=1, cfl=0.2)
        state = solver.project(_exact_primitive(0.0))
        state, _, _ = solver.evolve(state, 0.1)
        errors.append(solver.relative_errors(state, _exact_primitive(0.1))["worst"])
    order = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order < 1.5, (
        "the warped-mesh P1 design-order gate now passes -- delete this pin and "
        f"the corresponding section of docs/CASE-DG-CURVED-DESIGN-ORDER-RED.md (order={order:.3f})")
    assert errors[0] > 0.05, errors  # 5.4% at 8x8, i.e. far from asymptotic


def test_blow_up_reproduces_without_any_boundary():
    """P2: fully periodic mesh, no wall -- the march still explodes (defect pin)."""
    solver = ec.CurvilinearEuler(_periodic_mesh(8, 6), 1)
    s0 = solver.project(_rest)
    state = solver.project(_entropy_wave)
    dt = solver.stable_dt(state)
    n0 = float(np.abs(state - s0).max())
    for _ in range(260):
        state = solver.ssp_rk3_step(state, dt)
        if not np.all(np.isfinite(state)):
            break
        if float(np.abs(state - s0).max()) > 1e3 * n0:
            return  # reproduced
    pytest.skip(
        "the fully periodic march stayed bounded -- the interior blow-up is fixed: "
        "delete this pin, the 'unstable' paragraphs of tensordg/euler_curved.py and "
        "re-run the CKS1998 curved-wall leg")