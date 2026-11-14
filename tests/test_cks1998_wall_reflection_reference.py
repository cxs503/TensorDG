"""G3 regression gates for the interior-wall shock reflection (REPRO-DG-CKS1998).

These tests are deliberately *analytic and instant*: they pin the closed-form
reference and the plateau-sampling geometry, the two places where the merged
forward-facing-step work was silently wrong.

1. ``benchmarks.forward_facing_step`` failed at **import** time: its gate had a
   default argument ``incident_mach=mach`` that referenced a sibling parameter.
   Python evaluates default arguments in the enclosing scope, so every import of
   the module raised ``NameError`` and the G3 gate plus the Mach-3 step driver
   were dead code.  ``test_forward_step_module_imports`` is that regression.
2. ``WallReflection.state`` returned ``reflected`` for ``x < front`` and ``pre``
   for ``x > front`` *after* the reflection, i.e. it mirrored the plateau and
   dropped the post-shock region.  Every error norm computed against it
   (``wall_reflection()["l2_relative"]``, and the assertion in
   ``tests/test_euler_dg_2d_wall_bc.py``) was measuring a reference error.
"""
import importlib
import math

import numpy as np
import pytest
import torch

from benchmarks.euler_dg_2d_wall_bc import GAMMA, WallReflection


def test_forward_step_module_imports():
    module = importlib.import_module("benchmarks.forward_facing_step")
    assert callable(module.gate_wall_reflection)
    assert module.gate_wall_reflection.__defaults__ is not None
    # incident_mach must resolve to mach, not to a module-level accident
    gate = module.gate_wall_reflection
    assert gate.__defaults__[4] is None


def test_state_is_two_regions_before_wall_time_and_after():
    case = WallReflection(x_shock=0.4, x_wall=1.0, incident_mach=2.0)
    rho2, u2, p2 = case.post
    rho5, u5, p5 = case.reflected

    # (a) before the reflection: post-shock behind the incident front, pre ahead
    t = 0.5 * case.wall_time
    front = case.front(t)
    x = np.array([0.5 * (case.x_zero + front), 0.5 * (front + case.x_wall)])
    rho, u, p = case.state(x, t)
    assert rho[0] == pytest.approx(rho2) and rho[1] == pytest.approx(1.0)
    assert u[0] == pytest.approx(u2) and u[1] == pytest.approx(0.0)
    assert p[0] == pytest.approx(p2) and p[1] == pytest.approx(1.0)

    # (b) after the reflection: post-shock ahead of the *reflected* front, and the
    # reflected plateau between that front and the wall
    t = case.wall_time + 0.08
    front = case.front(t)
    assert front < case.x_wall
    x = np.array([0.5 * (case.x_zero + front), 0.5 * (front + case.x_wall)])
    rho, u, p = case.state(x, t)
    assert rho[0] == pytest.approx(rho2) and rho[1] == pytest.approx(rho5)
    assert u[0] == pytest.approx(u2) and u[1] == pytest.approx(u5)
    assert p[0] == pytest.approx(p2) and p[1] == pytest.approx(p5)


def test_reflected_state_satisfies_rankine_hugoniot_and_rest_condition():
    """Independent check of the closed form the gate compares against."""
    case = WallReflection(x_shock=0.4, x_wall=1.0, incident_mach=2.0)
    rho2, u2, p2 = case.post
    rho5, u5, p5 = case.reflected
    assert u5 == 0.0  # gas at rest against a solid wall
    # mass and momentum jump conditions across the reflected shock
    speed = case.reflected_speed
    assert rho2 * (u2 - speed) == pytest.approx(rho5 * (-speed), rel=1e-12)
    # momentum: the state-5 side carries the same flux, measured in the shock frame
    assert (p2 + rho2 * (u2 - speed) ** 2) == pytest.approx(
        p5 + rho5 * (-speed) ** 2, rel=1e-12)
    # energy: total enthalpy is conserved across the reflected shock
    h2 = GAMMA / (GAMMA - 1.0) * p2 / rho2 + 0.5 * (u2 - speed) ** 2
    h5 = GAMMA / (GAMMA - 1.0) * p5 / rho5 + 0.5 * (-speed) ** 2
    assert h2 == pytest.approx(h5, rel=1e-12)


def test_plateau_mask_never_samples_the_smeared_front_or_the_wall():
    from benchmarks.forward_facing_step import _plateau_mask, _snap_to_face

    # the wall must be a cell face: nx=480 is aligned, nx=64 leaves the wall a
    # third of a cell inside the last fluid cell (the first gate's geometry bug)
    face, offset = _snap_to_face(1.0, 1.2 / 480)
    assert offset == pytest.approx(0.0)
    _, offset64 = _snap_to_face(1.0, 1.2 / 64)
    assert abs(offset64) == pytest.approx(1.0 / 3.0)

    dx = 1.2 / 240
    centres = (torch.arange(240, dtype=torch.float64) + 0.5) * dx
    fluid = centres < face
    front = face - 20 * dx
    mask = _plateau_mask(centres, fluid, front, face, dx)
    kept = centres[mask]
    assert kept.numel() == pytest.approx(20 - 3 - 1)
    assert kept.min() > front + 3 * dx
    assert kept.max() < face - 1 * dx

    # a coarse mesh whose plateau is only ~5 cells wide keeps a single cell,
    # i.e. fewer than the gate's min_plateau_cells=3, so the gate reports
    # not_run instead of a 3 % FAIL that no scheme could meet
    dx64 = 1.2 / 64
    centres64 = (torch.arange(64, dtype=torch.float64) + 0.5) * dx64
    face64, _ = _snap_to_face(1.0, dx64)
    thin = _plateau_mask(centres64, centres64 < face64, face64 - 5 * dx64, face64, dx64)
    assert int(thin.sum()) == 1
