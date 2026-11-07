"""Ice-edge transport high-order reference benchmarks (campaign IN12-P5).

This module is the single source of truth for the *DG advection benchmark suite*
that supports the transport step of an ice-condition assimilation scheme
(``A2``) and passive frazil transport: it defines the benchmark cases, the
observed-order estimator, the physical metrics (peak retention, mass balance)
and the machine-checkable acceptance criteria.  The CLI wrapper
``tools/run_dg_advection_bench.py`` serialises the results to
``artifacts/dg_advection_bench.json``; ``tests/test_dg_advection_bench.py``
asserts the same criteria on a reduced (fast) configuration.

Benchmark cases
---------------
``shear_transport``
    Periodic transversal shear ``a = (a_x(y), 0)`` with ``a_x(y) = -omega (y-1/2)``
    (sign-changing, so upwinding flips inside the domain).  The exact solution
    is the rigid horizontal translation ``u(x, y, t) = u0(x - a_x(y) t, y)``,
    hence an analytic L2 norm on the *whole* periodic domain.  This is the
    analytic anchor for the variable-coefficient DG operator and the 1D element
    of a rotating front.
``solid_body_rotation_front``
    Rigid rotation ``a = omega (-(y-y0), x-x0)`` (``omega = 2 pi``, one
    revolution per unit time) transporting a Gaussian ice-edge front on a
    circular trajectory of radius 0.15 about the domain centre.  Peak retention
    after one exact revolution (a reentrant check, since the exact solution is
    the unrotated initial field) is measured, and the convergence order is
    measured in a disk of radius 0.28 about the centre.
``reentrant_deformational_vortex``
    LeVeque/Nair deformational flow, ``psi = A sin^2(pi x) sin^2(pi y)
    cos(pi t / T)`` with ``A = 1/pi``: divergence-free, periodic, strongly
    deforming, and the exact flow map over ``[0, T]`` is the identity, so the
    L2 error at ``t = T`` measured against the initial condition is a genuine
    error norm on the full domain.

Why a restricted norm for the rotation case
-------------------------------------------
The rigid-rotation velocity of a *periodic* square is discontinuous across the
seam (``a_y = omega(x-1/2)`` jumps from ``-pi`` to ``+pi`` at ``x = 1``), which
generates a thin O(1) layer of error at the seam that decays only as the seam
layer is resolved.  Because the characteristics of a rigid rotation are circles
about the centre, that seam noise stays at its own radius, so the disk norm
``r <= 0.28`` (bump trajectory at ``r = 0.15``, seam at ``r >= 0.5``) is a
well-posed error norm.  The inconsistency is *documented as a limitation*, not
hidden: the full-domain norm is also reported for the same run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from tensordg.advection_2d_var import (
    DGAdvection2D,
    gaussian_bump,
    reentrant_vortex,
    rotated_profile,
    solid_body_rotation,
    trig_profile,
)

OMEGA = 2.0 * math.pi

#: Acceptance criteria (machine checkable, mirrored into the artifact JSON).
CRITERIA = {
    "order_rule": "observed_order >= degree + 0.9 for degree = 1, 2 (degree = 3 reported)",
    "order_degrees_required": [1, 2],
    "order_margin": 0.9,
    "mass_rule": "relative global mass drift <= 1e-12 over the whole run",
    "mass_tolerance": 1e-12,
    "peak_rule": "Gaussian ice-edge front peak retention >= 0.99 after one revolution (p = 3)",
    "peak_tolerance": 0.99,
    "pytest_min_tests": 5,
}


def observed_order(errors: Sequence[float], cells: Sequence[int]) -> dict:
    """Least-squares slope of ``log(err)`` vs ``log(n_cells)`` plus pair rates.

    The least-squares slope over the full resolution set is the reported
    observed order of accuracy; the pairwise rates are kept as a diagnostic
    (the finest pair is the most asymptotic one).
    """
    if len(errors) != len(cells) or len(errors) < 2:
        raise ValueError("need matching sequences of length >= 2")
    xs = [math.log(float(n)) for n in cells]
    ys = [math.log(float(e)) for e in errors]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = -num / den   # errors decay with resolution -> positive order
    pairs = [
        -math.log(errors[i + 1] / errors[i]) / math.log(cells[i + 1] / cells[i])
        for i in range(len(errors) - 1)
    ]
    return {
        "observed_order": slope,
        "pairwise_orders": pairs,
        "finest_pair_order": pairs[-1],
    }


@dataclass
class OrderStudy:
    """Resolution study for one (case, degree) pair."""

    case: str
    degree: int
    cells: list[int]
    final_time: float
    errors: list[float] = field(default_factory=list)
    relative_errors: list[float] = field(default_factory=list)
    peaks: list[float] = field(default_factory=list)
    mass_drift: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    cfl: float = 0.3
    norm: str = "global"

    def as_dict(self) -> dict:
        out = {
            "case": self.case, "degree": self.degree, "cells": self.cells,
            "final_time": self.final_time, "cfl": self.cfl, "norm": self.norm,
            "errors_l2": self.errors,
            "relative_errors_l2": self.relative_errors,
            "peaks": self.peaks,
            "mass_drift_rel": self.mass_drift,
            "steps": self.steps,
        }
        out.update(observed_order(self.errors, self.cells))
        required = self.degree in CRITERIA["order_degrees_required"]
        out["required_order"] = self.degree + CRITERIA["order_margin"]
        out["order_pass"] = bool(
            out["observed_order"] >= out["required_order"]) if required else None
        return out


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    velocity: Callable
    profile: Callable
    exact: Callable | None      # None -> reentrant (exact = initial condition)
    final_time: float
    disk_radius: float | None   # restricted norm (rotation only)
    note: str


def _shear_velocity(x, y, t=0.0):
    """Sign-changing transversal shear ``a = (-omega (y-1/2), 0)`` (periodic)."""
    return -OMEGA * (y - 0.5), torch.zeros_like(y)


def _shear_exact(x, y, t):
    ax = -OMEGA * (y - 0.5)
    return trig_profile(x - ax * t, y)


def _rotation_bump(x, y, t=0.0):
    return gaussian_bump(x, y, t, center=(0.5, 0.35), sigma=0.075)


CASES: dict[str, CaseSpec] = {
    "shear_transport": CaseSpec(
        key="shear_transport",
        title="Periodic transversal shear transport of a smooth periodic field",
        velocity=_shear_velocity,
        profile=trig_profile,
        exact=_shear_exact,
        final_time=0.05,
        disk_radius=None,
        note="analytic solution on the whole periodic torus (sign-changing upwind)",
    ),
    "solid_body_rotation_front": CaseSpec(
        key="solid_body_rotation_front",
        title="Solid-body rotation of a Gaussian ice-edge front (one revolution)",
        velocity=solid_body_rotation(OMEGA),
        profile=_rotation_bump,
        exact=rotated_profile(_rotation_bump, OMEGA),
        final_time=0.5,
        disk_radius=0.28,
        note="rigid rotation; seam is non-periodic -> restricted disk norm",
    ),
    "reentrant_deformational_vortex": CaseSpec(
        key="reentrant_deformational_vortex",
        title="Reentrant deformational vortex (LeVeque/Nair), full period",
        velocity=reentrant_vortex(period=1.0),
        profile=trig_profile,
        exact=None,
        final_time=1.0,
        disk_radius=None,
        note="flow map over [0,T] is the identity -> exact = initial condition",
    ),
}


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def run_order_study(
    case_key: str,
    degree: int,
    cells: Sequence[int],
    cfl: float = 0.3,
    final_time: float | None = None,
) -> OrderStudy:
    """Run the resolution study for one case and polynomial degree."""
    spec = CASES[case_key]
    t_end = spec.final_time if final_time is None else float(final_time)
    study = OrderStudy(
        case=case_key, degree=degree, cells=list(cells), final_time=t_end,
        cfl=cfl, norm=(f"disk r<={spec.disk_radius}" if spec.disk_radius else "global"),
    )
    for n in cells:
        solver = DGAdvection2D(degree, n)
        coeff0 = solver.project(spec.profile)
        result = solver.solve(coeff0, spec.velocity, t_end, cfl=cfl)
        coeff = result["coeff"]
        if spec.exact is not None:
            err = solver.l2_error(
                coeff, spec.exact, t_end, radius=spec.disk_radius,
            )
        else:
            err = solver.l2_difference(coeff, coeff0)
            ref = solver.l2_error(coeff0, spec.profile, 0.0)
            err = dict(err, reference_l2=ref["reference_l2"])
        study.errors.append(err["absolute_l2"])
        study.relative_errors.append(
            err["absolute_l2"] / err["reference_l2"] if err["reference_l2"] else float("nan"))
        study.peaks.append(solver.peak(coeff)[1])
        study.mass_drift.append(result["meta"]["mass_drift_rel"])
        study.steps.append(result["meta"]["steps"])
    return study


def run_revolution_check(
    degree: int = 3,
    cells: int = 16,
    cfl: float = 0.3,
    revolutions: float = 1.0,
) -> dict:
    """Peak-retention check of the rotating ice-edge front after ``revolutions``.

    The exact solution after a whole number of revolutions is the initial field,
    so ``peak_final / peak_initial`` is a pure measure of numerical damping
    (``<= 1``) or overshoot (``> 1``).
    """
    spec = CASES["solid_body_rotation_front"]
    solver = DGAdvection2D(degree, cells)
    coeff0 = solver.project(spec.profile)
    peak0 = solver.peak(coeff0)[1]
    result = solver.solve(coeff0, spec.velocity, revolutions, cfl=cfl)
    peak1 = solver.peak(result["coeff"])[1]
    err0 = solver.l2_error(coeff0, spec.profile, 0.0)
    err = solver.l2_error(result["coeff"], spec.exact, revolutions,
                          radius=spec.disk_radius)
    mass = solver.mass(coeff0)
    return {
        "degree": degree, "cells": [cells, cells], "revolutions": revolutions,
        "cfl": cfl, "steps": result["meta"]["steps"],
        "peak_initial": peak0, "peak_final": peak1,
        "peak_retention": peak1 / peak0,
        "relative_l2_disk": err["relative_l2"],
        "relative_l2_global": solver.l2_error(
            result["coeff"], spec.exact, revolutions)["relative_l2"],
        "mass_drift_rel": result["meta"]["mass_drift_rel"],
        "mass_abs": mass,
        "error_norm_reference": err0["reference_l2"],
    }


def run_vortex_peak_check(degree: int = 3, cells: int = 16,
                          cfl: float = 0.3) -> dict:
    """Peak retention of a Gaussian bump under the reentrant deformational flow."""
    spec = CASES["reentrant_deformational_vortex"]
    profile = lambda x, y, t=0.0: gaussian_bump(  # noqa: E731
        x, y, t, center=(0.45, 0.45), sigma=0.1)
    solver = DGAdvection2D(degree, cells)
    coeff0 = solver.project(profile)
    peak0 = solver.peak(coeff0)[1]
    result = solver.solve(coeff0, spec.velocity, spec.final_time, cfl=cfl)
    peak1 = solver.peak(result["coeff"])[1]
    err = solver.l2_difference(result["coeff"], coeff0)
    ref = solver.l2_error(coeff0, profile, 0.0)
    return {
        "degree": degree, "cells": [cells, cells], "steps": result["meta"]["steps"],
        "peak_initial": peak0, "peak_final": peak1,
        "peak_retention": peak1 / peak0,
        "relative_l2": err["absolute_l2"] / ref["reference_l2"],
        "mass_drift_rel": result["meta"]["mass_drift_rel"],
    }


def mass_balance_trace(degree: int = 2, cells: int = 16, cfl: float = 0.3,
                       samples: int = 40) -> dict:
    """Mass drift history for the vortex case (used for the conservation figure)."""
    spec = CASES["reentrant_deformational_vortex"]
    solver = DGAdvection2D(degree, cells)
    coeff0 = solver.project(spec.profile)
    result = solver.solve(coeff0, spec.velocity, spec.final_time, cfl=cfl,
                          record_mass=True)
    hist = result["mass_history"]
    if len(hist) > samples:
        stride = int(math.ceil(len(hist) / samples))
        hist = hist[::stride] + [hist[-1]]
    m0 = hist[0]
    return {
        "degree": degree, "cells": [cells, cells], "steps": result["meta"]["steps"],
        "mass_initial": m0,
        "relative_drift": [(m - m0) / abs(m0) for m in hist],
        "relative_drift_max": result["meta"]["mass_drift_max_rel_history"],
        "relative_drift_final": result["meta"]["mass_drift_rel"],
    }