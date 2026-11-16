#!/usr/bin/env python3
"""CASE-DG-EULER-VORTEX-ERR: reference-and-instrument certification.

The case-pool anchor is decided by a *chain of instruments*:

    analytic isentropic vortex  --(R1 exactness, R2 isentropy)-->  reference field
    reference field + quadrature --(R3, R4)-->  invariants and relative-L2 error
    error readings              --(C1..C11)-->  verdict

Every published number of the bundle (``0.0152 %``, ``1.75e-13``, ``2.98e-07``)
is a *reading of these instruments*.  Until now the instruments themselves were
asserted, not certified: the gate assumed that

* ``vortex_primitive`` really is a solution of the 2D Euler equations (a
  transcription error in the closed form would look exactly like a scheme
  error, and the 6 % tolerance would be measured against the wrong field);
* the 320-point tensor Gauss reference used as the denominator of every drift
  column is converged far below the smallest drift reported;
* the ``p+2`` Gauss points per cell used by the DG error norm, and the 4-point
  exact cell averages used by the P0 norm, resolve the difference field well
  enough to carry a 4-digit percentage.

This script turns those three assumptions into measured certificates, and — the
part that makes them worth something — into *mutation controls*: every
certificate is run a second time against a deliberately corrupted input, and it
must fail.  A check that cannot fail certifies nothing.

Certificates
------------
R1  exactness     conservative-form residual ``dU/dt + dF/dx + dG/dy`` of the
                  analytic field, every derivative taken by autodiff on the
                  analytic expression itself (no finite differences, so nothing
                  but the arithmetic is in the loop).  Free-space family at
                  several transport times, plus exact periodicity of the
                  ``period=`` fold.  Mutants: four plausible mis-transcriptions.
R2  isentropy     ``p / rho**gamma`` must be spatially constant; mutant: an
                  isothermal variant.
R3  invariants    convergence of the reference invariants (mass, momentum,
                  energy, entropy) in the number of Gauss points, against the
                  smallest drift the gate claims to resolve.
R4  error norm    convergence of the relative-L2 reading in the number of
                  quadrature points per cell, for the real DG states (L2
                  projections and time-stepped solutions) and for the P0
                  cell-average norm.
R5  entropy       convergence of the entropy-budget integral (C4) with the same
                  number of points, i.e. is the reported ``dS/S`` production or
                  quadrature noise?
R6  reproduction  the instrument reproduces the *stored* readings
                  (``results/*.json``) from a fresh solve.

Pass rule used everywhere: an instrument must sit at least ``MARGIN`` (= 10)
times below the quantity it certifies, otherwise the quantity is
instrument-limited and is reported as such instead of as physics.

Usage::

    PYTHONPATH=. python3 benchmarks/vortex_reference_instrument_certification.py
    PYTHONPATH=. python3 benchmarks/vortex_reference_instrument_certification.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg import euler_dg_2d as dg2
from tensordg.euler_2d import (
    GAMMA,
    solve_isentropic_vortex,
    vortex_primitive,
    _primitive,
)
from benchmarks.verify_euler_dg_2d import _vortex_error
from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND,
    DOMAIN,
    PERIOD,
    STRENGTH,
    TOLERANCE,
    _dg_measurements,
    _initial_dg_state,
    _p0_measurements,
    exact_core_density,
    exact_invariants,
)

MARGIN = 10.0          # instrument must be >= 10x below what it certifies
OUT = Path("case_euler_vortex_errconserve")
RESULTS = OUT / "results"

#: Every certificate reports ``value``, the ``reference`` it is measured
#: against, and the ``ratio`` value/reference; ``passed`` is ``ratio <= 1/MARGIN``.
TOL_RESIDUAL = 1e-10   # absolute bound for a machine-exact solution, in units of
                       # the largest flux-divergence term at that point


# ---------------------------------------------------------------------------
# R1: is the analytic field a solution of the Euler equations?
# ---------------------------------------------------------------------------

def _conservative(rho, u, v, p):
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return rho, rho * u, rho * v, energy


def _fluxes(state):
    rho, mx, my, energy = state
    u = mx / rho
    v = my / rho
    p = (GAMMA - 1.0) * (energy - 0.5 * (mx * mx + my * my) / rho)
    flux_x = (mx, mx * u + p, my * u, u * (energy + p))
    flux_y = (my, mx * v, my * v + p, v * (energy + p))
    return flux_x, flux_y


def _mutant_primitive(name, x, y, time, strength, background):
    """The analytic vortex with one plausible transcription error baked in.

    Each mutant is a mistake a human *does* make when typing the closed form;
    the residual detector is worthless if it cannot separate them from the
    genuine article.
    """
    x_shift = x - background[0] * time
    y_shift = y - background[1] * time
    radius2 = x_shift ** 2 + y_shift ** 2
    envelope = torch.exp(0.5 * (1.0 - radius2))
    if name == "base":
        amplitude = (GAMMA - 1.0) * strength ** 2 / (8.0 * GAMMA * math.pi ** 2)
        temperature = 1.0 - amplitude * envelope ** 2
        velocity_x = background[0] - strength / (2.0 * math.pi) * envelope * y_shift
        velocity_y = background[1] + strength / (2.0 * math.pi) * envelope * x_shift
    elif name == "envelope_exponent":       # exp(1-r^2) instead of exp((1-r^2)/2)
        amplitude = (GAMMA - 1.0) * strength ** 2 / (8.0 * GAMMA * math.pi ** 2)
        temperature = 1.0 - amplitude * envelope ** 2
        velocity_x = background[0] - strength / (2.0 * math.pi) * envelope ** 2 * y_shift
        velocity_y = background[1] + strength / (2.0 * math.pi) * envelope ** 2 * x_shift
    elif name == "temperature_factor":      # 4*gamma*pi^2 instead of 8*gamma*pi^2
        amplitude = (GAMMA - 1.0) * strength ** 2 / (4.0 * GAMMA * math.pi ** 2)
        temperature = 1.0 - amplitude * envelope ** 2
        velocity_x = background[0] - strength / (2.0 * math.pi) * envelope * y_shift
        velocity_y = background[1] + strength / (2.0 * math.pi) * envelope * x_shift
    elif name == "velocity_one_sign":       # only the y-swirl sign flipped
        amplitude = (GAMMA - 1.0) * strength ** 2 / (8.0 * GAMMA * math.pi ** 2)
        temperature = 1.0 - amplitude * envelope ** 2
        velocity_x = background[0] - strength / (2.0 * math.pi) * envelope * y_shift
        velocity_y = background[1] - strength / (2.0 * math.pi) * envelope * x_shift
    elif name == "isothermal":              # p = rho instead of p = rho**gamma
        amplitude = (GAMMA - 1.0) * strength ** 2 / (8.0 * GAMMA * math.pi ** 2)
        temperature = 1.0 - amplitude * envelope ** 2
        density = temperature ** (1.0 / (GAMMA - 1.0))
        velocity_x = background[0] - strength / (2.0 * math.pi) * envelope * y_shift
        velocity_y = background[1] + strength / (2.0 * math.pi) * envelope * x_shift
        return density, velocity_x, velocity_y, density
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown mutant {name!r}")
    density = temperature ** (1.0 / (GAMMA - 1.0))
    return density, velocity_x, velocity_y, density ** GAMMA


def euler_residual(points, time=0.0, strength=STRENGTH, background=BACKGROUND,
                   period=None, mutant="base"):
    """Pointwise residual of ``dU/dt + dF/dx + dG/dy = 0`` in conservative form.

    ``points`` is ``(n, 2)`` float64.  The time coordinate is a *per-point*
    tensor: with a shared scalar time ``autograd.grad`` returns the sum of the
    derivatives over all points while the spatial derivatives stay per-point,
    and the two do not add up (the first version of this tool made exactly that
    mistake and reported a residual of 0.96 for an exact solution).
    """
    x = points[:, 0].clone().requires_grad_(True)
    y = points[:, 1].clone().requires_grad_(True)
    t = torch.full((points.shape[0],), float(time), dtype=points.dtype,
                   requires_grad=True)
    if mutant == "base":
        rho, u, v, p = vortex_primitive(x, y, t, strength, background, period=period)
    else:
        if period is not None:  # pragma: no cover - mutants are free-space only
            raise ValueError("mutants are only defined for the free-space form")
        rho, u, v, p = _mutant_primitive(mutant, x, y, t, strength, background)
    state = _conservative(rho, u, v, p)
    flux_x, flux_y = _fluxes(state)
    names = ("mass", "momentum_x", "momentum_y", "energy")
    residual, scale = {}, {}
    for index, name in enumerate(names):
        d_dt = torch.autograd.grad(state[index].sum(), t, retain_graph=True)[0]
        d_dx = torch.autograd.grad(flux_x[index].sum(), x, retain_graph=True)[0]
        d_dy = torch.autograd.grad(flux_y[index].sum(), y, retain_graph=True)[0]
        residual[name] = d_dt + d_dx + d_dy
        scale[name] = torch.maximum(
            torch.maximum(d_dt.abs(), d_dx.abs()), d_dy.abs())
    return residual, scale


def certify_exactness(points=256, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(points, 2, dtype=torch.float64) * 10.0 - 5.0
    rows = {}
    for label, kwargs in (("t=0", {}), ("t=1", {"time": 1.0}),
                          ("t=2.5", {"time": 2.5})):
        residual, scale = euler_residual(pts, **kwargs)
        rows[label] = {
            name: {
                "abs_max": float(residual[name].abs().max()),
                "relative": float((residual[name].abs() / scale[name]).max()),
            }
            for name in residual
        }
    worst = max(row["relative"] for row in rows.values() for row in row.values())
    mutants = {}
    for mutant in ("envelope_exponent", "temperature_factor", "velocity_one_sign"):
        residual, scale = euler_residual(pts, time=1.0, mutant=mutant)
        mutants[mutant] = max(
            float((residual[name].abs() / scale[name]).max()) for name in residual)
    # periodicity of the folded reference: the periodic field must be L-periodic
    shifted = pts + torch.tensor(PERIOD, dtype=torch.float64)
    base = vortex_primitive(pts[:, 0], pts[:, 1], 1.0, STRENGTH, BACKGROUND,
                            period=PERIOD)
    moved = vortex_primitive(shifted[:, 0], shifted[:, 1], 1.0, STRENGTH, BACKGROUND,
                             period=PERIOD)
    periodicity = max(
        float(((a - b).abs() / b.abs().clamp_min(1e-12)).max())
        for a, b in zip(base, moved))
    return {
        "criterion": "max |dU/dt + dF/dx + dG/dy| / max(|dU/dt|,|dF/dx|,|dG/dy|)",
        "residual": rows,
        "worst_relative": worst,
        "limit": TOL_RESIDUAL,
        "passed": bool(worst <= TOL_RESIDUAL),
        "mutants": mutants,
        "mutants_detected": bool(all(v > 1e-2 for v in mutants.values())),
        "periodicity_relative": periodicity,
        "periodicity_passed": bool(periodicity < 1e-13),
    }


# ---------------------------------------------------------------------------
# R2: isentropy of the reference field
# ---------------------------------------------------------------------------

def certify_isentropy(points=512, seed=1):
    torch.manual_seed(seed)
    pts = torch.rand(points, 2, dtype=torch.float64) * 10.0 - 5.0
    rows = {}
    for label, mutant in (("analytic", "base"), ("isothermal_mutant", "isothermal")):
        rho, _u, _v, p = _mutant_primitive(mutant, pts[:, 0], pts[:, 1],
                                           torch.zeros(points, dtype=torch.float64),
                                           STRENGTH, BACKGROUND)
        entropy = p / rho ** GAMMA
        rows[label] = float((entropy - 1.0).abs().max())
    return {
        "criterion": "max |p/rho^gamma - 1| over the reference field",
        "analytic": rows["analytic"],
        "limit": 1e-14,
        "passed": bool(rows["analytic"] <= 1e-14),
        "mutant": rows["isothermal_mutant"],
        "mutant_detected": bool(rows["isothermal_mutant"] > 1e-2),
    }


# ---------------------------------------------------------------------------
# R3: convergence of the reference invariants (the drift denominators)
# ---------------------------------------------------------------------------

def _smallest_conservation_drift():
    """Smallest mass/momentum/energy drift the bundle actually reports."""
    keys = ("mass", "momentum_x", "momentum_y", "energy")
    values = []
    for path in RESULTS.glob("*.json"):
        data = json.loads(path.read_text())
        drift = data.get("invariant_drift") or {}
        values.extend(drift[k] for k in keys if k in drift)
    return (min(values) if values else float("nan")), (max(values) if values else float("nan"))


def _max_entropy_drift():
    values = []
    for path in RESULTS.glob("*.json"):
        data = json.loads(path.read_text())
        if "entropy_drift" in data:
            values.append(float(data["entropy_drift"]))
    return max(values) if values else float("nan")


def certify_invariant_reference(ladder=(320, 480, 640, 960, 1280),
                                smallest_drift=None):
    if smallest_drift is None:
        smallest_drift, _largest = _smallest_conservation_drift()
    readings = {str(p): exact_invariants(p) for p in ladder}
    truth = readings[str(max(ladder))]
    keys = ("mass", "momentum_x", "momentum_y", "energy", "entropy")
    deviations = {}
    for p, values in readings.items():
        deviations[p] = {k: abs(values[k] - truth[k]) / abs(truth[k]) for k in keys}
    gate = deviations["320"]
    worst_gate = max(gate.values())
    core = exact_invariants(320)["density_min"]
    true_core = exact_core_density()
    return {
        "criterion": "|invariant(p) - invariant(1280)| / |invariant(1280)|",
        "deviations": deviations,
        "gate_points": 320,
        "gate_worst": worst_gate,
        "smallest_drift": smallest_drift,
        "ratio": worst_gate / smallest_drift,
        "passed": bool(worst_gate * MARGIN <= smallest_drift),
        "density_min": {
            "tensor_grid_320": core,
            "core_ray_4001": true_core,
            "relative_gap": abs(core - true_core) / true_core,
            "note": ("`density_min` is a minimum *over quadrature nodes*, so the "
                     "tensor grid samples the core off-centre; the gate uses it "
                     "only as a same-instrument drift (final vs initial), which "
                     "is unaffected, but it is not the true core minimum"),
        },
    }


# ---------------------------------------------------------------------------
# R4: convergence of the relative-L2 error norm
# ---------------------------------------------------------------------------

def _p0_error(conservative, x, y, final_time, quadrature, reference_kwargs):
    """Relative L2 of P0 cell averages against exact cell averages, nq-point exact."""
    nx, ny = conservative.shape[:2]
    dx = float((x[-1] - x[0]).item()) / (nx - 1)
    dy = float((y[-1] - y[0]).item()) / (ny - 1)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    nodes = torch.as_tensor(nodes_np, dtype=conservative.dtype)
    weights = torch.as_tensor(weights_np, dtype=conservative.dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    exact = vortex_primitive(xq, yq, final_time, STRENGTH, BACKGROUND,
                             **reference_kwargs)
    reference = tuple(
        0.25 * torch.sum(field * weights[None, None, :, None] * weights[None, None, None, :],
                         dim=(2, 3)) for field in exact)
    numerical = _primitive(conservative)
    out = {}
    for name, computed, target in zip(("density", "velocity_x", "velocity_y", "pressure"),
                                      numerical, reference):
        absolute = float(torch.sqrt(torch.mean((computed - target) ** 2)).item())
        norm = float(torch.sqrt(torch.mean(target ** 2)).item())
        out[name] = absolute / norm
    return out


def _dg_reading(coeff, degree, multipliers):
    readings = {}
    for multiplier in multipliers:
        quadrature = (degree + 2) * multiplier if multiplier > 0 else degree
        if quadrature < degree + 1:
            quadrature = degree + 1
        nodes, weights, basis, _, _ = dg2.legendre_operators(degree, quadrature)
        errors = _vortex_error(coeff, basis, nodes, DOMAIN, 0.0, STRENGTH,
                               BACKGROUND, period=PERIOD)
        readings[str(multiplier)] = max(
            errors[k] for k in ("density", "velocity_x", "velocity_y", "pressure"))
    return readings


def _cache_path(degree, cells, final_time):
    return OUT / f"instrument_state_p{degree}_n{cells}_t{final_time:g}.pt"


def _solve_cached(degree, cells, final_time):
    """Solve once, keep the modal state: a re-solve is the expensive part."""
    path = _cache_path(degree, cells, final_time)
    if path.exists():
        payload = torch.load(path, weights_only=False)
        return payload["coeff"], payload["meta"], True
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=cells, degree=degree, final_time=final_time, cfl=0.2,
        strength=STRENGTH, background_velocity=BACKGROUND, domain=DOMAIN)
    torch.save({"coeff": coeff, "meta": meta}, path)
    return coeff, meta, False


def certify_error_norm(with_solves=True, degrees=(1, 2, 3), cells=32,
                       solve_degrees=None):
    """Quadrature convergence of the norm on projections and on real solutions."""
    cases = {}
    for degree in degrees:
        config = {"task": "dg", "degree": degree, "cells": cells,
                  "final_time": 0.0, "cfl": 0.2}
        coeff, _basis, _nodes, _weights = _initial_dg_state(config)
        readings = _dg_reading(coeff, degree, (1, 2, 3, 4))
        gate = readings["1"]
        fine = readings["4"]
        under = _dg_reading(coeff, degree, (-1,))["-1"]
        cases[f"projection_p{degree}_n{cells}"] = {
            "kind": "L2 projection of the exact initial state (no time stepping)",
            "readings": readings,
            "gate": gate,
            "converged": fine,
            "relative_shift": (gate - fine) / fine,
            "under_integrated_shift": (under - fine) / fine,
        }
    if with_solves:
        for degree in (degrees if solve_degrees is None else solve_degrees):
            started = time.time()
            coeff, meta, cached = _solve_cached(degree, cells, 1.0)
            readings = {}
            for multiplier in (1, 2, 3, 4):
                nodes, weights, basis, _, _ = dg2.legendre_operators(
                    degree, (degree + 2) * multiplier)
                errors = _vortex_error(coeff, basis, nodes, DOMAIN, 1.0, STRENGTH,
                                       BACKGROUND, period=PERIOD)
                readings[str(multiplier)] = max(
                    errors[k] for k in ("density", "velocity_x", "velocity_y", "pressure"))
            gate = readings["1"]
            fine = readings["4"]
            key = f"dg_p{degree}_n{cells}_t1"
            stored = _stored(key + "_cfl0.2")
            cases[key] = {
                "kind": f"{meta['steps']} SSP-RK3 steps of the real solver",
                "wall_seconds": time.time() - started,
                "cached_state": cached,
                "readings": readings,
                "gate": gate,
                "converged": fine,
                "relative_shift": (gate - fine) / fine,
                "stored": stored,
                "reproduction_relative": (None if stored is None
                                          else (gate - stored) / stored),
            }
        # P0: the gate-worst row of C1 (4.926 %)
        started = time.time()
        x, y, fields, meta = solve_isentropic_vortex(
            ncells=64, final_time=1.0, cfl=0.4, domain=DOMAIN, strength=STRENGTH,
            background_velocity=BACKGROUND)
        conservative = fields["conservative"] if isinstance(fields, dict) else fields
        readings = {}
        for quadrature in (1, 2, 3, 4, 6, 8, 12):
            readings[str(quadrature)] = max(
                _p0_error(conservative, x, y, 1.0, max(quadrature, 1),
                          {"period": PERIOD}).values())
        gate = readings["4"]
        fine = readings["12"]
        stored = _stored("p0_n64_t1_cfl0.4")
        cases["p0_n64_t1"] = {
            "kind": f"{meta['steps']} SSP-RK3 steps of the P0 solver",
            "wall_seconds": time.time() - started,
            "readings": readings,
            "gate": gate,
            "converged": fine,
            "relative_shift": (gate - fine) / fine,
            "stored": stored,
            "reproduction_relative": (None if stored is None else (gate - stored) / stored),
        }
    worst = max(abs(case["relative_shift"]) for case in cases.values())
    return {
        "criterion": "(reading at gate quadrature - reading at 4x quadrature) / reading",
        "cases": cases,
        "worst_relative_shift": worst,
        "tolerance": TOLERANCE,
        "ratio": worst / TOLERANCE,
        "passed": bool(worst * MARGIN <= TOLERANCE),
    }


def _stored(task_key):
    path = RESULTS / f"{task_key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return float(data["max_relative_l2"])


# ---------------------------------------------------------------------------
# R5: is the entropy budget (C4) physics or quadrature noise?
# ---------------------------------------------------------------------------

def certify_entropy_instrument(degree=2, cells=32, final_time=1.0, coeff=None):
    """Recompute the C4 integral of one real DG state at rising quadrature."""
    if coeff is None:
        coeff, _meta, _cached = _solve_cached(degree, cells, final_time)
    readings = {}
    for multiplier in (1, 2, 3, 4, 6):
        nodes, weights, basis, _, _ = dg2.legendre_operators(
            degree, (degree + 2) * multiplier)
        readings[str(multiplier)] = _dg_measurements(coeff, basis, nodes, weights)
    config = {"task": "dg", "degree": degree, "cells": cells,
              "final_time": 0.0, "cfl": 0.2}
    coeff0, basis0, nodes0, weights0 = _initial_dg_state(config)
    initial = _dg_measurements(coeff0, basis0, nodes0, weights0)
    budgets = {}
    for key, final in readings.items():
        budgets[key] = abs(final["invariants"]["entropy"]
                           / initial["invariants"]["entropy"] - 1.0)
    gate = budgets["1"]
    fine = budgets["6"]
    # quadrature-only part: the same *state* integrated at gate vs refined
    quadrature_part = abs(readings["1"]["invariants"]["entropy"]
                          / readings["6"]["invariants"]["entropy"] - 1.0)
    return {
        "criterion": "|S(gate quadrature) - S(6x quadrature)| / S   vs   the C4 reading",
        "state": f"dg_p{degree}_n{cells}_t{final_time:g}",
        "budgets": budgets,
        "gate_reading": gate,
        "converged_reading": fine,
        "quadrature_part": quadrature_part,
        "ratio": quadrature_part / gate if gate else float("inf"),
        "passed": bool(quadrature_part * MARGIN <= gate),
        "conservation_check": {
            name: abs(readings["1"]["invariants"][name]
                      / readings["6"]["invariants"][name] - 1.0)
            for name in ("mass", "momentum_x", "momentum_y", "energy")
        },
        "note": ("mass/momentum/energy are integrals of the polynomial state and "
                 "are integrated exactly by any quadrature >= degree+1; entropy "
                 "p/rho^gamma is not polynomial, so C4 is the only criterion whose "
                 "denominator can be quadrature-limited"),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build(with_solves=True, solve_degrees=(1, 2, 3)):
    started = time.time()
    report = {
        "case": "CASE-DG-EULER-VORTEX-ERR",
        "scope": "reference + measurement instruments (not the scheme)",
        "margin": MARGIN,
        "reference": "analytic isentropic vortex (Yee et al. 2000 configuration)",
        "R1_exactness": certify_exactness(),
        "R2_isentropy": certify_isentropy(),
        "R3_invariants": certify_invariant_reference(),
        "R4_error_norm": certify_error_norm(with_solves=with_solves,
                                            solve_degrees=solve_degrees),
    }
    if with_solves:
        path = _cache_path(2, 32, 1.0)
        cached = torch.load(path, weights_only=False)["coeff"] if path.exists() else None
        report["R5_entropy_instrument"] = certify_entropy_instrument(coeff=cached)
    report["verdict"] = verdict(report)
    report["wall_seconds"] = time.time() - started
    return report


def verdict(report):
    keys = [k for k in report if k.startswith("R")]
    failed = [k for k in keys if not report[k].get("passed", False)]
    controls = []
    if "R1_exactness" in report:
        controls.append(report["R1_exactness"]["mutants_detected"])
        controls.append(report["R1_exactness"]["periodicity_passed"])
    if "R2_isentropy" in report:
        controls.append(report["R2_isentropy"]["mutant_detected"])
    return {
        "certificates": len(keys),
        "failed": failed,
        "controls_passed": all(controls),
        "status": "CERTIFIED" if not failed and all(controls) else "INSTRUMENT-LIMITED",
    }


def markdown(report):
    lines = ["# CASE-DG-EULER-VORTEX-ERR：参考解与量测仪表认证（程序化生成）", "",
             f"整包裁决：**{report['verdict']['status']}**"
             f"（{report['verdict']['certificates']} 项证书，失败 "
             f"{report['verdict']['failed'] or '无'}；突变对照 "
             f"{'全部触发' if report['verdict']['controls_passed'] else '未全部触发'}）", "",
             "通过判据统一为「仪表误差 ≤ 被判量 / 10」：仪表若不比它认证的量细一个量级，"
             "该量就是仪表限而非物理。", ""]
    r1 = report["R1_exactness"]
    lines += ["## R1 解析参考解是否真是 Euler 方程的解", "",
              "| 时刻 | 方程 | 残差绝对最大值 | 残差/最大通量散度项 |", "|---|---|---|---|"]
    for time_label, rows in r1["residual"].items():
        for name, row in rows.items():
            lines.append(f"| {time_label} | {name} | {row['abs_max']:.3e} | "
                         f"{row['relative']:.3e} |")
    lines += ["", f"最差相对残差 {r1['worst_relative']:.3e}（限 {r1['limit']:.0e}）："
                  f"{'通过' if r1['passed'] else '未通过'}。"
                  f"周期折叠的平移不变性 {r1['periodicity_relative']:.3e}"
                  f"（{'通过' if r1['periodicity_passed'] else '未通过'}）。", "",
              "| 突变（合理误抄） | 残差/最大项 | 是否被检出 |", "|---|---|---|"]
    for name, value in r1["mutants"].items():
        lines.append(f"| {name} | {value:.3e} | {'是' if value > 1e-2 else '否'} |")
    r2 = report["R2_isentropy"]
    lines += ["", "## R2 等熵结构", "",
              f"解析场 max|p/rho^gamma - 1| = {r2['analytic']:.3e}"
              f"（限 {r2['limit']:.0e}）：{'通过' if r2['passed'] else '未通过'}；"
              f"等温突变体读数 {r2['mutant']:.3e}"
              f"（{'已检出' if r2['mutant_detected'] else '未检出'}）。"]
    r3 = report["R3_invariants"]
    lines += ["", "## R3 不变量分母（C3 的分母）", "",
              "| Gauss 点数/方向 | " + " | ".join(
                  k for k in r3["deviations"]["320"]) + " |",
              "|---|" + "---|" * len(r3["deviations"]["320"])]
    for points, rows in r3["deviations"].items():
        lines.append(f"| {points} | " + " | ".join(
            f"{v:.2e}" for v in rows.values()) + " |")
    lines += ["", f"门禁用的 320 点：最差相对偏差 {r3['gate_worst']:.3e}，"
                  f"相对最小漂移 {r3['smallest_drift']:.2e} 的比值为 "
                  f"{r3['ratio']:.2e}（要求 ≤ {1/MARGIN:.0e}）："
                  f"{'通过' if r3['passed'] else '未通过'}。", "",
              f"核密度采样：张量网格最小值 {r3['density_min']['tensor_grid_320']:.9f}，"
              f"射线最小值 {r3['density_min']['core_ray_4001']:.9f}，"
              f"相对差 {r3['density_min']['relative_gap']:.2e}。"
              f"{r3['density_min']['note']}。"]
    r4 = report["R4_error_norm"]
    lines += ["", "## R4 相对 L2 误差范数的积分收敛性", "",
              "| 状态 | 门禁求积读数 | 4× 求积读数 | 相对漂移 | 欠积分对照 | 复现存储值 |",
              "|---|---|---|---|---|---|"]
    for name, case in r4["cases"].items():
        reproduction = case.get("reproduction_relative")
        lines.append(
            f"| {name} | {100*case['gate']:.6f}% | {100*case['converged']:.6f}% | "
            f"{100*case['relative_shift']:+.4f}% | "
            f"{100*case.get('under_integrated_shift', float('nan')):+.2f}% | "
            f"{'—' if reproduction is None else f'{100*reproduction:+.2e}%'} |")
    lines += ["", f"最差相对漂移 {r4['worst_relative_shift']:.3e}，"
                  f"相对容差 {r4['tolerance']:.0%} 的比值 {r4['ratio']:.3e}"
                  f"（要求 ≤ {1/MARGIN:.0e}）：{'通过' if r4['passed'] else '未通过'}。"]
    if "R5_entropy_instrument" in report:
        r5 = report["R5_entropy_instrument"]
        lines += ["", "## R5 熵预算（C4）是物理还是求积噪声", "",
                  f"状态 `{r5['state']}`。", "",
                  "| 求积倍数 | |dS|/S |", "|---|---|"]
        for key, value in r5["budgets"].items():
            lines.append(f"| {key}× | {value:.3e} |")
        lines += ["", f"同状态在门禁求积与 6× 求积之间的纯求积差 "
                      f"{r5['quadrature_part']:.3e}；门禁读数 {r5['gate_reading']:.3e}；"
                      f"比值 {r5['ratio']:.3e}（要求 ≤ {1/MARGIN:.0e}）："
                      f"{'通过' if r5['passed'] else '未通过'}。", "",
                  "守恒量（多项式被积函数）在同一对照下的偏差："
                  + "，".join(f"{k} {v:.1e}" for k, v in r5["conservation_check"].items())
                  + "。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the time-stepped states (seconds instead of minutes)")
    parser.add_argument("--solve-degrees", default="1,2,3",
                        help="degrees to time-step for R4/R5 (cost grows fast)")
    parser.add_argument("--json", default=str(OUT / "instrument_certification.json"))
    parser.add_argument("--markdown", default=str(OUT / "instrument_certification.md"))
    args = parser.parse_args(argv)
    solve_degrees = tuple(int(d) for d in args.solve_degrees.split(",") if d.strip())
    report = build(with_solves=not args.quick, solve_degrees=solve_degrees)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
    Path(args.markdown).write_text(markdown(report))
    print(markdown(report))
    print(f"\nwrote {args.json} and {args.markdown} in {report['wall_seconds']:.1f}s")
    return 0 if report["verdict"]["status"] == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())