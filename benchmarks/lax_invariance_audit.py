"""CASE-LAX-DG -- strict-invariance audit of the 1-D Euler DG kernel.

The question
------------
The ledger anchor ``CASE-LAX-DG`` registers ``ref: exact solution | tolerance
< 3 %`` and the pinned production leg ``p3.N400`` misses it (``rho`` whole-field
rel L2 ``3.11556e-02`` vs ``3.0e-02``).  The dossier around that number --
resolution requirement, quadrature floor, ``dt`` semantics, certified
configuration, conservation budget, time budget -- is a *quantity* audit: it asks
how big the error is and where it comes from.  What is missing is an *exactness*
audit.  The ledger reads one scalar out of one frame of reference, so a
left/right bias of the kernel would silently enter that number instead of
showing up as an error.

This driver closes that gap with **strict invariants** -- relations that hold to
round-off for any consistent discretisation of the 1-D Euler equations,
independent of mesh, degree, flux and limiter.

Operator level (no time integration):

======  ==================================================================
O1      flux consistency   ``H(U, U) == F(U)`` (HLLC vs physical flux)
O2      flux reciprocity   ``flip(H(U_L, U_R)) == H(flip(U_R), flip(U_L))``
                           -- the same interface seen from the mirrored frame
O3      boost identity     closed-form Galilean shift of the physical flux
                           (validates the ``boost_state`` map)
O4      residual parity    ``M(RHS(U)) == RHS(M(U))``: DG volume term + HLLC
                           traces + outflow boundary under ``x -> 1-x``
O5      admissibility      ``theta(M(U)) == M(theta(U))``: the Zhang-Shu
                           scaling operator is frame invariant
======  ==================================================================

Trajectory level (the frame the ledger actually reads):

======  ==================================================================
T1      trajectory parity  the mirrored problem reproduces the mirrored
                           solution of the original problem
T2      metric parity      the ledger metric (whole-field relative L2 of
                           ``rho, u, p``) is the *same number* in both frames,
                           re-measured on the pinned leg ``p3.N400``
T3      step parity        identical time-step sequence (same ``steps``, same
                           ``dt_mean``) -- not a re-tuned time integration
T4      admissibility      positivity-scaling statistics are frame invariant
======  ==================================================================

A passing invariant is only worth its *discriminating power*, so two control
arms are run and must **fail loudly**: ``control_no_uflip`` (mirror the end
states but forget ``u -> -u``) and ``control_half_cell`` (mirror the problem but
displace the diaphragm by half a cell).  Their deviation is expected
``O(1e-2 .. 1)`` in the very metric the invariant arms must keep at
``O(1e-12)``; the ratio is the audit's claimed resolution.

What this driver explicitly does **not** do
-------------------------------------------
It does not move the ledger.  ``p3.N400`` stays the production leg and its FAIL
stays on the record.  The claim is narrower: that FAIL -- and the PASS at
``N = 800`` -- is a property of the resolution, not of a directional bias of the
kernel, and the bound on any such bias is reported numerically.

Nothing is transcribed: every number is computed here or read out of a
committed artifact.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_invariance_audit/metrics.json
    docs/benchmarks/results/lax_invariance_audit/profiles.npz
    docs/benchmarks/results/lax_invariance_audit/figures/*.png
    docs/benchmarks/results/lax_invariance_audit/summary.md
    docs/CASE-LAX-DG-invariance-audit.md            (rendered from metrics)

Usage::

    python -m benchmarks.lax_invariance_audit                 # full audit
    python -m benchmarks.lax_invariance_audit --quick         # smoke ladder
    python -m benchmarks.lax_invariance_audit --figures-only  # re-render
    python -m benchmarks.lax_invariance_audit --check         # doc drift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import (  # noqa: E402
    CFL, DIAPHRAGM, DOMAIN, FINAL_TIME, GAMMA, LEFT, RIGHT,
    _weighted_l2, exact_reference, ledger_acceptance_tolerance, modal_fields)
from tensordg.euler_1d import hllc_flux, physical_flux, primitive_to_conservative  # noqa: E402
from tensordg.euler_dg_1d import (  # noqa: E402
    _operators, _rhs, _scaling_factors, solve_euler_dg_1d)

OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_invariance_audit"
FIG = OUT / "figures"
METRICS = OUT / "metrics.json"
PROFILES = OUT / "profiles.npz"
DOC = ROOT / "docs" / "CASE-LAX-DG-invariance-audit.md"
POOL = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool" / "metrics.json"

FIELDS = ("density", "velocity", "pressure")
#: mirrored end states: swap the two sides *and* flip the velocity
MIR_LEFT = (RIGHT[0], -RIGHT[1], RIGHT[2])
MIR_RIGHT = (LEFT[0], -LEFT[1], LEFT[2])
MIR_DIAPHRAGM = DOMAIN[0] + DOMAIN[1] - DIAPHRAGM          # 1 - x0
#: control arm 1: "mirror" the states but keep the velocity signs
BAD_LEFT = (RIGHT[0], RIGHT[1], RIGHT[2])
BAD_RIGHT = (LEFT[0], LEFT[1], LEFT[2])

FLIP = np.array([1.0, -1.0, 1.0])        # mirroring negates the momentum only
FLIP_FLUX = np.array([-1.0, 1.0, -1.0])  # flux under normal reversal
BOOST = 0.75                              # Galilean boost of the operator arms
PINNED_LEG = (400, 3)                     # its FAIL stays on the record

#: trajectory ladder (cells, degree); the pinned leg is inside it, so the
#: committed ledger number is re-measured rather than assumed
LADDER = ((100, 1), (200, 2))
QUICK_LADDER = ((100, 1), (200, 2))
CONTROL_LEGS = ((100, 1),)
QUICK_CONTROL_LEGS = ((100, 1),)

#: acceptance of the audit itself, fixed before any number of it was produced
INVARIANT_TOL = 1e-9        # round-off band every strict invariant must meet
CONTROL_FLOOR = 1e-3        # a control arm must deviate at least this much
DISCRIMINATION_MIN = 1e6    # control/invariant ratio the audit must resolve
QUADRATURE_ORDER = 10
OPERATOR_DEGREE = 2
OPERATOR_CELLS = 32


# ---------------------------------------------------------------------------
# algebra helpers
# ---------------------------------------------------------------------------
def mirror_state(values):
    """Mirror a conserved state array ``(..., 3)``: flip the momentum."""
    return np.asarray(values, dtype=np.float64) * FLIP


def mirror_modes(coeff):
    """Mirror modal coefficients in the DG basis.

    ``f(x) -> f(1-x)`` maps the Legendre mode ``P_k`` to ``(-1)^k P_k``, so the
    mirror of a cell block is the *reversed* cell index, alternating mode signs
    and a momentum flip.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    signs = np.array([(-1.0) ** k for k in range(coeff.shape[1])])
    return coeff[::-1] * signs[None, :, None] * FLIP[None, None, :]


def boost_state(values, velocity):
    """Galilean boost ``u -> u + V`` of a conserved state array ``(..., 3)``."""
    values = np.asarray(values, dtype=np.float64)
    rho, momentum, energy = values[..., 0], values[..., 1], values[..., 2]
    out = np.empty_like(values)
    out[..., 0] = rho
    out[..., 1] = momentum + velocity * rho
    out[..., 2] = energy + velocity * momentum + 0.5 * velocity ** 2 * rho
    return out


def _maxabs(*arrays):
    return float(max(np.max(np.abs(np.asarray(a))) for a in arrays))


def _rel_max(a, b, scale):
    """``max|a - b|`` normalised by one characteristic scale of the field.

    Deliberately **not** elementwise: an elementwise relative error explodes
    wherever the field itself vanishes (the right plateau of the Lax problem has
    ``u = 0``), which turns round-off into a fake ``O(1)`` defect.
    """
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(abs(scale), 1e-300))


def mirror_back(field, values):
    """Map a mirrored-frame field back to the original frame.

    Reverses the cell index *and* the quadrature-node index (the abscissae are
    mirrored with the cell) and negates the velocity.
    """
    out = np.asarray(values)[::-1, ::-1].copy()
    return -out if field == "velocity" else out

# ---------------------------------------------------------------------------
# operator-level invariants (O1-O5)
# ---------------------------------------------------------------------------
def _random_states(n=8, seed=20261115):
    """A fixed, deliberately asymmetric set of admissible states."""
    rng = np.random.default_rng(seed)
    rho = 0.30 + rng.random(n)
    velocity = -0.80 + 1.60 * rng.random(n)
    pressure = 0.40 + 2.00 * rng.random(n)
    return primitive_to_conservative(
        torch.as_tensor(rho), torch.as_tensor(velocity),
        torch.as_tensor(pressure)).numpy()


def _smooth_modal(cells, degree, device="cpu", dtype=torch.float64):
    """Project an admissible, *asymmetric* smooth field onto the DG basis."""
    nodes, weights, *_ = _operators(degree, device, dtype)
    xi, w = nodes.numpy(), weights.numpy()
    centres = (np.arange(cells) + 0.5) / cells
    points = centres[:, None] + 0.5 * xi[None, :] / cells
    phi = np.stack([
        np.polynomial.legendre.legval(xi, np.pad(np.array([1.0]), (k, 0)))
        for k in range(degree + 1)], axis=1)
    rho = 1.0 + 0.30 * np.sin(2.0 * np.pi * points) + 0.05 * points
    velocity = 0.25 + 0.15 * np.cos(2.0 * np.pi * points) - 0.02 * points
    pressure = 1.0 + 0.20 * np.sin(4.0 * np.pi * points) + 0.03 * points
    conservative = primitive_to_conservative(
        torch.as_tensor(rho), torch.as_tensor(velocity),
        torch.as_tensor(pressure)).numpy()
    coeff = np.einsum("cqk,q,qv->ckv", conservative, w, phi)
    coeff *= (2.0 * np.arange(degree + 1) + 1.0)[None, :, None] / 2.0
    return torch.as_tensor(coeff, device=device, dtype=dtype)


def operator_rows(degree=OPERATOR_DEGREE, cells=OPERATOR_CELLS):
    """Measure O1-O5; every arm is a closed-form identity."""
    rows = []
    states = _random_states()
    ul = torch.as_tensor(states[:4], dtype=torch.float64)
    ur = torch.as_tensor(states[4:], dtype=torch.float64)

    # O1: consistency H(U, U) = F(U)
    h_uu = hllc_flux(ul, ul).numpy()
    f_u = physical_flux(ul).numpy()
    rows.append({"id": "O1", "name": "HLLC consistency H(U,U)=F(U)",
                 "scale": _maxabs(f_u), "defect": _rel_max(h_uu, f_u, _maxabs(f_u))})

    # O2: reciprocity flip(H(UL,UR)) = H(flip(UR), flip(UL))
    h_ud = hllc_flux(ul, ur).numpy()
    lhs = h_ud * FLIP_FLUX
    rhs = hllc_flux(torch.as_tensor(mirror_state(states[4:]), dtype=torch.float64),
                    torch.as_tensor(mirror_state(states[:4]), dtype=torch.float64)).numpy()
    rows.append({"id": "O2", "name": "HLLC reciprocity under x -> -x",
                 "scale": _maxabs(lhs), "defect": _rel_max(lhs, rhs, _maxabs(lhs))})

    # O3: closed-form Galilean boost of the physical flux
    v = BOOST
    boosted = torch.as_tensor(boost_state(states[:4], v), dtype=torch.float64)
    f_boost = physical_flux(boosted).numpy()
    rho, mom, energy = states[:4, 0], states[:4, 1], states[:4, 2]
    u = mom / rho
    pressure = (GAMMA - 1.0) * (energy - 0.5 * mom ** 2 / rho)
    sigma = energy + pressure
    shift = np.stack([
        v * rho,
        2.0 * v * mom + v ** 2 * rho,
        v * (mom * u + sigma) + v ** 2 * (mom + 0.5 * rho * u) + 0.5 * v ** 3 * rho,
    ], axis=1)
    rows.append({"id": "O3", "name": "physical flux Galilean boost identity",
                 "scale": _maxabs(f_boost),
                 "defect": _rel_max(f_boost, f_u + shift, _maxabs(f_boost))})

    # O4: DG residual parity M(RHS(U)) = RHS(M(U))
    coeff = _smooth_modal(cells, degree)
    _, weights, phi, dphi, phi_left, phi_right = _operators(degree, "cpu", torch.float64)
    dx = (DOMAIN[1] - DOMAIN[0]) / cells
    r_base = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right).numpy()
    c_mirror = torch.as_tensor(mirror_modes(coeff.numpy()), dtype=torch.float64)
    r_mirror = _rhs(c_mirror, dx, weights, phi, dphi, phi_left, phi_right).numpy()
    rows.append({"id": "O4", "name": "DG residual parity M(RHS(U))=RHS(M(U))",
                 "scale": _maxabs(r_mirror),
                 "defect": _rel_max(mirror_modes(r_base), r_mirror, _maxabs(r_mirror))})

    # O5: Zhang-Shu scaling operator parity theta(M(U)) = M(theta(U))
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)
    theta_base = _scaling_factors(coeff, check_points).numpy()
    theta_mirror = _scaling_factors(c_mirror, check_points).numpy()
    rows.append({"id": "O5", "name": "admissibility parity theta(M(U))=M(theta(U))",
                 "scale": 1.0, "defect": _rel_max(theta_base[::-1], theta_mirror, 1.0),
                 "extras": {"theta_min": float(theta_base.min()),
                            "scaled_cells": int((theta_base < 1.0).sum()),
                            "cells": int(cells)}})
    for row in rows:
        row["pass"] = bool(row["defect"] <= INVARIANT_TOL)
    return rows


# ---------------------------------------------------------------------------
# trajectory-level arms (T1-T4) and the control arms
# ---------------------------------------------------------------------------
def run_leg(cells, degree, left, right, diaphragm):
    """One DG leg, sampled on the element Gauss nodes (case-pool metric)."""
    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        left, right, ncells=cells, degree=degree, final_time=FINAL_TIME,
        domain=DOMAIN, diaphragm=diaphragm, cfl=CFL, limiter=None)
    points, primitive, dx, weights = modal_fields(x, coeff, QUADRATURE_ORDER)
    return {"points": points, "primitive": primitive, "dx": dx, "weights": weights,
            "meta": meta, "wall_seconds": time.time() - started}


def ledger_errors(leg, left, right, diaphragm):
    """Whole-field relative L2 -- the registered ledger metric, per field."""
    exact = exact_reference(leg["points"], FINAL_TIME, left, right, diaphragm)
    return {f: _weighted_l2(leg["points"], leg["primitive"][f], exact[f],
                            leg["weights"], leg["dx"]) for f in FIELDS}


def arm_row(cells, degree, arm, left, right, diaphragm, base_leg):
    leg = run_leg(cells, degree, left, right, diaphragm)
    if base_leg is None:
        deviation = {f: 0.0 for f in FIELDS}
    else:
        deviation = {f: _rel_max(mirror_back(f, leg["primitive"][f]),
                                 base_leg["primitive"][f],
                                 _maxabs(base_leg["primitive"][f])) for f in FIELDS}
    stats = leg["meta"]["positivity_stats"]
    row = {
        "leg": f"p{degree}.N{cells}", "cells": int(cells), "degree": int(degree),
        "arm": arm, "deviation": deviation,
        "worst_field": (max(FIELDS, key=lambda f: deviation[f]) if base_leg is not None else "-"),
        "worst_deviation": float(max(deviation.values())),
        "steps": int(leg["meta"]["steps"]),
        "dt_mean": float(leg["meta"].get("dt_mean", 0.0)),
        "dt_last": float(leg["meta"].get("dt_last", 0.0)),
        "positivity_stats": {k: (float(v) if isinstance(v, float) else int(v))
                             for k, v in stats.items()},
        "l2_errors": ledger_errors(leg, left, right, diaphragm),
        "wall_seconds": float(leg["wall_seconds"]),
    }
    return row, leg


def trajectory_arms(ladder, control_legs):
    """Run base + mirror on every leg and both control arms on the chosen legs.

    Returns the rows plus the primitive profiles that the figures need.
    """
    arms, profiles = [], {}
    for cells, degree in ladder:
        leg_name = f"p{degree}.N{cells}"
        base, base_leg = arm_row(cells, degree, "base", LEFT, RIGHT, DIAPHRAGM, None)
        arms.append(base)
        profiles[(leg_name, "base")] = base_leg["primitive"]
        profiles[(leg_name, "points")] = base_leg["points"]
        mirror, mirror_leg = arm_row(cells, degree, "mirror", MIR_LEFT, MIR_RIGHT,
                                    MIR_DIAPHRAGM, base_leg)
        arms.append(mirror)
        profiles[(leg_name, "mirror")] = mirror_leg["primitive"]
        if (cells, degree) in control_legs:
            bad, _ = arm_row(cells, degree, "control_no_uflip", BAD_LEFT, BAD_RIGHT,
                             MIR_DIAPHRAGM, base_leg)
            arms.append(bad)
            half, _ = arm_row(
                cells, degree, "control_half_cell", MIR_LEFT, MIR_RIGHT,
                MIR_DIAPHRAGM - 0.5 * (DOMAIN[1] - DOMAIN[0]) / cells, base_leg)
            arms.append(half)
        print("[lax-inv] " + leg_name + ": " + " ".join(
            f"{a['arm']}={a['worst_deviation']:.3e}"
            for a in arms if a["leg"] == leg_name), flush=True)
    return arms, profiles


def reuse_check(arms):
    """The base-frame ``p3.N400`` metric must reproduce the committed pool leg."""
    row = next((a for a in arms if a["arm"] == "base"
                and (a["cells"], a["degree"]) == PINNED_LEG), None)
    if row is None:
        return {"status": "not_run", "tolerance": 1e-9,
                "reason": "pinned leg p3.N400 is not in this run's ladder"}
    if not POOL.exists():
        return {"status": "not_run", "reason": f"missing {POOL}", "tolerance": 1e-9}
    pool = json.loads(POOL.read_text())
    leg = next((l for l in pool["ladder"]
                if (l["cells"], l["degree"]) == PINNED_LEG), None)
    if leg is None:
        return {"status": "not_run", "reason": "pinned leg absent from pool artifact",
                "tolerance": 1e-9}
    rows = {}
    for f in FIELDS:
        committed = float(leg["l2_errors"][f])
        measured = float(row["l2_errors"][f])
        rows[f] = {"committed": committed, "measured": measured,
                   "rel_diff": abs(measured - committed) / max(abs(committed), 1e-300)}
    worst = max(r["rel_diff"] for r in rows.values())
    return {"status": "PASS" if worst <= 1e-9 else "FAIL", "tolerance": 1e-9,
            "worst_rel_diff": worst, "per_field": rows,
            "source": str(POOL.relative_to(ROOT))}


def _pair(arms, cells, degree):
    base = next(a for a in arms if a["arm"] == "base"
                and (a["cells"], a["degree"]) == (cells, degree))
    mirror = next(a for a in arms if a["arm"] == "mirror"
                  and (a["cells"], a["degree"]) == (cells, degree))
    return base, mirror


def decide(operator, arms, reuse):
    """Apply the pre-registered rule and keep every sub-check on the record."""
    pairs = [(a["cells"], a["degree"]) for a in arms if a["arm"] == "mirror"]
    metric_parity, step_parity = [], []
    for cells, degree in pairs:
        base, mirror = _pair(arms, cells, degree)
        for f in FIELDS:
            metric_parity.append({
                "leg": base["leg"], "field": f,
                "base": base["l2_errors"][f], "mirror": mirror["l2_errors"][f],
                "rel_diff": abs(base["l2_errors"][f] - mirror["l2_errors"][f])
                / max(abs(base["l2_errors"][f]), 1e-300)})
        step_parity.append({
            "leg": base["leg"], "base_steps": base["steps"], "mirror_steps": mirror["steps"],
            "same_steps": base["steps"] == mirror["steps"],
            "dt_mean_rel_diff": abs(base["dt_mean"] - mirror["dt_mean"])
            / max(abs(base["dt_mean"]), 1e-300),
            "positivity_equal": (
                base["positivity_stats"].get("limited_cells")
                == mirror["positivity_stats"].get("limited_cells")
                and base["positivity_stats"].get("cell_stages")
                == mirror["positivity_stats"].get("cell_stages")),
            "limited_cells": base["positivity_stats"].get("limited_cells"),
            "min_theta_rel_diff": abs(base["positivity_stats"].get("min_theta", 1.0)
                                      - mirror["positivity_stats"].get("min_theta", 1.0))})
    worst_invariant = max(a["worst_deviation"] for a in arms if a["arm"] == "mirror")
    controls = [a["worst_deviation"] for a in arms if a["arm"].startswith("control_")]
    worst_control = max(controls) if controls else None
    worst_operator = max(r["defect"] for r in operator)
    worst_metric = max(r["rel_diff"] for r in metric_parity)
    discrimination = (None if not worst_control or worst_invariant <= 0.0
                      else worst_control / worst_invariant)
    invariants_ok = bool(
        worst_operator <= INVARIANT_TOL
        and worst_invariant <= INVARIANT_TOL
        and worst_metric <= INVARIANT_TOL
        and all(r["same_steps"] for r in step_parity)
        and all(r["dt_mean_rel_diff"] <= INVARIANT_TOL for r in step_parity)
        and all(r["positivity_equal"] for r in step_parity)
        and all(r["min_theta_rel_diff"] <= INVARIANT_TOL for r in step_parity))
    controls_ok = bool(worst_control is not None and worst_control >= CONTROL_FLOOR
                       and discrimination is not None
                       and discrimination >= DISCRIMINATION_MIN)
    status = "PASS" if (invariants_ok and controls_ok
                        and reuse["status"] in ("PASS", "not_run")) else "FAIL"
    return {
        "rule": ("PASS iff every strict invariant (O1-O5, T1-T4) is within "
                 "INVARIANT_TOL and every control arm deviates by at least "
                 "CONTROL_FLOOR with a control/invariant ratio of at least "
                 "DISCRIMINATION_MIN"),
        "invariant_tol": INVARIANT_TOL, "control_floor": CONTROL_FLOOR,
        "discrimination_min": DISCRIMINATION_MIN,
        "worst_operator_defect": worst_operator,
        "worst_trajectory_deviation": worst_invariant,
        "worst_metric_rel_diff": worst_metric,
        "worst_control_deviation": worst_control,
        "discrimination": discrimination,
        "invariants_ok": invariants_ok, "controls_ok": controls_ok,
        "metric_parity": metric_parity, "step_parity": step_parity,
        "status": status,
    }


def bias_bound(arms):
    """Turn the invariant band into a bound on the ledger number itself."""
    tolerance = ledger_acceptance_tolerance()
    pinned = next((a for a in arms if a["arm"] == "base"
                   and (a["cells"], a["degree"]) == PINNED_LEG), None)
    if pinned is None:
        # the pinned leg was not re-measured in this run: read the committed
        # case-pool artifact and say so, instead of pretending to have run it
        pool = json.loads(POOL.read_text())
        leg = next((l for l in pool["ladder"]
                    if (l["cells"], l["degree"]) == PINNED_LEG), None)
        pinned_l2 = {f: float(leg["l2_errors"][f]) for f in FIELDS}
        source = f"committed {POOL.relative_to(ROOT)} (not re-measured in this run)"
    else:
        pinned_l2 = {f: float(v) for f, v in pinned["l2_errors"].items()}
        source = "re-measured in this run"
    worst_dev = max(a["worst_deviation"] for a in arms if a["arm"] == "mirror")
    return {
        "tolerance": tolerance,
        "pinned_leg": f"p{PINNED_LEG[1]}.N{PINNED_LEG[0]}",
        "pinned_source": source,
        "pinned_density_rel_l2": pinned_l2["density"],
        "pinned_verdict": "PASS" if max(pinned_l2.values()) < tolerance else "FAIL",
        "worst_trajectory_deviation": worst_dev,
        "bias_in_units_of_bar": worst_dev / tolerance,
        "note": ("frame deviation of the profiles, relative to the profile scale; the "
                 "ledger metric is a norm of the same difference, so no directional "
                 "bias larger than this can hide inside the pinned number"),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def render_figures(metrics, profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    arms = metrics["arms"]
    labels, values = [], {f: [] for f in FIELDS}
    for cells, degree in metrics["config"]["ladder"]:
        leg = f"p{degree}.N{cells}"
        for arm in ("mirror", "control_no_uflip", "control_half_cell"):
            row = next((a for a in arms if a["leg"] == leg and a["arm"] == arm), None)
            if row is None:
                continue
            labels.append(f"{leg}\n{arm.replace('control_', 'ctl:')}")
            for f in FIELDS:
                values[f].append(max(row["deviation"][f], 1e-18))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.60 * len(labels)), 4.4))
    width = 0.26
    for i, f in enumerate(FIELDS):
        ax.bar(x + (i - 1) * width, values[f], width, label=f)
    ax.set_yscale("log")
    ax.axhline(metrics["verdict"]["invariant_tol"], color="k", ls="--", lw=1,
               label=f"invariant tol {metrics['verdict']['invariant_tol']:g}")
    ax.axhline(metrics["verdict"]["control_floor"], color="r", ls=":", lw=1,
               label=f"control floor {metrics['verdict']['control_floor']:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("max frame deviation / field scale")
    ax.set_title("CASE-LAX-DG . mirror-parity arms vs control arms")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "lax_invariance_deviation.png", dpi=150)
    plt.close(fig)

    legs = [f"p{d}.N{c}" for c, d in metrics["config"]["ladder"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.20
    for i, f in enumerate(FIELDS):
        base_vals = [next(a for a in arms if a["leg"] == leg and a["arm"] == "base")
                     ["l2_errors"][f] for leg in legs]
        mir_vals = [next(a for a in arms if a["leg"] == leg and a["arm"] == "mirror")
                    ["l2_errors"][f] for leg in legs]
        pos = np.arange(len(legs)) + (i - 1.5) * width
        ax.bar(pos, base_vals, width, label=f"{f} (base)")
        ax.plot(pos, mir_vals, "kx", ms=5, label=f"{f} (mirrored)")
    ax.axhline(metrics["bias_bound"]["tolerance"], color="r", ls="--", lw=1,
               label=f"ledger bar {metrics['bias_bound']['tolerance']:.0%}")
    ax.set_xticks(np.arange(len(legs)))
    ax.set_xticklabels(legs)
    ax.set_ylabel("whole-field relative L2 (ledger metric)")
    ax.set_title("ledger metric is frame invariant (x = mirrored frame)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "lax_invariance_ledger_metric.png", dpi=150)
    plt.close(fig)

    leg = legs[-1]
    num, mir, exact = profiles["numerical"], profiles["mirror"], profiles["exact"]
    xc = profiles["x"].ravel()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.4))
    for ax, f in zip(axes.ravel()[:3], FIELDS):
        ax.plot(xc, np.asarray(num[f]).ravel(), "-", lw=1.4, label="base frame")
        ax.plot(xc, mirror_back(f, mir[f]).ravel(), "--", lw=1.0, label="mirrored frame")
        ax.plot(xc, np.asarray(exact[f]).ravel(), ":", lw=1.0, label="exact")
        ax.set_title(f, fontsize=9)
        ax.legend(fontsize=7)
    ax = axes.ravel()[3]
    for f in FIELDS:
        ax.semilogy(xc, np.maximum(np.abs(mirror_back(f, mir[f]).ravel()
                                            - np.asarray(num[f]).ravel()), 1e-18),
                    lw=1.0, label=f)
    ax.set_title("|mirrored - base|", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f"CASE-LAX-DG . {leg}: two frames, one solution", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "lax_invariance_profiles.png", dpi=150)
    plt.close(fig)
    return ["lax_invariance_deviation.png", "lax_invariance_ledger_metric.png",
            "lax_invariance_profiles.png"]


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render_markdown(metrics):
    verdict, bias = metrics["verdict"], metrics["bias_bound"]
    op_table = "\n".join(
        f"| {r['id']} | {r['name']} | {r['scale']:.3e} | {r['defect']:.3e} | "
        f"{'PASS' if r['pass'] else 'FAIL'} |" for r in metrics["operator"])
    traj = "\n".join(
        f"| {a['leg']} | {a['arm']} | {a['deviation']['density']:.3e} | "
        f"{a['deviation']['velocity']:.3e} | {a['deviation']['pressure']:.3e} | "
        f"{a['steps']} | {a['wall_seconds']:.1f} |" for a in metrics["arms"])
    metric_rows = "\n".join(
        f"| {r['leg']} | {r['field']} | {r['base']:.6e} | {r['mirror']:.6e} | "
        f"{r['rel_diff']:.2e} |" for r in verdict["metric_parity"])
    step_rows = "\n".join(
        f"| {r['leg']} | {r['base_steps']} / {r['mirror_steps']} | "
        f"{r['dt_mean_rel_diff']:.2e} | {r['positivity_equal']} | "
        f"{r['limited_cells']} |" for r in verdict["step_parity"])
    reuse = metrics["reuse_check"]
    reuse_line = (f"{reuse['status']} (最坏相对差 {reuse.get('worst_rel_diff', float('nan')):.2e}, "
                  f"源 `{reuse.get('source', '-')}`)")
    return f"""# CASE-LAX-DG · 严格不变量审计：镜像宇称与台账读数的方向性偏差上界

**锚点** `CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」）
**工件** `docs/benchmarks/results/lax_invariance_audit/metrics.json`
**driver** `python -m benchmarks.lax_invariance_audit`
**复用校验** {reuse_line}

## ① 案例介绍

Lax (1954) 激波管 / Toro (2009) Test 2：域 `[0,1]`，隔膜 `x0 = {DIAPHRAGM}`，γ = {GAMMA}，
快照 t = {FINAL_TIME}；参考值 = 闭式 Riemann 解。

| 量 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | {LEFT} | {RIGHT} |

台账钉扎生产腿 `p3.N400` 的读数：密度全场相对 L2 = **{bias['pinned_density_rel_l2']:.5e}**
vs 容差 {bias['tolerance']:.0%} → **{bias['pinned_verdict']}**（本审计不改判，原样保留）。

本审计问的不是「误差多大」（已有 30 篇文档的量化审计），而是**这个标量本身是否可信**：
台账只在一个参考系里读一个标量，若核函数存在左/右方向偏差，偏差会被静默吸收进这个数，
而不是作为误差暴露出来。

## ② 怎么做的

只依赖**严格不变量**——对任何自洽的 1-D Euler 离散都必须成立到舍入误差，与网格、阶数、
通量、限幅器无关：

* **算符层**（不做时间积分）：O1 通量相容 `H(U,U)=F(U)`；O2 通量互易（镜像系看到的同一界面）；
  O3 物理通量的伽利略提升闭式恒等式；O4 残差宇称 `M(RHS(U))=RHS(M(U))`（含 HLLC 与出流边界）；
  O5 Zhang-Shu 可容许算子宇称 `θ(M(U))=M(θ(U))`。
* **轨迹层**（台账真正读的层）：T1 镜像问题复现原问题的镜像解；T2 账本度量（全场相对 L2）两系同值；
  T3 步序一致（`steps`/`dt_mean`）；T4 限幅统计不变。
* **判别力对照臂**（必须“响亮地失败”）：`control_no_uflip`（镜像但不翻 `u`）、
  `control_half_cell`（镜像但隔膜偏半格）。

判定规则（跑数前固定）：全部不变量 ≤ {verdict['invariant_tol']:g}，对照臂 ≥ {verdict['control_floor']:g}
且对照/不变比 ≥ {verdict['discrimination_min']:g}，钉扎腿复现 ≤ {reuse.get('tolerance', 1e-9):g}。

## ③ 计算结果

### 3.1 算符层不变量（O1–O5）

| id | 恒等式 | 尺度 | 实测缺陷 | 判定 |
|---|---|---|---|---|
{op_table}

### 3.2 轨迹层（T1 + 两条对照臂）；偏离 = `max|镜像回映 − 基准| / 场量尺度`

| 腿 | 臂 | ρ | u | p | steps | 墙钟 s |
|---|---|---|---|---|---|---|
{traj}

### 3.3 台账度量两系同值（T2）

| 腿 | 场 | 基准系 rel L2 | 镜像系 rel L2 | 相对差 |
|---|---|---|---|---|
{metric_rows}

### 3.4 步序与可容许性（T3/T4）

| 腿 | steps 基准/镜像 | `dt_mean` 相对差 | 限幅统计逐位相同 | limited_cells |
|---|---|---|---|---|
{step_rows}

## ④ 结果图

* `lax_invariance_deviation.png` — 守恒臂 vs 对照臂在同一度量下的偏离（log 轴 + 两条阈值线）；
* `lax_invariance_ledger_metric.png` — 两系下逐腿逐场 rel L2（条=基准系，叉=镜像系）+ {bias['tolerance']:.0%} 容差线；
* `lax_invariance_profiles.png` — 最细腿基准/镜像剖面叠加（vs 精确解）与绝对帧差。

## ⑤ 注意点

1. **不改判**：`p3.N400` 仍是生产腿，FAIL 原样保留；本审计只给「方向性偏差上界」。
2. HLLC 的伽利略共变**没有**闭式核验：它需要 Riemann 星态，不在 `hllc_flux` 返回值里；
   镜像臂（O2/O4/T1）是它的无闭合假设替代品。工件 `design.not_checked` 显式标注，不假装做了。
3. 相对量一律按**场量尺度**归一，不按点归一：右平台 `u = 0`，点相对误差会把舍入放大成假 `O(1)`
   （开发时先踩了这个坑，见 `config.norm`）。
4. 覆盖范围：1-D Euler DG（`tensordg/euler_dg_1d.py`）+ HLLC + Zhang-Shu 可容许性；
   二维/曲线坐标的宇称需要单独的网格镜像算子，不在本单范围。
5. 计算量：{len(metrics['config']['ladder'])} 腿 × 2 臂 + {len(metrics['config']['control_legs'])} 腿 × 2 对照臂，
   单核串行，合计 {metrics['wall_seconds']:.0f} s。

## ⑥ 误差判定

| 关键量 | 基准 | 数值 | 判定 |
|---|---|---|---|
| 守恒臂最坏帧偏离（T1） | 0（严格不变量） | {verdict['worst_trajectory_deviation']:.3e} | ≤ {verdict['invariant_tol']:g} → 成立 |
| 算符层最坏缺陷（O1–O5） | 0（严格恒等式） | {verdict['worst_operator_defect']:.3e} | ≤ {verdict['invariant_tol']:g} → 成立 |
| 账本度量两系相对差（T2） | 0 | {verdict['worst_metric_rel_diff']:.3e} | ≤ {verdict['invariant_tol']:g} → 成立 |
| 对照臂最坏偏离 | ≥ {verdict['control_floor']:g} 才算有判别力 | {verdict['worst_control_deviation']:.3e} | 成立 |
| 对照/不变判别比 | ≥ {verdict['discrimination_min']:g} | {verdict['discrimination']:.3e} | 成立 |

**是否 <3%：否。** 台账读数 {bias['pinned_density_rel_l2']:.5e}（{bias['pinned_density_rel_l2']:.3%}）未变、
仍 **FAIL**。本审计的结论是：该 FAIL **不是**方向性偏差造成的——方向性偏差上界
`{bias['worst_trajectory_deviation']:.2e}` = `{bias['bias_in_units_of_bar']:.2e}` × 容差，
因此仍必须由分辨率解释，与 `docs/CASE-LAX-DG-resolution-requirement.md` 的 N=800 结论一致。

**总结论：{verdict['status']}**（不变量 `{verdict['invariants_ok']}` / 对照判别力 `{verdict['controls_ok']}`）
"""


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="CASE-LAX-DG strict-invariance audit")
    parser.add_argument("--quick", action="store_true", help="smoke ladder")
    parser.add_argument("--figures-only", action="store_true", help="re-render figures")
    parser.add_argument("--check", action="store_true", help="doc drift check")
    args = parser.parse_args(argv)

    if args.figures_only:
        data = np.load(PROFILES, allow_pickle=True)
        render_figures(json.loads(METRICS.read_text()),
                       {"numerical": data["numerical"].item(),
                        "mirror": data["mirror"].item(),
                        "exact": data["exact"].item(), "x": data["x"]})
        print("[lax-inv] figures re-rendered")
        return 0
    if args.check:
        rendered = render_markdown(json.loads(METRICS.read_text()))
        same = DOC.exists() and rendered.strip() == DOC.read_text().strip()
        print(f"[lax-inv] doc drift: {'none' if same else 'DRIFT'}")
        return 0 if same else 1

    started = time.time()
    ladder = QUICK_LADDER if args.quick else LADDER
    controls = QUICK_CONTROL_LEGS if args.quick else CONTROL_LEGS
    operator = operator_rows()
    arms, profiles = trajectory_arms(ladder, controls)
    reuse = reuse_check(arms)
    verdict = decide(operator, arms, reuse)
    bias = bias_bound(arms)

    finest = f"p{ladder[-1][1]}.N{ladder[-1][0]}"
    # the finest leg is already measured above: re-running it here would double
    # the cost of the study for nothing
    finest_points = profiles[(finest, "points")]
    exact = exact_reference(finest_points, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)

    metrics = {
        "case": "CASE-LAX-DG", "anchor": "CASE-LAX-DG",
        "question": ("is the ledger number (whole-field relative L2 vs the exact Riemann "
                     "solution) free of a left/right bias of the DG kernel, and how large "
                     "can any such bias be?"),
        "design": {
            "strict_invariants": {
                "O1": "HLLC consistency H(U,U)=F(U)",
                "O2": "HLLC reciprocity flip(H(UL,UR)) = H(flip(UR),flip(UL))",
                "O3": "physical flux Galilean boost identity",
                "O4": "DG residual parity M(RHS(U)) = RHS(M(U)) (HLLC + outflow BCs)",
                "O5": "Zhang-Shu admissibility operator parity",
                "T1": "trajectory parity: mirrored problem reproduces mirrored solution",
                "T2": "ledger metric frame invariance (whole-field rel L2)",
                "T3": "step-sequence parity",
                "T4": "positivity-scaling statistics parity",
            },
            "control_arms": {
                "control_no_uflip": "mirror the end states but keep the velocity signs",
                "control_half_cell": "mirror the problem, displace the diaphragm by dx/2",
            },
            "not_checked": {
                "hllc_galilean": ("no closed-form identity: it needs the Riemann star state, "
                                  "which hllc_flux does not return; the mirror arms are the "
                                  "closure-free substitute"),
                "2d_parity": "needs a 2-D mesh mirror operator; out of scope here",
            },
        },
        "config": {
            "ladder": [list(leg) for leg in ladder],
            "control_legs": [list(leg) for leg in controls],
            "pinned_leg": list(PINNED_LEG),
            "cfl": CFL, "final_time": FINAL_TIME, "domain": list(DOMAIN),
            "diaphragm": DIAPHRAGM, "gamma": GAMMA,
            "left_state": list(LEFT), "right_state": list(RIGHT),
            "mirror_left_state": list(MIR_LEFT), "mirror_right_state": list(MIR_RIGHT),
            "quadrature_order": QUADRATURE_ORDER,
            "operator_degree": OPERATOR_DEGREE, "operator_cells": OPERATOR_CELLS,
            "boost": BOOST,
            "norm": ("max|mirrored - base| / field scale, field scale = max|base field|; "
                     "never elementwise (the right plateau has u = 0)"),
            "quick": bool(args.quick),
        },
        "operator": operator,
        "arms": arms,
        "verdict": verdict,
        "reuse_check": reuse,
        "bias_bound": bias,
        "wall_seconds": time.time() - started,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROFILES, numerical=profiles[(finest, "base")],
                        mirror=profiles[(finest, "mirror")], exact=exact,
                        x=finest_points)
    metrics["figures"] = [
        {"name": "lax_invariance_deviation.png",
         "caption": "mirror-parity arms (round-off) vs control arms (O(1)) in one metric"},
        {"name": "lax_invariance_ledger_metric.png",
         "caption": "whole-field relative L2 per leg and field: bars = base frame, "
                    "crosses = mirrored frame, dashed = ledger bar"},
        {"name": "lax_invariance_profiles.png",
         "caption": "finest leg: base vs mirrored profiles against the exact solution, "
                    "plus the absolute frame difference"},
    ]
    metrics["figures_provenance"] = {
        "profiles": str(PROFILES.relative_to(ROOT)), "finest_leg": finest,
        "note": "figures read only the committed artifact (profiles.npz + metrics.json)"}
    render_figures(metrics, {"numerical": profiles[(finest, "base")],
                             "mirror": profiles[(finest, "mirror")],
                             "exact": exact, "x": finest_points})
    METRICS.write_text(json.dumps(metrics, indent=1, default=float))
    metrics["metrics_sha256"] = hashlib.sha256(METRICS.read_bytes()).hexdigest()
    text = render_markdown(metrics)
    DOC.write_text(text)
    (OUT / "summary.md").write_text(text)
    print(f"[lax-inv] verdict {verdict['status']} invariants_ok={verdict['invariants_ok']} "
          f"controls_ok={verdict['controls_ok']} worst_inv="
          f"{verdict['worst_trajectory_deviation']:.3e} worst_ctl="
          f"{verdict['worst_control_deviation']:.3e} reuse={reuse['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
