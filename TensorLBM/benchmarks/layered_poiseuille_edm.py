"""XVAL-LBM-TWO-COMPONENT-SC-LAYERED-POISEUILLE-EDM benchmark.

Scores ``tensorlbm.layered_poiseuille`` (two-component Shan-Chen channel with
EDM forcing and half-way bounce-back walls) against the exact layered Stokes
solution.  Four arms, all pre-registered:

===============  =============================================================
``L0``           reference self-check: the generic N-layer linear solve against
                 the hand-eliminated two-layer closed form, plus no-slip /
                 velocity / shear-stress continuity invariants (no solver).
``L1``           single-component EDM channel from rest -> analytic parabola.
                 Certifies the forcing scheme and the wall condition in the
                 absence of any two-phase physics.
``L2``           two-component layered channel, viscosity ratio
                 ``nu(0.6)/nu(1.0) = 2.33`` (component 0 below, richer heavy
                 fluid below the interface), from rest -> layered Stokes.
``L3``           same geometry, reversed contrast (omega = [0.6, 1.0]).
``L4``           low contrast ``omega = [1.0, 0.8]``.
===============  =============================================================

Acceptance (owner precision standard): the bulk-region relative error of the
stream velocity against the analytic solution must be ``< 15%``; the 3% target
is reported but never used to gate.  The diffuse-interface band (``|y-h| <= 4``)
is excluded from the bulk score *and reported separately* so the exclusion
cannot hide a defect.

Usage (one arm per call -- shardable)::

    python TensorLBM/benchmarks/layered_poiseuille_edm.py --arm L2 \
        --steps 40000 --ny 64 --nx 4

Every arm writes ``TensorLBM/outputs/layered_poiseuille/<arm>.json`` with the
measured profiles and the analytic reference sampled at the same nodes, so the
report figures and tables are produced from data, never hand-typed.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TensorLBM" / "src"))
sys.path.insert(0, str(ROOT / "TensorLBM" / "tools"))

from tensorlbm.layered_poiseuille import (  # noqa: E402
    ShanChenLayeredPoiseuille,
    layered_poiseuille_analytic,
    layered_poiseuille_two_layer_closed_form,
    local_viscosity,
    profile_error,
    viscosity_from_omega,
)
from scmp_planar_reference import coexistence_tieline  # noqa: E402

OUTDIR = ROOT / "TensorLBM" / "outputs" / "layered_poiseuille"
G_SC = -2.2          # Shan-Chen coupling (demixing, linear psi, rho_bar = 0.5)
RHO_BAR = 0.5
IFACE_FRACTION = 0.5  # interface at half the channel height


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_json_default(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def run_to_steady(sim, max_steps, check_every=250, tol=1e-9, log_every=2000):
    """Integrate until the stream-velocity profile stops moving.

    Returns ``(steps, residual_history)``.  The residual is the max-norm change
    of the mass-averaged stream profile between checks, normalised by its own
    peak, i.e. a direct steady-state residual in the same units as the error
    being reported downstream.
    """
    hist = []
    prev = sim.profile_u_mix()
    t0 = time.time()
    while sim.step_count < max_steps:
        sim.run(check_every)
        cur = sim.profile_u_mix()
        scale = max(np.max(np.abs(cur)), 1e-30)
        res = float(np.max(np.abs(cur - prev)) / scale)
        hist.append([sim.step_count, res, time.time() - t0])
        prev = cur
        if sim.step_count % log_every < check_every:
            print(f"    step {sim.step_count:6d} residual {res:.3e} "
                  f"u_max {np.max(np.abs(cur)):.6f} t={time.time() - t0:.1f}s",
                  flush=True)
        if not np.isfinite(res):
            break
        if res < tol:
            break
    return sim.step_count, hist


def build_arm(arm, ny, nx, steps, g_body):
    """Return (sim, analytic dict, meta) for the requested arm."""
    H = float(ny - 2)
    y = np.arange(1, ny - 1) - 0.5
    h1 = IFACE_FRACTION * H
    if arm == "L1":
        omega = [1.0]
        sim = ShanChenLayeredPoiseuille(ny, nx, omega, G12=0.0, g_body=g_body)
        analytic = layered_poiseuille_analytic(
            [viscosity_from_omega(omega[0])], [], H, g_body, y=y)
        meta = {"kind": "single-phase", "omega": omega, "G12": 0.0}
    elif arm in ("L2", "L3", "L4"):
        omega = {"L2": [1.0, 0.6], "L3": [0.6, 1.0], "L4": [1.0, 0.8]}[arm]
        rho_hi, rho_lo = coexistence_tieline(G_SC, RHO_BAR)
        sim = ShanChenLayeredPoiseuille(ny, nx, omega, G12=G_SC, g_body=g_body)
        sim.set_layered_profile(rho_hi, rho_lo, h1, width=2.0)
        nu = [viscosity_from_omega(o) for o in omega]
        analytic = layered_poiseuille_analytic(nu, [h1], H, g_body, y=y)
        meta = {"kind": "two-component", "omega": omega, "G12": G_SC,
                "rho_hi": rho_hi, "rho_lo": rho_lo, "nu": nu,
                "viscosity_ratio": max(nu) / min(nu), "interface_y": h1}
    else:
        raise SystemExit(f"unknown arm {arm}")
    meta.update({"ny": ny, "nx": nx, "H": H, "g_body": g_body,
                 "max_steps": steps})
    return sim, analytic, meta, y, h1


def score_arm(sim, analytic, y, h1, meta):
    """Measure, score and package one arm."""
    prof = sim.profiles()
    u_ref_nodes = analytic["u"]
    u_ref_fn = (lambda yy: layered_poiseuille_analytic(
        analytic["nu"], analytic["interfaces"], analytic["H"],
        analytic["g"], y=yy)["u"])

    two_layer = len(analytic["nu"]) == 2
    band = [(h1 - 4.0, h1 + 4.0)] if two_layer else []
    total = profile_error(prof["y"], prof["u_mix"], u_ref_fn)
    bulk = profile_error(prof["y"], prof["u_mix"], u_ref_fn, exclude=band)
    iface = profile_error(prof["y"], prof["u_mix"], u_ref_fn,
                          exclude=[(-1.0, h1 - 4.0), (h1 + 4.0, 1e9)])

    out = {
        "arm": meta["arm"],
        "meta": meta,
        "steps": sim.step_count,
        "y": prof["y"],
        "u_mix": prof["u_mix"],
        "u_analytic": u_ref_nodes,
        "rho_tot": prof["rho_tot"],
        "uy_max": sim.max_spurious_velocity(),
        "mass": sim.total_mass(),
        "err_all": total,
        "err_bulk": bulk,
        "err_interface_band": iface,
    }
    for a in range(sim.n_components):
        out[f"rho{a}"] = prof[f"rho{a}"]
        out[f"ux{a}"] = prof[f"ux{a}"]
        out[f"uy{a}"] = prof[f"uy{a}"]

    # model-free viscosity inversion in the two bulk halves
    nu_local = local_viscosity(prof["y"], prof["u_mix"], meta["g_body"],
                               half_width=3)
    out["nu_local"] = nu_local
    half = max(int(0.3 * prof["y"].size), 5)
    lower = float(np.nanmedian(nu_local[half:2 * half]))
    upper = float(np.nanmedian(nu_local[-2 * half:-half]))
    out["nu_measured"] = {"lower": lower, "upper": upper}
    if two_layer:
        nu_l, nu_u = meta["nu"]
        out["nu_reference"] = {"lower": nu_l, "upper": nu_u}
        out["nu_error"] = {
            "lower": abs(lower - nu_l) / nu_l,
            "upper": abs(upper - nu_u) / nu_u,
        }
    if "rho_hi" in meta:
        # interface position from the steepest total-composition gradient
        g = np.gradient(prof["rho0"] - prof["rho1"], prof["y"])
        out["interface_y_measured"] = float(prof["y"][int(np.argmax(np.abs(g)))])
    return out


def arm_L0():
    """Reference self-check: two independent routes + continuity invariants."""
    H, g = 62.0, 2e-5
    checks = []
    for nu1, nu2, h1 in [(1 / 6, 0.6, 31.0), (0.05, 0.5, 20.0), (0.4, 0.4, 30.0)]:
        y = np.linspace(0.0, H, 501)
        a = layered_poiseuille_analytic([nu1, nu2], [h1], H, g, y=y)
        b = layered_poiseuille_two_layer_closed_form(nu1, nu2, h1, H, g, y=y)
        scale = float(np.max(np.abs(a["u"])))
        # invariants at the interfaces
        coeffs = a["coeffs"]
        dud1 = coeffs[0, 0] - g * h1 / nu1
        dud2 = coeffs[1, 0] - g * h1 / nu2
        checks.append({
            "nu1": nu1, "nu2": nu2, "h1": h1, "u_max": scale,
            "route_difference": float(np.max(np.abs(a["u"] - b["u"])) / scale),
            "no_slip_bottom": float(abs(a["u"][0]) / scale),
            "no_slip_top": float(abs(a["u"][-1]) / scale),
            "shear_stress_jump_relative": float(
                abs(nu1 * dud1 - nu2 * dud2) / max(abs(nu1 * dud1), 1e-30)),
        })
    # equal-viscosity limit must be the single-fluid parabola
    y = np.linspace(0.0, H, 401)
    nu = 0.21
    a = layered_poiseuille_analytic([nu, nu], [H / 2], H, g, y=y)
    exact = g * y * (H - y) / (2.0 * nu)
    checks.append({
        "nu1": nu, "nu2": nu, "h1": H / 2, "u_max": float(np.max(exact)),
        "route_difference": float(np.max(np.abs(a["u"] - exact)) / np.max(exact)),
        "no_slip_bottom": float(abs(a["u"][0]) / np.max(exact)),
        "no_slip_top": float(abs(a["u"][-1]) / np.max(exact)),
        "shear_stress_jump_relative": 0.0,
    })
    worst = max(max(c["route_difference"], c["shear_stress_jump_relative"],
                    c["no_slip_bottom"], c["no_slip_top"]) for c in checks)
    return {"arm": "L0", "checks": checks, "worst_relative_defect": worst,
            "pass": bool(worst < 1e-9)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="all",
                    help="L0 | L1 | L2 | L3 | L4 | all")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--g-body", type=float, default=2e-5)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    arms = ["L0", "L1", "L2", "L3", "L4"] if args.arm == "all" else [args.arm]

    for arm in arms:
        print(f"[{arm}] start", flush=True)
        t0 = time.time()
        if arm == "L0":
            res = arm_L0()
        else:
            sim, analytic, meta, y, h1 = build_arm(
                arm, args.ny, args.nx, args.steps, args.g_body)
            meta["arm"] = arm
            steps, hist = run_to_steady(sim, args.steps, tol=args.tol)
            res = score_arm(sim, analytic, y, h1, meta)
            res["steady_history"] = hist
            res["wall_time_s"] = time.time() - t0
        res["pass"] = bool(
            res.get("err_bulk", {}).get("err_max", res.get("worst_relative_defect", 1.0))
            < 0.15)
        path = OUTDIR / f"{arm}.json"
        path.write_text(json.dumps(res, indent=1, default=_json_default))
        summary = {k: res.get(k) for k in ("err_bulk", "err_all",
                                           "err_interface_band", "nu_error",
                                           "nu_measured", "uy_max", "pass")}
        print(f"[{arm}] {json.dumps(summary, default=_json_default)}", flush=True)
        print(f"[{arm}] wrote {path} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()