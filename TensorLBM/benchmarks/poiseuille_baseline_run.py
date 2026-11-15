#!/usr/bin/env python3
"""Single-phase plane-Poiseuille baseline study (XVAL skeleton).

Anchor: ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO``
Reference pattern: ``lettucecfd/lettuce`` ``ext/_flows/poiseuille.py`` +
``ext/_reporter/error_reporter.py`` (analytic profile + report gate).

Sections
--------
A  delivery replica        legacy config of ``benchmarks/poiseuille.py``
                           (Nx=64, Ny=32, tau=0.8, F=1e-5, simple force term,
                           node-centred reference, 2000 fixed steps)
B  ablation                {simple, guo} x {node, halfway} at convergence:
                           separates the two defects of section A
C  step ladder             fixed-step runs -> error is not only discretisation
D  width ladder            Ny = 8/16/32/64 -> error vs L against |48 nu^2-1|/L^2
E  viscosity ladder        tau = 0.6 ... 1.5 -> error vs nu against the same law
F  slip-free relaxation    tau* = 1/2 + 3/sqrt(48) predicted round-off agreement
G  force independence      F = 1e-6/1e-5/1e-4 -> relative error invariant in F
H  invariants              mass conservation, initial-condition independence

Every section is appended to ``outputs/xval_poiseuille_baseline/summary.json``
as soon as it is done, so a partial run still yields usable evidence.
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.poiseuille_baseline import (  # noqa: E402
    BASELINE_SPEC, PoiseuilleConfig, ReL2Gate, run,
)

OUT = ROOT / "outputs" / "xval_poiseuille_baseline"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "summary.json"

TAU_STAR = 0.5 + 3.0 / math.sqrt(48.0)          # 48 nu^2 = 1  ->  nu = 1/sqrt(48)


def slip_law(tau: float, width: float) -> float:
    """Signed relative slip predicted by the closed form |(48 nu^2 - 1)| / L^2."""
    nu = (tau - 0.5) / 3.0
    return (48.0 * nu * nu - 1.0) / width ** 2


def _slim(res: dict) -> dict:
    """Drop the full profiles (kept separately for the canonical runs)."""
    return {
        "case": res["case"],
        "convergence": {k: v for k, v in res["convergence"].items()
                        if k != "residual_history"},
        "errors": res["errors"],
        "mass_drift": res["mass_drift"],
    }


def _merge(section: str, payload) -> None:
    data = {}
    if SUMMARY.exists():
        data = json.loads(SUMMARY.read_text())
    data[section] = payload
    data["_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    SUMMARY.write_text(json.dumps(data, indent=1))


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, round(time.time() - t0, 1)


# ---------------------------------------------------------------- section A
def section_a() -> dict:
    cfg = PoiseuilleConfig(Nx=64, Ny=32, tau=0.8, force=1e-5, forcing="simple",
                           wall_origin=0.0, init="rest", max_steps=2000,
                           check_every=2000, residual_tol=0.0)
    res, wall = timed(run, cfg)
    return {
        "description": "delivery replica: legacy force term + node-centred "
                       "analytic reference + 2000 fixed steps",
        "result": _slim(res),
        "wall_seconds": wall,
        "note": "`residual_tol=0` disables early stopping, so the step count is "
                "exactly the delivered 2000; nothing else is changed.",
    }


# ---------------------------------------------------------------- section B
def section_b() -> dict:
    variants = {
        "simple+node (legacy, converged)": dict(forcing="simple", wall_origin=0.0),
        "simple+halfway (converged)": dict(forcing="simple", wall_origin=0.5),
        "guo+node (converged)": dict(forcing="guo", wall_origin=0.0),
        "guo+halfway (baseline, converged)": dict(forcing="guo", wall_origin=0.5),
    }
    out = {}
    for label, kw in variants.items():
        cfg = PoiseuilleConfig(Nx=8, Ny=32, tau=0.8, init="rest",
                               residual_tol=1e-11, max_steps=200_000,
                               check_every=100, **kw)
        res, wall = timed(run, cfg)
        out[label] = _slim(res) | {"wall_seconds": wall}
    base = out["guo+halfway (baseline, converged)"]["errors"]["rel_l2"]
    legacy = out["simple+node (legacy, converged)"]["errors"]["rel_l2"]
    out["_reading"] = {
        "legacy_converged_rel_l2": legacy,
        "baseline_rel_l2": base,
        "improvement_factor": legacy / base if base else None,
        "law_prediction": slip_law(0.8, 30.0),
    }
    return out


# ---------------------------------------------------------------- section C
def section_c() -> dict:
    steps_ladder = [250, 500, 1000, 2000, 5000, 10000, 20000]
    rows = []
    for n in steps_ladder:
        cfg = PoiseuilleConfig(Nx=8, Ny=32, tau=0.8, wall_origin=0.5,
                               forcing="guo", init="rest", max_steps=n,
                               check_every=n, residual_tol=0.0)
        res, wall = timed(run, cfg)
        rows.append({"steps": n,
                     "rel_l2": res["errors"]["rel_l2"],
                     "rel_l2_shape": res["errors"]["rel_l2_shape"],
                     "wall_seconds": wall})
        print(f"  C steps={n:6d} relL2={rows[-1]['rel_l2']:.3e}", flush=True)
    return {
        "description": "fixed-step error at the canonical case (Ny=32, tau=0.8, "
                       "guo + halfway); the flat tail is the converged "
                       "discretisation error, the head is transient",
        "rows": rows,
        "slip_law_prediction": slip_law(0.8, 30.0),
    }


# ---------------------------------------------------------------- section D
def section_d() -> dict:
    rows = []
    for Ny in (8, 16, 32, 64):
        cfg = PoiseuilleConfig(Nx=4, Ny=Ny, tau=0.8, wall_origin=0.5,
                               forcing="guo", init="analytic",
                               residual_tol=1e-10, max_steps=300_000,
                               check_every=200)
        res, wall = timed(run, cfg)
        law = slip_law(0.8, cfg.width)
        rows.append({
            "Ny": Ny, "width": cfg.width, "nu": cfg.nu,
            "rel_l2": res["errors"]["rel_l2"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "offset_over_force": res["errors"]["offset_over_force"],
            "law": abs(law),
            "law_signed": law,
            "ratio_to_law": (res["errors"]["rel_l2"] / abs(law)) if law else None,
            "steps": res["convergence"]["steps"],
            "final_residual": res["convergence"]["final_residual"],
            "wall_seconds": wall,
        })
        print(f"  D Ny={Ny:3d} relL2={rows[-1]['rel_l2']:.3e} "
              f"law={abs(law):.3e}", flush=True)
    return {
        "description": "channel-width ladder at tau=0.8; error follows "
                       "|48 nu^2 - 1| / L^2",
        "rows": rows,
    }


# ---------------------------------------------------------------- section E
def section_e() -> dict:
    rows = []
    for tau in (0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5):
        cfg = PoiseuilleConfig(Nx=4, Ny=16, tau=tau, wall_origin=0.5,
                               forcing="guo", init="analytic",
                               residual_tol=1e-10, max_steps=200_000,
                               check_every=100)
        res, wall = timed(run, cfg)
        law = slip_law(tau, cfg.width)
        rows.append({
            "tau": tau, "nu": cfg.nu,
            "rel_l2": res["errors"]["rel_l2"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "offset_over_force": res["errors"]["offset_over_force"],
            "law": abs(law),
            "law_signed": law,
            "ratio_to_law": (res["errors"]["rel_l2"] / abs(law)) if law else None,
            "steps": res["convergence"]["steps"],
            "wall_seconds": wall,
        })
        print(f"  E tau={tau:4.2f} relL2={rows[-1]['rel_l2']:.3e} "
              f"law={abs(law):.3e}", flush=True)
    return {
        "description": "viscosity ladder at Ny=16 (L=14); signed offset follows "
                       "F (6 nu - 1/(8 nu)), i.e. relative |48 nu^2 - 1| / L^2",
        "rows": rows,
    }


# ---------------------------------------------------------------- section F
def section_f() -> dict:
    rows = []
    for Ny in (16, 32):
        cfg = PoiseuilleConfig(Nx=4, Ny=Ny, tau=TAU_STAR, wall_origin=0.5,
                               forcing="guo", init="analytic",
                               residual_tol=1e-12, max_steps=200_000,
                               check_every=100)
        res, wall = timed(run, cfg)
        rows.append({
            "Ny": Ny, "tau": TAU_STAR, "nu": cfg.nu,
            "rel_l2": res["errors"]["rel_l2"],
            "rel_linf": res["errors"]["rel_linf"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "steps": res["convergence"]["steps"],
            "final_residual": res["convergence"]["final_residual"],
            "wall_seconds": wall,
        })
        print(f"  F Ny={Ny:3d} tau*={TAU_STAR:.7f} relL2={rows[-1]['rel_l2']:.3e}",
              flush=True)
    return {
        "description": "tau* = 1/2 + 3/sqrt(48) cancels 48 nu^2 = 1, so the "
                       "residual error is the solver round-off level only",
        "tau_star": TAU_STAR,
        "rows": rows,
    }


# ---------------------------------------------------------------- section G
def section_g() -> dict:
    rows = []
    for F in (1e-6, 1e-5, 1e-4):
        cfg = PoiseuilleConfig(Nx=4, Ny=16, tau=1.0, force=F, wall_origin=0.5,
                               forcing="guo", init="analytic",
                               residual_tol=1e-10, max_steps=200_000,
                               check_every=100)
        res, wall = timed(run, cfg)
        rows.append({
            "force": F, "u_max_analytic": cfg.u_max_analytic,
            "mach": cfg.u_max_analytic / math.sqrt(1.0 / 3.0),
            "rel_l2": res["errors"]["rel_l2"],
            "offset_over_force": res["errors"]["offset_over_force"],
            "rel_l2_shape": res["errors"]["rel_l2_shape"],
            "wall_seconds": wall,
        })
        print(f"  G F={F:.0e} relL2={rows[-1]['rel_l2']:.3e} "
              f"off/F={rows[-1]['offset_over_force']:+.4f}", flush=True)
    vals = [r["rel_l2"] for r in rows]
    return {
        "description": "relative error and offset/force are independent of the "
                       "forcing amplitude (the residual error is O(F), not O(F^2))",
        "rows": rows,
        "spread_relative": (max(vals) - min(vals)) / min(vals),
    }


# ---------------------------------------------------------------- section H
def section_h() -> dict:
    out = {}
    cfg_rest = PoiseuilleConfig(Nx=4, Ny=16, tau=1.0, wall_origin=0.5,
                                forcing="guo", init="rest",
                                residual_tol=1e-10, max_steps=200_000,
                                check_every=100)
    cfg_ana = PoiseuilleConfig(Nx=4, Ny=16, tau=1.0, wall_origin=0.5,
                               forcing="guo", init="analytic",
                               residual_tol=1e-10, max_steps=200_000,
                               check_every=100)
    a, wa = timed(run, cfg_rest)
    b, wb = timed(run, cfg_ana)
    pa = torch.tensor(a["profile_u"])
    pb = torch.tensor(b["profile_u"])
    out["initial_condition_independence"] = {
        "rest_steps": a["convergence"]["steps"],
        "analytic_steps": b["convergence"]["steps"],
        "max_profile_diff": float((pa - pb).abs().max()),
        "u_max": a["errors"]["u_max_analytic"],
        "rel_max_profile_diff": float((pa - pb).abs().max())
        / a["errors"]["u_max_analytic"],
        "wall_seconds": [wa, wb],
    }
    out["mass_conservation"] = {
        "rest": a["mass_drift"],
        "analytic": b["mass_drift"],
        "max_abs": max(abs(a["mass_drift"]), abs(b["mass_drift"])),
        "relative": max(abs(a["mass_drift"]), abs(b["mass_drift"]))
        / (16 * 4.0),
    }
    print("  H", out, flush=True)
    return out


# ---------------------------------------------------------------- gate
def gate_section(data: dict) -> dict:
    gate = ReL2Gate(tolerance=BASELINE_SPEC["gate"]["tolerance"],
                    basis=BASELINE_SPEC["gate"]["basis"])
    checks = {}
    a = data.get("A_delivery_replica", {}).get("result", {}).get("errors", {})
    if a:
        checks["delivery_replica_2000_steps"] = gate.check(a["rel_l2"]).as_dict()
    b = data.get("B_ablation", {})
    for label in ("simple+node (legacy, converged)", "guo+halfway (baseline, converged)"):
        if label in b:
            checks[label] = gate.check(b[label]["errors"]["rel_l2"]).as_dict()
    d = data.get("D_width_ladder", {}).get("rows", [])
    for row in d:
        checks[f"width_ladder_Ny{row['Ny']}"] = gate.check(row["rel_l2"]).as_dict()
    e = data.get("E_tau_ladder", {}).get("rows", [])
    for row in e:
        checks[f"tau_ladder_tau{row['tau']}"] = gate.check(row["rel_l2"]).as_dict()
    f = data.get("F_slip_free_tau_star", {}).get("rows", [])
    for row in f:
        checks[f"tau_star_Ny{row['Ny']}"] = gate.check(row["rel_l2"]).as_dict()
    return {
        "tolerance": gate.tolerance,
        "basis": gate.basis,
        "spec": BASELINE_SPEC["gate"],
        "checks": checks,
        "n_pass": sum(1 for c in checks.values() if c["passed"]),
        "n_total": len(checks),
    }


SECTIONS = [
    ("A_delivery_replica", section_a),
    ("B_ablation", section_b),
    ("C_step_ladder", section_c),
    ("D_width_ladder", section_d),
    ("E_tau_ladder", section_e),
    ("F_slip_free_tau_star", section_f),
    ("G_force_independence", section_g),
    ("H_invariants", section_h),
]


def main(argv) -> int:
    only = set(argv[1:]) or None
    _merge("_spec", BASELINE_SPEC)
    for name, fn in SECTIONS:
        if only and name.split("_")[0] not in only:
            continue
        t0 = time.time()
        try:
            payload = fn()
        except Exception:                                   # keep going
            payload = {"error": traceback.format_exc()}
            print(f"[{name}] FAILED\n{payload['error']}", flush=True)
        payload["elapsed_seconds"] = round(time.time() - t0, 1)
        _merge(name, payload)
        print(f"[{name}] done in {payload['elapsed_seconds']}s", flush=True)
    data = json.loads(SUMMARY.read_text())
    _merge("I_gate", gate_section(data))
    print("wrote", SUMMARY, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))