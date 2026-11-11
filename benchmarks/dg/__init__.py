"""Unified verification surface for the advection--diffusion and dispersive modules.

Campaign node ``P165W3-N05``.  This package is the **single callable surface** of
the two production modules introduced for the node:

* :mod:`tensordg.advection_diffusion` -- strong-form DG advection (upwind/
  Rusanov flux) with optional SIPG diffusion, arbitrally space/time dependent
  divergence-free velocity fields;
* :mod:`tensordg.dispersive` -- linear dispersive-wave operators (Airy and the
  Boussinesq / [2/2], [6/6] Pade reductions of ``tanh(kh)/(kh)``) plus a
  pseudo-spectral KdV solver for ``sech^2`` solitary waves.

Criterion table (declared *before* the runs; see ``CRITERIA`` below for the
machine-readable form)
=====================================================================  ========
id  physics                                                            limit
=====================================================================  ========
E1  2-D solid-body rotation (``omega = 2*pi``, one revolution),        5 %
    cosine bell, relative L2 error against the analytic rotated field
E2  LeVeque deformation (cosine hill), one period, relative L2 error   5 %
    against the initial field (exact recovery)
E3  dispersive operator vs ``omega^2 = g k tanh(kh)`` for k in [0.1,5]  3 %
E4  KdV ``sech^2`` solitary wave propagated 10 soliton widths,         5 %
    relative L2 error against the exact travelling solution
E5  total-mass change of the advection(-diffusion) solver               1 %
=====================================================================  ========

Why a registry instead of a one-off script
------------------------------------------
The criteria are consumed by three callers that must not drift apart: the
standalone driver (:mod:`benchmarks.dg.unified_verification`, which writes the
report artifacts), the package entry point (:mod:`benchmarks.dg.__main__`, used
by CI) and the pytest suite (:mod:`tests.test_dg_unified_verification`).  All
three call :func:`run_criterion` / :func:`run_surface`, so the acceptance
numbers quoted in prose, in JSON and in tests are produced by one code path.

``mode="quick"`` shrinks the meshes for CI while keeping every criterion on the
same acceptance line; the declared (report) meshes are ``mode="declared"``.
"""
from __future__ import annotations

import math
import time

import torch

from tensordg import advection_diffusion as ad
from tensordg import dispersive as ds

#: ``k h`` samples used by the dispersion criterion (span fixed before the runs).
KH_SAMPLES = (0.1, 0.5, 1.0, 2.0, 3.5, 5.0)

#: acceptance limits -- identical to ``unified_verification.THRESHOLDS``; kept here
#: so that importing the registry does not import the 500-line driver module.
TOLERANCES = {"E1": 0.05, "E2": 0.05, "E3": 0.03, "E4": 0.05, "E5": 0.01}

CRITERIA = {
    "E1": {
        "physics": "2-D solid-body rotation of a cosine bell on the unit square "
                   "(omega = 2*pi, T = 1 s, one revolution)",
        "metric": "relative L2 error at T against the analytic rotated field",
        "reference": "exact solution u(x,T) = u0(R_{-omega T}(x - xc) + xc), "
                     "R the rotation about the square centre",
        "module": "tensordg.advection_diffusion",
        "callable": "run_solid_body_rotation",
        "declared": dict(ncells=24, degree=2, cfl=0.7),
        "quick": dict(ncells=20, degree=2, cfl=0.7),
    },
    "E2": {
        "physics": "LeVeque deformational flow (cosine hill) on the unit square, "
                   "flow cos(pi t / T) (sin^2(pi x) sin(2 pi y), -sin^2(pi y) sin(2 pi x)), "
                   "one period T = 1",
        "metric": "relative L2 error at T against the initial field (exact recovery)",
        "reference": "the exact solution at T equals the initial field because "
                     "int_0^T cos(pi t/T) dt = 0",
        "module": "tensordg.advection_diffusion",
        "callable": "run_deformational_flow",
        "declared": dict(ncells=24, degree=2, cfl=0.7),
        "quick": dict(ncells=20, degree=2, cfl=0.7),
    },
    "E3": {
        "physics": "linear dispersive water-wave operator: Airy kernel and its "
                   "Boussinesq / Pade reductions",
        "metric": "max relative error of omega (and of the time-marched phase) "
                  "against omega^2 = g k tanh(kh), k h in [0.1, 5]",
        "reference": "omega_airy(k) = sqrt(g k tanh(kh)), h = 1 m, g = 9.80665",
        "module": "tensordg.dispersive",
        "callable": "measure_dispersion / model_dispersion_error",
        "declared": dict(ncells=256, periods=10.0, depth=1.0),
        "quick": dict(ncells=128, periods=10.0, depth=1.0),
    },
    "E4": {
        "physics": "KdV solitary wave eta = a sech^2((x - ct)/w) propagated over "
                   "10 soliton widths in a periodic domain",
        "metric": "relative L2 error against the exact travelling solution; "
                  "measured celerity and peak retention reported alongside",
        "reference": "exact KdV soliton, c = sqrt(g h) (1 + a/(2h)), "
                     "w = sqrt(4 h^3/(3a)); measured speed also compared with "
                     "the shallow-water form sqrt(g (h + a))",
        "module": "tensordg.dispersive",
        "callable": "run_soliton_benchmark",
        "declared": dict(ncells=256, ampl=0.05, widths=10.0),
        "quick": dict(ncells=128, ampl=0.05, widths=10.0),
    },
    "E5": {
        "physics": "mass conservation of the strong-form DG advection operator "
                   "(single-valued upwind face flux, transparent boundary)",
        "metric": "max relative change of the total mass over the E1 and E2 runs",
        "reference": "0 (mass of a compactly supported field away from the "
                     "boundary is invariant)",
        "module": "tensordg.advection_diffusion",
        "callable": "solve",
        "declared": {"source": ["E1", "E2"]},
        "quick": {"source": ["E1", "E2"]},
    },
}


# --------------------------------------------------------------------------- #
# individual criteria
# --------------------------------------------------------------------------- #
def _rotation(cfg: dict) -> dict:
    metrics = ad.run_solid_body_rotation(**cfg)
    return {"metrics": metrics, "got": metrics["relative_l2"],
            "mass_drift": metrics["mass_drift"]}


def _deformation(cfg: dict) -> dict:
    metrics = ad.run_deformational_flow(snapshot_half=False, **cfg)
    return {"metrics": metrics, "got": metrics["relative_l2"],
            "mass_drift": metrics["mass_drift"]}


def _dispersion(cfg: dict) -> dict:
    measured = ds.measure_dispersion(list(KH_SAMPLES), depth=cfg["depth"],
                                     model="airy", periods=cfg["periods"],
                                     ncells=cfg["ncells"])
    ks = torch.linspace(0.1, 5.0, 200, dtype=torch.float64)
    table = {model: float(ds.model_dispersion_error(ks, cfg["depth"], ds.GRAVITY,
                                                   model)["max_relative_error_omega"])
             for model in ds.DISPERSION_MODELS}
    measured_max = max(r["relative_error_vs_airy"] for r in measured)
    # E3 is declared over the whole band [0.1, 5]; the operator that must meet it
    # is the one that is *physically valid* there (Airy kernel, [6/6] Pade).  The
    # low-kh models are reported, not silently dropped -- see ``model_table``.
    got = max(measured_max, table["pade66"])
    return {"got": got, "measured_airy": measured,
            "measured_max_airy": measured_max, "model_table": table}


def _soliton(cfg: dict) -> dict:
    r = ds.run_soliton_benchmark(amplitude=cfg["ampl"], depth=1.0,
                                 widths=cfg["widths"], ncells=cfg["ncells"])
    return {"got": r["relative_l2"], "metrics": r}


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _record(cid: str, got: float, detail: dict, tolerance: float) -> dict:
    return {
        "id": cid,
        "physics": CRITERIA[cid]["physics"],
        "metric": CRITERIA[cid]["metric"],
        "reference": CRITERIA[cid]["reference"],
        "tolerance": tolerance,
        "got": float(got),
        "pass": bool(got <= tolerance),
        "detail": detail,
    }


def run_criterion(name: str, mode: str = "declared") -> dict:
    """Evaluate one criterion with the production modules.

    ``mode`` is ``"declared"`` (report meshes) or ``"quick"`` (CI meshes).
    """
    if name not in CRITERIA:
        raise KeyError(f"unknown criterion {name!r}; expected one of {sorted(CRITERIA)}")
    if mode not in ("declared", "quick"):
        raise ValueError("mode must be 'declared' or 'quick'")
    spec = CRITERIA[name]
    cfg = dict(spec[mode])
    tol = TOLERANCES[name]

    if name == "E1":
        r = _rotation(cfg)
        return _record(name, r["got"], {"config": cfg, "metrics": r["metrics"]}, tol)
    if name == "E2":
        r = _deformation(cfg)
        return _record(name, r["got"], {"config": cfg, "metrics": r["metrics"]}, tol)
    if name == "E3":
        r = _dispersion(cfg)
        return _record(name, r["got"], {"config": cfg,
                                        "measured_airy": r["measured_airy"],
                                        "model_table_omega": r["model_table"]}, tol)
    if name == "E4":
        r = _soliton(cfg)
        m = r["metrics"]
        return _record(name, r["got"], {
            "config": cfg,
            "celerity_kdv": m["speed"],
            "celerity_sqrt_g_h_plus_a": m["speed_sqrt_g_h_plus_a"],
            "celerity_measured": m["speed_measured"],
            "relative_error_celerity_vs_sqrt": m["relative_error_speed_measured_vs_sqrt"],
            "peak_retention": m["peak_retention"],
            "mass_drift": m["mass_drift"],
            "dx": m["dx"], "dt": m["dt"], "steps": m["steps"],
        }, tol)
    # E5 is derived from the E1/E2 runs: mass conservation of the same operator.
    drift_rot = _rotation(dict(CRITERIA["E1"][mode]))["mass_drift"]
    drift_def = _deformation(dict(CRITERIA["E2"][mode]))["mass_drift"]
    return _record(name, max(drift_rot, drift_def),
                   {"config": {"source": ["E1", "E2"]},
                    "mass_drift_rotation": drift_rot,
                    "mass_drift_deformation": drift_def}, tol)


def run_surface(mode: str = "declared") -> dict:
    """Evaluate E1--E5 and return ``{id: record}`` plus a wall-clock stamp.

    The conservation criterion E5 is derived from the *same* E1/E2 runs (mass
    drift of the identical operator invocation), so the expensive advection runs
    are not repeated.
    """
    started = time.time()
    records = {}
    for cid in ("E1", "E2", "E3", "E4"):
        records[cid] = run_criterion(cid, mode=mode)
    drift_rot = records["E1"]["detail"]["metrics"]["mass_drift"]
    drift_def = records["E2"]["detail"]["metrics"]["mass_drift"]
    records["E5"] = _record("E5", max(drift_rot, drift_def),
                            {"config": {"source": ["E1", "E2"]},
                             "mass_drift_rotation": drift_rot,
                             "mass_drift_deformation": drift_def},
                            TOLERANCES["E5"])
    return {"mode": mode, "criteria": records,
            "verdict": "PASS" if all(r["pass"] for r in records.values()) else "FAIL",
            "wall_seconds": time.time() - started}


def benchmark_table(records: dict) -> dict:
    """Machine-readable ref-vs-got table (the ``benchmark`` evidence key)."""
    table = {}
    for cid, rec in records.items():
        row = {"reference": rec["reference"], "got": rec["got"],
               "error": rec["got"], "tolerance": rec["tolerance"],
               "pass": rec["pass"], "metric": rec["metric"]}
        detail = rec.get("detail", {})
        if cid == "E3":
            row["reference_values"] = [
                {"k": m["kh"], "omega_airy": m["omega_airy"],
                 "omega_measured": m["omega_measured"],
                 "relative_error": m["relative_error_vs_airy"]}
                for m in detail.get("measured_airy", [])]
            row["model_max_relative_error_omega"] = detail.get("model_table_omega", {})
        if cid == "E4" and detail:
            row["celerity"] = {"reference_sqrt_g_h_plus_a":
                               detail["celerity_sqrt_g_h_plus_a"],
                               "reference_kdv": detail["celerity_kdv"],
                               "measured": detail["celerity_measured"],
                               "relative_error_vs_sqrt":
                               detail["relative_error_celerity_vs_sqrt"]}
            row["peak_retention"] = detail["peak_retention"]
        if cid == "E5" and detail:
            row["mass_drift_rotation"] = detail["mass_drift_rotation"]
            row["mass_drift_deformation"] = detail["mass_drift_deformation"]
        table[cid] = row
    return table


__all__ = ["CRITERIA", "TOLERANCES", "KH_SAMPLES", "run_criterion", "run_surface",
           "benchmark_table"]