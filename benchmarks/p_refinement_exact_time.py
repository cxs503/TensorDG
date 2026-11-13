"""Exact-in-time p-refinement ladder for the smooth-solution DG case.

Case anchor: ``CASE-DG-PREFINEMENT-EXPONENTIAL``
Reference: Babuska & Suri (1994), SIAM Review 36(4), 578-632 (p/h-p versions).

Why this script exists
----------------------
The published study (``benchmarks/p_refinement_exponential.py``, p = 1..6, 16
cells, RK4 at CFL 0.08 with a *relative* temporal-refinement loop) stops at
p = 6.  The closeout of the criterion work (``docs/CLOSEOUT-m16918-...``)
declared p = 7 and p = 8 *extrapolated, not measured*, because "p = 7 takes
> 120 s".  This script measures them instead, and shows why the extrapolation
could not be tested with the production pipeline:

  With an explicit RK4 time integration the *temporal* error is independent of
  the polynomial degree (it scales as ``dt**4`` for a fixed CFL, i.e. as
  ``(CFL*h/(2p+1))**4`` only very weakly slower than the spatial part) while the
  *spatial* error falls exponentially in p.  At some degree the measured L2
  error stops decreasing in p and reflects the time integrator, not the scheme.
  The shipped temporal-convergence test is *relative* (``|e_new - e_old| /
  e_new <= 1%``), so a level can pass it and still be temporal-error dominated.

Method
------
For a linear semi-discrete system ``du/dt = L u`` the exact-in-time solution is
``exp(L t) u0``.  ``L`` is assembled column by column from the *shipped* right
hand side (:func:`tensordg.advection_1d._rhs`), so the operator is the
production one -- no re-derivation -- and the error measured against the
analytic wave is purely spatial.

Three independent measurement paths are recorded per degree:

1. ``scipy.linalg.expm`` (float64, LAPACK Pade);
2. ``tensordg.convergence.expm_scaled_squaring`` (80-bit ``np.longdouble``);
3. the production RK4 integrator at the published CFL (for comparison, and to
   quantify the temporal floor).

Predeclared gates (fixed before the high-degree numbers were seen):

* every degree p = 1..8 must be measurement-resolved: ``e > 10 * eps*||u_h||``;
* the composite discriminating criterion of the published study
  (``sigma > 1`` and non-decreasing pairwise decay rates and a >1e3 total
  reduction and monotone decrease) must hold on the *exact-time* ladder;
* the extrapolation published in ``measurement_floor.json`` must reproduce
  within a factor ``PROJECTION_FACTOR_TOL`` at p = 7 and p = 8;
* the longdouble path must be control invariant (``1e-10`` relative), and the
  float64 ``scipy`` path must agree within *its own* forward-error model wherever
  that model is smaller than a tenth of the signal (it is not at high p).

Run::

    python -m benchmarks.p_refinement_exact_time
    python -m benchmarks.p_refinement_exact_time --quick     # no CFL sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from benchmarks.p_refinement_exponential import check_discriminating_criterion
from benchmarks.p_refinement_floor import l2_error_extended, l2_error_numpy
from tensordg.advection_1d import (
    _operators,
    _rhs,
    l2_error,
    solve_advection,
)
from tensordg.convergence import (
    expm_scaled_squaring,
    modal_l2_norm_1d,
    operator_linearity_residual,
    p_decay_diagnostics,
    resolved_levels,
    roundoff_floor,
    semidiscrete_operator,
)

SCHEMA = "tensordg-p-refinement-exact-time/1"
ARTIFACT_RELPATH = Path("docs") / "benchmarks" / "results" / "p-refinement-exact-time"
PUBLISHED_RELPATH = (
    Path("docs") / "benchmarks" / "results" / "p-refinement-exponential"
)
PRIMARY = "p_refinement_results.json"
FLOOR_ARTIFACT = "measurement_floor.json"

# --- Configuration ---------------------------------------------------------
DEGREES_FULL = tuple(range(1, 9))       # 1..8: MAX_DEGREE of advection_1d is 8
DEGREES_QUICK = (5, 6, 7, 8)
NCELLS = 16
FINAL_TIME = 1.0
SPEED = 1.0
FALLBACK_CFL = 0.08

# --- Thresholds (predeclared) ---------------------------------------------
SNR_MIN = 10.0                    # a level is usable only above 10x the floor
EXPM_PATH_TOL = 1.0e-10           # relative agreement of the two expm paths
PROJECTION_FACTOR_TOL = 3.0       # published extrapolation vs measurement
MONOTONE_REQUIRED = True
TEMPORAL_SWEEP_DEGREES = (6, 7, 8)
TEMPORAL_SWEEP_CFLS = (0.08, 0.02)
CROSS_CHECK_DEGREE = 8            # top degree: RK4 with a 16x smaller CFL
CROSS_CHECK_CFL = 5.0e-3          # temporal error ~ (CFL/0.08)**4 of the CFL-0.08 one
PERTURBATION_SEEDS = (0, 1, 2)    # operator-representation sensitivity floor


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:  # pragma: no cover - git not always available
        return None


def _published(primary: dict | None, degree: int) -> dict | None:
    if not primary:
        return None
    for level in primary.get("results", {}).get("levels", []):
        if int(level.get("degree", -1)) == degree:
            return level
    return None


def _cfl_for(published_level: dict | None) -> float:
    return float(published_level["cfl"]) if published_level else FALLBACK_CFL


def assemble_operator(degree: int, ncells: int) -> tuple[np.ndarray, dict]:
    """Dense matrix of the shipped DG advection operator (periodic, upwind)."""
    h = 1.0 / ncells
    xq, wq, phi, volume, phi_l, phi_r = _operators(degree, torch.device("cpu"), torch.float64)
    shape = (ncells, degree + 1)

    def apply(vec):
        u = torch.as_tensor(
            np.asarray(vec, dtype=float).reshape(shape), dtype=torch.float64
        )
        return _rhs(u, h, SPEED, volume, phi_l, phi_r).numpy().ravel()

    ndof = ncells * (degree + 1)
    operator = semidiscrete_operator(apply, ndof)
    residual = operator_linearity_residual(apply, operator)
    info = {
        "ndof": ndof,
        "shape": list(shape),
        "linearity_residual": residual,
        "operator_norm_1": float(np.max(np.sum(np.abs(operator), axis=1))),
    }
    return operator, info


def exact_time_level(degree: int, ncells: int, published_level: dict | None) -> dict:
    """Purely spatial L2 error of the DG solution at final time."""
    operator, info = assemble_operator(degree, ncells)
    centers, coeff0, _ = solve_advection(
        ncells=ncells, degree=degree, final_time=0.0, speed=SPEED, cfl=FALLBACK_CFL,
    )
    u0 = coeff0.numpy().ravel()

    shape = (ncells, degree + 1)
    u_scipy = (expm(operator * FINAL_TIME) @ u0).reshape(shape)
    u_ld = (
        expm_scaled_squaring(operator, FINAL_TIME) @ np.asarray(u0, dtype=np.longdouble)
    ).reshape(shape)
    u_ld64 = np.asarray(u_ld, dtype=float)

    nq = 4 * degree + 16
    e_scipy_ld = l2_error_extended(centers, u_scipy, FINAL_TIME, SPEED, nq=nq)
    e_ld = l2_error_extended(centers, u_ld64, FINAL_TIME, SPEED, nq=nq)
    e_ld_hi = l2_error_extended(centers, u_ld64, FINAL_TIME, SPEED, nq=2 * nq)
    e_numpy_f64 = l2_error_numpy(centers, u_scipy, FINAL_TIME, SPEED, nq=nq)
    err = e_ld  # extended-precision path is the reference value

    # tighter expm controls: same answer only if the Taylor/scaling schedule is
    # not what limits the measurement
    u_ld_tight = (
        expm_scaled_squaring(
            operator, FINAL_TIME, taylor_terms=26, threshold=0.125,
        ) @ np.asarray(u0, dtype=np.longdouble)
    ).reshape(shape)
    e_ld_tight = l2_error_extended(centers, np.asarray(u_ld_tight, dtype=float),
                                   FINAL_TIME, SPEED, nq=nq)

    # operator-representation floor: the assembled L is float64, so its entries
    # carry a relative rounding of order eps; the response of the *solution* to
    # that rounding bounds what the float64 operator can certify at all.
    scale = float(np.max(np.sum(np.abs(operator), axis=1)))
    rng = np.random.default_rng(12345 + degree)
    perturbation_sensitivity = 0.0
    for _ in PERTURBATION_SEEDS:
        delta = rng.standard_normal(operator.shape) * (1.0e-16 * scale)
        u_pert = (
            expm_scaled_squaring(operator + delta, FINAL_TIME)
            @ np.asarray(u0, dtype=np.longdouble)
        ).reshape(shape)
        e_pert = l2_error_extended(centers, np.asarray(u_pert, dtype=float),
                                   FINAL_TIME, SPEED, nq=nq)
        perturbation_sensitivity = max(perturbation_sensitivity, abs(e_pert - e_ld))

    norm = modal_l2_norm_1d(centers, u_ld64)
    floor = roundoff_floor(norm)
    operator_floor = float(perturbation_sensitivity)

    # mass conservation of the exact-in-time solution (periodic, conservative DG).
    # The sine has zero mean, so normalise by the absolute mass, not by the mean.
    reference_mass = float(np.sum(np.abs(np.asarray(u0).reshape(shape)[:, 0])) / ncells)
    mass0 = float(np.sum(np.asarray(u0).reshape(shape)[:, 0]) / ncells)
    massT = float(np.sum(u_ld64[:, 0]) / ncells)

    # production RK4 at the CFL of the published study (or the fallback)
    cfl = _cfl_for(published_level)
    centers_rk4, coeff_rk4, meta_rk4 = solve_advection(
        ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
        cfl=cfl, time_integrator="rk4",
    )
    e_rk4 = l2_error(centers_rk4, coeff_rk4, FINAL_TIME, SPEED)

    return {
        "degree": degree,
        "ncells": ncells,
        "dofs": info["ndof"],
        "operator_linearity_residual": info["linearity_residual"],
        "operator_norm_1": info["operator_norm_1"],
        "exact_time": {
            "l2_longdouble_expm": e_ld,
            "l2_scipy_expm": e_scipy_ld,
            "l2_longdouble_expm_quadrature_2x": e_ld_hi,
            "l2_numpy_float64_quadrature": e_numpy_f64,
            "path_spread_rel": abs(e_scipy_ld - e_ld) / e_ld if e_ld else float("inf"),
            "quadrature_spread_rel": abs(e_ld_hi - e_ld) / e_ld if e_ld else float("inf"),
        },
        "exact_time_control_check": {
            "l2_longdouble_expm_tight_controls": e_ld_tight,
            "relative_difference": abs(e_ld_tight - e_ld) / e_ld if e_ld else float("inf"),
        },
        "operator_representation_floor": operator_floor,
        "solution_norm_l2": norm,
        "roundoff_floor": floor,
        "measurement_floor": max(floor, operator_floor),
        "snr_over_roundoff_floor": (err / floor) if floor else float("inf"),
        "snr_over_measurement_floor": (err / max(floor, operator_floor)) if err else float("inf"),
        "resolved": bool(err / max(floor, operator_floor) >= SNR_MIN) if err else True,
        "mass_relative_drift": (
            abs(massT - mass0) / reference_mass if reference_mass else float("inf")
        ),
        "rk4_production": {
            "cfl": cfl,
            "dt": meta_rk4["dt"],
            "steps": meta_rk4["steps"],
            "l2": e_rk4,
            "temporal_dominance_factor": e_rk4 / err if err else float("inf"),
        },
        "published_level": published_level,
    }


def temporal_sweep(degrees, cfls, ncells: int, exact_errors: dict[int, float]) -> list[dict]:
    """RK4 error vs CFL at the high degrees: is the ladder temporal limited?"""
    out = []
    for degree in degrees:
        history = []
        for cfl in cfls:
            centers, coeff, meta = solve_advection(
                ncells=ncells, degree=degree, final_time=FINAL_TIME, speed=SPEED,
                cfl=cfl, time_integrator="rk4",
            )
            err = l2_error(centers, coeff, FINAL_TIME, SPEED)
            history.append({
                "cfl": cfl,
                "steps": meta["steps"],
                "l2": err,
                "ratio_to_exact_time": err / exact_errors[degree],
            })
        # relative change of the shipped convergence test between successive CFLs
        for prev, cur in zip(history[:-1], history[1:]):
            cur["relative_change_vs_previous"] = abs(cur["l2"] - prev["l2"]) / cur["l2"]
        out.append({"degree": degree, "history": history})
    return out


def study(degrees: tuple[int, ...] = DEGREES_FULL, ncells: int = NCELLS,
          sweep: bool = True) -> dict:
    root = _root()
    published_path = root / PUBLISHED_RELPATH / PRIMARY
    floor_path = root / PUBLISHED_RELPATH / FLOOR_ARTIFACT
    primary = json.loads(published_path.read_text()) if published_path.exists() else None
    floor_artifact = json.loads(floor_path.read_text()) if floor_path.exists() else None

    levels = []
    for degree in degrees:
        levels.append(exact_time_level(degree, ncells, _published(primary, degree)))

    errors = [lv["exact_time"]["l2_longdouble_expm"] for lv in levels]
    diag = p_decay_diagnostics(degrees, errors)          # full ladder (reported)

    # A level feeds the decay-rate fit only if the measurement can resolve it:
    # above 10x of max(roundoff floor, float64-operator representation floor).
    floor = max(lv["measurement_floor"] for lv in levels)
    resolved = resolved_levels(degrees, errors, floor, snr_min=SNR_MIN)
    resolved_degrees = tuple(resolved["resolved_degrees"])
    resolved_errors = [
        e for p_, e in zip(degrees, errors) if p_ in resolved["resolved_degrees"]
    ]
    diag_resolved = (
        p_decay_diagnostics(resolved_degrees, resolved_errors)
        if len(resolved_degrees) >= 3 else None
    )
    criterion = (
        check_discriminating_criterion(diag_resolved) if diag_resolved
        else {"conditions": {}, "pass": False}
    )

    # float64 scipy expm is only a reference while its forward error (model
    # eps*||L||_1*t) stays far below the signal.
    for lv in levels:
        model = float(np.finfo(float).eps) * lv["operator_norm_1"] * FINAL_TIME
        lv["scipy_expm_forward_error_model"] = model
        lv["scipy_expm_adequate"] = bool(model < 0.1 * lv["exact_time"]["l2_longdouble_expm"])
        # a float64 reference can only be held to its own error model, not to 1e-10
        lv["scipy_expm_tolerance"] = float(
            max(EXPM_PATH_TOL, 10.0 * model / lv["exact_time"]["l2_longdouble_expm"])
        )

    # --- published extrapolation vs measurement (predeclared tolerance) -----
    projection = floor_artifact.get("projection") if floor_artifact else None
    projection_check = {"applicable": bool(projection), "levels": {}}
    if projection:
        for expected in projection["predicted"]:
            degree = int(expected)
            if degree not in degrees:
                continue
            predicted = float(projection["predicted"][expected]["predicted_error"])
            measured = errors[list(degrees).index(degree)]
            ratio = measured / predicted
            projection_check["levels"][expected] = {
                "predicted_error": predicted,
                "predicted_snr": float(projection["predicted"][expected]["predicted_snr"]),
                "measured_error": measured,
                "measured_over_predicted": ratio,
                "factor_tolerance": PROJECTION_FACTOR_TOL,
                "pass": bool(1.0 / PROJECTION_FACTOR_TOL <= ratio <= PROJECTION_FACTOR_TOL),
            }
        projection_check["all_pass"] = bool(
            all(v["pass"] for v in projection_check["levels"].values())
        ) if projection_check["levels"] else False

    # --- reproducibility of the published p = 1..6 RK4 numbers --------------
    repro = {}
    for lv in levels:
        pub = lv["published_level"]
        if pub is None:
            continue
        stored = float(pub["relative_l2"])
        measured = lv["rk4_production"]["l2"]
        repro[str(lv["degree"])] = {
            "published_l2": stored,
            "measured_l2": measured,
            "relative_difference": abs(measured - stored) / stored if stored else float("inf"),
        }
    repro_max = max((v["relative_difference"] for v in repro.values()), default=float("inf"))

    # --- temporal sweep ----------------------------------------------------
    exact_errors = {int(lv["degree"]): lv["exact_time"]["l2_longdouble_expm"] for lv in levels}
    sweep_degrees = [d for d in TEMPORAL_SWEEP_DEGREES if d in degrees]
    sweep_payload = (
        temporal_sweep(sweep_degrees, TEMPORAL_SWEEP_CFLS, ncells, exact_errors)
        if sweep and sweep_degrees else []
    )

    controls_spread = max(
        (lv["exact_time_control_check"]["relative_difference"] for lv in levels),
        default=float("inf"),
    )
    scipy_ratios = [
        lv["exact_time"]["path_spread_rel"] / lv["scipy_expm_tolerance"]
        for lv in levels if lv["scipy_expm_adequate"]
    ]
    scipy_spread = max(scipy_ratios, default=0.0)
    operator_floor = max(lv["operator_representation_floor"] for lv in levels)
    quadratic_spread = max(
        (lv["exact_time"]["quadrature_spread_rel"] for lv in levels), default=float("inf")
    )
    linearity = max(lv["operator_linearity_residual"] for lv in levels)
    mass_drift = max(lv["mass_relative_drift"] for lv in levels)

    cross_check = None
    if CROSS_CHECK_DEGREE in degrees:
        centers_x, coeff_x, meta_x = solve_advection(
            ncells=ncells, degree=CROSS_CHECK_DEGREE, final_time=FINAL_TIME,
            speed=SPEED, cfl=CROSS_CHECK_CFL, time_integrator="rk4",
        )
        e_x = l2_error(centers_x, coeff_x, FINAL_TIME, SPEED)
        e_exact = exact_errors[CROSS_CHECK_DEGREE]
        cross_check = {
            "degree": CROSS_CHECK_DEGREE,
            "cfl": CROSS_CHECK_CFL,
            "steps": meta_x["steps"],
            "rk4_l2": e_x,
            "exact_time_l2": e_exact,
            "ratio_rk4_over_exact_time": e_x / e_exact,
            "tolerance": 0.15,
            "pass": bool(abs(e_x / e_exact - 1.0) <= 0.15),
        }

    checks = {
        "all_levels_resolved_above_measurement_floor": {
            "measured": resolved["resolved_degrees"],
            "floor_limited": resolved["floor_limited_degrees"],
            "floor": floor,
            "floor_definition": "max over levels of max(eps64*||u_h||, float64-operator sensitivity)",
            "pass": CROSS_CHECK_DEGREE in resolved["resolved_degrees"]
            and resolved_degrees == tuple(range(min(degrees), max(resolved["resolved_degrees"]) + 1)),
        },
        "monotone_decreasing": {
            "measured": diag["monotone_decreasing"],
            "required": MONOTONE_REQUIRED,
            "pass": bool(diag["monotone_decreasing"]) or not MONOTONE_REQUIRED,
        },
        "discriminating_criterion": criterion,
        "expm_controls_agree": {
            "measured": controls_spread,
            "threshold": EXPM_PATH_TOL,
            "pass": bool(controls_spread <= EXPM_PATH_TOL),
        },
        "scipy_expm_agrees_where_adequate": {
            "measured": scipy_spread,
            "threshold": 1.0,
            "note": "measured = max over adequate levels of |e_scipy-e_ld|/e_ld divided by max(1e-10, 10*eps*||L||_1*t/e)",
            "adequate_degrees": [lv["degree"] for lv in levels if lv["scipy_expm_adequate"]],
            "pass": bool(scipy_spread <= EXPM_PATH_TOL),
        },
        "small_cfl_rk4_cross_check": cross_check or {"pass": True, "note": "not requested"},
        "quadrature_converged": {
            "measured": quadratic_spread,
            "threshold": 1.0e-6,
            "pass": bool(quadratic_spread <= 1.0e-6),
        },
        "operator_linear": {
            "measured": linearity,
            "threshold": 1.0e-12,
            "pass": bool(linearity <= 1.0e-12),
        },
        "mass_conserved": {
            "measured": mass_drift,
            "threshold": 1.0e-12,
            "pass": bool(mass_drift <= 1.0e-12),
        },
        "published_rk4_reproduced": {
            "max_relative_difference": repro_max,
            "threshold": 1.0e-6,
            "pass": bool(repro_max <= 1.0e-6),
        },
        "projection_test": projection_check,
    }
    gating = (
        "all_levels_resolved_above_measurement_floor", "monotone_decreasing",
        "expm_controls_agree", "scipy_expm_agrees_where_adequate",
        "quadrature_converged", "operator_linear", "mass_conserved",
        "published_rk4_reproduced", "small_cfl_rk4_cross_check",
    )
    overall = all(checks[key]["pass"] for key in gating) and bool(
        criterion.get("pass")
    )
    if projection_check["applicable"]:
        overall = overall and projection_check["all_pass"]

    return {
        "schema": SCHEMA,
        "case": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "question": (
            "Does the exponential p-convergence law of the smooth-solution case "
            "continue at p = 7 and p = 8, and does the published extrapolation of "
            "measurement_floor.json reproduce when the time integrator is removed?"
        ),
        "config": {
            "equation": "u_t + u_x = 0",
            "domain": "[0, 1] periodic",
            "initial_condition": "sin(2*pi*x)",
            "final_time": FINAL_TIME,
            "speed": SPEED,
            "ncells": ncells,
            "degrees": list(degrees),
            "time_integration": "exact (matrix exponential of the semi-discrete operator)",
            "snr_min": SNR_MIN,
            "projection_factor_tolerance": PROJECTION_FACTOR_TOL,
            "temporal_sweep": {
                "degrees": sweep_degrees,
                "cfls": list(TEMPORAL_SWEEP_CFLS),
            },
        },
        "levels": levels,
        "errors": errors,
        "diagnostics": diag,
        "diagnostics_resolved_ladder": diag_resolved,
        "operator_representation_floor_max": operator_floor,
        "floor": {
            "value": floor,
            "definition": "max over levels of eps64 * ||u_h||_L2",
            "snr_min": SNR_MIN,
            "resolved_degrees": resolved["resolved_degrees"],
            "floor_limited_degrees": resolved["floor_limited_degrees"],
            "max_resolvable_reduction": resolved["max_resolvable_reduction"],
        },
        "projection": projection,
        "projection_check": projection_check,
        "published_reproduction": repro,
        "temporal_sweep": sweep_payload,
        "checks": checks,
        "verdict": "PASS" if overall else "FAIL",
        "overall_pass": bool(overall),
        "artifact_provenance": {
            "published_artifact": str(PUBLISHED_RELPATH / PRIMARY),
            "published_artifact_sha256": _sha256(published_path),
            "floor_artifact": str(PUBLISHED_RELPATH / FLOOR_ARTIFACT),
            "floor_artifact_sha256": _sha256(floor_path),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git_commit": _git_commit(),
        },
    }


def generate_figures(payload: dict, outdir: Path) -> dict:
    """Semi-log error vs p: measured exact-time ladder, RK4 ladder, projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = np.asarray(payload["config"]["degrees"], dtype=float)
    exact = np.asarray(payload["errors"], dtype=float)
    rk4 = np.asarray([lv["rk4_production"]["l2"] for lv in payload["levels"]], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(degrees, exact, "o-", label="exact-in-time (longdouble expm)", lw=1.8)
    cfls = sorted({lv["rk4_production"]["cfl"] for lv in payload["levels"]})
    cfl_label = ", ".join(f"{c:.2f}" for c in cfls)
    ax.semilogy(degrees, rk4, "s--", label=f"RK4 production (CFL {cfl_label})")
    if payload["projection"]:
        proj = payload["projection"]
        anchor = float(proj["anchor_degree"])
        sigma = float(proj["sigma"])
        pred_deg = [anchor] + [float(k) for k in proj["predicted"]]
        pred_val = [float(proj["anchor_error"])] + [
            float(proj["predicted"][k]["predicted_error"]) for k in proj["predicted"]
        ]
        ax.semilogy(pred_deg, pred_val, "k:", label=f"published extrapolation (sigma={sigma:.3f})")
    ax.axhline(payload["floor"]["value"], color="gray", ls="-.", lw=1.0,
               label=r"roundoff floor $\epsilon_{64}\|u_h\|$")
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel(r"relative $L_2$ error")
    ax.set_title("p-refinement, smooth solution: temporal error masks p>=7")
    ax.set_xticks(degrees)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = "p_refinement_exact_time.png"
    fig.savefig(outdir / name, dpi=140)
    plt.close(fig)
    return {name: f"relative L2 error vs p ({name})"}


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES_FULL))
    ap.add_argument("--ncells", type=int, default=NCELLS)
    ap.add_argument("--quick", action="store_true", help="skip the CFL sweep")
    ap.add_argument("--out", type=str, default=str(ARTIFACT_RELPATH / "exact_time_results.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    degrees = tuple(sorted(set(args.degrees)))
    t0 = time.time()
    payload = study(degrees, ncells=args.ncells, sweep=not args.quick)

    out = Path(args.out)
    if not out.is_absolute():
        out = _root() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["runtime_seconds"] = time.time() - t0
    payload["figures"] = generate_figures(payload, out.parent)
    out.write_text(json.dumps(payload, indent=2))

    print(f"{'p':>3} {'exact L2':>14} {'RK4 L2':>14} {'RK4/exact':>10} "
          f"{'SNR':>10} {'resolved':>8}")
    for lv in payload["levels"]:
        print(f"{lv['degree']:>3} {lv['exact_time']['l2_longdouble_expm']:>14.4e} "
              f"{lv['rk4_production']['l2']:>14.4e} "
              f"{lv['rk4_production']['temporal_dominance_factor']:>10.2f} "
              f"{lv['snr_over_roundoff_floor']:>10.1f} {lv['resolved']!s:>8}")
    print(f"sigma (p=1..{max(degrees)}) = {payload['diagnostics']['lsq_decay_rate']:.4f}  "
          f"verdict = {payload['verdict']}")
    if payload["projection_check"]["levels"]:
        for key, val in payload["projection_check"]["levels"].items():
            print(f"  projection p={key}: measured/predicted = "
                  f"{val['measured_over_predicted']:.3f}  pass = {val['pass']}")
    print(f"written: {out}")
    return payload


if __name__ == "__main__":
    main()