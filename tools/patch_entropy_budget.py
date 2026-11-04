"""One-off patch script: extend the entropy-budget study (kept for provenance).

It applies four source edits that the commit history explains:
  * ``benchmarks/_entropy_tools.py`` -- gap-normalised entropy-profile metric;
  * ``benchmarks/entropy_budget_1d.py`` -- finest-mesh selector fix, ``limiter_stage``
    coverage, convergence order of the budget error, new profile metric.
Run:  python3 tools/patch_entropy_budget.py   (idempotence is asserted)
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _replace(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already applied: {path.name}")
        return
    if old not in text:
        raise SystemExit(f"anchor not found in {path.name}:\n{old[:200]}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path.name}")


# --------------------------------------------------------------------------- #
# 1. entropy-profile metric in the tools module
# --------------------------------------------------------------------------- #
TOOLS = ROOT / "benchmarks/_entropy_tools.py"
_OLD_METRIC = '''def pointwise_entropy_error(numerical, exact, x) -> float:
    """Relative ``L2`` error of the entropy density profile ``rho s``."""
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), 1e-300))
'''

_NEW_METRIC = '''def pointwise_entropy_error(numerical, exact, x) -> float:
    """Relative ``L2`` error of the entropy density profile ``rho s``.

    Deprecated for reporting: ``rho s = rho ln(p / rho**gamma)`` is *signed* and
    crosses zero inside shock-tube data, so ``||e|| / ||exact||`` divides an
    ``O(1)`` absolute error by a nearly cancelling norm.  On the Sod case it
    returns ``163068 %`` and carries no information about the solver.  Use
    :func:`entropy_profile_error` instead, which normalises by the intrinsic
    entropy-jump scale of the shock.
    """
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), 1e-300))


def entropy_jump_scale(upstream, downstream, gamma: float = GAMMA) -> float:
    """``|[s]| = |s_downstream - s_upstream|`` of a single shock.

    This is the intrinsic scale of the ``rho s`` profile: the entropy plateau
    jump the profile has to reproduce, and the only sensible denominator for a
    profile error.
    """
    rho_a, _, p_a = (float(value) for value in upstream)
    rho_b, _, p_b = (float(value) for value in downstream)
    return abs(float(gas_entropy(rho_b, p_b, gamma)) - float(gas_entropy(rho_a, p_a, gamma)))


def entropy_profile_error(numerical, exact, x, entropy_scale: float | None = None) -> dict:
    """Profile error of the signed entropy density, in interpretable units.

    Parameters
    ----------
    numerical, exact : array_like
        Pointwise ``rho s`` on the sample abscissae ``x``.
    entropy_scale : float, optional
        ``|[s]|`` of the shock (:func:`entropy_jump_scale`).  When given, the
        RMS error is also expressed as a fraction of the entropy jump.

    Returns
    -------
    dict with

    ``rms``
        ``||num - exact||_2 / sqrt(L)`` -- an RMS error in entropy-density units
        (units of ``rho s``, i.e. mass per volume; ``s`` itself is dimensionless).
    ``rms_over_entropy_jump``
        ``rms / |[s]|`` when ``entropy_scale`` is supplied: "how large is the
        profile error compared with the entropy step the shock has to produce".
    ``rms_over_mean_abs``
        ``rms / mean(|exact|)`` -- a scale-free fallback that does not rely on
        the shock data.
    ``legacy_relative_l2``
        The deprecated ``||e|| / ||exact||`` for provenance only.
    """
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)
    x = np.asarray(x, dtype=float)
    length = max(float(x[-1] - x[0]), 1e-300)
    residual = numerical - exact
    rms = float(np.linalg.norm(residual) / np.sqrt(length))
    out = {
        "rms": rms,
        "rms_over_mean_abs": rms / max(float(np.mean(np.abs(exact))), 1e-300),
        "legacy_relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(exact), 1e-300)),
    }
    if entropy_scale is not None:
        out["rms_over_entropy_jump"] = rms / max(float(entropy_scale), 1e-300)
    return out
'''

_replace(TOOLS, _OLD_METRIC, _NEW_METRIC)

# --------------------------------------------------------------------------- #
# 2. study driver: finest-mesh selector, limiter stages, budget EOC
# --------------------------------------------------------------------------- #
STUDY = ROOT / "benchmarks/entropy_budget_1d.py"

_OLD_IMPORT = """from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density,
    legendre_modes,"""
_NEW_IMPORT = """from benchmarks._entropy_tools import (
    analytic_right_shock_budget,
    entropy_density,
    entropy_jump_scale,
    entropy_profile_error,
    legendre_modes,"""
_replace(STUDY, _OLD_IMPORT, _NEW_IMPORT)

_OLD_STUDY = '''def shock_study(cells, limiter_settings, degree=2, cfl=0.3):
    """Run the three criteria on Sod and Lax and on the smooth periodic wave."""
    sod = _sod_shock_budget(SOD_TIME)
    lax = _lax_shock_budget(LAX_TIME)
    scenarios = [
        ("sod", LEFT, RIGHT, SOD_TIME, sod),
        ("lax", LAX_LEFT, LAX_RIGHT, LAX_TIME, lax),
    ]
    results = {}
    for name, left, right, final_time, budget in scenarios:
        rows = []
        for limiter, tvb_constant in limiter_settings:
            for ncells in cells:
                run = _entropy_of_run(left, right, ncells, degree, final_time,
                                      limiter, tvb_constant, "every", cfl=cfl)
                produced = run["final"] - run["initial"]
                rows.append({
                    "limiter": limiter or "none",
                    "tvb_constant": tvb_constant,
                    "cells": ncells,
                    "degree": degree,
                    "steps": run["steps"],
                    "dt": run["dt"],
                    "entropy_initial": run["initial"],
                    "entropy_final": run["final"],
                    "production_numeric": produced,
                    "production_analytic": budget["production_total"],
                    "relative_error": relative_error(budget["production_total"], produced),
                    "second_law": bool(produced > 0.0),
                })
'''

_NEW_STUDY = '''def finest_row(rows):
    """The finest-mesh row (largest ``cells``) of a study subset.

    Historical defect, fixed here: the "relative error at the finest mesh"
    verdict used ``min(rows, key=cells)``, i.e. it graded the **coarsest** mesh.
    That under-reported the gate (Sod / no limiter was scored 11.651 % at 64
    cells while the 256-cell value was 3.126 %), so the honest-FAIL verdict in
    the first version of `docs/entropy_verification.md` was partly a labelling
    bug, not a solver result.
    """
    return max(rows, key=lambda row: row["cells"])


def shock_study(cells, limiter_settings, degree=2, cfl=0.3, limiter_stages=("every",)):
    """Run the three criteria on Sod and Lax and on the smooth periodic wave."""
    sod = _sod_shock_budget(SOD_TIME)
    lax = _lax_shock_budget(LAX_TIME)
    scenarios = [
        ("sod", LEFT, RIGHT, SOD_TIME, sod),
        ("lax", LAX_LEFT, LAX_RIGHT, LAX_TIME, lax),
    ]
    results = {}
    for name, left, right, final_time, budget in scenarios:
        rows = []
        for limiter, tvb_constant in limiter_settings:
            # ``limiter_stage`` is inert without a limiter, so do not spend runs on it.
            stages = tuple(limiter_stages) if limiter else ("every",)
            for limiter_stage in stages:
                for ncells in cells:
                    run = _entropy_of_run(left, right, ncells, degree, final_time,
                                          limiter, tvb_constant, limiter_stage, cfl=cfl)
                    produced = run["final"] - run["initial"]
                    rows.append({
                        "limiter": limiter or "none",
                        "tvb_constant": tvb_constant,
                        "limiter_stage": limiter_stage,
                        "label": (limiter or "none")
                        + ("" if limiter_stage == "every" else f"@{limiter_stage}"),
                        "cells": ncells,
                        "degree": degree,
                        "steps": run["steps"],
                        "dt": run["dt"],
                        "entropy_initial": run["initial"],
                        "entropy_final": run["final"],
                        "production_numeric": produced,
                        "production_analytic": budget["production_total"],
                        "relative_error": relative_error(budget["production_total"], produced),
                        "second_law": bool(produced > 0.0),
                    })
'''
_replace(STUDY, _OLD_STUDY, _NEW_STUDY)

# CLI: limiter stages
_OLD_ARG = '''    parser.add_argument("--limiters", nargs="+", default=["none", "tvb"],
                        choices=["none", "tvb", "weno"])'''
_NEW_ARG = '''    parser.add_argument("--limiters", nargs="+", default=["none", "tvb"],
                        choices=["none", "tvb", "weno"])
    parser.add_argument("--limiter-stage", nargs="+", default=["every"],
                        choices=["every", "final"],
                        help="SSP-RK3 stage(s) at which the limiter acts (ignored without a limiter)")'''
_replace(STUDY, _OLD_ARG, _NEW_ARG)

_OLD_CALL = '''    shock_results = shock_study(args.cells, limiter_settings, degree=args.degree, cfl=args.cfl)
    lax_results = shock_study(args.lax_cells, limiter_settings, degree=args.degree, cfl=args.cfl)'''
_NEW_CALL = '''    shock_results = shock_study(args.cells, limiter_settings, degree=args.degree,
                                cfl=args.cfl, limiter_stages=args.limiter_stage)
    lax_results = shock_study(args.lax_cells, limiter_settings, degree=args.degree,
                              cfl=args.cfl, limiter_stages=args.limiter_stage)'''
_replace(STUDY, _OLD_CALL, _NEW_CALL)

# profile metric + verdicts
_OLD_PROFILE = '''        "profile_comparison": {
            "cells": profile_ncells,
            "relative_l2_entropy_density": {
                key: float(np.linalg.norm(profile[key][1] - profile[key][2])
                           / np.linalg.norm(profile[key][2]))
                for key in ("none", "tvb", "weno") if key in profile
            },
        },
    }

    finest, second_law = {}, all(
        row["second_law"] for case in shock_results.values() for row in case["rows"])
    for name, case in shock_results.items():
        for limiter in {row["limiter"] for row in case["rows"]}:
            subset = [row for row in case["rows"] if row["limiter"] == limiter]
            finest[f"{name}/{limiter}"] = min(subset, key=lambda row: row["cells"])["relative_error"]
'''
_NEW_PROFILE = '''        "profile_comparison": {
            "cells": profile_ncells,
            "entropy_jump_scale": entropy_jump_scale(
                _sod_shock_budget(SOD_TIME)["upstream"],
                _sod_shock_budget(SOD_TIME)["downstream"]),
            "metrics": {
                key: entropy_profile_error(profile[key][1], profile[key][2], profile[key][0],
                                           entropy_jump_scale(
                                               _sod_shock_budget(SOD_TIME)["upstream"],
                                               _sod_shock_budget(SOD_TIME)["downstream"]))
                for key in ("none", "tvb", "weno") if key in profile
            },
            "legacy_relative_l2_entropy_density": {
                key: float(np.linalg.norm(profile[key][1] - profile[key][2])
                           / np.linalg.norm(profile[key][2]))
                for key in ("none", "tvb", "weno") if key in profile
            },
        },
    }

    finest, coarsest, budget_orders, second_law = {}, {}, {}, all(
        row["second_law"] for case in shock_results.values() for row in case["rows"])
    for name, case in shock_results.items():
        for label in {row["label"] for row in case["rows"]}:
            subset = sorted([row for row in case["rows"] if row["label"] == label],
                            key=lambda row: row["cells"])
            finest[f"{name}/{label}"] = finest_row(subset)["relative_error"]
            coarsest[f"{name}/{label}"] = subset[0]["relative_error"]
            budget_orders[f"{name}/{label}"] = observed_order(
                [row["cells"] for row in subset],
                [row["relative_error"] for row in subset])
'''
_replace(STUDY, _OLD_PROFILE, _NEW_PROFILE)

_OLD_VERDICT = '''        "C2_relative_error_at_finest_mesh": finest,
        "C2_below_3_percent": {key: bool(value < 0.03) for key, value in finest.items()},'''
_NEW_VERDICT = '''        "C2_relative_error_at_finest_mesh": finest,
        "C2_relative_error_at_coarsest_mesh": coarsest,
        "C2_error_observed_orders": budget_orders,
        "C2_below_3_percent": {key: bool(value < 0.03) for key, value in finest.items()},'''
_replace(STUDY, _OLD_VERDICT, _NEW_VERDICT)

print("done")
sys.exit(0)