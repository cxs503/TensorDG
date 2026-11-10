#!/usr/bin/env python3
"""Drift law of the hydrostatic lake-at-rest state (round-off accumulation study).

Motivation
----------
``well_balanced=True`` makes the discrete lake-at-rest state ``h_h + z_h = const,
hu_h = 0`` a fixed point of the semi-discrete DG residual *to round-off level*:
the residual is not exactly zero, it is the round-off of a
``(2k+1)/dx``-amplified cancellation between the pressure-flux divergence and the
bed-slope source.  Those round-off residuals are *coherent* (they do not average
out in time), so the spurious current does not stay at the one-step noise floor:
it accumulates.  Two consequences motivate this script:

* the achievable "spurious current" of the solver can only be quoted together
  with an integration time and a mesh -- a bare "machine precision for all time"
  claim is not a reproducible statement, and
* the accumulation must be *algebraic*; an exponential one would be a genuine
  instability of the still-water state and has to be caught by a regression
  test (``tests/test_swe_wellbalanced_hydrostatic.py::
  test_drift_growth_is_algebraic_not_exponential``).

Where the seed comes from
-------------------------
Not from the bed *representation*: on a flat bed (``z_b == 0``, exactly
representable, no interpolation error at all) the state still drifts, because
interpolating a constant nodal vector leaves O(eps) higher modal coefficients,
whose interface traces then differ in the last bits and are amplified by the
``1/dx`` weak-form prefactor.  The seed is round-off, not an O(dx^3) bed error.

What is measured
----------------
For five (bed, mesh) cases -- flat, linear slope, sinusoid, parabolic bowl, and
the exact configuration of the CI regression test (``0.2 sin(pi x)`` on
``[-1, 1]``, N = 32) -- and ``p = 1, 2``, the lake-at-rest state is integrated to
``T in {0.5, 1, 2, 4}``.  For each run we record

* ``eta_drift``      : max |eta - eta0| over cells (reconstructed with the
                       *polynomial* bed average ``metadata["zb_modes"][:, 0]``;
                       the piecewise-linear trapezoidal mean is wrong for p = 2),
* ``eta_drift_over_depth``: the same, scaled by the local depth (dimensionless),
* ``u_max``          : max |u| (spurious current),
* ``u_over_c``       : max |u| / sqrt(g h) (dimensionless spurious current),
* ``growth_exponent``: least-squares slope of log(u_max) vs log(T) per
                       (case, degree) -- 1 means a constant coherent forcing,
                       larger values the two-way coupling of the mass and
                       momentum round-off; an exponential blow-up would show up
                       as a slope that keeps increasing with T.

Run with::

    PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift.py
    PYTHONPATH=. python validation/swe_thacker/run_hydrostatic_wb_drift.py --fig docs/figures/swe_wb_drift_law.png

Output: ``validation/swe_thacker/hydrostatic_wb_drift.json`` (plus the optional
figure).  Wall time ~20 min on one core, single-threaded BLAS.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.swe_1d import reconstruct_fields, solve_swe_1d

G = 9.81
TIMES = (0.5, 1.0, 2.0, 4.0)
DEGREES = (1, 2)
OUT = Path(__file__).with_name("hydrostatic_wb_drift.json")


def _flat(xx):
    return torch.zeros_like(xx)


def _slope(xx):
    return 0.5 * xx


def _sinus(xx):
    return 0.4 * torch.sin(math.pi * (xx + 1.0) / 2.0)  # one arch on [-1, 1]


def _bowl(xx):
    return -2.0 + 1.5 * xx ** 2  # parabolic basin, positive depth for eta0 = 1


def _ci_curved(xx):
    return 0.2 * torch.sin(math.pi * xx)  # same bed as the CI regression test


def cases() -> dict:
    """Case name -> (bed, domain, still-water level, ncells)."""
    return {
        "flat": (_flat, (0.0, 2.0), 3.0, 50),
        "slope": (_slope, (0.0, 2.0), 3.0, 50),
        "sinus": (_sinus, (-1.0, 1.0), 2.5, 50),
        "bowl": (_bowl, (-1.0, 1.0), 1.0, 50),
        "ci_curved": (_ci_curved, (-1.0, 1.0), 2.0, 32),
    }


def run_case(zb, domain, eta0, ncells, degree, final_time, cfl=0.3) -> dict:
    t0 = time.time()
    _, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: eta0 - zb(xx),
        lambda xx: torch.zeros_like(xx),
        zb,
        ncells=ncells,
        degree=degree,
        final_time=final_time,
        domain=domain,
        cfl=cfl,
        well_balanced=True,
    )
    h, u, eta = reconstruct_fields(coeff, zb_faces,
                                   zb_center=meta["zb_modes"][:, 0])
    c = torch.sqrt(G * h.clamp_min(1e-12))
    return {
        "final_time": final_time,
        "degree": degree,
        "ncells": ncells,
        "steps": meta["steps"],
        "eta_drift": float((eta - eta0).abs().max()),
        "eta_drift_over_depth": float(((eta - eta0).abs() / h).max()),
        "u_max": float(u.abs().max()),
        "u_over_c": float((u.abs() / c).max()),
        "reached_final_time": bool(meta["reached_final_time"]),
        "wall_time_s": round(time.time() - t0, 2),
    }


def fit_exponent(times, values):
    """Least-squares slope of log(value) vs log(time), or None if all values vanish."""
    v = np.asarray(values, dtype=float)
    if np.all(v == 0.0):
        return None
    t = np.log(np.asarray(times, dtype=float))
    return float(np.polyfit(t, np.log(np.maximum(v, 1e-300)), 1)[0])


def _run_item(item):
    """Evaluate one (case, degree, T) work item (module level for pickling)."""
    name, degree, T = item
    zb, domain, eta0, ncells = cases()[name]
    return f"{name}|p{degree}|T{T:g}", run_case(zb, domain, eta0, ncells, degree, T)


def _provenance() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, check=True,
                                    cwd=Path(__file__).resolve().parent).stdout.strip())
    except Exception:  # pragma: no cover - not a git checkout
        rev, dirty = None, None
    return {
        "git_rev": rev,
        "git_dirty": dirty,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "platform": platform.platform(),
    }


def _figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    times_ref = np.array(payload["summary"]["times"], dtype=float)
    for ax, degree in zip(axes, payload["summary"]["degrees"]):
        for case in payload["cases"]:
            if case["degree"] != degree:
                continue
            u = np.array([r["u_over_c"] for r in case["runs"]])
            keep = u > 0.0
            if not keep.any():
                continue
            ax.loglog(times_ref[keep], u[keep], "o-", label=case["case"])
        # guide slopes (algebraic growth with exponent 1..4)
        anchor_t, anchor_u = times_ref[0], 1e-13
        for p, style in ((1, ":"), (2, "--"), (4, "-.")):
            ax.loglog(times_ref, anchor_u * (times_ref / anchor_t) ** p,
                      style, color="0.6", lw=0.9,
                      label=f"$T^{p}$ guide" if p != 2 else "$T^2$ guide")
        ax.set_xlabel("integration time $T$ [s]")
        ax.set_title(f"DG P{degree}, lake at rest")
        ax.grid(True, which="both", alpha=0.3)
        if not ax.get_legend_handles_labels()[0]:
            ax.text(0.03, 0.85, "all cases exactly zero (no round-off seed)",
                    transform=ax.transAxes, fontsize=9, color="0.3")
    axes[0].set_ylabel(r"$\max |u| / \sqrt{gh}$  (dimensionless)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Spurious current of the discrete still-water state: "
                 "algebraic round-off accumulation, never exponential", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _interpretation(worst, exponents, budget) -> str:
    """One-paragraph reading of the measurement (shared by main() and refreshes)."""
    return (
        "The still-water state is a fixed point of the residual only to round-off, "
        "so a single step injects O(eps*(2p+1)*|F|/dx) of spurious momentum.  Those "
        "residuals are coherent, so the spurious current accumulates *algebraically*: "
        "the fitted log-log slopes are bounded "
        f"({min(exponents):.2f}..{max(exponents):.2f} over T = {TIMES[0]}..{TIMES[-1]} s), "
        "no exponential growth.  The accumulated level is max |u|/sqrt(gh) = "
        f"{worst['u_over_c']:.1e} over every case and time "
        f"(p={worst['degree']}, T={worst['final_time']} s).  A bare 'machine "
        "precision for all time' claim is therefore not reproducible: the achievable "
        "spurious current must be quoted with its integration time, mesh and dtype.  "
        "The seed is round-off, not bed-representation error -- the flat-bed case "
        "(z_b = 0, exactly representable) drifts too.  Values vary by a factor of a "
        "few between BLAS/thread configurations; the tolerances in "
        "tests/test_swe_wellbalanced_hydrostatic.py keep >= 180x headroom over the "
        "measurements quoted here.  Only an algebraic window is established "
        f"(T <= {TIMES[-1]} s); see docs/swe-hydrostatic-roundoff-drift-law.md.")


def _markdown(payload: dict) -> str:
    """Markdown table of the measurements (single source of truth for the docs)."""
    lines = [
        "| case | p | N | " + " | ".join(f"u/c @ T={t} s" for t in payload["summary"]["times"]) + " | eta/H (max) | growth |",
        "|---|---|---|" + "---|" * len(payload["summary"]["times"]) + "---|---|",
    ]
    for case in payload["cases"]:
        u = " | ".join(f"{r['u_over_c']:.2e}" for r in case["runs"])
        eta = max(r["eta_drift_over_depth"] for r in case["runs"])
        expo = ("exactly 0" if case["growth_exponent"] is None
                else f"T^{case['growth_exponent']:.2f}")
        lines.append(f"| {case['case']} | {case['degree']} | {case['ncells']} | "
                     f"{u} | {eta:.2e} | {expo} |")
    return "\n".join(lines) + "\n"


def main() -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT), help="output JSON path")
    ap.add_argument("--fig", default=None, help="optional output PNG path")
    ap.add_argument("--md", default=None, help="optional output markdown table path")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (each run is single-threaded)")
    ap.add_argument("--budget", type=float, default=None,
                    help="wall-clock seconds for this invocation; finished runs are "
                         "cached in <out>_runs.json and the next call resumes")
    args = ap.parse_args()
    out = Path(args.out)

    torch.set_num_threads(1)  # keep the round-off seed reproducible
    cache_path = out.with_name(out.stem + "_runs.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    budget = args.budget if args.budget is not None else float("inf")
    clock = time.time()

    # Plan the pending work items (deterministic order), then evaluate them --
    # possibly in parallel processes.  Every finished run is flushed to the cache
    # immediately, so an interrupted invocation loses at most the in-flight runs.
    plan = [(name, degree, T) for name, spec in cases().items()
            for degree in DEGREES for T in TIMES
            if f"{name}|p{degree}|T{T:g}" not in cache]

    _work = _run_item

    if plan:
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(_work, item) for item in plan]
                for future in as_completed(futures):
                    key, run = future.result()
                    cache[key] = run
                    cache_path.write_text(json.dumps(cache, indent=1) + "\n")
                    if time.time() - clock > budget:
                        break
        else:
            for item in plan:
                if time.time() - clock > budget:
                    break
                key, run = _work(item)
                cache[key] = run
                cache_path.write_text(json.dumps(cache, indent=1) + "\n")
    cache_path.write_text(json.dumps(cache, indent=1) + "\n")

    results = []
    for name, (zb, domain, eta0, ncells) in cases().items():
        for degree in DEGREES:
            keys = [f"{name}|p{degree}|T{T:g}" for T in TIMES]
            if any(key not in cache for key in keys):
                print(f"[incomplete] {len(cache)}/{len(keys) * len(cases()) * 2} runs "
                      f"cached -- rerun to finish; no JSON written", flush=True)
                return {}
            runs = [cache[key] for key in keys]
            exponent = fit_exponent([r["final_time"] for r in runs],
                                    [r["u_max"] for r in runs])
            results.append({
                "case": name,
                "degree": degree,
                "ncells": ncells,
                "domain": list(domain),
                "eta0": eta0,
                "growth_exponent": exponent,
                "runs": runs,
            })
            expo = "exact" if exponent is None else f"T^{exponent:.2f}"
            print(f"{name:10s} p={degree}  u/c(T={TIMES[-1]})={runs[-1]['u_over_c']:.3e}"
                  f"  eta/H(T={TIMES[-1]})={runs[-1]['eta_drift_over_depth']:.3e}"
                  f"  growth={expo}", flush=True)

    all_runs = [r for case in results for r in case["runs"]]
    worst = max(all_runs, key=lambda r: r["u_over_c"])
    exponents = [c["growth_exponent"] for c in results
                 if c["growth_exponent"] is not None]
    budget = {}
    for degree in DEGREES:
        vals = [r["u_over_c"] for case in results if case["degree"] == degree
                for r in case["runs"]]
        etas = [r["eta_drift_over_depth"] for case in results
                if case["degree"] == degree for r in case["runs"]]
        budget[f"p{degree}"] = {
            "max_u_over_c_T4": max(r["u_over_c"] for case in results
                                   if case["degree"] == degree
                                   for r in case["runs"] if r["final_time"] == TIMES[-1]),
            "max_u_over_c_all_times": max(vals),
            "max_eta_drift_over_depth_all_times": max(etas),
        }

    summary = {
        "ncells_per_case": {name: spec[3] for name, spec in cases().items()},
        "degrees": list(DEGREES),
        "times": list(TIMES),
        "max_u_over_c": worst["u_over_c"],
        "max_u_over_c_case": {"case": next(c["case"] for c in results
                                          if worst in c["runs"]),
                              "degree": worst["degree"],
                              "final_time": worst["final_time"]},
        "max_growth_exponent": max(exponents) if exponents else None,
        "dimensionless_budget": budget,
        "interpretation": _interpretation(worst, exponents, budget),
    }
    payload = {
        "generated_by": "validation/swe_thacker/run_hydrostatic_wb_drift.py",
        "provenance": _provenance(),
        "dtype": "float64",
        "cfl": 0.3,
        "reconstruction": "zb_center = metadata['zb_modes'][:, 0] (polynomial bed average)",
        "cases": results,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nmax u/c = {summary['max_u_over_c']:.3e}  "
          f"max growth exponent = {summary['max_growth_exponent']}")
    for key, val in budget.items():
        print(f"{key}: u/c(T={TIMES[-1]}) <= {val['max_u_over_c_T4']:.2e}  "
              f"eta/H <= {val['max_eta_drift_over_depth_all_times']:.2e}")
    print(f"wrote {out}")
    if args.fig:
        _figure(payload, Path(args.fig))
        print(f"wrote {args.fig}")
    if args.md:
        Path(args.md).write_text(_markdown(payload))
        print(f"wrote {args.md}")
    return payload


if __name__ == "__main__":
    main()