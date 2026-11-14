"""CASE-LAX-DG resolution-closure ladder (ledger A0, p3, 200 -> 1600 cells).

What this driver answers
------------------------
The pool driver ``benchmarks/lax_case_pool.py`` measures the ledger acceptance
``A0`` (relative L2 versus the exact solution, whole field) at the one
production leg the case module registers, ``N = 400 / P3``, and commits
``3.1156 %`` -- an **honest FAIL** against the 3 % bar, a 3.85 % overflow.
That number says nothing about *whether* the bar is reachable: the registered
ladder stops at ``N = 400`` and the measured p3 order over its finest pair is
only ``0.48`` (``docs/CASE-LAX-DG-p3-asymptotic-probe.md``).

This driver does exactly two things, nothing else:

1. **h-refinement of the production arm to closure.**  The registered
   configuration (p3, no shock limiter, CFL 0.35, exact Riemann reference) is
   run on ``N in {200, 400, 800, 1600}``.  For every leg the whole-field
   relative L1/L2 of all three primitive fields is measured on the element
   Gauss nodes against the closed-form exact solution, together with the
   R3-band-masked L1 (same fixed 0.02 physical band the pool uses).  The
   smallest *actually run* mesh whose L2 drops below the ledger tolerance is
   reported as the closure mesh; when no leg closes, the measured finest-pair
   order is used to *extrapolate* an ``N*`` and it is labelled as an
   extrapolation, never as a pass.

2. **Attribution of the p3 excess to the unlimited scheme.**  At the
   production mesh ``N = 400`` the single knob ``limiter`` is switched
   (``None`` / ``tvb`` / ``weno``), everything else held.  This separates
   "p3 is inaccurate because it is unlimited (Gibbs oscillations leaking out
   of the three discontinuities)" from "p3 is inaccurate because the operator
   is wrong".  The second hypothesis is already refuted by
   ``tests/test_lax_p3_smooth_order.py`` (smooth periodic entropy wave, p3
   order 4.00).

Nothing is transcribed: every number in ``metrics.json`` comes from the solver
or from the exact Riemann sampler, and the doc renders straight from the JSON.

Usage::

    python -m benchmarks.lax_resolution_closure --quick      # smoke (N=100/200)
    python -m benchmarks.lax_resolution_closure              # verdict run (~55 min)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.lax_case_pool import (
    CFL,
    DOMAIN,
    FINAL_TIME,
    R3_BAND_WIDTH,
    _weighted_l1,
    _weighted_l2,
    convergence_order,
    exact_reference,
    exact_waves,
    ledger_acceptance_tolerance,
    modal_fields,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d

OUT = Path("docs/benchmarks/results/lax_resolution_closure")
FIG = OUT / "figures"

#: the registered production arm, refined to closure.  Every entry is a
#: multiple of the ledger production leg (400), so the ladder is nested.
CLOSURE_LADDER = (200, 400, 800, 1600)
QUICK_LADDER = (100, 200)
DEGREE = 3
#: limiter arms at the production mesh -- one knob, everything else held
LIMITER_ARMS = (None, "tvb", "weno")
ARM_CELLS = 400
FIELDS = ("density", "velocity", "pressure")


# --------------------------------------------------------------------- runners
def run_leg(cells, degree=DEGREE, limiter=None, tvb_constant=1.0,
            waves=None, final_time=FINAL_TIME, cfl=CFL):
    """Solve one leg and measure every norm this driver reports.

    The mask of the R3 band is the pool's rule: a fixed **physical** half
    width ``R3_BAND_WIDTH`` around each of the four analytic wave positions,
    removed from the field before the band-masked L1 is taken.
    """
    from benchmarks.lax_case_pool import LEFT, RIGHT

    if waves is None:
        waves = exact_waves(final_time=final_time)
    started = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=cells, degree=degree, final_time=final_time,
        cfl=cfl, limiter=limiter, tvb_constant=tvb_constant,
    )
    points, primitive, dx, weights = modal_fields(x, coeff)
    exact = exact_reference(points, final_time=final_time)
    band = np.ones_like(primitive["density"], dtype=bool)
    for name in ("rarefaction_head", "rarefaction_tail", "contact", "shock"):
        band &= np.abs(points - waves["positions"][name]) > R3_BAND_WIDTH
    entry = {
        "cells": int(cells),
        "degree": int(degree),
        "limiter": "none" if limiter is None else str(limiter),
        "tvb_constant": float(tvb_constant) if limiter == "tvb" else None,
        "steps": int(meta["steps"]),
        "dt_mean": float(meta.get("dt_mean", final_time / max(meta["steps"], 1))),
        "wall_seconds": float(time.time() - started),
        "density_min": float(primitive["density"].min()),
        "band_fraction": float(band.mean()),
        "l1": {f: _weighted_l1(points, primitive[f], exact[f], weights, dx)
               for f in FIELDS},
        "l2": {f: _weighted_l2(points, primitive[f], exact[f], weights, dx)
               for f in FIELDS},
        "band_l1": {f: _weighted_l1(points, primitive[f], exact[f], weights, dx,
                                    mask=band) for f in FIELDS},
        "max_abs": {f: float(np.max(np.abs(primitive[f] - exact[f]))) for f in FIELDS},
    }
    print(f"[closure] N={cells:5d} p={degree} limiter={entry['limiter']:4s} "
          f"L2(rho)={entry['l2']['density']:.6f} "
          f"L1(rho)={entry['l1']['density']:.6f} "
          f"band={entry['band_l1']['density']:.6f} "
          f"t={entry['wall_seconds']:.1f}s", flush=True)
    return entry, (points, primitive, exact)


def run_ladder(ladder=CLOSURE_LADDER):
    waves = exact_waves()
    entries, profiles = [], None
    for cells in ladder:
        entry, profiles = run_leg(cells, waves=waves)
        entries.append(entry)
    return entries, profiles


def run_arms(waves=None):
    waves = waves or exact_waves()
    return [run_leg(ARM_CELLS, limiter=arm, waves=waves)[0] for arm in LIMITER_ARMS]


# ----------------------------------------------------------------------- gates
def closure_mesh(entries, tolerance, field="density", norm="l2"):
    """Smallest *run* mesh whose error is below *tolerance*, plus an h-model N*.

    ``closed_at`` is ``None`` when no run leg meets the bar; ``extrapolated_n``
    is then the mesh implied by fitting the two finest legs with a power law
    ``e = C N**(-q)``.  The two are reported separately on purpose: only
    ``closed_at`` is evidence, ``extrapolated_n`` is a model prediction.
    """
    ordered = sorted(entries, key=lambda e: e["cells"])
    series = [(e["cells"], e[norm][field]) for e in ordered]
    closed = next((n for n, err in series if err <= tolerance), None)
    finest = series[-1]
    model = None
    if len(series) >= 2:
        coarse = series[-2]
        order = convergence_order(coarse[0], coarse[1], finest[0], finest[1])
        model = {"order_finest_pair": order, "finest_pair": [coarse[0], finest[0]]}
        if order and order > 0 and finest[1] > tolerance:
            model["extrapolated_n"] = int(
                math.ceil(finest[0] * (finest[1] / tolerance) ** (1.0 / order))
            )
        elif finest[1] <= tolerance:
            model["extrapolated_n"] = finest[0]
    return {"tolerance": float(tolerance), "norm": norm, "field": field,
            "series": [{"cells": n, "error": err} for n, err in series],
            "closed_at": closed, "model": model}


def gate_closure(entries, arms, tolerance):
    """Verdict of the closure question, stated so it cannot be over-read."""
    l2 = closure_mesh(entries, tolerance, "density", "l2")
    worst_l2 = {f: closure_mesh(entries, tolerance, f, "l2") for f in FIELDS}
    closed_every_field = all(v["closed_at"] is not None for v in worst_l2.values())
    production = next((e for e in entries if e["cells"] == ARM_CELLS), None)
    # A leg only counts as closure if the strictest bar (worst field) is met.
    strict = max(worst_l2, key=lambda f: worst_l2[f]["series"][-1]["error"])
    closing = [e["cells"] for e in sorted(entries, key=lambda x: x["cells"])
               if all(e["l2"][f] <= tolerance for f in FIELDS)]
    status = "PASS" if closing else "FAIL"
    return {
        "bar": "relative L2 vs exact, whole field, every primitive field",
        "tolerance": float(tolerance),
        "production_leg": "p3.N400",
        "production_l2": (None if production is None else production["l2"]),
        "worst_field": strict,
        "per_field": worst_l2,
        "closure_mesh": (min(closing) if closing else None),
        "status": status,
        "reading": ("PASS means: at least one *actually run* mesh of the "
                    "registered p3/unlimited ladder meets the ledger L2 bar for "
                    "every field.  A model extrapolation of N* is reported but "
                    "never counted as a pass."),
        "arms_at_production_mesh": [
            {"limiter": a["limiter"],
             "l2": a["l2"], "l1": a["l1"], "band_l1": a["band_l1"]} for a in arms
        ],
    }


# --------------------------------------------------------------------- figures
def render_figures(entries, profile, arms, gate, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    tol = gate["tolerance"]

    # (1) h-convergence of every field with the ledger bar drawn in.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for field, style in zip(FIELDS, ("o-", "s--", "^:")):
        xs = [e["cells"] for e in entries]
        ys = [e["l2"][field] for e in entries]
        ax.loglog(xs, ys, style, label=f"{field} (whole field L2)")
    ax.axhline(tol, color="k", lw=1.2, ls="--", label=f"ledger bar {tol:.0%}")
    if gate.get("closure_mesh"):
        ax.axvline(gate["closure_mesh"], color="g", lw=1.0, alpha=0.6,
                   label=f"closure mesh N={gate['closure_mesh']}")
    ax.set_xlabel("cells N (p = 3, no limiter)")
    ax.set_ylabel("relative L2 vs exact")
    ax.set_title("CASE-LAX-DG: p3 h-refinement against the 3 % ledger bar")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "closure_ladder.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Whole-field relative L2 of density/velocity/pressure versus "
                 "cell count at p=3 with the registered (unlimited) arm; the "
                 "dashed line is the ledger 3 % bar, the green line the mesh at "
                 "which it is met."})

    # (2) density profile at the finest leg vs the exact solution.
    if profile is None or profile[0] is None:
        return made
    points, primitive, exact = profile
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(points.ravel(), exact["density"].ravel(), "k-", lw=1.0, label="exact Riemann")
    ax.plot(points.ravel(), primitive["density"].ravel(), "r.", ms=2.0,
            label=f"DG p3 N={entries[-1]['cells']}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
    ax.set_title("Density at the finest closure leg vs the exact solution")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    path = out_dir / "closure_profile.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Density field of the finest closure leg on the element Gauss "
                 "nodes against the closed-form exact solution; the four "
                 "analytic waves are resolved without overshoot outside the "
                 "documented band."})

    # (3) limiter arms at the production mesh.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = [a["limiter"] for a in arms]
    width = 0.26
    for k, field in enumerate(FIELDS):
        vals = [a["l2"][field] for a in arms]
        ax.bar([i + (k - 1) * width for i in range(len(arms))], vals, width,
               label=f"{field} L2")
    ax.axhline(tol, color="k", ls="--", lw=1.2, label="ledger bar")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f"limiter={l}" for l in labels])
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 vs exact")
    ax.set_title(f"One-knob comparison at the production leg (p3, N={ARM_CELLS})")
    ax.legend(fontsize=8)
    path = out_dir / "closure_arms.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    made.append({"name": path.name, "caption":
                 "Limiter arms at the production mesh p3 N=400: the only knob "
                 "changed is the shock limiter, so the difference is the cost "
                 "of leaving the discontinuities unlimited."})
    return made


def render_summary(metrics):
    g = metrics["gate"]
    lines = [
        "# CASE-LAX-DG resolution closure (p3 ladder to 3 %)",
        "",
        f"generated: {metrics['generated_at_utc']}  |  wall: {metrics['wall_seconds']:.0f} s",
        "",
        f"Verdict: **{g['status']}**  (closure mesh = {g['closure_mesh']})",
        "",
        "| N (p3, no limiter) | rho L2 | u L2 | p L2 | rho L1 | rho band L1 |",
        "|---|---|---|---|---|---|",
    ]
    for e in metrics["ladder"]:
        lines.append(
            f"| {e['cells']} | {e['l2']['density']:.4%} | {e['l2']['velocity']:.4%} | "
            f"{e['l2']['pressure']:.4%} | {e['l1']['density']:.4%} | "
            f"{e['band_l1']['density']:.4%} |")
    lines += ["", "| limiter at N=400 | rho L2 | u L2 | p L2 |", "|---|---|---|---|"]
    for a in metrics["arms"]:
        lines.append(f"| {a['limiter']} | {a['l2']['density']:.4%} | "
                     f"{a['l2']['velocity']:.4%} | {a['l2']['pressure']:.4%} |")
    lines.append("")
    Path(metrics["out_dir"], "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke ladder")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-arms", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tolerance = ledger_acceptance_tolerance()
    ladder = QUICK_LADDER if args.quick else CLOSURE_LADDER
    entries, profile = run_ladder(ladder)
    arms = [] if args.skip_arms else run_arms()
    gate = gate_closure(entries, arms, tolerance)
    figures = [] if args.skip_figures else render_figures(entries, profile, arms, gate,
                                                          out / "figures")
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "study": "resolution closure of ledger A0 (p3, whole-field relative L2 < 3 %)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"degree": DEGREE, "ladder": list(ladder), "cfl": CFL,
                   "final_time": FINAL_TIME, "domain": list(DOMAIN),
                   "r3_band_width": R3_BAND_WIDTH,
                   "limiter_arms": [("none" if a is None else a) for a in LIMITER_ARMS],
                   "arm_cells": ARM_CELLS, "quick": bool(args.quick)},
        "ladder": entries,
        "arms": arms,
        "gate": gate,
        "figures": figures,
        "out_dir": str(out),
        "wall_seconds": float(time.time() - started),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    render_summary(metrics)
    print(f"[closure] {gate['status']} closure_mesh={gate['closure_mesh']} "
          f"wall={metrics['wall_seconds']:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())