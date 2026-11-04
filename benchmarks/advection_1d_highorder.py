"""Verification of the 1-D modal DG advection kernel beyond its historical P3 cap.

Three independent sections, each with a criterion fixed *before* the numbers are
produced:

``smooth``
    Design-order study for ``p = 0..5``.  The exact solution
    ``sin(2*pi*(x - t))`` is smooth, so the L2 error of a P_p DG discretisation
    must decay like ``h^(p+1)``.  The time integrator is classical RK4, i.e.
    fourth order, which *caps* the measurable rate at 4 when ``p + 1 > 4``;
    the ``temporal`` control quantifies how much of the residual rate is time
    integration and how much is space, instead of blaming the spatial operator.
``floor``
    Round-off floor: at high ``p`` the error eventually stops decaying because
    float64 has ~1e-16 relative accuracy.  Measuring the floor is a negative
    control for the order study -- below it an "order collapse" is arithmetic,
    not a bug.
``discontinuous``
    Square-wave advection over half a period.  For a discontinuous solution the
    DG L1 rate is the classical ``p + 1/2`` (Johnson & Pitkaranta 1986; see also
    Cockburn & Shu, RKDG V, 1998), i.e. one half order *below* the smooth rate,
    and the L-infinity error does **not** go to zero without limiting (Gibbs).
    Both are measured here.  The estimator itself is controlled at ``t = 0``,
    where the L1 error of the L2 projection must be machine-precision small.

Run: ``PYTHONPATH=. python -m benchmarks.advection_1d_highorder``
Writes ``validation/advection_1d/highorder_summary.json`` and the figure
``docs/figures/advection_1d_highorder.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from tensordg.advection_1d import (
    MAX_DEGREE,
    l1_error,
    l2_error,
    linf_error,
    solve_advection,
)

OUT = pathlib.Path("validation/advection_1d")
FIG = pathlib.Path("docs/figures")

# Criterion constants (declared up front, reported verbatim in the summary).
ORDER_TOL = 0.05          # fitted order vs design order, relative
LP_HALF_BAND = 0.35       # accepted band around the p+1/2 discontinuity rate
CONTROL_TOL = 1e-12       # projection-only L1 at t=0 must be round-off
GATE = 0.03               # engineering gate: relative L1 < 3%


def square_wave(x):
    """Unit square wave on [0.25, 0.75), periodic."""
    f = x % 1.0
    return ((f >= 0.25) & (f < 0.75)).to(x.dtype)


def _fit_order(hs, errs):
    """Least-squares slope of log(error) against log(h)."""
    if len(hs) < 2:
        return float("nan")
    return float(abs(np.polyfit(np.log(hs), np.log(errs), 1)[0]))


def smooth_section(degrees=(0, 1, 2, 3, 4, 5), cfl=0.02, final_time=0.25):
    grids = {0: (8, 16, 32, 64), 1: (8, 16, 32, 64), 2: (8, 16, 32, 64),
             3: (8, 16, 32, 64), 4: (8, 16, 32), 5: (8, 16, 32)}
    out = {}
    for p in degrees:
        cells = grids.get(p, (8, 16, 32))
        hs, errs = [], []
        for n in cells:
            centers, coeff, _ = solve_advection(ncells=n, degree=p,
                                                final_time=final_time, cfl=cfl)
            hs.append(1.0 / n)
            errs.append(l2_error(centers, coeff, final_time))
        rate = _fit_order(hs, errs)
        design = p + 1
        deviations = [math.log(e1 / e2) / math.log(2.0)
                      for e1, e2 in zip(errs[:-1], errs[1:])]
        out[f"P{p}"] = {
            "cells": list(cells),
            "l2": errs,
            "pairwise_orders": deviations,
            "fitted_order": rate,
            "design_order": design,
            "order_matches_design": bool(abs(rate - design) <= ORDER_TOL * design),
            "rk4_capped": bool(design > 4),
        }
    return out


def temporal_control(degrees=(1, 2, 3, 5), ncells=16, final_time=0.25,
                     cfl_coarse=0.02, cfl_fine=0.005):
    """How much of the measured error is time integration?

    At fixed grid the two runs differ only in dt, so
    ``1 - |e_fine/e_coarse|`` is the fraction of the coarse error contributed by
    the RK4 stage.  Small values mean the reported order is spatially limited.
    """
    out = {}
    for p in degrees:
        errs = {}
        for tag, cfl in (("cfl_coarse", cfl_coarse), ("cfl_fine", cfl_fine)):
            centers, coeff, _ = solve_advection(ncells=ncells, degree=p,
                                                final_time=final_time, cfl=cfl)
            errs[tag] = l2_error(centers, coeff, final_time)
        out[f"P{p}"] = {
            **errs,
            "temporal_fraction": float(1.0 - errs["cfl_fine"] / errs["cfl_coarse"]),
            "negligible": bool(1.0 - errs["cfl_fine"] / errs["cfl_coarse"] < 0.01),
        }
    return out


def floor_section(degree=5, cells=(16, 32, 64, 128), cfl=0.02,
                  final_time=0.25):
    hs, errs = [], []
    for n in cells:
        centers, coeff, _ = solve_advection(ncells=n, degree=degree,
                                            final_time=final_time, cfl=cfl)
        hs.append(1.0 / n)
        errs.append(l2_error(centers, coeff, final_time))
    return {
        "degree": degree,
        "cells": list(cells),
        "l2": errs,
        "pairwise_orders": [math.log(e1 / e2) / math.log(2.0)
                            for e1, e2 in zip(errs[:-1], errs[1:])],
        "floor_level": float(min(errs)),
        "floor_reached": bool(errs[-1] / errs[-2] > 0.5),
    }


def discontinuous_section(degrees=(0, 1, 2, 3), cells=(16, 32, 64),
                          final_time=0.5, cfl=0.05):
    out = {}
    # Estimator control: at t=0 the DG solution *is* the L2 projection, so the
    # L1 error must be round-off small for the (non-polynomial) square wave.
    c0, cf0, _ = solve_advection(ncells=32, degree=3, final_time=0.0,
                                 initial=square_wave)
    control = l1_error(c0, cf0, 0.0, exact=square_wave, jumps=(0.25, 0.75))
    for p in degrees:
        hs, l1s, linfs = [], [], []
        for n in cells:
            centers, coeff, _ = solve_advection(ncells=n, degree=p,
                                                final_time=final_time, cfl=cfl,
                                                initial=square_wave)
            hs.append(1.0 / n)
            l1s.append(l1_error(centers, coeff, final_time, exact=square_wave,
                                jumps=(0.25, 0.75), nsub=6))
            linfs.append(linf_error(centers, coeff, final_time, exact=square_wave))
        rate = _fit_order(hs, l1s)
        predicted = p + 0.5
        out[f"P{p}"] = {
            "degree": p,
            "cells": list(cells),
            "l1_abs": l1s,
            "l1_relative": [e / 0.5 for e in l1s],
            "pairwise_orders": [math.log(e1 / e2) / math.log(2.0)
                                for e1, e2 in zip(l1s[:-1], l1s[1:])],
            "fitted_order": rate,
            "predicted_order": predicted,
            "within_band": bool(abs(rate - predicted) <= LP_HALF_BAND),
            "linf": linfs,
            "linf_decay_ratio": linfs[-1] / linfs[0],
            "gibbs_persists": bool(linfs[-1] / linfs[0] > 0.8),
            "relative_l1_at_finest": l1s[-1] / 0.5,
            "passes_3pct_gate": bool(l1s[-1] / 0.5 < GATE),
            "degree_above_old_cap": bool(p > 3),
        }
    return {"estimator_control_t0": control,
            "estimator_control_pass": bool(control < CONTROL_TOL),
            "cases": out}


def make_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, 6))
    ax = axes[0]
    for (label, d), col in zip(summary["smooth"].items(), colors):
        p = d["degree"]
        hs = [1.0 / n for n in d["cells"]]
        ax.loglog(hs, d["l2"], "o-", color=col, lw=1.4, ms=4,
                  label=f"P{p} (fitted {d['fitted_order']:.2f}, design {p + 1})")
    hs_ref = np.array([1 / 64.0, 1 / 8.0])
    for slope, y0 in ((2, 1e-5), (4, 1e-9)):
        ax.plot(hs_ref, y0 * (hs_ref / hs_ref[0]) ** slope, "k:", lw=1.0, alpha=0.6)
        ax.annotate(f"h^{slope}", (hs_ref[-1], y0), fontsize=8, alpha=0.7)
    ax.set_xlabel("h = 1/ncells")
    ax.set_ylabel("relative L2 error")
    ax.set_title("Smooth sine wave: design order p+1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    for (label, d), col in zip(summary["discontinuous"]["cases"].items(), colors):
        p = d["degree"]
        hs = [1.0 / n for n in d["cells"]]
        ax.loglog(hs, d["l1_relative"], "o-", color=col, lw=1.4, ms=4,
                  label=f"P{p} L1 (fitted {d['fitted_order']:.2f}, p+1/2={p + 0.5})")
    ax.set_xlabel("h = 1/ncells")
    ax.set_ylabel("relative L1 error")
    ax.set_title("Square wave (half period): L1 rate p+1/2, Gibbs in Linf")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="smooth,temporal,floor,discontinuous")
    ap.add_argument("--figure", default=str(FIG / "advection_1d_highorder.png"))
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the saved JSON (no recompute)")
    args = ap.parse_args()
    if args.figure_only:
        summary = json.loads((OUT / "highorder_summary.json").read_text())
        make_figure(summary, pathlib.Path(args.figure))
        print(f"figure -> {args.figure}")
        return
    want = {s.strip() for s in args.sections.split(",") if s.strip()}

    summary = {"max_degree": MAX_DEGREE,
               "criteria": {"order_tol_relative": ORDER_TOL,
                            "l1_half_order_band": LP_HALF_BAND,
                            "estimator_control_tol": CONTROL_TOL,
                            "relative_l1_gate": GATE}}
    if "smooth" in want:
        summary["smooth"] = smooth_section()
        for k, d in summary["smooth"].items():
            d["degree"] = int(k[1:])
    if "temporal" in want:
        summary["temporal"] = temporal_control()
    if "floor" in want:
        summary["floor"] = floor_section()
    if "discontinuous" in want:
        summary["discontinuous"] = discontinuous_section()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = OUT / "highorder_summary.json"
    stamp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if "smooth" in summary and "discontinuous" in summary:
        make_figure(summary, pathlib.Path(args.figure))

    print(f"{'case':>6} {'design':>7} {'fitted':>7} {'match':>6}  errors")
    for k, d in summary.get("smooth", {}).items():
        print(f"{k:>6} {d['design_order']:>7} {d['fitted_order']:>7.2f} "
              f"{str(d['order_matches_design']):>6}  "
              + " ".join(f"{e:.3e}" for e in d["l2"]))
    for k, d in summary.get("temporal", {}).items():
        print(f"temporal {k}: {d['temporal_fraction']:.2%} of the coarse error "
              f"(negligible={d['negligible']})")
    if "floor" in summary:
        f = summary["floor"]
        print(f"floor  P{f['degree']}  min={f['floor_level']:.3e} "
              f"reached={f['floor_reached']}  orders="
              f"{['%.2f' % o for o in f['pairwise_orders']]}")
    if "discontinuous" in summary:
        dc = summary["discontinuous"]
        print(f"estimator control t=0 L1 = {dc['estimator_control_t0']:.3e} "
              f"pass={dc['estimator_control_pass']}")
        print(f"{'case':>6} {'pred':>6} {'fitted':>7} {'band':>6} {'relL1@fine':>11} "
              f"{'3%gate':>7} {'Linf ratio':>10}")
        for k, d in dc["cases"].items():
            print(f"{k:>6} {d['predicted_order']:>6.1f} {d['fitted_order']:>7.2f} "
                  f"{str(d['within_band']):>6} {d['relative_l1_at_finest']:>11.4%} "
                  f"{str(d['passes_3pct_gate']):>7} {d['linf_decay_ratio']:>10.3f}")


if __name__ == "__main__":
    main()