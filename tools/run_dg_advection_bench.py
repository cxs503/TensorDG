#!/usr/bin/env python3
"""Generate the ice-edge transport DG benchmark artifact (campaign IN12-P5).

Runs the case suite of :mod:`tensordg.ice_edge_bench` (order of accuracy for
p = 1, 2, 3, peak retention of a rotating ice-edge front, reentrant vortex
error, global mass balance), evaluates the machine-checkable criteria, writes
``validation/dg_advection_bench/dg_advection_bench.json`` and the figures in ``docs/figures/``.

Usage::

    python tools/run_dg_advection_bench.py            # full campaign run (~5 min)
    python tools/run_dg_advection_bench.py --quick    # smoke configuration
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg import ice_edge_bench as bench  # noqa: E402
from tensordg.advection_2d_var import DGAdvection2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "dg_advection_bench" / "dg_advection_bench.json"
FIGDIR = ROOT / "docs" / "figures"

# Resolution sets per case and degree.  ``quick`` keeps the whole run inside a
# couple of minutes for smoke tests / CI; ``full`` is the campaign configuration
# used for the published artifact.
CONFIG = {
    "full": {
        "shear_transport": {1: [12, 16, 24], 2: [8, 12, 16], 3: [8, 12, 16]},
        "solid_body_rotation_front": {1: [16, 24, 32], 2: [12, 16, 24], 3: [12, 16, 24]},
        "reentrant_deformational_vortex": {1: [12, 16, 24], 2: [12, 16, 24], 3: [10, 14, 20]},
        "peak_rotation": [(1, 24), (2, 16), (3, 16)],
        "peak_rotation_resolution": [(3, 8), (3, 12), (3, 20)],
        "peak_vortex": [(2, 12), (3, 12), (3, 16)],
        "mass_trace": (2, 12),
        "field_snapshot": (3, 24),
    },
    "quick": {
        "shear_transport": {1: [12, 16], 2: [8, 12], 3: [8, 12]},
        "solid_body_rotation_front": {1: [12, 16], 2: [8, 12], 3: [8, 12]},
        "reentrant_deformational_vortex": {1: [12, 16], 2: [8, 12], 3: [8, 12]},
        "peak_rotation": [(2, 12)],
        "peak_rotation_resolution": [(2, 8), (2, 12)],
        "peak_vortex": [(2, 12)],
        "mass_trace": (2, 8),
        "field_snapshot": (2, 12),
    },
}
CFL = 0.3


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float64",
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_convergence(studies: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    styles = {"shear_transport": ("o", "#1f77b4", "periodic shear (analytic)"),
              "solid_body_rotation_front": ("s", "#d62728", "rotating front (disk norm)"),
              "reentrant_deformational_vortex": ("^", "#2ca02c", "reentrant vortex")}
    for case, per_degree in studies.items():
        marker, color, label = styles[case]
        for p, table in sorted(per_degree.items()):
            n = np.asarray(table["cells"], dtype=float)
            err = np.asarray(table["errors_l2"], dtype=float)
            ax.loglog(n, err, marker + "-", color=color, ms=5,
                      alpha=0.45 + 0.18 * p, label=f"{label}, p={p}")
    ref = np.asarray([8.0, 32.0])
    for p, ls in [(1, ":"), (2, "--"), (3, "-.")]:
        scale = 3e-2 / (8.0 ** (p + 1))
        ax.loglog(ref, scale * ref ** (p + 1), ls, color="k", lw=1.0,
                  label=f"slope p+1 = {p + 1}")
    ax.set_xlabel("cells per direction $n$")
    ax.set_ylabel(r"$L_2$ error")
    ax.set_title("Ice-edge transport: DG observed order of accuracy")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_fields(snapshot: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = snapshot["x"]
    y = snapshot["y"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    for ax, data, title in zip(
        axes,
        [snapshot["initial"], snapshot["numeric"], snapshot["error"]],
        ["initial ice-edge front", f"DG p={snapshot['degree']} after 1 revolution",
         "numerical $-$ exact (disk $r\\leq0.28$)"],
    ):
        im = ax.pcolormesh(x, y, data.T, shading="nearest", cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_peaks(rot_peak: list, rot_res: list, vortex_peak: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    degs = [r["degree"] for r in rot_peak]
    ret = [r["peak_retention"] for r in rot_peak]
    ax.bar([f"p={d}\nn={r['cells'][0]}" for d, r in zip(degs, rot_peak)], ret,
           color="#d62728", alpha=0.85)
    ax.axhline(0.99, ls="--", color="k", lw=1, label="0.99 acceptance")
    ax.set_ylim(min(0.97, min(ret) - 0.005), 1.005)
    ax.set_ylabel("peak retention (1 revolution)")
    ax.set_title("rotating front peak preservation", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    ax = axes[1]
    ns = [r["cells"][0] for r in rot_res]
    r3 = [r["peak_retention"] for r in rot_res]
    ax.semilogx(ns, r3, "s-", color="#1f77b4", label="rotating front, p=3")
    if vortex_peak:
        ax.semilogx([r["cells"][0] for r in vortex_peak],
                    [r["peak_retention"] for r in vortex_peak],
                    "^-", color="#2ca02c", label="reentrant vortex bump")
    ax.axhline(0.99, ls="--", color="k", lw=1)
    ax.set_xlabel("cells per direction $n$")
    ax.set_ylabel("peak retention")
    ax.set_title("peak retention vs resolution", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_mass(trace: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drift = np.asarray(trace["relative_drift"], dtype=float)
    steps = np.linspace(0, trace["steps"], len(drift))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogy(steps, np.maximum(np.abs(drift), 1e-18), "-", color="#1f77b4")
    ax.axhline(1e-12, ls="--", color="k", lw=1, label="1e-12 acceptance")
    ax.axhline(np.finfo(float).eps, ls=":", color="grey", lw=1,
               label="double precision eps")
    ax.set_xlabel("RK4 step")
    ax.set_ylabel("relative mass drift")
    ax.set_title(f"global mass balance, reentrant vortex p={trace['degree']}, "
                 f"n={trace['cells'][0]}", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def field_snapshot(degree: int, cells: int, revolutions: float = 1.0) -> dict:
    """Nodal fields for the field figure (initial / numerical / error)."""
    spec = bench.CASES["solid_body_rotation_front"]
    solver = DGAdvection2D(degree, cells)
    coeff0 = solver.project(spec.profile)
    result = solver.solve(coeff0, spec.velocity, revolutions, cfl=CFL)
    u0 = solver.nodal(coeff0)[:, :, solver.basis.nq // 2, solver.basis.nq // 2].numpy()
    un = solver.nodal(result["coeff"])[:, :, solver.basis.nq // 2,
                       solver.basis.nq // 2].numpy()
    ue = spec.exact(solver.Xn, solver.Yn, revolutions)[
        :, :, solver.basis.nq // 2, solver.basis.nq // 2].numpy()
    grid = solver.xq[:, solver.basis.nq // 2].numpy()
    gridy = solver.yq[:, solver.basis.nq // 2].numpy()
    return {"degree": degree, "cells": cells, "initial": u0, "numeric": un,
            "error": un - ue, "x": grid, "y": gridy}


# --------------------------------------------------------------------------
# Parallel task driver
# --------------------------------------------------------------------------
def _worker(task):
    """Run one benchmark task (executed in a worker process)."""
    torch.set_num_threads(1)
    kind, payload = task
    if kind == "order":
        case, degree, cells = payload
        return kind, payload, bench.run_order_study(case, degree, cells, cfl=CFL).as_dict()
    if kind == "rev":
        degree, cells, revolutions = payload
        return kind, payload, bench.run_revolution_check(
            degree=degree, cells=cells, cfl=CFL, revolutions=revolutions)
    if kind == "vpeak":
        degree, cells = payload
        return kind, payload, bench.run_vortex_peak_check(
            degree=degree, cells=cells, cfl=CFL)
    if kind == "mass":
        degree, cells = payload
        return kind, payload, bench.mass_balance_trace(degree, cells, cfl=CFL)
    if kind == "fields":
        degree, cells, revolutions = payload
        return kind, payload, field_snapshot(degree, cells, revolutions)
    raise ValueError(f"unknown task {kind}")


def run_tasks(tasks, jobs):
    from concurrent.futures import ProcessPoolExecutor
    if jobs <= 1:
        return [_worker(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_worker, tasks))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke configuration")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1),
                        help="parallel worker processes")
    args = parser.parse_args()
    cfg = CONFIG["quick" if args.quick else "full"]

    tasks = []
    for case in ("shear_transport", "solid_body_rotation_front",
                 "reentrant_deformational_vortex"):
        for degree, cells in sorted(cfg[case].items()):
            tasks.append(("order", (case, degree, cells)))
    for p, n in cfg["peak_rotation"]:
        tasks.append(("rev", (p, n, 1.0)))
    for p, n in cfg["peak_rotation_resolution"]:
        tasks.append(("rev", (p, n, 1.0)))
    for p, n in cfg["peak_vortex"]:
        tasks.append(("vpeak", (p, n)))
    tasks.append(("mass", cfg["mass_trace"]))
    tasks.append(("fields", tuple(cfg["field_snapshot"]) + (1.0,)))

    results = run_tasks(tasks, args.jobs)

    studies: dict[str, dict] = {"shear_transport": {},
                               "solid_body_rotation_front": {},
                               "reentrant_deformational_vortex": {}}
    rot_peak, rot_res, vortex_peak, trace, snap = [], [], [], None, None
    for kind, payload, value in results:
        if kind == "order":
            case, degree, _cells = payload
            studies[case][degree] = value
            print(f"[order] {case:32s} p={degree} n={value['cells']} "
                  f"order={value['observed_order']:.2f} "
                  f"(need {value['required_order']:.1f}) "
                  f"errors={['%.2e' % e for e in value['errors_l2']]}", flush=True)
        elif kind == "rev":
            key = (payload[0], payload[1])
            is_main = key in [tuple(x) for x in cfg["peak_rotation"]]
            (rot_peak if is_main else rot_res).append(value)
        elif kind == "vpeak":
            vortex_peak.append(value)
        elif kind == "mass":
            trace = value
        elif kind == "fields":
            snap = value
    rot_peak.sort(key=lambda r: r["degree"])
    rot_res.sort(key=lambda r: r["cells"][0])
    for r in rot_peak:
        print(f"[peak]   rotation p={r['degree']} n={r['cells'][0]} "
              f"retention={r['peak_retention']:.5f}", flush=True)
    for r in vortex_peak:
        print(f"[peak]   vortex   p={r['degree']} n={r['cells'][0]} "
              f"retention={r['peak_retention']:.5f}", flush=True)
    print(f"[mass]   max relative drift {trace['relative_drift_max']:.3e}", flush=True)

    # ---- criteria ------------------------------------------------------
    order_rows = []
    for case, per_degree in studies.items():
        for degree, table in per_degree.items():
            order_rows.append({
                "case": case, "degree": degree,
                "observed_order": table["observed_order"],
                "required_order": table["required_order"],
                "check_applied": table["order_pass"] is not None,
                "pass": table["order_pass"],
            })
    orders_required = [r for r in order_rows if r["check_applied"]]
    orders_pass = all(r["pass"] for r in orders_required)
    mass_values = [m for case in studies.values() for t in case.values()
                   for m in t["mass_drift_rel"]]
    mass_values += [r["mass_drift_rel"] for r in rot_peak + rot_res + vortex_peak]
    mass_values.append(trace["relative_drift_max"])
    mass_max = max(mass_values)
    mass_pass = mass_max <= bench.CRITERIA["mass_tolerance"]
    peak_val = max(r["peak_retention"] for r in rot_peak
                   if r["degree"] == 3) if any(r["degree"] == 3 for r in rot_peak) \
        else max(r["peak_retention"] for r in rot_peak)
    peak_pass = peak_val >= bench.CRITERIA["peak_tolerance"]

    figures = []
    if not args.no_figures:
        try:
            FIGDIR.mkdir(parents=True, exist_ok=True)
            figure_convergence(studies, FIGDIR / "dg_advection_convergence.png")
            figure_peaks(rot_peak, rot_res, vortex_peak,
                         FIGDIR / "dg_advection_peak_preservation.png")
            figure_mass(trace, FIGDIR / "dg_advection_mass_balance.png")
            figure_fields(snap, FIGDIR / "dg_advection_rotation_fields.png")
            figures = [
                {"name": "dg_advection_convergence.png",
                 "caption": "Observed L2 convergence for the three ice-edge transport "
                            "cases, p=1,2,3, with p+1 reference slopes."},
                {"name": "dg_advection_peak_preservation.png",
                 "caption": "Peak retention of the rotating ice-edge front after one "
                            "revolution (left) and versus resolution (right)."},
                {"name": "dg_advection_mass_balance.png",
                 "caption": "Relative global mass drift per RK4 step (reentrant vortex)."},
                {"name": "dg_advection_rotation_fields.png",
                 "caption": "Initial front, DG solution after one revolution and the "
                            "numerical-minus-exact field."},
            ]
        except Exception as exc:  # pragma: no cover
            print(f"[warn] figure generation failed: {exc}", flush=True)

    payload = {
        "benchmark": "dg_advection_bench",
        "campaign_node": "IN12-P5",
        "repo": "TensorDG",
        "branch": "agent/m4933",
        "commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": "quick" if args.quick else "full",
        "environment": environment(),
        "purpose": ("High-order reference for the transport step of ice-condition "
                    "assimilation (A2) and passive frazil transport: rotating front / "
                    "convection-dominated standards with verified DG convergence and "
                    "exact discrete mass conservation."),
        "discretisation": {
            "space": "modal tensor-product Legendre DG on a uniform periodic square",
            "flux": "upwind, node-wise sign of the face-normal velocity",
            "time": "classical RK4, dt = cfl / ((2p+1)(|a_x|/h_x + |a_y|/h_y))",
            "cfl": CFL,
        },
        "criteria": bench.CRITERIA,
        "cases": studies,
        "order_summary": order_rows,
        "peak_preservation": {
            "solid_body_rotation_front": rot_peak,
            "solid_body_rotation_resolution_study": rot_res,
            "reentrant_deformational_vortex": vortex_peak,
        },
        "mass_balance": {"tolerance": bench.CRITERIA["mass_tolerance"],
                         "max_relative_drift_observed": mass_max,
                         "trace": trace},
        "verdict": {
            "orders_pass": bool(orders_pass),
            "mass_pass": bool(mass_pass),
            "peak_pass": bool(peak_pass),
            "pytest_module": "tests/test_dg_advection_bench.py",
            "pytest_min_tests": bench.CRITERIA["pytest_min_tests"],
            "overall": "PASS" if (orders_pass and mass_pass and peak_pass) else "FAIL",
        },
        "figures": figures,
        "limitations": [
            "The rigid-rotation velocity of a periodic square is discontinuous at "
            "the seam (a_y jumps by 2*pi); the rotation convergence order is "
            "therefore measured in the disk r<=0.28 around the rotation centre "
            "(characteristics are circles, so seam noise stays at its own radius). "
            "The global norm of the same run is reported for transparency.",
            "The reentrant vortex uses the classical LeVeque/Nair amplitude A=1/pi "
            "(max speed 1); the peak/shape metrics are for smooth (C-infinity) "
            "fronts. Sharp (C0/discontinuous) ice edges require limiters and are "
            "out of scope for this route-level verification node.",
            "No physical ice model is integrated: this node fixes the *transport "
            "operator* reference only (no frazil thermodynamics, no coupling).",
        ],
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {ARTIFACT.relative_to(ROOT)}")
    print("verdict:", payload["verdict"])
    return 0 if payload["verdict"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())