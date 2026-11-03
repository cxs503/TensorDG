"""Figure + regional error diagnostic for the limiter verification pass.

Produces ``docs/figures/limiter_verification.png``:

* top row: density / velocity / pressure profiles of the exact Riemann
  solution versus the three configurations that matter for the verdict
  (unlimited, limiter inside the SSP-RK loop, limiter applied once to the
  unlimited final state);
* bottom-left: relative L2 error per field (bar chart, numbers read from the
  runs, not hand-copied);
* bottom-right: where the error lives (relative L2 error restricted to the
  left plateau, the rarefaction fan, and the shock/contact region).

Usage
-----
    PYTHONPATH=. python -m benchmarks.plot_limiter_verification
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import sample
from benchmarks.verify_limiter_sod import FINAL_TIME, SOD_LEFT, SOD_RIGHT
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.limiters import apply_limiter

FIELDS = ("density", "velocity", "pressure")
SAMPLES = 6


def _point_values(coeff, lower=0.0, upper=1.0):
    """Sample the modal polynomial at Gauss points inside every cell."""
    values = coeff.detach().cpu().numpy()
    cells, modes, _ = values.shape
    dx = (upper - lower) / cells
    reference = 2.0 * (np.arange(SAMPLES) + 0.5) / SAMPLES - 1.0
    phi = np.stack(
        [np.polynomial.legendre.legval(reference, np.eye(modes)[k]) for k in range(modes)], axis=1
    )
    numerical = np.einsum("cmv,qm->cqv", values, phi).reshape(-1, 3)
    points = (lower + (np.arange(cells)[:, None] + 0.5 * (reference[None, :] + 1.0)) * dx).reshape(-1)
    return points, numerical


def _primitive(numerical):
    rho = numerical[:, 0]
    u = numerical[:, 1] / rho
    p = 0.4 * (numerical[:, 2] - 0.5 * numerical[:, 1] ** 2 / rho)
    return {"density": rho, "velocity": u, "pressure": p}


def _regional_errors(points, primitive, exact, bounds):
    out = {}
    for name, (lo, hi) in bounds.items():
        mask = (points >= lo) & (points < hi)
        row = {}
        for field in FIELDS:
            num = primitive[field][mask]
            ref = exact[field][mask]
            ref_norm = float(np.linalg.norm(ref))
            row[field] = (
                float(np.linalg.norm(num - ref) / ref_norm * 100.0)
                if ref_norm > 1e-12
                else float("nan")
            )
            row[f"{field}_absolute_l2"] = float(np.linalg.norm(num - ref))
        out[name] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    x, coeff_none, meta = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=args.cells, degree=args.degree,
        final_time=FINAL_TIME, cfl=args.cfl, limiter=None,
    )
    dx = 1.0 / args.cells
    _, coeff_tvb, meta_tvb = solve_euler_dg_1d(
        SOD_LEFT, SOD_RIGHT, ncells=args.cells, degree=args.degree,
        final_time=FINAL_TIME, cfl=args.cfl, limiter="tvb", tvb_constant=1.0,
    )
    coeff_oneshot = apply_limiter(coeff_none, dx, kind="tvb", tvb_constant=1.0)

    runs = {"unlimited": coeff_none, "tvb in RK loop": coeff_tvb, "tvb one-shot (final state)": coeff_oneshot}
    errors = {name: riemann_dg_errors(x, c, FINAL_TIME, SOD_LEFT, SOD_RIGHT) for name, c in runs.items()}

    points, numerical = _point_values(coeff_none)
    primitive_none = _primitive(numerical)
    exact = sample(points, FINAL_TIME, SOD_LEFT, SOD_RIGHT)
    regional = {}
    bounds = {"left plateau x<0.25": (0.0, 0.25), "rarefaction 0.25<x<0.5": (0.25, 0.5),
              "contact/shock 0.5<x<0.95": (0.5, 0.95)}
    for name, coeff in runs.items():
        pts, num = _point_values(coeff)
        regional[name] = _regional_errors(pts, _primitive(num), sample(pts, FINAL_TIME, SOD_LEFT, SOD_RIGHT), bounds)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), gridspec_kw={"height_ratios": [1.0, 0.95]})
    colors = {"unlimited": "#1f77b4", "tvb in RK loop": "#d62728", "tvb one-shot (final state)": "#2ca02c"}
    for col, field in enumerate(FIELDS):
        ax = axes[0, col]
        ax.plot(points, exact[field], "k-", lw=2.0, label="exact Riemann")
        for name, coeff in runs.items():
            pts, num = _point_values(coeff)
            ax.plot(pts, _primitive(num)[field], lw=0.9, ls="-", color=colors[name],
                    label=f"{name}: {errors[name][field]['relative_l2_percent']:.2f}%")
        ax.set_title(f"{field} (P{args.degree}, {args.cells} cells)")
        ax.set_xlabel("x")
        ax.set_ylabel(field)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="best")

    ax = axes[1, 0]
    width = 0.26
    xs = np.arange(len(FIELDS))
    for i, (name, err) in enumerate(errors.items()):
        ax.bar(xs + (i - 1) * width, [err[f]["relative_l2_percent"] for f in FIELDS], width,
               color=colors[name], label=name)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3% door")
    ax.set_xticks(xs)
    ax.set_xticklabels(FIELDS)
    ax.set_ylabel("relative L2 error [%]")
    ax.set_title("error vs exact solution")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    regions = ["rarefaction 0.25<x<0.5", "contact/shock 0.5<x<0.95"]
    for i, name in enumerate(runs):
        ax.bar(np.arange(len(regions)) + (i - 1) * width,
               [regional[name][r]["velocity"] for r in regions], width, color=colors[name], label=name)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(["rarefaction", "contact/shock"])
    ax.set_ylabel("velocity relative L2 [%]")
    ax.set_title("where the velocity error lives")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 2]
    state = json.loads(Path("validation/limiter_sod/oneshot_post_fix.json").read_text())["detector_selectivity"]
    ax.plot([s["tvb_constant"] for s in state], [s["flagged_fraction"] * 100 for s in state], "o-")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("TVB constant M")
    ax.set_ylabel("flagged cells [%]")
    ax.set_title("troubled-cell detector selectivity (final state)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "TensorDG TVB/WENO limiter verification — Sod, modal DG P2, HLLC, SSP-RK3\n"
        f"steps: unlimited {meta['steps']}, tvb in loop {meta_tvb['steps']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "limiter_verification.png", dpi=140)
    print(f"wrote {out / 'limiter_verification.png'}")

    diag = {
        "ncells": args.cells, "degree": args.degree, "cfl": args.cfl,
        "steps": {"unlimited": meta["steps"], "tvb_in_loop": meta_tvb["steps"]},
        "relative_l2_percent": {n: {f: errors[n][f]["relative_l2_percent"] for f in FIELDS} for n in errors},
        "max_relative_l2_percent": {
            n: max(errors[n][f]["relative_l2_percent"] for f in FIELDS) for n in errors
        },
        "regional_velocity_percent": {n: {r: regional[n][r]["velocity"] for r in regions} for n in runs},
        "plateau_velocity_absolute_l2": {n: regional[n]["left plateau x<0.25"]["velocity_absolute_l2"] for n in runs},
    }
    path = Path("validation/limiter_sod/regional_diagnostic.json")
    path.write_text(json.dumps(diag, indent=2) + "\n")
    print(json.dumps(diag["max_relative_l2_percent"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()