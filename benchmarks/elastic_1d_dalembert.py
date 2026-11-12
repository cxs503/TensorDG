"""Verify the 1-D DG elastodynamics solver against the d'Alembert exact solution.

Two cases (both closed form, see ``tensordg/elastic_1d.py``):

1. ``pulse`` -- periodic bar, c = sqrt(E/rho) = 1, released with a Gaussian
   velocity pulse v0(x) = exp(-((x-0.5)/0.05)^2).  The exact solution is the
   d'Alembert superposition of two counter-propagating pulses.  Reported
   metrics: relative L2 error of v and sigma, observed order of accuracy on a
   mesh sequence, and the position/amplitude of the two stress pulses (a
   numerical-dispersion probe: the pulses must travel at +-c).
2. ``standing`` -- bar clamped at x=0 and free at x=L, released from the first
   longitudinal eigenmode.  Reported metrics: relative L2 error of v and sigma,
   observed order, and the discrete elastic-energy drift (one period of the
   oscillation must be energy conserving).

Usage::

    python -m benchmarks.elastic_1d_dalembert --output-dir validation/elastic_1d

writes ``elastic_1d_dalembert.json`` plus PNG figures into the output directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordg.elastic_1d import (
    dalembert_pulse,
    elastic_energy,
    peak_stress,
    relative_error,
    solve_elastic_1d,
    standing_wave,
    stress_profile,
)

PULSE_CENTER = 0.5
PULSE_WIDTH = 0.05


def pulse_case(meshes, degree, final_time, cfl, nsample):
    """Mesh sequence for the periodic d'Alembert pulse."""
    rows = []
    for n in meshes:
        centers, coeff, meta = solve_elastic_1d(
            ncells=n, degree=degree, final_time=final_time, cfl=cfl,
            bc="periodic",
        )
        x, v, s = stress_profile(centers, coeff, nsample=nsample)
        v_ex, s_ex = dalembert_pulse(x, final_time)
        c = meta["c"]
        shift = c * final_time
        # The pulse released at rest splits into a left-going wave
        # (sigma = +rho c/2) and a right-going wave (sigma = -rho c/2).
        x_left, s_left = peak_stress(x, s, 0.0, PULSE_CENTER)
        x_right, s_right = peak_stress(x, s, PULSE_CENTER, 1.0)
        x_left_ex, s_left_ex = peak_stress(x, s_ex, 0.0, PULSE_CENTER)
        x_right_ex, s_right_ex = peak_stress(x, s_ex, PULSE_CENTER, 1.0)
        centroid_left = _centroid(x, s, 0.0, PULSE_CENTER)
        centroid_right = _centroid(x, s, PULSE_CENTER, 1.0)
        rows.append({
            "cells": n,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "rel_l2_v": relative_error(x, v, v_ex),
            "rel_l2_sigma": relative_error(x, s, s_ex),
            "energy": elastic_energy(centers, coeff),
            "peak_left": {"x": x_left, "sigma": s_left},
            "peak_right": {"x": x_right, "sigma": s_right},
            "peak_left_exact": {"x": x_left_ex, "sigma": s_left_ex},
            "peak_right_exact": {"x": x_right_ex, "sigma": s_right_ex},
            "centroid_left": centroid_left,
            "centroid_right": centroid_right,
            "centroid_exact": {"left": PULSE_CENTER - shift,
                               "right": PULSE_CENTER + shift},
            "pulse_speed_error": max(
                abs(centroid_left - (PULSE_CENTER - shift)),
                abs(centroid_right - (PULSE_CENTER + shift)),
            ) / shift if shift else 0.0,
            "energy_exact": 0.5 * math.sqrt(math.pi / 2.0) * PULSE_WIDTH,
        })
    for a, b in zip(rows[:-1], rows[1:]):
        b["eoc_sigma"] = math.log(a["rel_l2_sigma"] / b["rel_l2_sigma"], 2.0)
        b["eoc_v"] = math.log(a["rel_l2_v"] / b["rel_l2_v"], 2.0)
    return rows


def _centroid(x, field, lo, hi):
    """First moment of ``|field|`` inside ``[lo, hi]`` (smooth pulse position)."""
    mask = (x >= lo) & (x <= hi)
    weight = field[mask].abs()
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("zero mass in window")
    return float((x[mask] * weight).sum()) / total


def standing_case(meshes, degree, final_time, cfl, nsample):
    """Mesh sequence for the clamped-free eigenmode release."""
    rows = []
    for n in meshes:
        centers, coeff, meta = solve_elastic_1d(
            ncells=n, degree=degree, final_time=final_time, cfl=cfl,
            bc="fixed-free",
        )
        x, v, s = stress_profile(centers, coeff, nsample=nsample)
        v_ex, s_ex, _u_ex = standing_wave(x, final_time)
        energy = elastic_energy(centers, coeff)
        k = math.pi / 2.0
        energy_exact = 0.25 * (k ** 2)          # 0.5 * int (E A k cos)^2 dx
        rows.append({
            "cells": n,
            "steps": meta["steps"],
            "dt": meta["dt"],
            "rel_l2_v": relative_error(x, v, v_ex),
            "rel_l2_sigma": relative_error(x, s, s_ex),
            "energy": energy,
            "energy_exact": energy_exact,
            "energy_drift": abs(energy - energy_exact) / energy_exact,
            "sigma_norm_at_wall": float(
                torch_max_abs(s[x < 1e-9]) if bool((x < 1e-9).any()) else 0.0),
            "sigma_at_free_end": float(
                torch_max_abs(s[x > 1.0 - 1e-9]) if bool((x > 1.0 - 1e-9).any()) else 0.0),
        })
    for a, b in zip(rows[:-1], rows[1:]):
        b["eoc_sigma"] = math.log(a["rel_l2_sigma"] / b["rel_l2_sigma"], 2.0)
    return rows


def torch_max_abs(t):
    return float(abs(t).max())


def figure_profiles(pulse_rows, standing_rows, final_time, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    best = pulse_rows[-1]
    centers, coeff, meta = solve_elastic_1d(
        ncells=best["cells"], degree=2, final_time=final_time, cfl=0.3,
        bc="periodic")
    x, v, s = stress_profile(centers, coeff, nsample=6)
    v_ex, s_ex = dalembert_pulse(x, final_time)
    axes[0, 0].plot(x, v_ex, "-", color="0.35", label="exact")
    axes[0, 0].plot(x, v, "--", color="C0", lw=1.2,
                    label=f"DG P2, {best['cells']} cells")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("v")
    axes[0, 0].set_title("d'Alembert pulse: velocity")
    axes[0, 0].legend()

    axes[0, 1].plot(x, s_ex, "-", color="0.35", label="exact")
    axes[0, 1].plot(x, s, "--", color="C3", lw=1.2,
                    label=f"DG P2, {best['cells']} cells")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel(r"$\sigma$")
    axes[0, 1].set_title("stress split into two counter-propagating pulses")
    axes[0, 1].legend()

    for row in pulse_rows:
        axes[1, 0].loglog(row["cells"], row["rel_l2_sigma"], "o-",
                          label=f"sigma (EOC {row.get('eoc_sigma', float('nan')):.2f})")
        axes[1, 0].loglog(row["cells"], row["rel_l2_v"], "s--",
                          label="v")
    axes[1, 0].set_xlabel("cells"); axes[1, 0].set_ylabel("relative L2 error")
    axes[1, 0].set_title("pulse case: convergence")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()

    best_s = standing_rows[-1]
    centers, coeff, meta = solve_elastic_1d(
        ncells=best_s["cells"], degree=2, final_time=final_time, cfl=0.3,
        bc="fixed-free")
    xs, vs, ss = stress_profile(centers, coeff, nsample=6)
    vs_ex, ss_ex, _ = standing_wave(xs, final_time)
    axes[1, 1].plot(xs, ss_ex, "-", color="0.35", label="exact")
    axes[1, 1].plot(xs, ss, "--", color="C2", lw=1.2,
                    label=f"DG P2, {best_s['cells']} cells")
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel(r"$\sigma$")
    axes[1, 1].set_title(f"clamped-free eigenmode, t = {final_time}")
    axes[1, 1].legend()

    fig.suptitle("TensorDG linear elastodynamics verification")
    fig.tight_layout()
    path = out_dir / "elastic_1d_dalembert.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--pulse-time", type=float, default=0.2)
    parser.add_argument("--standing-time", type=float, default=0.5)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--meshes", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--standing-meshes", type=int, nargs="+", default=[40, 80, 160])
    parser.add_argument("--output-dir", default="validation/elastic_1d")
    parser.add_argument("--max-relative-error", type=float, default=0.02)
    args = parser.parse_args()

    nsample = max(2 * args.degree + 3, 6)
    pulse = pulse_case(args.meshes, args.degree, args.pulse_time, args.cfl, nsample)
    standing = standing_case(args.standing_meshes, args.degree,
                             args.standing_time, args.cfl, nsample)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = figure_profiles(pulse, standing, args.pulse_time, out_dir)

    worst_pulse = max(r["rel_l2_sigma"] for r in pulse)
    worst_standing = max(r["rel_l2_sigma"] for r in standing)
    gate = {
        "criteria": {
            "rel_l2_sigma_pulse": args.max_relative_error,
            "rel_l2_sigma_standing": args.max_relative_error,
        },
        "worst": {
            "rel_l2_sigma_pulse": worst_pulse,
            "rel_l2_sigma_standing": worst_standing,
        },
        "passed": bool(worst_pulse < args.max_relative_error
                       and worst_standing < args.max_relative_error),
    }
    payload = {
        "case": "1-D linear elastodynamics (velocity-stress DG)",
        "parameters": {
            "degree": args.degree, "cfl": args.cfl,
            "pulse_time": args.pulse_time,
            "standing_time": args.standing_time,
            "meshes": args.meshes, "standing_meshes": args.standing_meshes,
            "rho": 1.0, "young": 1.0, "c": 1.0,
            "pulse_center": PULSE_CENTER, "pulse_width": PULSE_WIDTH,
            "reference": "Graff 1975 (d'Alembert bar waves); LeVeque 2002 Ch. 3 "
                         "(characteristic BCs)",
        },
        "pulse_case": pulse,
        "standing_case": standing,
        "gate": gate,
        "figure": str(figure),
    }
    json_path = out_dir / "elastic_1d_dalembert.json"
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"{'case':>9} {'cells':>6} {'steps':>7} {'relL2 v':>11} "
          f"{'relL2 sigma':>12} {'EOC sigma':>9}")
    for row in pulse:
        print(f"{'pulse':>9} {row['cells']:>6d} {row['steps']:>7d} "
              f"{row['rel_l2_v']:>11.3e} {row['rel_l2_sigma']:>12.3e} "
              f"{row.get('eoc_sigma', float('nan')):>9.3f}")
    for row in standing:
        print(f"{'standing':>9} {row['cells']:>6d} {row['steps']:>7d} "
              f"{row['rel_l2_v']:>11.3e} {row['rel_l2_sigma']:>12.3e} "
              f"{row.get('eoc_sigma', float('nan')):>9.3f}")
    print(f"gate (<{args.max_relative_error:.0%}): {'PASS' if gate['passed'] else 'FAIL'}"
          f"  worst pulse {worst_pulse:.3e}, worst standing {worst_standing:.3e}")
    print(f"json: {json_path}")
    print(f"figure: {figure}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())