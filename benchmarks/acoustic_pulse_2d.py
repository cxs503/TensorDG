"""2-D acoustic-pulse benchmark for the high-order Euler DG solver.

Metric definitions
------------------
``rel_l2_pressure``
    ``||p'_h - p'_exact||_2 / ||p'_exact||_2`` over the mesh, where both sides
    are **cell averages** (the numerical field is the modal P0 coefficient, the
    exact field is its 4x4 Gauss-Legendre average).  Comparing averages instead
    of point values removes the O(h^2) projection bias from the error.
``peak_rel_error``
    ``|max|p'_h| - max|p'_exact|| / max|p'_exact|`` with the same cell-average
    convention; this is the amplitude/dissipation measure and the quantity the
    ``CASE-ACOUSTIC-PULSE`` acceptance tolerance (``< 8 %``) is applied to.
``entropy_noise``
    ``max |rho' - p'/c0^2| / A``.  The initial data excites *only* the acoustic
    mode, so this quantity measures how much spurious entropy the scheme
    generates; for a consistent solver it is O(A) and must stay far below 1.
``phase_error_pct``
    Radial position of the first zero crossing of ``p'(r,t)`` along the ray
    ``y = 0`` (cell averages bilinearly interpolated onto the ray), relative to
    the exact crossing (``%`` of the exact radius): the dispersion measure,
    i.e. the numerical phase speed error accumulated over ``c0 t``.
``energy_ratio``
    Discrete linear acoustic energy ``E = sum( p'^2/(2 rho0 c0^2) + rho0 |u'|^2/2 ) dA``
    at ``t`` divided by the **discrete initial** energy on the same mesh (the L2
    projection is cell-average preserving and the initial velocity is zero, so
    this denominator is exact).  ``E`` is a conserved quantity of the linear
    system, hence ``1 - energy_ratio`` is the numerical dissipation of the
    scheme alone, without the projection loss of the continuum value.

Usage
-----
``python -m benchmarks.acoustic_pulse_2d --quick``   (smoke, seconds)
``python -m benchmarks.acoustic_pulse_2d``           (full ladder, minutes)
Artifacts land in ``runs/acoustic_pulse_2d/``; the module prints one JSON
document to stdout so the numbers can be injected into the report instead of
being transcribed by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from tensordg.acoustics import (
    acoustic_pulse_exact_pressure,
    acoustic_pulse_initial_state,
    acoustic_pulse_sound_speed,
    alpha_from_width,
)
from tensordg.euler_dg_2d import cell_fields, solve_euler_dg_2d

GAMMA = 1.4
RHO0 = 1.0
P0 = 1.0
WIDTH = 0.4
AMPLITUDE = 1.0e-3
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 2.0
CFL = 0.4

SOUND_SPEED = acoustic_pulse_sound_speed(GAMMA, RHO0, P0)


def _initial(x, y, amplitude=AMPLITUDE):
    return acoustic_pulse_initial_state(
        x, y, width=WIDTH, amplitude=amplitude, gamma=GAMMA, rho0=RHO0, p0=P0
    )


def _exact_cell_averages(centers, spacing, time, amplitude=AMPLITUDE, quadrature=4):
    """4x4 Gauss-Legendre cell average of the exact pressure perturbation."""
    nodes, weights = np.polynomial.legendre.leggauss(quadrature)
    x = centers[:, None, None, None] + 0.5 * spacing[0] * nodes[None, None, :, None]
    y = centers[None, :, None, None] + 0.5 * spacing[1] * nodes[None, None, None, :]
    radius = np.sqrt(x * x + y * y)
    values = acoustic_pulse_exact_pressure(
        radius, time, width=WIDTH, amplitude=amplitude, sound_speed=SOUND_SPEED
    )
    return 0.25 * np.einsum("ijab,a,b->ij", values, weights, weights)


def run_leg(ncells, degree, final_time=FINAL_TIME, cfl=CFL, amplitude=AMPLITUDE,
            limiter=None, threads=None):
    """Run one (mesh, degree) leg and return its metrics."""
    if threads is not None:
        torch.set_num_threads(int(threads))

    def initial(x, y):
        return _initial(x, y, amplitude)

    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=DOMAIN,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        periodic=(True, True),
    )
    wall = time.time() - started
    fields = cell_fields(coeff)
    u = amplitude / SOUND_SPEED  # reference velocity scale of the pulse
    pressure = fields["pressure"] - P0
    density = fields["density"] - RHO0
    momentum_x = fields["velocity_x"] * fields["density"]

    centers_x = x.numpy()
    centers_y = y.numpy()
    spacing = ((DOMAIN[0][1] - DOMAIN[0][0]) / ncells, (DOMAIN[1][1] - DOMAIN[1][0]) / ncells)
    exact = _exact_cell_averages(centers_x, spacing, final_time, amplitude)

    num = pressure.numpy()
    den = density.numpy()
    err = num - exact
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(exact))
    peak_num = float(np.max(np.abs(num)))
    peak_exact = float(np.max(np.abs(exact)))
    peak_rel = float(abs(peak_num - peak_exact) / peak_exact)
    entropy = float(np.max(np.abs(den - num / (SOUND_SPEED ** 2))) / amplitude)

    # --- dispersion: first zero crossing of p' along the ray y = 0 ----------
    ray = np.linspace(0.0, 0.9 * DOMAIN[0][1], 4001)
    interp = RegularGridInterpolator(
        (centers_x, centers_y), num, method="linear", bounds_error=False, fill_value=None
    )
    ray_num = interp(np.column_stack([ray, np.zeros_like(ray)]))
    ray_exact = acoustic_pulse_exact_pressure(
        ray, final_time, width=WIDTH, amplitude=amplitude, sound_speed=SOUND_SPEED
    )
    crossing_num = _first_crossing(ray, ray_num)
    crossing_exact = _first_crossing(ray, ray_exact)
    phase_error = (
        float(abs(crossing_num - crossing_exact) / crossing_exact * 100.0)
        if crossing_num is not None and crossing_exact is not None
        else None
    )

    # --- conserved linear acoustic energy ----------------------------------
    velocity_x = fields["velocity_x"].numpy()
    velocity_y = fields["velocity_y"].numpy()
    cell_area = spacing[0] * spacing[1]
    energy = float(
        np.sum(
            num * num / (2.0 * RHO0 * SOUND_SPEED ** 2)
            + 0.5 * RHO0 * (velocity_x ** 2 + velocity_y ** 2)
        )
        * cell_area
    )
    # Normalise by the *discrete* initial energy (the L2 projection is
    # cell-average preserving, so the initial pressure perturbation average is
    # the exact cell average; the initial velocity is zero).  Normalising by the
    # analytic continuum energy would fold the projection loss into the metric.
    initial_averages = _exact_cell_averages(centers_x, spacing, 0.0, amplitude)
    energy_exact0 = float(
        np.sum(initial_averages ** 2 / (2.0 * RHO0 * SOUND_SPEED ** 2)) * cell_area
    )

    return {
        "ncells": ncells,
        "degree": degree,
        "dofs": meta["dofs"],
        "steps": meta["steps"],
        "dt_last": meta["dt"],
        "cfl": cfl,
        "final_time": final_time,
        "amplitude": amplitude,
        "limiter": limiter,
        "wall_seconds": wall,
        "rel_l2_pressure": rel_l2,
        "rel_l2_density": float(
            np.linalg.norm(den - exact / (SOUND_SPEED ** 2)) / np.linalg.norm(exact / (SOUND_SPEED ** 2))
        ),
        "peak_pressure_numeric": peak_num,
        "peak_pressure_exact": peak_exact,
        "peak_rel_error": peak_rel,
        "entropy_noise_relative": entropy,
        "max_momentum": float(np.max(np.abs(momentum_x.numpy()))),
        "momentum_scale": amplitude / SOUND_SPEED,
        "phase_error_pct": phase_error,
        "zero_crossing_numeric": crossing_num,
        "zero_crossing_exact": crossing_exact,
        "energy_ratio": energy / energy_exact0,
        "_fields": {
            "x": centers_x,
            "y": centers_y,
            "pressure": num,
            "exact": exact,
            "density": den,
            "momentum_x": momentum_x.numpy(),
            "velocity_scale": u,
            "ray": ray,
            "ray_numeric": ray_num,
            "ray_exact": ray_exact,
        },
    }


def _first_crossing(radii, values):
    """Radius of the first sign change of ``values`` (None if it never changes)."""
    sign = np.sign(values)
    for i in range(1, len(sign)):
        if sign[i] != 0.0 and sign[i] != sign[0]:
            r0, r1 = radii[i - 1], radii[i]
            v0, v1 = values[i - 1], values[i]
            return float(r0 - v0 * (r1 - r0) / (v1 - v0))
    return None


def _write_figures(results, fields, outdir):
    """Two figures: final-time field + ray cut, and the error/convergence ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)

    x, y = fields["x"], fields["y"]
    pressure = fields["pressure"]
    limit = float(np.max(np.abs(pressure)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    image = axes[0].pcolormesh(x, y, pressure.T * 1e3, cmap="RdBu_r",
                               vmin=-limit * 1e3, vmax=limit * 1e3, shading="auto")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("p' [1e-3] at t = %.2f (DG p=%d, %d cells)"
                      % (FINAL_TIME, results["degree"], results["ncells"]))
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].plot(fields["ray"], fields["ray_exact"] * 1e3, "k-", lw=2, label="exact (Hankel integral)")
    axes[1].plot(fields["ray"], fields["ray_numeric"] * 1e3, "o", ms=2.5, label="DG p=%d" % results["degree"])
    axes[1].set_xlabel("r along y=0")
    axes[1].set_ylabel("p' [1e-3]")
    axes[1].set_title("waveform vs exact, rel L2 = %.2e" % results["rel_l2_pressure"])
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "field_and_ray.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for degree in sorted({r["degree"] for r in results}):
        legs = sorted((r for r in results if r["degree"] == degree), key=lambda r: r["ncells"])
        h = [1.0 / r["ncells"] for r in legs]
        axes[0].loglog(h, [r["rel_l2_pressure"] for r in legs], "o-", label="p=%d" % degree)
        axes[1].semilogx(h, [r["peak_rel_error"] * 100 for r in legs], "o-", label="p=%d" % degree)
    axes[0].set_xlabel("h = 1/N")
    axes[0].set_ylabel("rel L2 error of p'")
    axes[0].set_title("L2 convergence")
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()
    axes[1].axhline(8.0, color="r", ls="--", lw=1, label="8% acceptance tolerance")
    axes[1].set_xlabel("h = 1/N")
    axes[1].set_ylabel("peak amplitude error [%]")
    axes[1].set_title("dissipation vs the 8% gate")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "convergence.png"), dpi=150)
    plt.close(fig)
    return sorted(os.listdir(figdir))


def _write_artifacts(results, outdir, fields=None):
    payload = {
        "case": "CASE-ACOUSTIC-PULSE",
        "reference": "Tam & Webb 1993 / Tam & Dong 1996 acoustic pulse (linearised Euler), "
        "exact via Hankel integral; on-axis asymptote -A b^2/(2 ln2 c0^2 t^2)",
        "background": {"gamma": GAMMA, "rho0": RHO0, "p0": P0, "c0": SOUND_SPEED},
        "pulse": {"width": WIDTH, "amplitude": AMPLITUDE, "alpha": alpha_from_width(WIDTH)},
        "domain": DOMAIN,
        "final_time": FINAL_TIME,
        "cfl": CFL,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "legs": results,
    }
    ray_csv = os.path.join(outdir, "ray_profile.csv")
    if fields is not None:
        with open(ray_csv, "w") as fh:
            fh.write("r,pressure_numeric,pressure_exact\n")
            for r, a, b in zip(fields["ray"], fields["ray_numeric"], fields["ray_exact"]):
                fh.write("%.6f,%.12e,%.12e\n" % (r, a, b))
        payload["figures"] = _write_figures(results[-1] if results else {}, fields, outdir)
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny ladder (smoke test)")
    parser.add_argument("--outdir", default="runs/acoustic_pulse_2d")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--degrees", type=int, nargs="+", default=None, help="override the degree ladder"
    )
    parser.add_argument(
        "--ncells", type=int, nargs="+", default=None, help="override the mesh ladder"
    )
    parser.add_argument(
        "--nonlinearity",
        action="store_true",
        help="add an A=1e-2 leg to quantify the departure from linear acoustics",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    if args.quick:
        degrees = args.degrees or [1, 2]
        meshes = args.ncells or [24, 32]
        final_time = 0.4
    else:
        degrees = args.degrees or [1, 2, 3]
        meshes = args.ncells or [32, 48, 64]
        final_time = FINAL_TIME

    results = []
    finest = None
    for degree in degrees:
        for ncells in meshes:
            leg = run_leg(ncells, degree, final_time=final_time, threads=args.threads)
            fields = leg.pop("_fields")
            if not results or (leg["ncells"], leg["degree"]) >= (results[-1]["ncells"], results[-1]["degree"]):
                finest = fields
            results.append(leg)
            print(json.dumps(leg), flush=True)

    if args.nonlinearity:
        leg = run_leg(meshes[-1], degrees[-1], final_time=final_time,
                      amplitude=10.0 * AMPLITUDE, threads=args.threads)
        leg.pop("_fields")
        leg["note"] = "nonlinearity probe: A=1e-2, compare with the A=1e-3 leg"
        results.append(leg)

    payload = _write_artifacts(results, args.outdir, fields=finest)
    print(json.dumps({"artifacts": args.outdir, "legs": len(results)}))
    return payload


if __name__ == "__main__":
    main()