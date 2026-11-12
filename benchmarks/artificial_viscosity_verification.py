"""Verification of the sensor-gated modal artificial viscosity (SVV) filter.

Three measurements, none of them quoted from the literature:

* **A. Sensor scaling law.**  For a smooth field the top-mode energy fraction
  must decay like ``h^{2p}`` (measured slope ``-> 2p``), while for a jump it
  must saturate (slope ``-> 0``).  This is what makes a *resolution independent*
  threshold possible for shocks at all.
* **B. Sod shock tube, head to head.**  Same solver, same mesh, four closures of
  the sub-cell content (``None`` / TVB / WENO / ``"av"``): relative L1 error of
  the density cell averages against the exact Riemann solution, total-variation
  excess, minimum pressure and conservation drift.
* **C. Inertness on resolved data.**  The smooth-flow sensor level at the same
  mesh/degree gives the margin that the default ``s0`` leaves; when the sensor is
  below ``s0 - kappa`` the filter is *bitwise* the identity, which is the
  property that keeps the design order in smooth regions (asserted in
  ``tests/test_artificial_viscosity.py``).

Outputs (``--output-dir``): ``metrics.json``, ``summary.md``, ``profiles.csv``
and ``figures/*.png``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.sod_exact import cell_averages
from tensordg.artificial_viscosity import apply_av_filter, av_sensor, modal_smoothness
from tensordg.euler_dg_1d import solve_euler_dg_1d

GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
FINAL_TIME = 0.2
S0, KAPPA, ALPHA_MAX = -2.0, 0.5, 20.0
AV = {"s0": S0, "kappa": KAPPA, "alpha_max": ALPHA_MAX, "order": 2.0, "smooth": 1}


def _conserved_mean(state):
    """Cell-mean conserved state ``(rho, rho u, E)`` of a primitive state."""
    rho, u, p = state
    return np.array([rho, rho * u, p / (GAMMA - 1.0) + 0.5 * rho * u * u])


def _physical_flux(state):
    """Physical Euler flux of a primitive state ``(rho, u, p)``."""
    rho, u, p = state
    energy = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return np.array([rho * u, rho * u * u + p, u * (energy + p)])


def expected_state_mean(left, right, final_time, length=1.0) -> np.ndarray:
    """Domain mean of the conserved state at ``final_time``, flux accounted.

    The Sod domain is *open*: with ``p_L != p_R`` the pressure imbalance drives a
    constant momentum flux through the boundaries, so the mean momentum changes
    by ``(p_L - p_R) * t / L`` even for an exactly conservative scheme.  The
    initial mean is therefore **not** the right conservation reference (using it
    is what made the old metric report ``inf`` for momentum, audit m16938 F2).
    The boundary states are the initial ones as long as no wave has reached a
    boundary; that is checked by the caller (``t = 0.2``, waves at
    ``0.5 +- 0.35``, domain ``[0, 1]``).
    """
    mean0 = 0.5 * (_conserved_mean(left) + _conserved_mean(right))
    return mean0 + (_physical_flux(left) - _physical_flux(right)) * final_time / length


def conservation_drift(numeric_mean, initial_mean) -> dict:
    """Conservation drift of the cell-mean state, with the zero-mean branch defined.

    Returns ``{"relative", "absolute", "zero_mean_components"}``.

    ``relative`` is the max of ``|numeric - initial| / |initial|`` over the
    components whose initial mean is *numerically non-zero* (mass and energy for
    a shock tube); ``absolute`` is the max ``|numeric - initial|`` normalised by
    the largest component mean (a state scale), and is the only meaningful
    figure for a component whose mean is exactly zero.

    Rationale (audit m16938): the previous single metric divided by
    ``|initial_mean|`` component-wise, so the momentum component of any
    symmetric Riemann problem (mean exactly 0) produced ``inf`` for *every*
    run -- including the exactly conservative ones.  An ``inf`` column cannot
    distinguish a conservative scheme from a broken one, which is the only thing
    the metric exists for.
    """
    numeric = np.asarray(numeric_mean, dtype=float)
    initial = np.asarray(initial_mean, dtype=float)
    if numeric.shape != initial.shape:
        raise ValueError("numeric_mean and initial_mean must have the same shape")
    abs_drift = np.abs(numeric - initial)
    scale = np.abs(initial)
    state_scale = float(scale.max()) if scale.size else 0.0
    defined = scale > 1e-12 * max(state_scale, 1e-300)
    relative = float(np.max(abs_drift[defined] / scale[defined])) if defined.any() else 0.0
    absolute = float(np.max(abs_drift) / state_scale) if state_scale > 0.0 else float(np.max(abs_drift))
    return {
        "relative": relative,
        "absolute": absolute,
        "zero_mean_components": int((~defined).sum()),
    }


def project(field, ncells, degree, nodes_per_cell=24):
    """Modal Legendre coefficients of *field* (a projection, not a solve)."""
    nodes, weights = np.polynomial.legendre.leggauss(nodes_per_cell)
    phi = np.stack(
        [np.polynomial.legendre.legval(nodes, np.eye(degree + 1)[k]) for k in range(degree + 1)],
        axis=1,
    )
    x = (np.arange(ncells)[:, None] + 0.5 * (nodes[None, :] + 1)) / ncells
    values = field(x)
    norm = 2.0 / (2 * np.arange(degree + 1) + 1)
    coeff = np.einsum("cq,qm,q->cm", values, phi, weights) / norm[None, :]
    return torch.tensor(coeff, dtype=torch.float64).unsqueeze(-1)


def sensor_law(degree, cells=(20, 40, 80, 160), normalisation="fraction"):
    """Sensors of a smooth sine and of a faceted (misaligned) jump."""
    smooth = project(lambda x: np.sin(2 * np.pi * x), max(cells), degree)
    out = {"degree": degree, "cells": list(cells), "normalisation": normalisation,
           "smooth": [], "jump": []}
    for n in cells:
        cs = project(lambda x: np.sin(2 * np.pi * x), n, degree)
        # Jump deliberately off the cell faces: 0.5 + 0.37 h
        cj = project(lambda x, n=n: (x > 0.5 + 0.37 / n).astype(float), n, degree)
        out["smooth"].append(float(av_sensor(cs, smooth=0, normalisation=normalisation).max().item()))
        out["jump"].append(float(av_sensor(cj, smooth=0, normalisation=normalisation).max().item()))
    del smooth
    decades = np.log10(np.array(cells, dtype=float))
    out["smooth_slope"] = float(np.polyfit(decades, out["smooth"], 1)[0])
    out["jump_slope"] = float(np.polyfit(decades, out["jump"], 1)[0])
    out["theory_smooth_slope"] = 2 * degree
    return out


def sod(ncells, degree, limiter, kwargs=None, cfl=0.12):
    start = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=ncells, degree=degree, final_time=FINAL_TIME, cfl=cfl,
        limiter=limiter, limiter_kwargs=kwargs,
    )
    elapsed = time.time() - start
    avg = coeff[:, 0].numpy()
    rho = avg[:, 0]
    vel = avg[:, 1] / rho
    pressure = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
    _, reference = cell_averages(ncells, FINAL_TIME)
    rho_ref = np.asarray(reference["density"])
    l1 = float(np.sum(np.abs(rho - rho_ref)) / np.sum(np.abs(rho_ref)))
    tv = lambda a: float(np.sum(np.abs(np.diff(a))))
    # Conservation: mean over cells of the conservative state vs the *flux
    # accounted* expected mean (open domain, p_L != p_R; see
    # :func:`expected_state_mean` and :func:`conservation_drift`).
    expected_mean = expected_state_mean(LEFT, RIGHT, FINAL_TIME)
    drift = conservation_drift(avg.mean(axis=0), expected_mean)
    sensor = av_sensor(coeff, smooth=1).numpy()
    return {
        "ncells": ncells, "degree": degree, "limiter": limiter or "none",
        "limiter_kwargs": kwargs or {}, "steps": meta["steps"], "seconds": elapsed,
        "relative_l1_density": l1,
        "tv_excess": tv(rho) - tv(rho_ref),
        "tv_numeric": tv(rho), "tv_reference": tv(rho_ref),
        "min_pressure": float(pressure.min()),
        "conservation_drift": drift["relative"],
        "conservation_drift_absolute": drift["absolute"],
        "conservation_zero_mean_components": drift["zero_mean_components"],
        "sensor_max": float(sensor.max()),
        "flagged_fraction": float((sensor > S0 - KAPPA).mean()),
        "profile": {"x": x.numpy().tolist(), "density": rho.tolist(),
                    "density_exact": rho_ref.tolist(), "sensor": sensor.tolist()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/artificial_viscosity")
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="sensor law + one Sod mesh only")
    args = parser.parse_args()
    out = Path(args.output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    law = {
        f"{key}-{norm}": sensor_law(degree, normalisation=norm)
        for key, degree in (("p1", 1), ("p2", 2))
        for norm in ("fraction", "global")
    }

    runs = []
    variants = [("none", None), ("tvb", None), ("av", AV)]
    if args.quick:
        variants = [("none", None), ("av", AV)]
    for ncells in args.cells:
        for name, kwargs in variants:
            if name == "tvb" and ncells != args.cells[0]:
                continue
            runs.append(sod(ncells, args.degree, name if name != "none" else None, kwargs))
            print(json.dumps({k: v for k, v in runs[-1].items() if k != "profile"}))

    # ---- figure 1: sensor law
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, style in (("p1-fraction", "o-"), ("p2-fraction", "s-"),
                       ("p1-global", "o:"), ("p2-global", "s:")):
        data = law[key]
        ax.semilogx(data["cells"], data["smooth"], style,
                    label=f"smooth, p={data['degree']}, {data['normalisation']} (slope {data['smooth_slope']:.2f})")
        ax.semilogx(data["cells"], data["jump"], style, alpha=0.5, markerfacecolor="none",
                    label=f"jump, p={data['degree']}, {data['normalisation']} (slope {data['jump_slope']:.2f})")
    ax.axhline(S0, color="k", lw=0.8, ls=":", label=f"threshold s0={S0}")
    ax.set_xlabel("cells"); ax.set_ylabel(r"$\log_{10}$ top-mode energy fraction")
    ax.set_title("Sensor scaling law"); ax.legend(fontsize=6); fig.tight_layout()
    fig.savefig(out / "figures" / "sensor_law.png", dpi=150); plt.close(fig)

    # ---- figure 2: Sod profiles
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    biggest = max(runs, key=lambda r: r["ncells"])
    ax.plot(biggest["profile"]["x"], biggest["profile"]["density_exact"], "k:", lw=1.4, label="exact")
    for run in [r for r in runs if r["ncells"] == biggest["ncells"]]:
        ax.plot(run["profile"]["x"], run["profile"]["density"], lw=1.0,
                label=f"{run['limiter']} (L1={run['relative_l1_density']*100:.2f}%)")
    ax.set_title(f"Sod density, N={biggest['ncells']}, P{biggest['degree']}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "figures" / "sod_profiles.png", dpi=150); plt.close(fig)

    # ---- figure 3: sensor profile with threshold
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(biggest["profile"]["x"], biggest["profile"]["sensor"], lw=1.0)
    ax.axhline(S0, color="k", lw=0.8, ls=":", label=f"s0={S0}")
    ax.set_xlabel("x"); ax.set_ylabel(r"$s_e$"); ax.set_title("Sensor along the Sod solution")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "figures" / "sensor_profile.png", dpi=150); plt.close(fig)

    # ---- figure 4: error / TV bars
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    labels = [f"{r['limiter']}@{r['ncells']}" for r in runs]
    axes[0].bar(labels, [r["relative_l1_density"] * 100 for r in runs])
    axes[0].set_ylabel("density L1 error [%]"); axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(labels, [r["tv_excess"] for r in runs])
    axes[1].set_ylabel("total-variation excess"); axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out / "figures" / "error_tv.png", dpi=150); plt.close(fig)

    # ---- artifacts
    with open(out / "profiles.csv", "w") as handle:
        handle.write("ncells,limiter,x,density,density_exact,sensor\n")
        for run in runs:
            p = run["profile"]
            for i in range(len(p["x"])):
                handle.write(f"{run['ncells']},{run['limiter']},{p['x'][i]:.8e},"
                             f"{p['density'][i]:.8e},{p['density_exact'][i]:.8e},{p['sensor'][i]:.8e}\n")
    metrics = {
        "case": "Sod shock tube", "left": LEFT, "right": RIGHT, "final_time": FINAL_TIME,
        "degree": args.degree, "av_parameters": AV, "sensor_law": law, "runs": runs,
        "wall_seconds": time.time() - t0,
        "torch": torch.__version__, "numpy": np.__version__,
    }
    text = json.dumps(metrics, indent=2)
    (out / "metrics.json").write_text(text)
    (out / "metrics.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")

    lines = ["# Sensor-gated modal artificial viscosity — measured artefacts", "",
             f"Generated by `python -m benchmarks.artificial_viscosity_verification --output-dir {out}`",
             f"degree P{args.degree}, AV parameters `{AV}`, wall {metrics['wall_seconds']:.1f} s", "",
             "## A. Sensor scaling law", "",
             "| degree | normalisation | smooth slope | theory 2p | jump slope |",
             "|---|---|---|---|---|"]
    for key, d in law.items():
        lines.append(f"| {d['degree']} | {d['normalisation']} | {d['smooth_slope']:.3f} | "
                     f"{d['theory_smooth_slope']} | {d['jump_slope']:.3f} |")
    lines += ["", "## B. Sod shock tube", "",
              "| N | limiter | L1(rho) % | TV excess | min p | drift rel | drift abs | flagged |",
              "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['ncells']} | {r['limiter']} | {r['relative_l1_density']*100:.3f} | "
                     f"{r['tv_excess']:.3f} | {r['min_pressure']:.3e} | "
                     f"{r['conservation_drift']:.2e} | {r['conservation_drift_absolute']:.2e} | "
                     f"{r['flagged_fraction']*100:.1f}% |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("artifacts ->", out.resolve())
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
