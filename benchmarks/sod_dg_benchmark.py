"""Sod shock tube benchmark: DG P0–P3 vs exact Riemann solution.

Produces:
  - artifacts/sod_dg/metrics.json   (all errors, pass/fail)
  - artifacts/sod_dg/profiles.csv   (numerical + reference data)
  - artifacts/sod_dg/sod_profiles.png  (comparison plots)
  - artifacts/sod_dg/convergence.png   (error vs DOF)

Reference: Sod, G. A. (1978). "A survey of several finite difference methods
for systems of nonlinear hyperbolic conservation laws." J. Comput. Phys.
27(1), 1–31.  Initial states: (ρ,u,p)_L=(1,0,1), (ρ,u,p)_R=(0.125,0,0.1),
diaphragm at x=0.5, γ=1.4, evaluated at t=0.2 on [0,1].
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.sod_exact import cell_averages, sample, star_state
from benchmarks.euler_sod_1d import solve_sod
from tensordg.euler_dg_1d import solve_euler_dg_1d

GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
FINAL_TIME = 0.2
DOMAIN = (0.0, 1.0)
TOLERANCE = 0.03  # 3%


def compute_errors(x, coeff_or_fields, ncells, degree=0):
    """Compute relative L2 errors vs exact cell averages."""
    _, ref = cell_averages(ncells, FINAL_TIME)
    if degree == 0:
        # P0: fields is a dict
        fields = coeff_or_fields
    else:
        # Higher-order: coeff is (ncells, degree+1, 3)
        avg = coeff_or_fields[:, 0].detach().cpu().numpy()
        rho = avg[:, 0]
        vel = avg[:, 1] / rho
        pres = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
        fields = {"density": rho, "velocity": vel, "pressure": pres}

    errors = {}
    for key in ("density", "velocity", "pressure"):
        num = fields[key] if isinstance(fields[key], np.ndarray) else fields[key].detach().cpu().numpy()
        absolute = np.sqrt(np.mean((num - ref[key]) ** 2))
        norm = np.sqrt(np.mean(ref[key] ** 2))
        errors[key] = {
            "absolute_l2": float(absolute),
            "relative_l2": float(absolute / norm),
            "relative_l2_percent": float(100.0 * absolute / norm),
        }
    return errors, fields


def run_p0(ncells, cfl=0.4):
    """Run P0 DG (finite-volume) Sod."""
    t0 = time.time()
    x, fields, meta = solve_sod(ncells=ncells, final_time=FINAL_TIME, cfl=cfl)
    elapsed = time.time() - t0
    errors, fields_np = compute_errors(x, fields, ncells, degree=0)
    return x.detach().cpu().numpy(), fields_np, errors, {**meta, "wall_time": elapsed, "method": "P0"}


def run_dg(ncells, degree, cfl=0.12, limiter=None, limiter_stage="final"):
    """Run high-order DG Sod."""
    t0 = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=ncells, degree=degree, final_time=FINAL_TIME,
        diaphragm=DIAPHRAGM, domain=DOMAIN, cfl=cfl, limiter=limiter,
        limiter_stage=limiter_stage,
    )
    elapsed = time.time() - t0
    errors, fields_np = compute_errors(x, coeff, ncells, degree=degree)
    return x.detach().cpu().numpy(), fields_np, errors, {**meta, "wall_time": elapsed, "method": f"P{degree}"}


def plot_profiles(results, output_dir):
    """Plot numerical vs exact for all configurations."""
    dense_x = np.linspace(0.0, 1.0, 4001)
    dense_exact = sample(dense_x, FINAL_TIME)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    colors = {"P0": "red", "P1": "blue", "P2": "green", "P3": "purple"}

    for key, label, ax in zip(
        ("density", "velocity", "pressure"),
        (r"Density $\rho$", r"Velocity $u$", r"Pressure $p$"),
        axes,
    ):
        ax.plot(dense_x, dense_exact[key], "k-", linewidth=2.0, label="Exact Riemann", zorder=10)
        for name, (x, fields, errors, meta) in results.items():
            c = colors.get(name, "gray")
            num = fields[key] if isinstance(fields[key], np.ndarray) else fields[key].detach().cpu().numpy()
            err_pct = errors[key]["relative_l2_percent"]
            ax.plot(x, num, "--", color=c, linewidth=1.2, alpha=0.8,
                    label=f"{name} (err={err_pct:.2f}%)")
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("x", fontsize=12)
    fig.suptitle("Sod Shock Tube: DG P0–P3 vs Exact Riemann Solution (t=0.2)", fontsize=14)
    fig.savefig(output_dir / "sod_profiles.png", dpi=160)
    plt.close(fig)


def plot_convergence(convergence_data, output_dir):
    """Plot error vs degrees of freedom."""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"density": "o", "velocity": "s", "pressure": "^"}
    for key, marker in markers.items():
        for method, data in convergence_data.items():
            dofs = data["dofs"]
            errs = data[key]
            ax.loglog(dofs, errs, marker=marker, linewidth=1.5,
                      label=f"{method} {key}")
    ax.set_xlabel("Degrees of Freedom", fontsize=12)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.suptitle("Sod Shock Tube: Convergence Study", fontsize=14)
    fig.savefig(output_dir / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    output_dir = Path("artifacts/sod_dg")
    output_dir.mkdir(parents=True, exist_ok=True)

    p_star, u_star = star_state()
    print(f"Reference star state: p*={p_star:.12f}, u*={u_star:.12f}")

    all_results = {}
    convergence_data = {}

    # --- P0 at multiple resolutions ---
    print("\n=== P0 (finite-volume) ===")
    for ncells in [100, 200, 400]:
        x, fields, errors, meta = run_p0(ncells)
        all_results[f"P0_N{ncells}"] = (x, fields, errors, meta)
        max_err = max(errors[k]["relative_l2"] for k in errors)
        print(f"  N={ncells}: max_rel_err={max_err*100:.4f}% {'✓' if max_err<TOLERANCE else '✗'}")
        if "P0" not in convergence_data:
            convergence_data["P0"] = {"dofs": [], "density": [], "velocity": [], "pressure": []}
        convergence_data["P0"]["dofs"].append(ncells)
        for k in ("density", "velocity", "pressure"):
            convergence_data["P0"][k].append(errors[k]["relative_l2"])

    # --- P1, P2, P3 at multiple resolutions ---
    for degree in [1, 2, 3]:
        print(f"\n=== P{degree} DG ===")
        base_ncells = {1: 50, 2: 30, 3: 20}
        for mult in [1, 2]:
            ncells = base_ncells[degree] * mult
            x, fields, errors, meta = run_dg(ncells, degree, limiter=None, limiter_stage="final")
            name = f"P{degree}_N{ncells}"
            all_results[name] = (x, fields, errors, meta)
            max_err = max(errors[k]["relative_l2"] for k in errors)
            print(f"  N={ncells}: max_rel_err={max_err*100:.4f}% {'✓' if max_err<TOLERANCE else '✗'}")
            method_key = f"P{degree}"
            if method_key not in convergence_data:
                convergence_data[method_key] = {"dofs": [], "density": [], "velocity": [], "pressure": []}
            convergence_data[method_key]["dofs"].append(ncells * (degree + 1))
            for k in ("density", "velocity", "pressure"):
                convergence_data[method_key][k].append(errors[k]["relative_l2"])

    # --- Find best configuration ---
    print("\n=== Summary ===")
    best = None
    best_err = float("inf")
    for name, (x, fields, errors, meta) in all_results.items():
        max_err = max(errors[k]["relative_l2"] for k in errors)
        passed = max_err < TOLERANCE
        print(f"  {name}: max_rel_err={max_err*100:.4f}% {'PASS' if passed else 'FAIL'}")
        if passed and max_err < best_err:
            best = name
            best_err = max_err

    # --- Save metrics ---
    metrics = {
        "benchmark": "sod_dg",
        "reference": "Sod (1978) J. Comput. Phys. 27(1), 1-31; exact Riemann solver",
        "gamma": GAMMA,
        "left_state": LEFT,
        "right_state": RIGHT,
        "diaphragm": DIAPHRAGM,
        "final_time": FINAL_TIME,
        "domain": DOMAIN,
        "tolerance": TOLERANCE,
        "star_pressure": p_star,
        "star_velocity": u_star,
        "results": {},
        "best_configuration": best,
        "best_max_relative_error": best_err if best else None,
    }
    for name, (x, fields, errors, meta) in all_results.items():
        max_err = max(errors[k]["relative_l2"] for k in errors)
        metrics["results"][name] = {
            "method": meta.get("method", name.split("_")[0]),
            "ncells": meta.get("ncells", 0),
            "degree": meta.get("degree", 0),
            "steps": meta.get("steps", 0),
            "wall_time_s": meta.get("wall_time", 0),
            "errors": errors,
            "max_relative_l2": max_err,
            "max_relative_l2_percent": 100.0 * max_err,
            "passed": max_err < TOLERANCE,
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Save profiles for best config ---
    if best:
        x_best, fields_best, errors_best, meta_best = all_results[best]
        _, ref = cell_averages(meta_best.get("ncells", 0), FINAL_TIME)
        data = np.column_stack(
            [x_best]
            + [fields_best[k] if isinstance(fields_best[k], np.ndarray) else fields_best[k].detach().cpu().numpy()
               for k in ("density", "velocity", "pressure")]
            + [ref[k] for k in ("density", "velocity", "pressure")]
        )
        np.savetxt(
            output_dir / "profiles.csv", data, delimiter=",", comments="",
            header="x,density,velocity,pressure,density_ref,velocity_ref,pressure_ref",
        )

    # --- Plots ---
    # Use representative configs for profile plot
    profile_results = {}
    for name in ["P0_N400", "P1_N50", "P2_N30", "P3_N40"]:
        if name in all_results:
            profile_results[name] = all_results[name]
    if profile_results:
        plot_profiles(profile_results, output_dir)

    # Convergence plot
    if convergence_data:
        plot_convergence(convergence_data, output_dir)

    print(f"\nArtifacts written to {output_dir}")
    print(f"Best configuration: {best} (max error = {best_err*100:.4f}%)")
    if best:
        print(f"PASS: all fields < {TOLERANCE*100}%")
    else:
        print("FAIL: no configuration met the tolerance")


if __name__ == "__main__":
    main()
