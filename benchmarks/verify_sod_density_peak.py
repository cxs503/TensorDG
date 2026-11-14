"""Sod shock tube density peak (contact discontinuity) position verification.

Verifies that the TensorDG solver captures the contact discontinuity position
of the Sod (1978) shock tube problem within the <4% tolerance specified in
the feature ledger (CASE-DG-SOD-SHOCK).

Reference
---------
Sod, G. A. (1978). "A survey of several finite difference methods for systems
of nonlinear hyperbolic conservation laws." J. Comput. Phys. 27(1), 1–31.

Initial states: (ρ,u,p)_L = (1, 0, 1), (ρ,u,p)_R = (0.125, 0, 0.1)
Diaphragm at x = 0.5, γ = 1.4, evaluated at t = 0.2 on [0, 1].

The density peak in the Sod problem is the plateau ρ = 1.0 extending from
the rarefaction head to the contact discontinuity.  The key positional
quantity is the **contact discontinuity** at x_contact = 0.5 + u*·t,
where u* is the star-region velocity from the exact Riemann solver.

Verification strategy
---------------------
1. Compute the exact contact discontinuity position from the analytical
   Riemann solution (star-state velocity).
2. For high-order DG (P1+), locate the numerical contact by finding the
   maximum density gradient in the region x ∈ [0.5, 0.8] (between the
   contact and the shock).
3. For P0 (first-order), the contact is heavily smeared; locate it by
   finding where density crosses the mean of the two star-region densities.
4. Compute the position error |x_num − x_exact| / L_domain and verify < 4%.

Usage
-----
    PYTHONPATH=. python -m benchmarks.verify_sod_density_peak

Artifacts
---------
    artifacts/sod_density_peak/metrics.json   – all errors, pass/fail
    artifacts/sod_density_peak/profiles.png   – density profiles with contact markers
    artifacts/sod_density_peak/position_error.png – position error vs DOF
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

from benchmarks.sod_exact import star_state, sample, cell_averages
from benchmarks.euler_sod_1d import solve_sod
from tensordg.euler_dg_1d import solve_euler_dg_1d
from benchmarks.riemann_exact import sample as riemann_sample

# ── Problem definition (Sod 1978) ──────────────────────────────────────────
GAMMA = 1.4
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.1)
DIAPHRAGM = 0.5
FINAL_TIME = 0.2
DOMAIN = (0.0, 1.0)
DOMAIN_LENGTH = DOMAIN[1] - DOMAIN[0]

# Acceptance tolerance from feature ledger CASE-DG-SOD-SHOCK
POSITION_TOLERANCE = 0.04  # 4% of domain length
L2_TOLERANCE = 0.03        # 3% relative L2


def _star_densities():
    """Compute star-region densities for contact detection."""
    p_star, _ = star_state()
    rho_star_L = LEFT[0] * (p_star / LEFT[2]) ** (1.0 / GAMMA)
    ratio = p_star / RIGHT[2]
    rho_star_R = RIGHT[0] * (ratio + (GAMMA - 1) / (GAMMA + 1)) / (
        (GAMMA - 1) / (GAMMA + 1) * ratio + 1
    )
    return rho_star_L, rho_star_R


def exact_contact_position() -> float:
    """Return the exact contact discontinuity position x = 0.5 + u*·t."""
    _, u_star = star_state()
    return DIAPHRAGM + u_star * FINAL_TIME


def find_contact_high_order(x: np.ndarray, density: np.ndarray,
                            velocity: np.ndarray, u_star: float) -> float:
    """Locate contact discontinuity for high-order (P1+) solutions.

    Uses the velocity criterion: the contact is the density jump where
    velocity ≈ u* (velocity is continuous across contact, discontinuous
    across shock).
    """
    # Search in region between rarefaction tail and shock
    mask = (x > 0.45) & (x < 0.82)
    x_s = x[mask]
    rho_s = density[mask]
    u_s = velocity[mask]

    if len(x_s) < 3:
        return float(x_s[-1]) if len(x_s) > 0 else 0.5

    drho = np.diff(rho_s)
    dx = np.diff(x_s)
    grad = np.abs(drho / dx)

    # Find jumps where velocity is close to u*
    sorted_idx = np.argsort(grad)[::-1]
    for idx in sorted_idx[:5]:
        if idx < len(u_s):
            if abs(u_s[idx] - u_star) < 0.2:
                return float(x_s[idx])

    # Fallback: second largest jump (contact is usually smaller than shock)
    if len(sorted_idx) >= 2:
        return float(x_s[sorted_idx[1]])
    return float(x_s[sorted_idx[0]])


def find_contact_p0(x: np.ndarray, density: np.ndarray,
                    rho_star_avg: float) -> float:
    """Locate contact discontinuity for P0 (smeared) solutions.

    Uses the density threshold: find where density crosses the mean of
    the two star-region densities.
    """
    mask = (x > 0.4) & (x < 0.8)
    x_s = x[mask]
    rho_s = density[mask]

    if len(x_s) < 2:
        return float(x_s[-1]) if len(x_s) > 0 else 0.5

    idx = np.argmin(np.abs(rho_s - rho_star_avg))
    return float(x_s[idx])


def compute_l2_error(x: np.ndarray, density: np.ndarray) -> float:
    """Compute relative L2 error of density vs exact cell averages."""
    ncells = len(x)
    _, ref = cell_averages(ncells, FINAL_TIME)
    absolute = np.sqrt(np.mean((density - ref["density"]) ** 2))
    norm = np.sqrt(np.mean(ref["density"] ** 2))
    return float(absolute / norm)


def run_p0(ncells: int, cfl: float = 0.4):
    """Run P0 (finite-volume) Sod and extract contact position."""
    t0 = time.time()
    x, fields, meta = solve_sod(ncells=ncells, final_time=FINAL_TIME, cfl=cfl)
    x_np = x.detach().cpu().numpy()
    rho_np = fields["density"].detach().cpu().numpy()
    u_np = fields["velocity"].detach().cpu().numpy()

    rho_star_L, rho_star_R = _star_densities()
    x_contact = find_contact_p0(x_np, rho_np, (rho_star_L + rho_star_R) / 2)
    l2_err = compute_l2_error(x_np, rho_np)

    return {
        "x_contact": x_contact,
        "density_max": float(rho_np.max()),
        "density_max_pos": float(x_np[rho_np.argmax()]),
        "l2_error": l2_err,
        "steps": meta["steps"],
        "wall_time": time.time() - t0,
        "method": "P0",
        "ncells": ncells,
        "degree": 0,
    }


def run_dg(ncells: int, degree: int, cfl: float = 0.12,
           limiter: str | None = None, limiter_stage: str = "final"):
    """Run high-order DG Sod and extract contact position."""
    t0 = time.time()
    x, coeff, meta = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=ncells, degree=degree, final_time=FINAL_TIME,
        diaphragm=DIAPHRAGM, domain=DOMAIN, cfl=cfl, limiter=limiter,
        limiter_stage=limiter_stage,
    )
    x_np = x.detach().cpu().numpy()
    avg = coeff[:, 0].detach().cpu().numpy()
    rho_np = avg[:, 0]
    u_np = avg[:, 1] / rho_np

    _, u_star = star_state()
    x_contact = find_contact_high_order(x_np, rho_np, u_np, u_star)
    l2_err = compute_l2_error(x_np, rho_np)

    return {
        "x_contact": x_contact,
        "density_max": float(rho_np.max()),
        "density_max_pos": float(x_np[rho_np.argmax()]),
        "l2_error": l2_err,
        "steps": meta["steps"],
        "wall_time": time.time() - t0,
        "method": f"P{degree}",
        "ncells": ncells,
        "degree": degree,
    }


def plot_profiles(results: dict, x_exact: float, output_dir: Path):
    """Plot density profiles with exact and numerical contact positions."""
    dense_x = np.linspace(0.0, 1.0, 4001)
    dense_exact = sample(dense_x, FINAL_TIME)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"P0": "red", "P1": "blue", "P2": "green", "P3": "purple"}

    ax.plot(dense_x, dense_exact["density"], "k-", linewidth=2.0,
            label="Exact Riemann", zorder=10)
    ax.axvline(x_exact, color="k", linestyle="--", linewidth=1.0, alpha=0.5,
               label=f"Exact contact (x={x_exact:.4f})")

    for name, res in results.items():
        method = res["method"]
        c = colors.get(method, "gray")
        ncells = res["ncells"]
        degree = res["degree"]

        if degree == 0:
            x, fields, _ = solve_sod(ncells=ncells, final_time=FINAL_TIME, cfl=0.4)
            x_arr = x.detach().cpu().numpy()
            rho = fields["density"].detach().cpu().numpy()
        else:
            x, coeff, _ = solve_euler_dg_1d(
                LEFT, RIGHT, ncells=ncells, degree=degree, final_time=FINAL_TIME,
                diaphragm=DIAPHRAGM, domain=DOMAIN, cfl=0.12, limiter_stage="final",
            )
            x_arr = x.detach().cpu().numpy()
            rho = coeff[:, 0].detach().cpu().numpy()[:, 0]

        ax.plot(x_arr, rho, "--", color=c, linewidth=1.0, alpha=0.7,
                label=f"{name} (pos_err={res['position_error_percent']:.2f}%)")
        ax.axvline(res["x_contact"], color=c, linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel(r"Density $\rho$", fontsize=12)
    ax.set_title("Sod Shock Tube: Density Profile & Contact Discontinuity Position", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "profiles.png", dpi=160)
    plt.close(fig)


def plot_position_error(results: dict, x_exact: float, output_dir: Path):
    """Plot position error vs degrees of freedom."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = {}
    for name, res in results.items():
        method = res["method"]
        dofs = res["ncells"] * (res["degree"] + 1)
        if method not in methods:
            methods[method] = {"dofs": [], "errors": [], "names": []}
        methods[method]["dofs"].append(dofs)
        methods[method]["errors"].append(res["position_error_percent"])
        methods[method]["names"].append(name)

    markers = {"P0": "o", "P1": "s", "P2": "^", "P3": "D"}
    for method, data in methods.items():
        c = {"P0": "red", "P1": "blue", "P2": "green", "P3": "purple"}.get(method, "gray")
        ax.semilogy(data["dofs"], data["errors"], marker=markers.get(method, "o"),
                    color=c, linewidth=1.5, label=method)

    ax.axhline(POSITION_TOLERANCE * 100, color="k", linestyle="--", linewidth=1.0,
               alpha=0.5, label=f"Tolerance ({POSITION_TOLERANCE*100:.0f}%)")
    ax.set_xlabel("Degrees of Freedom", fontsize=12)
    ax.set_ylabel("Contact Position Error (%)", fontsize=12)
    ax.set_title("Sod Shock Tube: Contact Discontinuity Position Error vs Resolution", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "position_error.png", dpi=160)
    plt.close(fig)


def main():
    output_dir = Path("artifacts/sod_density_peak")
    output_dir.mkdir(parents=True, exist_ok=True)

    x_exact = exact_contact_position()
    _, u_star = star_state()
    print(f"Exact contact discontinuity position: x = {x_exact:.12f}")
    print(f"Star velocity: u* = {u_star:.12f}")
    print(f"Position tolerance: {POSITION_TOLERANCE*100:.1f}% of domain = {POSITION_TOLERANCE*DOMAIN_LENGTH:.4f}")
    print()

    results = {}

    # ── P0 at selected resolutions (lightweight) ───────────────────────────
    print("=== P0 (finite-volume) ===")
    for ncells in [200, 400, 800]:
        res = run_p0(ncells)
        pos_err = abs(res["x_contact"] - x_exact) / DOMAIN_LENGTH
        res["position_error"] = pos_err
        res["position_error_percent"] = pos_err * 100
        res["passed"] = pos_err < POSITION_TOLERANCE
        name = f"P0_N{ncells}"
        results[name] = res
        status = "✓ PASS" if res["passed"] else "✗ FAIL"
        print(f"  N={ncells:4d}: x_contact={res['x_contact']:.6f} "
              f"err={pos_err*100:.4f}%  L2={res['l2_error']*100:.4f}%  {status}")

    # ── P1, P2, P3 at selected resolutions ─────────────────────────────────
    for degree in [1, 2, 3]:
        print(f"\n=== P{degree} DG ===")
        base_ncells = {1: 50, 2: 30, 3: 20}
        for mult in [1, 2]:
            ncells = base_ncells[degree] * mult
            res = run_dg(ncells, degree, limiter_stage="final")
            pos_err = abs(res["x_contact"] - x_exact) / DOMAIN_LENGTH
            res["position_error"] = pos_err
            res["position_error_percent"] = pos_err * 100
            res["passed"] = pos_err < POSITION_TOLERANCE
            name = f"P{degree}_N{ncells}"
            results[name] = res
            status = "✓ PASS" if res["passed"] else "✗ FAIL"
            print(f"  N={ncells:4d}: x_contact={res['x_contact']:.6f} "
                  f"err={pos_err*100:.4f}%  L2={res['l2_error']*100:.4f}%  {status}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, res in sorted(results.items()):
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_passed = False
        print(f"  {name:25s}: pos_err={res['position_error_percent']:.4f}%  "
              f"L2={res['l2_error']*100:.4f}%  {status}")

    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILURES'}")
    print(f"Exact contact position: x = {x_exact:.12f}")

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "benchmark": "sod_density_peak_position",
        "anchor": "CASE-DG-SOD-SHOCK",
        "reference": "Sod (1978) J. Comput. Phys. 27(1), 1-31",
        "gamma": GAMMA,
        "left_state": list(LEFT),
        "right_state": list(RIGHT),
        "diaphragm": DIAPHRAGM,
        "final_time": FINAL_TIME,
        "domain": list(DOMAIN),
        "exact_contact_position": x_exact,
        "star_velocity": u_star,
        "position_tolerance": POSITION_TOLERANCE,
        "l2_tolerance": L2_TOLERANCE,
        "results": {},
        "all_passed": all_passed,
    }
    for name, res in results.items():
        metrics["results"][name] = {
            "method": res["method"],
            "ncells": res["ncells"],
            "degree": res["degree"],
            "steps": res["steps"],
            "wall_time_s": round(res["wall_time"], 3),
            "x_contact_numerical": res["x_contact"],
            "x_contact_exact": x_exact,
            "position_error": res["position_error"],
            "position_error_percent": res["position_error_percent"],
            "density_max": res["density_max"],
            "density_max_position": res["density_max_pos"],
            "l2_error": res["l2_error"],
            "l2_error_percent": res["l2_error"] * 100,
            "passed": res["passed"],
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {output_dir / 'metrics.json'}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_profiles(results, x_exact, output_dir)
    plot_position_error(results, x_exact, output_dir)
    print(f"Plots written to {output_dir}/")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
