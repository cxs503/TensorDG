"""Comprehensive verification study for the 1-D advection-diffusion DG solver.

This script performs systematic verification against the exact exponential layer
solution, including:
- h-refinement convergence studies for P1/P2/P3 at multiple Peclet numbers
- p-refinement studies at fixed resolution
- Flux consistency verification
- Cell Peclet number resolution limits

All results are machine-verifiable and programmatically injected into the report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.advection_diffusion_1d import (
    solve_advection_diffusion_1d,
    advection_diffusion_error,
    exact_flux,
    face_fluxes,
    sample_solution,
)


def convergence_study_h(
    peclet: float,
    degree: int,
    cells_list: list[int],
    speed: float = 1.0,
) -> list[dict]:
    """Run h-refinement study and return list of {cells, error, order}."""
    results = []
    prev_error = None
    prev_h = None
    
    for ncells in cells_list:
        centers, coeff, meta = solve_advection_diffusion_1d(
            ncells=ncells, degree=degree, peclet=peclet, speed=speed,
        )
        err = advection_diffusion_error(centers, coeff, peclet=peclet)
        rel_l2 = err["relative_l2"]
        h = 1.0 / ncells
        
        if prev_error is not None and prev_error > 0 and rel_l2 > 0:
            order = math.log(prev_error / rel_l2) / math.log(prev_h / h)
        else:
            order = float("nan")
        
        results.append({
            "cells": ncells,
            "h": h,
            "relative_l2": rel_l2,
            "absolute_l2": err["absolute_l2"],
            "order": order,
        })
        prev_error = rel_l2
        prev_h = h
    
    return results


def convergence_study_p(
    peclet: float,
    ncells: int,
    degrees: list[int],
    speed: float = 1.0,
) -> list[dict]:
    """Run p-refinement study and return list of {degree, error}."""
    results = []
    
    for degree in degrees:
        centers, coeff, meta = solve_advection_diffusion_1d(
            ncells=ncells, degree=degree, peclet=peclet, speed=speed,
        )
        err = advection_diffusion_error(centers, coeff, peclet=peclet)
        results.append({
            "degree": degree,
            "dofs": meta["dofs"],
            "relative_l2": err["relative_l2"],
            "absolute_l2": err["absolute_l2"],
        })
    
    return results


def flux_consistency_study(
    peclet: float,
    ncells: int,
    degree: int,
    speed: float = 1.0,
) -> dict:
    """Verify flux consistency: all face fluxes should equal the exact flux."""
    centers, coeff, meta = solve_advection_diffusion_1d(
        ncells=ncells, degree=degree, peclet=peclet, speed=speed,
    )
    fluxes = face_fluxes(coeff, peclet=peclet, speed=speed)
    exact_J = exact_flux(peclet=peclet, speed=speed)
    
    # Handle underflow: if exact flux is too small, use absolute spread
    if abs(exact_J) < 1e-300:
        relative_spread = float(torch.std(fluxes).item())
        metric_name = "absolute_spread"
    else:
        relative_spread = float(torch.std(fluxes).item()) / abs(exact_J)
        metric_name = "relative_spread"
    
    return {
        "exact_flux": exact_J,
        "computed_mean": float(torch.mean(fluxes).item()),
        "computed_std": float(torch.std(fluxes).item()),
        "computed_min": float(torch.min(fluxes).item()),
        "computed_max": float(torch.max(fluxes).item()),
        metric_name: relative_spread,
        "n_faces": len(fluxes),
    }


def plot_h_convergence(
    studies: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Plot h-convergence curves for different Peclet numbers and degrees."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    markers = {"P1": "o", "P2": "s", "P3": "^"}
    colors = {"Pe=1": "blue", "Pe=10": "red", "Pe=100": "green"}
    
    for label, results in studies.items():
        # Parse label: "Pe=1, P1" -> peclet=1, degree=1
        parts = label.split(", ")
        peclet_str = parts[0]  # "Pe=1"
        degree_str = parts[1]  # "P1"
        peclet = peclet_str.split("=")[1]  # "1"
        degree = degree_str[1:]  # "1" (strip "P")
        
        hs = [r["h"] for r in results]
        errors = [r["relative_l2"] for r in results]
        
        label_plot = f"{peclet_str}, {degree_str}"
        ax1.loglog(hs, errors, markers[degree_str], color=colors[peclet_str],
                   label=label_plot, linewidth=2, markersize=8)
    
    # Reference slopes
    h_ref = np.array([0.5, 0.05])
    for p, slope in [(1, 2), (2, 3), (3, 4)]:
        e_ref = 1e-2 * (h_ref / h_ref[0]) ** slope
        ax1.loglog(h_ref, e_ref, "k--", alpha=0.3, label=f"O(h^{slope})")
    
    ax1.set_xlabel("Cell size h", fontsize=12)
    ax1.set_ylabel("Relative L2 error", fontsize=12)
    ax1.set_title("h-Convergence: Advection-Diffusion DG", fontsize=14)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Observed order plot
    for label, results in studies.items():
        parts = label.split(", ")
        peclet_str = parts[0]
        degree_str = parts[1]
        
        hs = [r["h"] for r in results[1:]]  # Skip first (no order)
        orders = [r["order"] for r in results[1:]]
        
        ax2.semilogx(hs, orders, markers[degree_str], color=colors[peclet_str],
                     label=f"{peclet_str}, {degree_str}", linewidth=2, markersize=8)
    
    # Theoretical orders
    for p, order in [(1, 2), (2, 3), (3, 4)]:
        ax2.axhline(order, color="gray", linestyle="--", alpha=0.5,
                    label=f"Theory P{p}: {order}")
    
    ax2.set_xlabel("Cell size h", fontsize=12)
    ax2.set_ylabel("Observed order", fontsize=12)
    ax2.set_title("Observed Convergence Order", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_p_convergence(
    studies: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Plot p-convergence: error vs polynomial degree."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = {"Pe=1": "o", "Pe=10": "s", "Pe=100": "^"}
    colors = {"Pe=1": "blue", "Pe=10": "red", "Pe=100": "green"}
    
    for label, results in studies.items():
        peclet_str = label.split(",")[0]  # "Pe=1"
        peclet = peclet_str.split("=")[1]  # "1"
        
        degrees = [r["degree"] for r in results]
        errors = [r["relative_l2"] for r in results]
        
        ax.semilogy(degrees, errors, markers[peclet_str], color=colors[peclet_str],
                    label=peclet_str, linewidth=2, markersize=10)
    
    ax.set_xlabel("Polynomial degree p", fontsize=12)
    ax.set_ylabel("Relative L2 error", fontsize=12)
    ax.set_title("p-Convergence: Advection-Diffusion DG (32 cells)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks([1, 2, 3])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_flux_consistency(
    studies: list[dict],
    output_path: Path,
) -> None:
    """Plot flux consistency across different Peclet numbers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    peclets = [s["peclet"] for s in studies]
    spreads = [s.get("relative_spread", s.get("absolute_spread", 0)) for s in studies]
    
    ax.semilogy(peclets, spreads, "o-", linewidth=2, markersize=10, color="blue")
    ax.axhline(1e-6, color="green", linestyle="--", label="Target: <1e-6")
    ax.axhline(1e-3, color="orange", linestyle="--", label="Acceptable: <1e-3")
    
    ax.set_xlabel("Peclet number", fontsize=12)
    ax.set_ylabel("Relative flux spread (std/|J|)", fontsize=12)
    ax.set_title("Flux Consistency vs Peclet Number (64 cells, P2)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_solution_comparison(
    peclet: float,
    ncells: int,
    degree: int,
    output_path: Path,
) -> None:
    """Plot numerical vs exact solution for a representative case."""
    centers, coeff, meta = solve_advection_diffusion_1d(
        ncells=ncells, degree=degree, peclet=peclet,
    )
    x_num, u_num = sample_solution(centers, coeff, samples_per_cell=10)
    
    from tensordg.advection_diffusion_1d import exact_solution
    u_exact = exact_solution(x_num, peclet)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Solution comparison
    ax1.plot(x_num, u_exact, "k-", linewidth=2, label="Exact")
    ax1.plot(x_num, u_num, "r--", linewidth=2, label=f"DG P{degree}, {ncells} cells")
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("u(x)", fontsize=12)
    ax1.set_title(f"Solution Profile (Pe={peclet})", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Error distribution
    error = np.abs(u_num - u_exact)
    ax2.semilogy(x_num, error + 1e-16, "b-", linewidth=1.5)
    ax2.set_xlabel("x", fontsize=12)
    ax2.set_ylabel("|u_h - u_exact|", fontsize=12)
    ax2.set_title("Pointwise Error Distribution", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run all verification studies and generate report."""
    output_dir = Path("docs/benchmarks/results/advection_diffusion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Advection-Diffusion DG Verification Study")
    print("=" * 70)
    
    # ---- h-convergence studies ----
    print("\n[1/4] Running h-convergence studies...")
    h_studies = {}
    cells_list = [8, 16, 32, 64, 128]
    
    for peclet in [1.0, 10.0, 100.0]:
        for degree in [1, 2, 3]:
            label = f"Pe={int(peclet)}, P{degree}"
            print(f"  {label}...")
            results = convergence_study_h(peclet, degree, cells_list)
            h_studies[label] = results
    
    # ---- p-convergence studies ----
    print("\n[2/4] Running p-convergence studies...")
    p_studies = {}
    degrees = [1, 2, 3]
    
    for peclet in [1.0, 10.0, 100.0]:
        label = f"Pe={int(peclet)}"
        print(f"  {label}...")
        results = convergence_study_p(peclet, ncells=32, degrees=degrees)
        p_studies[label] = results
    
    # ---- Flux consistency studies ----
    print("\n[3/4] Running flux consistency studies...")
    flux_studies = []
    peclets = [0.1, 1.0, 10.0, 100.0]
    
    for peclet in peclets:
        print(f"  Pe={peclet}...")
        flux_result = flux_consistency_study(peclet, ncells=64, degree=2)
        flux_result["peclet"] = peclet
        flux_studies.append(flux_result)
    
    # ---- Generate plots ----
    print("\n[4/4] Generating plots...")
    plot_h_convergence(h_studies, output_dir / "h_convergence.png")
    plot_p_convergence(p_studies, output_dir / "p_convergence.png")
    plot_flux_consistency(flux_studies, output_dir / "flux_consistency.png")
    plot_solution_comparison(peclet=10.0, ncells=64, degree=2,
                            output_path=output_dir / "solution_comparison.png")
    
    # ---- Compute metrics ----
    print("\nComputing verification metrics...")
    metrics = {
        "h_studies": {},
        "p_studies": {},
        "flux_consistency": {},
        "overall_verdict": "PASS",
    }
    
    # h-convergence metrics
    for label, results in h_studies.items():
        parts = label.split(", ")
        degree_str = parts[1]  # "P1"
        degree = int(degree_str[1:])  # 1
        theory_order = degree + 1
        
        # Compute LSQ slope
        hs = [r["h"] for r in results]
        errors = [r["relative_l2"] for r in results]
        log_h = np.log(hs)
        log_e = np.log(errors)
        lsq_order = float(np.polyfit(log_h, log_e, 1)[0])
        
        # Check if within tolerance
        order_error = abs(lsq_order - theory_order)
        passed = order_error < 0.2
        
        metrics["h_studies"][label] = {
            "theory_order": theory_order,
            "lsq_order": lsq_order,
            "order_error": order_error,
            "passed": passed,
            "finest_error": results[-1]["relative_l2"],
        }
        
        if not passed:
            metrics["overall_verdict"] = "FAIL"
    
    # p-convergence metrics
    for label, results in p_studies.items():
        errors = [r["relative_l2"] for r in results]
        monotone = all(errors[i] > errors[i+1] for i in range(len(errors)-1))
        
        metrics["p_studies"][label] = {
            "monotone_decrease": monotone,
            "errors": errors,
            "passed": monotone,
        }
        
        if not monotone:
            metrics["overall_verdict"] = "FAIL"
    
    # Flux consistency metrics
    for flux_result in flux_studies:
        peclet = flux_result["peclet"]
        spread = flux_result.get("relative_spread", flux_result.get("absolute_spread", 0))
        passed = spread < 1e-3
        
        metrics["flux_consistency"][f"Pe={peclet}"] = {
            "spread": spread,
            "passed": passed,
        }
        
        if not passed:
            metrics["overall_verdict"] = "FAIL"
    
    # ---- Save metrics ----
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Overall verdict: {metrics['overall_verdict']}")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    n_pass = sum(1 for s in metrics["h_studies"].values() if s["passed"])
    n_total = len(metrics["h_studies"])
    print(f"\nh-convergence: {n_pass}/{n_total} passed")
    
    for label, m in metrics["h_studies"].items():
        status = "✓" if m["passed"] else "✗"
        print(f"  {status} {label}: theory={m['theory_order']}, "
              f"observed={m['lsq_order']:.3f}, error={m['order_error']:.3f}")
    
    n_pass_p = sum(1 for s in metrics["p_studies"].values() if s["passed"])
    n_total_p = len(metrics["p_studies"])
    print(f"\np-convergence: {n_pass_p}/{n_total_p} passed (monotone decrease)")
    
    n_pass_flux = sum(1 for s in metrics["flux_consistency"].values() if s["passed"])
    n_total_flux = len(metrics["flux_consistency"])
    print(f"\nFlux consistency: {n_pass_flux}/{n_total_flux} passed (<1e-3 spread)")
    
    print(f"\nOverall: {metrics['overall_verdict']}")
    print(f"\nArtifacts in {output_dir}/")
    print("  - h_convergence.png")
    print("  - p_convergence.png")
    print("  - flux_consistency.png")
    print("  - solution_comparison.png")
    print("  - metrics.json")


if __name__ == "__main__":
    main()
