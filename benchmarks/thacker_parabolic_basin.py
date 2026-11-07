"""Thacker (1981) parabolic basin oscillation benchmark for 1-D SWE.

Analytical solution for oscillating water in a parabolic basin:
    Bed: z_b(x) = -h_0 * (1 - (x/L)^2)
    Initial water surface: η(x,0) = h_0 + A * x/L
    Exact solution:
        η(x,t) = h_0 + A * cos(ω*t) * x/L
        u(x,t) = -A * ω * sin(ω*t) / g
        h(x,t) = η(x,t) - z_b(x)
    
    where ω = sqrt(2*g*h_0) / L

Reference:
    Thacker, W. C. (1981). "Some exact solutions of the nonlinear 
    shallow-water equations." Communications on Pure and Applied 
    Mathematics, 34(4), 543-552.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensordg.swe_1d import solve_swe_1d, reconstruct_fields


def thacker_exact(x: torch.Tensor, t: float, h0: float, A: float, L: float) -> tuple:
    """Compute exact Thacker solution at time t.
    
    Parameters
    ----------
    x : tensor of x-coordinates
    t : time
    h0 : reference depth at center
    A : amplitude of oscillation
    L : basin half-width
    
    Returns
    -------
    eta : water surface elevation (tensor)
    u : velocity (tensor)
    h : water depth (tensor)
    """
    g = 9.81
    omega = math.sqrt(2.0 * g * h0) / L
    
    eta = h0 + A * math.cos(omega * t) * x / L
    u = torch.full_like(x, -A * omega * math.sin(omega * t) / g)
    zb = -h0 * (1.0 - (x / L) ** 2)
    h = eta - zb
    
    return eta, u, h


def run_thacker_benchmark(
    ncells: int = 100,
    degree: int = 2,
    final_time: float = 1.0,
    h0: float = 1.0,
    A: float = 0.5,
    L: float = 1.0,
    cfl: float = 0.3,
):
    """Run Thacker benchmark and compute errors.
    
    Parameters
    ----------
    ncells : number of cells
    degree : polynomial degree (1 or 2)
    final_time : simulation time
    h0 : reference depth
    A : oscillation amplitude
    L : basin half-width
    cfl : CFL number
    
    Returns
    -------
    dict with errors and metadata
    """
    domain = (-L, L)
    
    # Initial conditions
    def init_h(x):
        zb = -h0 * (1.0 - (x / L) ** 2)
        eta = h0 + A * x / L
        return eta - zb
    
    def init_u(x):
        return torch.zeros_like(x)
    
    def zb(x):
        return -h0 * (1.0 - (x / L) ** 2)
    
    # Solve
    x_centers, coeff, zb_faces, meta = solve_swe_1d(
        init_h, init_u, zb,
        ncells=ncells, degree=degree,
        final_time=final_time, domain=domain,
        cfl=cfl, flux_kind="hll", limiter=None,
        rk_order=3, device="cpu", dtype=torch.float64,
    )
    
    # Reconstruct fields
    h_num, u_num, eta_num = reconstruct_fields(coeff, zb_faces)
    
    # Exact solution
    x_t = x_centers.clone() if isinstance(x_centers, torch.Tensor) else torch.tensor(x_centers, dtype=torch.float64)
    eta_exact, u_exact, h_exact = thacker_exact(x_t, final_time, h0, A, L)
    
    # Compute errors
    eta_err = torch.sqrt(torch.mean((eta_num - eta_exact) ** 2))
    u_err = torch.sqrt(torch.mean((u_num - u_exact) ** 2))
    h_err = torch.sqrt(torch.mean((h_num - h_exact) ** 2))
    
    eta_rel = eta_err / torch.max(torch.abs(eta_exact)) * 100
    u_rel = u_err / torch.max(torch.abs(u_exact)) * 100
    h_rel = h_err / torch.max(torch.abs(h_exact)) * 100
    
    return {
        "ncells": ncells,
        "degree": degree,
        "final_time": final_time,
        "steps": meta["steps"],
        "eta_l2": float(eta_err),
        "u_l2": float(u_err),
        "h_l2": float(h_err),
        "eta_rel_pct": float(eta_rel),
        "u_rel_pct": float(u_rel),
        "h_rel_pct": float(h_rel),
        "x": x_centers.numpy() if isinstance(x_centers, torch.Tensor) else x_centers,
        "eta_num": eta_num.numpy(),
        "u_num": u_num.numpy(),
        "h_num": h_num.numpy(),
        "eta_exact": eta_exact.numpy(),
        "u_exact": u_exact.numpy(),
        "h_exact": h_exact.numpy(),
    }


def convergence_study():
    """Run convergence study for Thacker benchmark."""
    print("=" * 80)
    print("Thacker (1981) Parabolic Basin Oscillation - Convergence Study")
    print("=" * 80)
    
    results = []
    
    # P1 convergence
    print("\nP1 (degree=1) convergence:")
    print(f"{'ncells':>8} {'steps':>6} {'η_rel%':>10} {'u_rel%':>10} {'h_rel%':>10}")
    print("-" * 50)
    
    for ncells in [50, 100, 200, 400]:
        res = run_thacker_benchmark(ncells=ncells, degree=1, final_time=1.0)
        results.append(res)
        print(f"{ncells:>8} {res['steps']:>6} {res['eta_rel_pct']:>10.4f} "
              f"{res['u_rel_pct']:>10.4f} {res['h_rel_pct']:>10.4f}")
    
    # Compute EOC for P1
    print("\nP1 EOC (Expected Order of Convergence):")
    p1_results = [r for r in results if r['degree'] == 1]
    for i in range(1, len(p1_results)):
        r1 = p1_results[i - 1]
        r2 = p1_results[i]
        ratio = r1['ncells'] / r2['ncells']
        eoc_eta = math.log(r1['eta_rel_pct'] / r2['eta_rel_pct']) / math.log(ratio)
        eoc_u = math.log(r1['u_rel_pct'] / r2['u_rel_pct']) / math.log(ratio)
        print(f"  {r1['ncells']}→{r2['ncells']}: η EOC={eoc_eta:.2f}, u EOC={eoc_u:.2f}")
    
    # P2 convergence
    print("\nP2 (degree=2) convergence:")
    print(f"{'ncells':>8} {'steps':>6} {'η_rel%':>10} {'u_rel%':>10} {'h_rel%':>10}")
    print("-" * 50)
    
    p2_results = []
    for ncells in [25, 50, 100, 200]:
        res = run_thacker_benchmark(ncells=ncells, degree=2, final_time=1.0)
        p2_results.append(res)
        print(f"{ncells:>8} {res['steps']:>6} {res['eta_rel_pct']:>10.4f} "
              f"{res['u_rel_pct']:>10.4f} {res['h_rel_pct']:>10.4f}")
    
    # Compute EOC for P2
    print("\nP2 EOC:")
    for i in range(1, len(p2_results)):
        r1 = p2_results[i - 1]
        r2 = p2_results[i]
        ratio = r1['ncells'] / r2['ncells']
        eoc_eta = math.log(r1['eta_rel_pct'] / r2['eta_rel_pct']) / math.log(ratio)
        eoc_u = math.log(r1['u_rel_pct'] / r2['u_rel_pct']) / math.log(ratio)
        print(f"  {r1['ncells']}→{r2['ncells']}: η EOC={eoc_eta:.2f}, u EOC={eoc_u:.2f}")
    
    return results + p2_results


def plot_solution_comparison(results, output_dir="outputs"):
    """Plot solution comparison with exact solution."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find P2 result with reasonable resolution
    p2_res = [r for r in results if r['degree'] == 2 and r['ncells'] == 100]
    if not p2_res:
        p2_res = [r for r in results if r['degree'] == 2]
    if not p2_res:
        print("No P2 results to plot")
        return
    
    res = p2_res[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Water surface elevation
    ax = axes[0]
    ax.plot(res['x'], res['eta_exact'], 'k-', linewidth=2, label='Exact')
    ax.plot(res['x'], res['eta_num'], 'r--', linewidth=1.5, label=f'DG-P{res["degree"]}')
    ax.set_xlabel('x')
    ax.set_ylabel('η (m)')
    ax.set_title(f'Water Surface Elevation (t={res["final_time"]:.2f}s, {res["ncells"]} cells)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Relative L2 error: {res["eta_rel_pct"]:.4f}%', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Velocity
    ax = axes[1]
    ax.plot(res['x'], res['u_exact'], 'k-', linewidth=2, label='Exact')
    ax.plot(res['x'], res['u_num'], 'r--', linewidth=1.5, label=f'DG-P{res["degree"]}')
    ax.set_xlabel('x')
    ax.set_ylabel('u (m/s)')
    ax.set_title('Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Relative L2 error: {res["u_rel_pct"]:.4f}%', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Water depth
    ax = axes[2]
    ax.plot(res['x'], res['h_exact'], 'k-', linewidth=2, label='Exact')
    ax.plot(res['x'], res['h_num'], 'r--', linewidth=1.5, label=f'DG-P{res["degree"]}')
    ax.set_xlabel('x')
    ax.set_ylabel('h (m)')
    ax.set_title('Water Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, f'Relative L2 error: {res["h_rel_pct"]:.4f}%', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thacker_solution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir}/thacker_solution_comparison.png")
    plt.close()


def plot_convergence(results, output_dir="outputs"):
    """Plot convergence curves."""
    Path(output_dir).mkdir(exist_ok=True)
    
    p1_results = sorted([r for r in results if r['degree'] == 1], key=lambda r: r['ncells'])
    p2_results = sorted([r for r in results if r['degree'] == 2], key=lambda r: r['ncells'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # η convergence
    ax = axes[0]
    if p1_results:
        ax.loglog([r['ncells'] for r in p1_results], 
                 [r['eta_rel_pct'] for r in p1_results], 
                 'bo-', linewidth=2, markersize=8, label='P1')
    if p2_results:
        ax.loglog([r['ncells'] for r in p2_results], 
                 [r['eta_rel_pct'] for r in p2_results], 
                 'rs-', linewidth=2, markersize=8, label='P2')
    
    # Reference slopes
    if p1_results:
        n_ref = np.array([p1_results[0]['ncells'], p1_results[-1]['ncells']])
        e_ref = p1_results[0]['eta_rel_pct'] * (n_ref[0] / n_ref) ** 2
        ax.loglog(n_ref, e_ref, 'b--', alpha=0.5, label='O(h²) ref')
    
    if p2_results:
        n_ref = np.array([p2_results[0]['ncells'], p2_results[-1]['ncells']])
        e_ref = p2_results[0]['eta_rel_pct'] * (n_ref[0] / n_ref) ** 3
        ax.loglog(n_ref, e_ref, 'r--', alpha=0.5, label='O(h³) ref')
    
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('η relative L2 error (%)')
    ax.set_title('Water Surface Elevation Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    # u convergence
    ax = axes[1]
    if p1_results:
        ax.loglog([r['ncells'] for r in p1_results], 
                 [r['u_rel_pct'] for r in p1_results], 
                 'bo-', linewidth=2, markersize=8, label='P1')
    if p2_results:
        ax.loglog([r['ncells'] for r in p2_results], 
                 [r['u_rel_pct'] for r in p2_results], 
                 'rs-', linewidth=2, markersize=8, label='P2')
    
    # Reference slopes
    if p1_results:
        n_ref = np.array([p1_results[0]['ncells'], p1_results[-1]['ncells']])
        e_ref = p1_results[0]['u_rel_pct'] * (n_ref[0] / n_ref) ** 2
        ax.loglog(n_ref, e_ref, 'b--', alpha=0.5, label='O(h²) ref')
    
    if p2_results:
        n_ref = np.array([p2_results[0]['ncells'], p2_results[-1]['ncells']])
        e_ref = p2_results[0]['u_rel_pct'] * (n_ref[0] / n_ref) ** 3
        ax.loglog(n_ref, e_ref, 'r--', alpha=0.5, label='O(h³) ref')
    
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('u relative L2 error (%)')
    ax.set_title('Velocity Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/thacker_convergence.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/thacker_convergence.png")
    plt.close()


if __name__ == "__main__":
    print("\nRunning Thacker (1981) parabolic basin benchmark...")
    print("This validates the well-balanced property and oscillatory flow accuracy.\n")
    
    results = convergence_study()
    
    print("\n" + "=" * 80)
    print("Generating plots...")
    plot_solution_comparison(results, output_dir="outputs")
    plot_convergence(results, output_dir="outputs")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)
