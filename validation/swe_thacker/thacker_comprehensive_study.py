#!/usr/bin/env python3
"""Comprehensive Thacker benchmark validation study.

This script performs a systematic study of the Thacker (1981) parabolic basin
oscillation benchmark to characterize the long-time error and identify fixes.

Key findings from preliminary studies:
1. Legacy slope + transmissive BC: ~12% error at t=0.5s, non-convergent
2. Exact slope + transmissive BC: ~55% error (catastrophic!)
3. Exact slope + exact BC: 0.04% error at N=40, t=0.3s (excellent!)

This suggests the issue is a consistency problem between source term and BC treatment.
"""
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/root/tp-agent/work/m7536')

from tensordg import swe_exact
from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

H0 = 1.0
AMP = 0.5
L = 1.0
G = 9.81
OMEGA = np.sqrt(2 * G * H0) / L

def thacker_bed(x):
    return -H0 * (1 - (x / L) ** 2)

def thacker_bed_slope(x):
    return 2 * H0 * x / L ** 2

def thacker_initial_h(x):
    h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)
    return torch.as_tensor(h, dtype=torch.float64)

def thacker_initial_u(x):
    return torch.zeros_like(x)

def run_case(ncells, degree, t_end, use_exact_slope, use_exact_bc):
    """Run a single test case."""
    zb_slope_fn = thacker_bed_slope if use_exact_slope else None
    
    if use_exact_bc:
        def bc_left(t):
            fl, _ = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            return fl
        def bc_right(t):
            _, fr = swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)
            return fr
    else:
        bc_left = None
        bc_right = None
    
    x, coeff, zb_faces, meta = solve_swe_1d(
        thacker_initial_h,
        thacker_initial_u,
        thacker_bed,
        ncells=ncells,
        degree=degree,
        final_time=t_end,
        domain=(-L, L),
        cfl=0.3,
        flux_kind="hll",
        limiter=None,
        rk_order=3,
        dtype=torch.float64,
        bc_left_flux=bc_left,
        bc_right_flux=bc_right,
        zb_slope=zb_slope_fn,
    )
    
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    
    # Exact solution
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(x.numpy(), t_end, H0, AMP, L, G)
    zb = thacker_bed(x.numpy())
    eta_e = torch.as_tensor(h_e + zb, dtype=torch.float64)
    u_e_t = torch.as_tensor(u_e, dtype=torch.float64)
    
    # Errors
    err_eta = float(torch.sqrt(torch.mean((eta - eta_e) ** 2)) / torch.sqrt(torch.mean(eta_e ** 2))) * 100
    err_u = float(torch.sqrt(torch.mean((u - u_e_t) ** 2)) / torch.sqrt(torch.mean(u_e_t ** 2))) * 100
    
    # Volume
    dx = 2 * L / ncells
    vol = float(h.sum()) * dx
    eta_c = H0 + (AMP ** 2 / (4 * H0)) * np.sin(OMEGA * t_end) ** 2
    vol_exact = 2.0 * eta_c + (4.0 / 3.0) * H0 * L
    vol_err = abs(vol - vol_exact) / vol_exact * 100
    
    return {
        'ncells': ncells,
        'degree': degree,
        't_end': t_end,
        'eta_err_pct': err_eta,
        'u_err_pct': err_u,
        'vol_err_pct': vol_err,
        'exact_slope': use_exact_slope,
        'exact_bc': use_exact_bc,
        'steps': meta['steps'],
        'reached_final': meta['reached_final_time'],
    }

def main():
    print("=" * 80)
    print("Thacker (1981) Parabolic Basin: Comprehensive Validation Study")
    print("=" * 80)
    
    results = []
    
    # Study 1: Convergence at t=0.3s with exact slope + exact BC
    print("\n" + "=" * 80)
    print("Study 1: Convergence with exact slope + exact BC (t=0.3s)")
    print("=" * 80)
    print(f"{'N':>6} {'P':>3} {'η err (%)':>12} {'u err (%)':>12} {'Vol err (%)':>12} {'EOC(η)':>10}")
    
    prev_err = None
    prev_n = None
    for n in [10, 20, 40, 80]:
        for deg in [1, 2]:
            try:
                res = run_case(n, deg, 0.3, use_exact_slope=True, use_exact_bc=True)
                results.append(res)
                
                eoc = ""
                if prev_err is not None and prev_err > 0 and res['eta_err_pct'] > 0:
                    eoc_val = np.log(prev_err / res['eta_err_pct']) / np.log(res['ncells'] / prev_n)
                    eoc = f"{eoc_val:.2f}"
                
                print(f"{res['ncells']:>6} {res['degree']:>3} {res['eta_err_pct']:>12.4f} "
                      f"{res['u_err_pct']:>12.4f} {res['vol_err_pct']:>12.4f} {eoc:>10}")
                
                prev_err = res['eta_err_pct']
                prev_n = res['ncells']
            except Exception as e:
                print(f"{n:>6} {deg:>3} FAILED: {e}")
    
    # Study 2: Long-time behavior (t=0.5s) with best configuration
    print("\n" + "=" * 80)
    print("Study 2: Long-time behavior (t=0.5s) with exact slope + exact BC")
    print("=" * 80)
    print(f"{'N':>6} {'P':>3} {'η err (%)':>12} {'u err (%)':>12} {'Vol err (%)':>12} {'Reached':>8}")
    
    for n in [20, 40, 80]:
        for deg in [1, 2]:
            try:
                res = run_case(n, deg, 0.5, use_exact_slope=True, use_exact_bc=True)
                results.append(res)
                print(f"{res['ncells']:>6} {res['degree']:>3} {res['eta_err_pct']:>12.4f} "
                      f"{res['u_err_pct']:>12.4f} {res['vol_err_pct']:>12.4f} {str(res['reached_final']):>8}")
            except Exception as e:
                print(f"{n:>6} {deg:>3} FAILED: {e}")
    
    # Study 3: Compare all configurations at t=0.3s, N=40, P1
    print("\n" + "=" * 80)
    print("Study 3: Configuration comparison (N=40, P1, t=0.3s)")
    print("=" * 80)
    
    configs = [
        (False, False, "Legacy slope, transmissive BC"),
        (True, False, "Exact slope, transmissive BC"),
        (True, True, "Exact slope, exact BC"),
    ]
    
    print(f"{'Configuration':<40} {'η err (%)':>12} {'u err (%)':>12} {'Vol err (%)':>12}")
    for exact_slope, exact_bc, label in configs:
        try:
            res = run_case(40, 1, 0.3, exact_slope, exact_bc)
            results.append(res)
            print(f"{label:<40} {res['eta_err_pct']:>12.4f} {res['u_err_pct']:>12.4f} {res['vol_err_pct']:>12.4f}")
        except Exception as e:
            print(f"{label:<40} FAILED: {e}")
    
    # Save results
    with open('/root/tp-agent/work/m7536/validation/swe_thacker/thacker_comprehensive_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating plots...")
    print("=" * 80)
    
    # Plot 1: Solution profiles at t=0.3s
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Best configuration
    x, coeff, zb_faces, meta = solve_swe_1d(
        thacker_initial_h, thacker_initial_u, thacker_bed,
        ncells=80, degree=1, final_time=0.3, domain=(-L, L),
        cfl=0.3, flux_kind="hll", limiter=None, rk_order=3,
        dtype=torch.float64,
        bc_left_flux=lambda t: swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[0],
        bc_right_flux=lambda t: swe_exact.thacker_planar_basin_boundary_flux(t, H0, AMP, L, G)[1],
        zb_slope=thacker_bed_slope,
    )
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    
    h_e, u_e, eta_e = swe_exact.thacker_planar_basin(x.numpy(), 0.3, H0, AMP, L, G)
    zb = thacker_bed(x.numpy())
    eta_e_t = torch.as_tensor(h_e + zb, dtype=torch.float64)
    
    axes[0].plot(x.numpy(), eta.numpy(), 'b-', label='DG (exact slope + exact BC)', linewidth=2)
    axes[0].plot(x.numpy(), eta_e_t.numpy(), 'r--', label='Exact', linewidth=2)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('η (m)')
    axes[0].set_title('Free Surface at t=0.3s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x.numpy(), u.numpy(), 'b-', label='DG', linewidth=2)
    axes[1].plot(x.numpy(), u_e, 'r--', label='Exact', linewidth=2)
    axes[1].set_xlabel('x (m)')
    axes[1].set_ylabel('u (m/s)')
    axes[1].set_title('Velocity at t=0.3s')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m7536/validation/swe_thacker/thacker_solution_profiles.png', dpi=150)
    print("Saved: thacker_solution_profiles.png")
    
    # Plot 2: Convergence study
    fig, ax = plt.subplots(figsize=(8, 6))
    
    exact_bc_results = [r for r in results if r['exact_bc'] and r['t_end'] == 0.3 and r['degree'] == 1]
    exact_bc_results.sort(key=lambda r: r['ncells'])
    
    ns = [r['ncells'] for r in exact_bc_results]
    errs = [r['eta_err_pct'] for r in exact_bc_results]
    
    ax.loglog(ns, errs, 'bo-', label='η error (exact slope + exact BC)', linewidth=2, markersize=8)
    
    # Reference slopes
    if len(ns) >= 2:
        ax.loglog([ns[0], ns[-1]], [errs[0], errs[0] * (ns[0]/ns[-1])**1], 'k--', alpha=0.5, label='O(h)')
        ax.loglog([ns[0], ns[-1]], [errs[0], errs[0] * (ns[0]/ns[-1])**2], 'k:', alpha=0.5, label='O(h²)')
    
    ax.set_xlabel('Number of cells N')
    ax.set_ylabel('Relative L2 error in η (%)')
    ax.set_title('Thacker Benchmark: Convergence Study (t=0.3s, P1)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/root/tp-agent/work/m7536/validation/swe_thacker/thacker_convergence_study.png', dpi=150)
    print("Saved: thacker_convergence_study.png")
    
    print("\n" + "=" * 80)
    print("Study complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
