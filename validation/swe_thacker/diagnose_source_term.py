#!/usr/bin/env python3
"""Diagnose the Thacker benchmark long-time error: source term discretization study.

Quick version - fewer configurations, shorter time.
"""
import sys
import json
import torch
import numpy as np

sys.path.insert(0, '/root/tp-agent/work/m7536')

from tensordg import swe_exact
from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

# Thacker parameters
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

def compute_error(ncells, degree, t_end, use_exact_slope=False, use_exact_bc=False):
    """Run DG solver and compute relative L2 error in eta."""
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
    
    # Relative L2 error
    err_eta = float(torch.sqrt(torch.mean((eta - eta_e) ** 2)) / torch.sqrt(torch.mean(eta_e ** 2))) * 100
    err_u = float(torch.sqrt(torch.mean((u - torch.as_tensor(u_e, dtype=torch.float64)) ** 2)) / 
                  torch.sqrt(torch.mean(torch.as_tensor(u_e, dtype=torch.float64) ** 2))) * 100
    
    # Volume conservation
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
    print("Thacker Benchmark: Source Term Study (Quick)")
    print("=" * 80)
    
    # Test at t=0.3s (before the error grows too large)
    t_end = 0.3
    
    results = []
    
    # Legacy slope, transmissive BC
    print("\n--- Legacy slope, transmissive BC ---")
    for n in [20, 40, 80]:
        print(f"N={n}...", end=" ", flush=True)
        res = compute_error(n, 1, t_end, use_exact_slope=False, use_exact_bc=False)
        results.append(res)
        print(f"η={res['eta_err_pct']:.3f}% u={res['u_err_pct']:.3f}% V={res['vol_err_pct']:.3f}%")
    
    # Exact slope, transmissive BC
    print("\n--- Exact slope, transmissive BC ---")
    for n in [20, 40, 80]:
        print(f"N={n}...", end=" ", flush=True)
        res = compute_error(n, 1, t_end, use_exact_slope=True, use_exact_bc=False)
        results.append(res)
        print(f"η={res['eta_err_pct']:.3f}% u={res['u_err_pct']:.3f}% V={res['vol_err_pct']:.3f}%")
    
    # Exact slope, exact BC (may fail)
    print("\n--- Exact slope, exact BC ---")
    for n in [20, 40]:
        print(f"N={n}...", end=" ", flush=True)
        try:
            res = compute_error(n, 1, t_end, use_exact_slope=True, use_exact_bc=True)
            results.append(res)
            print(f"η={res['eta_err_pct']:.3f}% u={res['u_err_pct']:.3f}% V={res['vol_err_pct']:.3f}% reached={res['reached_final']}")
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Save results
    with open('/root/tp-agent/work/m7536/validation/swe_thacker/source_term_study.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Convergence analysis
    print("\n" + "=" * 80)
    print("Convergence Analysis at t=0.3s")
    print("=" * 80)
    
    for exact_slope in [False, True]:
        subset = [r for r in results if r['exact_slope'] == exact_slope and 
                 not r['exact_bc'] and r['eta_err_pct'] > 0]
        if len(subset) < 2:
            continue
        
        label = f"Exact slope={exact_slope}"
        print(f"\n{label}:")
        print(f"{'N':>6} {'η err (%)':>12} {'u err (%)':>12} {'Vol err (%)':>12} {'EOC(η)':>10}")
        
        prev_err = None
        prev_n = None
        for r in subset:
            eoc = ""
            if prev_err is not None and prev_err > 0 and r['eta_err_pct'] > 0:
                eoc_val = np.log(prev_err / r['eta_err_pct']) / np.log(r['ncells'] / prev_n)
                eoc = f"{eoc_val:.2f}"
            print(f"{r['ncells']:>6} {r['eta_err_pct']:>12.3f} {r['u_err_pct']:>12.3f} "
                  f"{r['vol_err_pct']:>12.3f} {eoc:>10}")
            prev_err = r['eta_err_pct']
            prev_n = r['ncells']
    
    print("\n" + "=" * 80)
    print("Results saved to: validation/swe_thacker/source_term_study.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
