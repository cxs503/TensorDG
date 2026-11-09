#!/usr/bin/env python3
"""Deep dive: Why does exact slope + transmissive BC fail catastrophically?

Hypothesis: The exact slope treatment changes the energy balance, and without
exact BC to compensate, energy is injected/removed incorrectly at boundaries.
"""
import sys
import torch
import numpy as np

sys.path.insert(0, '/root/tp-agent/work/m7536')

from tensordg import swe_exact
from tensordg.swe_1d import solve_swe_1d, reconstruct_fields

H0 = 1.0
AMP = 0.5
L = 1.0
G = 9.81

def thacker_bed(x):
    return -H0 * (1 - (x / L) ** 2)

def thacker_bed_slope(x):
    return 2 * H0 * x / L ** 2

def thacker_initial_h(x):
    h, _ = swe_exact.thacker_planar_basin_initial(x.numpy(), H0, AMP, L)
    return torch.as_tensor(h, dtype=torch.float64)

def thacker_initial_u(x):
    return torch.zeros_like(x)

def compute_energy(ncells, t_end, use_exact_slope, use_exact_bc):
    """Compute total energy at different times."""
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
    
    times = [0.0, 0.1, 0.2, 0.3]
    energies = []
    
    for t in times:
        if t == 0.0:
            # Initial condition
            x = torch.linspace(-L + L/ncells, L - L/ncells, ncells, dtype=torch.float64)
            h = thacker_initial_h(x)
            u = thacker_initial_u(x)
        else:
            x, coeff, zb_faces, meta = solve_swe_1d(
                thacker_initial_h,
                thacker_initial_u,
                thacker_bed,
                ncells=ncells,
                degree=1,
                final_time=t,
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
        
        # Total energy: kinetic + potential
        dx = 2 * L / ncells
        zb = thacker_bed(x)
        ke = 0.5 * float((h * u**2).sum()) * dx
        pe = 0.5 * G * float((h**2).sum()) * dx
        pe_bed = G * float((h * zb).sum()) * dx
        total = ke + pe + pe_bed
        energies.append({'t': t, 'ke': ke, 'pe': pe, 'pe_bed': pe_bed, 'total': total})
    
    return energies

def main():
    print("=" * 80)
    print("Energy Balance Analysis")
    print("=" * 80)
    
    ncells = 40
    
    configs = [
        (False, False, "Legacy slope, transmissive BC"),
        (True, False, "Exact slope, transmissive BC"),
        (True, True, "Exact slope, exact BC"),
    ]
    
    for exact_slope, exact_bc, label in configs:
        print(f"\n--- {label} (N={ncells}) ---")
        print(f"{'t':>6} {'KE':>12} {'PE':>12} {'PE_bed':>12} {'Total':>12}")
        
        try:
            energies = compute_energy(ncells, 0.3, exact_slope, exact_bc)
            for e in energies:
                print(f"{e['t']:>6.2f} {e['ke']:>12.4f} {e['pe']:>12.4f} "
                      f"{e['pe_bed']:>12.4f} {e['total']:>12.4f}")
            
            # Energy drift
            dE = energies[-1]['total'] - energies[0]['total']
            dE_pct = abs(dE) / abs(energies[0]['total']) * 100
            print(f"Energy drift: {dE:+.6f} ({dE_pct:.3f}%)")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
