"""Probe 3: does the *shipped* DMR orientation (post-shock side down-right,
post-shock velocity up-left) ever produce a wall reflection?"""
import numpy as np
from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
from benchmarks.double_mach_reflection import _initial_double_mach, solve_double_mach

# (a) shipped config, short run: front must sweep *left* (out of the domain)
for t in (0.005, 0.05):
    x, y, coeff, meta = solve_double_mach(ncells=(60, 15), degree=1,
                                          final_time=t, cfl=0.15)
    rho = cell_fields(coeff)["density"].cpu().numpy()
    print(f"shipped t={t} steps={meta['steps']} rho range=({rho.min():.4f},{rho.max():.4f})")

# (b) shipped config, default final_time=0.2: whole domain post-shock, no structure
x, y, coeff, meta = solve_double_mach(ncells=(60, 15), degree=1, final_time=0.03)
rho = cell_fields(coeff)["density"].cpu().numpy()
print(f"shipped t=0.03 steps={meta['steps']} rho range=({rho.min():.4f},{rho.max():.4f}) "
      f"std={rho.std():.5f}")