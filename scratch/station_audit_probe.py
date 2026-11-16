"""Exploratory probe: E2 station-cell semantics + reference-instrument swap."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import sod_exact
from benchmarks.shock_reference_quadrature import cell_averages as exact_cell_averages
from benchmarks.shock_reference_quadrature import breakpoints

SOD_T = 0.2
LEFT, RIGHT = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
STATIONS = {"sod_rho_x030": ("density", 0.30),
            "sod_rho_x070": ("density", 0.70),
            "sod_vel_x055": ("velocity", 0.55)}

cuts = breakpoints(LEFT, RIGHT, SOD_T, 1.4, 0.5, (0.0, 1.0))
print("breakpoints:", np.array2string(np.asarray(cuts), precision=6))

for ncells in (48, 100, 200, 400, 800):
    dx = 1.0 / ncells
    _, gauss = sod_exact.cell_averages(ncells, SOD_T)
    _, exact = exact_cell_averages(ncells, SOD_T, LEFT, RIGHT)
    print(f"\n--- ncells={ncells} dx={dx:.6g}")
    for key, (field, x) in STATIONS.items():
        raw = x * ncells
        idx = min(int(raw), ncells - 1)
        tie = raw - np.floor(raw)
        lo, hi = idx * dx, (idx + 1) * dx
        contains = [c for c in cuts if lo < c < hi]
        dist = min(abs(np.asarray(cuts) - lo).min(), abs(np.asarray(cuts) - hi).min())
        rg, re_ = float(gauss[field][idx]), float(exact[field][idx])
        delta = abs(rg - re_) / abs(re_)
        # left-tie alternative (idx-1)
        i2 = max(idx - 1, 0)
        rg2, re2 = float(gauss[field][i2]), float(exact[field][i2])
        print(f" {key:14s} idx={idx:4d} tie={tie:.6f} cell=[{lo:.6f},{hi:.6f}] "
              f"contains_wave={bool(contains)} gap_to_wave={dist/dx:.3f}dx")
        print(f"   ref_gauss={rg:.8f} ref_exact={re_:.8f} instr_rel_delta={delta:.3e}")
        print(f"   left-cell idx={i2}: gauss={rg2:.8f} exact={re2:.8f} "
              f"instr_rel_delta={abs(rg2-re2)/abs(re2):.3e}")