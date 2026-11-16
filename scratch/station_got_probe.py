"""Probe 2b: reproduce the E2 station got/ref pairs + tie sensitivity, exactly as bench_workload."""
import sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks import sod_exact
from benchmarks.shock_reference_quadrature import cell_averages as exact_cell_avgs
from benchmarks.shock_reference_quadrature import breakpoints
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_1d import GAMMA

SOD_T = 0.2
LEFT, RIGHT = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
STATIONS = {"sod_rho_x030": ("density", 0.30),
            "sod_rho_x070": ("density", 0.70),
            "sod_vel_x055": ("velocity", 0.55)}
cuts = np.asarray(breakpoints(LEFT, RIGHT, SOD_T, 1.4, 0.5, (0.0, 1.0)))

for degree, ncells in ((1, 48), (1, 24)):
    t0 = time.time()
    x, coeff, meta = solve_euler_dg_1d(LEFT, RIGHT, ncells=ncells, degree=degree,
                                       final_time=SOD_T, cfl=0.12, limiter="tvb",
                                       tvb_constant=1.0, limiter_stage="final")
    avg = coeff[:, 0]
    rho = np.asarray(avg[:, 0]); mom = np.asarray(avg[:, 1]); en = np.asarray(avg[:, 2])
    got = {"density": rho, "velocity": mom / rho,
           "pressure": (GAMMA - 1.0) * (en - 0.5 * mom ** 2 / rho)}
    print(f"\n=== P{degree} N={ncells} ({time.time()-t0:.1f}s) limiter={meta.get('limiter')} "
          f"steps={meta.get('steps')}")
    _, rq = sod_exact.cell_averages(ncells, SOD_T)
    _, re = exact_cell_avgs(ncells, SOD_T, LEFT, RIGHT)
    dx = 1.0 / ncells
    for key, (field, xs) in STATIONS.items():
        raw = xs * ncells
        idx = min(int(raw), ncells - 1)
        for tag, i in (("coded", idx), ("left", max(idx-1, 0))):
            g = float(got[field][i]); r = float(re[field][i]); rq_v = float(rq[field][i])
            lo, hi = i*dx, (i+1)*dx
            near = min(abs(cuts - lo).min(), abs(cuts - hi).min()) / dx
            inside = bool(((cuts > lo) & (cuts < hi)).any())
            print(f" {key:14s} {tag:5s} i={i:4d} cell=[{lo:.5f},{hi:.5f}] wave={inside} gap={near:6.3f}dx")
            print(f"   got={g:.8f} ref_exact={r:.8f} err={abs(g-r)/abs(r):7.4%} | "
                  f"ref_gauss={rq_v:.8f} instr_delta={abs(rq_v-r)/abs(r):.2e}")