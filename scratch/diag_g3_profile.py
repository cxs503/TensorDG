"""Diagnostic: where is the reflected plateau and how wide is the numerical front?"""
import sys, time
import numpy as np
import torch

from benchmarks.euler_dg_2d_wall_bc import WallReflection
from tensordg.euler_dg_2d import solve_euler_dg_2d, _pressure


GAMMA = 1.4


def run(nx, degree=2, cfl=0.12, extra_time=0.08, limiter=True, x_shock=0.4,
        x_wall=1.0, xmax=1.2, ny=1, ymax=0.1):
    wr = WallReflection(x_shock=x_shock, x_wall=x_wall, incident_mach=2.0)
    dx = xmax / nx
    xc = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    solid = torch.zeros(nx, ny, dtype=torch.bool)
    solid[xc >= x_wall] = True
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        ahead = x < wr.x_zero
        return (torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
                torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
                torch.zeros_like(x),
                torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)))

    t_end = wr.wall_time + extra_time
    t0 = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=(nx, ny), degree=degree, final_time=t_end, cfl=cfl,
        domain=((0.0, xmax), (0.0, ymax)), periodic=(False, False), solid=solid,
        limiter=("tvb" if limiter else None),
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"})
    wall = time.time() - t0
    avg = coeff[..., 0, 0, :]
    rho = avg[..., 0].numpy()[:, 0]
    p = _pressure(avg).numpy()[:, 0]
    print(f"nx={nx} degree={degree} steps={meta['steps']} wall={wall:.1f}s "
          f"solid_cells={meta['solid_cells']} t_end={t_end:.4f}")
    print(f"  exact: pre rho={pre_rho} u={pre_u:.4f} p={pre_p}")
    print(f"         post(2) rho={post_rho:.4f} u={post_u:.4f} p={post_p:.4f}")
    print(f"         refl(5) rho={wr.reflected[0]:.4f} u={wr.reflected[1]:.4f} p={wr.reflected[2]:.4f}"
          f"  M_r={wr.reflected_mach:.4f}")
    front = wr.front(t_end)
    print(f"  exact reflected front x={front:.4f} (wall {x_wall:.4f}), plateau width={x_wall-front:.4f}"
          f" = {(x_wall-front)/dx:.2f} cells")
    for i in range(nx):
        xl = i * dx
        mark = ""
        if front - dx < xl < x_wall + dx:
            mark = "<-- reflected region"
        print(f"  i={i:4d} x=[{xl:.4f},{xl+dx:.4f}] solid={int(solid[i,0])} rho={rho[i]:9.4f} p={p[i]:9.4f} {mark}")
    return wr, rho, p


if __name__ == "__main__":
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run(nx, deg)