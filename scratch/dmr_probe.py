"""Probe: which orientation of the Woodward-Colella DMR initial data actually
produces a shock that sweeps down-right (and hence a wall reflection)?

sign=+1 : post-shock on the down-right side of the front (current repo code)
sign=-1 : post-shock on the up-left side of the front (standard Woodward-Colella,
          left boundary = post-shock inflow, shock sweeps right into the wall)
"""
import time
import numpy as np
import torch

from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
from benchmarks.oblique_shock import ShockState, oblique_shock

PRE = (1.4, 0.0, 0.0, 1.0)
N = (np.sin(np.deg2rad(60.0)), -np.cos(np.deg2rad(60.0)))   # (0.866, -0.5) down-right
W_SHOCK = 10.0
_POST = oblique_shock(10.0, 60.0, ShockState(1.4, W_SHOCK * N[0], W_SHOCK * N[1], 1.0))
POST = (_POST.downstream.rho,
        _POST.downstream.u - W_SHOCK * N[0],
        _POST.downstream.v - W_SHOCK * N[1],
        _POST.downstream.p)
print("POST", POST)
T60 = np.tan(np.deg2rad(60.0))


def make_ic(sign):
    def ic(x, y):
        front = T60 * (x.cpu().numpy() - 1.0 / 6.0)
        mask = (y.cpu().numpy() < front) if sign > 0 else (y.cpu().numpy() > front)
        m = torch.as_tensor(mask)
        rho = torch.where(m, torch.full_like(x, POST[0]), torch.full_like(x, PRE[0]))
        vx = torch.where(m, torch.full_like(x, POST[1]), torch.full_like(x, PRE[1]))
        vy = torch.where(m, torch.full_like(x, POST[2]), torch.full_like(x, PRE[2]))
        p = torch.where(m, torch.full_like(x, POST[3]), torch.full_like(x, PRE[3]))
        return rho, vx, vy, p
    return ic


for sign in (+1, -1):
    ic = make_ic(sign)
    t0 = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        ic, ncells=(60, 15), degree=1, domain=((0.0, 4.0), (0.0, 1.0)),
        final_time=0.01, cfl=0.15, limiter="tvb", tvb_constant=100.0,
        limiter_stage="final", periodic=(False, False),
        initial_projection="cell-average",
        bcs={"bottom": "wall"},
    )
    dt = time.time() - t0
    rho = cell_fields(coeff)["density"].cpu().numpy()
    xc = x.cpu().numpy().ravel()
    yc = y.cpu().numpy().ravel()
    j = int(np.argmin(np.abs(yc - 0.8)))
    prof = rho[:, j]
    grad = np.abs(np.diff(prof))
    i = int(np.argmax(grad))
    x_front = 0.5 * (xc[i] + xc[i + 1])
    shift = W_SHOCK / np.sin(np.deg2rad(60.0)) * 0.01
    analytic = 1.0 / 6.0 + 0.5773502691896258 * yc[j] + (shift if sign < 0 else -shift)
    print(f"sign={sign:+d} steps={meta['steps']} wall={dt:.1f}s rhoL={prof[0]:.3f} "
          f"rhoR={prof[-1]:.3f} x_front={x_front:.3f} analytic={analytic:.3f} "
          f"rng=({rho.min():.3f},{rho.max():.3f})")