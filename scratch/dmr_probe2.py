import time
import numpy as np
import torch

from tensordg.euler_dg_2d import solve_euler_dg_2d, cell_fields
from benchmarks.oblique_shock import ShockState, oblique_shock

PRE = (1.4, 0.0, 0.0, 1.0)
N = (np.sin(np.deg2rad(60.0)), -np.cos(np.deg2rad(60.0)))
W = 10.0
_POST = oblique_shock(10.0, 60.0, ShockState(1.4, W * N[0], W * N[1], 1.0))
POST = (_POST.downstream.rho, _POST.downstream.u - W * N[0],
        _POST.downstream.v - W * N[1], _POST.downstream.p)
T60 = np.tan(np.deg2rad(60.0))


def make_ic(sign):
    def ic(x, y):
        front = T60 * (x.cpu().numpy() - 1.0 / 6.0)
        mask = (y.cpu().numpy() < front) if sign > 0 else (y.cpu().numpy() > front)
        m = torch.as_tensor(mask)
        return tuple(torch.where(m, torch.full_like(x, s1), torch.full_like(x, s0))
                     for s1, s0 in zip(POST, PRE))
    return ic


for sign in (+1, -1):
    ic = make_ic(sign)
    for t in (0.005, 0.01):
        x, y, coeff, meta = solve_euler_dg_2d(
            ic, ncells=(120, 30), degree=1, domain=((0.0, 4.0), (0.0, 1.0)),
            final_time=t, cfl=0.15, limiter="tvb", tvb_constant=100.0,
            limiter_stage="final", periodic=(False, False),
            initial_projection="cell-average", bcs={"bottom": "wall"},
        )
        rho = cell_fields(coeff)["density"].cpu().numpy()
        xc = x.cpu().numpy().ravel()
        yc = y.cpu().numpy().ravel()
        j = int(np.argmin(np.abs(yc - 0.8)))
        prof = rho[:, j]
        # first crossing of the 2.5 level scanning from ahead side
        crossings = np.where(np.diff(np.sign(prof - 2.5)) != 0)[0]
        xs = [0.5 * (xc[i] + xc[i + 1]) for i in crossings]
        print(f"sign={sign:+d} t={t} steps={meta['steps']} crossings(2.5)={[round(v,3) for v in xs]} "
              f"profile={np.round(prof[::8],2)}")