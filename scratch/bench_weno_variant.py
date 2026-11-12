"""End-to-end: linear advection with the WENO limiter, js vs z weights.

Square-wave advection (u_t + u_x = 0, periodic, SSP-RK3, P2) is time-integrated
to t = 1; the reported error is the L1 norm of the cell-average error.  The
square wave keeps the limiter active, the sine wave keeps it (mostly) released,
so the two columns bound what the weight variant can change.
"""
import json
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
from tensordg.advection_1d import solve_advection


def square(x):
    return torch.where(x < 0.5, torch.ones_like(x), torch.zeros_like(x))


def sine(x):
    return torch.sin(2 * np.pi * x)


def cell_average_error(profile, ncells, coeff, shift=0.2):
    nodes, weights = np.polynomial.legendre.leggauss(8)
    centres = (np.arange(ncells) + 0.5) / ncells
    x = centres[:, None] + 0.5 * nodes[None, :] / ncells
    exact = profile(torch.tensor(x - shift)).detach().numpy()
    mean = 0.5 * (weights[None, :] * exact).sum(axis=1)
    avg = coeff[:, 0].detach().numpy()
    return float(np.abs(avg - mean).mean()), float(np.sqrt(((avg - mean) ** 2).mean()))


rows = []
for name, profile in (("square", square), ("sine", sine)):
    for ncells in (40, 80):
        for variant in ("js", "z"):
            _, coeff, meta = solve_advection(
                ncells=ncells, degree=2, final_time=0.2, cfl=0.45,
                initial=profile, limiter="weno", tvb_constant=1.0,
                time_integrator="ssp_rk3", weno_variant=variant,
            )
            l1, l2 = cell_average_error(profile, ncells, coeff)
            rows.append({"profile": name, "ncells": ncells, "variant": variant,
                         "steps": meta["steps"], "L1": l1, "L2": l2})
            print(name, ncells, variant, meta["steps"], f"{l1:.6e}", f"{l2:.6e}", flush=True)

with open("artifacts/weno_z_advection.json", "w") as fh:
    json.dump(rows, fh, indent=2)
